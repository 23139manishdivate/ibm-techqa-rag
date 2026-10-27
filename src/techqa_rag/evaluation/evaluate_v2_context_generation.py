"""Generate paired answers for the saved v2 BM25 plain/expanded contexts.

The script consumes contexts emitted by run_v2_context_ablation.py, uses the
same Qwen prompt and answer scoring as the archived oracle-context evaluation,
and checkpoints every condition to JSONL for safe resumption.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from techqa_rag.evaluation.gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    create_gemini_client,
    generate_with_gemini,
)


SYSTEM_PROMPT = (
    "Answer the question using only the supplied IBM technical documentation. "
    "Be concise and preserve exact technical identifiers. "
    "Do not invent fixes, APAR numbers, or commands. "
    "If the evidence does not support an answer, say so."
)
CONDITIONS = {
    "plain_top_chunks": "plain_top_chunks_context",
    "adjacent_expansion": "adjacent_expansion_context",
}
ID_PATTERN = re.compile(r"\b(?:PI|PM|IV|JR|LO|PK)\d{4,6}\b", re.IGNORECASE)


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def load_qa(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    answerable = [
        row for row in rows
        if str(row.get("ANSWERABLE", "")).strip().upper() in {"Y", "YES", "TRUE", "1"}
    ]
    output = {str(row["QUESTION_ID"]): row for row in answerable}
    if len(output) != len(answerable):
        raise ValueError("Development QA contains duplicate question IDs")
    return output


def build_prompt(tokenizer: Any, question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Documentation context:\n{context}\n\n"
                "Give a direct answer supported by the context."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = {(r["question_id"], r["condition"]): r for r in rows}
    summary: dict[str, Any] = {"conditions": {}, "paired": {}}
    per_condition: dict[str, dict[str, dict[str, Any]]] = {}

    def score_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [row for row in subset if row["reference_answer"].strip()]
        f1s = [float(row["token_f1"]) for row in scored]
        exact = [bool(row["normalized_exact_match"]) for row in scored]
        return {
            "questions": len(subset),
            "nonempty_references_scored": len(scored),
            "mean_token_f1": statistics.mean(f1s) if f1s else None,
            "normalized_exact_match_count": sum(exact),
            "normalized_exact_match_rate": statistics.mean(exact) if exact else None,
        }

    for condition in CONDITIONS:
        subset = {
            qid: row for (qid, label), row in latest.items()
            if label == condition and not row.get("generation_error")
        }
        per_condition[condition] = subset
        scored = [row for row in subset.values() if row["reference_answer"].strip()]
        f1s = [float(row["token_f1"]) for row in scored]
        exact = [bool(row["normalized_exact_match"]) for row in scored]
        identifier_rows = [row for row in scored if row["required_ids"]]
        summary["conditions"][condition] = {
            "generated_questions": len(subset),
            "nonempty_references_scored": len(scored),
            "mean_token_f1": statistics.mean(f1s) if f1s else None,
            "median_token_f1": statistics.median(f1s) if f1s else None,
            "normalized_exact_match_count": sum(exact),
            "normalized_exact_match_rate": statistics.mean(exact) if exact else None,
            "required_identifier_questions": len(identifier_rows),
            "required_identifier_all_match_count": sum(
                bool(row["required_ids_present"]) for row in identifier_rows
            ),
            "required_identifier_all_match_rate": (
                statistics.mean(bool(row["required_ids_present"]) for row in identifier_rows)
                if identifier_rows else None
            ),
            "mean_prompt_tokens": (
                statistics.mean(
                    int(row["prompt_tokens"]) for row in subset.values()
                    if row.get("prompt_tokens") is not None
                )
                if any(row.get("prompt_tokens") is not None for row in subset.values()) else None
            ),
            "prompt_truncations": sum(bool(row["prompt_truncated"]) for row in subset.values()),
            "generation_errors": sum(bool(row.get("generation_error")) for row in latest.values() if row["condition"] == condition),
            "by_context_evidence": {
                "reference_present": score_subset([
                    row for row in scored if row["selected_context_contains_reference"]
                ]),
                "reference_absent": score_subset([
                    row for row in scored if not row["selected_context_contains_reference"]
                ]),
                "offset_span_present": score_subset([
                    row for row in scored if row["selected_context_contains_offset_span"]
                ]),
                "offset_span_absent": score_subset([
                    row for row in scored if not row["selected_context_contains_offset_span"]
                ]),
            },
        }

    paired_ids = sorted(set(per_condition["plain_top_chunks"]) & set(per_condition["adjacent_expansion"]))
    paired = [
        qid for qid in paired_ids
        if per_condition["plain_top_chunks"][qid]["reference_answer"].strip()
    ]
    deltas = [
        float(per_condition["adjacent_expansion"][qid]["token_f1"])
        - float(per_condition["plain_top_chunks"][qid]["token_f1"])
        for qid in paired
    ]
    summary["paired"] = {
        "questions_with_both_conditions": len(paired_ids),
        "paired_questions_with_nonempty_reference": len(paired),
        "mean_token_f1_delta_expansion_minus_plain": statistics.mean(deltas) if deltas else None,
        "expansion_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "plain_wins": sum(delta < 0 for delta in deltas),
    }
    evidence_groups: dict[str, list[float]] = collections.defaultdict(list)
    for qid in paired:
        plain_has = bool(per_condition["plain_top_chunks"][qid]["selected_context_contains_reference"])
        expanded_has = bool(per_condition["adjacent_expansion"][qid]["selected_context_contains_reference"])
        label = (
            "both_contexts_contain_reference" if plain_has and expanded_has
            else "expansion_only_contains_reference" if expanded_has
            else "plain_only_contains_reference" if plain_has
            else "neither_context_contains_reference"
        )
        evidence_groups[label].append(
            float(per_condition["adjacent_expansion"][qid]["token_f1"])
            - float(per_condition["plain_top_chunks"][qid]["token_f1"])
        )
    summary["paired"]["by_reference_evidence_transition"] = {
        label: {
            "questions": len(values),
            "mean_token_f1_delta_expansion_minus_plain": statistics.mean(values),
            "expansion_wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "plain_wins": sum(value < 0 for value in values),
        }
        for label, values in evidence_groups.items()
    }
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=project_root / "results/evaluation/v2_context_ablation.json")
    parser.add_argument("--dev-qa", type=Path, default=project_root.parent / "TechQA/TechQA/training_and_dev/dev_Q_A.json")
    parser.add_argument("--provider", choices=("qwen", "gemini"), default="qwen")
    parser.add_argument("--model", default=None, help="Model ID; defaults to the selected provider's standard model")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--request-delay-seconds", type=float, default=3.0, help="Pause between Gemini calls to reduce rate-limit errors")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-run limit; omit for all questions")
    args = parser.parse_args()
    args.model = args.model or (
        DEFAULT_GEMINI_MODEL if args.provider == "gemini" else "Qwen/Qwen2.5-1.5B-Instruct"
    )
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    is_default_qwen = args.model == "Qwen/Qwen2.5-1.5B-Instruct"
    suffix = f"_{args.provider}_{model_tag}" if args.provider == "gemini" or not is_default_qwen else ""
    args.output_jsonl = args.output_jsonl or project_root / f"results/generation/v2_bm25_context_generation{suffix}.jsonl"
    args.output_csv = args.output_csv or project_root / f"results/evaluation/v2_bm25_context_generation{suffix}.csv"
    args.summary_json = args.summary_json or project_root / f"results/evaluation/v2_bm25_context_generation_summary{suffix}.json"
    if args.request_delay_seconds < 0:
        parser.error("--request-delay-seconds must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    torch = None

    if not args.contexts.exists():
        raise FileNotFoundError(f"Missing context-selection results: {args.contexts}")
    context_data = json.loads(args.contexts.read_text(encoding="utf-8"))
    qa_by_id = load_qa(args.dev_qa)
    questions = context_data["questions"]
    if len(questions) != len(qa_by_id):
        raise ValueError(f"Context rows ({len(questions)}) do not match answerable QA rows ({len(qa_by_id)})")
    if any(key not in questions[0] for key in CONDITIONS.values()):
        raise ValueError("Saved selection results do not include exact context text; rerun the context ablation first")
    if args.limit is not None:
        questions = questions[:args.limit]

    if args.provider == "gemini":
        client = create_gemini_client()
        tokenizer = model = None
        print(f"Using Gemini API model {args.model}; no local generation model or GPU is needed.")
    else:
        import torch as torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_module
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Loading {args.model} on {device} with {dtype}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        client = None
        if device == "cuda":
            print("GPU memory allocated after load:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output_jsonl.exists():
        with args.output_jsonl.open(encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing[(str(row["question_id"]), str(row["condition"]))] = row
    completed = {
        key for key, row in existing.items() if not row.get("generation_error")
    }
    print("Resuming; completed answer conditions:", len(completed))

    with args.output_jsonl.open("a", encoding="utf-8") as output_file:
        for q_idx, result in enumerate(questions, start=1):
            qid = str(result["question_id"])
            qa = qa_by_id[qid]
            question = str(qa.get("QUESTION_TEXT", ""))
            reference = str(qa.get("ANSWER") or "")
            for condition, context_key in CONDITIONS.items():
                record_key = (qid, condition)
                if record_key in completed:
                    continue
                context = str(result[context_key])
                encoded = None
                if args.provider == "qwen":
                    prompt = build_prompt(tokenizer, question, context)
                    encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        add_special_tokens=False,
                        truncation=False,
                    )
                    prompt_tokens = int(encoded["input_ids"].shape[1])
                    if prompt_tokens > 4096:
                        raise ValueError(f"{qid}/{condition} prompt exceeds 4096 tokens: {prompt_tokens}")
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                else:
                    prompt_tokens = None
                generation_start = time.perf_counter()
                try:
                    if args.provider == "gemini":
                        generated_result = generate_with_gemini(
                            client,
                            model=args.model,
                            system_prompt=SYSTEM_PROMPT,
                            question=question,
                            context=context,
                            max_output_tokens=args.max_new_tokens,
                        )
                        answer = generated_result["answer"]
                        prompt_tokens = generated_result["prompt_tokens"]
                        generated_token_count = generated_result["generated_token_count"]
                    else:
                        with torch.inference_mode():
                            generated = model.generate(
                                **encoded,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        if device == "cuda":
                            torch.cuda.synchronize()
                        answer = tokenizer.decode(
                            generated[0][prompt_tokens:], skip_special_tokens=True
                        ).strip()
                        generated_token_count = int(generated.shape[1] - prompt_tokens)
                    generation_seconds = time.perf_counter() - generation_start
                    error = ""
                except Exception as exc:
                    generation_seconds = time.perf_counter() - generation_start
                    answer = ""
                    generated_token_count = 0
                    error = repr(exc)
                    if args.provider == "qwen" and isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise

                normalized_prediction = normalize_answer(answer)
                normalized_reference = normalize_answer(reference)
                required_ids = sorted({value.upper() for value in ID_PATTERN.findall(reference)})
                predicted_ids = {value.upper() for value in ID_PATTERN.findall(answer)}
                row = {
                    "question_id": qid,
                    "condition": condition,
                    "provider": args.provider,
                    "model": args.model,
                    "question": question,
                    "reference_answer": reference,
                    "generated_answer": answer,
                    "token_f1": token_f1(answer, reference),
                    "normalized_exact_match": bool(normalized_reference) and normalized_prediction == normalized_reference,
                    "required_ids": required_ids,
                    "required_ids_present": (set(required_ids).issubset(predicted_ids) if required_ids else None),
                    "prompt_tokens": prompt_tokens,
                    "generation_seconds": round(generation_seconds, 3),
                    "generated_token_count": generated_token_count,
                    "prompt_truncated": False,
                    "context_characters": len(context),
                    "selected_context_contains_reference": result[condition]["normalized_reference_in_selected_context"],
                    "selected_context_contains_offset_span": result[condition]["offset_span_in_selected_context"],
                    "generation_error": error,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                existing[record_key] = row
                if not error:
                    completed.add(record_key)
                print(
                    f"[{q_idx}/{len(questions)}] {qid} {condition}: "
                    f"F1={row['token_f1']:.3f}; prompt={prompt_tokens}; "
                    f"generated={generated_token_count} tokens in {generation_seconds:.1f}s"
                )
                if args.provider == "gemini" and error:
                    raise RuntimeError(
                        f"Gemini call failed for {qid}/{condition}; the checkpoint row was saved. "
                        "Fix the API/model/quota issue and rerun; completed successful rows will resume."
                    )
                if encoded is not None:
                    del encoded
                if args.provider == "gemini" and not error and args.request_delay_seconds:
                    time.sleep(args.request_delay_seconds)

    final_rows = list(existing.values())
    summary = {
        "experiment": f"{args.provider} {args.model} on identical BM25 plain vs adjacent-expanded contexts",
        "provider": args.provider,
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "context_selection_tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
        "context_selection_input_token_limit": 4096,
        "generation_prompt_tokenizer": "Qwen tokenizer" if args.provider == "qwen" else "Gemini API usage metadata",
        "selection_results": str(args.contexts),
        "limitations": [
            "Only the 160 answerable development questions are evaluated.",
            "Two development references are empty and excluded from mean token F1 and exact-match denominators.",
            "Token F1 and normalized exact match are reference-overlap metrics, not human correctness judgments.",
            "The fixed document candidate rankings come from the prior retrieval/index version.",
            "The saved context candidates were built using the existing Qwen token budget; Gemini reports its own prompt-token usage.",
            "Gemini API results use the same question and selected context text, but Gemini tokenization and serving behavior differ from local Qwen.",
        ],
        **summarize(final_rows),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if final_rows:
        fieldnames = list(final_rows[-1])
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for row in final_rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in row.items()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSONL: {args.output_jsonl.resolve()}")
    print(f"CSV: {args.output_csv.resolve()}")
    print(f"Summary: {args.summary_json.resolve()}")


if __name__ == "__main__":
    main()
