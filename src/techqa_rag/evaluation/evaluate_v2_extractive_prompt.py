"""Compare a concise extractive prompt on fixed retrieved and oracle contexts."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from techqa_rag.evaluation.evaluate_v2_context_generation import (
    ID_PATTERN,
    load_qa,
    normalize_answer,
    token_f1,
)


SYSTEM_PROMPT = (
    "Find the shortest complete sentence or phrase in the supplied IBM documentation that directly answers the question. "
    "Copy that sentence or phrase exactly as written. Preserve product names, versions, identifiers, file paths, "
    "commands, and error text exactly. Do not paraphrase or add explanation. Return only copied text. "
    "If the context contains no relevant answer, return exactly: NO_ANSWER."
)


def build_extract_prompt(tokenizer: Any, question: str, context: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Documentation context:\n{context}\n\n"
                    "Return only the answer."
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-qa", type=Path, default=root.parent / "TechQA/TechQA/training_and_dev/dev_Q_A.json")
    parser.add_argument("--retrieved-contexts", type=Path, default=root / "results/evaluation/v2_context_ablation.json")
    parser.add_argument("--oracle-contexts", type=Path, default=root / "results/generation/v2_oracle_context_generation.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-jsonl", type=Path, default=root / "results/generation/v2_extractive_prompt_generation.jsonl")
    parser.add_argument("--output-csv", type=Path, default=root / "results/evaluation/v2_extractive_prompt_generation.csv")
    parser.add_argument("--summary-json", type=Path, default=root / "results/evaluation/v2_extractive_prompt_generation_summary.json")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--input-token-limit", type=int, default=4096)
    parser.add_argument("--question-ids", nargs="+", default=None, help="Optional stratified pilot question IDs")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    qa_by_id = load_qa(args.dev_qa)
    selected = json.loads(args.retrieved_contexts.read_text(encoding="utf-8"))["questions"]
    retrieved_by_id = {str(row["question_id"]): row for row in selected}
    oracle_by_id: dict[str, dict[str, Any]] = {}
    with args.oracle_contexts.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("condition") == "oracle_answer_passage":
                oracle_by_id[str(row["question_id"])] = row
    qids = sorted(set(qa_by_id) & set(retrieved_by_id) & set(oracle_by_id))
    if len(qids) != len(qa_by_id):
        raise ValueError(f"Expected contexts for all {len(qa_by_id)} answerable questions; found {len(qids)}")
    if args.question_ids:
        missing = sorted(set(args.question_ids) - set(qids))
        if missing:
            raise ValueError(f"Requested pilot question IDs are missing: {missing}")
        qids = [qid for qid in qids if qid in set(args.question_ids)]
    if args.limit is not None:
        qids = qids[:args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output_jsonl.exists():
        for line in args.output_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                completed[(str(row["question_id"]), str(row["condition"]))] = row
            except (json.JSONDecodeError, KeyError):
                continue

    conditions = ("retrieved_adjacent_expansion", "oracle_answer_passage")
    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for q_index, qid in enumerate(qids, 1):
            qa = qa_by_id[qid]
            question = str(qa.get("QUESTION_TEXT", ""))
            reference = str(qa.get("ANSWER") or "")
            inputs = {
                "retrieved_adjacent_expansion": (
                    str(retrieved_by_id[qid]["adjacent_expansion_context"]),
                    bool(retrieved_by_id[qid]["adjacent_expansion"].get("normalized_reference_in_selected_context")),
                    bool(retrieved_by_id[qid]["adjacent_expansion"].get("offset_span_in_selected_context")),
                ),
                "oracle_answer_passage": (
                    str(oracle_by_id[qid]["context_text"]),
                    bool(oracle_by_id[qid]["reference_present_in_context"]),
                    True,
                ),
            }
            for condition in conditions:
                key = (qid, condition)
                if key in completed and not completed[key].get("generation_error"):
                    continue
                context, contains_reference, contains_span = inputs[condition]
                prompt = build_extract_prompt(tokenizer, question, context)
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
                prompt_tokens = int(encoded["input_ids"].shape[1])
                if prompt_tokens > args.input_token_limit:
                    raise ValueError(f"{qid}/{condition} prompt exceeds {args.input_token_limit} tokens: {prompt_tokens}")
                encoded = {name: value.to(device) for name, value in encoded.items()}
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                if device == "cuda":
                    torch.cuda.synchronize()
                answer = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True).strip()
                required_ids = sorted({value.upper() for value in ID_PATTERN.findall(reference)})
                predicted_ids = {value.upper() for value in ID_PATTERN.findall(answer)}
                row = {
                    "question_id": qid,
                    "condition": condition,
                    "question": question,
                    "reference_answer": reference,
                    "generated_answer": answer,
                    "token_f1": token_f1(answer, reference),
                    "normalized_exact_match": bool(normalize_answer(reference)) and normalize_answer(answer) == normalize_answer(reference),
                    "required_ids": required_ids,
                    "required_ids_present": set(required_ids).issubset(predicted_ids) if required_ids else None,
                    "selected_context_contains_reference": contains_reference,
                    "selected_context_contains_offset_span": contains_span,
                    "prompt_tokens": prompt_tokens,
                    "context_characters": len(context),
                    "context_text": context,
                    "generated_token_count": int(generated.shape[1] - prompt_tokens),
                    "generation_seconds": round(time.perf_counter() - started, 3),
                    "generation_error": "",
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                completed[key] = row
                print(f"[{q_index}/{len(qids)}] {qid} {condition}: F1={row['token_f1']:.3f}; prompt={prompt_tokens}")

    latest = list(completed.values())
    summary: dict[str, Any] = {
        "experiment": "Controlled concise-extractive prompt comparison on fixed retrieved and oracle contexts",
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "input_token_limit": args.input_token_limit,
        "conditions": {},
        "paired": {},
        "limitations": [
            "Oracle answer passage uses development gold labels and is diagnostic only.",
            "Token F1 and normalized exact match are single-reference lexical measures, not human correctness judgments.",
            "Only answerable development questions are included; two empty references are excluded from answer-quality means.",
        ],
    }
    by_key = {(r["question_id"], r["condition"]): r for r in latest}
    for condition in conditions:
        subset = [r for r in latest if r["condition"] == condition]
        scored = [r for r in subset if r["reference_answer"].strip()]
        f1s = [float(r["token_f1"]) for r in scored]
        summary["conditions"][condition] = {
            "generated_questions": len(subset),
            "nonempty_references_scored": len(scored),
            "mean_token_f1": statistics.mean(f1s) if f1s else None,
            "median_token_f1": statistics.median(f1s) if f1s else None,
            "normalized_exact_match_count": sum(bool(r["normalized_exact_match"]) for r in scored),
            "normalized_exact_match_rate": statistics.mean(bool(r["normalized_exact_match"]) for r in scored) if scored else None,
            "mean_prompt_tokens": statistics.mean(int(r["prompt_tokens"]) for r in subset) if subset else None,
            "generation_errors": sum(bool(r.get("generation_error")) for r in subset),
            "by_context_reference_presence": {
                str(present).lower(): {
                    "questions": len(group := [r for r in scored if bool(r["selected_context_contains_reference"]) == present]),
                    "mean_token_f1": statistics.mean(float(r["token_f1"]) for r in group) if group else None,
                }
                for present in (True, False)
            },
        }
    paired_qids = sorted({q for q,c in by_key if c == conditions[0]} & {q for q,c in by_key if c == conditions[1]})
    paired_qids = [q for q in paired_qids if by_key[(q, conditions[0])]["reference_answer"].strip()]
    deltas = [float(by_key[(q, conditions[0])]["token_f1"]) - float(by_key[(q, conditions[1])]["token_f1"]) for q in paired_qids]
    summary["paired"] = {
        "questions": len(paired_qids),
        "mean_token_f1_delta_retrieved_minus_oracle": statistics.mean(deltas) if deltas else None,
        "retrieved_wins": sum(x > 0 for x in deltas),
        "ties": sum(x == 0 for x in deltas),
        "oracle_wins": sum(x < 0 for x in deltas),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(k for r in latest for k in r))
    with args.output_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in latest:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
