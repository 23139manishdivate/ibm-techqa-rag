"""Run a controlled reader diagnostic with oracle passage and gold document contexts.

The oracle conditions deliberately use development labels and are diagnostics only;
they must never be used as inference-time retrieval candidates.
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

from techqa_rag.evaluation.evaluate_v2_context_generation import (
    SYSTEM_PROMPT,
    build_prompt,
    load_qa,
    normalize_answer,
    token_f1,
)
from techqa_rag.evaluation.gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    create_gemini_client,
    generate_with_gemini,
)


def locate_normalized_span(text: str, reference: str) -> tuple[int, int] | None:
    """Locate normalized reference text while retaining offsets into source text."""
    normalized_chars: list[str] = []
    source_offsets: list[int] = []
    pending_space = False
    pending_offset = 0
    for index, char in enumerate(text.lower()):
        if char.isalnum():
            if pending_space and normalized_chars:
                normalized_chars.append(" ")
                source_offsets.append(pending_offset)
            pending_space = False
            normalized_chars.append(char)
            source_offsets.append(index)
        elif normalized_chars and normalized_chars[-1] != " ":
            pending_space = True
            pending_offset = index
    normalized_text = "".join(normalized_chars).strip()
    target = normalize_answer(reference)
    if not target:
        return None
    start = normalized_text.find(target)
    if start < 0:
        return None
    end = start + len(target) - 1
    # strip() can only remove outer spaces; find the corresponding map start.
    left_trim = len("".join(normalized_chars)) - len("".join(normalized_chars).lstrip())
    return source_offsets[start + left_trim], source_offsets[end + left_trim] + 1


def load_gold_texts(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            doc_id = str(row.get("document_id", row.get("id", "")))
            if doc_id in wanted:
                found[doc_id] = row
                if len(found) == len(wanted):
                    break
    missing = sorted(wanted - found.keys())
    if missing:
        raise FileNotFoundError(f"Canonical corpus is missing {len(missing)} gold documents: {missing[:10]}")
    return found


def gold_span(qa: dict[str, Any], text: str) -> tuple[int, int, str]:
    answer = str(qa.get("ANSWER") or "")
    try:
        start, end = int(qa.get("START_OFFSET", -1)), int(qa.get("END_OFFSET", -1))
    except (TypeError, ValueError):
        start, end = -1, -1
    if 0 <= start < end <= len(text) and answer and normalize_answer(text[start:end]) == normalize_answer(answer):
        return start, end, "validated_supplied_offsets"
    if answer:
        located = locate_normalized_span(text, answer)
        if located:
            return located[0], located[1], "normalized_reference_match"
    if 0 <= start < end <= len(text):
        return start, end, "supplied_offsets_reference_mismatch"
    raise ValueError(f"No usable answer offsets for {qa['QUESTION_ID']}")


def passage_context(doc: dict[str, Any], text: str, start: int, end: int, tokenizer: Any) -> tuple[str, dict[str, Any]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = enc["offset_mapping"]
    overlapping = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not overlapping:
        token_start = token_end = 0
    else:
        token_start, token_end = overlapping[0], overlapping[-1] + 1
    window_start = max(0, token_start - 256)
    window_end = min(len(offsets), token_end + 256)
    char_start = offsets[window_start][0] if offsets and window_start < len(offsets) else 0
    char_end = offsets[window_end - 1][1] if offsets and window_end else len(text)
    title = str(doc.get("title", ""))
    context = f"[Gold document: {doc.get('document_id', doc.get('id', ''))}; title: {title}]\n{text[char_start:char_end]}"
    return context, {
        "source_char_start": char_start,
        "source_char_end": char_end,
        "answer_start_in_context": start - char_start,
        "answer_end_in_context": end - char_start,
        "answer_token_count": token_end - token_start,
        "passage_window_token_count": window_end - window_start,
    }


def fit_full_document_context(doc: dict[str, Any], text: str, question: str, tokenizer: Any, limit: int) -> tuple[str, int, bool]:
    title = str(doc.get("title", ""))
    prefix = f"[Gold document: {doc.get('document_id', doc.get('id', ''))}; title: {title}]\n"
    doc_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    low, high = 0, len(doc_ids)
    best = ""
    best_token_count = 0
    while low <= high:
        mid = (low + high) // 2
        candidate = prefix + tokenizer.decode(doc_ids[:mid], skip_special_tokens=True)
        count = len(tokenizer(build_prompt(tokenizer, question, candidate), add_special_tokens=False)["input_ids"])
        if count <= limit:
            best = candidate
            best_token_count = mid
            low = mid + 1
        else:
            high = mid - 1
    used = len(tokenizer(build_prompt(tokenizer, question, best), add_special_tokens=False)["input_ids"])
    return best, used, best_token_count < len(doc_ids)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-qa", type=Path, default=root.parent / "TechQA/TechQA/training_and_dev/dev_Q_A.json")
    parser.add_argument("--canonical", type=Path, default=root / "data/processed/v2/canonical_technotes_v1.jsonl")
    parser.add_argument("--provider", choices=("qwen", "gemini"), default="qwen")
    parser.add_argument("--model", default=None, help="Model ID; defaults to the selected provider's standard model")
    parser.add_argument("--context-tokenizer", default=None, help="Tokenizer used only to build comparable oracle windows")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--input-token-limit", type=int, default=4096)
    parser.add_argument("--request-delay-seconds", type=float, default=3.0, help="Pause between Gemini calls to reduce rate-limit errors")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.model = args.model or (
        DEFAULT_GEMINI_MODEL if args.provider == "gemini" else "Qwen/Qwen2.5-1.5B-Instruct"
    )
    args.context_tokenizer = args.context_tokenizer or "Qwen/Qwen2.5-1.5B-Instruct"
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    is_default_qwen = args.model == "Qwen/Qwen2.5-1.5B-Instruct"
    suffix = f"_{args.provider}_{model_tag}" if args.provider == "gemini" or not is_default_qwen else ""
    args.output_jsonl = args.output_jsonl or root / f"results/generation/v2_oracle_context_generation{suffix}.jsonl"
    args.output_summary = args.output_summary or root / f"results/evaluation/v2_oracle_context_generation_summary{suffix}.json"
    args.output_csv = args.output_csv or root / f"results/evaluation/v2_oracle_context_generation{suffix}.csv"
    if args.request_delay_seconds < 0:
        parser.error("--request-delay-seconds must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    qa_by_id = load_qa(args.dev_qa)
    questions = list(qa_by_id.values())
    if args.limit is not None:
        questions = questions[:args.limit]
    gold_ids = {str(row["DOCUMENT"]) for row in questions}
    docs = load_gold_texts(args.canonical, gold_ids)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.context_tokenizer)
    if args.provider == "gemini":
        client = create_gemini_client()
        model = torch = device = None
        print(f"Using Gemini API model {args.model}; Qwen tokenizer is retained only to build comparable oracle windows.")
    else:
        import torch as torch_module
        from transformers import AutoModelForCausalLM

        torch = torch_module
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        if args.context_tokenizer != args.model:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
        model.eval()
        client = None
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output_jsonl.exists():
        for line in args.output_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                completed[(row["question_id"], row["condition"])] = row
            except (json.JSONDecodeError, KeyError):
                continue
    with args.output_jsonl.open("a", encoding="utf-8") as out:
        for q_index, qa in enumerate(questions, 1):
            qid = str(qa["QUESTION_ID"])
            doc = docs[str(qa["DOCUMENT"])]
            text = str(doc.get("text", ""))
            question = str(qa.get("QUESTION_TEXT", ""))
            reference = str(qa.get("ANSWER") or "")
            start, end, span_method = gold_span(qa, text)
            try:
                supplied_start = int(qa.get("START_OFFSET", -1))
                supplied_end = int(qa.get("END_OFFSET", -1))
            except (TypeError, ValueError):
                supplied_start, supplied_end = -1, -1
            supplied_range_valid = 0 <= supplied_start < supplied_end <= len(text)
            supplied_offset_matches_reference = bool(
                supplied_range_valid and reference
                and normalize_answer(text[supplied_start:supplied_end]) == normalize_answer(reference)
            )
            passage, passage_meta = passage_context(doc, text, start, end, tokenizer)
            full_doc, full_tokens, full_truncated = fit_full_document_context(
                doc, text, question, tokenizer, args.input_token_limit
            )
            contexts = {
                "oracle_answer_passage": (passage, passage_meta, False),
                "oracle_full_gold_document": (full_doc, {"full_document_prompt_tokens": full_tokens}, full_truncated),
            }
            for condition, (context, metadata, truncated) in contexts.items():
                key = (qid, condition)
                if key in completed and not completed[key].get("generation_error"):
                    continue
                prompt = build_prompt(tokenizer, question, context)
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
                qwen_prompt_tokens = int(encoded["input_ids"].shape[1])
                if args.provider == "qwen" and qwen_prompt_tokens > args.input_token_limit:
                    raise ValueError(f"{qid}/{condition} exceeds input token limit: {qwen_prompt_tokens}")
                if args.provider == "qwen":
                    prompt_tokens = qwen_prompt_tokens
                    encoded = {name: value.to(device) for name, value in encoded.items()}
                else:
                    prompt_tokens = None
                started = time.perf_counter()
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
                        generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    answer = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True).strip()
                    generated_token_count = int(generated.shape[1] - prompt_tokens)
                normalized_ref = normalize_answer(reference)
                normalized_context = normalize_answer(context)
                result = {
                    "question_id": qid,
                    "condition": condition,
                    "provider": args.provider,
                    "model": args.model,
                    "question": question,
                    "reference_answer": reference,
                    "generated_answer": answer,
                    "token_f1": token_f1(answer, reference),
                    "normalized_exact_match": bool(normalized_ref) and normalize_answer(answer) == normalized_ref,
                    "gold_document_id": str(qa["DOCUMENT"]),
                    "span_method": span_method,
                    "supplied_offset_range_valid": supplied_range_valid,
                    "supplied_offset_matches_reference": supplied_offset_matches_reference,
                    "reference_present_in_context": bool(normalized_ref and normalized_ref in normalized_context),
                    "prompt_tokens": prompt_tokens,
                    "prompt_truncated": truncated,
                    "context_characters": len(context),
                    "context_text": context,
                    "generation_seconds": round(time.perf_counter() - started, 3),
                    "generated_token_count": generated_token_count,
                    "context_metadata": metadata,
                    "generation_error": "",
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                completed[key] = result
                if args.provider == "gemini" and args.request_delay_seconds:
                    time.sleep(args.request_delay_seconds)
                print(f"[{q_index}/{len(questions)}] {qid} {condition}: F1={result['token_f1']:.3f}; prompt={prompt_tokens}; truncated={truncated}")

    latest = list(completed.values())
    summaries: dict[str, Any] = {}
    for condition in ("oracle_answer_passage", "oracle_full_gold_document"):
        subset = [r for r in latest if r["condition"] == condition and r["reference_answer"].strip()]
        f1s = [float(r["token_f1"]) for r in subset]
        summaries[condition] = {
            "questions": len(subset),
            "mean_token_f1": statistics.mean(f1s) if f1s else None,
            "median_token_f1": statistics.median(f1s) if f1s else None,
            "normalized_exact_match_count": sum(bool(r["normalized_exact_match"]) for r in subset),
            "reference_present_in_context": sum(bool(r["reference_present_in_context"]) for r in subset),
            "supplied_offset_range_valid": sum(bool(r["supplied_offset_range_valid"]) for r in subset),
            "supplied_offset_matches_reference": sum(bool(r["supplied_offset_matches_reference"]) for r in subset),
            "truncated_documents": sum(bool(r["prompt_truncated"]) for r in subset),
            "mean_prompt_tokens": (
                statistics.mean(int(r["prompt_tokens"]) for r in subset if r.get("prompt_tokens") is not None)
                if any(r.get("prompt_tokens") is not None for r in subset) else None
            ),
        }
    by_id = {(r["question_id"], r["condition"]): r for r in latest}
    shared = sorted({qid for qid, cond in by_id if cond == "oracle_answer_passage"} & {qid for qid, cond in by_id if cond == "oracle_full_gold_document"})
    paired = [qid for qid in shared if by_id[(qid, "oracle_answer_passage")]["reference_answer"].strip()]
    deltas = [by_id[(qid, "oracle_answer_passage")]["token_f1"] - by_id[(qid, "oracle_full_gold_document")]["token_f1"] for qid in paired]
    summary = {
        "experiment": f"{args.provider} {args.model} oracle answer passage vs truncated full gold document",
        "provider": args.provider,
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "input_token_limit": args.input_token_limit,
        "context_tokenizer": args.context_tokenizer,
        "generation_prompt_tokenizer": "Qwen tokenizer" if args.provider == "qwen" else "Gemini API usage metadata",
        "diagnostic_only_uses_dev_gold_document_and_answer_offsets": True,
        "conditions": summaries,
        "paired_passage_minus_full_document_mean_token_f1": statistics.mean(deltas) if deltas else None,
        "paired_questions": len(paired),
        "limitations": ["Oracle results are not deployable retrieval performance.", "Token F1 and exact match measure overlap with one reference, not human correctness.", "Gold-document context is truncated from the beginning to the fixed reader input limit."],
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if latest:
        fields = list(dict.fromkeys(k for row in latest for k in row))
        with args.output_csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(latest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
