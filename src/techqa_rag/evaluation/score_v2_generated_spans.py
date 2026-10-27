"""Conservatively map generated text to selected-source offsets and run TechQA scorer.

Only exact normalized answer strings that occur in both the saved context and a
retrieval candidate's canonical text receive a document/span prediction. The
adapter never uses the gold document or answer offsets to choose a retrieved
prediction. Oracle mode is explicitly diagnostic and may use only the labeled
gold document for source mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

from techqa_rag.evaluation.evaluate_v2_context_generation import load_qa, normalize_answer
from techqa_rag.evaluation.evaluate_v2_oracle_contexts import load_gold_texts, locate_normalized_span


def is_answerable(row: dict[str, Any]) -> bool:
    return str(row.get("ANSWERABLE", "")).strip().upper() in {"Y", "YES", "TRUE", "1"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-qa", type=Path, default=root.parent / "TechQA/TechQA/training_and_dev/dev_Q_A.json")
    parser.add_argument("--canonical", type=Path, default=root / "data/processed/v2/canonical_technotes_v1.jsonl")
    parser.add_argument("--rankings", type=Path, default=root / "artifacts/rankings/dev_fused_rankings.pkl")
    parser.add_argument("--contexts", type=Path, default=root / "results/evaluation/v2_context_ablation.json")
    parser.add_argument("--generation", type=Path, default=root / "results/generation/v2_bm25_context_generation.jsonl")
    parser.add_argument("--condition", choices=("plain_top_chunks", "adjacent_expansion", "oracle_answer_passage", "retrieved_adjacent_expansion"), default="adjacent_expansion")
    parser.add_argument("--output-dir", type=Path, default=root / "results/evaluation/official_span_v2")
    parser.add_argument("--official-evaluator", type=Path, default=root.parent / "TechQA/TechQA/evaluation.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dev_rows = json.loads(args.dev_qa.read_text(encoding="utf-8"))
    answerable_rows = [row for row in dev_rows if is_answerable(row)]
    qa_by_id = {str(row["QUESTION_ID"]): row for row in answerable_rows}
    if len(qa_by_id) != len(answerable_rows):
        raise ValueError("Duplicate question IDs in answerable dev QA")
    with args.rankings.open("rb") as stream:
        rank_rows = pickle.load(stream)
    if len(rank_rows) != len(answerable_rows):
        raise ValueError(f"Rankings have {len(rank_rows)} rows; expected {len(answerable_rows)}")
    ranking_by_id = {str(row["QUESTION_ID"]): [str(v) for v in rank_rows[i]] for i, row in enumerate(answerable_rows)}

    selection = json.loads(args.contexts.read_text(encoding="utf-8"))["questions"]
    selection_by_id = {str(row["question_id"]): row for row in selection}
    if set(selection_by_id) != set(qa_by_id):
        raise ValueError("Saved contexts and answerable dev question IDs do not align")
    order_matches = [
        (str(row["gold_document"]) in ranking_by_id[qid][:20]) == bool(row["gold_document_in_candidate_pool"])
        for qid, row in selection_by_id.items()
    ]
    if not all(order_matches):
        raise ValueError(f"Saved rankings do not align with context rows: {sum(order_matches)}/{len(order_matches)} candidate-pool checks match")

    generations: dict[str, dict[str, Any]] = {}
    with args.generation.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row["question_id"])
            if str(row.get("condition")) == args.condition:
                generations[qid] = row
    if set(generations) != set(qa_by_id):
        raise ValueError(f"Generation rows for {args.condition}: {len(generations)}; expected {len(qa_by_id)}")

    context_key = {
        "plain_top_chunks": "plain_top_chunks_context",
        "adjacent_expansion": "adjacent_expansion_context",
        "retrieved_adjacent_expansion": "adjacent_expansion_context",
    }.get(args.condition)
    candidate_docs: dict[str, list[str]] = {}
    needed_docs: set[str] = set()
    for qid in qa_by_id:
        if args.condition == "oracle_answer_passage":
            docs = [str(qa_by_id[qid]["DOCUMENT"])]
        else:
            docs = ranking_by_id[qid][:20]
        candidate_docs[qid] = docs
        needed_docs.update(docs)
    documents = load_gold_texts(args.canonical, needed_docs)

    predictions: dict[str, list[dict[str, Any]]] = {}
    mapped_rows: list[dict[str, Any]] = []
    located_count = 0
    for qid in sorted(qa_by_id):
        generation = generations[qid]
        answer = str(generation.get("generated_answer", "")).strip()
        normalized_answer = normalize_answer(answer)
        if args.condition == "oracle_answer_passage":
            context = str(generation.get("context_text", ""))
        else:
            context = str(selection_by_id[qid][context_key])
        context_has_answer = bool(normalized_answer and normalized_answer in normalize_answer(context))
        match: tuple[str, int, int] | None = None
        if context_has_answer:
            for doc_id in candidate_docs[qid]:
                found = locate_normalized_span(str(documents[doc_id].get("text", "")), answer)
                if found:
                    match = (doc_id, found[0], found[1])
                    break
        if match:
            doc_id, start, end = match
            score = 1.0
            located_count += 1
        else:
            doc_id, start, end, score = "", -1, -1, -1.0
        predictions[qid] = [{"doc_id": doc_id, "score": score, "start_offset": start, "end_offset": end}]
        mapped_rows.append({
            "question_id": qid,
            "condition": args.condition,
            "generated_answer": answer,
            "context_contains_answer_text": context_has_answer,
            "mapped_to_source_span": match is not None,
            "predicted_document_id": doc_id,
            "predicted_start_offset": start,
            "predicted_end_offset": end,
            "gold_document_id_for_scoring_only": str(qa_by_id[qid]["DOCUMENT"]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "answerable_dev_subset.json"
    pred_path = args.output_dir / f"{args.condition}_predictions.json"
    map_path = args.output_dir / f"{args.condition}_span_mapping.csv"
    eval_path = args.output_dir / f"{args.condition}_official_metrics.json"
    data_path.write_text(json.dumps(answerable_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_path.write_text(json.dumps({"threshold": 0.0, "predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
    with map_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(mapped_rows[0]))
        writer.writeheader()
        writer.writerows(mapped_rows)
    completed = subprocess.run(
        [sys.executable, str(args.official_evaluator), str(data_path), str(pred_path), "--out-file", str(eval_path), "--top_k", "1"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    (args.output_dir / f"{args.condition}_official_stdout.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    print(json.dumps({
        "condition": args.condition,
        "answerable_questions": len(qa_by_id),
        "answer_text_locatable_in_context_and_candidate_source": located_count,
        "unlocatable_predictions": len(qa_by_id) - located_count,
        "official_metrics": json.loads(eval_path.read_text(encoding="utf-8")),
        "mapping_csv": str(map_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
