"""Score saved V2 generated answers with ID-matched BERTScore.

Each input JSONL must contain one row per (question_id, condition), with
``reference_answer`` and ``generated_answer`` fields. The script writes new,
separate CSV/JSON outputs and never edits the source generation artifacts.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = ("plain_top_chunks", "adjacent_expansion")
DEFAULT_MODEL = "roberta-large"
DEFAULT_NUM_LAYERS = 17
BOOTSTRAP_SEED = 20261026
BOOTSTRAP_SAMPLES = 20_000


def read_generation_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            qid = str(row.get("question_id", "")).strip()
            condition = str(row.get("condition", "")).strip()
            if not qid or condition not in CONDITIONS:
                raise ValueError(
                    f"Unexpected/missing question_id or condition in {path} line {line_number}: "
                    f"{qid!r}, {condition!r}"
                )
            key = (qid, condition)
            if key in rows:
                raise ValueError(f"Duplicate generation row for question/condition: {key}")
            row["question_id"] = qid
            row["condition"] = condition
            row["reference_answer"] = str(row.get("reference_answer") or "")
            row["generated_answer"] = str(row.get("generated_answer") or "")
            rows[key] = row
    if not rows:
        raise ValueError(f"No generation rows found in {path}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile for no observations")
    location = (len(ordered) - 1) * fraction
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(deltas: list[float]) -> dict[str, float] | None:
    if not deltas:
        return None
    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    # Bounded batches keep the temporary index array modest on laptops.
    batch_size = 256
    for start in range(0, BOOTSTRAP_SAMPLES, batch_size):
        count = min(batch_size, BOOTSTRAP_SAMPLES - start)
        sample_indices = rng.integers(0, len(values), size=(count, len(values)))
        boot_means[start : start + count] = values[sample_indices].mean(axis=1)
    return {
        "confidence_level": 0.95,
        "lower": float(np.quantile(boot_means, 0.025)),
        "upper": float(np.quantile(boot_means, 0.975)),
        "method": "paired percentile bootstrap over question IDs",
        "resamples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def cached_hub_revision(model_id: str) -> str | None:
    """Return the resolved ``main`` commit when the standard HF cache has it."""
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    model_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    ref_path = model_dir / "refs" / "main"
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip() or None
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore dependencies are missing. Install them with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    generation_rows = read_generation_jsonl(args.input)
    qids_by_condition: dict[str, set[str]] = {
        condition: {
            qid for qid, row_condition in generation_rows if row_condition == condition
        }
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        if not qids_by_condition[condition]:
            raise ValueError(f"Input file has no rows for condition {condition!r}")
    if qids_by_condition[CONDITIONS[0]] != qids_by_condition[CONDITIONS[1]]:
        only_plain = sorted(qids_by_condition[CONDITIONS[0]] - qids_by_condition[CONDITIONS[1]])
        only_expanded = sorted(qids_by_condition[CONDITIONS[1]] - qids_by_condition[CONDITIONS[0]])
        raise ValueError(
            "Plain and expanded conditions must contain the same question IDs. "
            f"Only plain: {only_plain[:10]}; only expanded: {only_expanded[:10]}"
        )

    for qid in sorted(qids_by_condition[CONDITIONS[0]]):
        plain = generation_rows[(qid, CONDITIONS[0])]["reference_answer"]
        expanded = generation_rows[(qid, CONDITIONS[1])]["reference_answer"]
        if plain != expanded:
            raise ValueError(f"Reference answers differ between conditions for question {qid}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scored_keys: list[tuple[str, str]] = []
    candidates: list[str] = []
    references: list[str] = []
    rows_out: dict[tuple[str, str], dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}

    # Empty references are excluded from all aggregate and paired metrics.
    # Empty model answers with a non-empty reference receive P=R=F1=0 and
    # remain in the denominator (an unanswered item is not a semantic match).
    for condition in CONDITIONS:
        counts[condition] = {
            "input_rows": 0,
            "generation_error_rows_excluded": 0,
            "empty_reference_rows_excluded": 0,
            "empty_answer_rows_scored_as_zero": 0,
            "bert_score_model_scored_rows": 0,
            "included_in_aggregate": 0,
        }

    for key, source_row in generation_rows.items():
        qid, condition = key
        counts[condition]["input_rows"] += 1
        output_row = {
            "question_id": qid,
            "condition": condition,
            "reference_answer": source_row["reference_answer"],
            "generated_answer": source_row["generated_answer"],
            "bert_score_precision": None,
            "bert_score_recall": None,
            "bert_score_f1": None,
            "score_status": "pending",
        }
        if source_row.get("generation_error"):
            counts[condition]["generation_error_rows_excluded"] += 1
            output_row["score_status"] = "excluded_generation_error"
        elif not source_row["reference_answer"].strip():
            counts[condition]["empty_reference_rows_excluded"] += 1
            output_row["score_status"] = "excluded_empty_reference"
        elif not source_row["generated_answer"].strip():
            counts[condition]["empty_answer_rows_scored_as_zero"] += 1
            counts[condition]["included_in_aggregate"] += 1
            output_row.update({
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0,
                "score_status": "empty_answer_zero_score",
            })
        else:
            counts[condition]["bert_score_model_scored_rows"] += 1
            counts[condition]["included_in_aggregate"] += 1
            output_row["score_status"] = "scored"
            scored_keys.append(key)
            candidates.append(source_row["generated_answer"])
            references.append(source_row["reference_answer"])
        rows_out[key] = output_row

    if candidates:
        precision, recall, f1 = bert_score(
            candidates,
            references,
            model_type=args.model,
            num_layers=args.num_layers,
            lang=args.language,
            device=device,
            batch_size=args.batch_size,
            idf=False,
            rescale_with_baseline=False,
            verbose=args.verbose,
        )
        for key, p_value, r_value, f_value in zip(
            scored_keys, precision.tolist(), recall.tolist(), f1.tolist()
        ):
            rows_out[key]["bert_score_precision"] = float(p_value)
            rows_out[key]["bert_score_recall"] = float(r_value)
            rows_out[key]["bert_score_f1"] = float(f_value)

    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        included = [
            rows_out[(qid, condition)]
            for qid in sorted(qids_by_condition[condition])
            if rows_out[(qid, condition)]["score_status"] in {
                "scored", "empty_answer_zero_score"
            }
        ]
        condition_summaries[condition] = {
            "questions_in_input": len(qids_by_condition[condition]),
            "questions_in_aggregate": len(included),
            "mean_precision": statistics.mean(r["bert_score_precision"] for r in included)
            if included else None,
            "mean_recall": statistics.mean(r["bert_score_recall"] for r in included)
            if included else None,
            "mean_f1": statistics.mean(r["bert_score_f1"] for r in included)
            if included else None,
        }

    paired_qids = sorted(
        qid for qid in qids_by_condition[CONDITIONS[0]]
        if rows_out[(qid, CONDITIONS[0])]["score_status"] in {"scored", "empty_answer_zero_score"}
        and rows_out[(qid, CONDITIONS[1])]["score_status"] in {"scored", "empty_answer_zero_score"}
    )
    paired_deltas = {
        metric: [
            float(rows_out[(qid, CONDITIONS[1])][f"bert_score_{metric}"])
            - float(rows_out[(qid, CONDITIONS[0])][f"bert_score_{metric}"])
            for qid in paired_qids
        ]
        for metric in ("precision", "recall", "f1")
    }
    f1_deltas = paired_deltas["f1"]
    paired_summary: dict[str, Any] = {
        "paired_question_count": len(paired_qids),
        "difference_definition": "adjacent_expansion minus plain_top_chunks",
        "mean_difference": {
            metric: statistics.mean(deltas) if deltas else None
            for metric, deltas in paired_deltas.items()
        },
        "f1_expansion_wins": sum(delta > 0 for delta in f1_deltas),
        "f1_ties": sum(delta == 0 for delta in f1_deltas),
        "f1_plain_wins": sum(delta < 0 for delta in f1_deltas),
        "f1_mean_difference_95_percent_ci": bootstrap_mean_ci(f1_deltas),
        "paired_questions": [
            {
                "question_id": qid,
                "plain_precision": rows_out[(qid, CONDITIONS[0])]["bert_score_precision"],
                "plain_recall": rows_out[(qid, CONDITIONS[0])]["bert_score_recall"],
                "plain_f1": rows_out[(qid, CONDITIONS[0])]["bert_score_f1"],
                "expanded_precision": rows_out[(qid, CONDITIONS[1])]["bert_score_precision"],
                "expanded_recall": rows_out[(qid, CONDITIONS[1])]["bert_score_recall"],
                "expanded_f1": rows_out[(qid, CONDITIONS[1])]["bert_score_f1"],
                "precision_delta": paired_deltas["precision"][index],
                "recall_delta": paired_deltas["recall"][index],
                "f1_delta": f1_deltas[index],
            }
            for index, qid in enumerate(paired_qids)
        ],
    }

    try:
        package_version = importlib.metadata.version("bert-score")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = "unknown"
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = "unknown"
    try:
        huggingface_hub_version = importlib.metadata.version("huggingface-hub")
    except importlib.metadata.PackageNotFoundError:
        huggingface_hub_version = "unknown"

    summary = {
        "experiment": "BERTScore for saved V2 plain and adjacent-expanded generated answers",
        "input_generation_jsonl": str(args.input.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "output_json": str(args.output_json.resolve()),
        "configuration": {
            "model_type": args.model,
            "model_revision": cached_hub_revision(args.model),
            "num_layers": args.num_layers,
            "language": args.language,
            "idf": False,
            "rescale_with_baseline": False,
            "batch_size": args.batch_size,
            "device": device,
            "bert_score_version": package_version,
            "transformers_version": transformers_version,
            "huggingface_hub_version": huggingface_hub_version,
            "numpy_version": numpy_version,
            "torch_version": torch.__version__,
        },
        "question_id_matching": "Rows joined and paired by question_id and condition; no row-order matching.",
        "empty_text_policy": {
            "empty_reference": "Exclude from condition means, paired deltas, and bootstrap; retain an output row with null scores.",
            "empty_answer_with_nonempty_reference": "Assign precision, recall, and F1 of 0; retain in means and paired deltas.",
            "generation_error": "Exclude from metrics and paired comparisons; retain an output row with null scores.",
        },
        "counts_by_condition": counts,
        "conditions": condition_summaries,
        "paired": paired_summary,
        "limitations": [
            "BERTScore is a contextual semantic-similarity measure against one reference, not factual correctness or evidence-grounding evaluation.",
            "The results depend on the selected model checkpoint and do not establish statistical significance beyond the reported paired bootstrap interval.",
        ],
    }

    ordered_rows = [rows_out[key] for key in sorted(rows_out)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    import csv

    fieldnames = [
        "question_id", "condition", "reference_answer", "generated_answer",
        "bert_score_precision", "bert_score_recall", "bert_score_f1", "score_status",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=project_root / "results/generation/v2_bm25_context_generation.jsonl",
        help="Saved V2 generation JSONL with both plain and expanded conditions.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--language", default="en")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_layers < 1:
        parser.error("--num-layers must be positive")
    if args.output_csv is None or args.output_json is None:
        model_tag = args.input.stem.replace("v2_bm25_context_generation", "").strip("_")
        if not model_tag:
            model_tag = "qwen"
        suffix = f"_{model_tag}"
        if args.output_csv is None:
            args.output_csv = project_root / f"results/evaluation/v2_bertscore{suffix}.csv"
        if args.output_json is None:
            args.output_json = project_root / f"results/evaluation/v2_bertscore_summary{suffix}.json"
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
