"""Compare plain chunk selection with adjacent expansion on fixed document candidates.

This is a local diagnostic, not a replacement for the BGE-M3 dense/hybrid
retrieval benchmark. It holds the saved document-level candidate rankings fixed,
uses BM25 only to rank v2 chunks within those candidate documents, then compares
two context selectors on the same ranked chunk list and exact Qwen input budget.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Callable

from techqa_rag.context_selection.adjacent_chunks import (
    expand_adjacent_chunks,
    merge_contiguous_chunks,
    render_passages,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")
SYSTEM_PROMPT = (
    "Answer the question using only the supplied IBM technical documentation. "
    "Be concise and preserve exact technical identifiers. "
    "Do not invent fixes, APAR numbers, or commands. "
    "If the evidence does not support an answer, say so."
)


def tokenize(text: str) -> list[str]:
    return [term.casefold() for term in TOKEN_RE.findall(text)]


def bm25_scores(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Small per-question BM25 implementation; corpus is the fixed doc candidate set."""
    tokenized = [tokenize(text) for text in documents]
    n_docs = len(tokenized)
    if not n_docs:
        return []
    doc_freq: collections.Counter[str] = collections.Counter()
    term_freqs: list[collections.Counter[str]] = []
    lengths: list[int] = []
    for terms in tokenized:
        frequencies = collections.Counter(terms)
        term_freqs.append(frequencies)
        lengths.append(len(terms))
        doc_freq.update(frequencies.keys())
    average_length = sum(lengths) / n_docs or 1.0
    query_terms = tokenize(query)
    idf = {
        term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }
    scores: list[float] = []
    for frequencies, doc_length in zip(term_freqs, lengths):
        score = 0.0
        norm = k1 * (1.0 - b + b * doc_length / average_length)
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if tf:
                score += idf[term] * tf * (k1 + 1.0) / (tf + norm)
        scores.append(score)
    return scores


def span_covered(
    start: int,
    end: int,
    chunks: list[dict[str, Any]],
    *,
    document_id: str | None = None,
) -> bool:
    cursor = start
    eligible = [
        chunk for chunk in chunks
        if document_id is None or str(chunk.get("document_id", "")) == document_id
    ]
    for chunk in sorted(eligible, key=lambda row: (row["source_char_start"], row["source_char_end"])):
        chunk_start = int(chunk["source_char_start"])
        chunk_end = int(chunk["source_char_end"])
        if chunk_end <= cursor:
            continue
        if chunk_start > cursor:
            return False
        cursor = max(cursor, chunk_end)
        if cursor >= end:
            return True
    return cursor >= end


def normalized(text: str) -> str:
    return " ".join(str(text).casefold().split())


def load_jsonl_by_id(path: Path, *, nested_chunks: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if nested_chunks:
                output[str(row["document_id"])] = row["chunks"]
            else:
                output[str(row["document_id"])] = row
    return output


def load_qwen_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to run this context-budget comparison") from exc
    return AutoTokenizer.from_pretrained(model_name)


def prepare_bge_embeddings(
    chunks_by_document: dict[str, list[dict[str, Any]]],
    candidate_documents: set[str],
    *,
    model_name: str,
    batch_size: int,
    cache_dir: Path,
) -> tuple[Any, Any, dict[str, int], dict[str, Any]]:
    """Encode each unique v2 chunk in the fixed document candidate union once."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    rows = [
        chunk
        for document_id in sorted(candidate_documents)
        for chunk in chunks_by_document.get(document_id, [])
    ]
    chunk_ids = [str(row["chunk_id"]) for row in rows]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Candidate chunk IDs are not unique")
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / "bge_m3_chunk_embeddings.npy"
    ids_path = cache_dir / "bge_m3_chunk_ids.json"
    metadata_path = cache_dir / "bge_m3_embedding_manifest.json"
    metadata = {
        "embedding_model": model_name,
        "embedding_model_dtype": "float16" if torch.cuda.is_available() else "float32",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "candidate_documents": len(candidate_documents),
        "candidate_chunks": len(rows),
        "chunker_version": "section_paragraph_sentence_v2",
    }

    model_kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
    model = SentenceTransformer(
        model_name,
        device=metadata["device"],
        model_kwargs=model_kwargs,
    )
    index_by_chunk_id = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}

    cache_valid = False
    if embeddings_path.exists() and ids_path.exists() and metadata_path.exists():
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_valid = cached_ids == chunk_ids and all(
            cached_metadata.get(key) == metadata[key]
            for key in ("embedding_model", "candidate_documents", "candidate_chunks", "chunker_version")
        )
        if cache_valid:
            cached_embeddings = np.load(embeddings_path, mmap_mode="r")
            expected_dtype = np.dtype(cached_metadata.get("embedding_model_dtype", ""))
            cache_valid = (
                cached_embeddings.shape == (len(rows), 1024)
                and cached_embeddings.dtype == expected_dtype
                and cached_embeddings.dtype in (np.dtype("float16"), np.dtype("float32"))
            )
            if cache_valid:
                embeddings = cached_embeddings
                metadata = cached_metadata
    if cache_valid:
        print(f"Loaded cached v2 candidate embeddings: {embeddings.shape}")
    else:
        embeddings = model.encode(
            [str(row["text"]) for row in rows],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float16 if torch.cuda.is_available() else np.float32)
        np.save(embeddings_path, embeddings)
        ids_path.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")
        metadata["embedding_batch_size"] = batch_size
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved v2 candidate embeddings: {embeddings.shape} ({embeddings.dtype})")
    return model, embeddings, index_by_chunk_id, metadata


def prompt_token_counter(tokenizer: Any, question: str) -> Callable[[str], int]:
    def count(context: str) -> int:
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
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    return count


def baseline_select(
    ranked_chunks: list[dict[str, Any]],
    *,
    max_chunks: int,
    max_prompt_tokens: int,
    token_counter: Callable[[str], int],
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if len(selected) >= max_chunks:
            break
        candidate = dict(chunk)
        candidate["selection_role"] = "retrieved_seed"
        candidate["seed_rank"] = rank
        trial = selected + [candidate]
        rendered = render_passages(merge_contiguous_chunks(trial))
        if token_counter(rendered) <= max_prompt_tokens:
            selected.append(candidate)
    passages = merge_contiguous_chunks(selected)
    context = render_passages(passages)
    return {
        "chunks": selected,
        "passages": passages,
        "context": context,
        "prompt_tokens": token_counter(context),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    chunks_by_document = load_jsonl_by_id(args.chunks, nested_chunks=True)
    canonical = load_jsonl_by_id(args.canonical)
    qa_rows = json.loads(args.dev_qa.read_text(encoding="utf-8"))
    qa_rows = [
        row for row in qa_rows
        if str(row.get("ANSWERABLE", "")).strip().upper() in {"Y", "YES", "TRUE", "1"}
    ]
    rankings = pickle.loads(args.rankings.read_bytes())
    if len(rankings) != len(qa_rows):
        raise ValueError(f"Expected {len(qa_rows)} ranked rows, got {len(rankings)}")
    tokenizer = load_qwen_tokenizer(args.reader_model)
    bge_model = None
    chunk_embeddings = None
    embedding_index: dict[str, int] = {}
    embedding_metadata: dict[str, Any] = {}
    query_embeddings = None
    retrieval_queries = [
        " ".join([str(qa.get("QUESTION_TITLE", "")), str(qa.get("QUESTION_TEXT", ""))])
        for qa in qa_rows
    ]
    if args.passage_ranker == "bge-m3":
        import numpy as np
        import torch

        candidate_document_union = {
            str(doc_id)
            for ranking in rankings
            for doc_id in ranking[: args.candidate_documents]
        }
        bge_model, chunk_embeddings, embedding_index, embedding_metadata = prepare_bge_embeddings(
            chunks_by_document,
            candidate_document_union,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            cache_dir=args.embedding_cache_dir,
        )
        query_embeddings = bge_model.encode(
            retrieval_queries,
            batch_size=args.query_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        if torch.cuda.is_available():
            del bge_model
            bge_model = None
            torch.cuda.empty_cache()

    detail_rows: list[dict[str, Any]] = []
    for q_index, qa in enumerate(qa_rows):
        question_id = str(qa["QUESTION_ID"])
        gold_doc = str(qa["DOCUMENT"])
        doc_ranked = [str(doc_id) for doc_id in rankings[q_index]]
        candidate_doc_ids = doc_ranked[: args.candidate_documents]
        # Match the saved Qwen oracle-generation notebook, which supplies
        # QUESTION_TEXT to the reader and uses title + body only for retrieval.
        question = str(qa.get("QUESTION_TEXT", ""))
        candidate_chunks: list[dict[str, Any]] = []
        doc_rank_map = {doc_id: i + 1 for i, doc_id in enumerate(candidate_doc_ids)}
        for doc_id in candidate_doc_ids:
            metadata = canonical.get(doc_id, {})
            for source_chunk in chunks_by_document.get(doc_id, []):
                chunk = dict(source_chunk)
                chunk["title"] = metadata.get("title", "")
                chunk["document_rank"] = doc_rank_map[doc_id]
                candidate_chunks.append(chunk)

        query_text = retrieval_queries[q_index]
        if args.passage_ranker == "bm25":
            bm25_corpus = [
                f"{chunk.get('title', '')} {chunk.get('text', '')}"
                for chunk in candidate_chunks
            ]
            scores = bm25_scores(query_text, bm25_corpus)
        else:
            query_vector = query_embeddings[q_index]
            candidate_indices = [embedding_index[str(chunk["chunk_id"])] for chunk in candidate_chunks]
            scores = chunk_embeddings[candidate_indices].astype("float32") @ query_vector
        for chunk, score in zip(candidate_chunks, scores):
            chunk["score"] = score
        ranked_chunks = sorted(
            candidate_chunks,
            key=lambda chunk: (
                -float(chunk["score"]),
                int(chunk.get("document_rank", 10**9)),
                int(chunk.get("chunk_index", 10**9)),
            ),
        )
        for rank, chunk in enumerate(ranked_chunks, start=1):
            chunk["rank"] = rank

        token_counter = prompt_token_counter(tokenizer, question)
        plain = baseline_select(
            ranked_chunks,
            max_chunks=args.seed_chunks,
            max_prompt_tokens=args.max_prompt_tokens,
            token_counter=token_counter,
        )
        expanded = expand_adjacent_chunks(
            ranked_chunks[: args.seed_chunks],
            candidate_chunks,
            max_context_tokens=args.max_prompt_tokens,
            token_counter=token_counter,
            neighbor_radius=args.neighbor_radius,
        )
        expanded_prompt_tokens = token_counter(expanded["context"])

        try:
            answer_start, answer_end = int(qa["START_OFFSET"]), int(qa["END_OFFSET"])
        except (KeyError, TypeError, ValueError):
            answer_start, answer_end = -1, -1
        source = str(canonical.get(gold_doc, {}).get("text", ""))
        valid_span = 0 <= answer_start < answer_end <= len(source)
        offset_span = source[answer_start:answer_end] if valid_span else ""
        answer = str(qa.get("ANSWER") or "")
        candidate_pool_hits = [
            chunk for chunk in candidate_chunks
            if valid_span
            and int(chunk["source_char_start"]) < answer_end
            and int(chunk["source_char_end"]) > answer_start
        ]

        def summarize_context(selection: dict[str, Any]) -> dict[str, Any]:
            selected_chunks = selection["chunks"]
            passages = selection["passages"]
            passage_text = "\n".join(str(passage["text"]) for passage in passages)
            gold_passage_text = "\n".join(
                str(passage["text"])
                for passage in passages
                if str(passage.get("document_id", "")) == gold_doc
            )
            context_doc_ids = {str(chunk.get("document_id", "")) for chunk in selected_chunks}
            return {
                "gold_document_selected": gold_doc in context_doc_ids,
                "offset_span_in_selected_context": valid_span and span_covered(
                    answer_start,
                    answer_end,
                    [
                        {
                            "document_id": passage["document_id"],
                            "source_char_start": passage["source_char_start"],
                            "source_char_end": passage["source_char_end"],
                        }
                        for passage in passages
                    ],
                    document_id=gold_doc,
                ),
                "normalized_reference_in_selected_context": bool(normalized(answer))
                and normalized(answer) in normalized(passage_text),
                "normalized_reference_in_gold_context": bool(normalized(answer))
                and normalized(answer) in normalized(gold_passage_text),
                "selected_chunk_count": len(selected_chunks),
                "adjacent_chunk_count": sum(
                    chunk.get("selection_role") == "adjacent_expansion"
                    for chunk in selected_chunks
                ),
                "prompt_tokens": selection["prompt_tokens"]
                if "prompt_tokens" in selection
                else token_counter(selection["context"]),
                "context_characters": len(selection["context"]),
                "selected_chunk_ids": [str(chunk["chunk_id"]) for chunk in selected_chunks],
            }

        # Passage retrieval is measured separately from final context selection.
        passage_cutoffs: dict[str, bool] = {}
        for cutoff in (1, 5, 10, 20):
            passage_cutoffs[f"offset_span_covered_top_{cutoff}_chunks"] = valid_span and span_covered(
                answer_start, answer_end, ranked_chunks[:cutoff], document_id=gold_doc
            )
        detail_rows.append({
            "question_id": question_id,
            "gold_document": gold_doc,
            "candidate_document_count": len(candidate_doc_ids),
            "candidate_chunk_count": len(candidate_chunks),
            "gold_document_in_candidate_pool": gold_doc in candidate_doc_ids,
            "answer_span_covered_by_candidate_pool": valid_span and span_covered(
                answer_start, answer_end, candidate_chunks, document_id=gold_doc
            ),
            "answer_span_intersecting_candidate_chunk_count": len(candidate_pool_hits),
            **passage_cutoffs,
            "reference_matches_offset_span": normalized(answer) == normalized(offset_span),
            "plain_top_chunks": summarize_context(plain),
            "plain_top_chunks_context": plain["context"],
            "adjacent_expansion": summarize_context({
                **expanded,
                "prompt_tokens": expanded_prompt_tokens,
            }),
            "adjacent_expansion_context": expanded["context"],
        })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "v2 fixed document candidates; BM25 passage ranking; plain vs adjacent selection",
        "reader_model_tokenizer": args.reader_model,
        "prompt_token_budget": args.max_prompt_tokens,
        "candidate_documents_per_question": args.candidate_documents,
        "plain_selector_max_seed_chunks": args.seed_chunks,
        "adjacent_selector_neighbor_radius": args.neighbor_radius,
        "passage_ranker": (
            "Okapi BM25 over chunk text plus technote title, scoped to fixed saved document candidates"
            if args.passage_ranker == "bm25"
            else "BGE-M3 normalized dense similarity over v2 chunk text, scoped to fixed saved document candidates"
        ),
        "embedding_metadata": embedding_metadata,
        "document_rankings_source": str(args.rankings),
        "limitations": [
            "Document candidates come from a saved legacy fused ranking over the prior chunk/index version.",
            "BM25 is a diagnostic passage ranker and is not the existing dense/hybrid chunk ranker.",
            "This run measures evidence retrieval and context selection; it does not generate answers.",
        ],
        "questions": detail_rows,
    }
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for condition in ("plain_top_chunks", "adjacent_expansion"):
        summary: dict[str, Any] = {"condition": condition, "questions": len(detail_rows)}
        for key in (
            "gold_document_selected",
            "offset_span_in_selected_context",
            "normalized_reference_in_selected_context",
            "normalized_reference_in_gold_context",
        ):
            summary[key] = sum(bool(row[condition][key]) for row in detail_rows)
        summary["mean_selected_chunks"] = sum(row[condition]["selected_chunk_count"] for row in detail_rows) / len(detail_rows)
        summary["mean_adjacent_chunks"] = sum(row[condition]["adjacent_chunk_count"] for row in detail_rows) / len(detail_rows)
        summary["mean_prompt_tokens"] = sum(row[condition]["prompt_tokens"] for row in detail_rows) / len(detail_rows)
        summary["median_prompt_tokens"] = sorted(row[condition]["prompt_tokens"] for row in detail_rows)[len(detail_rows) // 2]
        summary["questions_at_prompt_budget"] = sum(row[condition]["prompt_tokens"] >= args.max_prompt_tokens for row in detail_rows)
        for cutoff in (1, 5, 10, 20):
            name = f"offset_span_covered_top_{cutoff}_chunks"
            summary[name] = sum(bool(row[name]) for row in detail_rows)
        summary_rows.append(summary)
    qa_pool_covered = sum(bool(row["answer_span_covered_by_candidate_pool"]) for row in detail_rows)
    qa_gold_doc = sum(bool(row["gold_document_in_candidate_pool"]) for row in detail_rows)
    output["summary"] = {
        "candidate_pool_gold_document_recall": qa_gold_doc / len(detail_rows),
        "candidate_pool_answer_span_coverage": qa_pool_covered / len(detail_rows),
        "conditions": summary_rows,
    }
    # Rewrite once with the computed aggregate included.
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    import csv

    flattened_rows = []
    for row in detail_rows:
        flat = {k: v for k, v in row.items() if k not in {"plain_top_chunks", "adjacent_expansion"}}
        for condition in ("plain_top_chunks", "adjacent_expansion"):
            for key, value in row[condition].items():
                flat[f"{condition}_{key}"] = (
                    json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                )
        flattened_rows.append(flat)
    with args.output_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(flattened_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flattened_rows)
    return output


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=project_root / "data/processed/v2/technote_chunks_v2.jsonl")
    parser.add_argument("--canonical", type=Path, default=project_root / "data/processed/v2/canonical_technotes_v1.jsonl")
    parser.add_argument("--dev-qa", type=Path, default=project_root.parent / "TechQA/TechQA/training_and_dev/dev_Q_A.json")
    parser.add_argument("--rankings", type=Path, default=project_root.parent / "TechQA/TechQA/Kaggle files/dev_fused_rankings.pkl")
    parser.add_argument("--reader-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--passage-ranker", choices=("bm25", "bge-m3"), default="bm25")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--embedding-cache-dir", type=Path, default=project_root / "artifacts/embeddings/v2_candidate_documents_top20")
    parser.add_argument("--candidate-documents", type=int, default=20)
    parser.add_argument("--seed-chunks", type=int, default=5)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-json", type=Path, default=project_root / "results/evaluation/v2_context_ablation.json")
    parser.add_argument("--output-csv", type=Path, default=project_root / "results/evaluation/v2_context_ablation.csv")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    metrics = run(cli_args)
    print(json.dumps(metrics["summary"], indent=2))
    print(f"JSON: {cli_args.output_json.resolve()}")
    print(f"CSV: {cli_args.output_csv.resolve()}")
