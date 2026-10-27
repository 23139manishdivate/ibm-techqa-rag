"""Build deterministic section-aware chunks with canonical source offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PARAGRAPH_SEPARATOR_RE = re.compile(r"\n[ \t]*\n+(?:[ \t]*\n)*")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+(?=[A-Z0-9\"'(])")
SPACE_NORMALIZE_RE = re.compile(r"\s+")


def paragraph_ranges(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in PARAGRAPH_SEPARATOR_RE.finditer(text):
        # Keep paragraph separators in the preceding range so no source chars vanish.
        ranges.append((start, match.end()))
        start = match.end()
    if start < len(text):
        ranges.append((start, len(text)))
    return ranges or [(0, len(text))]


def sentence_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    segment = text[start:end]
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(segment):
        boundary = match.start()
        if boundary > cursor:
            ranges.append((start + cursor, start + boundary))
        cursor = boundary
    if cursor < len(segment):
        ranges.append((start + cursor, end))
    return ranges or [(start, end)]


class OffsetTokenizer:
    """Thin adapter around a fast Hugging Face tokenizer with character offsets."""

    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Chunk building needs transformers. Install the project's requirements, "
                "then rerun with the same BGE-M3 tokenizer used for retrieval."
            ) from exc
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer with character offset mappings is required")

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        result = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids = result["input_ids"]
        offsets = [tuple(pair) for pair in result["offset_mapping"]]
        return ids, offsets


def split_long_range(
    text: str,
    start: int,
    end: int,
    tokenizer: OffsetTokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    local = text[start:end]
    token_ids, offsets = tokenizer.encode(local)
    if len(token_ids) <= max_tokens:
        return [(start, end)]
    if any(token_end <= token_start for token_start, token_end in offsets):
        raise ValueError("Tokenizer returned a non-character offset; cannot preserve exact source spans")

    windows: list[tuple[int, int]] = []
    step = max(1, max_tokens - overlap_tokens)
    for token_start in range(0, len(offsets), step):
        token_end = min(token_start + max_tokens, len(offsets))
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]
        if char_end <= char_start:
            continue
        windows.append((start + char_start, start + char_end))
        if token_end == len(offsets):
            break
    return windows


def section_units(
    text: str,
    section: dict[str, Any],
    tokenizer: OffsetTokenizer,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    section_start = section["source_char_start"]
    section_end = section["source_char_end"]
    local_text = text[section_start:section_end]
    candidates: list[tuple[int, int]] = []

    for local_start, local_end in paragraph_ranges(local_text):
        source_start = section_start + local_start
        source_end = section_start + local_end
        paragraph = text[source_start:source_end]
        if not paragraph.strip():
            continue
        token_ids, _ = tokenizer.encode(paragraph)
        if len(token_ids) <= target_tokens:
            candidates.append((source_start, source_end))
            continue

        sentences = sentence_ranges(text, source_start, source_end)
        for sentence_start, sentence_end in sentences:
            sentence_text = text[sentence_start:sentence_end]
            sentence_ids, _ = tokenizer.encode(sentence_text)
            if len(sentence_ids) <= max_tokens:
                candidates.append((sentence_start, sentence_end))
            else:
                candidates.extend(
                    split_long_range(
                        text,
                        sentence_start,
                        sentence_end,
                        tokenizer,
                        max_tokens=max_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                )

    # Pack adjacent units without crossing section boundaries or exceeding target size.
    packed: list[tuple[int, int]] = []
    if not candidates:
        return packed
    current_start, current_end = candidates[0]
    for next_start, next_end in candidates[1:]:
        proposed = text[current_start:next_end]
        proposed_count = len(tokenizer.encode(proposed)[0])
        current_count = len(tokenizer.encode(text[current_start:current_end])[0])
        if next_start <= current_end and proposed_count <= target_tokens:
            current_end = next_end
        elif next_start > current_end and current_count < target_tokens and proposed_count <= target_tokens:
            # Join neighboring paragraph text including intervening whitespace.
            current_end = next_end
        else:
            packed.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    packed.append((current_start, current_end))
    return packed


def normalize(text: str) -> str:
    return SPACE_NORMALIZE_RE.sub(" ", text).strip().casefold()


def span_covered_by_union(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    """Return whether ordered/overlapping chunk spans continuously cover [start, end)."""
    if start < 0 or end <= start:
        return False
    cursor = start
    for range_start, range_end in sorted(ranges):
        if range_end <= cursor:
            continue
        if range_start > cursor:
            return False
        cursor = max(cursor, range_end)
        if cursor >= end:
            return True
    return False


def build_chunks(
    sections_path: Path,
    output_path: Path,
    manifest_path: Path,
    tokenizer_name: str,
    target_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    dev_qa_path: Path | None = None,
) -> None:
    if target_tokens <= 0 or max_tokens < max(target_tokens, 4) or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Require 0 < target_tokens <= max_tokens and 0 <= overlap_tokens < max_tokens")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite chunk outputs; choose new versioned output and manifest paths"
        )
    tokenizer = OffsetTokenizer(tokenizer_name)
    qa_rows: list[dict[str, Any]] = []
    if dev_qa_path:
        qa_rows = json.loads(dev_qa_path.read_text(encoding="utf-8"))
    qa_by_document: dict[str, list[dict[str, Any]]] = {}
    for qa in qa_rows:
        if str(qa.get("ANSWERABLE", "")).strip().upper() in {"Y", "YES", "TRUE", "1"}:
            qa_by_document.setdefault(str(qa.get("DOCUMENT", "")), []).append(qa)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = documents_count = 0
    empty_chunk_ranges_skipped = 0
    token_counts: list[int] = []
    chunk_answer_coverage: dict[str, list[tuple[int, int]]] = {}
    section_counter: Counter[str] = Counter()
    qa_results: list[dict[str, Any]] = []
    processed_document_ids: set[str] = set()

    with sections_path.open(encoding="utf-8") as sections_file, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for doc_line in sections_file:
            doc = json.loads(doc_line)
            document_id = doc["document_id"]
            processed_document_ids.add(document_id)
            canonical_text = doc_text_from_sections(doc["sections"])
            if len(canonical_text) != doc["canonical_text_length"]:
                raise AssertionError(f"Section text length does not match source for {document_id}")
            chunks: list[dict[str, Any]] = []
            global_index = 0
            for section in doc["sections"]:
                section_counter[section["normalized_heading"]] += 1
                spans = section_units(
                    canonical_text,
                    section,
                    tokenizer,
                    target_tokens,
                    max_tokens,
                    overlap_tokens,
                )
                section_chunk_index = 0
                for original_start, original_end in spans:
                    # Text is taken from the canonical source range. Reconstructed text
                    # equals canonical text because sections are a lossless partition.
                    candidate_text = canonical_text[original_start:original_end]
                    candidate_count = len(tokenizer.encode(candidate_text)[0])
                    if candidate_count == 0 or not candidate_text.strip():
                        empty_chunk_ranges_skipped += 1
                        continue

                    final_spans = [(original_start, original_end)]
                    if candidate_count > max_tokens:
                        # Tokenizer boundary behavior can add a token when a source
                        # window is re-tokenized in isolation. Leave a small safety
                        # margin, then verify every emitted chunk against max_tokens.
                        final_spans = split_long_range(
                            canonical_text,
                            original_start,
                            original_end,
                            tokenizer,
                            max_tokens=max_tokens - 2,
                            overlap_tokens=min(overlap_tokens, max_tokens - 3),
                        )

                    for char_start, char_end in final_spans:
                        chunk_text = canonical_text[char_start:char_end]
                        token_count = len(tokenizer.encode(chunk_text)[0])
                        if not chunk_text.strip() or token_count == 0:
                            empty_chunk_ranges_skipped += 1
                            continue
                        if token_count > max_tokens:
                            raise ValueError(
                                f"Chunk exceeds max_tokens ({token_count}>{max_tokens}) "
                                f"for document {document_id} at [{char_start}:{char_end}]"
                            )
                        chunk_id = f"{document_id}:c{global_index:05d}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "section_id": section["section_id"],
                            "section_order": section["section_order"],
                            "chunk_index_in_section": section_chunk_index,
                            "chunk_index": global_index,
                            "heading": section["heading"],
                            "normalized_heading": section["normalized_heading"],
                            "source_char_start": char_start,
                            "source_char_end": char_end,
                            "text": chunk_text,
                            "token_count": token_count,
                            "tokenizer_name": tokenizer_name,
                            "source_document_hash": doc["source_document_hash"],
                            "chunker_version": "section_paragraph_sentence_v2",
                        })
                        chunk_answer_coverage.setdefault(document_id, []).append((char_start, char_end))
                        token_counts.append(token_count)
                        global_index += 1
                        section_chunk_index += 1
            if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
                raise AssertionError(f"Non-unique chunk IDs in {document_id}")
            if any(a["chunk_index"] >= b["chunk_index"] for a, b in zip(chunks, chunks[1:])):
                raise AssertionError(f"Global chunk order is not increasing in {document_id}")
            output_file.write(json.dumps({"document_id": document_id, "chunks": chunks}, ensure_ascii=False, separators=(",", ":")) + "\n")
            documents_count += 1
            chunk_count += len(chunks)
            for qa in qa_by_document.get(document_id, []):
                answer = str(qa.get("ANSWER") or "")
                try:
                    start, end = int(qa["START_OFFSET"]), int(qa["END_OFFSET"])
                except (KeyError, TypeError, ValueError):
                    start = end = -1
                valid_range = 0 <= start < end <= len(canonical_text)
                offset_text = canonical_text[start:end] if valid_range else ""
                offset_exact_match = valid_range and offset_text == answer
                normalized_answer = normalize(answer)
                ranges = chunk_answer_coverage.get(document_id, [])
                offset_contained = valid_range and any(a <= start and end <= b for a, b in ranges)
                offset_covered_by_union = valid_range and span_covered_by_union(start, end, ranges)
                normalized_answer_in_doc = bool(normalized_answer) and normalized_answer in normalize(canonical_text)
                normalized_answer_in_chunk = bool(normalized_answer) and any(
                    normalized_answer in normalize(canonical_text[a:b]) for a, b in ranges
                )
                qa_results.append({
                    "question_id": qa.get("QUESTION_ID"),
                    "document_id": document_id,
                    "reference_nonempty": bool(answer.strip()),
                    "valid_offset_range": valid_range,
                    "offset_text_exactly_matches_reference": offset_exact_match,
                    "normalized_reference_present_in_gold_document": normalized_answer_in_doc,
                    "answer_span_fully_contained_in_chunk": offset_contained,
                    "answer_span_covered_by_contiguous_chunk_union": offset_covered_by_union,
                    "normalized_reference_present_in_gold_document_chunk": normalized_answer_in_chunk,
                })

    for qa in qa_rows:
        if str(qa.get("ANSWERABLE", "")).strip().upper() not in {"Y", "YES", "TRUE", "1"}:
            continue
        document_id = str(qa.get("DOCUMENT", ""))
        if document_id not in processed_document_ids:
            qa_results.append({
                "question_id": qa.get("QUESTION_ID"),
                "document_id": document_id,
                "reference_nonempty": bool(str(qa.get("ANSWER") or "").strip()),
                "valid_offset_range": False,
                "offset_text_exactly_matches_reference": False,
                "normalized_reference_present_in_gold_document": False,
                "answer_span_fully_contained_in_chunk": False,
                "answer_span_covered_by_contiguous_chunk_union": False,
                "normalized_reference_present_in_gold_document_chunk": False,
                "failure_reason": "gold_document_missing_from_chunk_input",
            })

    chunk_answer_covered = sum(row["answer_span_fully_contained_in_chunk"] for row in qa_results)
    manifest = {
        "format_version": 1,
        "chunker_version": "section_paragraph_sentence_v2",
        "source_sections_path": str(sections_path.resolve()),
        "output_path": str(output_path.resolve()),
        "tokenizer_name": tokenizer_name,
        "target_tokens": target_tokens,
        "max_tokens": max_tokens,
        "overlap_tokens_for_long_sentence_windows": overlap_tokens,
        "documents_processed": documents_count,
        "chunks_created": chunk_count,
        "empty_chunk_ranges_skipped": empty_chunk_ranges_skipped,
        "token_count_min": min(token_counts) if token_counts else None,
        "token_count_median": sorted(token_counts)[len(token_counts) // 2] if token_counts else None,
        "token_count_p90": sorted(token_counts)[int(0.9 * (len(token_counts) - 1))] if token_counts else None,
        "token_count_max": max(token_counts) if token_counts else None,
        "section_counts": dict(section_counter.most_common()),
        "dev_answerable_rows_checked": len(qa_results),
        "dev_nonempty_reference_rows": sum(row["reference_nonempty"] for row in qa_results),
        "dev_valid_offset_rows": sum(row["valid_offset_range"] for row in qa_results),
        "dev_offset_exact_reference_matches": sum(row["offset_text_exactly_matches_reference"] for row in qa_results),
        "dev_normalized_reference_in_gold_document": sum(row["normalized_reference_present_in_gold_document"] for row in qa_results),
        "dev_answer_spans_contained": chunk_answer_covered,
        "dev_answer_span_containment_rate": chunk_answer_covered / len(qa_results) if qa_results else None,
        "dev_answer_spans_covered_by_contiguous_chunk_union": sum(
            row["answer_span_covered_by_contiguous_chunk_union"] for row in qa_results
        ),
        "dev_exact_reference_spans_contained_in_one_chunk": sum(
            row["offset_text_exactly_matches_reference"]
            and row["answer_span_fully_contained_in_chunk"]
            for row in qa_results
        ),
        "dev_exact_reference_spans_covered_by_chunk_union": sum(
            row["offset_text_exactly_matches_reference"]
            and row["answer_span_covered_by_contiguous_chunk_union"]
            for row in qa_results
        ),
        "dev_normalized_reference_in_gold_document_chunks": sum(row["normalized_reference_present_in_gold_document_chunk"] for row in qa_results),
        "dev_answer_span_diagnostics": qa_results,
        "offset_policy": "End-exclusive Python character offsets into canonical document text.",
        "chunk_text_policy": "Exact canonical source substring; section heading retained in metadata.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def doc_text_from_sections(sections: list[dict[str, Any]]) -> str:
    """Reconstruct the canonical source from lossless ordered section spans."""
    return "".join(section["text"] for section in sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tokenizer", default="BAAI/bge-m3")
    parser.add_argument("--target-tokens", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--dev-qa", type=Path)
    args = parser.parse_args()
    build_chunks(
        args.sections,
        args.output,
        args.manifest,
        args.tokenizer,
        args.target_tokens,
        args.max_tokens,
        args.overlap_tokens,
        args.dev_qa,
    )
    print(f"Chunks: {args.output.resolve()}")
    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
