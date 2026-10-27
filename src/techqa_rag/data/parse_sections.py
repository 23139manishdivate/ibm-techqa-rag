"""Split canonical TechQA text into conservative, exactly recoverable sections."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# This list is intentionally explicit. A phrase appearing in prose is not enough
# to create a boundary; the heading must occupy its own line.
HEADING_NAMES = (
    "TECHNOTE (TROUBLESHOOTING)",
    "PROBLEM(ABSTRACT)",
    "PROBLEM SUMMARY",
    "QUESTION",
    "ANSWER",
    "SYMPTOM",
    "CAUSE",
    "ENVIRONMENT",
    "RESOLVING THE PROBLEM",
    "LOCAL FIX",
    "TEMPORARY FIX",
    "ERROR DESCRIPTION",
    "ERROR MESSAGE",
    "BACKGROUND",
    "SOLUTION",
    "APAR INFORMATION",
    "RELATED INFORMATION",
    "DIAGNOSING THE PROBLEM",
    "PROBLEM CONCLUSION",
)


def _heading_pattern() -> re.Pattern[str]:
    alternatives = "|".join(re.escape(name) for name in sorted(HEADING_NAMES, key=len, reverse=True))
    return re.compile(rf"^\s*({alternatives})\s*:?[ \t]*$", re.IGNORECASE)


HEADING_RE = _heading_pattern()


def parse_sections(text: str) -> list[dict[str, Any]]:
    """Return ordered sections whose concatenated text reconstructs input exactly."""
    boundaries: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        match = HEADING_RE.fullmatch(line_body)
        if match:
            heading_start = cursor
            heading_end = cursor + len(line_body)
            boundaries.append((heading_start, heading_end, match.group(1).strip()))
        cursor += len(line)

    if not boundaries:
        return [{
            "section_order": 0,
            "section_id": "section-0000",
            "heading": None,
            "normalized_heading": "UNSECTIONED",
            "source_char_start": 0,
            "source_char_end": len(text),
            "heading_char_start": None,
            "heading_char_end": None,
            "text": text,
        }]

    spans: list[tuple[int, int, str | None, int | None, int | None]] = []
    if boundaries[0][0] > 0:
        spans.append((0, boundaries[0][0], None, None, None))
    for i, (heading_start, heading_end, heading) in enumerate(boundaries):
        section_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        spans.append((heading_start, section_end, heading, heading_start, heading_end))

    sections: list[dict[str, Any]] = []
    for order, (start, end, heading, heading_start, heading_end) in enumerate(spans):
        normalized = re.sub(r"[^A-Z0-9]+", " ", heading.upper()).strip() if heading else "UNSECTIONED"
        sections.append({
            "section_order": order,
            "section_id": f"section-{order:04d}",
            "heading": heading,
            "normalized_heading": normalized,
            "source_char_start": start,
            "source_char_end": end,
            "heading_char_start": heading_start,
            "heading_char_end": heading_end,
            "text": text[start:end],
        })

    if "".join(section["text"] for section in sections) != text:
        raise AssertionError("Section text does not reconstruct canonical source text")
    previous_end = 0
    for section in sections:
        if section["source_char_start"] != previous_end:
            raise AssertionError("Section spans contain a gap or overlap")
        previous_end = section["source_char_end"]
    if previous_end != len(text):
        raise AssertionError("Section spans do not cover the complete canonical text")
    return sections


def parse_jsonl(source_path: Path, output_path: Path, manifest_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heading_counts: Counter[str] = Counter()
    documents = sections_total = characters_total = 0
    with source_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            record = json.loads(line)
            text = record["text"]
            sections = parse_sections(text)
            for section in sections:
                heading_counts[section["normalized_heading"]] += 1
            row = {
                "document_id": record["document_id"],
                "source_document_hash": record["source_document_hash"],
                "canonical_text_hash": record["canonical_text_hash"],
                "canonical_text_length": len(text),
                "sections": sections,
            }
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            documents += 1
            sections_total += len(sections)
            characters_total += len(text)

    manifest = {
        "format_version": 1,
        "source_path": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
        "documents_processed": documents,
        "sections_created": sections_total,
        "canonical_characters_covered": characters_total,
        "heading_line_pattern": "Explicit known heading occupying its own line; case-insensitive.",
        "unknown_text_policy": "Retained as UNSECTIONED; no source text is discarded.",
        "heading_counts": dict(heading_counts.most_common()),
        "heading_dictionary": list(HEADING_NAMES),
        "output_sha256": _file_sha256(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Canonical JSONL from Phase B1")
    parser.add_argument("--output", required=True, type=Path, help="New section JSONL output path")
    parser.add_argument("--manifest", required=True, type=Path, help="New section audit manifest path")
    args = parser.parse_args()
    parse_jsonl(args.source, args.output, args.manifest)
    print(f"Sections: {args.output.resolve()}")
    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
