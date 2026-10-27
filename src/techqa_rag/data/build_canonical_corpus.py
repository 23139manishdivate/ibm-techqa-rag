"""Create a lossless, versioned canonical JSONL view of the TechQA technote corpus.

The source ``text`` field is preserved verbatim because TechQA answer offsets
refer to it. HTML ``content`` is retained separately for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


KEYWORD_DIV_RE = re.compile(
    r'<div\b(?=[^>]*\bid\s*=\s*["\']dBlue-keyword["\'])[^>]*>(.*?)</div\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic key ordering."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_keyword_terms(content: str) -> str:
    """Return the hidden keyword div's raw inner text as a separate field."""
    match = KEYWORD_DIV_RE.search(content)
    if not match:
        return ""
    inner = TAG_RE.sub(" ", match.group(1))
    # Decode entities without changing the canonical source fields.
    from html import unescape

    return WHITESPACE_RE.sub(" ", unescape(inner)).strip()


def load_source(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source_file:
        payload = json.load(source_file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected the technote corpus to be a JSON object, got {type(payload).__name__}")
    return payload


def build_canonical_records(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    source_key_mismatches = 0
    source_document_id_mismatches = 0
    missing_metadata = 0
    missing_required_fields: Counter[str] = Counter()
    duplicate_document_ids: Counter[str] = Counter()
    keyword_documents = 0
    content_text_raw_differences = 0

    for source_key, document in source.items():
        if not isinstance(document, dict):
            errors.append({"source_key": str(source_key), "error": "document value is not an object"})
            continue

        document_id = document.get("id")
        if not isinstance(document_id, str) or not document_id.strip():
            errors.append({"source_key": str(source_key), "error": "missing or invalid document id"})
            continue
        duplicate_document_ids[document_id] += 1

        if str(source_key) != document_id:
            source_key_mismatches += 1
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            missing_metadata += 1
        metadata_source_id = metadata.get("sourceDocumentId")
        if metadata_source_id is not None and str(metadata_source_id) != document_id:
            source_document_id_mismatches += 1

        for field in ("title", "text", "content"):
            if not isinstance(document.get(field), str):
                missing_required_fields[field] += 1
        title = document.get("title") if isinstance(document.get("title"), str) else ""
        text = document.get("text") if isinstance(document.get("text"), str) else ""
        content = document.get("content") if isinstance(document.get("content"), str) else ""
        if text != content:
            content_text_raw_differences += 1

        keywords = extract_keyword_terms(content)
        if keywords:
            keyword_documents += 1

        # Include all source metadata, but do not synthesize or rewrite source text.
        record = {
            "document_id": document_id,
            "title": title,
            "text": text,
            "content": content,
            "metadata": metadata,
            "keyword_terms": keywords,
            "source_key": str(source_key),
            "metadata_source_document_id": metadata_source_id,
            "source_document_hash": stable_hash(
                {
                    "document_id": document_id,
                    "title": title,
                    "text": text,
                    "content": content,
                    "metadata": metadata,
                }
            ),
            "canonical_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "canonical_text_length": len(text),
        }
        records.append(record)

    duplicate_ids = sorted(doc_id for doc_id, count in duplicate_document_ids.items() if count > 1)
    manifest = {
        "format_version": 1,
        "source_document_count": len(source),
        "canonical_record_count": len(records),
        "invalid_document_count": len(errors),
        "source_key_id_mismatch_count": source_key_mismatches,
        "metadata_source_id_mismatch_count": source_document_id_mismatches,
        "missing_metadata_count": missing_metadata,
        "missing_required_field_counts": dict(missing_required_fields),
        "duplicate_document_id_count": len(duplicate_ids),
        "duplicate_document_ids_sample": duplicate_ids[:50],
        "documents_with_hidden_keyword_terms": keyword_documents,
        "raw_content_text_differ_count": content_text_raw_differences,
        "canonical_text_policy": "Exact source document text field; never HTML-normalized or rewritten.",
        "keyword_policy": "Hidden dBlue-keyword inner text extracted to a separate field; never appended to canonical text.",
        "invalid_document_examples": errors[:50],
        "record_hash_policy": "SHA-256 over canonical JSON serialization of the source fields.",
    }
    return records, manifest


def write_outputs(source_path: Path, output_dir: Path) -> tuple[Path, Path]:
    source = load_source(source_path)
    records, manifest = build_canonical_records(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "canonical_technotes_v1.jsonl"
    manifest_path = output_dir / "canonical_technotes_v1_manifest.json"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")

    manifest["source_path"] = str(source_path.resolve())
    manifest["source_file_size_bytes"] = source_path.stat().st_size
    manifest["source_file_sha256"] = _file_sha256(source_path)
    manifest["output_path"] = str(jsonl_path.resolve())
    manifest["output_record_count"] = len(records)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl_path, manifest_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Source training/dev technote JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="New versioned output directory")
    args = parser.parse_args()
    jsonl_path, manifest_path = write_outputs(args.source, args.output_dir)
    print(f"Canonical JSONL: {jsonl_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
