"""Context selection that expands ranked chunks to adjacent source evidence.

The caller supplies a generation-model token counter so the rendered context,
including labels and separators, stays within its actual input budget.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence


Chunk = Mapping[str, Any]
TokenCounter = Callable[[str], int]


def _order_key(chunk: Chunk) -> tuple[int, int, int]:
    return (
        int(chunk.get("section_order", 0)),
        int(chunk.get("chunk_index_in_section", chunk.get("chunk_index", 0))),
        int(chunk.get("source_char_start", 0)),
    )


def _group_key(chunk: Chunk) -> tuple[str, str]:
    section = chunk.get("section_id")
    if section is None:
        section = str(chunk.get("section_order", chunk.get("section", "UNSECTIONED")))
    return str(chunk.get("document_id", "")), str(section)


def _rank_key(chunk: Chunk) -> tuple[float, int]:
    """Lower rank is better; score is used only when rank is absent."""
    if chunk.get("rank") is not None:
        return (float(chunk["rank"]), int(chunk.get("source_char_start", 0)))
    return (-float(chunk.get("score", chunk.get("hybrid_score", 0.0))), int(chunk.get("source_char_start", 0)))


def _ordered_inventory(chunks: Iterable[Chunk]) -> tuple[dict[str, Chunk], dict[str, list[Chunk]]]:
    by_id: dict[str, Chunk] = {}
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id:
            raise ValueError("Every chunk must have a stable chunk_id")
        if chunk_id in by_id:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        by_id[chunk_id] = chunk
        by_document[str(chunk.get("document_id", ""))].append(chunk)
    for document_chunks in by_document.values():
        document_chunks.sort(key=_order_key)
    return by_id, by_document


def expand_adjacent_chunks(
    ranked_chunks: Sequence[Chunk],
    all_chunks: Iterable[Chunk],
    *,
    max_context_tokens: int,
    token_counter: TokenCounter,
    neighbor_radius: int = 1,
) -> dict[str, Any]:
    """Select ranked chunks plus nearby same-section chunks under a rendered budget.

    Ranked seeds are considered from best to worst. Each seed is attempted first;
    then same-section neighbors are attempted by distance. Context cost is measured
    after merging contiguous source spans and rendering headings, using the caller's
    generation tokenizer. A neighbor can be included even if it was not retrieved
    into the original candidate list.
    """
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    if neighbor_radius < 0:
        raise ValueError("neighbor_radius cannot be negative")

    by_id, by_document = _ordered_inventory(all_chunks)
    if not ranked_chunks:
        return {"chunks": [], "passages": [], "context": "", "context_tokens": 0}

    seeds = sorted(ranked_chunks, key=_rank_key)
    selected: dict[str, dict[str, Any]] = {}
    ordered_by_document = {
        key: {str(c["chunk_id"]): i for i, c in enumerate(values)}
        for key, values in by_document.items()
    }

    def context_cost(chunks: Sequence[Mapping[str, Any]]) -> tuple[int, str, list[dict[str, Any]]]:
        passages = merge_contiguous_chunks(chunks)
        context = render_passages(passages)
        return token_counter(context), context, passages

    for seed in seeds:
        seed_id = str(seed.get("chunk_id", ""))
        if seed_id not in by_id:
            # Keep the ranked row usable when its full inventory row is absent.
            seed_full = dict(seed)
        else:
            seed_full = dict(by_id[seed_id])
            seed_full.update({k: v for k, v in seed.items() if k in {"rank", "score", "hybrid_score"}})
        document_id = str(seed_full.get("document_id", ""))
        document_chunks = by_document.get(document_id, [])
        position_map = ordered_by_document.get(document_id, {})
        position = position_map.get(seed_id)
        neighbors: list[tuple[int, Chunk]] = []
        if position is not None:
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                index = position + offset
                if not 0 <= index < len(document_chunks):
                    continue
                path = document_chunks[min(index, position):max(index, position) + 1]
                section_boundary_is_contiguous = all(
                    _group_key(left) == _group_key(right)
                    or int(left["source_char_end"]) >= int(right["source_char_start"])
                    for left, right in zip(path, path[1:])
                )
                if section_boundary_is_contiguous:
                    neighbors.append((abs(offset), document_chunks[index]))
        else:
            neighbors = [(0, seed_full)]
        neighbors.sort(key=lambda item: (item[0], _order_key(item[1])))

        seed_was_selected = seed_id in selected
        for distance, candidate in neighbors:
            # An expansion without its retrieved anchor would add context that
            # was never itself ranked as relevant.
            if distance > 0 and not seed_was_selected:
                continue
            candidate_id = str(candidate["chunk_id"])
            if candidate_id in selected:
                continue
            candidate_row = dict(candidate)
            if distance == 0:
                candidate_row["selection_role"] = "retrieved_seed"
                candidate_row["seed_rank"] = _rank_key(seed)[0]
            else:
                candidate_row["selection_role"] = "adjacent_expansion"
                candidate_row["expanded_from_chunk_id"] = seed_id
                candidate_row["seed_rank"] = _rank_key(seed)[0]
            trial = list(selected.values()) + [candidate_row]
            cost, _, _ = context_cost(trial)
            if cost <= max_context_tokens:
                selected[candidate_id] = candidate_row
                if candidate_id == seed_id:
                    seed_was_selected = True

    final_chunks = list(selected.values())
    context_tokens, context, passages = context_cost(final_chunks)
    return {
        "chunks": sorted(final_chunks, key=lambda c: (_group_key(c), _order_key(c))),
        "passages": passages,
        "context": context,
        "context_tokens": context_tokens,
        "max_context_tokens": max_context_tokens,
        "neighbor_radius": neighbor_radius,
    }


def merge_contiguous_chunks(chunks: Sequence[Chunk]) -> list[dict[str, Any]]:
    """Join selected chunks only when their exact source spans touch or overlap."""
    ordered = sorted(chunks, key=lambda c: (_group_key(c), _order_key(c)))
    passages: list[dict[str, Any]] = []
    for chunk in ordered:
        start = int(chunk["source_char_start"])
        end = int(chunk["source_char_end"])
        text = str(chunk["text"])
        if end < start or len(text) != end - start:
            raise ValueError(f"Chunk text length does not match source offsets: {chunk.get('chunk_id')}")
        group = _group_key(chunk)
        if (
            passages
            and passages[-1]["document_id"] == str(chunk.get("document_id", ""))
            and start <= passages[-1]["source_char_end"]
        ):
            prior = passages[-1]
            overlap_chars = prior["source_char_end"] - start
            if overlap_chars < len(text):
                prior["text"] += text[max(0, overlap_chars):]
            prior["source_char_end"] = max(prior["source_char_end"], end)
            prior["chunk_ids"].append(str(chunk["chunk_id"]))
            section_id = str(chunk.get("section_id", ""))
            if section_id and section_id not in prior["section_ids"]:
                prior["section_ids"].append(section_id)
            heading = str(chunk.get("heading") or "")
            if heading and heading not in prior["headings"]:
                prior["headings"].append(heading)
            prior["best_seed_rank"] = min(prior["best_seed_rank"], float(chunk.get("seed_rank", float("inf"))))
            continue

        passages.append({
            "document_id": str(chunk.get("document_id", "")),
            "section_id": group[1],
            "section_ids": [str(chunk.get("section_id", ""))],
            "section_order": int(chunk.get("section_order", 0)),
            "heading": chunk.get("heading"),
            "headings": [str(chunk.get("heading") or "")],
            "title": chunk.get("title", ""),
            "source_char_start": start,
            "source_char_end": end,
            "text": text,
            "chunk_ids": [str(chunk["chunk_id"])],
            "best_seed_rank": float(chunk.get("seed_rank", _rank_key(chunk)[0])),
        })

    passages.sort(key=lambda p: (p["best_seed_rank"], p["document_id"], p["source_char_start"]))
    return passages


def render_passages(passages: Sequence[Mapping[str, Any]]) -> str:
    """Render stitched evidence with source labels for the answer generator."""
    rendered: list[str] = []
    for passage in passages:
        title = str(passage.get("title") or "Untitled technote")
        heading = " / ".join(h for h in passage.get("headings", []) if h) or str(
            passage.get("heading") or "UNSECTIONED"
        )
        document_id = str(passage.get("document_id", ""))
        rendered.append(
            f"[Technote: {title} | ID: {document_id} | Section: {heading}]\n"
            f"{passage['text']}"
        )
    return "\n\n".join(rendered)
