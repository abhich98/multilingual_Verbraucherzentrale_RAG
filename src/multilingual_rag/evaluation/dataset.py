"""Build and refresh golden items from evidence quotes and a chunk set."""

from __future__ import annotations

from typing import Any

from multilingual_rag.evaluation.golden import ChunkLookup, UnresolvedEvidenceError


def _resolve_quotes(
    entries: list[dict[str, Any]], lookup: ChunkLookup, item_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach chunk IDs to each quote and return (entries, primary chunk per entry)."""
    resolved: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    for entry in entries:
        matches = lookup.find(entry["source_path"], entry["quote"])
        if not matches:
            raise UnresolvedEvidenceError(
                f"{item_id}: quote not found in {entry['source_path']!r}: {entry['quote'][:80]!r}"
            )
        resolved.append(
            {
                "source_path": entry["source_path"],
                "quote": entry["quote"],
                "chunk_ids": [chunk["chunk_id"] for chunk in matches],
            }
        )
        primary.append(matches[0])
    return resolved, primary


def refresh_item(item: dict[str, Any], lookup: ChunkLookup) -> dict[str, Any]:
    """Recompute chunk IDs, reference contexts and titles from the item's quotes."""
    evidence, primary = _resolve_quotes(item.get("evidence", []), lookup, item["id"])
    related, _ = _resolve_quotes(item.get("related", []), lookup, item["id"])

    contexts: list[str] = []
    chunk_ids: list[str] = []
    titles: list[str] = []
    for chunk in primary:
        if chunk["chunk_id"] in chunk_ids:
            continue
        chunk_ids.append(chunk["chunk_id"])
        contexts.append(chunk["text"])
        title = chunk.get("title") or chunk["source_path"]
        if title not in titles:
            titles.append(title)

    refreshed = dict(item)
    refreshed.update(
        reference_contexts=contexts,
        reference_chunk_ids=chunk_ids,
        source_titles=titles,
        evidence=evidence,
    )
    if related:
        refreshed["related"] = related
    return refreshed
