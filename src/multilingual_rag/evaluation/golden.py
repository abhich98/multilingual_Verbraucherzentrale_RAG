"""Load the golden dataset and resolve its evidence quotes to chunk IDs.

Each golden item stores its evidence as verbatim quotes from the source articles,
not only as chunk IDs. Chunk IDs depend on chunk size, overlap and page layout, so
they change whenever ingestion changes. Quotes do not, which lets the evaluation
re-map the gold labels onto whatever chunks the active index was built from.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace so quotes match across PDF line breaks."""
    return _WHITESPACE.sub(" ", text).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class ChunkLookup:
    """Whitespace-normalised chunk texts grouped by source document."""

    chunks: list[dict[str, Any]]
    _by_source: dict[str, list[tuple[str, dict[str, Any]]]] = field(init=False)

    def __post_init__(self) -> None:
        self._by_source = defaultdict(list)
        for chunk in self.chunks:
            self._by_source[chunk["source_path"]].append((normalize(chunk["text"]), chunk))

    def find(self, source_path: str, quote: str) -> list[dict[str, Any]]:
        """Return every chunk of ``source_path`` that contains ``quote``, in document order.

        Overlapping chunks can both contain a quote; all of them count as a hit.
        """
        needle = normalize(quote)
        return [chunk for text, chunk in self._by_source.get(source_path, []) if needle in text]


@dataclass(frozen=True)
class EvidenceGroup:
    """One required fact: retrieving any chunk that contains it satisfies the group."""

    quote: str
    source_path: str
    chunk_ids: frozenset[str]


class UnresolvedEvidenceError(ValueError):
    """Raised when a gold quote is not found in the chunks being evaluated."""


def resolve_item(
    item: dict[str, Any], lookup: ChunkLookup, *, allow_unresolved: bool = False
) -> list[EvidenceGroup]:
    """Map an item's evidence quotes onto chunk IDs of the given chunk set."""
    groups: list[EvidenceGroup] = []
    for evidence in item.get("evidence", []):
        matches = lookup.find(evidence["source_path"], evidence["quote"])
        if not matches:
            message = (
                f"{item['id']}: quote not found in {evidence['source_path']!r}: "
                f"{evidence['quote'][:80]!r}"
            )
            if allow_unresolved:
                continue
            raise UnresolvedEvidenceError(message)
        groups.append(
            EvidenceGroup(
                quote=evidence["quote"],
                source_path=evidence["source_path"],
                chunk_ids=frozenset(chunk["chunk_id"] for chunk in matches),
            )
        )
    return groups
