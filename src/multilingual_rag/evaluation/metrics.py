"""Retrieval metrics computed against evidence groups.

Definitions, per query with evidence groups G and ranked retrieved chunks R:

* Recall@K  = share of groups in G with at least one chunk in R[:K].
              A multi-hop question with three required facts scores 2/3 if only
              two facts are retrieved.
* Hit@K     = 1 if any group in G is hit within R[:K], else 0.
* MRR       = mean over queries of 1 / rank of the first retrieved chunk that
              belongs to any group (0 if none within the retrieved depth).
* DocRecall@K = share of gold source documents with any chunk in R[:K]; this
              ignores which chunk of the article was found.

Queries without evidence (truly unanswerable items) are excluded from all
retrieval metrics: there is nothing to recall.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from multilingual_rag.evaluation.golden import EvidenceGroup


@dataclass(frozen=True)
class QueryResult:
    item_id: str
    groups: list[EvidenceGroup]
    retrieved_ids: list[str]
    retrieved_sources: list[str]

    def first_relevant_rank(self) -> int | None:
        relevant = set().union(*(group.chunk_ids for group in self.groups)) if self.groups else set()
        for rank, chunk_id in enumerate(self.retrieved_ids, start=1):
            if chunk_id in relevant:
                return rank
        return None

    def group_ranks(self) -> list[int | None]:
        """Rank at which each evidence group is first satisfied."""
        ranks: list[int | None] = []
        for group in self.groups:
            rank = next(
                (index for index, chunk_id in enumerate(self.retrieved_ids, start=1) if chunk_id in group.chunk_ids),
                None,
            )
            ranks.append(rank)
        return ranks

    def recall_at(self, k: int) -> float:
        ranks = self.group_ranks()
        return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)

    def hit_at(self, k: int) -> float:
        rank = self.first_relevant_rank()
        return 1.0 if rank is not None and rank <= k else 0.0

    def reciprocal_rank(self) -> float:
        rank = self.first_relevant_rank()
        return 1.0 / rank if rank else 0.0

    def doc_recall_at(self, k: int) -> float:
        gold_docs = {group.source_path for group in self.groups}
        found = set(self.retrieved_sources[:k])
        return len(gold_docs & found) / len(gold_docs)


def summarize(results: Sequence[QueryResult], ks: Sequence[int]) -> dict[str, float]:
    if not results:
        return {"n": 0}
    count = len(results)
    summary: dict[str, float] = {"n": count, "MRR": sum(r.reciprocal_rank() for r in results) / count}
    for k in ks:
        summary[f"Recall@{k}"] = sum(r.recall_at(k) for r in results) / count
        summary[f"Hit@{k}"] = sum(r.hit_at(k) for r in results) / count
        summary[f"DocRecall@{k}"] = sum(r.doc_recall_at(k) for r in results) / count
    return summary


def summarize_by(
    results: Sequence[QueryResult],
    items: dict[str, dict[str, Any]],
    field_name: str,
    ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[QueryResult]] = defaultdict(list)
    for result in results:
        buckets[str(items[result.item_id].get(field_name))].append(result)
    return {key: summarize(bucket, ks) for key, bucket in sorted(buckets.items())}
