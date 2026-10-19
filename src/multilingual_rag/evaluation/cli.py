"""Command-line entry point for golden-dataset maintenance and retrieval evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from multilingual_rag.evaluation.dataset import refresh_item
from multilingual_rag.evaluation.export import to_giskard, to_ragas
from multilingual_rag.evaluation.golden import (
    ChunkLookup,
    UnresolvedEvidenceError,
    load_jsonl,
    resolve_item,
    write_jsonl,
)
from multilingual_rag.evaluation.metrics import QueryResult, summarize, summarize_by
from multilingual_rag.retrieval.embeddings import DEFAULT_MODEL_NAME

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("eval/golden_dataset.jsonl")


def _active_index_chunks(index_root: Path) -> tuple[str, list[dict]]:
    pointer = index_root / "current.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No active FAISS index found under {index_root}")
    build_id = json.loads(pointer.read_text(encoding="utf-8"))["build_id"]
    return build_id, load_jsonl(index_root / build_id / "chunks.jsonl")


def run_retrieval(args: argparse.Namespace) -> None:
    from multilingual_rag.retrieval.embeddings import E5Embedder
    from multilingual_rag.retrieval.faiss_store import search_index

    items = load_jsonl(args.dataset)
    build_id, index_chunks = _active_index_chunks(args.index_root)
    lookup = ChunkLookup(index_chunks)
    ks = sorted(set(args.k))
    depth = max(ks)

    embedder = E5Embedder(
        args.model, device=args.device, revision=args.revision, cache_dir=args.cache_dir
    )

    results: list[QueryResult] = []
    per_query: list[dict] = []
    skipped_unresolved: list[str] = []
    for item in items:
        if not item.get("evidence"):
            continue  # truly unanswerable: no gold chunks to recall
        try:
            groups = resolve_item(item, lookup)
        except UnresolvedEvidenceError as exc:
            if not args.allow_unresolved:
                raise SystemExit(
                    f"{exc}\nThe active index ({build_id}) was built from different chunks than the "
                    "golden dataset expects. Rebuild the index (uv run rag-index) or pass "
                    "--allow-unresolved to skip such items."
                ) from exc
            skipped_unresolved.append(item["id"])
            continue

        hits = search_index(
            args.index_root,
            item["question"],
            top_k=depth,
            model_name=args.model,
            device=args.device,
            revision=args.revision,
            embedder=embedder,
        )
        result = QueryResult(
            item_id=item["id"],
            groups=groups,
            retrieved_ids=[hit.chunk["chunk_id"] for hit in hits],
            retrieved_sources=[hit.chunk["source_path"] for hit in hits],
        )
        results.append(result)
        per_query.append(
            {
                "id": item["id"],
                "question": item["question"],
                "question_type": item["question_type"],
                "language": item["language"],
                "first_relevant_rank": result.first_relevant_rank(),
                "evidence_ranks": result.group_ranks(),
                **{f"recall@{k}": result.recall_at(k) for k in ks},
                "retrieved": [
                    {"rank": rank, "chunk_id": hit.chunk["chunk_id"], "score": round(hit.score, 4),
                     "title": hit.chunk.get("title"), "source_path": hit.chunk["source_path"]}
                    for rank, hit in enumerate(hits, start=1)
                ],
            }
        )

    by_id = {item["id"]: item for item in items}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "index_build_id": build_id,
        "embedding_model": args.model,
        "k": ks,
        "skipped_unresolved": skipped_unresolved,
        "overall": summarize(results, ks),
        "by_question_type": summarize_by(results, by_id, "question_type", ks),
        "by_language": summarize_by(results, by_id, "language", ks),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.output_dir / f"retrieval_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / f"retrieval_{stamp}_per_query.jsonl", per_query)

    _print_table("overall", {"all": report["overall"]}, ks)
    _print_table("question_type", report["by_question_type"], ks)
    _print_table("language", report["by_language"], ks)
    print(f"\nReport: {report_path}")


def _print_table(title: str, rows: dict[str, dict[str, float]], ks: list[int]) -> None:
    columns = ["n", "MRR", *[f"Recall@{k}" for k in ks], *[f"Hit@{k}" for k in ks]]
    print(f"\n{title:<16}" + "".join(f"{column:>11}" for column in columns))
    for name, metrics in rows.items():
        cells = []
        for column in columns:
            value = metrics.get(column, 0)
            cells.append(f"{int(value):>11d}" if column == "n" else f"{value:>11.3f}")
        print(f"{name:<16}" + "".join(cells))


def run_refresh(args: argparse.Namespace) -> None:
    items = load_jsonl(args.dataset)
    lookup = ChunkLookup(load_jsonl(args.chunks))
    refreshed = [refresh_item(item, lookup) for item in items]
    write_jsonl(args.dataset, refreshed)
    print(f"Refreshed {len(refreshed)} items against {args.chunks}")


def run_export(args: argparse.Namespace) -> None:
    items = load_jsonl(args.dataset)
    out = args.output_dir
    write_jsonl(out / "ragas_golden.jsonl", [to_ragas(item) for item in items if item["answerable"]])
    write_jsonl(out / "ragas_golden_unanswerable.jsonl", [to_ragas(item) for item in items if not item["answerable"]])
    write_jsonl(out / "giskard_qatestset.jsonl", [to_giskard(item) for item in items])
    print(f"Wrote Ragas and Giskard files to {out}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Golden-dataset tools for the multilingual RAG.")
    sub = parser.add_subparsers(dest="command", required=True)

    retrieval = sub.add_parser("retrieval", help="Compute Recall@K, Hit@K and MRR on the active index")
    retrieval.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    retrieval.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))
    retrieval.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    retrieval.add_argument("--model", default=DEFAULT_MODEL_NAME)
    retrieval.add_argument("--revision", default=None)
    retrieval.add_argument("--device", default="cpu")
    retrieval.add_argument("--cache-dir", default=".cache/")
    retrieval.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    retrieval.add_argument("--allow-unresolved", action="store_true",
                           help="Skip items whose evidence quotes are missing from the index chunks")
    retrieval.set_defaults(func=run_retrieval)

    refresh = sub.add_parser("refresh", help="Re-map gold chunk IDs and contexts after re-ingestion")
    refresh.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    refresh.add_argument("--chunks", type=Path, default=Path("artifacts/chunks.jsonl"))
    refresh.set_defaults(func=run_refresh)

    export = sub.add_parser("export", help="Write Ragas and Giskard RAGET input files")
    export.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    export.add_argument("--output-dir", type=Path, default=Path("eval/exports"))
    export.set_defaults(func=run_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
