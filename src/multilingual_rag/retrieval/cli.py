"""Command-line entry point for building the local embedding index."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from multilingual_rag.retrieval.embeddings import DEFAULT_MODEL_NAME
from multilingual_rag.retrieval.faiss_store import build_faiss_index


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Embed chunks and build a versioned FAISS index.")
    parser.add_argument("--chunks", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", help="Optional Hugging Face model revision to pin")
    parser.add_argument("--device", default="cpu", help="SentenceTransformer device, default: cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        version_path = build_faiss_index(
            args.chunks,
            args.index_root,
            model_name=args.model,
            device=args.device,
            revision=args.revision,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    logger.info("Built active retrieval index at %s", version_path)


if __name__ == "__main__":
    main()