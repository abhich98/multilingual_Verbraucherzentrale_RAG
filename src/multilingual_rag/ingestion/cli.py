"""Command-line entry point for raw PDF ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from multilingual_rag.ingestion.pipeline import IngestionError, ingest_pdf_directory


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Extract and chunk PDF source documents.")
    parser.add_argument("--input", type=Path, default=Path("data/Verbraucherzentrale_artikel"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Character overlap between chunks")
    parser.add_argument(
        "--skip-dvc-check",
        action="store_true",
        help="Skip checking that the input matches a committed DVC pointer",
    )
    args = parser.parse_args()

    try:
        count = ingest_pdf_directory(
            args.input,
            args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            skip_dvc_check=args.skip_dvc_check,
        )
    except (IngestionError, ValueError) as exc:
        logger.exception("Ingestion failed")
        parser.error(str(exc))
    logger.info("Wrote %d chunks to %s", count, args.output)


if __name__ == "__main__":
    main()