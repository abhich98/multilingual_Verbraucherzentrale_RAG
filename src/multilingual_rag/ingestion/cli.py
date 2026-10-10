"""Command-line entry point for raw PDF ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from multilingual_rag.ingestion.pipeline import IngestionError, ingest_pdf_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk PDF source documents.")
    parser.add_argument("--input", type=Path, default=Path("data/Verbraucherzentrale_artikel"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Character overlap between chunks")
    args = parser.parse_args()

    try:
        count = ingest_pdf_directory(
            args.input, args.output, chunk_size=args.chunk_size, overlap=args.overlap
        )
    except (IngestionError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {count} chunks to {args.output}")


if __name__ == "__main__":
    main()