"""Build provenance metadata for a completed ingestion artifact."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from multilingual_rag.ingestion.filtering import (
    PUBLISHER_PROFILES,
)
from multilingual_rag import INGESTION_MANIFEST_VERSION


def _get_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def build_ingestion_manifest(
    input_dir: Path,
    output_path: Path,
    source_digests: dict[str, str],
    *,
    chunk_size: int,
    overlap: int,
    chunk_count: int,
) -> dict[str, object]:
    """Describe the source dataset, transformation settings, and JSONL output."""
    source_files = [
        {"path": source_path, "sha256": digest}
        for source_path, digest in sorted(source_digests.items())
    ]
    # Include paths as well as bytes so renaming a source changes the dataset fingerprint.
    source_fingerprint = hashlib.sha256(
        "".join(f"{item['path']}\0{item['sha256']}\n" for item in source_files).encode("utf-8")
    ).hexdigest()

    return {
        "manifest_version": INGESTION_MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": {"git_commit": _get_git_commit(input_dir)},
        "input": {
            "directory": input_dir.name,
            "source_data_sha256": source_fingerprint,
            "files": source_files,
        },
        "filtering": {
            "publisher_profiles": [asdict(profile) for profile in PUBLISHER_PROFILES],
        },
        "chunking": {"chunk_size": chunk_size, "overlap": overlap},
        "output": {
            "path": output_path.name,
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "chunk_count": chunk_count,
        },
    }