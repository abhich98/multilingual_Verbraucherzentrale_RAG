"""Build and query versioned FAISS indexes over JSONL text chunks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from multilingual_rag.retrieval.embeddings import (
    DEFAULT_MODEL_NAME,
    E5Embedder,
    EmbeddingProvider,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """A chunk and cosine-similarity score returned by the index."""

    score: float
    chunk: dict[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_chunks(chunks_path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    source_bytes = chunks_path.read_bytes()
    chunks = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    if any(not isinstance(chunk.get("text"), str) or not chunk["text"].strip() for chunk in chunks):
        raise ValueError(f"Every chunk must contain non-empty text: {chunks_path}")
    if any("chunk_id" not in chunk for chunk in chunks):
        raise ValueError(f"Every chunk must contain a chunk_id: {chunks_path}")
    return source_bytes, chunks


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_faiss_index(
    chunks_path: Path,
    index_root: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    revision: str | None = None,
    batch_size: int = 32,
    embedder: EmbeddingProvider | None = None,
) -> Path:
    """Embed chunks, persist an immutable FAISS version, and promote its pointer."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file does not exist: {chunks_path}")

    source_bytes, chunks = _load_chunks(chunks_path)
    source_digest = _sha256(source_bytes)
    logger.info("Starting embedding build", extra={"chunk_count": len(chunks)})

    embedder = embedder or E5Embedder(model_name, device=device, revision=revision)
    vectors = np.asarray(
        embedder.encode_passages(
            [chunk["text"] for chunk in chunks], batch_size=batch_size
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
        raise ValueError("Embedder returned an invalid vector matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedder returned non-finite vector values")
    vectors = np.ascontiguousarray(vectors)

    import faiss

    index_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-")
    revision_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", revision or "default").strip("-")
    build_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{model_slug}-"
        f"{revision_slug}-{source_digest[:12]}"
    )
    build_directory = index_root / build_id
    with tempfile.TemporaryDirectory(prefix=".building-", dir=index_root) as temporary:
        temporary_directory = Path(temporary)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(vectors, np.arange(len(chunks), dtype=np.int64))
        index_path = temporary_directory / "index.faiss"
        faiss.write_index(index, str(index_path))
        metadata_path = temporary_directory / "chunks.jsonl"
        metadata_path.write_bytes(source_bytes)
        manifest = {
            "manifest_version": 1,
            "build_id": build_id,
            "created_at_utc": created_at.isoformat(),
            "embedding": {
                "provider": "sentence-transformers",
                "model": model_name,
                "revision": revision,
                "device": device,
                "dimension": int(vectors.shape[1]),
                "passage_prefix": "passage: ",
                "query_prefix": "query: ",
                "normalized": True,
            },
            "index": {
                "type": "IndexIDMap2(IndexFlatIP)",
                "similarity": "cosine",
                "vector_count": len(chunks),
                "file": index_path.name,
                "sha256": _sha256(index_path.read_bytes()),
            },
            "chunks": {
                "source_file": str(chunks_path),
                "file": metadata_path.name,
                "sha256": source_digest,
            },
        }
        manifest_path = temporary_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_directory.rename(build_directory)

    # Promote only after the complete immutable index directory is present.
    _atomic_write_json(index_root / "current.json", {"build_id": build_id})
    logger.info(
        "Embedding index ready",
        extra={"build_id": build_id, "chunk_count": len(chunks), "dimension": vectors.shape[1]},
    )
    return build_directory


def search_index(
    index_root: Path,
    query: str,
    *,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    revision: str | None = None,
    embedder: EmbeddingProvider | None = None,
) -> list[SearchResult]:
    """Search the active index and return ranked chunks with their source metadata."""
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    pointer_path = index_root / "current.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"No active FAISS index found under {index_root}")
    build_id = json.loads(pointer_path.read_text(encoding="utf-8"))["build_id"]
    build_directory = index_root / build_id
    manifest = json.loads((build_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["embedding"]["model"] != model_name:
        raise ValueError(
            f"Active index uses {manifest['embedding']['model']!r}, not {model_name!r}"
        )
    indexed_revision = manifest["embedding"].get("revision")
    if indexed_revision != revision:
        raise ValueError(
            f"Active index uses revision {indexed_revision!r}, "
            f"not {revision!r}"
        )

    chunks = [
        json.loads(line)
        for line in (build_directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embedder = embedder or E5Embedder(model_name, device=device, revision=revision)
    query_vector = np.ascontiguousarray(
        np.asarray(embedder.encode_query(query), dtype=np.float32).reshape(1, -1)
    )
    import faiss

    index = faiss.read_index(str(build_directory / manifest["index"]["file"]))
    distances, identifiers = index.search(query_vector, min(top_k, len(chunks)))
    results: list[SearchResult] = []
    for score, identifier in zip(distances[0], identifiers[0], strict=True):
        if identifier < 0:
            continue
        results.append(SearchResult(score=float(score), chunk=chunks[int(identifier)]))
    return results