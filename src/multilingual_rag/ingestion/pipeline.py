"""Extract PDF text and write reproducible, metadata-rich chunks as JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


SEPARATORS = ("\n\n", ". ", "? ", "! ") 
# "\n", " " can be separators, but these are removed to favor chunk splits containing full sentences

class IngestionError(RuntimeError):
    """Raised when source documents cannot be safely ingested."""


@dataclass(frozen=True)
class DocumentPage:
    source_path: str
    url: str
    source_sha256: str
    title: str
    language: str
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    source_path: str
    url: str
    source_sha256: str
    title: str
    language: str
    page_start: int
    page_end: int
    chunk_index: int


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path, root: Path) -> list[DocumentPage]:
    """Extract non-empty pages with source path, hash, page and title metadata."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise IngestionError(f"Encrypted PDF is not supported: {path}")
        raw_pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not read PDF {path}: {exc}") from exc

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    relative_path = path.relative_to(root).as_posix()
    title = path.stem
    pages = [
        DocumentPage(
            source_path=relative_path,
            url="",
            source_sha256=digest,
            title=title,
            language="de",
            page=page_number,
            text=normalized,
        )
        for page_number, raw_text in raw_pages
        if (normalized := _normalize_text(raw_text))
    ]
    if not pages:
        raise IngestionError(f"No extractable text found in {path}; OCR may be required")
    return pages


def _split_text(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Split on paragraph/sentence/word boundaries with a bounded character overlap."""
    remaining = text.strip()
    while remaining:
        if len(remaining) <= chunk_size:
            yield remaining
            break

        split_points = [
            (remaining.rfind(separator, 0, chunk_size + 1) + len(separator))
            for separator in SEPARATORS
            if remaining.rfind(separator, 0, chunk_size + 1) >= 0
        ]
        end = max(split_points, default=0)

        # A fallback: This ensures no chunks are smaller than half the chunk size, and a fallback when we have no suitable split points.
        if end < chunk_size // 2:
            end = chunk_size // 2

        piece = remaining[:end].strip()
        if not piece:
            end = chunk_size
            piece = remaining[:end].strip()
        yield piece

        next_start = max(0, end - overlap)
        while next_start < len(remaining) and remaining[next_start].isspace():
            next_start += 1
        if next_start <= 0 or next_start >= len(remaining):
            break
        remaining = remaining[next_start:]


def chunk_pages(
    pages: list[DocumentPage], *, chunk_size: int = 1000, overlap: int = 150
) -> list[TextChunk]:
    """Chunk pages without crossing page boundaries, retaining source provenance."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        for piece in _split_text(page.text, chunk_size, overlap):
            index = len(chunks)
            identity = f"{page.source_path}\0{page.page}\0{index}\0{piece}"
            chunks.append(
                TextChunk(
                    chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    text=piece,
                    source_path=page.source_path,
                    url="",
                    source_sha256=page.source_sha256,
                    title=page.title,
                    language=page.language,
                    page_start=page.page,
                    page_end=page.page,
                    chunk_index=index,
                )
            )
    return chunks


def _write_jsonl_atomic(output_path: Path, chunks: list[TextChunk]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for chunk in chunks:
                stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def ingest_pdf_directory(
    input_dir: Path, output_path: Path, *, chunk_size: int = 1000, overlap: int = 150
) -> int:
    """Ingest PDFs recursively and return the number of written chunks."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise IngestionError(f"Input directory does not exist: {input_dir}")
    pdf_paths = sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())
    if not pdf_paths:
        raise IngestionError(f"No PDF files found under {input_dir}")

    chunks: list[TextChunk] = []
    for path in pdf_paths:
        pages = extract_pdf(path, input_dir)
        chunks.extend(chunk_pages(pages, chunk_size=chunk_size, overlap=overlap))
    _write_jsonl_atomic(output_path, chunks)
    return len(chunks)