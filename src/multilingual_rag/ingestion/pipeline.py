"""Extract PDF text and write reproducible, metadata-rich chunks as JSONL."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.parse import urlsplit

from multilingual_rag.ingestion.filtering import (
    extract_page_text,
    filter_pages,
    get_publisher_profile,
)
from multilingual_rag.ingestion.manifest import build_ingestion_manifest
from multilingual_rag.json_utils import write_json_atomic, write_jsonl_atomic


logger = logging.getLogger(__name__)
pymupdf = import_module("pymupdf")

SEPARATORS = ("\n\n", ". ", "? ", "! ")
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


class _LinkPage(Protocol):
    def get_links(self) -> list[dict[str, Any]]: ...


class _PdfDocument(Protocol):
    is_encrypted: bool

    def __iter__(self) -> Iterator[_LinkPage]: ...


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


def extract_pdf_url(reader: _PdfDocument, page_texts: list[str], publisher_title: str) -> str:
    """Return the article URL in PDF text, falling back to the first PDF hyperlink."""

    # Printed footer URLs can be clipped in the PDF; do not publish an incomplete citation URL.
    raw_text_urls = [url.rstrip(".,;:!?)]}") for text in page_texts for url in URL_PATTERN.findall(text)]
    has_truncated_text_url = any(url.endswith(("…", "...")) for url in raw_text_urls)
    text_urls = [url for url in raw_text_urls if not url.endswith(("…", "..."))]
    linked_urls: list[str] = []
    for page in reader:
        linked_urls.extend(
            link["uri"] for link in page.get_links() if link.get("uri")
        )

    # An annotation fallback is unsafe when the document printed a clipped canonical URL;
    # its remaining links usually point to unrelated recommendations or site navigation.
    if has_truncated_text_url and not text_urls:
        return ""

    candidates = text_urls + linked_urls
    profile = get_publisher_profile(publisher_title)
    if profile is None:
        return ""

    for url in candidates:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        if (
            (host == profile.host or host.endswith(f".{profile.host}"))
            and parsed.path not in ("", "/")
            and not parsed.path.startswith("/media/")
        ):
            return url
    return ""


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path, root: Path) -> list[DocumentPage]:
    """Extract non-empty pages with source path, hash, page and title metadata."""
    title = path.stem
    try:
        with pymupdf.open(path) as reader:
            if reader.is_encrypted:
                raise IngestionError(f"Encrypted PDF is not supported: {path}")
            raw_pages = [
                (
                    index + 1,
                    extract_page_text(
                        page,
                        publisher_title=title,
                    ),
                )
                for index, page in enumerate(reader)
            ]
            # Capture footer URLs before margin removal; those strings may contain citation metadata.
            url = extract_pdf_url(
                reader,
                [page.get_text("text", sort=True) for page in reader],
                publisher_title=title,
            )
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Could not extract PDF", extra={"source_path": str(path)})
        raise IngestionError(f"Could not read PDF {path}: {exc}") from exc

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    relative_path = path.relative_to(root).as_posix()
    pages = [
        DocumentPage(
            source_path=relative_path,
            url=url,
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
                    url=page.url,
                    source_sha256=page.source_sha256,
                    title=page.title,
                    language=page.language,
                    page_start=page.page,
                    page_end=page.page,
                    chunk_index=index,
                )
            )
    return chunks


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

    logger.info("Starting PDF ingestion", extra={"pdf_count": len(pdf_paths)})
    chunks: list[TextChunk] = []
    source_digests: dict[str, str] = {}
    for path in pdf_paths:
        pages = extract_pdf(path, input_dir)
        source_digests[pages[0].source_path] = pages[0].source_sha256
        pages = filter_pages(pages)
        chunks.extend(chunk_pages(pages, chunk_size=chunk_size, overlap=overlap))
    write_jsonl_atomic(output_path, (asdict(chunk) for chunk in chunks))

    manifest = build_ingestion_manifest(
        input_dir,
        output_path,
        source_digests,
        chunk_size=chunk_size,
        overlap=overlap,
        chunk_count=len(chunks),
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    write_json_atomic(manifest_path, manifest)
    logger.info(
        "PDF ingestion complete",
        extra={
            "pdf_count": len(source_digests),
            "chunk_count": len(chunks),
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
        },
    )
    return len(chunks)