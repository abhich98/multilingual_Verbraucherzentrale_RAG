"""Publisher-specific cleanup for Verbraucherzentrale article PDFs."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from multilingual_rag.ingestion.pipeline import DocumentPage


logger = logging.getLogger(__name__)

ARTICLE_END_MARKERS = ("Artikel zum Thema", "Alle Wissens-Artikel")
READ_MORE_LINK_LABEL = "Weiterlesen"
SOURCE_LINE_PREFIX = "Quelle"
SOURCE_LINE_PATTERN = re.compile(
    rf"(?im)^\s*{re.escape(SOURCE_LINE_PREFIX)}\b[^\n]*(?:\n|$)"
)
VERBRAUCHERZENTRALE_TITLE_KEYWORD = "verbraucherzentrale"
PRINT_HEADER_PATTERN = re.compile(
    r"^\s*\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)\b",
    re.IGNORECASE,
)
# In sampled A4 exports, footer blocks start below 96% of page height; body text ends above it.
FOOTER_MARGIN_FRACTION = 0.04


class _PageRect(Protocol):
    height: float


class _PdfPage(Protocol):
    @property
    def rect(self) -> _PageRect: ...

    def get_text(self, option: str, *, sort: bool = False) -> Any: ...


def is_verbraucherzentrale_title(title: str) -> bool:
    return VERBRAUCHERZENTRALE_TITLE_KEYWORD in title.casefold()


def extract_page_text(page: _PdfPage, *, publisher_specific: bool) -> str:
    """Extract text blocks and omit repeated browser-print margins for target PDFs."""
    if not publisher_specific:
        return page.get_text("text", sort=True)

    footer_top = page.rect.height * (1 - FOOTER_MARGIN_FRACTION)
    retained_blocks: list[str] = []
    for block in page.get_text("blocks", sort=True):
        _, top, _, _, block_text, *_ = block
        text = block_text.strip()
        if not text or top >= footer_top or PRINT_HEADER_PATTERN.match(text):
            continue
        retained_blocks.append(text)
    return "\n".join(retained_blocks)


def filter_verbraucherzentrale_pages(
    pages: list[DocumentPage],
) -> list[DocumentPage]:
    """Trim article boilerplate without changing pages from other publishers."""
    if not pages or not is_verbraucherzentrale_title(pages[0].title):
        return pages

    filtered: list[DocumentPage] = []
    marker_found = False
    for page in pages:
        text = page.text
        marker_positions = [
            position
            for marker in ARTICLE_END_MARKERS
            if (position := text.find(marker)) >= 0
        ]
        read_more_match = re.search(
            rf"(?im)^\s*{re.escape(READ_MORE_LINK_LABEL)}\s*$", text
        )
        if read_more_match:
            marker_positions.append(read_more_match.start())
        marker_index = min(marker_positions, default=-1)
        if marker_index >= 0:
            marker_found = True
            text = text[:marker_index]
        text = SOURCE_LINE_PATTERN.sub("", text).strip()
        if text:
            filtered.append(replace(page, text=text))
        if marker_index >= 0:
            break

    if not marker_found:
        logger.warning(
            "Article end marker not found; retained all pages after source-line cleanup",
            extra={"source_path": pages[0].source_path},
        )
    return filtered