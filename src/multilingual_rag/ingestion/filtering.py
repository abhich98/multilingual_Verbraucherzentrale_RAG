"""Publisher-specific cleanup for Verbraucherzentrale article PDFs."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multilingual_rag.ingestion.pipeline import DocumentPage


logger = logging.getLogger(__name__)

ARTICLE_END_MARKER = "Artikel zum Thema"
SOURCE_LINE_PREFIX = "Quelle"
SOURCE_LINE_PATTERN = re.compile(
    rf"(?im)^\s*{re.escape(SOURCE_LINE_PREFIX)}\b[^\n]*(?:\n|$)"
)
VERBRAUCHERZENTRALE_TITLE_KEYWORD = "verbraucherzentrale"


def _is_verbraucherzentrale_title(title: str) -> bool:
    return VERBRAUCHERZENTRALE_TITLE_KEYWORD in title.casefold()


def filter_verbraucherzentrale_pages(
    pages: list[DocumentPage],
) -> list[DocumentPage]:
    """Trim article boilerplate without changing pages from other publishers."""
    if not pages or not _is_verbraucherzentrale_title(pages[0].title):
        return pages

    filtered: list[DocumentPage] = []
    marker_found = False
    for page in pages:
        text = page.text
        marker_index = text.find(ARTICLE_END_MARKER)
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