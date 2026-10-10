"""Publisher-specific cleanup for Verbraucherzentrale article PDFs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from multilingual_rag.ingestion.pipeline import DocumentPage


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublisherProfile:
    title_keyword: str
    host: str
    article_end_markers: tuple[str, ...]
    read_more_link_label: str
    source_line_pattern: str
    print_header_pattern: str
    footer_margin_fraction: float


PUBLISHER_PROFILES = (
    PublisherProfile(
        title_keyword="verbraucherzentrale",
        host="verbraucherzentrale.de",
        article_end_markers=("Artikel zum Thema", "Alle Wissens-Artikel"),
        read_more_link_label="Weiterlesen",
        source_line_pattern=r"(?im)^\s*Quelle\b[^\n]*(?:\n|$)",
        print_header_pattern=r"^\s*\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)\b",
        # Sampled A4 exports place footer blocks below 96% of page height.
        footer_margin_fraction=0.04,
    ),
)


class _PageRect(Protocol):
    height: float


class _PdfPage(Protocol):
    @property
    def rect(self) -> _PageRect: ...

    def get_text(self, option: str, *, sort: bool = False) -> Any: ...


def get_publisher_profile(title: str) -> PublisherProfile | None:
    """Find the configured publisher profile matching a document title."""
    normalized_title = title.casefold()
    return next(
        (profile for profile in PUBLISHER_PROFILES if profile.title_keyword in normalized_title),
        None,
    )


def extract_page_text(page: _PdfPage, *, publisher_title: str) -> str:
    """Extract text blocks and omit repeated browser-print margins for target PDFs."""
    profile = get_publisher_profile(publisher_title)
    if profile is None:
        return page.get_text("text", sort=True)

    footer_top = page.rect.height * (1 - profile.footer_margin_fraction)
    retained_blocks: list[str] = []
    for block in page.get_text("blocks", sort=True):
        _, top, _, _, block_text, *_ = block
        text = block_text.strip()
        if not text or top >= footer_top or re.match(profile.print_header_pattern, text):
            continue
        retained_blocks.append(text)
    return "\n".join(retained_blocks)


def filter_pages(
    pages: list[DocumentPage],
) -> list[DocumentPage]:
    """Trim article boilerplate without changing pages from other publishers."""
    if not pages:
        return pages

    profile = get_publisher_profile(pages[0].title)
    if profile is None:
        return pages

    filtered: list[DocumentPage] = []
    marker_found = False
    for page in pages:
        text = page.text
        marker_positions = [
            position
            for marker in profile.article_end_markers
            if (position := text.find(marker)) >= 0
        ]
        read_more_match = re.search(
            rf"(?im)^\s*{re.escape(profile.read_more_link_label)}\s*$", text
        )
        if read_more_match:
            marker_positions.append(read_more_match.start())
        marker_index = min(marker_positions, default=-1)
        if marker_index >= 0:
            marker_found = True
            text = text[:marker_index]
        text = re.sub(profile.source_line_pattern, "", text).strip()
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