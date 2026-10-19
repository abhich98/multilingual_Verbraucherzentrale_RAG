import hashlib
import json
from pathlib import Path

import pytest

import multilingual_rag.ingestion.filtering as filtering
from multilingual_rag.ingestion.pipeline import (
    DocumentPage,
    IngestionError,
    chunk_pages,
    extract_pdf_url,
    ingest_pdf_directory,
)
from multilingual_rag.ingestion.filtering import extract_page_text, filter_pages


def test_chunk_pages_preserves_metadata_and_overlap() -> None:
    page = DocumentPage(
        source_path="article.pdf",
        url="https://www.verbraucherzentrale.de/wissen/article",
        source_sha256="abc123",
        title="Article",
        language="de",
        page=2,
        text="Dies ist ein Satz. " * 20,
    )

    chunks = chunk_pages([page], chunk_size=80, overlap=15)

    assert len(chunks) > 1
    assert all(chunk.source_path == "article.pdf" for chunk in chunks)
    assert all(chunk.page_start == chunk.page_end == 2 for chunk in chunks)
    assert all(chunk.language == "de" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert chunks[0].text[-15:] in chunks[1].text


def test_chunk_pages_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=10, overlap=10)


def test_chunk_pages_keeps_split_punctuation_without_overlap() -> None:
    page = DocumentPage("article.pdf", "", "digest", "Article", "de", 1, "alpha. beta gamma")

    chunks = chunk_pages([page], chunk_size=8, overlap=0)

    assert " ".join(chunk.text for chunk in chunks) == page.text


def test_ingest_pdf_directory_writes_jsonl(tmp_path) -> None:
    input_dir = tmp_path / "source"
    input_dir.mkdir()
    (input_dir / "not-a-pdf.pdf").write_bytes(b"invalid PDF")
    output_path = tmp_path / "out" / "chunks.jsonl"

    with pytest.raises(IngestionError):
        ingest_pdf_directory(input_dir, output_path)
    assert not output_path.exists()


def test_jsonl_chunk_schema_is_serializable() -> None:
    page = DocumentPage("article.pdf", "", "digest", "Article", "de", 1, "Text")
    chunk = chunk_pages([page])[0]

    record = json.loads(json.dumps(chunk.__dict__, ensure_ascii=False))
    assert record["text"] == "Text"
    assert record["source_sha256"] == "digest"


def test_filter_trims_article_end_and_source_lines() -> None:
    pages = [
        DocumentPage(
            "article.pdf", "", "digest", "Article | Verbraucherzentrale.de", "de", 1,
            "Article body\nQuelle: stock image credit",
        ),
        DocumentPage(
            "article.pdf", "", "digest", "Article | Verbraucherzentrale.de", "de", 2,
            "Last article paragraph\nArtikel zum Thema\nSite footer",
        ),
        DocumentPage(
            "article.pdf", "", "digest", "Article | Verbraucherzentrale.de", "de", 3,
            "Print footer",
        ),
    ]

    filtered = filter_pages(pages)

    assert [page.page for page in filtered] == [1, 2]
    assert filtered[0].text == "Article body"
    assert filtered[1].text == "Last article paragraph"


def test_filter_leaves_non_verbraucherzentrale_pdfs_unchanged() -> None:
    page = DocumentPage("other.pdf", "", "digest", "Other publisher", "de", 1, "Quelle: keep this")

    assert filter_pages([page]) == [page]


def test_filter_uses_related_articles_heading_as_fallback_end_marker() -> None:
    pages = [
        DocumentPage(
            "article.pdf", "", "digest", "Article Verbraucherzentrale.de", "de", 1,
            "Final article paragraph\nAlle Wissens-Artikel\nRecommendation card",
        ),
        DocumentPage(
            "article.pdf", "", "digest", "Article Verbraucherzentrale.de", "de", 2,
            "Website footer",
        ),
    ]

    filtered = filter_pages(pages)

    assert len(filtered) == 1
    assert filtered[0].text == "Final article paragraph"


def test_filter_uses_standalone_read_more_link_but_not_prose() -> None:
    pages = [
        DocumentPage(
            "article.pdf", "", "digest", "Article Verbraucherzentrale.de", "de", 1,
            "We explain how to read more efficiently.\nWeiterlesen\nRecommendation card",
        ),
        DocumentPage(
            "article.pdf", "", "digest", "Article Verbraucherzentrale.de", "de", 2,
            "Website footer",
        ),
    ]

    filtered = filter_pages(pages)

    assert len(filtered) == 1
    assert filtered[0].text == "We explain how to read more efficiently."


def test_new_publisher_profile_reuses_generic_filter_and_url_logic(monkeypatch) -> None:
    profile = filtering.PublisherProfile(
        title_keyword="example press",
        host="example.com",
        article_end_markers=("Related stories",),
        read_more_link_label="Read more",
        source_line_pattern=r"(?im)^\s*Source\b[^\n]*(?:\n|$)",
        print_header_pattern=r"^PRINT HEADER$",
        footer_margin_fraction=0.04,
    )
    monkeypatch.setattr(filtering, "PUBLISHER_PROFILES", (*filtering.PUBLISHER_PROFILES, profile))
    pages = [
        DocumentPage(
            "article.pdf", "", "digest", "Article | Example Press", "en", 1,
            "Article body\nSource: image credit\nRelated stories\nRecommendation",
        )
    ]

    class Page:
        def get_links(self):
            return [{"uri": "https://example.com/articles/123"}]

    class Reader:
        def __iter__(self):
            return iter([Page()])

    assert filter_pages(pages)[0].text == "Article body"
    assert extract_pdf_url(Reader(), [], "Article | Example Press") == "https://example.com/articles/123"


def test_extract_pdf_url_prefers_article_url_in_text() -> None:
    class EmptyPage:
        def get_links(self):
            return []

    class Reader:
        is_encrypted = False

        def __iter__(self):
            return iter([EmptyPage()])

    article_url = "https://www.verbraucherzentrale.de/wissen/food/article-123"
    page_text = f"Header https://www.verbraucherzentrale.de/\n{article_url}"

    assert extract_pdf_url(Reader(), [page_text], "Article Verbraucherzentrale") == article_url


def test_extract_pdf_url_falls_back_to_link_annotation() -> None:
    article_url = "https://www.verbraucherzentrale.de/wissen/house/article-456"

    class Page:
        def get_links(self):
            return [{"uri": article_url}]

    class Reader:
        is_encrypted = False

        def __iter__(self):
            return iter([Page()])

    assert extract_pdf_url(Reader(), ["Article without a printed URL"], "Article Verbraucherzentrale") == article_url


def test_extract_pdf_url_rejects_truncated_printed_url() -> None:
    class Page:
        def get_links(self):
            return [{"uri": "https://www.verbraucherzentrale.de/"}]

    class Reader:
        is_encrypted = False

        def __iter__(self):
            return iter([Page()])

    truncated_url = "https://www.verbraucherzentrale.de/wissen/article-…"

    assert extract_pdf_url(Reader(), [truncated_url], "Article Verbraucherzentrale") == ""


def test_extract_page_text_removes_only_repeated_print_margins() -> None:
    from multilingual_rag.ingestion.filtering import extract_page_text

    class Rect:
        height = 842.0

    class Page:
        rect = Rect()

        def get_text(self, option, *, sort=False):
            if option == "text":
                return "Fallback text"
            return [
                (0, 15, 500, 26, "10/10/26, 12:09 PM Article | Verbraucherzentrale.de", 0, 0),
                (0, 778, 500, 807, "Article body ending near the bottom", 0, 0),
                (0, 816, 500, 827, "https://example.com/article 2/5", 0, 0),
            ]

    assert extract_page_text(
        Page(), publisher_title="Article | Verbraucherzentrale.de"
    ) == "Article body ending near the bottom"


def test_ingestion_writes_manifest_and_carries_url(tmp_path, monkeypatch) -> None:
    import multilingual_rag.ingestion.pipeline as pipeline
    from multilingual_rag.ingestion.dvc_utils import DVCSource

    input_dir = tmp_path / "source"
    input_dir.mkdir()
    source_path = input_dir / "Article Verbraucherzentrale.pdf"
    source_path.write_bytes(b"test pdf bytes")
    output_path = tmp_path / "out" / "chunks.jsonl"
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    article_url = "https://www.verbraucherzentrale.de/wissen/article-123"

    monkeypatch.setattr(
        pipeline,
        "extract_pdf",
        lambda path, root: [
            DocumentPage(path.name, article_url, digest, "Article Verbraucherzentrale", "de", 1, "Text")
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "get_dvc_source",
        lambda _: DVCSource("data/source.dvc", "source", "abc123.dir", False),
    )

    assert ingest_pdf_directory(
        input_dir, output_path, chunk_size=80, overlap=10, skip_dvc_check=True
    ) == 1

    record = json.loads(output_path.read_text(encoding="utf-8"))
    manifest_path = output_path.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plot_path = output_path.with_name("chunks_chunk_size_distribution.png")
    assert record["url"] == article_url
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert manifest["chunking"] == {"chunk_size": 80, "overlap": 10}
    assert manifest["input"]["files"] == [{"path": source_path.name, "sha256": digest}]
    assert manifest["input"]["dvc"] == {
        "pointer_path": "data/source.dvc",
        "output_path": "source",
        "md5": "abc123.dir",
        "verified": False,
    }
    assert manifest["output"]["chunk_count"] == 1
    assert manifest["output"]["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()


def test_validate_dvc_input_requires_clean_committed_pointer(tmp_path, monkeypatch) -> None:
    import subprocess

    import multilingual_rag.ingestion.dvc_utils as dvc_utils

    input_dir = tmp_path / "source"
    input_dir.mkdir()
    pointer = tmp_path / "source.dvc"
    pointer.write_text(
        "outs:\n- md5: abc123.dir\n  size: 10\n  nfiles: 1\n  hash: md5\n  path: source\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, str(tmp_path) if "--show-toplevel" in args else "", "")

    def run_dvc(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, "{}", "")

    monkeypatch.setattr(dvc_utils, "_run_git", run_git)
    monkeypatch.setattr(dvc_utils.subprocess, "run", run_dvc)

    source = dvc_utils.validate_dvc_input(input_dir)

    assert source.pointer_path == "source.dvc"
    assert source.output_path == "source"
    assert source.md5 == "abc123.dir"
    assert source.verified is True
    assert ["ls-files", "--error-unmatch", "--", "source.dvc"] in calls
    assert ["diff", "--quiet", "HEAD", "--", "source.dvc"] in calls


def test_validate_dvc_input_rejects_changed_dvc_data(tmp_path, monkeypatch) -> None:
    import subprocess

    import multilingual_rag.ingestion.dvc_utils as dvc_utils

    input_dir = tmp_path / "source"
    input_dir.mkdir()
    (tmp_path / "source.dvc").write_text(
        "outs:\n- md5: abc123.dir\n  path: source\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        dvc_utils,
        "_run_git",
        lambda args, *, cwd: subprocess.CompletedProcess(args, 0, str(tmp_path) if "--show-toplevel" in args else "", ""),
    )
    monkeypatch.setattr(
        dvc_utils.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, '{"source.dvc": {"changed outs": ["source"]}}', ""),
    )

    with pytest.raises(dvc_utils.DVCInputError, match="not up to date"):
        dvc_utils.validate_dvc_input(input_dir)


def test_validate_dvc_input_rejects_uncommitted_pointer(tmp_path, monkeypatch) -> None:
    import subprocess

    import multilingual_rag.ingestion.dvc_utils as dvc_utils

    input_dir = tmp_path / "source"
    input_dir.mkdir()
    (tmp_path / "source.dvc").write_text(
        "outs:\n- md5: abc123.dir\n  path: source\n", encoding="utf-8"
    )
    calls: list[list[str]] = []

    def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "diff":
            raise dvc_utils.DVCInputError("pointer is not committed")
        return subprocess.CompletedProcess(args, 0, str(tmp_path) if "--show-toplevel" in args else "", "")

    monkeypatch.setattr(dvc_utils, "_run_git", run_git)
    with pytest.raises(dvc_utils.DVCInputError, match="uncommitted changes"):
        dvc_utils.validate_dvc_input(input_dir)
    assert not any(args[0] == "dvc" for args in calls)