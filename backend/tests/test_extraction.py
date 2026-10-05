"""Extraction: text layers, tables as markdown, OCR for scanned pages, warnings."""

import shutil

import fitz
import pytest

from app.chunking import EmptyDocumentError
from app.config import get_settings
from app.pdf import (
    PDFError,
    _bbox_overlap,
    _build_warning,
    _clean_metadata,
    _render_table,
    _strip_repeated_furniture,
    extract_document,
    metadata_json,
)


def _text_pdf(text: str = "Hello GuardRAG", pages: int = 1, meta: dict | None = None) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    if meta:
        doc.set_metadata(meta)
    data = doc.tobytes()
    doc.close()
    return data


def _table_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    cols = [60, 200, 340, 480]
    rows = [80, 110, 140, 170, 200]
    for y in rows:
        page.draw_line(fitz.Point(cols[0], y), fitz.Point(cols[-1], y), width=0.8)
    for x in cols:
        page.draw_line(fitz.Point(x, rows[0]), fitz.Point(x, rows[-1]), width=0.8)
    labels = [
        ["Name", "Score", "Rank"],
        ["Ada", "10", "1"],
        ["Bo", "9", "2"],
        ["Cy", "8", "3"],
    ]
    for r, y in enumerate(rows[:-1]):
        for c, x in enumerate(cols[:-1]):
            page.insert_text((x + 4, y + 18), labels[r][c], fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _scanned_pdf() -> bytes:
    """A page whose text exists only as a raster image (no text layer)."""
    source = fitz.open()
    page = source.new_page()
    page.insert_text((60, 120), "SCANNED WORLD REPORT", fontsize=36)
    pixmap = page.get_pixmap(dpi=200)
    png = pixmap.tobytes("png")
    source.close()

    doc = fitz.open()
    image_page = doc.new_page(width=pixmap.width, height=pixmap.height)
    image_page.insert_image(image_page.rect, stream=png)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def ocr_on(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "ocr_languages", "eng")


@pytest.fixture
def ocr_off(monkeypatch):
    monkeypatch.setattr(get_settings(), "ocr_enabled", False)


def test_extract_document_returns_pages_and_metadata(ocr_off):
    extraction = extract_document(_text_pdf("Page body text", pages=2))
    assert [n for n, _ in extraction.pages] == [1, 2]
    assert extraction.pages[0][1] == "Page body text"
    assert extraction.total_pages == 2
    assert extraction.warning is None


def test_extract_document_cleans_pdf_metadata(ocr_off):
    extraction = extract_document(
        _text_pdf(meta={"title": "Deep Study", "author": "Ada", "trapped": "x"})
    )
    assert extraction.metadata["title"] == "Deep Study"
    assert extraction.metadata["author"] == "Ada"
    assert "trapped" not in extraction.metadata


def test_table_becomes_markdown_with_row_structure(ocr_off):
    extraction = extract_document(_table_pdf())
    text = extraction.pages[0][1]
    assert extraction.table_count >= 1
    assert "| Name | Score | Rank |" in text
    assert "| --- |" in text
    assert "| Ada | 10 | 1 |" in text
    # Cells appear once (inside the markdown), not again as loose text.
    assert text.count("Ada") == 1


def test_render_table_builds_markdown():
    class FakeTable:
        def extract(self):
            return [["A", "B"], ["1", "2"], ["3", "4"]]

    markdown = _render_table(FakeTable())
    assert markdown == "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"


def test_render_table_rejects_single_column():
    class FakeTable:
        def extract(self):
            return [["only"], ["value"]]

    assert _render_table(FakeTable()) is None


def test_render_table_drops_empty_rows_and_pads_cells():
    class FakeTable:
        def extract(self):
            return [["A", "B"], ["", ""], ["1"]]

    markdown = _render_table(FakeTable())
    assert markdown == "| A | B |\n| --- | --- |\n| 1 |  |"


def test_bbox_overlap():
    assert _bbox_overlap((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _bbox_overlap((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert _bbox_overlap((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_scanned_pdf_is_ocr_recovered(ocr_on):
    extraction = extract_document(_scanned_pdf())
    text = extraction.pages[0][1]
    assert extraction.ocr_pages == 1
    assert "SCANNED" in text.upper()
    assert extraction.warning is not None
    assert "OCR" in extraction.warning


def test_scanned_pdf_without_ocr_is_rejected(ocr_off):
    with pytest.raises(EmptyDocumentError):
        extract_document(_scanned_pdf())


def test_broken_pdf_raises_pdf_error():
    with pytest.raises(PDFError):
        extract_document(b"%PDF-1.4 broken")


def test_build_warning_none_when_all_pages_have_text():
    assert _build_warning(text_pages=5, total=5, ocr_pages=0) is None
    assert _build_warning(text_pages=2, total=2, ocr_pages=0) is None


def test_build_warning_reports_mostly_image_pdfs():
    warning = _build_warning(text_pages=2, total=10, ocr_pages=1)
    assert warning is not None
    assert "2 of 10 pages" in warning
    assert "OCR recovered" in warning


def test_build_warning_reports_ocr_usage():
    warning = _build_warning(text_pages=3, total=3, ocr_pages=3)
    assert "OCR was used" in warning


def test_clean_metadata_truncates_and_drops_blank():
    cleaned = _clean_metadata({"title": "x" * 600, "author": "  ", "subject": "ok", "key": None})
    assert len(cleaned["title"]) == 500
    assert "author" not in cleaned
    assert cleaned["subject"] == "ok"
    assert "key" not in cleaned


def test_metadata_json_encodes_or_none():
    assert metadata_json({}) is None
    assert metadata_json({"title": "T"}) == '{"title": "T"}'


def test_strip_repeated_furniture_removes_running_header():
    header = "GuardRAG Running Header"
    pages = [
        (n, f"{header}\nBody text of page {n} with unique content.") for n in range(1, 6)
    ]
    cleaned = _strip_repeated_furniture(pages)
    for _page, text in cleaned:
        assert header not in text
        assert "unique content" in text


def test_strip_repeated_furniture_never_empties_a_page():
    pages = [(n, "Same single line on every page") for n in range(1, 4)]
    cleaned = _strip_repeated_furniture(pages)
    assert all(text.strip() for _, text in cleaned)
    assert cleaned == pages


def test_strip_repeated_furniture_drops_repeated_page_numbers():
    pages = [(n, f"Body of page {n}.\nPage {n}") for n in range(1, 6)]
    # The digit-position heuristic removes edge lines that are pure page numbers.
    cleaned = _strip_repeated_furniture(pages)
    assert cleaned == [(n, f"Body of page {n}.") for n in range(1, 6)]


def test_strip_leaves_short_documents_alone():
    pages = [(1, "Only one page here."), (2, "Only one page here.")]
    assert _strip_repeated_furniture(pages) == pages
