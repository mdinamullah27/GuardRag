import io

import fitz
import pytest

from app.pdf import PDFError, extract_pages, validate_pdf


def make_pdf(text: str = "Hello GuardRAG", pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_rejects_empty_file():
    with pytest.raises(PDFError):
        validate_pdf("doc.pdf", b"")


def test_rejects_wrong_extension():
    with pytest.raises(PDFError):
        validate_pdf("notes.txt", b"%PDF-1.4")


def test_rejects_non_pdf_content():
    with pytest.raises(PDFError):
        validate_pdf("doc.pdf", b"not a pdf at all")


def test_rejects_oversized_file(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    with pytest.raises(PDFError):
        validate_pdf("doc.pdf", b"%PDF" + b"0" * (2 * 1024 * 1024))


def test_accepts_valid_pdf():
    validate_pdf("paper.pdf", make_pdf())


def test_extract_pages_returns_page_numbers():
    pages = extract_pages(make_pdf("page one text", pages=3))
    assert [number for number, _ in pages] == [1, 2, 3]
    assert pages[0][1] == "page one text"


def test_extract_pages_flags_unreadable_pdf():
    with pytest.raises(PDFError):
        extract_pages(b"%PDF-1.4 broken")
