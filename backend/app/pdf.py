import hashlib
import io
import json
import re
import subprocess
from dataclasses import dataclass, field

import fitz

from .chunking import EmptyDocumentError
from .config import get_settings

MIN_PAGE_CHARS = 30  # below this, the page is (re)tried through OCR
MIN_USABLE_CHARS = 10  # below this, the page counts as having no usable text


class PDFError(Exception):
    """Raised when an upload is not a usable PDF."""


@dataclass
class Extraction:
    """Everything extracted from one PDF."""

    pages: list[tuple[int, str]]
    metadata: dict[str, str] = field(default_factory=dict)
    total_pages: int = 0
    text_pages: int = 0
    ocr_pages: int = 0
    table_count: int = 0
    warning: str | None = None


def compute_file_hash(content: bytes) -> str:
    """Deterministic fingerprint of the uploaded file bytes (SHA-256)."""
    return hashlib.sha256(content).hexdigest()


def validate_pdf(filename: str | None, content: bytes) -> None:
    if not content:
        raise PDFError("The uploaded file is empty.")
    if not filename or not filename.lower().endswith(".pdf"):
        raise PDFError("Only PDF files are supported.")
    if not content.startswith(b"%PDF"):
        raise PDFError("File content is not a valid PDF.")
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise PDFError(f"File exceeds the {get_settings().max_upload_mb} MB limit.")


def extract_pages(content: bytes) -> list[tuple[int, str]]:
    """Return (1-based page number, page text) pairs."""
    return extract_document(content).pages


def extract_document(content: bytes) -> Extraction:
    """Full extraction: text (with tables rendered as markdown), metadata, OCR, warnings."""
    try:
        doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
    except Exception as exc:  # PyMuPDF raises its own error types
        raise PDFError("Unable to read the PDF file.") from exc

    try:
        if doc.page_count == 0:
            raise PDFError("The PDF has no pages.")
        metadata = _clean_metadata(doc.metadata or {})
        pages: list[tuple[int, str]] = []
        table_count = 0
        ocr_pages = 0
        for index in range(doc.page_count):
            page = doc.load_page(index)
            tables, count = _page_tables(page)
            table_count += count
            text = _assemble_page_text(page, tables)
            if len(text.strip()) < MIN_PAGE_CHARS:
                ocr_text = _ocr_page(page)
                # Only adopt OCR when it recovers more than the text layer had.
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    ocr_pages += 1
            pages.append((index + 1, text.strip()))
        pages = _strip_repeated_furniture(pages)

        text_pages = sum(1 for _, text in pages if len(text.strip()) >= MIN_USABLE_CHARS)
        total = len(pages)
        warning = _build_warning(text_pages, total, ocr_pages)
        if text_pages == 0:
            raise EmptyDocumentError(
                "This PDF appears to be scanned/image-based and contains little or no "
                "extractable text. OCR could not recover text from it either."
            )
        return Extraction(
            pages=pages,
            metadata=metadata,
            total_pages=total,
            text_pages=text_pages,
            ocr_pages=ocr_pages,
            table_count=table_count,
            warning=warning,
        )
    finally:
        doc.close()


# --- metadata --------------------------------------------------------------


def _clean_metadata(raw: dict) -> dict[str, str]:
    keep: dict[str, str] = {}
    for key in ("title", "author", "subject", "keywords", "creator", "producer",
                "creationDate", "modDate"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            keep[key] = value.strip()[:500]
    return keep


def metadata_json(metadata: dict[str, str]) -> str | None:
    return json.dumps(metadata, ensure_ascii=False) if metadata else None


# --- tables ----------------------------------------------------------------


def _bbox_overlap(a: tuple, b: tuple) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    block_area = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
    return ((x1 - x0) * (y1 - y0)) / block_area


def _render_table(table) -> str | None:
    try:
        rows = table.extract()
    except Exception:  # noqa: BLE001 - malformed tables are skipped, never fatal
        return None
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [(cell or "").replace("\n", " ").strip() for cell in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 2:
        return None
    width = max(len(row) for row in cleaned)
    if width < 2:
        return None
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    if sum(1 for row in cleaned if any(row)) < 2:
        return None
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _page_tables(page) -> tuple[list, int]:
    """Detect tables with the lines strategy, falling back to the text strategy."""
    for strategy in ("lines", "text"):
        try:
            finder = page.find_tables(strategy=strategy)
        except Exception:  # noqa: BLE001 - table finding must never break extraction
            continue
        rendered = []
        for table in getattr(finder, "tables", []) or []:
            markdown = _render_table(table)
            if markdown:
                rendered.append((tuple(table.bbox), markdown))
        if rendered:
            return rendered, len(rendered)
    return [], 0


def _assemble_page_text(page, tables: list[tuple[tuple, str]]) -> str:
    """Page text where table regions are replaced by markdown with row/column structure."""
    if not tables:
        return page.get_text("text")

    parts: list[str] = []
    used: set[int] = set()
    try:
        blocks = page.get_text("blocks")
    except Exception:  # noqa: BLE001
        return page.get_text("text")
    for block in blocks:
        x0, y0, x1, y1, text, _block_no, block_type = block
        if block_type != 0 or not text.strip():
            continue
        hit = None
        for index, (bbox, _markdown) in enumerate(tables):
            if _bbox_overlap((x0, y0, x1, y1), bbox) > 0.4:
                hit = index
                break
        if hit is None:
            parts.append(text.strip())
        elif hit not in used:
            used.add(hit)
            parts.append(tables[hit][1])
    for index, (_bbox, markdown) in enumerate(tables):
        if index not in used:
            parts.append(markdown)
    assembled = "\n".join(parts).strip()
    return assembled or page.get_text("text")


# --- OCR -------------------------------------------------------------------


def _ocr_page(page) -> str:
    """OCR a single image-only page through the tesseract CLI (no Python deps)."""
    settings = get_settings()
    if not settings.ocr_enabled:
        return ""
    try:
        pixmap = page.get_pixmap(dpi=settings.ocr_dpi)
        png = pixmap.tobytes("png")
        proc = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", settings.ocr_languages, "--psm", "3"],
            input=png,
            capture_output=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", "ignore").strip()


# --- repeated header/footer removal ---------------------------------------

_DIGIT_LINE = re.compile(r"^(?:page\s+)?[\-–—•\s]*\d{1,4}[\s\-–—\.]*$", re.IGNORECASE)


def _head_tail_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    candidates = lines[:2] + lines[-2:]
    return [line for line in candidates if 0 < len(line) <= 120]


def _strip_repeated_furniture(
    pages: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Conservatively remove running headers/footers/page numbers.

    A line is furniture only when it appears in the head/tail position of at least
    max(3, 40% of pages) pages. A page is never stripped down to nothing.
    """
    total = len(pages)
    if total < 3:
        return pages
    min_hits = max(3, int(total * 0.4))

    counts: dict[str, int] = {}
    digit_positions = 0
    for _, text in pages:
        seen: set[str] = set()
        for line in _head_tail_lines(text):
            if line not in seen:
                seen.add(line)
                counts[line] = counts.get(line, 0) + 1
        first = next((l for l in text.splitlines() if l.strip()), "")
        last = next((l for l in reversed(text.splitlines()) if l.strip()), "")
        if first.strip() and _DIGIT_LINE.match(first.strip()):
            digit_positions += 1
        if last.strip() and last.strip() != first.strip() and _DIGIT_LINE.match(last.strip()):
            digit_positions += 1

    furniture = {line for line, hits in counts.items() if hits >= min_hits}
    strip_digit_lines = digit_positions >= min_hits
    if not furniture and not strip_digit_lines:
        return pages

    cleaned: list[tuple[int, str]] = []
    for page_number, text in pages:
        lines = text.splitlines()
        kept: list[str] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            is_edge = index < 2 or index >= len(lines) - 2
            drop = is_edge and stripped in furniture
            if is_edge and strip_digit_lines and _DIGIT_LINE.match(stripped):
                drop = True
            if not drop:
                kept.append(line)
        result = "\n".join(kept).strip()
        cleaned.append((page_number, result if result else text))
    return cleaned


def _build_warning(text_pages: int, total: int, ocr_pages: int) -> str | None:
    if total >= 5 and text_pages < total / 2:
        message = (
            f"{text_pages} of {total} pages have no usable text layer "
            "(possibly scanned images); answers may be incomplete."
        )
        if ocr_pages:
            message += f" OCR recovered text from {ocr_pages} page(s)."
        return message
    if ocr_pages:
        return f"OCR was used to recover text from {ocr_pages} image-based page(s)."
    return None
