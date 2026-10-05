"""Exact/term text search, counting, and lightweight document structure helpers.

All searches run SQL-side (ILIKE) or over the overlap-free ``document_pages`` rows —
never by loading every vector or every chunk of a document into Python for a normal
query. Counting uses page text (not overlapping chunks) so occurrences are never
double-counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .models import DocumentChunk, DocumentPage
from .retriever import RetrievedChunk

MAX_SEARCH_TERMS = 12
DEFAULT_SEARCH_LIMIT = 120


@dataclass
class PageHit:
    page: int
    count: int
    snippet: str


def _escape(literal: str) -> str:
    return literal.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _term_conditions(terms: list[str], phrase: str | None):
    conditions = []
    if phrase:
        conditions.append(DocumentChunk.text.ilike(f"%{_escape(phrase)}%", escape="\\"))
    for term in terms[:MAX_SEARCH_TERMS]:
        conditions.append(DocumentChunk.text.ilike(f"%{_escape(term)}%", escape="\\"))
    return conditions


def search_chunks(
    db: Session,
    document_id: str,
    terms: list[str],
    phrase: str | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[RetrievedChunk]:
    """Every chunk matching ANY term or the exact phrase (case-insensitive), page-ordered."""
    conditions = _term_conditions(terms, phrase)
    if not conditions:
        return []
    statement = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id, or_(*conditions))
        .order_by(DocumentChunk.page_number, DocumentChunk.chunk_index)
        .limit(limit)
    )
    rows = db.execute(statement).scalars().all()
    return [
        RetrievedChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            text=chunk.text,
            similarity=1.0,
        )
        for chunk in rows
    ]


def _pattern_for(needle: str) -> re.Pattern[str] | None:
    """Word-boundary pattern for a plain word; literal substring otherwise."""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-']*", needle):
        return re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)
    return None


def count_in_text(text: str, needle: str) -> int:
    pattern = _pattern_for(needle)
    if pattern is not None:
        return len(pattern.findall(text))
    return text.lower().count(needle.lower())


def _snippet(text: str, needle: str, index: int) -> str:
    start = max(0, index - 70)
    end = min(len(text), index + len(needle) + 130)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ") + suffix


def count_on_pages(db: Session, document_id: str, needle: str) -> list[PageHit]:
    """Case-insensitive occurrence counting on overlap-free page text.

    Only pages that contain the needle are loaded. Returns per-page counts.
    """
    if not needle:
        return []
    statement = (
        select(DocumentPage)
        .where(
            DocumentPage.document_id == document_id,
            DocumentPage.page_text.ilike(f"%{_escape(needle)}%", escape="\\"),
        )
        .order_by(DocumentPage.page_number)
    )
    hits: list[PageHit] = []
    for row in db.execute(statement).scalars().all():
        count = count_in_text(row.page_text, needle)
        if count <= 0:
            continue
        lowered = row.page_text.lower()
        index = lowered.find(needle.lower())
        hits.append(
            PageHit(
                page=row.page_number,
                count=count,
                snippet=_snippet(row.page_text, needle, index if index >= 0 else 0),
            )
        )
    return hits


def _has_pages(db: Session, document_id: str) -> bool:
    """Cheap check: does this document have overlap-free document_pages rows?"""
    statement = (
        select(DocumentPage.page_number)
        .where(DocumentPage.document_id == document_id)
        .limit(1)
    )
    return db.execute(statement).first() is not None


def search_pages(db: Session, document_id: str, needle: str, limit: int = 200) -> list[PageHit]:
    """All pages containing the needle (case-insensitive), with snippets.

    SQL-side ILIKE for documents with page rows; merged chunk text for legacy docs.
    """
    if not needle:
        return []
    if _has_pages(db, document_id):
        statement = (
            select(DocumentPage)
            .where(
                DocumentPage.document_id == document_id,
                DocumentPage.page_text.ilike(f"%{_escape(needle)}%", escape="\\"),
            )
            .order_by(DocumentPage.page_number)
            .limit(limit)
        )
        hits: list[PageHit] = []
        for row in db.execute(statement).scalars().all():
            index = row.page_text.lower().find(needle.lower())
            hits.append(
                PageHit(
                    page=row.page_number,
                    count=count_in_text(row.page_text, needle),
                    snippet=_snippet(row.page_text, needle, index if index >= 0 else 0),
                )
            )
        return hits

    hits = []
    lowered = needle.lower()
    for page, text in legacy_page_texts(db, document_id).items():
        index = text.lower().find(lowered)
        if index < 0:
            continue
        hits.append(
            PageHit(page=page, count=count_in_text(text, needle), snippet=_snippet(text, needle, index))
        )
        if len(hits) >= limit:
            break
    return sorted(hits, key=lambda hit: hit.page)


def page_texts(db: Session, document_id: str) -> dict[int, str] | None:
    """All page texts in order, or None when the document has no document_pages rows."""
    statement = (
        select(DocumentPage.page_number, DocumentPage.page_text)
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number)
    )
    rows = db.execute(statement).all()
    if not rows:
        return None
    return {int(number): text for number, text in rows}


def merge_overlapping(texts: list[str]) -> str:
    """Re-join ordered overlapping chunks, removing duplicated overlap regions."""
    if not texts:
        return ""
    acc = texts[0]
    for text in texts[1:]:
        max_k = min(len(acc), len(text), 400)
        merged = False
        for k in range(max_k, 0, -1):
            if acc[-k:] == text[:k]:
                acc += text[k:]
                merged = True
                break
        if not merged:
            acc += "\n" + text
    return acc


def legacy_page_texts(db: Session, document_id: str) -> dict[int, str]:
    """Page text reconstructed from (overlapping) chunks for documents indexed
    before the document_pages table existed."""
    statement = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.page_number, DocumentChunk.chunk_index)
    )
    chunks = db.execute(statement).scalars().all()
    grouped: dict[int, list[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.page_number, []).append(chunk.text)
    return {page: merge_overlapping(texts) for page, texts in grouped.items()}


def resolve_page_texts(db: Session, document_id: str) -> dict[int, str]:
    """Prefer overlap-free page text; fall back to chunk reconstruction for legacy docs."""
    pages = page_texts(db, document_id)
    if pages is not None:
        return pages
    return legacy_page_texts(db, document_id)


def count_in_document(db: Session, document_id: str, needle: str) -> list[PageHit]:
    """Count occurrences across the document, falling back to merged chunks for legacy docs."""
    if _has_pages(db, document_id):
        return count_on_pages(db, document_id, needle)
    hits: list[PageHit] = []
    for page, text in legacy_page_texts(db, document_id).items():
        count = count_in_text(text, needle)
        if count <= 0:
            continue
        index = text.lower().find(needle.lower())
        hits.append(PageHit(page=page, count=count, snippet=_snippet(text, needle, index)))
    return sorted(hits, key=lambda hit: hit.page)


# --- lightweight structure -------------------------------------------------

_LABELED_RE = re.compile(r"^[ \t]*((?:chapter|section|part|lesson|lecture)[ \t]+\d+(?:\.\d+)*)", re.IGNORECASE | re.MULTILINE)
_BIB_RE = re.compile(r"^[ \t]*(references|bibliography|works cited)[ \t]*$", re.IGNORECASE | re.MULTILINE)
_CITE_LINE_RE = re.compile(
    r"^\s*(?:\[\d+\]|\(\d+\)|\d+\.|[A-Z][A-Za-z'’\-]+,)\s*\S"
)


def find_labeled_sections(page_texts_map: dict[int, str]) -> list[tuple[str, str, int]]:
    """Detect chapter/section headings anywhere in the text.

    Returns (kind, number, page) in document order.
    """
    found: list[tuple[str, str, int]] = []
    for page in sorted(page_texts_map):
        for match in _LABELED_RE.finditer(page_texts_map[page]):
            words = match.group(1).split()
            kind = words[0].lower()
            number = words[1]
            found.append((kind, number, page))
    return found


def page_range_for(
    page_texts_map: dict[int, str], kind: str, number: str
) -> tuple[int, int] | None:
    """Page range of a labeled section: its heading page until the next heading."""
    if not page_texts_map:
        return None
    sections = find_labeled_sections(page_texts_map)
    target_index = None
    for index, (found_kind, found_number, _) in enumerate(sections):
        if found_kind == kind and found_number == number:
            target_index = index
            break
    if target_index is None:
        return None
    start = sections[target_index][2]
    end = sections[target_index + 1][2] - 1 if target_index + 1 < len(sections) else max(page_texts_map)
    return start, max(start, min(end, max(page_texts_map)))


def bibliography_range(page_texts_map: dict[int, str]) -> tuple[int, int] | None:
    """Page range of the references/bibliography section, if a heading exists."""
    if not page_texts_map:
        return None
    start: int | None = None
    for page in sorted(page_texts_map):
        if _BIB_RE.search(page_texts_map[page]):
            start = page
            break
    if start is None:
        return None
    return start, max(page_texts_map)


def bib_entries(text: str) -> list[str]:
    """Split bibliography text into entry lines (heuristic, line-accurate).

    An entry starts on any line whose beginning looks like a citation (bracketed
    number, numbered list marker, or "Author," style) and continues until the next
    such marker. Non-entry lines (headers, page numbers) are skipped.
    """
    entries: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if _CITE_LINE_RE.match(stripped):
            if current:
                entries.append(" ".join(current))
            current = [stripped]
        elif current:
            # Continuation only when indented or lowercase-ish; otherwise a stray
            # header/page-number line, which we drop.
            if line[:1] in (" ", "\t") or stripped[:1].islower():
                current.append(stripped)
            else:
                entries.append(" ".join(current))
                current = []
    if current:
        entries.append(" ".join(current))
    return [entry for entry in entries if len(entry) >= 30]


def count_bibliography_entries(text: str) -> int:
    """Heuristic count of bibliography entries (numbered, bracketed, or Author, Year)."""
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < 40:
            continue
        if _CITE_LINE_RE.match(stripped):
            count += 1
    return count
