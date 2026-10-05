"""Hierarchical map-reduce document summaries for global questions.

Built lazily on the first GLOBAL question and cached in ``document_summaries`` so
ordinary questions and repeat global questions never re-scan the whole document.
A 1000-page book is summarized in ~30-page map calls (≈34 calls) once, not per query,
and never by sending the whole document to the LLM in one prompt.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import llm
from .models import DocumentSummary
from .search import resolve_page_texts

PART_MAX_PAGES = 30
PART_MAX_CHARS = 24_000

_MAP_PROMPT = """\
You are summarizing part of a document for later question answering.
You receive pages {page_start}-{page_end} of a larger document.

Write a faithful, dense summary (150-400 words) that preserves:
- the topics/sections covered
- key claims, facts, numbers, names, dates
- any list of items (contributions, findings, recommendations) — keep ALL items
- structure (chapter/section headings) when present

Use ONLY the provided page text. Never invent content. Plain text only.
"""

_REDUCE_PROMPT = """\
You are combining partial summaries of one document (in page order) into a single
document-level summary for later question answering.

Write a faithful, structured summary (max 500 words) that preserves:
- the document's purpose and structure
- all major topics, claims, results, and conclusions
- every enumerated item mentioned by the partial summaries (contributions, findings…)

Use ONLY the provided partial summaries. Never invent content. Plain text only.
"""


def _parts(page_map: dict[int, str]) -> list[tuple[int, int, str]]:
    """Group pages into map windows bounded by both page count and character budget."""
    parts: list[tuple[int, int, str]] = []
    start: int | None = None
    end: int | None = None
    buffer: list[str] = []
    chars = 0
    for page in sorted(page_map):
        text = page_map[page]
        if start is None:
            start = page
        would_exceed = chars + len(text) > PART_MAX_CHARS and buffer
        if would_exceed or (end is not None and end - start + 1 >= PART_MAX_PAGES):
            parts.append((start, end, "\n".join(buffer)))
            start, end, buffer, chars = page, page, [], 0
        end = page
        buffer.append(f"[page {page}]\n{text}")
        chars += len(text)
    if start is not None and end is not None and buffer:
        parts.append((start, end, "\n".join(buffer)))
    return parts


def _stored(db: Session, document_id: str, scope: str) -> DocumentSummary | None:
    statement = select(DocumentSummary).where(
        DocumentSummary.document_id == document_id,
        DocumentSummary.scope == scope,
    )
    return db.execute(statement).scalar_one_or_none()


def document_summary(db: Session, document_id: str) -> tuple[str, list[tuple[int, int]]]:
    """Return (document summary, part page ranges). Builds and caches on first use."""
    cached = _stored(db, document_id, "document")
    if cached is not None:
        return cached.summary, _ranges(db, document_id)

    page_map = resolve_page_texts(db, document_id)
    if not page_map:
        raise llm.LLMError("No extracted text available for summarization.")

    parts = _parts(page_map)
    part_summaries: list[str] = []
    stored_ranges: list[tuple[int, int]] = []
    for index, (page_start, page_end, text) in enumerate(parts):
        summary = llm.chat(
            [
                {"role": "user", "content": _MAP_PROMPT.format(page_start=page_start, page_end=page_end)},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        db.add(
            DocumentSummary(
                document_id=document_id,
                scope=f"part:{index}",
                summary=summary,
                page_start=page_start,
                page_end=page_end,
            )
        )
        part_summaries.append(f"(pages {page_start}-{page_end})\n{summary}")
        stored_ranges.append((page_start, page_end))

    if len(part_summaries) == 1:
        document_text = part_summaries[0]
    else:
        combined = "\n\n".join(part_summaries)
        document_text = llm.chat(
            [{"role": "user", "content": _REDUCE_PROMPT}, {"role": "user", "content": combined}],
            temperature=0.0,
        )
    db.add(
        DocumentSummary(
            document_id=document_id,
            scope="document",
            summary=document_text,
            page_start=min(page_map),
            page_end=max(page_map),
        )
    )
    db.commit()
    return document_text, stored_ranges or [(min(page_map), max(page_map))]


def _ranges(db: Session, document_id: str) -> list[tuple[int, int]]:
    statement = (
        select(DocumentSummary.page_start, DocumentSummary.page_end)
        .where(
            DocumentSummary.document_id == document_id,
            DocumentSummary.scope.like("part:%"),
        )
        .order_by(DocumentSummary.page_start)
    )
    ranges = [
        (int(start), int(end))
        for start, end in db.execute(statement).all()
        if start is not None and end is not None
    ]
    return ranges
