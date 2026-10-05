"""Shared helpers for database-backed tests (isolated guardrag_test database)."""

from __future__ import annotations

import json
import uuid
import zlib

from sqlalchemy.orm import Session

from app.models import EMBEDDING_DIM, Document, DocumentChunk, DocumentPage


def fake_embed(text: str) -> list[float]:
    """Deterministic per-question embedding.

    Identical questions map to the same vector (semantic cache hits work), while
    different questions map to orthogonal vectors (never cache-crossing).
    """
    vector = [0.0] * EMBEDDING_DIM
    vector[zlib.crc32(text.encode("utf-8")) % EMBEDDING_DIM] = 1.0
    return vector


def seed_document(
    db: Session,
    pages: dict[int, str],
    *,
    filename: str = "test.pdf",
    meta: dict | None = None,
    chunk_chars: int = 400,
) -> Document:
    """Insert a document with overlap-free page rows and simple fixed-embedding chunks."""
    document = Document(
        id=str(uuid.uuid4()),
        filename=filename,
        page_count=len(pages),
        meta_json=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(document)
    db.flush()
    embedding = [0.1] * EMBEDDING_DIM
    for page_number in sorted(pages):
        page_text = pages[page_number]
        db.add(DocumentPage(document_id=document.id, page_number=page_number, page_text=page_text))
        step = max(100, chunk_chars - 100)
        start = 0
        index = 0
        while start < len(page_text):
            piece = page_text[start : start + chunk_chars]
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=index,
                    page_number=page_number,
                    text=piece,
                    embedding=embedding,
                )
            )
            start += step
            index += 1
    db.commit()
    return document
