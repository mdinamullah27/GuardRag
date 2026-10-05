"""Semantic answer cache with query-mode and reader-context isolation.

An exhaustive question must never be answered from a normal-mode cache entry
(and vice versa), and an answer produced under one text selection is never
served for another. Legacy rows written before isolation existed carry NULL
``query_mode``/``context_hash`` and are only reusable for plain NORMAL questions
with no selection.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .models import SemanticCache

NORMAL_MODE = "NORMAL"


def context_hash(selected_text: str | None) -> str | None:
    """Stable key for the reader's selected text; None when nothing is selected."""
    if not selected_text or not selected_text.strip():
        return None
    normalized = " ".join(selected_text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def lookup(
    db: Session,
    document_id: str,
    question_embedding: list[float],
    threshold: float | None = None,
    *,
    mode: str = NORMAL_MODE,
    context_key: str | None = None,
) -> SemanticCache | None:
    """Return a cached answer for this document whose question is semantically similar.

    Only entries of the same query mode and the same selection context are eligible.
    """
    from .config import get_settings

    cutoff = threshold if threshold is not None else get_settings().cache_similarity_threshold
    distance = SemanticCache.question_embedding.cosine_distance(question_embedding).label("distance")

    if mode == NORMAL_MODE:
        mode_condition = or_(SemanticCache.query_mode.is_(None), SemanticCache.query_mode == mode)
    else:
        mode_condition = SemanticCache.query_mode == mode

    if context_key is None:
        context_condition = SemanticCache.context_hash.is_(None)
    else:
        context_condition = SemanticCache.context_hash == context_key

    statement = (
        select(SemanticCache, distance)
        .where(
            SemanticCache.document_id == document_id,
            mode_condition,
            context_condition,
        )
        .order_by(distance)
        .limit(1)
    )
    row = db.execute(statement).first()
    if row is None:
        return None
    entry, row_distance = row
    similarity = 1.0 - float(row_distance)
    return entry if similarity >= cutoff else None


def store(
    db: Session,
    document_id: str,
    question: str,
    question_embedding: list[float],
    answer: str,
    *,
    mode: str = NORMAL_MODE,
    context_key: str | None = None,
) -> SemanticCache:
    entry = SemanticCache(
        document_id=document_id,
        question=question,
        question_embedding=question_embedding,
        answer=answer,
        query_mode=mode,
        context_hash=context_key,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
