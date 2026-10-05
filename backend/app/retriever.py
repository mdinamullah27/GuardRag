from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import DocumentChunk


@dataclass
class RetrievedChunk:
    id: int
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    similarity: float


def retrieve(
    db: Session,
    document_id: str,
    question_embedding: list[float],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Cosine similarity search executed inside PostgreSQL via pgvector."""
    from .config import get_settings

    limit = top_k if top_k is not None else get_settings().top_k
    distance = DocumentChunk.embedding.cosine_distance(question_embedding).label("distance")
    statement = (
        select(DocumentChunk, distance)
        .where(DocumentChunk.document_id == document_id)
        .order_by(distance)
        .limit(limit)
    )
    rows = db.execute(statement).all()
    return [
        RetrievedChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            text=chunk.text,
            similarity=1.0 - float(row_distance),
        )
        for chunk, row_distance in rows
    ]
