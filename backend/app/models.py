import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base
from .embeddings import infer_dimension

EMBEDDING_DIM = infer_dimension()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # SHA-256 of the uploaded file bytes; nullable for documents indexed before dedup existed.
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default=None)
    # JSON-encoded PDF metadata (title/author/subject/…) captured at upload.
    meta_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentPage(Base):
    """Extracted text per page.

    Kept alongside (overlapping) chunks so exact search and counting can run over the
    original, overlap-free page text with reliable page attribution.
    """

    __tablename__ = "document_pages"

    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    page_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_text: Mapped[str] = mapped_column(Text, nullable=False)


class DocumentSummary(Base):
    """Cached hierarchical summaries for global/document-wide questions."""

    __tablename__ = "document_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    scope: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")


class SemanticCache(Base):
    __tablename__ = "semantic_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # Query class the answer was produced for (NORMAL, EXHAUSTIVE, COUNT, …).
    # NULL for entries stored before routing existed — they only serve NORMAL questions.
    query_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, default=None)
    # SHA-256 of the normalized selected_text context (NULL when nothing was selected).
    context_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
