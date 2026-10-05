from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


def _build_engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create the pgvector extension and GuardRAG tables if missing.

    Applies additive migrations only (never drops or rewrites existing data):
    - `documents.file_hash` (nullable, unique index) for dedup
    - `documents.meta_json` for PDF metadata
    - `semantic_cache.query_mode` / `context_hash` for cache isolation
    - `document_summaries.page_start` / `page_end` for part ranges
    - HNSW index for vector search at 1000-page scale
    """
    from . import models  # noqa: F401  (register mappings)

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64)"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS meta_json TEXT"))
        # Postgres unique indexes allow multiple NULLs, so legacy rows stay valid.
        conn.execute(
            text("CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_file_hash ON documents (file_hash)")
        )
        conn.execute(
            text("ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS query_mode VARCHAR(32)")
        )
        conn.execute(
            text("ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS context_hash VARCHAR(64)")
        )
        conn.execute(
            text("ALTER TABLE document_summaries ADD COLUMN IF NOT EXISTS page_start INTEGER")
        )
        conn.execute(
            text("ALTER TABLE document_summaries ADD COLUMN IF NOT EXISTS page_end INTEGER")
        )
        # HNSW keeps vector search sub-linear on large (1000-page) documents.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
                "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
