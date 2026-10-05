import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import pdf, rag
from .chunking import EmptyDocumentError, chunk_pages
from .config import get_settings, persist_env
from .database import SessionLocal, get_db, init_db
from .embeddings import embed_texts, warm_up
from .llm import LLMError
from .models import Document, DocumentChunk, DocumentPage
from .pdf import PDFError, compute_file_hash
from .schemas import ChatRequest, ChatResponse, LlmSettingsUpdate, LlmSettingsView, UploadResponse

logger = logging.getLogger("guardrag")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except Exception:
        logger.exception("Database initialization failed")
        raise
    warm_up()
    yield


app = FastAPI(title="GuardRAG", tagline="Chat with your PDF — without guessing.", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _sanitize(value: Any) -> Any:
    """Make validation errors JSON-safe (raw request bodies may be bytes)."""
    if isinstance(value, bytes):
        return f"<binary content: {len(value)} bytes>"
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(_sanitize(exc.errors()))})


def find_document_by_hash(db: Session, file_hash: str) -> Document | None:
    """Return the already-indexed document for this file fingerprint, if any."""
    from sqlalchemy import select

    return db.execute(select(Document).where(Document.file_hash == file_hash)).scalar_one_or_none()


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()
    try:
        pdf.validate_pdf(file.filename, content)
    except PDFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_hash = compute_file_hash(content)
    db: Session = SessionLocal()
    try:
        existing = find_document_by_hash(db, file_hash)
        if existing is not None:
            return UploadResponse(
                document_id=existing.id,
                filename=existing.filename,
                pages=existing.page_count,
                status="already_indexed",
                reused=True,
            )

        try:
            extraction = pdf.extract_document(content)
            chunks = chunk_pages(extraction.pages)
        except PDFError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmptyDocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Embeddings are generated only for documents that are not indexed yet.
        embeddings = embed_texts([chunk.text for chunk in chunks])

        document = Document(
            filename=file.filename or "document.pdf",
            page_count=len(extraction.pages),
            file_hash=file_hash,
            meta_json=pdf.metadata_json(extraction.metadata),
        )
        db.add(document)
        db.flush()
        db.add_all(
            DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                text=chunk.text,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        )
        # Overlap-free page text powers exact search, counting, and global summaries.
        db.add_all(
            DocumentPage(document_id=document.id, page_number=number, page_text=text)
            for number, text in extraction.pages
        )
        db.commit()
        return UploadResponse(
            document_id=document.id,
            filename=document.filename,
            pages=len(extraction.pages),
            status="ready",
            reused=False,
            warning=extraction.warning,
        )
    except IntegrityError:
        # Concurrent upload of the same file won the unique-index race.
        db.rollback()
        existing = find_document_by_hash(db, file_hash)
        if existing is None:
            raise HTTPException(status_code=409, detail="Document already exists.") from None
        return UploadResponse(
            document_id=existing.id,
            filename=existing.filename,
            pages=existing.page_count,
            status="already_indexed",
            reused=True,
        )
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    try:
        return rag.ask(
            db,
            request.document_id,
            request.question.strip(),
            current_page=request.current_page,
            selected_text=request.selected_text,
        )
    except rag.DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _mask_key(key: str) -> str | None:
    if not key:
        return None
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:7]}…{key[-4:]}"


def _settings_view() -> LlmSettingsView:
    settings = get_settings()
    return LlmSettingsView(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        llm_api_key_masked=_mask_key(settings.llm_api_key),
    )


@app.get("/api/settings", response_model=LlmSettingsView)
def read_settings() -> LlmSettingsView:
    return _settings_view()


@app.post("/api/settings", response_model=LlmSettingsView)
def update_settings(payload: LlmSettingsUpdate) -> LlmSettingsView:
    """Update LLM settings at runtime; values are persisted to the backend .env file."""
    settings = get_settings()
    updates: dict[str, str] = {}
    if payload.llm_api_key is not None:
        updates["llm_api_key"] = payload.llm_api_key
        settings.llm_api_key = payload.llm_api_key
    if payload.llm_model is not None:
        updates["llm_model"] = payload.llm_model
        settings.llm_model = payload.llm_model
    if payload.llm_base_url is not None:
        updates["llm_base_url"] = payload.llm_base_url
        settings.llm_base_url = payload.llm_base_url
    if updates:
        try:
            persist_env(updates)
        except OSError:
            logger.warning("Could not persist LLM settings to .env (applied in memory only)")
    return _settings_view()
