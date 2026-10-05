"""HTTP API tests: upload lifecycle (pages/meta/warning/dedup) and chat compatibility."""

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app import llm, rag
from app.database import get_db
from app.main import app as api_app
from app.models import EMBEDDING_DIM, Document, DocumentPage
from app.schemas import EvidenceVerdict
from tests.support import fake_embed, seed_document


def _pdf(text: str = "Hello GuardRAG body text", meta: dict | None = None) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    if meta:
        doc.set_metadata(meta)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(pg_engine, monkeypatch):
    """TestClient with upload sessions bound to the test DB; no real embeddings."""
    from app import main

    from sqlalchemy.orm import sessionmaker

    maker = sessionmaker(bind=pg_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(main, "SessionLocal", maker)
    monkeypatch.setattr(
        main, "embed_texts", lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts]
    )
    return TestClient(api_app)


def test_upload_stores_pages_metadata_and_chunks(client, db):
    content = _pdf(meta={"title": "Deep Study", "author": "Ada"})
    response = client.post(
        "/api/upload", files={"file": ("paper.pdf", content, "application/pdf")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["reused"] is False
    assert payload["pages"] == 1
    assert payload["warning"] is None

    document_id = payload["document_id"]
    pages = db.execute(
        select(DocumentPage).where(DocumentPage.document_id == document_id)
    ).scalars().all()
    assert len(pages) == 1
    assert "Hello GuardRAG" in pages[0].page_text

    document = db.get(Document, document_id)
    assert document is not None
    assert document.meta_json is not None
    assert "Deep Study" in document.meta_json
    assert document.file_hash is not None
    assert document.chunks  # chunk rows were written


def test_upload_same_bytes_is_deduplicated(client, db):
    content = _pdf(text="Dedup target text")
    first = client.post(
        "/api/upload", files={"file": ("paper.pdf", content, "application/pdf")}
    ).json()
    second = client.post(
        "/api/upload", files={"file": ("copy.pdf", content, "application/pdf")}
    ).json()
    assert second["status"] == "already_indexed"
    assert second["reused"] is True
    assert second["document_id"] == first["document_id"]


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/api/upload", files={"file": ("notes.txt", b"not a pdf", "application/pdf")}
    )
    assert response.status_code == 400


def test_chat_accepts_legacy_two_field_body(pg_engine, monkeypatch):
    session = Session(pg_engine)
    document = seed_document(
        session, {1: "The guard refuses when evidence is insufficient for the claim."}
    )
    monkeypatch.setattr(rag, "embed_text", fake_embed)
    monkeypatch.setattr(rag.llm, "retrieval_passage", lambda q, ctx: None)
    monkeypatch.setattr(
        rag.evidence,
        "check_evidence",
        lambda q, c: EvidenceVerdict(sufficient=True, reason="ok"),
    )
    monkeypatch.setattr(llm, "chat", lambda messages, temperature=0.0: "It refuses. [p. 1]")
    api_app.dependency_overrides[get_db] = lambda: session
    try:
        client = TestClient(api_app)
        response = client.post(
            "/api/chat",
            json={
                "document_id": document.id,
                "question": "What happens when evidence is insufficient?",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "answered"
        assert payload["answer"] == "It refuses. [p. 1]"
        assert payload["cache_hit"] is False
        assert payload["evidence_sufficient"] is True
        assert payload["query_mode"] == "NORMAL"
        assert "completeness" in payload
        assert [source["page"] for source in payload["sources"]] == [1]
    finally:
        api_app.dependency_overrides.clear()
        session.rollback()
        session.execute(text("DELETE FROM documents"))
        session.commit()
        session.close()


def test_chat_unknown_document_returns_404(pg_engine):
    session = Session(pg_engine)
    api_app.dependency_overrides[get_db] = lambda: session
    try:
        client = TestClient(api_app)
        response = client.post(
            "/api/chat", json={"document_id": "missing-doc", "question": "anything"}
        )
        assert response.status_code == 404
    finally:
        api_app.dependency_overrides.clear()
        session.close()
