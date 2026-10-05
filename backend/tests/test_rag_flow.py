from unittest.mock import MagicMock

import pytest

from app import cache, evidence, llm, rag
from app.models import SemanticCache
from app.retriever import RetrievedChunk
from app.schemas import EvidenceVerdict


class _Result:
    def scalars(self):
        return self

    def all(self):
        return ["The document opens with its title and abstract."]

    def first(self):
        return ("The document opens with its title and abstract.",)


class FakeDB:
    def __init__(self, document_id: str | None = "doc-1"):
        self._document_id = document_id
        self.committed = False

    def get(self, model, pk):
        if self._document_id is None or pk != self._document_id:
            return None
        return model(id=pk, filename="paper.pdf", page_count=5)

    def execute(self, statement):
        return _Result()

    def add(self, obj):
        pass

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        pass


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=11,
            document_id="doc-1",
            chunk_index=0,
            page_number=4,
            text="We address the problem of efficient retrieval.",
            similarity=0.93,
        ),
        RetrievedChunk(
            id=12,
            document_id="doc-1",
            chunk_index=1,
            page_number=7,
            text="Our main contribution is the evidence guard.",
            similarity=0.9,
        ),
    ]


def _patch_common(monkeypatch, chunks, sufficient=True):
    monkeypatch.setattr(rag, "embed_text", lambda q: [0.1] * 4)
    monkeypatch.setattr(rag, "retrieve", lambda db, doc_id, emb: chunks)
    monkeypatch.setattr(rag.llm, "retrieval_passage", lambda q, ctx: None)
    monkeypatch.setattr(
        rag.cache,
        "lookup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        rag.evidence,
        "check_evidence",
        lambda q, c: EvidenceVerdict(
            sufficient=sufficient,
            reason="ok" if sufficient else "The retrieved PDF content does not contain the requested information.",
        ),
    )


def test_cache_hit_skips_final_llm(monkeypatch):
    cached = SemanticCache(
        document_id="doc-1",
        question="What is the main problem discussed in this paper?",
        question_embedding=[0.1] * 4,
        answer="The paper addresses efficient retrieval.",
    )
    monkeypatch.setattr(rag, "embed_text", lambda q: [0.1] * 4)
    monkeypatch.setattr(rag.cache, "lookup", lambda *a, **k: cached)
    answer_spy = MagicMock()
    rewrite_spy = MagicMock(return_value=None)
    evidence_spy = MagicMock()
    monkeypatch.setattr(llm, "chat", answer_spy)
    monkeypatch.setattr(llm, "retrieval_passage", rewrite_spy)
    monkeypatch.setattr(rag.evidence, "check_evidence", evidence_spy)
    store_spy = MagicMock()
    monkeypatch.setattr(rag.cache, "store", store_spy)

    response = rag.ask(FakeDB(), "doc-1", "What is the main problem discussed in this paper?")

    assert response.status == "answered"
    assert response.cache_hit is True
    assert response.answer == "The paper addresses efficient retrieval."
    assert response.sources == []
    answer_spy.assert_not_called()
    rewrite_spy.assert_not_called()
    evidence_spy.assert_not_called()
    store_spy.assert_not_called()


def test_insufficient_evidence_skips_final_llm(monkeypatch):
    _patch_common(monkeypatch, _chunks(), sufficient=False)
    answer_spy = MagicMock()
    monkeypatch.setattr(llm, "chat", answer_spy)
    store_spy = MagicMock()
    monkeypatch.setattr(rag.cache, "store", store_spy)

    response = rag.ask(FakeDB(), "doc-1", "What was the author's phone number?")

    assert response.status == "insufficient_evidence"
    assert response.answer is None
    assert response.evidence_sufficient is False
    assert response.cache_hit is False
    assert response.reason
    answer_spy.assert_not_called()
    store_spy.assert_not_called()


def test_normal_answer_generates_and_caches(monkeypatch):
    _patch_common(monkeypatch, _chunks(), sufficient=True)
    monkeypatch.setattr(llm, "chat", lambda messages, temperature=0.0: "The evidence guard.")
    store_spy = MagicMock()
    monkeypatch.setattr(rag.cache, "store", store_spy)

    response = rag.ask(FakeDB(), "doc-1", "What does the evidence guard do?")

    assert response.status == "answered"
    assert response.cache_hit is False
    assert response.evidence_sufficient is True
    assert response.answer == "The evidence guard."
    assert [s.page for s in response.sources] == [4, 7]
    assert store_spy.call_count == 1


def test_unknown_document_raises(monkeypatch):
    monkeypatch.setattr(rag, "embed_text", lambda q: [0.1] * 4)
    with pytest.raises(rag.DocumentNotFound):
        rag.ask(FakeDB(document_id=None), "missing", "question")


def test_retrieval_query_uses_passage_when_available(monkeypatch):
    monkeypatch.setattr(rag.llm, "retrieval_passage", lambda q, ctx: "topical search query")
    assert rag._retrieval_query(FakeDB(), "doc-1", "generic question") == "topical search query"


def test_retrieval_query_falls_back_to_raw_question(monkeypatch):
    monkeypatch.setattr(rag.llm, "retrieval_passage", lambda q, ctx: None)
    assert rag._retrieval_query(FakeDB(), "doc-1", "generic question") == "generic question"


def test_retrieval_passage_prompt_forbids_negative_passages():
    # A passage like "the document does not provide details about X" embeds into
    # arbitrary text and hijacks the whole top-k (seen on a real 28-page paper).
    prompt = llm._RETRIEVAL_SYSTEM_PROMPT
    assert "Never state or imply" in prompt
    assert "lacks" in prompt
    assert "untrusted" in prompt.lower()
