from unittest.mock import MagicMock

from app import evidence, llm
from app.retriever import RetrievedChunk
from app.schemas import EvidenceVerdict


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=1,
        document_id="doc-1",
        chunk_index=0,
        page_number=3,
        text=text,
        similarity=0.9,
    )


def test_evidence_sufficient(monkeypatch):
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda messages, schema: EvidenceVerdict(
            sufficient=True,
            reason="The retrieved sections explicitly describe the research problem.",
        ),
    )
    verdict = evidence.check_evidence(
        "What problem does this paper address?", [_chunk("We address the problem of ...")]
    )
    assert verdict.sufficient is True
    assert "research problem" in verdict.reason


def test_evidence_insufficient(monkeypatch):
    monkeypatch.setattr(
        llm,
        "chat_json",
        lambda messages, schema: EvidenceVerdict(
            sufficient=False,
            reason="The retrieved PDF content does not contain the requested information.",
        ),
    )
    verdict = evidence.check_evidence("What was the author's phone number?", [_chunk("Abstract ...")])
    assert verdict.sufficient is False


def test_evidence_no_chunks_skips_llm(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(llm, "chat_json", spy)
    verdict = evidence.check_evidence("anything", [])
    assert verdict.sufficient is False
    spy.assert_not_called()
