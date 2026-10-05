from types import SimpleNamespace

from app import cache
from app.models import SemanticCache


class FakeResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class FakeSession:
    def __init__(self, row):
        self.row = row
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return FakeResult(self.row)


def _entry(answer: str = "cached answer") -> SemanticCache:
    return SemanticCache(
        document_id="doc-1",
        question="What problem does this paper address?",
        question_embedding=[0.1] * 4,
        answer=answer,
    )


def test_cache_hit_when_similarity_meets_threshold():
    session = FakeSession((_entry(), 0.04))  # similarity = 0.96
    hit = cache.lookup(session, "doc-1", [0.1] * 4, threshold=0.90)
    assert hit is not None
    assert hit.answer == "cached answer"


def test_cache_miss_when_similarity_below_threshold():
    session = FakeSession((_entry(), 0.25))  # similarity = 0.75
    assert cache.lookup(session, "doc-1", [0.1] * 4, threshold=0.90) is None


def test_cache_miss_when_no_rows():
    assert cache.lookup(FakeSession(None), "doc-1", [0.1] * 4, threshold=0.90) is None


def test_cache_is_scoped_to_document():
    session = FakeSession((_entry(), 0.0))
    cache.lookup(session, "doc-2", [0.1] * 4, threshold=0.90)
    compiled = str(session.statement)
    assert "document_id" in compiled
