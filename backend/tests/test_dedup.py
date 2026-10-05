from app.main import find_document_by_hash
from app.pdf import compute_file_hash
from app.schemas import UploadResponse


class FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class FakeSession:
    def __init__(self, row=None):
        self.row = row
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return FakeResult(self.row)


def test_file_hash_is_deterministic():
    assert compute_file_hash(b"%PDF-1.4 same bytes") == compute_file_hash(b"%PDF-1.4 same bytes")


def test_file_hash_differs_for_different_files():
    assert compute_file_hash(b"%PDF-1.4 file A") != compute_file_hash(b"%PDF-1.4 file B")


def test_find_document_by_hash_returns_existing_document():
    document = object()
    session = FakeSession(row=document)
    assert find_document_by_hash(session, "abc123") is document
    assert "file_hash" in str(session.statement)


def test_find_document_by_hash_returns_none_when_absent():
    assert find_document_by_hash(FakeSession(row=None), "abc123") is None


def test_upload_response_defaults_for_new_documents():
    response = UploadResponse(document_id="d1", filename="a.pdf", pages=3)
    assert response.status == "ready"
    assert response.reused is False


def test_upload_response_supports_reuse():
    response = UploadResponse(
        document_id="d1", filename="a.pdf", pages=3, status="already_indexed", reused=True
    )
    assert response.status == "already_indexed"
    assert response.reused is True
