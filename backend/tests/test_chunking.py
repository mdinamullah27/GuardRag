import pytest

from app.chunking import EmptyDocumentError, chunk_pages


def test_chunking_preserves_page_numbers():
    pages = [(1, "alpha " * 50), (2, "beta " * 50)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
    assert chunks
    assert {c.page_number for c in chunks} == {1, 2}
    assert [c.chunk_index for c in chunks if c.page_number == 1] == list(
        range(len([c for c in chunks if c.page_number == 1]))
    )


def test_chunks_respect_size_limit():
    pages = [(1, "word " * 500)]
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    pages = [(1, " ".join(f"w{i}" for i in range(200)))]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=60)
    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    shared = set(first_words) & set(second_words)
    assert shared
    assert int(second_words[0][1:]) < int(first_words[-1][1:])


def test_skips_empty_pages():
    pages = [(1, ""), (2, "actual content here")]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_raises_on_no_text():
    with pytest.raises(EmptyDocumentError):
        chunk_pages([(1, "   "), (2, "")])
