from dataclasses import dataclass

from .config import get_settings


class EmptyDocumentError(Exception):
    """Raised when a PDF yields no extractable text."""


@dataclass
class Chunk:
    chunk_index: int
    page_number: int
    text: str


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split page text into overlapping word-based chunks, preserving page numbers."""
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    overlap = max(0, min(overlap, size - 1))

    chunks: list[Chunk] = []
    for page_number, text in pages:
        if not text:
            continue
        chunks.extend(_split_page(page_number, text, size, overlap))

    if not chunks:
        raise EmptyDocumentError("No extractable text found in this PDF.")
    return chunks


def _split_page(page_number: int, text: str, size: int, overlap: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    pieces: list[Chunk] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words):
            next_len = length + len(words[end]) + 1
            if length > 0 and next_len > size:
                break
            length = next_len
            end += 1
        if end == start:
            end = start + 1

        piece = " ".join(words[start:end])
        pieces.append(Chunk(chunk_index=len(pieces), page_number=page_number, text=piece))

        if end >= len(words):
            break
        step = max(1, (end - start) - _words_fit(words[start:end], overlap))
        start += step
    return pieces


def _words_fit(words: list[str], overlap_chars: int) -> int:
    """Number of trailing words that fit within the overlap budget."""
    count = 0
    length = 0
    for word in reversed(words):
        next_len = length + len(word) + 1
        if count > 0 and next_len > overlap_chars:
            break
        length = next_len
        count += 1
    return count
