from functools import lru_cache
from typing import Sequence

import numpy as np

from .config import get_settings

# Preferred and fallback models are both 384-dimensional.
_KNOWN_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def infer_dimension() -> int:
    """Resolve the embedding dimension for the configured model."""
    name = get_settings().embedding_model
    if name in _KNOWN_DIMENSIONS:
        return _KNOWN_DIMENSIONS[name]
    return int(_load_model().get_sentence_embedding_dimension())


@lru_cache
def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model, device="cpu")


def warm_up() -> None:
    """Load the embedding model once at backend startup."""
    _load_model()


def embed_text(text: str) -> list[float]:
    model = _load_model()
    vector = model.encode(text, normalize_embeddings=True)
    return _to_list(vector)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _load_model()
    vectors = model.encode(list(texts), normalize_embeddings=True, batch_size=32)
    return [_to_list(v) for v in np.atleast_2d(vectors)]


def _to_list(vector: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(vector, dtype=np.float32)]
