"""
Shared embedding utility — module-level lazy singleton.

Loads the sentence-transformers model once on first call.
Returns plain Python lists, not numpy arrays, because ChromaDB expects lists.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a plain list[float]."""
    embedding = _get_model().encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns list[list[float]]."""
    embeddings = _get_model().encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
