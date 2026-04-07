from .base import BaseVectorStore
from .chroma import ChromaVectorStore

_STORE_REGISTRY: dict[str, type[BaseVectorStore]] = {
    "chroma": ChromaVectorStore,
}


def get_vector_store(backend: str = "chroma") -> BaseVectorStore:
    """
    Factory — returns an instance of the requested vector store backend.

    This is the only place in the codebase that imports concrete store classes.
    To add a new backend: subclass BaseVectorStore, add it to _STORE_REGISTRY.
    """
    if backend not in _STORE_REGISTRY:
        raise ValueError(
            f"Unknown vector store backend: '{backend}'. "
            f"Available: {list(_STORE_REGISTRY.keys())}"
        )
    return _STORE_REGISTRY[backend]()


__all__ = ["BaseVectorStore", "get_vector_store"]
