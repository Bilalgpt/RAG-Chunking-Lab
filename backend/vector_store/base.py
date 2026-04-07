from abc import ABC, abstractmethod

from chunkers.base import Chunk


class BaseVectorStore(ABC):
    """
    Abstract interface for all vector store backends.

    Services always hold a BaseVectorStore reference — never a concrete type.
    New backends plug in by subclassing this and registering in __init__.py.
    """

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection_name: str,
    ) -> None:
        """
        Store chunks with their pre-computed embeddings in the named collection.

        embeddings[i] corresponds to chunks[i]. Creates the collection if it
        does not already exist.
        """

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        collection_name: str,
        top_k: int,
    ) -> list[dict]:
        """
        Retrieve the top_k most similar chunks.

        Returns a list of dicts, each with keys:
            text (str), score (float), metadata (dict), index (int)
        Returns an empty list if the collection does not exist.
        """

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete the named collection. No-op if it does not exist."""

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Return True if the named collection exists."""

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return the names of all existing collections."""

    @abstractmethod
    def get_all_chunks(self, collection_name: str) -> list[dict]:
        """
        Return all chunks stored in the named collection.

        Each dict has keys: text (str), index (int), metadata (dict).
        Returns an empty list if the collection does not exist.
        """
