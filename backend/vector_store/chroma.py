import chromadb

from chunkers.base import Chunk
from .base import BaseVectorStore

# Module-level singleton — one client shared across all requests.
_client = chromadb.EphemeralClient()


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB in-memory vector store implementation."""

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection_name: str,
    ) -> None:
        collection = _client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            documents=[c.text for c in chunks],
            ids=[str(c.index) for c in chunks],
            metadatas=[_sanitize_metadata({**c.metadata, "chunk_index": c.index}) for c in chunks],
            embeddings=embeddings,
        )

    def query(
        self,
        query_embedding: list[float],
        collection_name: str,
        top_k: int,
    ) -> list[dict]:
        if not self.collection_exists(collection_name):
            return []

        collection = _client.get_collection(collection_name)
        n = min(top_k, collection.count())
        if n == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "score": float(1.0 - dist),  # cosine distance → similarity
                "metadata": meta,
                "index": meta.get("chunk_index", 0),
            })
        return output

    def delete_collection(self, collection_name: str) -> None:
        if self.collection_exists(collection_name):
            _client.delete_collection(collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        try:
            _client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def list_collections(self) -> list[str]:
        return [c.name for c in _client.list_collections()]

    def get_all_chunks(self, collection_name: str) -> list[dict]:
        if not self.collection_exists(collection_name):
            return []
        collection = _client.get_collection(collection_name)
        result = collection.get(include=["documents", "metadatas"])
        output = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            output.append({
                "text": doc,
                "index": meta.get("chunk_index", 0),
                "metadata": meta,
            })
        output.sort(key=lambda c: c["index"])
        return output


def _sanitize_metadata(meta: dict) -> dict:
    """
    Keep only ChromaDB-compatible metadata values (str, int, float, bool).
    Truncates strings longer than 512 characters to avoid storage issues.
    """
    clean: dict = {}
    for k, v in meta.items():
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float)):
            clean[k] = v
        elif isinstance(v, str):
            clean[k] = v[:512] if len(v) > 512 else v
        # silently drop list, dict, None — not supported by ChromaDB
    return clean
