# Vector Store — Abstraction Pattern

## What Lives Here
`base.py` defines the abstract interface. One file per backend implementation. `chroma.py` is the default and only required implementation.

## Abstract Interface (`base.py`)
All vector store implementations must subclass `BaseVectorStore` and implement:
```
add(chunks: list[Chunk], collection_name: str) -> None
query(embedding: list[float], collection_name: str, top_k: int) -> list[Chunk]
delete_collection(collection_name: str) -> None
collection_exists(collection_name: str) -> bool
```
No other methods are required by the interface. Extras are allowed but not called by services.

## ChromaDB Implementation (`chroma.py`)
- Uses an ephemeral in-memory client for simplicity (educational use, no persistence needed).
- Collection name is derived from: `{document_id}_{chunker_slug}`.
- Embeddings are stored alongside chunk text and metadata.

## How Services Use the Store
Services receive a `BaseVectorStore` instance via constructor injection. They never import `chroma.py` directly — always the base type. This is the only seam that needs changing to swap backends.

## Adding a New Vector Store
See `.claude/skills/adding-vector-store.md` for the full step-by-step.
