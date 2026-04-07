from chunkers import get_chunker
from models.schemas import ChunkResponse, IndexRequest, IndexResponse
from services.llm_service import get_llm_client
from utils.embedder import embed_text
from vector_store import get_vector_store

_LLM_CHUNKERS = {"contextual", "proposition"}


def index_document(request: IndexRequest) -> IndexResponse:
    """
    Chunk a document, embed each chunk, and store in the vector store.

    For LLM-dependent techniques (contextual, proposition), an LLM client is
    instantiated from request.llm_provider / request.api_key and injected into
    the chunker. If no API key is provided, the chunker falls back gracefully.
    """
    chunker = get_chunker(request.technique)

    # Build kwargs for chunker.chunk() — strip reserved keys that aren't chunker params
    chunk_kwargs = {
        k: v for k, v in request.params.items()
        if k not in ("llm_client", "llm_provider", "api_key")
    }

    if request.technique in _LLM_CHUNKERS and request.api_key:
        chunk_kwargs["llm_client"] = get_llm_client(request.llm_provider, request.api_key)
        chunk_kwargs["provider"] = request.llm_provider

    chunks = chunker.chunk(request.document_text, **chunk_kwargs)

    if not chunks:
        return IndexResponse(
            collection_name=request.collection_name,
            chunk_count=0,
            avg_chunk_size=0.0,
            technique=request.technique,
        )

    # Compute embeddings — late_chunking pre-populates chunk.embedding
    embeddings = [
        chunk.embedding if chunk.embedding is not None else embed_text(chunk.text)
        for chunk in chunks
    ]

    vs = get_vector_store()
    if vs.collection_exists(request.collection_name):
        vs.delete_collection(request.collection_name)

    vs.add_chunks(chunks, embeddings, request.collection_name)

    avg_size = sum(len(c.text.split()) for c in chunks) / len(chunks)

    chunk_responses = [
        ChunkResponse(
            text=c.text,
            index=c.index,
            token_count=len(c.text.split()),
            metadata={k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
        )
        for c in chunks
    ]

    return IndexResponse(
        collection_name=request.collection_name,
        chunk_count=len(chunks),
        avg_chunk_size=round(avg_size, 2),
        technique=request.technique,
        chunks=chunk_responses,
    )


def list_collections() -> list[str]:
    """Return the names of all indexed collections in the vector store."""
    return get_vector_store().list_collections()


def get_chunks(collection_name: str) -> list[ChunkResponse]:
    """Return all chunks stored in the named collection."""
    raw = get_vector_store().get_all_chunks(collection_name)
    return [
        ChunkResponse(
            text=c["text"],
            index=c["index"],
            token_count=len(c["text"].split()),
            metadata={k: v for k, v in c["metadata"].items() if isinstance(v, (str, int, float, bool))},
        )
        for c in raw
    ]
