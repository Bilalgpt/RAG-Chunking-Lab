import time

from models.schemas import ChunkResponse, QueryRequest, QueryResponse
from services.llm_service import generate_answer, get_llm_client
from utils.embedder import embed_text
from vector_store import get_vector_store


def query_document(request: QueryRequest) -> QueryResponse:
    """
    Embed the query, retrieve top-k chunks, generate an LLM answer.

    LLM answer generation is best-effort: if the provider call fails, the
    answer field contains the error message and retrieved_chunks are still
    returned. Latency covers the full round-trip including LLM.
    """
    start = time.monotonic()

    # 1. Embed query
    query_embedding = embed_text(request.query)

    # 2. Retrieve from vector store
    vs = get_vector_store()
    results = vs.query(query_embedding, request.collection_name, request.top_k)

    # 3. Generate answer (best-effort)
    try:
        client = get_llm_client(request.llm_provider, request.api_key)
        answer = generate_answer(
            client,
            request.llm_provider,
            [r["text"] for r in results],
            request.query,
        )
    except Exception as exc:
        answer = f"Error generating answer: {exc}"

    latency_ms = round((time.monotonic() - start) * 1000, 2)

    # 4. Build response
    chunk_responses = [
        ChunkResponse(
            text=r["text"],
            index=int(r.get("index", i)),
            token_count=len(r["text"].split()),
            metadata={k: v for k, v in r.get("metadata", {}).items()},
        )
        for i, r in enumerate(results)
    ]

    return QueryResponse(
        answer=answer,
        retrieved_chunks=chunk_responses,
        latency_ms=latency_ms,
    )
