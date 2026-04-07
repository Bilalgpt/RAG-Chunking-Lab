"""
End-to-end integration tests for the indexing + retrieval pipeline.

These tests use the real ChromaDB in-memory store and the real embedding model.
No LLM API key is required — answer generation is skipped in favour of testing
retrieval directly.

Each test creates a unique collection and deletes it in teardown so tests
remain independent and the in-process ChromaDB singleton stays clean.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.schemas import IndexRequest
from services.chunking_service import index_document, list_collections
from utils.embedder import embed_text
from vector_store import get_vector_store

# ── shared test document ──────────────────────────────────────────────────────

_DOC = (
    "Artificial intelligence is transforming many industries across the globe. "
    "Machine learning models can now perform tasks once considered uniquely human. "
    "Natural language processing enables computers to understand and generate text. "
    "Large language models are trained on billions of words from the internet. "
    "Retrieval-augmented generation combines language models with external knowledge. "
    "This approach grounds answers in retrieved documents and reduces hallucinations. "
    "The quality of retrieval depends heavily on how documents are chunked and indexed. "
    "Fixed-size chunking splits text into equal windows with configurable overlap. "
    "Semantic chunking places boundaries where embedding similarity drops sharply. "
    "Proposition chunking decomposes text into atomic verifiable factual claims. "
) * 3  # ~300 words

_COLLECTION = "test_pipeline_fixed_size"


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup():
    """Delete the test collection before and after each test."""
    vs = get_vector_store()
    vs.delete_collection(_COLLECTION)
    yield
    vs.delete_collection(_COLLECTION)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_index_returns_valid_response():
    request = IndexRequest(
        document_text=_DOC,
        technique="fixed_size",
        params={"chunk_size": 40, "overlap": 8},
        collection_name=_COLLECTION,
    )
    response = index_document(request)

    assert response.collection_name == _COLLECTION
    assert response.technique == "fixed_size"
    assert response.chunk_count > 0
    assert response.avg_chunk_size > 0.0


def test_index_and_query_returns_results():
    # Index
    request = IndexRequest(
        document_text=_DOC,
        technique="fixed_size",
        params={"chunk_size": 40, "overlap": 8},
        collection_name=_COLLECTION,
    )
    index_response = index_document(request)
    assert index_response.chunk_count > 0

    # Query directly via vector store (no LLM)
    query_embedding = embed_text("What is machine learning?")
    vs = get_vector_store()
    results = vs.query(query_embedding, _COLLECTION, top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)
    assert all(isinstance(r["text"], str) and r["text"] for r in results)
    assert all(isinstance(r["score"], float) for r in results)


def test_query_scores_are_in_valid_range():
    index_document(IndexRequest(
        document_text=_DOC,
        technique="fixed_size",
        params={"chunk_size": 40, "overlap": 8},
        collection_name=_COLLECTION,
    ))
    query_embedding = embed_text("natural language processing")
    results = get_vector_store().query(query_embedding, _COLLECTION, top_k=5)

    for r in results:
        # Cosine similarity in [-1, 1]; expect positive for a related query
        assert -1.0 <= r["score"] <= 1.0


def test_reindex_overwrites_collection():
    req = IndexRequest(
        document_text=_DOC,
        technique="fixed_size",
        params={"chunk_size": 40, "overlap": 8},
        collection_name=_COLLECTION,
    )
    r1 = index_document(req)
    r2 = index_document(req)  # second call should delete + re-index

    # Both should succeed and return the same chunk count
    assert r1.chunk_count == r2.chunk_count

    vs = get_vector_store()
    qemb = embed_text("machine learning")
    results = vs.query(qemb, _COLLECTION, top_k=10)
    # Should not have duplicate IDs (collection was cleared before re-adding)
    assert len(results) <= r2.chunk_count


def test_collection_appears_in_list_after_indexing():
    assert _COLLECTION not in list_collections()

    index_document(IndexRequest(
        document_text=_DOC,
        technique="fixed_size",
        params={"chunk_size": 40, "overlap": 8},
        collection_name=_COLLECTION,
    ))

    assert _COLLECTION in list_collections()


def test_query_missing_collection_returns_empty():
    query_embedding = embed_text("machine learning")
    results = get_vector_store().query(query_embedding, "nonexistent_collection_xyz", top_k=5)
    assert results == []


def test_index_recursive_chunker():
    """Regression test: recursive chunker flows through service correctly."""
    response = index_document(IndexRequest(
        document_text=_DOC,
        technique="recursive",
        params={"max_size": 50, "overlap": 10},
        collection_name=_COLLECTION,
    ))
    assert response.chunk_count > 0
    assert response.technique == "recursive"


def test_index_hierarchical_chunker():
    """Regression test: hierarchical chunker (leaf nodes only) stores correctly."""
    doc_with_paragraphs = (
        "Artificial intelligence transforms industries.\n\n"
        "Machine learning enables computers to learn from data.\n\n"
        "Natural language processing handles human text understanding.\n\n"
        "Retrieval augmented generation grounds LLMs in external knowledge."
    )
    response = index_document(IndexRequest(
        document_text=doc_with_paragraphs,
        technique="hierarchical",
        params={},
        collection_name=_COLLECTION,
    ))
    assert response.chunk_count > 0
    assert response.technique == "hierarchical"
