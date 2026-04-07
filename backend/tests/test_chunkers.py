"""
Tests for all 7 chunking techniques.

Chunkers 1-4 (no LLM/model dependency at test time): full behavioral tests.
Chunkers 5-7 (model or LLM dependent): fallback-path tests only.

The semantic chunker does load the embedding model — first run will download
all-MiniLM-L6-v2 (~90 MB) once. Subsequent runs are instant from cache.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from chunkers.base import Chunk
from chunkers.fixed_size import FixedSizeChunker
from chunkers.recursive import RecursiveChunker
from chunkers.semantic import SemanticChunker
from chunkers.hierarchical import HierarchicalChunker
from chunkers.late_chunking import LateChunkingChunker
from chunkers.contextual import ContextualChunker
from chunkers.proposition import PropositionChunker

# ── shared test document (~120 words, 3 paragraphs) ──────────────────────────

TEST_DOC = """
Artificial intelligence has transformed many industries over the past decade.
Machine learning models can now perform tasks that were once considered uniquely human.

Natural language processing allows computers to understand and generate human text.
Large language models are trained on vast amounts of internet data.
They learn statistical patterns in language through self-supervised learning.

Retrieval-augmented generation combines language models with external knowledge bases.
This approach reduces hallucinations by grounding answers in retrieved documents.
The quality of retrieval depends heavily on how documents are chunked and embedded.
""".strip()

# A longer document (~500 words) for size-sensitive tests
LONG_DOC = (
    "The history of computing spans many decades and crosses many disciplines. "
    "Early mechanical calculators gave way to electronic computers in the mid twentieth century. "
    "The invention of the transistor revolutionized circuit design and enabled miniaturization. "
    "Integrated circuits further compressed millions of transistors onto a single chip. "
    "The microprocessor brought programmable computing to everyday devices. "
) * 20  # ~500 words


# ─────────────────────────────────────────────────────────────────────────────
# Chunker 1 — Fixed Size
# ─────────────────────────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_instantiation(self):
        assert FixedSizeChunker() is not None

    def test_returns_list_of_chunks(self):
        result = FixedSizeChunker().chunk(TEST_DOC)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_fields(self):
        for chunk in FixedSizeChunker().chunk(TEST_DOC):
            assert isinstance(chunk.text, str) and chunk.text
            assert isinstance(chunk.index, int)
            assert isinstance(chunk.metadata, dict)
            for key in ("chunk_size", "overlap", "start_word", "end_word"):
                assert key in chunk.metadata, f"Missing metadata key: {key}"

    def test_reasonable_chunk_count(self):
        chunks = FixedSizeChunker().chunk(LONG_DOC, chunk_size=100, overlap=10)
        assert 2 < len(chunks) < len(LONG_DOC.split())

    def test_overlap_words_shared(self):
        chunks = FixedSizeChunker().chunk(LONG_DOC, chunk_size=50, overlap=10)
        assert len(chunks) >= 2
        last_words_of_first = chunks[0].text.split()[-10:]
        first_words_of_second = chunks[1].text.split()[:10]
        # The last 10 words of chunk 0 must appear at the start of chunk 1
        assert last_words_of_first == first_words_of_second

    def test_indices_are_sequential(self):
        chunks = FixedSizeChunker().chunk(LONG_DOC)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_empty_text_returns_empty(self):
        assert FixedSizeChunker().chunk("") == []

    def test_overlap_gte_chunk_size_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker().chunk(TEST_DOC, chunk_size=100, overlap=100)


# ─────────────────────────────────────────────────────────────────────────────
# Chunker 2 — Recursive
# ─────────────────────────────────────────────────────────────────────────────

class TestRecursiveChunker:
    def test_instantiation(self):
        assert RecursiveChunker() is not None

    def test_returns_list_of_chunks(self):
        result = RecursiveChunker().chunk(TEST_DOC)
        assert isinstance(result, list) and len(result) > 0
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_fields(self):
        for chunk in RecursiveChunker().chunk(TEST_DOC):
            assert chunk.text and isinstance(chunk.text, str)
            assert isinstance(chunk.index, int)
            for key in ("separator_used", "depth", "original_text"):
                assert key in chunk.metadata

    def test_no_chunk_exceeds_max_size(self):
        max_size = 30
        chunks = RecursiveChunker().chunk(LONG_DOC, max_size=max_size, overlap=5)
        for chunk in chunks:
            # original_text (before overlap prepended) must be within max_size
            original = chunk.metadata["original_text"]
            assert len(original.split()) <= max_size, (
                f"original_text exceeds max_size: {len(original.split())} words"
            )

    def test_reasonable_chunk_count(self):
        chunks = RecursiveChunker().chunk(LONG_DOC, max_size=50, overlap=5)
        assert 2 < len(chunks) < len(LONG_DOC.split())

    def test_empty_text_returns_empty(self):
        assert RecursiveChunker().chunk("") == []

    def test_paragraph_boundary_respected(self):
        # A doc with clear paragraph breaks — depth 0 should handle it
        doc = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = RecursiveChunker().chunk(doc, max_size=100, overlap=0)
        assert len(chunks) == 3
        assert all("\n\n" not in c.text for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Chunker 3 — Semantic
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticChunker:
    def test_instantiation(self):
        assert SemanticChunker() is not None

    def test_returns_list_of_chunks(self):
        # NOTE: loads embedding model on first call (~90 MB download on first run)
        result = SemanticChunker().chunk(TEST_DOC)
        assert isinstance(result, list) and len(result) > 0
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_fields(self):
        for chunk in SemanticChunker().chunk(TEST_DOC):
            assert chunk.text and isinstance(chunk.text, str)
            for key in ("sentence_count", "avg_similarity", "min_similarity"):
                assert key in chunk.metadata

    def test_min_chunk_words_enforced(self):
        min_words = 20
        chunks = SemanticChunker().chunk(TEST_DOC, k=0.0, min_chunk_words=min_words)
        for chunk in chunks:
            assert len(chunk.text.split()) >= min_words, (
                f"Chunk below min_chunk_words: {chunk.text!r}"
            )

    def test_higher_k_fewer_chunks(self):
        chunks_low_k = SemanticChunker().chunk(LONG_DOC, k=0.0, min_chunk_words=1)
        chunks_high_k = SemanticChunker().chunk(LONG_DOC, k=3.0, min_chunk_words=1)
        # Higher k → threshold is lower → fewer boundaries → fewer chunks
        assert len(chunks_high_k) <= len(chunks_low_k)


# ─────────────────────────────────────────────────────────────────────────────
# Chunker 4 — Hierarchical
# ─────────────────────────────────────────────────────────────────────────────

class TestHierarchicalChunker:
    def test_instantiation(self):
        assert HierarchicalChunker() is not None

    def test_returns_list_of_chunks(self):
        result = HierarchicalChunker().chunk(TEST_DOC)
        assert isinstance(result, list) and len(result) > 0
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_fields(self):
        for chunk in HierarchicalChunker().chunk(TEST_DOC):
            assert chunk.text and isinstance(chunk.text, str)
            for key in ("level", "section_idx", "paragraph_idx",
                        "section_text", "paragraph_text"):
                assert key in chunk.metadata, f"Missing key: {key}"

    def test_all_chunks_are_level_3(self):
        for chunk in HierarchicalChunker().chunk(TEST_DOC):
            assert chunk.metadata["level"] == 3

    def test_paragraph_text_contains_chunk_text(self):
        for chunk in HierarchicalChunker().chunk(TEST_DOC):
            assert chunk.text in chunk.metadata["paragraph_text"] or \
                   chunk.text.replace("  ", " ") in chunk.metadata["paragraph_text"]

    def test_markdown_headers_create_sections(self):
        doc = "# Introduction\nAI is powerful.\n\n# Methods\nWe used Python."
        chunks = HierarchicalChunker().chunk(doc)
        section_indices = {c.metadata["section_idx"] for c in chunks}
        assert len(section_indices) >= 2

    def test_section_idx_and_paragraph_idx_are_ints(self):
        for chunk in HierarchicalChunker().chunk(TEST_DOC):
            assert isinstance(chunk.metadata["section_idx"], int)
            assert isinstance(chunk.metadata["paragraph_idx"], int)


# ─────────────────────────────────────────────────────────────────────────────
# Chunker 5 — Late Chunking (fallback path — no LLM, uses embedding model)
# ─────────────────────────────────────────────────────────────────────────────

class TestLateChunkingChunker:
    def test_instantiation(self):
        assert LateChunkingChunker() is not None

    def test_returns_chunks_with_embeddings(self):
        # Will try Jina; falls back to all-MiniLM-L6-v2 if unavailable
        chunks = LateChunkingChunker().chunk(TEST_DOC)
        assert isinstance(chunks, list) and len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_fields_and_embeddings(self):
        chunks = LateChunkingChunker().chunk(TEST_DOC)
        for chunk in chunks:
            assert chunk.text and isinstance(chunk.text, str)
            assert chunk.embedding is not None
            assert isinstance(chunk.embedding, list)
            assert len(chunk.embedding) > 0
            assert isinstance(chunk.embedding[0], float)

    def test_metadata_has_required_keys(self):
        chunks = LateChunkingChunker().chunk(TEST_DOC)
        for chunk in chunks:
            for key in ("pooling", "model_used", "token_start", "token_end"):
                assert key in chunk.metadata, f"Missing metadata key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Chunker 6 — Contextual (fallback path — llm_client=None)
# ─────────────────────────────────────────────────────────────────────────────

class TestContextualChunker:
    def test_instantiation(self):
        assert ContextualChunker() is not None

    def test_fallback_returns_chunks_without_raising(self):
        chunks = ContextualChunker().chunk(TEST_DOC, llm_client=None)
        assert isinstance(chunks, list) and len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_fallback_metadata_keys(self):
        chunks = ContextualChunker().chunk(TEST_DOC, llm_client=None)
        for chunk in chunks:
            assert "has_context" in chunk.metadata
            assert "context_summary" in chunk.metadata
            assert "original_text" in chunk.metadata
            assert chunk.metadata["has_context"] is False

    def test_fallback_chunk_text_nonempty(self):
        chunks = ContextualChunker().chunk(TEST_DOC, llm_client=None)
        assert all(c.text for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Chunker 7 — Proposition (fallback path — llm_client=None)
# ─────────────────────────────────────────────────────────────────────────────

class TestPropositionChunker:
    def test_instantiation(self):
        assert PropositionChunker() is not None

    def test_fallback_returns_chunks_without_raising(self):
        chunks = PropositionChunker().chunk(TEST_DOC, llm_client=None)
        assert isinstance(chunks, list) and len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_fallback_metadata_keys(self):
        chunks = PropositionChunker().chunk(TEST_DOC, llm_client=None)
        for chunk in chunks:
            assert "source_paragraph" in chunk.metadata
            assert "paragraph_idx" in chunk.metadata
            assert "is_fallback" in chunk.metadata
            assert chunk.metadata["is_fallback"] is True

    def test_fallback_chunk_text_nonempty(self):
        chunks = PropositionChunker().chunk(TEST_DOC, llm_client=None)
        assert all(c.text for c in chunks)

    def test_paragraph_idx_values(self):
        chunks = PropositionChunker().chunk(TEST_DOC, llm_client=None)
        para_indices = {c.metadata["paragraph_idx"] for c in chunks}
        # TEST_DOC has 3 paragraphs
        assert len(para_indices) == 3
