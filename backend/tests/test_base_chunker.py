import sys
import os

# Ensure backend/ is on the path when running pytest from the backend directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunkers.base import BaseChunker, Chunk


class _MinimalChunker(BaseChunker):
    """Concrete minimal subclass used only in tests."""

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def description(self) -> str:
        return "Splits text into single-character chunks for testing."

    @property
    def default_params(self) -> dict:
        return {}

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        return [Chunk(text=ch, index=i) for i, ch in enumerate(text) if ch.strip()]


def test_minimal_chunker_can_be_instantiated():
    chunker = _MinimalChunker()
    assert chunker is not None


def test_chunk_returns_list():
    chunker = _MinimalChunker()
    result = chunker.chunk("hello world")
    assert isinstance(result, list)
    assert len(result) > 0


def test_chunk_items_have_required_keys():
    chunker = _MinimalChunker()
    result = chunker.chunk("abc")
    for chunk in result:
        assert isinstance(chunk, Chunk)
        assert isinstance(chunk.text, str) and chunk.text
        assert isinstance(chunk.index, int)
        assert isinstance(chunk.metadata, dict)


def test_chunk_index_is_zero_based():
    chunker = _MinimalChunker()
    result = chunker.chunk("ab")
    assert result[0].index == 0
    assert result[1].index == 1


def test_empty_input_returns_empty_list():
    chunker = _MinimalChunker()
    result = chunker.chunk("   ")
    assert result == []
