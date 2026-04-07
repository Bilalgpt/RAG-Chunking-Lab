from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single chunk of text produced by a chunker."""
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


class BaseChunker(ABC):
    """
    Abstract base class for all chunking techniques.

    Subclasses must implement chunk(), name, description, and default_params.
    Chunkers are stateless — do not store mutable state between calls.
    Chunkers must not call embeddings or LLM APIs directly (except contextual
    and proposition chunkers which accept an LLM client via __init__).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Technique slug, e.g. 'fixed_size'. Must match the registry key."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description shown in the UI technique card."""

    @property
    @abstractmethod
    def default_params(self) -> dict:
        """Default kwargs passed to chunk() when the caller provides none."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The full document text to chunk.
            **kwargs: Technique-specific parameters (chunk_size, overlap, etc.).
                      Merged with default_params — caller overrides take precedence.

        Returns:
            List of Chunk objects. Must be non-empty for non-empty input.
            Each Chunk.text must be non-empty. Chunk.index is 0-based.
        """
