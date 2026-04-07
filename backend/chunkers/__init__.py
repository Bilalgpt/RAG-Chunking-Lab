from .base import BaseChunker, Chunk
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .hierarchical import HierarchicalChunker
from .late_chunking import LateChunkingChunker
from .contextual import ContextualChunker
from .proposition import PropositionChunker

CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "hierarchical": HierarchicalChunker,
    "late_chunking": LateChunkingChunker,
    "contextual": ContextualChunker,
    "proposition": PropositionChunker,
}


def get_chunker(technique: str) -> BaseChunker:
    """Instantiate and return the chunker for the given technique slug."""
    if technique not in CHUNKER_REGISTRY:
        available = list(CHUNKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown chunking technique: '{technique}'. Available: {available}"
        )
    return CHUNKER_REGISTRY[technique]()


__all__ = ["BaseChunker", "Chunk", "CHUNKER_REGISTRY", "get_chunker"]
