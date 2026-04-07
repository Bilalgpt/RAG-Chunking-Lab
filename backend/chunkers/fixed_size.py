"""
Fixed-size chunking divides text into overlapping windows of equal word count.

Algorithm:
  chunk_i = words[ i·S : i·S + W ]
  where W = window size (chunk_size), S = stride = W − overlap

  N = ⌈(total_words − overlap) / S⌉

Each chunk overlaps with the previous by `overlap` words, preserving local
context across boundaries. Simple and fast — the baseline every other
technique is measured against.
"""

from math import ceil

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):

    @property
    def name(self) -> str:
        return "fixed_size"

    @property
    def description(self) -> str:
        return "Split text into equal-size windows by word count with configurable overlap."

    @property
    def default_params(self) -> dict:
        return {"chunk_size": 200, "overlap": 20}

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        params = {**self.default_params, **kwargs}
        chunk_size: int = int(params["chunk_size"])
        overlap: int = int(params["overlap"])

        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        words = text.split()
        if not words:
            return []

        stride = chunk_size - overlap
        chunks: list[Chunk] = []
        idx = 0
        i = 0

        while i < len(words):
            end = min(i + chunk_size, len(words))
            chunk_words = words[i:end]
            chunks.append(Chunk(
                text=" ".join(chunk_words),
                index=idx,
                metadata={
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "start_word": i,
                    "end_word": end,
                },
            ))
            idx += 1
            i += stride

        return chunks
