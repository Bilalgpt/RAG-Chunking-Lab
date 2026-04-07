"""
Semantic chunking places boundaries wherever the cosine similarity between
adjacent sentence embeddings drops below a dynamic threshold.

Algorithm:
  1. Split text into sentences on [". ", "? ", "! "].
  2. Embed each sentence with the shared all-MiniLM-L6-v2 model.
  3. Compute cosine similarity for every adjacent pair:
       sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
  4. Compute a dynamic threshold:
       threshold = μ(similarities) − k · σ(similarities)
     where k is a tunable parameter (default 1.0).
  5. Insert a boundary between s_i and s_{i+1} where sim < threshold.
  6. Merge consecutive sentences within each segment into one chunk.
  7. If a resulting chunk has fewer than min_chunk_words words, merge it
     forward into the next chunk.

Higher k → fewer, larger chunks (only very dissimilar splits).
Lower k  → more, smaller chunks (more sensitive to similarity drops).
"""

from __future__ import annotations

import re

import numpy as np

from utils.embedder import embed_batch
from .base import BaseChunker, Chunk


class SemanticChunker(BaseChunker):

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def description(self) -> str:
        return "Place boundaries where cosine similarity between adjacent sentence embeddings drops below a dynamic threshold."

    @property
    def default_params(self) -> dict:
        return {"k": 1.0, "min_chunk_words": 50}

    # ── public ────────────────────────────────────────────────────────────────

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        params = {**self.default_params, **kwargs}
        k: float = float(params["k"])
        min_chunk_words: int = int(params["min_chunk_words"])

        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [Chunk(text=sentences[0], index=0, metadata={
                "sentence_count": 1, "avg_similarity": 1.0, "min_similarity": 1.0,
            })]

        embeddings = np.array(embed_batch(sentences))  # (N, D)
        similarities = self._pairwise_cosine(embeddings)  # length N-1

        threshold = float(np.mean(similarities) - k * np.std(similarities))
        boundaries = {i + 1 for i, s in enumerate(similarities) if s < threshold}

        # Group sentences into segments
        segments: list[list[str]] = []
        seg_sims: list[list[float]] = []
        current: list[str] = [sentences[0]]
        current_sims: list[float] = []

        for i in range(1, len(sentences)):
            if i in boundaries:
                segments.append(current)
                seg_sims.append(current_sims)
                current = [sentences[i]]
                current_sims = []
            else:
                current.append(sentences[i])
                current_sims.append(float(similarities[i - 1]))

        segments.append(current)
        seg_sims.append(current_sims)

        # Enforce minimum chunk size — merge small trailing segments forward
        merged_segments, merged_sims = self._merge_small(segments, seg_sims, min_chunk_words)

        chunks: list[Chunk] = []
        for idx, (seg, sims) in enumerate(zip(merged_segments, merged_sims)):
            seg_text = " ".join(seg)
            chunks.append(Chunk(
                text=seg_text,
                index=idx,
                metadata={
                    "sentence_count": len(seg),
                    "avg_similarity": float(np.mean(sims)) if sims else 1.0,
                    "min_similarity": float(np.min(sims)) if sims else 1.0,
                },
            ))

        return chunks

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?]) +", text)
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity between each adjacent pair. Shape: (N-1,)."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = embeddings / norms
        return np.sum(normed[:-1] * normed[1:], axis=1)

    @staticmethod
    def _merge_small(
        segments: list[list[str]],
        sims: list[list[float]],
        min_words: int,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Merge segments that are below min_words into the next segment."""
        out_segs: list[list[str]] = []
        out_sims: list[list[float]] = []
        pending_s: list[str] = []
        pending_sim: list[float] = []

        for seg, sim in zip(segments, sims):
            pending_s.extend(seg)
            pending_sim.extend(sim)
            if len(" ".join(pending_s).split()) >= min_words:
                out_segs.append(pending_s)
                out_sims.append(pending_sim)
                pending_s = []
                pending_sim = []

        if pending_s:
            if out_segs:
                out_segs[-1].extend(pending_s)
                out_sims[-1].extend(pending_sim)
            else:
                out_segs.append(pending_s)
                out_sims.append(pending_sim)

        return out_segs, out_sims
