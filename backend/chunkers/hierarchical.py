"""
Hierarchical chunking builds a 3-level tree from a document's natural structure
and indexes only the leaf nodes, storing parent context in metadata.

Levels:
  L1 — Section:   split on triple newlines or Markdown headers (# / ## / …)
  L2 — Paragraph: within each section, split on double newlines
  L3 — Sentence:  within each paragraph, split on sentence-ending punctuation

chunk() returns ONLY the L3 (sentence) chunks — these are embedded and indexed.
Each leaf stores its full parent texts so a retriever can widen context:

  metadata["paragraph_text"] → expand to paragraph on retrieval
  metadata["section_text"]   → expand to section (Auto-Merge pattern)

If multiple leaf chunks from the same paragraph are retrieved, the caller
can deduplicate by returning metadata["paragraph_text"] instead, giving a
richer answer without re-indexing larger chunks.
"""

from __future__ import annotations

import re

from .base import BaseChunker, Chunk


class HierarchicalChunker(BaseChunker):

    @property
    def name(self) -> str:
        return "hierarchical"

    @property
    def description(self) -> str:
        return "Build a 3-level section → paragraph → sentence hierarchy and index leaf chunks with parent context in metadata."

    @property
    def default_params(self) -> dict:
        return {}  # structure-driven; no size parameters needed

    # ── public ────────────────────────────────────────────────────────────────

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        sections = self._split_sections(text)
        chunks: list[Chunk] = []
        chunk_idx = 0

        for sec_idx, section_text in enumerate(sections):
            paragraphs = self._split_paragraphs(section_text)

            for para_idx, para_text in enumerate(paragraphs):
                sentences = self._split_sentences(para_text)

                for sentence in sentences:
                    if not sentence:
                        continue
                    chunks.append(Chunk(
                        text=sentence,
                        index=chunk_idx,
                        metadata={
                            "level": 3,
                            "section_idx": sec_idx,
                            "paragraph_idx": para_idx,
                            "section_text": section_text,
                            "paragraph_text": para_text,
                        },
                    ))
                    chunk_idx += 1

        return chunks

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sections(text: str) -> list[str]:
        """Split on triple+ newlines or Markdown header boundaries."""
        # First split on 3+ newlines
        parts = re.split(r"\n{3,}", text)
        sections: list[str] = []
        for part in parts:
            # Further split at Markdown header lines (# ## ### …)
            sub = re.split(r"(?=\n#{1,6} )", part)
            sections.extend(s.strip() for s in sub if s.strip())
        return sections or [text.strip()]

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?]) +", text)
        return [s.strip() for s in parts if s.strip()]
