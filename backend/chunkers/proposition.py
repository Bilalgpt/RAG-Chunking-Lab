"""
Proposition chunking decomposes text into atomic, self-contained factual
claims using an LLM, then indexes each proposition as its own chunk.

Motivation: Traditional chunks contain multiple facts mixed together. A query
about one fact may retrieve a chunk dominated by different facts, diluting
relevance. Propositions are the minimal unit of meaning — each one addresses
exactly one verifiable claim, making retrieval far more precise.

Algorithm:
  1. Split document into paragraphs on double newlines.
  2. For each paragraph, prompt the LLM to extract atomic propositions:
       - Self-contained (replace pronouns with named entities)
       - Verifiable (a single claim, not a compound sentence)
       - JSON array output
  3. Parse the JSON. On failure, fall back to sentence splitting for that
     paragraph (logged as a warning, recorded in metadata["is_fallback"]).
  4. Each proposition becomes one Chunk.

If llm_client is None, sentence splitting is used for all paragraphs.
"""

from __future__ import annotations

import json
import re
import warnings

from .base import BaseChunker, Chunk

_PROPOSITION_PROMPT = """\
Extract all atomic factual propositions from the following paragraph.

Rules:
1. Each proposition must be a single, self-contained, verifiable claim.
2. Replace all pronouns (it, they, this, that) with the actual named entity.
3. Return ONLY a JSON array of strings. No preamble, no markdown fences.

Example: ["The Eiffel Tower was built in 1889.", "The Eiffel Tower is 330 m tall."]

Paragraph:
{paragraph}"""


class PropositionChunker(BaseChunker):

    @property
    def name(self) -> str:
        return "proposition"

    @property
    def description(self) -> str:
        return "Decompose text into atomic, self-contained factual propositions using an LLM."

    @property
    def default_params(self) -> dict:
        return {}  # proposition count is LLM-determined

    # ── public ────────────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        llm_client=None,
        provider: str = "anthropic",
        **kwargs,
    ) -> list[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        if llm_client is None:
            warnings.warn(
                "PropositionChunker: llm_client is None — "
                "falling back to sentence splitting for all paragraphs."
            )

        chunks: list[Chunk] = []
        chunk_idx = 0

        for para_idx, paragraph in enumerate(paragraphs):
            if llm_client is None:
                propositions = _sentence_split(paragraph)
                is_fallback = True
            else:
                try:
                    propositions = self._extract_propositions(llm_client, provider, paragraph)
                    is_fallback = False
                except Exception as exc:
                    warnings.warn(
                        f"PropositionChunker: extraction failed for paragraph {para_idx} ({exc}). "
                        "Using sentence fallback."
                    )
                    propositions = _sentence_split(paragraph)
                    is_fallback = True

            for prop in propositions:
                prop = prop.strip()
                if not prop:
                    continue
                chunks.append(Chunk(
                    text=prop,
                    index=chunk_idx,
                    metadata={
                        "source_paragraph": paragraph,
                        "paragraph_idx": para_idx,
                        "is_fallback": is_fallback,
                    },
                ))
                chunk_idx += 1

        return chunks

    # ── private ───────────────────────────────────────────────────────────────

    def _extract_propositions(
        self,
        client,
        provider: str,
        paragraph: str,
    ) -> list[str]:
        prompt = _PROPOSITION_PROMPT.format(paragraph=paragraph)
        raw = _call_llm(client, provider, prompt, max_tokens=512)

        # Strip markdown fences if the model wrapped the JSON
        raw = raw.strip()
        if raw.startswith("```"):
            inner = raw.split("```")
            raw = inner[1] if len(inner) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)


# ── module-level helpers ──────────────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?]) +", text)
    return [s.strip() for s in parts if s.strip()]


def _call_llm(client, provider: str, prompt: str, max_tokens: int = 512) -> str:
    if provider == "anthropic":
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    elif provider == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    elif provider == "groq":
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'")
