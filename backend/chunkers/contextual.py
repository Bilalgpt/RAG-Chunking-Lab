"""
Contextual retrieval prepends an LLM-generated summary to each chunk before
embedding, so that the vector representation captures not just the chunk's
own content but also its role within the broader document.

Motivation: A chunk like "The results were inconclusive." is ambiguous in
isolation. Prepending "This chunk discusses the clinical trial outcomes in
Section 3 of the medical report." makes the embedding far more informative
for retrieval.

Algorithm:
  1. Split the document with the recursive chunker to get base chunks.
  2. For each chunk, prompt the LLM with the document excerpt + chunk text.
  3. Prepend the returned 1-2 sentence context summary to the chunk text.
  4. The enriched text (summary + original) is what gets embedded.

The full document is truncated to 2000 characters in the prompt to keep
token costs low while still providing sufficient context.

If llm_client is None, context enrichment is skipped and plain recursive
chunks are returned with has_context=False in metadata.
"""

from __future__ import annotations

import warnings

from .base import BaseChunker, Chunk
from .recursive import RecursiveChunker

_CONTEXT_PROMPT = """\
You are a document analyst. Given the full document (excerpt) and one chunk \
from it, write 1-2 sentences describing what the chunk is about and where it \
sits in the document. Be concise. Output only the description, no preamble.

Document excerpt:
{doc_excerpt}

Chunk:
{chunk_text}"""


class ContextualChunker(BaseChunker):

    @property
    def name(self) -> str:
        return "contextual"

    @property
    def description(self) -> str:
        return "Prepend an LLM-generated context summary to each chunk before embedding."

    @property
    def default_params(self) -> dict:
        return {"chunk_size": 300, "overlap": 30}

    # ── public ────────────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        llm_client=None,
        provider: str = "anthropic",
        **kwargs,
    ) -> list[Chunk]:
        params = {**self.default_params, **kwargs}
        base_chunks = RecursiveChunker().chunk(
            text,
            max_size=params["chunk_size"],
            overlap=params["overlap"],
        )

        if llm_client is None:
            warnings.warn(
                "ContextualChunker: llm_client is None — "
                "skipping context enrichment. Returning plain recursive chunks."
            )
            for chunk in base_chunks:
                chunk.metadata.update({
                    "has_context": False,
                    "context_summary": "",
                    "original_text": chunk.metadata.get("original_text", chunk.text),
                })
            return base_chunks

        doc_excerpt = text[:2000]
        enriched: list[Chunk] = []

        for chunk in base_chunks:
            original_text = chunk.metadata.get("original_text", chunk.text)
            prompt = _CONTEXT_PROMPT.format(
                doc_excerpt=doc_excerpt,
                chunk_text=original_text,
            )
            try:
                summary = _call_llm(llm_client, provider, prompt, max_tokens=150)
            except Exception as exc:
                warnings.warn(f"ContextualChunker: LLM call failed ({exc}), skipping enrichment for chunk {chunk.index}.")
                summary = ""

            if summary:
                chunk.text = summary + "\n\n" + original_text
            chunk.metadata.update({
                "has_context": bool(summary),
                "context_summary": summary,
                "original_text": original_text,
            })
            enriched.append(chunk)

        return enriched


# ── shared LLM helper (module-level, not part of the class) ──────────────────

def _call_llm(client, provider: str, prompt: str, max_tokens: int = 200) -> str:
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
