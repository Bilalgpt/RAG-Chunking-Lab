"""
Late chunking embeds the full document first (preserving cross-chunk context
in every token's representation) and then pools within chunk boundaries,
rather than embedding each chunk independently.

Standard approach (loses cross-chunk context):
  embed(chunk_i) independently → each chunk's embedding is context-blind

Late chunking approach:
  embed(full_doc) → token vectors [v_1 … v_T]
  embed(chunk_i) = mean_pool(v_{start_i} … v_{end_i})

  mean_pool(v_s … v_e) = (1 / (e − s + 1)) · Σ v_t   for t ∈ [s, e]

This means the embedding for "RAG" in chunk 3 still carries information from
the introduction in chunk 1, because the transformer attended over the whole
document during the forward pass. Invented for long-context models like Jina.

Implementation:
  1. Build word-boundary spans using the recursive chunker.
  2. Load "jinaai/jina-embeddings-v2-base-en" (8192-token context window).
     Falls back to "all-MiniLM-L6-v2" if the Jina model is unavailable.
  3. Run the full document through the model to get per-token hidden states.
  4. Use the tokenizer's offset map to align chunk text spans → token indices.
  5. Mean-pool the token vectors within each span.
  6. Store the result in Chunk.embedding.
"""

from __future__ import annotations

import warnings

from .base import BaseChunker, Chunk
from .recursive import RecursiveChunker

_JINA_MODEL = "jinaai/jina-embeddings-v2-base-en"
_FALLBACK_MODEL = "all-MiniLM-L6-v2"


class LateChunkingChunker(BaseChunker):
    _model = None
    _model_name_used: str | None = None

    @property
    def name(self) -> str:
        return "late_chunking"

    @property
    def description(self) -> str:
        return "Embed the full document first to get context-aware token vectors, then mean-pool within chunk boundaries."

    @property
    def default_params(self) -> dict:
        return {"chunk_size": 200, "overlap": 20}

    # ── public ────────────────────────────────────────────────────────────────

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        params = {**self.default_params, **kwargs}

        # Step 1: get word-boundary spans via recursive chunker
        base_chunks = RecursiveChunker().chunk(
            text,
            max_size=params["chunk_size"],
            overlap=params["overlap"],
        )
        if not base_chunks:
            return []

        # Step 2: load model (cached after first call)
        model, model_name = self._get_or_load_model()

        # Step 3: annotate chunks with late-chunking embeddings
        base_chunks = self._late_embed(text, base_chunks, model, model_name)
        return base_chunks

    # ── private — model loading ───────────────────────────────────────────────

    @classmethod
    def _get_or_load_model(cls):
        if cls._model is None:
            cls._model, cls._model_name_used = cls._load_model()
        return cls._model, cls._model_name_used

    @staticmethod
    def _load_model():
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(_JINA_MODEL, trust_remote_code=True)
            return model, _JINA_MODEL
        except Exception as exc:
            warnings.warn(
                f"Could not load {_JINA_MODEL} ({exc}). "
                f"Falling back to {_FALLBACK_MODEL}."
            )
            return SentenceTransformer(_FALLBACK_MODEL), _FALLBACK_MODEL

    # ── private — late embedding ──────────────────────────────────────────────

    @staticmethod
    def _late_embed(
        full_text: str,
        chunks: list[Chunk],
        model,
        model_name: str,
    ) -> list[Chunk]:
        import numpy as np

        try:
            import torch

            # Forward pass on full document → token-level hidden states
            features = model.tokenize([full_text])
            features = {k: v.to(model.device) for k, v in features.items()}
            with torch.no_grad():
                output = model.forward(features)

            # token_embeddings: tensor (1, seq_len, hidden_dim)
            token_embeds = output["token_embeddings"][0].cpu().numpy()

            # Get character-to-token offset mapping from the tokenizer
            tokenizer = model.tokenizer
            encoding = tokenizer(
                full_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=token_embeds.shape[0],
                add_special_tokens=True,
            )
            offset_map: list[tuple[int, int]] = encoding["offset_mapping"]

            # Pool token vectors within each chunk's character span
            search_from = 0
            for chunk in chunks:
                char_start = full_text.find(chunk.text, search_from)
                if char_start == -1:
                    char_start = search_from
                char_end = char_start + len(chunk.text)

                token_indices = [
                    i for i, (ts, te) in enumerate(offset_map)
                    if te > char_start and ts < char_end and te > ts
                ]

                if token_indices:
                    t_start = token_indices[0]
                    t_end = min(token_indices[-1] + 1, len(token_embeds))
                    pooled = np.mean(token_embeds[t_start:t_end], axis=0)
                    chunk.embedding = pooled.tolist()
                    chunk.metadata.update({
                        "pooling": "mean",
                        "model_used": model_name,
                        "token_start": t_start,
                        "token_end": t_end - 1,
                    })
                else:
                    # Chunk beyond truncation limit — sentence-level fallback
                    emb = model.encode(chunk.text, convert_to_numpy=True)
                    chunk.embedding = emb.tolist()
                    chunk.metadata.update({
                        "pooling": "sentence_level_fallback",
                        "model_used": model_name,
                        "token_start": -1,
                        "token_end": -1,
                    })

                search_from = char_end

            return chunks

        except Exception as exc:
            warnings.warn(
                f"Token-level pooling failed ({exc}). "
                "Using sentence-level embeddings for all chunks."
            )
            for chunk in chunks:
                emb = model.encode(chunk.text, convert_to_numpy=True)
                chunk.embedding = emb.tolist()
                chunk.metadata.setdefault("pooling", "sentence_level_fallback")
                chunk.metadata.setdefault("model_used", model_name)
                chunk.metadata.setdefault("token_start", -1)
                chunk.metadata.setdefault("token_end", -1)
            return chunks
