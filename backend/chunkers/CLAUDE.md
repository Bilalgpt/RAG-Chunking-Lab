# Chunkers — Base Class Contract

## What Lives Here
One Python file per chunking technique, plus `base.py` defining the contract all chunkers must satisfy.

## The 7 Techniques (one file each)
1. `fixed_size.py` — split by token/character count with optional overlap
2. `recursive.py` — split by hierarchy of separators (paragraph → sentence → word)
3. `semantic.py` — split at embedding cosine-similarity drop points
4. `hierarchical.py` — parent/child chunks; store both, retrieve child, return parent context
5. `late_chunking.py` — embed full document first, then pool token embeddings into chunks
6. `contextual_retrieval.py` — prepend LLM-generated context summary to each chunk before embedding
7. `proposition.py` — decompose text into atomic factual propositions via LLM, embed propositions

## Base Class Contract (`base.py`)
Every chunker must subclass `BaseChunker` and implement exactly one method:
```
chunk(text: str, **kwargs) -> list[Chunk]
```
`Chunk` is a dataclass with fields: `id`, `text`, `metadata` (dict), `embedding` (optional list[float]).

## Rules for Adding a New Chunker
- See `.claude/skills/adding-chunker.md` for the full step-by-step.
- Do not add chunker-specific config to any file outside `chunkers/`.
- Chunkers must be stateless — no instance state that changes between calls.
- Chunkers do NOT call the vector store or LLM directly (except contextual and proposition which need LLM for generation — pass the LLM client in via constructor).

## Naming
File names are snake_case matching the technique slug used in API requests.
