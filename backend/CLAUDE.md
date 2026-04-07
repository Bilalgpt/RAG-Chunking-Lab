# Backend — FastAPI App

## What Lives Here
FastAPI application exposing endpoints for: document ingestion, chunking, embedding, retrieval, and LLM answer generation. Each concern is isolated in its own module.

## Directory Layout
- `main.py` — app factory, router registration, CORS config
- `routers/` — one file per feature group (documents, query, compare)
- `models/` — Pydantic request/response schemas only, no business logic
- `services/` — orchestration layer; calls chunkers, vector store, LLM clients
- `chunkers/` — one file per chunking technique + base class
- `vector_store/` — abstract base + one file per backend (ChromaDB default)
- `sample_docs/` — static text files served as bundled examples

## FastAPI Patterns
- Routers use `APIRouter` with a prefix; registered in `main.py`.
- All request/response bodies are Pydantic models defined in `models/`.
- Services are plain classes instantiated once and injected via FastAPI `Depends`.
- No business logic in router functions — delegate to services immediately.
- CORS is open in dev (`allow_origins=["*"]`); tighten for production.

## LLM Client Pattern
- A thin `LLMClient` class wraps each provider (Anthropic, OpenAI, Groq).
- The API key is passed in per-request from the frontend header `X-API-Key`.
- Provider is passed in per-request from the frontend header `X-LLM-Provider`.
- Never log or store API keys anywhere.

## Dependencies
All Python deps in `backend/requirements.txt`. Use a virtual environment (`venv/`).
