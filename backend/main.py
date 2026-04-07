from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from chunkers import CHUNKER_REGISTRY
from routers import documents, query, compare

app = FastAPI(
    title="RAG Chunking Lab API",
    description="Backend for the RAG Chunking Lab educational app.",
    version="0.1.0",
)

# CORS — open for educational/dev use. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(compare.router, prefix="/api", tags=["compare"])


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get("/api/chunkers", tags=["meta"])
async def list_chunkers():
    """Return metadata for all registered chunking techniques."""
    return [
        {
            "slug": cls().name,
            "name": cls().name.replace("_", " ").title(),
            "description": cls().description,
            "default_params": cls().default_params,
        }
        for cls in CHUNKER_REGISTRY.values()
    ]


@app.on_event("startup")
async def startup():
    print(f"[RAG Chunking Lab] Embedding model: {settings.embedding_model_name}")
    print(f"[RAG Chunking Lab] Registered chunkers: {list(CHUNKER_REGISTRY.keys()) or '(none yet)'}")
