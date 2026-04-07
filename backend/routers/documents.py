from fastapi import APIRouter, HTTPException

from models.schemas import ChunkResponse, IndexRequest, IndexResponse
from services import chunking_service

router = APIRouter()


@router.post("/index", response_model=IndexResponse)
async def index_document(request: IndexRequest) -> IndexResponse:
    """Chunk a document, embed it, and store it in the vector store."""
    try:
        return chunking_service.index_document(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")


@router.get("/list", response_model=list[str])
async def list_collections() -> list[str]:
    """Return the names of all indexed collections."""
    try:
        return chunking_service.list_collections()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not list collections: {exc}")


@router.get("/chunks/{collection_name}", response_model=list[ChunkResponse])
async def get_chunks(collection_name: str) -> list[ChunkResponse]:
    """Return all chunks stored in the named collection."""
    try:
        return chunking_service.get_chunks(collection_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not retrieve chunks: {exc}")
