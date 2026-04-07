import asyncio

from fastapi import APIRouter, HTTPException

from models.schemas import QueryRequest, QueryResponse
from services import query_service

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and generate an LLM answer."""
    try:
        return await asyncio.to_thread(query_service.query_document, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
