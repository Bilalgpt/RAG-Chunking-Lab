import asyncio

from fastapi import APIRouter, HTTPException

from models.schemas import CompareRequest, CompareResponse, QueryRequest
from services import query_service

router = APIRouter()


@router.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> CompareResponse:
    """Run the same query against two indexed collections concurrently."""
    if len(request.collection_names) != 2:
        raise HTTPException(
            status_code=422,
            detail="collection_names must contain exactly 2 items.",
        )

    try:
        def make_query(collection_name: str) -> QueryRequest:
            return QueryRequest(
                query=request.query,
                collection_name=collection_name,
                top_k=request.top_k,
                llm_provider=request.llm_provider,
                api_key=request.api_key,
            )

        results = await asyncio.gather(
            asyncio.to_thread(query_service.query_document, make_query(request.collection_names[0])),
            asyncio.to_thread(query_service.query_document, make_query(request.collection_names[1])),
        )
        return CompareResponse(results=list(results))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")
