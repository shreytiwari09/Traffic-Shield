import asyncio

from fastapi import APIRouter, HTTPException

from services.data_service import chroma_store, dataset_store
from services.shared.schemas import VectorSearchRequest, VectorSearchResponse, VectorSearchResult

router = APIRouter()


@router.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "chroma_count": chroma_store.count(),
        "dataset_records": dataset_store.count(),
    }


@router.post("/v1/vector-search", response_model=VectorSearchResponse)
async def vector_search(req: VectorSearchRequest):
    # chromadb's client is synchronous — run it in a thread so it doesn't
    # block this process's single event loop. Real bug found in testing:
    # without this, /v1/embeddings below (called ~20x concurrently per
    # request from Retrieval Service) serialized instead of running in
    # parallel, since every call blocked the same loop in turn.
    results = await asyncio.to_thread(chroma_store.query, req.embedding, req.top_k)
    return VectorSearchResponse(results=[VectorSearchResult(**r) for r in results])


@router.get("/v1/records/{record_id}")
async def get_record(record_id: str):
    record = dataset_store.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"record {record_id} not found")
    return record


@router.get("/v1/embeddings/{record_id}")
async def get_embeddings(record_id: str):
    embeddings = await asyncio.to_thread(chroma_store.get_embeddings, record_id)
    return {"embeddings": embeddings}
