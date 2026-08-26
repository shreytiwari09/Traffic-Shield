"""HTTP clients to the two services Retrieval depends on: LLM Service (only
for embedding the question) and Data Service (vector search + record
lookups). Retrieval never talks to Ollama directly."""

import httpx

from services.shared.settings import settings

_TIMEOUT = settings.request_timeout_seconds


async def embed_question(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(f"{settings.llm_service_url}/v1/embed", json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]


async def vector_search(embedding: list[float], top_k: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.data_service_url}/v1/vector-search",
            json={"embedding": embedding, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()["results"]


async def get_record(record_id: str) -> dict | None:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.data_service_url}/v1/records/{record_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


async def get_embeddings(record_id: str) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.data_service_url}/v1/embeddings/{record_id}")
        response.raise_for_status()
        return response.json()["embeddings"]
