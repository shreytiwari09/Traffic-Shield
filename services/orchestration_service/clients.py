"""HTTP clients to the two services Orchestration sequences: Retrieval
Service and LLM Service. Orchestration never touches Chroma, the graph JSON,
or Ollama/Gemini directly — it only calls the services that own them."""

import httpx

from services.shared.settings import settings

_TIMEOUT = settings.request_timeout_seconds


async def retrieve(question: str, top_k: int, mode: str = "hybrid") -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.retrieval_service_url}/v1/retrieve",
            json={"question": question, "top_k": top_k, "mode": mode},
        )
        response.raise_for_status()
        return response.json()


async def generate(question: str, context: list[dict], provider: str, use_persona: bool = True) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.llm_service_url}/v1/generate",
            json={
                "question": question,
                "context": context,
                "provider": provider,
                "use_persona": use_persona,
            },
        )
        response.raise_for_status()
        return response.json()


async def list_categories() -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.retrieval_service_url}/v1/categories")
        response.raise_for_status()
        return response.json()


async def category_sections(slug: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        return await client.get(f"{settings.retrieval_service_url}/v1/categories/{slug}/sections")
