"""HTTP client to Orchestration Service — Application Service's only
downstream dependency. No retrieval logic, no LLM calls, no data access
happen here."""

import httpx

from services.shared.settings import settings

_TIMEOUT = settings.request_timeout_seconds


async def ask(question: str, provider: str) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.orchestration_service_url}/v1/ask",
            json={"question": question, "provider": provider},
        )
        response.raise_for_status()
        return response.json()


async def eval_grid(question: str) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.orchestration_service_url}/v1/eval",
            json={"question": question},
        )
        response.raise_for_status()
        return response.json()


async def list_categories() -> dict:
    # Kept behind Orchestration, not called directly against Retrieval Service —
    # Application Service's only downstream dependency stays Orchestration.
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.orchestration_service_url}/v1/categories")
        response.raise_for_status()
        return response.json()


async def category_sections(slug: str) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.orchestration_service_url}/v1/categories/{slug}/sections")
        response.raise_for_status()
        return response.json()
