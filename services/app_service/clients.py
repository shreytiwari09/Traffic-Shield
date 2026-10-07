"""HTTP client to Orchestration Service — Application Service's only
downstream dependency. No retrieval logic, no LLM calls, no data access
happen here."""

import httpx

from services.shared.settings import settings

_TIMEOUT = settings.request_timeout_seconds


async def ask(question: str, provider: str, conversation_id: str | None = None) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.orchestration_service_url}/v1/ask",
            json={
                "question": question,
                "provider": provider,
                "conversation_id": conversation_id,
            },
        )
        response.raise_for_status()
        return response.json()


async def restore_conversation(conversation_id: str, messages: list[dict]) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.orchestration_service_url}/v1/conversations/{conversation_id}/restore",
            json={"messages": messages},
        )
        response.raise_for_status()
        return response.json()


async def reset_conversation(conversation_id: str) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.delete(
            f"{settings.orchestration_service_url}/v1/conversations/{conversation_id}"
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


async def submit_feedback(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(f"{settings.orchestration_service_url}/v1/feedback", json=payload)
        response.raise_for_status()
        return response.json()


async def feedback_summary() -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.orchestration_service_url}/v1/feedback/summary")
        response.raise_for_status()
        return response.json()


async def providers() -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.orchestration_service_url}/v1/providers")
        response.raise_for_status()
        return response.json()


async def model_registry() -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(f"{settings.orchestration_service_url}/v1/registry")
        response.raise_for_status()
        return response.json()
