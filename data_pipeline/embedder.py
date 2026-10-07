"""
Embedding client for Phase 6.

Embeds through a locally served Ollama model when this machine has one, per
the project's requirement that every stage of the RAG pipeline runs locally.
Without Ollama it runs the same nomic-embed-text v1.5 weights in-process
(services/shared/local_embedder.py) — still local, still no hosted API, and
the same vector space the services query with.
"""

import asyncio

import httpx

from data_pipeline.config import DEFAULT_EMBEDDING_MODEL
from services.shared import local_embedder
from services.shared.ollama_probe import ollama_status
from services.shared.settings import settings

_TIMEOUT = 180.0


async def choose_backend(model: str | None = None) -> str:
    """"ollama" or "local". Local only stands in for the default model: it
    runs nomic-embed-text, so a different --embedding-model needs Ollama."""
    model = model or DEFAULT_EMBEDDING_MODEL
    status = await ollama_status(force=True)
    if status.has_model(model):
        return "ollama"
    if model == DEFAULT_EMBEDDING_MODEL:
        return "local"
    raise RuntimeError(
        f"Embedding model '{model}' needs Ollama, which is not usable here ({status.reason}). "
        f"Start Ollama and `ollama pull {model}`, or use the default model."
    )


async def embed_texts(texts: list[str], model: str | None = None, backend: str = "ollama") -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input, in order."""
    if backend == "local":
        return await asyncio.to_thread(local_embedder.embed_texts, texts)

    model = model or DEFAULT_EMBEDDING_MODEL
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/embed",  # same URL the probe checked
            json={"model": model, "input": texts},
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama has no model '{model}'. Pull one first, e.g. "
                f"`ollama pull {DEFAULT_EMBEDDING_MODEL}`."
            )
        response.raise_for_status()
        return response.json().get("embeddings", [])


async def embed_single(text: str, model: str | None = None, backend: str = "ollama") -> list[float]:
    vectors = await embed_texts([text], model=model, backend=backend)
    return vectors[0] if vectors else []
