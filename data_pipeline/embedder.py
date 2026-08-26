"""
Ollama embedding client for Phase 6.

The pipeline embeds through a locally served Ollama model — no hosted
embedding API — per the project's requirement that every stage of the RAG
pipeline runs locally.
"""

import httpx

from data_pipeline.config import DEFAULT_EMBEDDING_MODEL, OLLAMA_BASE_URL

_TIMEOUT = 180.0


async def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input, in order."""
    model = model or DEFAULT_EMBEDDING_MODEL

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": texts},
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama has no model '{model}'. Pull one first, e.g. "
                f"`ollama pull {DEFAULT_EMBEDDING_MODEL}`."
            )
        response.raise_for_status()
        return response.json().get("embeddings", [])


async def embed_single(text: str, model: str | None = None) -> list[float]:
    vectors = await embed_texts([text], model=model)
    return vectors[0] if vectors else []
