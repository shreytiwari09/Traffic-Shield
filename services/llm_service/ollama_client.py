"""
Ollama provider — the only place in the whole app that talks to Ollama.
Generation via ``/api/chat`` (system/user role split carries the persona
cleanly); embeddings via ``/api/embed`` (same endpoint data_pipeline's own
embedder.py already uses, so retrieval-time and ingestion-time embeddings are
guaranteed to come from the same model/vector space).

Retries on 500s: on a GPU-constrained host (originally observed on a 6GB GPU
shared between llama3.1:8b (~4.7GB) and nomic-embed-text, with very little
headroom) Ollama's llama-runner subprocess occasionally crashes under memory
pressure ("llama runner process has terminated") when swapping models —
confirmed via nvidia-smi + `ollama ps` during testing, not a bug in the
prompt or this code. A short retry reliably recovers (observed repeatedly),
so it's cheaper and more honest than pretending prompt size was the cause.

Note this is hardware-specific and NOT true of every machine in the group: on
a CPU-only host (no discrete GPU) there is no GPU contention at all, and the
failure mode is the opposite one — everything simply runs slowly, so timeouts
rather than 500s are what bite. Both are handled below.
"""

import asyncio

import httpx

from services.shared.settings import settings

_TIMEOUT = settings.ollama_timeout_seconds
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2.0


async def _post_with_retry(url: str, json: dict) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(url, json=json)
        except httpx.TimeoutException as exc:
            # A timeout is a raised exception, not a status code, so it never
            # reached the 5xx retry branch below — it propagated straight out.
            # Worse, str(httpx.ReadTimeout()) is '', so the UI showed a bare
            # "ollama generation failed:" with no cause at all. Re-raise with a
            # message that actually says what happened and how to fix it.
            raise RuntimeError(
                f"Ollama did not respond within {_TIMEOUT:.0f}s for model "
                f"'{json.get('model')}'. On a CPU-only host a cold model load plus a long "
                f"RAG prompt can exceed this — raise OLLAMA_TIMEOUT_SECONDS in .env, or keep "
                f"the model warm (OLLAMA_KEEP_ALIVE)."
            ) from exc
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama has no model '{json.get('model')}'. Pull it first: "
                f"`ollama pull {json.get('model')}`."
            )
        if response.status_code >= 500 and attempt < _MAX_ATTEMPTS:
            last_exc = httpx.HTTPStatusError(
                f"Ollama returned {response.status_code} (attempt {attempt}/{_MAX_ATTEMPTS})",
                request=response.request,
                response=response,
            )
            await asyncio.sleep(_RETRY_DELAY_SECONDS)
            continue
        response.raise_for_status()
        return response
    raise last_exc  # only reached if every attempt hit a 5xx


async def generate(question: str, system_message: str | None, model: str | None = None) -> str:
    model = model or settings.ollama_model
    messages = []
    if system_message:  # None or "" -> no system turn at all (raw model, Eval tab's no-retrieval cells)
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": question})

    response = await _post_with_retry(
        f"{settings.ollama_base_url}/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            # Keeps the model resident past Ollama's default 5-minute idle
            # eviction, so the next question doesn't pay a multi-GB cold load.
            "keep_alive": settings.ollama_keep_alive,
        },
    )
    return response.json().get("message", {}).get("content", "")


async def embed(text: str, model: str | None = None) -> list[float]:
    model = model or settings.ollama_embedding_model
    response = await _post_with_retry(
        f"{settings.ollama_base_url}/api/embed",
        {"model": model, "input": [text], "keep_alive": settings.ollama_keep_alive},
    )
    vectors = response.json().get("embeddings", [])
    return vectors[0] if vectors else []
