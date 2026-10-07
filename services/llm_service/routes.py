import asyncio
import time

from fastapi import APIRouter, HTTPException

from opentelemetry.trace import Status, StatusCode

from services.llm_service import gemini_client, ollama_client
from services.llm_service.usage import estimate_cost_usd
from services.shared import local_embedder
from services.shared.observability import EMBEDDING_REQUESTS, LLM_GENERATION_SECONDS, record_usage
from services.shared.ollama_probe import ollama_status
from services.shared.prompts import PROMPT_REGISTRY, STABLE_PROMPT_VERSION, build_system_message, prompt_fingerprint
from services.shared.tracing import clip, tracer
from services.shared.schemas import EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse
from services.shared.settings import settings

router = APIRouter()

# Eval-tab-only providers that still run through Ollama, just with a
# different pulled model instead of settings.ollama_model. See EvalProvider
# in schemas.py for why these exist and why they're excluded from the
# regular Provider type used by the Ask tab.
_OLLAMA_MODEL_OVERRIDES = {"codellama": "codellama:7b", "starcoder2": "starcoder2:3b"}


@router.get("/v1/health")
async def health():
    ollama = await ollama_status(force=True)
    return {
        "status": "ok",
        "ollama_model": settings.ollama_model,
        "gemini_model": settings.gemini_model,
        "ollama_reachable": ollama.available,
        "ollama_status": ollama.reason,
        "ollama_model_pulled": ollama.has_model(settings.ollama_model),
        "embedding_backend": "ollama" if ollama.has_model(settings.ollama_embedding_model) else "local",
        "gemini_configured": bool(settings.gemini_api_key),
        "prompt_versions": {name: prompt_fingerprint(name) for name in PROMPT_REGISTRY},
    }


@router.post("/v1/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """Ollama's nomic-embed-text when this machine has it, otherwise the same
    model in-process (local_embedder.py). Both produce the vector space the
    Chroma index was built in, so retrieval works the same either way."""
    ollama = await ollama_status()
    if ollama.has_model(settings.ollama_embedding_model):
        try:
            vector = await ollama_client.embed(req.text)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
        if not vector:
            raise HTTPException(status_code=502, detail="Ollama returned no embedding")
        EMBEDDING_REQUESTS.labels("ollama").inc()
        return EmbedResponse(embedding=vector, model=settings.ollama_embedding_model, dimensions=len(vector))

    try:
        vector = await asyncio.to_thread(local_embedder.embed_text, req.text)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"embedding failed: Ollama is unavailable ({ollama.reason}) and the local model failed: {exc}",
        ) from exc
    EMBEDDING_REQUESTS.labels("local").inc()
    return EmbedResponse(embedding=vector, model=settings.local_embedding_model, dimensions=len(vector),
                         backend="local")


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    started = time.perf_counter()
    context_dicts = [c.model_dump() for c in req.context]
    prompt_version = (req.prompt_version or STABLE_PROMPT_VERSION) if req.use_persona else None
    try:
        system_message = build_system_message(context_dicts, prompt_version) if prompt_version else None
    except ValueError as exc:  # unknown prompt version
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    history = [m.model_dump() for m in req.history]
    model = settings.gemini_model if req.provider == "gemini" else _OLLAMA_MODEL_OVERRIDES.get(req.provider, settings.ollama_model)

    if req.provider != "gemini":
        ollama = await ollama_status()
        if not ollama.available:
            # Fail in milliseconds rather than letting the request wait out
            # ollama_timeout_seconds against a server that is not there.
            LLM_GENERATION_SECONDS.labels(req.provider, model, "unavailable").observe(time.perf_counter() - started)
            raise HTTPException(
                status_code=503,
                detail=f"Ollama is not available on this device ({ollama.reason}). Choose Gemini instead.",
            )

    with tracer().start_as_current_span("llm.generate") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.provider", req.provider)
        span.set_attribute("llm.model_name", model)
        span.set_attribute("llm.prompt_template.version", prompt_version or "none")
        span.set_attribute("input.value", clip(req.question))
        span.set_attribute("llm.context_chunks", len(req.context))
        span.set_attribute("llm.history_turns", len(history))
        try:
            if req.provider == "gemini":
                result = await gemini_client.generate_with_usage(req.question, system_message, history=history)
            else:
                result = await ollama_client.generate_with_usage(req.question, system_message, model=model,
                                                                 history=history)
        except gemini_client.GeminiNotConfigured as exc:
            LLM_GENERATION_SECONDS.labels(req.provider, model, "not_configured").observe(time.perf_counter() - started)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            LLM_GENERATION_SECONDS.labels(req.provider, model, "error").observe(time.perf_counter() - started)
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise HTTPException(status_code=502, detail=f"{req.provider} generation failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        cost_usd = estimate_cost_usd(req.provider, result.prompt_tokens, result.completion_tokens)
        LLM_GENERATION_SECONDS.labels(req.provider, model, "ok").observe(latency_ms / 1000)
        record_usage(req.provider, model, result.prompt_tokens, result.completion_tokens, cost_usd)

        span.set_attribute("output.value", clip(result.text))
        if result.prompt_tokens is not None:
            span.set_attribute("llm.token_count.prompt", result.prompt_tokens)
        if result.completion_tokens is not None:
            span.set_attribute("llm.token_count.completion", result.completion_tokens)
        if result.prompt_tokens is not None and result.completion_tokens is not None:
            span.set_attribute("llm.token_count.total", result.prompt_tokens + result.completion_tokens)
        span.set_attribute("llm.cost_usd", cost_usd)

    return GenerateResponse(
        answer=result.text,
        provider=req.provider,
        model=model,
        used_context=bool(req.context),
        latency_ms=round(latency_ms, 1),
        prompt_version=prompt_version,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=cost_usd,
    )
