import time

import httpx
from fastapi import APIRouter, HTTPException

from opentelemetry.trace import Status, StatusCode

from services.llm_service import gemini_client, ollama_client
from services.llm_service.usage import estimate_cost_usd
from services.shared.observability import LLM_GENERATION_SECONDS, record_usage
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
    ollama_reachable = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_reachable = r.status_code == 200
    except Exception:
        ollama_reachable = False

    return {
        "status": "ok",
        "ollama_model": settings.ollama_model,
        "gemini_model": settings.gemini_model,
        "ollama_reachable": ollama_reachable,
        "gemini_configured": bool(settings.gemini_api_key),
        "prompt_versions": {name: prompt_fingerprint(name) for name in PROMPT_REGISTRY},
    }


@router.post("/v1/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    try:
        vector = await ollama_client.embed(req.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
    if not vector:
        raise HTTPException(status_code=502, detail="Ollama returned no embedding")
    return EmbedResponse(embedding=vector, model=settings.ollama_embedding_model, dimensions=len(vector))


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
