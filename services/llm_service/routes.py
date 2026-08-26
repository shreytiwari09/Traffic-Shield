import time

import httpx
from fastapi import APIRouter, HTTPException

from services.llm_service import gemini_client, ollama_client
from services.shared.prompts import build_system_message
from services.shared.schemas import EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse
from services.shared.settings import settings

router = APIRouter()


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
    system_message = build_system_message(context_dicts) if req.use_persona else None

    try:
        if req.provider == "gemini":
            answer = await gemini_client.generate(req.question, system_message)
            model = settings.gemini_model
        else:
            answer = await ollama_client.generate(req.question, system_message)
            model = settings.ollama_model
    except gemini_client.GeminiNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{req.provider} generation failed: {exc}") from exc

    latency_ms = (time.perf_counter() - started) * 1000
    return GenerateResponse(
        answer=answer,
        provider=req.provider,
        model=model,
        used_context=bool(req.context),
        latency_ms=round(latency_ms, 1),
    )
