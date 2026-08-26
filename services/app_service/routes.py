from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from services.app_service import clients
from services.shared.schemas import AskRequest, EvalRequest
from services.shared.settings import settings

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter()


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return exc.response.json().get("detail", str(exc))
        except Exception:
            return str(exc)
    return str(exc)


@router.get("/", response_class=HTMLResponse)
async def ask_tab(request: Request):
    return templates.TemplateResponse("ask.html", {"request": request})


@router.get("/eval", response_class=HTMLResponse)
async def eval_tab(request: Request):
    return templates.TemplateResponse("eval.html", {"request": request})


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/api/ask")
async def api_ask(req: AskRequest):
    try:
        return await clients.ask(req.question, req.provider)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@router.post("/api/eval")
async def api_eval(req: EvalRequest):
    try:
        return await clients.eval_grid(req.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@router.get("/api/ask/stream")
async def api_ask_stream(question: str, provider: str = "ollama"):
    """Relays Orchestration's real live-progress SSE stream straight through —
    Application Service does not touch or reinterpret the events, it's a pure
    passthrough so the browser sees exactly what Orchestration actually did."""

    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET",
                f"{settings.orchestration_service_url}/v1/ask/stream",
                params={"question": question, "provider": provider},
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/categories")
async def api_list_categories():
    try:
        return await clients.list_categories()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@router.get("/api/categories/{slug}/sections")
async def api_category_sections(slug: str):
    try:
        return await clients.category_sections(slug)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc
