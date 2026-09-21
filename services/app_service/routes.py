import json
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from services.app_service import clients
from services.shared.schemas import AskRequest, EvalRequest, RestoreConversationRequest
from services.shared.settings import PROJECT_ROOT, settings

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
        return await clients.ask(req.question, req.provider, req.conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@router.post("/api/conversations/{conversation_id}/restore")
async def api_restore_conversation(conversation_id: str, req: RestoreConversationRequest):
    try:
        return await clients.restore_conversation(conversation_id, [m.model_dump() for m in req.messages])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@router.delete("/api/conversations/{conversation_id}")
async def api_reset_conversation(conversation_id: str):
    try:
        return await clients.reset_conversation(conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@router.post("/api/eval")
async def api_eval(req: EvalRequest):
    try:
        return await clients.eval_grid(req.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc)) from exc


@router.get("/api/ask/stream")
async def api_ask_stream(question: str, provider: str = "ollama", conversation_id: str | None = None):
    """Relays Orchestration's real live-progress SSE stream straight through —
    Application Service does not touch or reinterpret the events, it's a pure
    passthrough so the browser sees exactly what Orchestration actually did."""

    params = {"question": question, "provider": provider}
    if conversation_id:
        params["conversation_id"] = conversation_id

    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET",
                f"{settings.orchestration_service_url}/v1/ask/stream",
                params=params,
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/eval/report")
async def api_eval_report():
    """Serves the offline harness's metrics report to the RAG Evaluation tab.

    Read straight off disk rather than proxied through Orchestration, and that
    is deliberate: metrics_report.json is a build artifact of the OFFLINE
    evaluation sweep, not request-time data. No live service owns it — the same
    reasoning that keeps data_pipeline/ outside the five services. Routing it
    through Orchestration would imply it is part of answering a question, which
    it is not.

    404 (not 500) when absent: a fresh clone simply has not run the harness
    yet, and the tab renders a "run the harness" message instead of an error.
    """
    report_path = PROJECT_ROOT / "evaluation" / "metrics_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation report yet. Run: python -m evaluation.run_eval "
                   "then python -m evaluation.analyze",
        )
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"metrics_report.json is not valid JSON: {exc}") from exc


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
