"""
Application Service — the citizen-facing entry point (Ask tab) and the
evaluator/project-demo surface (Eval tab). No retrieval logic, no LLM calls,
no data access happen here — everything is a passthrough to Orchestration
Service.

Run: uvicorn services.app_service.main:app --port 8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from services.app_service.routes import router
from services.shared.observability import register_eval_history_collector, setup_metrics
from services.shared.settings import PROJECT_ROOT
from services.shared.tracing import setup_tracing

app = FastAPI(
    title="Haryana Traffic Legal Assistant",
    description="Ask tab (citizen-facing) + Eval tab (model/RAG comparison for evaluators).",
)
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="static",
)
app.include_router(router)
setup_metrics(app, "app_service")
setup_tracing(app, "app_service")
# Latest scheduled online-eval scores (evaluation/scheduled_eval.py) on /metrics.
register_eval_history_collector(PROJECT_ROOT / "evaluation" / "eval_history.jsonl")
