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
