"""
Retrieval Service — hybrid retrieval: query embedding (via LLM Service) ->
vector search (via Data Service) -> graph lookup (loaded from the flat-JSON
files at startup) -> fusion/ranking. Never talks to Ollama itself; never
generates text.

Run: uvicorn services.retrieval_service.main:app --port 8002
"""

from fastapi import FastAPI

from services.retrieval_service import graph_store
from services.retrieval_service.routes import router

app = FastAPI(
    title="Retrieval Service",
    description="Hybrid retrieval: vector search + flat-JSON graph lookup + fusion.",
)


@app.on_event("startup")
async def startup() -> None:
    graph_store.load()


app.include_router(router)
