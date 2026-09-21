"""
Retrieval Service — hybrid retrieval: query embedding (via LLM Service) ->
vector search (via Data Service) -> graph lookup (Cypher against Neo4j, or the
flat-JSON store as fallback — see graph_store.py) -> fusion/ranking. Never
talks to Ollama itself; never generates text.

Run: uvicorn services.retrieval_service.main:app --port 8002
"""

from fastapi import FastAPI

from services.retrieval_service import graph_store
from services.retrieval_service.routes import router

app = FastAPI(
    title="Retrieval Service",
    description="Hybrid retrieval: vector search + Neo4j graph lookup + fusion.",
)


@app.on_event("startup")
async def startup() -> None:
    graph_store.load()


@app.on_event("shutdown")
async def shutdown() -> None:
    # Closes the Neo4j driver's connection pool. Without this, reloading under
    # uvicorn --reload leaks a pool per reload.
    try:
        from services.retrieval_service import neo4j_store
        neo4j_store.close()
    except ImportError:
        pass


app.include_router(router)
