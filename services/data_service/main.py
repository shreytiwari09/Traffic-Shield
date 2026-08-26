"""
Data Service — owns DATA/dataset.jsonl (in-memory) and the Chroma vector
store built from DATA/chunks.jsonl. Never talks to Ollama; never reads the
graph JSON (that's Retrieval Service's job).

Run: uvicorn services.data_service.main:app --port 8004
"""

from fastapi import FastAPI

from services.data_service import dataset_store
from services.data_service.routes import router

app = FastAPI(
    title="Data Service",
    description="Owns dataset.jsonl and the Chroma vector store (trafficshield_sections).",
)


@app.on_event("startup")
async def startup() -> None:
    dataset_store.load()


app.include_router(router)
