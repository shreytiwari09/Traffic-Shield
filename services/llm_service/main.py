"""
LLM Service — the only service that talks to Ollama or Gemini. It knows
nothing about retrieval: it receives an opaque ``context`` list (possibly
empty) and a ``provider`` choice, and returns a generated answer.

Run: uvicorn services.llm_service.main:app --port 8003
"""

from fastapi import FastAPI

from services.llm_service.routes import router

app = FastAPI(
    title="LLM Service",
    description="Talks to Ollama (llama3.1:8b) and Gemini. No retrieval logic lives here.",
)
app.include_router(router)
