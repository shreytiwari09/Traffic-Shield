"""
Orchestration Service — coordinates one user request end to end: calls
Retrieval Service, then LLM Service (once for the Ask tab, four times for
the Eval tab's comparison grid), and assembles citations. Never touches
Chroma, the graph JSON, or Ollama/Gemini directly.

Run: uvicorn services.orchestration_service.main:app --port 8001
"""

from fastapi import FastAPI

from services.orchestration_service.routes import router

app = FastAPI(
    title="Orchestration Service",
    description="Sequences one user request across Retrieval Service and LLM Service.",
)
app.include_router(router)
