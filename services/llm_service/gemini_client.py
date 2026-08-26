"""
Gemini provider — the second model option, so a citizen (Ask tab) or an
evaluator (Eval tab) can compare it directly against Ollama's answer for the
same question. Uses Google's unified Gen AI SDK (``google-genai``).

``GEMINI_API_KEY`` is intentionally left blank in .env.example — the user
adds their own key later. Until then this raises ``GeminiNotConfigured``,
which the route layer turns into a clear 503 instead of a crash.
"""

import asyncio

from google import genai
from google.genai import types

from services.shared.settings import settings


class GeminiNotConfigured(RuntimeError):
    pass


def _client() -> genai.Client:
    if not settings.gemini_api_key:
        raise GeminiNotConfigured("GEMINI_API_KEY is not set — add it to .env to enable the Gemini provider.")
    return genai.Client(api_key=settings.gemini_api_key)


async def generate(question: str, system_message: str | None, model: str | None = None) -> str:
    model = model or settings.gemini_model
    client = _client()

    def _call() -> str:
        # None/"" -> no system_instruction at all: the raw model's own
        # behavior, used only by the Eval tab's no-retrieval cells.
        config = types.GenerateContentConfig(system_instruction=system_message) if system_message else None
        response = client.models.generate_content(model=model, contents=question, config=config)
        return response.text or ""

    # The SDK is synchronous; run it off the event loop so one slow Gemini
    # call doesn't block the other services' requests to this process.
    return await asyncio.to_thread(_call)
