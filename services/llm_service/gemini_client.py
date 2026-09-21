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


def _contents(question: str, history: list[dict] | None) -> list[types.Content]:
    """Prior turns + the current question, in the shape the Gen AI SDK wants.

    Note the role rename: Gemini calls the model's own turns "model", where
    Ollama (and this app's own schema) call them "assistant". Passing
    "assistant" through is silently accepted by the SDK but breaks the
    alternation the model is trained on, so it is mapped here rather than at
    the call site.
    """
    contents: list[types.Content] = []
    for turn in history or []:
        content = turn.get("content")
        if not content:
            continue
        role = turn.get("role")
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    return contents


async def generate(
    question: str,
    system_message: str | None,
    model: str | None = None,
    history: list[dict] | None = None,
) -> str:
    model = model or settings.gemini_model
    client = _client()
    contents = _contents(question, history)

    def _call() -> str:
        # None/"" -> no system_instruction at all: the raw model's own
        # behavior, used only by the Eval tab's no-retrieval cells.
        config = types.GenerateContentConfig(system_instruction=system_message) if system_message else None
        response = client.models.generate_content(model=model, contents=contents, config=config)
        return response.text or ""

    # The SDK is synchronous; run it off the event loop so one slow Gemini
    # call doesn't block the other services' requests to this process.
    return await asyncio.to_thread(_call)
