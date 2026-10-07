"""
Is Ollama usable on this machine?

Ollama is host infrastructure the app points at, not something it bundles, so
one teammate's laptop has it and another's does not. Without this check a
machine with no Ollama still sent every embedding and generation call to
localhost:11434, and each one either failed or hung until
ollama_timeout_seconds. With it, callers ask once and route around Ollama:
embeddings fall back to the same model run in-process (local_embedder.py), and
Ollama generation fails fast with a message saying to pick Gemini.

The answer is cached for ollama_probe_ttl_seconds, so starting Ollama later is
picked up without restarting any service, and a busy service does not ping
Ollama on every request.
"""

import time
from dataclasses import dataclass

import httpx

from services.shared.settings import settings

_PROBE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    reason: str
    # Pulled model names, e.g. {"llama3.1:8b", "nomic-embed-text:latest"}.
    # None means "not checked" (OLLAMA_MODE=always) — treated as having everything.
    models: frozenset[str] | None = None

    def has_model(self, name: str) -> bool:
        if not self.available:
            return False
        if self.models is None:
            return True
        # Ollama lists a model pulled as "nomic-embed-text" as "nomic-embed-text:latest".
        return name in self.models or f"{name}:latest" in self.models


_cache: tuple[float, OllamaStatus] | None = None


async def _probe() -> OllamaStatus:
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
    except httpx.HTTPError as exc:
        return OllamaStatus(False, f"not reachable at {settings.ollama_base_url} ({type(exc).__name__})")
    if response.status_code != 200:
        return OllamaStatus(False, f"{settings.ollama_base_url} answered HTTP {response.status_code}")
    models = frozenset(m.get("name", "") for m in response.json().get("models", []))
    return OllamaStatus(True, f"running at {settings.ollama_base_url}", models)


async def ollama_status(force: bool = False) -> OllamaStatus:
    global _cache
    if settings.ollama_mode == "never":
        return OllamaStatus(False, "disabled by OLLAMA_MODE=never")
    if settings.ollama_mode == "always":
        return OllamaStatus(True, "assumed present (OLLAMA_MODE=always)")

    now = time.monotonic()
    if not force and _cache is not None and now - _cache[0] < settings.ollama_probe_ttl_seconds:
        return _cache[1]
    status = await _probe()
    _cache = (now, status)
    return status


def clear_cache() -> None:
    global _cache
    _cache = None
