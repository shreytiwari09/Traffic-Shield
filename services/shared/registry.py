"""
Model + prompt registry and canary routing (LLMOps).

One place that answers "what is this deployment running?": which models are
serving production traffic, which exist only for the Eval-tab comparison, and
which prompt version is stable vs. under canary test. Exposed at
Orchestration's GET /v1/registry and recorded alongside every answer.

CANARY ROUTING
`canary_percent` of Ask traffic is sent to the candidate prompt. Assignment is
a deterministic hash of the routing key (the conversation id when there is
one), not random: a citizen mid-conversation must not flip between prompt
versions turn to turn, and the same key always reproduces the same arm when
debugging. Each answer's prompt_version label then lets the dashboard compare
hallucination rate, confidence and user feedback between the two arms on real
traffic before the candidate is promoted.
"""

import hashlib

from services.shared.prompts import PROMPT_REGISTRY, prompt_fingerprint
from services.shared.settings import settings

# Stage: "production" = serves citizen Ask traffic; "eval-only" = exists for the
# Week-4 model comparison (Eval tab / offline harness) and never answers users.
MODEL_REGISTRY: dict[str, dict] = {
    "llama3.1:8b": {"provider": "ollama", "stage": "production", "hosting": "local (CPU)"},
    "gemini-3.5-flash-lite": {"provider": "gemini", "stage": "production", "hosting": "Google API"},
    "codellama:7b": {"provider": "ollama", "stage": "eval-only", "hosting": "local (CPU)"},
    "starcoder2:3b": {"provider": "ollama", "stage": "eval-only", "hosting": "local (CPU)"},
}


def _bucket(routing_key: str) -> int:
    """Stable 0-99 bucket for a routing key."""
    return int(hashlib.sha256(routing_key.encode("utf-8")).hexdigest()[:8], 16) % 100


def choose_prompt_version(routing_key: str) -> tuple[str, str]:
    """Returns (prompt_version, arm) where arm is "stable" or "canary"."""
    percent = max(0, min(100, settings.canary_percent))
    canary = settings.canary_prompt_version
    if percent and canary in PROMPT_REGISTRY and _bucket(routing_key) < percent:
        return canary, "canary"
    return settings.stable_prompt_version, "stable"


def describe() -> dict:
    return {
        "models": MODEL_REGISTRY,
        "prompts": {
            name: {
                "fingerprint": prompt_fingerprint(name),
                "role": (
                    "stable" if name == settings.stable_prompt_version
                    else "canary" if name == settings.canary_prompt_version and settings.canary_percent > 0
                    else "registered"
                ),
            }
            for name in PROMPT_REGISTRY
        },
        "canary": {
            "prompt_version": settings.canary_prompt_version,
            "percent": settings.canary_percent,
        },
    }
