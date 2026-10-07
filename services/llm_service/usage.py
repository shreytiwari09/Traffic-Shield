"""Token usage + cost for one generation (LLMOps cost tracking).

Token counts come from the providers themselves (Ollama's prompt_eval_count /
eval_count, Gemini's usage_metadata), never estimated from string length.
"""

from dataclasses import dataclass

from services.shared.settings import settings


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def estimate_cost_usd(provider: str, prompt_tokens: int | None, completion_tokens: int | None) -> float:
    """Local Ollama inference has no per-token API price (its cost is the
    hardware, visible as latency instead). Gemini is priced per 1M tokens from
    settings, which are estimates until set to the account's real pricing."""
    if provider != "gemini":
        return 0.0
    return round(
        (prompt_tokens or 0) / 1_000_000 * settings.gemini_input_usd_per_1m
        + (completion_tokens or 0) / 1_000_000 * settings.gemini_output_usd_per_1m,
        8,
    )
