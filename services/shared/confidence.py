"""
A simple, honestly-labeled heuristic for how much retrieval evidence backs an
answer — NOT a calibrated probability, just a rough signal for the UI's
confidence badge ("Retrieved from official source"), based on vector
similarity plus graph corroboration. Shared by Orchestration's plain and
streaming Ask routes so both report the same number for the same request.
"""

from typing import Literal

Confidence = Literal["high", "medium", "low", "none"]

# Thresholds picked from what real queries against this corpus produced during
# testing: a strong on-topic match sits around 0.65-0.75 (score = 1 - cosine
# distance); a weak/tangential one sits below 0.5.
HIGH_SCORE = 0.62
MEDIUM_SCORE = 0.5


def compute_confidence(context: list[dict], matched_entities: list[str]) -> Confidence:
    if not context:
        return "none"

    vector_scores = [
        c["score"] for c in context if c.get("source") == "vector" and c.get("score") is not None
    ]
    top_score = max(vector_scores) if vector_scores else 0.0
    graph_support = bool(matched_entities)

    if top_score >= HIGH_SCORE and graph_support:
        return "high"
    if top_score >= MEDIUM_SCORE or graph_support:
        return "medium"
    return "low"
