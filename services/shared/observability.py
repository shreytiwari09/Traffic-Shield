"""
Prometheus metrics shared by all five services — the "monitor" stage of the
AIDevOps loop (see docs/AIDEVOPS_MIDTERM_NOTES.md).

Two kinds of metric live here:

1. Generic HTTP golden signals (traffic, errors, latency) recorded by one
   ASGI middleware for every service, labelled by the ROUTE TEMPLATE
   (e.g. /v1/categories/{slug}/sections) rather than the raw path, so a
   path parameter can never explode label cardinality.

2. AI-specific signals that plain HTTP metrics cannot see — the ones that
   tell you the *model* is degrading even while every request returns 200:
   guardrail decisions, answer confidence, grounding (hallucination) claims,
   LLM generation latency per provider/model, retrieval stage timings, the
   top retrieval similarity score (a cheap drift signal: if questions stop
   resembling the corpus, this distribution slides down), and whether the
   graph backend has silently fallen back from Neo4j to JSON.

Each service is a single uvicorn process, so the default prometheus_client
registry is correct here — no multiprocess mode needed.
"""

import json
import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# HTTP golden signals
# ---------------------------------------------------------------------------
HTTP_REQUESTS = Counter(
    "ts_http_requests_total",
    "HTTP requests handled, by service, method, route template and status code.",
    ["service", "method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "ts_http_request_duration_seconds",
    "Wall-clock time to fully handle an HTTP request (including streamed bodies).",
    ["service", "route"],
    # Wide on purpose: health checks take milliseconds, a CPU-only Ollama
    # answer takes minutes (see evaluation/run_meta.json).
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
HTTP_IN_PROGRESS = Gauge(
    "ts_http_requests_in_progress",
    "Requests currently being handled.",
    ["service"],
)

# ---------------------------------------------------------------------------
# AI / RAG-specific signals
# ---------------------------------------------------------------------------
GUARDRAIL_DECISIONS = Counter(
    "ts_guardrail_decisions_total",
    "Input guardrail outcomes: allowed, redacted (PII) or blocked.",
    ["outcome", "check"],
)
ANSWER_CONFIDENCE = Counter(
    "ts_answer_confidence_total",
    "Answers served, by the retrieval-evidence confidence badge shown to the user.",
    ["level", "prompt_version"],
)
GROUNDING_CLAIMS = Counter(
    "ts_grounding_claims_total",
    "Section/rupee claims in generated answers, checked against retrieved context.",
    ["result", "prompt_version"],
)
LLM_GENERATION_SECONDS = Histogram(
    "ts_llm_generation_seconds",
    "Time for the LLM provider to return a full answer.",
    ["provider", "model", "status"],
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120, 180, 300, 600, 900),
)
RETRIEVAL_STAGE_SECONDS = Histogram(
    "ts_retrieval_stage_seconds",
    "Time spent in each hybrid-retrieval stage.",
    ["stage"],
    # Up to 2 min: on the CPU-only dev box the graph stage alone measured
    # ~24 s/query, so a 30 s top bucket pinned its p95 at exactly "30 s".
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 45, 60, 120),
)
RETRIEVAL_TOP_SCORE = Histogram(
    "ts_retrieval_top_score",
    "Best cosine similarity among retrieved chunks per query (drift signal).",
    buckets=(0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0),
)
GRAPH_BACKEND_FALLBACK = Gauge(
    "ts_graph_backend_fell_back",
    "1 when the graph store silently fell back from Neo4j to the JSON store.",
)

# ---------------------------------------------------------------------------
# LLMOps signals: tokens, cost, user feedback, prompt versions
# ---------------------------------------------------------------------------
LLM_TOKENS = Counter(
    "ts_llm_tokens_total",
    "Tokens processed by the LLM, as reported by the provider.",
    ["provider", "model", "type"],  # type: prompt | completion
)
LLM_COST_USD = Counter(
    "ts_llm_cost_usd_total",
    "Estimated LLM API spend in USD (local Ollama = 0; Gemini priced from settings).",
    ["provider", "model"],
)
USER_FEEDBACK = Counter(
    "ts_user_feedback_total",
    "Citizen thumbs-up / thumbs-down on answers.",
    ["rating", "prompt_version"],
)
PROMPT_ROUTING = Counter(
    "ts_prompt_routing_total",
    "Ask requests per prompt version and canary arm.",
    ["prompt_version", "arm"],
)


# Pre-create the label combinations the dashboard queries, at zero. Prometheus'
# increase() cannot see a counter's FIRST value (there is no earlier sample to
# diff against), so without this the first burst of blocked requests after a
# restart never shows up — found when 3 blocked jailbreaks read as "0" in Grafana.
for _outcome, _check in (("allowed", "none"), ("blocked", "prompt_injection"), ("blocked", "illegal_conduct")):
    GUARDRAIL_DECISIONS.labels(outcome=_outcome, check=_check)

def _init_prompt_labels() -> None:
    from services.shared.prompts import PROMPT_REGISTRY

    for version in PROMPT_REGISTRY:
        for level in ("high", "medium", "low", "none"):
            ANSWER_CONFIDENCE.labels(level=level, prompt_version=version)
        for result in ("verified", "unverified"):
            GROUNDING_CLAIMS.labels(result=result, prompt_version=version)
        for rating in ("up", "down"):
            USER_FEEDBACK.labels(rating=rating, prompt_version=version)


_init_prompt_labels()


def record_guardrail(report) -> None:
    """report: services.shared.schemas.GuardrailReport."""
    if report.blocked:
        check = report.flags[0].check if report.flags else "unknown"
        GUARDRAIL_DECISIONS.labels(outcome="blocked", check=check).inc()
    elif report.pii_redacted:
        for flag in report.flags:
            GUARDRAIL_DECISIONS.labels(outcome="redacted", check=flag.pattern_matched or flag.check).inc()
    else:
        GUARDRAIL_DECISIONS.labels(outcome="allowed", check="none").inc()


def record_answer(confidence: str, grounding: dict, prompt_version: str) -> None:
    ANSWER_CONFIDENCE.labels(level=confidence, prompt_version=prompt_version).inc()
    GROUNDING_CLAIMS.labels(result="verified", prompt_version=prompt_version).inc(grounding.get("verified_claims", 0))
    GROUNDING_CLAIMS.labels(result="unverified", prompt_version=prompt_version).inc(
        grounding.get("unverified_claims", 0))


def record_usage(provider: str, model: str, prompt_tokens: int | None, completion_tokens: int | None,
                 cost_usd: float) -> None:
    if prompt_tokens:
        LLM_TOKENS.labels(provider, model, "prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(provider, model, "completion").inc(completion_tokens)
    LLM_COST_USD.labels(provider, model).inc(cost_usd)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
class PrometheusMiddleware:
    """Pure ASGI rather than BaseHTTPMiddleware: the Ask tab's SSE stream is a
    StreamingResponse, and timing must cover the whole streamed body — not
    just the moment headers were sent."""

    def __init__(self, app, service: str):
        self.app = app
        self.service = service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") == "/metrics":
            await self.app(scope, receive, send)
            return

        status_holder = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        HTTP_IN_PROGRESS.labels(self.service).inc()
        started = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            HTTP_IN_PROGRESS.labels(self.service).dec()
            # FastAPI's router writes the matched APIRoute into the shared
            # scope; anything unmatched (404 probes, scanners) collapses into
            # one label instead of one series per random path.
            route = scope.get("route")
            route_label = getattr(route, "path", None) or "unmatched"
            HTTP_REQUESTS.labels(self.service, scope["method"], route_label, str(status_holder["code"])).inc()
            HTTP_LATENCY.labels(self.service, route_label).observe(elapsed)


def setup_metrics(app, service: str) -> None:
    """Adds request instrumentation and a /metrics endpoint to a FastAPI app."""
    app.add_middleware(PrometheusMiddleware, service=service)

    @app.get("/metrics", include_in_schema=False)
    async def metrics(_: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Scheduled-eval scores (exposed by Application Service only)
# ---------------------------------------------------------------------------
class EvalHistoryCollector:
    """Exposes the LATEST line of evaluation/eval_history.jsonl (written by
    evaluation/scheduled_eval.py) as gauges at scrape time, so offline eval
    quality sits on the same dashboard as live traffic. Read on every scrape:
    a new eval run shows up without restarting anything."""

    def __init__(self, path):
        self.path = path

    def _latest(self) -> dict | None:
        try:
            lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return json.loads(lines[-1]) if lines else None
        except (OSError, ValueError):
            return None

    def collect(self):
        from prometheus_client.core import GaugeMetricFamily

        latest = self._latest()
        scores = GaugeMetricFamily("ts_eval_score", "Latest scheduled online-eval metric value.",
                                   labels=["metric", "provider"])
        last_run = GaugeMetricFamily("ts_eval_last_run_timestamp_seconds",
                                     "Unix time of the latest scheduled eval run.")
        if latest:
            for name, value in (latest.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    scores.add_metric([name, latest.get("provider", "unknown")], float(value))
            try:
                from datetime import datetime

                last_run.add_metric([], datetime.fromisoformat(latest["timestamp"]).timestamp())
            except (KeyError, ValueError):
                pass
        yield scores
        yield last_run


def register_eval_history_collector(path) -> None:
    from prometheus_client import REGISTRY

    REGISTRY.register(EvalHistoryCollector(path))
