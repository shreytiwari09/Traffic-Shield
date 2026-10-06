"""
Distributed LLM tracing (LLMOps) — OpenTelemetry, exported to Arize Phoenix.

Prometheus (observability.py) answers "is something wrong across all
requests?". A trace answers "what exactly happened on THIS request?": one
citizen question becomes one trace spanning all five services — guardrail
decision, the chunks retrieval returned and their scores, the exact prompt
version, the model, token counts and the answer — so a bad answer can be
opened and diagnosed (wrong chunk retrieved vs. model ignored the right one).

Spans use the OpenInference attribute conventions (openinference.span.kind,
llm.model_name, llm.token_count.*, input.value / output.value,
retrieval.documents.*), which Phoenix renders as LLM / retriever / chain
steps rather than generic HTTP spans.

Disabled unless OTEL_EXPORTER_OTLP_ENDPOINT is set. When disabled, every
span call in the code is a cheap no-op (OpenTelemetry's default tracer), so
tests and CI run without a trace backend.
"""

import logging

from opentelemetry import trace

from services.shared.settings import settings

log = logging.getLogger(__name__)

_configured = False

# Truncation for free-text attributes (answers, chunks): enough to read in the
# trace UI without shipping whole statute pages per span.
_MAX_TEXT = 4000


def tracer():
    return trace.get_tracer("trafficshield")


def clip(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + " …[truncated]"


def setup_tracing(app, service: str) -> bool:
    """Instruments a FastAPI app. Returns True if traces are being exported."""
    global _configured
    endpoint = settings.otel_exporter_otlp_endpoint
    if not endpoint:
        return False

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if not _configured:
        provider = TracerProvider(resource=Resource.create({
            "service.name": service,
            # Phoenix groups traces into projects by this attribute.
            "openinference.project.name": "traffic-shield",
        }))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        trace.set_tracer_provider(provider)
        # Outgoing inter-service calls carry the W3C traceparent header, which
        # is what stitches five services into ONE trace per question.
        HTTPXClientInstrumentor().instrument()
        _configured = True

    # exclude_spans: without it every ASGI send/receive message becomes its own
    # span — measured at ~220 spans for ONE question (retrieval makes ~40
    # internal calls), which buried the 3 spans that matter.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,health,v1/health",
                                       exclude_spans=["receive", "send"])
    log.info("tracing enabled: %s -> %s", service, endpoint)
    return True
