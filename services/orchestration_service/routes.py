import json
import time

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from services.orchestration_service import clients, conversation, regex_guardrails
from services.orchestration_service.grounding import check_grounding
from services.shared.confidence import compute_confidence
from services.shared.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    EvalCell,
    EvalRequest,
    EvalResponse,
    GroundingCheck,
    RestoreConversationRequest,
)
from services.shared.settings import settings

router = APIRouter()

_NO_CLAIMS = GroundingCheck(total_claims=0, verified_claims=0, unverified_claims=0)


def _citations(context: list[dict]) -> list[Citation]:
    return [
        Citation(act=c.get("act"), section=c.get("section"), page=c.get("page"), source_pdf=c.get("source_pdf"))
        for c in context
    ]


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return exc.response.json().get("detail", str(exc))
        except Exception:
            return str(exc)
    return str(exc)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.get("/v1/health")
async def health():
    return {"status": "ok"}


@router.get("/v1/categories")
async def list_categories():
    try:
        return await clients.list_categories()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"categories failed: {_error_detail(exc)}") from exc


@router.get("/v1/categories/{slug}/sections")
async def category_sections(slug: str):
    response = await clients.category_sections(slug)
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"unknown category '{slug}'")
    response.raise_for_status()
    return response.json()


@router.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Ask tab — resolves the turn against the conversation, retrieves hybrid
    RAG context, then generates with the citizen's chosen provider and the
    prior turns in scope."""
    # 1. Regex Guardrails input inspection (injection, illegal advice, PII)
    guardrail_report = regex_guardrails.inspect_input(req.question)
    if guardrail_report.blocked:
        return AskResponse(
            answer=guardrail_report.refusal_reason or "Request blocked by safety policy.",
            citations=[],
            provider=req.provider,
            model="guardrail-regex",
            used_context=False,
            confidence="none",
            context=[],
            matched_entities=[],
            grounding=_NO_CLAIMS,
            conversation_id=req.conversation_id,
            retrieval_query="",
            history_turns=0,
            guardrails=guardrail_report,
        )

    clean_question = guardrail_report.sanitized_question or req.question
    history = conversation.get_history(req.conversation_id)
    retrieval_query = conversation.resolve_retrieval_query(clean_question, history)

    try:
        retrieval = await clients.retrieve(retrieval_query, settings.default_top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"retrieval failed: {_error_detail(exc)}") from exc

    context = retrieval["context"]
    matched_entities = retrieval["matched_entities"]
    try:
        result = await clients.generate(clean_question, context, req.provider, history=history)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"generation failed: {_error_detail(exc)}") from exc

    # Recorded only after a successful generation: a failed turn leaves no
    # half-conversation behind for the next question to be resolved against.
    conversation.append_turn(req.conversation_id, "user", clean_question)
    conversation.append_turn(req.conversation_id, "assistant", result["answer"])

    return AskResponse(
        answer=result["answer"],
        citations=_citations(context),
        provider=result["provider"],
        model=result["model"],
        used_context=result["used_context"],
        confidence=compute_confidence(context, matched_entities),
        context=context,
        matched_entities=matched_entities,
        grounding=GroundingCheck(**check_grounding(result["answer"], context)),
        conversation_id=req.conversation_id,
        retrieval_query=retrieval_query,
        history_turns=len(history),
        guardrails=guardrail_report,
    )


@router.post("/v1/conversations/{conversation_id}/restore")
async def restore_conversation(conversation_id: str, req: RestoreConversationRequest):
    """Rehydrate a conversation this process no longer has (reload, restart,
    TTL expiry) from the transcript the browser still holds."""
    restored = conversation.restore(conversation_id, [m.model_dump() for m in req.messages])
    return {"status": "ok", "conversation_id": conversation_id, "restored_turns": restored}


@router.delete("/v1/conversations/{conversation_id}")
async def reset_conversation(conversation_id: str):
    """Starting a new chat — drops the server-side transcript so the next
    question is resolved and generated with no memory of this one."""
    conversation.reset(conversation_id)
    return {"status": "ok", "conversation_id": conversation_id}


@router.get("/v1/ask/stream")
async def ask_stream(question: str, provider: str = "ollama", conversation_id: str | None = None):
    """The live pipeline view's real data source — emits one real event per
    real step of the SAME request flow /v1/ask performs, as each one actually
    completes. No step here is simulated: every elapsed_ms and every count is
    measured from the real inter-service calls as they happen.

    `conversation_id` rather than the transcript itself: EventSource issues a
    GET and cannot send a body, and a multi-turn transcript in a query string
    runs into URL length limits — the turns live in this service's own store
    (see conversation.py)."""

    async def gen():
        overall_start = time.perf_counter()

        # 1. Regex Guardrails inspection
        guardrail_report = regex_guardrails.inspect_input(question)
        if guardrail_report.blocked:
            yield _sse({
                "step": "guardrail_blocked",
                "message": guardrail_report.refusal_reason,
                "flags": [f.model_dump() for f in guardrail_report.flags],
            })
            yield _sse({
                "step": "done",
                "message": "Request blocked by safety guardrails",
                "elapsed_ms": 0.0,
                "total_elapsed_ms": round((time.perf_counter() - overall_start) * 1000, 1),
                "answer": guardrail_report.refusal_reason,
                "citations": [],
                "provider": provider,
                "model": "guardrail-regex",
                "used_context": False,
                "confidence": "none",
                "context": [],
                "matched_entities": [],
                "grounding": _NO_CLAIMS.model_dump(),
                "conversation_id": conversation_id,
                "retrieval_query": "",
                "history_turns": 0,
                "guardrails": guardrail_report.model_dump(),
            })
            return

        if guardrail_report.pii_redacted:
            yield _sse({
                "step": "guardrail_sanitized",
                "message": "Sensitive personal information was detected and redacted before retrieval.",
                "flags": [f.model_dump() for f in guardrail_report.flags],
            })

        clean_question = guardrail_report.sanitized_question or question
        history = conversation.get_history(conversation_id)
        retrieval_query = conversation.resolve_retrieval_query(clean_question, history)

        yield _sse({"step": "start", "message": f"Received question — provider: {provider}"})

        if history:
            yield _sse({
                "step": "conversation_context",
                "message": (
                    f"Continuing conversation — {len(history)} earlier turn(s) in scope"
                    + (
                        f"; follow-up resolved for retrieval as: \"{retrieval_query}\""
                        if retrieval_query != question else
                        "; question is self-contained, retrieved as asked"
                    )
                ),
                "history_turns": len(history),
                "retrieval_query": retrieval_query,
            })

        yield _sse({"step": "retrieval_start", "message": "Calling Retrieval Service..."})
        try:
            retrieval = await clients.retrieve(retrieval_query, settings.default_top_k)
        except Exception as exc:
            yield _sse({"step": "error", "message": f"retrieval failed: {_error_detail(exc)}"})
            return

        context = retrieval["context"]
        matched_entities = retrieval["matched_entities"]
        timing = retrieval.get("timing", {})
        n_vector = sum(1 for c in context if c.get("source") == "vector")
        n_graph = sum(1 for c in context if c.get("source") == "graph")

        yield _sse({
            "step": "embed_done",
            "message": "Question embedded via Ollama (nomic-embed-text)",
            "elapsed_ms": timing.get("embed_ms"),
        })
        yield _sse({
            "step": "vector_search_done",
            "message": f"Vector search returned {n_vector} chunk(s) from Chroma",
            "elapsed_ms": timing.get("vector_search_ms"),
        })
        yield _sse({
            "step": "graph_done",
            "message": (
                f"Graph lookup matched entities: {', '.join(matched_entities)} "
                f"({n_graph} section(s) added from graph evidence)"
                if matched_entities else
                "Graph lookup matched no known entities in the question"
            ),
            "elapsed_ms": timing.get("graph_ms"),
            "matched_entities": matched_entities,
        })
        yield _sse({
            "step": "retrieval_done",
            "message": f"Fused into {len(context)} ranked context section(s)",
            "elapsed_ms": round((time.perf_counter() - overall_start) * 1000, 1),
            "context": context,
        })

        gen_start = time.perf_counter()
        yield _sse({
            "step": "generation_start",
            "message": (
                f"Asking {provider} with {len(history)} prior turn(s) in context..."
                if history else f"Asking {provider}..."
            ),
        })
        try:
            result = await clients.generate(clean_question, context, provider, history=history)
        except Exception as exc:
            yield _sse({"step": "error", "message": f"generation failed: {_error_detail(exc)}"})
            return
        gen_elapsed = round((time.perf_counter() - gen_start) * 1000, 1)

        conversation.append_turn(conversation_id, "user", clean_question)
        conversation.append_turn(conversation_id, "assistant", result["answer"])

        grounding = check_grounding(result["answer"], context)
        yield _sse({
            "step": "grounding_check",
            "message": (
                f"Checked {grounding['total_claims']} claim(s) against retrieved sources — "
                f"{grounding['verified_claims']} verified, {grounding['unverified_claims']} unverified"
                if grounding["total_claims"] else
                "No checkable section/rupee claims found in the answer"
            ),
        })

        yield _sse({
            "step": "done",
            "message": "Answer generated",
            "elapsed_ms": gen_elapsed,
            "total_elapsed_ms": round((time.perf_counter() - overall_start) * 1000, 1),
            "answer": result["answer"],
            "citations": [c.model_dump() for c in _citations(context)],
            "provider": result["provider"],
            "model": result["model"],
            "used_context": result["used_context"],
            "confidence": compute_confidence(context, matched_entities),
            "context": context,
            "matched_entities": matched_entities,
            "grounding": grounding,
            "conversation_id": conversation_id,
            "retrieval_query": retrieval_query,
            "history_turns": len(history),
            "guardrails": guardrail_report.model_dump(),
        })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/v1/eval", response_model=EvalResponse)
async def eval_grid(req: EvalRequest):
    """Eval tab — one hybrid retrieval (+ one graph-only retrieval for the
    graph cell), then: {ollama, gemini} x {no context, hybrid RAG context},
    ollama + graph-only RAG in isolation, and codellama/starcoder2 + hybrid
    RAG (the two extra local models from the Week 4 model-comparison
    exercise). Each cell fails independently (e.g. a missing Gemini key
    shows up only in the Gemini cells)."""
    try:
        retrieval = await clients.retrieve(req.question, settings.default_top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"retrieval failed: {_error_detail(exc)}") from exc

    context = retrieval["context"]

    async def cell(provider: str, ctx: list[dict], use_persona: bool) -> EvalCell:
        try:
            result = await clients.generate(req.question, ctx, provider, use_persona=use_persona)
        except Exception as exc:
            return EvalCell(
                answer=f"(unavailable — {_error_detail(exc)})",
                citations=[],
                provider=provider,
                model="-",
                used_context=bool(ctx),
                latency_ms=0.0,
                grounding=_NO_CLAIMS,
            )
        return EvalCell(
            answer=result["answer"],
            citations=_citations(ctx) if ctx else [],
            provider=result["provider"],
            model=result["model"],
            used_context=result["used_context"],
            latency_ms=result["latency_ms"],
            grounding=GroundingCheck(**check_grounding(result["answer"], ctx)),
        )

    # Persona/hard-rules prompt only applies alongside real retrieval — the
    # no-retrieval cells show the RAW model's own behavior instead of the
    # same persona artificially starved of context, so the contrast is
    # honest: "this app's pipeline" vs. "the model on its own."
    ollama_only = await cell("ollama", [], use_persona=False)
    ollama_rag = await cell("ollama", context, use_persona=True)
    gemini_only = await cell("gemini", [], use_persona=False)
    gemini_rag = await cell("gemini", context, use_persona=True)

    try:
        graph_retrieval = await clients.retrieve(req.question, settings.default_top_k, mode="graph_only")
        graph_context = graph_retrieval["context"]
    except Exception as exc:
        graph_context = []
        graph_retrieval_error = _error_detail(exc)
    else:
        graph_retrieval_error = None

    if graph_retrieval_error:
        ollama_graph_rag = EvalCell(
            answer=f"(unavailable — retrieval failed: {graph_retrieval_error})",
            citations=[], provider="ollama", model="-", used_context=False, latency_ms=0.0,
            grounding=_NO_CLAIMS,
        )
    else:
        ollama_graph_rag = await cell("ollama", graph_context, use_persona=True)

    codellama_rag = await cell("codellama", context, use_persona=True)
    starcoder2_rag = await cell("starcoder2", context, use_persona=True)

    return EvalResponse(
        retrieval=retrieval,
        ollama_only=ollama_only,
        ollama_rag=ollama_rag,
        gemini_only=gemini_only,
        gemini_rag=gemini_rag,
        ollama_graph_rag=ollama_graph_rag,
        codellama_rag=codellama_rag,
        starcoder2_rag=starcoder2_rag,
    )
