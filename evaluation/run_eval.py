"""
Evaluation harness — runs the fixed question set against every model under
test, through the SAME retrieval pipeline and the SAME persona prompt, so the
comparison isolates the effect of model choice.

WHAT CHANGED FROM THE FIRST VERSION, AND WHY

  * Generation is STREAMED. The previous version sent `"stream": False` and
    reported total latency only, which made Time To First Token structurally
    unmeasurable — there was no first-token event to observe. TTFT and
    tokens/sec are now real measurements taken from the token stream.

  * Hardware telemetry is CAPABILITY-DETECTED, not assumed. The old version
    shelled out to nvidia-smi unconditionally; on a host without an NVIDIA GPU
    that silently produced nothing, and the committed report still carried GPU
    figures from a different machine entirely (including, absurdly, "GPU
    memory" for Gemini — a network call). profiling.py now detects what this
    host can actually measure and records the absence explicitly.

  * Memory is sampled for PEAK during the call, on a background thread, rather
    than read once after it returns — by which time the allocation that caused
    the peak has usually been freed.

  * The host profile itself is written into the run, so latency numbers can
    never again be read without the hardware that produced them.

Run:   python -m evaluation.run_eval
Then:  python -m evaluation.analyze        (adds the RAGAS judge metrics)

Needs: Retrieval Service on :8002, Ollama running, GEMINI_API_KEY for the
Gemini row.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.profiling import ResourceSampler, host_profile, throughput  # noqa: E402
from services.orchestration_service.grounding import check_grounding  # noqa: E402
from services.shared.prompts import build_system_message, render_context_block  # noqa: E402
from services.shared.settings import settings  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"
RETRIEVAL_LOG_PATH = Path(__file__).parent / "retrieval_log.jsonl"
RUN_META_PATH = Path(__file__).parent / "run_meta.json"

MODELS = [
    {"id": "llama3.1:8b", "provider": "ollama"},
    {"id": "codellama:7b", "provider": "ollama"},
    {"id": "starcoder2:3b", "provider": "ollama"},
    {"id": "gemini-3.5-flash-lite", "provider": "gemini"},
]

_GEMINI_MIN_INTERVAL_S = 5.0  # free-tier RPM headroom
_last_gemini_call = 0.0

# Generation cap for the sweep. Not a tuning knob — a bound on pathological
# runs. starcoder2:3b is not instruction-tuned and, uncapped, rambled for 470
# SECONDS on a single question during testing (measured), which would put a
# 30-question x 4-model sweep into multi-hour territory on this CPU-only host.
#
# 512 tokens is comfortably above what a correct answer to any question in this
# set needs (the persona asks for two to five sentences), so it truncates
# runaway generation without truncating good answers. Applied identically to
# every local model, so the comparison stays fair.
MAX_EVAL_TOKENS = 512


def render_context_item(item: dict) -> str:
    """One context chunk in exactly the form the model received it.

    Reuses the app's own render_context_block on a single-item list rather than
    reimplementing the format, so the evaluation can never drift from what the
    live prompt actually contains.
    """
    return render_context_block([item])


async def retrieve(question: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{settings.retrieval_service_url}/v1/retrieve",
            json={"question": question, "top_k": settings.default_top_k},
        )
        r.raise_for_status()
        return r.json()


async def generate_ollama(model: str, question: str, system_message: str) -> dict:
    """Streamed so the first token's arrival is an observable event.

    Ollama emits newline-delimited JSON objects; the final one carries the
    real token counts (prompt_eval_count / eval_count), which is why the loop
    keeps reading after the text is complete rather than breaking early.
    """
    t0 = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    prompt_tokens = completion_tokens = None
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": question},
                    ],
                    "stream": True,
                    "keep_alive": settings.ollama_keep_alive,
                    "options": {"num_predict": MAX_EVAL_TOKENS},
                },
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode()[:200]
                    return {
                        "answer": None,
                        "error": f"HTTP {response.status_code}: {body}",
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    }
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = event.get("message", {}).get("content", "")
                    if piece and ttft is None:
                        ttft = time.perf_counter() - t0
                    if piece:
                        chunks.append(piece)
                    if event.get("done"):
                        prompt_tokens = event.get("prompt_eval_count")
                        completion_tokens = event.get("eval_count")

        total_s = time.perf_counter() - t0
        tp = throughput(completion_tokens, ttft, total_s, chunk_count=len(chunks))
        return {
            "answer": "".join(chunks),
            "latency_ms": round(total_s * 1000, 1),
            "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "stream_chunks": len(chunks),
            "tokens_per_second": tp["tokens_per_second"] if tp else None,
            "throughput_basis": tp["basis"] if tp else None,
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": None,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


async def generate_gemini(question: str, system_message: str) -> dict:
    """Streamed via generate_content_stream so TTFT is measurable for the
    hosted model too — otherwise the comparison would have TTFT for local
    models only, which is not a comparison."""
    global _last_gemini_call
    wait = _GEMINI_MIN_INTERVAL_S - (time.perf_counter() - _last_gemini_call)
    if wait > 0:
        await asyncio.sleep(wait)

    from google import genai
    from google.genai import types

    t0 = time.perf_counter()
    state: dict = {"ttft": None, "chunks": [], "usage": None}
    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        def _call() -> None:
            stream = client.models.generate_content_stream(
                model=settings.gemini_model,
                contents=question,
                config=types.GenerateContentConfig(system_instruction=system_message),
            )
            for event in stream:
                text = getattr(event, "text", None)
                if text:
                    if state["ttft"] is None:
                        state["ttft"] = time.perf_counter() - t0
                    state["chunks"].append(text)
                usage = getattr(event, "usage_metadata", None)
                if usage is not None:
                    state["usage"] = usage

        await asyncio.to_thread(_call)
        total_s = time.perf_counter() - t0
        usage = state["usage"]
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        tp = throughput(completion_tokens, state["ttft"], total_s, chunk_count=len(state["chunks"]))
        return {
            "answer": "".join(state["chunks"]),
            "latency_ms": round(total_s * 1000, 1),
            "ttft_ms": round(state["ttft"] * 1000, 1) if state["ttft"] is not None else None,
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "completion_tokens": completion_tokens,
            "stream_chunks": len(state["chunks"]),
            "tokens_per_second": tp["tokens_per_second"] if tp else None,
            "throughput_basis": tp["basis"] if tp else None,
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": None,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    finally:
        _last_gemini_call = time.perf_counter()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation sweep.")
    parser.add_argument("--models", nargs="*", help="Subset of model ids to run (default: all).")
    parser.add_argument("--limit", type=int, help="Only run the first N questions (smoke test).")
    args = parser.parse_args()

    models = [m for m in MODELS if not args.models or m["id"] in args.models]
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    profile = host_profile()
    gpu = profile["gpu"]
    print("Host profile:")
    print(f"  CPU  : {profile['cpu_physical_cores']} physical / {profile['cpu_logical_cores']} logical cores")
    print(f"  RAM  : {profile['total_ram_mb']:.0f} MB")
    print(f"  GPU  : {gpu['device'] if gpu['available'] else 'none detected'}")
    if not gpu["available"]:
        print(f"         {gpu['reason']}")
        print("         GPU metrics will be reported as unavailable, not as zero.")
    print()

    RUN_META_PATH.write_text(
        json.dumps(
            {
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "host": profile,
                "models": [m["id"] for m in models],
                "n_questions": len(questions),
                "top_k": settings.default_top_k,
                "graph_backend": settings.graph_backend,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results: list[dict] = []
    retrieval_logs: list[dict] = []

    for qi, q in enumerate(questions, 1):
        print(f"\n=== [{qi}/{len(questions)}] {q['id']} ({q['category']}): {q['question'][:66]}")
        try:
            retrieval = await retrieve(q["question"])
        except Exception as exc:
            print(f"  RETRIEVAL FAILED: {exc}")
            continue
        context = retrieval["context"]
        system_message = build_system_message(context)

        # Retrieval is shared across all models for this question (that is the
        # point — isolating model choice), so it is logged ONCE per question.
        # `context_sections` is ORDER-PRESERVING here: Context Precision is
        # rank-weighted, so sorting it (as the previous version did) would
        # destroy the very signal the metric measures.
        retrieval_logs.append({
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "context_sections": [c["section"].upper() for c in context if c.get("section")],
            "context_chunks": [c.get("text", "") for c in context],
            # The context EXACTLY as the model saw it, metadata header and all
            # — this is what the RAGAS judge must verify against.
            #
            # Judging against bare chunk text instead was a measurable error:
            # the persona requires citations shaped "(Act, Section N, Page P)",
            # and the page number lives in this header rather than in the chunk
            # body. A judge given only the body correctly reports that "Page 79"
            # is unsupported, and faithfulness collapses for every model on a
            # format the app itself mandates. Verified on a hand-checked answer
            # whose score went 0.333 -> 1.000 once the header was included.
            "context_rendered": [render_context_item(c) for c in context],
            "context_acts": [c.get("act") for c in context],
            "context_pages": [c.get("page") for c in context],
            "context_sources": [c.get("source") for c in context],
            "context_count": len(context),
            "matched_entities": retrieval["matched_entities"],
            "graph_relationship_count": len(retrieval["graph_relationships"]),
            "timing": retrieval["timing"],
        })
        RETRIEVAL_LOG_PATH.write_text(
            "\n".join(json.dumps(r) for r in retrieval_logs), encoding="utf-8"
        )

        for m in models:
            with ResourceSampler() as sampler:
                if m["provider"] == "ollama":
                    gen = await generate_ollama(m["id"], q["question"], system_message)
                else:
                    gen = await generate_gemini(q["question"], system_message)
            resources = sampler.result()

            grounding = check_grounding(gen["answer"], context) if gen.get("answer") else None

            results.append({
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "notes": q.get("notes", ""),
                "model": m["id"],
                "provider": m["provider"],
                "answer": gen.get("answer"),
                "error": gen.get("error"),
                "latency_ms": gen["latency_ms"],
                "ttft_ms": gen.get("ttft_ms"),
                "tokens_per_second": gen.get("tokens_per_second"),
                "throughput_basis": gen.get("throughput_basis"),
                "stream_chunks": gen.get("stream_chunks"),
                "prompt_tokens": gen.get("prompt_tokens"),
                "completion_tokens": gen.get("completion_tokens"),
                # Resource block is a nested dict that always carries
                # gpu_available, so a consumer cannot mistake a missing GPU
                # reading for a zero one.
                "resources": resources,
                "context_count": len(context),
                "matched_entities": retrieval["matched_entities"],
                "grounding": grounding,
            })

            status = "ERR" if gen.get("error") else "ok"
            g = f"{grounding['verified_claims']}/{grounding['total_claims']}" if grounding else "-"
            ttft = f"{gen.get('ttft_ms'):.0f}" if gen.get("ttft_ms") else "-"
            tps = f"{gen.get('tokens_per_second'):.1f}" if gen.get("tokens_per_second") else "-"
            print(
                f"    {m['id']:24s} {status:3s} {gen['latency_ms']:>7.0f}ms  "
                f"ttft={ttft:>7s}ms  tps={tps:>6s}  grounding={g}"
            )

            # Persist incrementally — a long sweep should not lose everything
            # to one crash.
            RESULTS_PATH.write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")

    print(f"\nDone. {len(results)} records -> {RESULTS_PATH}")
    print("Next: python -m evaluation.analyze")


if __name__ == "__main__":
    asyncio.run(main())
