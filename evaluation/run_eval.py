"""
Week 4, Exercises 1-3 — evaluation harness.

Runs the fixed question set (questions.json) against 4 models, using the
SAME retrieval pipeline (the already-running Retrieval Service — so the
knowledge base and retrieval logic are identical across all runs) and the
SAME persona system prompt for every model, so the comparison isolates the
effect of model choice, per the assignment's requirement.

Captures, per (question, model) pair:
  - latency_ms          — wall-clock time for the generation call only
  - prompt_tokens / completion_tokens — real token counts from each
    provider's own response (Ollama: prompt_eval_count/eval_count;
    Gemini: usage_metadata) — not estimated
  - gpu_mem_used_mb      — nvidia-smi snapshot right after the call
    (Ollama models only meaningfully affect this; Gemini is a network call)
  - cpu_percent / ram_used_mb — host-wide psutil snapshot right after the
    call (Ollama's llama.cpp runner shares the machine with everything
    else, so this is a system snapshot, not a per-process figure — noted
    as a limitation in the write-up, not hidden)
  - grounding            — reuses the SAME grounding.py used live in the
    app, so "hallucination rate" here is the identical, already-verified
    mechanism, not a separate one invented just for this exercise

Run: python -m evaluation.run_eval
Needs: Retrieval Service running (localhost:8002), Ollama running, and
GEMINI_API_KEY set in .env for the Gemini row.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.orchestration_service.grounding import check_grounding  # noqa: E402
from services.shared.prompts import build_system_message  # noqa: E402
from services.shared.settings import settings  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"
RETRIEVAL_LOG_PATH = Path(__file__).parent / "retrieval_log.jsonl"

MODELS = [
    {"id": "llama3.1:8b", "provider": "ollama"},
    {"id": "codellama:7b", "provider": "ollama"},
    {"id": "starcoder2:3b", "provider": "ollama"},
    {"id": "gemini-3.5-flash-lite", "provider": "gemini"},
]

_GEMINI_MIN_INTERVAL_S = 5.0  # free-tier RPM headroom — real rate limits hit earlier tonight
_last_gemini_call = 0.0


def gpu_mem_used_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def host_stats() -> tuple[float | None, float | None]:
    """(cpu_percent, ram_used_mb) — a whole-machine snapshot, taken right
    after each generation call. cpu_percent uses a short blocking interval
    so it reflects load during/just after the call rather than an
    instantaneous (and noisy) reading."""
    try:
        cpu = psutil.cpu_percent(interval=0.3)
        ram = psutil.virtual_memory().used / (1024 * 1024)
        return round(cpu, 1), round(ram, 1)
    except Exception:
        return None, None


async def retrieve(question: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{settings.retrieval_service_url}/v1/retrieve",
            json={"question": question, "top_k": settings.default_top_k},
        )
        r.raise_for_status()
        return r.json()


async def generate_ollama(model: str, question: str, system_message: str) -> dict:
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": question},
                    ],
                    "stream": False,
                },
            )
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return {"answer": None, "error": f"HTTP {r.status_code}: {r.text[:200]}", "latency_ms": round(elapsed, 1)}
        data = r.json()
        return {
            "answer": data.get("message", {}).get("content", ""),
            "latency_ms": round(elapsed, 1),
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "error": None,
        }
    except Exception as exc:
        return {"answer": None, "error": str(exc), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


async def generate_gemini(question: str, system_message: str) -> dict:
    global _last_gemini_call
    wait = _GEMINI_MIN_INTERVAL_S - (time.perf_counter() - _last_gemini_call)
    if wait > 0:
        await asyncio.sleep(wait)

    from google import genai
    from google.genai import types

    t0 = time.perf_counter()
    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        def _call():
            return client.models.generate_content(
                model=settings.gemini_model,
                contents=question,
                config=types.GenerateContentConfig(system_instruction=system_message),
            )

        response = await asyncio.to_thread(_call)
        elapsed = (time.perf_counter() - t0) * 1000
        usage = getattr(response, "usage_metadata", None)
        return {
            "answer": response.text or "",
            "latency_ms": round(elapsed, 1),
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "error": None,
        }
    except Exception as exc:
        return {"answer": None, "error": str(exc), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    finally:
        _last_gemini_call = time.perf_counter()


async def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results = []
    retrieval_logs = []

    for qi, q in enumerate(questions, 1):
        print(f"\n=== [{qi}/{len(questions)}] {q['id']} ({q['category']}): {q['question'][:70]}")
        try:
            retrieval = await retrieve(q["question"])
        except Exception as exc:
            print(f"  RETRIEVAL FAILED: {exc}")
            continue
        context = retrieval["context"]
        system_message = build_system_message(context)

        # Retrieval is shared across all 4 models for this question (that's
        # the point — isolating model choice) so its quality is logged ONCE
        # per question, not once per model.
        retrieval_logs.append({
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "context_sections": sorted({c["section"].upper() for c in context if c.get("section")}),
            "context_count": len(context),
            "matched_entities": retrieval["matched_entities"],
            "graph_relationship_count": len(retrieval["graph_relationships"]),
            "timing": retrieval["timing"],
        })
        RETRIEVAL_LOG_PATH.write_text(
            "\n".join(json.dumps(r) for r in retrieval_logs), encoding="utf-8"
        )

        for m in MODELS:
            if m["provider"] == "ollama":
                gen = await generate_ollama(m["id"], q["question"], system_message)
            else:
                gen = await generate_gemini(q["question"], system_message)
            gpu_after = gpu_mem_used_mb()
            cpu_pct, ram_mb = host_stats()

            grounding = check_grounding(gen["answer"], context) if gen.get("answer") else None

            record = {
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "notes": q.get("notes", ""),
                "model": m["id"],
                "provider": m["provider"],
                "answer": gen.get("answer"),
                "error": gen.get("error"),
                "latency_ms": gen["latency_ms"],
                "prompt_tokens": gen.get("prompt_tokens"),
                "completion_tokens": gen.get("completion_tokens"),
                "gpu_mem_used_mb": gpu_after,
                "cpu_percent": cpu_pct,
                "ram_used_mb": ram_mb,
                "context_count": len(context),
                "matched_entities": retrieval["matched_entities"],
                "grounding": grounding,
            }
            results.append(record)

            status = "ERR" if gen.get("error") else "ok"
            g = f"{grounding['verified_claims']}/{grounding['total_claims']}" if grounding else "-"
            print(f"    {m['id']:24s} {status:3s} {gen['latency_ms']:>8.0f} ms  grounding={g}")

            # Persist incrementally — a long run shouldn't lose everything to one crash.
            RESULTS_PATH.write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")

    print(f"\nDone. {len(results)} records saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
