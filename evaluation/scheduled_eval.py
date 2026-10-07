"""
Scheduled ONLINE evaluation (LLMOps) — the eval that CI cannot afford to run.

CI's quality gate re-scores the committed outputs of the last offline run;
this script produces NEW outputs. It sends the fixed question set through the
real, running production path (Orchestration /v1/ask: guardrails -> hybrid
retrieval -> LLM with the live prompt version -> grounding check), scores
every answer against ground truth, and appends one summary line to
evaluation/eval_history.jsonl.

That history is what catches regressions coming from OUTSIDE the code: a
provider silently updating the model behind its API, a corpus/index rebuild,
Neo4j falling back, or a prompt canary that is quietly worse. Application
Service exposes the latest run to Prometheus (ts_eval_score), so the Grafana
dashboard shows it next to the live traffic metrics, and an alert fires if a
scheduled run regresses.

Usage (services must be running):
    python -m evaluation.scheduled_eval                       # all questions, Gemini
    python -m evaluation.scheduled_eval --limit 5             # quick smoke eval
    python -m evaluation.scheduled_eval --judge               # + RAGAS faithfulness (LLM judge)
    python -m evaluation.scheduled_eval --fail-on-regression  # non-zero exit for schedulers

Scheduled nightly on Windows by scripts/register_nightly_eval.ps1.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.analyze import score_correctness  # noqa: E402
from evaluation.quality_gate import evaluate, render  # noqa: E402
from evaluation.retrieval_metrics import (  # noqa: E402
    context_precision_at_k,
    context_recall,
    first_relevant_rank,
    mean_reciprocal_rank,
)
from services.shared.settings import settings  # noqa: E402

EVAL_DIR = ROOT / "evaluation"
HISTORY_PATH = EVAL_DIR / "eval_history.jsonl"
RUNS_DIR = EVAL_DIR / "scheduled_runs"
# Metrics comparable between an online run and the committed baseline.
COMPARABLE = ("context_precision", "context_recall", "hit_rate", "mrr",
              "hallucination_rate", "test_pass_rate", "generation_errors")


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


async def ask(client: httpx.AsyncClient, base_url: str, question: str, provider: str) -> dict:
    started = time.perf_counter()
    try:
        r = await client.post(f"{base_url}/v1/ask", json={"question": question, "provider": provider})
        r.raise_for_status()
        body = r.json()
        body["_error"] = None
    except Exception as exc:
        body = {"_error": f"{type(exc).__name__}: {exc}"}
    body["_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return body


def summarize(rows: list[dict], ground_truth: dict) -> dict:
    precisions, recalls, ranks, verdicts = [], [], [], []
    total_claims = unverified = 0
    for row in rows:
        gt = ground_truth.get(row["question_id"], {})
        expected = gt.get("expected_sections")
        if row["error"]:
            continue
        if expected:
            ordered = list(dict.fromkeys(row["context_sections"]))
            precisions.append(context_precision_at_k(ordered, expected))
            recalls.append(context_recall(ordered, expected))
            ranks.append(first_relevant_rank(ordered, expected))
        g = row.get("grounding") or {}
        total_claims += g.get("total_claims", 0)
        unverified += g.get("unverified_claims", 0)
        verdicts.append(score_correctness(row.get("answer"), gt, row.get("grounding")))

    scorable = [v for v in verdicts if v in ("pass", "fail")]
    faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    return {
        "context_precision": _mean(precisions),
        "context_recall": _mean(recalls),
        "hit_rate": round(sum(1 for x in recalls if x) / len(recalls), 3) if recalls else None,
        "mrr": mean_reciprocal_rank(ranks),
        "hallucination_rate": round(unverified / total_claims, 3) if total_claims else None,
        "test_pass_rate": round(sum(v == "pass" for v in scorable) / len(scorable), 3) if scorable else None,
        "generation_errors": sum(1 for r in rows if r["error"]),
        "faithfulness": _mean(faith) if faith else None,
        "mean_latency_ms": _mean([r["latency_ms"] for r in rows if not r["error"]]),
        "total_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
        "total_completion_tokens": sum(r.get("completion_tokens") or 0 for r in rows),
        "total_cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 6),
        "n_questions": len(rows),
    }


async def run(args) -> int:
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    ground_truth = json.loads((EVAL_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    ground_truth.pop("_comment", None)

    judge = None
    if args.judge:
        from evaluation import judge as judge_mod
        judge = judge_mod

    started_at = datetime.now(timezone.utc)
    rows = []
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        for i, q in enumerate(questions, 1):
            body = await ask(client, args.base_url, q["question"], args.provider)
            usage = body.get("usage") or {}
            row = {
                "question_id": q["id"],
                "category": q.get("category"),
                "question": q["question"],
                "error": body["_error"],
                "latency_ms": body["_latency_ms"],
                "answer": body.get("answer"),
                "model": body.get("model"),
                "prompt_version": body.get("prompt_version"),
                "prompt_arm": body.get("prompt_arm"),
                "request_id": body.get("request_id"),
                "confidence": body.get("confidence"),
                "grounding": body.get("grounding"),
                "context_sections": [c.get("section") for c in body.get("context") or [] if c.get("section")],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": usage.get("cost_usd"),
            }
            if judge and row["answer"] and body.get("context"):
                f = await judge.faithfulness(q["question"], row["answer"], [c["text"] for c in body["context"]])
                row["faithfulness"] = f.get("score")
            rows.append(row)
            status = "ERROR " + row["error"] if row["error"] else (
                f"{row['confidence']:<6} grounding {row['grounding']['verified_claims']}/"
                f"{row['grounding']['total_claims']}  {row['latency_ms'] / 1000:.1f}s")
            print(f"[{i:02d}/{len(questions)}] {q['id']:<4} {status}", flush=True)

    metrics = summarize(rows, ground_truth)
    models = sorted({r["model"] for r in rows if r.get("model")})
    prompt_versions = sorted({r["prompt_version"] for r in rows if r.get("prompt_version")})
    summary = {
        "timestamp": started_at.isoformat(timespec="seconds"),
        "duration_s": round((datetime.now(timezone.utc) - started_at).total_seconds(), 1),
        "git_sha": _git_sha(),
        "provider": args.provider,
        "models": models,
        "prompt_versions": prompt_versions,
        "judge": bool(args.judge),
        "metrics": metrics,
    }

    RUNS_DIR.mkdir(exist_ok=True)
    detail_path = RUNS_DIR / f"{started_at.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    detail_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    baseline = json.loads((EVAL_DIR / "quality_baseline.json").read_text(encoding="utf-8"))
    baseline["metrics"] = {k: v for k, v in baseline["metrics"].items() if k in COMPARABLE}
    comparison = evaluate(metrics, baseline)
    print()
    print(render(comparison, []).replace("AI Quality Gate", "Scheduled eval vs baseline"))
    print(f"\nprovider={args.provider} models={models} prompts={prompt_versions}  "
          f"tokens={metrics['total_prompt_tokens']}+{metrics['total_completion_tokens']}  "
          f"cost=${metrics['total_cost_usd']}")
    if args.provider != "ollama":
        print("note: the committed baseline was measured on llama3.1:8b; generation metrics for another "
              "model are a comparison, not a like-for-like regression check.")
    print(f"history -> {HISTORY_PATH.relative_to(ROOT)}   detail -> {detail_path.relative_to(ROOT)}")

    regressed = not all(r["ok"] for r in comparison)
    return 1 if (regressed and args.fail_on_regression) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scheduled online evaluation against the running app.")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "ollama"])
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    parser.add_argument("--judge", action="store_true", help="add RAGAS faithfulness via the LLM judge")
    parser.add_argument("--base-url", default=settings.orchestration_service_url)
    parser.add_argument("--fail-on-regression", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
