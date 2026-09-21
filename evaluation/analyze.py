"""
Turns results.jsonl (+ retrieval_log.jsonl, ground_truth.json) into the metrics
report that backs the RAG Evaluation dashboard.

METRIC SET AND ATTRIBUTION
The four headline metrics follow the RAGAS framework (Es, S., James, J.,
Espinosa-Anke, L., & Schockaert, S., 2024, "RAGAS: Automated Evaluation of
Retrieval Augmented Generation", EACL 2024, arXiv:2309.15217):

  Faithfulness      — LLM-as-judge, statement extraction + NLI verification.
                      See evaluation/judge.py for the verbatim prompts.
  Answer Relevance  — LLM-as-judge reverse-question generation + cosine
                      similarity against the original question.
  Context Precision — rank-weighted Average Precision over retrieved chunks.
  Context Recall    — ground-truth sections surfaced by the retriever.

Rank weighting for Context Precision follows the standard IR Average Precision
formulation (Manning, Raghavan & Schütze, "Introduction to Information
Retrieval", 2008, ch. 8).

WHAT THE PREVIOUS VERSION GOT WRONG, AND WHY IT MATTERED

  * "Relevance" was content-word overlap between question and answer. On this
    project's own data that metric ranked codellama (0.784) above Gemini
    (0.47) while codellama hallucinated at 0.50 and Gemini at 0.075. Lexical
    overlap rewards restating the question. It is kept below as
    `relevance_lexical_overlap_legacy` so the two can be compared directly,
    but it is no longer a headline number.

  * Retrieval was scored by recall alone — a set intersection, blind to rank.
    With top_k=8 a correct section at position 8 scored the same as one at
    position 1, hiding exactly the signal-to-noise problem the retriever is
    most likely to have.

  * The refusal detector matched only the older no-source wording. The system
    prompt now also emits "nothing in the official sources I have makes ... an
    offence" for the distinct case where the user is accused of a non-offence;
    without that marker those answers were scored as failures regardless of
    correctness.

  * GPU memory was reported as a bare number even on hosts with no GPU, using
    figures that had been recorded on entirely different hardware. Reporting
    now carries availability explicitly — see evaluation/profiling.py.

Run: python -m evaluation.analyze              (judged; uses cache)
     python -m evaluation.analyze --no-judge   (deterministic metrics only)
"""

import argparse
import asyncio
import json
import re
import statistics as stats
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import judge as judge_mod  # noqa: E402
from evaluation.retrieval_metrics import (  # noqa: E402
    context_precision_at_k,
    context_recall,
    first_relevant_rank,
    mean_reciprocal_rank,
)

RESULTS_PATH = Path(__file__).parent / "results.jsonl"
RETRIEVAL_LOG_PATH = Path(__file__).parent / "retrieval_log.jsonl"
GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
RUN_META_PATH = Path(__file__).parent / "run_meta.json"
REPORT_PATH = Path(__file__).parent / "metrics_report.json"

_SECTION_MENTION = re.compile(r"(?:[Ss]ection|[Rr]ule)\s+(\d+[A-Za-z]{0,2})\b")

# Must stay in sync with the no-answer wording mandated by
# services/shared/prompts.py. The last three entries cover the "you are being
# accused of something that is not an offence" case added later — without them
# a correct, useful answer of that shape was scored as a failure.
_REFUSAL_MARKERS = (
    "don't have an official",
    "do not have an official",
    "can't answer reliably",
    "cannot answer reliably",
    "nothing in the official sources",
    "cannot find any provision",
    "can't find any provision",
    "outside",
    "don't cover",
    "do not cover",
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "for", "to", "of", "in", "on",
    "and", "or", "my", "me", "i", "can", "do", "does", "what", "if", "be", "it",
    "this", "that", "as", "at", "by", "with", "from", "not", "no", "am", "will",
}


def load_jsonl(path: Path) -> list[dict]:
    """Reads a JSONL file, tolerating a truncated final line.

    run_eval.py rewrites these files after every single generation so a long
    sweep survives a crash — which means analyze can legitimately be run
    against a file mid-write, and the last line may be half-flushed. Skipping
    one incomplete trailing record is correct; refusing to produce a report at
    all because the sweep is still going is not.
    """
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  (skipping incomplete line {i + 1} of {path.name} — sweep still writing)")
    return rows


def cited_sections(answer: str) -> set[str]:
    return {m.group(1).upper() for m in _SECTION_MENTION.finditer(answer)}


def looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def lexical_relevance(question: str, answer: str) -> float:
    """The superseded metric, retained only for comparison against the RAGAS
    judge score. See module docstring for why it is not trustworthy."""
    def words(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOPWORDS and len(w) > 2}
    qw, aw = words(question), words(answer)
    if not qw:
        return 0.0
    return round(len(qw & aw) / len(qw), 3)


def score_correctness(answer: str | None, gt: dict, grounding: dict | None) -> str:
    """'pass' | 'fail' | 'open' | 'no_answer' — this app's guardrail check.

    A citation either matches verified law or it does not, so this stays a
    deterministic rule rather than a judged one. It is reported alongside the
    RAGAS scores rather than replaced by them: they measure different things
    (is the answer supported by context vs. is it the legally correct answer).
    """
    if answer is None:
        return "no_answer"
    if gt.get("expect_refusal"):
        return "pass" if (looks_like_refusal(answer) and not cited_sections(answer)) else "fail"
    expected = gt.get("expected_sections")
    if expected is None:
        return "open"
    forbidden = set(gt.get("forbidden_sections") or [])
    cited = cited_sections(answer)
    hit_expected = bool(cited & {s.upper() for s in expected}) if expected else True
    hit_forbidden = bool(cited & {s.upper() for s in forbidden})
    grounded = grounding is not None and grounding.get("unverified_claims", 1) == 0
    return "pass" if (hit_expected and not hit_forbidden and grounded) else "fail"


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(stats.mean(clean), 3) if clean else None


def _p95(values: list[float]) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if len(clean) < 5:
        return None
    return round(clean[int(len(clean) * 0.95) - 1], 1)


async def judge_all(results: list[dict], retrieval_by_qid: dict, enabled: bool) -> dict:
    """Runs the RAGAS judge over every answered record, keyed by
    (question_id, model). Failures degrade to None rather than aborting: a
    quota limit mid-sweep should leave a partially-judged report, not no
    report, and judge.py caches everything it completes."""
    judged: dict[tuple[str, str], dict] = {}
    if not enabled:
        return judged

    answered = [r for r in results if r.get("answer")]
    print(f"Judging {len(answered)} answers with {judge_mod.settings.judge_model} "
          f"(cached results are reused)...")

    for i, r in enumerate(answered, 1):
        key = (r["question_id"], r["model"])
        rlog = retrieval_by_qid.get(r["question_id"], {})
        # Prefer the context as the model actually saw it (with its
        # "[Act — Section N, Page P]" header). Falls back to bare chunk text
        # for retrieval logs written before that field existed — those runs
        # will score lower on any answer citing a page number, which is a
        # property of the old log, not of the model.
        chunks = rlog.get("context_rendered") or rlog.get("context_chunks", [])
        try:
            faith = await judge_mod.faithfulness(r["question"], r["answer"], chunks)
            rel = await judge_mod.answer_relevance(r["question"], r["answer"])
            judged[key] = {"faithfulness": faith, "answer_relevance": rel}
            fs = faith.get("score")
            rs = rel.get("score")
            print(f"  [{i}/{len(answered)}] {r['question_id']:<5} {r['model']:<22} "
                  f"faith={fs if fs is not None else '-':<6} rel={rs if rs is not None else '-'}")
        except judge_mod.JudgeUnavailable as exc:
            print(f"  Judge stopped at {i}/{len(answered)}: {exc}")
            break
        except Exception as exc:
            print(f"  [{i}/{len(answered)}] {r['question_id']} {r['model']} judge error: {exc}")
    return judged


async def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation results.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-as-judge metrics (no API calls); deterministic metrics only.")
    args = parser.parse_args()

    results = load_jsonl(RESULTS_PATH)
    retrieval_logs = load_jsonl(RETRIEVAL_LOG_PATH)
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    ground_truth.pop("_comment", None)
    run_meta = json.loads(RUN_META_PATH.read_text(encoding="utf-8")) if RUN_META_PATH.exists() else {}

    if not results:
        print("No results yet — run evaluation/run_eval.py first.")
        return

    retrieval_by_qid = {r["question_id"]: r for r in retrieval_logs}

    # ---- Retrieval quality: precision (rank-aware) + recall ----
    retrieval_quality = []
    for r in retrieval_logs:
        gt = ground_truth.get(r["question_id"], {})
        expected = gt.get("expected_sections")
        if not expected:  # None (open) or [] (refusal expected) — not scorable
            continue
        # Order-preserving, de-duplicated: the same section can appear in
        # several retrieved chunks, and for ranking only its best position
        # matters.
        seen, ordered = set(), []
        for s in r["context_sections"]:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        precision = context_precision_at_k(ordered, expected)
        recall = context_recall(ordered, expected)
        rank = first_relevant_rank(ordered, expected)
        retrieval_quality.append({
            "question_id": r["question_id"],
            "category": r["category"],
            "expected_sections": sorted({s.upper() for s in expected}),
            "retrieved_sections": ordered,
            "context_precision": precision,
            "recall": recall,
            "first_relevant_rank": rank,
            "hit": bool(recall and recall > 0),
        })

    # ---- RAGAS judge ----
    judged = await judge_all(results, retrieval_by_qid, enabled=not args.no_judge)

    # ---- Per-model aggregation ----
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    model_reports = {}
    for model, rows in by_model.items():
        latencies = [r.get("latency_ms") for r in rows if r.get("latency_ms") is not None]
        ttfts = [r.get("ttft_ms") for r in rows if r.get("ttft_ms") is not None]
        tps = [r.get("tokens_per_second") for r in rows if r.get("tokens_per_second") is not None]

        # Resource block is nested in new runs and flat in older ones; read
        # both so a report can still be produced from pre-existing results.
        def _res(r, key, legacy=None):
            res = r.get("resources") or {}
            return res.get(key, r.get(legacy) if legacy else None)

        peak_ram_delta = [_res(r, "peak_ram_delta_mb") for r in rows]
        cpu_pct = [_res(r, "cpu_percent", "cpu_percent") for r in rows]
        gpu_vals = [_res(r, "peak_gpu_mem_mb", "gpu_mem_used_mb") for r in rows]
        gpu_available = any((r.get("resources") or {}).get("gpu_available") for r in rows)

        total_claims = sum(r["grounding"]["total_claims"] for r in rows if r.get("grounding"))
        unverified = sum(r["grounding"]["unverified_claims"] for r in rows if r.get("grounding"))

        faith_scores, rel_scores, rel_committal = [], [], []
        for r in rows:
            j = judged.get((r["question_id"], r["model"]))
            if not j:
                continue
            fs = (j.get("faithfulness") or {}).get("score")
            if fs is not None:
                faith_scores.append(fs)
            rel = j.get("answer_relevance") or {}
            if rel.get("score") is not None:
                rel_scores.append(rel["score"])
                if not rel.get("noncommittal"):
                    rel_committal.append(rel["score"])

        verdicts = [score_correctness(r.get("answer"), ground_truth.get(r["question_id"], {}), r.get("grounding")) for r in rows]
        scorable = [v for v in verdicts if v in ("pass", "fail")]

        model_reports[model] = {
            "n_questions": len(rows),
            "n_errors": sum(1 for r in rows if r.get("error")),
            # --- RAGAS (LLM-as-judge) ---
            "faithfulness": _mean(faith_scores),
            "faithfulness_n": len(faith_scores),
            "answer_relevance": _mean(rel_scores),
            "answer_relevance_n": len(rel_scores),
            # Refusals are scored 0.0 by RAGAS definition, but for this app a
            # refusal is often the CORRECT behavior — so the committal-only
            # figure is reported beside it to keep that readable.
            "answer_relevance_excluding_refusals": _mean(rel_committal),
            # --- deterministic guardrails ---
            "hallucination_rate_grounding": round(unverified / total_claims, 3) if total_claims else None,
            "total_claims_checked": total_claims,
            "test_pass_rate": round(sum(v == "pass" for v in scorable) / len(scorable), 3) if scorable else None,
            "scorable_questions": len(scorable),
            "open_questions_excluded": sum(v == "open" for v in verdicts),
            "verdict_breakdown": {v: verdicts.count(v) for v in set(verdicts)},
            "relevance_lexical_overlap_legacy": _mean(
                [lexical_relevance(r["question"], r["answer"]) for r in rows if r.get("answer")]
            ),
            # --- performance ---
            "latency_ms": {
                "mean": round(stats.mean(latencies), 1) if latencies else None,
                "median": round(stats.median(latencies), 1) if latencies else None,
                "p95": _p95(latencies),
            },
            "ttft_ms": {
                "mean": round(stats.mean(ttfts), 1) if ttfts else None,
                "median": round(stats.median(ttfts), 1) if ttfts else None,
                "measured": bool(ttfts),
            },
            "tokens_per_second_mean": _mean(tps),
            "tokens": {
                "mean_prompt": _mean([r.get("prompt_tokens") for r in rows]),
                "mean_completion": _mean([r.get("completion_tokens") for r in rows]),
            },
            # GPU is a structured absence, never a bare zero.
            "gpu": {
                "available": gpu_available,
                "peak_mem_mb_mean": _mean(gpu_vals) if gpu_available else None,
                "note": None if gpu_available else
                        "No GPU telemetry on this host — these models ran on CPU. "
                        "Not reported as 0, which would imply a measured value.",
            },
            "peak_ram_delta_mb_mean": _mean(peak_ram_delta),
            "cpu_percent_mean": _mean(cpu_pct),
        }

    # ---- Trace rows for the dashboard's inspector ----
    traces = []
    for r in results:
        j = judged.get((r["question_id"], r["model"])) or {}
        faith = j.get("faithfulness") or {}
        rel = j.get("answer_relevance") or {}
        rlog = retrieval_by_qid.get(r["question_id"], {})
        gt = ground_truth.get(r["question_id"], {})
        traces.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "category": r["category"],
            "model": r["model"],
            "provider": r["provider"],
            "answer": r.get("answer"),
            "error": r.get("error"),
            "faithfulness": faith.get("score"),
            "answer_relevance": rel.get("score"),
            "noncommittal": rel.get("noncommittal"),
            "guardrail": score_correctness(r.get("answer"), gt, r.get("grounding")),
            "grounding": r.get("grounding"),
            "latency_ms": r.get("latency_ms"),
            "ttft_ms": r.get("ttft_ms"),
            "tokens_per_second": r.get("tokens_per_second"),
            "expected_sections": gt.get("expected_sections"),
            "retrieved_context": [
                {"act": a, "section": s, "source": src, "text": t}
                for a, s, src, t in zip(
                    rlog.get("context_acts", []),
                    rlog.get("context_sections", []),
                    rlog.get("context_sources", []),
                    rlog.get("context_chunks", []),
                )
            ],
            "matched_entities": r.get("matched_entities", []),
            # The judge's own words — what makes a score auditable rather than
            # something to take on faith.
            "judge_faithfulness_evaluations": faith.get("evaluations"),
            "judge_generated_questions": rel.get("generated_questions"),
        })

    all_latencies = [r.get("latency_ms") for r in results if r.get("latency_ms") is not None]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_meta": run_meta,
        "judge": {
            "enabled": not args.no_judge,
            "provider": judge_mod.settings.judge_provider,
            "model": judge_mod.settings.judge_model,
            "n_judged": len(judged),
            "method": "RAGAS (Es et al., 2024, arXiv:2309.15217)",
        },
        "kpis": {
            "faithfulness": _mean([m["faithfulness"] for m in model_reports.values()]),
            "answer_relevance": _mean([m["answer_relevance"] for m in model_reports.values()]),
            "context_precision": _mean([rq["context_precision"] for rq in retrieval_quality]),
            "context_recall": _mean([rq["recall"] for rq in retrieval_quality]),
            "latency_p95_ms": _p95(all_latencies),
        },
        "retrieval_quality": {
            "mean_context_precision": _mean([rq["context_precision"] for rq in retrieval_quality]),
            "mean_recall": _mean([rq["recall"] for rq in retrieval_quality]),
            "hit_rate": round(sum(rq["hit"] for rq in retrieval_quality) / len(retrieval_quality), 3) if retrieval_quality else None,
            "mean_reciprocal_rank": mean_reciprocal_rank([rq["first_relevant_rank"] for rq in retrieval_quality]),
            "n_scored_questions": len(retrieval_quality),
            "per_question": retrieval_quality,
        },
        "models": model_reports,
        "traces": traces,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- Console summary ----
    rq = report["retrieval_quality"]
    print(f"\nRetrieval  precision={rq['mean_context_precision']}  recall={rq['mean_recall']}  "
          f"MRR={rq['mean_reciprocal_rank']}  (n={rq['n_scored_questions']})\n")
    print(f"{'Model':22s} {'Faith':>7s} {'Relev':>7s} {'Halluc':>7s} {'Pass':>7s} "
          f"{'Lat(ms)':>9s} {'TTFT':>8s} {'TPS':>7s} {'GPU':>10s}")
    for model, rep in model_reports.items():
        gpu = "n/a" if not rep["gpu"]["available"] else f"{rep['gpu']['peak_mem_mb_mean']:.0f}MB"
        def f(v, pct=False):
            if v is None:
                return "-"
            return f"{v * 100:.1f}%" if pct else f"{v:.3f}"
        print(
            f"{model:22s} {f(rep['faithfulness']):>7s} {f(rep['answer_relevance']):>7s} "
            f"{f(rep['hallucination_rate_grounding'], True):>7s} {f(rep['test_pass_rate'], True):>7s} "
            f"{(rep['latency_ms']['mean'] or 0):9.0f} "
            f"{(rep['ttft_ms']['mean'] or 0):8.0f} {(rep['tokens_per_second_mean'] or 0):7.1f} {gpu:>10s}"
        )

    host_gpu = (run_meta.get("host") or {}).get("gpu") or {}
    if host_gpu and not host_gpu.get("available"):
        print(f"\nGPU: {host_gpu.get('reason')}")
    print(f"\nReport -> {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
