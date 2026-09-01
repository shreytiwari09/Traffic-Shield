"""
Week 4, Exercise 3 — turns results.jsonl (+ retrieval_log.jsonl,
ground_truth.json) into the required quantitative metrics, per model and
per category. Exercise 4's trade-off analysis is written by hand from this
output, not generated here — a table alone isn't analysis.

Metric definitions (stated explicitly because "correctness" and "relevance"
have no single standard meaning for a legal Q&A app, unlike a code-gen task):

  - correctness  = for questions with a verified ground-truth section list,
    the answer cites >=1 expected section, cites 0 forbidden sections (a
    real section previously observed being cited OUT OF CONTEXT for that
    exact question), AND the grounding checker found 0 unverified claims.
    For expect_refusal questions, correctness = the answer declines to
    answer AND cites 0 sections (a "confident" refusal that still cites law
    would itself be a red flag). Questions with no verified ground truth
    (open/unverified topics) are excluded from this metric, not scored 0 —
    reported separately as "open" so the denominator stays honest.
  - test_pass_rate = same pass/fail signal as correctness, aggregated —
    this app's substitute for a code-gen "test suite": a legal citation
    either matches verified law or it doesn't.
  - hallucination_rate = unverified_claims / total_claims from grounding.py
    (the SAME mechanism already used live in the app), aggregated across
    all answers that made at least one checkable claim.
  - relevance = a coarse lexical-overlap proxy (content-word overlap
    between question and answer) — flagged clearly as weak; the write-up
    should spot-check a sample by hand rather than trust this number alone.
  - retrieval_quality (per question, shared across models) = recall of
    expected_sections within the retrieved context, i.e. did retrieval even
    hand the model the right law before generation started.

Run: python -m evaluation.analyze
"""

import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.jsonl"
RETRIEVAL_LOG_PATH = Path(__file__).parent / "retrieval_log.jsonl"
GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
REPORT_PATH = Path(__file__).parent / "metrics_report.json"

_SECTION_MENTION = re.compile(r"(?:[Ss]ection|[Rr]ule)\s+(\d+[A-Za-z]{0,2})\b")
_REFUSAL_MARKERS = (
    "don't have an official",
    "do not have an official",
    "can't answer reliably",
    "cannot answer reliably",
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
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cited_sections(answer: str) -> set[str]:
    return {m.group(1).upper() for m in _SECTION_MENTION.finditer(answer)}


def looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def lexical_relevance(question: str, answer: str) -> float:
    def words(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOPWORDS and len(w) > 2}
    qw, aw = words(question), words(answer)
    if not qw:
        return 0.0
    return round(len(qw & aw) / len(qw), 3)


def score_correctness(answer: str | None, gt: dict, grounding: dict | None) -> str:
    """Returns 'pass' | 'fail' | 'open' | 'no_answer'."""
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


def main() -> None:
    results = load_jsonl(RESULTS_PATH)
    retrieval_logs = load_jsonl(RETRIEVAL_LOG_PATH)
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    ground_truth.pop("_comment", None)

    if not results:
        print("No results yet — evaluation/results.jsonl is empty. Run evaluation/run_eval.py first.")
        return

    # ---- Retrieval quality (per question, model-independent) ----
    retrieval_quality = []
    for r in retrieval_logs:
        gt = ground_truth.get(r["question_id"], {})
        expected = gt.get("expected_sections")
        if expected is None:  # includes expect_refusal (empty list) and open topics
            continue
        got = set(r["context_sections"])
        exp = {s.upper() for s in expected}
        recall = round(len(got & exp) / len(exp), 3) if exp else 1.0
        retrieval_quality.append({
            "question_id": r["question_id"],
            "category": r["category"],
            "expected_sections": sorted(exp),
            "retrieved_sections": r["context_sections"],
            "recall": recall,
            "hit": recall > 0,
        })
    overall_recall = round(stats.mean(rq["recall"] for rq in retrieval_quality), 3) if retrieval_quality else None
    hit_rate = round(sum(rq["hit"] for rq in retrieval_quality) / len(retrieval_quality), 3) if retrieval_quality else None

    # ---- Per-model aggregation ----
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    model_reports = {}
    for model, rows in by_model.items():
        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        prompt_toks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
        completion_toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
        gpu_mem = [r["gpu_mem_used_mb"] for r in rows if r.get("gpu_mem_used_mb") is not None]
        cpu_pct = [r["cpu_percent"] for r in rows if r.get("cpu_percent") is not None]
        ram_mb = [r["ram_used_mb"] for r in rows if r.get("ram_used_mb") is not None]
        errors = [r for r in rows if r.get("error")]

        total_claims = sum(r["grounding"]["total_claims"] for r in rows if r.get("grounding"))
        unverified_claims = sum(r["grounding"]["unverified_claims"] for r in rows if r.get("grounding"))
        hallucination_rate = round(unverified_claims / total_claims, 3) if total_claims else None

        relevance_scores = [
            lexical_relevance(r["question"], r["answer"]) for r in rows if r.get("answer")
        ]

        verdicts = [score_correctness(r.get("answer"), ground_truth.get(r["question_id"], {}), r.get("grounding")) for r in rows]
        scorable = [v for v in verdicts if v in ("pass", "fail")]
        test_pass_rate = round(sum(v == "pass" for v in scorable) / len(scorable), 3) if scorable else None

        model_reports[model] = {
            "n_questions": len(rows),
            "n_errors": len(errors),
            "latency_ms": {
                "mean": round(stats.mean(latencies), 1) if latencies else None,
                "median": round(stats.median(latencies), 1) if latencies else None,
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1) if len(latencies) >= 5 else None,
            },
            "tokens": {
                "mean_prompt": round(stats.mean(prompt_toks), 1) if prompt_toks else None,
                "mean_completion": round(stats.mean(completion_toks), 1) if completion_toks else None,
            },
            "gpu_mem_used_mb_mean": round(stats.mean(gpu_mem), 1) if gpu_mem else None,
            "cpu_percent_mean": round(stats.mean(cpu_pct), 1) if cpu_pct else None,
            "ram_used_mb_mean": round(stats.mean(ram_mb), 1) if ram_mb else None,
            "hallucination_rate": hallucination_rate,
            "total_claims_checked": total_claims,
            "relevance_mean_lexical_overlap": round(stats.mean(relevance_scores), 3) if relevance_scores else None,
            "test_pass_rate": test_pass_rate,
            "scorable_questions": len(scorable),
            "open_questions_excluded": sum(v == "open" for v in verdicts),
            "verdict_breakdown": {v: verdicts.count(v) for v in set(verdicts)},
        }

    report = {
        "retrieval_quality": {
            "mean_recall": overall_recall,
            "hit_rate": hit_rate,
            "n_scored_questions": len(retrieval_quality),
            "per_question": retrieval_quality,
        },
        "models": model_reports,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nRetrieval quality: mean recall={overall_recall}, hit rate={hit_rate} (n={len(retrieval_quality)} scorable questions)\n")
    print(f"{'Model':22s} {'Lat(ms)':>9s} {'PromptTok':>9s} {'CompTok':>8s} {'GPU MB':>8s} {'CPU%':>6s} {'Halluc%':>8s} {'PassRate':>9s} {'Errs':>5s}")
    for model, rep in model_reports.items():
        print(
            f"{model:22s} {rep['latency_ms']['mean'] or 0:9.0f} "
            f"{rep['tokens']['mean_prompt'] or 0:9.0f} {rep['tokens']['mean_completion'] or 0:8.0f} "
            f"{rep['gpu_mem_used_mb_mean'] or 0:8.0f} {rep['cpu_percent_mean'] or 0:6.1f} "
            f"{(rep['hallucination_rate'] or 0) * 100:7.1f}% {(rep['test_pass_rate'] or 0) * 100:8.1f}% {rep['n_errors']:5d}"
        )

    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
