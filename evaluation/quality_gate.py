"""
AI quality gate — the "evaluate before you ship" stage of the AIDevOps loop.

Runs in CI on every push (see .github/workflows/ci.yml) and FAILS THE BUILD if
any AI-quality metric regresses past the committed baseline
(evaluation/quality_baseline.json). Ordinary unit tests answer "does the code
still work?"; this answers "is the AI still as good as it was?" — a change can
pass every unit test and still make retrieval worse or let a jailbreak through.

Three families of metric, all deterministic and all computable on a bare CI
runner with no GPU, no Ollama and no network:

  1. RETRIEVAL quality — Context Precision@k, Context Recall, Hit Rate and MRR,
     recomputed from the committed retrieval log (evaluation/retrieval_log.jsonl)
     against hand-verified ground truth (evaluation/ground_truth.json), using
     the exact same scoring code as analyze.py. Whenever someone re-runs the
     eval harness and commits a new log, a worse retriever cannot merge.

  2. GENERATION quality for the production model (llama3.1:8b) — hallucination
     rate from the grounding checker, deterministic test-pass rate and error
     count, recomputed from evaluation/results.jsonl.

  3. SAFETY — the input guardrails are executed LIVE against the red-team set
     (evaluation/guardrail_redteam.json): attack block rate, false-positive
     rate on genuine citizen questions, and PII redaction recall. This family
     re-runs on every commit regardless of whether the eval logs changed, so a
     regex edit that weakens a guardrail is caught immediately.

Why the LLM itself is not re-run here: a CPU-only llama3.1:8b answer takes
~175 s (evaluation/run_meta.json), so 30 questions = ~90 minutes per push.
The gate instead scores the committed outputs of the last real eval run, and
the eval run itself is a manual/offline step (run_eval.py + analyze.py).

Usage:
    python -m evaluation.quality_gate                  # check against baseline
    python -m evaluation.quality_gate --update-baseline  # accept current numbers
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.analyze import load_jsonl, score_correctness  # noqa: E402
from evaluation.retrieval_metrics import (  # noqa: E402
    context_precision_at_k,
    context_recall,
    first_relevant_rank,
    mean_reciprocal_rank,
)
from services.orchestration_service.regex_guardrails import inspect_input  # noqa: E402

EVAL_DIR = ROOT / "evaluation"
BASELINE_PATH = EVAL_DIR / "quality_baseline.json"
PRODUCTION_MODEL = "llama3.1:8b"


def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def retrieval_metrics(ground_truth: dict) -> dict:
    logs = load_jsonl(EVAL_DIR / "retrieval_log.jsonl")
    logged_ids = {r["question_id"] for r in logs}
    scorable_ids = {qid for qid, gt in ground_truth.items() if gt.get("expected_sections")}

    precisions, recalls, ranks = [], [], []
    for r in logs:
        expected = ground_truth.get(r["question_id"], {}).get("expected_sections")
        if not expected:
            continue
        ordered = list(dict.fromkeys(r["context_sections"]))  # de-dupe, keep best rank
        precisions.append(context_precision_at_k(ordered, expected))
        recalls.append(context_recall(ordered, expected))
        ranks.append(first_relevant_rank(ordered, expected))

    return {
        "context_precision": _mean(precisions),
        "context_recall": _mean(recalls),
        "hit_rate": round(sum(1 for x in recalls if x) / len(recalls), 3) if recalls else None,
        "mrr": mean_reciprocal_rank(ranks),
        # Guards against "improving" metrics by quietly dropping hard questions
        # from the eval set: every ground-truthed question must have been run.
        "eval_coverage": round(len(scorable_ids & logged_ids) / len(scorable_ids), 3) if scorable_ids else None,
    }


def generation_metrics(ground_truth: dict) -> dict:
    rows = [r for r in load_jsonl(EVAL_DIR / "results.jsonl") if r.get("model") == PRODUCTION_MODEL]
    total_claims = sum(r["grounding"]["total_claims"] for r in rows if r.get("grounding"))
    unverified = sum(r["grounding"]["unverified_claims"] for r in rows if r.get("grounding"))
    verdicts = [score_correctness(r.get("answer"), ground_truth.get(r["question_id"], {}), r.get("grounding"))
                for r in rows]
    scorable = [v for v in verdicts if v in ("pass", "fail")]
    return {
        "hallucination_rate": round(unverified / total_claims, 3) if total_claims else None,
        "test_pass_rate": round(sum(v == "pass" for v in scorable) / len(scorable), 3) if scorable else None,
        "generation_errors": sum(1 for r in rows if r.get("error")),
    }


def safety_metrics() -> tuple[dict, list[str]]:
    cases = json.loads((EVAL_DIR / "guardrail_redteam.json").read_text(encoding="utf-8"))["cases"]
    attacks = [c for c in cases if c["expect"] == "block"]
    benign = [c for c in cases if c["expect"] == "allow"]
    pii = [c for c in cases if c["expect"] == "redact"]
    failures = []

    blocked = 0
    for c in attacks:
        report = inspect_input(c["input"])
        if report.blocked and report.flags[0].check == c["check"]:
            blocked += 1
        else:
            failures.append(f"{c['id']} attack NOT blocked: {c['input']}")

    false_pos = 0
    for c in benign:
        report = inspect_input(c["input"])
        if report.blocked or report.pii_redacted:
            false_pos += 1
            failures.append(f"{c['id']} genuine question wrongly flagged: {c['input']}")

    redacted = 0
    for c in pii:
        report = inspect_input(c["input"])
        if not report.blocked and all(t in report.sanitized_question for t in c["redacted"]):
            redacted += 1
        else:
            failures.append(f"{c['id']} PII not redacted: {c['input']}")

    return {
        "attack_block_rate": round(blocked / len(attacks), 3),
        "false_positive_rate": round(false_pos / len(benign), 3),
        "pii_redaction_recall": round(redacted / len(pii), 3),
    }, failures


def compute_all() -> tuple[dict, list[str]]:
    ground_truth = json.loads((EVAL_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    ground_truth.pop("_comment", None)
    safety, safety_failures = safety_metrics()
    return {**retrieval_metrics(ground_truth), **generation_metrics(ground_truth), **safety}, safety_failures


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def evaluate(current: dict, baseline: dict) -> list[dict]:
    rows = []
    for name, rule in baseline["metrics"].items():
        value = current.get(name)
        direction, tol = rule["direction"], rule.get("tolerance", 0.0)
        if "min" in rule:  # absolute floor (safety metrics)
            threshold, ok = rule["min"], value is not None and value >= rule["min"]
            label = f">= {threshold}"
        elif "max" in rule:  # absolute ceiling
            threshold, ok = rule["max"], value is not None and value <= rule["max"]
            label = f"<= {threshold}"
        elif direction == "higher":
            threshold = round(rule["baseline"] - tol, 3)
            ok, label = value is not None and value >= threshold, f">= {threshold}"
        else:
            threshold = round(rule["baseline"] + tol, 3)
            ok, label = value is not None and value <= threshold, f"<= {threshold}"
        rows.append({"metric": name, "group": rule["group"], "value": value,
                     "baseline": rule.get("baseline"), "required": label, "ok": ok})
    return rows


def render(rows: list[dict], failures: list[str]) -> str:
    passed = all(r["ok"] for r in rows)
    lines = [
        f"## AI Quality Gate: {'PASSED' if passed else 'FAILED'}",
        "",
        "| Group | Metric | Current | Baseline | Required | Result |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['group']} | `{r['metric']}` | {r['value']} | {r['baseline'] if r['baseline'] is not None else '-'}"
                     f" | {r['required']} | {'PASS' if r['ok'] else '**FAIL**'} |")
    if failures:
        lines += ["", "### Red-team failures", *[f"- {f}" for f in failures]]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update-baseline", action="store_true",
                        help="Accept the current numbers as the new baseline (commit the result).")
    args = parser.parse_args()

    current, failures = compute_all()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    if args.update_baseline:
        for name, rule in baseline["metrics"].items():
            if "baseline" in rule:
                rule["baseline"] = current[name]
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Baseline updated -> {BASELINE_PATH.relative_to(ROOT)}")

    rows = evaluate(current, baseline)
    report = render(rows, failures)
    print(report)

    # In GitHub Actions this table is rendered on the run's summary page.
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
