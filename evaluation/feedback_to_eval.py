"""
Closes the LLMOps feedback loop: citizen 👎 answers -> candidate eval questions.

Reads feedback/feedback.jsonl (written by Orchestration's POST /v1/feedback),
keeps the thumbs-down entries, de-duplicates by question, skips questions the
eval set already contains, and writes evaluation/feedback_candidates.json.

The last step is deliberately HUMAN: someone checks the law and fills in
`expected_sections` (or `expect_refusal`) for each candidate, then moves it
into questions.json + ground_truth.json. From that commit on, the CI quality
gate guarantees that failure can never silently return. An LLM is not used to
write the expected answer — ground truth that a model generated would let the
model grade its own homework.

Usage:
    python -m evaluation.feedback_to_eval
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.shared.settings import settings  # noqa: E402

EVAL_DIR = ROOT / "evaluation"
OUT_PATH = EVAL_DIR / "feedback_candidates.json"


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def build_candidates(feedback_rows: list[dict], existing_questions: list[dict]) -> list[dict]:
    known = {_norm(q["question"]) for q in existing_questions}
    candidates: dict[str, dict] = {}
    for row in feedback_rows:
        if row.get("rating") != "down":
            continue
        key = _norm(row.get("question", ""))
        if not key or key in known:
            continue
        cand = candidates.setdefault(key, {
            "question": row["question"],
            "thumbs_down": 0,
            "comments": [],
            "observed": [],
            # To be filled in by a human after checking the statute text:
            "expected_sections": None,
            "forbidden_sections": [],
            "expect_refusal": False,
        })
        cand["thumbs_down"] += 1
        if row.get("comment"):
            cand["comments"].append(row["comment"])
        cand["observed"].append({
            "request_id": row.get("request_id"),
            "model": row.get("model"),
            "prompt_version": row.get("prompt_version"),
            "cited_sections": row.get("cited_sections", []),
            "grounding_unverified": row.get("grounding_unverified"),
            "answer_excerpt": (row.get("answer") or "")[:400],
        })
    return sorted(candidates.values(), key=lambda c: -c["thumbs_down"])


def main() -> int:
    path = settings.feedback_path
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    candidates = build_candidates(rows, questions)

    OUT_PATH.write_text(json.dumps({
        "_comment": "Thumbs-down answers from real users, not yet in the eval set. For each: verify the law, "
                    "fill expected_sections (or set expect_refusal), then add to questions.json + "
                    "ground_truth.json and commit. The CI quality gate then protects it.",
        "candidates": candidates,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ups = sum(1 for r in rows if r.get("rating") == "up")
    downs = sum(1 for r in rows if r.get("rating") == "down")
    print(f"feedback: {len(rows)} total ({ups} up / {downs} down)")
    print(f"new eval candidates: {len(candidates)} -> {OUT_PATH.relative_to(ROOT)}")
    for c in candidates[:10]:
        print(f"  [{c['thumbs_down']}x down] {c['question']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
