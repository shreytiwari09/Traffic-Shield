"""
Citizen feedback store (LLMOps feedback loop).

Every 👍/👎 is appended as one JSON line to settings.feedback_path. The loop it
closes: evaluation/feedback_to_eval.py turns 👎 answers into candidate eval
questions, a human writes the correct expected_sections for them, and once
they are in ground_truth.json the CI quality gate guarantees that failure can
never silently come back. The eval set then grows from real failures rather
than from guesses about what users will ask.

Privacy: the stored question is the PII-REDACTED one the pipeline actually
processed (AskResponse.question), never the raw input.
"""

import json
import threading
from collections import Counter
from datetime import datetime, timezone

from services.shared.settings import settings

_lock = threading.Lock()


def append(record: dict) -> dict:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    path = settings.feedback_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load() -> list[dict]:
    path = settings.feedback_path
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written last line must not break the summary
    return rows


def summary() -> dict:
    rows = load()
    by_version: dict[str, Counter] = {}
    for r in rows:
        by_version.setdefault(r.get("prompt_version") or "unknown", Counter())[r["rating"]] += 1
    total = Counter(r["rating"] for r in rows)
    n = total["up"] + total["down"]
    return {
        "total": n,
        "up": total["up"],
        "down": total["down"],
        "satisfaction": round(total["up"] / n, 3) if n else None,
        "by_prompt_version": {v: dict(c) for v, c in by_version.items()},
    }
