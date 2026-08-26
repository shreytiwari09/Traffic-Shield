"""
Phase 5 — Metadata Validation.

Every record must carry id, act, section, title, content, page, source_pdf and
state. Incomplete records are rejected: they are written to
``data/rejected_records.jsonl`` with the reason, and ``dataset.jsonl`` is
rewritten containing only records that passed.

Rejecting silently would hide extraction bugs, so every rejection is both
counted by reason in the report and preserved in full for inspection.
"""

import logging
from pathlib import Path

from data_pipeline.config import (
    DATASET_PATH,
    REJECTED_PATH,
    REQUIRED_FIELDS,
    STATE,
    VALIDATION_REPORT_PATH,
)
from data_pipeline.text_utils import read_jsonl, write_json, write_jsonl

logger = logging.getLogger(__name__)


def validate_record(record: dict, seen_ids: set[str]) -> list[str]:
    """Return the list of reasons *record* is invalid (empty when it passes)."""
    problems: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            problems.append(f"missing field: {field}")
            continue
        value = record[field]
        if value is None:
            problems.append(f"null field: {field}")
        elif isinstance(value, str) and not value.strip():
            problems.append(f"empty field: {field}")

    if record.get("id") in seen_ids:
        problems.append(f"duplicate id: {record.get('id')}")

    page = record.get("page")
    if page is not None and (not isinstance(page, int) or page < 1):
        problems.append(f"invalid page: {page!r}")

    if record.get("state") and record["state"] != STATE:
        problems.append(f"unexpected state: {record['state']!r}")

    return problems


def run(
    dataset_path: Path = DATASET_PATH,
    rejected_path: Path = REJECTED_PATH,
    report_path: Path = VALIDATION_REPORT_PATH,
) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found — run Phase 4 first.")

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_ids: set[str] = set()
    reason_counts: dict[str, int] = {}
    by_act: dict[str, dict[str, int]] = {}

    for record in read_jsonl(dataset_path):
        problems = validate_record(record, seen_ids)
        act = record.get("act", "unknown")
        bucket = by_act.setdefault(act, {"accepted": 0, "rejected": 0})

        if problems:
            rejected.append({"record": record, "problems": problems})
            bucket["rejected"] += 1
            for problem in problems:
                key = problem.split(":")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        else:
            seen_ids.add(record["id"])
            accepted.append(record)
            bucket["accepted"] += 1

    # dataset.jsonl is the deliverable, so it must hold only valid records.
    write_jsonl(dataset_path, accepted)
    write_jsonl(rejected_path, rejected)

    pages_missing = sum(1 for r in accepted if r.get("page") is None)

    report = {
        "phase": 5,
        "required_fields": REQUIRED_FIELDS,
        "records_in": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": reason_counts,
        "accepted_without_page_reference": pages_missing,
        "by_act": by_act,
    }
    write_json(report_path, report)

    logger.info("  %d accepted, %d rejected", len(accepted), len(rejected))
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        logger.info("    rejected — %s: %d", reason, count)
    if pages_missing:
        logger.warning("    %d accepted records carry no page reference", pages_missing)

    return report
