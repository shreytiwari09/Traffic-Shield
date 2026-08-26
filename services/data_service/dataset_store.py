"""
In-memory index over DATA/dataset.jsonl — the full legal-section records
(used for citation lookups and graph-evidence hydration). Read-only at
request time; rebuilt only by re-running data_pipeline offline.
"""

import json

from services.shared.settings import settings

_records: dict[str, dict] = {}


def load() -> None:
    global _records
    _records = {}
    if not settings.dataset_path.exists():
        return
    with open(settings.dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            _records[record["id"]] = record


def get(record_id: str) -> dict | None:
    return _records.get(record_id)


def count() -> int:
    return len(_records)
