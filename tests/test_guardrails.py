"""Input guardrails, driven by the same red-team set the CI quality gate uses
(evaluation/guardrail_redteam.json) — one source of truth for "what must be
blocked, what must be allowed, what must be redacted"."""

import json
from pathlib import Path

import pytest

from services.orchestration_service.regex_guardrails import inspect_input

REDTEAM = json.loads(
    (Path(__file__).resolve().parent.parent / "evaluation" / "guardrail_redteam.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", REDTEAM, ids=[c["id"] for c in REDTEAM])
def test_redteam_case(case):
    report = inspect_input(case["input"])

    if case["expect"] == "block":
        assert report.blocked, f"should have been blocked: {case['input']}"
        assert report.flags[0].check == case["check"]
        assert report.refusal_reason
    elif case["expect"] == "allow":
        assert not report.blocked, f"false positive: {case['input']}"
        assert not report.pii_redacted
        assert report.sanitized_question == case["input"]
    else:  # redact
        assert not report.blocked
        assert report.pii_redacted
        for token in case["redacted"]:
            assert token in report.sanitized_question


def test_redacted_question_no_longer_contains_raw_pii():
    report = inspect_input("My Aadhaar is 5432 1234 8765 and phone 9876543210")
    assert "5432 1234 8765" not in report.sanitized_question
    assert "9876543210" not in report.sanitized_question


def test_redteam_set_covers_every_group():
    groups = {c["group"] for c in REDTEAM}
    assert {"benign", "benign_tricky", "injection", "illegal", "pii"} <= groups
