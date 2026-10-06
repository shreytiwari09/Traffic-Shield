"""RAGAS-style retrieval metrics — these numbers are what the CI quality gate compares."""

from evaluation.retrieval_metrics import (
    context_precision_at_k,
    context_recall,
    first_relevant_rank,
    mean_reciprocal_rank,
)


def test_precision_is_rank_aware():
    assert context_precision_at_k(["158", "1", "2"], ["158"]) == 1.0
    assert context_precision_at_k(["1", "2", "3", "4", "5", "6", "7", "158"], ["158"]) == 0.125


def test_precision_edge_cases():
    assert context_precision_at_k([], ["158"]) is None
    assert context_precision_at_k(["1"], []) is None
    assert context_precision_at_k(["1", "2"], ["158"]) == 0.0


def test_recall_is_case_insensitive():
    assert context_recall(["194d", "1"], ["129", "194D"]) == 0.5
    assert context_recall(["1"], []) is None


def test_rank_and_mrr():
    assert first_relevant_rank(["1", "130", "158"], ["158", "130"]) == 2
    assert first_relevant_rank(["1"], ["158"]) is None
    assert mean_reciprocal_rank([1, 2, None, 4]) == round((1 + 0.5 + 0 + 0.25) / 4, 3)
