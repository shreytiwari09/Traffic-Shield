"""
Retrieval-side RAGAS metrics: Context Precision@k and Context Recall.

METHOD AND ATTRIBUTION
Follows the RAGAS framework (Es, S., James, J., Espinosa-Anke, L., &
Schockaert, S., 2024, "RAGAS: Automated Evaluation of Retrieval Augmented
Generation", EACL 2024, arXiv:2309.15217). Context Precision uses the
rank-weighted Average Precision formulation standard in information retrieval
(Manning, Raghavan & Schütze, "Introduction to Information Retrieval", 2008,
ch. 8) rather than a flat hit count.

WHY CONTEXT PRECISION HAD TO BE ADDED
The previous harness scored retrieval with recall only:

    recall = len(retrieved_sections & expected_sections) / len(expected_sections)

That is a set intersection, so it is completely blind to ORDER. With top_k=8,
a run that puts the correct section at rank 8 scored identically to one that
put it at rank 1 — even though the generator reads the context top-down and a
buried correct chunk competes with seven irrelevant ones for attention. A
retriever can therefore look healthy on recall while feeding the model mostly
noise, which is exactly the signal-to-noise failure this metric exists to
catch.

RELEVANCE JUDGEMENT IS NON-LLM HERE, ON PURPOSE
RAGAS normally asks a judge LLM whether each retrieved chunk was useful. This
project has something better for that: `ground_truth.json` records the exact
section numbers a correct answer must cite, hand-verified against the corpus
text in earlier hardening sessions. A section-number match is an objective,
reproducible, zero-cost relevance signal, so using an LLM to re-derive it would
add expense, latency and variance for strictly worse ground truth.
"""


def _relevant_flags(retrieved_sections: list[str], expected: set[str]) -> list[bool]:
    return [s.upper() in expected for s in retrieved_sections]


def context_precision_at_k(retrieved_sections: list[str], expected_sections: list[str]) -> float | None:
    """Rank-weighted precision — Average Precision over the retrieved list.

        AP = sum_k (Precision@k * rel_k) / total_relevant_retrieved

    Precision@k is computed over the first k chunks and only counted at ranks
    where a relevant chunk actually sits, so pushing the right section further
    down the list monotonically lowers the score. Two illustrative cases with
    one relevant chunk in eight:

        rank 1 -> 1/1 / 1 = 1.000
        rank 8 -> 1/8 / 1 = 0.125

    Returns None when there is nothing to score (no ground truth, or nothing
    relevant retrieved at all) so the metric is never silently averaged as 0
    against questions it does not apply to.
    """
    if not expected_sections or not retrieved_sections:
        return None
    expected = {s.upper() for s in expected_sections}
    flags = _relevant_flags(retrieved_sections, expected)
    n_relevant = sum(flags)
    if n_relevant == 0:
        # Nothing relevant was retrieved. That is a precision of 0 for this
        # question, not "not applicable" — the retriever returned pure noise.
        return 0.0

    running_hits = 0
    precision_sum = 0.0
    for k, is_relevant in enumerate(flags, start=1):
        if is_relevant:
            running_hits += 1
            precision_sum += running_hits / k
    return round(precision_sum / n_relevant, 3)


def context_recall(retrieved_sections: list[str], expected_sections: list[str]) -> float | None:
    """Fraction of the ground-truth sections the retriever actually surfaced.

    Unchanged in definition from the original harness — recall was the one
    retrieval metric already implemented correctly — but returned alongside
    precision so the two are read together. High recall with low precision is
    the specific pathology the pair is meant to expose.
    """
    if not expected_sections:
        return None
    expected = {s.upper() for s in expected_sections}
    got = {s.upper() for s in retrieved_sections}
    return round(len(got & expected) / len(expected), 3)


def first_relevant_rank(retrieved_sections: list[str], expected_sections: list[str]) -> int | None:
    """1-indexed rank of the first correct section, or None if absent.

    Reported next to precision because it is the number a human can actually
    act on: "the right law was at position 6" says what to fix in a way that
    an averaged 0.42 does not.
    """
    if not expected_sections:
        return None
    expected = {s.upper() for s in expected_sections}
    for i, section in enumerate(retrieved_sections, start=1):
        if section.upper() in expected:
            return i
    return None


def mean_reciprocal_rank(ranks: list[int | None]) -> float | None:
    """MRR across questions — 1/rank of the first correct hit, 0 when missed.

    A single number for "how near the top is the right answer, typically",
    which is the question Context Precision answers per-question but which is
    awkward to average across a mixed question set.
    """
    if not ranks:
        return None
    reciprocals = [1.0 / r if r else 0.0 for r in ranks]
    return round(sum(reciprocals) / len(reciprocals), 3)
