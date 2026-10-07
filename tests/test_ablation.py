"""Scoring rules of the ablation study (evaluation/ablation.py). The study is
only fair if an arm that was shown NO sources is still scored on whether its
claims are legally true — not failed by construction."""

from evaluation import ablation
from services.orchestration_service.grounding import check_grounding

CORPUS = {
    "194B": [{"section": "194B", "text": "Whoever drives without a safety belt shall be punishable with a fine of "
                                          "one thousand rupees."}],
    "177": [{"section": "177", "text": "fine which may extend to five hundred rupees"}],
}


def test_statute_check_verifies_a_correct_claim_the_model_was_not_shown():
    answer = "Under Section 194B the fine is Rs. 1000."
    # Shown-source check: no context was given, so both claims are unverified.
    assert check_grounding(answer, [])["unverified_claims"] == 2
    # Statute check: both claims are true in the real text of Section 194B.
    statute = ablation.statute_grounding(answer, CORPUS)
    assert statute["total_claims"] == 2 and statute["unverified_claims"] == 0


def test_statute_check_still_catches_a_wrong_amount_and_an_unknown_section():
    statute = ablation.statute_grounding("Under Section 194B the fine is Rs. 5000, see also Section 999.", CORPUS)
    assert statute["unverified_amounts"] == [5000]
    assert statute["unverified_sections"] == ["999"]


def test_retrieval_scores_skip_questions_without_known_sections():
    rows = [{"question_id": "Q1", "hybrid": ["48", "130", "158"]},
            {"question_id": "Q2", "hybrid": ["7"]}]
    gt = {"Q1": {"expected_sections": ["130", "158"]}, "Q2": {"expected_sections": []}}
    scores = ablation.retrieval_scores(rows, "hybrid", gt)
    assert scores["n"] == 1 and scores["context_recall"] == 1.0 and scores["mrr"] == 0.5


def test_rule_6_out_of_scope_reply_counts_as_a_correct_refusal():
    from evaluation.analyze import score_correctness

    gt = {"expect_refusal": True, "expected_sections": []}
    assert score_correctness("I cover Indian and Haryana motor vehicle and traffic law only.", gt, None) == "pass"
    assert score_correctness("There is no general speed limit on the Autobahn.", gt, None) == "fail"
