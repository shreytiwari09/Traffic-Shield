"""The two deterministic signals shown to citizens next to every answer:
the confidence badge and the hallucination (grounding) check."""

from services.orchestration_service.grounding import check_grounding
from services.shared.confidence import compute_confidence


def test_confidence_levels():
    strong = [{"source": "vector", "score": 0.7}]
    weak = [{"source": "vector", "score": 0.3}]
    assert compute_confidence([], []) == "none"
    assert compute_confidence(strong, ["Helmet"]) == "high"
    assert compute_confidence(strong, []) == "medium"
    assert compute_confidence(weak, ["Helmet"]) == "medium"
    assert compute_confidence(weak, []) == "low"


def test_confidence_ignores_graph_scores():
    assert compute_confidence([{"source": "graph", "score": 0.99}], []) == "low"


CONTEXT = [
    {"section": "177", "text": "fine which may extend to five hundred rupees, and for second offence 1500 rupees"},
    {"section": "185", "text": "fine which may extend to ten thousand rupees or Rs. 1000"},
]


def test_grounded_answer_is_fully_verified():
    g = check_grounding("Under Section 177 the fine is Rs. 500.", CONTEXT)
    assert g["total_claims"] == 2
    assert g["unverified_claims"] == 0


def test_hallucinated_section_is_flagged():
    g = check_grounding("Section 999 says you must pay.", CONTEXT)
    assert g["unverified_sections"] == ["999"]


def test_amount_is_checked_only_against_the_cited_section():
    # 1000 appears in context — but in Sec 185, not the cited Sec 177.
    g = check_grounding("Under Section 177 the fine is Rs. 1000.", CONTEXT)
    assert g["unverified_amounts"] == [1000]


def test_answer_without_claims():
    assert check_grounding("Please consult a lawyer.", CONTEXT)["total_claims"] == 0
