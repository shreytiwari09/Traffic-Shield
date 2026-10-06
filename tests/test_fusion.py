"""Hybrid retrieval fusion — the ranking step that decides which law the LLM sees."""

import pytest

from services.retrieval_service import fusion


def _vec(record_id, section, distance, text="text", priority=7):
    return {
        "record_id": record_id,
        "text": text,
        "distance": distance,
        "metadata": {"title": None, "act": "MVA", "section": section, "page": 1,
                     "source_pdf": "mva.pdf", "priority": priority},
    }


def _graph(record_id, section, score, text="text"):
    return {"record": {"id": record_id, "content": text, "section": section, "act": "MVA",
                       "page": 2, "source_pdf": "mva.pdf"}, "score": score}


def test_cosine_similarity_basic_cases():
    assert fusion.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert fusion.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert fusion.cosine_similarity([0, 0], [1, 1]) == 0.0  # zero vector guarded


def test_vector_and_graph_compete_on_one_scale():
    ranked = fusion.fuse([_vec("A", "1", distance=0.4)], [_graph("B", "2", score=0.9)], top_k=5)
    assert [r["section"] for r in ranked] == ["2", "1"]
    assert ranked[0]["source"] == "graph"
    assert ranked[1]["score"] == pytest.approx(0.6)


def test_duplicate_record_keeps_higher_score_and_is_not_duplicated():
    ranked = fusion.fuse([_vec("A", "1", distance=0.5)], [_graph("A", "1", score=0.8)], top_k=5)
    assert len(ranked) == 1
    assert ranked[0]["score"] == pytest.approx(0.8)


def test_missing_graph_score_ranks_last_but_is_kept():
    ranked = fusion.fuse([_vec("A", "1", distance=0.5)], [_graph("B", "2", score=None)], top_k=5)
    assert [r["section"] for r in ranked] == ["1", "2"]


def test_keyword_boost_matches_title_or_text():
    hits = [_vec("A", "1", distance=0.30, text="unrelated"),
            _vec("B", "2", distance=0.33, text="the registration certificate must be produced")]
    ranked = fusion.fuse(hits, [], top_k=5, boost_terms=["Registration Certificate"])
    assert ranked[0]["section"] == "2"
    assert ranked[0]["score"] == pytest.approx(0.67 + fusion.KEYWORD_BOOST)


def test_core_section_boost_is_a_nudge_and_top_k_cuts():
    hits = [_vec(str(i), str(i), distance=0.30 + i * 0.01) for i in range(10)]
    ranked = fusion.fuse(hits, [], top_k=3, core_ids={"9"})
    assert len(ranked) == 3
    assert "_priority" not in ranked[0]
    # 0.61 + 0.03 = 0.64 is still below 0.70/0.69/0.68 — a nudge, not an override
    assert "9" not in [r["section"] for r in ranked]


def test_score_never_exceeds_one():
    ranked = fusion.fuse([_vec("A", "1", distance=0.0, text="helmet")], [], top_k=1,
                         boost_terms=["helmet"], core_ids={"A"})
    assert ranked[0]["score"] == 1.0
