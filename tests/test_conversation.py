"""Follow-up detection decides what retrieval searches on in multi-turn chats."""

from services.orchestration_service.conversation import is_follow_up, resolve_retrieval_query

HISTORY = [
    {"role": "user", "content": "What is the fine for not wearing a helmet?"},
    {"role": "assistant", "content": "Under Section 194D the fine is Rs. 1000."},
]


def test_first_turn_is_never_a_follow_up():
    assert not is_follow_up("what about at night?", [])


def test_continuations_are_resolved_against_last_user_turn():
    assert is_follow_up("what about at night?", HISTORY)
    assert resolve_retrieval_query("what is the fine for that?", HISTORY) == (
        "What is the fine for not wearing a helmet? what is the fine for that?"
    )


def test_self_contained_question_is_left_alone():
    q = "Can they seize my driving licence?"
    assert not is_follow_up(q, HISTORY)
    assert resolve_retrieval_query(q, HISTORY) == q


def test_assistant_text_is_never_folded_into_retrieval():
    assert "194D" not in resolve_retrieval_query("why?", HISTORY)
