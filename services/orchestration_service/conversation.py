"""
Conversation memory for the Ask tab, plus the piece that makes RAG survive a
follow-up.

Two separate problems live here, and it is worth being clear that they are
separate:

1. THE MODEL needs the earlier turns, so "and if I refuse?" means something.
   That is just replaying prior user/assistant messages — solved by storing
   them and passing them to the provider.

2. RETRIEVAL needs the earlier turns too, and this is the part that silently
   breaks if you only solve (1). "What about at night?" embeds to nothing
   useful and alias-matches no entity, so vector search returns unrelated
   chunks and the graph matches nothing. The model then gets a correct
   conversation history alongside a context block about something else, and
   the hard rules make it refuse — or worse, it answers from the stale
   citations still visible in its own previous reply. So the retrieval query
   is resolved against the conversation before it is used.

Why the store is server-side rather than the client posting its transcript:
the live pipeline view streams over SSE, which is a GET (EventSource cannot
send a body). A full transcript in a query string hits URL length limits after
a few turns. The client sends an opaque conversation_id instead.

Why in-memory rather than Neo4j/SQLite: a conversation is per-session UX
state, not a deliverable — it is expected to be lost on restart, the same way
Chroma is expected to survive one. Bounded so a long-running process cannot
grow without limit.
"""

import re
import time
from collections import OrderedDict

# Turns kept per conversation (user+assistant each count as one). Beyond this
# the oldest are dropped: on a CPU-only host every extra token is real latency
# (~25 tok/s prompt ingestion), and a roadside legal question rarely depends
# on something said ten turns ago.
MAX_TURNS = 12

# Conversations kept in memory at once, evicted oldest-touched-first.
MAX_CONVERSATIONS = 200

# Abandoned conversations expire rather than occupying a slot forever.
TTL_SECONDS = 6 * 60 * 60

_conversations: "OrderedDict[str, dict]" = OrderedDict()


def _purge_expired() -> None:
    cutoff = time.time() - TTL_SECONDS
    for cid in [cid for cid, c in _conversations.items() if c["touched_at"] < cutoff]:
        del _conversations[cid]


def get_history(conversation_id: str | None) -> list[dict]:
    """Prior turns, oldest first. Empty for an unknown/absent id — an expired
    or unrecognised conversation degrades to a fresh one rather than failing."""
    if not conversation_id:
        return []
    _purge_expired()
    entry = _conversations.get(conversation_id)
    if entry is None:
        return []
    _conversations.move_to_end(conversation_id)
    return list(entry["messages"])


def append_turn(conversation_id: str | None, role: str, content: str) -> None:
    if not conversation_id or not content:
        return
    _purge_expired()
    entry = _conversations.get(conversation_id)
    if entry is None:
        entry = {"messages": [], "touched_at": time.time()}
        _conversations[conversation_id] = entry
    entry["messages"].append({"role": role, "content": content})
    # Trim from the front, keeping the tail — the recent turns are the ones a
    # follow-up actually refers back to.
    if len(entry["messages"]) > MAX_TURNS:
        del entry["messages"][: len(entry["messages"]) - MAX_TURNS]
    entry["touched_at"] = time.time()
    _conversations.move_to_end(conversation_id)
    while len(_conversations) > MAX_CONVERSATIONS:
        _conversations.popitem(last=False)


def reset(conversation_id: str | None) -> None:
    if conversation_id:
        _conversations.pop(conversation_id, None)


def restore(conversation_id: str, messages: list[dict]) -> int:
    """Repopulate a conversation from a transcript the client still holds.

    Needed because the two sides of a conversation have different lifetimes:
    the browser keeps its transcript in localStorage across a reload, while
    this store is in-memory and empties on a service restart (and after the
    TTL). Without this, a user could reload, see their whole chat still on
    screen, ask a follow-up — and get an answer generated with no memory of
    any of it, with nothing on screen indicating that. Rehydrating on load
    makes what the user sees and what the model sees agree.

    Replaces rather than merges: the client's transcript is authoritative
    here, and merging risks duplicating turns that survived the TTL.
    """
    clean = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ][-MAX_TURNS:]
    if not clean:
        _conversations.pop(conversation_id, None)
        return 0
    _conversations[conversation_id] = {"messages": clean, "touched_at": time.time()}
    _conversations.move_to_end(conversation_id)
    while len(_conversations) > MAX_CONVERSATIONS:
        _conversations.popitem(last=False)
    return len(clean)


def stats() -> dict:
    _purge_expired()
    return {
        "active_conversations": len(_conversations),
        "max_turns_per_conversation": MAX_TURNS,
    }


# ---------------------------------------------------------------------------
# Follow-up resolution
# ---------------------------------------------------------------------------

# Openers that are only ever continuations of something already said.
#
# Deliberately excludes "can they" / "do they" / "will they" and friends, even
# though they LOOK like continuations. In this domain they are the single most
# natural way to phrase a fully self-contained question — "Can they seize my
# licence?", "Can they fine me?" — so treating them as follow-ups fired on
# most ordinary questions and prepended the previous one to nearly every
# retrieval. A rewrite that triggers almost always is not a rewrite, it is a
# blanket dilution of the embedding.
_FOLLOW_UP_OPENERS = (
    "what about", "how about", "and if", "and what", "and can", "and is",
    "and do", "and the", "and then", "but what", "but if", "but can", "but the",
    "what if", "then what", "so can", "so what", "so is", "so do",
    "why", "why not", "how so", "and", "but", "so", "ok", "okay",
    "also", "then", "what else", "anything else", "what next", "after that",
)

# Weak back-references: suggestive, but only decisive when the message carries
# no concrete legal noun of its own.
#
# Third-person pronouns are NOT here. "they"/"he" in this app overwhelmingly
# mean "the police officer" generically rather than pointing at an earlier
# turn, so counting them as back-references misclassified self-contained
# questions constantly.
_ANAPHORA = re.compile(r"\b(it|its|there|above|earlier|instead|otherwise)\b", re.IGNORECASE)

# Demonstratives are a much stronger signal: they point at a specific thing
# already named. "What is the fine for that?" carries a real legal noun
# ("fine") and is still meaningless on its own, so these override the
# standalone-noun test below rather than deferring to it.
_DEMONSTRATIVE = re.compile(r"\b(that|this|those|these|the same)\b", re.IGNORECASE)

# A message this short is almost never self-contained.
_SHORT_MESSAGE_WORDS = 6

# Concrete legal nouns. If the user names one of these, the message can stand
# on its own for retrieval even if it also contains a pronoun — "can they seize
# my licence?" needs no rewriting, "can they do that?" does.
_STANDALONE_NOUNS = re.compile(
    r"\b(licence|license|dl|rc|registration|insurance|puc|pollution|permit|"
    r"fitness|helmet|seat ?belt|seatbelt|tint|challan|fine|penalty|arrest|"
    r"seiz\w*|impound\w*|tow\w*|breath\w*|alcohol|drunk|speed\w*|signal|"
    r"zebra|parking|number plate|numberplate|hsrp|fir|court|bail|"
    r"vehicle|car|bike|motorcycle|scooter|truck|documents?|papers?|"
    r"section|rule|act|officer|police|constable|inspector)\b",
    re.IGNORECASE,
)


def is_follow_up(message: str, history: list[dict]) -> bool:
    """Whether this message needs earlier turns to be retrievable at all.

    Deliberately a heuristic and not an LLM call: on this project's CPU-only
    Ollama path one extra generation is ~175s, which would more than double
    the wait for every follow-up. The failure modes are both cheap — a false
    positive just prepends the previous question to the embedding text (mild
    dilution, usually still correct), and a false negative degrades to exactly
    the old single-turn behavior.
    """
    if not history:
        return False

    text = message.strip().lower()
    if not text:
        return False

    # 1. An explicit continuation word. Unambiguous — "and if I refuse?"
    if text.startswith(_FOLLOW_UP_OPENERS):
        return True

    # 2. Points at something already named. Decisive even when the message
    #    does carry a legal noun: "what is the fine for that?" mentions a fine
    #    and is still unanswerable without knowing what "that" was.
    if _DEMONSTRATIVE.search(text):
        return True

    # 3. Otherwise a concrete legal noun makes the message self-contained,
    #    however short. "Can they seize my licence?" needs no earlier turn.
    if _STANDALONE_NOUNS.search(text):
        return False

    # 4. No legal noun to search on at all — too short to stand alone, or
    #    leaning on a weak back-reference. "for how long?", "why?"
    return len(text.split()) <= _SHORT_MESSAGE_WORDS or bool(_ANAPHORA.search(text))


def resolve_retrieval_query(message: str, history: list[dict]) -> str:
    """The text retrieval should actually search on for this turn.

    Returns `message` unchanged unless it is a follow-up, in which case the
    most recent USER turn is prepended. Only the last user turn, not the whole
    transcript: the retrieval query is embedded, and averaging in several
    turns' worth of unrelated text pulls the vector toward the conversation's
    centroid instead of what was just asked — which surfaced as the FIRST
    question's chunks being returned for every later turn.

    The assistant's own replies are never folded in: they contain the section
    numbers it already cited, so including them makes retrieval self-confirming
    — it would keep re-fetching whatever was cited before and the grounding
    check would rubber-stamp it.
    """
    if not is_follow_up(message, history):
        return message

    last_user = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"),
        None,
    )
    if not last_user:
        return message
    return f"{last_user.strip()} {message.strip()}"
