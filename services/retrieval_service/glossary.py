"""
A small, hand-curated glossary of common Indian-English abbreviations for
traffic-law concepts, used to normalize a citizen's question BEFORE both
vector embedding and graph alias-matching — so "Can I get my RC?" retrieves
the same context as "Can I get my registration certificate?" on both
retrieval paths at once (an alias-table edit alone only fixes the graph
path; the corpus itself never spells out these abbreviations, so vector
search needs the expansion too).

An LLM (the local llama3.1:8b) was tried first to brainstorm candidates, but
its output was unreliable — alongside real terms it fabricated several wrong
or nonsensical ones, including offering "baksheesh" (slang for a bribe) as a
synonym for "fine," which would have been actively misleading in a legal
assistant. Only manually verified, unambiguous terms are kept here.
"""

import re

# alias (lowercase, matched as a whole word) -> canonical term used by the
# corpus / ENTITY_ALIASES.
GLOSSARY: dict[str, str] = {
    "rc": "registration certificate",
    "dl": "driving licence",
    "cop": "police officer",
    "ticket": "challan",
    # "Tint film" is the everyday term; the corpus only ever says "safety
    # glass" (CMVR Rule 100) — without this, vector search misses that rule
    # entirely for a very common real question, confirmed during testing
    # (0/15 nearest results). Kept as the exact corpus phrase, not a longer
    # paraphrase — the keyword-boost in fusion.py only fires on a literal
    # substring match, so word order/punctuation has to match the source
    # text exactly, not just read close enough for embedding purposes.
    "tint": "safety glass",
    "tinted": "safety glass",
    "sun film": "safety glass",
    "black film": "safety glass",
    # Corpus (MVA Sec 16) phrases this as "disease or disability" — kept
    # short and exact for the same reason (the fuller phrase this used to be,
    # "disease or disability unfit to drive", never matched literally: the
    # corpus has a comma there — "disability, unfit" — so the boost never
    # fired; confirmed via testing, not assumed).
    "disabled": "disease or disability",
    "disability": "disease or disability",
    "handicapped": "disease or disability",
    # MVA Sec 129 (the actual helmet-wearing requirement) literally says
    # "protective headgear," never "helmet" — confirmed necessary during
    # testing: "no helmet" fine questions scored Sec 129 (0.629) and Sec 194D
    # (0.644, the actual ₹1,000 penalty) BELOW several unrelated sections,
    # so retrieval missed both and the model fabricated a citation instead.
    "helmet": "protective headgear",
}


def matched_canonical_terms(text: str) -> list[str]:
    """The canonical legal phrase(s) recognized in the text, without
    modifying it — used as an exact-phrase scoring signal in fusion.py,
    since a dense embedding can under-rate a passage where the relevant
    phrase is diluted among unrelated content in the same chunk (e.g. MVA
    Section 158 lists "the certificate of registration" alongside five
    other document types, scoring lower than sections purely about RC
    despite being the section that actually answers the question)."""
    lower = text.lower()
    return [
        canonical
        for alias, canonical in GLOSSARY.items()
        if re.search(rf"\b{re.escape(alias)}\b", lower) and canonical not in lower
    ]


def expand_query(text: str) -> str:
    """Append the canonical term for any recognized abbreviation found in
    the text, rather than replacing it — this keeps the citizen's original
    wording (still useful to the embedding model) while adding the term the
    legal corpus and the graph alias table actually use."""
    additions = matched_canonical_terms(text)
    if not additions:
        return text
    return f"{text} ({', '.join(additions)})"
