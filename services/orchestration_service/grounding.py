"""
Grounding/hallucination checker — extracts the checkable factual claims from
a generated answer (section numbers cited, rupee amounts stated) and
verifies each against the ACTUAL retrieved context, not against general
knowledge of the law. This exists because of three real, confirmed
hallucinations found during testing this project: a fabricated section
quote, a fabricated L-plate/certificate requirement, and a fabricated
₹10,000 fine — in every case the model cited a real section number but
stated something the retrieved text never actually said.

This is a text-matching check, not a full NLU fact-checker: it catches
exactly those failure modes (an invented section number, an invented rupee
figure) — it cannot verify that a paraphrase is faithful to its source, only
that the specific number the model stated was actually present in what it
was given. Rupee amounts are checked only against the context item(s) whose
section the model actually cited (not the whole retrieved batch) — an
earlier version checked against everything, which produced a real false
"verified": "Rs. 1000 under Section 177" passed only because 1000 appeared
in an unrelated section (185) also in context, while Section 177 itself
actually says 500/1500. Section numbers themselves are still matched by
number alone, not scoped to a specific Act, so two different Acts sharing a
section number (e.g. both have a "Section 2") could still produce a false
"verified" — a known, acceptable trade-off for a first version, not
something to oversell as perfect.
"""

import re

# CMVR/HMVR provisions are called "Rule" in the corpus, not "Section" (see
# their "unit" metadata field) — missing this meant a claim like "Rule 115"
# was silently never checked at all, confirmed during testing.
_SECTION_MENTION = re.compile(r"(?:[Ss]ection|[Rr]ule)\s+(\d+[A-Za-z]{0,2})\b")
_RUPEE_DIGITS = re.compile(r"(?:rs\.?|₹)\s*([\d,]+)|([\d,]+)\s*rupees", re.IGNORECASE)

# The specific spelled-out amounts actually seen in this corpus's penalty
# clauses. Not a general number-word parser (too easy to get subtly wrong
# without extensive testing) — a small, honest, extensible lookup instead.
_SPELLED_OUT_AMOUNTS: dict[str, int] = {
    "one hundred rupees": 100,
    "two hundred rupees": 200,
    "five hundred rupees": 500,
    "one thousand rupees": 1000,
    "one thousand and five hundred rupees": 1500,
    "two thousand rupees": 2000,
    "three thousand rupees": 3000,
    "five thousand rupees": 5000,
    "ten thousand rupees": 10000,
    "twenty thousand rupees": 20000,
    "one lakh rupees": 100000,
}


def _numeric_amounts_in(text: str) -> set[int]:
    values: set[int] = set()
    for match in _RUPEE_DIGITS.finditer(text):
        raw = match.group(1) or match.group(2)
        if raw:
            digits = raw.replace(",", "")
            # Guards a real edge case hit during testing: a stray comma with
            # no adjacent digits (messy OCR'd punctuation in the source
            # legal text) satisfies [\d,]+ but leaves nothing to parse.
            if digits.isdigit():
                values.add(int(digits))
    # Collapse whitespace before phrase-matching — confirmed necessary: the
    # source PDF text wraps mid-phrase ("...five hundred\nrupees"), so a
    # literal newline was breaking an otherwise-correct match.
    normalized = re.sub(r"\s+", " ", text.lower())
    for phrase, value in _SPELLED_OUT_AMOUNTS.items():
        if phrase in normalized:
            values.add(value)
    return values


def _sections_in(text: str) -> set[str]:
    return {m.group(1).upper() for m in _SECTION_MENTION.finditer(text)}


def check_grounding(answer: str, context: list[dict]) -> dict:
    claimed_sections = _sections_in(answer)
    claimed_amounts = _numeric_amounts_in(answer)

    context_sections = {c["section"].upper() for c in context if c.get("section")}
    unverified_sections = sorted(claimed_sections - context_sections)

    # Scope amount-checking to only the context item(s) whose section the
    # model actually cited — checking against the WHOLE retrieved batch was
    # a real bug, confirmed during testing: "Rs. 1000 under Section 177" was
    # marked verified only because 1000 happened to appear in a different,
    # unrelated section (185) also in context — Section 177 itself actually
    # says 500/1500, not 1000. Falls back to the full context only if the
    # model cited no section at all for its amount.
    cited_text = "\n".join(
        c.get("text", "") for c in context
        if c.get("section") and c["section"].upper() in claimed_sections
    )
    scope_text = cited_text if claimed_sections else "\n".join(c.get("text", "") for c in context)
    context_amounts = _numeric_amounts_in(scope_text)
    unverified_amounts = sorted(claimed_amounts - context_amounts)

    total = len(claimed_sections) + len(claimed_amounts)
    unverified = len(unverified_sections) + len(unverified_amounts)

    return {
        "total_claims": total,
        "verified_claims": total - unverified,
        "unverified_claims": unverified,
        "unverified_sections": unverified_sections,
        "unverified_amounts": unverified_amounts,
    }
