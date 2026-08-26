"""
Phase 4 — Legal Section Extraction.

Splits each cleaned document at legal-section boundaries — never at a character
count — so that one record is one section, rule, or regulation, and writes them
to ``data/dataset.jsonl`` in the blueprint's schema.

Section headers are matched by grammar rather than by fixed offsets, because
the five documents print them three different ways:

    act         27. Power of Central Government to make rules.—The Central …
    haryana     11.
                Mutilated license [Section 28(2)(c)].—(1) If at any time …
    regulation  6. Lane traffic.
                (1) Where any road is marked by lanes …

The ``amendment`` style reuses the ``act`` grammar: the Amendment Act's items
are numbered and dash-terminated in the same way.
"""

import bisect
import logging
import re
from pathlib import Path

from data_pipeline.config import (
    CLEANED_DIR,
    DATASET_PATH,
    DEFAULT_SOURCE_META,
    ID_PREFIXES,
    MIN_CONTENT_CHARS,
    SOURCES,
    STATE,
)
from data_pipeline.phase1_collect import normalise_stem
from data_pipeline.text_utils import (
    page_at_offset,
    read_front_matter,
    strip_page_markers,
    write_jsonl,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section-header grammars
# ---------------------------------------------------------------------------
# "27." / "2A." / "85-A." / "194C." — the number may sit on its own line. The
# trailing lookahead rejects decimals: fee tables are full of "100.00", and
# without it those read as a jump to section 100.
_NUM = r"(?P<num>\d{1,3}(?:-?[A-Z]{1,2})?)\.(?!\d)"

# A section that an amendment inserted or substituted is printed behind its
# footnote reference — "3[129. Wearing of protective headgear.—" — and the
# markers nest ("1[2[3["). Without this the amended provisions, which are
# exactly the current ones, would all be missed.
_FN = r"(?:\d{1,2}\[)*"

# None of these carry a "^" anchor: find_headers() controls the match position
# explicitly, because some headers begin mid-line, immediately after the last
# cell of the fee table belonging to the previous rule.
#
# Title runs to the em-dash that opens the body. Long titles wrap across lines
# in print, so the title may contain newlines; find_headers() then rejects a
# candidate whose "title" ran across a paragraph break, which is what stops
# this from swallowing body text.
_ACT_HEADER = re.compile(
    rf"[ \t]*{_FN}{_NUM}[ \t]*\n?[ \t]*(?P<title>[^—]{{2,180}}?)[ \t]*[.,:]?[ \t]*—"
)

# A section repealed by an amendment is printed with its old title in brackets
# and no body: "191. [Sale of vehicle …] Omitted by the Motor Vehicles
# (Amendment) Act, 2019". These are kept as records flagged ``omitted`` so the
# assistant can say a provision no longer applies instead of silently lacking it.
_OMITTED_HEADER = re.compile(
    rf"[ \t]*{_FN}{_NUM}[ \t]*\[(?P<title>[^\]\n]{{2,200}})\][ \t]*"
    r"(?=(?:Omitted|Rep\.|Repealed)\b)"
)

MAX_TITLE_LINES = 2

# The 2017 Regulations use no dash: the title is followed by "(1)". The full
# stop between them is optional — the CMVR runs them straight together, as in
# "108. Use of red, white or blue light (1) No motor vehicle shall…".
_REGULATION_HEADER = re.compile(
    rf"[ \t]*{_FN}{_NUM}[ \t]*(?P<title>[^\n—]{{2,120}}?)\.?[ \t]*\n?[ \t]*(?=\(1\))"
)

# The CMVR prints neither a dash nor a terminal stop — the title is simply its
# own short line ("3. General"), with the body starting on the next. The title
# must begin with a capital and end on a word character: rule bodies enumerate
# documents the same way ("8. School certificate,"), and the trailing comma is
# what separates those list items from a genuine heading. The monotonic-
# numbering filter in find_headers() catches what still slips through.
_HEADING_LINE_HEADER = re.compile(
    rf"[ \t]*{_NUM}[ \t]+(?P<title>[A-Z][^\n—;:]{{1,88}}?[A-Za-z0-9)\]])[ \t]*\n(?=[ \t]*\S)"
)

# Grammars strict enough to be safely tried mid-line: both demand a title
# terminated by an em-dash or a bracketed repeal note.
_MIDLINE_SAFE = (_OMITTED_HEADER, _ACT_HEADER)

_HEADER_PATTERNS: dict[str, list[re.Pattern]] = {
    "act": [_OMITTED_HEADER, _ACT_HEADER],
    "amendment": [_ACT_HEADER],
    "haryana": [_OMITTED_HEADER, _ACT_HEADER],
    "regulation": [_REGULATION_HEADER, _ACT_HEADER],
    # Order matters: a dash-terminated header is the most specific reading, and
    # the bare-heading grammar is the fallback.
    "rule_heading": [_OMITTED_HEADER, _ACT_HEADER, _REGULATION_HEADER, _HEADING_LINE_HEADER],
}

# Trailing footnote apparatus that belongs to the page, not to the section.
_FOOTNOTE_LINE = re.compile(
    r"(?m)^\s*\d{1,2}\.\s*(?:Subs\.|Ins\.|Omitted|Subs |Ins |Added|Rep\.)[^\n]*$"
)

# The CMVR footnotes every amendment at the foot of the page, numbered exactly
# like a rule ("1. Substituted by G.S.R. 338(E), dated 26-3-1993"). Left alone
# these outnumber the real rules and, being low-numbered, shatter the document
# into spurious parts — so they are rejected as headers by their wording.
_AMENDMENT_NOTE = re.compile(
    r"^(?:Subs?\b|Substituted|Ins\b|Inserted|Added|Omitted|Deleted|Renumbered"
    r"|Rep\b|Repealed|Earlier|Sub-R|Clause|Clauses|Vide|Now)\b",
    re.IGNORECASE,
)
_CITATION_MARKER = re.compile(r"\bG\.?S\.?R\.?\b|\bS\.?O\.?\s*\d|\bw\.e\.f\b", re.IGNORECASE)

# A restart of the numbering is legitimate only where the document announces a
# new division — a Chapter, Part, Schedule, Scheme, or Annexure.
_STRUCTURAL_HEADING = re.compile(
    r"^[ \t]*#{0,3}[ \t]*(?:"
    r"(?:CHAPTER|PART|SCHEDULE|SCHEME|ANNEXURE|APPENDIX|FORM)\b"
    r"|THE\b[^\n]{0,80}\b(?:SCHEME|RULES|SCHEDULE)\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _is_amendment_note(title: str) -> bool:
    """True when a candidate header is really a footnote about an amendment."""
    return bool(_AMENDMENT_NOTE.match(title) or _CITATION_MARKER.search(title))


def _sort_key(number: str) -> tuple[int, str]:
    """Order '2', '2A', '85-A', '194C' the way the statute prints them."""
    m = re.match(r"(\d+)(.*)", number)
    if not m:
        return (0, number)
    return (int(m.group(1)), m.group(2))


def _numeric(number: str) -> int:
    m = re.match(r"(\d+)", number)
    return int(m.group(1)) if m else 0


def find_headers(body: str, style: str) -> list[dict]:
    """
    Locate every section header in *body*.

    Candidates are gathered from the style's grammars, then filtered for
    plausibility: within a document the numbering runs upward, so a match that
    jumps backwards is a cross-reference quoted inside a proviso rather than a
    new section. A drop to 1 or 2 is treated as a genuine restart, which is how
    the CMVR's schedules and annexed schemes begin.
    """
    patterns = _HEADER_PATTERNS.get(style, [_ACT_HEADER])

    # Each candidate position is tried independently rather than scanning with
    # finditer(). finditer() cannot return overlapping matches, so one
    # over-reaching match — a footnote whose wrapped title runs past the next
    # header — would hide the real section behind it entirely.
    line_starts = {0} | {m.end() for m in re.finditer(r"\n", body)}
    # A header can also begin mid-line, straight after the previous rule's fee
    # table: "… 100.00 100.00 62. Fees for temporary permits.— …". Only the
    # dash-terminated grammars are tried there, and only where the preceding
    # token is itself a number — that is what distinguishes a run-on table row
    # from a cross-reference in prose ("… under section 62 …").
    midline = {m.end() for m in re.finditer(r"\d[ \t]+(?=\d)", body)} - line_starts

    positions = sorted(line_starts | midline)

    candidates: dict[int, dict] = {}
    for pos in positions:
        at_line_start = pos in line_starts
        for pattern in patterns:
            if not at_line_start and pattern not in _MIDLINE_SAFE:
                continue
            m = pattern.match(body, pos)
            if not m:
                continue
            raw_title = m.group("title")
            # A "title" that crossed a paragraph break, or wrapped more than a
            # printed line or two, is body text the dash-grammar over-reached into.
            if "\n\n" in raw_title or raw_title.count("\n") > MAX_TITLE_LINES:
                continue
            # Earlier grammars in the list win at the same offset.
            candidates.setdefault(m.start(), {
                "start": m.start(),
                "body_start": m.end(),
                "number": m.group("num"),
                "title": " ".join(raw_title.split()).strip(" .,:—"),
                "status": "omitted" if pattern is _OMITTED_HEADER else "in_force",
            })

    ordered = [
        cand for cand in sorted(candidates.values(), key=lambda c: c["start"])
        if not _is_amendment_note(cand["title"])
    ]
    return _select_consistent(ordered, body)


def _select_consistent(candidates: list[dict], body: str) -> list[dict]:
    """
    Choose the largest set of candidates whose numbering is self-consistent.

    Picking greedily — accept anything not lower than the last accepted number
    — lets a single spurious high match discard everything after it: a fee
    table reading as "section 100" once cost 38 consecutive real rules. This
    instead solves for the longest run, so an outlier is dropped in favour of
    the many headers that agree with each other.

    A drop back to 1 or 2 is allowed where the document announces a new
    division between the two headers, which is how the CMVR's annexed
    schedules and schemes restart their numbering.
    """
    n = len(candidates)
    if n == 0:
        return []

    heading_positions = [m.start() for m in _STRUCTURAL_HEADING.finditer(body)]

    def heading_between(start: int, end: int) -> bool:
        lo = bisect.bisect_left(heading_positions, start)
        hi = bisect.bisect_left(heading_positions, end)
        return hi > lo

    values = [_numeric(c["number"]) for c in candidates]

    best_len = [1] * n
    predecessor = [-1] * n
    for i in range(n):
        for j in range(i):
            if best_len[j] + 1 <= best_len[i]:
                continue
            restart = values[i] <= 2 and heading_between(
                candidates[j]["start"], candidates[i]["start"]
            )
            if values[i] >= values[j] or restart:
                best_len[i] = best_len[j] + 1
                predecessor[i] = j

    index = max(range(n), key=lambda i: best_len[i])
    chain: list[int] = []
    while index != -1:
        chain.append(index)
        index = predecessor[index]
    chain.reverse()

    accepted: list[dict] = []
    part = 1
    previous = 0
    for i in chain:
        if values[i] < previous:
            part += 1
        candidates[i]["part"] = part
        accepted.append(candidates[i])
        previous = values[i]
    return accepted


# The 2017 Regulations terminate a title with a spaced hyphen — "3. Duty
# towards other road users and the general public. - No vehicle shall be…" —
# where every other document prints an em-dash. Rewriting it on section-numbered
# lines only lets one grammar serve them all, without touching hyphens in prose.
_SPACED_HYPHEN_HEADER = re.compile(
    r"(?m)^([ \t]*(?:\d{1,2}\[)*\d{1,3}(?:-?[A-Z]{1,2})?\.(?!\d)[^\n]{2,180}?)"
    r"\.[ \t]*(?:\n[ \t]*)?-[ \t]+"
)


def _normalise_header_dashes(body: str) -> str:
    return _SPACED_HYPHEN_HEADER.sub(r"\1.— ", body)


def _tidy_content(text: str) -> str:
    text = _FOOTNOTE_LINE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document(md_path: Path) -> tuple[list[dict], dict]:
    raw = md_path.read_text(encoding="utf-8")
    front, body = read_front_matter(raw)
    body = _normalise_header_dashes(body)

    stem = normalise_stem(front.get("source_pdf", md_path.name))
    meta = SOURCES.get(stem, DEFAULT_SOURCE_META)
    prefix = ID_PREFIXES.get(stem, stem[:4].upper())
    style = meta["parser_style"]

    headers = find_headers(body, style)

    records: list[dict] = []
    skipped_short = 0
    seen_ids: dict[str, int] = {}

    for i, header in enumerate(headers):
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(body)
        content = _tidy_content(strip_page_markers(body[header["body_start"]:end]))

        # A repealed section is a one-line note by nature, so the length floor
        # applies only to provisions still in force.
        if header["status"] == "in_force" and len(content) < MIN_CONTENT_CHARS:
            # Too short to be an operative provision — almost always a stray
            # cross-reference the grammar mistook for a header.
            skipped_short += 1
            continue
        if not content:
            skipped_short += 1
            continue

        page = page_at_offset(body, header["start"])

        base_id = f"HR_{prefix}_{header['number']}"
        if header["part"] > 1:
            base_id = f"HR_{prefix}_P{header['part']}_{header['number']}"
        # Numbering can still legitimately repeat across annexed forms.
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        record_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"

        records.append({
            "id": record_id,
            "act": meta["act"],
            "section": header["number"],
            "title": header["title"],
            "content": content,
            "page": page,
            "source_pdf": front.get("source_pdf", md_path.name),
            "state": STATE,
            # "omitted" marks a provision repealed by amendment — retained so
            # the assistant can say it no longer applies rather than cite it.
            "status": header["status"],
            # Retained beyond the blueprint's required schema so retrieval can
            # rank by source authority (see the project brief).
            "unit": meta["unit"],
            "jurisdiction": meta["jurisdiction"],
            "authority": meta["authority"],
            "document_type": meta["document_type"],
            "priority": meta["priority"],
            "char_count": len(content),
        })

    stats = {
        "source_pdf": front.get("source_pdf", md_path.name),
        "act": meta["act"],
        "parser_style": style,
        "headers_found": len(headers),
        "records": len(records),
        "skipped_too_short": skipped_short,
        "section_range": (
            f"{records[0]['section']}–{records[-1]['section']}" if records else "—"
        ),
    }
    return records, stats


def run(cleaned_dir: Path = CLEANED_DIR, out_path: Path = DATASET_PATH) -> dict:
    md_files = sorted(cleaned_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown in {cleaned_dir} — run Phase 3 first.")

    all_records: list[dict] = []
    stats: list[dict] = []
    for md in md_files:
        records, stat = extract_document(md)
        logger.info("  %-46s %4d sections  (%s)  [%d short matches dropped]",
                    md.name, stat["records"], stat["section_range"],
                    stat["skipped_too_short"])
        all_records.extend(records)
        stats.append(stat)

    write_jsonl(out_path, all_records)

    return {
        "phase": 4,
        "dataset": str(out_path),
        "total_records": len(all_records),
        "documents": stats,
    }
