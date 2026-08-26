"""
Phase 3 — Data Cleaning.

Drops only what the blueprint permits — tables of contents, index pages, blank
pages, duplicate pages, and non-legal front matter — and never touches legal
sections, notes, penalty tables, or notifications.

Every decision is page-level and recorded in the returned report, so a page that
disappears can always be traced to the rule that removed it.
"""

import hashlib
import logging
import re
from pathlib import Path

from data_pipeline.config import CLEANED_DIR, PARSED_MD_DIR
from data_pipeline.text_utils import (
    PAGE_MARKER_RE,
    page_marker,
    read_front_matter,
    split_by_page,
    write_front_matter,
)

logger = logging.getLogger(__name__)

# --- Non-legal front matter, removable by explicit title -------------------
_FRONT_MATTER_TITLES = re.compile(
    r"ARRANGEMENT OF (?:SECTIONS|RULES)|LIST OF ABBREVIATIONS|LIST OF AMENDING ACTS"
    r"|TABLE OF CONTENTS|^\s*CONTENTS\s*$|LIST OF FORMS|SUBJECT[- ]INDEX|^\s*INDEX\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# --- Signatures of real legal text — a page showing any of these is kept ---
# A section header carries a title terminated by an em-dash; a body carries
# numbered sub-sections, provisos, or penalty language.
_SECTION_HEADER = re.compile(r"(?m)^\s*\d{1,3}(?:-?[A-Z]{1,2})?\.[ \t]*\n?[^\n—]{2,180}—")
_SUBSECTION = re.compile(r"(?m)^\s*\(\d{1,2}\)\s*\S")
_LEGAL_PHRASE = re.compile(
    r"\bshall be punishable\b|\bprovided that\b|\bnotwithstanding\b|\bfine which may extend\b"
    r"|\bimprisonment\b|\bin the Official Gazette\b|\bNOTIFICATION\b",
    re.IGNORECASE,
)

# --- Table-of-contents signatures ------------------------------------------
# "102. Cancellation or modification of scheme."  (entry, no body)
_TOC_ENTRY = re.compile(r"^\s*\d{1,3}(?:-?[A-Z]{1,2})?\.\s*\S[^\n]{0,120}$")
# "82. Tourist permits  63"  (entry trailed by a page number)
_TOC_WITH_PAGE = re.compile(r"^\s*\d{1,3}(?:-?[A-Z]{1,2})?\.?\s+\D[^\n]{0,110}?\s+\d{1,3}\s*$")
# A bare page-number line left over from a two-column contents layout.
_BARE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

# A contents listing rendered as a markdown table: "| Rule | CONTENTS | Page |".
_TOC_TABLE_HEADER = re.compile(
    r"^\|[^\n|]*\|\s*(?:CONTENTS|TABLE OF CONTENTS)\s*\|", re.IGNORECASE | re.MULTILINE
)
# "| 82. | Tourist permits | 63 |" — numbered entry, short title, page number.
_TOC_TABLE_ROW = re.compile(
    r"^\|\s*(?:\d{1,3}(?:-?[A-Z]{1,2})?\.?)?\s*\|[^|\n]{0,120}\|\s*\d{0,4}\s*\|\s*$",
    re.MULTILINE,
)
# The CMVR also indexes its forms and annexures without a "CONTENTS" header:
# "| FORM 30 | Application for … | 296 |".
_TABLE_ROW = re.compile(r"^\|(?!\s*---).*\|\s*$", re.MULTILINE)
_TABLE_ROW_PAGE_REF = re.compile(r"^\|.*\|\s*\d{1,4}\s*\|\s*$", re.MULTILINE)
_INDEX_LABEL = re.compile(r"^\|\s*(?:FORM|ANNEXURE|SCHEDULE|RULE|CHAPTER|PART)\b", re.IGNORECASE | re.MULTILINE)


def _is_index_table(page_text: str) -> tuple[str, str]:
    """
    Detect a listing table — rows of labels each trailed by a page number.

    Requires the label signal as well as the page-number signal, so a fee or
    penalty table whose last column happens to be numeric is not mistaken for
    an index.
    """
    rows = _TABLE_ROW.findall(page_text)
    if len(rows) < 5:
        return "", ""
    with_page = len(_TABLE_ROW_PAGE_REF.findall(page_text))
    labelled = len(_INDEX_LABEL.findall(page_text))
    ratio = with_page / len(rows)
    if ratio >= 0.6 and labelled >= 3:
        return "strong", f"index table ({with_page}/{len(rows)} rows trail a page number)"
    return "", ""

_TOC_RATIO_THRESHOLD = 0.55
_TOC_MIN_LINES = 5


def _content_lines(page_text: str) -> list[str]:
    return [ln.strip() for ln in page_text.split("\n") if ln.strip()]


def _looks_like_toc(page_text: str) -> tuple[str, str]:
    """
    Classify a page as a table of contents / index.

    Returns ``(confidence, reason)`` where confidence is ``"strong"`` (an
    explicit contents heading or contents table), ``"weak"`` (inferred purely
    from the shape of the lines), or ``""`` (not contents). Weak verdicts are
    overridden by legal-content signals; strong ones are not.
    """
    # The CMVR's contents pages are laid out as tables, so they must be judged
    # on their rows rather than on prose lines.
    if _TOC_TABLE_HEADER.search(page_text):
        rows = len(_TOC_TABLE_ROW.findall(page_text))
        return "strong", f"contents table ({rows} index rows)"

    index_verdict = _is_index_table(page_text)
    if index_verdict[0]:
        return index_verdict

    lines = _content_lines(page_text)
    if len(lines) < _TOC_MIN_LINES:
        return "", ""

    # An explicit contents/index heading is decisive on its own.
    titled = bool(_FRONT_MATTER_TITLES.search(page_text))

    toc_like = 0
    for line in lines:
        body = line.lstrip("#").strip()
        if _BARE_NUMBER.match(body):
            toc_like += 1
        elif "—" in body:
            continue  # a real section header or body sentence
        elif _TOC_WITH_PAGE.match(body) or _TOC_ENTRY.match(body):
            toc_like += 1

    ratio = toc_like / len(lines)
    if titled and ratio >= 0.35:
        return "strong", f"contents heading + {ratio:.0%} index-style lines"
    if titled and len(lines) < 25:
        return "strong", "contents/abbreviations front matter"
    if ratio >= _TOC_RATIO_THRESHOLD:
        return "weak", f"{ratio:.0%} index-style lines"
    return "", ""


def _has_legal_content(page_text: str) -> bool:
    """
    True when the page carries operative legal text that must be kept.

    Deliberately excludes the "page contains a table" signal: a contents
    listing is also a table, so that test is applied only after the page has
    been cleared of being a contents page.
    """
    if _SECTION_HEADER.search(page_text):
        return True
    if _LEGAL_PHRASE.search(page_text):
        return True
    if len(_SUBSECTION.findall(page_text)) >= 2:
        return True
    return False


def _fingerprint(page_text: str) -> str:
    """Hash of the page's alphanumeric content, for duplicate detection."""
    normalised = re.sub(r"[^a-z0-9]+", "", page_text.lower())
    return hashlib.sha1(normalised.encode()).hexdigest()


def clean_document(md_path: Path, out_dir: Path = CLEANED_DIR) -> dict:
    text = md_path.read_text(encoding="utf-8")
    meta, body = read_front_matter(text)
    pages = split_by_page(body)

    kept: list[tuple[int, str]] = []
    removed: list[dict] = []
    seen: dict[str, int] = {}

    for page_no, page_text in pages:
        stripped = page_text.strip()

        if not stripped:
            removed.append({"page": page_no, "reason": "blank"})
            continue

        # A real section header outranks every removal rule. Failing that, a
        # strong contents verdict removes the page, while a weak one yields to
        # any other sign of operative legal text.
        if not _SECTION_HEADER.search(stripped):
            confidence, why = _looks_like_toc(stripped)
            if confidence == "strong" or (confidence == "weak" and not _has_legal_content(stripped)):
                removed.append({"page": page_no, "reason": f"table of contents / index ({why})"})
                continue

        fp = _fingerprint(stripped)
        # Only substantial pages can be duplicates; short repeated headers are not.
        if len(stripped) > 400 and fp in seen:
            removed.append({
                "page": page_no,
                "reason": f"duplicate of page {seen[fp]}",
            })
            continue
        seen.setdefault(fp, page_no)

        kept.append((page_no, stripped))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / md_path.name

    meta_out = dict(meta)
    meta_out["pages_kept"] = len(kept)
    meta_out["pages_removed"] = len(removed)

    parts = [write_front_matter(meta_out), f"# {meta.get('act', md_path.stem)}\n"]
    for page_no, page_text in kept:
        parts.append(page_marker(page_no))
        parts.append(page_text)
        parts.append("")
    out_path.write_text("\n".join(parts), encoding="utf-8")

    by_reason: dict[str, int] = {}
    for entry in removed:
        key = entry["reason"].split(" (")[0].split(" of page")[0]
        by_reason[key] = by_reason.get(key, 0) + 1

    return {
        "source_pdf": meta.get("source_pdf", md_path.stem),
        "act": meta.get("act", ""),
        "cleaned_file": out_path.name,
        "pages_in": len(pages),
        "pages_kept": len(kept),
        "pages_removed": len(removed),
        "removed_by_reason": by_reason,
        "removed_pages": removed,
    }


def run(parsed_dir: Path = PARSED_MD_DIR, out_dir: Path = CLEANED_DIR) -> dict:
    md_files = sorted(parsed_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown in {parsed_dir} — run Phase 2 first.")

    results = []
    for md in md_files:
        result = clean_document(md, out_dir)
        logger.info("  %-46s kept %3d / %3d pages  %s",
                    md.name, result["pages_kept"], result["pages_in"],
                    result["removed_by_reason"] or "")
        results.append(result)

    return {
        "phase": 3,
        "cleaned_dir": str(out_dir),
        "documents": results,
        "pages_in": sum(r["pages_in"] for r in results),
        "pages_kept": sum(r["pages_kept"] for r in results),
        "pages_removed": sum(r["pages_removed"] for r in results),
    }
