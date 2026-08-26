"""
Shared text helpers for the data pipeline.

The Gazette PDFs in the knowledge base embed fonts whose custom encodings
PyMuPDF cannot map, so em-dashes and curly quotes extract as U+FFFD. Since the
em-dash is the delimiter between a section title and its body, repairing those
characters is a prerequisite for Phase 4, not cosmetic polish.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator

REPLACEMENT = "�"

# ---------------------------------------------------------------------------
# Encoding repair
# ---------------------------------------------------------------------------
# A U+FFFD sitting between two word characters is an apostrophe (driver�s);
# one following a word and preceding a space/paren/digit is an em-dash
# (…make rules�The Central Government). Everything else falls back to a dash.
_APOSTROPHE_CTX = re.compile(rf"(?<=\w){REPLACEMENT}(?=[a-z])")
_QUOTE_OPEN_CTX = re.compile(rf"(?<=[\s(\[]){REPLACEMENT}(?=\w)")
_QUOTE_CLOSE_CTX = re.compile(rf"(?<=\w){REPLACEMENT}(?=[\s,.;:)\]])")


def repair_encoding(text: str) -> str:
    """Map unmappable-glyph placeholders back to their intended punctuation."""
    if REPLACEMENT not in text:
        return text
    text = _APOSTROPHE_CTX.sub("’", text)
    text = _QUOTE_OPEN_CTX.sub("“", text)
    text = _QUOTE_CLOSE_CTX.sub("”", text)
    # Anything left is a dash — overwhelmingly the section-title em-dash.
    return text.replace(REPLACEMENT, "—")


def normalise_dashes(text: str) -> str:
    """Collapse the several dash spellings these documents mix into an em-dash."""
    return re.sub(r"—|–|(?<!-)--(?!-)", "—", text)


def normalise_whitespace(text: str) -> str:
    """Trim trailing spaces and collapse 3+ blank lines, keeping paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_extracted_text(text: str) -> str:
    """Full repair chain applied to raw PyMuPDF output."""
    text = unicodedata.normalize("NFKC", text)
    text = repair_encoding(text)
    text = normalise_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Token estimation
#
# tiktoken is not a dependency (and would be the wrong tokenizer for an Ollama
# model anyway), so token counts are estimated. Sub-word tokenizers split legal
# English at roughly 4 characters per token; numbering like "sub-section (2)(a)"
# pushes that down, so the word-based floor keeps the estimate honest.
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Approximate the token count of *text*."""
    if not text:
        return 0
    by_chars = len(text) / 4
    by_words = len(_WORD_RE.findall(text)) * 0.85
    return max(1, int(round(max(by_chars, by_words))))


# ---------------------------------------------------------------------------
# Page markers
#
# Phase 2 stamps every page boundary into the markdown so Phases 3 and 4 can
# still resolve a section back to its printed PDF page.
# ---------------------------------------------------------------------------
PAGE_MARKER = "<!-- page: {n} -->"
PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


def page_marker(n: int) -> str:
    return PAGE_MARKER.format(n=n)


def split_by_page(text: str) -> list[tuple[int, str]]:
    """Split marker-stamped markdown into ``(page_number, page_text)`` pairs."""
    parts = PAGE_MARKER_RE.split(text)
    pages: list[tuple[int, str]] = []
    # split() yields [preamble, num, body, num, body, …]
    for i in range(1, len(parts) - 1, 2):
        pages.append((int(parts[i]), parts[i + 1]))
    return pages


def page_at_offset(text: str, offset: int) -> int | None:
    """Return the page number in effect at character *offset* of *text*."""
    last: int | None = None
    for m in PAGE_MARKER_RE.finditer(text):
        if m.start() > offset:
            break
        last = int(m.group(1))
    return last


def strip_page_markers(text: str) -> str:
    return PAGE_MARKER_RE.sub("", text)


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------
_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def write_front_matter(meta: dict) -> str:
    lines = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def read_front_matter(text: str) -> tuple[dict, str]:
    """Split a markdown document into ``(front_matter_dict, body)``."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, text[m.end():]


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
