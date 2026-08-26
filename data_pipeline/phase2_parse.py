"""
Phase 2 — PDF Parsing.

Converts each raw PDF into markdown while preserving the structure the
blueprint calls out: headings, section numbers, tables, lists, and the original
page reference.

Page references survive as ``<!-- page: N -->`` markers, which Phase 3 keeps and
Phase 4 uses to stamp each extracted section with its printed page number.
"""

import logging
import re
from pathlib import Path

import pymupdf

from data_pipeline.config import DEFAULT_SOURCE_META, PARSED_MD_DIR, RAW_PDF_DIR, SOURCES
from data_pipeline.phase1_collect import normalise_stem
from data_pipeline.text_utils import (
    clean_extracted_text,
    normalise_dashes,
    page_marker,
    write_front_matter,
)

logger = logging.getLogger(__name__)

# Structural headings in these documents are printed in full caps.
_CHAPTER_RE = re.compile(r"^\s*(CHAPTER\s+[IVXLC\d]+[A-Z]?)\s*$", re.IGNORECASE)
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,.\-—’'()/&]{4,80}$")
_LIST_ITEM_RE = re.compile(r"^\s*\(([a-z]{1,3}|[ivxlc]{1,6}|\d{1,3})\)\s")
_SECTION_START_RE = re.compile(r"^\s*\d{1,3}(?:-?[A-Z]{1,2})?\.\s")
# The header line PyMuPDF picks up from the printed page number.
_RUNNING_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
# The CMVR PDF was generated from a web page and carries navigation furniture.
_BOILERPLATE_RE = re.compile(r"^\s*(Back to Top|Top of Form|Bottom of Form)\s*$", re.IGNORECASE)


def _is_plausible_table(rows: list[list[str]]) -> bool:
    """
    Reject "tables" the detector hallucinated around ordinary prose.

    A genuine table in these documents has several short cells across two or
    more columns; a false positive is one long paragraph boxed into one cell.
    """
    if len(rows) < 2:
        return False
    width = max(len(row) for row in rows)
    if width < 2:
        return False

    cells = [cell for row in rows for cell in row]
    if not cells:
        return False
    if max(len(cell) for cell in cells) > 300:
        return False  # a paragraph, not a cell

    filled = [cell for cell in cells if cell]
    if len(filled) / len(cells) < 0.4:
        return False  # mostly empty grid
    if sum(len(cell) for cell in filled) / len(filled) > 120:
        return False  # cells read as prose

    # A column that is almost entirely blank means the detector drew a grid
    # around a paragraph rather than finding real columns.
    for col in range(width):
        column = [row[col] for row in rows if col < len(row)]
        if column and sum(1 for cell in column if cell) / len(column) < 0.3:
            return False
    return True


def get_source_meta(pdf_path: Path) -> dict:
    """Registry metadata for a PDF, or a flagged generic fallback."""
    return SOURCES.get(normalise_stem(pdf_path.name), DEFAULT_SOURCE_META)


def _extract_tables(page: pymupdf.Page) -> tuple[list[str], list[tuple[float, float, float, float]]]:
    """
    Render detected tables as markdown.

    Returns the markdown blocks plus their bounding boxes, so the caller can
    drop the text that PyMuPDF would otherwise emit a second time as loose
    lines inside those regions.
    """
    blocks: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    try:
        found = page.find_tables()
    except Exception as exc:  # table finder is best-effort on scanned layouts
        logger.debug("    table detection failed on page %d: %s", page.number + 1, exc)
        return blocks, boxes

    for table in found.tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
        rows = [row for row in rows if any(row)]
        if not _is_plausible_table(rows):
            # Leave the region to the normal text path rather than emitting a
            # fake table — and, crucially, do not record its bbox as consumed.
            continue

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header, *body = rows
        header = [h or f"col{i + 1}" for i, h in enumerate(header)]

        md = ["| " + " | ".join(header) + " |",
              "| " + " | ".join("---" for _ in header) + " |"]
        md += ["| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |" for row in body]
        blocks.append("\n".join(md))
        boxes.append(tuple(table.bbox))
    return blocks, boxes


def _in_any_box(bbox, boxes, pad: float = 2.0) -> bool:
    x0, y0, x1, y1 = bbox
    for bx0, by0, bx1, by1 in boxes:
        if x0 >= bx0 - pad and y0 >= by0 - pad and x1 <= bx1 + pad and y1 <= by1 + pad:
            return True
    return False


def _format_line(line: str) -> str:
    """Apply markdown structure to a single extracted line."""
    stripped = line.strip()
    if not stripped:
        return ""
    if _CHAPTER_RE.match(stripped):
        return f"## {stripped}"
    # A section start keeps its own paragraph so Phase 4 can anchor on it.
    if _SECTION_START_RE.match(stripped):
        return stripped
    if _ALLCAPS_HEADING_RE.match(stripped) and len(stripped.split()) <= 12:
        return f"### {stripped}"
    if _LIST_ITEM_RE.match(stripped):
        return stripped
    return stripped


def _page_to_markdown(page: pymupdf.Page) -> str:
    table_md, table_boxes = _extract_tables(page)

    # Pull text block-by-block so table regions can be excluded by geometry.
    raw_blocks = page.get_text("blocks")
    text_parts: list[str] = []
    for block in sorted(raw_blocks, key=lambda b: (round(b[1], 1), round(b[0], 1))):
        x0, y0, x1, y1, content, _no, btype = block[:7]
        if btype != 0 or not content.strip():
            continue
        if _in_any_box((x0, y0, x1, y1), table_boxes):
            continue
        text_parts.append(content)

    body = clean_extracted_text("\n".join(text_parts))
    body = normalise_dashes(body)

    lines: list[str] = []
    for raw_line in body.split("\n"):
        if _RUNNING_NUMBER_RE.match(raw_line):
            continue  # printed page number — the marker already records it
        if _BOILERPLATE_RE.match(raw_line):
            continue
        formatted = _format_line(raw_line)
        if formatted:
            lines.append(formatted)

    out = "\n".join(lines)
    if table_md:
        out = (out + "\n\n" + "\n\n".join(table_md)).strip()
    return out


def parse_pdf(pdf_path: Path, out_dir: Path = PARSED_MD_DIR) -> dict:
    meta = get_source_meta(pdf_path)
    doc = pymupdf.open(str(pdf_path))

    sections: list[str] = [
        write_front_matter({
            "act": meta["act"],
            "jurisdiction": meta["jurisdiction"],
            "authority": meta["authority"],
            "document_type": meta["document_type"],
            "state": meta["state"],
            "priority": meta["priority"],
            "unit": meta["unit"],
            "parser_style": meta["parser_style"],
            "source_pdf": pdf_path.name,
            "total_pages": doc.page_count,
        }),
        f"# {meta['act']}\n",
    ]

    non_empty = 0
    image_only: list[int] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        markdown = _page_to_markdown(page)
        sections.append(page_marker(page_index + 1))
        if markdown:
            sections.append(markdown)
            non_empty += 1
        elif page.get_images():
            # A plate with no text layer — e.g. the MV Act's First Schedule of
            # road-sign images. Recorded so the gap is visible, not silent.
            image_only.append(page_index + 1)
        sections.append("")
    doc.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")

    return {
        "source_pdf": pdf_path.name,
        "act": meta["act"],
        "markdown_file": out_path.name,
        "pages_total": page_index + 1,
        "pages_with_text": non_empty,
        "image_only_pages": image_only,
        "chars": out_path.stat().st_size,
    }


def run(raw_dir: Path = RAW_PDF_DIR, out_dir: Path = PARSED_MD_DIR) -> dict:
    pdfs = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {raw_dir} — run Phase 1 first.")

    results = []
    for pdf in pdfs:
        logger.info("  parsing %s", pdf.name)
        result = parse_pdf(pdf, out_dir)
        logger.info("    %d/%d pages carried text", result["pages_with_text"], result["pages_total"])
        if result["image_only_pages"]:
            pages = result["image_only_pages"]
            logger.info("    %d image-only pages (no text layer): %d–%d",
                        len(pages), pages[0], pages[-1])
        results.append(result)

    return {
        "phase": 2,
        "parsed_markdown_dir": str(out_dir),
        "documents_parsed": len(results),
        "pages_total": sum(r["pages_total"] for r in results),
        "image_only_pages_total": sum(len(r["image_only_pages"]) for r in results),
        "documents": results,
    }
