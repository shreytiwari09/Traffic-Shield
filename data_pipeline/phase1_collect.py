"""
Phase 1 — Data Collection.

Gathers the official legal PDFs into ``data/raw_pdfs/`` under their original
filenames and records a manifest (size + SHA-256) so later phases can tell
whether the corpus changed.

The blueprint's five required sources are all present in ``trafficshield_kb/``;
this phase copies rather than re-downloads, and reports any required source
that is missing so the gap is visible instead of silent.
"""

import hashlib
import logging
import shutil
from pathlib import Path

from data_pipeline.config import ID_PREFIXES, KB_DIR, MANIFEST_PATH, RAW_PDF_DIR, SOURCES
from data_pipeline.text_utils import write_json

logger = logging.getLogger(__name__)


def normalise_stem(filename: str) -> str:
    """Lowercase filename stem with separators normalised to underscores."""
    return Path(filename).stem.lower().replace("-", "_").replace(" ", "_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_pdfs(kb_dir: Path = KB_DIR) -> list[Path]:
    """Find every PDF in the knowledge base, case-insensitively, deduplicated."""
    found: dict[str, Path] = {}
    for path in kb_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            found.setdefault(str(path.resolve()).lower(), path)
    return sorted(found.values())


def run(kb_dir: Path = KB_DIR, out_dir: Path = RAW_PDF_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(kb_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {kb_dir}")

    entries: list[dict] = []
    for src in pdfs:
        stem = normalise_stem(src.name)
        dest = out_dir / src.name

        # Copy only when the destination is absent or stale.
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
            action = "copied"
        else:
            action = "up-to-date"

        known = stem in SOURCES
        entries.append({
            "filename": src.name,
            "stem": stem,
            "registered": known,
            "act": SOURCES.get(stem, {}).get("act", "UNREGISTERED"),
            "id_prefix": ID_PREFIXES.get(stem, ""),
            "size_bytes": dest.stat().st_size,
            "sha256": sha256(dest),
        })
        logger.info("  %-12s %s", action, src.name)
        if not known:
            logger.warning("    ! %s is not in the source registry — Phase 4 will "
                           "fall back to generic metadata", src.name)

    collected_stems = {e["stem"] for e in entries}
    missing = sorted(set(SOURCES) - collected_stems)
    for stem in missing:
        logger.warning("  MISSING required source: %s (%s)", stem, SOURCES[stem]["act"])

    summary = {
        "phase": 1,
        "raw_pdf_dir": str(out_dir),
        "documents_collected": len(entries),
        "required_sources": len(SOURCES),
        "missing_sources": missing,
        "documents": entries,
    }
    write_json(MANIFEST_PATH, summary)
    return summary
