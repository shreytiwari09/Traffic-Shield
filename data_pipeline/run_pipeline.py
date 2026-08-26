"""
Pipeline orchestrator.

    python -m data_pipeline.run_pipeline                  # all seven phases
    python -m data_pipeline.run_pipeline --from 4         # resume at Phase 4
    python -m data_pipeline.run_pipeline --only 6         # one phase
    python -m data_pipeline.run_pipeline --skip-embeddings

Phases 1–5 and 7 need nothing but the PDFs. Phase 6 needs a running Ollama;
``--skip-embeddings`` lets the rest of the pipeline complete without one.
"""

import argparse
import json
import logging
import sys
import time

from data_pipeline import (
    phase1_collect,
    phase2_parse,
    phase3_clean,
    phase4_sections,
    phase5_validate,
    phase6_vectorize,
    phase7_graph,
)
from data_pipeline.config import ALL_DIRS, DATA_DIR
from data_pipeline.text_utils import write_json

logger = logging.getLogger("pipeline")

PHASES = [
    (1, "Data Collection", phase1_collect.run),
    (2, "PDF Parsing", phase2_parse.run),
    (3, "Data Cleaning", phase3_clean.run),
    (4, "Legal Section Extraction", phase4_sections.run),
    (5, "Metadata Validation", phase5_validate.run),
    (6, "Vectorization", phase6_vectorize.run),
    (7, "GraphRAG Data", phase7_graph.run),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrafficShield data pipeline")
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="first phase to run (default 1)")
    parser.add_argument("--to", dest="end", type=int, default=7,
                        help="last phase to run (default 7)")
    parser.add_argument("--only", type=int, help="run a single phase")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Phase 6: write chunks without calling Ollama")
    parser.add_argument("--embedding-model", default=None,
                        help="Phase 6: override the Ollama embedding model")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    start, end = (args.only, args.only) if args.only else (args.start, args.end)

    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    overall = time.perf_counter()

    for number, name, run_phase in PHASES:
        if not (start <= number <= end):
            continue

        logger.info("")
        logger.info("Phase %d — %s", number, name)
        logger.info("%s", "-" * 60)
        began = time.perf_counter()

        try:
            if number == 6:
                result = run_phase(
                    skip_embeddings=args.skip_embeddings,
                    embedding_model=args.embedding_model,
                )
            else:
                result = run_phase()
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            summary[f"phase_{number}"] = {"status": "failed", "error": str(exc)}
            write_json(DATA_DIR / "pipeline_report.json", summary)
            return 1

        elapsed = time.perf_counter() - began
        result["elapsed_seconds"] = round(elapsed, 1)
        result["status"] = "ok"
        summary[f"phase_{number}"] = result
        logger.info("  done in %.1fs", elapsed)

    report_path = DATA_DIR / "pipeline_report.json"
    write_json(report_path, summary)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline finished in %.1fs", time.perf_counter() - overall)
    logger.info("Report: %s", report_path)
    _print_deliverables(summary)
    return 0


def _print_deliverables(summary: dict) -> None:
    def get(phase: str, key: str, default="—"):
        return summary.get(phase, {}).get(key, default)

    logger.info("")
    logger.info("Deliverables under %s/", DATA_DIR.name)
    rows = [
        ("raw_pdfs/", f"{get('phase_1', 'documents_collected')} PDFs"),
        ("parsed_markdown/", f"{get('phase_2', 'documents_parsed')} documents, "
                             f"{get('phase_2', 'pages_total')} pages"),
        ("cleaned_sections/", f"{get('phase_3', 'pages_kept')} pages kept, "
                              f"{get('phase_3', 'pages_removed')} removed"),
        ("dataset.jsonl", f"{get('phase_5', 'accepted')} validated records "
                          f"({get('phase_5', 'rejected')} rejected)"),
        ("chunks.jsonl", f"{get('phase_6', 'total_chunks')} chunks, "
                         f"embeddings: {get('phase_6', 'embeddings')}"),
        ("graph/entities.json", f"{get('phase_7', 'concept_entities')} concepts + "
                                f"{get('phase_7', 'legal_section_entities')} sections"),
        ("graph/relationships.json", f"{get('phase_7', 'relationships')} edges"),
    ]
    for path, detail in rows:
        logger.info("  %-28s %s", path, detail)


if __name__ == "__main__":
    raise SystemExit(main())
