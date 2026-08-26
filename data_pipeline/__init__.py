"""
TrafficShield data pipeline.

Implements the seven phases of DATA/Haryana_Traffic_Data_Blueprint (1).md:

    1. Collection          trafficshield_kb/ -> data/raw_pdfs/
    2. PDF parsing         data/raw_pdfs/    -> data/parsed_markdown/
    3. Cleaning            parsed_markdown/  -> data/cleaned_sections/
    4. Section extraction  cleaned_sections/ -> data/dataset.jsonl
    5. Metadata validation dataset.jsonl     -> data/validation_report.json
    6. Vectorization       dataset.jsonl     -> data/chunks.jsonl (+ Chroma)
    7. GraphRAG            dataset.jsonl     -> data/graph/{entities,relationships}.json

Scope is data only — no app or model development.
"""

__all__ = ["config", "text_utils"]
