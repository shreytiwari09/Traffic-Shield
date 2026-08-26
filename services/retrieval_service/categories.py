"""
Rights Library — browsing the knowledge base by category, rather than asking
a question. Reuses the same 12 graph concept entities already built by
data_pipeline Phase 7 (no new categorization system): each library category
maps onto one or more existing entity names, and "browsing" a category means
walking that entity's real graph relationships for evidence sections, the
same lookup the Q&A path already does — just without a question to match
entities against first, and without the top-k cap.
"""

import asyncio

from services.retrieval_service import clients, graph_store

CATEGORIES: dict[str, dict] = {
    "driver-rights": {"label": "Driver Rights", "entities": ["Driver"]},
    "police-powers": {"label": "Police Powers", "entities": ["Police Officer"]},
    "documents-required": {
        "label": "Documents Required",
        "entities": ["Driving Licence", "Registration Certificate"],
    },
    "traffic-signals": {"label": "Traffic Signals", "entities": ["Traffic Signal"]},
    "challans-fines": {"label": "Challans & Fines", "entities": ["Challan", "Fine"]},
}

_MAX_SECTIONS = 60  # a browse list, not a paginated API — keep it to a sane page size


def list_categories() -> list[dict]:
    return [
        {"slug": slug, "label": info["label"], "entity_names": info["entities"]}
        for slug, info in CATEGORIES.items()
    ]


async def sections_for_category(slug: str) -> dict | None:
    info = CATEGORIES.get(slug)
    if info is None:
        return None

    relationships = graph_store.relationships_for(info["entities"])
    # No top_k truncation here (unlike the Q&A path) — a per_relationship_limit
    # high enough to surface effectively every evidenced section for the category.
    evidence_ids = graph_store.evidence_record_ids(relationships, per_relationship_limit=1000)
    evidence_ids = evidence_ids[:_MAX_SECTIONS]

    fetched = await asyncio.gather(*(clients.get_record(rid) for rid in evidence_ids))
    sections = [record for record in fetched if record]

    return {
        "category": {"slug": slug, "label": info["label"], "entity_names": info["entities"]},
        "sections": sections,
    }
