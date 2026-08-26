"""
Phase 7 — GraphRAG Data.

Builds ``graph/entities.json`` and ``graph/relationships.json`` from the
validated dataset.

Every edge is evidenced by real legal text: the blueprint's domain
relationships are emitted only for the sections whose text actually mentions
both endpoints, and each edge carries the section ids and pages that support
it. Nothing here invents law — it indexes what Phase 4 already extracted.
"""

import logging
import re
from pathlib import Path

from data_pipeline.config import (
    DATASET_PATH,
    DOMAIN_RELATIONSHIPS,
    ENTITIES_PATH,
    ENTITY_ALIASES,
    ENTITY_TYPES,
    RELATIONSHIPS_PATH,
)
from data_pipeline.text_utils import read_jsonl, write_json

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 25  # cap the evidence list per edge so the files stay readable


def _build_matchers() -> dict[str, re.Pattern]:
    """One word-boundary alternation per canonical entity."""
    matchers: dict[str, re.Pattern] = {}
    for entity, aliases in ENTITY_ALIASES.items():
        if not aliases:
            continue
        # Longest first so "driving licence" wins over "licence" style overlaps.
        ordered = sorted(aliases, key=len, reverse=True)
        pattern = "|".join(re.escape(a) for a in ordered)
        matchers[entity] = re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)
    return matchers


def _mentions(text: str, matchers: dict[str, re.Pattern]) -> dict[str, int]:
    """Count entity mentions in a section's text."""
    found: dict[str, int] = {}
    for entity, matcher in matchers.items():
        hits = len(matcher.findall(text))
        if hits:
            found[entity] = hits
    return found


def run(
    dataset_path: Path = DATASET_PATH,
    entities_path: Path = ENTITIES_PATH,
    relationships_path: Path = RELATIONSHIPS_PATH,
) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found — run Phases 4–5 first.")

    matchers = _build_matchers()
    records = list(read_jsonl(dataset_path))

    # --- Concept entities ---------------------------------------------------
    concept_entities: dict[str, dict] = {
        name: {
            "id": f"ENT_{name.upper().replace(' ', '_')}",
            "name": name,
            "type": ENTITY_TYPES.get(name, "Concept"),
            "kind": "concept",
            "mention_count": 0,
            "section_count": 0,
            "aliases": ENTITY_ALIASES.get(name, []),
        }
        for name in ENTITY_ALIASES
    }

    # --- Legal-section entities, one per dataset record ---------------------
    section_entities: list[dict] = []
    per_record_mentions: dict[str, dict[str, int]] = {}

    for record in records:
        text = f"{record['title']}\n{record['content']}"
        found = _mentions(text, matchers)
        per_record_mentions[record["id"]] = found

        for entity, hits in found.items():
            concept_entities[entity]["mention_count"] += hits
            concept_entities[entity]["section_count"] += 1

        section_entities.append({
            "id": record["id"],
            "name": f"{record['act']} — {record['section']}",
            "type": "Law",
            "kind": "legal_section",
            "act": record["act"],
            "section": record["section"],
            "title": record["title"],
            "page": record["page"],
            "source_pdf": record["source_pdf"],
            "state": record["state"],
            "jurisdiction": record.get("jurisdiction", ""),
            "priority": record.get("priority", 7),
            "mentions": sorted(found),
        })

    concept_entities["Legal Section"]["section_count"] = len(section_entities)
    concept_entities["Legal Section"]["mention_count"] = len(section_entities)

    entities = {
        "entity_count": len(concept_entities) + len(section_entities),
        "concepts": list(concept_entities.values()),
        "legal_sections": section_entities,
    }

    # --- Relationships ------------------------------------------------------
    relationships: list[dict] = []

    # 1. Domain edges from the blueprint, evidenced by co-occurrence.
    for spec in DOMAIN_RELATIONSHIPS:
        source, target = spec["source"], spec["target"]
        evidence: list[dict] = []

        for record in records:
            found = per_record_mentions[record["id"]]
            # "Legal Section" is satisfied by the record itself, not by a keyword.
            source_ok = source == "Legal Section" or source in found
            target_ok = target == "Legal Section" or target in found
            if source_ok and target_ok:
                evidence.append({
                    "section_id": record["id"],
                    "act": record["act"],
                    "section": record["section"],
                    "page": record["page"],
                })

        relationships.append({
            "id": f"REL_{source.upper().replace(' ', '_')}_{spec['relation']}_"
                  f"{target.upper().replace(' ', '_')}",
            "source": concept_entities[source]["id"],
            "source_name": source,
            "relation": spec["relation"],
            "target": concept_entities[target]["id"],
            "target_name": target,
            "kind": "domain",
            "evidence_count": len(evidence),
            "evidence": evidence[:MAX_EVIDENCE],
        })

    # 2. Section → concept edges, so retrieval can walk from a topic to the
    #    provisions that actually speak to it.
    for section in section_entities:
        for entity_name in section["mentions"]:
            if entity_name == "Legal Section":
                continue
            relationships.append({
                "id": f"REL_{section['id']}_MENTIONS_"
                      f"{entity_name.upper().replace(' ', '_')}",
                "source": section["id"],
                "source_name": section["name"],
                "relation": "MENTIONS",
                "target": concept_entities[entity_name]["id"],
                "target_name": entity_name,
                "kind": "derived",
                "evidence_count": 1,
                "evidence": [{
                    "section_id": section["id"],
                    "act": section["act"],
                    "section": section["section"],
                    "page": section["page"],
                }],
            })

    domain_edges = [r for r in relationships if r["kind"] == "domain"]
    payload = {
        "relationship_count": len(relationships),
        "domain_relationship_count": len(domain_edges),
        "derived_relationship_count": len(relationships) - len(domain_edges),
        "relationships": relationships,
    }

    write_json(entities_path, entities)
    write_json(relationships_path, payload)

    for edge in domain_edges:
        logger.info("  %-16s %-10s %-20s  %d sections",
                    edge["source_name"], edge["relation"], edge["target_name"],
                    edge["evidence_count"])
        if edge["evidence_count"] == 0:
            logger.warning("    ! no section evidences this relationship")

    return {
        "phase": 7,
        "entities_file": str(entities_path),
        "relationships_file": str(relationships_path),
        "concept_entities": len(concept_entities),
        "legal_section_entities": len(section_entities),
        "relationships": len(relationships),
        "domain_relationships": [
            {
                "edge": f"{e['source_name']} -{e['relation']}-> {e['target_name']}",
                "evidence_count": e["evidence_count"],
            }
            for e in domain_edges
        ],
    }
