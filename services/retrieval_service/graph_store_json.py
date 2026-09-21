"""
The original flat-JSON graph store (DATA/graph/entities.json +
relationships.json), loaded once at startup and answered in-process.

This was the deliberate stand-in for a real graph database while none existed.
It is no longer the default — see neo4j_store.py — but it is kept as a working
backend for two reasons that are not sentimental:

  * it is the fallback when Neo4j is unreachable, so a stopped database
    degrades the graph half of hybrid retrieval instead of taking retrieval
    down with it;
  * it is the reference implementation scripts/verify_neo4j.py checks the
    Cypher backend against, which is what makes "the Neo4j path behaves
    identically" a verified claim rather than an assertion.

Small enough (1,331 entities / 1,390 edges) that an in-memory dict is instant.
"""

import json
import re

from services.shared.settings import settings

_alias_to_entity: dict[str, str] = {}
_entity_relationships: dict[str, list[dict]] = {}
_loaded = False


def load() -> bool:
    global _loaded

    if not settings.entities_path.exists() or not settings.relationships_path.exists():
        _loaded = False
        return False

    with open(settings.entities_path, encoding="utf-8") as f:
        entities = json.load(f)
    with open(settings.relationships_path, encoding="utf-8") as f:
        relationships = json.load(f)

    _alias_to_entity.clear()
    for concept in entities.get("concepts", []):
        name = concept["name"]
        aliases = concept.get("aliases") or [name.lower()]
        for alias in aliases:
            _alias_to_entity[alias.lower()] = name

    _entity_relationships.clear()
    for rel in relationships.get("relationships", []):
        for name in (rel["source_name"], rel["target_name"]):
            _entity_relationships.setdefault(name, []).append(rel)

    _loaded = True
    return True


def is_loaded() -> bool:
    return _loaded


def status() -> dict:
    return {
        "backend": "json",
        "loaded": _loaded,
        "entities": len(set(_alias_to_entity.values())),
        "aliases": len(_alias_to_entity),
    }


def match_entities(question: str) -> list[str]:
    """Alias/keyword match against the question text, using the same alias
    table data_pipeline Phase 7 used to build entities.json — so retrieval
    and graph-construction stay in sync."""
    q = question.lower()
    matched = set()
    for alias, entity_name in _alias_to_entity.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            matched.add(entity_name)
    return sorted(matched)


def relationships_for(entity_names: list[str]) -> list[dict]:
    seen_ids: set[str] = set()
    out: list[dict] = []
    for name in entity_names:
        for rel in _entity_relationships.get(name, []):
            if rel["id"] in seen_ids:
                continue
            seen_ids.add(rel["id"])
            out.append(rel)
    return out


# Mirrors data_pipeline/config.py's SOURCES priority tiers (1 = primary
# legislation ... 4 = supporting/procedural law). Kept as a local, small
# lookup rather than importing data_pipeline.config directly, matching the
# same reasoning as settings.py's own DATA_DIR: retrieval_service stays a
# self-contained reader, not coupled to the offline pipeline's module.
_ACT_PRIORITY = {
    "Motor Vehicles Act, 1988": 1,
    "Motor Vehicles (Amendment) Act, 2019": 1,
    "Central Motor Vehicles Rules, 1989": 2,
    "Haryana Motor Vehicle Rules, 1993": 2,
    "Motor Vehicles (Driving) Regulations, 2017": 3,
    "Bharatiya Sakshya Adhiniyam, 2023": 4,
}


def evidence_record_ids(relationships: list[dict], per_relationship_limit: int = 5) -> list[str]:
    """
    Collects every evidenced section across the given relationships, then
    ranks by the SOURCE ACT's real legal priority before any caller truncates
    the list — NOT by whatever order relationships.json happens to store
    them in. Without this, a heavily cross-referenced entity (e.g.
    "Registration Certificate" — mentioned in 111 relationships, mostly
    lower-priority CMVR/Haryana rules) can bury the one primary-Act section
    that actually answers the question dozens of positions deep, purely
    because of file order, and a small evidence cap (kept small on purpose,
    for latency — see routes.py's _MAX_GRAPH_EVIDENCE) would silently drop it.

    Ties within a priority tier are broken by section id. They previously kept
    file discovery order, which left the same complaint above only half-fixed:
    a tie-break by file position is still file order deciding what survives the
    cap. It also made this store impossible to compare against the Cypher one,
    since Neo4j guarantees no row order without an ORDER BY — so the two
    backends could not have been proven equivalent at all.
    """
    seen: set[str] = set()
    ranked: list[tuple[int, str]] = []
    for rel in relationships:
        for evidence in rel.get("evidence", [])[:per_relationship_limit]:
            section_id = evidence["section_id"]
            if section_id in seen:
                continue
            seen.add(section_id)
            priority = _ACT_PRIORITY.get(evidence.get("act", ""), 7)
            ranked.append((priority, section_id))
    ranked.sort()
    return [section_id for _, section_id in ranked]
