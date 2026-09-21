"""
The graph half of hybrid retrieval, behind one interface with two
interchangeable backends.

    settings.graph_backend = "neo4j"  -> neo4j_store      (Cypher, default)
    settings.graph_backend = "json"   -> graph_store_json (in-process dicts)

Nothing above this module knows which one is active: retrieval_service/routes.py
calls the same four functions either way. That is deliberate — it is what let
a real graph database replace the flat-JSON stand-in without touching the
fusion, scoring, or ranking code that the app's retrieval quality depends on.

FALLBACK
If `graph_backend` is "neo4j" and the database cannot be reached (not started,
wrong password, empty graph), this falls back to the JSON store rather than
leaving retrieval with no graph path. The reason is quantitative: on
entity-bearing questions the graph contributes roughly half the fused context,
so failing closed would silently halve answer quality while every service
still reported healthy. Falling back keeps answers correct and reports the
degradation in /v1/health, where it can actually be noticed.
"""

from services.retrieval_service import graph_store_json
from services.shared.settings import settings

# Imported lazily-ish: neo4j_store imports the driver at module level, and a
# json-only deployment should not need the package installed at all.
_neo4j_store = None
_active = graph_store_json
_fell_back = False


def _neo4j():
    global _neo4j_store
    if _neo4j_store is None:
        from services.retrieval_service import neo4j_store
        _neo4j_store = neo4j_store
    return _neo4j_store


def load() -> None:
    global _active, _fell_back
    _fell_back = False

    if settings.graph_backend == "neo4j":
        driver_missing = False
        try:
            if _neo4j().load():
                _active = _neo4j()
                return
        except ImportError:
            # Driver not installed — handled like an unreachable database.
            driver_missing = True
        if not settings.neo4j_fallback_to_json and not driver_missing:
            _active = _neo4j()  # stays unloaded; is_loaded() reports False
            return
        # With the driver absent there is no neo4j_store to point at, so the
        # JSON store is the only thing left to load regardless of the fallback
        # setting — the alternative is an ImportError escaping startup.
        _fell_back = True

    graph_store_json.load()
    _active = graph_store_json


def is_loaded() -> bool:
    return _active.is_loaded()


def status() -> dict:
    """What /v1/health reports. `fell_back` is the field that matters: it is
    the difference between "running on Neo4j as configured" and "quietly
    running on the JSON store because Neo4j is down"."""
    detail = _active.status()
    detail["configured_backend"] = settings.graph_backend
    detail["fell_back"] = _fell_back
    if settings.graph_backend == "neo4j" and _neo4j_store is not None:
        detail["neo4j_error"] = _neo4j_store.status().get("error")
    return detail


def match_entities(question: str) -> list[str]:
    return _active.match_entities(question)


def relationships_for(entity_names: list[str]) -> list[dict]:
    return _active.relationships_for(entity_names)


def evidence_record_ids(relationships: list[dict], per_relationship_limit: int = 5) -> list[str]:
    return _active.evidence_record_ids(relationships, per_relationship_limit)
