"""
Neo4j-backed graph store — the real graph database behind the hybrid
retrieval path, replacing the flat-JSON stand-in (graph_store_json.py, kept
as a fallback).

It answers the same four questions the JSON store did, so the retrieval route
above it did not have to change:

    is_loaded() / match_entities() / relationships_for() / evidence_record_ids()

WHAT ACTUALLY MOVED INTO THE DATABASE

Not just storage. Each of these was a Python loop over an in-memory dict and
is now a Cypher query the database plans and executes:

  * alias matching — was a regex scan over ~3,000 aliases per question
  * relationship lookup — was two dict lookups plus manual de-duplication
  * evidence ranking — was a Python sort by a hardcoded act-priority table;
    the priority now lives on the Section node itself, so ranking is an
    ORDER BY over indexed values rather than application logic

WHAT THE JSON STORE STRUCTURALLY COULD NOT DO

Evidence sections are real (:Section) nodes here, not string ids buried inside
a relationship's JSON payload. That makes the corpus an actual traversable
graph — entity to relationship to section to the other entities that cite the
same section — which is what `settings.graph_expand_hops = 2` uses to reach
evidence one hop further out than any dict lookup could.

PARITY IS THE POINT
Hop-1 results are intended to be byte-identical to the JSON store's, because
the graph path contributes roughly half the fused context on entity-bearing
questions and a silent behavior change here would look like a model
regression, not a retrieval one. scripts/verify_neo4j.py asserts that parity
across a question set rather than asking anyone to take it on trust.
"""

import re

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from services.shared.settings import settings

_driver = None
_loaded = False
_last_error: str | None = None


# The same act-priority tiers the JSON store used (mirroring
# data_pipeline/config.py's SOURCES). Applied once at load time and stored on
# each Section node, so ranking happens in the database instead of in Python.
ACT_PRIORITY = {
    "Motor Vehicles Act, 1988": 1,
    "Motor Vehicles (Amendment) Act, 2019": 1,
    "Central Motor Vehicles Rules, 1989": 2,
    "Haryana Motor Vehicle Rules, 1993": 2,
    "Motor Vehicles (Driving) Regulations, 2017": 3,
    "Bharatiya Sakshya Adhiniyam, 2023": 4,
}
DEFAULT_PRIORITY = 7


def alias_pattern(alias: str) -> str:
    r"""A Java-regex word-boundary matcher for one alias.

    \Q...\E quotes the alias literally, which matters because aliases come
    from the corpus and can contain regex metacharacters — an unescaped "(" in
    an alias would raise a PatternSyntaxException inside Cypher and fail the
    whole query, not just that one alias. Python's re.escape() is not reusable
    here: it produces Python-flavored escaping, and Cypher runs Java regexes.
    """
    return f"(?i).*\\b\\Q{alias}\\E\\b.*"


def _connect():
    global _driver, _loaded, _last_error
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    driver.verify_connectivity()
    _driver = driver
    _loaded = True
    _last_error = None


def load() -> bool:
    """Open the driver and confirm the graph is actually populated.

    Returns False (rather than raising) on any failure, so the caller can fall
    back to the JSON store — a Neo4j that is down should degrade retrieval,
    not take the service down with it.
    """
    global _loaded, _last_error
    try:
        _connect()
    except (ServiceUnavailable, Neo4jError, ValueError) as exc:
        _loaded = False
        _last_error = f"{type(exc).__name__}: {exc}"
        return False

    # Connectivity alone is not readiness: an empty database would answer every
    # query successfully with zero rows, which reads as "the question matched
    # no entities" and would silently disable the graph half of hybrid
    # retrieval on every request.
    try:
        with _driver.session(database=settings.neo4j_database) as session:
            count = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
    except Neo4jError as exc:
        _loaded = False
        _last_error = f"{type(exc).__name__}: {exc}"
        return False

    if count == 0:
        _loaded = False
        _last_error = "connected, but the graph is empty — run scripts/load_neo4j.py"
        return False

    _loaded = True
    return True


def close() -> None:
    global _driver, _loaded
    if _driver is not None:
        _driver.close()
    _driver = None
    _loaded = False


def is_loaded() -> bool:
    return _loaded


def status() -> dict:
    return {
        "backend": "neo4j",
        "loaded": _loaded,
        "uri": settings.neo4j_uri,
        "database": settings.neo4j_database,
        "error": _last_error,
    }


def _run(query: str, **params):
    with _driver.session(database=settings.neo4j_database) as session:
        return list(session.run(query, **params))


# The question text is matched against every entity's stored alias patterns
# inside the database. `any(... IN e.alias_patterns ...)` is the Cypher
# equivalent of the JSON store's per-alias regex loop, and returns the same
# set for the same input — see scripts/verify_neo4j.py.
_MATCH_ENTITIES = """
MATCH (e:Entity)
WHERE any(p IN e.alias_patterns WHERE $question =~ p)
RETURN e.name AS name
ORDER BY name
"""


def match_entities(question: str) -> list[str]:
    if not _loaded:
        return []
    # Newlines would make `.*` fail to span the question under Java regex
    # defaults (DOTALL is off), so a multi-line question could match nothing.
    flat = re.sub(r"\s+", " ", question)
    return [r["name"] for r in _run(_MATCH_ENTITIES, question=flat)]


# Undirected traversal, because the JSON store indexed each relationship under
# BOTH endpoints — an entity matched as a relationship's target must find it
# too. startNode/endNode are used for the returned names so the payload keeps
# the relationship's real stored direction rather than the traversal direction.
_RELATIONSHIPS_FOR = """
MATCH (a:Entity)-[r:RELATES]-(:Entity)
WHERE a.name IN $names
RETURN DISTINCT
  r.id             AS id,
  r.relation       AS relation,
  startNode(r).name AS source_name,
  endNode(r).name   AS target_name,
  r.evidence_count AS evidence_count
"""


def relationships_for(entity_names: list[str]) -> list[dict]:
    if not _loaded or not entity_names:
        return []
    return [dict(r) for r in _run(_RELATIONSHIPS_FOR, names=entity_names)]


# Hop 1: evidence from exactly the relationships passed in.
#
# Selected by relationship id, NOT by endpoint name. That distinction is the
# whole correctness of this query: the endpoint names of a matched entity's
# relationships include the entity's NEIGHBOURS, so re-querying by name would
# silently pull in the neighbours' own relationships too — hop-2 behavior
# wearing a hop-1 label, and a large, invisible widening of the candidate pool.
#
# Directed `-[r:RELATES]->` rather than undirected: the filter is on the id, so
# direction cannot affect which relationships match, and undirected would
# return every one of them twice.
#
# The [0..$per_rel] slice reproduces the JSON store's per-relationship evidence
# cap, and is applied BEFORE ranking, not after: capping after ranking would
# let one heavily-evidenced relationship crowd out every other relationship's
# contribution entirely.
_EVIDENCE_HOP1 = """
MATCH ()-[r:RELATES]->()
WHERE r.id IN $rel_ids
UNWIND r.evidence_ids[0..$per_rel] AS sid
MATCH (s:Section {id: sid})
RETURN DISTINCT sid AS id, s.priority AS priority
ORDER BY priority, id
"""

# Hop 2: evidence held by the relationships of entities one step out from a
# matched entity — reachable only because sections and entities are real nodes
# here. Ranked strictly after every hop-1 result rather than interleaved: a
# second-hop section is topically adjacent, not equally relevant, and
# interleaving would let it displace a directly-evidenced section from the
# candidate cap in retrieval_service/routes.py.
_EVIDENCE_HOP2 = """
MATCH ()-[r:RELATES]->()
WHERE r.id IN $rel_ids
WITH collect(DISTINCT startNode(r).name) + collect(DISTINCT endNode(r).name) AS names
UNWIND names AS n
MATCH (e:Entity {name: n})-[r2:RELATES]-(:Entity)
WHERE NOT r2.id IN $rel_ids
UNWIND r2.evidence_ids[0..$per_rel] AS sid
MATCH (s:Section {id: sid})
RETURN DISTINCT sid AS id, s.priority AS priority
ORDER BY priority, id
"""


def evidence_record_ids(relationships: list[dict], per_relationship_limit: int = 5) -> list[str]:
    """Evidence section ids for the current match, ranked by source-act
    priority (1 = primary legislation ... 4 = supporting law).

    Signature is inherited from the JSON store so retrieval_service/routes.py
    is unchanged. The ranking key itself lives on the Section nodes, applied
    once at load time rather than recomputed per request from a hardcoded
    table.

    Ties are broken by section id, in both this store and the JSON one. The
    JSON store originally left ties in whatever order relationships.json
    happened to list them — its own docstring already called that out as a
    problem, and it made the two backends impossible to compare, since Cypher
    guarantees no row order without an ORDER BY. Sorting ties by id makes the
    result deterministic and lets scripts/verify_neo4j.py assert exact
    equality instead of "close enough".
    """
    if not _loaded or not relationships:
        return []

    rel_ids = [r["id"] for r in relationships]
    ordered = [r["id"] for r in _run(_EVIDENCE_HOP1, rel_ids=rel_ids, per_rel=per_relationship_limit)]

    if settings.graph_expand_hops >= 2:
        seen = set(ordered)
        for r in _run(_EVIDENCE_HOP2, rel_ids=rel_ids, per_rel=per_relationship_limit):
            if r["id"] not in seen:
                seen.add(r["id"])
                ordered.append(r["id"])

    return ordered
