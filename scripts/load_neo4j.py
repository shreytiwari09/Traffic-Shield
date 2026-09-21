"""
Builds the Neo4j graph from the pipeline's output (DATA/graph/entities.json +
relationships.json).

Offline and idempotent, like the rest of data_pipeline/: the live services only
ever read this graph, never write it. Re-running replaces the graph wholesale,
so it is safe to run after any pipeline rerun.

    ./.venv/Scripts/python.exe -m scripts.load_neo4j
    ./.venv/Scripts/python.exe -m scripts.load_neo4j --wipe

THE MODEL

    (:Entity)          {id, name, type, kind, aliases, alias_patterns, ...}
    (:Entity:Section)  {id, name, act, section, title, page, priority, ...}

    (:Entity)-[:RELATES     {id, relation, kind, evidence_count, evidence_ids}]->(:Entity)
    (:Entity)-[:EVIDENCED_BY {relation}]->(:Section)

WHY LEGAL SECTIONS CARRY BOTH LABELS

entities.json holds two populations under one `entity_count` of 1,331: 14
`concepts` (Driver, Police Officer, Fine — the alias-bearing domain nouns) and
1,317 `legal_sections`. Both are relationship endpoints: only 5 of the 1,390
relationships run concept-to-concept, and the other 1,385 are
section-MENTIONS-concept. So a legal section has to be an :Entity or those
1,385 edges have no source node to attach to — an earlier version of this
loader created only the concepts and silently produced a graph with 5
relationships in it, because MERGE_RELATIONSHIPS MATCHes both endpoints and
simply matches nothing when one is absent.

They are :Section as well because they are the same nodes the evidence lists
point at. One node with two labels, rather than a section node and a near-
duplicate entity node that would have to be kept in sync.

Two further decisions worth stating:

`RELATES` is one relationship type carrying a `relation` property, rather than
a distinct Cypher type per relation (MUST_OBEY, MENTIONS, ...). Typed edges
would read better in the Neo4j browser, but creating them requires either
dynamic relationship types or APOC, and every query this app runs filters on
the endpoints rather than the relation name — so the typed version would cost a
dependency and buy nothing at query time.

`evidence_ids` is a list property on the edge, and the same sections are ALSO
real (:Section) nodes joined by :EVIDENCED_BY. That looks redundant and is not:
the list preserves each relationship's own evidence ORDER, which the
per-relationship cap in neo4j_store.py depends on, while the nodes are what
make the corpus traversable — entity to section to the other entities citing
that section, which is the hop the flat-JSON store could never make.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase  # noqa: E402
from neo4j.exceptions import AuthError, ServiceUnavailable  # noqa: E402

from services.retrieval_service.neo4j_store import (  # noqa: E402
    ACT_PRIORITY,
    DEFAULT_PRIORITY,
    alias_pattern,
)
from services.shared.settings import settings  # noqa: E402

# Written in batches rather than one transaction per row: 1,331 entities and
# 1,390 relationships as individual transactions is thousands of round trips,
# and as a single transaction it is one oversized write. UNWIND over batches is
# the shape Neo4j actually wants.
BATCH = 500

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
    # Not a uniqueness rule but the index behind every match_entities() call —
    # relationships_for() and the hop-2 query both look entities up by name.
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    # evidence_record_ids() selects relationships by id; without this the
    # `r.id IN $rel_ids` filter is a full relationship scan on every question.
    "CREATE INDEX relates_id IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.id)",
    "CREATE INDEX section_priority IF NOT EXISTS FOR (s:Section) ON (s.priority)",
]

MERGE_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.id})
SET e.name = row.name,
    e.type = row.type,
    e.kind = row.kind,
    e.aliases = row.aliases,
    e.alias_patterns = row.alias_patterns,
    e.mention_count = row.mention_count,
    e.section_count = row.section_count
"""

# Legal sections are relationship endpoints as well as evidence targets, so
# they get :Entity too — see the module docstring. alias_patterns is set to []
# rather than left null so match_entities()'s `any(p IN e.alias_patterns ...)`
# evaluates to false for them instead of null: only the 14 concepts carry
# aliases, and only they should ever match a question.
MERGE_SECTIONS = """
UNWIND $rows AS row
MERGE (s:Section {id: row.id})
SET s:Entity,
    s.name = row.name,
    s.act = row.act,
    s.section = row.section,
    s.title = row.title,
    s.page = row.page,
    s.source_pdf = row.source_pdf,
    s.priority = row.priority,
    s.kind = 'legal_section',
    s.alias_patterns = []
"""

MERGE_RELATIONSHIPS = """
UNWIND $rows AS row
MATCH (a:Entity {id: row.source})
MATCH (b:Entity {id: row.target})
MERGE (a)-[r:RELATES {id: row.id}]->(b)
SET r.relation = row.relation,
    r.kind = row.kind,
    r.evidence_count = row.evidence_count,
    r.evidence_ids = row.evidence_ids
"""

MERGE_EVIDENCE_EDGES = """
UNWIND $rows AS row
MATCH (e:Entity {id: row.entity_id})
MATCH (s:Section {id: row.section_id})
MERGE (e)-[ev:EVIDENCED_BY]->(s)
SET ev.relation = row.relation
"""


def _batches(rows, size=BATCH):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def build(session, entities: dict, relationships: dict) -> dict:
    entity_rows = []
    for concept in entities.get("concepts", []):
        name = concept["name"]
        aliases = concept.get("aliases") or [name.lower()]
        entity_rows.append({
            "id": concept["id"],
            "name": name,
            "type": concept.get("type"),
            "kind": concept.get("kind"),
            "aliases": aliases,
            # Precomputed at load time so match_entities() is a plain regex
            # comparison at query time rather than string-building per request.
            "alias_patterns": [alias_pattern(a) for a in aliases],
            "mention_count": concept.get("mention_count", 0),
            "section_count": concept.get("section_count", 0),
        })

    rels = relationships.get("relationships", [])

    # Sections come from entities.json's `legal_sections` first, because that
    # is the authoritative record — it carries the title and source_pdf that
    # the abbreviated evidence entries do not.
    #
    # Priority is recomputed from the act name rather than taken from the
    # file's own `priority` field, even though the two agree today. The JSON
    # store ranks by exactly this table, and the whole point of
    # scripts/verify_neo4j.py is that the two backends rank identically —
    # sourcing the key differently would make that equivalence coincidental
    # rather than structural.
    sections: dict[str, dict] = {}
    for sec in entities.get("legal_sections", []):
        act = sec.get("act", "")
        sections[sec["id"]] = {
            "id": sec["id"],
            "name": sec.get("name"),
            "act": act,
            "section": sec.get("section"),
            "title": sec.get("title"),
            "page": sec.get("page"),
            "source_pdf": sec.get("source_pdf"),
            "priority": ACT_PRIORITY.get(act, DEFAULT_PRIORITY),
        }

    rel_rows = []
    evidence_rows = []
    for rel in rels:
        evidence_ids = []
        for ev in rel.get("evidence", []):
            sid = ev["section_id"]
            evidence_ids.append(sid)
            # Evidence entries are the fallback source for a section, not the
            # primary one: they carry no title. In practice every evidenced
            # section is already in legal_sections, so this is a guard against
            # a future pipeline change silently dropping evidence edges the
            # way the missing-entity bug dropped relationships.
            if sid not in sections:
                act = ev.get("act", "")
                sections[sid] = {
                    "id": sid,
                    "name": None,
                    "act": act,
                    "section": ev.get("section"),
                    "title": None,
                    "page": ev.get("page"),
                    "source_pdf": None,
                    "priority": ACT_PRIORITY.get(act, DEFAULT_PRIORITY),
                }
            # Both endpoints get an :EVIDENCED_BY edge, because the JSON store
            # indexed each relationship under both of them — an entity matched
            # as a target must be able to reach the same evidence.
            for endpoint in (rel["source"], rel["target"]):
                evidence_rows.append({
                    "entity_id": endpoint,
                    "section_id": sid,
                    "relation": rel["relation"],
                })

        rel_rows.append({
            "id": rel["id"],
            "source": rel["source"],
            "target": rel["target"],
            "relation": rel["relation"],
            "kind": rel.get("kind"),
            "evidence_count": rel.get("evidence_count", len(evidence_ids)),
            "evidence_ids": evidence_ids,
        })

    # Sections are written BEFORE relationships, because 1,385 of the 1,390
    # relationships have a legal section as their source and MERGE_RELATIONSHIPS
    # MATCHes both endpoints — reversing this order silently drops them.
    for batch in _batches(entity_rows):
        session.run(MERGE_ENTITIES, rows=batch)
    for batch in _batches(list(sections.values())):
        session.run(MERGE_SECTIONS, rows=batch)
    for batch in _batches(rel_rows):
        session.run(MERGE_RELATIONSHIPS, rows=batch)
    for batch in _batches(evidence_rows):
        session.run(MERGE_EVIDENCE_EDGES, rows=batch)

    # Counted from the database, not from len() of what was submitted. Those
    # two are not the same number, and assuming they were is what hid a graph
    # containing 5 of 1,390 relationships: MERGE...MATCH silently writes
    # nothing when an endpoint is missing, and reports no error.
    written = {
        "entities": session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"],
        "sections": session.run("MATCH (s:Section) RETURN count(s) AS n").single()["n"],
        "relationships": session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS n").single()["n"],
        "evidence_edges": session.run("MATCH ()-[r:EVIDENCED_BY]->() RETURN count(r) AS n").single()["n"],
    }
    submitted = {
        "entities": len(entity_rows) + len(sections),
        "sections": len(sections),
        "relationships": len(rel_rows),
        "evidence_edges": len({(e["entity_id"], e["section_id"]) for e in evidence_rows}),
    }
    return written, submitted


def main() -> int:
    parser = argparse.ArgumentParser(description="Load the traffic-law graph into Neo4j.")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete all existing nodes first. Use after a pipeline rerun that removed entities — "
             "a plain reload MERGEs and would leave the removed ones behind.",
    )
    args = parser.parse_args()

    if not settings.neo4j_password:
        print("NEO4J_PASSWORD is not set in .env — cannot connect.", file=sys.stderr)
        return 2

    for path in (settings.entities_path, settings.relationships_path):
        if not path.exists():
            print(f"Missing {path}. Run the data pipeline first.", file=sys.stderr)
            return 2

    with open(settings.entities_path, encoding="utf-8") as f:
        entities = json.load(f)
    with open(settings.relationships_path, encoding="utf-8") as f:
        relationships = json.load(f)

    print(f"Connecting to {settings.neo4j_uri} (database: {settings.neo4j_database})...")
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        driver.verify_connectivity()
    except AuthError:
        print("Authentication failed — check NEO4J_USER / NEO4J_PASSWORD in .env.", file=sys.stderr)
        return 1
    except ServiceUnavailable as exc:
        print(f"Cannot reach Neo4j at {settings.neo4j_uri}: {exc}", file=sys.stderr)
        print("Is the server running? See README 'Graph database (Neo4j)'.", file=sys.stderr)
        return 1

    with driver:
        with driver.session(database=settings.neo4j_database) as session:
            for constraint in CONSTRAINTS:
                session.run(constraint)

            if args.wipe:
                print("Wiping existing graph...")
                # Batched rather than a bare DETACH DELETE: deleting ~1,300
                # nodes and their ~50k evidence edges in one transaction builds
                # a transaction log larger than the default heap allows.
                while True:
                    summary = session.run(
                        "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS n"
                    ).single()
                    if summary["n"] == 0:
                        break

            print("Loading...")
            written, submitted = build(session, entities, relationships)

    print()
    print(f"{'':<16} {'in graph':>10} {'expected':>10}")
    shortfall = False
    for key in ("entities", "sections", "relationships", "evidence_edges"):
        flag = ""
        if written[key] < submitted[key]:
            flag = "  <-- MISSING"
            shortfall = True
        print(f"{key:<16} {written[key]:>10} {submitted[key]:>10}{flag}")

    if shortfall:
        print(
            "\nThe graph is missing rows that were submitted. MERGE...MATCH writes nothing "
            "(without erroring) when an endpoint node is absent — check that every relationship "
            "endpoint id exists in entities.json's concepts or legal_sections.",
            file=sys.stderr,
        )
        return 1

    print("\nVerify parity against the JSON store with: python -m scripts.verify_neo4j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
