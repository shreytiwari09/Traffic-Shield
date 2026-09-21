"""
Proves the Neo4j graph backend behaves identically to the flat-JSON one it
replaced.

    ./.venv/Scripts/python.exe -m scripts.verify_neo4j

Why this exists rather than a smoke test: the graph contributes roughly half
the fused context on entity-bearing questions, and it does so *upstream* of
generation. A subtle divergence — one alias no longer matching, evidence
ranked differently, a per-relationship cap applied in the wrong place — would
not raise anything. It would surface as slightly worse answers, and would be
naturally misread as the model or the prompt regressing rather than retrieval.

So both backends are run over the same questions and compared exactly:
matched entities, relationship ids, and the ordered evidence list that feeds
the candidate cap in retrieval_service/routes.py.

Hop-2 expansion (settings.graph_expand_hops = 2) is deliberately NOT expected
to match: it reaches evidence the JSON store structurally cannot. This script
checks hop-1 parity and reports what hop 2 adds on top, which is the honest
way to show what the graph database actually bought.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.retrieval_service import graph_store_json, neo4j_store  # noqa: E402
from services.shared.settings import settings  # noqa: E402

# Spread across the corpus's topics on purpose: entity-heavy questions,
# abbreviation-only questions (the glossary expands these before retrieval
# sees them), a multi-entity question, and one that should match nothing.
QUESTIONS = [
    "Can the officer ask for my RC?",
    "Can my driving licence be seized on the spot?",
    "What is the fine for not wearing a helmet?",
    "Is a tinted windscreen allowed?",
    "Can a police officer arrest me without a warrant during a traffic stop?",
    "What documents must I carry while driving a motor vehicle?",
    "Can the police impound my vehicle for no insurance?",
    "What happens if I jump a red traffic signal?",
    "Do I need a permit for a transport vehicle?",
    "Can they check my pollution under control certificate?",
    "What are the rules about parking on a public road?",
    "Tell me a recipe for chocolate cake",
]


def compare(question: str) -> dict:
    json_entities = graph_store_json.match_entities(question)
    neo_entities = neo4j_store.match_entities(question)

    json_rels = graph_store_json.relationships_for(json_entities)
    neo_rels = neo4j_store.relationships_for(neo_entities)

    json_evidence = graph_store_json.evidence_record_ids(json_rels)
    neo_evidence = neo4j_store.evidence_record_ids(neo_rels)

    return {
        "question": question,
        "entities_match": json_entities == neo_entities,
        "json_entities": json_entities,
        "neo_entities": neo_entities,
        "rel_ids_match": {r["id"] for r in json_rels} == {r["id"] for r in neo_rels},
        "n_rels": (len(json_rels), len(neo_rels)),
        "evidence_match": json_evidence == neo_evidence,
        "n_evidence": (len(json_evidence), len(neo_evidence)),
        "json_evidence": json_evidence,
        "neo_evidence": neo_evidence,
    }


def main() -> int:
    # Hop-1 is the only comparable setting; force it regardless of .env so the
    # parity result does not silently depend on local configuration.
    original_hops = settings.graph_expand_hops
    settings.graph_expand_hops = 1

    if not graph_store_json.load():
        print("Could not load the JSON graph store — check DATA/graph/.", file=sys.stderr)
        return 2
    if not neo4j_store.load():
        print(f"Could not load the Neo4j store: {neo4j_store.status()['error']}", file=sys.stderr)
        return 2

    results = [compare(q) for q in QUESTIONS]
    failures = [r for r in results if not (r["entities_match"] and r["rel_ids_match"] and r["evidence_match"])]

    print(f"{'question':<58} {'entities':>9} {'rels':>7} {'evidence':>10}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['question'][:56]:<58} "
            f"{('OK' if r['entities_match'] else 'DIFF'):>9} "
            f"{('OK' if r['rel_ids_match'] else 'DIFF'):>7} "
            f"{('OK' if r['evidence_match'] else 'DIFF'):>10}"
        )

    if failures:
        print(f"\n{len(failures)} question(s) diverged:\n")
        for r in failures:
            print(f"  {r['question']}")
            if not r["entities_match"]:
                only_json = sorted(set(r["json_entities"]) - set(r["neo_entities"]))
                only_neo = sorted(set(r["neo_entities"]) - set(r["json_entities"]))
                print(f"    entities  json-only={only_json} neo4j-only={only_neo}")
            if not r["rel_ids_match"]:
                print(f"    relationships  json={r['n_rels'][0]} neo4j={r['n_rels'][1]}")
            if not r["evidence_match"]:
                print(f"    evidence  json={r['json_evidence'][:8]}")
                print(f"              neo4j={r['neo_evidence'][:8]}")
        settings.graph_expand_hops = original_hops
        return 1

    print(f"\nAll {len(results)} questions identical across both backends.")

    # Now show what the graph database adds that the JSON store could not do.
    settings.graph_expand_hops = 2
    print("\nHop-2 expansion (Neo4j only — no JSON equivalent exists):")
    for q in QUESTIONS[:6]:
        entities = neo4j_store.match_entities(q)
        rels = neo4j_store.relationships_for(entities)
        if not rels:
            continue
        settings.graph_expand_hops = 1
        hop1 = neo4j_store.evidence_record_ids(rels)
        settings.graph_expand_hops = 2
        hop2 = neo4j_store.evidence_record_ids(rels)
        print(f"  {q[:52]:<54} {len(hop1):>4} -> {len(hop2):>5} candidate sections")

    settings.graph_expand_hops = original_hops
    neo4j_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
