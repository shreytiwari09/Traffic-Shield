import asyncio
import time

from fastapi import APIRouter, HTTPException

from services.retrieval_service import categories, clients, core_sections, fusion, glossary, graph_store
from services.shared.schemas import (
    CategorySectionsResponse,
    GraphRelationship,
    RetrievalTiming,
    RetrieveRequest,
    RetrieveResponse,
)

router = APIRouter()

# A high-frequency entity (e.g. Registration Certificate) can be evidenced by
# ~100 relationships. Fetching that many individual records only to keep the
# top `top_k` after fusion is wasteful — cap what's fetched per request, and
# fetch concurrently rather than one-at-a-time (a real, pre-existing bug: the
# sequential version took 48s on a 100-evidence-id query in testing, which
# would make the live pipeline view look frozen).
_MAX_GRAPH_EVIDENCE = 20

# The general default-penalty provision (MVA Sec 177: "if no penalty is
# provided for the offence, fine up to five hundred rupees..."). It can
# never win on relevance score against a specific question — its whole text
# is a generic catch-all that names no violation — so it's guaranteed a
# slot instead whenever "Fine" is asked about and no more specific
# fine-bearing section was already found. Confirmed necessary during
# testing: without this, Ollama fabricated a specific fine amount rather
# than citing the real (much smaller) default penalty.
_GENERAL_PENALTY_RECORD_ID = "HR_MVA_177"


async def _score_graph_candidate(record_id: str, query_embedding: list[float]) -> dict | None:
    """Fetches one graph-evidenced record plus its already-stored chunk
    embedding(s), and scores it by real cosine similarity against the same
    question embedding vector search used — so a graph candidate competes on
    the same scale as a vector hit, not a blind priority guess."""
    record, embeddings = await asyncio.gather(
        clients.get_record(record_id),
        clients.get_embeddings(record_id),
    )
    if record is None:
        return None
    score = max((fusion.cosine_similarity(query_embedding, e) for e in embeddings), default=None)
    return {"record": record, "score": score}


@router.get("/v1/health")
async def health():
    return {"status": "ok", "graph_loaded": graph_store.is_loaded()}


@router.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    # Normalize known abbreviations ("RC", "DL", ...) before EITHER retrieval
    # path — fixes vector search and graph alias-matching with one glossary,
    # since the corpus itself never spells out these abbreviations either.
    expanded_question = glossary.expand_query(req.question)

    t_embed_start = time.perf_counter()
    try:
        embedding = await clients.embed_question(expanded_question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc
    embed_ms = (time.perf_counter() - t_embed_start) * 1000

    t_vector_start = time.perf_counter()
    vector_hits = []
    if req.mode == "hybrid":
        try:
            vector_hits = await clients.vector_search(embedding, req.top_k)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"vector search failed: {exc}") from exc
    vector_search_ms = (time.perf_counter() - t_vector_start) * 1000

    t_graph_start = time.perf_counter()
    matched_entities = graph_store.match_entities(expanded_question)
    relationships = graph_store.relationships_for(matched_entities)
    evidence_ids = graph_store.evidence_record_ids(relationships)[:_MAX_GRAPH_EVIDENCE]

    # Hand-verified core sections supplement the noisy auto-derived MENTIONS
    # edges above — same candidate pool, same fair scoring below, just a
    # higher-precision source. Confirmed necessary during testing across
    # several topics (helmet, RC, driving licence, police authority during a
    # stop) where the correct section either scored too low among the noisy
    # candidates to be fetched at all, or wasn't evidenced by any graph edge
    # in the first place.
    core_ids = set(core_sections.core_section_ids(matched_entities))
    candidate_ids = list(dict.fromkeys(evidence_ids + list(core_ids)))

    graph_candidates = await asyncio.gather(
        *(_score_graph_candidate(rid, embedding) for rid in candidate_ids)
    )
    graph_candidates = [c for c in graph_candidates if c is not None]
    graph_ms = (time.perf_counter() - t_graph_start) * 1000

    boost_terms = glossary.matched_canonical_terms(req.question)
    context = fusion.fuse(vector_hits, graph_candidates, req.top_k, boost_terms=boost_terms, core_ids=core_ids)

    if "Fine" in matched_entities and not any(c["source_pdf"] == "motor_vehicles_act_1988.pdf" and c["section"] == "177" for c in context):
        general_penalty = await clients.get_record(_GENERAL_PENALTY_RECORD_ID)
        if general_penalty:
            context.append({
                "text": general_penalty["content"],
                "act": general_penalty["act"],
                "section": general_penalty["section"],
                "page": general_penalty["page"],
                "source_pdf": general_penalty["source_pdf"],
                "score": None,
                "source": "default_penalty_fallback",
            })

    return RetrieveResponse(
        context=context,
        matched_entities=matched_entities,
        graph_relationships=[
            GraphRelationship(
                relation=r["relation"],
                source_name=r["source_name"],
                target_name=r["target_name"],
                evidence_count=r["evidence_count"],
            )
            for r in relationships
        ],
        timing=RetrievalTiming(
            embed_ms=round(embed_ms, 1),
            vector_search_ms=round(vector_search_ms, 1),
            graph_ms=round(graph_ms, 1),
        ),
    )


@router.get("/v1/categories")
async def list_categories():
    return {"categories": categories.list_categories()}


@router.get("/v1/categories/{slug}/sections", response_model=CategorySectionsResponse)
async def category_sections(slug: str):
    result = await categories.sections_for_category(slug)
    if result is None:
        raise HTTPException(status_code=404, detail=f"unknown category '{slug}'")
    return result
