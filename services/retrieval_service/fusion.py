"""
Fuses vector-search hits with graph-evidence sections into one ranked
context list — the "hybrid retrieval" merge step.

Both pools are now scored on the SAME scale — real cosine similarity against
the question embedding — so the best candidate wins regardless of which path
found it. Earlier versions of this file either let vector hits always
out-rank every graph hit (graph given `score: None`, which sorts after any
real number no matter how relevant), or reserved a fixed number of slots for
graph candidates regardless of relevance. Both were found, during testing, to
either suppress or force-include candidates without regard to whether they
actually answered the question. A graph candidate's score is computed in
routes.py from its own chunk embedding(s), already stored by data_pipeline
Phase 6 — no extra Ollama call needed.
"""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# A small, explainable nudge for candidates that literally contain the
# glossary's canonical legal phrase (e.g. "registration certificate") —
# confirmed necessary during testing: MVA Section 158 genuinely scores lower
# by pure embedding similarity (0.613) than five vector hits that are only
# tangentially related (0.659-0.671), because 158 lists the phrase among five
# other unrelated document types in the same chunk, diluting its embedding —
# despite being the section that actually answers the question. Applied to
# every candidate regardless of source (vector or graph), so it's a genuine
# scoring correction, not a thumb on the scale for one path over the other.
KEYWORD_BOOST = 0.06

# A smaller boost for hand-verified "core section" candidates (see
# core_sections.py) — reflects genuine, human-checked topical relevance to
# the matched entity, not a literal phrase match. Kept smaller than
# KEYWORD_BOOST since "verified relevant to this entity" is a weaker signal
# than "literally contains the exact legal phrase." Confirmed necessary:
# MVA Sec 130 (defines the limited scope of a roadside stop) scored 0.6368
# against a real question — 0.0014 short of the 8th-place cutoff — despite
# being exactly the section needed to correctly frame the answer.
CORE_SECTION_BOOST = 0.03


def fuse(
    vector_hits: list[dict],
    graph_candidates: list[dict],
    top_k: int,
    boost_terms: list[str] | None = None,
    core_ids: set[str] | None = None,
) -> list[dict]:
    """
    vector_hits: Data Service's raw vector-search results.
    graph_candidates: [{"record": <dataset record>, "score": float | None}, ...]
        — score is a real cosine similarity when the record's embedding was
        found, or None when it couldn't be computed (record has no stored
        chunk, e.g. a parsing gap) — treated as the lowest possible score
        rather than skipped, so it can still surface if literally nothing
        else was found, but never displaces something actually relevant.
    """
    fused: dict[str, dict] = {}

    for hit in vector_hits:
        meta = hit["metadata"]
        fused[hit["record_id"]] = {
            "text": hit["text"],
            "title": meta.get("title"),
            "act": meta.get("act"),
            "section": meta.get("section"),
            "page": meta.get("page"),
            "source_pdf": meta.get("source_pdf"),
            "score": 1.0 - hit["distance"],  # cosine distance -> similarity
            "source": "vector",
            "_priority": meta.get("priority", 7),
        }

    for candidate in graph_candidates:
        record = candidate["record"]
        record_id = record["id"]
        score = candidate["score"] if candidate["score"] is not None else 0.0

        if record_id in fused:
            # Already present via vector search — keep whichever score is
            # actually higher rather than assuming the vector one wins.
            if score > fused[record_id]["score"]:
                fused[record_id]["score"] = score
            continue

        fused[record_id] = {
            "text": record.get("content", ""),
            "title": record.get("title"),
            "act": record.get("act"),
            "section": record.get("section"),
            "page": record.get("page"),
            "source_pdf": record.get("source_pdf"),
            "score": score,
            "source": "graph",
            "_priority": record.get("priority", 7),
        }

    # Checked against title as well as body text — confirmed necessary: MVA
    # Sec 194D's own penalty text never says "protective headgear" (only
    # cross-references "section 129" by number), but its TITLE, "Penalty for
    # not wearing protective headgear," does. Missing this meant the real
    # ₹1,000 penalty section never got boosted even once Sec 129 itself did.
    for term in boost_terms or []:
        term_lower = term.lower()
        for item in fused.values():
            haystack = item["text"].lower() + " " + (item.get("title") or "").lower()
            if term_lower in haystack:
                item["score"] = min(1.0, item["score"] + KEYWORD_BOOST)

    for record_id in core_ids or set():
        if record_id in fused:
            fused[record_id]["score"] = min(1.0, fused[record_id]["score"] + CORE_SECTION_BOOST)

    ranked = sorted(fused.values(), key=lambda item: (-item["score"], item["_priority"]))
    for item in ranked:
        item.pop("_priority", None)
    return ranked[:top_k]
