"""
Read-time wrapper over the Chroma collection data_pipeline's Phase 6 builds
offline. Data Service only ever queries it here — it never writes at request
time (single-writer discipline: never run Phase 6 while this service is up
against the same chroma_data/ directory).
"""

import chromadb

from services.shared.settings import settings

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0


def query(embedding: list[float], top_k: int) -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    # IMPORTANT: query with query_embeddings, never query_texts — Chroma
    # would silently fall back to its own default sentence-transformers
    # embedding function, a different vector space than what was indexed.
    result = collection.query(query_embeddings=[embedding], n_results=top_k)

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {
            "chunk_id": chunk_id,
            "record_id": metadata.get("record_id", ""),
            "text": text,
            "distance": distance,
            "metadata": metadata,
        }
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def get_embeddings(record_id: str) -> list[list[float]]:
    """The already-computed chunk embedding(s) for one dataset record — lets
    Retrieval Service score a graph-sourced candidate on the exact same
    vector space as a vector-search hit (real cosine similarity against the
    question embedding), instead of leaving it unscored. No new embedding
    call needed; these vectors were already stored by data_pipeline Phase 6."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(where={"record_id": record_id}, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return []
    return [list(vector) for vector in embeddings]
