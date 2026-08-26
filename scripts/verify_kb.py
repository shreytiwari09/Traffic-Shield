"""
Exercise 2 smoke test — confirms the knowledge base is actually queryable.

Queries the persisted Chroma collection directly (no services need to be
running) for a handful of real driver-rights questions, and prints the
nearest legal sections. This is the direct evidence that
"Documents -> Chunking -> Embeddings -> Vector Representation" produced a
real, searchable knowledge base, not just files on disk.

Run from the repo root:
    python scripts/verify_kb.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from data_pipeline.config import CHROMA_COLLECTION, PROJECT_ROOT  # noqa: E402
from data_pipeline.embedder import embed_single  # noqa: E402

SAMPLE_QUESTIONS = [
    "Can a police officer ask me to produce my driving licence?",
    "What is the fine for not wearing a helmet?",
    "Can my registration certificate be seized on the spot?",
]


async def main() -> None:
    persist_dir = PROJECT_ROOT / "chroma_data"
    if not persist_dir.exists():
        print(f"No Chroma data at {persist_dir} yet.")
        print("Run: ollama pull nomic-embed-text")
        print("Then: python -m data_pipeline.run_pipeline --only 6")
        return

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        print(f"Collection '{CHROMA_COLLECTION}' not found in {persist_dir}.")
        return

    if collection.count() == 0:
        print(f"Collection '{CHROMA_COLLECTION}' exists but is empty — rerun Phase 6 without --skip-embeddings.")
        return

    print(f"Collection '{CHROMA_COLLECTION}': {collection.count()} vectors\n")

    for question in SAMPLE_QUESTIONS:
        vector = await embed_single(question)
        result = collection.query(query_embeddings=[vector], n_results=3)
        print(f"Q: {question}")
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            print(f"  [{dist:.3f}] {meta['act']} — Sec {meta['section']} (p.{meta['page']})")
            snippet = doc[:140].replace("\n", " ")
            print(f"         {snippet}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
