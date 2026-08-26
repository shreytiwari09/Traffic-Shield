"""
Phase 6 — Vectorization.

Chunks the validated dataset at section boundaries only, embeds each chunk via
Ollama, and stores the result in ChromaDB. Chunk records are also written to
``data/chunks.jsonl`` in the blueprint's ``{text, embedding, metadata}`` shape.

Blueprint rules honoured here:
  • split only at section boundaries — a chunk never spans two sections;
  • 500–800 tokens per chunk, with a 15 % overlap;
  • metadata stays attached to every chunk.

Embeddings are produced by Ollama, per the project brief's requirement that no
hosted embedding API is used. With ``--skip-embeddings`` the phase still writes
chunks.jsonl (embeddings left empty) so the corpus can be inspected without a
running Ollama.
"""

import asyncio
import logging
from pathlib import Path

from data_pipeline.config import (
    CHROMA_COLLECTION,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_RATIO,
    CHUNK_TARGET_TOKENS,
    CHUNKS_PATH,
    DATASET_PATH,
)
from data_pipeline.text_utils import estimate_tokens, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _paragraphs(text: str, target: int = CHUNK_TARGET_TOKENS) -> list[str]:
    """
    Split a section's body into the largest units below chunk size.

    Paragraphs are preferred; a paragraph that alone exceeds the target is
    split further at sentence ends so no unit is oversized to begin with.
    """
    units: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if estimate_tokens(para) <= target:
            units.append(para)
            continue
        # Oversized paragraph: fall back to sentences, then to a hard word
        # split. Legal text runs to long unpunctuated tables and schedules, so
        # without the last fallback a single "sentence" can blow past the band.
        buffer = ""
        for piece in _sentences(para):
            for unit in _split_oversized(piece, target):
                candidate = f"{buffer} {unit}".strip()
                if buffer and estimate_tokens(candidate) > target:
                    units.append(buffer)
                    buffer = unit
                else:
                    buffer = candidate
        if buffer:
            units.append(buffer)
    return units


def _split_oversized(text: str, target: int) -> list[str]:
    """Split at word boundaries when nothing else brings a unit under target."""
    if estimate_tokens(text) <= target:
        return [text]

    pieces: list[str] = []
    buffer: list[str] = []
    for word in text.split(" "):
        # Test before committing: appending first and checking after would let
        # the final word carry the piece past the target.
        if buffer and estimate_tokens(" ".join(buffer + [word])) > target:
            pieces.append(" ".join(buffer))
            buffer = []
        buffer.append(word)
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def _sentences(text: str) -> list[str]:
    import re
    # Legal text is dense with "sub-section (2)." and "rule 5." — split only
    # where a full stop is followed by whitespace and a capital or a "(n)".
    parts = re.split(r"(?<=\.)\s+(?=[A-Z(])", text)
    return [p for p in (p.strip() for p in parts) if p]


def chunk_record(record: dict) -> list[dict]:
    """
    Split one dataset record into 500–800-token chunks with 15 % overlap.

    A section below the lower bound is emitted whole rather than padded — the
    band governs how large sections are divided, not a minimum a section must
    reach.
    """
    content = record["content"]
    header = f"{record['act']} — Section {record['section']}: {record['title']}"

    total_tokens = estimate_tokens(content)
    if total_tokens <= CHUNK_TARGET_TOKENS:
        return [_build_chunk(record, content, 0, 1, header)]

    # Split chunks are prefixed with the citation line, so the text budget has
    # to leave room for it or the finished chunk overshoots the band.
    target = max(200, CHUNK_TARGET_TOKENS - estimate_tokens(header) - 4)

    units = _paragraphs(content, target)
    overlap_tokens = int(target * CHUNK_OVERLAP_RATIO)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target:
            chunks.append("\n\n".join(current))
            # Carry the tail of this chunk into the next one as overlap.
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev)
                if carry_tokens + prev_tokens > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tokens += prev_tokens
            current = carry
            current_tokens = carry_tokens
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        tail = "\n\n".join(current)
        # Avoid a runt final chunk made only of the overlap already emitted —
        # but never at the cost of pushing the previous chunk past the target.
        merged = f"{chunks[-1]}\n\n{tail}" if chunks else ""
        if (chunks and estimate_tokens(tail) < CHUNK_MIN_TOKENS * 0.4
                and estimate_tokens(merged) <= target):
            chunks[-1] = merged
        else:
            chunks.append(tail)

    return [
        _build_chunk(record, text, i, len(chunks), header)
        for i, text in enumerate(chunks)
    ]


def _build_chunk(record: dict, text: str, index: int, total: int, header: str) -> dict:
    """Wrap chunk text with the metadata the blueprint requires on every chunk."""
    # Prefixing the citation keeps a mid-section chunk self-identifying once it
    # is retrieved out of context.
    body = text if total == 1 else f"[{header}]\n{text}"
    return {
        "text": body,
        "embedding": [],
        "metadata": {
            "record_id": record["id"],
            "chunk_id": f"{record['id']}__{index}",
            "chunk_index": index,
            "chunk_total": total,
            "section": record["section"],
            "title": record["title"],
            "page": record["page"],
            "act": record["act"],
            "state": record["state"],
            "source_pdf": record["source_pdf"],
            "jurisdiction": record.get("jurisdiction", ""),
            "authority": record.get("authority", ""),
            "document_type": record.get("document_type", ""),
            "priority": record.get("priority", 7),
            "token_estimate": estimate_tokens(body),
        },
    }


def build_chunks(dataset_path: Path = DATASET_PATH) -> list[dict]:
    chunks: list[dict] = []
    for record in read_jsonl(dataset_path):
        chunks.extend(chunk_record(record))
    return chunks


# ---------------------------------------------------------------------------
# Embedding + storage
# ---------------------------------------------------------------------------
async def _embed_all(chunks: list[dict], model: str | None) -> None:
    from data_pipeline.embedder import embed_texts

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        vectors = await embed_texts([c["text"] for c in batch], model=model)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(vectors)} embeddings for {len(batch)} inputs"
            )
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector
        logger.info("    embedded %d/%d", min(start + EMBED_BATCH_SIZE, len(chunks)), len(chunks))


def _store_in_chroma(chunks: list[dict], collection_name: str) -> int:
    import chromadb

    from data_pipeline.config import PROJECT_ROOT

    persist_dir = PROJECT_ROOT / "chroma_data"
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma metadata values must be scalars.
    def flatten(meta: dict) -> dict:
        return {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in meta.items()
            if v is not None
        }

    batch = 2000
    for start in range(0, len(chunks), batch):
        window = chunks[start:start + batch]
        collection.add(
            ids=[c["metadata"]["chunk_id"] for c in window],
            embeddings=[c["embedding"] for c in window],
            documents=[c["text"] for c in window],
            metadatas=[flatten(c["metadata"]) for c in window],
        )
    return collection.count()


def run(
    dataset_path: Path = DATASET_PATH,
    out_path: Path = CHUNKS_PATH,
    skip_embeddings: bool = False,
    embedding_model: str | None = None,
) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found — run Phases 4–5 first.")

    chunks = build_chunks(dataset_path)
    if not chunks:
        raise RuntimeError("Chunking produced no chunks.")

    token_counts = [c["metadata"]["token_estimate"] for c in chunks]
    multi = [c for c in chunks if c["metadata"]["chunk_total"] > 1]

    report = {
        "phase": 6,
        "chunks_file": str(out_path),
        "total_chunks": len(chunks),
        "sections_split_into_multiple_chunks": len({c["metadata"]["record_id"] for c in multi}),
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_overlap_ratio": CHUNK_OVERLAP_RATIO,
        "token_estimate": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(sum(token_counts) / len(token_counts), 1),
            "over_target": sum(1 for t in token_counts if t > CHUNK_TARGET_TOKENS),
        },
        "embeddings": "skipped",
        "chroma_collection": None,
        "chroma_count": 0,
    }

    logger.info("  %d chunks from %d sections (mean %.0f tokens, max %d)",
                len(chunks), len(set(c["metadata"]["record_id"] for c in chunks)),
                report["token_estimate"]["mean"], report["token_estimate"]["max"])

    if skip_embeddings:
        logger.warning("  embeddings skipped — chunks.jsonl written without vectors")
        write_jsonl(out_path, chunks)
        return report

    logger.info("  embedding via Ollama …")
    asyncio.run(_embed_all(chunks, embedding_model))

    dims = len(chunks[0]["embedding"])
    count = _store_in_chroma(chunks, CHROMA_COLLECTION)

    write_jsonl(out_path, chunks)

    report["embeddings"] = "ollama"
    report["embedding_dimensions"] = dims
    report["chroma_collection"] = CHROMA_COLLECTION
    report["chroma_count"] = count
    logger.info("  stored %d vectors (%d dims) in collection '%s'",
                count, dims, CHROMA_COLLECTION)
    return report
