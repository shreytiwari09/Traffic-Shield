"""
nomic-embed-text v1.5 run in-process (ONNX via fastembed) — the embedding path
for machines without Ollama.

It must be the same model as the index, not just "a" local model: Chroma holds
vectors Ollama's nomic-embed-text produced, and a question embedded by any
other model lands in a different vector space, so vector search would return
nonsense with no error. Checked on 6 stored chunks spread across the corpus:
these vectors match Ollama's at cosine 1.0000 (nomic-embed-text v1, by
contrast, only reached ~0.84).

The first call downloads the weights (~550 MB) into local_embedding_cache_dir
and loads them (~1 s once cached); after that a question embeds in tens of
milliseconds on CPU.
"""

import threading

from services.shared.settings import settings

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            # Imported lazily: only machines that actually lack Ollama pay for
            # loading onnxruntime and the model.
            from fastembed import TextEmbedding

            settings.local_embedding_cache_dir.mkdir(parents=True, exist_ok=True)
            _model = TextEmbedding(settings.local_embedding_model, cache_dir=str(settings.local_embedding_cache_dir))
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """One vector per input, in order. Blocking (CPU-bound) — call through
    asyncio.to_thread from async code."""
    return [vector.tolist() for vector in _get_model().embed(texts)]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
