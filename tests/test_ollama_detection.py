"""Ollama auto-detection: a machine without Ollama must skip it — local
embeddings instead of Ollama's, fast 503s instead of timed-out generations —
and a machine with it must keep using it. Ollama, the probe's network call and
the local model are all faked; nothing here touches localhost:11434."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from data_pipeline import phase6_vectorize
from data_pipeline.text_utils import write_jsonl
from services.llm_service import ollama_client
from services.llm_service.main import app as llm_app
from services.orchestration_service import clients as orch_clients
from services.orchestration_service.main import app as orch_app
from services.shared import local_embedder, ollama_probe
from services.shared.ollama_probe import OllamaStatus
from services.shared.settings import settings


def metric(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def auto_mode(monkeypatch):
    """OLLAMA_MODE=auto with a controllable fake probe."""
    monkeypatch.setattr(settings, "ollama_mode", "auto")
    ollama_probe.clear_cache()
    state = {"status": OllamaStatus(False, "not reachable (test)"), "probes": 0}

    async def fake_probe():
        state["probes"] += 1
        return state["status"]

    monkeypatch.setattr(ollama_probe, "_probe", fake_probe)
    yield state
    ollama_probe.clear_cache()


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------
def test_never_mode_reports_unavailable_without_probing(monkeypatch, auto_mode):
    monkeypatch.setattr(settings, "ollama_mode", "never")
    status = asyncio.run(ollama_probe.ollama_status())
    assert not status.available and "never" in status.reason
    assert auto_mode["probes"] == 0


def test_auto_mode_probe_is_cached_until_forced(auto_mode):
    asyncio.run(ollama_probe.ollama_status())
    asyncio.run(ollama_probe.ollama_status())
    assert auto_mode["probes"] == 1
    asyncio.run(ollama_probe.ollama_status(force=True))
    assert auto_mode["probes"] == 2


def test_has_model_matches_ollamas_latest_tag():
    status = OllamaStatus(True, "ok", frozenset({"nomic-embed-text:latest", "llama3.1:8b"}))
    assert status.has_model("nomic-embed-text") and status.has_model("llama3.1:8b")
    assert not status.has_model("codellama:7b")
    assert not OllamaStatus(False, "down").has_model("llama3.1:8b")
    assert OllamaStatus(True, "assumed").has_model("anything")  # models unknown -> assume present


# ---------------------------------------------------------------------------
# LLM service routing
# ---------------------------------------------------------------------------
def test_without_ollama_questions_are_embedded_locally(monkeypatch, auto_mode):
    async def must_not_call(*args, **kwargs):
        raise AssertionError("Ollama must not be called when it is unavailable")

    monkeypatch.setattr(ollama_client, "embed", must_not_call)
    monkeypatch.setattr(local_embedder, "embed_text", lambda text: [0.1, 0.2, 0.3])
    before = metric("ts_embedding_requests_total", {"backend": "local"})

    r = TestClient(llm_app).post("/v1/embed", json={"text": "fine for no helmet"})

    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "local" and body["model"] == settings.local_embedding_model
    assert body["embedding"] == [0.1, 0.2, 0.3]
    assert metric("ts_embedding_requests_total", {"backend": "local"}) == before + 1


def test_with_ollama_questions_are_embedded_by_ollama(monkeypatch, auto_mode):
    auto_mode["status"] = OllamaStatus(True, "running", frozenset({"nomic-embed-text:latest"}))

    async def fake_embed(text, model=None):
        return [0.9, 0.8]

    monkeypatch.setattr(ollama_client, "embed", fake_embed)
    monkeypatch.setattr(local_embedder, "embed_text", lambda text: pytest.fail("local model used despite Ollama"))

    body = TestClient(llm_app).post("/v1/embed", json={"text": "q"}).json()
    assert body["backend"] == "ollama" and body["embedding"] == [0.9, 0.8]


def test_ollama_running_without_embedding_model_falls_back_to_local(monkeypatch, auto_mode):
    auto_mode["status"] = OllamaStatus(True, "running", frozenset({"llama3.1:8b"}))
    monkeypatch.setattr(local_embedder, "embed_text", lambda text: [0.5])
    assert TestClient(llm_app).post("/v1/embed", json={"text": "q"}).json()["backend"] == "local"


def test_ollama_generation_fails_fast_when_ollama_is_missing(monkeypatch, auto_mode):
    async def must_not_call(*args, **kwargs):
        raise AssertionError("Ollama must not be called when it is unavailable")

    monkeypatch.setattr(ollama_client, "generate_with_usage", must_not_call)
    labels = {"provider": "ollama", "model": "llama3.1:8b", "status": "unavailable"}
    before = metric("ts_llm_generation_seconds_count", labels)

    r = TestClient(llm_app).post("/v1/generate", json={"question": "q", "provider": "ollama"})

    assert r.status_code == 503
    assert "Choose Gemini" in r.json()["detail"]
    assert metric("ts_llm_generation_seconds_count", labels) == before + 1


def test_llm_health_reports_detection(monkeypatch, auto_mode):
    body = TestClient(llm_app).get("/v1/health").json()
    assert body["ollama_reachable"] is False and body["embedding_backend"] == "local"


# ---------------------------------------------------------------------------
# What the Chat tab is told
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ollama,gemini,default", [
    (True, True, "ollama"), (False, True, "gemini"), (False, False, None),
])
def test_providers_endpoint_picks_a_working_default(monkeypatch, ollama, gemini, default):
    async def fake_health():
        return {"ollama_reachable": ollama, "gemini_configured": gemini, "ollama_model": "llama3.1:8b",
                "gemini_model": "g", "ollama_status": "test", "embedding_backend": "local"}

    monkeypatch.setattr(orch_clients, "llm_health", fake_health)
    body = TestClient(orch_app).get("/v1/providers").json()
    assert body["ollama"]["available"] is ollama and body["gemini"]["available"] is gemini
    assert body["default"] == default


# ---------------------------------------------------------------------------
# Phase 6 can rebuild the index from stored vectors
# ---------------------------------------------------------------------------
def test_reuse_keeps_vectors_for_unchanged_chunks_only(tmp_path):
    def chunk(cid, text, emb):
        return {"text": text, "embedding": emb, "metadata": {"chunk_id": cid}}

    previous = tmp_path / "chunks.jsonl"
    write_jsonl(previous, [chunk("A__0", "same text", [1.0, 0.0]), chunk("B__0", "old text", [0.0, 1.0])])
    fresh = [chunk("A__0", "same text", []), chunk("B__0", "edited text", []), chunk("C__0", "new", [])]

    assert phase6_vectorize._reuse_stored_embeddings(fresh, previous) == 1
    assert fresh[0]["embedding"] == [1.0, 0.0]
    assert fresh[1]["embedding"] == [] and fresh[2]["embedding"] == []  # edited / new -> re-embedded
