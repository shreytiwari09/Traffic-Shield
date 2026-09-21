"""
Shared configuration for every service, loaded from environment variables /
``.env`` via pydantic-settings. One source of truth so ports, model names, and
paths are never hardcoded twice across five processes.

Note: deliberately defines its own, correctly-cased ``DATA_DIR`` rather than
importing ``data_pipeline.config.DATA_DIR`` (which is spelled lowercase
``"data"`` while the real folder on disk is ``DATA`` — only works today
because Windows is case-insensitive; would break on Linux/Docker later).
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), extra="ignore")

    # --- paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "DATA"
    dataset_path: Path = PROJECT_ROOT / "DATA" / "dataset.jsonl"
    chunks_path: Path = PROJECT_ROOT / "DATA" / "chunks.jsonl"
    entities_path: Path = PROJECT_ROOT / "DATA" / "graph" / "entities.json"
    relationships_path: Path = PROJECT_ROOT / "DATA" / "graph" / "relationships.json"
    chroma_dir: Path = PROJECT_ROOT / "chroma_data"
    chroma_collection: str = "trafficshield_sections"

    # --- Ollama --------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"          # config-driven; swap to "codellama" with one edit
    ollama_embedding_model: str = "nomic-embed-text"

    # How long to wait on one Ollama call. Was hardcoded at 180s in
    # ollama_client.py, which is not enough on a CPU-only host: a *cold* model
    # load (~5GB off disk) plus ~2k prompt tokens at ~25 tok/s plus generation
    # at ~6.5 tok/s overruns it, and the request dies with an empty-message
    # httpx.ReadTimeout. Config-driven now so slow hardware is a .env change,
    # not a code edit.
    ollama_timeout_seconds: float = 600.0

    # Passed to Ollama as `keep_alive`, so the model stays resident between
    # questions instead of being evicted after its default 5-minute idle. The
    # cold reload is what pushed the first question after an idle gap over the
    # old timeout; keeping it warm removes that cliff. Set "0" to free RAM
    # immediately after each call on memory-constrained machines.
    ollama_keep_alive: str = "30m"

    # --- Gemini --------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"  # override in .env if your account has a different model enabled/quota'd

    # --- service URLs (inter-service calls) ----------------------------------
    data_service_url: str = "http://localhost:8004"
    llm_service_url: str = "http://localhost:8003"
    retrieval_service_url: str = "http://localhost:8002"
    orchestration_service_url: str = "http://localhost:8001"

    # --- Neo4j (graph backend) -----------------------------------------------
    # "neo4j": entity/relationship lookups run as Cypher against a real graph
    # database. "json": the original in-process flat-JSON store. Both answer
    # the identical graph_store interface and are verified to return identical
    # matches (scripts/verify_neo4j.py), so this is a genuine switch rather
    # than two divergent code paths.
    graph_backend: Literal["neo4j", "json"] = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # If Neo4j is unreachable at startup, fall back to the JSON store rather
    # than leaving retrieval with no graph path at all. The graph contributes
    # roughly half the fused context on entity-bearing questions, so failing
    # closed here would quietly halve answer quality; failing over keeps the
    # app answering and says so in /v1/health.
    neo4j_fallback_to_json: bool = True

    # How far to walk out from a matched entity when collecting evidence.
    # 1 = only relationships the matched entity is itself an endpoint of (the
    # flat-JSON store's only possible behavior). 2 additionally pulls evidence
    # from the relationships of directly-connected entities — something the
    # JSON store structurally cannot do.
    #
    # IMPORTANT, measured: setting this to 2 currently changes NOTHING for most
    # questions, and the reason is the interaction with _MAX_GRAPH_EVIDENCE in
    # retrieval_service/routes.py. Hop-2 ids are appended strictly AFTER every
    # hop-1 id (deliberately — a second-hop section must not displace a
    # directly-evidenced one), and routes.py then keeps only the first 20. A
    # typical entity-bearing question already produces far more than 20 hop-1
    # candidates ("fine for not wearing a helmet" produces 86), so the hop-2
    # tail is truncated away before it is ever scored.
    #
    # It can only matter where direct evidence is genuinely thin (<20
    # candidates) — which is arguably the only case where widening is wanted
    # anyway. Making it apply more broadly means reserving cap slots for hop-2,
    # which is a retrieval-ranking change that should be measured with
    # evaluation/run_eval.py rather than assumed to help.
    graph_expand_hops: int = 1

    # --- evaluation judge (offline harness only) -------------------------------
    # The LLM-as-a-judge used by evaluation/judge.py for the RAGAS metrics
    # (Es et al., 2024, arXiv:2309.15217). Deliberately NOT one of the models
    # under test: a model grading its own answers is not an evaluation. Nothing
    # in the live request path reads these — they exist for the offline sweep.
    judge_provider: Literal["gemini", "ollama"] = "gemini"
    judge_model: str = "gemini-3.5-flash-lite"
    # Floor between judge calls. A full sweep is ~27 questions x 4 models x 3
    # judge calls; free-tier rate limits bite long before that without a gap.
    judge_min_interval_seconds: float = 4.0

    # --- retrieval tuning ------------------------------------------------------
    default_top_k: int = 8  # was 5 — repeatedly found the correct section scoring just under a 5-slot cutoff
    request_timeout_seconds: float = 120.0


settings = Settings()
