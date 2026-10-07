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

    # Whether this machine has Ollama at all (services/shared/ollama_probe.py).
    # "auto": ping it (cached for ollama_probe_ttl_seconds) and only use it if
    # it answers — so a laptop without Ollama skips it instead of every request
    # hanging until ollama_timeout_seconds. "always": assume it is there without
    # pinging (the original behaviour; what the test suite runs with). "never":
    # ignore Ollama even if it is installed.
    ollama_mode: Literal["auto", "always", "never"] = "auto"
    ollama_probe_ttl_seconds: float = 30.0

    # Question embeddings when Ollama is not available. These are the SAME
    # nomic-embed-text v1.5 weights Ollama serves, run in-process via ONNX
    # (fastembed) — measured cosine 1.0000 against the Ollama vectors stored in
    # DATA/chunks.jsonl, so the existing Chroma index needs no rebuild. A
    # different model here would put questions in a different vector space
    # from the index and silently break retrieval.
    local_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    # Outside the repo and outside /tmp: the ONNX weights are ~550 MB, and the
    # library's default (a temp dir) would re-download them after every reboot.
    local_embedding_cache_dir: Path = Path.home() / ".cache" / "fastembed"

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
    # Was gemini-3.5-flash-lite — the same model as the Gemini contestant, i.e.
    # the model grading its own answers (Run 1's judged scores carry that bias).
    # A stronger judge is better still, but on the free tier gemini-3.5-flash is
    # capped at 20 requests/day, far below a ~170-call judged sweep.
    judge_model: str = "gemini-3.1-flash-lite"
    # Floor between judge calls. A full sweep is ~27 questions x 4 models x 3
    # judge calls; free-tier rate limits bite long before that without a gap.
    judge_min_interval_seconds: float = 4.0

    # --- LLMOps: prompt registry + canary routing (services/shared/registry.py) --
    # Stable prompt every request gets unless it lands in the canary bucket.
    stable_prompt_version: str = "legal-persona-v3"
    # Candidate prompt and the share of Ask traffic (0-100) routed to it. 0 =
    # canary off (the default, so nothing changes unless deliberately enabled).
    canary_prompt_version: str = "legal-persona-v4-strict-amounts"
    canary_percent: int = 0

    # --- LLMOps: token cost accounting ------------------------------------------
    # USD per 1M tokens. Local Ollama models cost 0 in API terms. Gemini prices
    # are ESTIMATES for a flash-lite-class model — set them to your plan's real
    # pricing in .env; the cost panel is only as right as these two numbers.
    gemini_input_usd_per_1m: float = 0.10
    gemini_output_usd_per_1m: float = 0.40

    # --- LLMOps: tracing ----------------------------------------------------------
    # OTLP/gRPC endpoint of the trace backend (Arize Phoenix, see
    # docker-compose.monitoring.yml), e.g. http://localhost:4317. Empty = tracing
    # disabled (spans are no-ops) — what tests and CI run with.
    otel_exporter_otlp_endpoint: str = ""

    # --- LLMOps: user feedback -----------------------------------------------------
    feedback_path: Path = PROJECT_ROOT / "feedback" / "feedback.jsonl"

    # --- retrieval tuning ------------------------------------------------------
    default_top_k: int = 8  # was 5 — repeatedly found the correct section scoring just under a 5-slot cutoff
    request_timeout_seconds: float = 120.0


settings = Settings()
