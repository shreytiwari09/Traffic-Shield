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

    # --- retrieval tuning ------------------------------------------------------
    default_top_k: int = 8  # was 5 — repeatedly found the correct section scoring just under a 5-slot cutoff
    request_timeout_seconds: float = 120.0


settings = Settings()
