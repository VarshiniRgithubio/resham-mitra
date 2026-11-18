"""Central configuration, read from environment / .env.

Everything an experiment may vary (LLM, embedder, pipeline mode, safety method
and thresholds) lives here, so experiments change config, not code.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    # LLM
    llm_provider: str = "fake"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_base_url: str = "http://localhost:11434"  # for LLM_PROVIDER=ollama

    # Embeddings: "hash" (offline baseline) or "st" (sentence-transformers)
    embedder: str = "hash"
    embed_model: str = "BAAI/bge-m3"

    # Storage. QDRANT_URL may be ":memory:" or "path:<dir>" for no-Docker runs.
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "resham_mitra"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sericulture_chunks"

    # RAG
    retrieval_top_k: int = 5
    pipeline_mode: str = "direct"  # direct | translate

    # Safety. Thresholds are placeholders until tuned on the dev split.
    safety_method: str = "s1"  # none | s1 | s2 | s3 | s1+s2
    s1_threshold: float = 0.35
    s2_threshold: float = 0.8
    s3_samples: int = 3
    s3_threshold: float = 0.5
    officer_contact: str = "<OFFICER_CONTACT_TBD>"

    # Auth. CHANGE jwt_secret outside local dev (any long random string).
    jwt_secret: str = "dev-only-secret-change-me"
    token_days: int = 30
    max_login_attempts: int = 5
    lockout_minutes: int = 15


@lru_cache
def get_settings() -> Settings:
    return Settings()
