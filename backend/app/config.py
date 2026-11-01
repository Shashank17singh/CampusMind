"""Application settings, read only from environment variables and .env files."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Secrets have no usable default outside development."""

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    secret_key: str = "dev-only-secret-key-change-me-0123456789"  # noqa: S105

    database_url: str = "postgresql+psycopg://campusmind:campusmind@localhost:5432/campusmind"
    redis_url: str = "redis://localhost:6379/0"
    celery_eager: bool = False

    llm_provider: Literal["anthropic", "ollama"] = "ollama"
    llm_model: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = 120.0

    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 25
    chunk_size_tokens: int = Field(default=800, ge=50)
    chunk_overlap_tokens: int = Field(default=120, ge=0)
    ocr_min_chars: int = 50
    ocr_dpi: int = 300
    tesseract_cmd: str = ""
    header_footer_threshold: float = 0.6
    near_duplicate_threshold: float = 0.7

    vector_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_path: str = "./data/chroma"
    chroma_host: str = ""
    chroma_port: int = 8000
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = ""
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = False
    rrf_vector_weight: float = 1.0
    rrf_bm25_weight: float = 1.0

    rag_top_k: int = 6
    rag_similarity_threshold: float = 0.45
    rag_min_results: int = 3
    rag_context_tokens: int = 6000
    rag_cache_ttl_seconds: int = 3600
    rag_followups_enabled: bool = True

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = "CampusMind <no-reply@campusmind.local>"

    access_token_minutes: int = 15

    @property
    def max_upload_bytes(self) -> int:
        """Upload limit in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_llm_model(self) -> str:
        """The configured model, or the default for the selected provider."""
        if self.llm_model:
            return self.llm_model
        return "claude-opus-5-5" if self.llm_provider == "anthropic" else "llama3.2:1b"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        """Reject combinations that can only fail later, at a worse time."""
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.environment == "production":
            if self.secret_key.startswith("dev-only") or len(self.secret_key) < 32:
                raise ValueError("SECRET_KEY must be a real secret of 32+ characters in production")
            if self.llm_provider == "anthropic" and not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings object."""
    return Settings()
