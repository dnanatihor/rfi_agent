from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://rfi:rfi@127.0.0.1:5433/rfi"

    llm_provider: str = "openai_compatible"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str | None = None

    embedding_provider: str = "openai_compatible"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str | None = None
    embedding_dim: int = 1536

    rerank_provider: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""
    rerank_base_url: str | None = None

    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/oauth/google/callback"

    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    frontend_origin: str = "http://localhost:3000"
    data_dir: Path = _REPO_ROOT / "data"
    fetch_allowlist: str = ""

    ask_wall_seconds: float = 45
    ask_max_steps: int = 8
    ask_max_input_tokens: int = 80_000
    ask_max_cost_usd: float = 0
    ask_usd_per_million_input: float = 0.15
    ask_usd_per_million_output: float = 0.60
    batch_question_wall_seconds: float = 30
    batch_question_max_steps: int = 4
    trace_retention_days: int = 90

    default_org_id: str = "org_local"
    default_org_name: str = "Local"
    sitemap_max_pages: int = 5000
    max_upload_bytes: int = 50 * 1024 * 1024
    max_project_bytes: int = 500 * 1024 * 1024

    memory_accept: float = 0.88
    doc_min_score: float = 0.15
    memory_ttl_days: int = 365
    session_index_retention_days: int = 30

    questionnaire_max_questions: int = 500
    questionnaire_plan_threshold: int = 20
    questionnaire_wall_seconds: int = 1800
    questionnaire_max_input_tokens: int = 4_000_000
    parser_version: str = "qparse-1"

    def resolve_llm_key(self) -> str:
        import os

        if self.llm_api_key.strip():
            return self.llm_api_key.strip()
        if self.llm_provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if self.llm_provider == "openai_compatible":
            return os.environ.get("OPENAI_API_KEY", "").strip()
        return ""

    def resolve_embedding_key(self) -> str:
        import os

        if self.embedding_api_key.strip():
            return self.embedding_api_key.strip()
        if self.resolve_llm_key():
            return self.resolve_llm_key()
        return os.environ.get("OPENAI_API_KEY", "").strip()

    def resolve_rerank_key(self) -> str:
        import os

        if self.rerank_api_key.strip():
            return self.rerank_api_key.strip()
        if self.resolve_llm_key():
            return self.resolve_llm_key()
        return os.environ.get("COHERE_API_KEY", os.environ.get("JINA_API_KEY", "")).strip()

    def llm_ready(self) -> bool:
        if self.llm_provider == "bedrock":
            return True
        if self.llm_base_url and self.llm_base_url.strip():
            return True
        return bool(self.resolve_llm_key())

    def embeddings_ready(self) -> bool:
        if self.embedding_provider == "bedrock":
            return True
        if (self.embedding_base_url or self.llm_base_url or "").strip():
            return True
        return bool(self.resolve_embedding_key())

    def rerank_ready(self) -> bool:
        return bool(self.rerank_provider.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
