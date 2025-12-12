from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://hiero:hiero@localhost:5432/hiero"

    # Embedding providers
    openai_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    default_embedding_provider: str = "openai"
    default_embedding_model: str = "text-embedding-3-small"

    # LLM providers
    anthropic_api_key: SecretStr | None = None
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o"

    # Retrieval defaults
    default_top_k: int = 10
    rerank_top_k: int = 5
    hybrid_alpha: float = 0.7

    # Chunking defaults
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

