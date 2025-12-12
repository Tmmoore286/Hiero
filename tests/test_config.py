import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.default_embedding_provider == "openai"
    assert settings.default_embedding_model == "text-embedding-3-small"
    assert settings.default_llm_provider == "openai"
    assert settings.default_llm_model == "gpt-4o"
    assert settings.default_top_k == 10
    assert settings.rerank_top_k == 5
    assert settings.hybrid_alpha == 0.7
    assert settings.default_chunk_size == 512
    assert settings.default_chunk_overlap == 64


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/testdb")
    monkeypatch.setenv("DEFAULT_TOP_K", "3")
    settings = Settings()
    assert settings.database_url.endswith("/testdb")
    assert settings.default_top_k == 3
