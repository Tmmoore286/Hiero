import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.config import Settings
from hiero.embedding import EmbedderFactory, OpenAIEmbedder


class _FakeDatum:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeUsage:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, embeddings, total_tokens=0):
        self.data = [_FakeDatum(e) for e in embeddings]
        self.usage = _FakeUsage(total_tokens)


@pytest.mark.asyncio
async def test_openai_embedder_batches_and_normalizes(monkeypatch):
    embedder = OpenAIEmbedder(api_key="sk-test")

    async def fake_call(texts):
        assert texts == ["a", "b"]
        return _FakeResponse([[3.0, 4.0], [0.0, 0.0]], total_tokens=5)

    monkeypatch.setattr(embedder, "_embed_call", fake_call)

    result = await embedder.embed_batch(["a", "b"], batch_size=2, normalize=True)
    assert result.api_calls == 1
    assert result.total_tokens == 5
    assert len(result.results) == 2
    v0 = result.results[0].vector
    assert abs(sum(x * x for x in v0) - 1.0) < 1e-6


def test_embedder_factory_requires_openai_key():
    settings = Settings(openai_api_key=None)
    with pytest.raises(ValueError):
        EmbedderFactory(settings).create()

