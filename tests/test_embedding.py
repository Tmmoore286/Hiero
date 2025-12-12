import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hashlib import sha256

from hiero.config import Settings
from hiero.embedding import EmbedderFactory, OpenAIEmbedder
from hiero.embedding.base import EmbeddingResult


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


@pytest.mark.asyncio
async def test_openai_embedder_uses_cache(monkeypatch):
    cached_text = "cached"
    miss_text = "miss"
    cached_hash = sha256(cached_text.encode("utf-8")).hexdigest()

    class _FakeCache:
        async def get(self, text_hash, model_id):
            return None

        async def get_batch(self, text_hashes, model_id):
            if cached_hash in text_hashes:
                return {
                    cached_hash: EmbeddingResult(
                        text_hash=cached_hash,
                        vector=[0.0, 1.0],
                        model_id=model_id,
                        dimensions=2,
                        cached=True,
                    )
                }
            return {}

        async def set(self, result):
            return None

    embedder = OpenAIEmbedder(api_key="sk-test", cache=_FakeCache())

    async def fake_call(texts):
        assert texts == [miss_text]
        return _FakeResponse([[1.0, 0.0]], total_tokens=3)

    monkeypatch.setattr(embedder, "_embed_call", fake_call)

    res = await embedder.embed_batch([cached_text, miss_text], normalize=False)
    assert res.cache_hits == 1
    assert res.api_calls == 1
    assert res.results[0].cached is True
    assert res.results[1].cached is False
