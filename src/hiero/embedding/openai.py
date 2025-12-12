from __future__ import annotations

import math
from hashlib import sha256
from time import perf_counter

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import (
    BatchEmbeddingResult,
    EMBEDDING_MODELS,
    EmbeddingCache,
    EmbeddingError,
    EmbeddingModel,
    EmbeddingResult,
    EmbedderProtocol,
)


class OpenAIEmbedder(EmbedderProtocol):
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        base_url: str | None = None,
        cache: EmbeddingCache | None = None,
    ):
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {model_name}")
        self._model: EmbeddingModel = EMBEDDING_MODELS[model_name]
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.cache = cache

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    async def embed(self, text: str) -> EmbeddingResult:
        batch = await self.embed_batch([text], batch_size=1)
        return batch.results[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int = 100, normalize: bool = True
    ) -> BatchEmbeddingResult:
        start_time = perf_counter()
        total_tokens = 0
        api_calls = 0
        cache_hits = 0

        text_hashes = [sha256(t.encode("utf-8")).hexdigest() for t in texts]
        cached: dict[str, EmbeddingResult] = {}
        if self.cache:
            cached = await self.cache.get_batch(text_hashes, self.model.model_id)
            cache_hits = len(cached)

        results_by_hash: dict[str, EmbeddingResult] = dict(cached)
        missing_texts = [
            (t, h) for t, h in zip(texts, text_hashes) if h not in cached
        ]

        for i in range(0, len(missing_texts), batch_size):
            batch_texts = [t for t, _ in missing_texts[i : i + batch_size]]
            response = await self._embed_call(batch_texts)
            api_calls += 1

            vectors = [d.embedding for d in response.data]
            if normalize:
                vectors = [self._l2_normalize(v) for v in vectors]

            for text, vector in zip(batch_texts, vectors):
                result = EmbeddingResult.from_text(text, vector, self.model)
                results_by_hash[result.text_hash] = result
                if self.cache:
                    await self.cache.set(result)

            usage = getattr(response, "usage", None)
            if usage and getattr(usage, "total_tokens", None):
                total_tokens += int(usage.total_tokens)

        ordered_results = [results_by_hash[h] for h in text_hashes]
        elapsed_ms = (perf_counter() - start_time) * 1000
        return BatchEmbeddingResult(
            results=ordered_results,
            total_tokens=total_tokens,
            processing_time_ms=elapsed_ms,
            cache_hits=cache_hits,
            api_calls=api_calls,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def _embed_call(self, texts: list[str]):
        try:
            return await self.client.embeddings.create(model=self.model.model_name, input=texts)
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc

    def _l2_normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
