from __future__ import annotations

import math
from time import perf_counter

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import (
    BatchEmbeddingResult,
    EMBEDDING_MODELS,
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
    ):
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {model_name}")
        self._model: EmbeddingModel = EMBEDDING_MODELS[model_name]
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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
        results: list[EmbeddingResult] = []
        total_tokens = 0
        api_calls = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            response = await self._embed_call(batch_texts)
            api_calls += 1

            vectors = [d.embedding for d in response.data]
            if normalize:
                vectors = [self._l2_normalize(v) for v in vectors]

            for text, vector in zip(batch_texts, vectors):
                results.append(EmbeddingResult.from_text(text, vector, self.model))

            usage = getattr(response, "usage", None)
            if usage and getattr(usage, "total_tokens", None):
                total_tokens += int(usage.total_tokens)

        elapsed_ms = (perf_counter() - start_time) * 1000
        return BatchEmbeddingResult(
            results=results,
            total_tokens=total_tokens,
            processing_time_ms=elapsed_ms,
            cache_hits=0,
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

