from __future__ import annotations

import asyncio
import json
import re

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import (
    RerankRequest,
    RerankResult,
    RerankerConfig,
    RerankerProtocol,
    RerankedChunk,
    RerankingMethod,
    _Timer,
)


class LLMReranker(RerankerProtocol):
    SCORE_RE = re.compile(r"([01](?:\.\d+)?)")

    def __init__(
        self,
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        if provider != "openai":
            raise ValueError(f"Unsupported reranker provider for Phase 2: {provider}")
        self.provider = provider
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def rerank(self, request: RerankRequest) -> RerankResult:
        timer = _Timer()
        candidates = request.chunks[: request.config.candidates]

        if not candidates:
            return RerankResult(
                query=request.query,
                chunks=[],
                method_used=request.config.method,
                latency_ms=timer.ms(),
                tokens_used=0,
                rank_changes=0,
            )

        if request.config.method != RerankingMethod.POINTWISE:
            raise NotImplementedError("Only pointwise reranking is supported in Phase 2.")

        scored, tokens = await self._rerank_pointwise(
            request.query, candidates, request.config
        )
        scored.sort(key=lambda c: c.reranked_score, reverse=True)

        rank_changes = 0
        for i, c in enumerate(scored, start=1):
            c.reranked_rank = i
            if c.reranked_rank != c.original_rank:
                rank_changes += 1

        return RerankResult(
            query=request.query,
            chunks=scored[: request.config.top_k],
            method_used=request.config.method,
            latency_ms=timer.ms(),
            tokens_used=tokens,
            rank_changes=rank_changes,
        )

    async def _rerank_pointwise(
        self,
        query: str,
        chunks: list,
        config: RerankerConfig,
    ) -> tuple[list[RerankedChunk], int]:
        total_tokens = 0

        async def score_one(idx: int, chunk):
            nonlocal total_tokens
            score, reasoning, tokens = await self._score_chunk(query, chunk.content, config)
            total_tokens += tokens
            return RerankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                original_rank=chunk.rank,
                reranked_rank=chunk.rank,
                original_score=chunk.score,
                reranked_score=score,
                reasoning=reasoning,
                metadata=chunk.metadata or {},
            )

        scored = await asyncio.gather(*(score_one(i, c) for i, c in enumerate(chunks)))
        return list(scored), total_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def _score_chunk(
        self, query: str, content: str, config: RerankerConfig
    ) -> tuple[float, str | None, int]:
        prompt = self._build_prompt(query, content, config)
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=150 if config.include_reasoning else 50,
        )

        text = completion.choices[0].message.content or ""
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0) or 0

        score, reasoning = self._parse_score(text, config.include_reasoning)
        return score, reasoning, int(tokens)

    def _build_prompt(self, query: str, content: str, config: RerankerConfig) -> str:
        base = (
            "You are a reranker. Given a user query and a candidate chunk, "
            "produce a relevance score between 0 and 1.\n\n"
            f"Query:\n{query}\n\nChunk:\n{content}\n\n"
        )
        if config.include_reasoning:
            base += (
                "Respond in JSON: {\"score\": <float 0-1>, \"reasoning\": \"...\"}"
            )
        else:
            base += "Respond in JSON: {\"score\": <float 0-1>}"
        return base

    def _parse_score(self, text: str, include_reasoning: bool) -> tuple[float, str | None]:
        try:
            data = json.loads(text)
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning") if include_reasoning else None
            return max(0.0, min(1.0, score)), reasoning
        except Exception:
            m = self.SCORE_RE.search(text)
            if not m:
                return 0.0, None
            score = float(m.group(1))
            return max(0.0, min(1.0, score)), None

