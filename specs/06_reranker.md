# Component Spec: LLM-based Reranker

## Overview

The Reranker takes retrieval candidates and reorders them using an LLM to score relevance more accurately. It provides higher precision than embedding-only retrieval by considering query-document interactions in context.

## Requirements

### Functional
- **FR-1**: Rerank retrieved chunks by relevance to query
- **FR-2**: Support multiple LLM providers (OpenAI, Anthropic)
- **FR-3**: Batch processing for efficiency
- **FR-4**: Configurable reranking prompts
- **FR-5**: Score calibration (consistent 0-1 scale)
- **FR-6**: Explanation/reasoning for rankings (optional)
- **FR-7**: Fallback to original ranking on failure

### Non-Functional
- **NFR-1**: Rerank 20 candidates in < 3 seconds
- **NFR-2**: Cost-efficient (minimize tokens)
- **NFR-3**: Graceful degradation if LLM unavailable

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID


class RerankingMethod(str, Enum):
    POINTWISE = "pointwise"    # Score each doc independently
    PAIRWISE = "pairwise"      # Compare pairs of docs
    LISTWISE = "listwise"      # Rank full list at once


class RerankerConfig(BaseModel):
    """Configuration for reranking behavior."""
    method: RerankingMethod = RerankingMethod.POINTWISE
    top_k: int = 5              # Final number to return
    candidates: int = 20        # Number of candidates to rerank
    include_reasoning: bool = False
    temperature: float = 0.0    # Deterministic scoring
    max_retries: int = 2


class RerankRequest(BaseModel):
    """Request to rerank chunks."""
    query: str
    chunks: list[RetrievedChunk]
    config: RerankerConfig = Field(default_factory=RerankerConfig)


class RerankedChunk(BaseModel):
    """A chunk after reranking."""
    chunk_id: UUID
    document_id: UUID
    content: str

    # Scores
    original_rank: int
    reranked_rank: int
    original_score: float
    reranked_score: float

    # Optional reasoning
    reasoning: str | None = None

    # Metadata
    metadata: dict = Field(default_factory=dict)


class RerankResult(BaseModel):
    """Complete result from reranking."""
    query: str
    chunks: list[RerankedChunk]
    method_used: RerankingMethod
    latency_ms: float
    tokens_used: int
    rank_changes: int  # How many positions changed
```

## Interfaces

```python
from abc import ABC, abstractmethod


class RerankerProtocol(ABC):
    """Base protocol for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResult:
        """
        Rerank chunks by relevance to query.

        Args:
            request: RerankRequest with query and chunks

        Returns:
            RerankResult with reordered chunks
        """
        ...
```

## Implementation Details

### LLM Reranker

```python
import asyncio
import json
import time
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


class LLMReranker(RerankerProtocol):
    """
    LLM-based reranking using pointwise, pairwise, or listwise methods.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = None,
    ):
        self.provider = provider
        self.model = model

        if provider == "openai":
            self.client = AsyncOpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = AsyncAnthropic(api_key=api_key)

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResult:
        start_time = time.perf_counter()

        # Limit candidates
        candidates = request.chunks[:request.config.candidates]

        # Choose method
        if request.config.method == RerankingMethod.POINTWISE:
            scored_chunks, tokens = await self._rerank_pointwise(
                request.query, candidates, request.config
            )
        elif request.config.method == RerankingMethod.PAIRWISE:
            scored_chunks, tokens = await self._rerank_pairwise(
                request.query, candidates, request.config
            )
        elif request.config.method == RerankingMethod.LISTWISE:
            scored_chunks, tokens = await self._rerank_listwise(
                request.query, candidates, request.config
            )

        # Sort by reranked score
        scored_chunks.sort(key=lambda x: x.reranked_score, reverse=True)

        # Assign new ranks and calculate changes
        rank_changes = 0
        for i, chunk in enumerate(scored_chunks):
            chunk.reranked_rank = i + 1
            if chunk.reranked_rank != chunk.original_rank:
                rank_changes += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RerankResult(
            query=request.query,
            chunks=scored_chunks[:request.config.top_k],
            method_used=request.config.method,
            latency_ms=elapsed_ms,
            tokens_used=tokens,
            rank_changes=rank_changes,
        )

    async def _rerank_pointwise(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        config: RerankerConfig,
    ) -> tuple[list[RerankedChunk], int]:
        """
        Score each chunk independently.

        Pros: Parallelizable, consistent scoring
        Cons: Doesn't consider relative comparisons
        """
        total_tokens = 0

        async def score_chunk(chunk: RetrievedChunk, idx: int) -> RerankedChunk:
            nonlocal total_tokens

            prompt = self._build_pointwise_prompt(query, chunk.content, config)

            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    max_tokens=150 if config.include_reasoning else 50,
                )
                result_text = response.choices[0].message.content
                total_tokens += response.usage.total_tokens

            elif self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=150 if config.include_reasoning else 50,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = response.content[0].text
                total_tokens += response.usage.input_tokens + response.usage.output_tokens

            # Parse response
            score, reasoning = self._parse_pointwise_response(result_text, config)

            return RerankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                original_rank=idx + 1,
                reranked_rank=0,  # Set later
                original_score=chunk.score,
                reranked_score=score,
                reasoning=reasoning,
                metadata=chunk.metadata,
            )

        # Score all chunks in parallel
        tasks = [score_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        scored = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle failures - keep original score
        results = []
        for i, result in enumerate(scored):
            if isinstance(result, Exception):
                results.append(RerankedChunk(
                    chunk_id=chunks[i].chunk_id,
                    document_id=chunks[i].document_id,
                    content=chunks[i].content,
                    original_rank=i + 1,
                    reranked_rank=0,
                    original_score=chunks[i].score,
                    reranked_score=chunks[i].score,  # Fallback
                    metadata=chunks[i].metadata,
                ))
            else:
                results.append(result)

        return results, total_tokens

    async def _rerank_pairwise(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        config: RerankerConfig,
    ) -> tuple[list[RerankedChunk], int]:
        """
        Compare pairs of chunks to determine relative ranking.

        Uses a tournament-style comparison.
        Pros: Better relative judgments
        Cons: O(n^2) comparisons in worst case
        """
        total_tokens = 0
        n = len(chunks)

        # Win counts for each chunk
        wins = {chunk.chunk_id: 0 for chunk in chunks}

        # Compare top chunks more thoroughly
        # Use Swiss-system tournament style
        comparisons = []
        for i in range(min(n, 10)):
            for j in range(i + 1, min(n, 10)):
                comparisons.append((i, j))

        async def compare_pair(i: int, j: int) -> tuple[int, int, int]:
            """Compare chunks i and j, return winner index and tokens used."""
            prompt = self._build_pairwise_prompt(
                query,
                chunks[i].content,
                chunks[j].content,
            )

            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    max_tokens=20,
                )
                result = response.choices[0].message.content.strip()
                tokens = response.usage.total_tokens
            else:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=20,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.content[0].text.strip()
                tokens = response.usage.input_tokens + response.usage.output_tokens

            # Parse: expect "A" or "B"
            if "A" in result.upper():
                return i, j, tokens
            else:
                return j, i, tokens

        # Run comparisons in parallel (batched)
        for batch_start in range(0, len(comparisons), 10):
            batch = comparisons[batch_start:batch_start + 10]
            tasks = [compare_pair(i, j) for i, j in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if not isinstance(result, Exception):
                    winner_idx, loser_idx, tokens = result
                    wins[chunks[winner_idx].chunk_id] += 1
                    total_tokens += tokens

        # Convert to RerankedChunk with win-based scores
        max_wins = max(wins.values()) if wins.values() else 1
        results = [
            RerankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                original_rank=i + 1,
                reranked_rank=0,
                original_score=chunk.score,
                reranked_score=wins[chunk.chunk_id] / max_wins,
                metadata=chunk.metadata,
            )
            for i, chunk in enumerate(chunks)
        ]

        return results, total_tokens

    async def _rerank_listwise(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        config: RerankerConfig,
    ) -> tuple[list[RerankedChunk], int]:
        """
        Rank all chunks at once in a single LLM call.

        Pros: Most efficient, considers full context
        Cons: Limited by context window, less consistent
        """
        prompt = self._build_listwise_prompt(query, chunks, config)

        if self.provider == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=200,
            )
            result_text = response.choices[0].message.content
            tokens = response.usage.total_tokens
        else:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

        # Parse ranking
        ranking = self._parse_listwise_response(result_text, len(chunks))

        # Map back to chunks
        results = []
        for i, chunk in enumerate(chunks):
            # Find this chunk's new rank
            new_rank = ranking.get(i + 1, i + 1)  # Default to original
            score = 1.0 - (new_rank - 1) / len(chunks)  # Convert rank to score

            results.append(RerankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                original_rank=i + 1,
                reranked_rank=0,  # Set later after sorting
                original_score=chunk.score,
                reranked_score=score,
                metadata=chunk.metadata,
            ))

        return results, tokens

    def _build_pointwise_prompt(
        self,
        query: str,
        content: str,
        config: RerankerConfig,
    ) -> str:
        """Build prompt for pointwise scoring."""
        base_prompt = f"""Rate how relevant this passage is to answering the query.

Query: {query}

Passage: {content[:1500]}

Rate the relevance from 0 to 10, where:
- 0: Completely irrelevant
- 5: Somewhat relevant, partially answers the query
- 10: Highly relevant, directly answers the query

"""
        if config.include_reasoning:
            base_prompt += """Respond in this format:
Score: [0-10]
Reasoning: [Brief explanation]"""
        else:
            base_prompt += "Respond with only the numeric score."

        return base_prompt

    def _build_pairwise_prompt(
        self,
        query: str,
        content_a: str,
        content_b: str,
    ) -> str:
        """Build prompt for pairwise comparison."""
        return f"""Given this query, which passage is MORE relevant?

Query: {query}

Passage A: {content_a[:800]}

Passage B: {content_b[:800]}

Which passage is more relevant to the query? Answer with only "A" or "B"."""

    def _build_listwise_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        config: RerankerConfig,
    ) -> str:
        """Build prompt for listwise ranking."""
        passages = "\n\n".join([
            f"[{i+1}] {chunk.content[:500]}"
            for i, chunk in enumerate(chunks[:10])  # Limit for context
        ])

        return f"""Rank these passages by relevance to the query.

Query: {query}

Passages:
{passages}

Return the passage numbers in order of relevance, most relevant first.
Format: 1, 2, 3, ... (just the numbers, comma-separated)"""

    def _parse_pointwise_response(
        self,
        response: str,
        config: RerankerConfig,
    ) -> tuple[float, str | None]:
        """Parse pointwise scoring response."""
        reasoning = None

        if config.include_reasoning:
            # Parse structured response
            lines = response.strip().split('\n')
            score_line = next((l for l in lines if 'score' in l.lower()), lines[0])
            reasoning_line = next((l for l in lines if 'reasoning' in l.lower()), None)

            # Extract score
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', score_line)
            score = float(numbers[0]) / 10 if numbers else 0.5

            if reasoning_line:
                reasoning = reasoning_line.split(':', 1)[-1].strip()
        else:
            # Just a number
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            score = float(numbers[0]) / 10 if numbers else 0.5

        return min(max(score, 0), 1), reasoning

    def _parse_listwise_response(
        self,
        response: str,
        num_chunks: int,
    ) -> dict[int, int]:
        """Parse listwise ranking response. Returns {original_idx: new_rank}"""
        import re

        # Extract numbers from response
        numbers = re.findall(r'\d+', response)
        numbers = [int(n) for n in numbers if 1 <= int(n) <= num_chunks]

        # Remove duplicates while preserving order
        seen = set()
        ranking = []
        for n in numbers:
            if n not in seen:
                seen.add(n)
                ranking.append(n)

        # Map original index to new rank
        result = {}
        for new_rank, orig_idx in enumerate(ranking, 1):
            result[orig_idx] = new_rank

        return result
```

### Calibrated Reranker

```python
class CalibratedReranker(RerankerProtocol):
    """
    Wraps an LLM reranker with score calibration.

    Ensures consistent scoring across different queries
    by normalizing based on calibration examples.
    """

    def __init__(
        self,
        base_reranker: LLMReranker,
        calibration_queries: list[tuple[str, list[str], list[float]]] = None,
    ):
        self.base = base_reranker
        self.calibration = calibration_queries or []
        self._calibration_offset = 0.0
        self._calibration_scale = 1.0

    async def calibrate(self):
        """Run calibration to determine score adjustment."""
        if not self.calibration:
            return

        expected_scores = []
        actual_scores = []

        for query, docs, expected in self.calibration:
            chunks = [
                RetrievedChunk(
                    chunk_id=uuid4(),
                    document_id=uuid4(),
                    content=doc,
                    score=0.5,
                    rank=i+1,
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    metadata={},
                    chunk_index=i,
                )
                for i, doc in enumerate(docs)
            ]

            result = await self.base.rerank(RerankRequest(
                query=query,
                chunks=chunks,
            ))

            expected_scores.extend(expected)
            actual_scores.extend([c.reranked_score for c in result.chunks])

        # Linear regression for calibration
        import numpy as np
        A = np.vstack([actual_scores, np.ones(len(actual_scores))]).T
        self._calibration_scale, self._calibration_offset = np.linalg.lstsq(
            A, expected_scores, rcond=None
        )[0]

    async def rerank(self, request: RerankRequest) -> RerankResult:
        result = await self.base.rerank(request)

        # Apply calibration
        for chunk in result.chunks:
            chunk.reranked_score = (
                chunk.reranked_score * self._calibration_scale
                + self._calibration_offset
            )
            chunk.reranked_score = min(max(chunk.reranked_score, 0), 1)

        return result
```

### Ensemble Reranker

```python
class EnsembleReranker(RerankerProtocol):
    """
    Combines multiple rerankers for more robust results.
    """

    def __init__(
        self,
        rerankers: list[RerankerProtocol],
        weights: list[float] | None = None,
    ):
        self.rerankers = rerankers
        self.weights = weights or [1.0 / len(rerankers)] * len(rerankers)

    async def rerank(self, request: RerankRequest) -> RerankResult:
        import time
        start_time = time.perf_counter()

        # Run all rerankers in parallel
        results = await asyncio.gather(*[
            r.rerank(request) for r in self.rerankers
        ])

        # Combine scores
        chunk_scores: dict[UUID, float] = {}
        chunk_data: dict[UUID, RerankedChunk] = {}

        for result, weight in zip(results, self.weights):
            for chunk in result.chunks:
                if chunk.chunk_id not in chunk_scores:
                    chunk_scores[chunk.chunk_id] = 0
                    chunk_data[chunk.chunk_id] = chunk

                chunk_scores[chunk.chunk_id] += chunk.reranked_score * weight

        # Sort by combined score
        sorted_chunks = sorted(
            chunk_data.values(),
            key=lambda c: chunk_scores[c.chunk_id],
            reverse=True,
        )

        for i, chunk in enumerate(sorted_chunks):
            chunk.reranked_score = chunk_scores[chunk.chunk_id]
            chunk.reranked_rank = i + 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RerankResult(
            query=request.query,
            chunks=sorted_chunks[:request.config.top_k],
            method_used=RerankingMethod.POINTWISE,  # Ensemble
            latency_ms=elapsed_ms,
            tokens_used=sum(r.tokens_used for r in results),
            rank_changes=sum(r.rank_changes for r in results) // len(results),
        )
```

## Reranking Methods Comparison

| Method | Latency | Cost | Quality | Best For |
|--------|---------|------|---------|----------|
| Pointwise | Medium | Medium | Good | General use, parallelizable |
| Pairwise | High | High | Better | High precision needs |
| Listwise | Low | Low | Variable | Fast reranking, cost-sensitive |

## Cost Optimization

```python
class CostAwareReranker(RerankerProtocol):
    """
    Adjusts reranking strategy based on cost budget.
    """

    def __init__(
        self,
        reranker: LLMReranker,
        max_cost_per_query: float = 0.01,  # $0.01 per query
    ):
        self.reranker = reranker
        self.max_cost = max_cost_per_query

        # Approximate costs per method (GPT-4o-mini)
        self.cost_estimates = {
            RerankingMethod.POINTWISE: 0.0002,  # per chunk
            RerankingMethod.PAIRWISE: 0.0005,   # per comparison
            RerankingMethod.LISTWISE: 0.001,    # fixed
        }

    async def rerank(self, request: RerankRequest) -> RerankResult:
        n = len(request.chunks)

        # Estimate costs
        pointwise_cost = n * self.cost_estimates[RerankingMethod.POINTWISE]
        pairwise_cost = (n * (n-1) / 2) * self.cost_estimates[RerankingMethod.PAIRWISE]
        listwise_cost = self.cost_estimates[RerankingMethod.LISTWISE]

        # Choose method within budget
        if listwise_cost <= self.max_cost and n <= 10:
            request.config.method = RerankingMethod.LISTWISE
        elif pointwise_cost <= self.max_cost:
            request.config.method = RerankingMethod.POINTWISE
        else:
            # Reduce candidates to fit budget
            max_chunks = int(self.max_cost / self.cost_estimates[RerankingMethod.POINTWISE])
            request.chunks = request.chunks[:max_chunks]
            request.config.method = RerankingMethod.POINTWISE

        return await self.reranker.rerank(request)
```

## Dependencies

```toml
[project.dependencies]
openai = "^1.14"
anthropic = "^0.21"
numpy = "^1.26"
```

## Testing Strategy

1. **Unit Tests**: Each reranking method with mocked LLM
2. **Integration Tests**: Real LLM calls with test queries
3. **Quality Tests**: Compare reranked vs original ordering
4. **Calibration Tests**: Verify score consistency
5. **Cost Tests**: Token usage tracking
