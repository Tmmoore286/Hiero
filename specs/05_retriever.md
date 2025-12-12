# Component Spec: Hybrid Retriever

## Overview

The Hybrid Retriever orchestrates multiple retrieval strategies (dense vector, sparse BM25, hybrid fusion) and provides a unified interface for querying the vector store. It handles query preprocessing, multi-stage retrieval, and result aggregation.

## Requirements

### Functional
- **FR-1**: Support dense (vector) retrieval
- **FR-2**: Support sparse (BM25/full-text) retrieval
- **FR-3**: Support hybrid retrieval with configurable fusion
- **FR-4**: Query preprocessing (expansion, reformulation)
- **FR-5**: Multi-hop retrieval for complex queries
- **FR-6**: Metadata filtering during retrieval
- **FR-7**: Deduplication of results across strategies
- **FR-8**: Configurable retrieval pipelines

### Non-Functional
- **NFR-1**: Retrieval latency < 100ms at 95th percentile
- **NFR-2**: Support concurrent queries (100+ QPS)
- **NFR-3**: Graceful degradation if one strategy fails

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID


class RetrievalStrategy(str, Enum):
    DENSE = "dense"           # Vector similarity
    SPARSE = "sparse"         # BM25 / full-text
    HYBRID = "hybrid"         # Combined dense + sparse
    MULTI_HOP = "multi_hop"   # Iterative retrieval


class FusionMethod(str, Enum):
    RRF = "rrf"                    # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # Linear combination of scores
    DBSF = "dbsf"                  # Distribution-Based Score Fusion


class RetrievalConfig(BaseModel):
    """Configuration for retrieval behavior."""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10

    # Hybrid settings
    dense_weight: float = 0.7  # Alpha for hybrid (1-alpha for sparse)
    fusion_method: FusionMethod = FusionMethod.RRF

    # Retrieval limits
    dense_top_k: int = 50    # Candidates from dense search
    sparse_top_k: int = 50   # Candidates from sparse search

    # Multi-hop settings
    max_hops: int = 3
    hop_top_k: int = 5

    # Filtering
    score_threshold: float | None = None
    metadata_filter: dict | None = None
    document_ids: list[UUID] | None = None

    # Query preprocessing
    expand_query: bool = False  # Use LLM to expand query
    use_hyde: bool = False      # Hypothetical Document Embedding


class RetrievalQuery(BaseModel):
    """A query to the retriever."""
    text: str
    namespace: str = "default"
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)

    # Optional: pre-computed query embedding
    query_embedding: list[float] | None = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk with relevance info."""
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    rank: int

    # Source tracking
    retrieval_strategy: RetrievalStrategy
    dense_score: float | None = None
    sparse_score: float | None = None

    # Metadata
    metadata: dict = Field(default_factory=dict)
    section_title: str | None = None
    chunk_index: int


class RetrievalResult(BaseModel):
    """Complete result from retrieval."""
    query: str
    chunks: list[RetrievedChunk]
    total_candidates: int
    strategy_used: RetrievalStrategy
    latency_ms: float

    # Debugging info
    dense_candidates: int = 0
    sparse_candidates: int = 0
    query_embedding_time_ms: float = 0
```

## Interfaces

```python
from abc import ABC, abstractmethod


class RetrieverProtocol(ABC):
    """Base protocol for retrievers."""

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: RetrievalQuery with text and config

        Returns:
            RetrievalResult with ranked chunks
        """
        ...

    @abstractmethod
    async def retrieve_batch(
        self,
        queries: list[RetrievalQuery],
    ) -> list[RetrievalResult]:
        """Retrieve for multiple queries concurrently."""
        ...


class QueryPreprocessor(ABC):
    """Preprocesses queries before retrieval."""

    @abstractmethod
    async def preprocess(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[str]:
        """
        Preprocess query, potentially returning multiple variants.

        Returns:
            List of query strings (original + expansions)
        """
        ...
```

## Implementation Details

### Core Retriever

```python
import asyncio
import time
from typing import Callable


class HybridRetriever(RetrieverProtocol):
    """
    Main retriever supporting dense, sparse, and hybrid strategies.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        embedder: EmbedderProtocol,
        preprocessor: QueryPreprocessor | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.preprocessor = preprocessor or DefaultPreprocessor()

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> RetrievalResult:
        start_time = time.perf_counter()

        # Preprocess query
        query_variants = await self.preprocessor.preprocess(
            query.text, query.config
        )

        # Get query embedding if needed
        embedding_time = 0
        if query.config.strategy in (RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID):
            if query.query_embedding:
                query_embedding = query.query_embedding
            else:
                embed_start = time.perf_counter()
                embed_result = await self.embedder.embed(query.text)
                query_embedding = embed_result.vector
                embedding_time = (time.perf_counter() - embed_start) * 1000

        # Execute retrieval based on strategy
        if query.config.strategy == RetrievalStrategy.DENSE:
            chunks, candidates = await self._retrieve_dense(
                query_embedding, query
            )
            dense_candidates = candidates
            sparse_candidates = 0

        elif query.config.strategy == RetrievalStrategy.SPARSE:
            chunks, candidates = await self._retrieve_sparse(
                query_variants, query
            )
            dense_candidates = 0
            sparse_candidates = candidates

        elif query.config.strategy == RetrievalStrategy.HYBRID:
            chunks, dense_cand, sparse_cand = await self._retrieve_hybrid(
                query_embedding, query_variants, query
            )
            dense_candidates = dense_cand
            sparse_candidates = sparse_cand

        elif query.config.strategy == RetrievalStrategy.MULTI_HOP:
            chunks = await self._retrieve_multi_hop(
                query_embedding, query
            )
            dense_candidates = len(chunks) * query.config.max_hops
            sparse_candidates = 0

        # Apply score threshold
        if query.config.score_threshold:
            chunks = [
                c for c in chunks
                if c.score >= query.config.score_threshold
            ]

        # Deduplicate and rank
        chunks = self._deduplicate(chunks)
        chunks = chunks[:query.config.top_k]

        # Assign final ranks
        for i, chunk in enumerate(chunks):
            chunk.rank = i + 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query.text,
            chunks=chunks,
            total_candidates=dense_candidates + sparse_candidates,
            strategy_used=query.config.strategy,
            latency_ms=elapsed_ms,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            query_embedding_time_ms=embedding_time,
        )

    async def _retrieve_dense(
        self,
        query_embedding: list[float],
        query: RetrievalQuery,
    ) -> tuple[list[RetrievedChunk], int]:
        """Pure vector similarity search."""
        search_query = SearchQuery(
            query_vector=query_embedding,
            namespace=query.namespace,
            top_k=query.config.dense_top_k,
            metadata_filter=query.config.metadata_filter,
            document_ids=query.config.document_ids,
        )

        results = await self.vector_store.search_dense(search_query)

        chunks = [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                rank=0,
                retrieval_strategy=RetrievalStrategy.DENSE,
                dense_score=r.score,
                metadata=r.metadata,
                section_title=r.section_title,
                chunk_index=r.chunk_index,
            )
            for r in results
        ]

        return chunks, len(results)

    async def _retrieve_sparse(
        self,
        query_variants: list[str],
        query: RetrievalQuery,
    ) -> tuple[list[RetrievedChunk], int]:
        """Full-text / BM25 search."""
        # Search with all query variants
        all_results = []

        for variant in query_variants:
            search_query = SearchQuery(
                query_text=variant,
                namespace=query.namespace,
                top_k=query.config.sparse_top_k,
                metadata_filter=query.config.metadata_filter,
                document_ids=query.config.document_ids,
            )
            results = await self.vector_store.search_sparse(search_query)
            all_results.extend(results)

        # Deduplicate by chunk_id, keeping highest score
        seen = {}
        for r in all_results:
            if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                seen[r.chunk_id] = r

        results = sorted(seen.values(), key=lambda x: x.score, reverse=True)

        chunks = [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                rank=0,
                retrieval_strategy=RetrievalStrategy.SPARSE,
                sparse_score=r.score,
                metadata=r.metadata,
                section_title=r.section_title,
                chunk_index=r.chunk_index,
            )
            for r in results[:query.config.sparse_top_k]
        ]

        return chunks, len(results)

    async def _retrieve_hybrid(
        self,
        query_embedding: list[float],
        query_variants: list[str],
        query: RetrievalQuery,
    ) -> tuple[list[RetrievedChunk], int, int]:
        """Combined dense + sparse with fusion."""
        # Run both in parallel
        dense_task = self._retrieve_dense(query_embedding, query)
        sparse_task = self._retrieve_sparse(query_variants, query)

        (dense_chunks, dense_count), (sparse_chunks, sparse_count) = await asyncio.gather(
            dense_task, sparse_task
        )

        # Fuse results
        if query.config.fusion_method == FusionMethod.RRF:
            fused = self._fuse_rrf(
                dense_chunks, sparse_chunks,
                query.config.dense_weight
            )
        elif query.config.fusion_method == FusionMethod.WEIGHTED_SUM:
            fused = self._fuse_weighted_sum(
                dense_chunks, sparse_chunks,
                query.config.dense_weight
            )
        elif query.config.fusion_method == FusionMethod.DBSF:
            fused = self._fuse_dbsf(dense_chunks, sparse_chunks)

        # Update strategy
        for chunk in fused:
            chunk.retrieval_strategy = RetrievalStrategy.HYBRID

        return fused, dense_count, sparse_count

    def _fuse_rrf(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_chunks: list[RetrievedChunk],
        alpha: float,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion.

        RRF(d) = Σ 1 / (k + rank(d))

        Alpha weights the contribution of each source.
        """
        chunk_scores: dict[UUID, float] = {}
        chunk_data: dict[UUID, RetrievedChunk] = {}

        # Dense scores
        for rank, chunk in enumerate(dense_chunks):
            rrf_score = alpha / (k + rank + 1)
            chunk_scores[chunk.chunk_id] = chunk_scores.get(chunk.chunk_id, 0) + rrf_score
            chunk_data[chunk.chunk_id] = chunk

        # Sparse scores
        for rank, chunk in enumerate(sparse_chunks):
            rrf_score = (1 - alpha) / (k + rank + 1)
            chunk_scores[chunk.chunk_id] = chunk_scores.get(chunk.chunk_id, 0) + rrf_score

            if chunk.chunk_id in chunk_data:
                # Merge scores
                chunk_data[chunk.chunk_id].sparse_score = chunk.sparse_score
            else:
                chunk_data[chunk.chunk_id] = chunk

        # Sort by fused score
        sorted_ids = sorted(
            chunk_scores.keys(),
            key=lambda cid: chunk_scores[cid],
            reverse=True,
        )

        result = []
        for cid in sorted_ids:
            chunk = chunk_data[cid]
            chunk.score = chunk_scores[cid]
            result.append(chunk)

        return result

    def _fuse_weighted_sum(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_chunks: list[RetrievedChunk],
        alpha: float,
    ) -> list[RetrievedChunk]:
        """
        Linear combination of normalized scores.

        score = alpha * norm_dense + (1-alpha) * norm_sparse
        """
        # Normalize scores to [0, 1]
        def normalize(chunks: list[RetrievedChunk]) -> dict[UUID, float]:
            if not chunks:
                return {}
            scores = [c.score for c in chunks]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1
            return {
                c.chunk_id: (c.score - min_s) / range_s
                for c in chunks
            }

        dense_norm = normalize(dense_chunks)
        sparse_norm = normalize(sparse_chunks)

        # Combine
        all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        chunk_data = {c.chunk_id: c for c in dense_chunks + sparse_chunks}

        results = []
        for cid in all_ids:
            d_score = dense_norm.get(cid, 0)
            s_score = sparse_norm.get(cid, 0)
            combined = alpha * d_score + (1 - alpha) * s_score

            chunk = chunk_data[cid]
            chunk.score = combined
            results.append(chunk)

        return sorted(results, key=lambda c: c.score, reverse=True)

    def _fuse_dbsf(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Distribution-Based Score Fusion.

        Normalizes scores based on their statistical distribution,
        then combines. More robust to score scale differences.
        """
        import numpy as np

        def normalize_dbsf(chunks: list[RetrievedChunk]) -> dict[UUID, float]:
            if not chunks:
                return {}
            scores = np.array([c.score for c in chunks])
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                std = 1
            # Z-score normalization, then sigmoid to [0, 1]
            z_scores = (scores - mean) / std
            normalized = 1 / (1 + np.exp(-z_scores))
            return {
                c.chunk_id: float(normalized[i])
                for i, c in enumerate(chunks)
            }

        dense_norm = normalize_dbsf(dense_chunks)
        sparse_norm = normalize_dbsf(sparse_chunks)

        # Average the normalized scores
        all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        chunk_data = {c.chunk_id: c for c in dense_chunks + sparse_chunks}

        results = []
        for cid in all_ids:
            scores = []
            if cid in dense_norm:
                scores.append(dense_norm[cid])
            if cid in sparse_norm:
                scores.append(sparse_norm[cid])

            chunk = chunk_data[cid]
            chunk.score = sum(scores) / len(scores)
            results.append(chunk)

        return sorted(results, key=lambda c: c.score, reverse=True)

    async def _retrieve_multi_hop(
        self,
        initial_embedding: list[float],
        query: RetrievalQuery,
    ) -> list[RetrievedChunk]:
        """
        Multi-hop retrieval for complex queries.

        Iteratively retrieves, then uses retrieved content
        to find additional relevant chunks.
        """
        all_chunks = []
        seen_ids = set()
        current_embedding = initial_embedding

        for hop in range(query.config.max_hops):
            # Retrieve
            search_query = SearchQuery(
                query_vector=current_embedding,
                namespace=query.namespace,
                top_k=query.config.hop_top_k,
                metadata_filter=query.config.metadata_filter,
            )
            results = await self.vector_store.search_dense(search_query)

            # Filter already seen
            new_results = [r for r in results if r.chunk_id not in seen_ids]

            if not new_results:
                break

            # Add to results
            for r in new_results:
                chunk = RetrievedChunk(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=r.score * (0.9 ** hop),  # Decay by hop
                    rank=0,
                    retrieval_strategy=RetrievalStrategy.MULTI_HOP,
                    dense_score=r.score,
                    metadata={**r.metadata, "hop": hop},
                    section_title=r.section_title,
                    chunk_index=r.chunk_index,
                )
                all_chunks.append(chunk)
                seen_ids.add(r.chunk_id)

            # Generate new query embedding from retrieved content
            # Combine original query with retrieved content
            combined_text = query.text + " " + " ".join(
                r.content[:200] for r in new_results[:3]
            )
            embed_result = await self.embedder.embed(combined_text)
            current_embedding = embed_result.vector

        # Sort by score
        return sorted(all_chunks, key=lambda c: c.score, reverse=True)

    def _deduplicate(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Remove duplicate chunks, keeping highest score."""
        seen = {}
        for chunk in chunks:
            if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
                seen[chunk.chunk_id] = chunk
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    async def retrieve_batch(
        self,
        queries: list[RetrievalQuery],
    ) -> list[RetrievalResult]:
        """Retrieve for multiple queries concurrently."""
        return await asyncio.gather(*[
            self.retrieve(q) for q in queries
        ])
```

### Query Preprocessor

```python
class DefaultPreprocessor(QueryPreprocessor):
    """Basic query preprocessing."""

    async def preprocess(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[str]:
        # Just return original query
        return [query]


class LLMQueryExpander(QueryPreprocessor):
    """Expands queries using an LLM."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def preprocess(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[str]:
        if not config.expand_query:
            return [query]

        # Generate query expansions
        expansion_prompt = f"""Given this search query, generate 3 alternative phrasings
that would help find relevant documents. Return only the alternatives, one per line.

Query: {query}

Alternatives:"""

        response = await self.llm.complete(expansion_prompt, max_tokens=150)

        expansions = [
            line.strip()
            for line in response.split('\n')
            if line.strip() and not line.strip().startswith('-')
        ]

        return [query] + expansions[:3]


class HyDEPreprocessor(QueryPreprocessor):
    """
    Hypothetical Document Embedding (HyDE).

    Generates a hypothetical answer, then embeds that
    for retrieval instead of the raw query.
    """

    def __init__(self, llm_client, embedder: EmbedderProtocol):
        self.llm = llm_client
        self.embedder = embedder

    async def preprocess(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[str]:
        if not config.use_hyde:
            return [query]

        # Generate hypothetical document
        hyde_prompt = f"""Write a short paragraph that would be a perfect answer
to this question. Write as if you're quoting from an authoritative source.

Question: {query}

Answer paragraph:"""

        hypothetical_doc = await self.llm.complete(hyde_prompt, max_tokens=200)

        # Return both original and hypothetical for retrieval
        return [query, hypothetical_doc.strip()]
```

### Retrieval Pipeline Builder

```python
class RetrievalPipeline:
    """
    Configurable retrieval pipeline with stages.
    """

    def __init__(self):
        self.stages: list[Callable] = []

    def add_stage(self, stage: Callable) -> "RetrievalPipeline":
        """Add a processing stage."""
        self.stages.append(stage)
        return self

    async def execute(
        self,
        query: RetrievalQuery,
        retriever: RetrieverProtocol,
    ) -> RetrievalResult:
        """Execute the pipeline."""
        result = await retriever.retrieve(query)

        for stage in self.stages:
            result = await stage(result)

        return result


# Example stages
async def filter_by_recency(result: RetrievalResult) -> RetrievalResult:
    """Filter to only recent documents."""
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=30)

    result.chunks = [
        c for c in result.chunks
        if c.metadata.get('created_at', datetime.min) > cutoff
    ]
    return result


async def boost_section_matches(result: RetrievalResult) -> RetrievalResult:
    """Boost chunks from relevant sections."""
    priority_sections = ['overview', 'summary', 'introduction']

    for chunk in result.chunks:
        if chunk.section_title and chunk.section_title.lower() in priority_sections:
            chunk.score *= 1.2

    result.chunks.sort(key=lambda c: c.score, reverse=True)
    return result


# Usage
pipeline = (
    RetrievalPipeline()
    .add_stage(filter_by_recency)
    .add_stage(boost_section_matches)
)
```

## Retrieval Strategies Comparison

| Strategy | Strengths | Weaknesses | Best For |
|----------|-----------|------------|----------|
| Dense | Semantic similarity, handles synonyms | Misses exact keyword matches | Conceptual queries |
| Sparse | Exact matches, interpretable | Misses semantic similarity | Keyword queries, names |
| Hybrid | Best of both | More compute, tuning needed | General use |
| Multi-hop | Complex reasoning | Slower, may drift | Multi-part questions |

## Dependencies

```toml
[project.dependencies]
numpy = "^1.26"
```

## ARCnet Integration

The retriever supports ARCnet's agent selection workflow:

```python
class ARCnetRetriever(HybridRetriever):
    """Retriever optimized for ARCnet doctrine lookup."""

    async def find_relevant_doctrine(
        self,
        mission_text: str,
        mos_codes: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find doctrine relevant to a mission.
        Used for agent context injection.
        """
        # Build metadata filter for MOS codes
        metadata_filter = None
        if mos_codes:
            metadata_filter = {"mos_code": {"$in": mos_codes}}

        query = RetrievalQuery(
            text=mission_text,
            namespace="arcnet_doctrine",
            config=RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID,
                top_k=10,
                dense_weight=0.6,  # Slightly favor semantic for doctrine
                metadata_filter=metadata_filter,
            ),
        )

        result = await self.retrieve(query)
        return result.chunks
```

## Testing Strategy

1. **Unit Tests**: Each retrieval strategy in isolation
2. **Integration Tests**: Full pipeline with real vector store
3. **Fusion Tests**: Verify RRF, weighted sum, DBSF produce expected rankings
4. **Performance Tests**: Latency benchmarks at scale
5. **Quality Tests**: Measure Recall@k, nDCG on test datasets
