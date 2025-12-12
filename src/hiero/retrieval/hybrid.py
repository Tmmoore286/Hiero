from __future__ import annotations

import asyncio
from collections import defaultdict

from hiero.embedding.base import EmbedderProtocol
from hiero.storage.repository import PgVectorStore, SearchResult

from .base import (
    FusionMethod,
    RetrievedChunk,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    RetrieverProtocol,
    _Timer,
)


class HybridRetriever(RetrieverProtocol):
    RRF_K = 60

    def __init__(self, vector_store: PgVectorStore, embedder: EmbedderProtocol):
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        timer = _Timer()

        dense_results: list[SearchResult] = []
        sparse_results: list[SearchResult] = []

        embedding_time_ms = 0.0
        query_embedding = query.query_embedding
        if query.config.strategy in (RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID):
            if query_embedding is None:
                embed_timer = _Timer()
                embed_result = await self.embedder.embed(query.text)
                query_embedding = embed_result.vector
                embedding_time_ms = embed_timer.ms()

        tasks = []
        if query.config.strategy in (RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID):
            k = query.config.dense_top_k if query.config.strategy == RetrievalStrategy.HYBRID else query.config.top_k
            tasks.append(
                self.vector_store.search_dense(
                    namespace=query.namespace,
                    query_embedding=query_embedding or [],
                    top_k=k,
                    document_ids=query.config.document_ids,
                    metadata_filter=query.config.metadata_filter,
                )
            )
        if query.config.strategy in (RetrievalStrategy.SPARSE, RetrievalStrategy.HYBRID):
            k = query.config.sparse_top_k if query.config.strategy == RetrievalStrategy.HYBRID else query.config.top_k
            tasks.append(
                self.vector_store.search_sparse(
                    namespace=query.namespace,
                    query_text=query.text,
                    top_k=k,
                    document_ids=query.config.document_ids,
                    metadata_filter=query.config.metadata_filter,
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks)
            if query.config.strategy == RetrievalStrategy.DENSE:
                dense_results = results[0]
            elif query.config.strategy == RetrievalStrategy.SPARSE:
                sparse_results = results[0]
            else:
                dense_results, sparse_results = results

        if query.config.strategy == RetrievalStrategy.DENSE:
            chunks = self._wrap_dense(dense_results)
            return RetrievalResult(
                query=query.text,
                chunks=chunks[: query.config.top_k],
                total_candidates=len(chunks),
                strategy_used=RetrievalStrategy.DENSE,
                latency_ms=timer.ms(),
                dense_candidates=len(dense_results),
                query_embedding_time_ms=embedding_time_ms,
            )

        if query.config.strategy == RetrievalStrategy.SPARSE:
            chunks = self._wrap_sparse(sparse_results)
            return RetrievalResult(
                query=query.text,
                chunks=chunks[: query.config.top_k],
                total_candidates=len(chunks),
                strategy_used=RetrievalStrategy.SPARSE,
                latency_ms=timer.ms(),
                sparse_candidates=len(sparse_results),
            )

        fused = self._fuse(dense_results, sparse_results, query.config.fusion_method, query.config.dense_weight)
        return RetrievalResult(
            query=query.text,
            chunks=fused[: query.config.top_k],
            total_candidates=len(fused),
            strategy_used=RetrievalStrategy.HYBRID,
            latency_ms=timer.ms(),
            dense_candidates=len(dense_results),
            sparse_candidates=len(sparse_results),
            query_embedding_time_ms=embedding_time_ms,
        )

    async def retrieve_batch(self, queries: list[RetrievalQuery]) -> list[RetrievalResult]:
        return await asyncio.gather(*(self.retrieve(q) for q in queries))

    def _wrap_dense(self, results: list[SearchResult]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for rank, r in enumerate(results, start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=r.score,
                    rank=rank,
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    dense_score=r.score,
                    metadata=r.metadata,
                    section_title=r.section_title,
                    chunk_index=r.chunk_index,
                )
            )
        return chunks

    def _wrap_sparse(self, results: list[SearchResult]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for rank, r in enumerate(results, start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=r.score,
                    rank=rank,
                    retrieval_strategy=RetrievalStrategy.SPARSE,
                    sparse_score=r.score,
                    metadata=r.metadata,
                    section_title=r.section_title,
                    chunk_index=r.chunk_index,
                )
            )
        return chunks

    def _fuse(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
        method: FusionMethod,
        dense_weight: float,
    ) -> list[RetrievedChunk]:
        if method != FusionMethod.RRF and method != FusionMethod.WEIGHTED_SUM:
            method = FusionMethod.RRF

        dense_ranks = {r.chunk_id: i + 1 for i, r in enumerate(dense)}
        sparse_ranks = {r.chunk_id: i + 1 for i, r in enumerate(sparse)}

        dense_scores = {r.chunk_id: r.score for r in dense}
        sparse_scores = {r.chunk_id: r.score for r in sparse}

        fused_scores: dict = defaultdict(float)

        if method == FusionMethod.RRF:
            for cid, rank in dense_ranks.items():
                fused_scores[cid] += dense_weight * (1.0 / (self.RRF_K + rank))
            for cid, rank in sparse_ranks.items():
                fused_scores[cid] += (1.0 - dense_weight) * (1.0 / (self.RRF_K + rank))
        else:
            max_dense = max(dense_scores.values(), default=1.0)
            max_sparse = max(sparse_scores.values(), default=1.0)
            for cid, score in dense_scores.items():
                fused_scores[cid] += dense_weight * (score / max_dense)
            for cid, score in sparse_scores.items():
                fused_scores[cid] += (1.0 - dense_weight) * (score / max_sparse)

        all_ids = set(dense_ranks) | set(sparse_ranks)
        fused_list: list[RetrievedChunk] = []
        for cid in all_ids:
            d_rank = dense_ranks.get(cid)
            s_rank = sparse_ranks.get(cid)
            rank_val = min(r for r in [d_rank, s_rank] if r is not None)

            # Prefer dense content if present, else sparse
            dense_item = next((r for r in dense if r.chunk_id == cid), None)
            sparse_item = next((r for r in sparse if r.chunk_id == cid), None)
            item = dense_item or sparse_item
            if item is None:
                continue

            fused_list.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    content=item.content,
                    score=fused_scores[cid],
                    rank=rank_val,
                    retrieval_strategy=RetrievalStrategy.HYBRID,
                    dense_score=dense_scores.get(cid),
                    sparse_score=sparse_scores.get(cid),
                    metadata=item.metadata,
                    section_title=item.section_title,
                    chunk_index=item.chunk_index,
                )
            )

        fused_list.sort(key=lambda c: c.score, reverse=True)
        for i, c in enumerate(fused_list, start=1):
            c.rank = i
        return fused_list

