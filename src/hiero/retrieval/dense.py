from __future__ import annotations

import asyncio

from hiero.embedding.base import EmbedderProtocol
from hiero.storage.repository import PgVectorStore

from .base import RetrievedChunk, RetrievalQuery, RetrievalResult, RetrievalStrategy, RetrieverProtocol, _Timer


class DenseRetriever(RetrieverProtocol):
    def __init__(self, vector_store: PgVectorStore, embedder: EmbedderProtocol):
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        timer = _Timer()

        embed_timer = _Timer()
        if query.query_embedding is None:
            embed_result = await self.embedder.embed(query.text)
            query_embedding = embed_result.vector
        else:
            query_embedding = query.query_embedding
        embedding_time_ms = embed_timer.ms()

        search_results = await self.vector_store.search_dense(
            namespace=query.namespace,
            query_embedding=query_embedding,
            top_k=query.config.top_k,
            document_ids=query.config.document_ids,
            metadata_filter=query.config.metadata_filter,
        )

        chunks: list[RetrievedChunk] = []
        for rank, result in enumerate(search_results, start=1):
            chunks.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    score=result.score,
                    rank=rank,
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    dense_score=result.score,
                    metadata=result.metadata,
                    section_title=result.section_title,
                    chunk_index=result.chunk_index,
                )
            )

        return RetrievalResult(
            query=query.text,
            chunks=chunks,
            total_candidates=len(search_results),
            strategy_used=RetrievalStrategy.DENSE,
            latency_ms=timer.ms(),
            dense_candidates=len(search_results),
            query_embedding_time_ms=embedding_time_ms,
        )

    async def retrieve_batch(self, queries: list[RetrievalQuery]) -> list[RetrievalResult]:
        return await asyncio.gather(*(self.retrieve(q) for q in queries))

