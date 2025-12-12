from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

import sqlalchemy as sa
from pydantic import BaseModel

from hiero.chunking import AdaptiveChunker, ChunkingConfig, Tokenizer
from hiero.config import Settings
from hiero.embedding import EmbedderFactory
from hiero.generation import GenerationConfig, GenerationRequest, GenerationResponse, GroundedGenerator, SourceContext
from hiero.ingestion import DocumentMetadata, IngestionRouter
from hiero.reranking import LLMReranker, RerankRequest, RerankerConfig
from hiero.retrieval import HybridRetriever, RetrievalConfig, RetrievalQuery, RetrievalResult
from hiero.storage import ChunkORM, DocumentORM, PgVectorStore


class QueryResponse(BaseModel):
    answer: str
    citations: list
    retrieval: RetrievalResult
    generation: GenerationResponse


class Hiero:
    def __init__(
        self,
        database_url: str,
        openai_api_key: str | None = None,
        cohere_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        namespace: str = "default",
        **kwargs,
    ):
        self.settings = Settings(
            database_url=database_url,
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
            anthropic_api_key=anthropic_api_key,
            **kwargs,
        )
        self.namespace = namespace
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        self.vector_store = PgVectorStore(self.settings.database_url)
        self.embedder = EmbedderFactory(self.settings).create()
        self.tokenizer = Tokenizer()
        self.chunker = AdaptiveChunker(self.tokenizer)
        self.ingestor = IngestionRouter()
        self.retriever = HybridRetriever(self.vector_store, self.embedder)

        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for generation")
        self.generator = GroundedGenerator(
            api_key=self.settings.openai_api_key.get_secret_value()
        )
        self.reranker = LLMReranker(
            api_key=self.settings.openai_api_key.get_secret_value()
        )

        self._initialized = True

    async def __aenter__(self) -> "Hiero":
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        if getattr(self, "vector_store", None):
            await self.vector_store.dispose()
        cache = getattr(self.embedder, "cache", None)
        if cache is not None and hasattr(cache, "dispose"):
            await cache.dispose()
        self._initialized = False

    async def ingest(
        self,
        source: str | Path | BinaryIO,
        metadata: dict | None = None,
        namespace: str | None = None,
        file_type: str | None = None,
    ) -> UUID:
        await self.initialize()
        namespace = namespace or self.namespace
        meta = DocumentMetadata(**metadata) if metadata else DocumentMetadata()

        if isinstance(source, (str, Path)):
            source_str = str(source)
            doc = await self.ingestor.ingest_path(source_str, meta)
        else:
            ft = file_type or meta.file_type
            if not ft:
                raise ValueError("file_type is required when ingesting from a file object")
            doc = await self.ingestor.ingest_file(source, ft, meta)

        # Idempotent ingest: return existing doc if already present.
        async with self.vector_store.db.session() as session:
            existing_id = (
                await session.execute(
                    sa.select(DocumentORM.id).where(
                        DocumentORM.namespace == namespace,
                        DocumentORM.content_hash == doc.content_hash,
                    )
                )
            ).scalar_one_or_none()
        if existing_id:
            return existing_id

        chunk_cfg = ChunkingConfig(
            target_chunk_size=self.settings.default_chunk_size,
            chunk_overlap=self.settings.default_chunk_overlap,
        )
        chunking_result = await self.chunker.chunk(doc, chunk_cfg)

        texts = [c.content for c in chunking_result.chunks]
        embeddings = await self.embedder.embed_batch(texts)

        document_orm = DocumentORM(
            id=doc.id,
            namespace=namespace,
            content_hash=doc.content_hash,
            metadata_=doc.metadata.model_dump(),
            source=doc.source.value,
        )

        chunk_orms: list[ChunkORM] = []
        for chunk, embed in zip(chunking_result.chunks, embeddings.results):
            chunk_hash = sha256(chunk.content.encode("utf-8")).hexdigest()
            chunk_orms.append(
                ChunkORM(
                    document_id=doc.id,
                    namespace=namespace,
                    content=chunk.content,
                    content_hash=chunk_hash,
                    embedding=embed.vector,
                    embedding_model=embed.model_id,
                    embedding_dimensions=embed.dimensions,
                    chunk_index=chunk.metadata.chunk_index,
                    start_char=chunk.metadata.start_char,
                    end_char=chunk.metadata.end_char,
                    token_count=chunk.metadata.token_count,
                    section_title=chunk.metadata.section_title,
                    section_hierarchy=chunk.metadata.section_hierarchy,
                    strategy_used=chunk.metadata.strategy_used.value,
                    metadata_={},
                )
            )

        await self.vector_store.insert_document(document_orm, chunk_orms)
        return doc.id

    async def retrieve(
        self,
        query: str,
        namespace: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        await self.initialize()
        rq = RetrievalQuery(
            text=query,
            namespace=namespace or self.namespace,
            config=config or RetrievalConfig(top_k=self.settings.default_top_k),
        )
        return await self.retriever.retrieve(rq)

    async def query(
        self,
        question: str,
        namespace: str | None = None,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> QueryResponse:
        await self.initialize()
        effective_retrieval_config = retrieval_config or RetrievalConfig(
            top_k=self.settings.default_top_k
        )
        retrieval = await self.retrieve(
            question,
            namespace=namespace,
            config=effective_retrieval_config,
        )

        context = [
            SourceContext(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                metadata=c.metadata,
                relevance_score=c.score,
            )
            for c in retrieval.chunks
        ]

        gen_request = GenerationRequest(
            query=question,
            context=context,
            config=generation_config or GenerationConfig(),
        )
        generation = await self.generator.generate(gen_request)

        if effective_retrieval_config.rerank_results:
            rerank_cfg = RerankerConfig(
                top_k=self.settings.rerank_top_k,
                candidates=effective_retrieval_config.rerank_candidates,
            )
            reranked = await self.reranker.rerank(
                RerankRequest(query=question, chunks=retrieval.chunks, config=rerank_cfg)
            )
            reranked_chunks = []
            by_id = {c.chunk_id: c for c in retrieval.chunks}
            for i, rc in enumerate(reranked.chunks, start=1):
                orig = by_id[rc.chunk_id]
                reranked_chunks.append(
                    orig.model_copy(update={"score": rc.reranked_score, "rank": i})
                )
            retrieval = retrieval.model_copy(update={"chunks": reranked_chunks})

        return QueryResponse(
            answer=generation.response,
            citations=generation.citations,
            retrieval=retrieval,
            generation=generation,
        )
