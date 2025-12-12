# Component Spec: Vector Store (pgvector)

## Overview

The Vector Store component provides persistent storage and retrieval of embeddings using PostgreSQL with the pgvector extension. It supports both dense vector similarity search (HNSW index) and sparse full-text search (tsvector), enabling hybrid retrieval strategies.

## Requirements

### Functional
- **FR-1**: Store embeddings with metadata in PostgreSQL
- **FR-2**: HNSW index for approximate nearest neighbor search
- **FR-3**: Full-text search via tsvector for BM25-like sparse retrieval
- **FR-4**: Metadata filtering (by document ID, date range, custom fields)
- **FR-5**: Support multiple embedding dimensions (384 to 4096)
- **FR-6**: Batch insert/update operations
- **FR-7**: Document-level and chunk-level operations
- **FR-8**: Tenant isolation (namespace support for multi-tenant deployments)

### Non-Functional
- **NFR-1**: Vector search < 50ms for 1M vectors at 95th percentile
- **NFR-2**: Support 10M+ vectors per namespace
- **NFR-3**: Insert throughput > 1000 vectors/second
- **NFR-4**: 99.9% query availability

## Database Schema

### Tables

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table (stores original document metadata)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    content_hash VARCHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    source VARCHAR(32) NOT NULL,  -- file_upload, url, api, arcnet_sync
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_documents_namespace_hash UNIQUE (namespace, content_hash)
);

CREATE INDEX idx_documents_namespace ON documents(namespace);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);


-- Chunks table (stores text chunks with embeddings)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',

    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,

    -- Embedding (dimension specified per-row to support multiple models)
    embedding vector,  -- Will store vectors of any dimension
    embedding_model VARCHAR(128),
    embedding_dimensions INT,

    -- Full-text search
    content_tsvector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,

    -- Chunk metadata
    chunk_index INT NOT NULL,
    start_char INT,
    end_char INT,
    token_count INT,
    section_title VARCHAR(256),
    section_hierarchy TEXT[],
    strategy_used VARCHAR(32),

    -- Custom metadata (for filtering)
    metadata JSONB NOT NULL DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_chunks_doc_index UNIQUE (document_id, chunk_index)
);

-- HNSW index for vector similarity search
-- Create separate indexes for common dimensions
CREATE INDEX idx_chunks_embedding_1536 ON chunks
    USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
    WHERE embedding_dimensions = 1536;

CREATE INDEX idx_chunks_embedding_3072 ON chunks
    USING hnsw ((embedding::vector(3072)) vector_cosine_ops)
    WHERE embedding_dimensions = 3072;

CREATE INDEX idx_chunks_embedding_768 ON chunks
    USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WHERE embedding_dimensions = 768;

CREATE INDEX idx_chunks_embedding_384 ON chunks
    USING hnsw ((embedding::vector(384)) vector_cosine_ops)
    WHERE embedding_dimensions = 384;

-- Full-text search index
CREATE INDEX idx_chunks_content_tsvector ON chunks USING GIN (content_tsvector);

-- Metadata and filtering indexes
CREATE INDEX idx_chunks_namespace ON chunks(namespace);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_metadata ON chunks USING GIN (metadata);
CREATE INDEX idx_chunks_created_at ON chunks(created_at);


-- Namespaces table (for multi-tenant support)
CREATE TABLE namespaces (
    name VARCHAR(64) PRIMARY KEY,
    description TEXT,
    default_embedding_model VARCHAR(128) DEFAULT 'text-embedding-3-small',
    default_embedding_dimensions INT DEFAULT 1536,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default namespace
INSERT INTO namespaces (name, description)
VALUES ('default', 'Default namespace');


-- Vector search statistics (for monitoring)
CREATE TABLE vector_search_stats (
    id SERIAL PRIMARY KEY,
    namespace VARCHAR(64) NOT NULL,
    query_type VARCHAR(32) NOT NULL,  -- dense, sparse, hybrid
    latency_ms FLOAT NOT NULL,
    result_count INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_search_stats_namespace_time
    ON vector_search_stats(namespace, created_at DESC);
```

### Index Tuning

```sql
-- HNSW index parameters for 1M+ vectors
-- m: max connections per node (higher = better recall, more memory)
-- ef_construction: build-time search depth (higher = better index, slower build)

ALTER INDEX idx_chunks_embedding_1536
    SET (hnsw.m = 16, hnsw.ef_construction = 64);

-- Query-time parameter (higher = better recall, slower search)
SET hnsw.ef_search = 100;
```

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class DocumentModel(BaseModel):
    """Database model for documents."""
    id: UUID = Field(default_factory=uuid4)
    namespace: str = "default"
    content_hash: str
    metadata: dict = Field(default_factory=dict)
    source: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkModel(BaseModel):
    """Database model for chunks with embeddings."""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    namespace: str = "default"

    # Content
    content: str
    content_hash: str

    # Embedding
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None

    # Chunk metadata
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None
    token_count: int | None = None
    section_title: str | None = None
    section_hierarchy: list[str] = Field(default_factory=list)
    strategy_used: str | None = None

    # Custom metadata
    metadata: dict = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """Result from vector/text search."""
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float  # Similarity or relevance score
    metadata: dict
    section_title: str | None = None
    chunk_index: int


class SearchQuery(BaseModel):
    """Query parameters for search."""
    query_vector: list[float] | None = None  # For dense search
    query_text: str | None = None  # For sparse/hybrid search
    namespace: str = "default"
    top_k: int = 10
    score_threshold: float | None = None  # Minimum score filter
    metadata_filter: dict | None = None  # JSONB filter
    document_ids: list[UUID] | None = None  # Limit to specific docs
    include_content: bool = True
```

## Interfaces

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator


class VectorStoreProtocol(ABC):
    """Base protocol for vector storage."""

    @abstractmethod
    async def insert_document(
        self,
        document: DocumentModel,
        chunks: list[ChunkModel],
    ) -> UUID:
        """
        Insert a document with its chunks and embeddings.

        Args:
            document: Document metadata
            chunks: List of chunks with embeddings

        Returns:
            Document ID

        Raises:
            DuplicateDocumentError: If document hash already exists
        """
        ...

    @abstractmethod
    async def search_dense(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """
        Search by vector similarity (cosine).

        Args:
            query: Search parameters with query_vector

        Returns:
            List of SearchResult sorted by similarity
        """
        ...

    @abstractmethod
    async def search_sparse(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """
        Search by full-text (BM25-like).

        Args:
            query: Search parameters with query_text

        Returns:
            List of SearchResult sorted by relevance
        """
        ...

    @abstractmethod
    async def search_hybrid(
        self,
        query: SearchQuery,
        alpha: float = 0.7,  # Weight for dense vs sparse
    ) -> list[SearchResult]:
        """
        Combined dense + sparse search with score fusion.

        Args:
            query: Search parameters with both vector and text
            alpha: Weight for dense score (1-alpha for sparse)

        Returns:
            List of SearchResult with fused scores
        """
        ...

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document and all its chunks."""
        ...

    @abstractmethod
    async def get_document(self, document_id: UUID) -> DocumentModel | None:
        """Get document by ID."""
        ...

    @abstractmethod
    async def list_documents(
        self,
        namespace: str = "default",
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentModel]:
        """List documents in namespace."""
        ...
```

## Implementation

### SQLAlchemy Models

```python
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, ForeignKey,
    Index, text, func
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY, TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector


class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(PGUUID, primary_key=True, server_default=text("gen_random_uuid()"))
    namespace = Column(String(64), nullable=False, default="default")
    content_hash = Column(String(64), nullable=False)
    metadata = Column(JSONB, nullable=False, default={})
    source = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("ChunkORM", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_documents_namespace", "namespace"),
        Index("idx_documents_metadata", "metadata", postgresql_using="gin"),
    )


class ChunkORM(Base):
    __tablename__ = "chunks"

    id = Column(PGUUID, primary_key=True, server_default=text("gen_random_uuid()"))
    document_id = Column(PGUUID, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    namespace = Column(String(64), nullable=False, default="default")

    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)

    # Vector column - pgvector supports dynamic dimensions
    embedding = Column(Vector)
    embedding_model = Column(String(128))
    embedding_dimensions = Column(Integer)

    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer)
    end_char = Column(Integer)
    token_count = Column(Integer)
    section_title = Column(String(256))
    section_hierarchy = Column(ARRAY(Text))
    strategy_used = Column(String(32))

    metadata = Column(JSONB, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("DocumentORM", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_namespace", "namespace"),
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_metadata", "metadata", postgresql_using="gin"),
    )
```

### Repository Implementation

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete, and_, or_
from sqlalchemy.dialects.postgresql import insert
import numpy as np


class PgVectorStore(VectorStoreProtocol):
    """PostgreSQL + pgvector implementation."""

    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url)
        self.session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def insert_document(
        self,
        document: DocumentModel,
        chunks: list[ChunkModel],
    ) -> UUID:
        async with self.session_factory() as session:
            # Check for duplicate
            existing = await session.execute(
                select(DocumentORM).where(
                    DocumentORM.namespace == document.namespace,
                    DocumentORM.content_hash == document.content_hash,
                )
            )
            if existing.scalar_one_or_none():
                raise DuplicateDocumentError(
                    f"Document with hash {document.content_hash} exists"
                )

            # Insert document
            doc_orm = DocumentORM(
                id=document.id,
                namespace=document.namespace,
                content_hash=document.content_hash,
                metadata=document.metadata,
                source=document.source,
            )
            session.add(doc_orm)

            # Insert chunks
            for chunk in chunks:
                chunk_orm = ChunkORM(
                    id=chunk.id,
                    document_id=document.id,
                    namespace=document.namespace,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    embedding=chunk.embedding,
                    embedding_model=chunk.embedding_model,
                    embedding_dimensions=chunk.embedding_dimensions,
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    token_count=chunk.token_count,
                    section_title=chunk.section_title,
                    section_hierarchy=chunk.section_hierarchy,
                    strategy_used=chunk.strategy_used,
                    metadata=chunk.metadata,
                )
                session.add(chunk_orm)

            await session.commit()
            return document.id

    async def search_dense(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Vector similarity search using pgvector."""
        if not query.query_vector:
            raise ValueError("query_vector required for dense search")

        async with self.session_factory() as session:
            # Build base query with cosine distance
            # pgvector: <=> is cosine distance, lower is better
            # Convert to similarity: 1 - distance
            distance_expr = ChunkORM.embedding.cosine_distance(query.query_vector)

            stmt = (
                select(
                    ChunkORM,
                    (1 - distance_expr).label("similarity"),
                )
                .where(ChunkORM.namespace == query.namespace)
                .order_by(distance_expr)
                .limit(query.top_k)
            )

            # Apply filters
            stmt = self._apply_filters(stmt, query)

            result = await session.execute(stmt)
            rows = result.all()

            return [
                SearchResult(
                    chunk_id=row.ChunkORM.id,
                    document_id=row.ChunkORM.document_id,
                    content=row.ChunkORM.content if query.include_content else "",
                    score=float(row.similarity),
                    metadata=row.ChunkORM.metadata,
                    section_title=row.ChunkORM.section_title,
                    chunk_index=row.ChunkORM.chunk_index,
                )
                for row in rows
                if query.score_threshold is None or row.similarity >= query.score_threshold
            ]

    async def search_sparse(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Full-text search using PostgreSQL tsvector."""
        if not query.query_text:
            raise ValueError("query_text required for sparse search")

        async with self.session_factory() as session:
            # Convert query to tsquery
            tsquery = func.plainto_tsquery('english', query.query_text)

            # ts_rank for relevance scoring
            rank_expr = func.ts_rank(
                text("content_tsvector"),
                tsquery,
            )

            stmt = (
                select(
                    ChunkORM,
                    rank_expr.label("relevance"),
                )
                .where(
                    ChunkORM.namespace == query.namespace,
                    text("content_tsvector @@ plainto_tsquery('english', :query)")
                )
                .params(query=query.query_text)
                .order_by(rank_expr.desc())
                .limit(query.top_k)
            )

            stmt = self._apply_filters(stmt, query)

            result = await session.execute(stmt)
            rows = result.all()

            return [
                SearchResult(
                    chunk_id=row.ChunkORM.id,
                    document_id=row.ChunkORM.document_id,
                    content=row.ChunkORM.content if query.include_content else "",
                    score=float(row.relevance),
                    metadata=row.ChunkORM.metadata,
                    section_title=row.ChunkORM.section_title,
                    chunk_index=row.ChunkORM.chunk_index,
                )
                for row in rows
            ]

    async def search_hybrid(
        self,
        query: SearchQuery,
        alpha: float = 0.7,
    ) -> list[SearchResult]:
        """
        Hybrid search with Reciprocal Rank Fusion (RRF).

        Combines dense and sparse results using:
        RRF(d) = Σ 1 / (k + rank(d))

        where k is a constant (typically 60) and rank is position in result list.
        """
        if not query.query_vector or not query.query_text:
            raise ValueError("Both query_vector and query_text required for hybrid search")

        # Get results from both methods (fetch more for fusion)
        expanded_query = query.model_copy()
        expanded_query.top_k = query.top_k * 3

        dense_results = await self.search_dense(expanded_query)
        sparse_results = await self.search_sparse(expanded_query)

        # Build rank maps
        k = 60  # RRF constant
        chunk_scores: dict[UUID, float] = {}
        chunk_data: dict[UUID, SearchResult] = {}

        # Dense scores
        for rank, result in enumerate(dense_results):
            rrf_score = alpha / (k + rank + 1)
            chunk_scores[result.chunk_id] = chunk_scores.get(result.chunk_id, 0) + rrf_score
            chunk_data[result.chunk_id] = result

        # Sparse scores
        for rank, result in enumerate(sparse_results):
            rrf_score = (1 - alpha) / (k + rank + 1)
            chunk_scores[result.chunk_id] = chunk_scores.get(result.chunk_id, 0) + rrf_score
            if result.chunk_id not in chunk_data:
                chunk_data[result.chunk_id] = result

        # Sort by fused score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:query.top_k]

        return [
            SearchResult(
                chunk_id=chunk_id,
                document_id=chunk_data[chunk_id].document_id,
                content=chunk_data[chunk_id].content,
                score=score,
                metadata=chunk_data[chunk_id].metadata,
                section_title=chunk_data[chunk_id].section_title,
                chunk_index=chunk_data[chunk_id].chunk_index,
            )
            for chunk_id, score in sorted_chunks
        ]

    def _apply_filters(self, stmt, query: SearchQuery):
        """Apply metadata and document filters to query."""
        if query.document_ids:
            stmt = stmt.where(ChunkORM.document_id.in_(query.document_ids))

        if query.metadata_filter:
            # JSONB containment query
            stmt = stmt.where(
                ChunkORM.metadata.contains(query.metadata_filter)
            )

        return stmt

    async def delete_document(self, document_id: UUID) -> bool:
        async with self.session_factory() as session:
            result = await session.execute(
                delete(DocumentORM).where(DocumentORM.id == document_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def get_document(self, document_id: UUID) -> DocumentModel | None:
        async with self.session_factory() as session:
            result = await session.execute(
                select(DocumentORM).where(DocumentORM.id == document_id)
            )
            doc = result.scalar_one_or_none()

            if doc:
                return DocumentModel(
                    id=doc.id,
                    namespace=doc.namespace,
                    content_hash=doc.content_hash,
                    metadata=doc.metadata,
                    source=doc.source,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
            return None

    async def list_documents(
        self,
        namespace: str = "default",
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentModel]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(DocumentORM)
                .where(DocumentORM.namespace == namespace)
                .order_by(DocumentORM.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            docs = result.scalars().all()

            return [
                DocumentModel(
                    id=doc.id,
                    namespace=doc.namespace,
                    content_hash=doc.content_hash,
                    metadata=doc.metadata,
                    source=doc.source,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
                for doc in docs
            ]

    async def update_embeddings(
        self,
        document_id: UUID,
        embeddings: dict[int, list[float]],  # chunk_index -> embedding
        model: str,
        dimensions: int,
    ) -> int:
        """Update embeddings for existing chunks (re-embedding)."""
        async with self.session_factory() as session:
            updated = 0
            for chunk_index, embedding in embeddings.items():
                result = await session.execute(
                    ChunkORM.__table__.update()
                    .where(
                        ChunkORM.document_id == document_id,
                        ChunkORM.chunk_index == chunk_index,
                    )
                    .values(
                        embedding=embedding,
                        embedding_model=model,
                        embedding_dimensions=dimensions,
                    )
                )
                updated += result.rowcount

            await session.commit()
            return updated

    async def get_stats(self, namespace: str = "default") -> dict:
        """Get storage statistics for namespace."""
        async with self.session_factory() as session:
            doc_count = await session.execute(
                select(func.count()).select_from(DocumentORM)
                .where(DocumentORM.namespace == namespace)
            )
            chunk_count = await session.execute(
                select(func.count()).select_from(ChunkORM)
                .where(ChunkORM.namespace == namespace)
            )
            models = await session.execute(
                select(ChunkORM.embedding_model, func.count())
                .where(ChunkORM.namespace == namespace)
                .group_by(ChunkORM.embedding_model)
            )

            return {
                "namespace": namespace,
                "document_count": doc_count.scalar(),
                "chunk_count": chunk_count.scalar(),
                "embedding_models": dict(models.all()),
            }
```

### Batch Operations

```python
class BatchVectorStore(PgVectorStore):
    """Extended vector store with batch operations."""

    async def insert_documents_batch(
        self,
        documents: list[tuple[DocumentModel, list[ChunkModel]]],
        batch_size: int = 100,
    ) -> list[UUID]:
        """Insert multiple documents in batches."""
        inserted_ids = []

        async with self.session_factory() as session:
            for doc, chunks in documents:
                # Check duplicate
                existing = await session.execute(
                    select(DocumentORM.id).where(
                        DocumentORM.namespace == doc.namespace,
                        DocumentORM.content_hash == doc.content_hash,
                    )
                )
                if existing.scalar_one_or_none():
                    continue  # Skip duplicates in batch

                # Insert document
                doc_orm = DocumentORM(
                    id=doc.id,
                    namespace=doc.namespace,
                    content_hash=doc.content_hash,
                    metadata=doc.metadata,
                    source=doc.source,
                )
                session.add(doc_orm)

                # Batch insert chunks
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    chunk_orms = [
                        ChunkORM(
                            id=chunk.id,
                            document_id=doc.id,
                            namespace=doc.namespace,
                            content=chunk.content,
                            content_hash=chunk.content_hash,
                            embedding=chunk.embedding,
                            embedding_model=chunk.embedding_model,
                            embedding_dimensions=chunk.embedding_dimensions,
                            chunk_index=chunk.chunk_index,
                            metadata=chunk.metadata,
                        )
                        for chunk in batch
                    ]
                    session.add_all(chunk_orms)

                inserted_ids.append(doc.id)

            await session.commit()

        return inserted_ids

    async def search_multi_vector(
        self,
        query_vectors: list[list[float]],
        namespace: str = "default",
        top_k: int = 10,
    ) -> list[list[SearchResult]]:
        """Search with multiple query vectors (for query expansion)."""
        results = []
        for vector in query_vectors:
            query = SearchQuery(
                query_vector=vector,
                namespace=namespace,
                top_k=top_k,
            )
            results.append(await self.search_dense(query))
        return results
```

## Error Handling

```python
class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


class DuplicateDocumentError(VectorStoreError):
    """Document already exists."""
    pass


class DocumentNotFoundError(VectorStoreError):
    """Document not found."""
    pass


class DimensionMismatchError(VectorStoreError):
    """Query vector dimension doesn't match stored vectors."""
    pass


class NamespaceNotFoundError(VectorStoreError):
    """Namespace doesn't exist."""
    pass
```

## Migrations

Using Alembic for database migrations:

```python
# alembic/versions/001_initial_schema.py
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

def upgrade():
    # Enable pgvector
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("namespace", sa.String(64), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create chunks table
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="CASCADE")),
        sa.Column("namespace", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector()),
        sa.Column("embedding_model", sa.String(128)),
        sa.Column("embedding_dimensions", sa.Integer()),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        # ... other columns
    )

    # Create indexes
    op.execute("""
        CREATE INDEX idx_chunks_embedding_1536 ON chunks
        USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WHERE embedding_dimensions = 1536
    """)

def downgrade():
    op.drop_table("chunks")
    op.drop_table("documents")
```

## Dependencies

```toml
[project.dependencies]
sqlalchemy = {extras = ["asyncio"], version = "^2.0"}
asyncpg = "^0.29"
pgvector = "^0.2"
alembic = "^1.13"
```

## ARCnet Integration

The vector store provides the persistent storage ARCnet lacks:

```python
class ARCnetDoctrineStore(PgVectorStore):
    """Specialized store for ARCnet doctrine documents."""

    ARCNET_NAMESPACE = "arcnet_doctrine"

    async def store_doctrine(
        self,
        mos_code: str,
        doctrine_id: str,
        content: str,
        embedding: list[float],
        metadata: dict,
    ) -> UUID:
        """Store a doctrine document for ARCnet agent selection."""
        doc = DocumentModel(
            namespace=self.ARCNET_NAMESPACE,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            metadata={
                "mos_code": mos_code,
                "doctrine_id": doctrine_id,
                **metadata,
            },
            source="arcnet_sync",
        )

        chunk = ChunkModel(
            document_id=doc.id,
            namespace=self.ARCNET_NAMESPACE,
            content=content,
            content_hash=doc.content_hash,
            embedding=embedding,
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            chunk_index=0,
            metadata={"mos_code": mos_code},
        )

        return await self.insert_document(doc, [chunk])

    async def get_doctrine_centroid(self, mos_code: str) -> list[float] | None:
        """
        Compute centroid of all doctrine embeddings for an MOS.
        Mirrors ARCnet's EmbeddingService.computeCentroid().
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(ChunkORM.embedding)
                .where(
                    ChunkORM.namespace == self.ARCNET_NAMESPACE,
                    ChunkORM.metadata["mos_code"].astext == mos_code,
                )
            )
            embeddings = [row[0] for row in result.all()]

            if not embeddings:
                return None

            # Compute centroid (average)
            centroid = np.mean(embeddings, axis=0)
            # Normalize
            centroid = centroid / np.linalg.norm(centroid)
            return centroid.tolist()

    async def find_similar_agents(
        self,
        mission_embedding: list[float],
        threshold: float = 0.3,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Find agents similar to mission embedding.
        Mirrors ARCnet's EmbeddingService.findSimilarAgents().
        """
        query = SearchQuery(
            query_vector=mission_embedding,
            namespace=self.ARCNET_NAMESPACE,
            top_k=top_k,
            score_threshold=threshold,
        )

        results = await self.search_dense(query)

        return [
            {
                "mos_code": r.metadata.get("mos_code"),
                "similarity": r.score,
                "doctrine_id": r.metadata.get("doctrine_id"),
            }
            for r in results
        ]
```

## Testing Strategy

1. **Unit Tests**: CRUD operations with test database
2. **Integration Tests**: Full search pipeline with real pgvector
3. **Performance Tests**: Search latency at scale (1M vectors)
4. **Index Tests**: Verify HNSW index usage with EXPLAIN ANALYZE
5. **Edge Cases**:
   - Empty namespace
   - Zero-vector queries
   - Very high-dimensional vectors
   - Concurrent writes
