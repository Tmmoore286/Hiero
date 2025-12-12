from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocumentORM(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    namespace: Mapped[str] = mapped_column(
        sa.String(64),
        nullable=False,
        server_default=text("'default'"),
    )
    content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    source: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
    )

    chunks: Mapped[list["ChunkORM"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkORM(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    namespace: Mapped[str] = mapped_column(
        sa.String(64),
        nullable=False,
        server_default=text("'default'"),
    )

    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(sa.String(128))
    embedding_dimensions: Mapped[int | None] = mapped_column(sa.Integer)

    content_tsvector: Mapped[Any] = mapped_column(
        TSVECTOR,
        sa.Computed("to_tsvector('english', content)", persisted=True),
    )

    chunk_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    start_char: Mapped[int | None] = mapped_column(sa.Integer)
    end_char: Mapped[int | None] = mapped_column(sa.Integer)
    token_count: Mapped[int | None] = mapped_column(sa.Integer)
    section_title: Mapped[str | None] = mapped_column(sa.String(256))
    section_hierarchy: Mapped[list[str] | None] = mapped_column(ARRAY(sa.Text))
    strategy_used: Mapped[str | None] = mapped_column(sa.String(32))

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )

    document: Mapped[DocumentORM] = relationship(back_populates="chunks")

    __table_args__ = (
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
    )


class NamespaceORM(Base):
    __tablename__ = "namespaces"

    name: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    description: Mapped[str | None] = mapped_column(sa.Text)
    default_embedding_model: Mapped[str | None] = mapped_column(
        sa.String(128), server_default=text("'text-embedding-3-small'")
    )
    default_embedding_dimensions: Mapped[int | None] = mapped_column(
        sa.Integer, server_default=text("1536")
    )
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )


class VectorSearchStatORM(Base):
    __tablename__ = "vector_search_stats"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    namespace: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    query_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    latency_ms: Mapped[float] = mapped_column(sa.Float, nullable=False)
    result_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )


class EmbeddingCacheORM(Base):
    __tablename__ = "embedding_cache"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    text_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    dimensions: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )
    access_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=text("0")
    )

    __table_args__ = (
        sa.UniqueConstraint("text_hash", "model_id", name="uq_embedding_cache_text_model"),
    )
