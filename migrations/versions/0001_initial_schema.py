"""Initial Hiero schema

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2025-12-12
"""

from __future__ import annotations

from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            namespace VARCHAR(64) NOT NULL DEFAULT 'default',
            content_hash VARCHAR(64) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            source VARCHAR(32) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_documents_namespace_hash UNIQUE (namespace, content_hash)
        );
        """
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            namespace VARCHAR(64) NOT NULL DEFAULT 'default',

            content TEXT NOT NULL,
            content_hash VARCHAR(64) NOT NULL,

            embedding vector,
            embedding_model VARCHAR(128),
            embedding_dimensions INT,

            content_tsvector TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('english', content)
            ) STORED,

            chunk_index INT NOT NULL,
            start_char INT,
            end_char INT,
            token_count INT,
            section_title VARCHAR(256),
            section_hierarchy TEXT[],
            strategy_used VARCHAR(32),

            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT uq_chunks_doc_index UNIQUE (document_id, chunk_index)
        );
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_1536 ON chunks
            USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
            WHERE embedding_dimensions = 1536;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_3072 ON chunks
            USING hnsw ((embedding::vector(3072)) vector_cosine_ops)
            WHERE embedding_dimensions = 3072;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_768 ON chunks
            USING hnsw ((embedding::vector(768)) vector_cosine_ops)
            WHERE embedding_dimensions = 768;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_384 ON chunks
            USING hnsw ((embedding::vector(384)) vector_cosine_ops)
            WHERE embedding_dimensions = 384;
        """
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_tsvector ON chunks USING GIN (content_tsvector);"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_namespace ON chunks(namespace);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING GIN (metadata);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at);")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS namespaces (
            name VARCHAR(64) PRIMARY KEY,
            description TEXT,
            default_embedding_model VARCHAR(128) DEFAULT 'text-embedding-3-small',
            default_embedding_dimensions INT DEFAULT 1536,
            settings JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    op.execute(
        """
        INSERT INTO namespaces (name, description)
        VALUES ('default', 'Default namespace')
        ON CONFLICT (name) DO NOTHING;
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_search_stats (
            id SERIAL PRIMARY KEY,
            namespace VARCHAR(64) NOT NULL,
            query_type VARCHAR(32) NOT NULL,
            latency_ms FLOAT NOT NULL,
            result_count INT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_search_stats_namespace_time
            ON vector_search_stats(namespace, created_at DESC);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS vector_search_stats;")
    op.execute("DROP TABLE IF EXISTS namespaces;")
    op.execute("DROP TABLE IF EXISTS chunks;")
    op.execute("DROP TABLE IF EXISTS documents;")
    op.execute("DROP EXTENSION IF EXISTS vector;")
    op.execute("DROP EXTENSION IF EXISTS pgcrypto;")

