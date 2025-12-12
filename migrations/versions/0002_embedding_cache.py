"""Embedding cache table

Revision ID: 0002_embedding_cache
Revises: 0001_initial_schema
Create Date: 2025-12-12
"""

from __future__ import annotations

from alembic import op

revision = "0002_embedding_cache"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            text_hash VARCHAR(64) NOT NULL,
            model_id VARCHAR(128) NOT NULL,
            embedding vector NOT NULL,
            dimensions INT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            access_count INT NOT NULL DEFAULT 0,
            CONSTRAINT uq_embedding_cache_text_model UNIQUE (text_hash, model_id)
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_cache_text_model ON embedding_cache(text_hash, model_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_cache;")

