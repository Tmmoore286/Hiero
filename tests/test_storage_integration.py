import os
import sys
import uuid

import pytest
import sqlalchemy as sa

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

TEST_URL = os.getenv("HIERO_TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
if not TEST_URL:
    pytest.skip(
        "Set HIERO_TEST_DATABASE_URL to run storage integration tests",
        allow_module_level=True,
    )

from hiero.storage.models import Base, ChunkORM, DocumentORM
from hiero.storage.repository import PgVectorStore


@pytest.mark.asyncio
async def test_insert_and_dense_search():
    store = PgVectorStore(TEST_URL)

    async with store.db.engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    doc_id = uuid.uuid4()
    document = DocumentORM(
        id=doc_id,
        namespace="default",
        content_hash="hash",
        metadata_={},
        source="file_upload",
    )
    chunk = ChunkORM(
        document_id=doc_id,
        namespace="default",
        content="hello world",
        content_hash="chunkhash",
        embedding=[1.0, 0.0],
        embedding_model="test",
        embedding_dimensions=2,
        chunk_index=0,
        start_char=0,
        end_char=11,
        token_count=2,
        strategy_used="semantic",
        metadata_={},
    )

    await store.insert_document(document, [chunk])

    results = await store.search_dense(
        namespace="default",
        query_embedding=[1.0, 0.0],
        top_k=1,
    )
    assert len(results) == 1
    assert results[0].content == "hello world"

