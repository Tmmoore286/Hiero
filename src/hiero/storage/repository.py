from __future__ import annotations

from typing import Any, Iterable
from uuid import UUID

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import delete, select

from hiero.db import Database

from .models import ChunkORM, DocumentORM


class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    metadata: dict[str, Any]
    section_title: str | None = None
    chunk_index: int


class PgVectorStore:
    def __init__(self, database_url: str, echo: bool = False):
        self.db = Database(database_url, echo=echo)

    async def insert_document(
        self, document: DocumentORM, chunks: Iterable[ChunkORM]
    ) -> None:
        async with self.db.session() as session:
            session.add(document)
            for chunk in chunks:
                session.add(chunk)

    async def delete_document(self, document_id: UUID) -> None:
        async with self.db.session() as session:
            await session.execute(
                delete(DocumentORM).where(DocumentORM.id == document_id)
            )

    async def get_document(self, document_id: UUID) -> DocumentORM | None:
        async with self.db.session() as session:
            result = await session.execute(
                select(DocumentORM).where(DocumentORM.id == document_id)
            )
            return result.scalar_one_or_none()

    async def search_dense(
        self,
        namespace: str,
        query_embedding: list[float],
        top_k: int = 10,
        document_ids: list[UUID] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        async with self.db.session() as session:
            distance = ChunkORM.embedding.cosine_distance(query_embedding)
            stmt = (
                select(ChunkORM, distance.label("distance"))
                .where(
                    ChunkORM.namespace == namespace,
                    ChunkORM.embedding.is_not(None),
                )
                .order_by(distance)
                .limit(top_k)
            )

            if document_ids:
                stmt = stmt.where(ChunkORM.document_id.in_(document_ids))

            if metadata_filter:
                for key, value in metadata_filter.items():
                    stmt = stmt.where(ChunkORM.metadata_[key].astext == sa.cast(value, sa.Text))

            rows = (await session.execute(stmt)).all()

            results: list[SearchResult] = []
            for chunk, dist in rows:
                score = 1.0 - float(dist)
                results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        score=score,
                        metadata=chunk.metadata_ or {},
                        section_title=chunk.section_title,
                        chunk_index=chunk.chunk_index,
                    )
                )
            return results

    async def search_sparse(
        self,
        namespace: str,
        query_text: str,
        top_k: int = 10,
        document_ids: list[UUID] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        async with self.db.session() as session:
            ts_query = sa.func.plainto_tsquery("english", query_text)
            rank = sa.func.ts_rank(ChunkORM.content_tsvector, ts_query)
            stmt = (
                select(ChunkORM, rank.label("rank"))
                .where(ChunkORM.namespace == namespace)
                .where(ChunkORM.content_tsvector.op("@@")(ts_query))
                .order_by(sa.desc(rank))
                .limit(top_k)
            )

            if document_ids:
                stmt = stmt.where(ChunkORM.document_id.in_(document_ids))

            if metadata_filter:
                for key, value in metadata_filter.items():
                    stmt = stmt.where(ChunkORM.metadata_[key].astext == sa.cast(value, sa.Text))

            rows = (await session.execute(stmt)).all()
            results: list[SearchResult] = []
            for chunk, r in rows:
                results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        score=float(r),
                        metadata=chunk.metadata_ or {},
                        section_title=chunk.section_title,
                        chunk_index=chunk.chunk_index,
                    )
                )
            return results

    async def dispose(self) -> None:
        await self.db.dispose()
