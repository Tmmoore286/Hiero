from __future__ import annotations

from datetime import datetime
from typing import Iterable

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from hiero.db import Database
from hiero.storage.models import EmbeddingCacheORM

from .base import EmbeddingCache, EmbeddingResult


class PostgresEmbeddingCache(EmbeddingCache):
    def __init__(self, database_url: str):
        self.db = Database(database_url)

    async def get(self, text_hash: str, model_id: str) -> EmbeddingResult | None:
        async with self.db.session() as session:
            row = (
                await session.execute(
                    sa.select(EmbeddingCacheORM).where(
                        EmbeddingCacheORM.text_hash == text_hash,
                        EmbeddingCacheORM.model_id == model_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            await session.execute(
                sa.update(EmbeddingCacheORM)
                .where(EmbeddingCacheORM.id == row.id)
                .values(
                    last_accessed=sa.func.now(),
                    access_count=EmbeddingCacheORM.access_count + 1,
                )
            )

            return EmbeddingResult(
                text_hash=row.text_hash,
                vector=list(row.embedding),
                model_id=row.model_id,
                dimensions=row.dimensions,
                created_at=row.created_at,
                cached=True,
            )

    async def get_batch(
        self, text_hashes: list[str], model_id: str
    ) -> dict[str, EmbeddingResult]:
        if not text_hashes:
            return {}
        async with self.db.session() as session:
            rows: Iterable[EmbeddingCacheORM] = (
                await session.execute(
                    sa.select(EmbeddingCacheORM).where(
                        EmbeddingCacheORM.model_id == model_id,
                        EmbeddingCacheORM.text_hash.in_(text_hashes),
                    )
                )
            ).scalars()

            rows_list = list(rows)
            if rows_list:
                ids = [r.id for r in rows_list]
                await session.execute(
                    sa.update(EmbeddingCacheORM)
                    .where(EmbeddingCacheORM.id.in_(ids))
                    .values(
                        last_accessed=sa.func.now(),
                        access_count=EmbeddingCacheORM.access_count + 1,
                    )
                )

            return {
                r.text_hash: EmbeddingResult(
                    text_hash=r.text_hash,
                    vector=list(r.embedding),
                    model_id=r.model_id,
                    dimensions=r.dimensions,
                    created_at=r.created_at,
                    cached=True,
                )
                for r in rows_list
            }

    async def set(self, result: EmbeddingResult) -> None:
        async with self.db.session() as session:
            stmt = insert(EmbeddingCacheORM).values(
                text_hash=result.text_hash,
                model_id=result.model_id,
                embedding=result.vector,
                dimensions=result.dimensions,
                created_at=result.created_at or datetime.utcnow(),
                last_accessed=sa.func.now(),
                access_count=0,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["text_hash", "model_id"],
                set_={
                    "embedding": result.vector,
                    "dimensions": result.dimensions,
                    "last_accessed": sa.func.now(),
                },
            )
            await session.execute(stmt)

    async def dispose(self) -> None:
        await self.db.dispose()

