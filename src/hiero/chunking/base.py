from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from time import perf_counter
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from hiero.ingestion.base import Document


class ChunkStrategy(str, Enum):
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    FIXED = "fixed"
    CODE = "code"
    ADAPTIVE = "adaptive"


class ChunkMetadata(BaseModel):
    document_id: UUID
    chunk_index: int
    start_char: int
    end_char: int
    section_title: str | None = None
    section_hierarchy: list[str] = Field(default_factory=list)
    page_number: int | None = None
    parent_chunk_id: UUID | None = None
    strategy_used: ChunkStrategy
    token_count: int


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: ChunkMetadata


class ChunkingConfig(BaseModel):
    strategy: ChunkStrategy = ChunkStrategy.ADAPTIVE
    target_chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    respect_sentence_boundaries: bool = True
    preserve_code_blocks: bool = True
    tokenizer: str = "cl100k_base"


class ChunkingResult(BaseModel):
    document_id: UUID
    chunks: list[Chunk]
    total_chunks: int
    strategy_used: ChunkStrategy
    avg_chunk_size: float
    processing_time_ms: float


class ChunkerProtocol(ABC):
    @abstractmethod
    async def chunk(self, document: Document, config: ChunkingConfig) -> ChunkingResult:
        ...

    @property
    @abstractmethod
    def strategy(self) -> ChunkStrategy:
        ...


def _finish_result(
    document: Document,
    chunks: list[Chunk],
    strategy: ChunkStrategy,
    start_time: float,
) -> ChunkingResult:
    token_counts = [c.metadata.token_count for c in chunks] or [0]
    return ChunkingResult(
        document_id=document.id,
        chunks=chunks,
        total_chunks=len(chunks),
        strategy_used=strategy,
        avg_chunk_size=sum(token_counts) / max(len(token_counts), 1),
        processing_time_ms=(perf_counter() - start_time) * 1000,
    )
