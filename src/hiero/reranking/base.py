from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from time import perf_counter
from uuid import UUID

from pydantic import BaseModel, Field

from hiero.retrieval.base import RetrievedChunk


class RerankingMethod(str, Enum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    LISTWISE = "listwise"


class RerankerConfig(BaseModel):
    method: RerankingMethod = RerankingMethod.POINTWISE
    top_k: int = 5
    candidates: int = 20
    include_reasoning: bool = False
    temperature: float = 0.0
    max_retries: int = 2


class RerankRequest(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    config: RerankerConfig = Field(default_factory=RerankerConfig)


class RerankedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    original_rank: int
    reranked_rank: int
    original_score: float
    reranked_score: float
    reasoning: str | None = None
    metadata: dict = Field(default_factory=dict)


class RerankResult(BaseModel):
    query: str
    chunks: list[RerankedChunk]
    method_used: RerankingMethod
    latency_ms: float
    tokens_used: int
    rank_changes: int


class RerankerProtocol(ABC):
    @abstractmethod
    async def rerank(self, request: RerankRequest) -> RerankResult:
        ...


class _Timer:
    def __init__(self):
        self._start = perf_counter()

    def ms(self) -> float:
        return (perf_counter() - self._start) * 1000

