from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from time import perf_counter
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    MULTI_HOP = "multi_hop"


class FusionMethod(str, Enum):
    RRF = "rrf"
    WEIGHTED_SUM = "weighted_sum"
    DBSF = "dbsf"


class RetrievalConfig(BaseModel):
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    dense_weight: float = 0.7
    fusion_method: FusionMethod = FusionMethod.RRF
    dense_top_k: int = 50
    sparse_top_k: int = 50
    max_hops: int = 3
    hop_top_k: int = 5
    score_threshold: float | None = None
    metadata_filter: dict | None = None
    document_ids: list[UUID] | None = None
    rerank_results: bool = False
    rerank_candidates: int = 20


class RetrievalQuery(BaseModel):
    text: str
    namespace: str = "default"
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    query_embedding: list[float] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    rank: int
    retrieval_strategy: RetrievalStrategy
    dense_score: float | None = None
    sparse_score: float | None = None
    metadata: dict = Field(default_factory=dict)
    section_title: str | None = None
    chunk_index: int


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    total_candidates: int
    strategy_used: RetrievalStrategy
    latency_ms: float
    dense_candidates: int = 0
    sparse_candidates: int = 0
    query_embedding_time_ms: float = 0


class RetrieverProtocol(ABC):
    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        ...

    @abstractmethod
    async def retrieve_batch(self, queries: list[RetrievalQuery]) -> list[RetrievalResult]:
        ...


class _Timer:
    def __init__(self):
        self._start = perf_counter()

    def ms(self) -> float:
        return (perf_counter() - self._start) * 1000
