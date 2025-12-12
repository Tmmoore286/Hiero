from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from hashlib import sha256
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    LOCAL = "local"


class EmbeddingModel(BaseModel):
    provider: EmbeddingProvider
    model_name: str
    dimensions: int
    max_tokens: int
    version: str

    @property
    def model_id(self) -> str:
        return f"{self.provider.value}:{self.model_name}:{self.version}"


EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    "text-embedding-3-small": EmbeddingModel(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-small",
        dimensions=1536,
        max_tokens=8191,
        version="2024-01",
    ),
    "text-embedding-3-large": EmbeddingModel(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-large",
        dimensions=3072,
        max_tokens=8191,
        version="2024-01",
    ),
    "all-MiniLM-L6-v2": EmbeddingModel(
        provider=EmbeddingProvider.LOCAL,
        model_name="all-MiniLM-L6-v2",
        dimensions=384,
        max_tokens=256,
        version="v2",
    ),
}


class EmbeddingResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text_hash: str
    vector: list[float]
    model_id: str
    dimensions: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    cached: bool = False

    @classmethod
    def from_text(
        cls, text: str, vector: list[float], model: EmbeddingModel, cached: bool = False
    ) -> "EmbeddingResult":
        text_hash = sha256(text.encode("utf-8")).hexdigest()
        return cls(
            text_hash=text_hash,
            vector=vector,
            model_id=model.model_id,
            dimensions=len(vector),
            cached=cached,
        )


class BatchEmbeddingResult(BaseModel):
    results: list[EmbeddingResult]
    total_tokens: int
    processing_time_ms: float
    cache_hits: int
    api_calls: int


class EmbeddingError(Exception):
    pass


class EmbedderProtocol(ABC):
    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        ...

    @abstractmethod
    async def embed_batch(
        self, texts: list[str], batch_size: int = 100, normalize: bool = True
    ) -> BatchEmbeddingResult:
        ...

    @property
    @abstractmethod
    def model(self) -> EmbeddingModel:
        ...


class EmbeddingCache(ABC):
    @abstractmethod
    async def get(self, text_hash: str, model_id: str) -> EmbeddingResult | None:
        ...

    @abstractmethod
    async def set(self, result: EmbeddingResult) -> None:
        ...

    @abstractmethod
    async def get_batch(
        self, text_hashes: list[str], model_id: str
    ) -> dict[str, EmbeddingResult]:
        ...
