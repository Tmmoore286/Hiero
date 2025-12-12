# Component Spec: Embedding Layer

## Overview

The Embedding Layer converts text chunks into dense vector representations using pluggable embedding providers. It supports multiple providers (OpenAI, Cohere, local models) with a unified interface, enabling model-agnostic storage and retrieval.

## Requirements

### Functional
- **FR-1**: Support multiple embedding providers (OpenAI, Cohere, sentence-transformers)
- **FR-2**: Batch embedding for efficiency (process multiple chunks in one call)
- **FR-3**: Automatic retry with exponential backoff for API failures
- **FR-4**: Rate limiting to avoid provider throttling
- **FR-5**: Caching layer to avoid re-embedding identical text
- **FR-6**: Dimension normalization (store vectors as-is, handle at query time)
- **FR-7**: Model versioning (track which model produced each embedding)

### Non-Functional
- **NFR-1**: Embed 100 chunks in < 5 seconds (with batching)
- **NFR-2**: < 1% API failure rate after retries
- **NFR-3**: Support vectors up to 4096 dimensions
- **NFR-4**: Cache hit rate > 80% for repeated content

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    LOCAL = "local"  # sentence-transformers


class EmbeddingModel(BaseModel):
    """Metadata about an embedding model."""
    provider: EmbeddingProvider
    model_name: str
    dimensions: int
    max_tokens: int
    version: str  # For tracking model updates

    @property
    def model_id(self) -> str:
        """Unique identifier for this model configuration."""
        return f"{self.provider.value}:{self.model_name}:{self.version}"


# Pre-defined model configurations
EMBEDDING_MODELS = {
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
    "embed-english-v3.0": EmbeddingModel(
        provider=EmbeddingProvider.COHERE,
        model_name="embed-english-v3.0",
        dimensions=1024,
        max_tokens=512,
        version="v3.0",
    ),
    "embed-multilingual-v3.0": EmbeddingModel(
        provider=EmbeddingProvider.COHERE,
        model_name="embed-multilingual-v3.0",
        dimensions=1024,
        max_tokens=512,
        version="v3.0",
    ),
    "all-MiniLM-L6-v2": EmbeddingModel(
        provider=EmbeddingProvider.LOCAL,
        model_name="all-MiniLM-L6-v2",
        dimensions=384,
        max_tokens=256,
        version="v2",
    ),
    "all-mpnet-base-v2": EmbeddingModel(
        provider=EmbeddingProvider.LOCAL,
        model_name="all-mpnet-base-v2",
        dimensions=768,
        max_tokens=384,
        version="v2",
    ),
}


class EmbeddingRequest(BaseModel):
    """Request to embed one or more texts."""
    texts: list[str]
    model: str = "text-embedding-3-small"
    normalize: bool = True  # L2 normalize vectors


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    id: UUID = Field(default_factory=uuid4)
    text_hash: str  # SHA-256 of input text for caching
    vector: list[float]
    model_id: str
    dimensions: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    cached: bool = False  # Whether this was a cache hit


class BatchEmbeddingResult(BaseModel):
    """Result of batch embedding operation."""
    results: list[EmbeddingResult]
    total_tokens: int
    processing_time_ms: float
    cache_hits: int
    api_calls: int
```

## Interfaces

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator


class EmbedderProtocol(ABC):
    """Base protocol for all embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with vector

        Raises:
            EmbeddingError: If embedding fails
            RateLimitError: If rate limited
        """
        ...

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> BatchEmbeddingResult:
        """
        Embed multiple texts with automatic batching.

        Args:
            texts: List of texts to embed
            batch_size: Max texts per API call

        Returns:
            BatchEmbeddingResult with all vectors
        """
        ...

    @property
    @abstractmethod
    def model(self) -> EmbeddingModel:
        """Return the model configuration."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available."""
        ...


class EmbeddingCache(ABC):
    """Cache interface for embedding results."""

    @abstractmethod
    async def get(self, text_hash: str, model_id: str) -> EmbeddingResult | None:
        """Get cached embedding if exists."""
        ...

    @abstractmethod
    async def set(self, result: EmbeddingResult) -> None:
        """Cache an embedding result."""
        ...

    @abstractmethod
    async def get_batch(
        self, text_hashes: list[str], model_id: str
    ) -> dict[str, EmbeddingResult]:
        """Get multiple cached embeddings."""
        ...
```

## Implementation Details

### OpenAI Embedder

```python
import asyncio
import hashlib
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class OpenAIEmbedder(EmbedderProtocol):
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        cache: EmbeddingCache | None = None,
        max_retries: int = 3,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self._model = EMBEDDING_MODELS[model_name]
        self.cache = cache
        self.max_retries = max_retries

        # Rate limiting: OpenAI allows 3000 RPM for embeddings
        self._semaphore = asyncio.Semaphore(50)  # Max concurrent requests
        self._rate_limiter = TokenBucketRateLimiter(
            rate=50,  # requests per second
            capacity=100,
        )

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed single text with caching."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Check cache
        if self.cache:
            cached = await self.cache.get(text_hash, self._model.model_id)
            if cached:
                cached.cached = True
                return cached

        # Call API
        result = await self._embed_with_retry([text])
        embedding_result = EmbeddingResult(
            text_hash=text_hash,
            vector=result[0],
            model_id=self._model.model_id,
            dimensions=self._model.dimensions,
        )

        # Cache result
        if self.cache:
            await self.cache.set(embedding_result)

        return embedding_result

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> BatchEmbeddingResult:
        """Embed multiple texts with batching and caching."""
        import time
        start_time = time.perf_counter()

        # Compute hashes
        text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

        # Check cache for all
        cached_results = {}
        if self.cache:
            cached_results = await self.cache.get_batch(
                text_hashes, self._model.model_id
            )

        # Identify uncached texts
        uncached_indices = [
            i for i, h in enumerate(text_hashes) if h not in cached_results
        ]
        uncached_texts = [texts[i] for i in uncached_indices]

        # Batch embed uncached
        api_calls = 0
        new_embeddings = []

        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i + batch_size]
            vectors = await self._embed_with_retry(batch)
            new_embeddings.extend(vectors)
            api_calls += 1

        # Build results list in original order
        results = []
        new_idx = 0

        for i, text_hash in enumerate(text_hashes):
            if text_hash in cached_results:
                result = cached_results[text_hash]
                result.cached = True
                results.append(result)
            else:
                result = EmbeddingResult(
                    text_hash=text_hash,
                    vector=new_embeddings[new_idx],
                    model_id=self._model.model_id,
                    dimensions=self._model.dimensions,
                )
                results.append(result)

                # Cache new result
                if self.cache:
                    await self.cache.set(result)

                new_idx += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return BatchEmbeddingResult(
            results=results,
            total_tokens=sum(len(t.split()) for t in texts),  # Approximate
            processing_time_ms=elapsed_ms,
            cache_hits=len(cached_results),
            api_calls=api_calls,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    )
    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI API with retry logic."""
        async with self._semaphore:
            await self._rate_limiter.acquire()

            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self._model.model_name,
                )
                return [e.embedding for e in response.data]

            except openai.RateLimitError as e:
                raise RateLimitError(str(e))
            except openai.APIConnectionError as e:
                raise APIConnectionError(str(e))
            except openai.APIError as e:
                raise EmbeddingError(str(e))

    async def health_check(self) -> bool:
        """Check OpenAI API availability."""
        try:
            await self.client.embeddings.create(
                input=["health check"],
                model=self._model.model_name,
            )
            return True
        except Exception:
            return False
```

### Cohere Embedder

```python
import cohere


class CohereEmbedder(EmbedderProtocol):
    """Cohere embedding provider."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "embed-english-v3.0",
        cache: EmbeddingCache | None = None,
    ):
        self.client = cohere.AsyncClient(api_key=api_key)
        self._model = EMBEDDING_MODELS[model_name]
        self.cache = cache
        self._semaphore = asyncio.Semaphore(10)

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    async def embed(self, text: str) -> EmbeddingResult:
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if self.cache:
            cached = await self.cache.get(text_hash, self._model.model_id)
            if cached:
                cached.cached = True
                return cached

        async with self._semaphore:
            response = await self.client.embed(
                texts=[text],
                model=self._model.model_name,
                input_type="search_document",  # or "search_query" for queries
            )

        result = EmbeddingResult(
            text_hash=text_hash,
            vector=response.embeddings[0],
            model_id=self._model.model_id,
            dimensions=self._model.dimensions,
        )

        if self.cache:
            await self.cache.set(result)

        return result

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 96,  # Cohere limit
    ) -> BatchEmbeddingResult:
        # Similar implementation to OpenAI...
        pass

    async def health_check(self) -> bool:
        try:
            await self.client.embed(
                texts=["health"],
                model=self._model.model_name,
                input_type="search_document",
            )
            return True
        except Exception:
            return False
```

### Local Embedder (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer
import numpy as np


class LocalEmbedder(EmbedderProtocol):
    """Local embedding using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache: EmbeddingCache | None = None,
        device: str = "cpu",  # or "cuda", "mps"
    ):
        self._model_config = EMBEDDING_MODELS[model_name]
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = cache
        self._lock = asyncio.Lock()  # Serialize model access

    @property
    def model(self) -> EmbeddingModel:
        return self._model_config

    async def embed(self, text: str) -> EmbeddingResult:
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if self.cache:
            cached = await self.cache.get(text_hash, self._model_config.model_id)
            if cached:
                cached.cached = True
                return cached

        # Run in thread pool to avoid blocking
        vector = await asyncio.get_event_loop().run_in_executor(
            None, self._embed_sync, text
        )

        result = EmbeddingResult(
            text_hash=text_hash,
            vector=vector,
            model_id=self._model_config.model_id,
            dimensions=self._model_config.dimensions,
        )

        if self.cache:
            await self.cache.set(result)

        return result

    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding (called in thread pool)."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> BatchEmbeddingResult:
        import time
        start_time = time.perf_counter()

        text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

        cached_results = {}
        if self.cache:
            cached_results = await self.cache.get_batch(
                text_hashes, self._model_config.model_id
            )

        uncached_indices = [
            i for i, h in enumerate(text_hashes) if h not in cached_results
        ]
        uncached_texts = [texts[i] for i in uncached_indices]

        # Batch encode in thread pool
        if uncached_texts:
            new_embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self._batch_embed_sync, uncached_texts, batch_size
            )
        else:
            new_embeddings = []

        # Build results
        results = []
        new_idx = 0

        for i, text_hash in enumerate(text_hashes):
            if text_hash in cached_results:
                result = cached_results[text_hash]
                result.cached = True
                results.append(result)
            else:
                result = EmbeddingResult(
                    text_hash=text_hash,
                    vector=new_embeddings[new_idx],
                    model_id=self._model_config.model_id,
                    dimensions=self._model_config.dimensions,
                )
                results.append(result)
                if self.cache:
                    await self.cache.set(result)
                new_idx += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return BatchEmbeddingResult(
            results=results,
            total_tokens=0,  # Local models don't track tokens
            processing_time_ms=elapsed_ms,
            cache_hits=len(cached_results),
            api_calls=0,
        )

    def _batch_embed_sync(
        self, texts: list[str], batch_size: int
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def health_check(self) -> bool:
        try:
            await self.embed("health check")
            return True
        except Exception:
            return False
```

### Embedding Cache (PostgreSQL-backed)

```python
from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from sqlalchemy.ext.asyncio import AsyncSession


class EmbeddingCacheModel(Base):
    """PostgreSQL table for embedding cache."""
    __tablename__ = "embedding_cache"

    text_hash = Column(String(64), primary_key=True)
    model_id = Column(String(128), primary_key=True)
    vector = Column(ARRAY(REAL))
    dimensions = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)
    last_accessed = Column(DateTime, default=datetime.utcnow)


class PostgresEmbeddingCache(EmbeddingCache):
    """PostgreSQL-backed embedding cache."""

    def __init__(self, session_factory, max_entries: int = 100_000):
        self.session_factory = session_factory
        self.max_entries = max_entries

    async def get(
        self, text_hash: str, model_id: str
    ) -> EmbeddingResult | None:
        async with self.session_factory() as session:
            result = await session.execute(
                select(EmbeddingCacheModel).where(
                    EmbeddingCacheModel.text_hash == text_hash,
                    EmbeddingCacheModel.model_id == model_id,
                )
            )
            cached = result.scalar_one_or_none()

            if cached:
                # Update access stats
                cached.access_count += 1
                cached.last_accessed = datetime.utcnow()
                await session.commit()

                return EmbeddingResult(
                    text_hash=cached.text_hash,
                    vector=cached.vector,
                    model_id=cached.model_id,
                    dimensions=cached.dimensions,
                    created_at=cached.created_at,
                )

            return None

    async def set(self, result: EmbeddingResult) -> None:
        async with self.session_factory() as session:
            # Upsert
            stmt = insert(EmbeddingCacheModel).values(
                text_hash=result.text_hash,
                model_id=result.model_id,
                vector=result.vector,
                dimensions=result.dimensions,
            ).on_conflict_do_update(
                index_elements=['text_hash', 'model_id'],
                set_={'last_accessed': datetime.utcnow()},
            )
            await session.execute(stmt)
            await session.commit()

    async def get_batch(
        self, text_hashes: list[str], model_id: str
    ) -> dict[str, EmbeddingResult]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(EmbeddingCacheModel).where(
                    EmbeddingCacheModel.text_hash.in_(text_hashes),
                    EmbeddingCacheModel.model_id == model_id,
                )
            )
            cached = result.scalars().all()

            return {
                c.text_hash: EmbeddingResult(
                    text_hash=c.text_hash,
                    vector=c.vector,
                    model_id=c.model_id,
                    dimensions=c.dimensions,
                )
                for c in cached
            }

    async def evict_old(self) -> int:
        """Evict least recently used entries if over capacity."""
        async with self.session_factory() as session:
            count = await session.execute(
                select(func.count()).select_from(EmbeddingCacheModel)
            )
            total = count.scalar()

            if total <= self.max_entries:
                return 0

            # Delete oldest entries
            to_delete = total - self.max_entries
            oldest = await session.execute(
                select(EmbeddingCacheModel.text_hash, EmbeddingCacheModel.model_id)
                .order_by(EmbeddingCacheModel.last_accessed)
                .limit(to_delete)
            )

            for text_hash, model_id in oldest:
                await session.execute(
                    delete(EmbeddingCacheModel).where(
                        EmbeddingCacheModel.text_hash == text_hash,
                        EmbeddingCacheModel.model_id == model_id,
                    )
                )

            await session.commit()
            return to_delete
```

### Embedder Factory

```python
class EmbedderFactory:
    """Factory for creating embedders based on configuration."""

    def __init__(self, settings: Settings, cache: EmbeddingCache | None = None):
        self.settings = settings
        self.cache = cache

    def create(
        self,
        provider: EmbeddingProvider | None = None,
        model_name: str | None = None,
    ) -> EmbedderProtocol:
        """Create embedder based on provider and model."""
        provider = provider or EmbeddingProvider(
            self.settings.default_embedding_provider
        )
        model_name = model_name or self.settings.default_embedding_model

        if provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbedder(
                api_key=self.settings.openai_api_key.get_secret_value(),
                model_name=model_name,
                cache=self.cache,
            )

        elif provider == EmbeddingProvider.COHERE:
            return CohereEmbedder(
                api_key=self.settings.cohere_api_key.get_secret_value(),
                model_name=model_name,
                cache=self.cache,
            )

        elif provider == EmbeddingProvider.LOCAL:
            return LocalEmbedder(
                model_name=model_name,
                cache=self.cache,
            )

        raise ValueError(f"Unknown provider: {provider}")
```

### Rate Limiter

```python
import asyncio
import time


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until tokens are available."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.rate,
                )
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Wait for tokens to refill
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
```

## Error Handling

```python
class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class RateLimitError(EmbeddingError):
    """Rate limit exceeded."""
    pass


class APIConnectionError(EmbeddingError):
    """Failed to connect to embedding API."""
    pass


class ModelNotFoundError(EmbeddingError):
    """Requested model not found."""
    pass


class DimensionMismatchError(EmbeddingError):
    """Vector dimensions don't match expected."""
    pass
```

## ARCnet Compatibility

Hiero embeddings are compatible with ARCnet's `EmbeddingService`:

```python
# ARCnet uses text-embedding-3-small (1536-dim) by default
# Hiero should default to the same for compatibility

class ARCnetCompatibleEmbedder:
    """Wrapper ensuring ARCnet compatibility."""

    def __init__(self, embedder: EmbedderProtocol):
        self.embedder = embedder

    async def embed_for_arcnet(
        self,
        text: str,
        doc_type: str = "doctrine",  # "doctrine" | "mission"
    ) -> list[float]:
        """
        Embed text in ARCnet-compatible format.

        ARCnet expects:
        - 1536-dimensional vectors (text-embedding-3-small)
        - L2 normalized vectors
        - Cosine similarity for matching
        """
        result = await self.embedder.embed(text)

        # Verify dimensions match ARCnet
        if result.dimensions != 1536:
            raise DimensionMismatchError(
                f"ARCnet expects 1536 dims, got {result.dimensions}"
            )

        return result.vector
```

## Dependencies

```toml
[project.dependencies]
openai = "^1.14"
cohere = "^5.0"
sentence-transformers = "^2.6"
tenacity = "^8.2"
tiktoken = "^0.7"
numpy = "^1.26"
```

## Testing Strategy

1. **Unit Tests**: Each embedder with mocked APIs
2. **Integration Tests**: Real API calls (use small test texts)
3. **Cache Tests**: Hit/miss behavior, eviction
4. **Rate Limit Tests**: Verify backoff behavior
5. **Dimension Tests**: Verify output dimensions match spec
6. **Performance Tests**: Batch embedding throughput
