# Component Spec: API & Library Layer

## Overview

The API Layer exposes Hiero capabilities through both a Python library (for direct integration) and a REST API (for service deployment). Designed for serverless deployment (AWS Lambda, Vercel) while supporting traditional hosting.

## Requirements

### Functional
- **FR-1**: Python library with simple, intuitive API
- **FR-2**: REST API with FastAPI
- **FR-3**: OpenAPI documentation auto-generation
- **FR-4**: Authentication and API key management
- **FR-5**: Rate limiting per tenant/API key
- **FR-6**: Streaming responses for generation
- **FR-7**: Health and metrics endpoints
- **FR-8**: Serverless deployment support (Mangum adapter)

### Non-Functional
- **NFR-1**: API response time < 200ms (excluding LLM calls)
- **NFR-2**: Support 100+ concurrent requests
- **NFR-3**: 99.9% availability target
- **NFR-4**: Comprehensive request logging

## Python Library API

### High-Level Interface

```python
"""
Hiero Python Library

Simple, intuitive API for RAG operations.

Usage:
    from hiero import Hiero

    # Initialize
    rag = Hiero(
        database_url="postgresql://...",
        openai_api_key="sk-...",
    )

    # Ingest documents
    doc_id = await rag.ingest("path/to/document.pdf")

    # Query
    response = await rag.query("What is the main topic?")
    print(response.answer)
    print(response.citations)
"""

from typing import AsyncIterator, BinaryIO
from pathlib import Path
from uuid import UUID

from .config import Settings
from .ingestion import IngestionRouter, Document
from .chunking import AdaptiveChunker, ChunkingConfig
from .embedding import EmbedderFactory
from .storage import PgVectorStore
from .retrieval import HybridRetriever, RetrievalConfig
from .reranking import LLMReranker
from .agent import ReActAgent, AgentConfig
from .generation import GroundedGenerator, GenerationConfig


class Hiero:
    """
    Main entry point for Hiero library.

    Provides a simple interface for document ingestion,
    retrieval, and question answering.
    """

    def __init__(
        self,
        database_url: str,
        openai_api_key: str | None = None,
        cohere_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        namespace: str = "default",
        **kwargs,
    ):
        """
        Initialize Hiero.

        Args:
            database_url: PostgreSQL connection URL
            openai_api_key: OpenAI API key (optional)
            cohere_api_key: Cohere API key (optional)
            anthropic_api_key: Anthropic API key (optional)
            namespace: Default namespace for operations
            **kwargs: Additional configuration options
        """
        self.settings = Settings(
            database_url=database_url,
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
            **kwargs,
        )
        self.namespace = namespace
        self._initialized = False

    async def initialize(self):
        """Initialize all components. Called automatically on first use."""
        if self._initialized:
            return

        # Initialize components
        self.vector_store = PgVectorStore(self.settings.database_url)
        self.embedder = EmbedderFactory(self.settings).create()
        self.chunker = AdaptiveChunker(self.embedder.tokenizer)
        self.ingestor = IngestionRouter(...)
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        self.reranker = LLMReranker(
            provider="openai",
            api_key=self.settings.openai_api_key.get_secret_value(),
        )
        self.generator = GroundedGenerator(
            provider="openai",
            api_key=self.settings.openai_api_key.get_secret_value(),
        )
        self.agent = ReActAgent(
            llm_client=...,
            retriever=self.retriever,
            reranker=self.reranker,
        )

        self._initialized = True

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        """Clean up resources."""
        # Close database connections, etc.
        pass

    # =========== Document Ingestion ===========

    async def ingest(
        self,
        source: str | Path | BinaryIO,
        metadata: dict | None = None,
        namespace: str | None = None,
    ) -> UUID:
        """
        Ingest a document from file path, URL, or file object.

        Args:
            source: File path, URL string, or file-like object
            metadata: Optional metadata to attach
            namespace: Namespace override (default: instance namespace)

        Returns:
            Document ID

        Example:
            # From file
            doc_id = await rag.ingest("report.pdf")

            # From URL
            doc_id = await rag.ingest("https://example.com/page")

            # With metadata
            doc_id = await rag.ingest("doc.pdf", metadata={"author": "Tim"})
        """
        await self.initialize()
        namespace = namespace or self.namespace

        # Ingest document
        if isinstance(source, (str, Path)):
            source = str(source)
            if source.startswith(('http://', 'https://')):
                doc = await self.ingestor.ingest_url(source, metadata)
            else:
                with open(source, 'rb') as f:
                    doc = await self.ingestor.ingest_file(f, metadata)
        else:
            doc = await self.ingestor.ingest_file(source, metadata)

        # Chunk document
        chunks = await self.chunker.chunk(doc, ChunkingConfig())

        # Embed chunks
        texts = [c.content for c in chunks.chunks]
        embeddings = await self.embedder.embed_batch(texts)

        # Store
        for chunk, embed_result in zip(chunks.chunks, embeddings.results):
            chunk.embedding = embed_result.vector
            chunk.embedding_model = embed_result.model_id
            chunk.embedding_dimensions = len(embed_result.vector)

        await self.vector_store.insert_document(
            DocumentModel(
                id=doc.id,
                namespace=namespace,
                content_hash=doc.content_hash,
                metadata=doc.metadata.model_dump(),
                source=doc.source.value,
            ),
            [ChunkModel(...) for chunk in chunks.chunks],
        )

        return doc.id

    async def ingest_batch(
        self,
        sources: list[str | Path],
        metadata: dict | None = None,
        namespace: str | None = None,
    ) -> list[UUID]:
        """
        Ingest multiple documents concurrently.

        Args:
            sources: List of file paths or URLs
            metadata: Metadata to attach to all documents
            namespace: Namespace override

        Returns:
            List of document IDs
        """
        tasks = [self.ingest(s, metadata, namespace) for s in sources]
        return await asyncio.gather(*tasks)

    # =========== Retrieval ===========

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        strategy: str = "hybrid",
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            namespace: Namespace override
            strategy: "dense", "sparse", or "hybrid"
            metadata_filter: Filter by metadata fields

        Returns:
            List of chunk dictionaries with content and scores

        Example:
            results = await rag.retrieve("machine learning", top_k=10)
            for r in results:
                print(r["content"], r["score"])
        """
        await self.initialize()
        namespace = namespace or self.namespace

        from .retrieval import RetrievalQuery, RetrievalConfig, RetrievalStrategy

        result = await self.retriever.retrieve(RetrievalQuery(
            text=query,
            namespace=namespace,
            config=RetrievalConfig(
                strategy=RetrievalStrategy(strategy),
                top_k=top_k,
                metadata_filter=metadata_filter,
            ),
        ))

        return [
            {
                "chunk_id": str(c.chunk_id),
                "document_id": str(c.document_id),
                "content": c.content,
                "score": c.score,
                "metadata": c.metadata,
            }
            for c in result.chunks
        ]

    # =========== Question Answering ===========

    async def query(
        self,
        question: str,
        namespace: str | None = None,
        mode: str = "grounded",
        use_agent: bool = False,
        **kwargs,
    ) -> "QueryResponse":
        """
        Answer a question using RAG.

        Args:
            question: The question to answer
            namespace: Namespace override
            mode: "simple", "grounded", or "analytical"
            use_agent: Use agentic RAG for complex queries
            **kwargs: Additional configuration

        Returns:
            QueryResponse with answer and citations

        Example:
            response = await rag.query("What are the key findings?")
            print(response.answer)
            for citation in response.citations:
                print(f"- {citation.snippet}")
        """
        await self.initialize()
        namespace = namespace or self.namespace

        if use_agent:
            return await self._query_agent(question, namespace, **kwargs)
        else:
            return await self._query_simple(question, namespace, mode, **kwargs)

    async def _query_simple(
        self,
        question: str,
        namespace: str,
        mode: str,
        **kwargs,
    ) -> "QueryResponse":
        """Simple retrieve-then-generate pipeline."""
        # Retrieve
        chunks = await self.retrieve(question, top_k=10, namespace=namespace)

        # Rerank
        from .reranking import RerankRequest, RerankerConfig
        reranked = await self.reranker.rerank(RerankRequest(
            query=question,
            chunks=[...],  # Convert to RetrievedChunk
            config=RerankerConfig(top_k=5),
        ))

        # Generate
        from .generation import GenerationRequest, SourceContext, GenerationMode
        gen_response = await self.generator.generate(GenerationRequest(
            query=question,
            context=[
                SourceContext(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    metadata=c.metadata,
                    relevance_score=c.reranked_score,
                )
                for c in reranked.chunks
            ],
            config=GenerationConfig(mode=GenerationMode(mode)),
        ))

        return QueryResponse(
            question=question,
            answer=gen_response.response,
            citations=[
                Citation(
                    chunk_id=c.chunk_id,
                    snippet=c.quoted_text,
                    source_index=c.source_index,
                )
                for c in gen_response.citations
            ],
            sources_used=len(gen_response.sources_used),
            latency_ms=gen_response.latency_ms,
        )

    async def _query_agent(
        self,
        question: str,
        namespace: str,
        **kwargs,
    ) -> "QueryResponse":
        """Agentic RAG for complex queries."""
        from .agent import AgentQuery

        response = await self.agent.run(AgentQuery(
            question=question,
            namespace=namespace,
        ))

        return QueryResponse(
            question=question,
            answer=response.answer,
            citations=[
                Citation(
                    chunk_id=c.chunk_id,
                    snippet=c.content_snippet,
                    source_index=0,
                )
                for c in response.citations
            ],
            sources_used=len(response.sources_used),
            latency_ms=response.total_latency_ms,
            agent_steps=response.total_steps,
        )

    async def query_streaming(
        self,
        question: str,
        namespace: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Answer a question with streaming output.

        Yields:
            Response tokens as they're generated

        Example:
            async for token in rag.query_streaming("Summarize the document"):
                print(token, end="", flush=True)
        """
        await self.initialize()
        namespace = namespace or self.namespace

        # Retrieve and rerank
        chunks = await self.retrieve(question, top_k=10, namespace=namespace)

        # Stream generation
        from .generation import GenerationRequest, SourceContext

        async for token in self.generator.generate_streaming(GenerationRequest(
            query=question,
            context=[SourceContext(...) for c in chunks],
        )):
            yield token

    # =========== Document Management ===========

    async def list_documents(
        self,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List documents in a namespace."""
        await self.initialize()
        namespace = namespace or self.namespace

        docs = await self.vector_store.list_documents(namespace, limit, offset)
        return [
            {
                "id": str(d.id),
                "metadata": d.metadata,
                "source": d.source,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ]

    async def delete_document(self, document_id: str | UUID) -> bool:
        """Delete a document and its chunks."""
        await self.initialize()
        if isinstance(document_id, str):
            document_id = UUID(document_id)
        return await self.vector_store.delete_document(document_id)

    async def get_stats(self, namespace: str | None = None) -> dict:
        """Get statistics for a namespace."""
        await self.initialize()
        namespace = namespace or self.namespace
        return await self.vector_store.get_stats(namespace)


# Response types
class Citation:
    chunk_id: UUID
    snippet: str
    source_index: int


class QueryResponse:
    question: str
    answer: str
    citations: list[Citation]
    sources_used: int
    latency_ms: float
    agent_steps: int | None = None
```

## REST API

### FastAPI Application

```python
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional
import asyncio


# Create FastAPI app
app = FastAPI(
    title="Hiero API",
    description="Research-grade Retrieval-Augmented Generation API",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# Request/Response Schemas
class IngestRequest(BaseModel):
    """Request to ingest a document."""
    url: Optional[str] = Field(None, description="URL to fetch and ingest")
    content: Optional[str] = Field(None, description="Raw content to ingest")
    metadata: dict = Field(default_factory=dict)
    namespace: str = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/document",
                "metadata": {"source": "web", "category": "research"},
                "namespace": "default"
            }
        }


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    document_id: str
    chunks_created: int
    status: str


class QueryRequest(BaseModel):
    """Request to query the knowledge base."""
    question: str = Field(..., description="The question to answer")
    namespace: str = "default"
    top_k: int = Field(5, ge=1, le=50)
    mode: str = Field("grounded", pattern="^(simple|grounded|analytical)$")
    use_agent: bool = False
    stream: bool = False
    metadata_filter: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main findings of the study?",
                "namespace": "default",
                "top_k": 5,
                "mode": "grounded"
            }
        }


class QueryResponse(BaseModel):
    """Response from a query."""
    question: str
    answer: str
    citations: list[dict]
    sources_used: int
    confidence: float
    latency_ms: float


class RetrieveRequest(BaseModel):
    """Request to retrieve documents."""
    query: str
    namespace: str = "default"
    top_k: int = Field(10, ge=1, le=100)
    strategy: str = Field("hybrid", pattern="^(dense|sparse|hybrid)$")
    metadata_filter: Optional[dict] = None


class RetrieveResponse(BaseModel):
    """Response from retrieval."""
    query: str
    results: list[dict]
    total: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    embedding_model: str
    uptime_seconds: float


# Dependencies
async def get_rag() -> Hiero:
    """Get Hiero instance from app state."""
    return app.state.rag


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """Verify API key and return tenant ID."""
    api_key = credentials.credentials

    # Validate API key (implement your auth logic)
    tenant_id = await validate_api_key(api_key)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return tenant_id


# Routes
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(rag: Hiero = Depends(get_rag)):
    """Check API health and component status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="connected",
        embedding_model=rag.embedder.model.model_name,
        uptime_seconds=time.time() - app.state.start_time,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    rag: Hiero = Depends(get_rag),
    tenant_id: str = Depends(verify_api_key),
):
    """
    Ingest a document from URL or raw content.

    The document will be chunked, embedded, and stored for retrieval.
    """
    if not request.url and not request.content:
        raise HTTPException(
            status_code=400,
            detail="Either 'url' or 'content' must be provided",
        )

    try:
        if request.url:
            doc_id = await rag.ingest(
                request.url,
                metadata=request.metadata,
                namespace=request.namespace,
            )
        else:
            # Create temp file for content
            from io import BytesIO
            content_file = BytesIO(request.content.encode())
            doc_id = await rag.ingest(
                content_file,
                metadata=request.metadata,
                namespace=request.namespace,
            )

        # Get chunk count
        stats = await rag.vector_store.get_stats(request.namespace)

        return IngestResponse(
            document_id=str(doc_id),
            chunks_created=stats.get("chunk_count", 0),
            status="completed",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    rag: Hiero = Depends(get_rag),
    tenant_id: str = Depends(verify_api_key),
):
    """
    Query the knowledge base.

    Retrieves relevant documents and generates an answer.
    """
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            rag.query_streaming(
                request.question,
                namespace=request.namespace,
            ),
            media_type="text/event-stream",
        )

    try:
        response = await rag.query(
            question=request.question,
            namespace=request.namespace,
            mode=request.mode,
            use_agent=request.use_agent,
        )

        return QueryResponse(
            question=request.question,
            answer=response.answer,
            citations=[
                {
                    "chunk_id": str(c.chunk_id),
                    "snippet": c.snippet,
                    "source_index": c.source_index,
                }
                for c in response.citations
            ],
            sources_used=response.sources_used,
            confidence=0.8,  # From generation response
            latency_ms=response.latency_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve(
    request: RetrieveRequest,
    rag: Hiero = Depends(get_rag),
    tenant_id: str = Depends(verify_api_key),
):
    """
    Retrieve relevant chunks without generation.

    Useful for search interfaces or custom generation pipelines.
    """
    try:
        results = await rag.retrieve(
            query=request.query,
            top_k=request.top_k,
            namespace=request.namespace,
            strategy=request.strategy,
            metadata_filter=request.metadata_filter,
        )

        return RetrieveResponse(
            query=request.query,
            results=results,
            total=len(results),
            latency_ms=0,  # Add timing
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Documents"])
async def list_documents(
    namespace: str = "default",
    limit: int = 100,
    offset: int = 0,
    rag: Hiero = Depends(get_rag),
    tenant_id: str = Depends(verify_api_key),
):
    """List documents in a namespace."""
    return await rag.list_documents(namespace, limit, offset)


@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    rag: Hiero = Depends(get_rag),
    tenant_id: str = Depends(verify_api_key),
):
    """Delete a document and its chunks."""
    success = await rag.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": document_id}


@app.get("/stats", tags=["System"])
async def get_stats(
    namespace: str = "default",
    rag: Hiero = Depends(get_rag),
    tenant_id: str = Depends(verify_api_key),
):
    """Get statistics for a namespace."""
    return await rag.get_stats(namespace)


# Startup/Shutdown
@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    app.state.start_time = time.time()
    app.state.rag = Hiero(
        database_url=os.environ["DATABASE_URL"],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    await app.state.rag.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await app.state.rag.close()
```

### Serverless Deployment

```python
# handler.py - AWS Lambda / Vercel deployment
from mangum import Mangum
from api.main import app

# Lambda handler
handler = Mangum(app, lifespan="off")


# For Vercel
# vercel.json:
# {
#   "builds": [{"src": "handler.py", "use": "@vercel/python"}],
#   "routes": [{"src": "/(.*)", "dest": "handler.py"}]
# }
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/query")
@limiter.limit("10/minute")  # 10 queries per minute
async def query(...):
    ...


@app.post("/ingest")
@limiter.limit("5/minute")  # 5 ingestions per minute
async def ingest(...):
    ...
```

## API Endpoints Summary

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/health` | GET | Health check | - |
| `/ingest` | POST | Ingest document | 5/min |
| `/query` | POST | Query with RAG | 10/min |
| `/retrieve` | POST | Retrieve chunks | 30/min |
| `/documents` | GET | List documents | 30/min |
| `/documents/{id}` | DELETE | Delete document | 10/min |
| `/stats` | GET | Get statistics | 30/min |

## Dependencies

```toml
[project.dependencies]
fastapi = "^0.109"
uvicorn = {extras = ["standard"], version = "^0.27"}
mangum = "^0.17"            # Serverless adapter
slowapi = "^0.1"            # Rate limiting
python-multipart = "^0.0.9" # File uploads
```

## Testing

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ingest():
    response = client.post(
        "/ingest",
        json={"content": "Test content", "namespace": "test"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    assert "document_id" in response.json()


def test_query():
    response = client.post(
        "/query",
        json={"question": "What is this about?", "namespace": "test"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    assert "answer" in response.json()
```
