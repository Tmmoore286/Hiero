# Hiero Phased Build Plan

## Executive Summary

This document outlines a phased approach to building **Hiero**, a general-purpose RAG framework. Each phase delivers working, testable functionality.

> *Named after King Hiero II of Syracuse, Sicily—Hiero retrieves authentic, grounded answers from your documents.*

---

## Phase Overview

| Phase | Focus | Deliverables | Dependencies |
|-------|-------|--------------|--------------|
| **0** | Foundation | Project setup, database, config | - |
| **1** | Core Pipeline | Ingest → Chunk → Embed → Store → Retrieve → Generate | Phase 0 |
| **2** | Production Quality | Hybrid retrieval, reranking, API | Phase 1 |
| **3** | Advanced Features | Agentic RAG, full evaluation | Phase 2 |
| **4** | Integrations | ARCnet adapter, LangChain/LlamaIndex (optional) | Phase 3 |

---

## Phase 0: Foundation

**Goal**: Set up project infrastructure and database schema.

### Tasks

#### 0.1 Project Structure
- [ ] Create project directory `hiero/`
- [ ] Initialize Python project with `pyproject.toml`
- [ ] Set up `src/hiero/` package structure
- [ ] Configure development tools (ruff, mypy, pytest)
- [ ] Create `.env.example` with required variables
- [ ] Add `docker-compose.yml` for local PostgreSQL + pgvector

```toml
# pyproject.toml
[project]
name = "hiero"
version = "0.1.0"
description = "Research-grade Retrieval-Augmented Generation framework"
requires-python = ">=3.11"
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.29",
    "pgvector>=0.2",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "openai>=1.14",
    "tiktoken>=0.7",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff", "mypy"]
api = ["fastapi>=0.109", "uvicorn[standard]>=0.27"]
all = ["hiero[dev,api]"]
```

#### 0.2 Configuration System
- [ ] Create `config.py` with Pydantic Settings
- [ ] Support environment variables and `.env` files
- [ ] Validate required API keys on startup

#### 0.3 Database Setup
- [ ] Create Alembic migration structure
- [ ] Write initial migration with schema from spec
- [ ] Add pgvector extension initialization
- [ ] Create HNSW indexes for common dimensions
- [ ] Test migration up/down

#### 0.4 Base Models
- [ ] Define SQLAlchemy ORM models (DocumentORM, ChunkORM)
- [ ] Define Pydantic schemas for API contracts
- [ ] Add base repository interface

### Exit Criteria
- [x] `docker-compose up` starts PostgreSQL with pgvector
- [x] `alembic upgrade head` creates all tables
- [x] Unit tests pass for config loading

---

## Phase 1: Core Pipeline

**Goal**: End-to-end RAG pipeline: ingest a document, query it, get an answer.

### Tasks

#### 1.1 Ingestion (Basic)
- [ ] Implement `IngestorProtocol` base class
- [ ] Build `PDFIngestor` using pymupdf
- [ ] Build `TextIngestor` for plain text/markdown
- [ ] Add file type detection with python-magic
- [ ] Create sync ingestion path (skip queue for now)

**Test**: Ingest a PDF, verify document stored in DB

#### 1.2 Chunking (Semantic)
- [ ] Implement `Tokenizer` wrapper around tiktoken
- [ ] Build `SemanticChunker` with sentence boundaries
- [ ] Add chunk overlap handling
- [ ] Preserve basic metadata (index, char offsets)

**Test**: Chunk a document, verify chunks are reasonable size

#### 1.3 Embedding (OpenAI)
- [ ] Implement `EmbedderProtocol` base class
- [ ] Build `OpenAIEmbedder` with batching
- [ ] Add retry logic with tenacity
- [ ] Implement basic rate limiting

**Test**: Embed text, verify vector dimensions match

#### 1.4 Vector Store (Basic CRUD)
- [ ] Implement `PgVectorStore` repository
- [ ] Add `insert_document()` with chunks
- [ ] Add `search_dense()` with cosine similarity
- [ ] Add `delete_document()` and `get_document()`

**Test**: Insert document, search, verify results

#### 1.5 Retriever (Dense Only)
- [ ] Implement `RetrieverProtocol` base class
- [ ] Build `DenseRetriever` wrapping vector store
- [ ] Add query embedding before search
- [ ] Return ranked `RetrievedChunk` objects

**Test**: Query returns relevant chunks

#### 1.6 Generator (Basic)
- [ ] Implement `GeneratorProtocol` base class
- [ ] Build `GroundedGenerator` with OpenAI
- [ ] Create basic prompt template with sources
- [ ] Parse inline citations from response

**Test**: Generate answer with citations

#### 1.7 Library Interface (Minimal)
- [ ] Create `Hiero` class with basic methods
- [ ] Implement `ingest()`, `retrieve()`, `query()`
- [ ] Add async context manager support

**Test**: End-to-end: ingest PDF → query → get answer

### Exit Criteria
```python
from hiero import Hiero

async with Hiero(database_url=..., openai_api_key=...) as hiero:
    await hiero.ingest("document.pdf")
    response = await hiero.query("What is this about?")
    print(response.answer)  # Works!
```

---

## Phase 2: Production Quality

**Goal**: Hybrid retrieval, reranking, REST API, proper error handling.

### Tasks

#### 2.1 Sparse Retrieval
- [ ] Add `content_tsvector` column generation
- [ ] Implement `search_sparse()` using ts_rank
- [ ] Test full-text search independently

#### 2.2 Hybrid Retrieval
- [ ] Implement `HybridRetriever` with fusion
- [ ] Add RRF score fusion
- [ ] Add configurable alpha weighting
- [ ] Benchmark dense vs sparse vs hybrid

#### 2.3 LLM Reranker
- [ ] Implement `RerankerProtocol` base class
- [ ] Build `LLMReranker` with pointwise scoring
- [ ] Add to retrieval pipeline (retrieve → rerank)
- [ ] Make reranking optional via config

#### 2.4 Additional Ingestors
- [ ] Build `DOCXIngestor` using python-docx
- [ ] Build `URLIngestor` using trafilatura
- [ ] Add async ingestion queue (PostgreSQL-backed)

#### 2.5 Adaptive Chunking
- [ ] Implement `AdaptiveChunker` with doc type detection
- [ ] Add `RecursiveChunker` for structured docs
- [ ] Add `FixedChunker` as fallback

#### 2.6 Embedding Cache
- [ ] Create `embedding_cache` table
- [ ] Implement `PostgresEmbeddingCache`
- [ ] Add cache lookup before API calls
- [ ] Add cache eviction for size limits

#### 2.7 REST API
- [ ] Create FastAPI application
- [ ] Implement `/ingest` endpoint
- [ ] Implement `/query` endpoint
- [ ] Implement `/retrieve` endpoint
- [ ] Add `/health` and `/stats` endpoints
- [ ] Add API key authentication
- [ ] Add rate limiting with slowapi

#### 2.8 Streaming Support
- [ ] Add `generate_streaming()` to generator
- [ ] Add streaming endpoint to API
- [ ] Test with SSE client

#### 2.9 Error Handling & Logging
- [ ] Add structured logging throughout
- [ ] Implement proper exception hierarchy
- [ ] Add request ID tracking
- [ ] Create error response schemas

### Exit Criteria
- [x] API serves queries with < 500ms latency (excluding LLM)
- [x] Hybrid retrieval improves Recall@5 over dense-only
- [x] Reranking improves precision in top results
- [x] All endpoints documented in OpenAPI

---

## Phase 3: Advanced Features

**Goal**: Agentic RAG, comprehensive evaluation, production hardening.

### Tasks

#### 3.1 Agent Controller
- [ ] Implement ReAct loop structure
- [ ] Build `RetrieveTool` for agent
- [ ] Build `CalculateTool` for math
- [ ] Build `SummarizeTool` for content
- [ ] Add query decomposition
- [ ] Add self-evaluation step
- [ ] Implement step tracing/logging

#### 3.2 Multi-hop Retrieval
- [ ] Add iterative retrieval in agent
- [ ] Implement content-based query expansion
- [ ] Track and deduplicate across hops

#### 3.3 Evaluation Suite - Retrieval
- [ ] Implement Recall@k calculator
- [ ] Implement Precision@k calculator
- [ ] Implement nDCG@k calculator
- [ ] Implement MRR calculator
- [ ] Create evaluation runner

#### 3.4 Evaluation Suite - Generation
- [ ] Implement LLM-as-judge evaluator
- [ ] Add factuality scoring
- [ ] Add groundedness scoring
- [ ] Add relevance scoring

#### 3.5 CLI Tools
- [ ] Create `hiero` CLI entry point
- [ ] Add `hiero eval` command
- [ ] Add `hiero ingest` command
- [ ] Add `hiero serve` command

#### 3.6 Hallucination Detection
- [ ] Implement `HallucinationDetector`
- [ ] Add claim extraction
- [ ] Add source verification
- [ ] Optional filtering in generation

#### 3.7 Multiple Embedding Providers
- [ ] Add `CohereEmbedder`
- [ ] Add `LocalEmbedder` (sentence-transformers)
- [ ] Create `EmbedderFactory` for provider selection

#### 3.8 Serverless Deployment
- [ ] Add Mangum adapter for Lambda
- [ ] Create deployment configs
- [ ] Test cold start performance
- [ ] Document deployment process

### Exit Criteria
- [x] Agent can answer multi-hop questions
- [x] Evaluation suite produces reproducible metrics
- [x] Can deploy to AWS Lambda successfully

---

## Phase 4: Integrations (Optional)

**Goal**: Platform-specific integrations as optional extras.

### Tasks

#### 4.1 ARCnet Integration
- [ ] Create `hiero/integrations/arcnet/` module
- [ ] Implement `DoctrineStore` for doctrine documents
- [ ] Implement `AgentSelector` support
- [ ] Add MOS code filtering
- [ ] Compute doctrine centroids
- [ ] Test with ARCnet backend

#### 4.2 LangChain Adapter (Future)
- [ ] Create retriever adapter
- [ ] Create document loader adapter
- [ ] Publish as optional extra

#### 4.3 LlamaIndex Adapter (Future)
- [ ] Create index adapter
- [ ] Create query engine adapter
- [ ] Publish as optional extra

### Exit Criteria
- [x] ARCnet can use Hiero as doctrine store
- [x] Integrations are optional (core works without them)

---

## Implementation Order (Recommended)

```
Week 1-2: Phase 0 + Phase 1.1-1.4
├── Project setup
├── Database schema
├── Basic ingestion
├── Chunking
└── Embedding

Week 3-4: Phase 1.5-1.7
├── Vector store CRUD
├── Dense retrieval
├── Basic generation
└── Library interface (MVP working!)

Week 5-6: Phase 2.1-2.4
├── Sparse retrieval
├── Hybrid retrieval
├── LLM reranker
└── Additional ingestors

Week 7-8: Phase 2.5-2.9
├── Adaptive chunking
├── Embedding cache
├── REST API
├── Streaming
└── Error handling

Week 9-10: Phase 3.1-3.4
├── Agent controller
├── Multi-hop retrieval
├── Retrieval evaluation
└── Generation evaluation

Week 11-12: Phase 3.5-3.8 + Phase 4
├── CLI tools
├── Hallucination detection
├── Multiple providers
├── Serverless deployment
└── Integrations (optional)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| pgvector performance at scale | Benchmark early with 100k+ vectors; tune HNSW parameters |
| LLM API costs during development | Use smaller models (gpt-4o-mini) for testing; cache aggressively |
| Complex async code bugs | Comprehensive async testing; use structured concurrency |
| Scope creep | Strict phase gates; MVP first mentality |

---

## Success Metrics

### Phase 1 (MVP)
- End-to-end query works
- < 5 second response time

### Phase 2 (Production)
- Recall@5 > 0.7 on test dataset
- API handles 10 QPS
- 99% uptime in testing

### Phase 3 (Advanced)
- Agent answers multi-hop questions correctly > 70%
- Evaluation correlates with human judgment
- Deploy to serverless in < 10 minutes

### Phase 4 (Integrations)
- Integrations work without modifying core
- Clear documentation for each integration

---

## Dependencies Summary

```
Phase 0 Dependencies:
├── PostgreSQL 15+
├── pgvector extension
└── Python 3.11+

Phase 1 Dependencies:
├── openai
├── tiktoken
├── pymupdf
├── sqlalchemy[asyncio]
└── asyncpg

Phase 2 Dependencies (additional):
├── fastapi
├── uvicorn
├── slowapi
├── python-docx
├── trafilatura
└── tenacity

Phase 3 Dependencies (additional):
├── sentence-transformers (optional)
├── cohere (optional)
├── click
└── mangum

Phase 4 Dependencies:
└── Integration-specific (optional extras)
```

---

## Getting Started

```bash
# Clone and setup
git clone <repo>
cd hiero
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Start database
docker-compose up -d

# Run migrations
alembic upgrade head

# Run tests
pytest

# Start development
# Begin with Phase 0 tasks...
```

---

## Quick Reference

### Library Usage (Phase 1+)
```python
from hiero import Hiero

async with Hiero(database_url="...", openai_api_key="...") as h:
    # Ingest
    doc_id = await h.ingest("document.pdf")

    # Query
    response = await h.query("What is the main topic?")
    print(response.answer)
    print(response.citations)

    # Retrieve only
    chunks = await h.retrieve("search query", top_k=10)
```

### API Usage (Phase 2+)
```bash
# Start server
hiero serve --port 8000

# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

### CLI Usage (Phase 3+)
```bash
# Ingest documents
hiero ingest ./documents/

# Run evaluation
hiero eval --dataset test_queries.json --output results.json

# Start API server
hiero serve --port 8000
```
