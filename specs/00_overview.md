# Hiero Technical Specifications

## Overview

**Hiero** is a research-grade, production-ready Retrieval-Augmented Generation framework designed for 2025.

> *Named after King Hiero II of Syracuse, Sicily, who challenged Archimedes to solve the first "information authenticity" problem—detecting whether his crown was pure gold. Hiero continues that tradition: retrieving authentic, grounded answers from your documents.*

## Design Principles

1. **Pluggable by Default** - Every major component uses abstract interfaces
2. **PostgreSQL as SSOT** - Single database for vectors, metadata, and full-text search
3. **Cloud-Native** - Designed for serverless deployment (AWS Lambda, Vercel, etc.)
4. **General-Purpose** - Works for any domain; optional integrations for specific platforms
5. **Research-Grade** - Comprehensive evaluation, ablation support, full observability

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Hiero Core                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐    │
│  │  Ingestion   │───▶│   Chunking   │───▶│   Embedding Layer      │    │
│  │  (Hybrid)    │    │  (Adaptive)  │    │   (Pluggable)          │    │
│  └──────────────┘    └──────────────┘    └────────────────────────┘    │
│         │                                           │                   │
│         ▼                                           ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    pgvector (PostgreSQL 15+)                    │   │
│  │    vectors (HNSW) + metadata (JSONB) + full-text (tsvector)     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐    │
│  │   Hybrid     │───▶│  LLM-based   │───▶│   Agent Controller     │    │
│  │  Retriever   │    │   Reranker   │    │   (Full Agent)         │    │
│  └──────────────┘    └──────────────┘    └────────────────────────┘    │
│         │                                           │                   │
│         ▼                                           ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Generation Layer                            │   │
│  │        Structured prompts • Grounding • Citations               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                         Exposure Layer                                  │
│  ┌─────────────────────────┐      ┌─────────────────────────────────┐  │
│  │    Python Library       │      │    FastAPI REST API             │  │
│  │    (direct import)      │      │    (serverless-ready)           │  │
│  └─────────────────────────┘      └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Index

| # | Component | Spec File | Priority |
|---|-----------|-----------|----------|
| 1 | Ingestion | `01_ingestion.md` | P0 |
| 2 | Chunking | `02_chunking.md` | P0 |
| 3 | Embedding Layer | `03_embedding.md` | P0 |
| 4 | Vector Store | `04_vector_store.md` | P0 |
| 5 | Hybrid Retriever | `05_retriever.md` | P0 |
| 6 | LLM Reranker | `06_reranker.md` | P1 |
| 7 | Agent Controller | `07_agent.md` | P1 |
| 8 | Generation Layer | `08_generation.md` | P0 |
| 9 | API Layer | `09_api.md` | P0 |
| 10 | Evaluation Suite | `10_evaluation.md` | P1 |

## Tech Stack

### Core
- **Python**: 3.11+
- **Database**: PostgreSQL 15+ with pgvector extension
- **API Framework**: FastAPI 0.104+
- **Async**: asyncio + asyncpg

### ML/AI
- **Embeddings**: OpenAI, Cohere, sentence-transformers (pluggable)
- **LLM**: OpenAI GPT-4/4o, Anthropic Claude (pluggable)
- **Sparse Retrieval**: PostgreSQL tsvector (built-in BM25-like scoring)

### Infrastructure
- **ORM**: SQLAlchemy 2.0 (async)
- **Migrations**: Alembic
- **Validation**: Pydantic 2.0
- **Testing**: pytest + pytest-asyncio
- **Deployment**: Docker, AWS Lambda (Mangum), or Vercel

## Directory Structure

```
hiero/
├── pyproject.toml
├── alembic.ini
├── docker-compose.yml
├── .env.example
│
├── src/
│   └── hiero/
│       ├── __init__.py
│       ├── config.py              # Settings via pydantic-settings
│       │
│       ├── core/                  # Core RAG components
│       │   ├── ingestion/
│       │   │   ├── __init__.py
│       │   │   ├── base.py        # IngestorProtocol
│       │   │   ├── file.py        # PDF, DOCX, MD, TXT
│       │   │   ├── web.py         # URL fetching
│       │   │   └── queue.py       # Async job queue
│       │   │
│       │   ├── chunking/
│       │   │   ├── __init__.py
│       │   │   ├── base.py        # ChunkerProtocol
│       │   │   ├── semantic.py    # Sentence-boundary chunking
│       │   │   ├── recursive.py   # Header/section aware
│       │   │   ├── fixed.py       # Token windows
│       │   │   └── adaptive.py    # Doc-type detection + routing
│       │   │
│       │   ├── embedding/
│       │   │   ├── __init__.py
│       │   │   ├── base.py        # EmbedderProtocol
│       │   │   ├── openai.py      # OpenAI embeddings
│       │   │   ├── cohere.py      # Cohere embeddings
│       │   │   └── local.py       # sentence-transformers
│       │   │
│       │   ├── storage/
│       │   │   ├── __init__.py
│       │   │   ├── models.py      # SQLAlchemy models
│       │   │   └── repository.py  # Vector CRUD operations
│       │   │
│       │   ├── retrieval/
│       │   │   ├── __init__.py
│       │   │   ├── base.py        # RetrieverProtocol
│       │   │   ├── dense.py       # Vector similarity
│       │   │   ├── sparse.py      # tsvector full-text
│       │   │   └── hybrid.py      # Weighted combination
│       │   │
│       │   ├── reranking/
│       │   │   ├── __init__.py
│       │   │   ├── base.py        # RerankerProtocol
│       │   │   └── llm.py         # LLM-based reranking
│       │   │
│       │   ├── agent/
│       │   │   ├── __init__.py
│       │   │   ├── controller.py  # ReAct loop
│       │   │   ├── tools.py       # Retrieve, calculate, etc.
│       │   │   ├── planner.py     # Query decomposition
│       │   │   └── memory.py      # Conversation state
│       │   │
│       │   └── generation/
│       │       ├── __init__.py
│       │       ├── base.py        # GeneratorProtocol
│       │       ├── generator.py   # LLM generation with grounding
│       │       └── prompts.py     # Prompt templates
│       │
│       ├── api/                   # REST API
│       │   ├── __init__.py
│       │   ├── main.py            # FastAPI app
│       │   ├── routes/
│       │   │   ├── ingest.py
│       │   │   ├── query.py
│       │   │   └── health.py
│       │   └── schemas.py         # Pydantic request/response
│       │
│       ├── evaluation/            # Eval suite
│       │   ├── __init__.py
│       │   ├── retrieval.py       # Recall@k, nDCG, MRR
│       │   ├── generation.py      # Factuality, groundedness
│       │   └── runner.py          # One-command eval
│       │
│       └── integrations/          # Optional platform integrations
│           ├── __init__.py
│           ├── arcnet/            # ARCnet integration
│           │   ├── __init__.py
│           │   ├── doctrine.py    # Doctrine store
│           │   └── selector.py    # Agent selection support
│           ├── langchain/         # Future: LangChain adapter
│           └── llamaindex/        # Future: LlamaIndex adapter
│
├── migrations/                    # Alembic migrations
│   └── versions/
│
├── tests/
│   ├── conftest.py
│   ├── test_chunking/
│   ├── test_embedding/
│   ├── test_retrieval/
│   └── test_integration/
│
└── notebooks/
    ├── 01_getting_started.ipynb
    ├── 02_chunking_ablation.ipynb
    └── 03_evaluation.ipynb
```

## Configuration

All configuration via environment variables with Pydantic settings:

```python
class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://user:pass@localhost/hiero"

    # Embedding providers
    openai_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    default_embedding_provider: str = "openai"
    default_embedding_model: str = "text-embedding-3-small"

    # LLM providers
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o"

    # Retrieval defaults
    default_top_k: int = 10
    rerank_top_k: int = 5
    hybrid_alpha: float = 0.7  # weight for dense vs sparse

    # Chunking defaults
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64

    model_config = SettingsConfigDict(env_file=".env")
```

## Integrations

Hiero is designed as a general-purpose RAG framework. Platform-specific integrations are optional and isolated in the `integrations/` module.

### Available Integrations

| Integration | Status | Description |
|-------------|--------|-------------|
| ARCnet | Planned | Doctrine storage, agent selection support |
| LangChain | Future | Adapter for LangChain pipelines |
| LlamaIndex | Future | Adapter for LlamaIndex workflows |

### Adding Custom Integrations

Integrations should:
1. Live in `hiero/integrations/{name}/`
2. Use only public Hiero APIs
3. Be optional (not required for core functionality)
4. Have their own dependencies (extras in pyproject.toml)
