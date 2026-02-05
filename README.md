# Hiero

A research-grade Retrieval-Augmented Generation (RAG) framework built on PostgreSQL, pgvector, and OpenAI. Hiero implements a full RAG pipeline — ingestion, chunking, embedding, hybrid retrieval, reranking, grounded generation, and agentic reasoning — as a single, modular Python library.

Named after [King Hiero II of Syracuse](https://en.wikipedia.org/wiki/Hiero_II_of_Syracuse), who posed the problem that led Archimedes to his principle of displacement — the original "information authenticity" challenge.

## Architecture

```
Document ─→ Ingest ─→ Chunk ─→ Embed ─→ PostgreSQL + pgvector
                                               │
                                    ┌──────────┴──────────┐
Query ─→ Embed ─→              Dense Search        Sparse Search
                                    │                     │
                                    └───── RRF Fusion ────┘
                                               │
                                           Rerank (optional)
                                               │
                                    Grounded Generation ─→ Answer + Citations
                                               │
                                       Agent (optional)
                                    ReAct loop with tools
                                    for multi-hop reasoning
```

### Design Principles

- **Single source of truth.** PostgreSQL handles vectors (pgvector HNSW), full-text search (tsvector), metadata (JSONB), and relational data. No external vector database required.
- **Hybrid retrieval by default.** Dense (semantic) and sparse (keyword) searches run in parallel, fused via Reciprocal Rank Fusion. Configurable weights and strategy selection.
- **Grounded generation.** Every answer includes source citations. LLM-as-judge evaluation scores factuality, groundedness, and relevance.
- **Pluggable components.** Protocol-based interfaces for embedders, retrievers, generators, and agent tools. Swap providers without changing pipeline code.
- **Async-first.** All I/O is asynchronous (asyncpg, httpx, SQLAlchemy async). Concurrent embedding, retrieval, and generation where possible.

## Components

| Module | Description |
|--------|-------------|
| `ingestion` | Multi-format document ingestion: PDF (PyMuPDF), DOCX, plain text, URLs (trafilatura) |
| `chunking` | Semantic, fixed-size, and adaptive chunking strategies with token-aware splitting |
| `embedding` | Pluggable embedding providers (OpenAI, extensible to Cohere/local models) with PostgreSQL-backed cache |
| `storage` | pgvector store with HNSW indexing, tsvector full-text search, namespace isolation, CRUD operations |
| `retrieval` | Dense, sparse, and hybrid retrieval with RRF or weighted-sum fusion |
| `reranking` | Optional LLM-based reranking to refine top-k results |
| `generation` | Grounded generation with structured citation extraction |
| `agent` | ReAct reasoning loop with tool use: retrieve, retrieve_more, calculate, summarize, finish |
| `evaluation` | Retrieval metrics (Recall@k, Precision@k, nDCG, MRR, MAP) and generation metrics (factuality, groundedness, relevance, correctness) via LLM-as-judge |

## Technology Stack

- **Language:** Python 3.11+
- **Database:** PostgreSQL 15 with pgvector extension
- **ORM / Migrations:** SQLAlchemy 2.0 (async) + Alembic
- **LLM / Embeddings:** OpenAI API (GPT-4o, text-embedding-3-small)
- **Document Parsing:** PyMuPDF, python-docx, trafilatura, BeautifulSoup
- **Validation:** Pydantic 2.0
- **Async I/O:** asyncpg, httpx, tenacity (retry logic)
- **Testing:** pytest + pytest-asyncio
- **Linting / Types:** Ruff, mypy
- **API (optional):** FastAPI + uvicorn

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension (or Docker)
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/Tmmoore286/Hiero.git
cd Hiero

# Start PostgreSQL with pgvector
docker compose up -d

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your database URL and OpenAI API key

# Run database migrations
alembic upgrade head
```

### Usage

```python
from hiero import Hiero

async with Hiero() as h:
    # Ingest a document
    await h.ingest("research_paper.pdf")

    # Query with grounded generation
    result = await h.query("What are the key findings?")
    print(result.answer)
    for citation in result.citations:
        print(f"  [{citation.chunk_id}] {citation.text}")

    # Agent query for multi-hop reasoning
    result = await h.agent_query("Compare the methodology in sections 2 and 4")
```

### Agent CLI

```bash
python -m demo.agent_cli \
  --ingest paper.pdf \
  --question "What methodology was used?" \
  --self-eval
```

### Running Tests

```bash
pytest
```

## Project Structure

```
src/hiero/
├── agent/              # ReAct agent loop and tools
│   ├── react.py        #   Reason → Act → Observe cycle
│   ├── llm.py          #   LLM interface for agent reasoning
│   └── tools/          #   retrieve, calculate, summarize, finish
├── chunking/           # Document chunking strategies
│   ├── semantic.py     #   Embedding-based boundary detection
│   ├── adaptive.py     #   Content-aware chunk sizing
│   └── fixed.py        #   Token-count splitting
├── embedding/          # Embedding providers + caching
│   ├── openai.py       #   OpenAI embeddings with batching
│   ├── cache.py        #   PostgreSQL embedding cache
│   └── factory.py      #   Provider factory
├── evaluation/         # RAG evaluation suite
│   ├── metrics.py      #   Retrieval + generation metrics
│   ├── runner.py       #   Batch evaluation harness
│   └── dataset.py      #   Eval dataset models
├── generation/         # Grounded generation
│   └── grounded.py     #   Citation-aware response generation
├── ingestion/          # Document ingestion
│   ├── pdf.py          #   PDF parsing (PyMuPDF)
│   ├── docx.py         #   DOCX parsing
│   ├── url.py          #   Web page extraction
│   └── router.py       #   Format detection + routing
├── retrieval/          # Retrieval strategies
│   ├── dense.py        #   Vector similarity search
│   └── hybrid.py       #   Dense + sparse fusion (RRF)
├── reranking/          # Result reranking
│   └── llm.py          #   LLM-based relevance reranking
├── storage/            # Database layer
│   ├── models.py       #   SQLAlchemy ORM models
│   └── repository.py   #   pgvector store operations
├── config.py           # Pydantic settings
├── db.py               # Async database utilities
└── hiero.py            # Main library interface
```

## Evaluation

Hiero includes a built-in evaluation framework for measuring retrieval and generation quality:

**Retrieval metrics:** Recall@k, Precision@k, nDCG@k, MRR, MAP — computed against ground-truth relevant document sets.

**Generation metrics (LLM-as-judge):** Factuality (are claims supported by context?), groundedness (are sources properly attributed?), relevance (does the answer address the question?), correctness (does it match ground truth?).

```python
from hiero.evaluation import RAGEvaluator, EvalDataset

evaluator = RAGEvaluator(hiero_instance)
results = await evaluator.run(dataset, metrics=["recall@5", "factuality"])
```

## Specifications

Detailed technical specifications for each component are in [`/specs`](specs/). These were written before implementation to define interfaces, data models, and behavioral contracts.

## License

MIT
