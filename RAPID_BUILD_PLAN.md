# Hiero Rapid Build Plan

## Goal

Ship a working RAG implementation in ~4 weeks with:
- End-to-end functionality (ingest → query → answer)
- Interactive demo
- Clean, documented code

---

## 4-Week Sprint Plan

### Week 1: Foundation + Core Pipeline

**Goal**: Ingest a PDF and store chunks with embeddings.

```
Day 1-2: Project setup
├── pyproject.toml with dependencies
├── docker-compose.yml (PostgreSQL + pgvector)
├── Basic directory structure: src/hiero/
└── Config system with pydantic-settings

Day 3-4: Database + Models
├── Alembic migration for documents/chunks tables
├── SQLAlchemy ORM models
└── Basic repository pattern

Day 5-7: Ingest + Chunk + Embed
├── PDF ingestion with pymupdf
├── Semantic chunking with tiktoken
├── OpenAI embeddings with batching
└── Store in pgvector
```

**Exit Criteria**: Can run `python -m hiero.ingest document.pdf` and see chunks in database.

---

### Week 2: Retrieval + Generation

**Goal**: Query the system and get answers with citations.

```
Day 1-2: Dense retrieval
├── Vector similarity search
├── Query embedding
└── Return ranked chunks

Day 3-4: Basic generation
├── Grounded generation prompt
├── OpenAI completion with context
└── Parse citations from response

Day 5-7: Library interface
├── Hiero class with ingest/query methods
├── Async context manager
└── End-to-end test
```

**Exit Criteria**:
```python
async with Hiero(...) as h:
    await h.ingest("doc.pdf")
    response = await h.query("What is this about?")
    print(response.answer)  # Works!
```

---

### Week 3: Polish + Demo

**Goal**: Interactive demo with hybrid retrieval.

```
Day 1-2: Hybrid retrieval
├── Add sparse search with tsvector
├── RRF fusion
└── Benchmark improvement

Day 3-4: Demo UI
├── Streamlit or Gradio app
├── Upload PDF
├── Ask questions
├── Show answers with sources

Day 5-7: Documentation
├── README with architecture diagram
├── Quickstart guide
├── Record demo video
```

**Exit Criteria**: Someone can clone the repo, run `docker-compose up` + `streamlit run demo.py`, and use it.

---

### Week 4: Advanced Features (pick 1-2)

| Option | Complexity | Effort |
|--------|------------|--------|
| **LLM Reranking** | Medium | 2 days |
| **REST API** | Medium | 2 days |
| **Eval metrics** | Medium | 3 days |
| **Multiple embedding providers** | Low | 2 days |
| **Basic agent (ReAct)** | High | 3-4 days |

---

## Priority Tiers

```
Must Have (Weeks 1-2):
├── Working ingest → query pipeline
├── PostgreSQL + pgvector storage
├── OpenAI embeddings + generation
└── Python library interface

Should Have (Week 3):
├── Hybrid retrieval (dense + sparse)
├── Demo UI (CLI or Streamlit)
└── README + quickstart

Nice to Have (Week 4):
├── Reranking
├── REST API
└── Evaluation metrics
```

---

## Target File Structure

```
hiero/
├── src/hiero/
│   ├── __init__.py           # Hiero class export
│   ├── config.py             # Settings
│   ├── models.py             # SQLAlchemy + Pydantic
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf.py
│   │   └── chunker.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── openai.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── pgvector.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   └── hybrid.py
│   └── generation/
│       ├── __init__.py
│       └── grounded.py
├── demo/
│   └── app.py                # Streamlit demo
├── tests/
├── specs/                    # Design documentation
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Key Differentiators

1. **Design-first**: Comprehensive specs before implementation
2. **Single database**: PostgreSQL + pgvector (no separate vector DB)
3. **Hybrid retrieval**: Dense + sparse fusion, not just vector search
4. **Grounded generation**: Citations required, hallucination-resistant
5. **Clean architecture**: Pluggable components via protocols

---

## Getting Started

```bash
# Create project structure
mkdir -p src/hiero/{ingestion,embedding,storage,retrieval,generation}
touch src/hiero/__init__.py

# Initialize project
cd /Users/timmoore/Documents/projects/Hiero
python -m venv .venv
source .venv/bin/activate

# Start with:
# 1. Create pyproject.toml
# 2. Create docker-compose.yml
# 3. Begin Phase 0 from BUILD_PLAN.md
```

---

## Development Principles

- Ship working code over perfect code
- End-to-end flow before optimizations
- Demo-able progress at each milestone
- Specs guide implementation, not constrain it
