import sys
import uuid
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero import Hiero
from hiero.agent import AgentResponse
from hiero.generation import GenerationResponse, InlineCitation
from hiero.reranking import RerankResult, RerankedChunk, RerankingMethod
from hiero.retrieval import RetrievedChunk, RetrievalConfig, RetrievalResult, RetrievalStrategy


@pytest.mark.asyncio
async def test_hiero_query_wires_retrieval_and_generation():
    hiero = Hiero(database_url="postgresql+asyncpg://u:p@localhost:5432/db", openai_api_key="sk-test")
    await hiero.initialize()

    fake_chunk_id = uuid.uuid4()
    fake_doc_id = uuid.uuid4()

    async def fake_retrieve(rq):
        return RetrievalResult(
            query=rq.text,
            chunks=[
                RetrievedChunk(
                    chunk_id=fake_chunk_id,
                    document_id=fake_doc_id,
                    content="context",
                    score=0.9,
                    rank=1,
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    dense_score=0.9,
                    chunk_index=0,
                )
            ],
            total_candidates=1,
            strategy_used=RetrievalStrategy.DENSE,
            latency_ms=1.0,
            dense_candidates=1,
        )

    async def fake_generate(req):
        return GenerationResponse(
            query=req.query,
            response="answer [1]",
            citations=[
                InlineCitation(
                    source_index=0,
                    chunk_id=fake_chunk_id,
                    quoted_text="",
                    start_pos=7,
                    end_pos=10,
                )
            ],
            sources_used=[fake_chunk_id],
            sources_provided=1,
            grounding_score=1.0,
            confidence=0.9,
            tokens_used=0,
            latency_ms=1.0,
        )

    hiero.retriever = SimpleNamespace(retrieve=fake_retrieve)
    hiero.generator = SimpleNamespace(generate=fake_generate)

    resp = await hiero.query("q")
    assert resp.answer == "answer [1]"
    assert resp.citations[0].chunk_id == fake_chunk_id


@pytest.mark.asyncio
async def test_hiero_query_applies_reranking():
    hiero = Hiero(database_url="postgresql+asyncpg://u:p@localhost:5432/db", openai_api_key="sk-test")
    await hiero.initialize()

    doc_id = uuid.uuid4()
    a_id = uuid.uuid4()
    b_id = uuid.uuid4()

    async def fake_retrieve(rq):
        return RetrievalResult(
            query=rq.text,
            chunks=[
                RetrievedChunk(
                    chunk_id=a_id,
                    document_id=doc_id,
                    content="a",
                    score=0.9,
                    rank=1,
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    dense_score=0.9,
                    chunk_index=0,
                ),
                RetrievedChunk(
                    chunk_id=b_id,
                    document_id=doc_id,
                    content="b",
                    score=0.8,
                    rank=2,
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    dense_score=0.8,
                    chunk_index=1,
                ),
            ],
            total_candidates=2,
            strategy_used=RetrievalStrategy.DENSE,
            latency_ms=1.0,
            dense_candidates=2,
        )

    async def fake_rerank(req):
        return RerankResult(
            query=req.query,
            chunks=[
                RerankedChunk(
                    chunk_id=b_id,
                    document_id=doc_id,
                    content="b",
                    original_rank=2,
                    reranked_rank=1,
                    original_score=0.8,
                    reranked_score=0.95,
                ),
                RerankedChunk(
                    chunk_id=a_id,
                    document_id=doc_id,
                    content="a",
                    original_rank=1,
                    reranked_rank=2,
                    original_score=0.9,
                    reranked_score=0.1,
                ),
            ],
            method_used=RerankingMethod.POINTWISE,
            latency_ms=1.0,
            tokens_used=0,
            rank_changes=2,
        )

    async def fake_generate(req):
        return GenerationResponse(
            query=req.query,
            response="answer",
            citations=[],
            sources_used=[],
            sources_provided=2,
            grounding_score=0.0,
            confidence=0.9,
            tokens_used=0,
            latency_ms=1.0,
        )

    hiero.retriever = SimpleNamespace(retrieve=fake_retrieve)
    hiero.reranker = SimpleNamespace(rerank=fake_rerank)
    hiero.generator = SimpleNamespace(generate=fake_generate)

    resp = await hiero.query("q", retrieval_config=RetrievalConfig(rerank_results=True, top_k=2))
    assert resp.retrieval.chunks[0].chunk_id == b_id
    assert resp.retrieval.chunks[0].score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_hiero_agent_query_calls_agent():
    hiero = Hiero(database_url="postgresql+asyncpg://u:p@localhost:5432/db", openai_api_key="sk-test")
    await hiero.initialize()

    async def fake_run(query):
        return AgentResponse(question=query.question, answer="agent", confidence=0.9)

    hiero.agent = SimpleNamespace(run=fake_run)
    resp = await hiero.agent_query("q")
    assert resp.answer == "agent"
