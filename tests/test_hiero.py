import sys
import uuid
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero import Hiero
from hiero.generation import GenerationResponse, InlineCitation
from hiero.retrieval import RetrievedChunk, RetrievalResult, RetrievalStrategy


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

