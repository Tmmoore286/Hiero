import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.reranking import LLMReranker, RerankRequest, RerankerConfig
from hiero.retrieval import RetrievedChunk, RetrievalStrategy


@pytest.mark.asyncio
async def test_llm_reranker_orders_by_score(monkeypatch):
    reranker = LLMReranker(api_key="sk-test")

    async def fake_score(query, content, config):
        if "first" in content:
            return 0.2, None, 1
        return 0.9, None, 1

    monkeypatch.setattr(reranker, "_score_chunk", fake_score)

    doc_id = uuid.uuid4()
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            content="first chunk",
            score=0.9,
            rank=1,
            retrieval_strategy=RetrievalStrategy.DENSE,
            chunk_index=0,
        ),
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            content="second chunk",
            score=0.8,
            rank=2,
            retrieval_strategy=RetrievalStrategy.DENSE,
            chunk_index=1,
        ),
    ]

    result = await reranker.rerank(
        RerankRequest(query="q", chunks=chunks, config=RerankerConfig(top_k=2))
    )
    assert result.chunks[0].content == "second chunk"
    assert result.rank_changes >= 1

