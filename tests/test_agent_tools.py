import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.agent.tools import CalculateTool, RetrieveTool, RetrieveMoreTool, SummarizeTool
from hiero.retrieval import RetrievedChunk, RetrievalConfig, RetrievalQuery, RetrievalResult, RetrievalStrategy


@pytest.mark.asyncio
async def test_calculate_tool_evaluates_expression():
    tool = CalculateTool()
    result = await tool.execute(expression="2 + 3 * 4")
    assert result["value"] == 14.0


class _FakeRetriever:
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        assert isinstance(query.config, RetrievalConfig)
        assert query.text == "hello"
        assert query.config.top_k == 3
        return RetrievalResult(
            query=query.text,
            chunks=[],
            total_candidates=0,
            strategy_used=RetrievalStrategy.DENSE,
            latency_ms=1.0,
        )

    async def retrieve_batch(self, queries):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_retrieve_tool_calls_retriever():
    tool = RetrieveTool(_FakeRetriever(), namespace="default")
    out = await tool.execute(query="hello", top_k=3, document_ids=[str(uuid.uuid4())])
    assert out["query"] == "hello"


@pytest.mark.asyncio
async def test_retrieve_more_tool_excludes_chunk_ids():
    cid = uuid.uuid4()

    class _FakeRetrieverMore(_FakeRetriever):
        async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
            return RetrievalResult(
                query=query.text,
                chunks=[
                    RetrievedChunk(
                        chunk_id=cid,
                        document_id=uuid.uuid4(),
                        content="x",
                        score=0.1,
                        rank=1,
                        retrieval_strategy=RetrievalStrategy.DENSE,
                        chunk_index=0,
                    )
                ],
                total_candidates=1,
                strategy_used=RetrievalStrategy.DENSE,
                latency_ms=1.0,
            )

    tool = RetrieveMoreTool(_FakeRetrieverMore(), namespace="default")
    out = await tool.execute(query="hello", top_k=3, exclude_chunk_ids=[str(cid)])
    assert out["query"] == "hello"
    assert out["chunks"] == []


@pytest.mark.asyncio
async def test_summarize_tool_returns_brief():
    tool = SummarizeTool()
    out = await tool.execute(chunks=[{"content": "abc"}, {"content": "def"}], max_chars=4)
    assert len(out["summary"]) <= 4 + 2  # allow separator/newlines
