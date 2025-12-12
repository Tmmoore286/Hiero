import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.agent.tools import CalculateTool, RetrieveTool
from hiero.retrieval import RetrievalConfig, RetrievalQuery, RetrievalResult, RetrievalStrategy


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

