import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.agent.tools.finish import FinishTool
from hiero.generation import GenerationResponse, InlineCitation, SourceContext
from hiero.retrieval import RetrievedChunk, RetrievalStrategy


class _FakeGenerator:
    async def generate(self, request):
        return GenerationResponse(
            query=request.query,
            response="answer [1]",
            citations=[
                InlineCitation(
                    source_index=0,
                    chunk_id=request.context[0].chunk_id,
                    quoted_text="",
                    start_pos=7,
                    end_pos=10,
                )
            ],
            sources_used=[request.context[0].chunk_id],
            sources_provided=len(request.context),
            grounding_score=1.0,
            confidence=0.9,
            tokens_used=0,
            latency_ms=1.0,
        )

    async def generate_streaming(self, request):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_finish_tool_generates_answer_with_citations():
    tool = FinishTool(_FakeGenerator())
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            content="context",
            score=0.9,
            rank=1,
            retrieval_strategy=RetrievalStrategy.DENSE,
            chunk_index=0,
        )
    ]
    result = await tool.execute(question="q", chunks=chunks)
    assert result["answer"].startswith("answer")
    assert result["citations"][0]["chunk_id"] == chunk_id

