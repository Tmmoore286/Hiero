import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.agent import AgentQuery, ReActAgent, ToolType
from hiero.agent.llm import ChatMessage
from hiero.agent.tools import CalculateTool
from hiero.agent.base import ToolProtocol


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.0):
        self.calls += 1
        if self.calls == 1:
            return (
                '{"tool":"calculate","tool_input":{"expression":"2+2"},"thought":"math"}',
                5,
            )
        return (
            '{"tool":"finish","tool_input":{},"thought":"finish"}',
            5,
        )


class _FakeFinishTool(ToolProtocol):
    @property
    def name(self) -> ToolType:
        return ToolType.FINISH

    @property
    def description(self) -> str:
        return "finish"

    @property
    def parameters(self) -> dict:
        return {"type": "object"}

    async def execute(self, **kwargs):
        return {"answer": "done", "citations": [], "confidence": 0.9}


@pytest.mark.asyncio
async def test_react_agent_finishes():
    agent = ReActAgent(llm=_FakeLLM(), tools=[CalculateTool(), _FakeFinishTool()])
    resp = await agent.run(AgentQuery(question="q"))
    assert resp.answer == "done"
    assert resp.total_steps == 2
    assert resp.steps[0].action.tool == ToolType.CALCULATE
