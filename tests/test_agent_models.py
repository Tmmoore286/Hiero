import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.agent import AgentAction, AgentObservation, AgentStep, Citation, ToolType


def test_agent_models_construct():
    action = AgentAction(tool=ToolType.RETRIEVE, tool_input={"q": "x"}, thought="t")
    obs = AgentObservation(action=action, result={"ok": True}, success=True, latency_ms=1.0)
    step = AgentStep(step_number=1, action=action, observation=obs)
    cit = Citation(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content_snippet="s",
    )
    assert step.step_number == 1
    assert cit.content_snippet == "s"

