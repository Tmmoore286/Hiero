import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.generation import GenerationConfig, GenerationRequest, GroundedGenerator, SourceContext


@pytest.mark.asyncio
async def test_grounded_generator_parses_inline_citations(monkeypatch):
    generator = GroundedGenerator(api_key="sk-test")

    async def fake_generate_openai(system_prompt, user_prompt, config):
        return {"content": "Answer based on sources [1] and [2].", "tokens": 12}

    monkeypatch.setattr(generator, "_generate_openai", fake_generate_openai)

    context = [
        SourceContext(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="source one",
        ),
        SourceContext(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="source two",
        ),
    ]

    request = GenerationRequest(
        query="What?",
        context=context,
        config=GenerationConfig(require_grounding=True),
    )

    response = await generator.generate(request)
    assert response.response.startswith("Answer based on sources")
    assert len(response.citations) == 2
    assert {c.chunk_id for c in response.citations} == {c.chunk_id for c in context}

