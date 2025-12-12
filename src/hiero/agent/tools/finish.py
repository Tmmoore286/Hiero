from __future__ import annotations

from typing import Any

from hiero.generation import GenerationConfig, GenerationRequest, GeneratorProtocol, SourceContext
from hiero.retrieval import RetrievedChunk

from ..base import Citation, ToolProtocol, ToolType


class FinishTool(ToolProtocol):
    def __init__(self, generator: GeneratorProtocol):
        self.generator = generator

    @property
    def name(self) -> ToolType:
        return ToolType.FINISH

    @property
    def description(self) -> str:
        return "Produce the final grounded answer with citations."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "chunks": {"type": "array"},
            },
            "required": ["question", "chunks"],
        }

    async def execute(self, **kwargs) -> Any:
        question: str = kwargs["question"]
        chunks: list[RetrievedChunk] = kwargs["chunks"]

        context = [
            SourceContext(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                metadata=c.metadata,
                relevance_score=c.score,
            )
            for c in chunks
        ]

        gen_req = GenerationRequest(
            query=question,
            context=context,
            config=GenerationConfig(require_grounding=True),
        )
        gen = await self.generator.generate(gen_req)

        citations: list[Citation] = []
        by_chunk_id = {c.chunk_id: c for c in context}
        for cit in gen.citations:
            src = by_chunk_id.get(cit.chunk_id)
            if not src:
                continue
            snippet = src.content[:240]
            citations.append(
                Citation(
                    chunk_id=src.chunk_id,
                    document_id=src.document_id,
                    content_snippet=snippet,
                )
            )

        return {
            "answer": gen.response,
            "citations": [c.model_dump() for c in citations],
            "confidence": gen.confidence,
            "grounding_score": gen.grounding_score,
            "tokens_used": gen.tokens_used,
        }

