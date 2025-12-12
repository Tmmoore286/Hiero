from __future__ import annotations

from typing import Any

from uuid import UUID

from hiero.generation import GenerationConfig, GenerationRequest, GeneratorProtocol, SourceContext

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
        chunks = kwargs["chunks"]

        context: list[SourceContext] = []
        for c in chunks:
            if isinstance(c, dict):
                chunk_id = UUID(str(c["chunk_id"]))
                document_id = UUID(str(c["document_id"]))
                content = str(c.get("content", ""))
                metadata = c.get("metadata") or {}
                score = float(c.get("score", 0.0))
            else:
                chunk_id = c.chunk_id
                document_id = c.document_id
                content = c.content
                metadata = c.metadata
                score = c.score

            context.append(
                SourceContext(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=content,
                    metadata=metadata,
                    relevance_score=score,
                )
            )

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
