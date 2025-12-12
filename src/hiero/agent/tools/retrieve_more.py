from __future__ import annotations

from typing import Any
from uuid import UUID

from hiero.retrieval import RetrievalConfig, RetrievalQuery, RetrieverProtocol

from ..base import ToolProtocol, ToolType


class RetrieveMoreTool(ToolProtocol):
    def __init__(self, retriever: RetrieverProtocol, namespace: str = "default"):
        self.retriever = retriever
        self.namespace = namespace

    @property
    def name(self) -> ToolType:
        return ToolType.RETRIEVE_MORE

    @property
    def description(self) -> str:
        return "Retrieve additional chunks, excluding ones already seen."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                "exclude_chunk_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> Any:
        query_text = kwargs["query"]
        top_k = int(kwargs.get("top_k", 20))
        exclude = {UUID(x) for x in (kwargs.get("exclude_chunk_ids") or [])}

        rq = RetrievalQuery(
            text=query_text,
            namespace=self.namespace,
            config=RetrievalConfig(top_k=top_k),
        )
        result = await self.retriever.retrieve(rq)
        if exclude:
            filtered = [c for c in result.chunks if c.chunk_id not in exclude]
            result = result.model_copy(
                update={"chunks": filtered, "total_candidates": len(filtered)}
            )
        return result.model_dump(mode="json")
