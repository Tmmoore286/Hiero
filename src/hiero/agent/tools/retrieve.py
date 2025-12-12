from __future__ import annotations

from typing import Any
from uuid import UUID

from hiero.retrieval import RetrievalConfig, RetrievalQuery, RetrieverProtocol

from ..base import ToolProtocol, ToolType


class RetrieveTool(ToolProtocol):
    def __init__(self, retriever: RetrieverProtocol, namespace: str = "default"):
        self.retriever = retriever
        self.namespace = namespace

    @property
    def name(self) -> ToolType:
        return ToolType.RETRIEVE

    @property
    def description(self) -> str:
        return "Retrieve relevant chunks from the document store."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> Any:
        query_text = kwargs["query"]
        top_k = int(kwargs.get("top_k", 10))
        document_ids = kwargs.get("document_ids")
        parsed_ids: list[UUID] | None = None
        if document_ids:
            parsed_ids = [UUID(d) for d in document_ids]

        config = RetrievalConfig(top_k=top_k, document_ids=parsed_ids)
        rq = RetrievalQuery(text=query_text, namespace=self.namespace, config=config)
        result = await self.retriever.retrieve(rq)
        return result.model_dump(mode="json")
