from __future__ import annotations

from typing import Any

from ..base import ToolProtocol, ToolType


class SummarizeTool(ToolProtocol):
    @property
    def name(self) -> ToolType:
        return ToolType.SUMMARIZE

    @property
    def description(self) -> str:
        return "Summarize retrieved chunks into a short brief."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "chunks": {"type": "array"},
                "max_chars": {"type": "integer"},
            },
            "required": ["chunks"],
        }

    async def execute(self, **kwargs) -> Any:
        chunks = kwargs.get("chunks") or []
        max_chars = int(kwargs.get("max_chars", 800))
        texts = []
        used = 0
        for c in chunks:
            content = c.get("content") if isinstance(c, dict) else str(c)
            if not content:
                continue
            remaining = max_chars - used
            if remaining <= 0:
                break
            snippet = content[:remaining]
            texts.append(snippet)
            used += len(snippet)
        return {"summary": "\n\n".join(texts)}

