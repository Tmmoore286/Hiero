from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ChatModel(ABC):
    @abstractmethod
    async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.0) -> tuple[str, int]:
        ...


class OpenAIChatModel(ChatModel):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.0) -> tuple[str, int]:
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
        )
        content = completion.choices[0].message.content or ""
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0) or 0
        return content, int(tokens)

