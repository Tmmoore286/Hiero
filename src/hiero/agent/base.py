from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ToolType(str, Enum):
    RETRIEVE = "retrieve"
    RETRIEVE_MORE = "retrieve_more"
    CALCULATE = "calculate"
    SUMMARIZE = "summarize"
    FINISH = "finish"


class AgentAction(BaseModel):
    tool: ToolType
    tool_input: dict[str, Any] = Field(default_factory=dict)
    thought: str = ""


class AgentObservation(BaseModel):
    action: AgentAction
    result: Any = None
    success: bool
    error: str | None = None
    latency_ms: float


class AgentStep(BaseModel):
    step_number: int
    action: AgentAction
    observation: AgentObservation
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentConfig(BaseModel):
    max_steps: int = 10
    max_retrieval_calls: int = 5
    temperature: float = 0.1
    enable_self_evaluation: bool = False
    enable_query_decomposition: bool = False
    require_citations: bool = True
    verbose_logging: bool = False

    retrieval_top_k: int = 5
    rerank_results: bool = False

    enabled_tools: list[ToolType] = Field(
        default_factory=lambda: [
            ToolType.RETRIEVE,
            ToolType.RETRIEVE_MORE,
            ToolType.CALCULATE,
            ToolType.SUMMARIZE,
            ToolType.FINISH,
        ]
    )


class Citation(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content_snippet: str
    relevance: str | None = None


class AgentQuery(BaseModel):
    question: str
    namespace: str = "default"
    context: str | None = None
    config: AgentConfig = Field(default_factory=AgentConfig)
    history: list[tuple[str, str]] = Field(default_factory=list)


class AgentResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)

    steps: list[AgentStep] = Field(default_factory=list)
    total_steps: int = 0
    total_retrieval_calls: int = 0

    confidence: float = 0.0
    self_evaluation: str | None = None

    total_latency_ms: float = 0.0
    llm_tokens_used: int = 0


class SubQuery(BaseModel):
    query: str
    reasoning: str
    depends_on: list[int] = Field(default_factory=list)


class AgentProtocol(ABC):
    @abstractmethod
    async def run(self, query: AgentQuery) -> AgentResponse:
        ...

    @abstractmethod
    async def run_streaming(self, query: AgentQuery) -> AsyncIterator[AgentStep | AgentResponse]:
        ...


class ToolProtocol(ABC):
    @property
    @abstractmethod
    def name(self) -> ToolType:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        ...

