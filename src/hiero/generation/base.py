from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class GenerationMode(str, Enum):
    SIMPLE = "simple"
    GROUNDED = "grounded"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


class OutputFormat(str, Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    JSON = "json"
    BULLETS = "bullets"


class GenerationConfig(BaseModel):
    mode: GenerationMode = GenerationMode.GROUNDED
    output_format: OutputFormat = OutputFormat.MARKDOWN
    max_tokens: int = 1000
    temperature: float = 0.3
    include_citations: bool = True
    citation_style: str = "inline"
    require_grounding: bool = True
    model: str = "gpt-4o"


class SourceContext(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    metadata: dict = Field(default_factory=dict)
    relevance_score: float = 0.0


class GenerationRequest(BaseModel):
    query: str
    context: list[SourceContext]
    config: GenerationConfig = Field(default_factory=GenerationConfig)
    system_instructions: str | None = None
    conversation_history: list[tuple[str, str]] = Field(default_factory=list)


class InlineCitation(BaseModel):
    source_index: int
    chunk_id: UUID
    quoted_text: str
    start_pos: int
    end_pos: int


class GenerationResponse(BaseModel):
    query: str
    response: str
    citations: list[InlineCitation]
    sources_used: list[UUID]
    sources_provided: int
    grounding_score: float
    confidence: float
    tokens_used: int
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GeneratorProtocol(ABC):
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        ...

    @abstractmethod
    async def generate_streaming(self, request: GenerationRequest):
        ...

