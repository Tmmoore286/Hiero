# Component Spec: Generation Layer

## Overview

The Generation Layer synthesizes final responses from retrieved context using LLMs. It enforces grounding (answers must be supported by sources), handles structured output, and provides attribution/citations.

## Requirements

### Functional
- **FR-1**: Generate answers grounded in retrieved context
- **FR-2**: Support multiple LLM providers (OpenAI, Anthropic)
- **FR-3**: Structured output with citations
- **FR-4**: Configurable prompting strategies
- **FR-5**: Streaming response support
- **FR-6**: Hallucination detection/prevention
- **FR-7**: Response formatting (markdown, JSON, etc.)

### Non-Functional
- **NFR-1**: Generation latency < 5 seconds for typical queries
- **NFR-2**: Token-efficient prompts
- **NFR-3**: Consistent citation format

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID
from typing import Any


class GenerationMode(str, Enum):
    SIMPLE = "simple"              # Direct answer
    GROUNDED = "grounded"          # Strict grounding to sources
    ANALYTICAL = "analytical"      # Multi-source synthesis
    CONVERSATIONAL = "conversational"  # Chat-style


class OutputFormat(str, Enum):
    PLAIN = "plain"          # Plain text
    MARKDOWN = "markdown"    # Markdown formatted
    JSON = "json"            # Structured JSON
    BULLETS = "bullets"      # Bullet points


class GenerationConfig(BaseModel):
    """Configuration for generation behavior."""
    mode: GenerationMode = GenerationMode.GROUNDED
    output_format: OutputFormat = OutputFormat.MARKDOWN
    max_tokens: int = 1000
    temperature: float = 0.3
    include_citations: bool = True
    citation_style: str = "inline"  # inline, footnote, endnote
    require_grounding: bool = True  # Reject ungrounded claims
    model: str = "gpt-4o"


class SourceContext(BaseModel):
    """A source chunk provided as context."""
    chunk_id: UUID
    document_id: UUID
    content: str
    metadata: dict = Field(default_factory=dict)
    relevance_score: float = 0.0


class GenerationRequest(BaseModel):
    """Request to generate a response."""
    query: str
    context: list[SourceContext]
    config: GenerationConfig = Field(default_factory=GenerationConfig)

    # Optional additional context
    system_instructions: str | None = None
    conversation_history: list[tuple[str, str]] = Field(default_factory=list)


class InlineCitation(BaseModel):
    """A citation within the response."""
    source_index: int      # Index in the context list
    chunk_id: UUID
    quoted_text: str       # The relevant quote
    start_pos: int         # Position in response
    end_pos: int


class GenerationResponse(BaseModel):
    """Complete generated response."""
    query: str
    response: str
    citations: list[InlineCitation]
    sources_used: list[UUID]  # chunk_ids actually referenced
    sources_provided: int

    # Quality metrics
    grounding_score: float  # 0-1, how grounded is the response
    confidence: float       # 0-1, model confidence

    # Performance
    tokens_used: int
    latency_ms: float
```

## Interfaces

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator


class GeneratorProtocol(ABC):
    """Base protocol for generators."""

    @abstractmethod
    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """
        Generate a response to the query.

        Args:
            request: GenerationRequest with query and context

        Returns:
            GenerationResponse with answer and citations
        """
        ...

    @abstractmethod
    async def generate_streaming(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """
        Generate response with streaming output.

        Yields:
            Response tokens as they're generated
        """
        ...
```

## Implementation Details

### Grounded Generator

```python
import asyncio
import json
import re
import time
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


class GroundedGenerator(GeneratorProtocol):
    """
    Generator that ensures responses are grounded in provided sources.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = None,
    ):
        self.provider = provider

        if provider == "openai":
            self.client = AsyncOpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        start_time = time.perf_counter()

        # Build prompt
        system_prompt = self._build_system_prompt(request.config)
        user_prompt = self._build_user_prompt(request)

        # Generate response
        if self.provider == "openai":
            response = await self._generate_openai(
                system_prompt, user_prompt, request
            )
        else:
            response = await self._generate_anthropic(
                system_prompt, user_prompt, request
            )

        # Parse response and extract citations
        parsed_response, citations = self._parse_response(
            response["content"],
            request.context,
            request.config,
        )

        # Calculate grounding score
        grounding_score = self._calculate_grounding(
            parsed_response, request.context, citations
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResponse(
            query=request.query,
            response=parsed_response,
            citations=citations,
            sources_used=list(set(c.chunk_id for c in citations)),
            sources_provided=len(request.context),
            grounding_score=grounding_score,
            confidence=response.get("confidence", 0.8),
            tokens_used=response["tokens"],
            latency_ms=elapsed_ms,
        )

    async def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        request: GenerationRequest,
    ) -> dict:
        """Generate using OpenAI API."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for user_msg, assistant_msg in request.conversation_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": user_prompt})

        response = await self.client.chat.completions.create(
            model=request.config.model,
            messages=messages,
            max_tokens=request.config.max_tokens,
            temperature=request.config.temperature,
        )

        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
        }

    async def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        request: GenerationRequest,
    ) -> dict:
        """Generate using Anthropic API."""
        messages = []

        for user_msg, assistant_msg in request.conversation_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": user_prompt})

        response = await self.client.messages.create(
            model=request.config.model,
            system=system_prompt,
            messages=messages,
            max_tokens=request.config.max_tokens,
        )

        return {
            "content": response.content[0].text,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

    def _build_system_prompt(self, config: GenerationConfig) -> str:
        """Build the system prompt based on configuration."""
        base_prompt = """You are a helpful assistant that answers questions based on provided source documents.

"""
        if config.mode == GenerationMode.GROUNDED:
            base_prompt += """IMPORTANT: You must ONLY use information from the provided sources.
- Every claim must be supported by the sources
- If the sources don't contain enough information, say so
- Never make up information or use external knowledge
- Use citations to reference sources

"""
        elif config.mode == GenerationMode.ANALYTICAL:
            base_prompt += """Synthesize information from multiple sources to provide a comprehensive answer.
- Compare and contrast information across sources
- Identify patterns and connections
- Note any contradictions between sources
- Cite sources for all claims

"""
        elif config.mode == GenerationMode.CONVERSATIONAL:
            base_prompt += """Respond in a natural, conversational tone.
- Be helpful and informative
- Use the sources to ground your response
- Feel free to elaborate but stay accurate

"""

        if config.include_citations:
            if config.citation_style == "inline":
                base_prompt += """Citation format: Use [Source N] inline citations where N is the source number.
Example: "The project started in 2020 [Source 1] and expanded significantly [Source 2]."

"""
            elif config.citation_style == "footnote":
                base_prompt += """Citation format: Use superscript numbers for citations, list sources at the end.
Example: "The project started in 2020¹ and expanded significantly²."

"""

        if config.output_format == OutputFormat.MARKDOWN:
            base_prompt += "Format your response using markdown for readability.\n"
        elif config.output_format == OutputFormat.BULLETS:
            base_prompt += "Format your response as bullet points.\n"
        elif config.output_format == OutputFormat.JSON:
            base_prompt += "Format your response as JSON with 'answer' and 'sources' fields.\n"

        return base_prompt

    def _build_user_prompt(self, request: GenerationRequest) -> str:
        """Build the user prompt with context."""
        parts = []

        # Add sources
        parts.append("Sources:")
        for i, source in enumerate(request.context):
            parts.append(f"\n[Source {i+1}]")
            parts.append(source.content)
            if source.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in source.metadata.items() if v)
                if meta_str:
                    parts.append(f"(Metadata: {meta_str})")

        parts.append(f"\n\nQuestion: {request.query}")

        if request.system_instructions:
            parts.append(f"\nAdditional instructions: {request.system_instructions}")

        return "\n".join(parts)

    def _parse_response(
        self,
        response: str,
        context: list[SourceContext],
        config: GenerationConfig,
    ) -> tuple[str, list[InlineCitation]]:
        """Parse the response and extract citations."""
        citations = []

        if not config.include_citations:
            return response, citations

        # Find citation patterns like [Source 1], [Source 2], etc.
        citation_pattern = r'\[Source\s*(\d+)\]'
        matches = list(re.finditer(citation_pattern, response, re.IGNORECASE))

        for match in matches:
            source_num = int(match.group(1))
            if 1 <= source_num <= len(context):
                source = context[source_num - 1]

                # Find the sentence containing this citation
                start = response.rfind('.', 0, match.start()) + 1
                end = response.find('.', match.end())
                if end == -1:
                    end = len(response)

                quoted_text = response[start:end].strip()

                citations.append(InlineCitation(
                    source_index=source_num - 1,
                    chunk_id=source.chunk_id,
                    quoted_text=quoted_text,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        return response, citations

    def _calculate_grounding(
        self,
        response: str,
        context: list[SourceContext],
        citations: list[InlineCitation],
    ) -> float:
        """
        Calculate how well the response is grounded in sources.

        Uses simple heuristics - in production, use NLI model.
        """
        if not context:
            return 0.0

        # Count sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 1.0

        # Check how many sentences have citations
        cited_sentences = 0
        for sentence in sentences:
            # Check if sentence contains a citation marker
            if re.search(r'\[Source\s*\d+\]', sentence, re.IGNORECASE):
                cited_sentences += 1
            else:
                # Check for content overlap with sources
                source_texts = " ".join(s.content.lower() for s in context)
                sentence_words = set(sentence.lower().split())
                source_words = set(source_texts.split())
                overlap = len(sentence_words & source_words) / max(len(sentence_words), 1)
                if overlap > 0.5:
                    cited_sentences += 0.5

        return min(cited_sentences / len(sentences), 1.0)

    async def generate_streaming(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        """Generate with streaming output."""
        system_prompt = self._build_system_prompt(request.config)
        user_prompt = self._build_user_prompt(request)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.provider == "openai":
            stream = await self.client.chat.completions.create(
                model=request.config.model,
                messages=messages,
                max_tokens=request.config.max_tokens,
                temperature=request.config.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif self.provider == "anthropic":
            async with self.client.messages.stream(
                model=request.config.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=request.config.max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
```

### Prompt Templates

```python
class PromptTemplates:
    """Collection of prompt templates for different scenarios."""

    GROUNDED_QA = """Based on the provided sources, answer the following question.

{sources}

Question: {question}

Instructions:
- Only use information from the sources above
- Cite sources using [Source N] format
- If the sources don't contain the answer, say "The provided sources don't contain information about this."
- Be concise but complete

Answer:"""

    ANALYTICAL_SYNTHESIS = """Analyze the following sources to provide a comprehensive answer.

{sources}

Question: {question}

Instructions:
- Synthesize information across all relevant sources
- Note any agreements or contradictions
- Cite sources for each major point
- Provide a balanced, well-reasoned response

Analysis:"""

    COMPARISON = """Compare and contrast the information from these sources.

{sources}

Topic: {question}

Create a comparison that:
- Identifies key similarities
- Highlights important differences
- Notes any conflicting information
- Cites sources throughout

Comparison:"""

    SUMMARIZATION = """Summarize the key information from these sources.

{sources}

Focus: {question}

Create a summary that:
- Captures the main points
- Maintains factual accuracy
- Uses clear, concise language
- Cites sources for major claims

Summary:"""

    NO_CONTEXT = """I don't have any sources to answer this question.

Question: {question}

The provided sources don't contain information relevant to this question.
Please provide additional context or rephrase your question."""


def get_template(mode: GenerationMode) -> str:
    """Get appropriate template for generation mode."""
    templates = {
        GenerationMode.SIMPLE: PromptTemplates.GROUNDED_QA,
        GenerationMode.GROUNDED: PromptTemplates.GROUNDED_QA,
        GenerationMode.ANALYTICAL: PromptTemplates.ANALYTICAL_SYNTHESIS,
    }
    return templates.get(mode, PromptTemplates.GROUNDED_QA)
```

### Structured Output Generator

```python
from pydantic import BaseModel as PydanticBaseModel


class StructuredGenerator:
    """
    Generator that produces structured (JSON) outputs.
    Uses OpenAI's structured output feature.
    """

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate_structured(
        self,
        request: GenerationRequest,
        output_schema: type[PydanticBaseModel],
    ) -> PydanticBaseModel:
        """Generate a response conforming to a schema."""
        system_prompt = self._build_system_prompt(request.config)
        user_prompt = self._build_user_prompt(request)

        # Add schema instruction
        schema_instruction = f"""
Output your response as JSON matching this schema:
{output_schema.model_json_schema()}
"""
        system_prompt += schema_instruction

        response = await self.client.chat.completions.create(
            model=request.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=request.config.temperature,
        )

        content = response.choices[0].message.content
        return output_schema.model_validate_json(content)


# Example structured output schemas
class AnswerWithSources(PydanticBaseModel):
    """Structured answer with sources."""
    answer: str
    confidence: float
    sources_used: list[int]
    key_facts: list[str]
    limitations: str | None = None


class ComparisonResult(PydanticBaseModel):
    """Structured comparison output."""
    similarities: list[str]
    differences: list[str]
    contradictions: list[str]
    conclusion: str
```

### Hallucination Detector

```python
class HallucinationDetector:
    """
    Detects potential hallucinations in generated responses.
    """

    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = llm_client
        self.model = model

    async def check_grounding(
        self,
        response: str,
        sources: list[SourceContext],
    ) -> dict:
        """
        Check if response claims are grounded in sources.

        Returns:
            {
                "grounded": bool,
                "ungrounded_claims": list[str],
                "confidence": float
            }
        """
        prompt = f"""Analyze if this response is fully grounded in the provided sources.

Response to check:
{response}

Available sources:
{self._format_sources(sources)}

For each claim in the response, determine if it's supported by the sources.

Output JSON:
{{
    "grounded": true/false,
    "claims": [
        {{"claim": "...", "supported": true/false, "source": "Source N or null"}}
    ],
    "ungrounded_claims": ["..."],
    "confidence": 0.0-1.0
}}"""

        result = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        return json.loads(result.choices[0].message.content)

    def _format_sources(self, sources: list[SourceContext]) -> str:
        return "\n\n".join([
            f"[Source {i+1}]: {s.content}"
            for i, s in enumerate(sources)
        ])

    async def filter_hallucinations(
        self,
        response: str,
        sources: list[SourceContext],
    ) -> str:
        """
        Remove or flag ungrounded claims from response.
        """
        check_result = await self.check_grounding(response, sources)

        if check_result["grounded"]:
            return response

        # Add warnings for ungrounded claims
        warnings = []
        for claim in check_result.get("ungrounded_claims", []):
            warnings.append(f"⚠️ Unverified claim: {claim}")

        if warnings:
            return response + "\n\n---\n" + "\n".join(warnings)

        return response
```

## Generation Modes Comparison

| Mode | Use Case | Grounding | Style |
|------|----------|-----------|-------|
| Simple | Quick answers | Moderate | Direct |
| Grounded | Factual Q&A | Strict | Cited |
| Analytical | Research | High | Comprehensive |
| Conversational | Chatbots | Moderate | Natural |

## Dependencies

```toml
[project.dependencies]
openai = "^1.14"
anthropic = "^0.21"
```

## Testing Strategy

1. **Unit Tests**: Prompt building, citation parsing
2. **Integration Tests**: Full generation with real LLM
3. **Grounding Tests**: Verify citations accuracy
4. **Hallucination Tests**: Known ungrounded claims detection
5. **Format Tests**: Output format compliance
