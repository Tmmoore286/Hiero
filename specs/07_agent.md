# Component Spec: Agent Controller

## Overview

The Agent Controller implements agentic RAG with multi-step reasoning, tool use, query decomposition, and self-evaluation. It uses a ReAct (Reason-Act-Observe) loop to iteratively retrieve and reason over information.

## Requirements

### Functional
- **FR-1**: ReAct loop for iterative reasoning
- **FR-2**: Tool use (retrieve, calculate, web search)
- **FR-3**: Query decomposition for complex questions
- **FR-4**: Self-evaluation and refinement
- **FR-5**: Conversation memory/context management
- **FR-6**: Structured output with citations
- **FR-7**: Full trace logging for debugging
- **FR-8**: Configurable stopping conditions

### Non-Functional
- **NFR-1**: Complete most queries in < 30 seconds
- **NFR-2**: Maximum 10 reasoning steps (configurable)
- **NFR-3**: Graceful handling of tool failures
- **NFR-4**: Reproducible results with same seed

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import Any


class ToolType(str, Enum):
    RETRIEVE = "retrieve"          # Search vector store
    RETRIEVE_MORE = "retrieve_more"  # Get additional results
    CALCULATE = "calculate"        # Math operations
    WEB_SEARCH = "web_search"      # External search
    LOOKUP = "lookup"              # Specific doc lookup
    SUMMARIZE = "summarize"        # Summarize retrieved content
    FINISH = "finish"              # Complete the task


class AgentAction(BaseModel):
    """A single action taken by the agent."""
    tool: ToolType
    tool_input: dict[str, Any]
    thought: str  # Reasoning for this action


class AgentObservation(BaseModel):
    """Result of executing an action."""
    action: AgentAction
    result: Any
    success: bool
    error: str | None = None
    latency_ms: float


class AgentStep(BaseModel):
    """A complete step in the agent loop."""
    step_number: int
    action: AgentAction
    observation: AgentObservation
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    max_steps: int = 10
    max_retrieval_calls: int = 5
    temperature: float = 0.1
    enable_self_evaluation: bool = True
    enable_query_decomposition: bool = True
    require_citations: bool = True
    verbose_logging: bool = False

    # Retrieval settings
    retrieval_top_k: int = 5
    rerank_results: bool = True

    # Tool availability
    enabled_tools: list[ToolType] = Field(default_factory=lambda: [
        ToolType.RETRIEVE,
        ToolType.RETRIEVE_MORE,
        ToolType.CALCULATE,
        ToolType.SUMMARIZE,
        ToolType.FINISH,
    ])


class Citation(BaseModel):
    """A citation to source material."""
    chunk_id: UUID
    document_id: UUID
    content_snippet: str
    relevance: str  # How it supports the answer


class AgentQuery(BaseModel):
    """Input query to the agent."""
    question: str
    namespace: str = "default"
    context: str | None = None  # Additional context
    config: AgentConfig = Field(default_factory=AgentConfig)

    # Optional: conversation history
    history: list[tuple[str, str]] = Field(default_factory=list)  # (user, assistant)


class AgentResponse(BaseModel):
    """Complete response from the agent."""
    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str
    citations: list[Citation]

    # Trace
    steps: list[AgentStep]
    total_steps: int
    total_retrieval_calls: int

    # Quality
    confidence: float  # 0-1 self-assessed confidence
    self_evaluation: str | None = None

    # Performance
    total_latency_ms: float
    llm_tokens_used: int


class SubQuery(BaseModel):
    """A decomposed sub-query."""
    query: str
    reasoning: str  # Why this sub-query is needed
    depends_on: list[int] = Field(default_factory=list)  # Indices of prerequisite sub-queries
```

## Interfaces

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator


class AgentProtocol(ABC):
    """Base protocol for agents."""

    @abstractmethod
    async def run(
        self,
        query: AgentQuery,
    ) -> AgentResponse:
        """
        Execute the agent to answer a query.

        Args:
            query: AgentQuery with question and config

        Returns:
            AgentResponse with answer and trace
        """
        ...

    @abstractmethod
    async def run_streaming(
        self,
        query: AgentQuery,
    ) -> AsyncIterator[AgentStep | AgentResponse]:
        """
        Execute agent with streaming step updates.

        Yields:
            AgentStep for each reasoning step
            AgentResponse as final yield
        """
        ...


class ToolProtocol(ABC):
    """Base protocol for agent tools."""

    @property
    @abstractmethod
    def name(self) -> ToolType:
        """Tool identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description for the LLM."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        ...
```

## Implementation Details

### ReAct Agent

```python
import asyncio
import json
import time
from openai import AsyncOpenAI


class ReActAgent(AgentProtocol):
    """
    ReAct (Reason-Act-Observe) agent for agentic RAG.
    """

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        retriever: RetrieverProtocol,
        reranker: RerankerProtocol | None = None,
        model: str = "gpt-4o",
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.reranker = reranker
        self.model = model

        # Initialize tools
        self.tools: dict[ToolType, ToolProtocol] = {}
        self._init_tools()

    def _init_tools(self):
        """Initialize available tools."""
        self.tools[ToolType.RETRIEVE] = RetrieveTool(self.retriever, self.reranker)
        self.tools[ToolType.RETRIEVE_MORE] = RetrieveMoreTool(self.retriever)
        self.tools[ToolType.CALCULATE] = CalculateTool()
        self.tools[ToolType.SUMMARIZE] = SummarizeTool(self.llm, self.model)
        self.tools[ToolType.FINISH] = FinishTool()

    async def run(
        self,
        query: AgentQuery,
    ) -> AgentResponse:
        start_time = time.perf_counter()

        # Initialize state
        steps: list[AgentStep] = []
        retrieval_calls = 0
        total_tokens = 0
        retrieved_chunks: list[RetrievedChunk] = []
        context_buffer = ""

        # Query decomposition if enabled
        sub_queries = None
        if query.config.enable_query_decomposition:
            sub_queries = await self._decompose_query(query.question)

        # Main ReAct loop
        step_num = 0
        final_answer = None

        while step_num < query.config.max_steps:
            step_num += 1

            # Generate next action
            action, tokens = await self._generate_action(
                question=query.question,
                steps=steps,
                retrieved_chunks=retrieved_chunks,
                sub_queries=sub_queries,
                config=query.config,
            )
            total_tokens += tokens

            # Check for finish
            if action.tool == ToolType.FINISH:
                final_answer = action.tool_input.get("answer", "")
                steps.append(AgentStep(
                    step_number=step_num,
                    action=action,
                    observation=AgentObservation(
                        action=action,
                        result=final_answer,
                        success=True,
                        latency_ms=0,
                    ),
                ))
                break

            # Check retrieval limit
            if action.tool in (ToolType.RETRIEVE, ToolType.RETRIEVE_MORE):
                if retrieval_calls >= query.config.max_retrieval_calls:
                    # Force finish
                    action = AgentAction(
                        tool=ToolType.FINISH,
                        tool_input={"answer": "Unable to find sufficient information."},
                        thought="Reached maximum retrieval calls.",
                    )
                    continue
                retrieval_calls += 1

            # Execute tool
            observation = await self._execute_tool(action, query.namespace)

            # Update state
            steps.append(AgentStep(
                step_number=step_num,
                action=action,
                observation=observation,
            ))

            # Accumulate retrieved chunks
            if action.tool in (ToolType.RETRIEVE, ToolType.RETRIEVE_MORE):
                if observation.success and isinstance(observation.result, list):
                    retrieved_chunks.extend(observation.result)

            if query.config.verbose_logging:
                print(f"Step {step_num}: {action.tool} -> {observation.success}")

        # Self-evaluation if enabled
        self_eval = None
        confidence = 0.8
        if query.config.enable_self_evaluation and final_answer:
            self_eval, confidence, eval_tokens = await self._self_evaluate(
                question=query.question,
                answer=final_answer,
                steps=steps,
            )
            total_tokens += eval_tokens

        # Extract citations
        citations = self._extract_citations(final_answer, retrieved_chunks)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return AgentResponse(
            question=query.question,
            answer=final_answer or "Unable to determine an answer.",
            citations=citations,
            steps=steps,
            total_steps=len(steps),
            total_retrieval_calls=retrieval_calls,
            confidence=confidence,
            self_evaluation=self_eval,
            total_latency_ms=elapsed_ms,
            llm_tokens_used=total_tokens,
        )

    async def _generate_action(
        self,
        question: str,
        steps: list[AgentStep],
        retrieved_chunks: list[RetrievedChunk],
        sub_queries: list[SubQuery] | None,
        config: AgentConfig,
    ) -> tuple[AgentAction, int]:
        """Generate the next action using the LLM."""
        # Build prompt
        system_prompt = self._build_system_prompt(config)
        user_prompt = self._build_user_prompt(
            question, steps, retrieved_chunks, sub_queries
        )

        # Get available tools as functions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name.value,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool_type, tool in self.tools.items()
            if tool_type in config.enabled_tools
        ]

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice="required",
            temperature=config.temperature,
        )

        message = response.choices[0].message
        tokens = response.usage.total_tokens

        # Parse tool call
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_type = ToolType(tool_call.function.name)
            tool_input = json.loads(tool_call.function.arguments)

            return AgentAction(
                tool=tool_type,
                tool_input=tool_input,
                thought=message.content or "Executing tool.",
            ), tokens

        # Fallback - shouldn't happen with tool_choice="required"
        return AgentAction(
            tool=ToolType.FINISH,
            tool_input={"answer": message.content or ""},
            thought="No tool selected.",
        ), tokens

    def _build_system_prompt(self, config: AgentConfig) -> str:
        """Build the system prompt for the agent."""
        return f"""You are a research assistant that answers questions by searching a knowledge base.

Use the available tools to find information and answer the user's question.

Guidelines:
1. Think step-by-step about what information you need
2. Use the retrieve tool to search for relevant information
3. If initial results are insufficient, use retrieve_more or refine your search
4. Use calculate for any mathematical operations
5. Use summarize to condense large amounts of retrieved text
6. When you have enough information, use finish to provide your answer

{"Include citations to source material in your answer." if config.require_citations else ""}

Be thorough but efficient. Maximum {config.max_steps} steps allowed."""

    def _build_user_prompt(
        self,
        question: str,
        steps: list[AgentStep],
        retrieved_chunks: list[RetrievedChunk],
        sub_queries: list[SubQuery] | None,
    ) -> str:
        """Build the user prompt with context."""
        parts = [f"Question: {question}"]

        if sub_queries:
            sq_text = "\n".join([
                f"- {sq.query} ({sq.reasoning})"
                for sq in sub_queries
            ])
            parts.append(f"\nSub-questions to address:\n{sq_text}")

        if steps:
            history = "\n\n".join([
                f"Step {s.step_number}:\n"
                f"Thought: {s.action.thought}\n"
                f"Action: {s.action.tool.value}({json.dumps(s.action.tool_input)})\n"
                f"Observation: {str(s.observation.result)[:500]}"
                for s in steps[-5:]  # Last 5 steps
            ])
            parts.append(f"\nPrevious steps:\n{history}")

        if retrieved_chunks:
            chunks_text = "\n\n".join([
                f"[Source {i+1}] {chunk.content[:300]}..."
                for i, chunk in enumerate(retrieved_chunks[:10])
            ])
            parts.append(f"\nRetrieved information:\n{chunks_text}")

        parts.append("\nWhat should be the next step? Think carefully, then select a tool.")

        return "\n".join(parts)

    async def _execute_tool(
        self,
        action: AgentAction,
        namespace: str,
    ) -> AgentObservation:
        """Execute a tool and return the observation."""
        start_time = time.perf_counter()

        tool = self.tools.get(action.tool)
        if not tool:
            return AgentObservation(
                action=action,
                result=None,
                success=False,
                error=f"Unknown tool: {action.tool}",
                latency_ms=0,
            )

        try:
            # Inject namespace for retrieval tools
            if action.tool in (ToolType.RETRIEVE, ToolType.RETRIEVE_MORE):
                action.tool_input["namespace"] = namespace

            result = await tool.execute(**action.tool_input)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return AgentObservation(
                action=action,
                result=result,
                success=True,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return AgentObservation(
                action=action,
                result=None,
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
            )

    async def _decompose_query(
        self,
        question: str,
    ) -> list[SubQuery]:
        """Decompose a complex query into sub-queries."""
        prompt = f"""Break down this question into simpler sub-questions that can be answered independently.

Question: {question}

For each sub-question, explain why it's needed.
If the question is simple and doesn't need decomposition, return just the original question.

Format as JSON:
[
  {{"query": "sub-question 1", "reasoning": "why needed", "depends_on": []}},
  {{"query": "sub-question 2", "reasoning": "why needed", "depends_on": [0]}}
]"""

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            content = response.choices[0].message.content
            # Handle both array and object responses
            data = json.loads(content)
            if isinstance(data, dict):
                data = data.get("queries", data.get("sub_queries", [data]))
            return [SubQuery(**sq) for sq in data]
        except:
            return [SubQuery(query=question, reasoning="Original question")]

    async def _self_evaluate(
        self,
        question: str,
        answer: str,
        steps: list[AgentStep],
    ) -> tuple[str, float, int]:
        """Self-evaluate the answer quality."""
        prompt = f"""Evaluate this answer to the question.

Question: {question}

Answer: {answer}

Consider:
1. Does the answer fully address the question?
2. Is the answer supported by the retrieved information?
3. Are there any gaps or inaccuracies?

Provide:
1. A brief evaluation (2-3 sentences)
2. A confidence score from 0.0 to 1.0

Format as JSON:
{{"evaluation": "...", "confidence": 0.X}}"""

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        tokens = response.usage.total_tokens

        try:
            data = json.loads(response.choices[0].message.content)
            return data["evaluation"], data["confidence"], tokens
        except:
            return "Unable to evaluate.", 0.5, tokens

    def _extract_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        """Extract citations from retrieved chunks used in the answer."""
        citations = []

        # Simple heuristic: chunks whose content appears in the answer
        # In production, use more sophisticated attribution
        for chunk in chunks[:10]:  # Limit citations
            # Check for content overlap
            chunk_words = set(chunk.content.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(chunk_words & answer_words) / len(chunk_words)

            if overlap > 0.3:  # 30% word overlap threshold
                citations.append(Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content_snippet=chunk.content[:200],
                    relevance=f"{overlap:.0%} content overlap",
                ))

        return citations

    async def run_streaming(
        self,
        query: AgentQuery,
    ) -> AsyncIterator[AgentStep | AgentResponse]:
        """Streaming version that yields steps as they complete."""
        # Similar to run() but yields steps
        # Implementation would follow same pattern with yields
        response = await self.run(query)
        for step in response.steps:
            yield step
        yield response
```

### Agent Tools

```python
class RetrieveTool(ToolProtocol):
    """Tool for searching the vector store."""

    def __init__(self, retriever: RetrieverProtocol, reranker: RerankerProtocol | None):
        self.retriever = retriever
        self.reranker = reranker

    @property
    def name(self) -> ToolType:
        return ToolType.RETRIEVE

    @property
    def description(self) -> str:
        return "Search the knowledge base for information relevant to a query."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 5,
        **kwargs,
    ) -> list[dict]:
        retrieval_query = RetrievalQuery(
            text=query,
            namespace=namespace,
            config=RetrievalConfig(
                strategy=RetrievalStrategy.HYBRID,
                top_k=top_k * 2 if self.reranker else top_k,
            ),
        )

        result = await self.retriever.retrieve(retrieval_query)

        # Rerank if available
        if self.reranker and result.chunks:
            rerank_result = await self.reranker.rerank(RerankRequest(
                query=query,
                chunks=result.chunks,
                config=RerankerConfig(top_k=top_k),
            ))
            chunks = [
                {
                    "chunk_id": str(c.chunk_id),
                    "content": c.content,
                    "score": c.reranked_score,
                }
                for c in rerank_result.chunks
            ]
        else:
            chunks = [
                {
                    "chunk_id": str(c.chunk_id),
                    "content": c.content,
                    "score": c.score,
                }
                for c in result.chunks[:top_k]
            ]

        return chunks


class CalculateTool(ToolProtocol):
    """Tool for mathematical calculations."""

    @property
    def name(self) -> ToolType:
        return ToolType.CALCULATE

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Supports basic arithmetic and common functions."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str, **kwargs) -> str:
        import math
        import re

        # Safe evaluation with limited functions
        safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "sqrt": math.sqrt,
            "pow": pow,
            "log": math.log,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }

        # Sanitize expression
        expression = re.sub(r'[^0-9+\-*/().a-z_ ]', '', expression.lower())

        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"


class SummarizeTool(ToolProtocol):
    """Tool for summarizing retrieved content."""

    def __init__(self, llm_client, model: str):
        self.llm = llm_client
        self.model = model

    @property
    def name(self) -> ToolType:
        return ToolType.SUMMARIZE

    @property
    def description(self) -> str:
        return "Summarize a collection of text passages into a concise overview."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of text passages to summarize",
                },
                "focus": {
                    "type": "string",
                    "description": "What aspect to focus the summary on",
                },
            },
            "required": ["texts"],
        }

    async def execute(
        self,
        texts: list[str],
        focus: str | None = None,
        **kwargs,
    ) -> str:
        combined = "\n\n---\n\n".join(texts[:10])

        prompt = f"""Summarize these passages concisely.

Passages:
{combined}

{"Focus on: " + focus if focus else ""}

Summary:"""

        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        return response.choices[0].message.content


class FinishTool(ToolProtocol):
    """Tool for completing the agent task."""

    @property
    def name(self) -> ToolType:
        return ToolType.FINISH

    @property
    def description(self) -> str:
        return "Complete the task and provide the final answer."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the question",
                },
            },
            "required": ["answer"],
        }

    async def execute(self, answer: str, **kwargs) -> str:
        return answer
```

### Memory Manager

```python
class ConversationMemory:
    """Manages conversation context for multi-turn interactions."""

    def __init__(self, max_turns: int = 10, max_tokens: int = 4000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history: list[tuple[str, str]] = []  # (user, assistant)
        self.retrieved_chunks: list[RetrievedChunk] = []

    def add_turn(self, user: str, assistant: str):
        """Add a conversation turn."""
        self.history.append((user, assistant))
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def add_chunks(self, chunks: list[RetrievedChunk]):
        """Add retrieved chunks to memory."""
        self.retrieved_chunks.extend(chunks)
        # Deduplicate by chunk_id
        seen = set()
        unique = []
        for chunk in self.retrieved_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique.append(chunk)
        self.retrieved_chunks = unique[-50:]  # Keep last 50

    def get_context(self) -> str:
        """Get formatted context for the agent."""
        parts = []

        if self.history:
            history_text = "\n".join([
                f"User: {u}\nAssistant: {a}"
                for u, a in self.history[-5:]
            ])
            parts.append(f"Conversation history:\n{history_text}")

        if self.retrieved_chunks:
            chunks_text = "\n".join([
                f"- {c.content[:200]}..."
                for c in self.retrieved_chunks[-10:]
            ])
            parts.append(f"Previously retrieved:\n{chunks_text}")

        return "\n\n".join(parts)

    def clear(self):
        """Clear all memory."""
        self.history = []
        self.retrieved_chunks = []
```

## Agent Loop Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                     User Question                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Query Decompose │ (optional)
                    └─────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │         ReAct Loop            │
              │  ┌─────────────────────────┐  │
              │  │       REASON            │  │
              │  │  (Think about next step)│  │
              │  └───────────┬─────────────┘  │
              │              │                │
              │              ▼                │
              │  ┌─────────────────────────┐  │
              │  │         ACT             │  │
              │  │   (Execute tool)        │  │
              │  │  - retrieve             │  │
              │  │  - calculate            │  │
              │  │  - summarize            │  │
              │  │  - finish               │  │
              │  └───────────┬─────────────┘  │
              │              │                │
              │              ▼                │
              │  ┌─────────────────────────┐  │
              │  │       OBSERVE           │  │
              │  │  (Process tool result)  │  │
              │  └───────────┬─────────────┘  │
              │              │                │
              │              ▼                │
              │     ┌───────────────┐         │
              │     │ Done? / Max?  │─── No ──┼─┐
              │     └───────────────┘         │ │
              │              │ Yes            │ │
              └──────────────┼────────────────┘ │
                             │                  │
                             ▼                  │
                    ┌─────────────────┐         │
                    │ Self-Evaluate   │ ◄───────┘
                    └─────────────────┘
                             │
                             ▼
              ┌───────────────────────────────┐
              │     Final Answer + Citations  │
              └───────────────────────────────┘
```

## Dependencies

```toml
[project.dependencies]
openai = "^1.14"
```

## Testing Strategy

1. **Unit Tests**: Each tool in isolation
2. **Integration Tests**: Full agent loop with mocked LLM
3. **Behavior Tests**: Specific query patterns (multi-hop, calculations)
4. **Regression Tests**: Known questions with expected answers
5. **Performance Tests**: Step counts and latency distribution
