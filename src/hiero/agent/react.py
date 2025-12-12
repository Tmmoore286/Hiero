from __future__ import annotations

import json
from time import perf_counter
from typing import Any, AsyncIterator

from .base import (
    AgentAction,
    AgentConfig,
    AgentObservation,
    AgentProtocol,
    AgentQuery,
    AgentResponse,
    AgentStep,
    ToolProtocol,
    ToolType,
)
from .llm import ChatMessage, ChatModel


class ReActAgent(AgentProtocol):
    def __init__(self, llm: ChatModel, tools: list[ToolProtocol]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    async def run(self, query: AgentQuery) -> AgentResponse:
        start = perf_counter()
        steps: list[AgentStep] = []
        tokens_used = 0
        retrieval_calls = 0
        scratch: dict[str, Any] = {"retrieved": []}

        for step_num in range(1, query.config.max_steps + 1):
            action, tokens = await self._next_action(query, query.config, steps, scratch)
            tokens_used += tokens

            obs = await self._execute_action(action, query, scratch)
            if action.tool in (ToolType.RETRIEVE, ToolType.RETRIEVE_MORE):
                retrieval_calls += 1

            steps.append(AgentStep(step_number=step_num, action=action, observation=obs))

            if action.tool == ToolType.FINISH and obs.success:
                result = obs.result or {}
                return AgentResponse(
                    question=query.question,
                    answer=str(result.get("answer", "")),
                    citations=result.get("citations", []) or [],
                    steps=steps,
                    total_steps=len(steps),
                    total_retrieval_calls=retrieval_calls,
                    confidence=float(result.get("confidence", 0.8)),
                    total_latency_ms=(perf_counter() - start) * 1000,
                    llm_tokens_used=tokens_used,
                )

        return AgentResponse(
            question=query.question,
            answer="I couldn't complete the task within the step limit.",
            citations=[],
            steps=steps,
            total_steps=len(steps),
            total_retrieval_calls=retrieval_calls,
            confidence=0.0,
            total_latency_ms=(perf_counter() - start) * 1000,
            llm_tokens_used=tokens_used,
        )

    async def run_streaming(self, query: AgentQuery) -> AsyncIterator[AgentStep | AgentResponse]:
        resp = await self.run(query)
        for step in resp.steps:
            yield step
        yield resp

    async def _execute_action(self, action: AgentAction, query: AgentQuery, scratch: dict[str, Any]) -> AgentObservation:
        start = perf_counter()
        try:
            if action.tool == ToolType.FINISH:
                return AgentObservation(
                    action=action,
                    result=action.tool_input,
                    success=True,
                    latency_ms=(perf_counter() - start) * 1000,
                )

            tool = self.tools.get(action.tool)
            if tool is None:
                raise ValueError(f"Tool not available: {action.tool}")

            result = await tool.execute(**action.tool_input)
            if action.tool in (ToolType.RETRIEVE, ToolType.RETRIEVE_MORE):
                scratch["retrieved"] = result.get("chunks", [])
            return AgentObservation(
                action=action,
                result=result,
                success=True,
                latency_ms=(perf_counter() - start) * 1000,
            )
        except Exception as exc:
            return AgentObservation(
                action=action,
                result=None,
                success=False,
                error=str(exc),
                latency_ms=(perf_counter() - start) * 1000,
            )

    async def _next_action(
        self,
        query: AgentQuery,
        config: AgentConfig,
        steps: list[AgentStep],
        scratch: dict[str, Any],
    ) -> tuple[AgentAction, int]:
        tool_list = ", ".join([t.value for t in config.enabled_tools if t in self.tools or t == ToolType.FINISH])
        last_obs = steps[-1].observation.result if steps else None
        retrieved = scratch.get("retrieved", [])

        system = (
            "You are an agent that answers questions using tools. "
            "Pick the next tool and inputs. Respond ONLY as JSON.\n\n"
            "JSON schema:\n"
            "{\"tool\": \"retrieve|retrieve_more|calculate|summarize|finish\", "
            "\"tool_input\": {..}, \"thought\": \"...\"}\n\n"
            f"Available tools: {tool_list}\n"
            "Rules:\n"
            "- Use retrieve first unless you already have enough context.\n"
            "- finish.tool_input MUST include: {\"answer\": \"...\", \"citations\": [], \"confidence\": 0-1}\n"
        )
        user = {
            "question": query.question,
            "context": query.context,
            "history": query.history[-5:],
            "last_observation": last_obs,
            "retrieved_chunks": retrieved[: config.retrieval_top_k],
        }
        content, tokens = await self.llm.complete(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=json.dumps(user))],
            temperature=config.temperature,
        )
        action = _parse_action(content)
        if action.tool == ToolType.RETRIEVE and "query" not in action.tool_input:
            action.tool_input["query"] = query.question
            action.tool_input.setdefault("top_k", config.retrieval_top_k)
        if action.tool == ToolType.RETRIEVE_MORE and "query" not in action.tool_input:
            action.tool_input["query"] = query.question
            action.tool_input.setdefault("top_k", config.retrieval_top_k)
        return action, tokens


def _parse_action(text: str) -> AgentAction:
    try:
        data = json.loads(text)
        return AgentAction(
            tool=ToolType(data["tool"]),
            tool_input=data.get("tool_input") or {},
            thought=data.get("thought") or "",
        )
    except Exception:
        # Safe fallback: do a retrieval first.
        return AgentAction(tool=ToolType.RETRIEVE, tool_input={}, thought="fallback")

