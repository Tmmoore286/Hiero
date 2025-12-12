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
        scratch: dict[str, Any] = {"retrieved": [], "seen_chunk_ids": set()}

        for step_num in range(1, query.config.max_steps + 1):
            action, tokens = await self._next_action(query, query.config, steps, scratch)
            tokens_used += tokens

            obs = await self._execute_action(action, query, scratch)
            if action.tool in (ToolType.RETRIEVE, ToolType.RETRIEVE_MORE):
                retrieval_calls += 1
                if isinstance(obs.result, dict) and "chunks" in obs.result:
                    for c in obs.result["chunks"]:
                        cid = c.get("chunk_id") if isinstance(c, dict) else None
                        if cid:
                            scratch["seen_chunk_ids"].add(str(cid))
                if retrieval_calls >= query.config.max_retrieval_calls:
                    scratch["retrieval_budget_exhausted"] = True

            steps.append(AgentStep(step_number=step_num, action=action, observation=obs))

            if action.tool == ToolType.FINISH and obs.success:
                result = obs.result or {}
                confidence = float(result.get("confidence", 0.8))
                self_eval = None
                if query.config.enable_self_evaluation:
                    confidence, self_eval_tokens, self_eval = await self._self_evaluate(
                        query=query.question,
                        answer=str(result.get("answer", "")),
                        citations=result.get("citations", []) or [],
                        retrieved=scratch.get("retrieved", []),
                        temperature=query.config.temperature,
                    )
                    tokens_used += self_eval_tokens
                return AgentResponse(
                    question=query.question,
                    answer=str(result.get("answer", "")),
                    citations=result.get("citations", []) or [],
                    steps=steps,
                    total_steps=len(steps),
                    total_retrieval_calls=retrieval_calls,
                    confidence=confidence,
                    self_evaluation=self_eval,
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
            tool = self.tools.get(action.tool)
            if tool is None:
                raise ValueError(f"Tool not available: {action.tool}")

            tool_input = dict(action.tool_input or {})
            if action.tool in (ToolType.RETRIEVE_MORE,) and "exclude_chunk_ids" not in tool_input:
                tool_input["exclude_chunk_ids"] = list(scratch.get("seen_chunk_ids", set()))
            if action.tool == ToolType.FINISH:
                tool_input.setdefault("question", query.question)
                tool_input.setdefault("chunks", scratch.get("retrieved", []))

            result = await tool.execute(**tool_input)
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
        tool_list = ", ".join([t.value for t in config.enabled_tools if t in self.tools])
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
            "- Use retrieve_more if you need more sources.\n"
            "- Use finish when ready; finish.tool_input can be empty.\n"
        )
        if scratch.get("retrieval_budget_exhausted"):
            system += "\nRetrieval budget exhausted. You must finish with available sources.\n"
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
        if scratch.get("retrieval_budget_exhausted") and action.tool in (
            ToolType.RETRIEVE,
            ToolType.RETRIEVE_MORE,
        ):
            action = AgentAction(tool=ToolType.FINISH, tool_input={}, thought="budget exhausted")
        return action, tokens

    async def _self_evaluate(
        self,
        *,
        query: str,
        answer: str,
        citations: list,
        retrieved: list,
        temperature: float,
    ) -> tuple[float, int, str | None]:
        system = (
            "Evaluate the assistant answer for groundedness and relevance to the question. "
            "Return ONLY JSON: {\"confidence\": <0-1>, \"critique\": \"...\"}."
        )
        user = {
            "question": query,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved[:10],
        }
        text, tokens = await self.llm.complete(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=json.dumps(user)),
            ],
            temperature=temperature,
        )
        data = _safe_parse_json(text) or {}
        conf = data.get("confidence", 0.5)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.5
        conf_f = max(0.0, min(1.0, conf_f))
        critique = data.get("critique")
        return conf_f, int(tokens), str(critique) if critique is not None else None


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


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None
