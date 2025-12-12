from __future__ import annotations

import re
from time import perf_counter

from openai import AsyncOpenAI

from .base import (
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    GeneratorProtocol,
    InlineCitation,
    SourceContext,
)


class GroundedGenerator(GeneratorProtocol):
    CITATION_RE = re.compile(r"\[(\d+)\]")

    def __init__(self, api_key: str, provider: str = "openai", base_url: str | None = None):
        if provider != "openai":
            raise ValueError(f"Unsupported provider for MVP: {provider}")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = perf_counter()
        system_prompt = self._build_system_prompt(request.config, request.system_instructions)
        user_prompt = self._build_user_prompt(request.query, request.context)

        response = await self._generate_openai(system_prompt, user_prompt, request.config)
        parsed_text, citations = self._parse_citations(response["content"], request.context)

        grounding_score = (
            min(len(citations) / max(len(request.context), 1), 1.0)
            if request.config.require_grounding
            else 0.0
        )

        return GenerationResponse(
            query=request.query,
            response=parsed_text,
            citations=citations,
            sources_used=list({c.chunk_id for c in citations}),
            sources_provided=len(request.context),
            grounding_score=grounding_score,
            confidence=0.8 if citations or not request.config.require_grounding else 0.5,
            tokens_used=response.get("tokens", 0),
            latency_ms=(perf_counter() - start_time) * 1000,
        )

    async def generate_streaming(self, request: GenerationRequest):
        raise NotImplementedError("Streaming will be added in Phase 2.")

    async def _generate_openai(
        self, system_prompt: str, user_prompt: str, config: GenerationConfig
    ) -> dict:
        completion = await self.client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        content = completion.choices[0].message.content or ""
        tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0) or 0
        return {"content": content, "tokens": int(tokens)}

    def _build_system_prompt(
        self, config: GenerationConfig, system_instructions: str | None
    ) -> str:
        base = (
            "Answer the user's question using ONLY the provided sources. "
            "Every factual claim must be followed by an inline citation like [1]. "
            "If the sources do not contain the answer, say you don't know."
        )
        if system_instructions:
            base = f"{system_instructions}\n\n{base}"
        return base

    def _build_user_prompt(self, query: str, context: list[SourceContext]) -> str:
        sources = "\n\n".join(
            f"[{i+1}] {c.content}" for i, c in enumerate(context)
        )
        return f"Question:\n{query}\n\nSources:\n{sources}\n\nAnswer:"

    def _parse_citations(
        self, text: str, context: list[SourceContext]
    ) -> tuple[str, list[InlineCitation]]:
        citations: list[InlineCitation] = []
        for match in self.CITATION_RE.finditer(text):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(context):
                source = context[idx]
                citations.append(
                    InlineCitation(
                        source_index=idx,
                        chunk_id=source.chunk_id,
                        quoted_text="",
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )
        return text, citations

