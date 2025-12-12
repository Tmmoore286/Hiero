from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from hiero import Hiero
from hiero.agent import AgentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hiero agent CLI demo")
    p.add_argument("--database-url", default=os.getenv("DATABASE_URL"), required=False)
    p.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), required=False)
    p.add_argument("--namespace", default=os.getenv("HIERO_NAMESPACE", "default"))

    p.add_argument(
        "--ingest",
        action="append",
        default=[],
        help="Path(s) or URL(s) to ingest before running the agent (repeatable).",
    )
    p.add_argument("--question", required=True, help="Question to ask.")

    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-retrieval-calls", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--self-eval", action="store_true", help="Enable self-evaluation pass.")
    p.add_argument("--json", dest="as_json", action="store_true", help="Print full response JSON.")
    return p


async def main_async(args: argparse.Namespace) -> int:
    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")
    if not args.openai_api_key:
        raise SystemExit("--openai-api-key or OPENAI_API_KEY is required")

    async with Hiero(
        database_url=args.database_url,
        openai_api_key=args.openai_api_key,
        namespace=args.namespace,
    ) as h:
        for item in args.ingest:
            await h.ingest(item)

        config = AgentConfig(
            max_steps=args.max_steps,
            max_retrieval_calls=args.max_retrieval_calls,
            retrieval_top_k=args.top_k,
            enable_self_evaluation=args.self_eval,
        )
        resp = await h.agent_query(args.question, namespace=args.namespace, config=config)

        if args.as_json:
            print(resp.model_dump_json(indent=2))
            return 0

        print(resp.answer)
        if resp.citations:
            print("\nCitations:")
            for i, c in enumerate(resp.citations, start=1):
                print(f"- [{i}] doc={c.document_id} chunk={c.chunk_id}")
                snippet = (c.content_snippet or "").strip().replace("\n", " ")
                if snippet:
                    print(f"      {snippet[:200]}")

        print("\nTrace:")
        for s in resp.steps:
            status = "ok" if s.observation.success else "err"
            print(f"- step {s.step_number}: {s.action.tool.value} ({status})")
        if resp.self_evaluation:
            print(f"\nSelf-eval: {resp.self_evaluation} (confidence={resp.confidence:.2f})")
        return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

