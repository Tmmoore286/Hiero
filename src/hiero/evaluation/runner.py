from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from hiero import Hiero
from hiero.evaluation.dataset import EvalDataset
from hiero.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


@dataclass(frozen=True)
class QueryMetrics:
    query_id: str
    recall_at_5: float
    precision_at_5: float
    ndcg_at_5: float
    mrr: float
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    latency_ms: float


def load_dataset(path: str | Path) -> EvalDataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalDataset.model_validate(data)


async def run_retrieval_eval(
    dataset: EvalDataset,
    *,
    database_url: str,
    openai_api_key: str,
    namespace: str = "default",
) -> dict:
    start = perf_counter()
    per_query: list[QueryMetrics] = []

    async with Hiero(database_url=database_url, openai_api_key=openai_api_key, namespace=namespace) as h:
        for q in dataset.queries:
            t0 = perf_counter()
            retrieval = await h.retrieve(q.question, namespace=namespace)
            retrieved_doc_ids = [str(c.document_id) for c in retrieval.chunks]
            relevant_doc_ids = q.relevant_doc_ids

            per_query.append(
                QueryMetrics(
                    query_id=q.query_id,
                    recall_at_5=recall_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
                    precision_at_5=precision_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
                    ndcg_at_5=ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
                    mrr=mrr(retrieved_doc_ids, relevant_doc_ids),
                    retrieved_doc_ids=retrieved_doc_ids,
                    relevant_doc_ids=relevant_doc_ids,
                    latency_ms=(perf_counter() - t0) * 1000,
                )
            )

    avg = lambda xs: sum(xs) / max(len(xs), 1)
    summary = {
        "dataset": dataset.name,
        "queries": len(per_query),
        "avg_recall@5": avg([m.recall_at_5 for m in per_query]),
        "avg_precision@5": avg([m.precision_at_5 for m in per_query]),
        "avg_ndcg@5": avg([m.ndcg_at_5 for m in per_query]),
        "avg_mrr": avg([m.mrr for m in per_query]),
        "total_time_s": perf_counter() - start,
    }
    return {"summary": summary, "per_query": [asdict(m) for m in per_query]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hiero evaluation runner")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--database-url", required=True, help="Postgres database URL")
    parser.add_argument("--openai-api-key", required=True, help="OpenAI API key")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--out", default="eval_results.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    import asyncio

    results = asyncio.run(
        run_retrieval_eval(
            dataset,
            database_url=args.database_url,
            openai_api_key=args.openai_api_key,
            namespace=args.namespace,
        )
    )
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

