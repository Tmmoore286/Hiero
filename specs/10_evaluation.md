# Component Spec: Evaluation Suite

## Overview

The Evaluation Suite provides comprehensive metrics for both retrieval quality and generation quality. It supports automated evaluation, LLM-as-judge scoring, and benchmark comparison with one-command execution.

## Requirements

### Functional
- **FR-1**: Retrieval metrics (Recall@k, nDCG, MRR, Precision@k)
- **FR-2**: Generation metrics (factuality, groundedness, coherence)
- **FR-3**: LLM-as-judge evaluation pipeline
- **FR-4**: End-to-end RAG metrics
- **FR-5**: Benchmark datasets support
- **FR-6**: Ablation experiment support
- **FR-7**: One-command evaluation runner
- **FR-8**: Results export (JSON, CSV, plots)

### Non-Functional
- **NFR-1**: Evaluate 100 queries in < 10 minutes
- **NFR-2**: Reproducible results with seed control
- **NFR-3**: Detailed per-query breakdowns

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID
from typing import Any


class MetricType(str, Enum):
    # Retrieval metrics
    RECALL_AT_K = "recall@k"
    PRECISION_AT_K = "precision@k"
    NDCG_AT_K = "ndcg@k"
    MRR = "mrr"
    MAP = "map"

    # Generation metrics
    FACTUALITY = "factuality"
    GROUNDEDNESS = "groundedness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"

    # End-to-end
    ANSWER_CORRECTNESS = "answer_correctness"
    CITATION_ACCURACY = "citation_accuracy"


class EvalQuery(BaseModel):
    """A query in an evaluation dataset."""
    query_id: str
    question: str
    ground_truth_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """Evaluation dataset with queries and ground truth."""
    name: str
    queries: list[EvalQuery]
    description: str | None = None
    version: str = "1.0"


class MetricResult(BaseModel):
    """Result for a single metric."""
    metric: MetricType
    value: float
    k: int | None = None  # For @k metrics
    details: dict = Field(default_factory=dict)


class QueryEvalResult(BaseModel):
    """Evaluation result for a single query."""
    query_id: str
    question: str
    predicted_answer: str | None = None
    ground_truth_answer: str | None = None
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    metrics: list[MetricResult]
    latency_ms: float


class EvalRunResult(BaseModel):
    """Complete evaluation run result."""
    run_id: str
    dataset_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Aggregate metrics
    aggregate_metrics: dict[str, float]

    # Per-query results
    query_results: list[QueryEvalResult]

    # Configuration
    config: dict

    # Performance
    total_queries: int
    total_time_seconds: float


class EvalConfig(BaseModel):
    """Configuration for evaluation run."""
    dataset_name: str
    namespace: str = "default"

    # Retrieval config
    retrieval_top_k: list[int] = [1, 3, 5, 10]

    # Generation evaluation
    evaluate_generation: bool = True
    use_llm_judge: bool = True
    judge_model: str = "gpt-4o"

    # Execution
    batch_size: int = 10
    seed: int = 42
```

## Interfaces

```python
from abc import ABC, abstractmethod


class MetricCalculator(ABC):
    """Base class for metric calculators."""

    @abstractmethod
    def calculate(
        self,
        retrieved: list[str],
        relevant: list[str],
        **kwargs,
    ) -> MetricResult:
        """Calculate the metric value."""
        ...


class EvaluatorProtocol(ABC):
    """Base protocol for evaluators."""

    @abstractmethod
    async def evaluate(
        self,
        dataset: EvalDataset,
        config: EvalConfig,
    ) -> EvalRunResult:
        """
        Run evaluation on a dataset.

        Args:
            dataset: Evaluation dataset
            config: Evaluation configuration

        Returns:
            Complete evaluation results
        """
        ...
```

## Implementation Details

### Retrieval Metrics

```python
import numpy as np
from typing import List


class RecallAtK(MetricCalculator):
    """
    Recall@k: Proportion of relevant documents retrieved in top k.

    Recall@k = |{relevant} ∩ {retrieved@k}| / |{relevant}|
    """

    def __init__(self, k: int):
        self.k = k

    def calculate(
        self,
        retrieved: list[str],
        relevant: list[str],
        **kwargs,
    ) -> MetricResult:
        if not relevant:
            return MetricResult(
                metric=MetricType.RECALL_AT_K,
                value=1.0 if not retrieved else 0.0,
                k=self.k,
            )

        retrieved_at_k = set(retrieved[:self.k])
        relevant_set = set(relevant)
        hits = len(retrieved_at_k & relevant_set)

        return MetricResult(
            metric=MetricType.RECALL_AT_K,
            value=hits / len(relevant_set),
            k=self.k,
            details={
                "hits": hits,
                "total_relevant": len(relevant_set),
            },
        )


class PrecisionAtK(MetricCalculator):
    """
    Precision@k: Proportion of retrieved documents that are relevant.

    Precision@k = |{relevant} ∩ {retrieved@k}| / k
    """

    def __init__(self, k: int):
        self.k = k

    def calculate(
        self,
        retrieved: list[str],
        relevant: list[str],
        **kwargs,
    ) -> MetricResult:
        retrieved_at_k = set(retrieved[:self.k])
        relevant_set = set(relevant)
        hits = len(retrieved_at_k & relevant_set)

        return MetricResult(
            metric=MetricType.PRECISION_AT_K,
            value=hits / self.k if self.k > 0 else 0.0,
            k=self.k,
            details={"hits": hits},
        )


class NDCG(MetricCalculator):
    """
    Normalized Discounted Cumulative Gain.

    Measures ranking quality with position-aware relevance.

    DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k
    """

    def __init__(self, k: int):
        self.k = k

    def calculate(
        self,
        retrieved: list[str],
        relevant: list[str],
        relevance_scores: dict[str, float] | None = None,
        **kwargs,
    ) -> MetricResult:
        if not relevant:
            return MetricResult(
                metric=MetricType.NDCG_AT_K,
                value=1.0,
                k=self.k,
            )

        # Binary relevance if no scores provided
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant}

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:self.k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)

        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:self.k]
        idcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(ideal_scores)
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return MetricResult(
            metric=MetricType.NDCG_AT_K,
            value=ndcg,
            k=self.k,
            details={"dcg": dcg, "idcg": idcg},
        )


class MRR(MetricCalculator):
    """
    Mean Reciprocal Rank.

    MRR = 1/|Q| Σ 1/rank_i

    Where rank_i is the rank of the first relevant document for query i.
    """

    def calculate(
        self,
        retrieved: list[str],
        relevant: list[str],
        **kwargs,
    ) -> MetricResult:
        relevant_set = set(relevant)

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return MetricResult(
                    metric=MetricType.MRR,
                    value=1.0 / (i + 1),
                    details={"first_relevant_rank": i + 1},
                )

        return MetricResult(
            metric=MetricType.MRR,
            value=0.0,
            details={"first_relevant_rank": None},
        )


class MAP(MetricCalculator):
    """
    Mean Average Precision.

    AP = (1/R) Σ (P@k × rel_k)

    Average over queries for MAP.
    """

    def calculate(
        self,
        retrieved: list[str],
        relevant: list[str],
        **kwargs,
    ) -> MetricResult:
        if not relevant:
            return MetricResult(metric=MetricType.MAP, value=1.0)

        relevant_set = set(relevant)
        hits = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i

        ap = sum_precisions / len(relevant_set) if relevant_set else 0.0

        return MetricResult(
            metric=MetricType.MAP,
            value=ap,
            details={"total_hits": hits},
        )
```

### Generation Metrics (LLM-as-Judge)

```python
import json
from openai import AsyncOpenAI


class LLMJudge:
    """
    LLM-based evaluation for generation quality.
    """

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    async def evaluate_factuality(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> MetricResult:
        """
        Evaluate factual accuracy of the answer.

        Checks if claims are supported by the context.
        """
        prompt = f"""Evaluate the factual accuracy of this answer based on the provided context.

Question: {question}

Context:
{self._format_context(context)}

Answer: {answer}

For each claim in the answer, determine if it's:
1. SUPPORTED: Directly supported by the context
2. NOT_SUPPORTED: Cannot be verified from context
3. CONTRADICTED: Contradicts the context

Output JSON:
{{
    "claims": [
        {{"claim": "...", "verdict": "SUPPORTED|NOT_SUPPORTED|CONTRADICTED", "evidence": "..."}}
    ],
    "factuality_score": 0.0-1.0,
    "explanation": "..."
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)

        return MetricResult(
            metric=MetricType.FACTUALITY,
            value=result["factuality_score"],
            details={
                "claims": result["claims"],
                "explanation": result["explanation"],
            },
        )

    async def evaluate_groundedness(
        self,
        answer: str,
        context: list[str],
    ) -> MetricResult:
        """
        Evaluate how well the answer is grounded in sources.

        Measures citation accuracy and source attribution.
        """
        prompt = f"""Evaluate how well this answer is grounded in the provided sources.

Sources:
{self._format_context(context)}

Answer: {answer}

Consider:
1. Does the answer cite sources appropriately?
2. Are the citations accurate?
3. Is any information presented without source attribution?

Output JSON:
{{
    "groundedness_score": 0.0-1.0,
    "cited_claims": ["..."],
    "uncited_claims": ["..."],
    "incorrect_citations": ["..."],
    "explanation": "..."
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)

        return MetricResult(
            metric=MetricType.GROUNDEDNESS,
            value=result["groundedness_score"],
            details=result,
        )

    async def evaluate_relevance(
        self,
        question: str,
        answer: str,
    ) -> MetricResult:
        """
        Evaluate how relevant the answer is to the question.
        """
        prompt = f"""Evaluate how relevant and complete this answer is to the question.

Question: {question}

Answer: {answer}

Consider:
1. Does the answer address the question directly?
2. Is the answer complete?
3. Is there any irrelevant information?

Output JSON:
{{
    "relevance_score": 0.0-1.0,
    "addresses_question": true/false,
    "completeness": "complete|partial|incomplete",
    "irrelevant_parts": ["..."],
    "missing_aspects": ["..."],
    "explanation": "..."
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)

        return MetricResult(
            metric=MetricType.RELEVANCE,
            value=result["relevance_score"],
            details=result,
        )

    async def evaluate_answer_correctness(
        self,
        question: str,
        predicted_answer: str,
        ground_truth: str,
    ) -> MetricResult:
        """
        Evaluate if predicted answer matches ground truth.

        Uses semantic similarity, not exact match.
        """
        prompt = f"""Compare these two answers to the same question.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted_answer}

Evaluate:
1. Do they convey the same information?
2. Are there factual disagreements?
3. Is the predicted answer more/less complete?

Output JSON:
{{
    "correctness_score": 0.0-1.0,
    "semantic_match": true/false,
    "factual_disagreements": ["..."],
    "missing_info": ["..."],
    "extra_info": ["..."],
    "explanation": "..."
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)

        return MetricResult(
            metric=MetricType.ANSWER_CORRECTNESS,
            value=result["correctness_score"],
            details=result,
        )

    def _format_context(self, context: list[str]) -> str:
        return "\n\n".join([
            f"[Source {i+1}]: {c}"
            for i, c in enumerate(context)
        ])
```

### Evaluation Runner

```python
import asyncio
import time
from uuid import uuid4


class RAGEvaluator(EvaluatorProtocol):
    """
    Full RAG evaluation pipeline.
    """

    def __init__(
        self,
        rag: "Hiero",
        llm_judge: LLMJudge,
    ):
        self.rag = rag
        self.judge = llm_judge

        # Metric calculators
        self.retrieval_metrics = {
            1: [RecallAtK(1), PrecisionAtK(1), NDCG(1)],
            3: [RecallAtK(3), PrecisionAtK(3), NDCG(3)],
            5: [RecallAtK(5), PrecisionAtK(5), NDCG(5)],
            10: [RecallAtK(10), PrecisionAtK(10), NDCG(10)],
        }

    async def evaluate(
        self,
        dataset: EvalDataset,
        config: EvalConfig,
    ) -> EvalRunResult:
        start_time = time.time()
        run_id = str(uuid4())[:8]

        query_results = []

        # Process in batches
        for i in range(0, len(dataset.queries), config.batch_size):
            batch = dataset.queries[i:i + config.batch_size]
            batch_results = await asyncio.gather(*[
                self._evaluate_query(q, config)
                for q in batch
            ])
            query_results.extend(batch_results)

        # Calculate aggregate metrics
        aggregate = self._aggregate_metrics(query_results)

        total_time = time.time() - start_time

        return EvalRunResult(
            run_id=run_id,
            dataset_name=dataset.name,
            aggregate_metrics=aggregate,
            query_results=query_results,
            config=config.model_dump(),
            total_queries=len(dataset.queries),
            total_time_seconds=total_time,
        )

    async def _evaluate_query(
        self,
        query: EvalQuery,
        config: EvalConfig,
    ) -> QueryEvalResult:
        start_time = time.time()

        # Retrieve
        retrieved_results = await self.rag.retrieve(
            query=query.question,
            top_k=max(config.retrieval_top_k),
            namespace=config.namespace,
        )
        retrieved_ids = [r["chunk_id"] for r in retrieved_results]

        # Calculate retrieval metrics
        metrics = []
        for k in config.retrieval_top_k:
            for calculator in self.retrieval_metrics.get(k, []):
                metric = calculator.calculate(
                    retrieved=retrieved_ids,
                    relevant=query.relevant_doc_ids,
                )
                metrics.append(metric)

        # MRR (doesn't depend on k)
        mrr = MRR().calculate(retrieved_ids, query.relevant_doc_ids)
        metrics.append(mrr)

        # Generation evaluation
        predicted_answer = None
        if config.evaluate_generation:
            response = await self.rag.query(
                question=query.question,
                namespace=config.namespace,
            )
            predicted_answer = response.answer

            if config.use_llm_judge:
                context = [r["content"] for r in retrieved_results[:5]]

                # Factuality
                factuality = await self.judge.evaluate_factuality(
                    query.question, predicted_answer, context
                )
                metrics.append(factuality)

                # Groundedness
                groundedness = await self.judge.evaluate_groundedness(
                    predicted_answer, context
                )
                metrics.append(groundedness)

                # Relevance
                relevance = await self.judge.evaluate_relevance(
                    query.question, predicted_answer
                )
                metrics.append(relevance)

                # Answer correctness (if ground truth available)
                if query.ground_truth_answer:
                    correctness = await self.judge.evaluate_answer_correctness(
                        query.question,
                        predicted_answer,
                        query.ground_truth_answer,
                    )
                    metrics.append(correctness)

        latency_ms = (time.time() - start_time) * 1000

        return QueryEvalResult(
            query_id=query.query_id,
            question=query.question,
            predicted_answer=predicted_answer,
            ground_truth_answer=query.ground_truth_answer,
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=query.relevant_doc_ids,
            metrics=metrics,
            latency_ms=latency_ms,
        )

    def _aggregate_metrics(
        self,
        results: list[QueryEvalResult],
    ) -> dict[str, float]:
        """Aggregate metrics across all queries."""
        aggregated = {}

        # Collect all metric values
        metric_values: dict[str, list[float]] = {}
        for result in results:
            for metric in result.metrics:
                key = f"{metric.metric.value}"
                if metric.k:
                    key += f"@{metric.k}"
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(metric.value)

        # Calculate means
        for key, values in metric_values.items():
            aggregated[key] = sum(values) / len(values)
            aggregated[f"{key}_std"] = np.std(values)

        return aggregated
```

### CLI Runner

```python
import click
import json
import asyncio


@click.command()
@click.option("--dataset", required=True, help="Path to evaluation dataset JSON")
@click.option("--output", default="eval_results.json", help="Output file")
@click.option("--namespace", default="default", help="Namespace to evaluate")
@click.option("--no-generation", is_flag=True, help="Skip generation evaluation")
@click.option("--no-llm-judge", is_flag=True, help="Skip LLM-as-judge")
def run_eval(dataset, output, namespace, no_generation, no_llm_judge):
    """Run evaluation on a dataset."""

    async def _run():
        # Load dataset
        with open(dataset) as f:
            data = json.load(f)
        eval_dataset = EvalDataset(**data)

        # Initialize
        rag = Hiero(...)
        await rag.initialize()

        judge = LLMJudge(...) if not no_llm_judge else None
        evaluator = RAGEvaluator(rag, judge)

        # Run evaluation
        config = EvalConfig(
            dataset_name=eval_dataset.name,
            namespace=namespace,
            evaluate_generation=not no_generation,
            use_llm_judge=not no_llm_judge,
        )

        result = await evaluator.evaluate(eval_dataset, config)

        # Save results
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

        # Print summary
        click.echo(f"\n{'='*50}")
        click.echo(f"Evaluation Results: {result.run_id}")
        click.echo(f"{'='*50}")
        click.echo(f"Dataset: {result.dataset_name}")
        click.echo(f"Queries: {result.total_queries}")
        click.echo(f"Time: {result.total_time_seconds:.2f}s")
        click.echo(f"\nAggregate Metrics:")
        for metric, value in sorted(result.aggregate_metrics.items()):
            if not metric.endswith("_std"):
                click.echo(f"  {metric}: {value:.4f}")

    asyncio.run(_run())


if __name__ == "__main__":
    run_eval()
```

### Benchmark Datasets

```python
# Built-in benchmark datasets
BENCHMARK_DATASETS = {
    "squad_mini": {
        "description": "Mini SQuAD subset for quick evaluation",
        "size": 100,
        "url": "https://...",
    },
    "natural_questions": {
        "description": "Natural Questions dataset",
        "size": 1000,
        "url": "https://...",
    },
    "hotpotqa": {
        "description": "HotPotQA multi-hop dataset",
        "size": 500,
        "url": "https://...",
    },
}


async def load_benchmark(name: str) -> EvalDataset:
    """Load a benchmark dataset."""
    if name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark: {name}")

    # Download and parse dataset
    ...
```

## Evaluation Metrics Summary

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| Recall@k | Retrieval | [0, 1] | Relevant docs in top k |
| Precision@k | Retrieval | [0, 1] | Precision at cutoff k |
| nDCG@k | Retrieval | [0, 1] | Position-aware relevance |
| MRR | Retrieval | [0, 1] | Rank of first relevant |
| MAP | Retrieval | [0, 1] | Mean average precision |
| Factuality | Generation | [0, 1] | Claims supported by context |
| Groundedness | Generation | [0, 1] | Source attribution quality |
| Relevance | Generation | [0, 1] | Answer addresses question |
| Correctness | Generation | [0, 1] | Match to ground truth |

## Dependencies

```toml
[project.dependencies]
numpy = "^1.26"
click = "^8.1"
matplotlib = "^3.8"  # For plots
pandas = "^2.0"      # For data analysis
```

## Example Dataset Format

```json
{
    "name": "my_eval_dataset",
    "description": "Custom evaluation dataset",
    "version": "1.0",
    "queries": [
        {
            "query_id": "q1",
            "question": "What is the capital of France?",
            "ground_truth_answer": "Paris is the capital of France.",
            "relevant_doc_ids": ["doc_1", "doc_5"],
            "metadata": {"category": "geography"}
        }
    ]
}
```
