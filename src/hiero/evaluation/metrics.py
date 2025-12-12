from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not relevant:
        return 1.0 if not retrieved else 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / max(len(relevant_set), 1)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / len(retrieved_k)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    gains = [1.0 if r in relevant_set else 0.0 for r in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = [1.0] * min(len(relevant_set), k)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0

