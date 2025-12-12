import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, mrr


def test_recall_precision_mrr():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "x"]
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert mrr(retrieved, relevant) == 0.5


def test_ndcg():
    retrieved = ["a", "b", "c"]
    relevant = ["b", "c"]
    score = ndcg_at_k(retrieved, relevant, 3)
    assert 0.0 < score <= 1.0

