import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.embedding.base import EMBEDDING_MODELS, BatchEmbeddingResult, EmbeddingResult
from hiero.retrieval import HybridRetriever, RetrievalConfig, RetrievalQuery, RetrievalStrategy, FusionMethod
from hiero.storage.repository import SearchResult


class _FakeEmbedder:
    @property
    def model(self):
        return EMBEDDING_MODELS["text-embedding-3-small"]

    async def embed(self, text: str):
        return EmbeddingResult.from_text(text, [1.0, 0.0], self.model)

    async def embed_batch(self, texts, batch_size=100, normalize=True):
        results = [EmbeddingResult.from_text(t, [1.0, 0.0], self.model) for t in texts]
        return BatchEmbeddingResult(
            results=results,
            total_tokens=0,
            processing_time_ms=0.0,
            cache_hits=0,
            api_calls=1,
        )


class _FakeStore:
    def __init__(self):
        self.a = uuid.uuid4()
        self.b = uuid.uuid4()
        self.c = uuid.uuid4()
        self.doc = uuid.uuid4()

    async def search_dense(self, namespace, query_embedding, top_k=10, **kwargs):
        return [
            SearchResult(
                chunk_id=self.a,
                document_id=self.doc,
                content="dense-a",
                score=0.9,
                metadata={},
                chunk_index=0,
            ),
            SearchResult(
                chunk_id=self.b,
                document_id=self.doc,
                content="dense-b",
                score=0.8,
                metadata={},
                chunk_index=1,
            ),
        ]

    async def search_sparse(self, namespace, query_text, top_k=10, **kwargs):
        return [
            SearchResult(
                chunk_id=self.c,
                document_id=self.doc,
                content="sparse-c",
                score=2.0,
                metadata={},
                chunk_index=2,
            ),
            SearchResult(
                chunk_id=self.a,
                document_id=self.doc,
                content="sparse-a",
                score=1.5,
                metadata={},
                chunk_index=0,
            ),
        ]


@pytest.mark.asyncio
async def test_hybrid_retriever_rrf_fuses_and_sorts():
    store = _FakeStore()
    retriever = HybridRetriever(store, _FakeEmbedder())
    cfg = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
        dense_top_k=2,
        sparse_top_k=2,
        fusion_method=FusionMethod.RRF,
        dense_weight=0.7,
    )
    result = await retriever.retrieve(RetrievalQuery(text="q", config=cfg))
    ids = [c.chunk_id for c in result.chunks]
    assert set(ids) == {store.a, store.b, store.c}
    assert ids[0] == store.a

