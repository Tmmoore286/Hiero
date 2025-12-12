import sys
import uuid

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.embedding.base import EMBEDDING_MODELS, BatchEmbeddingResult, EmbeddingResult
from hiero.retrieval import DenseRetriever, RetrievalConfig, RetrievalQuery
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
    async def search_dense(self, namespace, query_embedding, top_k=10, **kwargs):
        doc_id = uuid.uuid4()
        return [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=doc_id,
                content="first",
                score=0.9,
                metadata={},
                chunk_index=0,
            ),
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=doc_id,
                content="second",
                score=0.8,
                metadata={},
                chunk_index=1,
            ),
        ][:top_k]


@pytest.mark.asyncio
async def test_dense_retriever_ranks_results():
    retriever = DenseRetriever(_FakeStore(), _FakeEmbedder())
    query = RetrievalQuery(text="q", namespace="default", config=RetrievalConfig(top_k=2))
    result = await retriever.retrieve(query)
    assert result.total_candidates == 2
    assert result.chunks[0].rank == 1
    assert result.chunks[0].content == "first"
    assert result.chunks[1].rank == 2

