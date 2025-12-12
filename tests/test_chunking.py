import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.chunking import (
    AdaptiveChunker,
    ChunkingConfig,
    FixedChunker,
    SemanticChunker,
    Tokenizer,
)
from hiero.ingestion.base import Document, DocumentMetadata, DocumentSource


@pytest.mark.asyncio
async def test_semantic_chunker_respects_max_tokens():
    doc = Document.from_content(
        "One. Two. Three.",
        DocumentMetadata(),
        DocumentSource.FILE_UPLOAD,
    )
    tokenizer = Tokenizer()
    chunker = SemanticChunker(tokenizer)
    cfg = ChunkingConfig(max_chunk_size=3, target_chunk_size=3, chunk_overlap=0)

    result = await chunker.chunk(doc, cfg)
    assert result.total_chunks == 3
    assert all(c.metadata.document_id == doc.id for c in result.chunks)


@pytest.mark.asyncio
async def test_fixed_chunker_splits_text():
    doc = Document.from_content(
        "A B C D E F G H I J K L",
        DocumentMetadata(file_type="txt"),
        DocumentSource.FILE_UPLOAD,
    )
    tokenizer = Tokenizer()
    chunker = FixedChunker(tokenizer)
    cfg = ChunkingConfig(strategy="fixed", target_chunk_size=2, chunk_overlap=0)
    result = await chunker.chunk(doc, cfg)
    assert result.total_chunks > 1
    assert all(c.metadata.strategy_used.value == "fixed" for c in result.chunks)


@pytest.mark.asyncio
async def test_adaptive_chunker_uses_semantic_for_text():
    doc = Document.from_content(
        "One. Two. Three.",
        DocumentMetadata(file_type="txt"),
        DocumentSource.FILE_UPLOAD,
    )
    tokenizer = Tokenizer()
    chunker = AdaptiveChunker(tokenizer)
    cfg = ChunkingConfig(strategy="adaptive", max_chunk_size=3, target_chunk_size=3, chunk_overlap=0)
    result = await chunker.chunk(doc, cfg)
    assert result.strategy_used.value == "semantic"
