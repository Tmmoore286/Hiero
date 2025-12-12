from __future__ import annotations

from hiero.ingestion.base import Document

from .base import ChunkStrategy, ChunkerProtocol, ChunkingConfig
from .fixed import FixedChunker
from .semantic import SemanticChunker
from .tokenizer import Tokenizer


class AdaptiveChunker(ChunkerProtocol):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.semantic = SemanticChunker(tokenizer)
        self.fixed = FixedChunker(tokenizer)

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.ADAPTIVE

    async def chunk(self, document: Document, config: ChunkingConfig):
        if config.strategy == ChunkStrategy.SEMANTIC:
            return await self.semantic.chunk(document, config)
        if config.strategy == ChunkStrategy.FIXED:
            return await self.fixed.chunk(document, config)
        if config.strategy != ChunkStrategy.ADAPTIVE:
            raise NotImplementedError(
                f"Chunking strategy not supported yet: {config.strategy}"
            )

        file_type = (document.metadata.file_type or "").lower()
        if file_type in {"pdf", "txt", "md", "markdown", "docx", "html"}:
            semantic_config = config.model_copy(update={"strategy": ChunkStrategy.SEMANTIC})
            return await self.semantic.chunk(document, semantic_config)

        fixed_config = config.model_copy(update={"strategy": ChunkStrategy.FIXED})
        return await self.fixed.chunk(document, fixed_config)

