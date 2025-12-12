from __future__ import annotations

from time import perf_counter

from hiero.ingestion.base import Document

from .base import Chunk, ChunkMetadata, ChunkStrategy, ChunkerProtocol, ChunkingConfig, _finish_result
from .tokenizer import Tokenizer


class FixedChunker(ChunkerProtocol):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.FIXED

    async def chunk(self, document: Document, config: ChunkingConfig):
        start_time = perf_counter()
        splits = self.tokenizer.split_by_tokens(
            document.content, config.target_chunk_size, config.chunk_overlap
        )
        chunks: list[Chunk] = []
        for i, (text, start_char, end_char) in enumerate(splits):
            token_count = self.tokenizer.count_tokens(text)
            chunks.append(
                Chunk(
                    content=text.strip(),
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        chunk_index=i,
                        start_char=start_char,
                        end_char=end_char,
                        strategy_used=self.strategy,
                        token_count=token_count,
                    ),
                )
            )
        return _finish_result(document, chunks, self.strategy, start_time)

