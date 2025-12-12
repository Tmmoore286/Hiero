from __future__ import annotations

import re
from time import perf_counter

from hiero.ingestion.base import Document

from .base import Chunk, ChunkMetadata, ChunkStrategy, ChunkerProtocol, ChunkingConfig, _finish_result
from .tokenizer import Tokenizer


class SemanticChunker(ChunkerProtocol):
    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.SEMANTIC

    async def chunk(self, document: Document, config: ChunkingConfig) -> "ChunkingResult":
        start_time = perf_counter()
        sentences = self._split_sentences(document.content)

        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        current_start = 0
        current_end = 0
        chunk_index = 0

        overlap_text = ""

        for sentence, s_start, s_end in sentences:
            sentence_tokens = self.tokenizer.count_tokens(sentence)

            if not current_parts:
                current_start = s_start

            if current_tokens + sentence_tokens > config.max_chunk_size and current_parts:
                chunk_text = "".join(current_parts).strip()
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        metadata=ChunkMetadata(
                            document_id=document.id,
                            chunk_index=chunk_index,
                            start_char=current_start,
                            end_char=current_end,
                            strategy_used=self.strategy,
                            token_count=current_tokens,
                        ),
                    )
                )
                chunk_index += 1

                overlap_text = (
                    self.tokenizer.take_last_tokens(chunk_text, config.chunk_overlap)
                    if config.chunk_overlap > 0
                    else ""
                )
                current_parts = [overlap_text] if overlap_text else []
                current_tokens = self.tokenizer.count_tokens(overlap_text) if overlap_text else 0
                current_start = s_start

            current_parts.append(sentence)
            current_tokens += sentence_tokens
            current_end = s_end

        if current_parts:
            chunk_text = "".join(current_parts).strip()
            chunks.append(
                Chunk(
                    content=chunk_text,
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_end,
                        strategy_used=self.strategy,
                        token_count=current_tokens,
                    ),
                )
            )

        return _finish_result(document, chunks, self.strategy, start_time)

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        parts: list[tuple[str, int, int]] = []
        start = 0
        for match in self.SENTENCE_ENDINGS.finditer(text):
            end = match.start()
            sentence = text[start:end]
            if sentence.strip():
                parts.append((sentence, start, end))
            start = match.end()
        tail = text[start:]
        if tail.strip():
            parts.append((tail, start, len(text)))
        return parts

