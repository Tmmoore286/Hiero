from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def _get_encoding(name: str):
    return tiktoken.get_encoding(name)


class Tokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self.encoding = _get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    def take_last_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[-max_tokens:])

    def split_by_tokens(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[str, int, int]]:
        tokens = self.encoding.encode(text)
        chunks: list[tuple[str, int, int]] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            char_start = len(self.encoding.decode(tokens[:start]))
            char_end = len(self.encoding.decode(tokens[:end]))
            chunks.append((chunk_text, char_start, char_end))

            start = end - overlap if end < len(tokens) else end
        return chunks

