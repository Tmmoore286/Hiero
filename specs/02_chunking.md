# Component Spec: Chunking

## Overview

The Chunking component splits documents into semantically meaningful segments optimized for embedding and retrieval. It uses an adaptive strategy that detects document type and applies the appropriate chunking method.

## Requirements

### Functional
- **FR-1**: Support multiple chunking strategies (semantic, recursive, fixed)
- **FR-2**: Auto-detect document type and select optimal strategy
- **FR-3**: Configurable chunk size and overlap
- **FR-4**: Preserve document structure (headers, sections) in metadata
- **FR-5**: Handle code blocks specially (don't split mid-function)
- **FR-6**: Support manual strategy override
- **FR-7**: Maintain parent-child relationships for hierarchical chunks

### Non-Functional
- **NFR-1**: Chunk 1MB document in < 1 second
- **NFR-2**: Consistent chunk sizes (within 20% of target)
- **NFR-3**: No semantic breaks mid-sentence

## Data Models

```python
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from enum import Enum


class ChunkStrategy(str, Enum):
    SEMANTIC = "semantic"      # Sentence-boundary aware
    RECURSIVE = "recursive"    # Header/section hierarchy
    FIXED = "fixed"           # Token windows with overlap
    CODE = "code"             # AST-aware for code files
    ADAPTIVE = "adaptive"     # Auto-select based on content


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk."""
    document_id: UUID
    chunk_index: int                    # Position in document
    start_char: int                     # Character offset start
    end_char: int                       # Character offset end
    section_title: str | None = None    # Enclosing section header
    section_hierarchy: list[str] = Field(default_factory=list)  # ["Chapter 1", "Section 1.1"]
    page_number: int | None = None      # For PDFs
    parent_chunk_id: UUID | None = None # For hierarchical chunking
    strategy_used: ChunkStrategy
    token_count: int


class Chunk(BaseModel):
    """A single chunk of text with metadata."""
    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: ChunkMetadata


class ChunkingConfig(BaseModel):
    """Configuration for chunking behavior."""
    strategy: ChunkStrategy = ChunkStrategy.ADAPTIVE
    target_chunk_size: int = 512        # Target tokens per chunk
    chunk_overlap: int = 64             # Overlap tokens between chunks
    min_chunk_size: int = 100           # Minimum chunk size (avoid tiny chunks)
    max_chunk_size: int = 1024          # Maximum chunk size
    respect_sentence_boundaries: bool = True
    preserve_code_blocks: bool = True
    tokenizer: str = "cl100k_base"      # OpenAI's tokenizer


class ChunkingResult(BaseModel):
    """Result of chunking a document."""
    document_id: UUID
    chunks: list[Chunk]
    total_chunks: int
    strategy_used: ChunkStrategy
    avg_chunk_size: float
    processing_time_ms: float
```

## Interfaces

```python
from abc import ABC, abstractmethod
from ..ingestion.base import Document


class ChunkerProtocol(ABC):
    """Base protocol for all chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> ChunkingResult:
        """
        Split document into chunks.

        Args:
            document: Normalized document from ingestion
            config: Chunking configuration

        Returns:
            ChunkingResult with list of Chunk objects
        """
        ...

    @property
    @abstractmethod
    def strategy(self) -> ChunkStrategy:
        """Return the strategy this chunker implements."""
        ...
```

## Implementation Details

### Tokenizer Wrapper

```python
import tiktoken
from functools import lru_cache


class Tokenizer:
    """Wrapper around tiktoken for token counting."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    def split_by_tokens(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[str, int, int]]:
        """
        Split text into token-bounded chunks.

        Returns:
            List of (chunk_text, start_char, end_char)
        """
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Find character offsets (approximate)
            # This is tricky - we need to map token positions to char positions
            char_start = len(self.encoding.decode(tokens[:start]))
            char_end = len(self.encoding.decode(tokens[:end]))

            chunks.append((chunk_text, char_start, char_end))

            # Move start with overlap
            start = end - overlap if end < len(tokens) else end

        return chunks
```

### Semantic Chunker (Sentence-Boundary Aware)

```python
import re
from typing import Iterator


class SemanticChunker(ChunkerProtocol):
    """
    Splits on sentence boundaries while respecting token limits.
    Best for: Prose documents, policies, manuals.
    """

    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    PARAGRAPH_BREAK = re.compile(r'\n\s*\n')

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.SEMANTIC

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> ChunkingResult:
        import time
        start_time = time.perf_counter()

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        char_offset = 0

        # Split into sentences
        sentences = self._split_sentences(document.content)

        for sentence in sentences:
            sentence_tokens = self.tokenizer.count_tokens(sentence)

            # If single sentence exceeds max, split it further
            if sentence_tokens > config.max_chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(self._create_chunk(
                        content=" ".join(current_chunk),
                        document_id=document.id,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                        token_count=current_tokens,
                    ))
                    chunk_index += 1
                    char_offset += len(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large sentence by tokens
                sub_chunks = self._split_large_sentence(sentence, config)
                for sub in sub_chunks:
                    chunks.append(self._create_chunk(
                        content=sub,
                        document_id=document.id,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                        token_count=self.tokenizer.count_tokens(sub),
                    ))
                    chunk_index += 1
                    char_offset += len(sub)
                continue

            # Check if adding sentence exceeds target
            if current_tokens + sentence_tokens > config.target_chunk_size:
                # Only flush if we have content and meet minimum
                if current_chunk and current_tokens >= config.min_chunk_size:
                    chunks.append(self._create_chunk(
                        content=" ".join(current_chunk),
                        document_id=document.id,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                        token_count=current_tokens,
                    ))
                    chunk_index += 1
                    char_offset += len(" ".join(current_chunk)) + 1

                    # Handle overlap - keep last N tokens worth of sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, config.chunk_overlap
                    )
                    current_chunk = overlap_sentences
                    current_tokens = sum(
                        self.tokenizer.count_tokens(s) for s in overlap_sentences
                    )
                else:
                    # Below minimum, keep accumulating
                    pass

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Flush remaining
        if current_chunk:
            chunks.append(self._create_chunk(
                content=" ".join(current_chunk),
                document_id=document.id,
                chunk_index=chunk_index,
                start_char=char_offset,
                token_count=current_tokens,
            ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ChunkingResult(
            document_id=document.id,
            chunks=chunks,
            total_chunks=len(chunks),
            strategy_used=self.strategy,
            avg_chunk_size=sum(c.metadata.token_count for c in chunks) / len(chunks) if chunks else 0,
            processing_time_ms=elapsed_ms,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # First split by paragraphs, then by sentences
        paragraphs = self.PARAGRAPH_BREAK.split(text)
        sentences = []

        for para in paragraphs:
            para_sentences = self.SENTENCE_ENDINGS.split(para.strip())
            sentences.extend([s.strip() for s in para_sentences if s.strip()])

        return sentences

    def _get_overlap_sentences(
        self, sentences: list[str], overlap_tokens: int
    ) -> list[str]:
        """Get sentences from end that fit within overlap tokens."""
        result = []
        tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.tokenizer.count_tokens(sentence)
            if tokens + sentence_tokens > overlap_tokens:
                break
            result.insert(0, sentence)
            tokens += sentence_tokens

        return result

    def _split_large_sentence(
        self, sentence: str, config: ChunkingConfig
    ) -> list[str]:
        """Split a sentence that exceeds max chunk size."""
        # Try splitting on clauses first
        clause_splits = re.split(r'[,;:]\s+', sentence)

        if len(clause_splits) > 1:
            # Recursively chunk clauses
            return self._merge_small_parts(clause_splits, config.target_chunk_size)

        # Fall back to token-based splitting
        return [
            chunk for chunk, _, _ in
            self.tokenizer.split_by_tokens(
                sentence,
                config.target_chunk_size,
                config.chunk_overlap,
            )
        ]

    def _merge_small_parts(
        self, parts: list[str], target_size: int
    ) -> list[str]:
        """Merge small parts into chunks approaching target size."""
        result = []
        current = []
        current_tokens = 0

        for part in parts:
            part_tokens = self.tokenizer.count_tokens(part)

            if current_tokens + part_tokens > target_size and current:
                result.append(", ".join(current))
                current = []
                current_tokens = 0

            current.append(part)
            current_tokens += part_tokens

        if current:
            result.append(", ".join(current))

        return result

    def _create_chunk(
        self,
        content: str,
        document_id: UUID,
        chunk_index: int,
        start_char: int,
        token_count: int,
    ) -> Chunk:
        return Chunk(
            content=content,
            metadata=ChunkMetadata(
                document_id=document_id,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(content),
                strategy_used=self.strategy,
                token_count=token_count,
            ),
        )
```

### Recursive Chunker (Structure-Aware)

```python
import re
from dataclasses import dataclass


@dataclass
class Section:
    """Represents a document section with hierarchy."""
    title: str
    level: int  # 1 = H1, 2 = H2, etc.
    content: str
    children: list["Section"]


class RecursiveChunker(ChunkerProtocol):
    """
    Splits by document structure (headers, sections) then recursively.
    Best for: Technical docs, manuals with clear hierarchy.
    """

    # Markdown header patterns
    MD_HEADERS = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    # Common document markers
    SECTION_MARKERS = [
        r'^(?:Chapter|Section|Part)\s+\d+',
        r'^\d+\.\d*\s+\w+',  # Numbered sections like "1.2 Overview"
    ]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.semantic_chunker = SemanticChunker(tokenizer)

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.RECURSIVE

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> ChunkingResult:
        import time
        start_time = time.perf_counter()

        # Parse document structure
        sections = self._parse_structure(document.content)

        # Recursively chunk sections
        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = await self._chunk_section(
                section=section,
                document_id=document.id,
                config=config,
                hierarchy=[],
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ChunkingResult(
            document_id=document.id,
            chunks=chunks,
            total_chunks=len(chunks),
            strategy_used=self.strategy,
            avg_chunk_size=sum(c.metadata.token_count for c in chunks) / len(chunks) if chunks else 0,
            processing_time_ms=elapsed_ms,
        )

    def _parse_structure(self, content: str) -> list[Section]:
        """Parse document into hierarchical sections."""
        sections = []
        current_section = None
        current_content = []

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]
            header_match = self.MD_HEADERS.match(line)

            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                current_section = Section(
                    title=title,
                    level=level,
                    content="",
                    children=[],
                )
                current_content = []
            else:
                current_content.append(line)

            i += 1

        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)
        elif current_content:
            # No headers found - treat whole doc as one section
            sections.append(Section(
                title="",
                level=0,
                content='\n'.join(current_content).strip(),
                children=[],
            ))

        return sections

    async def _chunk_section(
        self,
        section: Section,
        document_id: UUID,
        config: ChunkingConfig,
        hierarchy: list[str],
        start_index: int,
    ) -> list[Chunk]:
        """Recursively chunk a section and its children."""
        chunks = []
        current_hierarchy = hierarchy + [section.title] if section.title else hierarchy

        section_tokens = self.tokenizer.count_tokens(section.content)

        if section_tokens <= config.target_chunk_size:
            # Section fits in one chunk
            if section.content.strip():
                chunks.append(Chunk(
                    content=section.content,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        chunk_index=start_index + len(chunks),
                        start_char=0,  # Would need tracking for accurate offsets
                        end_char=len(section.content),
                        section_title=section.title or None,
                        section_hierarchy=current_hierarchy,
                        strategy_used=self.strategy,
                        token_count=section_tokens,
                    ),
                ))
        else:
            # Section too large - use semantic chunker on content
            temp_doc = Document(
                id=document_id,
                content=section.content,
                metadata=document.metadata,
                source=document.source,
                content_hash="",
            )

            sub_result = await self.semantic_chunker.chunk(temp_doc, config)

            for sub_chunk in sub_result.chunks:
                # Preserve hierarchy in sub-chunks
                sub_chunk.metadata.section_title = section.title
                sub_chunk.metadata.section_hierarchy = current_hierarchy
                sub_chunk.metadata.chunk_index = start_index + len(chunks)
                chunks.append(sub_chunk)

        # Process children recursively
        for child in section.children:
            child_chunks = await self._chunk_section(
                section=child,
                document_id=document_id,
                config=config,
                hierarchy=current_hierarchy,
                start_index=start_index + len(chunks),
            )
            chunks.extend(child_chunks)

        return chunks
```

### Fixed Chunker (Simple Token Windows)

```python
class FixedChunker(ChunkerProtocol):
    """
    Simple fixed-size token windows with overlap.
    Best for: Uniform content, when structure doesn't matter.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.FIXED

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> ChunkingResult:
        import time
        start_time = time.perf_counter()

        chunks = []
        raw_chunks = self.tokenizer.split_by_tokens(
            document.content,
            config.target_chunk_size,
            config.chunk_overlap,
        )

        for i, (content, start_char, end_char) in enumerate(raw_chunks):
            # Optionally adjust to sentence boundaries
            if config.respect_sentence_boundaries:
                content = self._adjust_boundaries(content)

            chunks.append(Chunk(
                content=content,
                metadata=ChunkMetadata(
                    document_id=document.id,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char,
                    strategy_used=self.strategy,
                    token_count=self.tokenizer.count_tokens(content),
                ),
            ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ChunkingResult(
            document_id=document.id,
            chunks=chunks,
            total_chunks=len(chunks),
            strategy_used=self.strategy,
            avg_chunk_size=sum(c.metadata.token_count for c in chunks) / len(chunks) if chunks else 0,
            processing_time_ms=elapsed_ms,
        )

    def _adjust_boundaries(self, content: str) -> str:
        """Adjust chunk to end at sentence boundary if possible."""
        # Find last sentence ending
        last_period = content.rfind('. ')
        last_question = content.rfind('? ')
        last_exclaim = content.rfind('! ')

        last_boundary = max(last_period, last_question, last_exclaim)

        if last_boundary > len(content) * 0.7:  # Only if not losing too much
            return content[:last_boundary + 1].strip()

        return content.strip()
```

### Adaptive Chunker (Auto-Selection)

```python
import re
from collections import Counter


class AdaptiveChunker(ChunkerProtocol):
    """
    Auto-detects document type and selects optimal chunking strategy.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.chunkers = {
            ChunkStrategy.SEMANTIC: SemanticChunker(tokenizer),
            ChunkStrategy.RECURSIVE: RecursiveChunker(tokenizer),
            ChunkStrategy.FIXED: FixedChunker(tokenizer),
            ChunkStrategy.CODE: CodeChunker(tokenizer),
        }

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.ADAPTIVE

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> ChunkingResult:
        # Detect optimal strategy
        detected_strategy = self._detect_strategy(document)

        # Use detected strategy
        chunker = self.chunkers[detected_strategy]
        result = await chunker.chunk(document, config)

        # Mark that adaptive routing was used
        result.strategy_used = ChunkStrategy.ADAPTIVE

        return result

    def _detect_strategy(self, document: Document) -> ChunkStrategy:
        """Analyze document content to select best strategy."""
        content = document.content
        file_type = document.metadata.file_type

        # Check for code files
        if file_type in ("py", "js", "ts", "go", "rs", "java", "cpp", "c"):
            return ChunkStrategy.CODE

        # Check for markdown/structured content
        header_count = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        if header_count >= 3:
            return ChunkStrategy.RECURSIVE

        # Check for numbered sections
        numbered_sections = len(re.findall(r'^\d+\.\d*\s+\w+', content, re.MULTILINE))
        if numbered_sections >= 3:
            return ChunkStrategy.RECURSIVE

        # Analyze sentence structure
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0

        # Long sentences suggest prose -> semantic chunking
        if avg_sentence_len > 100:
            return ChunkStrategy.SEMANTIC

        # Check paragraph density
        paragraphs = re.split(r'\n\s*\n', content)
        if len(paragraphs) > 10:
            return ChunkStrategy.SEMANTIC

        # Default to fixed for uniform/unclear content
        return ChunkStrategy.FIXED

    def _detect_code_language(self, content: str) -> str | None:
        """Attempt to detect programming language."""
        indicators = {
            "python": [r'def \w+\(', r'import \w+', r'class \w+:', r'if __name__'],
            "javascript": [r'function \w+\(', r'const \w+ =', r'let \w+ =', r'=>'],
            "typescript": [r'interface \w+', r': string', r': number', r'export '],
            "go": [r'func \w+\(', r'package \w+', r'import \('],
            "rust": [r'fn \w+\(', r'let mut', r'impl \w+', r'use \w+::'],
        }

        scores = Counter()
        for lang, patterns in indicators.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    scores[lang] += 1

        if scores:
            return scores.most_common(1)[0][0]
        return None
```

### Code Chunker (AST-Aware)

```python
import ast
import re


class CodeChunker(ChunkerProtocol):
    """
    AST-aware chunking for code files.
    Keeps functions/classes intact when possible.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def strategy(self) -> ChunkStrategy:
        return ChunkStrategy.CODE

    async def chunk(
        self,
        document: Document,
        config: ChunkingConfig,
    ) -> ChunkingResult:
        import time
        start_time = time.perf_counter()

        chunks = []
        content = document.content

        # Try Python AST parsing
        try:
            tree = ast.parse(content)
            chunks = self._chunk_python_ast(tree, content, document.id, config)
        except SyntaxError:
            # Not Python or invalid - fall back to regex-based splitting
            chunks = self._chunk_by_patterns(content, document.id, config)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ChunkingResult(
            document_id=document.id,
            chunks=chunks,
            total_chunks=len(chunks),
            strategy_used=self.strategy,
            avg_chunk_size=sum(c.metadata.token_count for c in chunks) / len(chunks) if chunks else 0,
            processing_time_ms=elapsed_ms,
        )

    def _chunk_python_ast(
        self,
        tree: ast.AST,
        content: str,
        document_id: UUID,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Extract functions and classes as chunks."""
        chunks = []
        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno

                node_content = '\n'.join(lines[start_line:end_line])
                node_tokens = self.tokenizer.count_tokens(node_content)

                # If too large, split by methods (for classes) or by logic blocks
                if node_tokens > config.max_chunk_size:
                    sub_chunks = self._split_large_node(
                        node, lines, document_id, config, len(chunks)
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(Chunk(
                        content=node_content,
                        metadata=ChunkMetadata(
                            document_id=document_id,
                            chunk_index=len(chunks),
                            start_char=sum(len(l) + 1 for l in lines[:start_line]),
                            end_char=sum(len(l) + 1 for l in lines[:end_line]),
                            section_title=f"{type(node).__name__}: {node.name}",
                            strategy_used=ChunkStrategy.CODE,
                            token_count=node_tokens,
                        ),
                    ))

        return chunks

    def _chunk_by_patterns(
        self,
        content: str,
        document_id: UUID,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Regex-based chunking for non-Python code."""
        # Common function/class patterns
        patterns = [
            r'^(?:async\s+)?function\s+\w+\s*\([^)]*\)\s*\{',  # JS functions
            r'^(?:export\s+)?(?:async\s+)?function\s+\w+',     # JS/TS
            r'^(?:pub\s+)?fn\s+\w+',                           # Rust
            r'^func\s+\w+\s*\(',                               # Go
            r'^class\s+\w+',                                    # Various
        ]

        combined_pattern = '|'.join(f'({p})' for p in patterns)
        matches = list(re.finditer(combined_pattern, content, re.MULTILINE))

        if not matches:
            # No structure found - use fixed chunking
            return FixedChunker(self.tokenizer).chunk(
                Document(id=document_id, content=content, metadata=None, source=None, content_hash=""),
                config,
            ).chunks

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            chunk_content = content[start:end].strip()
            chunks.append(Chunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    document_id=document_id,
                    chunk_index=i,
                    start_char=start,
                    end_char=end,
                    strategy_used=ChunkStrategy.CODE,
                    token_count=self.tokenizer.count_tokens(chunk_content),
                ),
            ))

        return chunks
```

## Chunking Strategy Selection Guide

| Document Type | Recommended Strategy | Rationale |
|---------------|---------------------|-----------|
| Policy/Legal | Semantic | Preserve sentence meaning, clause boundaries |
| Technical Manual | Recursive | Leverage header structure |
| API Docs | Recursive | Section hierarchy important |
| Code Files | Code | Keep functions/classes intact |
| Chat Logs | Fixed | Uniform turn structure |
| Research Papers | Recursive | Abstract, sections, references |
| Prose/Articles | Semantic | Paragraph and sentence flow |
| Mixed/Unknown | Adaptive | Auto-detect best approach |

## Dependencies

```toml
[project.dependencies]
tiktoken = "^0.7"          # OpenAI tokenizer
```

## Testing Strategy

1. **Unit Tests**: Each chunker with fixture documents
2. **Property Tests**: Chunk size distribution within bounds
3. **Regression Tests**: Same doc produces same chunks
4. **Edge Cases**:
   - Empty documents
   - Single-sentence documents
   - Documents with only headers
   - Code with syntax errors
   - Mixed content (prose + code)
5. **Performance**: Chunking throughput benchmarks
