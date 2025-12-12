# Component Spec: Ingestion

## Overview

The Ingestion component handles document intake from multiple sources, with hybrid sync/async processing based on document size. It normalizes all inputs into a common `Document` schema for downstream chunking.

## Requirements

### Functional
- **FR-1**: Support file upload (PDF, DOCX, MD, TXT, HTML)
- **FR-2**: Support URL ingestion (fetch and extract content)
- **FR-3**: Sync processing for small documents (< 1MB)
- **FR-4**: Async queue for large documents (≥ 1MB)
- **FR-5**: Extract and preserve metadata (title, author, date, source URL)
- **FR-6**: Idempotent ingestion (re-ingest same doc updates, doesn't duplicate)
- **FR-7**: Progress tracking for async jobs

### Non-Functional
- **NFR-1**: Handle documents up to 100MB
- **NFR-2**: Process small docs in < 2 seconds
- **NFR-3**: Queue large docs in < 500ms (actual processing async)
- **NFR-4**: Support concurrent ingestion (10+ simultaneous uploads)

## Data Models

```python
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class DocumentSource(str, Enum):
    FILE_UPLOAD = "file_upload"
    URL = "url"
    API = "api"
    ARCNET_SYNC = "arcnet_sync"  # Synced from ARCnet doctrine repo


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Extracted and user-provided metadata."""
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    source_url: str | None = None
    file_type: str | None = None
    file_size_bytes: int | None = None
    page_count: int | None = None
    language: str | None = "en"
    custom: dict = Field(default_factory=dict)  # User-defined metadata


class Document(BaseModel):
    """Normalized document after ingestion."""
    id: UUID = Field(default_factory=uuid4)
    content: str  # Full extracted text
    metadata: DocumentMetadata
    source: DocumentSource
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str  # SHA-256 for deduplication


class IngestionJob(BaseModel):
    """Tracks async ingestion progress."""
    job_id: UUID = Field(default_factory=uuid4)
    document_id: UUID | None = None  # Set after processing
    status: IngestionStatus = IngestionStatus.PENDING
    progress_pct: float = 0.0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
```

## Interfaces

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, BinaryIO


class IngestorProtocol(ABC):
    """Base protocol for all ingestors."""

    @abstractmethod
    async def ingest(
        self,
        source: BinaryIO | str,  # File handle or URL
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        """
        Ingest a document and return normalized Document.

        Args:
            source: File handle (BinaryIO) or URL string
            metadata: Optional user-provided metadata to merge

        Returns:
            Normalized Document with extracted content

        Raises:
            IngestionError: If extraction fails
            UnsupportedFormatError: If file type not supported
        """
        ...

    @abstractmethod
    def supports(self, file_type: str) -> bool:
        """Check if this ingestor supports the given file type."""
        ...


class IngestionQueue(ABC):
    """Async job queue for large documents."""

    @abstractmethod
    async def enqueue(
        self,
        source: BinaryIO | str,
        metadata: DocumentMetadata | None = None,
    ) -> IngestionJob:
        """Queue a document for async processing."""
        ...

    @abstractmethod
    async def get_status(self, job_id: UUID) -> IngestionJob:
        """Get current status of an ingestion job."""
        ...

    @abstractmethod
    async def poll_completed(self) -> AsyncIterator[Document]:
        """Yield completed documents from the queue."""
        ...
```

## Implementation Details

### File Type Detection

```python
import magic  # python-magic library
from pathlib import Path

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/markdown": "md",
    "text/plain": "txt",
    "text/html": "html",
}

def detect_file_type(file: BinaryIO) -> str:
    """Detect file type using libmagic."""
    mime = magic.from_buffer(file.read(2048), mime=True)
    file.seek(0)  # Reset for subsequent reads

    if mime not in SUPPORTED_TYPES:
        raise UnsupportedFormatError(f"Unsupported MIME type: {mime}")

    return SUPPORTED_TYPES[mime]
```

### Extraction Strategies

| File Type | Library | Notes |
|-----------|---------|-------|
| PDF | `pymupdf` (fitz) | Fast, handles scanned PDFs with OCR |
| DOCX | `python-docx` | Preserves structure, extracts tables |
| MD | `markdown-it-py` | Parse to AST, extract text |
| TXT | Built-in | Direct read with encoding detection |
| HTML | `beautifulsoup4` + `trafilatura` | Article extraction, removes boilerplate |

### PDF Extraction

```python
import fitz  # pymupdf
from hashlib import sha256


class PDFIngestor(IngestorProtocol):
    async def ingest(
        self,
        source: BinaryIO,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        doc = fitz.open(stream=source.read(), filetype="pdf")

        # Extract text from all pages
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))

        content = "\n\n".join(text_parts)
        content_hash = sha256(content.encode()).hexdigest()

        # Extract PDF metadata
        pdf_meta = doc.metadata

        extracted_metadata = DocumentMetadata(
            title=pdf_meta.get("title") or metadata.title if metadata else None,
            author=pdf_meta.get("author") or metadata.author if metadata else None,
            page_count=len(doc),
            file_type="pdf",
            **(metadata.model_dump(exclude_unset=True) if metadata else {}),
        )

        doc.close()

        return Document(
            content=content,
            metadata=extracted_metadata,
            source=DocumentSource.FILE_UPLOAD,
            content_hash=content_hash,
        )

    def supports(self, file_type: str) -> bool:
        return file_type == "pdf"
```

### URL Ingestion

```python
import httpx
from trafilatura import extract


class URLIngestor(IngestorProtocol):
    def __init__(self, timeout: float = 30.0):
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def ingest(
        self,
        source: str,  # URL
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        response = await self.client.get(source)
        response.raise_for_status()

        # Extract main content, removing boilerplate
        content = extract(
            response.text,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )

        if not content:
            raise IngestionError(f"Could not extract content from {source}")

        content_hash = sha256(content.encode()).hexdigest()

        return Document(
            content=content,
            metadata=DocumentMetadata(
                source_url=source,
                file_type="html",
                **(metadata.model_dump(exclude_unset=True) if metadata else {}),
            ),
            source=DocumentSource.URL,
            content_hash=content_hash,
        )

    def supports(self, file_type: str) -> bool:
        return file_type in ("html", "url")
```

### Hybrid Sync/Async Router

```python
from typing import BinaryIO

SIZE_THRESHOLD = 1_000_000  # 1MB


class IngestionRouter:
    """Routes documents to sync or async processing based on size."""

    def __init__(
        self,
        ingestors: dict[str, IngestorProtocol],
        queue: IngestionQueue,
    ):
        self.ingestors = ingestors
        self.queue = queue

    async def ingest(
        self,
        source: BinaryIO | str,
        metadata: DocumentMetadata | None = None,
    ) -> Document | IngestionJob:
        """
        Route to sync or async based on size.

        Returns:
            Document if processed sync, IngestionJob if queued async
        """
        if isinstance(source, str):
            # URL - always sync (fetch is already async)
            return await self.ingestors["url"].ingest(source, metadata)

        # File - check size
        source.seek(0, 2)  # Seek to end
        size = source.tell()
        source.seek(0)  # Reset

        if size >= SIZE_THRESHOLD:
            # Large file - queue for async processing
            return await self.queue.enqueue(source, metadata)

        # Small file - process sync
        file_type = detect_file_type(source)
        ingestor = self.ingestors.get(file_type)

        if not ingestor:
            raise UnsupportedFormatError(f"No ingestor for type: {file_type}")

        return await ingestor.ingest(source, metadata)
```

### Async Queue Implementation

For serverless, use PostgreSQL as the queue (simple, no extra infra):

```python
from sqlalchemy import Column, String, DateTime, Float, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID, BYTEA
from sqlalchemy.ext.asyncio import AsyncSession


class IngestionJobModel(Base):
    __tablename__ = "ingestion_jobs"

    job_id = Column(PGUUID, primary_key=True)
    status = Column(SQLEnum(IngestionStatus), default=IngestionStatus.PENDING)
    progress_pct = Column(Float, default=0.0)
    error_message = Column(String, nullable=True)
    file_data = Column(BYTEA)  # Store file bytes for processing
    metadata_json = Column(String)  # Serialized DocumentMetadata
    document_id = Column(PGUUID, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class PostgresIngestionQueue(IngestionQueue):
    """Uses PostgreSQL for job queue - no additional infrastructure."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def enqueue(
        self,
        source: BinaryIO,
        metadata: DocumentMetadata | None = None,
    ) -> IngestionJob:
        job = IngestionJob()

        async with self.session_factory() as session:
            job_model = IngestionJobModel(
                job_id=job.job_id,
                file_data=source.read(),
                metadata_json=metadata.model_dump_json() if metadata else None,
            )
            session.add(job_model)
            await session.commit()

        return job

    async def process_pending(self):
        """Worker function to process pending jobs. Run as background task."""
        async with self.session_factory() as session:
            # Fetch oldest pending job (SELECT FOR UPDATE SKIP LOCKED)
            result = await session.execute(
                select(IngestionJobModel)
                .where(IngestionJobModel.status == IngestionStatus.PENDING)
                .order_by(IngestionJobModel.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            job_model = result.scalar_one_or_none()

            if not job_model:
                return None

            # Update status to processing
            job_model.status = IngestionStatus.PROCESSING
            await session.commit()

            try:
                # Process the document
                file_type = detect_file_type(BytesIO(job_model.file_data))
                metadata = (
                    DocumentMetadata.model_validate_json(job_model.metadata_json)
                    if job_model.metadata_json else None
                )

                document = await self.ingestors[file_type].ingest(
                    BytesIO(job_model.file_data),
                    metadata,
                )

                # Mark completed
                job_model.status = IngestionStatus.COMPLETED
                job_model.document_id = document.id
                job_model.completed_at = datetime.utcnow()
                job_model.progress_pct = 100.0

            except Exception as e:
                job_model.status = IngestionStatus.FAILED
                job_model.error_message = str(e)

            await session.commit()
```

## Error Handling

```python
class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class UnsupportedFormatError(IngestionError):
    """File format not supported."""
    pass


class ExtractionError(IngestionError):
    """Failed to extract content from document."""
    pass


class ContentTooLargeError(IngestionError):
    """Document exceeds size limit."""
    pass
```

## Dependencies

```toml
[project.dependencies]
pymupdf = "^1.24"          # PDF extraction
python-docx = "^1.1"       # DOCX extraction
python-magic = "^0.4"      # File type detection
trafilatura = "^1.8"       # Web content extraction
beautifulsoup4 = "^4.12"   # HTML parsing
httpx = "^0.27"            # Async HTTP client
chardet = "^5.2"           # Encoding detection
```

## Testing Strategy

1. **Unit Tests**: Each ingestor with fixture files
2. **Integration Tests**: Full pipeline from file upload to Document
3. **Edge Cases**:
   - Corrupted files
   - Password-protected PDFs
   - Empty documents
   - Non-UTF8 encodings
   - Very large files (100MB)
4. **Performance Tests**: Concurrent ingestion throughput

## ARCnet Integration

For doctrine documents synced from ARCnet:

```python
class ARCnetDoctrineIngestor(IngestorProtocol):
    """Ingest doctrine documents from ARCnet's doctrine repository."""

    async def ingest_doctrine(
        self,
        doctrine_id: str,
        mos_code: str,
        content: str,
        metadata: dict,
    ) -> Document:
        """
        Special ingestion path for ARCnet doctrine sync.

        Args:
            doctrine_id: ARCnet's doctrine document ID
            mos_code: Military Occupational Specialty code
            content: Pre-extracted doctrine text
            metadata: ARCnet-provided metadata
        """
        return Document(
            content=content,
            metadata=DocumentMetadata(
                title=metadata.get("title"),
                custom={
                    "mos_code": mos_code,
                    "doctrine_id": doctrine_id,
                    **metadata,
                },
            ),
            source=DocumentSource.ARCNET_SYNC,
            content_hash=sha256(content.encode()).hexdigest(),
        )
```
