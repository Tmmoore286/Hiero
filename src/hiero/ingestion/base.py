from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import BinaryIO
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentSource(str, Enum):
    FILE_UPLOAD = "file_upload"
    URL = "url"
    API = "api"
    ARCNET_SYNC = "arcnet_sync"


class DocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    source_url: str | None = None
    file_type: str | None = None
    file_size_bytes: int | None = None
    page_count: int | None = None
    language: str | None = "en"
    custom: dict = Field(default_factory=dict)


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: DocumentMetadata
    source: DocumentSource
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str

    @classmethod
    def from_content(
        cls,
        content: str,
        metadata: DocumentMetadata,
        source: DocumentSource,
        document_id: UUID | None = None,
    ) -> "Document":
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        return cls(
            id=document_id or uuid4(),
            content=content,
            metadata=metadata,
            source=source,
            content_hash=content_hash,
        )


class IngestionError(Exception):
    pass


class UnsupportedFormatError(IngestionError):
    pass


class IngestorProtocol(ABC):
    @abstractmethod
    async def ingest(
        self,
        source: BinaryIO,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        ...

    @abstractmethod
    def supports(self, file_type: str) -> bool:
        ...

