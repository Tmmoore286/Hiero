from __future__ import annotations

from hashlib import sha256
from typing import BinaryIO

from .base import Document, DocumentMetadata, DocumentSource, IngestorProtocol


class TextIngestor(IngestorProtocol):
    async def ingest(
        self,
        source: BinaryIO,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        raw = source.read()
        content = raw.decode("utf-8", errors="ignore")
        content_hash = sha256(content.encode("utf-8")).hexdigest()

        extracted = DocumentMetadata(
            file_type=(metadata.file_type if metadata else "txt"),
            file_size_bytes=len(raw),
            **(metadata.model_dump(exclude_unset=True) if metadata else {}),
        )

        return Document(
            content=content,
            metadata=extracted,
            source=DocumentSource.FILE_UPLOAD,
            content_hash=content_hash,
        )

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in {"txt", "md", "markdown", "text"}

