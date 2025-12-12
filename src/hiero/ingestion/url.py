from __future__ import annotations

from hashlib import sha256

import httpx
from trafilatura import extract

from .base import Document, DocumentMetadata, DocumentSource, IngestionError, IngestorProtocol


class URLIngestor(IngestorProtocol):
    def __init__(self, timeout: float = 30.0):
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def ingest(
        self,
        source: str,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        try:
            response = await self.client.get(source)
            response.raise_for_status()
        except Exception as exc:
            raise IngestionError(f"Failed to fetch URL: {exc}") from exc

        content = extract(
            response.text,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if not content:
            raise IngestionError(f"Could not extract content from {source}")

        content_hash = sha256(content.encode("utf-8")).hexdigest()

        extracted = DocumentMetadata(
            source_url=source,
            file_type="html",
            file_size_bytes=len(response.text.encode("utf-8")),
            **(metadata.model_dump(exclude_unset=True) if metadata else {}),
        )

        return Document(
            content=content,
            metadata=extracted,
            source=DocumentSource.URL,
            content_hash=content_hash,
        )

    def supports(self, file_type: str) -> bool:
        return file_type.lower() in {"url", "html"}

