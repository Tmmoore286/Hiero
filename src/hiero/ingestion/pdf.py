from __future__ import annotations

from hashlib import sha256
from typing import BinaryIO

import fitz  # pymupdf

from .base import Document, DocumentMetadata, DocumentSource, IngestionError, IngestorProtocol


class PDFIngestor(IngestorProtocol):
    async def ingest(
        self,
        source: BinaryIO,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        try:
            pdf = fitz.open(stream=source.read(), filetype="pdf")
        except Exception as exc:
            raise IngestionError(f"Failed to open PDF: {exc}") from exc

        try:
            text_parts: list[str] = []
            for page in pdf:
                text_parts.append(page.get_text("text"))

            content = "\n\n".join(text_parts)
            content_hash = sha256(content.encode("utf-8")).hexdigest()

            pdf_meta = pdf.metadata or {}
            extracted = DocumentMetadata(
                title=pdf_meta.get("title") or (metadata.title if metadata else None),
                author=pdf_meta.get("author") or (metadata.author if metadata else None),
                page_count=len(pdf),
                file_type="pdf",
                **(metadata.model_dump(exclude_unset=True) if metadata else {}),
            )

            return Document(
                content=content,
                metadata=extracted,
                source=DocumentSource.FILE_UPLOAD,
                content_hash=content_hash,
            )
        finally:
            pdf.close()

    def supports(self, file_type: str) -> bool:
        return file_type.lower() == "pdf"

