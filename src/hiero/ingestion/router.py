from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .base import Document, DocumentMetadata, IngestorProtocol, UnsupportedFormatError
from .pdf import PDFIngestor
from .text import TextIngestor


def detect_file_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "pdf":
        return "pdf"
    if ext in {"txt", "md", "markdown"}:
        return ext
    raise UnsupportedFormatError(f"Unsupported file extension: {ext or '<none>'}")


class IngestionRouter:
    def __init__(self, ingestors: list[IngestorProtocol] | None = None):
        self.ingestors = ingestors or [PDFIngestor(), TextIngestor()]

    async def ingest_file(
        self,
        source: BinaryIO,
        file_type: str,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        for ingestor in self.ingestors:
            if ingestor.supports(file_type):
                return await ingestor.ingest(source, metadata)
        raise UnsupportedFormatError(f"No ingestor for type: {file_type}")

    async def ingest_path(
        self,
        path: str | Path,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        file_type = detect_file_type(path)
        with open(path, "rb") as f:
            return await self.ingest_file(f, file_type=file_type, metadata=metadata)

