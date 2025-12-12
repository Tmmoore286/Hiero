import io
import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.ingestion.router import detect_file_type
from hiero.ingestion import DocumentMetadata, IngestionRouter, TextIngestor, UnsupportedFormatError


def test_detect_file_type_by_extension():
    assert detect_file_type("doc.pdf") == "pdf"
    assert detect_file_type("notes.md") == "md"
    assert detect_file_type("notes.markdown") == "markdown"
    assert detect_file_type("notes.txt") == "txt"
    with pytest.raises(UnsupportedFormatError):
        detect_file_type("file.exe")


@pytest.mark.asyncio
async def test_text_ingestor_reads_content_and_hash():
    buf = io.BytesIO(b"hello world")
    meta = DocumentMetadata(title="Test")
    doc = await TextIngestor().ingest(buf, meta)
    assert doc.content == "hello world"
    assert doc.metadata.title == "Test"
    assert doc.metadata.file_size_bytes == 11
    assert len(doc.content_hash) == 64


@pytest.mark.asyncio
async def test_router_ingests_text_path(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("sample text", encoding="utf-8")
    router = IngestionRouter()
    doc = await router.ingest_path(path)
    assert "sample text" in doc.content

