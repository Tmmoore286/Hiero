import io
import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.ingestion.router import detect_file_type
from hiero.ingestion import (
    DOCXIngestor,
    DocumentMetadata,
    IngestionRouter,
    TextIngestor,
    URLIngestor,
    UnsupportedFormatError,
)


def test_detect_file_type_by_extension():
    assert detect_file_type("doc.pdf") == "pdf"
    assert detect_file_type("doc.docx") == "docx"
    assert detect_file_type("notes.md") == "md"
    assert detect_file_type("notes.markdown") == "markdown"
    assert detect_file_type("notes.txt") == "txt"
    assert detect_file_type("https://example.com") == "url"
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


@pytest.mark.asyncio
async def test_docx_ingestor_reads_paragraphs():
    from docx import Document as DocxDocument

    buf = io.BytesIO()
    d = DocxDocument()
    d.add_paragraph("Hello DOCX")
    d.save(buf)
    buf.seek(0)

    doc = await DOCXIngestor().ingest(buf, DocumentMetadata(title="T"))
    assert "Hello DOCX" in doc.content
    assert doc.metadata.file_type == "docx"


@pytest.mark.asyncio
async def test_url_ingestor_uses_extractor(monkeypatch):
    ingestor = URLIngestor()

    class _Resp:
        text = "<html><body>ignored</body></html>"

        def raise_for_status(self):
            return None

    async def fake_get(url):
        return _Resp()

    monkeypatch.setattr(ingestor.client, "get", fake_get)
    monkeypatch.setattr("hiero.ingestion.url.extract", lambda *a, **k: "URL content")

    doc = await ingestor.ingest("https://example.com")
    assert doc.content == "URL content"
    assert doc.source.value == "url"
