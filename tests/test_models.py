import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Hiero requires Python >= 3.11", allow_module_level=True)

from hiero.storage.models import Base


def test_base_metadata_tables_present():
    table_names = set(Base.metadata.tables.keys())
    assert "documents" in table_names
    assert "chunks" in table_names
    assert "namespaces" in table_names
    assert "vector_search_stats" in table_names
    assert "embedding_cache" in table_names
