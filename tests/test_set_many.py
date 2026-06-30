"""Tests for set_many() operation.

Verifies bulk insert correctness, partial failure handling, and transaction semantics.
"""

from __future__ import annotations

import sys

import pytest

import omendb


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


def _create_collection(tmp_path, name="test"):
    """Create a test collection."""
    db = omendb.create(str(tmp_path / "db"))
    col = db.collection(name, config=omendb.CollectionConfig(dim=2, text=True))
    return col


class TestSetManyBasic:
    """Test basic set_many() functionality."""

    def test_set_many_dense_vectors(self, tmp_path) -> None:
        """set_many with dense vectors."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": "a", "vector": [1.0, 0.0]},
            {"id": "b", "vector": [0.0, 1.0]},
            {"id": "c", "vector": [0.5, 0.5]},
        ]
        col.set_many(records)

        # All should be searchable
        results = col.search(vector=[1.0, 0.0], k=3)
        assert len(results) == 3
        assert results[0].id == "a"

    def test_set_many_text_only(self, tmp_path) -> None:
        """set_many with text-only records."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": "a", "text": "Hello world"},
            {"id": "b", "text": "Goodbye world"},
            {"id": "c", "text": "Hello again"},
        ]
        col.set_many(records)

        # All should be searchable by text
        results = col.search(text="Hello", k=2)
        assert len(results) == 2

    def test_set_many_mixed(self, tmp_path) -> None:
        """set_many with mixed vector and text records."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": "a", "vector": [1.0, 0.0], "text": "Hello"},
            {"id": "b", "vector": [0.0, 1.0], "text": "World"},
            {"id": "c", "text": "Text only"},
        ]
        col.set_many(records)

        # All should be present
        assert col.get("a") is not None
        assert col.get("b") is not None
        assert col.get("c") is not None

    def test_set_many_with_metadata(self, tmp_path) -> None:
        """set_many with metadata."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": "a", "vector": [1.0, 0.0], "metadata": {"category": "test"}},
            {"id": "b", "vector": [0.0, 1.0], "metadata": {"category": "other"}},
        ]
        col.set_many(records)

        # Metadata should be present
        doc = col.get("a", include_vector=False)
        assert doc["metadata"]["category"] == "test"

    def test_set_many_with_source(self, tmp_path) -> None:
        """set_many with source spans."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": "a", "text": "Hello", "source": {"path": "test.txt"}},
            {"id": "b", "text": "World", "source": {"path": "other.txt"}},
        ]
        col.set_many(records)

        # Source should be present
        doc = col.get("a", include_source=True)
        assert doc["source"]["path"] == "test.txt"

    def test_set_many_empty_list(self, tmp_path) -> None:
        """set_many with empty list is a no-op."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)
        col.set_many([])

        # No items should exist
        results = col.search(vector=[0.0, 0.0], k=10)
        assert len(results) == 0

    def test_set_many_single_record(self, tmp_path) -> None:
        """set_many with single record."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [{"id": "a", "vector": [1.0, 0.0]}]
        col.set_many(records)

        assert col.get("a") is not None


class TestSetManyUpdates:
    """Test set_many() with updates and upserts."""

    def test_set_many_upsert(self, tmp_path) -> None:
        """set_many can update existing records."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        # Initial insert
        col.set("a", vector=[1.0, 0.0], text="Original")

        # Update via set_many
        col.set_many([{"id": "a", "vector": [0.0, 1.0], "text": "Updated"}])

        # Should be updated
        doc = col.get("a", include_text=True)
        assert doc["text"] == "Updated"

    def test_set_many_partial_update(self, tmp_path) -> None:
        """set_many can update some records while inserting others."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        # Initial insert
        col.set("a", vector=[1.0, 0.0], text="Original")

        # Update a, insert b
        col.set_many(
            [
                {"id": "a", "text": "Updated"},
                {"id": "b", "vector": [0.0, 1.0], "text": "New"},
            ]
        )

        # Both should exist
        assert col.get("a") is not None
        assert col.get("b") is not None


class TestSetManyPersistence:
    """Test set_many() survives flush/reopen."""

    def test_set_many_survives_reopen(self, tmp_path) -> None:
        """set_many records persist across flush/reopen."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        records = [
            {"id": "a", "vector": [1.0, 0.0], "text": "Hello"},
            {"id": "b", "vector": [0.0, 1.0], "text": "World"},
        ]
        col.set_many(records)
        db.flush()

        # Reopen
        db2 = omendb.open(str(tmp_path / "db"))
        col2 = db2.collection("test")

        assert col2.get("a") is not None
        assert col2.get("b") is not None

        results = col2.search(vector=[1.0, 0.0], k=2)
        assert len(results) == 2


class TestSetManyAtScale:
    """Test set_many() with larger datasets."""

    def test_set_many_100_records(self, tmp_path) -> None:
        """set_many with 100 records."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": f"item_{i}", "vector": [float(i), 0.0], "text": f"Text {i}"}
            for i in range(100)
        ]
        col.set_many(records)

        # All should be present
        for i in range(100):
            assert col.get(f"item_{i}") is not None

    def test_set_many_1000_records(self, tmp_path) -> None:
        """set_many with 1000 records."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": f"item_{i}", "vector": [float(i % 100), float(i // 100)]}
            for i in range(1000)
        ]
        col.set_many(records)

        # All should be present
        for i in range(1000):
            assert col.get(f"item_{i}") is not None

    def test_set_many_search_quality(self, tmp_path) -> None:
        """set_many preserves search quality."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        # Create records with known ordering
        records = [
            {"id": "close", "vector": [1.0, 0.0]},
            {"id": "medium", "vector": [0.5, 0.5]},
            {"id": "far", "vector": [0.0, 1.0]},
        ]
        col.set_many(records)

        # Search should return in correct order
        results = col.search(vector=[1.0, 0.0], k=3)
        assert results[0].id == "close"
        assert results[1].id == "medium"
        assert results[2].id == "far"


class TestSetManyErrorHandling:
    """Test set_many() error handling."""

    def test_set_many_missing_id(self, tmp_path) -> None:
        """set_many raises error if id is missing."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        with pytest.raises(KeyError):
            col.set_many([{"vector": [1.0, 0.0]}])

    def test_set_many_invalid_vector(self, tmp_path) -> None:
        """set_many raises error if vector is invalid."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        with pytest.raises(Exception):
            col.set_many([{"id": "a", "vector": [1.0]}])  # Wrong dimension

    def test_set_many_duplicate_ids(self, tmp_path) -> None:
        """set_many with duplicate ids uses last value."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [
            {"id": "a", "vector": [1.0, 0.0], "text": "First"},
            {"id": "a", "vector": [0.0, 1.0], "text": "Second"},
        ]
        col.set_many(records)

        # Should use last value
        doc = col.get("a", include_text=True)
        assert doc["text"] == "Second"


class TestSetManyTransactionSemantics:
    """Test set_many() transaction semantics."""

    def test_set_many_atomic_on_success(self, tmp_path) -> None:
        """set_many commits all records on success."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        records = [{"id": f"item_{i}", "vector": [float(i), 0.0]} for i in range(10)]
        col.set_many(records)

        # All should be present
        for i in range(10):
            assert col.get(f"item_{i}") is not None

    def test_set_many_partial_failure(self, tmp_path) -> None:
        """set_many stops on first error (not atomic)."""
        _skip_if_free_threaded()

        col = _create_collection(tmp_path)

        # Valid, then invalid, then valid
        records = [
            {"id": "a", "vector": [1.0, 0.0]},
            {"id": "b", "vector": [1.0]},  # Invalid dimension
            {"id": "c", "vector": [0.0, 1.0]},
        ]

        with pytest.raises(Exception):
            col.set_many(records)

        # First record should be present (not atomic)
        assert col.get("a") is not None

    def test_set_many_flush_after_failure(self, tmp_path) -> None:
        """set_many can flush after partial failure."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        # Valid, then invalid
        records = [
            {"id": "a", "vector": [1.0, 0.0]},
            {"id": "b", "vector": [1.0]},  # Invalid dimension
        ]

        with pytest.raises(Exception):
            col.set_many(records)

        # Should be able to flush
        db.flush()

        # Reopen and check
        db2 = omendb.open(str(tmp_path / "db"))
        col2 = db2.collection("test")
        assert col2.get("a") is not None
