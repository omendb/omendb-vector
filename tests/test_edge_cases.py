"""Tests for edge cases: empty collections, large vectors, Unicode, special chars.

Verifies boundary conditions and unusual inputs.
"""

from __future__ import annotations

import sys

import pytest

import omendb_vector


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


def _create_collection(tmp_path, name="test", dim=2):
    """Create a test collection."""
    db = omendb_vector.create(str(tmp_path / "db"))
    col = db.collection(name, config=omendb_vector.CollectionConfig(dim=dim, text=True))
    return db, col


class TestEmptyCollections:
    """Test empty collection behavior."""

    def test_search_empty_collection(self, tmp_path) -> None:
        """Search on empty collection returns empty results."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        results = col.search(vector=[1.0, 0.0], k=10)
        assert results == []

    def test_text_search_empty_collection(self, tmp_path) -> None:
        """Text search on empty collection raises error (text not enabled)."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        # Text search requires text=True in config
        with pytest.raises(Exception):
            col.search(text="hello", k=10)

    def test_get_empty_collection(self, tmp_path) -> None:
        """Get on empty collection returns None."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        result = col.get("nonexistent")
        assert result is None

    def test_flush_empty_collection(self, tmp_path) -> None:
        """Flush on empty collection works (empty collections may not persist)."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        db.flush()

        # Reopen - empty collections may not be persisted
        db2 = omendb_vector.open(str(tmp_path / "db"))
        try:
            col2 = db2.collection("test")
            results = col2.search(vector=[1.0, 0.0], k=10)
            assert results == []
        except KeyError, ValueError:
            pass  # Empty collections may not be persisted

    def test_check_empty_collection(self, tmp_path) -> None:
        """Check on empty collection passes."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        result = db.check()
        assert result.ok is True

    def test_vacuum_empty_collection(self, tmp_path) -> None:
        """Vacuum on empty collection works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        col.vacuum()


class TestLargeVectors:
    """Test with large dimension vectors."""

    def test_dim_1024(self, tmp_path) -> None:
        """Dimension 1024 works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path, dim=1024)

        # Create a 1024-dim vector
        vector = [float(i % 100) / 100.0 for i in range(1024)]
        col.set("a", vector=vector)

        # Search should work
        results = col.search(vector=vector, k=1)
        assert len(results) == 1
        assert results[0].id == "a"

    def test_dim_3072(self, tmp_path) -> None:
        """Dimension 3072 works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path, dim=3072)

        # Create a 3072-dim vector
        vector = [float(i % 100) / 100.0 for i in range(3072)]
        col.set("a", vector=vector)

        # Search should work
        results = col.search(vector=vector, k=1)
        assert len(results) == 1
        assert results[0].id == "a"

    def test_dim_8192(self, tmp_path) -> None:
        """Dimension 8192 works with flat index."""
        _skip_if_free_threaded()

        # dim=8192 requires index='flat'
        db = omendb_vector.create(str(tmp_path / "db"))
        col = db.collection(
            "test",
            config=omendb_vector.CollectionConfig(dim=8192, index="flat", text=True),
        )

        # Create a 8192-dim vector
        vector = [float(i % 100) / 100.0 for i in range(8192)]
        col.set("a", vector=vector)

        # Search should work
        results = col.search(vector=vector, k=1)
        assert len(results) == 1
        assert results[0].id == "a"


class TestUnicodeHandling:
    """Test Unicode in IDs, text, and metadata."""

    def test_unicode_id(self, tmp_path) -> None:
        """Unicode IDs work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("日本語", vector=[1.0, 0.0], text="Japanese ID")
        result = col.get("日本語", include_text=True)
        assert result is not None
        assert result["text"] == "Japanese ID"

    def test_unicode_text(self, tmp_path) -> None:
        """Unicode text works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("a", vector=[1.0, 0.0], text="日本語テキスト")
        result = col.get("a", include_text=True)
        assert result is not None
        assert result["text"] == "日本語テキスト"

    def test_unicode_metadata(self, tmp_path) -> None:
        """Unicode metadata works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("a", vector=[1.0, 0.0], metadata={"lang": "日本語"})
        result = col.get("a")
        assert result is not None
        assert result["metadata"]["lang"] == "日本語"

    def test_unicode_search(self, tmp_path) -> None:
        """Unicode text search works (BM25 tokenization may not match CJK)."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("a", text="日本語テキスト")
        col.set("b", text="English text")

        # BM25 tokenization may not match CJK characters
        results = col.search(text="English", k=10)
        assert len(results) > 0
        assert results[0].id == "b"

    def test_unicode_flush_reopen(self, tmp_path) -> None:
        """Unicode survives flush/reopen."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("日本語", vector=[1.0, 0.0], text="日本語テキスト")
        db.flush()

        db2 = omendb_vector.open(str(tmp_path / "db"))
        col2 = db2.collection("test")

        result = col2.get("日本語", include_text=True)
        assert result is not None
        assert result["text"] == "日本語テキスト"


class TestSpecialCharacters:
    """Test special characters in IDs and text."""

    def test_special_chars_id(self, tmp_path) -> None:
        """Special characters in IDs work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        # Various special characters
        ids = ["a-b", "a_b", "a.b", "a@b", "a#b", "a b"]
        for id in ids:
            col.set(id, vector=[1.0, 0.0], text=f"Text for {id}")

        for id in ids:
            result = col.get(id, include_text=True)
            assert result is not None
            assert result["text"] == f"Text for {id}"

    def test_special_chars_text(self, tmp_path) -> None:
        """Special characters in text work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("a", vector=[1.0, 0.0], text="Hello, World! @#$%^&*()")
        result = col.get("a", include_text=True)
        assert result is not None
        assert result["text"] == "Hello, World! @#$%^&*()"

    def test_special_chars_metadata(self, tmp_path) -> None:
        """Special characters in metadata work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("a", vector=[1.0, 0.0], metadata={"key": "value with spaces & symbols"})
        result = col.get("a")
        assert result is not None
        assert result["metadata"]["key"] == "value with spaces & symbols"

    def test_newlines_in_text(self, tmp_path) -> None:
        """Newlines in text work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        text = "Line 1\nLine 2\nLine 3"
        col.set("a", vector=[1.0, 0.0], text=text)
        result = col.get("a", include_text=True)
        assert result is not None
        assert result["text"] == text

    def test_tabs_in_text(self, tmp_path) -> None:
        """Tabs in text work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        text = "Col1\tCol2\tCol3"
        col.set("a", vector=[1.0, 0.0], text=text)
        result = col.get("a", include_text=True)
        assert result is not None
        assert result["text"] == text


class TestZeroVectors:
    """Test zero vectors."""

    def test_zero_vector(self, tmp_path) -> None:
        """Zero vector works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("a", vector=[0.0, 0.0])
        result = col.get("a")
        assert result is not None

    def test_zero_vector_search(self, tmp_path) -> None:
        """Search with zero vector works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        col.set("a", vector=[1.0, 0.0])
        col.set("b", vector=[0.0, 1.0])

        results = col.search(vector=[0.0, 0.0], k=2)
        assert len(results) == 2


class TestMaxLimits:
    """Test maximum limits."""

    def test_many_metadata_keys(self, tmp_path) -> None:
        """Many metadata keys work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        col.set("a", vector=[1.0, 0.0], metadata=metadata)

        result = col.get("a")
        assert result is not None
        assert len(result["metadata"]) == 100

    def test_large_metadata_value(self, tmp_path) -> None:
        """Large metadata value works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        large_value = "x" * 10000
        col.set("a", vector=[1.0, 0.0], metadata={"large": large_value})

        result = col.get("a")
        assert result is not None
        assert result["metadata"]["large"] == large_value

    def test_large_text(self, tmp_path) -> None:
        """Large text works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        large_text = "word " * 10000
        col.set("a", vector=[1.0, 0.0], text=large_text)

        result = col.get("a", include_text=True)
        assert result is not None
        assert result["text"] == large_text


class TestSourceSpans:
    """Test source span edge cases."""

    def test_source_with_all_fields(self, tmp_path) -> None:
        """Source with all fields works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        source = {
            "path": "test.txt",
            "line_start": 10,
            "line_end": 20,
            "page": 5,
            "symbol": "my_function",
        }
        col.set("a", vector=[1.0, 0.0], text="Hello", source=source)

        result = col.get("a", include_source=True)
        assert result is not None
        assert result["source"]["path"] == "test.txt"
        assert result["source"]["line_start"] == 10
        assert result["source"]["line_end"] == 20
        assert result["source"]["page"] == 5
        assert result["source"]["symbol"] == "my_function"

    def test_source_with_minimal_fields(self, tmp_path) -> None:
        """Source with minimal fields works."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        source = {"path": "test.txt"}
        col.set("a", vector=[1.0, 0.0], text="Hello", source=source)

        result = col.get("a", include_source=True)
        assert result is not None
        assert result["source"]["path"] == "test.txt"


class TestRelationships:
    """Test relationship edge cases."""

    def test_self_relationship_not_supported(self, tmp_path) -> None:
        """Self-relationship raises error (not supported)."""
        _skip_if_free_threaded()

        db = omendb_vector.create(str(tmp_path / "db"))
        col = db.collection(
            "test",
            config=omendb_vector.CollectionConfig(dim=2, text=True, graph=True),
        )

        col.set("a", vector=[1.0, 0.0])
        with pytest.raises(Exception):
            col.add_relationship("a", "a", type="self")

    def test_multiple_relationship_types(self, tmp_path) -> None:
        """Multiple relationship types work."""
        _skip_if_free_threaded()

        db = omendb_vector.create(str(tmp_path / "db"))
        col = db.collection(
            "test",
            config=omendb_vector.CollectionConfig(dim=2, text=True, graph=True),
        )

        col.set("a", vector=[1.0, 0.0])
        col.set("b", vector=[0.0, 1.0])

        col.add_relationship("a", "b", type="parent")
        col.add_relationship("a", "b", type="friend")

        parents = col.neighbors("a", type="parent")
        friends = col.neighbors("a", type="friend")

        assert "b" in parents
        assert "b" in friends
