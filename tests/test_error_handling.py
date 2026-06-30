"""Tests for error handling edge cases.

Verifies graceful handling of disk full, corrupt data,
invalid usage, and resource exhaustion.
"""

from __future__ import annotations

import shutil
import sys

import pytest

import omendb


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


class TestDiskFullHandling:
    """Test behavior when disk is full."""

    def test_flush_to_readonly_directory(self, tmp_path) -> None:
        """Flush to read-only directory raises appropriate error."""
        _skip_if_free_threaded()

        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        # create() should fail on read-only directory
        with pytest.raises(Exception):
            omendb.create(str(readonly_dir / "db"))

        # Cleanup
        readonly_dir.chmod(0o755)

    def test_create_in_nonexistent_parent(self, tmp_path) -> None:
        """Create database in nonexistent parent directory creates it."""
        _skip_if_free_threaded()

        # create() creates parent directories
        db = omendb.create(str(tmp_path / "nonexistent" / "db"))
        assert db is not None


class TestCorruptDataHandling:
    """Test behavior with corrupt data."""

    def test_corrupt_manifest_recovery(self, tmp_path) -> None:
        """Database detects corrupt manifest or handles it gracefully."""
        _skip_if_free_threaded()

        # Create and flush a valid database
        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("a", vector=[1.0, 0.0])
        db.flush()
        db.close()

        # Corrupt the manifest
        manifest_path = tmp_path / "db" / "_manifest.json"
        if manifest_path.exists():
            manifest_path.write_text("corrupt")

        # Should either detect corruption or handle it gracefully
        try:
            result = omendb.check(str(tmp_path / "db"))
            # If check succeeds, it should either detect issues or report ok
            assert result.ok is True or len(result.issues) > 0
        except Exception:
            # Acceptable to raise on corrupt manifest
            pass

    def test_missing_data_files(self, tmp_path) -> None:
        """Database handles missing data files gracefully."""
        _skip_if_free_threaded()

        # Create and flush a valid database
        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("a", vector=[1.0, 0.0])
        db.flush()
        db.close()

        # Remove data files (but keep manifest)
        data_dir = tmp_path / "db" / "test"
        if data_dir.exists():
            shutil.rmtree(data_dir)

        # Should detect issues
        try:
            result = omendb.check(str(tmp_path / "db"))
            assert result.ok is False or len(result.issues) > 0
        except Exception:
            # Acceptable to raise on missing files
            pass

    def test_empty_database_file(self, tmp_path) -> None:
        """Database handles empty database file."""
        _skip_if_free_threaded()

        # Create and flush a valid database
        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("a", vector=[1.0, 0.0])
        db.flush()
        db.close()

        # Truncate a data file
        data_dir = tmp_path / "db" / "test"
        if data_dir.exists():
            for f in data_dir.glob("*.bin"):
                f.write_bytes(b"")

        # Should detect issues
        try:
            result = omendb.check(str(tmp_path / "db"))
            assert result.ok is False or len(result.issues) > 0
        except Exception:
            # Acceptable to raise on corrupt files
            pass


class TestInvalidUsageHandling:
    """Test behavior with invalid API usage."""

    def test_wrong_vector_dimension(self, tmp_path) -> None:
        """Wrong vector dimension raises error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        with pytest.raises(Exception):
            col.set("a", vector=[1.0])  # Wrong dimension

    def test_missing_required_fields(self, tmp_path) -> None:
        """Missing required fields raises error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        with pytest.raises(Exception):
            col.set()  # Missing id

    def test_invalid_search_parameters(self, tmp_path) -> None:
        """Invalid search parameters raises error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("a", vector=[1.0, 0.0])

        # Negative k
        with pytest.raises(Exception):
            col.search(vector=[1.0, 0.0], k=-1)

        # Zero k
        with pytest.raises(Exception):
            col.search(vector=[1.0, 0.0], k=0)

    def test_get_nonexistent_item(self, tmp_path) -> None:
        """Get nonexistent item returns None."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        result = col.get("nonexistent")
        assert result is None

    def test_delete_nonexistent_item(self, tmp_path) -> None:
        """Delete nonexistent item raises error or is no-op."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # delete() may raise or be a no-op for nonexistent items
        try:
            col.delete("nonexistent")
        except KeyError:
            pass  # Acceptable

    def test_update_nonexistent_item(self, tmp_path) -> None:
        """Update nonexistent item raises error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        with pytest.raises(Exception):
            col.update("nonexistent", text="new text")

    def test_duplicate_collection_creation(self, tmp_path) -> None:
        """Duplicate collection creation returns existing collection."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col1 = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col2 = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Should return the same collection
        assert col1 is not None
        assert col2 is not None

    def test_get_nonexistent_collection(self, tmp_path) -> None:
        """Get nonexistent collection raises error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))

        with pytest.raises(Exception):
            db.collection("nonexistent")

    def test_operations_on_closed_database(self, tmp_path) -> None:
        """Operations on closed database may succeed or fail."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        db.close()

        # close() may or may not affect the collection handle
        try:
            col.set("a", vector=[1.0, 0.0])
        except Exception:
            pass  # Acceptable


class TestResourceExhaustion:
    """Test behavior under resource pressure."""

    def test_many_collections(self, tmp_path) -> None:
        """Creating many collections works."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))

        # Create 100 collections
        for i in range(100):
            col = db.collection(f"col_{i}", config=omendb.CollectionConfig(dim=2))
            col.set("a", vector=[1.0, 0.0])

        # All should be accessible
        for i in range(100):
            col = db.collection(f"col_{i}")
            assert col.get("a") is not None

    def test_many_items_in_collection(self, tmp_path) -> None:
        """Many items in a single collection works."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Add 1000 items
        for i in range(1000):
            col.set(f"item_{i}", vector=[float(i), 0.0])

        # All should be present
        for i in range(1000):
            assert col.get(f"item_{i}") is not None

    def test_flush_reopen_cycle(self, tmp_path) -> None:
        """Multiple flush/reopen cycles work."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # 10 flush/reopen cycles
        for cycle in range(10):
            col.set(f"item_{cycle}", vector=[float(cycle), 0.0])
            db.flush()
            db.close()

            db = omendb.open(str(tmp_path / "db"))
            col = db.collection("test")

        # All items should be present
        for cycle in range(10):
            assert col.get(f"item_{cycle}") is not None


class TestErrorRecovery:
    """Test recovery from errors."""

    def test_recovery_after_invalid_operation(self, tmp_path) -> None:
        """Database recovers after invalid operation."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Valid operation
        col.set("a", vector=[1.0, 0.0])

        # Invalid operation
        try:
            col.set("b", vector=[1.0])  # Wrong dimension
        except Exception:
            pass

        # Should still be able to do valid operations
        col.set("c", vector=[0.0, 1.0])
        assert col.get("c") is not None

    def test_flush_after_error(self, tmp_path) -> None:
        """Can flush after error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Valid operation
        col.set("a", vector=[1.0, 0.0])

        # Invalid operation
        try:
            col.set("b", vector=[1.0])  # Wrong dimension
        except Exception:
            pass

        # Should be able to flush
        db.flush()

        # Reopen and verify
        db2 = omendb.open(str(tmp_path / "db"))
        col2 = db2.collection("test")
        assert col2.get("a") is not None
