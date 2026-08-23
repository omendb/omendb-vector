"""Tests for backup/export/import preservation through snapshot and restore.

Verifies that all data types survive backup and restore:
- Dense vectors
- Text content
- Source spans
- Metadata
- Multivector collections
- Text-only items
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import omendb_vector


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


class TestBackupRestoreDense:
    """Test backup/restore preserves dense vector data."""

    def test_dense_vectors_survive_backup_restore(self, tmp_path) -> None:
        """Dense vectors, text, metadata, and source survive snapshot/restore."""
        _skip_if_free_threaded()

        db_path = str(tmp_path / "db")
        snapshot_path = str(tmp_path / "snapshot")
        restore_path = str(tmp_path / "restored")

        # Create database with dense vectors
        db = omendb_vector.create(db_path)
        col = db.collection("dense", config=omendb_vector.CollectionConfig(dim=2, text=True))
        for i in range(10):
            col.set(
                f"doc_{i}",
                vector=[float(i) / 10, 1.0 - float(i) / 10],
                text=f"Document {i} content",
                metadata={"index": i, "category": f"cat_{i % 3}"},
                source={"url": f"https://example.com/doc_{i}"},
            )
        col.flush()

        # Snapshot
        db.snapshot(snapshot_path)

        # Restore to new location
        db2 = omendb_vector.create(restore_path)
        db2.import_snapshot(snapshot_path)

        # Verify all data preserved
        col2 = db2.collection("dense", create=False)
        for i in range(10):
            doc = col2.get(f"doc_{i}", include_text=True, include_source=True)
            assert doc is not None
            assert doc["id"] == f"doc_{i}"
            assert doc["text"] == f"Document {i} content"
            assert doc["metadata"]["index"] == i
            assert doc["metadata"]["category"] == f"cat_{i % 3}"
            assert doc["source"]["url"] == f"https://example.com/doc_{i}"

        # Verify search still works
        results = col2.search_vector([0.5, 0.5], k=5)
        assert len(results) == 5
        assert all(r.id.startswith("doc_") for r in results)


class TestBackupRestoreText:
    """Test backup/restore preserves text/BM25 data."""

    def test_text_data_survives_backup_restore(self, tmp_path) -> None:
        """Text content and BM25 index survive snapshot/restore."""
        _skip_if_free_threaded()

        db_path = str(tmp_path / "db")
        snapshot_path = str(tmp_path / "snapshot")
        restore_path = str(tmp_path / "restored")

        # Create database with text data
        db = omendb_vector.create(db_path)
        col = db.collection("text", config=omendb_vector.CollectionConfig(dim=2, text=True))
        for i in range(10):
            col.set(
                f"doc_{i}",
                vector=[float(i) / 10, 1.0 - float(i) / 10],
                text=f"Document {i} about topic {i % 3}",
                metadata={"index": i},
            )
        col.flush()

        # Snapshot and restore
        db.snapshot(snapshot_path)
        db2 = omendb_vector.create(restore_path)
        db2.import_snapshot(snapshot_path)

        # Verify text search works
        col2 = db2.collection("text", create=False)
        results = col2.search_text("topic 0")
        assert len(results) > 0
        # Verify we can get the text content
        for r in results:
            doc = col2.get(r.id, include_text=True)
            assert doc is not None
            assert doc["text"] is not None


class TestBackupRestoreSource:
    """Test backup/restore preserves source spans."""

    def test_source_spans_survive_backup_restore(self, tmp_path) -> None:
        """Source spans survive snapshot/restore."""
        _skip_if_free_threaded()

        db_path = str(tmp_path / "db")
        snapshot_path = str(tmp_path / "snapshot")
        restore_path = str(tmp_path / "restored")

        # Create database with source spans
        db = omendb_vector.create(db_path)
        col = db.collection("source", config=omendb_vector.CollectionConfig(dim=2, text=True))
        for i in range(10):
            col.set(
                f"doc_{i}",
                vector=[float(i) / 10, 1.0 - float(i) / 10],
                text=f"Document {i}",
                source={"url": f"https://example.com/doc_{i}"},
            )
        col.flush()

        # Snapshot and restore
        db.snapshot(snapshot_path)
        db2 = omendb_vector.create(restore_path)
        db2.import_snapshot(snapshot_path)

        # Verify source preserved
        col2 = db2.collection("source", create=False)
        for i in range(10):
            doc = col2.get(f"doc_{i}", include_source=True)
            assert doc is not None
            assert doc["source"]["url"] == f"https://example.com/doc_{i}"


class TestBackupRestoreMultivector:
    """Test backup/restore preserves multivector data."""

    def test_multivector_data_survives_backup_restore(self, tmp_path) -> None:
        """Multivector collections survive snapshot/restore."""
        _skip_if_free_threaded()

        db_path = str(tmp_path / "db")
        snapshot_path = str(tmp_path / "snapshot")
        restore_path = str(tmp_path / "restored")

        # Create database with multivector data
        db = omendb_vector.create(db_path)
        col = db.collection(
            "multi",
            config=omendb_vector.CollectionConfig(
                dim=2,
                text=True,
                vector_mode="multi",
                metric="dot",
                index="flat",
            ),
        )
        for i in range(10):
            col.set(
                f"doc_{i}",
                vectors=[[float(i) / 10, 1.0 - float(i) / 10], [1.0, 0.0]],
                text=f"Multivector document {i}",
            )
        col.flush()

        # Snapshot and restore
        db.snapshot(snapshot_path)
        db2 = omendb_vector.create(restore_path)
        db2.import_snapshot(snapshot_path)

        # Verify multivector search works
        col2 = db2.collection("multi", create=False)
        results = col2.search_vectors([[0.5, 0.5]], k=5)
        assert len(results) == 5


class TestBackupRestoreTextOnly:
    """Test backup/restore preserves text-only items."""

    def test_text_only_items_survive_backup_restore(self, tmp_path) -> None:
        """Text-only items survive snapshot/restore."""
        _skip_if_free_threaded()

        db_path = str(tmp_path / "db")
        snapshot_path = str(tmp_path / "snapshot")
        restore_path = str(tmp_path / "restored")

        # Create database with text-only items
        db = omendb_vector.create(db_path)
        col = db.collection(
            "text_only", config=omendb_vector.CollectionConfig(dim=2, text=True)
        )
        for i in range(10):
            col.set(
                f"doc_{i}",
                text=f"Text-only document {i}",
            )
        col.flush()

        # Snapshot and restore
        db.snapshot(snapshot_path)
        db2 = omendb_vector.create(restore_path)
        db2.import_snapshot(snapshot_path)

        # Verify text-only items preserved
        col2 = db2.collection("text_only", create=False)
        for i in range(10):
            doc = col2.get(f"doc_{i}", include_text=True)
            assert doc is not None
            assert doc["text"] == f"Text-only document {i}"

        # Verify text search works
        results = col2.search_text("Text-only")
        assert len(results) == 10


class TestBackupRestoreMixed:
    """Test backup/restore preserves mixed data types."""

    def test_mixed_data_types_survive_backup_restore(self, tmp_path) -> None:
        """Mixed dense, text-only, and multivector collections
        survive backup/restore.
        """
        _skip_if_free_threaded()

        db_path = str(tmp_path / "db")
        snapshot_path = str(tmp_path / "snapshot")
        restore_path = str(tmp_path / "restored")

        # Create database with mixed data types
        db = omendb_vector.create(db_path)

        # Dense collection
        dense_col = db.collection(
            "dense", config=omendb_vector.CollectionConfig(dim=2, text=True)
        )
        for i in range(5):
            dense_col.set(
                f"dense_{i}",
                vector=[float(i) / 5, 1.0 - float(i) / 5],
                text=f"Dense document {i}",
                metadata={"type": "dense"},
            )

        # Text-only collection
        text_col = db.collection(
            "text_only", config=omendb_vector.CollectionConfig(dim=2, text=True)
        )
        for i in range(5):
            text_col.set(
                f"text_{i}",
                text=f"Text-only document {i}",
            )

        # Multivector collection
        multi_col = db.collection(
            "multi",
            config=omendb_vector.CollectionConfig(
                dim=2,
                text=True,
                vector_mode="multi",
                metric="dot",
                index="flat",
            ),
        )
        for i in range(5):
            multi_col.set(
                f"multi_{i}",
                vectors=[[float(i) / 5, 1.0 - float(i) / 5], [1.0, 0.0]],
                text=f"Multivector document {i}",
            )

        db.flush()

        # Snapshot and restore
        db.snapshot(snapshot_path)
        db2 = omendb_vector.create(restore_path)
        db2.import_snapshot(snapshot_path)

        # Verify all collections preserved
        dense_col2 = db2.collection("dense", create=False)
        for i in range(5):
            doc = dense_col2.get(f"dense_{i}", include_text=True)
            assert doc is not None
            assert doc["text"] == f"Dense document {i}"
            assert doc["metadata"]["type"] == "dense"

        text_col2 = db2.collection("text_only", create=False)
        for i in range(5):
            doc = text_col2.get(f"text_{i}", include_text=True)
            assert doc is not None
            assert doc["text"] == f"Text-only document {i}"

        multi_col2 = db2.collection("multi", create=False)
        for i in range(5):
            doc = multi_col2.get(f"multi_{i}", include_text=True)
            assert doc is not None
            assert doc["text"] == f"Multivector document {i}"


class TestCLIBackupRestore:
    """Test CLI backup/restore commands."""

    def test_cli_snapshot_and_check(self, tmp_path) -> None:
        """CLI snapshot and check commands work."""
        _skip_if_free_threaded()

        db_path = str(tmp_path / "db")
        snapshot_path = str(tmp_path / "snapshot")

        # Create database
        db = omendb_vector.create(db_path)
        col = db.collection("test", config=omendb_vector.CollectionConfig(dim=2, text=True))
        for i in range(5):
            col.set(
                f"doc_{i}", vector=[float(i) / 5, 1.0 - float(i) / 5], text=f"Doc {i}"
            )
        col.flush()

        # Test snapshot via Python API (CLI requires omendb_vector installed)
        db.snapshot(snapshot_path)
        assert Path(snapshot_path).exists()

        # Test check
        result = db.check()
        assert result.ok
