"""Tests for the omendb.ingest module.

Verifies file/directory ingestion with source spans, chunking, and metadata.
"""

from __future__ import annotations

import sys

import pytest

import omendb
from omendb.ingest import ingest_directory, ingest_file, ingest_text


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


class TestIngestText:
    """Test ingest_text helper."""

    def test_ingest_text_basic(self, tmp_path) -> None:
        """Basic text ingestion works."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ingest_text(
            col,
            "doc_1",
            "Hello, world!",
            source={"path": "test.txt"},
            metadata={"type": "greeting"},
        )

        doc = col.get("doc_1", include_text=True, include_source=True)
        assert doc is not None
        assert doc["text"] == "Hello, world!"
        assert doc["source"]["path"] == "test.txt"
        assert doc["metadata"]["type"] == "greeting"


class TestIngestFile:
    """Test ingest_file helper."""

    def test_ingest_file_whole(self, tmp_path) -> None:
        """Ingest entire file as one item."""
        _skip_if_free_threaded()

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!\nThis is a test.")

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_file(col, test_file)
        assert len(ids) == 1
        assert ids[0].startswith("file:")

        doc = col.get(ids[0], include_text=True, include_source=True)
        assert doc is not None
        assert doc["text"] == "Hello, world!\nThis is a test."
        assert doc["source"]["path"] == str(test_file)
        assert doc["metadata"]["file"] == str(test_file)

    def test_ingest_file_chunked(self, tmp_path) -> None:
        """Ingest file with chunking."""
        _skip_if_free_threaded()

        # Create test file with enough content for chunking
        test_file = tmp_path / "test.txt"
        test_file.write_text("A" * 100 + "\n" + "B" * 100 + "\n" + "C" * 100)

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_file(col, test_file, chunk_size=150)
        assert len(ids) > 1

        # Verify all chunks have source metadata
        for item_id in ids:
            doc = col.get(item_id, include_text=True, include_source=True)
            assert doc is not None
            assert doc["source"]["path"] == str(test_file)
            assert "chunk_index" in doc["metadata"]

    def test_ingest_file_with_overlap(self, tmp_path) -> None:
        """Ingest file with chunk overlap."""
        _skip_if_free_threaded()

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("A" * 100 + "\n" + "B" * 100)

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_file(col, test_file, chunk_size=80, chunk_overlap=20)
        assert len(ids) > 1

    def test_ingest_file_not_found(self, tmp_path) -> None:
        """Ingesting non-existent file raises error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        with pytest.raises(FileNotFoundError):
            ingest_file(col, tmp_path / "nonexistent.txt")

    def test_ingest_file_empty(self, tmp_path) -> None:
        """Ingesting empty file returns no IDs."""
        _skip_if_free_threaded()

        # Create empty file
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_file(col, test_file)
        assert len(ids) == 0


class TestIngestDirectory:
    """Test ingest_directory helper."""

    def test_ingest_directory_recursive(self, tmp_path) -> None:
        """Ingest directory recursively."""
        _skip_if_free_threaded()

        # Create test files
        (tmp_dir := tmp_path / "src").mkdir()
        (tmp_dir / "a.py").write_text("print('a')")
        (tmp_dir / "b.py").write_text("print('b')")
        sub_dir = tmp_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "c.py").write_text("print('c')")

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_directory(col, tmp_dir, glob="*.py")
        assert len(ids) == 3

    def test_ingest_directory_non_recursive(self, tmp_path) -> None:
        """Ingest directory non-recursively."""
        _skip_if_free_threaded()

        # Create test files
        (tmp_dir := tmp_path / "src").mkdir()
        (tmp_dir / "a.py").write_text("print('a')")
        sub_dir = tmp_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "b.py").write_text("print('b')")

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_directory(col, tmp_dir, glob="*.py", recursive=False)
        assert len(ids) == 1

    def test_ingest_directory_not_found(self, tmp_path) -> None:
        """Ingesting non-existent directory raises error."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        with pytest.raises(FileNotFoundError):
            ingest_directory(col, tmp_path / "nonexistent")

    def test_ingest_directory_not_a_directory(self, tmp_path) -> None:
        """Ingesting a file as directory raises error."""
        _skip_if_free_threaded()

        # Create a file
        test_file = tmp_path / "file.txt"
        test_file.write_text("hello")

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        with pytest.raises(NotADirectoryError):
            ingest_directory(col, test_file)

    def test_ingest_directory_with_metadata(self, tmp_path) -> None:
        """Ingest directory with custom metadata."""
        _skip_if_free_threaded()

        # Create test file
        (tmp_dir := tmp_path / "src").mkdir()
        (tmp_dir / "a.py").write_text("print('a')")

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_directory(col, tmp_dir, glob="*.py", metadata={"project": "test"})
        assert len(ids) == 1

        doc = col.get(ids[0], include_text=True)
        assert doc is not None
        assert doc["metadata"]["project"] == "test"

    def test_ingest_directory_chunked(self, tmp_path) -> None:
        """Ingest directory with chunking."""
        _skip_if_free_threaded()

        # Create test file with enough content
        (tmp_dir := tmp_path / "src").mkdir()
        (tmp_dir / "large.py").write_text("# " + "A" * 500 + "\n" + "# " + "B" * 500)

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2, text=True))

        ids = ingest_directory(col, tmp_dir, glob="*.py", chunk_size=300)
        assert len(ids) > 1
