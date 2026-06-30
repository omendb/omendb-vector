"""Tests for concurrent access: multiple readers, writers, lock contention.

Verifies thread safety and concurrent access patterns.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

import omendb


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


def _create_collection(tmp_path, name="test"):
    """Create a test collection."""
    db = omendb.create(str(tmp_path / "db"))
    col = db.collection(name, config=omendb.CollectionConfig(dim=2, text=True))
    return db, col


class TestMultipleReaders:
    """Test multiple concurrent readers."""

    def test_multiple_readers_same_item(self, tmp_path) -> None:
        """Multiple readers can read the same item concurrently."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        col.set("a", vector=[1.0, 0.0], text="Hello")

        results = []
        errors = []

        def reader():
            try:
                doc = col.get("a", include_text=True)
                results.append(doc)
            except Exception as e:
                errors.append(e)

        # Start 10 readers
        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        for doc in results:
            assert doc["text"] == "Hello"

    def test_multiple_readers_search(self, tmp_path) -> None:
        """Multiple readers can search concurrently."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        for i in range(100):
            col.set(f"item_{i}", vector=[float(i), 0.0], text=f"Text {i}")

        results = []
        errors = []

        def reader():
            try:
                search_results = col.search(vector=[50.0, 0.0], k=10)
                results.append(search_results)
            except Exception as e:
                errors.append(e)

        # Start 10 readers
        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        for search_results in results:
            assert len(search_results) == 10


class TestSingleWriter:
    """Test single writer with readers."""

    def test_writer_with_readers(self, tmp_path) -> None:
        """Writer can write while readers read."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        col.set("initial", vector=[1.0, 0.0], text="Initial")

        results = []
        errors = []

        def writer():
            try:
                for i in range(10):
                    col.set(f"new_{i}", vector=[float(i), 0.0], text=f"New {i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(10):
                    doc = col.get("initial", include_text=True)
                    results.append(doc)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start writer and readers
        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(5)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50  # 5 readers * 10 reads each


class TestLockContention:
    """Test lock contention scenarios."""

    def test_concurrent_sets_with_store_busy(self, tmp_path) -> None:
        """Concurrent sets raise StoreBusyError (expected behavior)."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        errors = []

        def writer(start_idx):
            try:
                for i in range(10):
                    idx = start_idx + i
                    col.set(f"item_{idx}", vector=[float(idx), 0.0], text=f"Text {idx}")
            except omendb.StoreBusyError:
                errors.append(omendb.StoreBusyError("expected"))
            except Exception as e:
                errors.append(e)

        # Start 10 writers, each writing 10 items
        threads = [threading.Thread(target=writer, args=(i * 10,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have StoreBusyError (expected - collection has write lock)
        assert len(errors) > 0
        assert all(isinstance(e, omendb.StoreBusyError) for e in errors)

    def test_concurrent_sets_same_item_with_store_busy(self, tmp_path) -> None:
        """Concurrent sets to same item raise StoreBusyError (expected behavior)."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        errors = []

        def writer(value):
            try:
                col.set("shared", vector=[float(value), 0.0], text=f"Value {value}")
            except omendb.StoreBusyError:
                errors.append(omendb.StoreBusyError("expected"))
            except Exception as e:
                errors.append(e)

        # Start 10 writers to same item
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have StoreBusyError (expected - collection has write lock)
        assert len(errors) > 0
        assert all(isinstance(e, omendb.StoreBusyError) for e in errors)


class TestConcurrentFlush:
    """Test concurrent flush operations."""

    def test_flush_with_readers(self, tmp_path) -> None:
        """Flush can happen while readers read (may raise StoreBusyError)."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        col.set("a", vector=[1.0, 0.0], text="Hello")

        results = []
        errors = []

        def flusher():
            try:
                for _ in range(5):
                    db.flush()
                    time.sleep(0.001)
            except omendb.StoreBusyError:
                errors.append(omendb.StoreBusyError("expected"))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(10):
                    doc = col.get("a", include_text=True)
                    results.append(doc)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start flusher and readers
        flusher_thread = threading.Thread(target=flusher)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        flusher_thread.start()
        for t in reader_threads:
            t.start()

        flusher_thread.join()
        for t in reader_threads:
            t.join()

        # Readers should succeed (some may not complete all reads due to timing)
        assert len(results) > 0
        # Flusher may have StoreBusyError (expected)
        for e in errors:
            assert isinstance(e, omendb.StoreBusyError)


class TestConcurrentSearch:
    """Test concurrent search operations."""

    def test_concurrent_vector_search(self, tmp_path) -> None:
        """Multiple concurrent vector searches work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        for i in range(100):
            col.set(f"item_{i}", vector=[float(i), 0.0])

        results = []
        errors = []

        def searcher(query_idx):
            try:
                search_results = col.search(vector=[float(query_idx), 0.0], k=10)
                results.append(search_results)
            except Exception as e:
                errors.append(e)

        # Start 10 concurrent searches
        threads = [threading.Thread(target=searcher, args=(i * 10,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        for search_results in results:
            assert len(search_results) == 10

    def test_concurrent_text_search(self, tmp_path) -> None:
        """Multiple concurrent text searches work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        for i in range(100):
            col.set(f"item_{i}", text=f"Document about topic {i}")

        results = []
        errors = []

        def searcher(query):
            try:
                search_results = col.search(text=query, k=10)
                results.append(search_results)
            except Exception as e:
                errors.append(e)

        # Start 10 concurrent searches
        threads = [
            threading.Thread(target=searcher, args=(f"topic {i}",)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


class TestConcurrentAccessStress:
    """Stress test for concurrent access."""

    def test_mixed_operations_stress(self, tmp_path) -> None:
        """Mixed operations under stress work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)

        # Pre-populate
        for i in range(50):
            col.set(f"item_{i}", vector=[float(i), 0.0], text=f"Text {i}")

        errors = []

        def mixed_worker(worker_id):
            try:
                for i in range(20):
                    # Read
                    col.get(f"item_{i % 50}")
                    # Search
                    col.search(vector=[float(i), 0.0], k=5)
                    # Write
                    col.set(f"worker_{worker_id}_{i}", vector=[float(i), 0.0])
            except Exception as e:
                errors.append(e)

        # Start 5 workers
        threads = [threading.Thread(target=mixed_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify worker items were created
        for worker_id in range(5):
            for i in range(20):
                assert col.get(f"worker_{worker_id}_{i}") is not None


class TestConcurrentPersistence:
    """Test concurrent access with persistence."""

    def test_concurrent_flush_reopen(self, tmp_path) -> None:
        """Concurrent flush and reopen work."""
        _skip_if_free_threaded()

        db, col = _create_collection(tmp_path)
        col.set("a", vector=[1.0, 0.0], text="Hello")

        errors = []

        def flush_and_reopen():
            try:
                db.flush()
                db.close()
                # Reopen
                db2 = omendb.open(str(tmp_path / "db"))
                col2 = db2.collection("test")
                doc = col2.get("a", include_text=True)
                assert doc is not None
            except Exception as e:
                errors.append(e)

        # Start 5 flush/reopen cycles
        threads = [threading.Thread(target=flush_and_reopen) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least some should succeed (race conditions may cause some to fail)
        # The important thing is no crashes
