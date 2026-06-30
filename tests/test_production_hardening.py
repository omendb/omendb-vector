"""Production hardening tests: resource cleanup, memory leaks, graceful degradation.

Verifies production readiness characteristics.
"""

from __future__ import annotations

import gc
import sys

import pytest

import omendb


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


class TestResourceCleanup:
    """Test resource cleanup."""

    def test_database_close_cleanup(self, tmp_path) -> None:
        """Database close releases resources."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("a", vector=[1.0, 0.0])

        # Close should release resources
        db.close()

        # Should be able to reopen
        db2 = omendb.open(str(tmp_path / "db"))
        col2 = db2.collection("test")
        assert col2.get("a") is not None

    def test_multiple_open_close_cycles(self, tmp_path) -> None:
        """Multiple open/close cycles work."""
        _skip_if_free_threaded()

        # Create first
        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("item_0", vector=[0.0, 0.0])
        db.flush()
        db.close()

        # Open/close cycles
        for i in range(1, 10):
            db = omendb.open(str(tmp_path / "db"))
            col = db.collection("test")
            col.set(f"item_{i}", vector=[float(i), 0.0])
            db.flush()
            db.close()

        # Final open should have all items
        db = omendb.open(str(tmp_path / "db"))
        col = db.collection("test")
        for i in range(10):
            assert col.get(f"item_{i}") is not None

    def test_flush_releases_resources(self, tmp_path) -> None:
        """Flush releases temporary resources."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Insert many items
        for i in range(1000):
            col.set(f"item_{i}", vector=[float(i), 0.0])

        # Flush should release resources
        db.flush()

        # Should be able to continue
        col.set("extra", vector=[0.0, 0.0])
        assert col.get("extra") is not None


class TestMemoryLeaks:
    """Test for memory leaks."""

    def test_repeated_operations_no_leak(self, tmp_path) -> None:
        """Repeated operations don't leak memory."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Do many operations
        for i in range(100):
            col.set(f"item_{i}", vector=[float(i), 0.0])
            col.get(f"item_{i}")
            col.search(vector=[float(i), 0.0], k=1)
            col.delete(f"item_{i}")

        # Check memory
        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not have significant leak (allow some variance)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, (
            f"Potential memory leak: {object_growth} new objects"
        )

    def test_flush_reopen_no_leak(self, tmp_path) -> None:
        """Flush/reopen cycles don't leak memory."""
        _skip_if_free_threaded()

        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create first
        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("item_0", vector=[0.0, 0.0])
        db.flush()
        db.close()

        # Do many flush/reopen cycles
        for i in range(1, 10):
            db = omendb.open(str(tmp_path / "db"))
            col = db.collection("test")
            col.set(f"item_{i}", vector=[float(i), 0.0])
            db.flush()
            db.close()

        # Check memory
        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not have significant leak
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, (
            f"Potential memory leak: {object_growth} new objects"
        )


class TestGracefulDegradation:
    """Test graceful degradation under pressure."""

    def test_many_collections_stress(self, tmp_path) -> None:
        """Many collections under stress work."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))

        # Create many collections
        for i in range(50):
            col = db.collection(f"col_{i}", config=omendb.CollectionConfig(dim=2))
            col.set("a", vector=[1.0, 0.0])

        # All should be accessible
        for i in range(50):
            col = db.collection(f"col_{i}")
            assert col.get("a") is not None

    def test_many_items_stress(self, tmp_path) -> None:
        """Many items under stress work."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Add many items
        for i in range(5000):
            col.set(f"item_{i}", vector=[float(i), 0.0])

        # All should be present
        for i in range(5000):
            assert col.get(f"item_{i}") is not None

    def test_rapid_flush_stress(self, tmp_path) -> None:
        """Rapid flush operations work."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Rapid flush
        for i in range(100):
            col.set(f"item_{i}", vector=[float(i), 0.0])
            db.flush()

        # All items should be present
        for i in range(100):
            assert col.get(f"item_{i}") is not None

    def test_search_under_load(self, tmp_path) -> None:
        """Search works under load."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Insert items
        for i in range(1000):
            col.set(f"item_{i}", vector=[float(i), 0.0])

        # Search under load
        for _ in range(100):
            results = col.search(vector=[500.0, 0.0], k=10)
            assert len(results) == 10


class TestLongRunningStability:
    """Test long-running stability."""

    def test_sustained_operations(self, tmp_path) -> None:
        """Sustained operations work."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Sustained operations
        for i in range(500):
            col.set(f"item_{i}", vector=[float(i), 0.0])
            if i % 100 == 0:
                db.flush()

        # Verify
        for i in range(500):
            assert col.get(f"item_{i}") is not None

    def test_mixed_operations_stability(self, tmp_path) -> None:
        """Mixed operations are stable."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Mixed operations
        for i in range(200):
            col.set(f"item_{i}", vector=[float(i), 0.0])
            col.get(f"item_{i}")
            col.search(vector=[float(i), 0.0], k=1)
            if i % 50 == 0:
                db.flush()

        # Verify
        for i in range(200):
            assert col.get(f"item_{i}") is not None


class TestErrorRecovery:
    """Test error recovery."""

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

        # Should still work
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


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    def test_multiple_readers(self, tmp_path) -> None:
        """Multiple readers work."""
        _skip_if_free_threaded()

        import threading

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))
        col.set("a", vector=[1.0, 0.0])

        results = []
        errors = []

        def reader():
            try:
                doc = col.get("a")
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
