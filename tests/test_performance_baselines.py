"""Performance baseline tests.

Documents performance baselines for OmenDB operations.
"""

from __future__ import annotations

import math
import random
import sys
import time

import pytest

import omendb


def _skip_if_free_threaded() -> None:
    if "t" in sys.abiflags:
        pytest.skip("Mojo extension is built for standard CPython ABI")


def _generate_random_vector(dim: int) -> list[float]:
    """Generate a random unit vector."""
    vector = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


class TestInsertPerformance:
    """Test insert performance baselines."""

    def test_insert_10k_items(self, tmp_path) -> None:
        """Insert 10K items in reasonable time."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Generate vectors
        vectors = [_generate_random_vector(2) for _ in range(10000)]

        # Insert
        start = time.perf_counter()
        for i, vector in enumerate(vectors):
            col.set(f"item_{i}", vector=vector)
        elapsed = time.perf_counter() - start

        qps = 10000 / elapsed
        # Should be able to insert at least 10K items/s
        assert qps > 10000, f"Insert QPS too low: {qps:.0f}"

    def test_insert_100k_items(self, tmp_path) -> None:
        """Insert 100K items in reasonable time."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Generate vectors
        vectors = [_generate_random_vector(2) for _ in range(100000)]

        # Insert
        start = time.perf_counter()
        for i, vector in enumerate(vectors):
            col.set(f"item_{i}", vector=vector)
            if (i + 1) % 10000 == 0:
                pass  # Just for timing
        elapsed = time.perf_counter() - start

        qps = 100000 / elapsed
        # Should be able to insert at least 10K items/s
        assert qps > 10000, f"Insert QPS too low: {qps:.0f}"


class TestSearchPerformance:
    """Test search performance baselines."""

    def test_search_latency_10k_items(self, tmp_path) -> None:
        """Search latency on 10K items is reasonable."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Insert items
        for i in range(10000):
            col.set(f"item_{i}", vector=_generate_random_vector(2))

        # Warmup
        for _ in range(10):
            col.search(vector=_generate_random_vector(2), k=10)

        # Search
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            col.search(vector=_generate_random_vector(2), k=10)
            latencies.append((time.perf_counter() - start) * 1000)

        p50 = sorted(latencies)[50]
        # p50 should be under 1ms
        assert p50 < 1.0, f"Search p50 too high: {p50:.2f}ms"

    def test_search_latency_100k_items(self, tmp_path) -> None:
        """Search latency on 100K items is reasonable."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Insert items
        for i in range(100000):
            col.set(f"item_{i}", vector=_generate_random_vector(2))

        # Warmup
        for _ in range(10):
            col.search(vector=_generate_random_vector(2), k=10)

        # Search
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            col.search(vector=_generate_random_vector(2), k=10)
            latencies.append((time.perf_counter() - start) * 1000)

        p50 = sorted(latencies)[50]
        # p50 should be under 1ms
        assert p50 < 1.0, f"Search p50 too high: {p50:.2f}ms"


class TestFlushPerformance:
    """Test flush performance baselines."""

    def test_flush_10k_items(self, tmp_path) -> None:
        """Flush 10K items in reasonable time."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Insert items
        for i in range(10000):
            col.set(f"item_{i}", vector=_generate_random_vector(2))

        # Flush
        start = time.perf_counter()
        db.flush()
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Flush took too long: {elapsed:.2f}s"

    def test_flush_100k_items(self, tmp_path) -> None:
        """Flush 100K items in reasonable time."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Insert items
        for i in range(100000):
            col.set(f"item_{i}", vector=_generate_random_vector(2))

        # Flush
        start = time.perf_counter()
        db.flush()
        elapsed = time.perf_counter() - start

        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Flush took too long: {elapsed:.2f}s"


class TestReopenPerformance:
    """Test reopen performance baselines."""

    def test_reopen_10k_items(self, tmp_path) -> None:
        """Reopen 10K items in reasonable time."""
        _skip_if_free_threaded()

        db = omendb.create(str(tmp_path / "db"))
        col = db.collection("test", config=omendb.CollectionConfig(dim=2))

        # Insert items
        for i in range(10000):
            col.set(f"item_{i}", vector=_generate_random_vector(2))
        db.flush()
        db.close()

        # Reopen
        start = time.perf_counter()
        db2 = omendb.open(str(tmp_path / "db"))
        col2 = db2.collection("test")
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Reopen took too long: {elapsed:.2f}s"

        # Verify data is present
        results = col2.search(vector=_generate_random_vector(2), k=10)
        assert len(results) == 10
