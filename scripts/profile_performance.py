#!/usr/bin/env python3
"""Performance profiling script for OmenDB.

Profiles performance on realistic datasets (100K+ vectors).
"""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import omendb


def generate_random_vector(dim: int) -> list[float]:
    """Generate a random unit vector."""
    import math

    vector = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def profile_insert(db_path: str, num_items: int, dim: int) -> dict:
    """Profile insert performance."""
    print(f"\n{'=' * 60}")
    print(f"Profiling insert: {num_items} items, dim={dim}")
    print(f"{'=' * 60}")

    db = omendb.create(db_path)
    col = db.collection("test", config=omendb.CollectionConfig(dim=dim))

    # Generate vectors
    print("Generating vectors...")
    vectors = [generate_random_vector(dim) for _ in range(num_items)]

    # Insert
    print("Inserting...")
    start = time.perf_counter()
    for i, vector in enumerate(vectors):
        col.set(f"item_{i}", vector=vector)
        if (i + 1) % 10000 == 0:
            print(f"  Inserted {i + 1}/{num_items}")
    elapsed = time.perf_counter() - start

    qps = num_items / elapsed
    print(f"Insert: {elapsed:.2f}s, {qps:.0f} items/s")

    return {"elapsed": elapsed, "qps": qps, "count": num_items}


def profile_search(db_path: str, num_queries: int, dim: int, k: int = 10) -> dict:
    """Profile search performance."""
    print(f"\n{'=' * 60}")
    print(f"Profiling search: {num_queries} queries, k={k}")
    print(f"{'=' * 60}")

    import shutil

    search_db_path = db_path + "_search"
    if Path(search_db_path).exists():
        shutil.rmtree(search_db_path)

    db = omendb.create(search_db_path)
    col = db.collection("test", config=omendb.CollectionConfig(dim=dim))

    # Insert some items first
    print("Inserting items for search...")
    for i in range(1000):
        col.set(f"item_{i}", vector=generate_random_vector(dim))
    db.flush()

    # Generate query vectors
    queries = [generate_random_vector(dim) for _ in range(num_queries)]

    # Warmup
    print("Warming up...")
    for i in range(10):
        col.search(vector=queries[i], k=k)

    # Search
    print("Searching...")
    latencies = []
    for i, query in enumerate(queries):
        start = time.perf_counter()
        results = col.search(vector=query, k=k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms
        if (i + 1) % 100 == 0:
            print(f"  Searched {i + 1}/{num_queries}")

    # Calculate stats
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)
    qps = num_queries / (sum(latencies) / 1000)

    print(f"Search: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
    print(f"Mean={mean:.2f}ms, QPS={qps:.0f}")

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean": mean,
        "qps": qps,
        "count": num_queries,
    }


def profile_flush(db_path: str) -> dict:
    """Profile flush performance."""
    print(f"\n{'=' * 60}")
    print("Profiling flush")
    print(f"{'=' * 60}")

    db = omendb.open(db_path)

    start = time.perf_counter()
    db.flush()
    elapsed = time.perf_counter() - start

    print(f"Flush: {elapsed:.2f}s")
    return {"elapsed": elapsed}


def profile_memory(db_path: str, dim: int) -> dict:
    """Profile memory usage (simplified - no psutil dependency)."""
    print(f"\n{'=' * 60}")
    print("Profiling memory")
    print(f"{'=' * 60}")

    import os

    # Get process memory from /proc on Linux, or estimate on macOS
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_kb = int(line.split()[1])
                    mem_mb = mem_kb / 1024
                    print(f"Memory: {mem_mb:.1f}MB")
                    return {"rss_mb": mem_mb}
    except FileNotFoundError, PermissionError:
        pass

    # Fallback: just report database size
    db_size = sum(f.stat().st_size for f in Path(db_path).rglob("*") if f.is_file())
    db_mb = db_size / 1024 / 1024
    print(f"Database size: {db_mb:.1f}MB")
    return {"db_size_mb": db_mb}


def main():
    """Run performance profiling."""
    import argparse

    parser = argparse.ArgumentParser(description="OmenDB Performance Profiling")
    parser.add_argument(
        "--items", type=int, default=100000, help="Number of items to insert"
    )
    parser.add_argument(
        "--queries", type=int, default=1000, help="Number of search queries"
    )
    parser.add_argument("--dim", type=int, default=2, help="Vector dimension")
    parser.add_argument(
        "--db-path", default="/tmp/omendb_profile", help="Database path"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean database before profiling"
    )
    args = parser.parse_args()

    if args.clean:
        import shutil

        if Path(args.db_path).exists():
            shutil.rmtree(args.db_path)

    print("OmenDB Performance Profiling")
    print(f"Items: {args.items}, Queries: {args.queries}, Dim: {args.dim}")
    print(f"Database: {args.db_path}")

    # Run profiling
    insert_stats = profile_insert(args.db_path, args.items, args.dim)
    search_stats = profile_search(args.db_path, args.queries, args.dim)
    flush_stats = profile_flush(args.db_path)
    memory_stats = profile_memory(args.db_path, args.dim)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Insert: {insert_stats['qps']:.0f} items/s")
    print(
        f"Search: p50={search_stats['p50']:.2f}ms, p95={search_stats['p95']:.2f}ms, p99={search_stats['p99']:.2f}ms"
    )
    print(f"Search QPS: {search_stats['qps']:.0f}")
    print(f"Flush: {flush_stats['elapsed']:.2f}s")
    if "delta_mb" in memory_stats:
        print(f"Memory delta: {memory_stats['delta_mb']:.1f}MB")
    elif "db_size_mb" in memory_stats:
        print(f"Database size: {memory_stats['db_size_mb']:.1f}MB")


if __name__ == "__main__":
    main()
