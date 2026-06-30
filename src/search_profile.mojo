from store import (
    HNSWParams,
    Metric,
    SearchOptions,
    VectorStore,
    VectorStoreOptions,
)
from std.time import perf_counter_ns


def elapsed_ms(start_ns: UInt, end_ns: UInt) -> Float64:
    return Float64(end_ns - start_ns) / 1_000_000.0


def main() raises:
    comptime dim = 128
    var n_items = 10_000
    var n_queries = 100
    var k = 10

    print("=== Search Performance Profile ===")
    print("Items: " + String(n_items))
    print("Queries: " + String(n_queries))
    print("Dimension: " + String(dim))
    print("k: " + String(k))
    print("")

    # Create store
    var opts = VectorStoreOptions()
    opts.hnsw.M = 16
    opts.hnsw.ef_construction = 100
    var store = VectorStore[dim].create_in_memory(opts)

    # Insert items with random-ish vectors
    var t0 = perf_counter_ns()
    for i in range(n_items):
        var vec = List[Float32]()
        for d in range(dim):
            vec.append(Float32(i % 1000) * 0.001 + Float32(d) * 0.0001)
        _ = store.set(
            "item_" + String(i), Span(ptr=vec.unsafe_ptr(), length=dim)
        )
    var t1 = perf_counter_ns()
    print(
        "Insert: "
        + String(elapsed_ms(t0, t1))
        + " ms ("
        + String(Float64(n_items) / elapsed_ms(t0, t1) * 1000.0)
        + " items/sec)"
    )

    # Prepare queries
    var queries = List[List[Float32]]()
    for q in range(n_queries):
        var vec = List[Float32]()
        for d in range(dim):
            vec.append(Float32(q) * 0.01 + Float32(d) * 0.001)
        queries.append(vec^)

    # Benchmark: no-filter search
    var t2 = perf_counter_ns()
    for q in range(n_queries):
        var results = store.search(
            Span(ptr=queries[q].unsafe_ptr(), length=dim),
            SearchOptions(k=k),
        )
    var t3 = perf_counter_ns()
    print(
        "No-filter search: "
        + String(elapsed_ms(t2, t3))
        + " ms ("
        + String(Float64(n_queries) / elapsed_ms(t2, t3) * 1000.0)
        + " queries/sec)"
    )

    # Benchmark: 50% selectivity filter
    var allowlist_50 = List[UInt8]()
    for i in range(n_items):
        if i < n_items / 2:
            allowlist_50.append(1)
        else:
            allowlist_50.append(0)
    var t4 = perf_counter_ns()
    for q in range(n_queries):
        var results = store.search(
            Span(ptr=queries[q].unsafe_ptr(), length=dim),
            SearchOptions(k=k),
            allowlist=Optional[List[UInt8]](allowlist_50.copy()),
        )
    var t5 = perf_counter_ns()
    print(
        "50% selectivity: "
        + String(elapsed_ms(t4, t5))
        + " ms ("
        + String(Float64(n_queries) / elapsed_ms(t4, t5) * 1000.0)
        + " queries/sec)"
    )

    # Benchmark: 10% selectivity filter
    var allowlist_10 = List[UInt8]()
    for i in range(n_items):
        if i < n_items / 10:
            allowlist_10.append(1)
        else:
            allowlist_10.append(0)
    var t6 = perf_counter_ns()
    for q in range(n_queries):
        var results = store.search(
            Span(ptr=queries[q].unsafe_ptr(), length=dim),
            SearchOptions(k=k),
            allowlist=Optional[List[UInt8]](allowlist_10.copy()),
        )
    var t7 = perf_counter_ns()
    print(
        "10% selectivity: "
        + String(elapsed_ms(t6, t7))
        + " ms ("
        + String(Float64(n_queries) / elapsed_ms(t6, t7) * 1000.0)
        + " queries/sec)"
    )

    # Benchmark: 1% selectivity filter (exact scan path)
    var allowlist_1 = List[UInt8]()
    for i in range(n_items):
        if i < n_items / 100:
            allowlist_1.append(1)
        else:
            allowlist_1.append(0)
    var t8 = perf_counter_ns()
    for q in range(n_queries):
        var results = store.search(
            Span(ptr=queries[q].unsafe_ptr(), length=dim),
            SearchOptions(k=k),
            allowlist=Optional[List[UInt8]](allowlist_1.copy()),
        )
    var t9 = perf_counter_ns()
    print(
        "1% selectivity: "
        + String(elapsed_ms(t8, t9))
        + " ms ("
        + String(Float64(n_queries) / elapsed_ms(t8, t9) * 1000.0)
        + " queries/sec)"
    )

    print("\n=== Done ===")
