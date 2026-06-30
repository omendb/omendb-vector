"""
SIFT-128 benchmark harness for OmenDB-Mojo.

Measures HNSW build speed, search QPS, recall@10, and latency percentiles.
Supports siftsmall (10K) dataset.

Usage:
    pixi run mojo run src/sift_benchmark.mojo
"""
from std.time import perf_counter_ns
from std.math import min
from hnsw import HNSWIndex, Candidate
from persistence import save_hnsw, load_hnsw
from std.python import Python


def load_f32bin(path: String, num: Int, dim: Int) raises -> List[Float32]:
    """Load an f32 binary file into a flat Float32 list."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    var expected = num * dim * 4
    if len(bytes) < expected:
        raise Error(
            "SIFT file too small: expected "
            + String(expected)
            + " bytes, got "
            + String(len(bytes))
        )
    var data = List[Float32]()
    data.reserve(num * dim)
    var ptr = bytes.unsafe_ptr().bitcast[Float32]()
    for i in range(num * dim):
        data.append(ptr[i])
    return data^


def load_u32bin(path: String, num: Int, k: Int) raises -> List[UInt32]:
    """Load a u32 groundtruth file (num * k entries)."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    var expected = num * k * 4
    if len(bytes) < expected:
        raise Error(
            "groundtruth file too small: expected "
            + String(expected)
            + " bytes, got "
            + String(len(bytes))
        )
    var data = List[UInt32]()
    data.reserve(num * k)
    var ptr = bytes.unsafe_ptr().bitcast[UInt32]()
    for i in range(num * k):
        data.append(ptr[i])
    return data^


def percentile(sorted_vals: List[Float64], p: Float64) -> Float64:
    """Get the p-th percentile from a sorted list."""
    if len(sorted_vals) == 0:
        return 0.0
    var idx = Int(Float64(len(sorted_vals) - 1) * p / 100.0)
    return sorted_vals[idx]


def main() raises:
    comptime dim = 128
    var num_base = 100_000
    var num_query = 10_000
    var gt_k = 10
    var M = 16
    var ef_construction = 200
    var ef_search = 100
    var search_k = 10

    var base_path = "data/sift100k_base.f32bin"
    var query_path = "data/sift100k_query.f32bin"
    var gt_path = "data/sift100k_groundtruth.u32bin"

    print("=== SIFT Benchmark ===")
    print("Dataset:   sift100k")
    print("Base:      " + String(num_base))
    print("Query:     " + String(num_query))
    print("M:         " + String(M))
    print("ef_c:      " + String(ef_construction))
    print("ef_s:      " + String(ef_search))
    print("")

    # --- Load data ---
    print("Loading data...")
    var t0 = Int(perf_counter_ns())
    var base_data = load_f32bin(base_path, num_base, dim)
    var query_data = load_f32bin(query_path, num_query, dim)
    var gt_data = load_u32bin(gt_path, num_query, gt_k)
    var t1 = Int(perf_counter_ns())
    var load_sec = Float64(t1 - t0) / 1e9
    print("Loaded in " + String(load_sec) + "s")

    # --- Build HNSW ---
    print("\nBuilding HNSW index...")
    var index = HNSWIndex[dim](M=M, ef_construction=ef_construction, alpha=1.0)

    var t2 = Int(perf_counter_ns())
    for vi in range(num_base):
        var vec = Span(ptr=base_data.unsafe_ptr() + vi * dim, length=dim)
        index.insert(vec)
    var t3 = Int(perf_counter_ns())

    var build_sec = Float64(t3 - t2) / 1e9
    var vectors_per_sec = Float64(num_base) / build_sec
    print("Build time:      " + String(build_sec) + "s")
    print("Vectors/sec:     " + String(Int(vectors_per_sec)))

    # --- Search and recall ---
    print("\nSearching...")
    var latencies = List[Float64]()
    latencies.reserve(num_query)
    var recall_sum: Float64 = 0.0

    # Warmup
    for wi in range(min(10, num_query)):
        var q = Span(ptr=query_data.unsafe_ptr() + wi * dim, length=dim)
        _ = index.search(q, search_k, ef_search=ef_search)

    # Timed search
    var t4 = Int(perf_counter_ns())
    for qi in range(num_query):
        var q = Span(ptr=query_data.unsafe_ptr() + qi * dim, length=dim)

        var qs = Int(perf_counter_ns())
        var results = index.search(q, search_k, ef_search=ef_search)
        var qe = Int(perf_counter_ns())

        var q_ms = Float64(qe - qs) / 1e6
        latencies.append(q_ms)

        # Recall
        var gt_start = qi * gt_k
        var hits = 0
        for j in range(len(results)):
            var pred_id = results[j].id
            for x in range(search_k):
                if pred_id == gt_data[gt_start + x]:
                    hits += 1
                    break
        recall_sum += Float64(hits) / Float64(search_k)

    var t5 = Int(perf_counter_ns())

    # Sort latencies for percentiles (insertion sort, num_query is small)
    for si in range(1, len(latencies)):
        var key = latencies[si]
        var sj = si - 1
        while sj >= 0 and latencies[sj] > key:
            latencies[sj + 1] = latencies[sj]
            sj -= 1
        latencies[sj + 1] = key

    var search_sec = Float64(t5 - t4) / 1e9
    var qps = Float64(num_query) / search_sec
    var recall = recall_sum / Float64(num_query)
    var p50 = percentile(latencies, 50.0)
    var p95 = percentile(latencies, 95.0)
    var p99 = percentile(latencies, 99.0)

    print("\n=== Results ===")
    print(
        "Build:       "
        + String(build_sec)
        + "s  ("
        + String(Int(vectors_per_sec))
        + " vec/s)"
    )
    print("Search QPS:  " + String(Int(qps)))
    print("Recall@10:   " + String(recall))
    print("Latency p50: " + String(p50) + "ms")
    print("Latency p95: " + String(p95) + "ms")
    print("Latency p99: " + String(p99) + "ms")
    print("")

    # --- Persistence roundtrip ---
    print("Testing persistence roundtrip...")
    var persist_path = "/tmp/omendb_sift_bench"
    save_hnsw(persist_path, index)
    var loaded = load_hnsw[dim](persist_path)

    var recall_after_reload: Float64 = 0.0
    for qi in range(num_query):
        var q = Span(ptr=query_data.unsafe_ptr() + qi * dim, length=dim)
        var results = loaded.search(q, search_k, ef_search=ef_search)
        var gt_start = qi * gt_k
        var hits = 0
        for j in range(len(results)):
            var pred_id = results[j].id
            for x in range(search_k):
                if pred_id == gt_data[gt_start + x]:
                    hits += 1
                    break
        recall_after_reload += Float64(hits) / Float64(search_k)
    recall_after_reload /= Float64(num_query)
    print("Recall@10 after reload: " + String(recall_after_reload))
    print("=== Done ===")
