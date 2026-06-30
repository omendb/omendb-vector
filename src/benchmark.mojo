from distance import l2_distance
from flat_index import FlatIndex
from hnsw import HNSWIndex
from std.math import sqrt
from std.time import perf_counter_ns


def elapsed_ms(start_ns: Int, end_ns: Int) -> Float64:
    return Float64(end_ns - start_ns) / 1_000_000.0


def mean(values: List[Float64]) -> Float64:
    var total: Float64 = 0.0
    for i in range(len(values)):
        total += values[i]
    return total / Float64(len(values))


def stddev(values: List[Float64], avg: Float64) -> Float64:
    var total: Float64 = 0.0
    for i in range(len(values)):
        var delta = values[i] - avg
        total += delta * delta
    return sqrt(total / Float64(len(values)))


def print_stats(name: String, values: List[Float64], unit: String):
    var avg = mean(values)
    var sd = stddev(values, avg)
    print(
        name, ": mean=", avg, " ", unit, " stddev=", sd, " runs=", len(values)
    )


def fill_vector(mut data: List[Float32], count: Int, dim: Int):
    data.clear()
    data.reserve(count * dim)
    for i in range(count):
        for d in range(dim):
            var raw = ((i + 1) * (d + 3) * 37 + d * 101) % 1000003
            var value = Float32(raw) / 1000003.0
            data.append(value)


def bench_distance[dim: Int](warmups: Int, runs: Int, iterations: Int) raises:
    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(dim):
        a.append(Float32(i % 17) / 17.0)
        b.append(Float32((dim - i) % 19) / 19.0)

    var a_span = Span(ptr=a.unsafe_ptr(), length=dim)
    var b_span = Span(ptr=b.unsafe_ptr(), length=dim)
    var sink: Float32 = 0.0
    var samples = List[Float64]()

    for run in range(warmups + runs):
        var start = Int(perf_counter_ns())
        for _ in range(iterations):
            sink += l2_distance[dim](a_span, b_span)
        var end = Int(perf_counter_ns())
        if run >= warmups:
            samples.append(elapsed_ms(start, end))

    print_stats("distance_l2_dim_" + String(dim), samples, "ms")
    print("  iterations_per_run=", iterations, " sink=", sink)


def build_flat[dim: Int](data: List[Float32], count: Int) -> FlatIndex[dim]:
    var index = FlatIndex[dim]()
    for i in range(count):
        var vec = Span(ptr=data.unsafe_ptr() + i * dim, length=dim)
        index.insert(vec)
    return index^


def build_hnsw[dim: Int](data: List[Float32], count: Int) -> HNSWIndex[dim]:
    var index = HNSWIndex[dim](M=16, ef_construction=100, alpha=1.0)
    index.reserve(count)
    for i in range(count):
        var vec = Span(ptr=data.unsafe_ptr() + i * dim, length=dim)
        index.insert(vec)
    return index^


def recall_at_10[
    dim: Int
](
    ref flat: FlatIndex[dim],
    ref hnsw: HNSWIndex[dim],
    queries: List[Float32],
    num_queries: Int,
) -> Float32:
    var k = 10
    var recall_sum: Float32 = 0.0
    for i in range(num_queries):
        var query = Span(ptr=queries.unsafe_ptr() + i * dim, length=dim)
        var exact = flat.search(query, k)
        var approx = hnsw.search(query, k=k, ef_search=100)
        var hits = 0
        for j in range(len(approx)):
            for x in range(len(exact)):
                if approx[j].id == exact[x].id:
                    hits += 1
                    break
        recall_sum += Float32(hits) / Float32(k)
    return recall_sum / Float32(num_queries)


def estimate_hnsw_bytes[dim: Int](ref index: HNSWIndex[dim]) -> Int:
    var total = len(index.data) * 4
    total += len(index.codes)
    total += len(index.node_levels) * 8
    for layer in range(index.graph.layer_count()):
        total += index.graph.layer_neighbors_len(layer) * 4
        total += index.graph.layer_counts_len(layer) * 4
    return total


def bench_ann[
    dim: Int
](warmups: Int, runs: Int, count: Int, num_queries: Int) raises:
    var data = List[Float32]()
    var queries = List[Float32]()
    fill_vector(data, count, dim)
    fill_vector(queries, num_queries, dim)

    var flat = build_flat[dim](data, count)
    var flat_samples = List[Float64]()
    for run in range(warmups + runs):
        var start = Int(perf_counter_ns())
        for i in range(num_queries):
            var query = Span(ptr=queries.unsafe_ptr() + i * dim, length=dim)
            _ = flat.search(query, 10)
        var end = Int(perf_counter_ns())
        if run >= warmups:
            flat_samples.append(elapsed_ms(start, end))
    print_stats("flat_search_dim_" + String(dim), flat_samples, "ms")
    print("  queries_per_run=", num_queries)

    var build_samples = List[Float64]()
    var hnsw = HNSWIndex[dim](M=16, ef_construction=100, alpha=1.0)
    for run in range(warmups + runs):
        var start = Int(perf_counter_ns())
        hnsw = build_hnsw[dim](data, count)
        var end = Int(perf_counter_ns())
        if run >= warmups:
            build_samples.append(elapsed_ms(start, end))
    print_stats("hnsw_build_dim_" + String(dim), build_samples, "ms")
    print("  vectors_per_build=", count)

    var hnsw_samples = List[Float64]()
    for run in range(warmups + runs):
        var start = Int(perf_counter_ns())
        for i in range(num_queries):
            var query = Span(ptr=queries.unsafe_ptr() + i * dim, length=dim)
            _ = hnsw.search(query, k=10, ef_search=100)
        var end = Int(perf_counter_ns())
        if run >= warmups:
            hnsw_samples.append(elapsed_ms(start, end))
    print_stats("hnsw_search_dim_" + String(dim), hnsw_samples, "ms")
    print("  queries_per_run=", num_queries)

    var recall = recall_at_10[dim](flat, hnsw, queries, num_queries)
    var memory_bytes = estimate_hnsw_bytes[dim](hnsw)
    print("hnsw_recall_at_10_ef_100=", recall)
    print("hnsw_estimated_index_bytes=", memory_bytes)
    print("recall_gate: use src/hnsw_sift_test.mojo for the SIFT reload gate")


def main() raises:
    var warmups = 2
    var runs = 7
    var count = 2000
    var num_queries = 100
    print("OmenDB-Mojo benchmark gate")
    print(
        "methodology: deterministic synthetic vectors, warmups=",
        warmups,
        " runs=",
        runs,
    )
    print("dataset: count=", count, " query_count=", num_queries, " dim=32")
    print("machine: run `uname -a` with this output for reporting context")
    bench_distance[32](warmups, runs, 100000)
    bench_distance[128](warmups, runs, 50000)
    bench_ann[32](warmups, runs, count, num_queries)
