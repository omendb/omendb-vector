"""
HNSW vs SymphonyQG — search QPS benchmark on VectorIndex directly.

Usage: pixi run mojo run src/oracle_bench.mojo
"""

from vector_index import VectorIndex
from store_types import Metric
from std.memory import Span
from std.math import sqrt
from std.time import perf_counter_ns
from std.python import Python


def _random_unit_vector[dim: Int](seed: Int) raises -> List[Float32]:
    var v = List[Float32]()
    var sum_sq: Float32 = 0.0
    for d in range(dim):
        var raw = Float32((seed * 7919 + d * 6271) % 10007) / 10007.0 - 0.5
        sum_sq += raw * raw
        v.append(raw)
    var norm_val = sqrt(Float64(sum_sq))
    if norm_val > 0:
        var nf = Float32(norm_val)
        for d in range(dim):
            v[d] = v[d] / nf
    return v^


def main() raises:
    var py = Python.import_module("builtins")
    comptime DIM: Int = 128

    py.print("── HNSW vs SymphonyQG — Search QPS (VectorIndex) ──")
    py.print()

    for nv in [1000, 10000]:
        py.print("───", nv, "vectors ───")
        var flat = List[Float32]()
        for i in range(nv):
            var v = _random_unit_vector[DIM](i)
            for j in range(DIM):
                flat.append(v[j])

        # HNSW
        var idx_hnsw = VectorIndex[DIM](M=16, ef_construction=100, metric=Metric.L2, backend=0)
        var t0 = perf_counter_ns()
        for i in range(nv):
            var span = Span[Float32, _](ptr=flat.unsafe_ptr() + i * DIM, length=DIM)
            idx_hnsw.insert(span)
        var hnsw_build = perf_counter_ns() - t0

        # SymphonyQG
        var idx_qg = VectorIndex[DIM](M=16, ef_construction=100, metric=Metric.L2, backend=1)
        t0 = perf_counter_ns()
        for i in range(nv):
            var span = Span[Float32, _](ptr=flat.unsafe_ptr() + i * DIM, length=DIM)
            idx_qg.insert(span)
        var qg_build = perf_counter_ns() - t0

        var nq = 100
        var queries = List[List[Float32]]()
        for qi in range(nq):
            queries.append(_random_unit_vector[DIM](50000 + qi))

        # HNSW search QPS
        var total_ns: Int = 0
        for qi in range(nq):
            var q = Span[Float32, _](ptr=queries[qi].unsafe_ptr(), length=DIM)
            t0 = perf_counter_ns()
            var results1 = idx_hnsw.search(q, 10, 100)
            total_ns += Int(perf_counter_ns() - t0)
        var hnsw_ms = Float64(total_ns) / Float64(nq) / 1_000_000.0
        py.print("hnsw:    build=", hnsw_build/1_000_000, "ms", "search=", hnsw_ms, "ms/q  QPS=", 1000.0/hnsw_ms)

        # QG search QPS
        total_ns = 0
        for qi in range(nq):
            var q = Span[Float32, _](ptr=queries[qi].unsafe_ptr(), length=DIM)
            t0 = perf_counter_ns()
            var results2 = idx_qg.search(q, 10, 50)
            total_ns += Int(perf_counter_ns() - t0)
        var qg_ms = Float64(total_ns) / Float64(nq) / 1_000_000.0
        py.print("symphonyqg: build=", qg_build/1_000_000, "ms", "search=", qg_ms, "ms/q  QPS=", 1000.0/qg_ms)

        var ratio = (1000.0 / hnsw_ms) / (1000.0 / qg_ms) if qg_ms > 0 else -1.0
        py.print("ratio (hnsw/qg):", ratio, "x")
        py.print()

    py.print("done.")
