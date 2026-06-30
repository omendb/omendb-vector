"""
Recall oracle for SymphonyQG — full calibration.

Measures recall@10 vs brute-force at various ef values.
Uses parallel lists for brute-force (avoids Mojo tuple Copyable issues).

Usage: pixi run mojo run src/oracle_symphonyqg.mojo
"""

from symphonyqg_index import SymphonyQGIndex
from rabitq import RaBitQEncoder, _popcount_u64
from store_types import Metric
from std.memory import Span
from std.math import sqrt
from std.time import perf_counter_ns
from std.python import Python


def _l2_distance[dim: Int](a: Span[Float32, _], b: Span[Float32, _]) -> Float32:
    var s: Float32 = 0.0
    for d in range(dim):
        var diff = a[d] - b[d]
        s += diff * diff
    return s


def _random_unit_vector[dim: Int](seed: Int) raises -> List[Float32]:
    """Deterministic pseudorandom unit vector from seed."""
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


def _brute_force_topk[dim: Int](
    data: Span[Float32, _], n: Int, query: Span[Float32, _], k: Int,
) raises -> List[Int]:
    """Brute-force top-k using parallel index/distance arrays (no tuples)."""
    # Collect all distances
    var dists = List[Float32]()
    var indices = List[Int]()
    for i in range(n):
        var v = Span[Float32, _](ptr=data.unsafe_ptr() + i * dim, length=dim)
        dists.append(_l2_distance[dim](query, v))
        indices.append(i)

    # Selection sort top-k
    for i in range(k):
        var best = i
        for j in range(i + 1, n):
            if dists[j] < dists[best]:
                best = j
        var tmp_d = dists[i]
        dists[i] = dists[best]
        dists[best] = tmp_d
        var tmp_i = indices[i]
        indices[i] = indices[best]
        indices[best] = tmp_i

    var result = List[Int]()
    for i in range(k):
        result.append(indices[i])
    return result^


def main() raises:
    var py = Python.import_module("builtins")
    comptime DIM: Int = 128

    py.print("── SymphonyQG Recall Calibration ──")
    py.print()

    # Parameters
    var n = 1000       # database size
    var n_queries = 100
    var k = 10

    # Generate data
    py.print("generating", n, "vectors...")
    var flat = List[Float32]()
    for i in range(n):
        var v = _random_unit_vector[DIM](i)
        for j in range(DIM):
            flat.append(v[j])
    py.print("done,", len(flat), "floats")

    # Build SymphonyQG index
    var idx = SymphonyQGIndex[DIM](Metric.L2)
    for i in range(n):
        var span = Span[Float32, _](
            ptr=flat.unsafe_ptr() + i * DIM, length=DIM
        )
        if i == 0:
            idx.insert(span, idx.encoder.encode_first(span))
        else:
            idx.insert(span, idx.encoder.encode(span))
    py.print("index built,", len(idx.raw_f32) // DIM, "vectors")
    py.print("entry_point:", idx.entry_point)
    # Check neighbor distribution
    var deg_sum = 0
    for i in range(min(20, n)):
        deg_sum += idx.neighbor_counts[i]
    py.print("first 20 avg degree:", Float64(deg_sum) / 20.0)

    # Quick RaBitQ correlation check
    py.print("RaBitQ hamming vs true L2 for first 10 pairs:")
    for i in range(10):
        var v1 = Span[Float32, _](ptr=flat.unsafe_ptr() + i * DIM, length=DIM)
        var v2 = Span[Float32, _](ptr=flat.unsafe_ptr() + (i + 1) * DIM, length=DIM)
        var s: Float32 = 0.0
        for d in range(DIM):
            var diff = v1[d] - v2[d]
            s += diff * diff
        var true_l2 = s
        var c1 = idx.encoder.encode_first(v1) if i == 0 else idx.encoder.encode(v1)
        var c2 = idx.encoder.encode(v2)
        var ham = 0
        for wi in range(idx.words_per_vec):
            ham += _popcount_u64(c1.bits[wi] ^ c2.bits[wi])
        py.print("  pair", i, "→ L2=", true_l2, "ham=", ham, "/", DIM, "bits")

    # Generate queries and ground truth
    py.print("computing ground truth for", n_queries, "queries...")
    var q_flat = List[Float32]()
    for qi in range(n_queries):
        var qv = _random_unit_vector[DIM](10000 + qi)
        for j in range(DIM):
            q_flat.append(qv[j])

    var gt_list = List[Int]()  # flattened: q0_top10, q1_top10, ...
    var gt_starts = List[Int]()
    for qi in range(n_queries):
        var q = Span[Float32, _](ptr=q_flat.unsafe_ptr() + qi * DIM, length=DIM)
        var dspan = Span[Float32, _](ptr=flat.unsafe_ptr(), length=len(flat))
        var gt = _brute_force_topk[DIM](dspan, n, q, k)
        gt_starts.append(len(gt_list))
        for gi in range(k):
            gt_list.append(gt[gi])
    py.print("done")

    # Calibrate
    py.print()
    py.print("─── recall@10 at 1K scale ───")
    py.print("ef       recall    avg_time_us")

    var ef_values = List[Int]()
    ef_values.append(50)
    ef_values.append(100)
    ef_values.append(200)
    ef_values.append(400)

    for ei in range(len(ef_values)):
        var ef = ef_values[ei]
        var total_hits = 0
        var total_ns: Int = 0

        for qi in range(n_queries):
            var q = Span[Float32, _](
                ptr=q_flat.unsafe_ptr() + qi * DIM, length=DIM
            )
            var t0 = perf_counter_ns()
            var results = idx.search(q, k, ef)
            total_ns += Int(perf_counter_ns() - t0)

            var gt_off = gt_starts[qi]
            for gi in range(k):
                var gid = gt_list[gt_off + gi]
                for ri in range(len(results)):
                    if results[ri].id == UInt32(gid):
                        total_hits += 1
                        break

        var recall = Float64(total_hits) / Float64(n_queries * k)
        var avg_us = Float64(total_ns) / Float64(n_queries) / 1000.0

        py.print("ef=", ef, "  ", recall * 100.0, "%   ", avg_us, "us")

    py.print()
    py.print("done — ef for >95% recall is the minimum viable setting")
