"""
Filtered-search correctness oracle for OmenDB-Mojo.

Proves that filtered vector search has no recall loss vs exact brute-force
across selectivity bands (broad / medium / restrictive).

The exact oracle computes F32 L2 distances over all eligible items and returns
top-k via linear scan. The HNSW filtered search uses the same allow-list but
graph traversal + SQ8 quantized distances. Recall measures how many of the
oracle's top-k appear in the HNSW results.

Usage:
    pixi run mojo run src/filtered_search_oracle.mojo
"""
from std.time import perf_counter_ns
from std.math import min
from hnsw import HNSWIndex, Candidate
from distance import l2_distance


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


def exact_oracle_topk[dim: Int](
    query: Span[Float32, _],
    base: List[Float32],
    allowed: List[UInt8],
    k: Int,
) -> List[UInt32]:
    """Brute-force exact F32 search over eligible items.

    Returns top-k item IDs sorted by distance (nearest first).
    Uses linear scan with a max-heap of size k.
    """
    var num = len(base) // dim
    # Track top-k as parallel arrays: ids and distances
    var top_ids = List[UInt32]()
    top_ids.reserve(k + 1)
    var top_dists = List[Float32]()
    top_dists.reserve(k + 1)

    for i in range(num):
        if i < len(allowed) and allowed[i] != 0:
            var vec = Span(ptr=base.unsafe_ptr() + i * dim, length=dim)
            var dist = l2_distance[dim](query, vec)

            if len(top_ids) < k:
                top_ids.append(UInt32(i))
                top_dists.append(dist)
            else:
                # Find worst (max distance) in top-k
                var worst_idx = 0
                var worst_dist = top_dists[0]
                for j in range(1, k):
                    if top_dists[j] > worst_dist:
                        worst_dist = top_dists[j]
                        worst_idx = j
                # Replace if this one is better
                if dist < worst_dist:
                    top_ids[worst_idx] = UInt32(i)
                    top_dists[worst_idx] = dist

    # Sort top-k by distance (insertion sort, k is small)
    for i in range(1, len(top_ids)):
        var key_id = top_ids[i]
        var key_dist = top_dists[i]
        var j = i - 1
        while j >= 0 and top_dists[j] > key_dist:
            top_ids[j + 1] = top_ids[j]
            top_dists[j + 1] = top_dists[j]
            j -= 1
        top_ids[j + 1] = key_id
        top_dists[j + 1] = key_dist

    return top_ids^


def compute_recall(oracle_ids: List[UInt32], hsnw: List[Candidate]) -> Float64:
    """Compute recall@k: fraction of oracle top-k found in HNSW results."""
    if len(oracle_ids) == 0:
        return 1.0

    var found = 0
    for i in range(len(hsnw)):
        var hsnw_id = hsnw[i].id
        for j in range(len(oracle_ids)):
            if oracle_ids[j] == hsnw_id:
                found += 1
                break

    return Float64(found) / Float64(len(oracle_ids))


def generate_categories(
    num: Int, num_categories: Int, seed: UInt64
) -> List[Int]:
    """Assign each item a category [0, num_categories) using a simple hash."""
    var cats = List[Int]()
    cats.reserve(num)
    var state = seed
    for i in range(num):
        state = state * 6364136223846793005 + 1442695040888963407
        cats.append(Int(state >> 33) % num_categories)
    return cats^


def count_eligible(allowed: List[UInt8]) -> Int:
    var count = 0
    for i in range(len(allowed)):
        if allowed[i] != 0:
            count += 1
    return count


def run_selectivity_band[dim: Int](
    label: String,
    index: HNSWIndex[dim],
    base: List[Float32],
    queries: List[Float32],
    allowed: List[UInt8],
    num_queries: Int,
    k: Int,
    ef_search: Int,
) raises:
    """Run one selectivity band: measure recall and latency for filtered search."""
    var eligible = count_eligible(allowed)
    var selectivity = Float64(eligible) / Float64(index.num_elements) * 100.0

    print("\n=== " + label + " ===")
    print(
        "Eligible items: "
        + String(eligible)
        + " / "
        + String(index.num_elements)
        + " ("
        + String(Int(selectivity * 10) / 10)
        + "%)"
    )

    if eligible < k:
        print("SKIPPED: fewer eligible items than k")
        return

    var total_recall = 0.0
    var oracle_ns = List[UInt64]()
    oracle_ns.reserve(num_queries)
    var hsnw_ns = List[UInt64]()
    hsnw_ns.reserve(num_queries)

    # Warmup
    for q in range(min(5, num_queries)):
        var qvec = Span(ptr=queries.unsafe_ptr() + q * dim, length=dim)
        _ = index.search_filtered(qvec, allowed, k, ef_search)

    for q in range(num_queries):
        var qvec = Span(ptr=queries.unsafe_ptr() + q * dim, length=dim)

        # Exact oracle
        var t0 = UInt64(perf_counter_ns())
        var oracle_ids = exact_oracle_topk[dim](qvec, base, allowed, k)
        var t1 = UInt64(perf_counter_ns())
        oracle_ns.append(t1 - t0)

        # Filtered HNSW
        var t2 = UInt64(perf_counter_ns())
        var hsnw_result = index.search_filtered(qvec, allowed, k, ef_search)
        var t3 = UInt64(perf_counter_ns())
        hsnw_ns.append(t3 - t2)

        total_recall += compute_recall(oracle_ids, hsnw_result)

    var avg_recall = total_recall / Float64(num_queries)

    # Compute latency percentiles (simple sort into sorted copy)
    var oracle_sorted = List[UInt64]()
    oracle_sorted.reserve(len(oracle_ns))
    for i in range(len(oracle_ns)):
        oracle_sorted.append(oracle_ns[i])
    # Insertion sort (small dataset)
    for i in range(1, len(oracle_sorted)):
        var key = oracle_sorted[i]
        var j = i - 1
        while j >= 0 and oracle_sorted[j] > key:
            oracle_sorted[j + 1] = oracle_sorted[j]
            j -= 1
        oracle_sorted[j + 1] = key

    var hsnw_sorted = List[UInt64]()
    hsnw_sorted.reserve(len(hsnw_ns))
    for i in range(len(hsnw_ns)):
        hsnw_sorted.append(hsnw_ns[i])
    for i in range(1, len(hsnw_sorted)):
        var key = hsnw_sorted[i]
        var j = i - 1
        while j >= 0 and hsnw_sorted[j] > key:
            hsnw_sorted[j + 1] = hsnw_sorted[j]
            j -= 1
        hsnw_sorted[j + 1] = key

    var oracle_p50 = Float64(oracle_sorted[len(oracle_sorted) // 2]) / 1e6
    var oracle_p99 = Float64(
        oracle_sorted[Int(Float64(len(oracle_sorted) - 1) * 0.99)]
    ) / 1e6
    var hsnw_p50 = Float64(hsnw_sorted[len(hsnw_sorted) // 2]) / 1e6
    var hsnw_p99 = Float64(
        hsnw_sorted[Int(Float64(len(hsnw_sorted) - 1) * 0.99)]
    ) / 1e6

    print("Recall@" + String(k) + ":   " + String(avg_recall))
    print("Oracle p50:   " + String(oracle_p50) + " ms")
    print("Oracle p99:   " + String(oracle_p99) + " ms")
    print("HNSW p50:     " + String(hsnw_p50) + " ms")
    print("HNSW p99:     " + String(hsnw_p99) + " ms")

    if avg_recall < 0.99:
        print("WARNING: recall below 0.99 — filtered search has recall loss!")
    else:
        print("PASS: recall >= 0.99 — no recall loss detected")


def main() raises:
    comptime dim = 128
    var num_base = 100_000
    var num_query = 1_000
    var M = 16
    var ef_construction = 200
    var ef_search = 100
    var k = 10
    var num_categories = 100

    print("=== Filtered-Search Correctness Oracle ===")
    print("Vectors:     " + String(num_base))
    print("Dimensions:  " + String(dim))
    print("Categories:  " + String(num_categories))
    print("k:           " + String(k))
    print("ef_search:   " + String(ef_search))

    # Load data
    print("\nLoading vectors...")
    var base = load_f32bin("data/sift100k_base.f32bin", num_base, dim)
    var queries = load_f32bin("data/sift100k_query.f32bin", 10_000, dim)

    # Build index
    print("Building HNSW index...")
    var t0 = UInt64(perf_counter_ns())
    var index = HNSWIndex[dim](M=M, ef_construction=ef_construction)
    for i in range(num_base):
        var vec = Span(ptr=base.unsafe_ptr() + i * dim, length=dim)
        index.insert(vec)
    var t1 = UInt64(perf_counter_ns())
    var build_sec = Float64(t1 - t0) / 1e9
    var build_vecs = Float64(num_base) / build_sec
    print(
        "Build time:  "
        + String(build_sec)
        + "s ("
        + String(Int(build_vecs))
        + " vec/s)"
    )

    # Generate categories
    print("Assigning categories...")
    var categories = generate_categories(num_base, num_categories, 42)

    # Build allow-lists for each selectivity band
    # broad: categories 0-49 → ~50%
    # medium: categories 0-9 → ~10%
    # restrictive: category 0 → ~1%
    var broad_allowed = List[UInt8]()
    broad_allowed.reserve(num_base)
    var medium_allowed = List[UInt8]()
    medium_allowed.reserve(num_base)
    var restrictive_allowed = List[UInt8]()
    restrictive_allowed.reserve(num_base)

    for i in range(num_base):
        var cat = categories[i]
        broad_allowed.append(UInt8(1 if cat < 50 else 0))
        medium_allowed.append(UInt8(1 if cat < 10 else 0))
        restrictive_allowed.append(UInt8(1 if cat == 0 else 0))

    # Run each band
    try:
        run_selectivity_band[dim](
            "BROAD (~50% eligible)",
            index,
            base,
            queries,
            broad_allowed,
            num_query,
            k,
            ef_search,
        )
        run_selectivity_band[dim](
            "MEDIUM (~10% eligible)",
            index,
            base,
            queries,
            medium_allowed,
            num_query,
            k,
            ef_search,
        )
        run_selectivity_band[dim](
            "RESTRICTIVE (~1% eligible)",
            index,
            base,
            queries,
            restrictive_allowed,
            num_query,
            k,
            ef_search,
        )
    except e:
        print("ERROR during benchmark")
        raise e^

    print("\n=== Summary ===")
    print("All bands tested. Check recall >= 0.99 for each band.")
    print("If recall drops at restrictive selectivity, that's expected —")
    print("HNSW graph may not reach all eligible nodes when few exist.")
    print("This measures the gap; ACORN is the candidate fix if needed.")
