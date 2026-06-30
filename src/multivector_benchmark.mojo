from multivector import (
    MultiVectorExactStore,
    MultiVectorFDEIndex,
    MultiVectorResult,
)
from std.time import perf_counter_ns


def elapsed_ms(start_ns: Int, end_ns: Int) -> Float64:
    return Float64(end_ns - start_ns) / 1_000_000.0


def mean(values: List[Float64]) -> Float64:
    var total: Float64 = 0.0
    for i in range(len(values)):
        total += values[i]
    return total / Float64(len(values))


def percentile(ref values: List[Float64], pct: Int) -> Float64:
    if len(values) == 0:
        return 0.0
    var sorted = List[Float64]()
    sorted.reserve(len(values))
    for i in range(len(values)):
        sorted.append(values[i])
    for i in range(1, len(sorted)):
        var current = sorted[i]
        var j = i
        while j > 0 and sorted[j - 1] > current:
            sorted[j] = sorted[j - 1]
            j -= 1
        sorted[j] = current
    var idx = (len(values) * pct) // 100
    if idx >= len(sorted):
        idx = len(sorted) - 1
    return sorted[idx]


def deterministic_value(seed: Int, dim_idx: Int) -> Float32:
    var raw = ((seed + 17) * (dim_idx + 31) * 1103515245 + 12345) % 1_000_003
    return (Float32(raw) / 500_001.5) - 1.0


def fill_vectors[
    dim: Int
](mut vectors: List[Float32], seed: Int, vector_count: Int):
    vectors.clear()
    vectors.reserve(vector_count * dim)
    for vector_idx in range(vector_count):
        var base = seed * 131 + vector_idx * 17
        for dim_idx in range(dim):
            vectors.append(deterministic_value(base, dim_idx))


def record_text(record_idx: Int) -> String:
    var module = record_idx % 16
    var kind = record_idx % 7
    return (
        "module_"
        + String(module)
        + " symbol_"
        + String(kind)
        + " async function memory graph retrieval"
    )


def query_text(query_idx: Int) -> String:
    return "module_" + String(query_idx % 16) + " async retrieval"


def build_store[
    dim: Int
](record_count: Int, vectors_per_record: Int) raises -> MultiVectorExactStore[
    dim
]:
    var store = MultiVectorExactStore[dim]()
    var vectors = List[Float32]()
    for i in range(record_count):
        fill_vectors[dim](vectors, i, vectors_per_record)
        _ = store.set_vectors_text(
            "doc-" + String(i),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vectors_per_record,
            record_text(i),
        )
    return store^


def build_muvera_fde_index[
    dim: Int, partition_count: Int, repetition_count: Int
](
    record_count: Int, vectors_per_record: Int, M: Int, ef_construction: Int
) raises -> MultiVectorFDEIndex[dim, partition_count, repetition_count]:
    var index = MultiVectorFDEIndex[dim, partition_count, repetition_count](
        M=M, ef_construction=ef_construction
    )
    index.reserve(record_count)
    var vectors = List[Float32]()
    for i in range(record_count):
        fill_vectors[dim](vectors, i, vectors_per_record)
        index.insert(
            UInt64(i),
            Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
            vectors_per_record,
        )
    return index^


def result_checksum(
    ref results: List[MultiVectorResult], query_idx: Int
) -> Int:
    var checksum = 0
    for j in range(len(results)):
        checksum += (Int(results[j].doc_id) + 1) * (j + 1) * (query_idx + 1)
    return checksum


def overlap_at_k(
    ref oracle: List[MultiVectorResult],
    ref results: List[MultiVectorResult],
    k: Int,
) -> Int:
    var hits = 0
    var limit = min(k, len(oracle))
    for i in range(limit):
        for j in range(min(k, len(results))):
            if oracle[i].doc_id == results[j].doc_id:
                hits += 1
                break
    return hits


def bench_exact_vectors[
    dim: Int
](
    ref store: MultiVectorExactStore[dim],
    query_count: Int,
    query_vectors_per_record: Int,
    k: Int,
) raises:
    var query = List[Float32]()
    var latencies = List[Float64]()
    var checksum = 0
    var total_ms: Float64 = 0.0

    for i in range(query_count):
        fill_vectors[dim](query, 10_000 + i, query_vectors_per_record)
        var start = Int(perf_counter_ns())
        var results = store.search_vectors(
            Span(ptr=query.unsafe_ptr(), length=len(query)),
            query_vectors_per_record,
            k=k,
        )
        var end = Int(perf_counter_ns())
        var ms = elapsed_ms(start, end)
        total_ms += ms
        latencies.append(ms)
        checksum += result_checksum(results, i)

    var qps = Float64(query_count) / (total_ms / 1000.0)
    print(
        "BENCH_MV_SEARCH schema_version=1 mode=exact_maxsim"
        + " query_count="
        + String(query_count)
        + " query_vectors="
        + String(query_vectors_per_record)
        + " k="
        + String(k)
        + " total_ms="
        + String(total_ms)
        + " qps="
        + String(qps)
        + " avg_ms="
        + String(mean(latencies))
        + " p50_ms="
        + String(percentile(latencies, 50))
        + " p95_ms="
        + String(percentile(latencies, 95))
        + " p99_ms="
        + String(percentile(latencies, 99))
        + " oracle_checksum="
        + String(checksum)
    )


def bench_muvera_hnsw_vectors[
    dim: Int, partition_count: Int, repetition_count: Int
](
    ref store: MultiVectorExactStore[dim],
    ref candidate_index: MultiVectorFDEIndex[
        dim, partition_count, repetition_count
    ],
    query_count: Int,
    query_vectors_per_record: Int,
    k: Int,
    candidate_k: Int,
    ef_search: Int,
) raises:
    var query = List[Float32]()
    var latencies = List[Float64]()
    var oracle_checksum = 0
    var result_checksum_total = 0
    var total_ms: Float64 = 0.0
    var encode_total_ms: Float64 = 0.0
    var hnsw_total_ms: Float64 = 0.0
    var rerank_total_ms: Float64 = 0.0
    var recall_hits = 0
    var recall_total = 0
    var candidate_total = 0

    for i in range(query_count):
        fill_vectors[dim](query, 10_000 + i, query_vectors_per_record)
        var query_span = Span(ptr=query.unsafe_ptr(), length=len(query))
        var oracle = store.search_vectors(
            query_span, query_vectors_per_record, k=k
        )

        var start = Int(perf_counter_ns())
        var encoded_query = MultiVectorFDEIndex[
            dim, partition_count, repetition_count
        ].encode_vectors(query_span, query_vectors_per_record)
        var encode_end = Int(perf_counter_ns())
        var candidate_doc_ids = candidate_index.search_encoded(
            Span(ptr=encoded_query.unsafe_ptr(), length=len(encoded_query)),
            candidate_k=candidate_k,
            ef_search=ef_search,
        )
        var hnsw_end = Int(perf_counter_ns())
        var results = store.search_vectors_from_doc_ids(
            query_span,
            query_vectors_per_record,
            Span(
                ptr=candidate_doc_ids.unsafe_ptr(),
                length=len(candidate_doc_ids),
            ),
            k=k,
        )
        var end = Int(perf_counter_ns())

        var ms = elapsed_ms(start, end)
        encode_total_ms += elapsed_ms(start, encode_end)
        hnsw_total_ms += elapsed_ms(encode_end, hnsw_end)
        rerank_total_ms += elapsed_ms(hnsw_end, end)
        total_ms += ms
        latencies.append(ms)
        oracle_checksum += result_checksum(oracle, i)
        result_checksum_total += result_checksum(results, i)
        recall_hits += overlap_at_k(oracle, results, k)
        recall_total += min(k, len(oracle))
        candidate_total += len(candidate_doc_ids)

    var qps = Float64(query_count) / (total_ms / 1000.0)
    var recall = Float64(recall_hits) / Float64(recall_total)
    print(
        "BENCH_MV_SEARCH schema_version=1 mode=muvera_fde_hnsw_exact_rerank"
        + " query_count="
        + String(query_count)
        + " query_vectors="
        + String(query_vectors_per_record)
        + " k="
        + String(k)
        + " fde_dim="
        + String(
            MultiVectorFDEIndex[dim, partition_count, repetition_count].FDE_DIM
        )
        + " partition_count="
        + String(partition_count)
        + " repetition_count="
        + String(repetition_count)
        + " candidate_k="
        + String(candidate_k)
        + " ef_search="
        + String(ef_search)
        + " avg_candidates="
        + String(Float64(candidate_total) / Float64(query_count))
        + " recall_at_"
        + String(k)
        + "_vs_exact="
        + String(recall)
        + " encode_total_ms="
        + String(encode_total_ms)
        + " hnsw_total_ms="
        + String(hnsw_total_ms)
        + " rerank_total_ms="
        + String(rerank_total_ms)
        + " total_ms="
        + String(total_ms)
        + " qps="
        + String(qps)
        + " avg_encode_ms="
        + String(encode_total_ms / Float64(query_count))
        + " avg_hnsw_ms="
        + String(hnsw_total_ms / Float64(query_count))
        + " avg_rerank_ms="
        + String(rerank_total_ms / Float64(query_count))
        + " avg_ms="
        + String(mean(latencies))
        + " p50_ms="
        + String(percentile(latencies, 50))
        + " p95_ms="
        + String(percentile(latencies, 95))
        + " p99_ms="
        + String(percentile(latencies, 99))
        + " oracle_checksum="
        + String(oracle_checksum)
        + " result_checksum="
        + String(result_checksum_total)
    )


def bench_hybrid[
    dim: Int
](
    ref store: MultiVectorExactStore[dim],
    query_count: Int,
    query_vectors_per_record: Int,
    k: Int,
    text_candidate_k: Int,
) raises:
    var query = List[Float32]()
    var latencies = List[Float64]()
    var checksum = 0
    var total_ms: Float64 = 0.0

    for i in range(query_count):
        fill_vectors[dim](query, 20_000 + i, query_vectors_per_record)
        var start = Int(perf_counter_ns())
        var results = store.search_hybrid_vectors(
            query_text(i),
            Span(ptr=query.unsafe_ptr(), length=len(query)),
            query_vectors_per_record,
            k=k,
            text_candidate_k=text_candidate_k,
        )
        var end = Int(perf_counter_ns())
        var ms = elapsed_ms(start, end)
        total_ms += ms
        latencies.append(ms)
        checksum += result_checksum(results, i)

    var qps = Float64(query_count) / (total_ms / 1000.0)
    print(
        "BENCH_MV_SEARCH schema_version=1 mode=bm25_candidates_exact_maxsim"
        + " query_count="
        + String(query_count)
        + " query_vectors="
        + String(query_vectors_per_record)
        + " k="
        + String(k)
        + " text_candidate_k="
        + String(text_candidate_k)
        + " total_ms="
        + String(total_ms)
        + " qps="
        + String(qps)
        + " avg_ms="
        + String(mean(latencies))
        + " p50_ms="
        + String(percentile(latencies, 50))
        + " p95_ms="
        + String(percentile(latencies, 95))
        + " p99_ms="
        + String(percentile(latencies, 99))
        + " oracle_checksum="
        + String(checksum)
    )


def main() raises:
    comptime dim = 48
    var record_count = 2_000
    var vectors_per_record = 16
    var query_count = 50
    var query_vectors = 8
    var k = 10
    var text_candidate_k = 200
    comptime muvera_partition_count = 2
    comptime muvera_repetition_count = 2
    var muvera_candidate_k = 500
    var muvera_ef_search = 100
    var muvera_M = 16
    var muvera_ef_construction = 100

    print("OmenDB-Mojo multi-vector exact and MuVERA candidate baseline")
    print(
        "BENCH_MV_META schema_version=1"
        + " benchmark=multivector_omengrep_shape"
        + " dim="
        + String(dim)
        + " record_count="
        + String(record_count)
        + " vectors_per_record="
        + String(vectors_per_record)
        + " query_count="
        + String(query_count)
        + " query_vectors="
        + String(query_vectors)
        + " k="
        + String(k)
        + " text_candidate_k="
        + String(text_candidate_k)
        + " muvera_partition_count="
        + String(muvera_partition_count)
        + " muvera_repetition_count="
        + String(muvera_repetition_count)
        + " muvera_candidate_k="
        + String(muvera_candidate_k)
        + " muvera_ef_search="
        + String(muvera_ef_search)
        + " oracle=exact_maxsim"
    )

    var build_start = Int(perf_counter_ns())
    var store = build_store[dim](record_count, vectors_per_record)
    var build_end = Int(perf_counter_ns())
    var build_ms = elapsed_ms(build_start, build_end)
    var vector_bytes = record_count * vectors_per_record * dim * 4
    print(
        "BENCH_MV_BUILD schema_version=1"
        + " mode=exact_maxsim"
        + " build_ms="
        + String(build_ms)
        + " records_per_s="
        + String(Float64(record_count) / (build_ms / 1000.0))
        + " vector_bytes="
        + String(vector_bytes)
        + " bytes_per_record_vectors="
        + String(vector_bytes // record_count)
    )

    var muvera_build_start = Int(perf_counter_ns())
    var candidate_index = build_muvera_fde_index[
        dim, muvera_partition_count, muvera_repetition_count
    ](
        record_count,
        vectors_per_record,
        muvera_M,
        muvera_ef_construction,
    )
    var muvera_build_end = Int(perf_counter_ns())
    var muvera_build_ms = elapsed_ms(muvera_build_start, muvera_build_end)
    var fde_vector_bytes = candidate_index.encoded_float_count() * 4
    var fde_code_bytes = candidate_index.encoded_code_count()
    var hnsw_graph_u32_bytes = candidate_index.graph_u32_count() * 4
    print(
        "BENCH_MV_BUILD schema_version=1"
        + " mode=muvera_fde_hnsw"
        + " build_ms="
        + String(muvera_build_ms)
        + " records_per_s="
        + String(Float64(record_count) / (muvera_build_ms / 1000.0))
        + " fde_dim="
        + String(
            MultiVectorFDEIndex[
                dim, muvera_partition_count, muvera_repetition_count
            ].FDE_DIM
        )
        + " partition_count="
        + String(muvera_partition_count)
        + " repetition_count="
        + String(muvera_repetition_count)
        + " M="
        + String(muvera_M)
        + " ef_construction="
        + String(muvera_ef_construction)
        + " fde_vector_bytes="
        + String(fde_vector_bytes)
        + " fde_code_bytes="
        + String(fde_code_bytes)
        + " hnsw_graph_u32_bytes="
        + String(hnsw_graph_u32_bytes)
        + " bytes_per_record_fde="
        + String(fde_vector_bytes // record_count)
    )

    bench_exact_vectors[dim](store, query_count, query_vectors, k)
    bench_muvera_hnsw_vectors[
        dim, muvera_partition_count, muvera_repetition_count
    ](
        store,
        candidate_index,
        query_count,
        query_vectors,
        k,
        muvera_candidate_k,
        muvera_ef_search,
    )
    bench_hybrid[dim](store, query_count, query_vectors, k, text_candidate_k)
