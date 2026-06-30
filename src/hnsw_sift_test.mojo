from hnsw import HNSWIndex, Candidate
from persistence import save_hnsw, load_hnsw
from std.python import Python
from std.testing import assert_true
from std.math import min


def load_sift_f32(path: String, num: Int, dim: Int) raises -> List[Float32]:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    var data = List[Float32]()
    data.reserve(num * dim)
    var ptr = bytes.unsafe_ptr().bitcast[Float32]()
    for i in range(num * dim):
        data.append(ptr[i])
    return data^


def test_hnsw_sift() raises:
    var num_base = 10000
    var dim = 128
    print("Loading SIFT-10K base vectors...")
    var base_data = load_sift_f32("data/siftsmall_base.f32bin", num_base, dim)

    var M = 16
    var ef_construction = 200
    var alpha: Float32 = 1.0
    var idx = HNSWIndex[128](
        M=M,
        ef_construction=ef_construction,
        min_val=0.0,
        max_val=255.0,
        alpha=alpha,
    )
    idx.reserve(num_base)

    print(
        "Building HNSW index (M=",
        M,
        " ef_construction=",
        ef_construction,
        " alpha=",
        alpha,
        ")...",
    )
    for i in range(num_base):
        if i % 1000 == 0:
            print("  inserting vector", i, "/", num_base)
        var vec = Span(ptr=base_data.unsafe_ptr() + i * dim, length=dim)
        idx.insert(vec)

    print(
        "Index built. num_elements=",
        idx.num_elements,
        " max_level=",
        idx.max_level,
    )

    var num_query = 100
    var query_data = load_sift_f32(
        "data/siftsmall_query.f32bin", num_query, dim
    )

    var gt_f = open("data/siftsmall_groundtruth.u32bin", "r")
    var gt_bytes = gt_f.read_bytes()
    var gt_ptr = gt_bytes.unsafe_ptr().bitcast[UInt32]()

    var k = 10
    var ef_search_values = [50, 100, 200, 400]

    for q_idx in range(len(ef_search_values)):
        var ef_search = ef_search_values[q_idx]
        var recall_sum: Float32 = 0.0

        for i in range(num_query):
            var q = Span(ptr=query_data.unsafe_ptr() + i * dim, length=dim)
            var results = idx.search(q, k=k, ef_search=ef_search)

            var gt_start = i * 100
            var hits = 0
            for j in range(len(results)):
                var pred_id = results[j].id
                for x in range(k):
                    if pred_id == gt_ptr[gt_start + x]:
                        hits += 1
                        break
            recall_sum += Float32(hits) / Float32(k)

        var recall = recall_sum / Float32(num_query)
        print("  ef_search=", ef_search, " recall@10=", recall)

    print("Recall gate: checking ef_search=200 >= 0.98...")
    var recall_200: Float32 = 0.0
    for i in range(num_query):
        var q = Span(ptr=query_data.unsafe_ptr() + i * dim, length=dim)
        var results = idx.search(q, k=k, ef_search=200)
        var gt_start = i * 100
        var hits = 0
        for j in range(len(results)):
            var pred_id = results[j].id
            for x in range(k):
                if pred_id == gt_ptr[gt_start + x]:
                    hits += 1
                    break
        recall_200 += Float32(hits) / Float32(k)
    recall_200 /= Float32(num_query)
    print("Final recall@10 (ef=200):", recall_200)
    assert_true(recall_200 >= 0.98)
    print("PASS: recall gate >= 0.98")

    print("Saving and reloading HNSW index for persistence recall gate...")
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var reload_dir = "test_hnsw_sift_reload"
    if os.path.exists(reload_dir):
        shutil.rmtree(reload_dir)
    os.makedirs(reload_dir)
    var reload_prefix = reload_dir + "/hnsw"
    save_hnsw(reload_prefix, idx)
    var loaded_idx = load_hnsw[128](reload_prefix)

    var reload_recall_200: Float32 = 0.0
    for i in range(num_query):
        var q = Span(ptr=query_data.unsafe_ptr() + i * dim, length=dim)
        var results = loaded_idx.search(q, k=k, ef_search=200)
        var gt_start = i * 100
        var hits = 0
        for j in range(len(results)):
            var pred_id = results[j].id
            for x in range(k):
                if pred_id == gt_ptr[gt_start + x]:
                    hits += 1
                    break
        reload_recall_200 += Float32(hits) / Float32(k)
    reload_recall_200 /= Float32(num_query)
    print("Reload recall@10 (ef=200):", reload_recall_200)
    assert_true(reload_recall_200 >= 0.98)
    shutil.rmtree(reload_dir)
    print("PASS: reload recall gate >= 0.98")


def main() raises:
    test_hnsw_sift()
