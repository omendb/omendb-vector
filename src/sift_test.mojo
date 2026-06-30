from std.memory import UnsafePointer
from flat_index import FlatIndex
from hnsw import Candidate
from std.testing import assert_true
from std.math import min


def load_sift_base(path: String, num: Int, dim: Int) raises -> List[Float32]:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    var data = List[Float32]()
    data.reserve(num * dim)

    var ptr = bytes.unsafe_ptr().bitcast[Float32]()
    for i in range(num * dim):
        data.append(ptr[i])
    return data^


def test_sift() raises:
    print("Loading siftsmall dataset (List-based)...")
    var num_base = 10000
    var base_data = load_sift_base("data/siftsmall_base.f32bin", num_base, 128)

    print("Base data length:", len(base_data))

    var index = FlatIndex[128]()
    index.reserve(num_base)
    for i in range(num_base):
        if i % 1000 == 0:
            print("Inserting vector", i)
        var vec = Span(ptr=base_data.unsafe_ptr() + i * 128, length=128)
        index.insert(vec)

    print("Inserted all. Testing recall@10...")
    # Load query and groundtruth too
    var num_query = 100
    var query_data = load_sift_base(
        "data/siftsmall_query.f32bin", num_query, 128
    )

    # Load groundtruth
    var gt_f = open("data/siftsmall_groundtruth.u32bin", "r")
    var gt_bytes = gt_f.read_bytes()
    var gt_ptr = gt_bytes.unsafe_ptr().bitcast[UInt32]()

    var recall_10: Float32 = 0.0
    var k = 10

    for i in range(num_query):
        var q = Span(ptr=query_data.unsafe_ptr() + i * 128, length=128)
        var results = index.search(q, k)

        var gt_start = i * 100
        var hits = 0
        for j in range(len(results)):
            var pred_id = results[j].id
            for x in range(k):
                if pred_id == gt_ptr[gt_start + x]:
                    hits += 1
                    break
        recall_10 += Float32(hits) / Float32(k)

    recall_10 /= Float32(num_query)
    print("Recall@10:", recall_10)
    assert_true(recall_10 > 0.99)


def main() raises:
    test_sift()
