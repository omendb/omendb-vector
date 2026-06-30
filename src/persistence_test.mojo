from std.testing import assert_equal
from persistence import save_hnsw, load_hnsw
from hnsw import HNSWIndex


def test_hnsw_persistence() raises:
    var dim = 4
    var idx = HNSWIndex[4](M=16, ef_construction=200)

    var vec: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    idx.insert(Span(ptr=vec.unsafe_ptr(), length=dim))

    var path = "test_hnsw"
    save_hnsw(path, idx)

    var loaded_idx = load_hnsw[4](path)
    assert_equal(loaded_idx.num_elements, 1)
    assert_equal(loaded_idx.max_level, idx.max_level)

    var results = loaded_idx.search(Span(ptr=vec.unsafe_ptr(), length=dim), k=1)
    assert_equal(len(results), 1)
    assert_equal(results[0].id, 0)


def main() raises:
    test_hnsw_persistence()
    print("PASS test_hnsw_persistence")
