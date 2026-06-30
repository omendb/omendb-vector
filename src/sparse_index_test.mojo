from std.testing import assert_equal
from sparse_index import SparseInvertedIndex, SparseVector


def test_sparse_search_returns_doc_zero() raises:
    var index = SparseInvertedIndex()

    var doc0 = SparseVector()
    doc0.dims.append(UInt32(7))
    doc0.weights.append(10.0)
    index.insert(doc0, UInt32(0))

    var doc1 = SparseVector()
    doc1.dims.append(UInt32(7))
    doc1.weights.append(1.0)
    index.insert(doc1, UInt32(1))

    var query = SparseVector()
    query.dims.append(UInt32(7))
    query.weights.append(1.0)

    var results = index.search(query, 2)
    assert_equal(len(results), 2)
    assert_equal(results[0].doc_id, UInt32(0))
    assert_equal(results[1].doc_id, UInt32(1))


def main() raises:
    test_sparse_search_returns_doc_zero()
    print("PASS test_sparse_search_returns_doc_zero")
