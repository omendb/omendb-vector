from std.testing import assert_equal, assert_true
from std.sys import argv
from hnsw import HNSWIndex, HNSWSearchScratch, MinHeap, MaxHeap, Candidate


def test_heaps() raises:
    var h = MinHeap()
    h.push(Candidate(0, 3.0))
    h.push(Candidate(1, 1.0))
    h.push(Candidate(2, 2.0))
    assert_equal(h.len(), 3)
    var p = h.peek()
    assert_equal(p.id, 1)
    var c = h.pop()
    assert_equal(c.id, 1)
    c = h.pop()
    assert_equal(c.id, 2)
    c = h.pop()
    assert_equal(c.id, 0)

    var mh = MaxHeap()
    mh.push(Candidate(0, 1.0))
    mh.push(Candidate(1, 3.0))
    mh.push(Candidate(2, 2.0))
    assert_true(mh.worst_distance() > 2.99)
    mh.push_bounded(Candidate(3, 0.5), 2)
    assert_equal(mh.len(), 3)


def test_hnsw_5pt() raises:
    var idx = HNSWIndex[2](M=4, ef_construction=50, min_val=0.0, max_val=1.0)
    var v0: List[Float32] = [0.0, 0.0]
    var v1: List[Float32] = [1.0, 0.0]
    var v2: List[Float32] = [0.0, 1.0]
    var v3: List[Float32] = [1.0, 1.0]
    var v4: List[Float32] = [0.5, 0.5]
    idx.insert(Span(ptr=v0.unsafe_ptr(), length=2))
    idx.insert(Span(ptr=v1.unsafe_ptr(), length=2))
    idx.insert(Span(ptr=v2.unsafe_ptr(), length=2))
    idx.insert(Span(ptr=v3.unsafe_ptr(), length=2))
    idx.insert(Span(ptr=v4.unsafe_ptr(), length=2))
    assert_equal(idx.num_elements, 5)

    var query: List[Float32] = [0.1, 0.1]
    var q = Span(ptr=query.unsafe_ptr(), length=2)
    var results = idx.search(q, k=3, ef_search=10)
    assert_true(len(results) >= 1)
    assert_equal(results[0].id, 0)


def test_hnsw_100pt() raises:
    var idx = HNSWIndex[2](M=4, ef_construction=50, min_val=0.0, max_val=1.0)
    for i in range(100):
        var fi = Float32(i)
        var vec: List[Float32] = [fi * 0.01, fi * 0.01]
        idx.insert(Span(ptr=vec.unsafe_ptr(), length=2))
    assert_equal(idx.num_elements, 100)

    var query: List[Float32] = [0.5, 0.5]
    var q = Span(ptr=query.unsafe_ptr(), length=2)
    var results = idx.search(q, k=5, ef_search=50)
    assert_true(len(results) >= 3)
    print("  closest result id=", results[0].id, " dist=", results[0].distance)


def test_hnsw_1000pt() raises:
    var idx = HNSWIndex[2](M=16, ef_construction=200, min_val=0.0, max_val=1.0)
    for i in range(1000):
        var fi = Float32(i)
        var vec: List[Float32] = [fi * 0.001, fi * 0.001]
        idx.insert(Span(ptr=vec.unsafe_ptr(), length=2))
    assert_equal(idx.num_elements, 1000)

    var query: List[Float32] = [0.5, 0.5]
    var q = Span(ptr=query.unsafe_ptr(), length=2)
    var results = idx.search(q, k=5, ef_search=100)
    assert_true(len(results) >= 3)
    print("PASS test_hnsw_1000pt")


def test_add_neighbor_full_row_does_not_overflow() raises:
    var idx = HNSWIndex[2](M=2, ef_construction=10)

    idx.graph.ensure_node_storage(UInt32(1), 0, idx.M0, idx.M)

    # Layer 0 has M0 slots per node. Fill node 0 exactly to capacity and place
    # sentinels in node 1's row so cross-row writes are visible.
    for i in range(8):
        idx.graph.set_neighbor(0, 0, i, idx.M0, idx.M, UInt32(i))
    idx.graph.set_count(0, 0, 4)
    idx.graph.set_count(1, 0, 0)

    idx._add_neighbor(0, 0, UInt32(99))

    assert_equal(idx.graph.count(0, 0), 4)
    assert_equal(idx.graph.neighbor(1, 0, 0, idx.M0, idx.M), UInt32(4))
    assert_equal(idx.graph.count(1, 0), 0)


def test_hnsw_search_edge_cases() raises:
    var empty = HNSWIndex[2](M=4, ef_construction=20, min_val=0.0, max_val=1.0)
    var q_empty: List[Float32] = [0.0, 0.0]
    var empty_results = empty.search(
        Span(ptr=q_empty.unsafe_ptr(), length=2), k=10, ef_search=1
    )
    assert_equal(len(empty_results), 0)

    var small = HNSWIndex[2](M=4, ef_construction=20, min_val=0.0, max_val=1.0)
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 1.0]
    small.insert(Span(ptr=a.unsafe_ptr(), length=2))
    small.insert(Span(ptr=b.unsafe_ptr(), length=2))
    var small_results = small.search(
        Span(ptr=a.unsafe_ptr(), length=2), k=10, ef_search=1
    )
    assert_true(len(small_results) <= 2)
    assert_true(len(small_results) > 0)
    assert_equal(small_results[0].id, 0)

    var dup = HNSWIndex[2](M=4, ef_construction=20, min_val=0.0, max_val=1.0)
    var d0: List[Float32] = [0.25, 0.25]
    var d1: List[Float32] = [0.25, 0.25]
    var d2: List[Float32] = [0.25, 0.25]
    dup.insert(Span(ptr=d0.unsafe_ptr(), length=2))
    dup.insert(Span(ptr=d1.unsafe_ptr(), length=2))
    dup.insert(Span(ptr=d2.unsafe_ptr(), length=2))
    var dup_results = dup.search(
        Span(ptr=d0.unsafe_ptr(), length=2), k=5, ef_search=5
    )
    assert_true(len(dup_results) <= 3)
    assert_true(len(dup_results) > 0)
    assert_equal(dup_results[0].distance, 0.0)

    var idx4 = HNSWIndex[4](M=4, ef_construction=20, min_val=0.0, max_val=1.0)
    var x0: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    var x1: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    idx4.insert(Span(ptr=x0.unsafe_ptr(), length=4))
    idx4.insert(Span(ptr=x1.unsafe_ptr(), length=4))
    var results4 = idx4.search(
        Span(ptr=x0.unsafe_ptr(), length=4), k=2, ef_search=2
    )
    assert_true(len(results4) > 0)
    assert_equal(results4[0].id, 0)


def test_hnsw_search_with_scratch_matches_search() raises:
    var idx = HNSWIndex[2](M=4, ef_construction=50, min_val=0.0, max_val=1.0)
    for i in range(128):
        var x = Float32(i % 16) / 16.0
        var y = Float32(i // 16) / 8.0
        var vec: List[Float32] = [x, y]
        idx.insert(Span(ptr=vec.unsafe_ptr(), length=2))

    var scratch = HNSWSearchScratch[2](idx.quantizer)
    var q0_data: List[Float32] = [0.25, 0.5]
    var q0 = Span(ptr=q0_data.unsafe_ptr(), length=2)
    var baseline = idx.search(q0, k=5, ef_search=40)
    var scratched = idx.search_with_scratch(q0, scratch, k=5, ef_search=40)
    assert_equal(len(scratched), len(baseline))
    for i in range(len(baseline)):
        assert_equal(scratched[i].id, baseline[i].id)
        assert_equal(scratched[i].distance, baseline[i].distance)

    var q1_data: List[Float32] = [0.75, 0.125]
    var q1 = Span(ptr=q1_data.unsafe_ptr(), length=2)
    baseline = idx.search(q1, k=5, ef_search=40)
    scratched = idx.search_with_scratch(q1, scratch, k=5, ef_search=40)
    assert_equal(len(scratched), len(baseline))
    for i in range(len(baseline)):
        assert_equal(scratched[i].id, baseline[i].id)
        assert_equal(scratched[i].distance, baseline[i].distance)


def test_hnsw_filtered_search_returns_only_allowed() raises:
    var idx = HNSWIndex[2](M=4, ef_construction=50, min_val=0.0, max_val=1.0)
    for i in range(128):
        var x = Float32(i % 16) / 16.0
        var y = Float32(i // 16) / 8.0
        var vec: List[Float32] = [x, y]
        idx.insert(Span(ptr=vec.unsafe_ptr(), length=2))

    var allowed = List[UInt8]()
    for i in range(128):
        if i % 2 == 0:
            allowed.append(1)
        else:
            allowed.append(0)

    var q_data: List[Float32] = [0.5, 0.5]
    var q = Span(ptr=q_data.unsafe_ptr(), length=2)
    var results = idx.search_filtered(q, allowed, k=5, ef_search=64)
    assert_true(len(results) > 0)
    for i in range(len(results)):
        assert_equal(Int(results[i].id) % 2, 0)


def test_hnsw_graph_invariants() raises:
    var idx = HNSWIndex[2](M=4, ef_construction=50, min_val=0.0, max_val=1.0)

    for i in range(64):
        var original = (i * 37) % 64
        var x = Float32(original % 8) / 8.0
        var y = Float32(original // 8) / 8.0
        var vec: List[Float32] = [x, y]
        idx.insert(Span(ptr=vec.unsafe_ptr(), length=2))

    assert_equal(idx.num_elements, 64)
    assert_equal(idx.graph.layer_count(), idx.max_level + 1)

    for layer in range(idx.graph.layer_count()):
        var max_conn = idx.M0 if layer == 0 else idx.M
        assert_true(
            idx.graph.slot_capacity(layer, idx.M0, idx.M)
            >= idx.graph.node_count(layer) * max_conn
        )
        assert_true(idx.graph.node_count(layer) <= idx.num_elements)
        assert_equal(
            idx.graph.row_lock_count(layer), idx.graph.node_count(layer)
        )
        if idx.graph.node_count(layer) > 0:
            var owner = 1000 + layer
            idx.graph.lock_row(0, layer, owner)
            assert_true(idx.graph.unlock_row(0, layer, owner))

        for node in range(idx.graph.node_count(layer)):
            var count = idx.graph.count(node, layer)
            assert_true(count <= max_conn)

            for i in range(count):
                var nbr = idx.graph.neighbor(node, layer, i, idx.M0, idx.M)
                assert_true(nbr < UInt32(idx.num_elements))
                assert_true(nbr != UInt32(node))

                for j in range(i + 1, count):
                    assert_true(
                        nbr != idx.graph.neighbor(node, layer, j, idx.M0, idx.M)
                    )


def test_hnsw_insert_batch_serial() raises:
    var idx = HNSWIndex[2](M=4, ef_construction=50, min_val=0.0, max_val=1.0)
    var data = List[Float32]()
    for i in range(64):
        var x = Float32(i % 8) / 8.0
        var y = Float32(i // 8) / 8.0
        data.append(x)
        data.append(y)

    idx.insert_batch_serial(Span(ptr=data.unsafe_ptr(), length=len(data)), 64)
    assert_equal(idx.num_elements, 64)
    assert_equal(idx.graph.layer_count(), idx.max_level + 1)

    var query: List[Float32] = [0.25, 0.25]
    var results = idx.search(
        Span(ptr=query.unsafe_ptr(), length=2), k=5, ef_search=40
    )
    assert_true(len(results) > 0)
    assert_equal(idx.ef_construction, 50)

    var idx_ef2 = HNSWIndex[2](
        M=4, ef_construction=50, min_val=0.0, max_val=1.0
    )
    idx_ef2.insert_batch_serial(
        Span(ptr=data.unsafe_ptr(), length=len(data)), 64, ef_multiplier=2
    )
    assert_equal(idx_ef2.num_elements, 64)
    assert_equal(idx_ef2.ef_construction, 50)


def main() raises:
    test_heaps()
    print("PASS test_heaps")

    var args = argv()
    if len(args) > 1:
        var test_name = String(args[1])
        if test_name == "5pt":
            test_hnsw_5pt()
            print("PASS test_hnsw_5pt")
        elif test_name == "100pt":
            test_hnsw_100pt()
            print("PASS test_hnsw_100pt")
        elif test_name == "1000pt":
            test_hnsw_1000pt()
        elif test_name == "neighbor_overflow":
            test_add_neighbor_full_row_does_not_overflow()
            print("PASS test_add_neighbor_full_row_does_not_overflow")
        elif test_name == "edge_cases":
            test_hnsw_search_edge_cases()
            print("PASS test_hnsw_search_edge_cases")
        elif test_name == "invariants":
            test_hnsw_graph_invariants()
            print("PASS test_hnsw_graph_invariants")
        elif test_name == "scratch":
            test_hnsw_search_with_scratch_matches_search()
            print("PASS test_hnsw_search_with_scratch_matches_search")
        elif test_name == "filtered":
            test_hnsw_filtered_search_returns_only_allowed()
            print("PASS test_hnsw_filtered_search_returns_only_allowed")
        elif test_name == "batch_serial":
            test_hnsw_insert_batch_serial()
            print("PASS test_hnsw_insert_batch_serial")
    else:
        test_hnsw_5pt()
        print("PASS test_hnsw_5pt")
        test_hnsw_100pt()
        print("PASS test_hnsw_100pt")
        test_hnsw_1000pt()
        test_add_neighbor_full_row_does_not_overflow()
        print("PASS test_add_neighbor_full_row_does_not_overflow")
        test_hnsw_search_edge_cases()
        print("PASS test_hnsw_search_edge_cases")
        test_hnsw_search_with_scratch_matches_search()
        print("PASS test_hnsw_search_with_scratch_matches_search")
        test_hnsw_filtered_search_returns_only_allowed()
        print("PASS test_hnsw_filtered_search_returns_only_allowed")
        test_hnsw_graph_invariants()
        print("PASS test_hnsw_graph_invariants")
        test_hnsw_insert_batch_serial()
        print("PASS test_hnsw_insert_batch_serial")
    print("All tests passed!")
