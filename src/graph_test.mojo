from std.testing import assert_equal, assert_true
from graph import GraphDirection, PropertyGraph, Node, Edge


def test_graph_basic() raises:
    var g = PropertyGraph()

    # Add nodes
    var n0 = g.add_node("Person")
    var n1 = g.add_node("Person")
    var n2 = g.add_node("Person")

    assert_equal(n0, 0)
    assert_equal(n1, 1)
    assert_equal(n2, 2)
    assert_equal(len(g.nodes), 3)

    # Add edges
    _ = g.add_edge(n0, n1, "KNOWS")
    _ = g.add_edge(n1, n2, "KNOWS")

    assert_equal(len(g.edges), 2)

    # Check CSR
    var n0_nbrs = g.index.out_neighbors_of(n0)
    assert_equal(len(n0_nbrs), 1)
    assert_equal(n0_nbrs[0], n1)

    var n1_nbrs = g.index.out_neighbors_of(n1)
    assert_equal(len(n1_nbrs), 1)
    assert_equal(n1_nbrs[0], n2)

    var n2_nbrs = g.index.out_neighbors_of(n2)
    assert_equal(len(n2_nbrs), 0)

    # Check incoming
    var n2_in = g.index.in_neighbors_of(n2)
    assert_equal(len(n2_in), 1)
    assert_equal(n2_in[0], n1)


def test_graph_bfs() raises:
    var g = PropertyGraph()
    # Create a star graph
    var center = g.add_node("Center")
    for _ in range(5):
        var leaf = g.add_node("Leaf")
        _ = g.add_edge(center, leaf, "CONNECTS")

    var results = g.bfs(center, max_hops=1)
    assert_equal(len(results), 6)  # center + 5 leaves
    assert_equal(results[0], center)


def test_joint_search() raises:
    from hnsw import HNSWIndex

    var v_index = HNSWIndex[2](M=16, ef_construction=200)
    var g = PropertyGraph()

    # Add nodes and vectors
    # 0 -> [0,0]
    # 1 -> [1,1]
    # 2 -> [2,2]
    # Link 0-1 and 1-2
    for i in range(3):
        var fi = Float32(i)
        var vec: List[Float32] = [fi, fi]
        v_index.insert(Span(ptr=vec.unsafe_ptr(), length=2))
        _ = g.add_node("Point", vector_id=UInt64(i))

    _ = g.add_edge(0, 1, "LINK")
    _ = g.add_edge(1, 2, "LINK")

    # Query near 2 [1.9, 1.9]
    var q: List[Float32] = [1.9, 1.9]
    # With ef=1, search might find 2 or 1.
    var results = g.joint_search[2](
        v_index, Span(ptr=q.unsafe_ptr(), length=2), k=1
    )
    assert_equal(len(results), 1)
    assert_equal(results[0].id, 2)


def test_external_id_mapping() raises:
    from hnsw import HNSWIndex

    var g = PropertyGraph()
    var v_index = HNSWIndex[2](M=16, ef_construction=200)

    var v0: List[Float32] = [0.0, 0.0]
    var v1: List[Float32] = [1.0, 1.0]
    var v2: List[Float32] = [2.0, 2.0]
    v_index.insert(Span(ptr=v0.unsafe_ptr(), length=2))
    v_index.insert(Span(ptr=v1.unsafe_ptr(), length=2))
    v_index.insert(Span(ptr=v2.unsafe_ptr(), length=2))

    var c = g.add_node_with_id("doc-c", "Point", vector_id=UInt64(2))
    var a = g.add_node_with_id("doc-a", "Point", vector_id=UInt64(0))
    var b = g.add_node_with_id("doc-b", "Point", vector_id=UInt64(1))

    assert_equal(c, 0)
    assert_equal(a, 1)
    assert_equal(b, 2)
    assert_equal(g.node_id_for("doc-c").value(), c)
    assert_equal(g.node_id_for_vector(0).value(), a)
    assert_equal(g.node_id_for_vector(2).value(), c)
    assert_equal(g.external_id_for(b).value(), "doc-b")
    assert_true(not g.node_id_for("missing"))

    var duplicate = g.add_node_with_id("doc-c", "Point", vector_id=UInt64(1))
    assert_equal(duplicate, c)
    assert_equal(len(g.nodes), 3)

    _ = g.add_edge(a, b, "LINK")
    _ = g.add_edge(b, c, "LINK")

    var q: List[Float32] = [1.9, 1.9]
    var results = g.joint_search[2](
        v_index, Span(ptr=q.unsafe_ptr(), length=2), k=1
    )
    assert_equal(len(results), 1)
    assert_equal(results[0].id, 2)


def test_edge_validation_and_removal() raises:
    var g = PropertyGraph()
    var a = g.add_node_with_id("doc-a", "Point")
    var b = g.add_node_with_id("doc-b", "Point")
    var c = g.add_node_with_id("doc-c", "Point")

    var first = g.add_edge(a, b, "LINK", weight=Float32(1.5))
    var duplicate = g.add_edge(a, b, "LINK", weight=Float32(9.0))
    assert_equal(duplicate, first)
    assert_equal(len(g.edges), 1)
    assert_true(g.edges[Int(first)].weight)
    assert_equal(g.edges[Int(first)].weight.value(), 1.5)

    var second = g.add_edge(b, c, "LINK")
    assert_equal(second, 1)
    assert_equal(g.edge_id_for(a, b, "LINK").value(), first)
    assert_true(not g.edge_id_for(a, c, "LINK"))

    assert_true(g.remove_edge(a, b, "LINK"))
    assert_equal(len(g.edges), 1)
    assert_equal(g.edges[0].id, 0)
    assert_equal(g.edges[0].src, b)
    assert_equal(g.edges[0].dst, c)
    assert_true(not g.remove_edge(a, b, "LINK"))

    var b_nbrs = g.index.out_neighbors_of(b)
    assert_equal(len(b_nbrs), 1)
    assert_equal(b_nbrs[0], c)
    var a_nbrs = g.index.out_neighbors_of(a)
    assert_equal(len(a_nbrs), 0)

    var failed = False
    try:
        _ = g.add_edge(a, a, "SELF")
    except e:
        failed = True
    assert_true(failed)

    failed = False
    try:
        _ = g.add_edge(a, c, "")
    except e:
        failed = True
    assert_true(failed)

    failed = False
    try:
        _ = g.add_edge(UInt64(999), c, "MISSING")
    except e:
        failed = True
    assert_true(failed)

    failed = False
    try:
        _ = g.remove_edge(a, UInt64(999), "MISSING")
    except e:
        failed = True
    assert_true(failed)


def test_direction_aware_traversal() raises:
    var g = PropertyGraph()
    var a = g.add_node_with_id("a", "Node")
    var b = g.add_node_with_id("b", "Node")
    var c = g.add_node_with_id("c", "Node")
    var d = g.add_node_with_id("d", "Node")

    _ = g.add_edge(a, b, "link")
    _ = g.add_edge(c, b, "ref")
    _ = g.add_edge(b, d, "link")
    _ = g.add_edge(d, a, "link")

    var out_a = g.neighbors(a, GraphDirection.OUTGOING)
    assert_equal(len(out_a), 1)
    assert_equal(out_a[0], b)

    var in_b = g.neighbors(b, GraphDirection.INCOMING)
    assert_equal(len(in_b), 2)
    assert_equal(in_b[0], a)
    assert_equal(in_b[1], c)

    var both_b = g.neighbors(b, GraphDirection.BOTH)
    assert_equal(len(both_b), 3)
    assert_equal(both_b[0], d)
    assert_equal(both_b[1], a)
    assert_equal(both_b[2], c)

    var link_in_b = g.neighbors(
        b, GraphDirection.INCOMING, edge_type=String("link")
    )
    assert_equal(len(link_in_b), 1)
    assert_equal(link_in_b[0], a)

    var missing_filter = g.neighbors(
        b, GraphDirection.OUTGOING, edge_type=String("missing")
    )
    assert_equal(len(missing_filter), 0)

    var depth1 = g.traverse(a, GraphDirection.OUTGOING, max_hops=1)
    assert_equal(len(depth1), 2)
    assert_equal(depth1[0], a)
    assert_equal(depth1[1], b)

    var depth2 = g.traverse(a, GraphDirection.OUTGOING, max_hops=2)
    assert_equal(len(depth2), 3)
    assert_equal(depth2[0], a)
    assert_equal(depth2[1], b)
    assert_equal(depth2[2], d)

    var incoming = g.traverse(d, GraphDirection.INCOMING, max_hops=2)
    assert_equal(len(incoming), 4)
    assert_equal(incoming[0], d)
    assert_equal(incoming[1], b)
    assert_equal(incoming[2], a)
    assert_equal(incoming[3], c)

    var filtered = g.traverse(
        b, GraphDirection.BOTH, max_hops=2, edge_type=String("ref")
    )
    assert_equal(len(filtered), 2)
    assert_equal(filtered[0], b)
    assert_equal(filtered[1], c)

    var zero_depth = g.traverse(a, GraphDirection.BOTH, max_hops=0)
    assert_equal(len(zero_depth), 1)
    assert_equal(zero_depth[0], a)

    var failed = False
    try:
        _ = g.traverse(a, GraphDirection.OUTGOING, max_hops=-1)
    except e:
        failed = True
    assert_true(failed)

    failed = False
    try:
        _ = g.neighbors(UInt64(999), GraphDirection.OUTGOING)
    except e:
        failed = True
    assert_true(failed)


def test_external_id_path_queries() raises:
    var g = PropertyGraph()
    _ = g.add_node_with_id("a", "Node")
    _ = g.add_node_with_id("b", "Node")
    _ = g.add_node_with_id("c", "Node")
    _ = g.add_node_with_id("d", "Node")
    _ = g.add_node_with_id("e", "Node")

    _ = g.add_edge(
        g.node_id_for("a").value(), g.node_id_for("b").value(), "link"
    )
    _ = g.add_edge(
        g.node_id_for("b").value(), g.node_id_for("c").value(), "link"
    )
    _ = g.add_edge(
        g.node_id_for("c").value(), g.node_id_for("d").value(), "link"
    )
    _ = g.add_edge(
        g.node_id_for("d").value(), g.node_id_for("b").value(), "link"
    )
    _ = g.add_edge(
        g.node_id_for("a").value(), g.node_id_for("e").value(), "ref"
    )

    assert_true(g.has_path("a", "d", GraphDirection.OUTGOING, max_depth=3))
    assert_true(not g.has_path("a", "d", GraphDirection.OUTGOING, max_depth=2))
    assert_true(not g.has_path("d", "a", GraphDirection.OUTGOING, max_depth=10))
    assert_true(g.has_path("d", "a", GraphDirection.INCOMING, max_depth=3))
    assert_true(g.has_path("a", "a", GraphDirection.OUTGOING, max_depth=0))
    assert_true(not g.has_path("missing", "a", GraphDirection.OUTGOING))

    var path = g.shortest_path("a", "d", GraphDirection.OUTGOING, max_depth=10)
    assert_equal(len(path), 4)
    assert_equal(path[0], "a")
    assert_equal(path[1], "b")
    assert_equal(path[2], "c")
    assert_equal(path[3], "d")

    var ref_path = g.shortest_path(
        "a", "e", GraphDirection.OUTGOING, max_depth=1, edge_type=String("ref")
    )
    assert_equal(len(ref_path), 2)
    assert_equal(ref_path[0], "a")
    assert_equal(ref_path[1], "e")

    var filtered_out = g.shortest_path(
        "a", "e", GraphDirection.OUTGOING, max_depth=1, edge_type=String("link")
    )
    assert_equal(len(filtered_out), 0)

    var self_path = g.shortest_path("a", "a", GraphDirection.BOTH, max_depth=0)
    assert_equal(len(self_path), 1)
    assert_equal(self_path[0], "a")

    var failed = False
    try:
        _ = g.shortest_path("a", "d", GraphDirection.OUTGOING, max_depth=-1)
    except e:
        failed = True
    assert_true(failed)


def main() raises:
    test_graph_basic()
    print("PASS test_graph_basic")
    test_graph_bfs()
    print("PASS test_graph_bfs")
    test_joint_search()
    print("PASS test_joint_search")
    test_external_id_mapping()
    print("PASS test_external_id_mapping")
    test_edge_validation_and_removal()
    print("PASS test_edge_validation_and_removal")
    test_direction_aware_traversal()
    print("PASS test_direction_aware_traversal")
    test_external_id_path_queries()
    print("PASS test_external_id_path_queries")
