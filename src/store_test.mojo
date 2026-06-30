from std.testing import assert_equal, assert_true
from std.python import Python
from graph import GraphDirection
from store import (
    HNSWParams,
    Metric,
    SearchOptions,
    SearchResult,
    VectorStore,
    VectorStoreOptions,
)
from consistency_check import verify_consistency


def test_metric_values() raises:
    assert_true(Metric.L2 == Metric.L2)
    assert_true(Metric.L2 != Metric.COSINE)
    assert_true(Metric.DOT != Metric.COSINE)


def test_hnsw_params_defaults() raises:
    var params = HNSWParams()
    assert_equal(params.M, 16)
    assert_equal(params.ef_construction, 100)
    assert_equal(params.ef_search, 100)
    assert_equal(params.alpha, 1.0)

    var custom = HNSWParams(M=32, ef_construction=400, ef_search=100, alpha=1.1)
    assert_equal(custom.M, 32)
    assert_equal(custom.ef_construction, 400)
    assert_equal(custom.ef_search, 100)
    assert_equal(custom.alpha, 1.1)


def test_search_options_defaults() raises:
    var options = SearchOptions()
    assert_equal(options.k, 10)
    assert_equal(options.ef_search, 100)
    assert_true(not options.max_distance)

    var limited = SearchOptions(k=5, ef_search=64, max_distance=Float32(3.5))
    assert_equal(limited.k, 5)
    assert_equal(limited.ef_search, 64)
    assert_true(limited.max_distance)
    assert_equal(limited.max_distance.value(), 3.5)


def test_vector_store_options_defaults() raises:
    var options = VectorStoreOptions()
    assert_true(options.metric == Metric.L2)
    assert_equal(options.hnsw.M, 16)
    assert_true(not options.text_enabled)
    assert_true(not options.graph_enabled)

    var custom = VectorStoreOptions(
        metric=Metric.COSINE,
        hnsw=HNSWParams(M=24, ef_construction=300, ef_search=80, alpha=1.3),
        text_enabled=True,
        graph_enabled=True,
    )
    assert_true(custom.metric == Metric.COSINE)
    assert_equal(custom.hnsw.M, 24)
    assert_equal(custom.hnsw.ef_search, 80)
    assert_true(custom.text_enabled)
    assert_true(custom.graph_enabled)


def test_search_result_construction() raises:
    var result = SearchResult("doc-1", UInt64(7), 0.25)
    assert_equal(result.id, "doc-1")
    assert_equal(result.vector_id, 7)
    assert_equal(result.distance, 0.25)
    assert_true(not result.metadata)

    var with_metadata = SearchResult(
        "doc-2", UInt64(8), 0.5, metadata=String('{"kind":"note"}')
    )
    assert_equal(with_metadata.id, "doc-2")
    assert_true(with_metadata.metadata)
    assert_equal(with_metadata.metadata.value(), '{"kind":"note"}')


def test_vector_store_set_get_search() raises:
    var store = VectorStore[2].create_in_memory()
    assert_true(not store.is_persistent())
    store.flush()

    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [2.0, 2.0]
    var c: List[Float32] = [1.0, 1.0]

    var a_id = store.set(
        "a", Span(ptr=a.unsafe_ptr(), length=2), metadata=String("first")
    )
    var b_id = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    var c_id = store.set(
        "c", Span(ptr=c.unsafe_ptr(), length=2), metadata=String("middle")
    )
    assert_equal(a_id, 0)
    assert_equal(b_id, 1)
    assert_equal(c_id, 2)
    assert_equal(store.len(), 3)

    var got = store.get("c")
    assert_equal(len(got), 2)
    assert_equal(got[0], 1.0)
    assert_equal(got[1], 1.0)

    var missing = store.get("missing")
    assert_equal(len(missing), 0)

    var q: List[Float32] = [0.1, 0.1]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2), SearchOptions(k=3, ef_search=64)
    )
    assert_equal(len(results), 3)
    assert_equal(results[0].id, "a")
    assert_equal(results[0].vector_id, 0)
    assert_true(results[0].metadata)
    assert_equal(results[0].metadata.value(), "first")
    assert_equal(results[1].id, "c")
    assert_equal(results[2].id, "b")

    var limited = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=3, ef_search=64, max_distance=Float32(0.1)),
    )
    assert_equal(len(limited), 1)
    assert_equal(limited[0].id, "a")

    var empty = store.search(
        Span(ptr=q.unsafe_ptr(), length=2), SearchOptions(k=0)
    )
    assert_equal(len(empty), 0)


def test_vector_store_failure_paths() raises:
    var store = VectorStore[2].create_in_memory()
    var short_vec: List[Float32] = [1.0]
    var vec: List[Float32] = [1.0, 1.0]

    var failed = False
    try:
        _ = store.set("bad", Span(ptr=short_vec.unsafe_ptr(), length=1))
    except e:
        failed = True
    assert_true(failed)

    _ = store.set("ok", Span(ptr=vec.unsafe_ptr(), length=2))

    failed = False
    try:
        _ = store.set("ok", Span(ptr=vec.unsafe_ptr(), length=2))
    except e:
        failed = True
    assert_true(failed)

    failed = False
    try:
        _ = VectorStore[2].create_in_memory(
            VectorStoreOptions(metric=Metric.COSINE)
        )
    except e:
        failed = True
    assert_true(failed)


def test_vector_store_persistence_lifecycle() raises:
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var json = Python.import_module("json")
    var builtins = Python.import_module("builtins")
    var path = "test_vector_store_lifecycle"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    assert_true(store.is_persistent())
    assert_equal(store.len(), 0)

    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 1.0]
    _ = store.set(
        "a", Span(ptr=a.unsafe_ptr(), length=2), metadata=String("alpha")
    )
    _ = store.set(
        "b", Span(ptr=b.unsafe_ptr(), length=2), metadata=String("beta")
    )
    store.flush()

    assert_true(os.path.exists(path + "/manifest.json"))
    assert_true(os.path.exists(path + "/records.json"))
    assert_true(os.path.exists(path + "/tombstones.json"))
    assert_true(os.path.exists(path + "/hnsw.meta"))

    var manifest_file = builtins.open(path + "/manifest.json", "r")
    var manifest = json.loads(manifest_file.read())
    _ = manifest_file.close()
    assert_equal(String(manifest["store_layout"]), "single_segment_v1")
    assert_equal(String(manifest["index_mode"]), "hnsw")
    assert_equal(String(manifest["segment_id"]), "segment0")
    assert_equal(Int(py=manifest["record_count"]), 2)
    assert_equal(Int(py=manifest["tombstone_count"]), 0)
    assert_true(String(manifest["config_fingerprint"]).byte_length() > 0)
    assert_true("records.json" in manifest["checksums"])
    assert_true("hnsw.data" in manifest["checksums"])

    var reopened = VectorStore[2].open(path)
    assert_true(reopened.is_persistent())
    assert_equal(reopened.len(), 2)

    var got = reopened.get("b")
    assert_equal(len(got), 2)
    assert_equal(got[0], 1.0)
    assert_equal(got[1], 1.0)

    var q: List[Float32] = [0.9, 0.9]
    var results = reopened.search(
        Span(ptr=q.unsafe_ptr(), length=2), SearchOptions(k=2)
    )
    assert_equal(len(results), 2)
    assert_equal(results[0].id, "b")
    assert_true(results[0].metadata)
    assert_equal(results[0].metadata.value(), "beta")

    var failed = False
    try:
        _ = VectorStore[2].create(path)
    except e:
        failed = True
    assert_true(failed)

    failed = False
    try:
        _ = VectorStore[3].open(path)
    except e:
        failed = True
    assert_true(failed)

    _ = reopened.set("c", Span(ptr=a.unsafe_ptr(), length=2))
    reopened.flush()
    assert_true(os.path.exists(path + "/manifest.json"))
    assert_true(not os.path.exists(path + ".bak"))

    shutil.rmtree(path)


def test_vector_store_persistence_failures() raises:
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")
    var json = Python.import_module("json")
    var path = "test_vector_store_failures"
    if os.path.exists(path):
        shutil.rmtree(path)

    var failed = False
    try:
        _ = VectorStore[2].open(path)
    except e:
        failed = True
    assert_true(failed)

    var store = VectorStore[2].create(path)
    var vec: List[Float32] = [1.0, 1.0]
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    os.remove(path + "/records.json")
    failed = False
    try:
        _ = VectorStore[2].open(path)
    except e:
        failed = True
    assert_true(failed)

    shutil.rmtree(path)
    store = VectorStore[2].create(path)
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    os.remove(path + "/hnsw.data")
    failed = False
    try:
        _ = VectorStore[2].open(path)
    except e:
        failed = True
    assert_true(failed)

    shutil.rmtree(path)
    store = VectorStore[2].create(path)
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    var data_file = builtins.open(path + "/hnsw.data", "wb")
    _ = data_file.write(builtins.bytes("corrupt", "utf-8"))
    _ = data_file.close()
    failed = False
    try:
        _ = VectorStore[2].open(path)
    except e:
        failed = True
    assert_true(failed)

    shutil.rmtree(path)
    store = VectorStore[2].create(path)
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    var meta_file = builtins.open(path + "/hnsw.meta", "wb")
    _ = meta_file.write(builtins.bytes("abc", "utf-8"))
    _ = meta_file.close()
    failed = False
    try:
        _ = VectorStore[2].open(path)
    except e:
        failed = True
    assert_true(failed)

    shutil.rmtree(path)
    store = VectorStore[2].create(path)
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    os.remove(path + "/tombstones.json")
    failed = False
    try:
        _ = VectorStore[2].open(path)
    except e:
        failed = True
    assert_true(failed)

    shutil.rmtree(path)
    store = VectorStore[2].create(path)
    _ = store.set("a", Span(ptr=vec.unsafe_ptr(), length=2))
    store.flush()

    var manifest_file = builtins.open(path + "/manifest.json", "r")
    var manifest = json.loads(manifest_file.read())
    _ = manifest_file.close()
    manifest["M"] = 32
    manifest_file = builtins.open(path + "/manifest.json", "w")
    _ = manifest_file.write(json.dumps(manifest))
    _ = manifest_file.close()
    failed = False
    try:
        _ = VectorStore[2].open(path)
    except e:
        failed = True
    assert_true(failed)

    shutil.rmtree(path)


def test_vector_store_text_graph_reload() raises:
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var path = "test_vector_store_text_graph_reload"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = VectorStore[2].create(path)
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 1.0]
    var c: List[Float32] = [2.0, 2.0]

    _ = store.set_text(
        "doc-a",
        Span(ptr=a.unsafe_ptr(), length=2),
        "alpha alpha memory",
        metadata=String("a-meta"),
    )
    _ = store.set_text(
        "doc-b",
        Span(ptr=b.unsafe_ptr(), length=2),
        "beta graph",
        metadata=String("b-meta"),
    )
    _ = store.set_text(
        "doc-c",
        Span(ptr=c.unsafe_ptr(), length=2),
        "alpha graph",
        metadata=String("c-meta"),
    )
    _ = store.add_edge("doc-a", "doc-b", "link", weight=Float32(1.5))
    _ = store.add_edge("doc-b", "doc-c", "link")
    store.flush()

    assert_true(os.path.exists(path + "/text_docs.json"))
    assert_true(os.path.exists(path + "/graph_edges.json"))

    var reopened = VectorStore[2].open(path)
    assert_true(reopened.options.text_enabled)
    assert_true(reopened.options.graph_enabled)

    var text_results = reopened.search_text("alpha", k=2)
    assert_equal(len(text_results), 2)
    assert_equal(text_results[0].id, "doc-a")
    assert_true(text_results[0].metadata)
    assert_equal(text_results[0].metadata.value(), "a-meta")

    var neighbors = reopened.neighbors("doc-a", GraphDirection.OUTGOING)
    assert_equal(len(neighbors), 1)
    assert_equal(neighbors[0], "doc-b")

    var path_results = reopened.shortest_path(
        "doc-a", "doc-c", GraphDirection.OUTGOING, max_depth=2
    )
    assert_equal(len(path_results), 3)
    assert_equal(path_results[0], "doc-a")
    assert_equal(path_results[1], "doc-b")
    assert_equal(path_results[2], "doc-c")

    shutil.rmtree(path)


def test_vector_store_text_facade() raises:
    var store = VectorStore[2].create_in_memory()
    var v0: List[Float32] = [0.0, 0.0]
    var v1: List[Float32] = [1.0, 1.0]
    var v2: List[Float32] = [2.0, 2.0]

    _ = store.set_text(
        "doc-a",
        Span(ptr=v0.unsafe_ptr(), length=2),
        "alpha alpha memory",
        metadata=String("a-meta"),
    )
    _ = store.set_text(
        "doc-b", Span(ptr=v1.unsafe_ptr(), length=2), "beta graph"
    )
    _ = store.set_text(
        "doc-c", Span(ptr=v2.unsafe_ptr(), length=2), "alpha graph"
    )

    assert_true(store.options.text_enabled)
    var results = store.search_text("alpha", k=2)
    assert_equal(len(results), 2)
    assert_equal(results[0].id, "doc-a")
    assert_true(results[0].metadata)
    assert_equal(results[0].metadata.value(), "a-meta")
    assert_true(results[0].distance <= results[1].distance)

    var empty = store.search_text("missing", k=10)
    assert_equal(len(empty), 0)

    var zero = store.search_text("alpha", k=0)
    assert_equal(len(zero), 0)


def test_vector_store_text_requires_opt_in() raises:
    var store = VectorStore[2].create_in_memory()
    var failed = False
    try:
        _ = store.search_text("alpha", k=1)
    except e:
        failed = True
    assert_true(failed)


def test_vector_store_graph_facade() raises:
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 1.0]
    var c: List[Float32] = [2.0, 2.0]

    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=2))

    var edge0 = store.add_edge("a", "b", "link")
    var edge1 = store.add_edge("b", "c", "link")
    assert_equal(edge0, 0)
    assert_equal(edge1, 1)
    assert_true(store.options.graph_enabled)

    var out_a = store.neighbors("a", GraphDirection.OUTGOING)
    assert_equal(len(out_a), 1)
    assert_equal(out_a[0], "b")

    var in_c = store.neighbors("c", GraphDirection.INCOMING)
    assert_equal(len(in_c), 1)
    assert_equal(in_c[0], "b")

    var traversed = store.traverse("a", GraphDirection.OUTGOING, max_depth=2)
    assert_equal(len(traversed), 3)
    assert_equal(traversed[0], "a")
    assert_equal(traversed[1], "b")
    assert_equal(traversed[2], "c")

    assert_true(store.has_path("a", "c", GraphDirection.OUTGOING, max_depth=2))
    assert_true(
        not store.has_path("c", "a", GraphDirection.OUTGOING, max_depth=2)
    )

    var path = store.shortest_path(
        "a", "c", GraphDirection.OUTGOING, max_depth=2
    )
    assert_equal(len(path), 3)
    assert_equal(path[0], "a")
    assert_equal(path[1], "b")
    assert_equal(path[2], "c")

    assert_true(store.remove_edge("a", "b", "link"))
    var no_out = store.neighbors("a", GraphDirection.OUTGOING)
    assert_equal(len(no_out), 0)
    assert_true(not store.remove_edge("a", "b", "link"))


def test_vector_store_graph_failure_paths() raises:
    var store = VectorStore[2].create_in_memory()
    var failed = False
    try:
        _ = store.neighbors("missing", GraphDirection.OUTGOING)
    except e:
        failed = True
    assert_true(failed)

    var a: List[Float32] = [0.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    failed = False
    try:
        _ = store.add_edge("a", "missing", "link")
    except e:
        failed = True
    assert_true(failed)


def test_vector_store_delete_detaches_graph_edges() raises:
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]
    var c: List[Float32] = [2.0, 0.0]
    var updated: List[Float32] = [10.0, 0.0]

    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=2))
    _ = store.add_edge("a", "b", "link")
    _ = store.add_edge("b", "c", "link")

    assert_true(store.delete("a"))
    assert_equal(len(store.neighbors("a", GraphDirection.OUTGOING)), 0)
    assert_true(
        not store.has_path(
            "a", "b", GraphDirection.OUTGOING, edge_type=String("link")
        )
    )

    _ = store.set("a", Span(ptr=updated.unsafe_ptr(), length=2))
    assert_equal(len(store.neighbors("a", GraphDirection.OUTGOING)), 0)
    assert_true(
        not store.has_path(
            "a", "b", GraphDirection.OUTGOING, edge_type=String("link")
        )
    )

    _ = store.add_edge("a", "c", "link")
    var out_a = store.neighbors(
        "a", GraphDirection.OUTGOING, edge_type=String("link")
    )
    assert_equal(len(out_a), 1)
    assert_equal(out_a[0], "c")


def test_consistency_check_clean_store() raises:
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]
    var c: List[Float32] = [2.0, 0.0]

    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=2))

    var report = verify_consistency(store)
    assert_true(
        report.ok(), "clean store should be consistent: " + report.summary()
    )
    assert_true(report.checked > 0)


def test_consistency_check_with_deletes() raises:
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]

    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    assert_true(store.delete("a"))

    var report = verify_consistency(store)
    assert_true(
        report.ok(),
        "store with deletes should be consistent: " + report.summary(),
    )


def test_consistency_check_with_text() raises:
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(text_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]

    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set_text_only("b", "text for b")

    var report = verify_consistency(store)
    assert_true(
        report.ok(),
        "store with text_only should be consistent: " + report.summary(),
    )


def test_consistency_check_with_graph() raises:
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(graph_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]

    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.add_edge("a", "b", "link")

    var report = verify_consistency(store)
    assert_true(
        report.ok(),
        "store with graph should be consistent: " + report.summary(),
    )


def test_consistency_check_counts_match() raises:
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(text_enabled=True, graph_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]
    var c: List[Float32] = [2.0, 0.0]

    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set_text_only("c", "text for c")
    _ = store.add_edge("a", "b", "link")
    assert_true(store.delete("b"))

    var report = verify_consistency(store)
    assert_true(
        report.ok(),
        "complex store should be consistent: " + report.summary(),
    )


def test_consistency_catches_corrupted_deleted_count() raises:
    """The checker must detect when deleted_count doesn't match actual deleted items.
    """
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    assert_true(store.delete("a"))

    # Corrupt: set deleted_count to 0 while "a" is still deleted
    store.deleted_count = 0
    var report = verify_consistency(store)
    assert_true(not report.ok(), "corrupted deleted_count must be caught")

    # Restore
    store.deleted_count = 1
    assert_true(verify_consistency(store).ok())


def test_consistency_catches_metadata_length_mismatch() raises:
    """The checker must detect when metadata array length differs from vector_external_ids.
    """
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))

    # Corrupt: remove metadata entry
    _ = store.metadata.pop()
    var report = verify_consistency(store)
    assert_true(not report.ok(), "metadata length mismatch must be caught")


def test_consistency_catches_dual_status() raises:
    """The checker must detect items that are both deleted and text_only."""
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(text_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set_text_only("b", "text for b")
    # Now: item 0 = vector "a", item 1 = text_only "b"
    # text_only = [False, True]

    # Corrupt: mark item 1 as also deleted (dual status)
    store.deleted[1] = True
    store.deleted_count = 1
    var report = verify_consistency(store)
    assert_true(
        not report.ok(), "dual status must be caught: " + report.summary()
    )

    # Restore
    store.deleted[1] = False
    store.deleted_count = 0
    assert_true(verify_consistency(store).ok())


def test_consistency_catches_text_only_count_mismatch() raises:
    """The checker must detect when text_only_count doesn't match actual text_only items.
    """
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(text_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set_text_only("b", "text")

    # Corrupt: set text_only_count to 0
    store.text_only_count = 0
    var report = verify_consistency(store)
    assert_true(not report.ok(), "corrupted text_only_count must be caught")


def test_consistency_passes_on_clean_complex_store() raises:
    """The checker must pass on a store with vectors, text, deletes, and graph edges.
    """
    var store = VectorStore[4].create_in_memory(
        VectorStoreOptions(text_enabled=True, graph_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var c: List[Float32] = [2.0, 0.0, 0.0, 0.0]
    var d: List[Float32] = [3.0, 0.0, 0.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=4))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=4))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=4))
    _ = store.set_text_only("d", "text for d")
    _ = store.add_edge("a", "b", "link")
    _ = store.add_edge("b", "c", "link")
    assert_true(store.delete("c"))

    var report = verify_consistency(store)
    assert_true(
        report.ok(),
        "complex clean store should be consistent: " + report.summary(),
    )
    assert_true(report.checked >= 10, "should run at least 10 checks")
    assert_equal(store.deleted_count, 1)
    assert_equal(store.text_only_count, 1)


def test_count_exists_close() raises:
    """count(), exists(), and close() work correctly."""
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(text_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]
    var c: List[Float32] = [2.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=2))

    # count/len
    assert_equal(store.count(), 3)
    assert_equal(store.len(), 3)

    # exists
    assert_true(store.exists("a"))
    assert_true(store.exists("b"))
    assert_true(not store.exists("nonexistent"))

    # exists after delete
    assert_true(store.delete("b"))
    assert_true(not store.exists("b"))
    assert_equal(store.count(), 2)

    # exists after text_only
    _ = store.set_text_only("d", "text")
    assert_true(not store.exists("d"))
    # count() = len() = total items minus deleted (text-only items are counted)
    assert_equal(store.count(), 3)

    # close
    store.close()


def test_search_with_allowlist_high_selectivity() raises:
    """Allowlist selecting >50% should use standard HNSW + post-filter."""
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]
    var c: List[Float32] = [2.0, 0.0]
    var d: List[Float32] = [3.0, 0.0]
    var e: List[Float32] = [4.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=2))
    _ = store.set("d", Span(ptr=d.unsafe_ptr(), length=2))
    _ = store.set("e", Span(ptr=e.unsafe_ptr(), length=2))

    # Allow 4 of 5 (80%) — should use standard HNSW path
    var allowlist = List[UInt8]()
    allowlist.append(1)
    allowlist.append(1)
    allowlist.append(1)
    allowlist.append(1)
    allowlist.append(0)
    var q: List[Float32] = [0.1, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=3),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    assert_equal(len(results), 3)
    assert_equal(results[0].id, "a")


def test_search_with_allowlist_low_selectivity() raises:
    """Allowlist selecting <10% should use exact scan."""
    var store = VectorStore[2].create_in_memory()
    # Insert 20 items
    for i in range(20):
        var v = List[Float32]()
        v.append(Float32(i))
        v.append(0.0)
        _ = store.set("item_" + String(i), Span(ptr=v.unsafe_ptr(), length=2))

    # Allow only 1 of 20 (5%) — should use exact scan
    var allowlist = List[UInt8]()
    for i in range(20):
        allowlist.append(0)
    allowlist[7] = 1

    var q: List[Float32] = [7.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=5),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    assert_equal(len(results), 1)
    assert_equal(results[0].id, "item_7")


def test_search_with_allowlist_empty() raises:
    """Empty allowlist should return no results."""
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))

    var allowlist = List[UInt8]()
    allowlist.append(0)
    var q: List[Float32] = [0.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=5),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    assert_equal(len(results), 0)


def test_search_with_allowlist_respects_deleted() raises:
    """Deleted items should be excluded even if in allowlist."""
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [1.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    assert_true(store.delete("a"))

    # Allowlist says both are allowed, but a is deleted
    var allowlist = List[UInt8]()
    allowlist.append(1)
    allowlist.append(1)
    var q: List[Float32] = [0.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=5),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    assert_equal(len(results), 1)
    assert_equal(results[0].id, "b")


def test_search_with_allowlist_respects_text_only() raises:
    """Text-only items should be excluded even if in allowlist."""
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(text_enabled=True)
    )
    var a: List[Float32] = [0.0, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set_text_only("b", "text")

    # Allowlist says both are allowed, but b is text-only
    var allowlist2 = List[UInt8]()
    allowlist2.append(1)
    allowlist2.append(1)
    var q2: List[Float32] = [0.0, 0.0]
    var results2 = store.search(
        Span(ptr=q2.unsafe_ptr(), length=2),
        SearchOptions(k=5),
        allowlist=Optional[List[UInt8]](allowlist2^),
    )
    assert_equal(len(results2), 1)
    assert_equal(results2[0].id, "a")


def test_search_with_max_distance_and_allowlist() raises:
    """max_distance should work with allowlist."""
    var store = VectorStore[2].create_in_memory()
    var a: List[Float32] = [0.0, 0.0]
    var b: List[Float32] = [10.0, 0.0]
    var c: List[Float32] = [0.1, 0.0]
    _ = store.set("a", Span(ptr=a.unsafe_ptr(), length=2))
    _ = store.set("b", Span(ptr=b.unsafe_ptr(), length=2))
    _ = store.set("c", Span(ptr=c.unsafe_ptr(), length=2))

    var allowlist = List[UInt8]()
    allowlist.append(1)
    allowlist.append(1)
    allowlist.append(1)
    var q: List[Float32] = [0.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=5, max_distance=Optional[Float32](1.0)),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    # b is at distance 100, should be excluded
    assert_equal(len(results), 2)


def test_exact_scan_does_not_return_deleted() raises:
    """Regression: _exact_scan must skip deleted items even if allowlisted."""
    var store = VectorStore[2].create_in_memory()
    for i in range(20):
        var v = List[Float32]()
        v.append(Float32(i))
        v.append(0.0)
        _ = store.set("item_" + String(i), Span(ptr=v.unsafe_ptr(), length=2))

    # Delete item_0
    assert_true(store.delete("item_0"))

    # Allowlist selects only index 0 (selectivity=0.05 < 0.1, exact scan path)
    var allowlist = List[UInt8]()
    allowlist.append(1)  # item_0 (deleted)
    for i in range(19):
        allowlist.append(0)

    var q: List[Float32] = [0.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=5),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    # Deleted item must not appear
    assert_equal(len(results), 0, "deleted item must not appear in exact scan")


def test_text_only_items_dont_collapse_recall() raises:
    """Regression: text_only items in HNSW should not dominate top-k near origin.
    """
    var store = VectorStore[2].create_in_memory(
        VectorStoreOptions(text_enabled=True)
    )

    # Add 5 vectors at distance 1.0 from origin
    for i in range(5):
        var v = List[Float32]()
        v.append(1.0)
        v.append(Float32(i))
        _ = store.set("vec_" + String(i), Span(ptr=v.unsafe_ptr(), length=2))

    # Add 10 text_only items (zero vectors — at distance 0 from origin)
    for i in range(10):
        _ = store.set_text_only("txt_" + String(i), "text " + String(i))

    # Search near origin — text_only items are at distance 0
    var q: List[Float32] = [0.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=5),
    )
    # Must return 5 vector items, not text_only items
    assert_equal(len(results), 5, "text_only items must not collapse recall")
    for r in results:
        assert_true(
            r.id.startswith("vec_"),
            "result must be a vector item, not: " + r.id,
        )


def test_overfetch_scales_with_selectivity() raises:
    """Regression: moderate-selectivity band must return k results even at 10% selectivity.
    """
    var store = VectorStore[2].create_in_memory()
    for i in range(200):
        var v = List[Float32]()
        v.append(Float32(i))
        v.append(0.0)
        _ = store.set("item_" + String(i), Span(ptr=v.unsafe_ptr(), length=2))

    # Allowlist selects 20 of 200 (10% selectivity — moderate band)
    var allowlist = List[UInt8]()
    for i in range(200):
        if i < 20:
            allowlist.append(1)
        else:
            allowlist.append(0)

    var q: List[Float32] = [0.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=10),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    # Must return exactly k results (all 20 allowed items are near query)
    assert_equal(len(results), 10, "overfetch must scale with selectivity")


def test_standard_path_overfetch_for_allowlist() raises:
    """Regression: standard path must overfetch when allowlist excludes nearest items.
    """
    var store = VectorStore[2].create_in_memory()
    # 4 items: 2 near origin (excluded), 2 far (allowed)
    var near1: List[Float32] = [0.0, 0.0]
    var near2: List[Float32] = [0.1, 0.0]
    var far1: List[Float32] = [100.0, 0.0]
    var far2: List[Float32] = [101.0, 0.0]
    _ = store.set("near1", Span(ptr=near1.unsafe_ptr(), length=2))
    _ = store.set("near2", Span(ptr=near2.unsafe_ptr(), length=2))
    _ = store.set("far1", Span(ptr=far1.unsafe_ptr(), length=2))
    _ = store.set("far2", Span(ptr=far2.unsafe_ptr(), length=2))

    # Allowlist: exclude near, include far (50% selectivity)
    var allowlist = List[UInt8]()
    allowlist.append(0)  # near1 excluded
    allowlist.append(0)  # near2 excluded
    allowlist.append(1)  # far1 allowed
    allowlist.append(1)  # far2 allowed

    var q: List[Float32] = [0.0, 0.0]
    var results = store.search(
        Span(ptr=q.unsafe_ptr(), length=2),
        SearchOptions(k=2),
        allowlist=Optional[List[UInt8]](allowlist^),
    )
    # Must return both far items, even though near items are closer
    assert_equal(len(results), 2, "standard path must overfetch for allowlist")
    assert_equal(results[0].id, "far1")
    assert_equal(results[1].id, "far2")


def main() raises:
    test_metric_values()
    print("PASS test_metric_values")
    test_hnsw_params_defaults()
    print("PASS test_hnsw_params_defaults")
    test_search_options_defaults()
    print("PASS test_search_options_defaults")
    test_vector_store_options_defaults()
    print("PASS test_vector_store_options_defaults")
    test_search_result_construction()
    print("PASS test_search_result_construction")
    test_vector_store_set_get_search()
    print("PASS test_vector_store_set_get_search")
    test_vector_store_failure_paths()
    print("PASS test_vector_store_failure_paths")
    test_vector_store_persistence_lifecycle()
    print("PASS test_vector_store_persistence_lifecycle")
    test_vector_store_persistence_failures()
    print("PASS test_vector_store_persistence_failures")
    test_vector_store_text_graph_reload()
    print("PASS test_vector_store_text_graph_reload")
    test_vector_store_text_facade()
    print("PASS test_vector_store_text_facade")
    test_vector_store_text_requires_opt_in()
    print("PASS test_vector_store_text_requires_opt_in")
    test_vector_store_graph_facade()
    print("PASS test_vector_store_graph_facade")
    test_vector_store_graph_failure_paths()
    print("PASS test_vector_store_graph_failure_paths")
    test_vector_store_delete_detaches_graph_edges()
    print("PASS test_vector_store_delete_detaches_graph_edges")
    test_consistency_check_clean_store()
    print("PASS test_consistency_check_clean_store")
    test_consistency_check_with_deletes()
    print("PASS test_consistency_check_with_deletes")
    test_consistency_check_with_text()
    print("PASS test_consistency_check_with_text")
    test_consistency_check_with_graph()
    print("PASS test_consistency_check_with_graph")
    test_consistency_check_counts_match()
    print("PASS test_consistency_check_counts_match")
    test_consistency_catches_corrupted_deleted_count()
    print("PASS test_consistency_catches_corrupted_deleted_count")
    test_consistency_catches_metadata_length_mismatch()
    print("PASS test_consistency_catches_metadata_length_mismatch")
    test_consistency_catches_dual_status()
    print("PASS test_consistency_catches_dual_status")
    test_consistency_catches_text_only_count_mismatch()
    print("PASS test_consistency_catches_text_only_count_mismatch")
    test_consistency_passes_on_clean_complex_store()
    print("PASS test_consistency_passes_on_clean_complex_store")
    test_count_exists_close()
    print("PASS test_count_exists_close")
    test_search_with_allowlist_high_selectivity()
    print("PASS test_search_with_allowlist_high_selectivity")
    test_search_with_allowlist_low_selectivity()
    print("PASS test_search_with_allowlist_low_selectivity")
    test_search_with_allowlist_empty()
    print("PASS test_search_with_allowlist_empty")
    test_search_with_allowlist_respects_deleted()
    print("PASS test_search_with_allowlist_respects_deleted")
    test_search_with_allowlist_respects_text_only()
    print("PASS test_search_with_allowlist_respects_text_only")
    test_search_with_max_distance_and_allowlist()
    print("PASS test_search_with_max_distance_and_allowlist")
    test_exact_scan_does_not_return_deleted()
    print("PASS test_exact_scan_does_not_return_deleted")
    test_text_only_items_dont_collapse_recall()
    print("PASS test_text_only_items_dont_collapse_recall")
    test_overfetch_scales_with_selectivity()
    print("PASS test_overfetch_scales_with_selectivity")
    test_standard_path_overfetch_for_allowlist()
    print("PASS test_standard_path_overfetch_for_allowlist")
