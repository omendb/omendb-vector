from multivector import (
    MultiVectorExactStore,
    MultiVectorFDEIndex,
    MultiVectorMuveraStore,
)
from std.python import Python
from std.testing import assert_equal, assert_true


def test_store_and_query_multi() raises:
    var store = MultiVectorExactStore[2]()
    var doc_a: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var doc_b: List[Float32] = [0.1, 0.0, 0.0, 0.1]
    _ = store.set_vectors(
        "doc-a",
        Span(ptr=doc_a.unsafe_ptr(), length=len(doc_a)),
        2,
        metadata=String("a-meta"),
    )
    _ = store.set_vectors(
        "doc-b", Span(ptr=doc_b.unsafe_ptr(), length=len(doc_b)), 2
    )

    var query: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var results = store.search_vectors(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=2
    )
    assert_equal(len(results), 2)
    assert_equal(results[0].id, "doc-a")
    assert_equal(results[0].distance, -2.0)
    assert_true(results[0].metadata)
    assert_equal(results[0].metadata.value(), "a-meta")
    assert_equal(results[1].id, "doc-b")


def test_text_candidates_rerank_by_maxsim() raises:
    var store = MultiVectorExactStore[2]()
    var doc_a: List[Float32] = [0.0, 1.0, 0.0, 1.0]
    var doc_b: List[Float32] = [1.0, 0.0, 1.0, 0.0]
    var doc_c: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    _ = store.set_vectors_text(
        "doc-a",
        Span(ptr=doc_a.unsafe_ptr(), length=len(doc_a)),
        2,
        "alpha shared",
    )
    _ = store.set_vectors_text(
        "doc-b",
        Span(ptr=doc_b.unsafe_ptr(), length=len(doc_b)),
        2,
        "alpha shared",
    )
    _ = store.set_vectors_text(
        "doc-c",
        Span(ptr=doc_c.unsafe_ptr(), length=len(doc_c)),
        2,
        "other",
    )

    var query: List[Float32] = [1.0, 0.0]
    var results = store.search_hybrid_vectors(
        "alpha", Span(ptr=query.unsafe_ptr(), length=len(query)), 1, k=2
    )
    assert_equal(len(results), 2)
    assert_equal(results[0].id, "doc-b")
    assert_equal(results[0].distance, -1.0)
    assert_equal(results[1].id, "doc-a")


def test_muvera_candidates_rerank_by_exact_maxsim() raises:
    var store = MultiVectorExactStore[2]()
    var candidate_index = MultiVectorFDEIndex[2, 4, 2](M=4, ef_construction=20)
    var doc_a: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var doc_b: List[Float32] = [0.1, 0.0, 0.0, 0.1]
    var doc_c: List[Float32] = [0.0, 1.0, 1.0, 0.0]

    var doc_id = store.set_vectors(
        "doc-a", Span(ptr=doc_a.unsafe_ptr(), length=len(doc_a)), 2
    )
    candidate_index.insert(
        doc_id, Span(ptr=doc_a.unsafe_ptr(), length=len(doc_a)), 2
    )
    doc_id = store.set_vectors(
        "doc-b", Span(ptr=doc_b.unsafe_ptr(), length=len(doc_b)), 2
    )
    candidate_index.insert(
        doc_id, Span(ptr=doc_b.unsafe_ptr(), length=len(doc_b)), 2
    )
    doc_id = store.set_vectors(
        "doc-c", Span(ptr=doc_c.unsafe_ptr(), length=len(doc_c)), 2
    )
    candidate_index.insert(
        doc_id, Span(ptr=doc_c.unsafe_ptr(), length=len(doc_c)), 2
    )

    var query: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var exact = store.search_vectors(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=2
    )
    var candidate_doc_ids = candidate_index.search(
        Span(ptr=query.unsafe_ptr(), length=len(query)),
        2,
        candidate_k=3,
        ef_search=20,
    )
    var reranked = store.search_vectors_from_doc_ids(
        Span(ptr=query.unsafe_ptr(), length=len(query)),
        2,
        Span(ptr=candidate_doc_ids.unsafe_ptr(), length=len(candidate_doc_ids)),
        k=2,
    )
    assert_equal(len(candidate_doc_ids), 3)
    assert_equal(len(reranked), 2)
    assert_equal(reranked[0].id, exact[0].id)
    assert_equal(reranked[1].id, exact[1].id)


def test_muvera_store_small_collection_matches_exact() raises:
    var store = MultiVectorMuveraStore[2]()
    for i in range(55):
        var id = "doc-" + String(i)
        if i == 0:
            var vectors: List[Float32] = [1.0, 0.0, 0.0, 1.0]
            _ = store.set_vectors(
                id, Span(ptr=vectors.unsafe_ptr(), length=len(vectors)), 2
            )
        else:
            var vectors: List[Float32] = [0.0, 1.0, 0.0, 0.8]
            _ = store.set_vectors(
                id, Span(ptr=vectors.unsafe_ptr(), length=len(vectors)), 2
            )

    assert_equal(store.candidate_index.len(), 55)
    var query: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var exact = store.search_vectors_exact(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=1
    )
    var muvera = store.search_vectors_with_options(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=1
    )
    assert_equal(len(exact), 1)
    assert_equal(len(muvera), 1)
    assert_equal(muvera[0].id, exact[0].id)
    assert_equal(muvera[0].id, "doc-0")


def test_muvera_store_filters_deleted_candidate() raises:
    var store = MultiVectorMuveraStore[2]()
    for i in range(55):
        var id = "doc-" + String(i)
        if i == 0:
            var vectors: List[Float32] = [1.0, 0.0, 0.0, 1.0]
            _ = store.set_vectors(
                id, Span(ptr=vectors.unsafe_ptr(), length=len(vectors)), 2
            )
        else:
            var vectors: List[Float32] = [0.0, 1.0, 0.0, 0.8]
            _ = store.set_vectors(
                id, Span(ptr=vectors.unsafe_ptr(), length=len(vectors)), 2
            )

    assert_true(store.delete("doc-0"))
    assert_equal(store.len(), 54)
    assert_equal(store.candidate_index.len(), 55)
    var query: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var results = store.search_vectors_with_options(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=3
    )
    assert_equal(len(results), 3)
    for i in range(len(results)):
        assert_true(results[i].id != "doc-0")


def test_muvera_store_reopen_rebuilds_candidate_index() raises:
    var shutil = Python.import_module("shutil")
    var builtins = Python.import_module("builtins")
    var json = Python.import_module("json")
    var os = Python.import_module("os")
    var path = "test_multivector_muvera_store"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = MultiVectorMuveraStore[2].create(path)
    for i in range(55):
        var id = "doc-" + String(i)
        if i == 0:
            var vectors: List[Float32] = [1.0, 0.0, 0.0, 1.0]
            _ = store.set_vectors_text(
                id,
                Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
                2,
                "target code path",
            )
        else:
            var vectors: List[Float32] = [0.0, 1.0, 0.0, 0.8]
            _ = store.set_vectors_text(
                id,
                Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
                2,
                "background code path",
            )
    store.flush()

    var query: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var before = store.search_vectors_with_options(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=1
    )
    var reopened = MultiVectorMuveraStore[2].open(path)
    assert_equal(reopened.len(), 55)
    assert_equal(reopened.candidate_index.len(), 55)
    var after = reopened.search_vectors_with_options(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=1
    )
    assert_equal(len(before), 1)
    assert_equal(len(after), 1)
    assert_equal(after[0].id, before[0].id)

    var manifest_file = builtins.open(path + "/manifest.json", "r")
    var manifest = json.loads(manifest_file.read())
    _ = manifest_file.close()
    assert_equal(String(manifest["encoding_mode"]), "muvera")
    assert_equal(
        String(manifest["candidate_index_mode"]), "fde_hnsw_rebuild_v0"
    )

    shutil.rmtree(path)


def test_get_vectors_metadata_and_delete() raises:
    var store = MultiVectorExactStore[2]()
    var vectors: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    _ = store.set_vectors(
        "doc",
        Span(ptr=vectors.unsafe_ptr(), length=len(vectors)),
        2,
        metadata=String('{"file":"src/main.rs"}'),
    )

    var stored = store.get_vectors("doc")
    assert_equal(len(stored), 4)
    assert_equal(stored[2], 3.0)
    assert_equal(store.get_vector_count("doc"), 2)
    var metadata = store.get_metadata("doc")
    assert_true(metadata)
    assert_equal(metadata.value(), '{"file":"src/main.rs"}')

    assert_true(store.delete("doc"))
    assert_equal(store.len(), 0)
    assert_equal(len(store.get_vectors("doc")), 0)
    assert_equal(store.get_vector_count("doc"), 0)
    assert_true(not store.get_metadata("doc"))
    assert_true(not store.delete("doc"))


def test_validation_errors() raises:
    var store = MultiVectorExactStore[2]()
    var vectors: List[Float32] = [1.0, 2.0, 3.0]
    var failed = False
    try:
        _ = store.set_vectors(
            "bad", Span(ptr=vectors.unsafe_ptr(), length=len(vectors)), 2
        )
    except:
        failed = True
    assert_true(failed)

    var query: List[Float32] = [1.0]
    failed = False
    try:
        _ = store.search_vectors(
            Span(ptr=query.unsafe_ptr(), length=len(query)), 1, k=1
        )
    except:
        failed = True
    assert_true(failed)


def test_persistence_reopen() raises:
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var path = "test_multivector_exact_store"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = MultiVectorExactStore[2].create(path)
    var doc_a: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var doc_b: List[Float32] = [0.0, 1.0, 0.0, 1.0]
    _ = store.set_vectors_text(
        "doc-a",
        Span(ptr=doc_a.unsafe_ptr(), length=len(doc_a)),
        2,
        "alpha memory",
        metadata=String("a-meta"),
    )
    _ = store.set_vectors_text(
        "doc-b",
        Span(ptr=doc_b.unsafe_ptr(), length=len(doc_b)),
        2,
        "alpha graph",
    )
    assert_true(store.delete("doc-b"))
    store.flush()

    assert_true(os.path.exists(path + "/manifest.json"))
    assert_true(os.path.exists(path + "/manifest.meta"))
    assert_true(os.path.exists(path + "/records.meta"))
    assert_true(os.path.exists(path + "/records.strings"))
    assert_true(os.path.exists(path + "/vectors.f32"))
    assert_true(os.path.exists(path + "/text_docs.meta"))
    assert_true(os.path.exists(path + "/text_docs.strings"))

    var reopened = MultiVectorExactStore[2].open(path)
    assert_equal(reopened.len(), 1)
    var query: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var results = reopened.search_vectors(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=5
    )
    assert_equal(len(results), 1)
    assert_equal(results[0].id, "doc-a")
    assert_equal(results[0].distance, -2.0)
    var text_results = reopened.search_hybrid_vectors(
        "alpha", Span(ptr=query.unsafe_ptr(), length=len(query)), 2, k=5
    )
    assert_equal(len(text_results), 1)
    assert_equal(text_results[0].id, "doc-a")
    assert_equal(reopened.get_vector_count("doc-a"), 2)
    assert_equal(reopened.get_vector_count("doc-b"), 0)

    shutil.rmtree(path)


def test_persistence_rejects_corrupt_vectors() raises:
    var shutil = Python.import_module("shutil")
    var builtins = Python.import_module("builtins")
    var os = Python.import_module("os")
    var path = "test_multivector_corrupt_vectors"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = MultiVectorExactStore[2].create(path)
    var doc: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    _ = store.set_vectors("doc", Span(ptr=doc.unsafe_ptr(), length=len(doc)), 2)
    store.flush()

    var vectors = builtins.open(path + "/vectors.f32", "w")
    _ = vectors.write("bad")
    _ = vectors.close()

    var failed = False
    try:
        _ = MultiVectorExactStore[2].open(path)
    except:
        failed = True
    assert_true(failed)

    shutil.rmtree(path)


def main() raises:
    test_store_and_query_multi()
    test_text_candidates_rerank_by_maxsim()
    test_muvera_candidates_rerank_by_exact_maxsim()
    test_muvera_store_small_collection_matches_exact()
    test_muvera_store_filters_deleted_candidate()
    test_muvera_store_reopen_rebuilds_candidate_index()
    test_get_vectors_metadata_and_delete()
    test_validation_errors()
    test_persistence_reopen()
    test_persistence_rejects_corrupt_vectors()
    print("multivector tests passed")
