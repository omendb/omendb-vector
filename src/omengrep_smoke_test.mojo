from multivector import MultiVectorExactStore
from std.python import Python
from std.testing import assert_equal, assert_true


def test_omengrep_index_search_similar_and_reopen() raises:
    var shutil = Python.import_module("shutil")
    var os = Python.import_module("os")
    var path = "test_omengrep_store"
    if os.path.exists(path):
        shutil.rmtree(path)

    var store = MultiVectorExactStore[2].create(path)

    var fn_vectors: List[Float32] = [1.0, 0.0, 0.9, 0.1]
    var trait_vectors: List[Float32] = [0.0, 1.0, 0.1, 0.9]
    var stale_vectors: List[Float32] = [1.0, 0.0, 1.0, 0.0]

    _ = store.set_vectors_text(
        "src/lib.rs:function:10",
        Span(ptr=fn_vectors.unsafe_ptr(), length=len(fn_vectors)),
        2,
        "async function store with text and metadata",
        metadata=String('{"file":"src/lib.rs","kind":"function","line":10}'),
    )
    _ = store.set_vectors_text(
        "src/lib.rs:trait:30",
        Span(ptr=trait_vectors.unsafe_ptr(), length=len(trait_vectors)),
        2,
        "trait implementation graph traversal",
        metadata=String('{"file":"src/lib.rs","kind":"trait","line":30}'),
    )
    _ = store.set_vectors_text(
        "src/old.rs:function:4",
        Span(ptr=stale_vectors.unsafe_ptr(), length=len(stale_vectors)),
        2,
        "old deleted function store",
    )
    assert_true(store.delete("src/old.rs:function:4"))

    var query: List[Float32] = [1.0, 0.0]
    var hybrid = store.search_hybrid_vectors(
        "function store",
        Span(ptr=query.unsafe_ptr(), length=len(query)),
        1,
        k=5,
    )
    assert_equal(len(hybrid), 1)
    assert_equal(hybrid[0].id, "src/lib.rs:function:10")
    assert_equal(hybrid[0].distance, -1.0)

    var semantic = store.search_vectors_with_options(
        Span(ptr=query.unsafe_ptr(), length=len(query)), 1, k=5
    )
    assert_equal(semantic[0].id, "src/lib.rs:function:10")

    var vectors = store.get_vectors("src/lib.rs:function:10")
    assert_equal(len(vectors), 4)
    assert_equal(store.get_vector_count("src/lib.rs:function:10"), 2)

    var metadata = store.get_metadata("src/lib.rs:function:10")
    assert_true(metadata)
    assert_equal(
        metadata.value(), '{"file":"src/lib.rs","kind":"function","line":10}'
    )

    store.flush()
    var reopened = MultiVectorExactStore[2].open(path)
    var reopened_hybrid = reopened.search_hybrid_vectors(
        "function store",
        Span(ptr=query.unsafe_ptr(), length=len(query)),
        1,
        k=5,
    )
    assert_equal(len(reopened_hybrid), 1)
    assert_equal(reopened_hybrid[0].id, "src/lib.rs:function:10")
    assert_equal(reopened.get_vector_count("src/old.rs:function:4"), 0)

    shutil.rmtree(path)


def main() raises:
    test_omengrep_index_search_similar_and_reopen()
    print("omengrep smoke passed")
