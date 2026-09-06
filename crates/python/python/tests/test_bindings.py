"""End-to-end tests for the OmenDB Vector Python bindings."""
import os
import shutil
import tempfile

import pytest

from omendb_vector import Store


@pytest.fixture
def store_dir():
    d = tempfile.mkdtemp(prefix="omendb-py-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_lifecycle_and_durability(store_dir):
    store, recovery = Store.open(os.path.join(store_dir, "db"))
    assert recovery.committed_seq == 0
    assert store.dim == 0
    assert len(store) == 0

    seq1 = store.upsert(1, [0.1, 0.2], text="hello vector engine")
    seq2 = store.upsert(2, [0.3, 0.4], text="hello database", meta={"k": 1})
    assert seq2 > seq1
    store.commit()

    assert len(store) == 2
    assert store.dim == 2

    got = store.get(1)
    assert got["external_id"] == 1
    assert got["text"] == "hello vector engine"

    got2 = store.get(2)
    assert got2["meta"]["k"] == 1

    # checkpoint and reopen: state preserved
    store.checkpoint()
    store2, rec2 = Store.open(os.path.join(store_dir, "db"))
    assert len(store2) == 2
    assert store2.get(2)["meta"]["k"] == 1


def test_exact_search_oracle(store_dir):
    store, _ = Store.open(os.path.join(store_dir, "db"))
    store.upsert(1, [1.0, 0.0])
    store.upsert(2, [0.0, 1.0])
    store.upsert(3, [0.5, 0.5])
    store.commit()

    hits = store.exact_search("dot", [1.0, 0.0], 3)
    assert [h.external_id for h in hits] == [1, 3, 2]
    assert hits[0].score > hits[1].score > hits[2].score


def test_hnsw_backend_search(store_dir):
    import random
    random.seed(42)
    store, _ = Store.open(
        os.path.join(store_dir, "db"),
        backend="hnsw",
        metric="l2",
    )
    for i in range(1, 301):
        v = [random.random() for _ in range(8)]
        store.upsert(i, v, meta={"bucket": i % 3})
    store.commit()
    store.checkpoint()

    query = [0.5] * 8
    hits = store.exact_search("l2", query, 5)
    assert len(hits) == 5

    # filtered search
    f = [{"eq": {"field": "bucket", "value": 1}}]
    # backend search uses filtered_exact_top_k when filters given
    fhits = store.search("l2", query, 5, 5, filters=f)
    assert len(fhits) == 5
    # filtered: only ids with bucket == 1... need engine get to verify
    # (uses filtered_exact_top_k internally, oracle behavior)
    for h in fhits:
        rid = store.get(h.external_id)["external_id"]
        assert rid % 3 == 1


def test_text_search(store_dir):
    store, _ = Store.open(os.path.join(store_dir, "db"))
    store.upsert(1, [0.0], text="install omendb guide")
    store.upsert(2, [0.0], text="troubleshooting steps")
    store.commit()
    hits = store.text_search("install guide", 5)
    assert hits[0].external_id == 1
    assert len(hits) == 1


def test_hybrid_search_rrf(store_dir):
    store, _ = Store.open(os.path.join(store_dir, "db"))
    store.upsert(1, [1.0, 0.0], text="zzz qqq")
    store.upsert(2, [0.0, 1.0], text="install omendb")
    store.commit()
    fused = store.hybrid_search(2, 2, vector_query=[1.0, 0.0], text_query="install omendb")
    ids = [h.external_id for h in fused]
    assert set(ids) == {1, 2}


def test_errors(store_dir):
    store, _ = Store.open(os.path.join(store_dir, "db"))
    store.upsert(1, [0.1, 0.2])
    # mixed dim rejected
    with pytest.raises(Exception):
        store.upsert(2, [0.1, 0.2, 0.3])
    # unknown delete rejected
    with pytest.raises(Exception):
        store.delete(99)
    # unknown metric
    with pytest.raises(Exception):
        store.exact_search("manhattan", [1.0, 1.0], 1)
    # hybrid with no paths
    with pytest.raises(Exception):
        store.hybrid_search(2, 2)


def test_uncommitted_vanish_on_reopen(store_dir):
    path = os.path.join(store_dir, "db")
    store, _ = Store.open(path)
    store.upsert(1, [0.5])
    store.commit()
    store.upsert(2, [0.6])  # never committed
    del store  # simulate close

    store2, rec = Store.open(path)
    assert len(store2) == 1
    assert store2.get(2) is None
