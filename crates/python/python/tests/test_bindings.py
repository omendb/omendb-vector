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
    assert [h.id for h in hits] == [1, 3, 2]
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

    # filtered search: legacy predicate-dict list via where= (v2 form)
    f = [{"eq": {"field": "bucket", "value": 1}}]
    fhits = store.search(query, 5, where=f, metric="l2")
    assert len(fhits) == 5
    # filtered: only ids with bucket == 1... need engine get to verify
    # (uses filtered_exact_top_k internally, oracle behavior)
    for h in fhits:
        rid = store.get(h.id)["external_id"]
        assert rid % 3 == 1


def test_text_search(store_dir):
    store, _ = Store.open(os.path.join(store_dir, "db"))
    store.upsert(1, [0.0], text="install omendb guide")
    store.upsert(2, [0.0], text="troubleshooting steps")
    store.commit()
    hits = store.text_search("install guide", 5)
    assert hits[0].id == 1
    assert len(hits) == 1


def test_hybrid_search_rrf(store_dir):
    store, _ = Store.open(os.path.join(store_dir, "db"))
    store.upsert(1, [1.0, 0.0], text="zzz qqq")
    store.upsert(2, [0.0, 1.0], text="install omendb")
    store.commit()
    fused = store.hybrid_search(2, 2, vector_query=[1.0, 0.0], text_query="install omendb")
    ids = [h.id for h in fused]
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


def test_string_ids_end_to_end(tmp_path):
    store, _ = Store.open(str(tmp_path / "db"))
    store.upsert("doc-a", [0.1, 0.2], text="alpha beta")
    store.upsert("doc-b", [0.3, 0.4], text="gamma")
    store.commit()

    # get by string id
    got = store.get("doc-a")
    assert got["external_id"] == "doc-a"

    # search returns string ids
    hits = store.exact_search("dot", [1.0, 1.0], 2)
    assert {h.id for h in hits} == {"doc-a", "doc-b"}

    # mixed kind rejected
    try:
        store.upsert(7, [0.5, 0.5])
        assert False, "mixed id kind must raise"
    except Exception:
        pass

    # delete (batch shape) + checkpoint + reopen: kind and records persist
    store.delete(["doc-b"])
    store.commit()
    store.checkpoint()
    store2, _ = Store.open(str(tmp_path / "db"))
    assert store2.get("doc-a")["external_id"] == "doc-a"
    assert store2.get("doc-b") is None
    # string ids still locked after reopen
    try:
        store2.upsert(9, [0.5, 0.5])
        assert False
    except Exception:
        pass


# ---- v2 surface (docs/api-v2-draft.md) ----

def test_v2_connect_and_add(tmp_path):
    import omendb_vector as omendb

    db = omendb.connect(str(tmp_path / "db"), metric="dot")
    # columnar batch, n>=1: single = batch of 1
    db.add([1, 2], [[1.0, 0.0], [0.0, 1.0]], text=[None, "hello world"])
    db.commit()

    assert db.get([1])["external_id"] == 1
    assert db.get([1, 2])[1]["text"] == "hello world"

    # keyed last-wins: re-add id 1 with a new vector
    db.add([1], [[0.5, 0.5]])
    db.commit()
    assert db.get([1])["vector"] == [0.5, 0.5]
    assert db.__len__() == 2


def test_v2_where_dict_and_string(tmp_path):
    import omendb_vector as omendb

    db = omendb.connect(str(tmp_path / "db"), metric="dot")
    db.add(
        ["a", "b", "c"],
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        text=["alpha", "beta", "gamma"],
        metadata=[{"lang": "en", "year": 2023},
                  {"lang": "fr", "year": 2024},
                  {"lang": "en", "year": 2024}],
    )
    db.commit()

    # dict form: AND of equalities
    hits = db.search([1.0, 0.0], k=3, where={"lang": "en"})
    assert sorted(h.id for h in hits) == ["a", "c"]

    # string form: OR (engine AND-only planner routes exact; exact eval is oracle)
    hits = db.search([1.0, 0.0], k=3, where="lang = 'en' OR year >= 2024")
    assert {h.id for h in hits} == {"a", "b", "c"}

    # string form with NOT + parens
    hits = db.search([1.0, 0.0], k=3, where="(lang = 'fr' OR year = 2023) AND NOT year = 2023")
    assert [h.id for h in hits] == ["b"]

    # text search with where
    hits = db.text_search("alpha", 5, where="year >= 2024")
    assert hits == []  # 'alpha' is in record a (year 2023)

    # hybrid with where
    hits = db.hybrid_search(k=2, window=3, vector_query=[1.0, 0.0], text_query="beta",
                           where={"lang": "fr"})
    assert [h.id for h in hits] == ["b"]


def test_v2_rollback_and_transaction_semantics(tmp_path):
    import omendb_vector as omendb

    db = omendb.connect(str(tmp_path / "db"))
    db.add([1], [[0.1, 0.2]])
    db.commit()
    db.add([2], [[0.3, 0.4]])  # unacked

    db.rollback()
    assert db.__len__() == 1
    assert db.get([2]) is None

    # post-rollback commit cannot resurrect the rolled-back record
    db.add([3], [[0.5, 0.6]])
    db.commit()
    assert db.__len__() == 2
    assert db.get([2]) is None


def test_v2_metric_locked_at_connect(tmp_path):
    import omendb_vector as omendb

    db = omendb.connect(str(tmp_path / "db"), metric="dot")
    db.add([1], [[1.0, 0.0]])
    db.commit()

    # reopen via connect with the same metric: fine (idempotent)
    db2 = omendb.connect(str(tmp_path / "db"), metric="dot")
    assert db2.get([1])["external_id"] == 1

    # conflicting metric on reopen: loud error
    try:
        omendb.connect(str(tmp_path / "db"), metric="l2")
        assert False
    except Exception:
        pass

    # search defaults to the locked collection metric
    hits = db2.search([1.0, 0.0], k=1)
    assert hits[0].id == 1
