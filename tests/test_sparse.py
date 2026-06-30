"""Sparse vector tests — insertion and basic retrieval.

Three-way hybrid search (dense + sparse + BM25) is implemented in the Mojo
engine (store.mojo search_hybrid) and wired through Collection.search(sparse=...).
"""

import pytest
from omendb import Collection, CollectionConfig


def test_sparse_set_and_query():
    """set_sparse succeeds and dense search still works."""
    config = CollectionConfig(dim=128, metric="cosine", sparse=True)
    c = Collection(":memory:", config=config)

    c.set("doc1", vector=[1.0] * 128)
    c.set_sparse("doc1", {0: 0.5, 1: 0.8, 3: 0.3})

    c.set("doc2", vector=[0.5] * 128)
    c.set_sparse("doc2", {2: 0.7, 4: 0.2})

    # Dense search still works
    results = c.search(vector=[1.0] * 128, k=2)
    assert len(results) == 2
    assert results[0].id == "doc1"
    assert results[0].score is not None

    c


def test_sparse_missing_id_raises():
    """set_sparse on unknown ID raises error."""
    config = CollectionConfig(dim=128, sparse=True)
    c = Collection(":memory:", config=config)
    with pytest.raises(Exception):
        c.set_sparse("nonexistent", {0: 1.0})
    c


def test_sparse_auto_enables():
    """set_sparse auto-enables sparse on the store (like text)."""
    config = CollectionConfig(dim=128)  # sparse not explicitly set
    c = Collection(":memory:", config=config)
    c.set("a", vector=[1.0] * 128)
    # Auto-enable: since store.set_sparse auto-enables, this should work
    c.set_sparse("a", {10: 1.0, 20: 0.5})
    c


def test_sparse_with_text():
    """Sparse can be set alongside text."""
    config = CollectionConfig(dim=128, metric="cosine", text=True, sparse=True)
    c = Collection(":memory:", config=config)
    c.set("d1", vector=[0.1] * 128, text="hello world")
    c.set_sparse("d1", {5: 1.0, 15: 0.5, 25: 0.3})

    # Text search still works
    results = c.search(text="hello", k=5)
    assert len(results) >= 1
    c


def test_search_sparse_parameter_reaches_hybrid_engine():
    """Collection.search(sparse=...) contributes to native 3-way RRF."""
    config = CollectionConfig(dim=2, metric="cosine", text=True, sparse=True)
    c = Collection(":memory:", config=config)
    c.set("dense_only", vector=[1.0, 0.0], text="dense document")
    c.set_sparse("dense_only", {1: 0.1})
    c.set("sparse_winner", vector=[0.0, 1.0], text="sparse document")
    c.set_sparse("sparse_winner", {42: 10.0})

    results = c.search(
        vector=[1.0, 0.0],
        text="term-that-does-not-exist",
        sparse={42: 1.0},
        mode="hybrid",
        hybrid_alpha=0.0,
        k=2,
    )
    assert results[0].id == "sparse_winner"


def test_sparse_high_dimensional():
    """Sparse with high-dimensional spaces (BGE-M3 style dims)."""
    config = CollectionConfig(dim=128, metric="cosine", sparse=True)
    c = Collection(":memory:", config=config)
    c.set("doc", vector=[0.1] * 128)
    # BGE-M3 sparse dims can be up to 250K
    sparse = {i: 0.5 for i in range(0, 1000, 10)}  # 100 nonzero dims
    c.set_sparse("doc", sparse)
    c


def test_sparse_persistence_roundtrip(tmp_path):
    """Sparse vectors survive flush/reopen."""
    import os
    path = str(tmp_path / "sparse_persist")

    config = CollectionConfig(dim=128, text=True, sparse=True, metric="cosine")
    c = Collection(path, config=config)
    c.set("d1", vector=[0.1] * 128, text="hello world")
    c.set_sparse("d1", {5: 1.0, 15: 0.5, 25: 0.3})
    c.flush()

    # Reopen
    c2 = Collection(path, config=config, create=False)
    # Insert another doc — sparse index should still be functional
    c2.set("d2", vector=[0.2] * 128, text="world")
    c2.set_sparse("d2", {10: 0.7})
    # Dense search still works after reopen
    results = c2.search(vector=[0.1] * 128, k=2)
    assert len(results) >= 1
