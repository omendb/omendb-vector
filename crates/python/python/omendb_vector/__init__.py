"""OmenDB Vector Python bindings — v2 surface.

    import omendb_vector as omendb

    db = omendb.connect("./mydata", metric="cosine", safety="strict")
    db.add(ids, vectors, text=texts, metadata=metadatas)
    db.delete(ids)
    db.get(ids)
    hits = db.search(query_vector=v, k=10, where={"lang": "en"})
    hits = db.search(query_text="hello", k=10, where="year >= 2024 OR lang != 'xx'")
    for hit in hits:
        print(hit.id, hit.score, hit.record)

`add` replaces existing records with the same id (INSERT OR REPLACE,
not append) and writes whole records (omitted fields are not
preserved).
"""
from omendb_vector._omendb_vector import PyStore as Store, PyHit as Hit, PyRecovery as Recovery

__all__ = ["Store", "Hit", "Recovery", "connect"]


def connect(path, *, metric="l2", safety="strict", backend="exact",
            hnsw_m=16, hnsw_m0=32, hnsw_ef_construction=192, hnsw_ef_search=200):
    """Open (or create) a store — the single constructor.

    Endpoint-neutral: today a local directory path; the same verb
    will accept a server endpoint later.

    Args:
        path: directory for the store (created if missing).
        metric: "cosine" | "l2" | "dot" — fixed for the collection
            once set (persists in the manifest).
        safety: "strict" (default) fsyncs every commit — never lose
            an acked write, even on power loss. "normal" fsyncs at
            checkpoints — process-crash-safe with a power-crash
            window (the sqlite `PRAGMA synchronous=NORMAL`-in-WAL
            analogy). The "normal" tier requires the durable-fs
            class threading (lands with its adoption merge); it
            raises until then.
        backend: "exact" (default) | "hnsw" for sealed segments.
    """
    if safety == "normal":
        raise ValueError(
            'safety="normal" lands with the durable-fs adoption merge; '
            'use safety="strict" for now'
        )
    if safety not in ("strict", "normal"):
        raise ValueError('safety must be "strict" or "normal"')
    store, recovery = Store.open(
        path, backend=backend, metric=metric,
        hnsw_m=hnsw_m, hnsw_m0=hnsw_m0,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )
    # Fix the collection metric on first connect (v2: metric locked
    # at connect; idempotent on reopen).
    store.set_metric(metric)
    return store
