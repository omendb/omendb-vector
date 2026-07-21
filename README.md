# OmenDB

> **Developer preview (`0.1.0a1`).** OmenDB-Mojo is an early public preview.
> APIs, persistence formats, supported platforms, and performance are still
> subject to change. It is not a production-stability guarantee.

Embedded local database for hybrid vector + text search with source evidence.

OmenDB stores items with dense vectors, text, metadata, source evidence, and
bounded relationships. It maintains HNSW, BM25, filter, and relationship indexes
over those items. Search is hybrid: vector, text, or combined with RRF.

Build this preview from source (requires [Pixi](https://pixi.sh)):

```bash
git clone https://github.com/omendb/omendb.git
cd omendb
pixi install
pixi run python -m pip install -e .
```

This Mojo preview is source-only and is not yet the package served by
`pip install omendb`; use the source-build commands above.

The Rust implementation and its existing language bindings live in the
separate [`omendb-rs`](https://github.com/omendb/omendb-rs) repository. This
repository is the active Mojo engine and does not replace that repository's
published package history retroactively.

## Quickstart

```python
import omendb

db = omendb.create("./my-db")
docs = db.collection("docs", config=omendb.CollectionConfig(dim=128, text=True))

# Insert with vector, text, metadata, and source evidence
docs.set(
    "install",
    vector=[0.0] * 128,
    text="install package and open a local store",
    metadata={"kind": "guide"},
    source={"path": "README.md", "line_start": 1, "line_end": 10},
)
docs.set("search", vector=[1.0] * 128, text="hybrid vector and text search")

# Search
docs.search_text("package", k=5)          # BM25 text search
docs.search_vector([0.0] * 128, k=5)      # HNSW vector search
docs.search_hybrid(vector=[0.0] * 128, text="package", k=5)  # RRF hybrid

# Metadata filters (equality, $in, $gt/$lt/$gte/$lte)
docs.search_vector([0.0] * 128, k=5, filter={"kind": "guide"})
docs.search_vector([0.0] * 128, k=5, filter={"score": {"$gt": 40}})

# Source evidence in results
r = docs.search_vector([0.0] * 128, k=1, explain=True)
print(r[0].explanation)  # diagnostics: method, eligible_count, selectivity
print(r[0].source)       # {"path": "README.md", "line_start": 1, ...}

# Batch insert (4-7x faster than individual sets)
items = [
    {"id": "doc1", "vector": [0.0] * 128, "metadata": {"source": "web"}},
    {"id": "doc2", "vector": [1.0] * 128, "metadata": {"source": "api"}},
]
docs.set_many(items)

# Lifecycle: supersede old items with new ones
docs.set("old", vector=[0.0] * 128, metadata={"version": 1})
docs.set("new", vector=[0.1] * 128, metadata={"version": 2})
docs.supersede("old", "new")  # old is hidden, new is active

# Batch retrieve
items = docs.multi_get(["install", "search", "missing"])
# Returns list in same order; None for missing/deleted

# Timestamps (auto-managed)
item = docs.get("install", include_timestamps=True)
print(item["created_at"])   # Unix timestamp
print(item["updated_at"])   # set on every update

# Relationships (graph)
docs.add_relationship("install", "search", type="mentions")
docs.neighbors("install")

# Persistence
db.flush()
db2 = omendb.open("./my-db")
docs2 = db2.collection("docs", config=omendb.CollectionConfig(dim=128, text=True))
docs2.search_text("package", k=5)  # data survived reopen

# Integrity
db2.check()       # verify collection integrity
docs2.vacuum()    # compact deleted records
```

## Search Modes

| Mode | Method | Index | Description |
|------|--------|-------|-------------|
| Vector | `search_vector()` | HNSW | ANN search over dense vectors |
| Text | `search_text()` | BM25 | Full-text search with tokenization |
| Hybrid | `search_hybrid()` | HNSW + BM25 | RRF fusion of vector and text |
| Multivector | `search_vectors()` | MaxSim | Multi-vector per item (ColBERT-style) |

## Metrics

```python
# L2 (default)
docs = db.collection("docs", config=omendb.CollectionConfig(dim=128, metric="l2"))

# Cosine similarity
docs = db.collection("docs", config=omendb.CollectionConfig(dim=128, metric="cosine"))

# Dot product
docs = db.collection("docs", config=omendb.CollectionConfig(dim=128, metric="dot"))
```

## Filters

```python
# Equality
docs.search_vector(v, k=5, filter={"kind": "guide"})

# Not-equal
docs.search_vector(v, k=5, filter={"kind": {"$ne": "api"}})

# Allow-list
docs.search_vector(v, k=5, filter={"kind": {"$in": ["guide", "api"]}})

# Range (numeric and string)
docs.search_vector(v, k=5, filter={"score": {"$gte": 10, "$lte": 50}})

# Conjunction (AND)
docs.search_vector(v, k=5, filter={"kind": "guide", "status": "active"})

# Existence
docs.search_vector(v, k=5, filter={"score": {"$exists": True}})

# Boolean logic
docs.search_vector(v, k=5, filter={"$not": {"kind": "api"}})
docs.search_vector(v, k=5, filter={"$and": [{"kind": "guide"}, {"score": {"$gte": 10}}]})
docs.search_vector(v, k=5, filter={"$or": [{"kind": "guide"}, {"kind": "api"}]})
```

Filter routing: broad filters (>20% selectivity) use HNSW with bitmap
constraints for fast traversal; narrow filters (≤20%) fall back to exact
brute-force for guaranteed recall.

## Diagnostics

```python
results = docs.search_vector(v, k=5, filter={"kind": "guide"}, explain=True)
e = results[0].explanation
# {
#   "method": "hnsw_filtered_l2",
#   "filter": {
#     "strategy": "engine_bitmap_filter",
#     "eligible_count": 42,
#     "selectivity": 0.15,
#   },
# }
```

## Supported Dimensions

| Index | Dims | Notes |
|-------|------|-------|
| HNSW | 2, 128, 256, 384, 512, 768, 1024, 1536, 3072 | Compile-time optimized |
| Flat | Any | Exact search, no HNSW |

## Persistence

- One writer per collection path.
- Concurrent readers open committed snapshots.
- `flush()` commits to disk.
- `check()` reports integrity issues.
- `vacuum()` compacts deleted records.
- `snapshot()`/`import_snapshot()` for backup/restore.

## CLI

```bash
omendb-check ./my-db           # verify database integrity
omendb-engine-check            # print engine version
```

## Platform

- macOS ARM64 (Apple Silicon)
- Linux x86_64 (Fedora, Ubuntu)
- CPython 3.14 (standard ABI)
- No external runtime dependencies

## Documentation

- **[API Reference](docs/API.md)** — full Python API reference

## Benchmarks

These are early local reference measurements, not release guarantees. Results
depend on hardware, data distribution, index settings, and query mix.

M3 Max (macOS 15, Apple Silicon), 128-dimensional L2 vectors, Python API:

| Scale | Build | Search QPS | Search Latency |
|-------|-------|------------|----------------|
| 1K vectors | 65ms | 32,600 | 31 µs |
| 10K vectors | 180ms | 27,900 | 36 µs |

HNSW backend (M=16, efConstruction=100, ef=100). The public preview uses F32
traversal; quantized traversal remains experimental and is not a release
guarantee. Metadata filtering and hybrid search are workload-dependent and
are not covered by these early reference measurements.

## License

AGPL-3.0-only. Contact for commercial AGPL-free licensing.
