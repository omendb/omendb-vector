# OmenDB Vector

Embedded retrieval database in Rust for mutable application data. Combines
vector search, BM25 text search, metadata filtering, and hybrid ranking in a
local store.

**Work in progress.** The Rust engine is implemented and under development.
APIs and storage formats may change; this is not a production release.

## Current engine

The [`omendb-vector-engine`](crates/engine) crate provides:

- Record insertion, replacement, deletion, and explicit transactions.
- Dense vector search, text search, and reciprocal-rank fusion for hybrid queries.
- Metadata filters and exact-search paths for checking retrieval results.
- WAL recovery, checkpoints, and persisted segments.

Canonical records own the stored data. Derived search indexes can be rebuilt
from those records. See the [architecture](docs/architecture.md) for the storage
and retrieval contracts, and the [evidence register](docs/evidence-register.md)
for validation status and limitations.

## Build and test

Run from the repository root with Rust installed:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

The Rust workspace currently contains the engine crate. Its public entry point
is `Store`, with APIs for records, transactions, search, and checkpointing.
Generate the API documentation with:

```sh
cargo doc --package omendb-vector-engine --no-deps --open
```

## License

[AGPL-3.0-only](LICENSE).
