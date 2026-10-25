# OmenDB Vector

## Source and design owners

This repository implements the standalone Rust retrieval engine. Read the
[README](README.md) and [current architecture](docs/architecture.md), then
inspect source and current CI before changing code. The
[evidence register](docs/evidence-register.md) records external research and
unqualified candidates; [API v2 draft](docs/api-v2-draft.md) is **not an
accepted/implemented API contract**.

[OmenDB's specialized-engine contract](https://github.com/omendb/omendb/blob/main/docs/design/engines-and-scale.md)
defines the long-term integration boundary: canonical relational records,
constraints and catalog share **one** SeerDB/OmenDB transaction authority.
Standalone Vector can retain its independently qualified WAL and storage
when it owns independently ingested datasets. Never embed this standalone
`Store` and its WAL as another independently committing component of one
OmenDB SQL transaction. Sharing retrieval algorithms does not require
the standalone Vector engine to adopt SeerDB's OLTP B-tree or object layout.

## Reliability and performance

- Keep canonical identity, replacements, deletions, metadata/filters and
  transaction visibility consistent across exact, dense, text and hybrid
  paths. ANN approximates ranking, never eligibility/authorization.
- An ACK requires declared durable WAL and canonical visible state. Correct
  rollback, same-handle continuation, checkpoints, corruption refusal,
  retained readers and safe reclaim before trusting unrebuildable data.
- Changes to indexing/search, mutation, memory, WAL/checkpoint, segments
  or tiering require representative exactness, recall, churn, p99,
  memory/space and recovery comparisons at matching revision/hardware.
  Historical SIFT and papers are references, not replacement qualification.
- Keep reusable ranking/scoring/index components in **one** maintained owner
  once integrated; do not publish private OLAP source or start an incompatible
  duplicate implementation merely for repository symmetry.

## Local checks

~~~sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
~~~

Workspace checks do not exercise the currently excluded Python binding.
Run the relevant binding/real-client checks separately when changing it.

