# OmenDB Vector architecture

**Status:** standalone Rust engine under development. This document separates
current behavior from required contracts and measured-later proposals. It does
not establish production readiness. The 2026-10-13 repository/reuse decision below
is a target, not a completed migration.

## Current implementation

`crates/engine` owns canonical records, WAL framing/recovery, an in-memory L0,
persisted segments, exact/HNSW dense indexes, text scoring, metadata predicates
and hybrid RRF. `Store` is the public Rust entry point. The Python binding lives
in `crates/python` but is excluded from the Rust workspace; workspace checks do
not verify its public write/ACK behavior.

Records currently require a dense vector and may also carry text and typed
metadata. Exact metadata evaluation and BM25 reconstructed from current records
are implemented; bitmap routing, persistent posting snapshots and path-quality
fusion guards are not. File segments are decoded into memory, not mmap-backed
reader generations. Checkpoint currently rewrites the complete live collection,
and WAL history is not bounded by an implemented truncation policy.

Existing code is not proof that every intended contract below holds. Mutable
visibility/rollback, binding ACK parity, corruption handling, exclusive writer
ownership and persisted configuration/generation identity require qualification
before trusting the store with unrebuildable data.

## Ownership

Canonical records and collection configuration define truth. HNSW topology,
quantized codes, posting lists and filter acceleration are derived state.
Rebuildable indexes cannot define different deletion, replacement or transaction
visibility from the canonical record view.

The current standalone engine owns its own WAL and commit outcome. Sharing
`durable-fs` with OmenDB/OLAP shares filesystem primitives, not transaction
authority. Integrated OmenDB retrieval uses its canonical rows and shared kernel;
it does not embed this independently committing `Store` inside a SQL transaction.

### Repository and component direction

Keep this repository and its existing API available until replacement consumers
are qualified. The long-term direction is reusable retrieval components and
embedded/server facades, with native OmenDB vector/text access methods for queries
that need relational snapshot and transaction composition. Independently operated
or object-backed retrieval remains a valid deployment goal, not an automatic
requirement for a separate canonical store in every application.

Port concrete useful scoring, indexing, filtering or ranking components with their
exact/recall/mutation oracles. Once a component serves integrated OmenDB, its shared
source belongs in the OmenDB workspace and standalone consumers may depend on it.
Keep one maintained implementation; do not copy the entire engine or create speculative
shared crates. Retire replaced code only after migration and qualification. No repository
archival, package release, API removal or source relocation happened in this design review.

The cross-project decision is owned by
[OmenDB ADR 0012](https://github.com/omendb/omendb/blob/storage-kernel-vnext/docs/adr/0012-unified-modalities-and-rewrite-strategy.md).
An independent CDC-fed projection has its own durability/freshness contract and needs
source identity, retained bootstrap/change history and consumer progress committed
with its effects. It is not part of an atomic OmenDB transaction merely because its
code or branding is shared. Preserve licenses when extracting reusable components.

## Required mutation and durability contract

One engine lifecycle owns validation, staging, commit, rollback and failure
fencing. Bindings call that lifecycle rather than reimplement mutation loops.

- Validate an entire batch against its prospective ID-kind, dimension and
  collection configuration before appending any mutation.
- Keep uncommitted changes in a private overlay; preserve committed predecessors
  so rollback restores the same query-visible state without reopening.
- Commit acknowledges only after the declared WAL durability boundary and
  coherent canonical visibility. Python and Rust must share that boundary.
- Drop/cancel is not a fabricated successful abort after uncertain I/O. Fence the
  handle if safe continuation cannot be established; reopen resolves outcomes.
- Distinguish provably incomplete final writes from complete corrupt/unknown
  frames. Only the former may be repaired automatically. Corruption fails closed
  without deleting acknowledged history.
- Open and create have distinct authority requirements. Acquire exclusive
  writable ownership before destructive recovery; bind WAL, configuration and
  generations to one persistent store incarnation.
- Persist authoritative ID kind, dimension and fixed metric independently of
  derived indexes. Generation references prove schema/configuration identity,
  content integrity and complete canonical coverage before replacing replay.
- Checkpoint publishes a complete generation only after dependencies are durable.
  Retention advances only after publication and reader/recovery requirements
  permit it.

These are acceptance requirements, not assertions that current recovery and
bindings already satisfy them. Identity reuse, schema migration and WAL
reclamation need explicit protocols before formats become stable.

## Retrieval correctness before ranking

Every search path resolves one canonical view. Sealed candidates shadowed by
newer L0 replacements/tombstones are not eligible merely because their old score
is better. Validate candidate identity/version, score the current visible vector
where required, and suppress deleted/nonmatching records before final ranking.

Exact scan is the oracle for eligibility and scoring. ANN may approximate the
ranking/recall contract, not deletion or predicate semantics. If candidate
validation leaves fewer than `k` results while enough eligible records exist,
refill or use a declared exact fallback rather than silently returning a short
result as complete.

Standalone metadata predicates are not SQL's three-valued expressions and are
not trusted authorization policy. Their missing/NULL/type/numeric comparison
rules must be explicit. An eventual OmenDB operator translates typed SQL
predicates deliberately and uses trusted snapshot-bound eligibility. Replicated
ACL fields alone do not establish immediate revocation.

A published completeness frontier is not historical snapshot support. Future
reader generations must retain the values needed for their declared snapshot
interval, or use complete transaction-aware deltas/fallback. Capture canonical
visibility and physical index generations together; do not independently pick
the latest instance of each.

## Concurrency and resource lifecycle

Current ownership is a mutable `Store` handle, not a qualified single-writer plus
generation-pinned-reader system. A process-local borrow does not exclude another
process opening the same files.

The target starts with one enforced writer and coherent retained readers.
Multiple writers, RCU, server sessions and distributed placement are separate
requirements, not scaffolding to add now. Explicit reader leases must govern
checkpoint/compaction reclamation once physical generations are shared.

Bound private writes, L0, retained WAL and maintenance work before promising
larger-than-memory operation. Measure checkpoint pause, duplicated vector/index
memory and recovery history growth. A storage-provider trait does not make RAM,
mmap, NVMe and object-storage economics interchangeable.

## Performance and experiment policy

Exact plus HNSW are current dense backends. Quantization, filtered traversal,
segment routing, NVMe/object tiers and Mojo GPU kernels remain candidates, not
required infrastructure. Introduce a concrete backend or kernel before
generalizing a seam; keep batch-oriented scoring/reranking where useful.

Choose filtered exact scan, predicate-aware graph traversal, clustered/IVF or SSD
layouts using selectivity, filter/vector correlation, mutation rate, memory and
I/O. No unfiltered resident-memory result selects every deployment. Preserve
canonical values at their declared precision; compact routing codes are not a
license to change exact scoring or visibility. Large vector/posting extents need
not inherit the small-page layout of an OLTP index.

Use representative mutable retrieval, deletion/replacement, restrictive and
correlated filters, OOD embeddings and text/hybrid workloads. Report recall,
eligible-result count, p50/p95/p99, build/checkpoint/reopen time, memory and
dataset/hardware/configuration together. Separate ingest+commit timing from ANN
construction; a benchmark named “build” must identify what it actually includes.

The [evidence register](evidence-register.md) records external research and
candidate mechanisms. Literature performance and novelty claims are not measured
Vector results or mandatory routing/format choices. Promotion requires matched
end-to-end evidence after correctness gates pass.
