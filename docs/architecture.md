# Architecture — OmenDB Vector Fresh Rust Engine

Status: draft for review. Research phase (see `docs/evidence-register.md`); no engine code before approval.
Decisions locked: B1 language posture (Rust core, Mojo GPU earned lane); v0 surface dense+text+filters+hybrid; storage scope from research (this doc); SeerDB patterns-only.

## 1. Goals and non-goals

- Embedded local-first retrieval engine (dense, BM25 text, metadata filters, hybrid RRF) that runs **local or server** behind one engine core.
- Storage-neutral: RAM, mmap file segments, and (as contracted seams) NVMe and object tiers. Tier invisible to the caller.
- Performance as default; API is the product (`search()` leaks no algorithm, quantization, or tier).
- Exact flat search is the correctness oracle for every ANN, filtered, and quantized mode.
- Out: SQL/OLTP, full graph DB, lakehouse, authz policy, sparse/multivector/relationships in v0 (deferred per v0 scope; design inputs recorded in `docs/evidence-register.md` §5–§6).

## 2. Crate layout

```text
crates/engine          # core: zero binding deps
  records/             # canonical record store: ids, vectors, text, metadata, lifecycle
  wal/                 # write-ahead log: framing, checksums, replay, truncation
  segments/            # L0 + immutable segments, generation manifests
  index/               # VectorIndex trait + backends (exact, hnsw, quant)
  text/                # BM25 postings over the segment model
  filter/              # typed metadata indexes, bitmap predicate eval
  planner/             # selectivity routing, fusion (RRF + weakest-link guard), rerank
  storage/             # file/mmap providers; NVMe + object seams live here
  kernels/             # kernel-provider trait (CPU impls now; Mojo GPU plugin later)
crates/engine-ffi      # later: C ABI
crates/python          # later: PyO3 bindings (after core API stabilizes)
crates/node            # later: NAPI bindings
```

Bindings wait until the core API shape stabilizes. No `kernels/` Mojo lane until a benchmark demands it.

## 3. Data model

- **Canonical records** are the sole mutation authority: stable external IDs, dense vector (F32), text, typed metadata, lifecycle state (live / deleted-tombstone / superseded).
- **Derived everything else:** HNSW topology, quantized traversal codes, BM25 postings, filter bitmaps, adjacency sidecars. Generation-bound, checksummed, rebuildable from canonical. Damaged derived state fails closed or rebuilds; it never makes canonical state ambiguous.
- **Segments:** mutable in-RAM **L0** (WAL-backed, exact scan) + **immutable segments** (canonical payloads + derived artifacts + manifest). Queries fan out over L0 + segments, merge by exact final score, apply lifecycle visibility after every candidate path.
- **Manifest** per generation: segment list, capability/checksum metadata, commit sequence, config fingerprint. Atomic publish; incomplete generations fail closed.

## 4. Durability (day-one design, not a later lane)

- WAL-first: every mutation durable in the WAL before visibility; checkpoints publish immutable generations; WAL truncation only after manifest publish.
- Crash matrix from the start: torn-tail discard, generation-rename atomicity, manifest-publish-before-reset ordering, corruption fail-closed (no silent fallback).
- Rationale: durability is the product wedge (DuckDB-vss-class failures are the competitive gap) and the #1 recorded failure of both prior engines. omendb-rs WAL patterns are the reference; SeerDB WAL framing/checksum discipline is pattern input. No code carried over.

## 5. Index backends (v0)

- **exact**: correctness oracle, L0 scan, tiny segments, highly selective filters.
- **hnsw+f32**: default traversal backend for immutable segments (M=16, ef auto-scaled with collection size; ef_search dynamic).
- **sq8 (gated opt-in)**: calibrated scalar-quantized traversal + mandatory F32 canonical rerank, oversample ≥2k. Promoted to default only on matched recall/build/memory evidence (pattern triply validated: priors, AQR-HNSW, sqlite-vec rescore).
- All backends behind the **`VectorIndex` trait**; the store never touches HNSW concretely. Per-backend recall harness vs the exact oracle is the acceptance gate for any backend change (rejection: >0.002 absolute recall@10 regression at matched budget).
- Reproduction tracks (not v0): PAG, AQR-HNSW.

## 6. Filtered search and fusion (v0)

- Predicate eval: typed per-field indexes (equality inverted, range sorted, presence bitmaps) → single allow-list bitmap. AND=intersect, OR=union.
- Routing by selectivity estimate: tiny allowed set → exact scan; large → in-filter HNSW traversal; **exact fallback guarantees full-k** whenever enough eligible docs exist.
- Fanout rule (from CGIF theory, lifted as a planner invariant): exploration recall is determined by local satisfying fanout — connected above ~(1+e)log k, fragmented below. The planner **measures fanout and routes to exact when unmet**, rather than trusting traversal into a fragmented subgraph.
- In-filter mechanism (specified, implementation sequenced after core): **RACORN-1 ASF** — filter-failing nodes admitted as transient bridges (candidate queue only, never results), count = deficit vs `n × bridge_ratio` (default 1.0), stride-sampled for spatial diversity, failing nodes registered visited inside the ASF branch; auto-activates at selectivity ≲ 1/M. Formal cost/recall analysis + 4 datasets × 21 selectivities + production-family adoption (Weaviate/Lucene/Vespa/Qdrant ACORN lineage). RACORN-1+ exact fallback is the same guarantee as our exact-fallback rule.
- Reproduction-then-promote: **PAG** as HNSW-challenger backend (code + bench scripts available); promotes to default on verified independent evidence.
- Text module is Tantivy-shaped: immutable per-segment postings with UUID segment ids, alive-bitset deletes, compact doc-id space, background merges, snapshot readers, block-max pruning on postings. Closest production analog of our segment model — study as worked example.
- Future multivector serving shape (from Qdrant production): prefetch (BM25/dense) then rescore with unindexed quantized token vectors; no token-vector graph. Applies when multivector lands.

### Segment-native co-design (novel-opportunity thesis)

Every existing graph+partition hybrid is filtered-only (CGIF), static/rebuild-bound (ScaNN/SOAR), or bolted (per-partition graphs with no co-design). Our segment model offers a co-design no one has built: immutable segments as natural IVF partitions with per-partition graphs and a centroid router, where the partition layer *is* the mutation/lifecycle unit (churn via L0 + compaction, never rebuilds). Pursue only with ann-benchmarks harness + ablations + writeup; the engine build itself is the prerequisite regardless.
- Fusion: **RRF default** + path-wise quality assessment (weakest-link guard: a degraded path is down-weighted/excluded before fusion). DBSF later for large candidate windows.

## 7. Storage tiers and seams

- **v0 implements: RAM + mmap file segments.** One segment format, page-friendly; mmap is a provider over the same layout, not a second format.
- **NVMe seam** (contracted, not built): partitioned async disk-graph operator. Acceptance contract covers page locality, cache policy, async I/O (io_uring-class), filter handling, update policy. Named candidates: DiskANN-family baseline; Starling layout + PipeANN pipelining + OctopusANN composition.
- **Object seam** (contracted, not built): SPFresh-family hierarchical clustering + binary quantization; WAL + async-index + exact-tail lifecycle (turbopuffer-validated shape). Graph-over-object explicitly excluded.
- Seam rule: a new tier adds a conforming segment operator + provider, never a store rewrite. Logical contracts (records, manifests, query semantics, oracles) are tier-independent.

## 8. Kernel-provider seam (the Mojo lane, earned)

- Batch-granular, data-owning, versioned-ABI provider trait: scan / quantized-distance / rerank / calibration op families.
- CPU providers implemented in Rust first (SIMD parity is measured; no gap to close).
- Mojo GPU provider per-op, only on benchmark evidence (first plausible op: quantizer calibration at build time — zero query-latency risk). Consumers never install Mojo; we ship the pinned-toolchain `.so`.
- The batch-granular discipline is good kernel architecture regardless; if the lane never justifies itself the seam costs ~nothing.

## 9. Concurrency (v0)

Single-writer + snapshot readers. Readers pin generations; checkpoint reclamation only after release. Multi-writer/server sessions are a later deliberate design (embedded-first default stands unless vetoed).

## 10. Benchmark plan

- **Regression floor:** SIFT-100K, 128D, M=16, ef=100 — match-or-beat the omendb-rs reference (build ~24,881 vec/s, single ~2,324 QPS, batch ~39,905 QPS, recall@10 99.8%).
- **Beyond the floor:** modern embedding dims (384/768/1536 — behavior differs strongly by dimension, per PAG evidence); **out-of-distribution sets required** (Glass-class methods collapse to ~70% on hard OOD — in-distribution-only numbers are not citable); filtered matrix (unfiltered / 50% / 5% / <1% / correlated); cold-start + mmap page-fault accounting; insert/delete churn with recall-drift tracking; p50/p95/p99 always.
- **Honesty rules (settled, non-negotiable):** no claim without recall + dataset/config/hardware + build mode + latency percentiles. Symmetric setups; headline multipliers treated as guilty until reproduced.

## 11. Open questions for review

1. Approve this architecture → implementation sprints?
2. ann-benchmarks integration: keep the omendb-rs approach (adapted) or lighter in-tree harness first?
3. First backend set confirmed as exact + HNSW (+gated SQ8), or pull SQ8 into the default set earlier?
