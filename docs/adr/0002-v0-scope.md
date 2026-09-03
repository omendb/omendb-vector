# ADR 0002: v0 scope

Date: 2026-09-02. Decided by Nick (storage line from research evidence).

- Surface: dense + BM25 text + metadata filters + RRF hybrid. Sparse,
  multivector, relationships defer to measured demand.
- Storage: RAM + mmap file segments implemented; NVMe + object as contracted
  seams with named candidates (DiskANN-family/Starling/PipeANN; SPFresh-family
  hierarchical clustering + BQ).
- Deployment: embedded-first (single-writer + snapshot readers); server shapes
  as seams. SeerDB patterns-only, no dependency.
