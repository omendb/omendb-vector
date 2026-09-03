# Evidence Register — Fresh SOTA Pass, September 2026

Language-neutral research backing the fresh Rust engine design. Priors (Apr–Jul 2026 monorepo docs) are the baseline; this register records what the **2025–26 literature and production systems** add or change. Every entry: finding → v0 implication → sources.

## 1. Graph ANN: HNSW still the default; PAG is the challenger to watch

- **PAG (Projection-Augmented Graph, arxiv 2603.06660, Mar 2026):** up to **5× HNSW QPS-recall** on modern datasets (384–3072d, text/image/multimodal), 20–40% of HNSW build time, robustness across K and dimensionality. Strongest HNSW challenger on current evidence. Status: single paper, needs independent reproduction.
- **AQR-HNSW (arxiv 2602.21600, Feb 2026):** density-aware adaptive quantization + multi-stage rerank, 2.5–3.3× HNSW QPS at 98%+ recall, 4× compression. Independently validates the **quantized-traversal + exact-rerank** pattern as the local performance shape.
- **VSAG (VLDB, Ant Group):** production framework — prefetch + contiguous neighbor copies (L3 miss 94%→39%), **automated parameter tuning** (60h brute-force → 2–3× build time), SQ + selective rerank. Two portable lessons: cache layout work and auto-defaults beat new algorithms for shipped performance.
- **v0 implication:** exact flat + **HNSW+F32** stays the default backend; calibrated SQ8+rerank as gated opt-in. PAG and AQR-HNSW are reproduction tracks, not v0 backends. VSAG's auto-tuning lesson supports the performance-as-default pillar (ef auto-scale with collection size).

## 2. Filtered search: selectivity routing confirmed; RACORN-1 fixes the low end

- **RACORN-1 (arxiv 2607.00768, Jul 2026):** in-place ACORN-1 extension. Fixes ACORN-1's recall collapse below 5% (collapse to 0.03–0.10 at 0.3% → recovered to 0.77–0.98) via adaptive bridge fallback; **RACORN-1+** adds adaptive exact fallback → recall 1.00 with 20–75× speedup at ≤0.1% selectivity. Directly closes the known ACORN low-selectivity hole.
- **Query-aware routing (arxiv 2606.19898):** no single filtered-ANN method dominates across datasets/predicates; per-query ML routing over methods wins. Validates selectivity/correlation-aware routing as the durable principle; the ML router itself is future work.
- **v0 implication:** bitmap predicate → selectivity estimate → **exact scan vs in-filter HNSW**, with exact fallback guaranteeing full-k (RACORN-1+'s AEF is the same idea, independently derived). RACORN-1 is the named upgrade track for the low-selectivity regime. No method hard-coded as "the" filtered path.

## 3. Quantization: RaBitQ healthy; the pattern matters more than the codec

- **RaBitQ vs TurboQuant symmetric comparison (alphaxiv 2604.19528, Apr 2026):** RaBitQ wins most settings; several TurboQuant speedup claims did not reproduce under symmetric setup. Cautionary tale for headline numbers — symmetric, reproducible setups only.
- **GPU-native IVF-RaBitQ (VLDB vol.19):** cluster methods expose GEMM-friendly regular compute → natural GPU fit; RaBitQ needs no codebook training, has error bounds. Confirms: **IVF-family is the GPU-friendly shape; graph-family is the CPU-local shape.** Direct support for the kernel-seam split (CPU graph now, GPU cluster later).
- **v0 implication:** SQ8 traversal + F32 canonical rerank (settled pattern, now triply validated: our priors, AQR-HNSW, sqlite-vec rescore below). RaBitQ as research track; codec choice stays behind the traversal/rerank contract, never in the query API.

## 4. NVMe tier: page/cache/async dominate the named algorithm (seam, not v0)

- **OctopusANN (VLDB vol.19):** I/O is **70–90% of query latency** on SSD; memory-resident navigation + dynamic width strongest standalone; page-shuffle + page-search complementary; composition beats Starling 4–38% and DiskANN 87–150% at recall@10=90%.
- **PipeANN (OSDI'25):** io_uring + SQ polling + compute/I/O pipelining → 39–48% of DiskANN/Starling latency at 0.9 recall; beats SPANN by ~70% at recall≥0.9.
- **v0 implication:** NVMe operator is a **designed seam with named candidates** (DiskANN-family baseline; Starling layout + PipeANN pipelining + OctopusANN composition as the evaluated set), not v0 code. The seam's acceptance contract must cover page locality, cache policy, and async I/O — the things that actually decide the tier.

## 5. Sparse retrieval (deferred to phase 2; design input recorded)

- **BMP (block-max pruning)** is the modern engine for learned-sparse indexes (2–60× over MaxScore/BMW on SPLADE-family).
- **Cross-engine pruning study (arxiv 2608.16309, Aug 2026):** index-side static pruning is portable (1.2–6.6× latency, 18–82% smaller — sparse retrieval is memory-bound); query pruning is already internalized by modern engines (BMP's β, SEISMIC's query_cut); static + dynamic compose (2.5× at NDCG within 0.003).
- **v0 implication:** none (sparse deferred per v0 scope). Phase-2 design input: inverted index + BMP-style dynamic pruning; encoders stay external (we index weights, we don't train models).

## 6. Multivector (deferred; TACHIOM replaces MUVERA as the named candidate)

- **TACHIOM (arxiv 2604.28142, Apr 2026):** token-aware clustering (247× faster than k-means at 262k centroids), HNSW over centroids + PQ residuals + exact MaxSim refine; 2.5–9.8× retrieval speedup at comparable effectiveness — **without model retraining**. This is the first approximate backend that survives the FDE/"Party is Over" caveat.
- **MUVERA** stands as the FDE baseline with guarantees, but the STE-retraining requirement keeps it out as a general engine-side default.
- **v0 implication:** none (multivector deferred). The multivector strategy updates: **exact MaxSim** documented path; **TACHIOM** named approximate candidate; MUVERA demoted to baseline reference.

## 7. Object tier (seam, not v0): turbopuffer validates the whole shape

- **Turbopuffer architecture + ANN v3 (primary production source):** SPFresh **centroid-based** index (not graph — roundtrip-bounded tree height instead of dependent walks), hierarchical clustering (100× branching, DRAM/SSD-matched), binary quantization, per-namespace **WAL on object storage**, async indexing with **exhaustive-search tail** for fresh data, exact metadata indexes. Cold p50 ~500–874ms at 1M, warm 14ms; ANN v3 to 100B vectors.
- **v0 implication:** the object seam names **SPFresh-family hierarchical clustering + BQ**. Critically, turbopuffer's WAL + async-index + exact-tail is the same L0+segments shape as our local design — one lifecycle model spans all three tiers. Graph-over-object-storage stays explicitly excluded (matches priors).

## 8. Hybrid fusion: RRF stays default; add the weakest-link guard

- **"Balancing the Blend" (VLDB vol.19, Infinity framework, 11 datasets):** **weakest-link phenomenon** — a weak path degrades fused results; no single hybrid method wins accuracy+efficiency+cost jointly.
- **Production comparison (Aug 2026):** RRF (no calibration, blind to score margins) vs RSF/min-max (outlier compression) vs DBSF (needs N≥30 window; Qdrant's choice).
- **v0 implication:** **RRF stays the default** (prior practice + omendb-rs); add **path-wise quality assessment** before fusion (weakest-link guard); DBSF as the later option when candidate windows are large. Late-interaction (exact MaxSim) stays the documented rerank path.

## 9. Competitive corrections (primary-source verified, Sep 2026)

- **sqlite-vec now HAS ANN (alpha):** v0.1.10-alpha.1+ (Mar–May 2026) ships `rescore` (quantize → oversample → rerank, 5.8× at 0.988 recall@10 on 1M×1024d) plus DiskANN; stable line still brute-force-only. Their rescore is our traversal+rerank pattern — validates the design, removes any "only we do this" framing.
- **DuckDB vss durability gap confirmed worse than reported:** official docs admit the WAL-recovery hole; issue #81 documents repeated corruption incidents and workarounds. Strongest third-party evidence for the durability wedge.
- **Mojo 1.0 facts (for the language record):** compiler fully open (Apache 2.0, Aug 2026) but contributions gated; source-stable only, **no ABI stability**; small stable stdlib set; no robust async; no native Windows. CPU kernel parity with Rust confirmed (single-microbenchmark + ORNL GPU study: ~87% of CUDA memory-bound).

## 10. Deep-pass additions (Sep 2026: full-paper reads + code checks)

- **CGIF (VLDB'26, graph+IVF co-design for filtered search):** HNSW navigation phase unchanged; exploration phase draws from HNSW neighbors + two-hop + IVF cluster augmentation with per-cluster cursors. MNNG theory (satisfying-subgraph equivalence) + local-satisfying-fanout threshold. Laion: 1600 QPS @ 0.95 recall (+34% over next best); Wiki-Range 2×; 677 distance comps vs 925 (NaVix) / 20000 (Faiss-IVF) at 0.80 recall. O((M+1)n) storage.
- **Production IVF+HNSW hybrids:** LanceDB ships `IVF_HNSW_FLAT/SQ/PQ` (IVF partitions, per-partition HNSW) + `IVF_RQ` (RaBitQ-style, 1/32 size, preferred for filtered workloads); FAISS guidance: HNSW as coarse quantizer attractive (fewer params than IVFPQFastScan).
- **SOAR (Google, spillover assignment):** orthogonality-amplified residual loss for redundant partition assignment — directly attacks clustering accuracy loss; SOTA ANN benchmark results with fast build, low memory.
- **RACORN-1 (full read):** Naver Corp; ASF (filter-failing nodes as transient bridges) + stride sampling + RACORN-1+ exact fallback. Formal cost/recall analysis under random-graph model; evaluated 4 datasets × 21 selectivities. Mechanism is traversal-level — applies to any graph backend (NSG named), not HNSW-only.
- **TACHIOM (full abs + code check):** SIGIR'26, 6pp. TAC clustering + HNSW-over-centroids gather + PQ-residual MaxSim refine. Code: github.com/TusKANNy/tachiom (**Rust + PyO3**, MIT, PyPI `tachiom` v0.3.4, SIGIR repro configs in-repo). Easiest reproduction track we have — same language, permissive license (study, don't vendor blindly).
- **PAG (code check):** github.com/KejingLu-810/PAG/ (C++/Python, AVX-512 kernels, bench + repro scripts). Reproducible challenger.
- **MUVERA status (LightOn, Jun 2026):** root cause confirmed (anisotropic token cone, cosine ~0.95 → random projections degenerate); **STE regularization fixes it** — model lightonai/LateOn-regularized on HF; transfers across methods/seeds; PLAID preserved. MUVERA viable *with model co-design only*; dead on arbitrary user models.
- **RaBitQ vindicated:** symmetric comparison (Gao et al., Apr 2026) — RaBitQ beats TurboQuant most settings; TurboQuant's 8× claim traced to asymmetric setup, some results unreproducible. GPU-native IVF-RaBitQ (VLDB) + LanceDB IVF_RQ = production validation. Our Mojo `rabitq.mojo` is algorithm reference.
- **LeanVec/LVQ (Intel SVS, production):** LVQ4x8 two-level (4-bit traversal + 8-bit rerank) mirrors our traversal+rerank contract; LeanVec adds dim-reduction for 768d+. OOD variant handles query/db distribution shift. Evaluation track alongside SQ8/RaBitQ.

### Thoroughness audit (honest)

- Pass 1: 8 areas × 1 search step, excerpt-level, zero paper reads. Survey-grade.
- Pass 2 (this): 3 full reads (RACORN-1 HTML, TACHIOM/PAG/RaBitQ-note abs+code pages) + 6 deep searches. Claim-level verification on the decision-critical points (MUVERA viability, code availability, CGIF mechanism).
- Not yet done: full PDF reads of PAG/TACHIOM/CGIF/OctopusANN/PipeANN; reproduction runs; code audits. Required before any 'we beat X' claim or novel-work writeup.

## What this changes vs the priors

1. Storage scope: **RAM + mmap in v0; NVMe and object as contracted seams** (§4, §7 name the candidates and acceptance contracts).
2. Multivector candidate: **TACHIOM in, MUVERA demoted** to baseline (§6).
3. Filtered routing: exact-fallback guarantee + **RACORN-1 as the named low-selectivity upgrade** (§2).
4. Fusion: **weakest-link guard** added to RRF default (§8).
5. New reproduction tracks: **PAG, AQR-HNSW** (§1). Challenge queue for HNSW, not v0 backends.
