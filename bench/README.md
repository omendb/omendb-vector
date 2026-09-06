# Benchmarks

## Honesty rules (architecture §10)

Every number carries: dataset + config + hardware + build mode + percentiles.
`BenchRecord` enforces this structurally — see `crates/engine/src/bench/util.rs`.

## Run

```sh
cargo run --release -q --bin bench -- smoke              # synthetic, CI-safe
cargo run --release -q --bin bench -- floor bench/data/sift100k.bin
```

## SIFT-100K floor data

Fetched, never committed (`bench/data/` is gitignored):

```sh
curl -L -o bench/data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5
```

Convert once (needs `h5py` + `numpy`):

```python
import struct, h5py
f = h5py.File('bench/data/sift-128-euclidean.hdf5', 'r')
with open('bench/data/sift100k.bin', 'wb') as out:
    train = f['train'][:100000]; test = f['test'][:1000]
    out.write(struct.pack('<QQ', 100000, 128))
    out.write(train.astype('<f4').tobytes())
    out.write(struct.pack('<Q', 1000))
    out.write(test.astype('<f4').tobytes())
```

## Recorded baselines

`results/` holds JSONL runs. Current floor (SIFT-100K, release, Apple M3 Max,
HNSW M=16/M0=32/ef_construction=192/ef_search=200, k=10, L2):

| op | mean | p50 | p99 | ops/s | recall@10 |
|---|---|---|---|---|---|
| build (upsert+commit) | 454ms | — | — | 220,379 vec/s | — |
| single query (segment, HNSW) | 686us | 667us | 1,150us | 1,458 | **0.9997** |
| single query (L0 exact) | 46ms | — | — | 22 | — |
| batch (1000 q) | 574ms | — | — | 1,742 | — |

Architecture floor targets (M1 Pro estimates): build 24,881 vec/s, single 2,324
QPS at recall 99.8%, batch 39,905 QPS.

**Honest comparison:** build 8.9x above target; recall 99.97% above the 99.8%
gate; single-query QPS 1,458 vs 2,324 target (63%) — the recall-leaning
ef=200 default trades QPS for recall, and the naive beam (BinaryHeap-based,
boxed trait objects) has a clear optimization lane; batch 1,742 vs 39,905
(4.4%) — dominated by the same per-query overheads (no SIMD, no batching
amortization) and the oracle-free search path being heap-bound.

The v0 posture: correctness and provenance first; these are unoptimized
numbers with the optimization lanes named. Per-op improvements land only
with before/after runs in `results/`.

## ef_search knob

`HnswConfig::ef_search_default` (default 200) sets the recall/latency point.
Measured Pareto on SIFT-100K (release, M3 Max, k=10):

| ef | recall@10 | mean | QPS |
|---|---|---|---|
| 40 | 0.9805 | 181us | 5,517 |
| 200 | 0.9995 | 634us | 1,577 |
| 400 | 1.0000 | 1,161us | 861 |
| 800 | 1.0000 | 1,796us | 557 |

The default (200) clears the 99.8% recall gate with margin.
