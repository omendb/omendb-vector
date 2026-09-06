//! Store-level benchmark scenarios: the ops the harness measures,
//! reusable by the CLI runner and tests.
//!
//! Every scenario returns `BenchRecord`s with provenance filled in
//! and recall computed against the store's own exact oracle — the
//! same `recall.rs` code path as the acceptance gates.

use super::synth;
use super::{environment_string, BenchRecord, Timing};
use crate::index::Metric;
use crate::store::{IndexBackend, Store};
use std::path::Path;

/// Fresh store per scenario (its own tmp dir, caller decides lifetime).
fn fresh_store(dir: &Path, backend: IndexBackend) -> Store {
    let (s, _) = Store::open_with(dir, backend).expect("open store");
    s
}

/// Build: n upserts + one commit. One timing sample (total wall).
pub fn bench_build(
    dir: &Path,
    records: &[crate::records::Record],
    dataset_label: &str,
    backend: &IndexBackend,
) -> BenchRecord {
    let mut s = fresh_store(dir, backend.clone());
    let mut t = Timing::new();
    t.time(|| {
        for r in records {
            s.upsert(r.clone()).expect("upsert");
        }
        s.commit().expect("commit");
    });
    let n = records.len();
    let config = format!("{backend:?}");
    let mut rec = BenchRecord::from_timing(
        "build(upsert+commit)",
        dataset_label,
        config,
        environment_string(),
        &t,
        None,
    );
    rec.ops_per_sec = n as f64 / t.mean().as_secs_f64().max(1e-9);
    rec
}

/// Single-query latency: `queries.len()` searches, k each.
pub fn bench_single_query(
    dir: &Path,
    records: &[crate::records::Record],
    queries: &[Vec<f32>],
    k: usize,
    metric: Metric,
    dataset_label: &str,
    backend: &IndexBackend,
) -> Vec<BenchRecord> {
    let mut s = fresh_store(dir, backend.clone());
    for r in records {
        s.upsert(r.clone()).expect("upsert");
    }
    s.commit().expect("commit");
    // Warmup: JIT-free but cache/JIT-of-nothing still matters — page in
    // the graph/segment, prime allocator, spin up turbo. Untimed.
    for q in queries.iter().take(queries.len().min(20)) {
        let _ = s.exact_top_k(metric, q, k).expect("warmup query");
    }
    // Multi-pass: run the full query set REPEAT times so percentiles
    // and the mean carry real variance (single-pass means proved
    // noisy: 497-808us run-to-run on the same machine).
    const PASSES: usize = 3;
    let mut t_l0 = Timing::new();
    for q in queries {
        t_l0.time(|| {
            s.exact_top_k(metric, q, k).expect("query");
        });
    }
    s.checkpoint().expect("checkpoint");
    let mut t_seg = Timing::new();
    for _ in 0..PASSES {
        for q in queries {
            t_seg.time(|| {
                s.search(metric, q, k, k).expect("query");
            });
        }
    }
    // Recall vs oracle on a bounded query subset (the oracle is
    // O(n) per query; 200 queries give a tight estimate without
    // minutes of brute force).
    let recall_queries = &queries[..queries.len().min(200)];
    let mut recall_sum = 0.0f64;
    for q in recall_queries {
        let got = s.search(metric, q, k, k).expect("query");
        let want = s.exact_top_k(metric, q, k).expect("oracle");
        let want_ids: std::collections::HashSet<u64> = want.iter().map(|h| h.external_id).collect();
        let overlap = got
            .iter()
            .filter(|h| want_ids.contains(&h.external_id))
            .count();
        recall_sum += overlap as f64 / k.min(want.len()).max(1) as f64;
    }
    let recall = recall_sum / recall_queries.len() as f64;
    vec![
        BenchRecord::from_timing(
            "single_query_l0_exact",
            dataset_label,
            format!("{backend:?} {metric:?} k={k}"),
            environment_string(),
            &t_l0,
            None,
        ),
        BenchRecord::from_timing(
            "single_query_segment",
            dataset_label,
            format!("{backend:?} {metric:?} k={k}"),
            environment_string(),
            &t_seg,
            Some(recall),
        ),
    ]
}

/// Batch throughput: all queries served back-to-back, measured as
/// total time for Q queries (report ops/s from mean).
pub fn bench_batch_query(
    dir: &Path,
    records: &[crate::records::Record],
    queries: &[Vec<f32>],
    k: usize,
    metric: Metric,
    dataset_label: &str,
    backend: &IndexBackend,
) -> BenchRecord {
    let mut s = fresh_store(dir, backend.clone());
    for r in records {
        s.upsert(r.clone()).expect("upsert");
    }
    s.commit().expect("commit");
    s.checkpoint().expect("checkpoint");
    // Warmup + multi-pass batch: single-pass batch numbers proved
    // noisy (455-525ms on identical config); PASSES batches amortize
    // the per-run variance into a mean.
    for q in queries.iter().take(queries.len().min(20)) {
        let _ = s.search(metric, q, k, k).expect("warmup query");
    }
    let mut t = Timing::new();
    let mut all_hits = 0usize;
    const PASSES: usize = 3;
    t.time(|| {
        for _ in 0..PASSES {
            for q in queries {
                all_hits += s.search(metric, q, k, k).expect("query").len();
            }
        }
    });
    assert!(all_hits > 0);
    let mut rec = BenchRecord::from_timing(
        "batch_query",
        dataset_label,
        format!("{backend:?} {metric:?} k={k} nq={}", queries.len()),
        environment_string(),
        &t,
        None,
    );
    rec.ops_per_sec = queries.len() as f64 * PASSES as f64 / t.mean().as_secs_f64().max(1e-9);
    rec
}

/// Checkpoint + reopen cost at n records.
pub fn bench_checkpoint_reopen(
    dir: &Path,
    records: &[crate::records::Record],
    dataset_label: &str,
    backend: &IndexBackend,
) -> Vec<BenchRecord> {
    let mut s = fresh_store(dir, backend.clone());
    for r in records {
        s.upsert(r.clone()).expect("upsert");
    }
    s.commit().expect("commit");
    let mut t_cp = Timing::new();
    t_cp.time(|| {
        s.checkpoint().expect("checkpoint");
    });
    let backend2 = backend.clone();
    let mut t_re = Timing::new();
    t_re.time(|| {
        drop(s);
        let (_, _) = Store::open_with(dir, backend2).expect("reopen");
    });
    vec![
        BenchRecord::from_timing(
            "checkpoint",
            dataset_label,
            format!("{backend:?}"),
            environment_string(),
            &t_cp,
            None,
        ),
        BenchRecord::from_timing(
            "reopen",
            dataset_label,
            format!("{backend:?}"),
            environment_string(),
            &t_re,
            None,
        ),
    ]
}

/// Full synthetic smoke scenario — CI-safe, no network.
pub fn synth_smoke(dir: &Path) -> Vec<BenchRecord> {
    let n = 2_000;
    let dim = 32;
    let dataset = format!("synthetic-clustered n={n} d={dim} clusters=16");
    let records = synth::clustered(n, dim, 16, 0.05, 1234);
    let queries = synth::queries_from(100, dim, 5678);
    let mut out = Vec::new();
    out.push(bench_build(
        &dir.join("build-exact"),
        &records,
        &dataset,
        &IndexBackend::Exact,
    ));
    let hnsw = IndexBackend::Hnsw(crate::index::HnswConfig::default());
    out.extend(bench_single_query(
        &dir.join("q-hnsw"),
        &records,
        &queries,
        10,
        Metric::L2,
        &dataset,
        &hnsw,
    ));
    out.push(bench_batch_query(
        &dir.join("b-hnsw"),
        &records,
        &queries,
        10,
        Metric::L2,
        &dataset,
        &hnsw,
    ));
    out.extend(bench_checkpoint_reopen(
        &dir.join("cp-hnsw"),
        &records,
        &dataset,
        &hnsw,
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir()
            .join("omendb-bench-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn smoke_scenario_produces_records() {
        let dir = tmp("smoke");
        let recs = synth_smoke(&dir);
        // build + 2 single + batch + checkpoint + reopen = 6 records
        assert!(recs.len() >= 6, "got {}", recs.len());
        for r in &recs {
            assert!(!r.dataset.is_empty());
            assert!(!r.environment.is_empty());
            assert!(r.samples >= 1);
            assert!(serde_json::to_string(r).is_ok());
        }
        // The segment query record must carry a recall value.
        let seg = recs
            .iter()
            .find(|r| r.op == "single_query_segment")
            .expect("segment record");
        let recall = seg.recall_at_10.expect("recall present");
        assert!(recall >= 0.9, "smoke recall {recall}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
