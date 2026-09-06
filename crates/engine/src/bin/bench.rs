//! Benchmark CLI: runs scenarios and prints JSON records.
//!
//! Usage:
//!   bench smoke                 — synthetic CI-safe smoke (default)
//!   bench floor <sift-dir>     — SIFT-100K floor (needs prepared data dir)
//!
//! Output: one JSON object per line (BenchRecord) — machine-readable,
//! provenance embedded per record.

use omendb_vector_engine::bench;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("smoke");
    let out = std::io::stdout();
    let mut w = std::io::BufWriter::new(out.lock());

    match mode {
        "smoke" => {
            let dir = std::env::temp_dir().join(format!("omendb-bench-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("tmpdir");
            for r in bench::synth_smoke(&dir) {
                serde_json::to_writer(&mut w, &r).unwrap();
                writeln!(w).unwrap();
            }
            std::fs::remove_dir_all(&dir).ok();
        }
        "floor" => {
            let data = args
                .get(2)
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from("bench/data/sift100k.bin"));
            if !data.exists() {
                eprintln!(
                    "sift data not found at {} — fetch and convert first (see bench/README)",
                    data.display()
                );
                std::process::exit(2);
            }
            let ds = bench::load(&data).expect("load sift");
            eprintln!(
                "floor: {} train x {} dim, {} queries",
                ds.train.len(),
                ds.train.first().map(|r| r.vector.len()).unwrap_or(0),
                ds.queries.len()
            );
            let dir = std::env::temp_dir().join(format!("omendb-floor-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("tmpdir");
            let label = format!(
                "sift-100k d={} nq={}",
                ds.queries.first().map(|q| q.len()).unwrap_or(0),
                ds.queries.len()
            );
            let hnsw = omendb_vector_engine::store::IndexBackend::Hnsw(
                omendb_vector_engine::index::HnswConfig::default(),
            );
            let build_rec = bench::bench_build(&dir.join("b"), &ds.train, &label, &hnsw);
            {
                serde_json::to_writer(&mut w, &build_rec).unwrap();
                writeln!(w).unwrap();
            }
            for r in bench::bench_single_query(
                &dir.join("q"),
                &ds.train,
                &ds.queries,
                10,
                omendb_vector_engine::index::Metric::L2,
                &label,
                &hnsw,
            ) {
                serde_json::to_writer(&mut w, &r).unwrap();
                writeln!(w).unwrap();
            }
            let batch_rec = bench::bench_batch_query(
                &dir.join("t"),
                &ds.train,
                &ds.queries,
                10,
                omendb_vector_engine::index::Metric::L2,
                &label,
                &hnsw,
            );
            {
                serde_json::to_writer(&mut w, &batch_rec).unwrap();
                writeln!(w).unwrap();
            }
            std::fs::remove_dir_all(&dir).ok();
        }
        other => {
            eprintln!("unknown mode {other:?}; use: smoke | floor <sift-dir>");
            std::process::exit(2);
        }
    }
}
