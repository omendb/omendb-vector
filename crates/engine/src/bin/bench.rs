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
        other => {
            eprintln!("unknown mode {other:?}; use: smoke | floor <sift-dir>");
            std::process::exit(2);
        }
    }
}
