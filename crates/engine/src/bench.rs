//! Benchmark harness: timing, provenance-carrying records, synthetic
//! generators, and store-level scenarios.
//!
//! Library module (not a separate bin) so tests and the future CLI
//! share the exact same measurement code — a benchmark number and an
//! acceptance number must come from the same implementation, per the
//! recall-gate rule.

pub mod scenarios;
pub mod sift;
pub mod synth;
pub mod util;

pub use scenarios::{
    bench_batch_query, bench_build, bench_checkpoint_reopen, bench_single_query, synth_smoke,
};
pub use sift::load;
pub use synth::{clustered, queries_from, uniform, SynthRng};
pub use util::{environment_string, BenchRecord, Timing};
