//! Timing collection and provenance-carrying benchmark records.
//!
//! Honesty rules (architecture §10): every reported number carries
//! its dataset, config, hardware, and build mode. `BenchRecord`
//! makes that structural — you cannot emit a number without the
//! context that makes it interpretable.

use std::time::{Duration, Instant};

/// Wall-clock sample set with percentiles.
#[derive(Debug, Clone, Default)]
pub struct Timing {
    pub samples: Vec<Duration>,
}

impl Timing {
    pub fn new() -> Self {
        Timing::default()
    }

    pub fn record(&mut self, d: Duration) {
        self.samples.push(d);
    }

    /// Run `f`, record its duration, return its output.
    pub fn time<R, F: FnOnce() -> R>(&mut self, f: F) -> R {
        let start = Instant::now();
        let out = f();
        self.record(start.elapsed());
        out
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn mean(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.samples.iter().sum();
        total / self.samples.len() as u32
    }

    /// Percentile 0..=100 (linear interpolation between order
    /// statistics; p50 with even n interpolates the middle pair).
    pub fn percentile(&self, p: u64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted: Vec<Duration> = self.samples.clone();
        sorted.sort();
        if sorted.len() == 1 {
            return sorted[0];
        }
        let rank = (p as f64 / 100.0) * (sorted.len() - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        if lo == hi {
            return sorted[lo];
        }
        let frac = rank - lo as f64;
        let lo_ns = sorted[lo].as_nanos() as f64;
        let hi_ns = sorted[hi].as_nanos() as f64;
        Duration::from_nanos((lo_ns + (hi_ns - lo_ns) * frac) as u64)
    }

    pub fn p50(&self) -> Duration {
        self.percentile(50)
    }
    pub fn p95(&self) -> Duration {
        self.percentile(95)
    }
    pub fn p99(&self) -> Duration {
        self.percentile(99)
    }
}

/// One benchmark result with mandatory provenance.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchRecord {
    /// What was measured (op name).
    pub op: String,
    /// Dataset identity (name, n, dim).
    pub dataset: String,
    /// Engine config summary (backend, metric, k, window).
    pub config: String,
    /// Hardware + build mode (debug/release, CPU string).
    pub environment: String,
    pub samples: usize,
    pub mean_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    /// Throughput where meaningful (ops/s), else 0.
    pub ops_per_sec: f64,
    /// Recall@10 vs exact oracle where meaningful, else None.
    pub recall_at_10: Option<f64>,
}

impl BenchRecord {
    pub fn from_timing(
        op: impl Into<String>,
        dataset: impl Into<String>,
        config: impl Into<String>,
        environment: impl Into<String>,
        t: &Timing,
        recall_at_10: Option<f64>,
    ) -> Self {
        let mean = t.mean();
        let ops = if mean.as_nanos() > 0 {
            1e9 / mean.as_nanos() as f64
        } else {
            0.0
        };
        BenchRecord {
            op: op.into(),
            dataset: dataset.into(),
            config: config.into(),
            environment: environment.into(),
            samples: t.len(),
            mean_us: mean.as_micros() as u64,
            p50_us: t.p50().as_micros() as u64,
            p95_us: t.p95().as_micros() as u64,
            p99_us: t.p99().as_micros() as u64,
            ops_per_sec: ops,
            recall_at_10,
        }
    }
}

/// Environment string: build profile + CPU. Deterministic within a
/// machine; cross-machine comparisons carry the string.
pub fn environment_string() -> String {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .or_else(|| {
            std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        })
        .unwrap_or_else(|| "unknown-cpu".into());
    format!("{profile}/{cpu}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_known_values() {
        let mut t = Timing::new();
        for i in 1..=100u64 {
            t.record(Duration::from_nanos(i * 100));
        }
        assert_eq!(t.len(), 100);
        assert_eq!(t.mean(), Duration::from_nanos(5050));
        assert_eq!(t.p50(), Duration::from_nanos(5050));
        // p95: rank = .95*99 = 94.05 -> 9500 + 0.05*100 = 9505ns
        let p95 = t.p95().as_nanos();
        assert_eq!(p95, 9505, "p95={p95}");
        // p99: rank = .99*99 = 98.01 -> 9900 + 0.01*100 = 9901ns
        assert_eq!(t.p99(), Duration::from_nanos(9901));
    }

    #[test]
    fn empty_timing_is_zero_everywhere() {
        let t = Timing::new();
        assert_eq!(t.mean(), Duration::ZERO);
        assert_eq!(t.p50(), Duration::ZERO);
        assert_eq!(t.p99(), Duration::ZERO);
    }

    #[test]
    fn single_sample_all_percentiles_equal() {
        let mut t = Timing::new();
        t.record(Duration::from_micros(7));
        assert_eq!(t.p50(), Duration::from_micros(7));
        assert_eq!(t.p99(), Duration::from_micros(7));
    }

    #[test]
    fn even_n_interpolates_middle() {
        let mut t = Timing::new();
        for i in 1..=2u64 {
            t.record(Duration::from_nanos(i * 1000));
        }
        assert_eq!(t.p50(), Duration::from_nanos(1500));
    }

    #[test]
    fn record_carries_provenance() {
        let mut t = Timing::new();
        t.record(Duration::from_micros(10));
        let r = BenchRecord::from_timing("op", "ds", "cfg", "env", &t, Some(0.99));
        assert_eq!(r.op, "op");
        assert_eq!(r.dataset, "ds");
        assert_eq!(r.recall_at_10, Some(0.99));
        assert!(r.ops_per_sec > 0.0);
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"dataset\":\"ds\""));
    }
}
