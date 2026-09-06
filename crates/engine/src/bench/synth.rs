//! Deterministic synthetic vector generators for benchmark smoke
//! runs (CI-safe: no network, no downloads).
//!
//! Two shapes matter for honest measurement:
//! - `uniform`: iid coordinates — the *hard* case for ANN (no
//!   cluster structure to exploit) and a useful lower bound.
//! - `clustered`: k Gaussian clusters — resembles real embedding
//!   geometry and gives the recall picture ANN methods advertise.

use crate::records::Record;

/// splitmix64 — same PRNG as HNSW level assignment (deterministic
/// across platforms; floats via 53-bit mantissa).
pub struct SynthRng(u64);

impl SynthRng {
    pub fn new(seed: u64) -> Self {
        SynthRng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Uniform iid vectors in [0,1)^dim, ids 1..=n.
pub fn uniform(n: usize, dim: usize, seed: u64) -> Vec<Record> {
    let mut rng = SynthRng::new(seed);
    (1..=n as u64)
        .map(|i| Record::new(i, (0..dim).map(|_| rng.next_f32()).collect()))
        .collect()
}

/// `k` Gaussian clusters (std `sigma`), ids 1..=n; queries drawn from
/// the same clusters land near real points.
pub fn clustered(n: usize, dim: usize, clusters: usize, sigma: f32, seed: u64) -> Vec<Record> {
    let mut rng = SynthRng::new(seed);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect();
    (1..=n as u64)
        .map(|i| {
            let c = &centers[(i as usize) % clusters];
            let v: Vec<f32> = c.iter().map(|&x| x + sigma * rng.next_normal()).collect();
            Record::new(i, v)
        })
        .collect()
}

/// Query set: points perturbed from the data distribution (same
/// generator stream offset), so recall numbers are meaningful rather
/// than trivially-1 (exact points) or worst-case (random far away).
pub fn queries_from(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Reuse `clustered` shapes: a query is a cluster point + noise.
    // Kept as raw vectors (queries are not records).
    let mut rng = SynthRng::new(seed);
    let one = clustered(1, dim, 1, 0.05, seed);
    let base = &one[0].vector;
    (0..n)
        .map(|_| {
            base.iter()
                .map(|&x| x + 0.1 * rng.next_normal())
                .collect::<Vec<f32>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let a = uniform(10, 4, 42);
        let b = uniform(10, 4, 42);
        assert_eq!(a, b);
        let c = uniform(10, 4, 43);
        assert_ne!(a, c);
    }

    #[test]
    fn clustered_stays_finite_and_dim() {
        let rs = clustered(50, 8, 5, 0.1, 7);
        assert_eq!(rs.len(), 50);
        for r in rs {
            assert_eq!(r.vector.len(), 8);
            assert!(r.vector.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn uniforms_in_unit_range() {
        let rs = uniform(100, 3, 1);
        for r in rs {
            assert!(r.vector.iter().all(|x| (0.0..1.0).contains(x)));
        }
    }

    #[test]
    fn queries_well_formed() {
        let qs = queries_from(5, 16, 9);
        assert_eq!(qs.len(), 5);
        for q in qs {
            assert_eq!(q.len(), 16);
            assert!(q.iter().all(|x| x.is_finite()));
        }
    }
}
