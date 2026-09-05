//! Vector index backends and the trait boundary.
//!
//! Architecture §5: all backends (exact oracle, hnsw+f32, later sq8)
//! sit behind the `VectorIndex` trait; the store never touches a
//! backend concretely. Exact is the correctness oracle for every
//! other mode — a backend's acceptance gate is recall vs exact
//! (rejection: >0.002 absolute recall@10 at matched budget).
//!
//! An index is a derived artifact: built from canonical records at
//! segment-seal time (or open), bound to the segment generation it
//! indexes, rebuildable, never authoritative. Backends cover **sealed
//! segments only** — fresh writes live in L0 and are always served by
//! exact scan, so every write is instantly visible without index
//! surgery. A missing or unbuilt index fails safe: the caller falls
//! back to inline exact scan (correct, slower), never a panic.

pub mod exact;
pub mod recall;

pub use exact::ExactIndex;
pub use recall::{oracle_of, recall_at_k, RecallReport};

use crate::error::{EngineError, EngineResult};

/// Distance/similarity metric. Higher score = better everywhere (L2
/// is negated squared distance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Inner product.
    Dot,
    /// Cosine similarity; needs nonzero norms on both sides.
    Cosine,
    /// Squared L2 distance (negated for score).
    L2,
}

/// Ranked hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub external_id: u64,
    pub score: f32,
    pub seq: u64,
}

/// The minimal data a vector index backend needs: identity plus
/// full-dim vector.
#[derive(Debug, Clone)]
pub struct IndexedVector {
    pub external_id: u64,
    pub seq: u64,
    pub vector: Vec<f32>,
}

impl IndexedVector {
    pub fn new(external_id: u64, seq: u64, vector: Vec<f32>) -> Self {
        IndexedVector {
            external_id,
            seq,
            vector,
        }
    }
}

/// Ordered score over the query-prefix of the stored vector; higher =
/// better. Cosine against a zero-norm side is a schema error (the
/// caller decides whether that excludes the record or fails).
pub(crate) fn score(metric: Metric, q: &[f32], v: &[f32]) -> EngineResult<f32> {
    debug_assert!(q.len() <= v.len(), "query dim validated by caller");
    let v = &v[..q.len()];
    Ok(match metric {
        Metric::Dot => q.iter().zip(v).map(|(a, b)| a * b).sum(),
        Metric::Cosine => {
            let qn = crate::records::l2_norm(q);
            let vn = crate::records::l2_norm(v);
            if qn == 0.0 || vn == 0.0 {
                return Err(EngineError::Schema("cosine of zero-norm vector".into()));
            }
            let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
            dot / (qn * vn)
        }
        Metric::L2 => {
            let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            -d
        }
    })
}

/// Validate a query against a stored dim: non-empty, prefix-dim only,
/// cosine queries must have nonzero norm. Backends call this first so
/// the trait is safe to use directly (the store also validates — the
/// logical contract lives there).
pub(crate) fn validate_query(metric: Metric, query: &[f32], stored_dim: u32) -> EngineResult<()> {
    if query.is_empty() {
        return Err(EngineError::Schema("empty query vector".into()));
    }
    if stored_dim != 0 && query.len() as u32 > stored_dim {
        return Err(EngineError::Schema(format!(
            "query dim {} exceeds collection dim {}",
            query.len(),
            stored_dim
        )));
    }
    if metric == Metric::Cosine && crate::records::l2_norm(query) == 0.0 {
        return Err(EngineError::Schema("cosine of zero query vector".into()));
    }
    Ok(())
}

/// One vector index backend over one sealed segment's live set.
pub trait VectorIndex: Send + Sync {
    /// Backend name (diagnostics, recall reports).
    fn name(&self) -> &'static str;
    /// Number of indexed vectors.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Top-k search, hits sorted best-first. `query` may be a prefix
    /// of the stored dim. Zero-norm stored vectors under cosine are
    /// excluded, not fatal.
    fn search(&self, metric: Metric, query: &[f32], k: usize) -> EngineResult<Vec<Hit>>;
}
