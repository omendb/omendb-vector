//! Recall harness: every backend's acceptance gate (architecture §5).
//!
//! Rule: a backend change is rejected if recall@k vs the exact oracle
//! drops by more than `0.002` absolute at matched search budget.
//! The harness compares a candidate backend's top-k against the exact
//! flat oracle over the same vector set, for many queries, and
//! reports per-query and aggregate recall.
//!
//! This module is test/diagnostic infrastructure: it lives in the
//! library (not tests/) so benchmarks and future gate scripts reuse
//! the identical computation — a benchmark number and an acceptance
//! number must come from the same code.

use crate::index::{ExactIndex, IndexedVector, Metric, VectorIndex};

/// Result of one recall run.
#[derive(Debug, Clone)]
pub struct RecallReport {
    pub backend: &'static str,
    pub metric: Metric,
    pub queries: usize,
    pub k: usize,
    /// Mean recall@k over all queries.
    pub mean_recall: f64,
    /// Worst per-query recall@k.
    pub min_recall: f64,
}

impl RecallReport {
    /// Would this report pass the acceptance gate?
    pub fn passes_gate(&self, threshold: f64) -> bool {
        self.mean_recall >= 1.0 - threshold
    }
}

/// Compute recall@k of `candidate` against `oracle` for `queries`
/// (each query is a full-dim or prefix-dim vector).
///
/// Recall@k = |top_k(candidate) ∩ top_k(oracle)| / k using external
/// ids. Ties may cost fractionally; that is the conservative choice
/// (never inflates recall).
pub fn recall_at_k(
    candidate: &dyn VectorIndex,
    oracle: &dyn VectorIndex,
    metric: Metric,
    queries: &[Vec<f32>],
    k: usize,
) -> crate::error::EngineResult<RecallReport> {
    let backend = candidate.name();
    let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());
    for q in queries {
        let want = oracle.search(metric, q, k)?;
        let got = candidate.search(metric, q, k)?;
        let want_ids: std::collections::HashSet<u64> = want.iter().map(|h| h.external_id).collect();
        let overlap = got
            .iter()
            .filter(|h| want_ids.contains(&h.external_id))
            .count();
        let denom = k.min(want.len()).max(1);
        recalls.push(overlap as f64 / denom as f64);
    }
    let mean = if recalls.is_empty() {
        1.0 // vacuous: no queries, nothing to recall
    } else {
        recalls.iter().sum::<f64>() / recalls.len() as f64
    };
    let min = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
    Ok(RecallReport {
        backend,
        metric,
        queries: queries.len(),
        k,
        mean_recall: mean,
        min_recall: if min.is_infinite() { 1.0 } else { min },
    })
}

/// Convenience: build the oracle from the same vectors as a candidate.
pub fn oracle_of(vectors: &[IndexedVector]) -> ExactIndex {
    ExactIndex::build(vectors.iter().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(id: u64, seq: u64, v: Vec<f32>) -> IndexedVector {
        IndexedVector::new(id, seq, v)
    }

    #[test]
    fn identical_backend_scores_full_recall() {
        let vectors = vec![
            iv(1, 1, vec![0.1, 0.9]),
            iv(2, 2, vec![0.5, 0.5]),
            iv(3, 3, vec![0.9, 0.1]),
        ];
        let oracle = oracle_of(&vectors);
        let twin = ExactIndex::build(vectors.clone());
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let report = recall_at_k(&twin, &oracle, Metric::Dot, &queries, 2).unwrap();
        assert!((report.mean_recall - 1.0).abs() < 1e-9);
        assert!((report.min_recall - 1.0).abs() < 1e-9);
        assert!(report.passes_gate(0.002));
    }

    #[test]
    fn half_wrong_backend_scores_half_recall() {
        // Candidate ranks 2 and 3; oracle's top-2 are 1 and 2 → overlap {2} = 1/2.
        let vectors = vec![
            iv(1, 1, vec![1.0, 0.0]),
            iv(2, 2, vec![0.8, 0.2]),
            iv(3, 3, vec![0.6, 0.4]),
        ];
        let oracle = oracle_of(&vectors);
        // Degraded twin: drop vector 1 (recall of id-2-neighboring set).
        let degraded = ExactIndex::build(vec![iv(2, 2, vec![0.8, 0.2]), iv(3, 3, vec![0.6, 0.4])]);
        let queries = vec![vec![1.0, 0.0]];
        let report = recall_at_k(&degraded, &oracle, Metric::Dot, &queries, 2).unwrap();
        assert!((report.mean_recall - 0.5).abs() < 1e-9);
        assert!(!report.passes_gate(0.002));
    }

    #[test]
    fn empty_queries_report_is_not_nan() {
        let vectors = vec![iv(1, 1, vec![1.0])];
        let oracle = oracle_of(&vectors);
        let twin = ExactIndex::build(vectors.clone());
        let report = recall_at_k(&twin, &oracle, Metric::Dot, &[], 2).unwrap();
        assert_eq!(report.queries, 0);
        assert!((report.mean_recall - 1.0).abs() < 1e-9); // vacuous pass, no NaN
    }

    #[test]
    fn k_zero_is_not_div_zero() {
        let vectors = vec![iv(1, 1, vec![1.0])];
        let oracle = oracle_of(&vectors);
        let twin = ExactIndex::build(vectors.clone());
        let report = recall_at_k(&twin, &oracle, Metric::Dot, &[vec![1.0]], 0).unwrap();
        assert!(report.mean_recall.is_finite());
    }
}
