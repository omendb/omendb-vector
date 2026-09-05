//! Exact flat backend: brute-force scan, the correctness oracle.
//!
//! Serves three roles (architecture §5): the recall oracle for every
//! other backend, the L0/fresh-write path, and the query path for
//! tiny segments and highly selective filters where brute force is
//! simply the best plan. O(n·d) per query, no build step, indexes
//! nothing — it stores its input so the trait is self-contained.

use super::{score, validate_query, Hit, IndexedVector, Metric, VectorIndex};
use crate::error::EngineResult;

/// Flat array of indexed vectors; scan on search.
#[derive(Debug, Default)]
pub struct ExactIndex {
    vectors: Vec<IndexedVector>,
}

impl ExactIndex {
    pub fn new() -> Self {
        ExactIndex::default()
    }

    /// Build from an iterator of live records.
    pub fn build<I: IntoIterator<Item = IndexedVector>>(vectors: I) -> Self {
        ExactIndex {
            vectors: vectors.into_iter().collect(),
        }
    }

    /// Indexed vectors (for tooling/diagnostics).
    pub fn vectors(&self) -> &[IndexedVector] {
        &self.vectors
    }
}

impl VectorIndex for ExactIndex {
    fn name(&self) -> &'static str {
        "exact"
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn search(&self, metric: Metric, query: &[f32], k: usize) -> EngineResult<Vec<Hit>> {
        validate_query(metric, query, self.stored_dim())?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut hits: Vec<Hit> = self
            .vectors
            .iter()
            .filter_map(|iv| {
                score(metric, query, &iv.vector).ok().map(|s| Hit {
                    external_id: iv.external_id,
                    score: s,
                    seq: iv.seq,
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }
}

impl ExactIndex {
    /// Max stored dim for validation (0 when empty: no constraint yet).
    fn stored_dim(&self) -> u32 {
        self.vectors
            .first()
            .map(|v| v.vector.len() as u32)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(id: u64, seq: u64, v: Vec<f32>) -> IndexedVector {
        IndexedVector::new(id, seq, v)
    }

    #[test]
    fn exact_matches_bruteforce_ordering() {
        let idx = ExactIndex::build([
            iv(1, 1, vec![1.0, 0.0]),
            iv(2, 2, vec![0.9, 0.1]),
            iv(3, 3, vec![0.0, 1.0]),
            iv(4, 4, vec![-1.0, 0.0]),
        ]);
        let hits = idx.search(Metric::Dot, &[1.0, 0.0], 4).unwrap();
        let ids: Vec<u64> = hits.iter().map(|h| h.external_id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn k_truncation_and_k_zero() {
        let idx = ExactIndex::build([iv(1, 1, vec![1.0]), iv(2, 2, vec![0.5])]);
        assert_eq!(idx.search(Metric::Dot, &[1.0], 0).unwrap().len(), 0);
        assert_eq!(idx.search(Metric::Dot, &[1.0], 1).unwrap().len(), 1);
        assert_eq!(idx.search(Metric::Dot, &[1.0], 99).unwrap().len(), 2);
    }

    #[test]
    fn prefix_dim_query() {
        let idx = ExactIndex::build([iv(1, 1, vec![1.0, 0.0, 9.0]), iv(2, 2, vec![0.0, 1.0, 9.0])]);
        let hits = idx.search(Metric::Dot, &[1.0], 2).unwrap();
        assert_eq!(hits[0].external_id, 1); // only dim 0 counts
        assert!((hits[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn query_dim_exceeding_stored_rejected() {
        let idx = ExactIndex::build([iv(1, 1, vec![1.0, 1.0])]);
        assert!(idx.search(Metric::Dot, &[1.0, 1.0, 1.0], 1).is_err());
    }

    #[test]
    fn cosine_zero_norm_excluded_not_fatal() {
        let idx = ExactIndex::build([
            iv(1, 1, vec![0.0, 0.0]), // zero-norm stored
            iv(2, 2, vec![1.0, 0.5]),
        ]);
        let hits = idx.search(Metric::Cosine, &[1.0, 0.5], 2).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].external_id, 2);
    }

    #[test]
    fn cosine_zero_query_is_error() {
        let idx = ExactIndex::build([iv(1, 1, vec![1.0, 0.5])]);
        assert!(idx.search(Metric::Cosine, &[0.0, 0.0], 1).is_err());
    }

    #[test]
    fn empty_query_is_error() {
        let idx = ExactIndex::build([iv(1, 1, vec![1.0])]);
        assert!(idx.search(Metric::Dot, &[], 1).is_err());
    }

    #[test]
    fn l2_negated_distance_ordering() {
        let idx = ExactIndex::build([
            iv(1, 1, vec![1.0, 1.0]), // dist^2 0 from [1,1]
            iv(2, 2, vec![2.0, 2.0]), // dist^2 2
        ]);
        let hits = idx.search(Metric::L2, &[1.0, 1.0], 2).unwrap();
        assert_eq!(hits[0].external_id, 1);
        assert!(hits[0].score >= hits[1].score);
        assert!((hits[1].score + 2.0).abs() < 1e-5);
    }

    #[test]
    fn empty_index_searches_clean() {
        let idx = ExactIndex::new();
        assert!(idx.is_empty());
        assert!(idx.search(Metric::Dot, &[1.0], 5).unwrap().is_empty());
        // empty index, empty query: still an error (contract holds)
        assert!(idx.search(Metric::Dot, &[], 1).is_err());
    }

    #[test]
    fn ties_keep_stable_order() {
        // Two identical vectors: both returned, no panic on NaN-free ties.
        let idx = ExactIndex::build([iv(1, 1, vec![1.0, 0.0]), iv(2, 2, vec![1.0, 0.0])]);
        let hits = idx.search(Metric::Cosine, &[0.5, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].score, hits[1].score);
    }
}
