//! Query planning: fusion (RRF) and result assembly.
//!
//! v0 ships **RRF** (reciprocal rank fusion) as the default hybrid
//! combiner — evidence register §6: rank-based fusion is robust when
//! path scores are incomparable (BM25 vs cosine), and RRF beat score-
//! normalization alternatives on weak-path robustness. The
//! **weakest-link guard** (down-weight/exclude a degraded path
//! before fusion) is specified in the architecture but deferred:
//! it needs per-path quality signals that only exist once the
//! benchmark harness produces path-quality telemetry.
//!
//! RRF: score(id) = Σ_paths 1/(rrf_k + rank_path(id)), rrf_k = 60
//! (the standard constant from the original paper, validated by
//! subsequent production use). Deterministic: ties break by
//! (score desc, external_id asc).

use crate::index::Hit;

pub const RRF_K: u32 = 60;

/// One ranked candidate list (a path's output).
pub type PathHits = Vec<Hit>;

/// Reciprocal rank fusion over any number of paths.
///
/// A hit absent from a path contributes nothing from that path. The
/// output is top-k fused, deterministic under ties.
pub fn rrf_fuse(paths: &[PathHits], k: usize, rrf_k: u32) -> Vec<Hit> {
    if k == 0 {
        return Vec::new();
    }
    // id -> (best fused score, representative hit for seq)
    let mut fused: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for path in paths {
        // Rank 0 is the best hit (paths arrive sorted best-first).
        // RRF is rank-only: every hit in the caller's window
        // contributes; window sizing is the caller's contract.
        for (rank, hit) in path.iter().enumerate() {
            *fused.entry(hit.external_id).or_insert(0.0) +=
                1.0 / (rrf_k as f64 + rank as f64 + 1.0);
        }
    }
    let mut out: Vec<Hit> = Vec::with_capacity(fused.len());
    // Reproduce per-id seq by looking the id up in the paths (first
    // occurrence wins; seq is the same record).
    let mut seqs: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for path in paths {
        for hit in path {
            seqs.entry(hit.external_id).or_insert(hit.seq);
        }
    }
    for (id, score) in fused {
        out.push(Hit {
            external_id: id,
            score: score as f32,
            seq: seqs.get(&id).copied().unwrap_or(0),
        });
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.external_id.cmp(&b.external_id))
    });
    out.truncate(k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: u64, score: f32) -> Hit {
        Hit {
            external_id: id,
            score,
            seq: id,
        }
    }

    #[test]
    fn both_paths_rank_first_wins_fused() {
        let a = vec![hit(1, 0.9), hit(2, 0.8), hit(3, 0.7)];
        let b = vec![hit(1, 5.0), hit(3, 4.0), hit(2, 3.0)];
        let fused = rrf_fuse(&[a, b], 3, RRF_K);
        assert_eq!(fused[0].external_id, 1);
        // 2 vs 3: 2 is rank1 in a, rank2 in b => 1/62+1/63;
        // 3 is rank2 in a, rank1 in b => 1/63+1/62 — tie! id asc wins.
        assert_eq!(fused[1].external_id, 2);
        assert_eq!(fused[2].external_id, 3);
    }

    #[test]
    fn single_path_passthrough_order() {
        let a = vec![hit(5, 1.0), hit(4, 0.9), hit(3, 0.8), hit(2, 0.7)];
        let fused = rrf_fuse(&[a], 2, RRF_K);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].external_id, 5);
        assert_eq!(fused[1].external_id, 4);
    }

    #[test]
    fn absent_from_one_path_uses_other_only() {
        let a = vec![hit(1, 0.9)];
        let b = vec![hit(9, 5.0), hit(1, 4.0)];
        let fused = rrf_fuse(&[a, b], 5, RRF_K);
        // 1: 1/61 + 1/62 > 9: 1/61
        assert_eq!(fused[0].external_id, 1);
        assert_eq!(fused[1].external_id, 9);
    }

    #[test]
    fn disjoint_paths_fair_tie() {
        let a = vec![hit(7, 1.0)];
        let b = vec![hit(3, 1.0)];
        let fused = rrf_fuse(&[a, b], 2, RRF_K);
        // Same fused score: smaller id first (deterministic).
        assert_eq!(fused[0].external_id, 3);
        assert_eq!(fused[1].external_id, 7);
    }

    #[test]
    fn k_zero_and_empty_paths() {
        assert!(rrf_fuse(&[], 0, RRF_K).is_empty());
        assert!(rrf_fuse(&[vec![hit(1, 1.0)]], 0, RRF_K).is_empty());
        let fused = rrf_fuse(&[], 5, RRF_K);
        assert!(fused.is_empty());
    }

    #[test]
    fn k_truncates() {
        let a: Vec<Hit> = (1..=10u64).map(|i| hit(i, 1.0 / i as f32)).collect();
        let fused = rrf_fuse(&[a], 3, RRF_K);
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn rrf_ignores_score_scale_completely() {
        // RRF must be invariant to the score scales of the paths:
        // multiplying one path's scores by 1000 changes nothing.
        let a = vec![hit(1, 0.1), hit(2, 0.09)];
        let a_scaled = vec![hit(1, 100.0), hit(2, 90.0)];
        let b = vec![hit(2, 1.0), hit(1, 0.5)];
        let f1 = rrf_fuse(&[a, b.clone()], 2, RRF_K);
        let f2 = rrf_fuse(&[a_scaled, b], 2, RRF_K);
        let ids1: Vec<u64> = f1.iter().map(|h| h.external_id).collect();
        let ids2: Vec<u64> = f2.iter().map(|h| h.external_id).collect();
        assert_eq!(ids1, ids2);
        // And their fused scores are equal (rank-only).
        for (h1, h2) in f1.iter().zip(f2.iter()) {
            assert_eq!(h1.score, h2.score);
        }
    }
}
