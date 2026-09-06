//! HNSW graph backend (Malkov & Yashunin, Algorithm 1/2/4/5).
//!
//! One graph per sealed segment, built for ONE metric (L2 or cosine —
//! both true metrics). Dot-product graphs are out of v0: inner product
//! is not a metric (no triangle inequality) and MIPS-specific graphs
//! are a research area; building a dot HNSW returns a schema error,
//! and dot *queries* against an L2/cosine graph fall back to an
//! internal exact scan over the index's own vectors (correct, slower
//! — the routing layer may prefer exact anyway for dot).
//!
//! Parameters follow the architecture doc: M=16 (M0=32 at layer 0),
//! ef_construction = max(M*12, 100), ef_search = max(4k, 32)
//! ("ef_search dynamic"). Level assignment uses splitmix64 seeded from
//! config, drawn in insertion order: the same vectors + same seed
//! produce the identical graph, so a rebuilt derived artifact is
//! byte-comparable.
//!
//! Score/distance conventions: `index::score` returns higher=better;
//! graph distance is lower=closer — squared L2, and `1 - cosine` for
//! the cosine space (both non-negative, so f32 ordering is total).

use super::{score, validate_query, Hit, IndexedVector, Metric, VectorIndex};
use crate::error::{EngineError, EngineResult};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// HNSW build configuration.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max out-degree per node on layers > 0.
    pub m: usize,
    /// Max out-degree at layer 0 (paper default: 2*M).
    pub m0: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// The metric space the graph is built for. Must be L2 or Cosine.
    pub metric: Metric,
    /// RNG seed for level assignment (deterministic rebuilds).
    pub seed: u64,
    /// Default ef for searches that do not specify one. The SIFT-100K
    /// floor showed max(4k,32) (k=window=10 -> ef=40) at recall@10
    /// 0.9805; ef=200 reaches 0.9995. The default stays conservative
    /// (recall-leaning) at max(4k, 200).
    pub ef_search_default: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        HnswConfig {
            m: 16,
            m0: 32,
            ef_construction: 192,
            metric: Metric::L2,
            seed: 0x5EED_5EED,
            ef_search_default: 200,
        }
    }
}

/// Distance in the graph's space. Both terms are non-negative so f32
/// comparisons are total (vectors are validated finite at build).
fn dist(metric: Metric, q: &[f32], v: &[f32]) -> f32 {
    match metric {
        Metric::L2 => {
            let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            d
        }
        Metric::Cosine => {
            let qn = crate::records::l2_norm(q);
            let vn = crate::records::l2_norm(v);
            if qn == 0.0 || vn == 0.0 {
                // No direction: treat as maximally far; such records
                // are excluded from cosine results by the caller's
                // score mapping anyway (score() errs on zero norm).
                return 2.0;
            }
            let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
            1.0 - dot / (qn * vn)
        }
        Metric::Dot => unreachable!("dot graphs are rejected at build"),
    }
}

/// Deterministic splitmix64.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn next_unit(&mut self) -> f64 {
        let x = self.next() >> 11; // 53 bits
        (x as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
}

/// HNSW index over one segment's live set.
#[derive(Debug, Default)]
pub struct HnswIndex {
    config: HnswConfig,
    vectors: Vec<IndexedVector>,
    /// Per-node top level.
    levels: Vec<u8>,
    /// node → layer → neighbors (layers 0..=level[node]).
    links: Vec<Vec<Vec<u32>>>,
    entry: Option<u32>,
    max_level: i32,
}

impl HnswIndex {
    /// Build a graph from live vectors. Input order defines node
    /// indexes; same input + same seed ⇒ identical graph.
    pub fn build<I: IntoIterator<Item = IndexedVector>>(
        config: HnswConfig,
        vectors: I,
    ) -> EngineResult<Self> {
        if config.metric == Metric::Dot {
            return Err(EngineError::Schema(
                "HNSW v0 does not build dot-product graphs (use L2 or cosine; \
                 dot queries fall back to exact)"
                    .into(),
            ));
        }
        if config.m == 0 || config.m0 < config.m {
            return Err(EngineError::Schema(
                "HNSW requires m >= 1 and m0 >= m".into(),
            ));
        }
        let vectors: Vec<IndexedVector> = vectors.into_iter().collect();
        let mut idx = HnswIndex {
            config,
            vectors,
            levels: Vec::new(),
            links: Vec::new(),
            entry: None,
            max_level: -1,
        };
        // Validate vectors once: finite, uniform dim.
        let dim = idx.vectors.first().map(|v| v.vector.len()).unwrap_or(0);
        for iv in &idx.vectors {
            if iv.vector.len() != dim {
                return Err(EngineError::Schema(format!(
                    "HNSW vector dim {} != {} (mixed dims)",
                    iv.vector.len(),
                    dim
                )));
            }
            if iv.vector.iter().any(|x| !x.is_finite()) {
                return Err(EngineError::Schema("HNSW vector not finite".into()));
            }
        }
        let ml = 1.0 / (idx.config.m as f64).ln();
        let mut rng = SplitMix64(idx.config.seed);
        for node in 0..idx.vectors.len() as u32 {
            let level = (-rng.next_unit().ln() * ml).floor() as i64;
            let level = level.clamp(0, 16) as u8;
            idx.insert_node(node, level);
        }
        Ok(idx)
    }

    /// Stored dim (0 when empty).
    fn stored_dim(&self) -> u32 {
        self.vectors
            .first()
            .map(|v| v.vector.len() as u32)
            .unwrap_or(0)
    }

    fn vec_of(&self, node: u32) -> &[f32] {
        &self.vectors[node as usize].vector
    }

    /// Algorithm 1 insertion (vector already stored at `node`).
    fn insert_node(&mut self, node: u32, level: u8) {
        self.levels.push(level);
        self.links.push(vec![Vec::new(); level as usize + 1]);
        let Some(entry) = self.entry else {
            self.entry = Some(node);
            self.max_level = level as i32;
            return;
        };
        let q = self.vec_of(node).to_vec(); // owned: avoid borrow tangles
        let metric = self.config.metric;
        let m = self.config.m;
        let m0 = self.config.m0;
        let efc = self.config.ef_construction;

        // Phase 1 (immutable): greedy descent above `level`.
        let mut ep = entry;
        let top = self.max_level;
        for lc in ((level as i32 + 1)..=top).rev() {
            ep = self.greedy_closest(metric, &q, ep, lc as u8);
        }

        // Phase 2: connect at each layer from min(level, top) down.
        for lc in (0..=level.min(top as u8)).rev() {
            let candidates = self.search_layer(metric, &q, &[ep], efc, lc);
            if candidates.is_empty() {
                break;
            }
            let m_max = if lc == 0 { m0 } else { m };
            let neighbors = self.select_heuristic(metric, &q, &candidates, m);
            // Mutation phase: bidirectional links + shrink overfull.
            self.links[node as usize][lc as usize] = neighbors.clone();
            for &nb in &neighbors {
                let nb_vec = self.vec_of(nb).to_vec();
                let mut nb_list = std::mem::take(&mut self.links[nb as usize][lc as usize]);
                nb_list.push(node);
                if nb_list.len() > m_max {
                    // Re-select nb's neighborhood by the heuristic.
                    let mut cands: Vec<(f32, u32)> = nb_list
                        .iter()
                        .map(|&c| (dist(metric, &nb_vec, self.vec_of(c)), c))
                        .collect();
                    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    let kept = self.select_heuristic(metric, &nb_vec, &cands, m_max);
                    nb_list = kept;
                }
                self.links[nb as usize][lc as usize] = nb_list;
            }
            ep = candidates[0].1; // closest carries down (paper: ep = W)
        }

        if level as i32 > self.max_level {
            self.max_level = level as i32;
            self.entry = Some(node);
        }
    }

    /// Greedy walk to the local minimum at layer `lc`.
    fn greedy_closest(&self, metric: Metric, q: &[f32], start: u32, lc: u8) -> u32 {
        let mut cur = start;
        let mut best = dist(metric, q, self.vec_of(cur));
        loop {
            let mut moved = false;
            for &nb in &self.links[cur as usize][lc as usize] {
                let d = dist(metric, q, self.vec_of(nb));
                if d < best {
                    best = d;
                    cur = nb;
                    moved = true;
                }
            }
            if !moved {
                return cur;
            }
        }
    }

    /// Algorithm 2: beam search at layer `lc`. Returns (distance, node)
    /// sorted nearest-first, at most `ef` entries.
    fn search_layer(
        &self,
        metric: Metric,
        q: &[f32],
        eps: &[u32],
        ef: usize,
        lc: u8,
    ) -> Vec<(f32, u32)> {
        let mut visited: HashSet<u32> = eps.iter().copied().collect();
        let mut candidates: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(u64, u32)> = BinaryHeap::new();
        let key = |d: f32| d.to_bits() as u64; // non-negative => monotonic

        for &ep in eps {
            let d = dist(metric, q, self.vec_of(ep));
            candidates.push(Reverse((key(d), ep)));
            results.push((key(d), ep));
        }
        while let Some(Reverse((cd, c))) = candidates.pop() {
            if let Some(&(fd, _)) = results.peek() {
                if results.len() >= ef && cd > fd {
                    break;
                }
            }
            for &nb in &self.links[c as usize][lc as usize].clone() {
                if !visited.insert(nb) {
                    continue;
                }
                let d = dist(metric, q, self.vec_of(nb));
                let kd = key(d);
                let full = results.len() >= ef;
                let better = match results.peek() {
                    Some(&(fd, _)) => kd < fd,
                    None => true,
                };
                if !full || better {
                    candidates.push(Reverse((kd, nb)));
                    results.push((kd, nb));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out: Vec<(f32, u32)> = results
            .into_sorted_vec()
            .into_iter()
            .map(|(k, n)| (f32::from_bits(k as u32), n))
            .collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    /// Algorithm 4 neighbor selection with keepPruned: a candidate is
    /// kept iff it is closer to q than to every already-selected
    /// neighbor; pruned entries refill up to m.
    fn select_heuristic(
        &self,
        metric: Metric,
        _q: &[f32],
        candidates: &[(f32, u32)],
        m: usize,
    ) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        let mut pruned: Vec<u32> = Vec::new();
        for &(d, c) in candidates {
            if selected.len() >= m {
                pruned.push(c);
                continue;
            }
            let closer_than_all = selected.iter().all(|&s| {
                let ds = dist(metric, self.vec_of(s), self.vec_of(c));
                d < ds
            });
            if closer_than_all {
                selected.push(c);
            } else {
                pruned.push(c);
            }
        }
        // keepPruned: refill from pruned in candidate (distance) order.
        for &c in &pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(c);
        }
        selected
    }

    /// Internal exact scan over this index's own vectors (fallback
    /// path for metric mismatch / tiny sets).
    fn exact_scan(&self, metric: Metric, query: &[f32], k: usize) -> EngineResult<Vec<Hit>> {
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

impl VectorIndex for HnswIndex {
    fn name(&self) -> &'static str {
        "hnsw"
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Algorithm 5. Graph traversal for the build metric; internal
    /// exact scan for any other metric (dot on an L2/cosine graph)
    /// and for sets small enough that brute force is the right plan.
    fn search(&self, metric: Metric, query: &[f32], k: usize) -> EngineResult<Vec<Hit>> {
        validate_query(metric, query, self.stored_dim())?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let n = self.vectors.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        // Small sets or cross-metric queries: exact is correct and
        // competitive below ~100 nodes regardless.
        if metric != self.config.metric || k >= n || n <= 100 {
            return self.exact_scan(metric, query, k);
        }
        let ef = (k * 4).max(self.config.ef_search_default.max(32));
        let Some(entry) = self.entry else {
            return self.exact_scan(metric, query, k);
        };
        // Descend upper layers greedily.
        let mut ep = entry;
        let top = self.max_level;
        for lc in (1..=top).rev() {
            ep = self.greedy_closest(self.config.metric, query, ep, lc as u8);
        }
        let beam = self.search_layer(self.config.metric, query, &[ep], ef, 0);
        // Map graph distance back to trait score semantics.
        let mut hits: Vec<Hit> = beam
            .into_iter()
            .filter_map(|(_d, node)| {
                let iv = &self.vectors[node as usize];
                // Recompute via `score` so scores match the oracle
                // exactly (including prefix-dim handling); a zero-norm
                // cosine record drops out here.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::recall::{oracle_of, recall_at_k};

    /// Deterministic random vectors for tests.
    fn rand_vecs(n: usize, d: usize, seed: u64) -> Vec<IndexedVector> {
        let mut rng = SplitMix64(seed);
        (0..n as u64)
            .map(|i| {
                let v: Vec<f32> = (0..d)
                    .map(|_| (rng.next() % 10_000) as f32 / 10_000.0)
                    .collect();
                IndexedVector::new(i + 1, i + 1, v)
            })
            .collect()
    }

    #[test]
    fn clustered_queries_match_oracle_exactly() {
        // Two tight clusters + generous ef: HNSW must return the
        // oracle's top-k on every query.
        let mut vectors = Vec::new();
        for i in 0..32u64 {
            vectors.push(IndexedVector::new(i, i, vec![0.01 * i as f32, 0.0, 0.0]));
        }
        for i in 0..32u64 {
            vectors.push(IndexedVector::new(
                100 + i,
                100 + i,
                vec![10.0 + 0.01 * i as f32, 0.0, 0.0],
            ));
        }
        let idx = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        let oracle = oracle_of(&vectors);
        for q in [
            vec![0.1, 0.0, 0.0],
            vec![10.1, 0.0, 0.0],
            vec![5.0, 0.0, 0.0],
        ] {
            let got = idx.search(Metric::L2, &q, 5).unwrap();
            let want = oracle.search(Metric::L2, &q, 5).unwrap();
            assert_eq!(
                got.iter().map(|h| h.external_id).collect::<Vec<_>>(),
                want.iter().map(|h| h.external_id).collect::<Vec<_>>(),
                "query {q:?}"
            );
        }
    }

    #[test]
    fn recall_gate_random_data_l2() {
        let vectors = rand_vecs(2_000, 32, 7);
        let idx = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        let oracle = oracle_of(&vectors);
        let queries: Vec<Vec<f32>> = (0..100)
            .map(|i| rand_vecs(1, 32, 1_000 + i)[0].vector.clone())
            .collect();
        let report = recall_at_k(&idx, &oracle, Metric::L2, &queries, 10).unwrap();
        assert!(
            report.mean_recall >= 0.95,
            "HNSW L2 recall@10 = {} (gate 0.95)",
            report.mean_recall
        );
    }

    #[test]
    fn recall_gate_random_data_cosine() {
        let vectors = rand_vecs(2_000, 32, 11);
        let cfg = HnswConfig {
            metric: Metric::Cosine,
            ..HnswConfig::default()
        };
        let idx = HnswIndex::build(cfg, vectors.clone()).unwrap();
        let oracle = oracle_of(&vectors);
        let queries: Vec<Vec<f32>> = (0..100)
            .map(|i| rand_vecs(1, 32, 2_000 + i)[0].vector.clone())
            .collect();
        let report = recall_at_k(&idx, &oracle, Metric::Cosine, &queries, 10).unwrap();
        assert!(
            report.mean_recall >= 0.95,
            "HNSW cosine recall@10 = {} (gate 0.95)",
            report.mean_recall
        );
    }

    #[test]
    fn deterministic_rebuild() {
        let vectors = rand_vecs(500, 16, 3);
        let a = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        let b = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        assert_eq!(a.links, b.links);
        assert_eq!(a.entry, b.entry);
        assert_eq!(a.max_level, b.max_level);
        // Different seed ⇒ (almost surely) different level assignment.
        let c = HnswIndex::build(
            HnswConfig {
                seed: 99,
                ..HnswConfig::default()
            },
            vectors,
        )
        .unwrap();
        assert_ne!(a.levels, c.levels);
    }

    #[test]
    fn dot_falls_back_to_exact() {
        let vectors = rand_vecs(300, 8, 5); // >100: graph path would run
        let idx = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        let oracle = oracle_of(&vectors);
        let q = vec![0.5f32; 8];
        let got = idx.search(Metric::Dot, &q, 5).unwrap();
        let want = oracle.search(Metric::Dot, &q, 5).unwrap();
        assert_eq!(
            got.iter().map(|h| h.external_id).collect::<Vec<_>>(),
            want.iter().map(|h| h.external_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn dot_graph_build_rejected() {
        let vectors = rand_vecs(10, 4, 6);
        let cfg = HnswConfig {
            metric: Metric::Dot,
            ..HnswConfig::default()
        };
        assert!(matches!(
            HnswIndex::build(cfg, vectors),
            Err(EngineError::Schema(_))
        ));
    }

    #[test]
    fn prefix_dim_query_returns_valid_results() {
        // Tiny set ⇒ exact path; prefix scoring must match the oracle.
        let vectors: Vec<IndexedVector> = (0..16u64)
            .map(|i| IndexedVector::new(i, i, vec![i as f32 * 0.1, 1.0, 3.0]))
            .collect();
        let idx = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        let oracle = oracle_of(&vectors);
        let got = idx.search(Metric::L2, &[0.7], 4).unwrap();
        let want = oracle.search(Metric::L2, &[0.7], 4).unwrap();
        assert_eq!(
            got.iter().map(|h| h.external_id).collect::<Vec<_>>(),
            want.iter().map(|h| h.external_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_k_zero_k_over_n_edges() {
        let empty = HnswIndex::build(HnswConfig::default(), Vec::<IndexedVector>::new()).unwrap();
        assert!(empty.search(Metric::L2, &[1.0], 5).unwrap().is_empty());
        let vectors = rand_vecs(10, 4, 8);
        let idx = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        assert!(idx.search(Metric::L2, &[0.5; 4], 0).unwrap().is_empty());
        let all = idx.search(Metric::L2, &[0.5; 4], 50).unwrap();
        assert_eq!(all.len(), 10); // k >= n ⇒ everything
    }

    #[test]
    fn duplicate_vectors_all_indexed() {
        let v = vec![0.25f32; 8];
        let vectors: Vec<IndexedVector> = (0..40u64)
            .map(|i| IndexedVector::new(i, i, v.clone()))
            .collect();
        let idx = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        let got = idx.search(Metric::L2, &[0.25; 8], 5).unwrap();
        assert_eq!(got.len(), 5);
        // all scores ~0 distance
        assert!(got.iter().all(|h| h.score >= -1e-5));
    }

    #[test]
    fn cosine_zero_norm_stored_record_excluded() {
        let mut vectors = rand_vecs(30, 8, 9);
        vectors.push(IndexedVector::new(999, 999, vec![0.0; 8]));
        let cfg = HnswConfig {
            metric: Metric::Cosine,
            ..HnswConfig::default()
        };
        let idx = HnswIndex::build(cfg, vectors.clone()).unwrap();
        let got = idx
            .search(
                Metric::Cosine,
                &[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                30,
            )
            .unwrap();
        assert!(got.iter().all(|h| h.external_id != 999));
        assert!(!got.is_empty());
    }

    #[test]
    fn nonfinite_or_mixed_dim_input_rejected() {
        let bad = vec![IndexedVector::new(1, 1, vec![f32::NAN, 0.0])];
        assert!(HnswIndex::build(HnswConfig::default(), bad).is_err());
        let mixed = vec![
            IndexedVector::new(1, 1, vec![1.0, 2.0]),
            IndexedVector::new(2, 2, vec![1.0, 2.0, 3.0]),
        ];
        assert!(HnswIndex::build(HnswConfig::default(), mixed).is_err());
    }

    #[test]
    fn scores_match_oracle_values() {
        // Scores returned by the graph path must equal index::score
        // exactly (same computation, not graph distance).
        let vectors = rand_vecs(600, 12, 13);
        let idx = HnswIndex::build(HnswConfig::default(), vectors.clone()).unwrap();
        let oracle = oracle_of(&vectors);
        let q = rand_vecs(1, 12, 77)[0].vector.clone();
        let got = idx.search(Metric::L2, &q, 10).unwrap();
        let want = oracle.search(Metric::L2, &q, 10).unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            if g.external_id == w.external_id {
                assert_eq!(g.score, w.score);
            }
        }
    }
}
