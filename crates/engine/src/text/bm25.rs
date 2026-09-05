//! BM25 scoring for one segment's postings.
//!
//! Segment-local BM25 with **Lucene's IDF**: `ln(1 + (N - df + 0.5) /
//! (df + 0.5))` — strictly positive for every df, so a term present
//! in one small segment always scores > 0 (the classic Robertson
//! formula goes negative past half the corpus, which breaks
//! multi-segment merges: a term can be rare in one segment, common
//! corpus-wide, and would score negative there while positive
//! elsewhere). k1 = 1.2, b = 0.75 (Lucene/Tantivy defaults).
//!
//! Per-segment stats (N, avgdl) are the Tantivy pattern: each sealed
//! segment carries its own doc lengths and average; queries score
//! segment-locally and merge by score. This is an approximation of
//! global BM25 that converges as segments merge — the same trade
//! Lucene and Tantivy make in production.

use super::tokenizer::tokenize;
use crate::index::{Hit, Metric};
use crate::records::Record;
use std::collections::HashMap;

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

/// One posting: doc ordinal in the segment + term frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct Posting {
    pub doc: u32,
    pub tf: u32,
}

/// Immutable inverted index for one segment's documents.
#[derive(Debug, Default, Clone)]
pub struct Postings {
    /// term -> postings (docs ascending).
    map: HashMap<String, Vec<Posting>>,
    /// doc ordinal -> token count (doc length).
    doc_lengths: Vec<u32>,
    /// Number of documents.
    num_docs: u32,
}

impl Postings {
    /// Build postings for an ordered slice of documents (already
    /// tokenized). Doc ordinals are indexes into `docs`.
    pub fn build_from_tokens(docs: &[Vec<String>]) -> Self {
        let mut map: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut doc_lengths = Vec::with_capacity(docs.len());
        for (doc, toks) in docs.iter().enumerate() {
            doc_lengths.push(toks.len() as u32);
            let mut tfs: HashMap<&str, u32> = HashMap::new();
            for t in toks {
                *tfs.entry(t.as_str()).or_insert(0) += 1;
            }
            for (t, tf) in tfs {
                map.entry(t.to_string()).or_default().push(Posting {
                    doc: doc as u32,
                    tf,
                });
            }
        }
        for plist in map.values_mut() {
            plist.sort_by_key(|p| p.doc);
        }
        Postings {
            map,
            doc_lengths,
            num_docs: docs.len() as u32,
        }
    }

    /// Convenience: build by tokenizing texts.
    pub fn build<I, S>(texts: I) -> Self
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        let docs: Vec<Vec<String>> = texts
            .into_iter()
            .map(|t| t.map(|s| tokenize(s.as_ref())).unwrap_or_default())
            .collect();
        Postings::build_from_tokens(&docs)
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Average doc length (tokens); 1.0 floor avoids div-by-zero on
    /// empty-doc corpora.
    pub fn avgdl(&self) -> f32 {
        if self.num_docs == 0 {
            return 1.0;
        }
        let total: u64 = self.doc_lengths.iter().map(|&l| l as u64).sum();
        (total as f32 / self.num_docs as f32).max(1.0)
    }

    /// Postings for one term, docs ascending.
    pub fn postings(&self, term: &str) -> &[Posting] {
        self.map.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// BM25 score of `term` for `doc`: positive IDF, length norm b.
    pub fn score_term(&self, term: &str, doc: u32) -> f32 {
        let postings = self.postings(term);
        let df = postings.len() as u32;
        if df == 0 || doc >= self.num_docs {
            return 0.0;
        }
        let tf = postings
            .binary_search_by_key(&doc, |p| p.doc)
            .ok()
            .map(|i| postings[i].tf)
            .unwrap_or(0);
        if tf == 0 {
            return 0.0;
        }
        let tf = tf as f32;
        let idf = (1.0 + (self.num_docs as f32 - df as f32 + 0.5) / (df as f32 + 0.5)).ln();
        let dl = self.doc_lengths[doc as usize] as f32;
        let tf_norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / self.avgdl()));
        idf * tf_norm
    }

    /// BM25 score of a tokenized query against one doc.
    pub fn score_doc(&self, query_tokens: &[String], doc: u32) -> f32 {
        query_tokens.iter().map(|t| self.score_term(t, doc)).sum()
    }

    /// Top-k docs by BM25 for a raw query text. `alive` masks doc
    /// ordinals that must not surface (tombstones / caller policy).
    pub fn top_k(&self, query: &str, k: usize, alive: &dyn Fn(u32) -> bool) -> Vec<(u32, f32)> {
        let toks = tokenize(query);
        let mut scored: Vec<(u32, f32)> = (0..self.num_docs)
            .filter(|&d| alive(d))
            .map(|d| (d, self.score_doc(&toks, d)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Global-view text oracle (correctness reference for the segment
/// merge): one postings set over ALL live docs, exact BM25. The
/// store's merged per-segment results approximate this; the
/// acceptance test bounds the gap on real-shaped data.
pub fn exact_text_top_k(records: &[&Record], query: &str, k: usize) -> Vec<(u64, f32)> {
    let texts: Vec<Option<&str>> = records.iter().map(|r| r.text.as_deref()).collect();
    let postings = Postings::build(texts);
    let ids: Vec<u64> = records.iter().map(|r| r.external_id).collect();
    postings
        .top_k(query, k, &|_| true)
        .into_iter()
        .map(|(d, s)| (ids[d as usize], s))
        .collect()
}

/// Text hits reuse `Hit` so the planner fuses text and vectors in
/// one shape. Metric is meaningless for text; seq carries the doc's
/// WAL seq.
pub fn hits_from(scores: Vec<(u64, f32)>, seqs: &dyn Fn(u64) -> u64) -> Vec<Hit> {
    scores
        .into_iter()
        .map(|(id, score)| Hit {
            external_id: id,
            score,
            seq: seqs(id),
        })
        .collect()
}

/// Text scoring is metric-independent; this exists so callers that
/// pass a Metric uniformly get a loud error instead of silent
/// nonsense.
pub fn metric_guard(metric: Metric) -> crate::error::EngineResult<()> {
    Err(crate::error::EngineError::Schema(format!(
        "text search does not take a metric (got {metric:?})"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(texts: &[&str]) -> Postings {
        Postings::build(texts.iter().map(|t| Some(*t)))
    }

    #[test]
    fn builds_postings_with_tf_and_df() {
        let p = build(&["a b a", "c c", ""]);
        assert_eq!(p.num_docs(), 3);
        assert_eq!(p.postings("a").len(), 1); // df 1
        assert_eq!(p.postings("a")[0].tf, 2); // tf 2 in doc 0
        assert_eq!(p.postings("c").len(), 1);
        assert_eq!(p.postings("c")[0].tf, 2); // tf 2 in doc 1
        assert!(p.postings("missing").is_empty());
        assert!((p.avgdl() - 5.0 / 3.0).abs() < 1e-6); // (3+2+0)/3
    }

    #[test]
    fn lucene_idf_always_positive() {
        // Term in ALL docs: classic Robertson IDF goes negative;
        // Lucene's stays positive.
        let p = build(&["x y", "x z", "x w"]);
        assert!(p.score_term("x", 0) > 0.0);
    }

    #[test]
    fn rare_term_beats_common() {
        let p = build(&[
            "omnedb install guide",    // doc 0 (typo-ish rare term)
            "install omendb guide",    // doc 1: the real rare term
            "install install install", // doc 2: common-term spam
        ]);
        // "omendb" appears only in doc 1 → highest IDF; doc 1 must win.
        let top = p.top_k("omendb", 3, &|_| true);
        assert_eq!(top[0].0, 1);
        // "install" everywhere: doc 2 has tf 3 but no length
        // advantage after norm; scores must all be positive.
        let top2 = p.top_k("install", 3, &|_| true);
        assert_eq!(top2.len(), 3);
        assert!(top2.iter().all(|(_, s)| *s > 0.0));
    }

    #[test]
    fn length_norm_penalizes_long_docs() {
        let p = build(&[
            "vector search engine",
            "vector search engine and many many more tokens here to dilute the term",
        ]);
        // Same tf, longer doc → lower score for "vector".
        assert!(p.score_term("vector", 0) > p.score_term("vector", 1));
    }

    #[test]
    fn empty_and_missing_terms_score_zero() {
        let p = build(&["hello world"]);
        assert!(p.top_k("nothere", 5, &|_| true).is_empty());
        assert!(p.top_k("", 5, &|_| true).is_empty());
        assert_eq!(p.score_term("hello", 99), 0.0); // doc out of range
    }

    #[test]
    fn alive_mask_hides_docs() {
        let p = build(&["alpha beta", "alpha gamma", "alpha delta"]);
        let top = p.top_k("alpha", 3, &|d| d != 1);
        let docs: Vec<u32> = top.iter().map(|(d, _)| *d).collect();
        assert!(!docs.contains(&1));
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn k_truncation() {
        let p = build(&["a", "a", "a", "a"]);
        assert_eq!(p.top_k("a", 2, &|_| true).len(), 2);
    }

    #[test]
    fn oracle_ranks_rare_term_doc_first() {
        use crate::records::Record;
        let rs: Vec<Record> = vec![
            Record::new(1, vec![0.0]).with_text("the quick brown fox"),
            Record::new(2, vec![0.0]).with_text("the slow brown dog"),
            Record::new(3, vec![0.0]).with_text("quick quick quick fox"),
        ];
        let refs: Vec<&Record> = rs.iter().collect();
        let top = exact_text_top_k(&refs, "quick", 3);
        assert_eq!(top[0].0, 3); // highest tf of the rare term
        assert!(top[0].1 > top[1].1);
        // "dog" appears once → id 2 first
        let top2 = exact_text_top_k(&refs, "dog", 3);
        assert_eq!(top2[0].0, 2);
    }
}
