//! BM25 text search over the segment model (v0).
//!
//! Tantivy-shaped: each sealed segment carries its own immutable
//! postings (built at seal time like the vector indexes); queries
//! score per segment with segment-local stats (N, avgdl) and merge by
//! score — the Lucene/Tantivy production pattern. L0 (fresh writes)
//! is scored as a mini-segment. `exact_text_top_k` provides the
//! global-view oracle for the merge-approximation acceptance test.

pub mod bm25;
pub mod tokenizer;

pub use bm25::{exact_text_top_k, hits_from, Postings, B, K1};
pub use tokenizer::{tokenize, tokenize_with_positions};
