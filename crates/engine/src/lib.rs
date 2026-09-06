//! OmenDB Vector engine core.
//!
//! Embedded local-first retrieval engine. v0 scope: dense + BM25 text +
//! metadata filters + hybrid RRF; RAM and mmap storage tiers; single
//! writer with snapshot readers. The design lives in `docs/architecture.md`;
//! the prior Mojo engine line is archived under tag `v0.1.0a1-mojo`.
//!
//! Crate layout follows the mutation-authority split: canonical
//! records are the sole source of truth; every derived artifact
//! (indexes, text postings) is generation-bound, checksummed, and
//! rebuildable from records.

pub mod error;

pub mod filter;

pub mod planner;

pub mod codec;

pub mod fsutil;

pub mod index;

pub mod records;

pub mod segments;

pub mod store;

pub mod text;

pub mod wal;

pub use error::{EngineError, EngineResult};
pub use filter::{Filter, Num, Predicate};
pub use index::{Hit, Metric, RecallReport};
pub use planner::{rrf_fuse, RRF_K};
pub use records::{l2_norm, Lifecycle, MetaValue, Record};
pub use store::{Store, StoreRecovery};
