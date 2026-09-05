//! The store: single-writer coordinator over WAL + segments + L0.
//!
//! Mutation authority is the record stream (WAL first, then segments
//! as sealed history). `L0` is the RAM tier for records appended since
//! the last checkpoint. `Store` owns the full lifecycle:
//!
//! - `upsert`/`delete` append to the WAL (nothing is acked until
//!   `commit` fsyncs a barrier). The first upsert fixes the
//!   collection dim; later mixed-dim writes are rejected.
//! - `checkpoint` = commit, seal the live set (post-tombstone-absorb,
//!   full records preserved) into one new segment, publish the
//!   manifest atomically, drop consumed L0. After a checkpoint the
//!   WAL's covered prefix is redundant but retained (v0: no
//!   rotation), so segments stay derived and rebuildable.
//! - `get`/`scan`/`exact_top_k` read the merged view: segments (seq
//!   ≤ `checkpoint_seq`) + L0 (seq > `checkpoint_seq`), deduplicated
//!   by external id, last seq wins, tombstones honored.
//! - Open/reopen: WAL recovery (destructive prefix rule) then either
//!   manifest fast path (all segments valid, `checkpoint_seq` ≤ WAL
//!   committed seq) or flagged full rebuild from the WAL. Reopen
//!   equivalence — the store view equals the pre-close view — is the
//!   acceptance test for the whole layer.
//!
//! `exact_top_k` is the flat-scan correctness oracle for every future
//! ANN/filtered/quantized mode (architecture §5): brute force over the
//! merged live view, prefix-dim queries allowed (Matryoshka), metrics
//! dot/cosine/L2. Cosine needs nonzero norms: a zero-norm query is an
//! error; a zero-norm stored record is unscorable and excluded.

use crate::error::{EngineError, EngineResult};
use crate::index::{ExactIndex, HnswConfig, HnswIndex, IndexedVector, VectorIndex};
use crate::records::{Lifecycle, Record};
use crate::segments::{
    gc_segments, load_manifest, publish_manifest, write_segment, Manifest, SegmentEntry,
    SegmentReader,
};
use crate::text::Postings;
use crate::wal::Wal;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Re-export the query types so callers see one coherent API surface;
// the definitions live in `index` where every backend shares them.
pub use crate::index::{Hit, Metric};
/// Which index backend serves sealed segments.
#[derive(Debug, Clone, Default)]
pub enum IndexBackend {
    /// Flat exact scan (default): correctness oracle behavior,
    /// zero build cost.
    #[default]
    Exact,
    /// HNSW graph for one metric; queries under other metrics fall
    /// back to the backend's internal exact scan.
    Hnsw(HnswConfig),
}

/// What `Store::open` recovered.
#[derive(Debug)]
pub struct StoreRecovery {
    /// WAL-level recovery info.
    pub wal: crate::wal::WalRecovery,
    /// Manifest generation loaded, if any.
    pub manifest_generation: Option<u64>,
    /// Set when the manifest/segments were unusable and the live view
    /// was rebuilt from the WAL instead.
    pub rebuilt_from_wal: bool,
    /// Segment files removed after a rebuild.
    pub gc_removed: usize,
}

/// The L0 RAM tier: records since the last checkpoint, keyed by
/// external id; latest seq per id wins.
#[derive(Debug, Default)]
struct L0 {
    /// external_id -> (seq, record)
    live: HashMap<u64, (u64, Record)>,
}

impl L0 {
    fn apply(&mut self, seq: u64, record: Record) {
        // WAL seqs are strictly increasing, so an existing entry
        // always has a lower seq; last write wins.
        self.live.insert(record.external_id, (seq, record));
    }
}

/// Open store: single writer, crash-safe, derived-artifact rebuild.
pub struct Store {
    dir: PathBuf,
    wal: Wal,
    dim: u32,
    /// Segment files in the manifest, opened.
    segments: Vec<SegmentReader>,
    /// Per-segment index (same order as `segments`). Built at seal
    /// time and on open per `backend`; trait objects so any
    /// `VectorIndex` drops in. A missing/unbuilt index is rebuilt
    /// inline (fail-safe: never panic).
    seg_indexes: Vec<Arc<dyn VectorIndex>>,
    /// Backend used for sealed segments.
    backend: IndexBackend,
    /// Live ids in segments: id -> seq (sealed segments hold only
    /// live records, so membership == liveness).
    seg_live: HashMap<u64, u64>,
    checkpoint_seq: u64,
    generation: u64,
    l0: L0,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("dir", &self.dir)
            .field("dim", &self.dim)
            .field("segments", &self.segments.len())
            .field("checkpoint_seq", &self.checkpoint_seq)
            .field("generation", &self.generation)
            .field("l0_records", &self.l0.live.len())
            .finish_non_exhaustive()
    }
}

impl Store {
    /// Open (or create) a store at `dir`. Exactly one writer process;
    /// no file locking in v0.
    pub fn open<P: AsRef<Path>>(dir: P) -> EngineResult<(Store, StoreRecovery)> {
        Self::open_with(dir, IndexBackend::default())
    }

    /// Open with an explicit index backend for sealed segments.
    pub fn open_with<P: AsRef<Path>>(
        dir: P,
        backend: IndexBackend,
    ) -> EngineResult<(Store, StoreRecovery)> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        // Validate the backend config up front: an invalid choice
        // (e.g. HNSW with dot metric) must fail the open, not surface
        // at first checkpoint.
        if let IndexBackend::Hnsw(cfg) = &backend {
            if cfg.metric == Metric::Dot {
                return Err(EngineError::Schema(
                    "HNSW backend cannot use Dot metric (use L2 or Cosine; Dot queries \
                     fall back to exact)"
                        .into(),
                ));
            }
            if cfg.m == 0 || cfg.m0 < cfg.m {
                return Err(EngineError::Schema(
                    "HNSW backend requires m >= 1 and m0 >= m".into(),
                ));
            }
        }
        let (wal, wal_rec) = Wal::open(dir.join("wal.log"))?;

        let mut rebuilt = false;
        let mut manifest_gen = None;
        let mut segments: Vec<SegmentReader> = Vec::new();
        let mut checkpoint_seq = 0u64;
        let mut dim = 0u32;

        if let Some(m) = load_manifest(&dir)? {
            let mut usable = m.checkpoint_seq <= wal_rec.committed_seq;
            if usable {
                for name in &m.segments {
                    match SegmentReader::open(&dir, name) {
                        Ok(seg) => segments.push(seg),
                        Err(_) => {
                            usable = false;
                            break;
                        }
                    }
                }
            }
            if usable {
                manifest_gen = Some(m.generation);
                checkpoint_seq = m.checkpoint_seq;
                dim = m.dim;
            } else {
                // A manifest we cannot honor is a lie; rebuild from the
                // WAL (the sole authority) and replace it.
                segments.clear();
                rebuilt = true;
            }
        } else {
            rebuilt = true; // no manifest: everything from the WAL
        }

        let seg_live: HashMap<u64, u64> = segments
            .iter()
            .flat_map(|seg| seg.entries())
            .map(|e| (e.record.external_id, e.seq))
            .collect();
        // Build per-segment indexes per the configured backend.
        let seg_indexes = segments
            .iter()
            .map(|seg| {
                let vectors: Vec<IndexedVector> = seg
                    .entries()
                    .iter()
                    .map(|e| {
                        IndexedVector::new(e.record.external_id, e.seq, e.record.vector.clone())
                    })
                    .collect();
                build_index(&backend, vectors)
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let mut store = Store {
            backend,
            seg_live,
            seg_indexes,
            dir,
            wal,
            dim,
            segments,
            checkpoint_seq,
            generation: manifest_gen.unwrap_or(0),
            l0: L0::default(),
        };

        // WAL replay: records above checkpoint_seq are L0; older ones
        // are already sealed (or, after a rebuild, define the live set).
        for (seq, record) in &wal_rec.records {
            if *seq > checkpoint_seq {
                store.apply_replayed(*seq, record.clone())?;
            }
        }

        // Rebuild housekeeping: publish an honest (empty) manifest and
        // remove orphaned segment files so reopen is stable.
        let mut gc_removed = 0;
        if rebuilt {
            let m = Manifest {
                generation: store.generation,
                checkpoint_seq: 0,
                dim: store.dim,
                segments: Vec::new(),
            };
            publish_manifest(&store.dir, &m)?;
            for entry in std::fs::read_dir(&store.dir)? {
                let entry = entry?;
                let name = entry.file_name();
                let Some(name) = name.to_str() else { continue };
                if name.starts_with("seg-") && name.ends_with(".seg") {
                    std::fs::remove_file(entry.path())?;
                    gc_removed += 1;
                }
            }
        }

        let recovery = StoreRecovery {
            wal: wal_rec,
            manifest_generation: manifest_gen,
            rebuilt_from_wal: rebuilt,
            gc_removed,
        };
        Ok((store, recovery))
    }

    /// Apply a replayed WAL record. Validates dim consistency; the
    /// first record defines the dim when the manifest had none.
    fn apply_replayed(&mut self, seq: u64, record: Record) -> EngineResult<()> {
        if self.dim == 0 {
            self.dim = record.vector.len() as u32;
        }
        if record.vector.len() as u32 != self.dim {
            return Err(EngineError::Schema(format!(
                "record dim {} != collection dim {}",
                record.vector.len(),
                self.dim
            )));
        }
        self.l0.apply(seq, record);
        Ok(())
    }

    /// Collection dim (0 until the first record defines it).
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Append an upsert (live record). Not durable until `commit`.
    /// The first upsert fixes the collection dim.
    pub fn upsert(&mut self, record: Record) -> EngineResult<u64> {
        self.validate_new_record(&record)?;
        if self.dim == 0 {
            self.dim = record.vector.len() as u32;
        }
        let seq = self.wal.append(&record)?;
        self.l0.apply(seq, record);
        Ok(seq)
    }

    /// Append a tombstone. Not durable until `commit`. Deleting an
    /// unknown/dead id is an error — v0 makes that explicit rather
    /// than guessing caller intent.
    pub fn delete(&mut self, external_id: u64) -> EngineResult<u64> {
        if !self.is_live(external_id) {
            return Err(EngineError::Schema(format!(
                "delete of unknown or dead id {external_id}"
            )));
        }
        let mut tomb = Record::new(external_id, vec![0.0; self.dim as usize]);
        tomb.lifecycle = Lifecycle::Tombstone;
        let seq = self.wal.append(&tomb)?;
        self.l0.apply(seq, tomb);
        Ok(seq)
    }

    /// Commit barrier + fsync. Everything appended is now durable.
    pub fn commit(&mut self) -> EngineResult<u64> {
        self.wal.commit()
    }

    fn validate_new_record(&self, record: &Record) -> EngineResult<()> {
        if self.dim == 0 {
            if record.vector.is_empty() {
                return Err(EngineError::Schema(
                    "first record must have a vector".into(),
                ));
            }
        } else if record.vector.len() as u32 != self.dim {
            return Err(EngineError::Schema(format!(
                "record dim {} != collection dim {}",
                record.vector.len(),
                self.dim
            )));
        }
        // encode() performs the finite/oversize checks.
        record.encode()?;
        Ok(())
    }

    /// Is this external id live anywhere (L0 first, then segments)?
    fn is_live(&self, external_id: u64) -> bool {
        if let Some((_, r)) = self.l0.live.get(&external_id) {
            return r.lifecycle == Lifecycle::Live;
        }
        self.seg_live.contains_key(&external_id)
    }

    /// The merged live view: full records (vector, text, meta, norm),
    /// latest seq per id across segments + L0, tombstones winning
    /// when newest, output sorted by seq (the order segment writes
    /// require).
    fn merged_live(&self) -> Vec<(u64, u64, Record)> {
        let mut best: HashMap<u64, (u64, Record)> = HashMap::new();
        for seg in &self.segments {
            for e in seg.entries() {
                match best.get(&e.record.external_id) {
                    Some((s, _)) if *s >= e.seq => {}
                    _ => {
                        best.insert(e.record.external_id, (e.seq, e.record.clone()));
                    }
                }
            }
        }
        for (id, (seq, r)) in &self.l0.live {
            match best.get(id) {
                Some((s, _)) if *s >= *seq => {}
                _ => {
                    best.insert(*id, (*seq, r.clone()));
                }
            }
        }
        let mut out: Vec<(u64, u64, Record)> = best
            .into_iter()
            .filter(|(_, (_, r))| r.lifecycle == Lifecycle::Live)
            .map(|(id, (seq, r))| (id, seq, r))
            .collect();
        out.sort_by_key(|(_, seq, _)| *seq);
        out
    }

    /// Unified search: per-segment index backends + L0 exact scan,
    /// merged, top-k overall. Same contract as `exact_top_k` (which
    /// remains the direct oracle: single scan, no index involved).
    /// `per_segment_k` is the candidate pool each backend contributes
    /// before the final merge (>= k).
    pub fn search(
        &self,
        metric: Metric,
        query: &[f32],
        k: usize,
        per_segment_k: usize,
    ) -> EngineResult<Vec<Hit>> {
        if per_segment_k < k {
            return Err(EngineError::Schema(format!(
                "per_segment_k {per_segment_k} < k {k}"
            )));
        }
        let pool = per_segment_k;
        let mut all: Vec<Hit> = Vec::new();
        // Sealed segments: their backend index answers.
        for idx in &self.seg_indexes {
            all.extend(idx.search(metric, query, pool)?);
        }
        // L0: fresh writes, always exact.
        let l0_vecs: Vec<IndexedVector> = self
            .l0
            .live
            .values()
            .filter(|(_, r)| r.lifecycle == Lifecycle::Live)
            .map(|(seq, r)| IndexedVector::new(r.external_id, *seq, r.vector.clone()))
            .collect();
        if !l0_vecs.is_empty() {
            let l0_idx: Arc<dyn VectorIndex> = Arc::new(ExactIndex::build(l0_vecs));
            all.extend(l0_idx.search(metric, query, pool)?);
        }
        // Merge: dedupe by external id (segments hold disjoint ids in
        // v0's single-segment layout, but stay correct for any
        // layout), keep best score, sort, truncate.
        let mut best: HashMap<u64, Hit> = HashMap::new();
        for h in all {
            match best.get(&h.external_id) {
                Some(prev) if prev.score >= h.score => {}
                _ => {
                    best.insert(h.external_id, h);
                }
            }
        }
        let mut hits: Vec<Hit> = best.into_values().collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// BM25 text search over the merged live view. Scoring uses
    /// **merged stats** (one postings set over all live docs across
    /// sealed segments + L0, computed in one pass): with a single
    /// sealed segment + L0 both in RAM, global stats are free and
    /// make the result exactly the global-view oracle — no
    /// per-segment IDF divergence (an L0 vocabulary cluster cannot
    /// deflate its own scores). L0 entries shadow sealed entries
    /// with the same id. Live records only.
    pub fn text_search(&self, query: &str, k: usize) -> Vec<Hit> {
        if k == 0 {
            return Vec::new();
        }
        let q_tokens = crate::text::tokenize(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        // Shadowing: ids whose latest state is in L0.
        let shadowed: HashSet<u64> = self.l0.live.keys().copied().collect();

        // Candidate docs: live, not shadowed, from segments; all
        // live from L0.
        let mut cands: Vec<(u64, u64, Option<&str>)> = Vec::new();
        for seg in &self.segments {
            for e in seg.entries() {
                if e.record.lifecycle == Lifecycle::Live
                    && !shadowed.contains(&e.record.external_id)
                {
                    cands.push((e.record.external_id, e.seq, e.record.text.as_deref()));
                }
            }
        }
        for (id, (seq, r)) in &self.l0.live {
            if r.lifecycle == Lifecycle::Live {
                cands.push((*id, *seq, r.text.as_deref()));
            }
        }
        if cands.is_empty() {
            return Vec::new();
        }

        // One postings set over all candidates: merged stats AND
        // merged scoring; top_k is exact over this set.
        let texts: Vec<Option<&str>> = cands.iter().map(|(_, _, t)| *t).collect();
        let postings = Postings::build(texts);
        let top = postings.top_k(query, k, &|_| true);
        top.into_iter()
            .map(|(doc, score)| {
                let (external_id, seq, _) = cands[doc as usize];
                Hit {
                    external_id,
                    score,
                    seq,
                }
            })
            .collect()
    }

    /// Exact flat scan (correctness oracle): brute force over the
    /// merged live view, prefix-dim queries allowed (Matryoshka),
    /// metrics Dot/Cosine/L2. Cosine zero-norm query is a schema
    /// error; a zero-norm stored record is unscorable and excluded.
    pub fn exact_top_k(&self, metric: Metric, query: &[f32], k: usize) -> EngineResult<Vec<Hit>> {
        if query.is_empty() {
            return Err(EngineError::Schema("empty query vector".into()));
        }
        if self.dim != 0 && query.len() as u32 > self.dim {
            return Err(EngineError::Schema(format!(
                "query dim {} exceeds collection dim {}",
                query.len(),
                self.dim
            )));
        }
        if metric == Metric::Cosine && crate::records::l2_norm(query) == 0.0 {
            return Err(EngineError::Schema("cosine of zero query vector".into()));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut hits: Vec<Hit> = Vec::new();
        for (id, seq, r) in self.merged_live() {
            match crate::index::score(metric, query, &r.vector) {
                Ok(s) => hits.push(Hit {
                    external_id: id,
                    score: s,
                    seq,
                }),
                // Only reachable for cosine against a zero-norm stored
                // vector: no direction, unscorable, excluded.
                Err(_) => continue,
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Fetch a live record by external id.
    pub fn get(&self, external_id: u64) -> Option<Record> {
        if !self.is_live(external_id) {
            return None;
        }
        if let Some((_, r)) = self.l0.live.get(&external_id) {
            if r.lifecycle == Lifecycle::Live {
                return Some(r.clone());
            }
        }
        // Sealed segments hold at most one entry per id (merged_live
        // dedupe on write), so the first live match is the record.
        self.segments
            .iter()
            .flat_map(|seg| seg.entries())
            .find(|e| e.record.external_id == external_id && e.record.lifecycle == Lifecycle::Live)
            .map(|e| e.record.clone())
    }

    /// Scan the merged live view (seq order).
    pub fn scan(&self) -> Vec<Record> {
        self.merged_live().into_iter().map(|(_, _, r)| r).collect()
    }

    /// Live record count.
    pub fn len(&self) -> usize {
        self.merged_live().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checkpoint: commit the WAL, seal the live set into a new
    /// segment (full records, tombstones absorbed — a deleted id
    /// vanishes; the WAL keeps history so nothing resurrects),
    /// publish the manifest atomically, reset L0, GC replaced
    /// segments.
    pub fn checkpoint(&mut self) -> EngineResult<u64> {
        let committed = self.wal.commit()?;
        let live = self.merged_live();
        let entries: Vec<SegmentEntry> = live
            .into_iter()
            .map(|(_id, seq, r)| SegmentEntry {
                seq,
                record: r, // full record: vector, text, meta, norm
            })
            .collect();
        self.generation += 1;
        let name = write_segment(&self.dir, self.generation, &entries)?;
        let manifest = Manifest {
            generation: self.generation,
            checkpoint_seq: committed,
            dim: self.dim,
            segments: vec![name.clone()],
        };
        publish_manifest(&self.dir, &manifest)?;

        // Swap in the sealed view and reset L0.
        let new_seg = SegmentReader::open(&self.dir, &name)?;
        let new_index = build_index(
            &self.backend,
            new_seg
                .entries()
                .iter()
                .map(|e| IndexedVector::new(e.record.external_id, e.seq, e.record.vector.clone()))
                .collect(),
        );
        self.seg_live = new_seg
            .entries()
            .iter()
            .map(|e| (e.record.external_id, e.seq))
            .collect();
        self.segments = vec![new_seg];
        self.seg_indexes = vec![new_index?];
        self.checkpoint_seq = committed;
        self.l0 = L0::default();

        gc_segments(&self.dir, &manifest)?;
        Ok(committed)
    }
}

/// Build one segment index for the chosen backend. Invalid backend
/// configuration (e.g. HNSW with dot metric) is a caller error and
/// fails the open/checkpoint — silent degradation would hide it.
fn build_index(
    backend: &IndexBackend,
    vectors: Vec<IndexedVector>,
) -> EngineResult<Arc<dyn VectorIndex>> {
    match backend {
        IndexBackend::Exact => Ok(Arc::new(ExactIndex::build(vectors))),
        IndexBackend::Hnsw(cfg) => Ok(Arc::new(
            HnswIndex::build(cfg.clone(), vectors)
                .map_err(|e| EngineError::Schema(format!("index backend unusable: {e}")))?,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("omendb-store-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn rec(id: u64, x: f32) -> Record {
        Record::new(id, vec![x, x * 0.5])
    }

    #[test]
    fn hnsw_backend_matches_oracle_across_boundaries() {
        let dir = tmp_dir("hnswire");
        let cfg = HnswConfig::default();
        let mut s = Store::open_with(&dir, IndexBackend::Hnsw(cfg)).unwrap().0;
        // 300 vectors: enough that the graph path actually runs
        // (n > 100, k < n).
        let mut rng = 12345u64;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as f32 / 2.0_f32.powi(31)
        };
        for i in 0..300u64 {
            s.upsert(Record::new(i, vec![next(), next(), next()]))
                .unwrap();
        }
        s.commit().unwrap();
        s.checkpoint().unwrap(); // segment built with HNSW
        for i in 300..400u64 {
            s.upsert(Record::new(i, vec![next(), next(), next()]))
                .unwrap();
        }
        s.commit().unwrap(); // 100 fresh in L0
                             // L2 (graph metric): search must match the oracle on the
                             // overlap of returned ids (within the pool).
        for k in [1usize, 5, 10] {
            let merged = s.search(Metric::L2, &[0.5, 0.5, 0.5], k, k).unwrap();
            let oracle = s.exact_top_k(Metric::L2, &[0.5, 0.5, 0.5], k).unwrap();
            let m: Vec<u64> = merged.iter().map(|h| h.external_id).collect();
            let o: Vec<u64> = oracle.iter().map(|h| h.external_id).collect();
            // HNSW is approximate: require >= 80% overlap at each k
            // (the gate for store-level recall, stricter than the
            // 0.002 rule which applies to backend-vs-oracle only).
            let overlap = m.iter().filter(|id| o.contains(id)).count();
            assert!(
                overlap as f64 >= (k as f64) * 0.8,
                "k={k}: overlap {overlap}/{k} — merged {m:?} vs oracle {o:?}"
            );
            // Scores for the ids both paths returned must be equal.
            for h in &merged {
                if let Some(oh) = oracle.iter().find(|x| x.external_id == h.external_id) {
                    assert_eq!(h.score, oh.score);
                }
            }
        }
        // Dot on an L2 graph: internal fallback must be EXACT equal.
        let merged = s.search(Metric::Dot, &[1.0, 1.0, 1.0], 5, 5).unwrap();
        let oracle = s.exact_top_k(Metric::Dot, &[1.0, 1.0, 1.0], 5).unwrap();
        assert_eq!(
            merged.iter().map(|h| h.external_id).collect::<Vec<_>>(),
            oracle.iter().map(|h| h.external_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_hnsw_backend_fails_open() {
        let dir = tmp_dir("badbackend");
        // dot metric config: must fail loud, not silently degrade
        let cfg = HnswConfig {
            metric: Metric::Dot,
            ..HnswConfig::default()
        };
        let err = Store::open_with(&dir, IndexBackend::Hnsw(cfg)).unwrap_err();
        assert!(matches!(err, EngineError::Schema(_)));
    }

    #[test]
    fn text_search_matches_oracle_ranking() {
        let dir = tmp_dir("textoracle");
        let mut s = Store::open(&dir).unwrap().0;
        // Two clusters of vocabulary: 20 docs about installing, 20 about
        // troubleshooting; some share tokens.
        for i in 0..40u64 {
            let text = if i < 20 {
                format!("install guide step {i} for the omendb vector engine")
            } else {
                format!("troubleshooting steps {i} when the install fails")
            };
            s.upsert(Record::new(i, vec![0.1, 0.1]).with_text(text))
                .unwrap();
        }
        s.commit().unwrap();
        s.checkpoint().unwrap();
        for i in 40..50u64 {
            s.upsert(Record::new(i, vec![0.1, 0.1]).with_text(format!("install omendb fresh {i}")))
                .unwrap();
        }
        s.commit().unwrap();

        for (q, k) in [
            ("install omendb", 10usize),
            ("troubleshooting fails", 5),
            ("vector engine", 8),
        ] {
            let got = s.text_search(q, k);
            let live: Vec<Record> = s.scan();
            let refs: Vec<&Record> = live.iter().collect();
            let want = crate::text::exact_text_top_k(&refs, q, k);
            // Same id SET (order may differ slightly between segment
            // stats and global stats on ties).
            let gs: std::collections::HashSet<u64> = got.iter().map(|h| h.external_id).collect();
            let ws: std::collections::HashSet<u64> = want.iter().map(|(id, _)| *id).collect();
            assert_eq!(gs, ws, "q={q}: got {gs:?} want {ws:?}");
        }
    }

    #[test]
    fn text_search_l0_shadows_sealed_copy() {
        let dir = tmp_dir("textshadow");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(Record::new(1, vec![0.1]).with_text("install guide for the database"))
            .unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap(); // sealed with old text
                                 // Update in L0: new text must win, old must not surface
        s.upsert(
            Record::new(1, vec![0.1]).with_text("completely different troubleshooting content"),
        )
        .unwrap();
        s.commit().unwrap();
        let hits = s.text_search("install guide", 5);
        assert!(
            hits.iter().all(|h| h.external_id != 1) || {
                // id 1 may surface only via its NEW text
                hits.iter().all(|h| h.external_id != 1 || h.score == 0.0)
            }
        );
        assert!(s
            .text_search("troubleshooting content", 5)
            .iter()
            .any(|h| h.external_id == 1));
    }

    #[test]
    fn text_search_survives_reopen() {
        let dir = tmp_dir("textreopen");
        {
            let mut s = Store::open(&dir).unwrap().0;
            s.upsert(Record::new(1, vec![0.1]).with_text("hello vector world"))
                .unwrap();
            s.upsert(Record::new(2, vec![0.1]).with_text("hello database world"))
                .unwrap();
            s.commit().unwrap();
            s.checkpoint().unwrap();
        }
        let (s2, _) = Store::open(&dir).unwrap();
        let hits = s2.text_search("hello world", 5);
        let ids: Vec<u64> = hits.iter().map(|h| h.external_id).collect();
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    #[test]
    fn text_search_empty_and_no_text() {
        let dir = tmp_dir("textempty");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(Record::new(1, vec![0.1])).unwrap(); // no text
        s.upsert(Record::new(2, vec![0.1]).with_text("words here"))
            .unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap();
        assert!(s.text_search("words", 5)[0].external_id == 2);
        assert!(s.text_search("nothing matches this", 5).is_empty());
        assert!(s.text_search("anything", 0).is_empty());
    }

    #[test]
    fn search_matches_exact_oracle_across_boundaries() {
        let dir = tmp_dir("searchequiv");
        let mut s = Store::open(&dir).unwrap().0;
        // segment tier + L0 tier, overlapping score ranges
        for i in 0..20u64 {
            s.upsert(Record::new(i, vec![i as f32 * 0.05, 1.0 - i as f32 * 0.05]))
                .unwrap();
        }
        s.commit().unwrap();
        s.checkpoint().unwrap(); // all 20 sealed + indexed
        for i in 20..35u64 {
            s.upsert(Record::new(i, vec![i as f32 * 0.05, 1.0 - i as f32 * 0.05]))
                .unwrap();
        }
        s.commit().unwrap(); // 15 fresh in L0
        for k in [1usize, 3, 7, 15] {
            let merged = s.search(Metric::Dot, &[1.0, 0.0], k, k).unwrap();
            let oracle = s.exact_top_k(Metric::Dot, &[1.0, 0.0], k).unwrap();
            let m: Vec<(u64, f32)> = merged.iter().map(|h| (h.external_id, h.score)).collect();
            let o: Vec<(u64, f32)> = oracle.iter().map(|h| (h.external_id, h.score)).collect();
            assert_eq!(m, o, "k={k}");
        }
        // cosine too
        let merged = s.search(Metric::Cosine, &[1.0, 0.0], 5, 5).unwrap();
        let oracle = s.exact_top_k(Metric::Cosine, &[1.0, 0.0], 5).unwrap();
        assert_eq!(
            merged.iter().map(|h| h.external_id).collect::<Vec<_>>(),
            oracle.iter().map(|h| h.external_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn search_pool_smaller_than_k_rejected() {
        let dir = tmp_dir("poolcheck");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 0.1)).unwrap();
        s.commit().unwrap();
        assert!(matches!(
            s.search(Metric::Dot, &[1.0, 1.0], 5, 3),
            Err(EngineError::Schema(_))
        ));
    }

    #[test]
    fn upsert_commit_checkpoint_reopen_equivalence() {
        let dir = tmp_dir("equivalence");
        let ids_scores_before: Vec<(u64, f32)> = {
            let mut s = Store::open(&dir).unwrap().0;
            s.upsert(rec(1, 0.1)).unwrap();
            s.upsert(rec(2, 0.5)).unwrap();
            s.upsert(rec(3, 0.9)).unwrap();
            s.commit().unwrap();
            let top = s.exact_top_k(Metric::Dot, &[1.0, 0.5], 3).unwrap();
            top.iter().map(|h| (h.external_id, h.score)).collect()
        };
        {
            let mut s = Store::open(&dir).unwrap().0;
            s.checkpoint().unwrap();
            let top = s.exact_top_k(Metric::Dot, &[1.0, 0.5], 3).unwrap();
            let after: Vec<(u64, f32)> = top.iter().map(|h| (h.external_id, h.score)).collect();
            assert_eq!(ids_scores_before, after);
        }
        // Full reopen equivalence: same view from segments+manifest.
        let (s2, recovery) = Store::open(&dir).unwrap();
        assert!(!recovery.rebuilt_from_wal);
        assert!(recovery.manifest_generation.is_some());
        let top2 = s2.exact_top_k(Metric::Dot, &[1.0, 0.5], 3).unwrap();
        let after2: Vec<(u64, f32)> = top2.iter().map(|h| (h.external_id, h.score)).collect();
        assert_eq!(ids_scores_before, after2);
        assert_eq!(s2.len(), 3);
        assert_eq!(s2.dim(), 2);
    }

    #[test]
    fn upsert_overrides_by_last_seq() {
        let dir = tmp_dir("override");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 0.1)).unwrap();
        s.upsert(rec(1, 0.9)).unwrap(); // same id, new vector
        s.commit().unwrap();
        let got = s.get(1).unwrap();
        assert_eq!(got.vector, vec![0.9, 0.45]);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn text_and_meta_survive_checkpoint_and_reopen() {
        let dir = tmp_dir("richrecords");
        let full = Record::new(7, vec![0.3, 0.1])
            .with_text("hello vector engine")
            .with_meta("lang", crate::records::MetaValue::Str("en".into()))
            .with_norm();
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(full.clone()).unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap();
        let (s2, _) = Store::open(&dir).unwrap();
        let got = s2.get(7).unwrap();
        assert_eq!(got.text.as_deref(), Some("hello vector engine"));
        assert_eq!(got.meta.len(), 1);
        assert!(got.norm.is_some());
        assert_eq!(got.norm, full.norm);
    }

    #[test]
    fn seq_order_not_id_order_checkpoint_writes() {
        let dir = tmp_dir("seqorder");
        let mut s = Store::open(&dir).unwrap().0;
        // id 5 first, id 1 second: id order != seq order
        s.upsert(rec(5, 0.1)).unwrap();
        s.upsert(rec(1, 0.5)).unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap(); // must not error on seq ordering
        let (s2, _) = Store::open(&dir).unwrap();
        assert_eq!(s2.len(), 2);
        assert!(s2.get(5).is_some());
        assert!(s2.get(1).is_some());
    }

    #[test]
    fn delete_tombstones_and_checkpoint_absorbs() {
        let dir = tmp_dir("tombstone");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 0.1)).unwrap();
        s.upsert(rec(2, 0.2)).unwrap();
        s.commit().unwrap();
        s.delete(1).unwrap();
        s.commit().unwrap();
        assert_eq!(s.len(), 1);
        assert!(s.get(1).is_none());

        s.checkpoint().unwrap();
        assert_eq!(s.len(), 1);

        let (s2, _) = Store::open(&dir).unwrap();
        assert_eq!(s2.len(), 1);
        assert!(s2.get(1).is_none());
        assert!(s2.get(2).is_some());
    }

    #[test]
    fn delete_unknown_fails() {
        let dir = tmp_dir("delunk");
        let mut s = Store::open(&dir).unwrap().0;
        assert!(matches!(s.delete(99), Err(EngineError::Schema(_))));
        s.upsert(rec(1, 0.1)).unwrap();
        s.commit().unwrap();
        s.delete(1).unwrap();
        s.commit().unwrap();
        // dead id: delete again must fail
        assert!(matches!(s.delete(1), Err(EngineError::Schema(_))));
    }

    #[test]
    fn mixed_dim_rejected() {
        let dir = tmp_dir("mixeddim");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 0.1)).unwrap();
        assert_eq!(s.dim(), 2);
        let wrong = Record::new(2, vec![0.1, 0.2, 0.3]);
        assert!(matches!(s.upsert(wrong), Err(EngineError::Schema(_))));
        // and via a different call path: reopen keeps the dim
        let (mut s2, _) = Store::open(&dir).unwrap();
        s2.upsert(rec(3, 0.2)).unwrap(); // dim 2 ok
        assert!(matches!(
            s2.upsert(Record::new(4, vec![1.0])),
            Err(EngineError::Schema(_))
        ));
    }

    #[test]
    fn uncommitted_writes_vanish_but_committed_survive() {
        let dir = tmp_dir("uncommitted");
        {
            let mut s = Store::open(&dir).unwrap().0;
            s.upsert(rec(1, 0.1)).unwrap();
            s.commit().unwrap();
            s.upsert(rec(2, 0.2)).unwrap(); // never committed
        }
        let (s2, recovery) = Store::open(&dir).unwrap();
        assert_eq!(s2.len(), 1);
        assert!(s2.get(1).is_some());
        assert!(s2.get(2).is_none());
        assert_eq!(recovery.wal.dropped_uncommitted, 1);
    }

    #[test]
    fn rebuild_from_wal_when_manifest_lost() {
        let dir = tmp_dir("manifestloss");
        {
            let mut s = Store::open(&dir).unwrap().0;
            s.upsert(rec(1, 0.1)).unwrap();
            s.upsert(rec(2, 0.5)).unwrap();
            s.commit().unwrap();
            s.checkpoint().unwrap();
        }
        // destroy the manifest: segments are orphaned
        fs::remove_file(dir.join("MANIFEST")).unwrap();
        let (s2, recovery) = Store::open(&dir).unwrap();
        assert!(recovery.rebuilt_from_wal);
        assert_eq!(s2.len(), 2);
        let top = s2.exact_top_k(Metric::Dot, &[1.0, 0.5], 2).unwrap();
        assert_eq!(top.len(), 2);
        // orphaned segments were removed
        let segs = fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .unwrap()
                    .ends_with(".seg")
            })
            .count();
        assert_eq!(segs, 0);
    }

    #[test]
    fn rebuild_from_wal_when_segment_corrupt() {
        let dir = tmp_dir("segcorrupt");
        {
            let mut s = Store::open(&dir).unwrap().0;
            s.upsert(rec(1, 0.1)).unwrap();
            s.commit().unwrap();
            s.checkpoint().unwrap();
        }
        // Corrupt the one segment.
        let seg = fs::read_dir(&dir)
            .unwrap()
            .find(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .unwrap()
                    .ends_with(".seg")
            })
            .unwrap()
            .unwrap()
            .file_name();
        let mut bytes = fs::read(dir.join(&seg)).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(dir.join(&seg), &bytes).unwrap();

        let (s2, recovery) = Store::open(&dir).unwrap();
        assert!(recovery.rebuilt_from_wal);
        assert_eq!(s2.len(), 1);
        assert!(s2.get(1).is_some());
    }

    #[test]
    fn prefix_dim_query_supported() {
        let dir = tmp_dir("prefixdim");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(Record::new(1, vec![1.0, 0.0, 0.0])).unwrap();
        s.upsert(Record::new(2, vec![0.0, 1.0, 0.0])).unwrap();
        s.commit().unwrap();
        // 1-dim query matches the first dim only
        let top = s.exact_top_k(Metric::Dot, &[1.0], 2).unwrap();
        assert_eq!(top[0].external_id, 1);
        // 2-dim query
        let top2 = s.exact_top_k(Metric::Dot, &[0.0, 1.0], 2).unwrap();
        assert_eq!(top2[0].external_id, 2);
        // query longer than stored dim rejected
        assert!(matches!(
            s.exact_top_k(Metric::Dot, &[1.0, 1.0, 1.0, 1.0], 1),
            Err(EngineError::Schema(_))
        ));
    }

    #[test]
    fn cosine_and_l2_metrics() {
        let dir = tmp_dir("metrics");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 1.0)).unwrap(); // [1.0, 0.5]
        s.upsert(rec(2, 2.0)).unwrap(); // [2.0, 1.0] (same direction)
        s.commit().unwrap();
        let cos = s.exact_top_k(Metric::Cosine, &[1.0, 0.5], 2).unwrap();
        // identical direction: both score 1.0
        assert_eq!(cos.len(), 2);
        assert!((cos[0].score - 1.0).abs() < 1e-6);
        let l2 = s.exact_top_k(Metric::L2, &[1.0, 0.5], 2).unwrap();
        assert_eq!(l2[0].external_id, 1); // exact match wins on L2
        assert!(l2[0].score >= -1e-9); // ~0 distance
    }

    #[test]
    fn checkpoint_seq_boundaries_kept() {
        let dir = tmp_dir("seqbounds");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 0.1)).unwrap();
        let c1 = s.commit().unwrap();
        s.checkpoint().unwrap();
        s.upsert(rec(2, 0.2)).unwrap();
        let c2 = s.commit().unwrap();
        assert!(c2 > c1);
        // 2 is in L0 (post-checkpoint), 1 in the segment; view merged.
        assert_eq!(s.len(), 2);
        let (s2, _) = Store::open(&dir).unwrap();
        assert_eq!(s2.len(), 2);
        let top = s2.exact_top_k(Metric::Dot, &[1.0, 0.5], 2).unwrap();
        assert_eq!(top[0].external_id, 2); // 0.25 vs 0.125 dot
    }

    #[test]
    fn empty_store_queries() {
        let dir = tmp_dir("emptyq");
        let s = Store::open(&dir).unwrap().0;
        assert!(s.is_empty());
        let top = s.exact_top_k(Metric::Dot, &[1.0, 1.0], 3).unwrap();
        assert!(top.is_empty());
        assert!(s
            .exact_top_k(Metric::Dot, &[1.0, 1.0], 0)
            .unwrap()
            .is_empty());
        assert!(matches!(
            s.exact_top_k(Metric::Dot, &[], 1),
            Err(EngineError::Schema(_))
        ));
    }

    #[test]
    fn zero_vector_semantics() {
        let dir = tmp_dir("zeroqv");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 0.0)).unwrap(); // stored zero vector
        s.upsert(rec(2, 1.0)).unwrap();
        s.commit().unwrap();
        // zero query under cosine: error
        assert!(matches!(
            s.exact_top_k(Metric::Cosine, &[0.0, 0.0], 1),
            Err(EngineError::Schema(_))
        ));
        // zero-norm stored record under cosine: excluded, not fatal
        let cos = s.exact_top_k(Metric::Cosine, &[1.0, 0.5], 2).unwrap();
        assert_eq!(cos.len(), 1);
        assert_eq!(cos[0].external_id, 2);
        // zero query under dot/L2: fine (all-zero scores)
        let dot = s.exact_top_k(Metric::Dot, &[0.0, 0.0], 2).unwrap();
        assert_eq!(dot.len(), 2);
    }

    #[test]
    fn checkpoint_twice_with_gc() {
        let dir = tmp_dir("twice");
        let mut s = Store::open(&dir).unwrap().0;
        s.upsert(rec(1, 0.1)).unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap();
        s.upsert(rec(2, 0.2)).unwrap();
        s.commit().unwrap();
        s.checkpoint().unwrap();
        assert_eq!(s.len(), 2);
        let (s2, _) = Store::open(&dir).unwrap();
        assert_eq!(s2.len(), 2);
        // exactly one segment file remains
        let segs = fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .unwrap()
                    .ends_with(".seg")
            })
            .count();
        assert_eq!(segs, 1);
    }
}
