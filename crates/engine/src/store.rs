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
use crate::records::{Lifecycle, Record};
use crate::segments::{
    gc_segments, load_manifest, publish_manifest, write_segment, Manifest, SegmentEntry,
    SegmentReader,
};
use crate::wal::Wal;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Distance/similarity metric for `exact_top_k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Inner product (larger = better).
    Dot,
    /// Cosine similarity (larger = better); needs nonzero norms.
    Cosine,
    /// Squared L2 distance (smaller = better; score is negated so
    /// higher = better everywhere).
    L2,
}

/// Ranked hit from `exact_top_k`.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub external_id: u64,
    pub score: f32,
    pub seq: u64,
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

/// Ordered search score; higher = better (L2 is negated).
fn score(metric: Metric, q: &[f32], v: &[f32]) -> EngineResult<f32> {
    debug_assert!(q.len() <= v.len(), "query dim validated by caller");
    // Prefix-dim query (Matryoshka): score over the prefix only.
    let v = &v[..q.len()];
    Ok(match metric {
        Metric::Dot => q.iter().zip(v).map(|(a, b)| a * b).sum(),
        Metric::Cosine => {
            let qn = crate::records::l2_norm(q);
            let vn = crate::records::l2_norm(v);
            if qn == 0.0 || vn == 0.0 {
                // Zero norm has no direction; 0/0 would be NaN. The
                // caller pre-validates the query; a zero-norm stored
                // record is unscorable and excluded by the caller.
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
#[derive(Debug)]
pub struct Store {
    dir: PathBuf,
    wal: Wal,
    dim: u32,
    /// Segment files in the manifest, opened.
    segments: Vec<SegmentReader>,
    /// Live ids in segments: id -> seq (sealed segments hold only
    /// live records, so membership == liveness).
    seg_live: HashMap<u64, u64>,
    checkpoint_seq: u64,
    generation: u64,
    l0: L0,
}

impl Store {
    /// Open (or create) a store at `dir`. Exactly one writer process;
    /// no file locking in v0.
    pub fn open<P: AsRef<Path>>(dir: P) -> EngineResult<(Store, StoreRecovery)> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
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

        let mut store = Store {
            seg_live: segments
                .iter()
                .flat_map(|seg| seg.entries())
                .map(|e| (e.record.external_id, e.seq))
                .collect(),
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

    /// Exact flat scan (correctness oracle). `query.len()` may be a
    /// prefix of the stored dim. Live records only.
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
            match score(metric, query, &r.vector) {
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
        self.seg_live = new_seg
            .entries()
            .iter()
            .map(|e| (e.record.external_id, e.seq))
            .collect();
        self.segments = vec![new_seg];
        self.checkpoint_seq = committed;
        self.l0 = L0::default();

        gc_segments(&self.dir, &manifest)?;
        Ok(committed)
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
