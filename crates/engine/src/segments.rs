//! Immutable on-disk segments: sealed record files + atomic manifest.
//!
//! A segment is a write-once file of canonical records, one per frame
//! (`seq: u64` + record payload, same body codec as the WAL). Segments
//! are named `seg-<gen>-<uuid>.seg` and are never mutated after
//! publish. `MANIFEST` (JSON, atomically replaced) names the active
//! generation, the WAL seq the segments cover (`checkpoint_seq`), the
//! collection dim, and the live segment list.
//!
//! Derivation rule (v0): the WAL never rotates, so segments and the
//! manifest are *derived, rebuildable* artifacts. A missing manifest,
//! a missing/corrupt segment, or a segment claiming a seq above the
//! WAL's committed seq triggers a full rebuild from the WAL (flagged
//! in recovery) rather than data loss. This invariant expires when
//! WAL rotation lands; rotation must pair with archived-segment
//! checksums to keep the rebuild path bounded.
//!
//! Segment file layout:
//!
//! ```text
//! magic "OMVS" | version: u16 | count: u32 | frames...
//! ```
//!
//! Unlike the WAL, sealed segments have no prefix-recovery semantics:
//! a frame that fails CRC, length, or record-body validation fails the
//! whole open (loud), because the manifest promises a complete file
//! and the caller falls back to WAL rebuild.

use crate::codec::{decode_frames, encode_frame, FrameKind};
use crate::error::{EngineError, EngineResult};
use crate::fsutil::atomic_write;
use crate::records::Record;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[cfg(test)]
use std::path::PathBuf;

const SEG_MAGIC: [u8; 4] = *b"OMVS";
const SEG_VERSION: u16 = 1;
/// magic(4) + version(2) + count(4).
const SEG_HEADER_LEN: usize = 10;
const MANIFEST_NAME: &str = "MANIFEST";

/// Manifest naming the active generation and its live segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Generation number; every publish increments it.
    pub generation: u64,
    /// Highest WAL seq fully covered by the listed segments.
    pub checkpoint_seq: u64,
    /// Collection vector dimension (records must match).
    pub dim: u32,
    /// Active segment file names (bare names, no directories).
    pub segments: Vec<String>,
}

/// A segment held open for reads.
#[derive(Debug)]
pub struct SegmentReader {
    /// Bare file name (unique id; also the sort key for stable scan order).
    pub name: String,
    /// Records in seq order, tombstones included.
    records: Vec<SegmentEntry>,
}

/// One entry in a sealed segment.
#[derive(Debug, Clone)]
pub struct SegmentEntry {
    /// WAL seq the record was appended under.
    pub seq: u64,
    pub record: Record,
}

impl SegmentReader {
    /// Open and fully validate a segment file.
    pub fn open(dir: &Path, name: &str) -> EngineResult<Self> {
        let path = dir.join(name);
        let bytes = fs::read(&path)
            .map_err(|e| EngineError::Segment(format!("segment {name} unreadable: {e}")))?;
        if bytes.len() < SEG_HEADER_LEN {
            return Err(EngineError::Segment(format!("segment {name} too short")));
        }
        if bytes[0..4] != SEG_MAGIC {
            return Err(EngineError::Segment(format!("segment {name} bad magic")));
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != SEG_VERSION {
            return Err(EngineError::Segment(format!(
                "segment {name} version {version} != {SEG_VERSION}"
            )));
        }
        let count = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let (frames, consumed) = decode_frames(&bytes[SEG_HEADER_LEN..]);
        if consumed != bytes.len() - SEG_HEADER_LEN {
            // Sealed files must end cleanly; a torn tail means the
            // publish did not complete (temp rename is atomic, so a
            // live segment can never look like this) — corruption.
            return Err(EngineError::Segment(format!(
                "segment {name} has trailing bytes after last frame"
            )));
        }
        if frames.len() != count {
            return Err(EngineError::Segment(format!(
                "segment {name} header count {count} != {} frames",
                frames.len()
            )));
        }
        let mut records = Vec::with_capacity(count);
        let mut last_seq: Option<u64> = None;
        for f in frames {
            if f.kind != FrameKind::Record {
                return Err(EngineError::Segment(format!(
                    "segment {name} contains non-record frame kind {:?}",
                    f.kind
                )));
            }
            if f.payload.len() < 8 {
                return Err(EngineError::Segment(format!(
                    "segment {name} frame body missing seq"
                )));
            }
            let seq = u64::from_le_bytes(f.payload[..8].try_into().unwrap());
            if let Some(prev) = last_seq {
                if seq <= prev {
                    return Err(EngineError::Segment(format!(
                        "segment {name} seq {seq} not monotonic after {prev}"
                    )));
                }
            }
            last_seq = Some(seq);
            let record = Record::decode(&f.payload[8..]).map_err(|e| {
                EngineError::Segment(format!("segment {name} record body invalid: {e}"))
            })?;
            records.push(SegmentEntry { seq, record });
        }
        Ok(SegmentReader {
            name: name.to_string(),
            records,
        })
    }

    /// All entries, seq order, tombstones included. The caller applies
    /// lifecycle/supersede rules.
    pub fn entries(&self) -> &[SegmentEntry] {
        &self.records
    }
}

/// Write a sealed segment file: all records, one frame each, temp
/// write + rename via `atomic_write`. Returns the bare file name.
pub fn write_segment(
    dir: &Path,
    generation: u64,
    entries: &[SegmentEntry],
) -> EngineResult<String> {
    let mut body = Vec::new();
    body.extend_from_slice(&SEG_MAGIC);
    body.extend_from_slice(&SEG_VERSION.to_le_bytes());
    body.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    let mut last_seq: Option<u64> = None;
    for e in entries {
        if let Some(prev) = last_seq {
            if e.seq <= prev {
                return Err(EngineError::Segment(format!(
                    "segment write requires strictly increasing seqs ({} after {})",
                    e.seq, prev
                )));
            }
        }
        last_seq = Some(e.seq);
        let mut fbody = Vec::with_capacity(8 + 16);
        fbody.extend_from_slice(&e.seq.to_le_bytes());
        fbody.extend_from_slice(&e.record.encode()?);
        encode_frame(FrameKind::Record, &fbody, &mut body);
    }
    let name = format!("seg-{generation:020}-{}.seg", uuid_simple());
    atomic_write(dir, &name, &body)?;
    Ok(name)
}

/// Publish the manifest atomically (temp, fsync, rename, dir fsync).
pub fn publish_manifest(dir: &Path, manifest: &Manifest) -> EngineResult<()> {
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|e| EngineError::Segment(format!("manifest serialize: {e}")))?;
    atomic_write(dir, MANIFEST_NAME, &bytes)
}

/// Load the manifest, if present and parseable.
pub fn load_manifest(dir: &Path) -> EngineResult<Option<Manifest>> {
    let bytes = match fs::read(dir.join(MANIFEST_NAME)) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(EngineError::Segment(format!("manifest unreadable: {e}"))),
    };
    let m: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| EngineError::Segment(format!("manifest corrupt: {e}")))?;
    Ok(Some(m))
}

/// A tiny, dependency-free id for segment file names. Not a security
/// boundary — uniqueness only (process id + counter + time).
fn uuid_simple() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}", nanos, c)
}

/// Garbage-collect segment files not named by the current manifest.
/// Non-segment files and the manifest itself are never touched.
pub fn gc_segments(dir: &Path, manifest: &Manifest) -> EngineResult<()> {
    let live: BTreeSet<&str> = manifest.segments.iter().map(|s| s.as_str()).collect();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with("seg-") && name.ends_with(".seg") && !live.contains(name) {
            fs::remove_file(entry.path())?;
        }
    }
    fsutil::fsync_dir_public(dir)
}

/// Public dir-fsync passthrough for `gc_segments` (module boundary).
mod fsutil {
    use super::*;
    pub(crate) fn fsync_dir_public(dir: &Path) -> EngineResult<()> {
        crate::fsutil::fsync_dir(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("omendb-seg-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(seq: u64, id: u64) -> SegmentEntry {
        SegmentEntry {
            seq,
            record: Record::new(id, vec![0.1 * id as f32; 2]),
        }
    }

    #[test]
    fn segment_write_read_round_trip() {
        let dir = tmp_dir("roundtrip");
        let entries = vec![entry(1, 10), entry(2, 20), entry(3, 30)];
        let name = write_segment(&dir, 1, &entries).unwrap();
        let seg = SegmentReader::open(&dir, &name).unwrap();
        assert_eq!(seg.entries().len(), 3);
        assert_eq!(seg.entries()[0].record.external_id, 10);
        assert_eq!(seg.entries()[2].seq, 3);
    }

    #[test]
    fn empty_segment_round_trips() {
        let dir = tmp_dir("empty");
        let name = write_segment(&dir, 0, &[]).unwrap();
        let seg = SegmentReader::open(&dir, &name).unwrap();
        assert_eq!(seg.entries().len(), 0);
    }

    #[test]
    fn nonmonotonic_write_rejected() {
        let dir = tmp_dir("nonmono");
        let entries = vec![entry(5, 1), entry(5, 2)];
        assert!(matches!(
            write_segment(&dir, 1, &entries),
            Err(EngineError::Segment(_))
        ));
    }

    #[test]
    fn corrupt_segment_fails_loud() {
        let dir = tmp_dir("corrupt");
        let name = write_segment(&dir, 1, &[entry(1, 1)]).unwrap();
        let mut bytes = fs::read(dir.join(&name)).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF; // flip a byte inside the frame CRC
        fs::write(dir.join(&name), &bytes).unwrap();
        assert!(matches!(
            SegmentReader::open(&dir, &name),
            Err(EngineError::Segment(_))
        ));
    }

    #[test]
    fn trailing_bytes_fail_loud() {
        let dir = tmp_dir("trailing");
        let name = write_segment(&dir, 1, &[entry(1, 1)]).unwrap();
        let mut bytes = fs::read(dir.join(&name)).unwrap();
        bytes.push(0); // garbage after last frame
        fs::write(dir.join(&name), &bytes).unwrap();
        assert!(matches!(
            SegmentReader::open(&dir, &name),
            Err(EngineError::Segment(_))
        ));
    }

    #[test]
    fn manifest_publish_load_round_trip() {
        let dir = tmp_dir("manifest");
        let m = Manifest {
            generation: 3,
            checkpoint_seq: 42,
            dim: 4,
            segments: vec!["a.seg".into(), "b.seg".into()],
        };
        publish_manifest(&dir, &m).unwrap();
        let back = load_manifest(&dir).unwrap().unwrap();
        assert_eq!(back.generation, 3);
        assert_eq!(back.checkpoint_seq, 42);
        assert_eq!(back.dim, 4);
        assert_eq!(back.segments.len(), 2);
    }

    #[test]
    fn missing_manifest_is_none_not_error() {
        let dir = tmp_dir("nomani");
        assert!(load_manifest(&dir).unwrap().is_none());
    }

    #[test]
    fn corrupt_manifest_fails_loud() {
        let dir = tmp_dir("badmani");
        fs::write(dir.join(MANIFEST_NAME), b"{not json").unwrap();
        assert!(matches!(load_manifest(&dir), Err(EngineError::Segment(_))));
    }

    #[test]
    fn gc_removes_stale_segments_only() {
        let dir = tmp_dir("gc");
        let live = write_segment(&dir, 1, &[entry(1, 1)]).unwrap();
        let stale = write_segment(&dir, 1, &[entry(1, 2)]).unwrap();
        fs::write(dir.join("wal.log"), b"not a wal, just debris").unwrap(); // non-seg file: gc must not touch
        let m = Manifest {
            generation: 1,
            checkpoint_seq: 1,
            dim: 2,
            segments: vec![live.clone()],
        };
        publish_manifest(&dir, &m).unwrap();
        gc_segments(&dir, &m).unwrap();
        assert!(dir.join(&live).exists());
        assert!(!dir.join(&stale).exists());
        assert!(dir.join("wal.log").exists()); // gc never touches non-seg files
    }

    #[test]
    fn segment_names_are_unique() {
        let dir = tmp_dir("uniq");
        let a = write_segment(&dir, 1, &[entry(1, 1)]).unwrap();
        let b = write_segment(&dir, 1, &[entry(1, 2)]).unwrap();
        assert_ne!(a, b);
        assert!(a.starts_with("seg-00000000000000000001-"));
        assert!(b.starts_with("seg-00000000000000000001-"));
    }
}
