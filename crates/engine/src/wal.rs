//! Write-ahead log: the durability backbone of the store.
//!
//! Every mutation is a frame appended here before it exists anywhere
//! else. A frame is `len | kind | payload | crc32c` (see `codec`),
//! where the WAL payload is `seq: u64` followed by kind-specific bytes:
//!
//! - `Record` frame: seq + canonical record payload
//! - `Commit` frame: seq only (exactly 8 bytes)
//! - `Checkpoint` frame: seq + checkpoint body (written by the segment
//!   layer; this module validates seq and skips the body, which is
//!   defined with segments)
//!
//! File layout: `magic "OMVW" | version: u16 | start_seq: u64 | frames`.
//! `start_seq` lets a future rotation start a fresh file at a higher
//! seq (segments own that flow).
//!
//! Durability contract:
//!
//! - `commit()` is the only acknowledgment. It appends the Commit frame
//!   and fsyncs; one fsync covers the record frames before it.
//! - On open, the log is scanned. Record frames with a later Commit
//!   frame are committed; everything else is unacknowledged and
//!   discarded.
//! - **Recovery truncates to the end of the last Commit frame**, not to
//!   the last valid frame. Without this, records appended after a crash
//!   would sit before a *new* Commit frame and be resurrected as
//!   committed data on the next recovery.
//! - A torn tail (partial frame, bad CRC, invalid declared length)
//!   ends the scan: prefix recovery, like SQLite. Physical damage
//!   before the last commit barrier loses that batch — bit rot inside
//!   a committed frame is data loss, detected but not repairable from
//!   the WAL alone.
//! - Frames that scan fine but fail semantic validation (record body
//!   undecodable, non-monotonic seq, malformed Commit) raise
//!   `EngineError::Wal`: loud corruption, never silent truncation.
//! - If `append` returns an error the file may hold a partial frame;
//!   drop the handle and reopen (recovery truncates it) before
//!   further use.
//!
//! v0 limits (pre-rotation): the whole log is read on open and the file
//! grows until the segment layer rotates it; single writer, no file
//! locking (concurrency is out of sprint-1 scope). Fsync correctness is
//! only provable under power loss, which CI cannot simulate — crash
//! tests here are file-level state simulation (torn/partial/uncommitted
//! bytes); power-failure coverage belongs to the production-readiness
//! gate.

use crate::codec::{decode_frames, encode_frame, FrameKind};
use crate::error::{EngineError, EngineResult};
use crate::records::Record;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic bytes at the start of every WAL file.
const MAGIC: [u8; 4] = *b"OMVW";
/// On-disk format version. Bumped on any breaking frame/header change;
/// pre-release 0.x does not keep cross-version compatibility.
const VERSION: u16 = 1;
/// magic(4) + version(2) + start_seq(8).
const HEADER_LEN: usize = 14;
/// First seq handed out by a fresh WAL. Seq 0 is reserved as the "no
/// commit yet" sentinel in `committed_seq`.
const FIRST_SEQ: u64 = 1;
/// Framing bytes per frame beyond the payload: len(4) + kind(1) + crc(4).
const FRAME_BASE: usize = 9;

/// What `Wal::open` recovered from the log.
#[derive(Debug)]
pub struct WalRecovery {
    /// Seq of the last Commit frame (0 = the log never committed).
    pub committed_seq: u64,
    /// Seq the next appended frame will receive.
    pub next_seq: u64,
    /// Seq the file header declares the log starts at.
    pub start_seq: u64,
    /// Committed records in seq order (uncommitted ones are absent).
    pub records: Vec<(u64, Record)>,
    /// Bytes discarded on open: torn tail plus uncommitted-but-valid
    /// frames after the last commit.
    pub truncated_bytes: u64,
    /// Byte offset of the end of the last commit frame (or the header
    /// when nothing committed): where the log ends after recovery.
    pub valid_end: u64,
    /// Valid record frames that were appended after the last commit and
    /// therefore dropped.
    pub dropped_uncommitted: usize,
}

/// Open write-ahead log handle. Single writer; one instance per file.
#[derive(Debug)]
pub struct Wal {
    file: File,
    path: PathBuf,
    start_seq: u64,
    next_seq: u64,
    committed_seq: u64,
}

impl Wal {
    /// Open (creating if needed) and recover a WAL file.
    ///
    /// Recovery is destructive by design: torn tails and uncommitted
    /// frames are truncated away so the log ends at the last commit
    /// barrier. The returned `WalRecovery` describes what survived.
    pub fn open<P: AsRef<Path>>(path: P) -> EngineResult<(Wal, WalRecovery)> {
        let path = path.as_ref().to_path_buf();
        let (mut file, recovery) = open_or_recover(&path)?;
        // Position at the recovery point (end of last commit frame, or
        // the header) so appends continue from there.
        file.seek(SeekFrom::Start(recovery_valid_end(&recovery)))?;
        let wal = Wal {
            file,
            path,
            start_seq: recovery.start_seq,
            next_seq: recovery.next_seq,
            committed_seq: recovery.committed_seq,
        };
        Ok((wal, recovery))
    }

    /// Append a record frame. Not durable until `commit` returns.
    /// Returns the seq assigned to the record.
    pub fn append(&mut self, record: &Record) -> EngineResult<u64> {
        let payload = record.encode()?;
        let seq = self.next_seq;
        let mut body = Vec::with_capacity(8 + payload.len());
        body.extend_from_slice(&seq.to_le_bytes());
        body.extend_from_slice(&payload);
        let mut frame = Vec::with_capacity(FRAME_BASE + body.len());
        encode_frame(FrameKind::Record, &body, &mut frame);
        self.file.write_all(&frame)?;
        self.next_seq += 1;
        Ok(seq)
    }

    /// Append the commit barrier and fsync. Once this returns, every
    /// record appended before it is durable. Returns the barrier's seq.
    pub fn commit(&mut self) -> EngineResult<u64> {
        let seq = self.next_seq;
        let body = seq.to_le_bytes();
        let mut frame = Vec::with_capacity(FRAME_BASE + body.len());
        encode_frame(FrameKind::Commit, &body, &mut frame);
        self.file.write_all(&frame)?;
        self.file.sync_all()?;
        self.next_seq += 1;
        self.committed_seq = seq;
        Ok(seq)
    }

    /// Seq of the last durable commit (0 = nothing committed yet).
    pub fn committed_seq(&self) -> u64 {
        self.committed_seq
    }

    /// Seq the next appended frame will receive.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Seq this file's header declares the log starts at.
    pub fn start_seq(&self) -> u64 {
        self.start_seq
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Byte offset where appends resume after recovery: the end of the
/// last commit frame, or the header when the log never committed.
fn recovery_valid_end(recovery: &WalRecovery) -> u64 {
    // The walk in open_or_recover computes this from frame sizes; the
    // recovery struct carries it implicitly through truncated_bytes
    // against the file length at open time. Store it explicitly instead
    // of recomputing: see `valid_end` in the struct below.
    recovery.valid_end
}

fn header_bytes(start_seq: u64) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..4].copy_from_slice(&MAGIC);
    h[4..6].copy_from_slice(&VERSION.to_le_bytes());
    h[6..HEADER_LEN].copy_from_slice(&start_seq.to_le_bytes());
    h
}

/// Validate the fixed header; returns start_seq.
fn parse_header(buf: &[u8]) -> EngineResult<u64> {
    let (magic, rest) = buf.split_at(4);
    if magic != MAGIC {
        return Err(EngineError::Wal("not a WAL file (bad magic)".into()));
    }
    let (version, rest) = rest.split_at(2);
    let version = u16::from_le_bytes(version.try_into().unwrap());
    if version != VERSION {
        return Err(EngineError::Wal(format!(
            "unsupported WAL version {version} (expected {VERSION})"
        )));
    }
    let start_seq = u64::from_le_bytes(rest.try_into().unwrap());
    if start_seq == 0 {
        return Err(EngineError::Wal("header start_seq must be >= 1".into()));
    }
    Ok(start_seq)
}

/// fsync the parent directory so a freshly created file's directory
/// entry is durable.
fn fsync_dir(path: &Path) -> EngineResult<()> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let dir = File::open(&parent)?;
    dir.sync_all()?;
    Ok(())
}

/// Open (or create) the file and run recovery. Returns the handle and
/// what was recovered. On any semantic corruption the file is left
/// untouched and an error is returned.
fn open_or_recover(path: &Path) -> EngineResult<(File, WalRecovery)> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let file_len = file.metadata()?.len() as usize;
    let created = file_len == 0;

    if created || file_len < HEADER_LEN {
        // Fresh create, or a torn header from a crashed create (the
        // header is written and fsynced before any frame, so a short
        // file can hold no records). Accept only if the bytes present
        // are a prefix of a valid header; anything else is not ours.
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let expect = header_bytes(FIRST_SEQ);
        if !expect.starts_with(buf.as_slice()) {
            return Err(EngineError::Wal("not a WAL file (bad header)".into()));
        }
        let start_seq = FIRST_SEQ;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header_bytes(start_seq))?;
        file.sync_all()?;
        fsync_dir(path)?;
        return Ok((
            file,
            WalRecovery {
                committed_seq: 0,
                next_seq: start_seq,
                start_seq,
                records: Vec::new(),
                truncated_bytes: 0,
                dropped_uncommitted: 0,
                valid_end: HEADER_LEN as u64,
            },
        ));
    }

    let mut buf = Vec::with_capacity(file_len);
    file.read_to_end(&mut buf)?;
    let start_seq = parse_header(&buf[..HEADER_LEN])?;

    let (frames, _consumed) = decode_frames(&buf[HEADER_LEN..]);

    let mut records: Vec<(u64, Record)> = Vec::new();
    let mut pending: Vec<(u64, Record)> = Vec::new();
    let mut last_seq: Option<u64> = None;
    let mut committed_seq = 0u64;
    let mut valid_end = HEADER_LEN as u64;
    let mut frame_end = HEADER_LEN as u64;

    for f in &frames {
        let seq = frame_seq(&f.payload, f.kind)?;
        if let Some(prev) = last_seq {
            if seq <= prev {
                return Err(EngineError::Wal(format!(
                    "non-monotonic seq {seq} after {prev}"
                )));
            }
        }
        if seq < start_seq {
            return Err(EngineError::Wal(format!(
                "seq {seq} below header start_seq {start_seq}"
            )));
        }
        match f.kind {
            FrameKind::Record => {
                let record = Record::decode(&f.payload[8..])
                    .map_err(|e| EngineError::Wal(format!("record frame body invalid: {e}")))?;
                pending.push((seq, record));
            }
            FrameKind::Commit => {
                if f.payload.len() != 8 {
                    return Err(EngineError::Wal(format!(
                        "commit frame body must be exactly 8 bytes, got {}",
                        f.payload.len()
                    )));
                }
                committed_seq = seq;
                records.append(&mut pending);
                // The barrier moves the survive-point past this frame.
                frame_end += (FRAME_BASE + f.payload.len()) as u64;
                valid_end = frame_end;
                last_seq = Some(seq);
                continue;
            }
            FrameKind::Checkpoint => {
                // Written by the segment layer; body is opaque here.
                // Seq was validated above; the frame does not commit
                // records, so it never moves `valid_end` on its own.
            }
        }
        frame_end += (FRAME_BASE + f.payload.len()) as u64;
        last_seq = Some(seq);
    }

    let dropped_uncommitted = pending.len();
    let next_seq = last_seq.map(|s| s + 1).unwrap_or(start_seq);
    let truncated_bytes = file_len as u64 - valid_end;
    if truncated_bytes > 0 {
        // Torn tail and/or uncommitted frames: end the log at the last
        // commit barrier. This is what prevents post-crash appends from
        // resurrecting aborted records under a new commit.
        file.set_len(valid_end)?;
        file.sync_all()?;
    }

    Ok((
        file,
        WalRecovery {
            committed_seq,
            next_seq,
            start_seq,
            records,
            truncated_bytes,
            dropped_uncommitted,
            valid_end,
        },
    ))
}

/// Read and validate the seq prefix of a frame payload.
fn frame_seq(payload: &[u8], kind: FrameKind) -> EngineResult<u64> {
    if payload.len() < 8 {
        return Err(EngineError::Wal(format!(
            "{kind:?} frame body too short for seq ({})",
            payload.len()
        )));
    }
    let _ = kind;
    Ok(u64::from_le_bytes(payload[..8].try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("omendb-wal-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn rec(id: u64, dim_val: f32, dim: usize) -> Record {
        Record::new(id, vec![dim_val; dim])
    }

    #[test]
    fn append_commit_replay_round_trip() {
        let dir = tmp_dir("roundtrip");
        let path = dir.join("wal.log");
        let (mut wal, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 0);
        assert_eq!(recovery.committed_seq, 0);
        let s1 = wal.append(&rec(1, 0.1, 3)).unwrap();
        wal.append(&rec(2, 0.2, 3)).unwrap();
        wal.append(&rec(3, 0.3, 3)).unwrap();
        let c = wal.commit().unwrap();
        assert!(c > s1);
        drop(wal);

        let (wal2, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.committed_seq, c);
        assert_eq!(recovery.records.len(), 3);
        assert_eq!(recovery.records[0].0, s1);
        assert_eq!(recovery.records[0].1.external_id, 1);
        assert_eq!(recovery.records[2].1.external_id, 3);
        assert_eq!(recovery.truncated_bytes, 0);
        // seqs continue from the last frame seen
        assert_eq!(wal2.next_seq(), recovery.next_seq);
    }

    #[test]
    fn uncommitted_records_discarded_and_not_resurrected() {
        let dir = tmp_dir("resurrection");
        let path = dir.join("wal.log");
        let (mut wal, _) = Wal::open(&path).unwrap();
        wal.append(&rec(1, 0.1, 2)).unwrap();
        wal.commit().unwrap();
        wal.append(&rec(2, 0.2, 2)).unwrap(); // appended, never committed
        wal.append(&rec(3, 0.3, 2)).unwrap();
        drop(wal);

        let (mut wal2, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 1);
        assert_eq!(recovery.records[0].1.external_id, 1);
        assert_eq!(recovery.dropped_uncommitted, 2);
        assert!(recovery.truncated_bytes > 0);
        // the log now ends at the commit barrier
        let len_after = fs::metadata(&path).unwrap().len();
        assert!(recovery.valid_end < len_after + 1); // sanity
                                                     // append a new record under a NEW commit; the aborted ones must stay dead
        wal2.append(&rec(4, 0.4, 2)).unwrap();
        wal2.commit().unwrap();
        drop(wal2);

        let (_, recovery) = Wal::open(&path).unwrap();
        let ids: Vec<u64> = recovery
            .records
            .iter()
            .map(|(_, r)| r.external_id)
            .collect();
        assert_eq!(ids, vec![1, 4]);
    }

    #[test]
    fn torn_tail_truncated_and_recovery_idempotent() {
        let dir = tmp_dir("torntail");
        let path = dir.join("wal.log");
        let (mut wal, _) = Wal::open(&path).unwrap();
        wal.append(&rec(1, 0.1, 2)).unwrap();
        wal.commit().unwrap();
        drop(wal);

        // simulate a crash mid-append: garbage partial frame bytes
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0x00, 0x01, 0x02, 0x03, 0x04]).unwrap();
        drop(f);
        let len_with_garbage = fs::metadata(&path).unwrap().len();

        let (_, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 1);
        assert_eq!(recovery.truncated_bytes, 5);
        assert_eq!(fs::metadata(&path).unwrap().len(), len_with_garbage - 5);

        // second recovery changes nothing
        let (_, recovery2) = Wal::open(&path).unwrap();
        assert_eq!(recovery2.truncated_bytes, 0);
        assert_eq!(recovery2.records.len(), 1);
    }

    #[test]
    fn empty_and_missing_files_start_fresh() {
        let dir = tmp_dir("fresh");
        let touched = dir.join("touched.log");
        fs::write(&touched, b"").unwrap();
        let (_, recovery) = Wal::open(&touched).unwrap();
        assert_eq!(recovery.records.len(), 0);
        assert_eq!(recovery.start_seq, FIRST_SEQ);
        // header was written
        let bytes = fs::read(&touched).unwrap();
        assert_eq!(bytes[0..4], MAGIC);
        assert_eq!(bytes.len(), HEADER_LEN);

        let missing = dir.join("missing.log");
        let (mut wal, _) = Wal::open(&missing).unwrap();
        wal.append(&rec(1, 0.5, 1)).unwrap();
        wal.commit().unwrap();
        drop(wal);
        let (_, recovery) = Wal::open(&missing).unwrap();
        assert_eq!(recovery.records.len(), 1);
    }

    #[test]
    fn partial_header_recreated_as_fresh() {
        let dir = tmp_dir("tornheader");
        let path = dir.join("wal.log");
        // crashed create: first 7 bytes of a valid header only
        fs::write(&path, &header_bytes(FIRST_SEQ)[..7]).unwrap();
        let (mut wal, _) = Wal::open(&path).unwrap();
        wal.append(&rec(9, 0.1, 1)).unwrap();
        wal.commit().unwrap();
        drop(wal);
        let (_, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 1);
    }

    #[test]
    fn bad_magic_and_version_fail_loud() {
        let dir = tmp_dir("badheader");
        let path = dir.join("wal.log");
        let mut bad = vec![0u8; HEADER_LEN];
        bad[0..4].copy_from_slice(b"NOPE");
        fs::write(&path, &bad).unwrap();
        assert!(matches!(Wal::open(&path), Err(EngineError::Wal(_))));

        let path2 = dir.join("wal2.log");
        let mut future = header_bytes(FIRST_SEQ).to_vec();
        future[4..6].copy_from_slice(&99u16.to_le_bytes());
        fs::write(&path2, &future).unwrap();
        assert!(matches!(Wal::open(&path2), Err(EngineError::Wal(_))));
    }

    #[test]
    fn midfile_crc_corruption_is_prefix_recovery() {
        let dir = tmp_dir("crcflip");
        let path = dir.join("wal.log");
        let (mut wal, _) = Wal::open(&path).unwrap();
        wal.append(&rec(1, 0.1, 2)).unwrap();
        wal.append(&rec(2, 0.2, 2)).unwrap(); // carries unique text marker
        wal.commit().unwrap();
        drop(wal);

        // flip one payload byte inside record 2's frame
        let mut bytes = fs::read(&path).unwrap();
        let marker = b"0.2-is-not-text"; // not present; find by vector bytes instead
        let _ = marker;
        // record 2's vector is [0.2f32; 2]: locate its bit pattern
        let needle = 0.2f32.to_le_bytes();
        let pos = bytes
            .windows(4)
            .position(|w| w == needle)
            .expect("vector bytes present");
        bytes[pos] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();

        // scan stops at the damaged frame: no commit frame is reachable,
        // so nothing is acknowledged and everything is truncated.
        let (_, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 0);
        assert_eq!(recovery.committed_seq, 0);
        assert_eq!(fs::metadata(&path).unwrap().len(), HEADER_LEN as u64);
    }

    #[test]
    fn seq_regression_fails_loud() {
        let dir = tmp_dir("seqregress");
        let path = dir.join("wal.log");
        let r = rec(1, 0.1, 1);
        let mut buf = header_bytes(FIRST_SEQ).to_vec();
        let mut body = Vec::new();
        body.extend_from_slice(&5u64.to_le_bytes());
        body.extend_from_slice(&r.encode().unwrap());
        encode_frame(FrameKind::Record, &body, &mut buf);
        let mut body2 = Vec::new();
        body2.extend_from_slice(&3u64.to_le_bytes()); // regresses
        body2.extend_from_slice(&r.encode().unwrap());
        encode_frame(FrameKind::Record, &body2, &mut buf);
        fs::write(&path, &buf).unwrap();

        assert!(matches!(Wal::open(&path), Err(EngineError::Wal(_))));
    }

    #[test]
    fn malformed_commit_and_record_bodies_fail_loud() {
        let dir = tmp_dir("badbody");
        // commit frame with 9-byte body
        let path = dir.join("commit9.log");
        let mut buf = header_bytes(FIRST_SEQ).to_vec();
        encode_frame(FrameKind::Commit, &[0u8; 9], &mut buf);
        fs::write(&path, &buf).unwrap();
        assert!(matches!(Wal::open(&path), Err(EngineError::Wal(_))));

        // record frame with garbage body (valid CRC)
        let path2 = dir.join("garbage.log");
        let mut buf2 = header_bytes(FIRST_SEQ).to_vec();
        let mut body = Vec::new();
        body.extend_from_slice(&1u64.to_le_bytes());
        body.extend_from_slice(&[0xEE; 5]); // not a record payload
        encode_frame(FrameKind::Record, &body, &mut buf2);
        fs::write(&path2, &buf2).unwrap();
        assert!(matches!(Wal::open(&path2), Err(EngineError::Wal(_))));
    }

    #[test]
    fn header_only_log_replays_empty() {
        let dir = tmp_dir("headeronly");
        let path = dir.join("wal.log");
        let (wal, _) = Wal::open(&path).unwrap();
        drop(wal);
        let (_, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 0);
        assert_eq!(recovery.committed_seq, 0);
        assert_eq!(recovery.next_seq, FIRST_SEQ);
    }

    #[test]
    fn tombstones_replay_with_lifecycle() {
        let dir = tmp_dir("tombstone");
        let path = dir.join("wal.log");
        let (mut wal, _) = Wal::open(&path).unwrap();
        wal.append(&rec(1, 0.1, 1)).unwrap();
        let t = rec(1, 0.1, 1).tombstone();
        wal.append(&t).unwrap();
        wal.commit().unwrap();
        drop(wal);
        let (_, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 2);
        assert_eq!(
            recovery.records[1].1.lifecycle,
            crate::records::Lifecycle::Tombstone
        );
    }

    #[test]
    fn checkpoint_frame_validated_but_skipped() {
        let dir = tmp_dir("checkpoint");
        let path = dir.join("wal.log");
        let mut buf = header_bytes(FIRST_SEQ).to_vec();
        let mut body = Vec::new();
        body.extend_from_slice(&1u64.to_le_bytes());
        body.extend_from_slice(&[0xAA; 4]); // body beyond seq: opaque here
        encode_frame(FrameKind::Checkpoint, &body, &mut buf);
        fs::write(&path, &buf).unwrap();

        // No commit exists, so even the valid checkpoint frame is
        // truncated away — but the open itself must succeed and the
        // seq must be consumed by the monotonic walk.
        let (_, recovery) = Wal::open(&path).unwrap();
        assert_eq!(recovery.records.len(), 0);
        assert_eq!(recovery.committed_seq, 0);
        assert_eq!(recovery.next_seq, 2);
        assert!(recovery.truncated_bytes > 0);
    }

    #[test]
    fn multiple_commits_boundaries_respected() {
        let dir = tmp_dir("multicommit");
        let path = dir.join("wal.log");
        let (mut wal, _) = Wal::open(&path).unwrap();
        wal.append(&rec(1, 0.1, 1)).unwrap();
        wal.append(&rec(2, 0.2, 1)).unwrap();
        let c1 = wal.commit().unwrap();
        wal.append(&rec(3, 0.3, 1)).unwrap();
        let c2 = wal.commit().unwrap();
        wal.append(&rec(4, 0.4, 1)).unwrap(); // never committed
        drop(wal);

        let (_, recovery) = Wal::open(&path).unwrap();
        let ids: Vec<u64> = recovery
            .records
            .iter()
            .map(|(_, r)| r.external_id)
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(recovery.committed_seq, c2);
        assert!(c2 > c1);
        assert_eq!(recovery.dropped_uncommitted, 1);
        let seqs: Vec<u64> = recovery.records.iter().map(|(s, _)| *s).collect();
        assert!(seqs.windows(2).all(|w| w[0] < w[1]));
    }
}
