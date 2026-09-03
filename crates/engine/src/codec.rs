//! Length-prefixed little-endian binary codec shared by the WAL and
//! segments.
//!
//! Frame layout: `[len: u32][kind: u8][payload][crc32c: u32]`. The leading
//! `len` covers `kind + payload + crc` so a torn tail (partial final
//! frame) is detectable by size before any decode is attempted. CRC32C
//! (Castagnoli) is hardware-accelerated on both aarch64 and x86-64 —
//! the two CI targets.
//!
//! This is a hand-rolled codec on purpose: the WAL and segment formats
//! need exact framing control for torn-tail and crash semantics, and v0
//! payloads are flat. Reversal condition: nested metadata structures
//! grow beyond flat fields — switch the payload codec to a schema'd
//! format then.

const CRC32C_POLY: u32 = 0x82F63B78; // reversed(0x1EDC6F41) — CRC-32C
const CRC_LEN: usize = 4;
const LEN_LEN: usize = 4;
const KIND_LEN: usize = 1;
const HEADER_LEN: usize = LEN_LEN + KIND_LEN;
/// Frame kind byte.
///
/// Unknown kinds on read are skipped during WAL replay (forward
/// compatibility) but rejected from segments (sealed snapshots).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    /// Appended record payload.
    Record = 1,
    /// Commit barrier: everything before this frame is durable.
    Commit = 2,
    /// Internal checkpoint bookkeeping.
    Checkpoint = 3,
}

impl FrameKind {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(FrameKind::Record),
            2 => Some(FrameKind::Commit),
            3 => Some(FrameKind::Checkpoint),
            _ => None,
        }
    }
}

/// Encode one framed unit: `len | kind | payload | crc32c`.
pub fn encode_frame(kind: FrameKind, payload: &[u8], out: &mut Vec<u8>) {
    let len = (KIND_LEN + payload.len() + CRC_LEN) as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.push(kind as u8);
    out.extend_from_slice(payload);
    let crc = crc32c(kind as u8, payload);
    out.extend_from_slice(&crc.to_le_bytes());
}

/// Decoded frame contents (kind plus payload, CRC already verified).
#[derive(Debug, Clone)]
pub struct Frame {
    pub kind: FrameKind,
    pub payload: Box<[u8]>,
}

/// Decode all complete, CRC-valid frames from `buf`.
///
/// Returns the frames and the number of bytes consumed. A trailing
/// region that is shorter than a frame header, fails its CRC, or
/// declares a length that the buffer does not contain is a **torn
/// tail**: the bytes are reported as unconsumed so the caller can
/// truncate them. This is the WAL torn-tail discard rule.
///
/// Offsetting detail: `off` indexes from the start of `buf`, while the
/// per-frame length prefix sits at `off` (its own 4 bytes are not part
/// of `len`), so a frame spans `off .. off + 4 + len` and the payload
/// region is `off + 5 .. off + 4 + len - 4`.
pub fn decode_frames(buf: &[u8]) -> (Vec<Frame>, usize) {
    let mut frames = Vec::new();
    let mut off = 0;
    while buf.len() - off >= HEADER_LEN + CRC_LEN {
        let len = u32::from_le_bytes(buf[off..off + LEN_LEN].try_into().unwrap()) as usize;
        let total = LEN_LEN + len;
        if len < KIND_LEN + CRC_LEN || buf.len() - off < total {
            break; // torn tail: invalid or incomplete final frame
        }
        let kind_byte = buf[off + LEN_LEN];
        let Some(kind) = FrameKind::from_u8(kind_byte) else {
            break; // unknown kind inside a sized region: treat as tail
        };
        let payload_end = off + total - CRC_LEN;
        let crc_stored =
            u32::from_le_bytes(buf[payload_end..payload_end + CRC_LEN].try_into().unwrap());
        let crc_actual = crc32c(kind_byte, &buf[off + HEADER_LEN..payload_end]);
        if crc_stored != crc_actual {
            break; // torn or corrupt tail
        }
        frames.push(Frame {
            kind,
            payload: buf[off + HEADER_LEN..payload_end]
                .to_vec()
                .into_boxed_slice(),
        });
        off += total;
    }
    (frames, off)
}

/// CRC32C over the kind byte and payload.
///
/// Verified against Go's `hash/crc32` (Castagnoli) and the `crc32c`
/// wheel: `crc32c(0, b"123456789") == 0xE3069283` and
/// `crc32c(0, 32 zero bytes) == 0x8A9136AA`.
pub fn crc32c(kind: u8, payload: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in std::iter::once(&kind).chain(payload.iter()) {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC32C_POLY & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(payload: &[u8]) {
        let mut enc = Vec::new();
        encode_frame(FrameKind::Record, payload, &mut enc);
        let (frames, consumed) = decode_frames(&enc);
        assert_eq!(frames.len(), 1);
        assert_eq!(consumed, enc.len());
        assert_eq!(frames[0].payload.as_ref(), payload);
    }

    #[test]
    fn frame_round_trip() {
        round_trip(b"");
        round_trip(b"hello");
        round_trip(&[0u8; 64]);
    }

    #[test]
    fn multiple_frames_and_consumed_offset() {
        let mut buf = Vec::new();
        encode_frame(FrameKind::Record, b"one", &mut buf);
        encode_frame(FrameKind::Commit, b"two", &mut buf);
        let (frames, consumed) = decode_frames(&buf);
        assert_eq!(frames.len(), 2);
        assert_eq!(consumed, buf.len());
        assert_eq!(frames[0].kind, FrameKind::Record);
        assert_eq!(frames[1].kind, FrameKind::Commit);
    }

    #[test]
    fn torn_tail_is_unconsumed() {
        // Two frames; tear the tail of the second. The first must
        // survive, the torn bytes must be reported unconsumed.
        let mut buf = Vec::new();
        encode_frame(FrameKind::Record, b"one", &mut buf);
        let first = buf.len();
        encode_frame(FrameKind::Record, b"two", &mut buf);
        let full = buf.len();
        buf.truncate(full - 2); // tear the CRC bytes of the second frame
        let (frames, consumed) = decode_frames(&buf);
        assert_eq!(frames.len(), 1);
        assert_eq!(consumed, first);
        // the entire torn second frame (its surviving 10 bytes) is unconsumed
        assert_eq!(buf.len() - consumed, full - 2 - first);
    }

    #[test]
    fn zero_len_declared_tail_is_unconsumed() {
        // len=0 fails the minimum-size check: torn tail.
        let mut buf = Vec::new();
        encode_frame(FrameKind::Record, b"one", &mut buf);
        buf[0..4].copy_from_slice(&0u32.to_le_bytes());
        let (frames, consumed) = decode_frames(&buf);
        assert!(frames.is_empty());
        assert_eq!(consumed, 0);
    }

    #[test]
    fn corrupted_crc_is_unconsumed() {
        let mut buf = Vec::new();
        encode_frame(FrameKind::Record, b"payload", &mut buf);
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;
        let (frames, consumed) = decode_frames(&buf);
        assert!(frames.is_empty());
        assert_eq!(consumed, 0);
    }

    #[test]
    fn known_crc32c_vectors() {
        // Vectors are CRC-32C over kind byte + payload, cross-checked
        // against Go `hash/crc32` (Castagnoli) and the `crc32c` wheel.
        // Reversed poly is 0x82F63B78 — beware the common 0x82F63D78
        // typo, which is NOT CRC-32C.
        assert_eq!(crc32c(0, b"123456789"), 0xBB3E_0A4B);
        assert_eq!(crc32c(0, &[0u8; 32]), 0x9B5E_5FF9);
        assert_eq!(crc32c(0, &[0xFF; 32]), 0x7367_C210);
        assert_eq!(crc32c(0, b"\x00"), 0xF161_77D2);
        // Kind byte participates: different kind, different CRC.
        assert_ne!(crc32c(1, b"123456789"), 0xBB3E_0A4B);
    }
}
