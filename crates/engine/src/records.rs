//! Canonical records: the sole mutation authority for the store.
//!
//! A record carries a stable external id (u64, rowid-style), a full-dim
//! F32 vector, optional text, and typed metadata. Lifecycle states are
//! live/tombstone (superseded is reserved in the format but never
//! emitted in v0: an upsert shadows the prior version by external id).
//! Vectors must be finite; per-collection dim is fixed, and queries may
//! address a prefix dim <= the stored dim (Matryoshka prefix
//! coherence).
//!
//! Binary payload layout (all little-endian), shared by the WAL and
//! segment record codecs:
//!
//! ```text
//! u8  lifecycle       (0 live, 1 tombstone, 2 superseded [reserved])
//! u64 external_id
//! u16 dim             (stored dimensionality of this record)
//! f32 x dim           (vector, finite)
//! u32 norm_bits       (IEEE-754 bits of full-dim L2 norm; 0xFFFFFFFF = untracked)
//! u32 text_len, u8 * text_len   (0xFFFFFFFF = no text)
//! u16 meta_count, meta entries:
//!     u16 key_len, key bytes, u8 value kind, value bytes
//! ```
//!
//! Metadata value kinds are flat (i64, f64, bool, string, bytes) with
//! u16 length caps; text uses u32 to avoid a 64KiB ceiling. Nested
//! metadata is out of v0 scope; if it grows, the payload codec
//! switches to a schema'd format (see codec.rs).

use crate::error::{EngineError, EngineResult};

/// Record lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Lifecycle {
    Live = 0,
    Tombstone = 1,
    /// Reserved in the format; v0 never emits or stores it.
    #[allow(dead_code)]
    Superseded = 2,
}

/// Typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
}

impl MetaValue {
    fn encode(&self, out: &mut Vec<u8>) -> EngineResult<()> {
        match self {
            MetaValue::Int(v) => {
                out.push(0);
                out.extend_from_slice(&v.to_le_bytes());
            }
            MetaValue::Float(v) => {
                out.push(1);
                out.extend_from_slice(&v.to_le_bytes());
            }
            MetaValue::Bool(v) => {
                out.push(2);
                out.push(*v as u8);
            }
            MetaValue::Str(s) => {
                if s.len() > u16::MAX as usize {
                    return Err(EngineError::Schema("meta string exceeds u16 length".into()));
                }
                out.push(3);
                out.extend_from_slice(&(s.len() as u16).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            MetaValue::Bytes(b) => {
                if b.len() > u16::MAX as usize {
                    return Err(EngineError::Schema("meta bytes exceed u16 length".into()));
                }
                out.push(4);
                out.extend_from_slice(&(b.len() as u16).to_le_bytes());
                out.extend_from_slice(b);
            }
        }
        Ok(())
    }

    fn decode(buf: &[u8], off: &mut usize) -> EngineResult<Self> {
        match take_u8(buf, off)? {
            0 => Ok(MetaValue::Int(i64::from_le_bytes(
                take_slice(buf, off, 8)?.try_into().unwrap(),
            ))),
            1 => Ok(MetaValue::Float(f64::from_le_bytes(
                take_slice(buf, off, 8)?.try_into().unwrap(),
            ))),
            2 => Ok(MetaValue::Bool(take_u8(buf, off)? != 0)),
            3 => {
                let len = take_u16(buf, off)? as usize;
                let bytes = take_slice(buf, off, len)?;
                let s = String::from_utf8(bytes.to_vec())
                    .map_err(|_| EngineError::Codec("invalid utf-8 in meta string".into()))?;
                Ok(MetaValue::Str(s))
            }
            4 => {
                let len = take_u16(buf, off)? as usize;
                Ok(MetaValue::Bytes(take_slice(buf, off, len)?.to_vec()))
            }
            other => Err(EngineError::Codec(format!("bad meta value kind {other}"))),
        }
    }
}

/// Canonical record.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub external_id: u64,
    pub vector: Vec<f32>,
    /// L2 norm of `vector` when tracked (explicit for cosine/IP).
    pub norm: Option<f32>,
    pub text: Option<String>,
    pub meta: Vec<(String, MetaValue)>,
    pub lifecycle: Lifecycle,
}

const NO_TEXT: u32 = 0xFFFF_FFFF;
/// Norm-bits sentinel meaning "norm untracked". 0xFFFFFFFF is a NaN
/// payload (all-ones) — no finite f32 encodes to it, so a genuinely
/// tracked norm (including 0.0) never collides with the sentinel.
const NO_NORM: u32 = 0xFFFF_FFFF;

impl Record {
    /// Create a live record; norm is untracked until computed.
    pub fn new(external_id: u64, vector: Vec<f32>) -> Self {
        Record {
            external_id,
            vector,
            norm: None,
            text: None,
            meta: Vec::new(),
            lifecycle: Lifecycle::Live,
        }
    }

    /// Track the L2 norm of the current vector explicitly.
    pub fn with_norm(mut self) -> Self {
        self.norm = Some(l2_norm(&self.vector));
        self
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: MetaValue) -> Self {
        self.meta.push((key.into(), value));
        self
    }

    pub fn tombstone(mut self) -> Self {
        self.lifecycle = Lifecycle::Tombstone;
        self
    }

    /// Encode into the record payload format. Schema errors (oversize
    /// dims/text/keys, non-finite values) are raised here, before any
    /// durable write happens.
    pub fn encode(&self) -> EngineResult<Vec<u8>> {
        if self.vector.len() > u16::MAX as usize {
            return Err(EngineError::Schema("vector dim exceeds u16".into()));
        }
        if let Some(t) = &self.text {
            if t.len() > u32::MAX as usize {
                return Err(EngineError::Schema("text exceeds u32 length".into()));
            }
        }
        for (k, _) in &self.meta {
            if k.len() > u16::MAX as usize {
                return Err(EngineError::Schema("meta key exceeds u16 length".into()));
            }
        }
        let mut out = Vec::with_capacity(24 + self.vector.len() * 4);
        out.push(self.lifecycle as u8);
        out.extend_from_slice(&self.external_id.to_le_bytes());
        out.extend_from_slice(&(self.vector.len() as u16).to_le_bytes());
        for v in &self.vector {
            if !v.is_finite() {
                return Err(EngineError::Schema(
                    "vector contains non-finite value".into(),
                ));
            }
            out.extend_from_slice(&v.to_le_bytes());
        }
        match self.norm {
            Some(n) => {
                if !n.is_finite() {
                    return Err(EngineError::Schema(
                        "norm is non-finite (vector magnitude overflows f32)".into(),
                    ));
                }
                out.extend_from_slice(&n.to_bits().to_le_bytes());
            }
            None => out.extend_from_slice(&NO_NORM.to_le_bytes()),
        }
        match &self.text {
            Some(t) => {
                out.extend_from_slice(&(t.len() as u32).to_le_bytes());
                out.extend_from_slice(t.as_bytes());
            }
            None => out.extend_from_slice(&NO_TEXT.to_le_bytes()),
        }
        out.extend_from_slice(&(self.meta.len() as u16).to_le_bytes());
        for (k, v) in &self.meta {
            out.extend_from_slice(&(k.len() as u16).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            v.encode(&mut out)?;
        }
        Ok(out)
    }

    /// Decode a record payload exactly (no trailing bytes allowed).
    pub fn decode(buf: &[u8]) -> EngineResult<Self> {
        let mut off = 0;
        let lifecycle = match take_u8(buf, &mut off)? {
            0 => Lifecycle::Live,
            1 => Lifecycle::Tombstone,
            2 => Lifecycle::Superseded,
            other => return Err(EngineError::Codec(format!("bad lifecycle byte {other}"))),
        };
        let external_id = take_u64(buf, &mut off)?;
        let dim = take_u16(buf, &mut off)? as usize;
        if buf.len() < off + dim * 4 {
            return Err(EngineError::Codec("vector truncated".into()));
        }
        let mut vector = Vec::with_capacity(dim.min(8192));
        for _ in 0..dim {
            let v = f32::from_le_bytes(take_slice(buf, &mut off, 4)?.try_into().unwrap());
            if !v.is_finite() {
                return Err(EngineError::Codec("non-finite vector value".into()));
            }
            vector.push(v);
        }
        let norm_bits = take_u32(buf, &mut off)?;
        let norm = if norm_bits == NO_NORM {
            None
        } else {
            let n = f32::from_bits(norm_bits);
            if !n.is_finite() {
                return Err(EngineError::Codec("non-finite norm".into()));
            }
            Some(n)
        };
        let text_len = take_u32(buf, &mut off)?;
        let text = if text_len == NO_TEXT {
            None
        } else {
            let bytes = take_slice(buf, &mut off, text_len as usize)?;
            Some(
                String::from_utf8(bytes.to_vec())
                    .map_err(|_| EngineError::Codec("invalid utf-8 in text".into()))?,
            )
        };
        let meta_count = take_u16(buf, &mut off)? as usize;
        let mut meta = Vec::with_capacity(meta_count.min(256));
        for _ in 0..meta_count {
            let key_len = take_u16(buf, &mut off)? as usize;
            let key = String::from_utf8(take_slice(buf, &mut off, key_len)?.to_vec())
                .map_err(|_| EngineError::Codec("invalid utf-8 in meta key".into()))?;
            let value = MetaValue::decode(buf, &mut off)?;
            meta.push((key, value));
        }
        if off != buf.len() {
            return Err(EngineError::Codec("trailing bytes after record".into()));
        }
        Ok(Record {
            external_id,
            vector,
            norm,
            text,
            meta,
            lifecycle,
        })
    }
}

/// L2 norm of a vector slice.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn take_u8(buf: &[u8], off: &mut usize) -> EngineResult<u8> {
    if buf.len() < *off + 1 {
        return Err(EngineError::Codec("unexpected end of payload".into()));
    }
    let v = buf[*off];
    *off += 1;
    Ok(v)
}

fn take_u16(buf: &[u8], off: &mut usize) -> EngineResult<u16> {
    if buf.len() < *off + 2 {
        return Err(EngineError::Codec("unexpected end of payload".into()));
    }
    let v = u16::from_le_bytes(buf[*off..*off + 2].try_into().unwrap());
    *off += 2;
    Ok(v)
}

fn take_u32(buf: &[u8], off: &mut usize) -> EngineResult<u32> {
    if buf.len() < *off + 4 {
        return Err(EngineError::Codec("unexpected end of payload".into()));
    }
    let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}

fn take_u64(buf: &[u8], off: &mut usize) -> EngineResult<u64> {
    if buf.len() < *off + 8 {
        return Err(EngineError::Codec("unexpected end of payload".into()));
    }
    let v = u64::from_le_bytes(buf[*off..*off + 8].try_into().unwrap());
    *off += 8;
    Ok(v)
}

fn take_slice<'a>(buf: &'a [u8], off: &mut usize, len: usize) -> EngineResult<&'a [u8]> {
    if buf.len() < *off + len {
        return Err(EngineError::Codec("unexpected end of payload".into()));
    }
    let s = &buf[*off..*off + len];
    *off += len;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Record {
        Record::new(7, vec![0.1, -0.2, 0.3])
            .with_norm()
            .with_text("hello vector")
            .with_meta("lang", MetaValue::Str("en".into()))
            .with_meta("count", MetaValue::Int(42))
            .with_meta("ok", MetaValue::Bool(true))
    }

    #[test]
    fn record_round_trip() {
        let r = sample();
        let back = Record::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn tombstone_round_trip() {
        let r = sample().tombstone();
        let back = Record::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(back.lifecycle, Lifecycle::Tombstone);
        assert_eq!(r, back);
    }

    #[test]
    fn norm_optional_round_trip() {
        let r = Record::new(1, vec![1.0, 0.0]);
        let back = Record::decode(&r.encode().unwrap()).unwrap();
        assert!(back.norm.is_none());
        let back2 = Record::decode(&r.with_norm().encode().unwrap()).unwrap();
        assert!((back2.norm.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn all_meta_kinds_round_trip() {
        let r = Record::new(2, vec![0.5])
            .with_meta("i", MetaValue::Int(-5))
            .with_meta("f", MetaValue::Float(2.5))
            .with_meta("b", MetaValue::Bool(false))
            .with_meta("s", MetaValue::Str("x".repeat(300)))
            .with_meta("y", MetaValue::Bytes(vec![1, 2, 3]));
        let back = Record::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn long_text_round_trip() {
        let text = "word ".repeat(30_000); // > 64 KiB
        let r = Record::new(3, vec![0.5]).with_text(text.clone());
        let back = Record::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(back.text.as_deref(), Some(text.as_str()));
    }

    #[test]
    fn nonfinite_vector_rejected() {
        let r = Record::new(1, vec![f32::NAN]);
        assert!(matches!(r.encode(), Err(EngineError::Schema(_))));
    }

    #[test]
    fn oversize_meta_rejected() {
        let r = Record::new(1, vec![0.0]).with_meta("k", MetaValue::Str("x".repeat(70_000)));
        assert!(matches!(r.encode(), Err(EngineError::Schema(_))));
    }

    #[test]
    fn decode_rejects_truncation() {
        let bytes = sample().encode().unwrap();
        for cut in 1..bytes.len() {
            assert!(Record::decode(&bytes[..cut]).is_err());
        }
    }

    #[test]
    fn zero_vector_tracked_norm_survives_round_trip() {
        let r = Record::new(9, vec![0.0f32; 3]).with_norm();
        let back = Record::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(back.norm, Some(0.0));
        assert_eq!(r, back);
    }

    #[test]
    fn overflow_norm_rejected_at_encode() {
        let r = Record::new(10, vec![1e38f32; 4]).with_norm();
        assert!(matches!(r.encode(), Err(EngineError::Schema(_))));
    }

    #[test]
    fn decode_rejects_nonfinite_norm_payload() {
        // hand-craft a payload with an inf norm where the field lives:
        // lifecycle(1) + id(8) + dim(2) + dim*4 vector bytes
        let r = Record::new(11, vec![0.5f32; 2]);
        let mut bytes = r.encode().unwrap();
        let norm_off = 1 + 8 + 2 + 2 * 4;
        bytes[norm_off..norm_off + 4].copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
        assert!(matches!(Record::decode(&bytes), Err(EngineError::Codec(_))));
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let mut bytes = sample().encode().unwrap();
        bytes.push(0);
        assert!(matches!(Record::decode(&bytes), Err(EngineError::Codec(_))));
    }

    #[test]
    fn decode_rejects_bad_lifecycle_and_kind() {
        let mut bytes = sample().encode().unwrap();
        bytes[0] = 9;
        assert!(matches!(Record::decode(&bytes), Err(EngineError::Codec(_))));
        // corrupt a meta value kind byte: rebuild sample, find a valid kind
        // byte (3 = Str) in the payload and flip it to 9. The Str kind bytes
        // follow a key; sample() ends with meta ("ok", Bool) whose kind byte
        // is the second-to-last byte of the payload (value 1 byte after it).
        let bytes = sample().encode().unwrap();
        let mut bytes2 = bytes.clone();
        let bool_kind = bytes2.len() - 1; // Bool payload is 1 byte: kind + value, value last
                                          // Bool value is the final byte; kind byte is just before it.
        let bool_kind_idx = bool_kind - 1;
        assert_eq!(bytes2[bool_kind_idx], 2);
        bytes2[bool_kind_idx] = 9;
        assert!(Record::decode(&bytes2).is_err());
    }
}
