//! SIFT-100K floor dataset loader (pre-converted binary format).
//!
//! Source: ann-benchmarks sift-128-euclidean.hdf5 (train/test),
//! converted once to a raw little-endian format so the Rust loader
//! has no HDF5 dependency:
//!
//! ```text
//! u64 n_train, u64 dim, f32 x n_train*dim,
//! u64 n_query,         f32 x n_query*dim
//! ```
//!
//! The conversion script lives in the repo docs (bench/README);
//! the data itself is fetched, cached under bench/data/, and never
//! committed. 100K train + 1K queries = the architecture §10 floor
//! calibration set.

use crate::error::{EngineError, EngineResult};
use crate::records::Record;

pub const SIFT_DIM: usize = 128;
pub const SIFT_TRAIN: usize = 100_000;
pub const SIFT_QUERY: usize = 1_000;

/// Parsed floor dataset.
pub struct SiftData {
    pub train: Vec<Record>,
    pub queries: Vec<Vec<f32>>,
}

/// Load a `sift100k.bin` file (see module docs for the layout).
pub fn load(path: &std::path::Path) -> EngineResult<SiftData> {
    let bytes = std::fs::read(path)
        .map_err(|e| EngineError::Codec(format!("sift data unreadable: {e}")))?;
    if bytes.len() < 24 {
        return Err(EngineError::Codec("sift data too short".into()));
    }
    let mut off = 0usize;
    let take_u64 = |off: &mut usize| -> u64 {
        let v = u64::from_le_bytes(bytes[*off..*off + 8].try_into().unwrap());
        *off += 8;
        v
    };
    let n_train = take_u64(&mut off) as usize;
    let dim = take_u64(&mut off) as usize;
    if dim != SIFT_DIM || n_train > 10_000_000 {
        return Err(EngineError::Codec(format!(
            "unexpected header: n_train={n_train} dim={dim} (want dim {SIFT_DIM})"
        )));
    }
    let need = n_train * dim * 4;
    if bytes.len() < off + need + 8 {
        return Err(EngineError::Codec("sift train section truncated".into()));
    }
    let mut train = Vec::with_capacity(n_train);
    for i in 0..n_train as u64 {
        let start = off + i as usize * dim * 4;
        let v: Vec<f32> = (0..dim)
            .map(|d| {
                f32::from_le_bytes(bytes[start + d * 4..start + d * 4 + 4].try_into().unwrap())
            })
            .collect();
        train.push(Record::new(i + 1, v));
    }
    off += need;
    let n_query = take_u64(&mut off) as usize;
    if bytes.len() < off + n_query * dim * 4 {
        return Err(EngineError::Codec("sift query section truncated".into()));
    }
    let mut queries = Vec::with_capacity(n_query);
    for i in 0..n_query {
        let start = off + i * dim * 4;
        let q: Vec<f32> = (0..dim)
            .map(|d| {
                f32::from_le_bytes(bytes[start + d * 4..start + d * 4 + 4].try_into().unwrap())
            })
            .collect();
        queries.push(q);
    }
    Ok(SiftData { train, queries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_tiny_sift_format() {
        // Hand-write a 3x2-train 2x2-query file and load it.
        let mut buf = Vec::new();
        buf.extend_from_slice(&3u64.to_le_bytes()); // n_train
        buf.extend_from_slice(&128u64.to_le_bytes()); // dim
        for i in 0..3u64 {
            for d in 0..128 {
                buf.extend_from_slice(&((i * 128 + d) as f32).to_le_bytes());
            }
        }
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_query
        for _ in 0..2 {
            for d in 0..128 {
                buf.extend_from_slice(&(d as f32).to_le_bytes());
            }
        }
        let dir = std::env::temp_dir().join("omendb-sift-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tiny.bin");
        std::fs::write(&p, &buf).unwrap();
        let d = load(&p).unwrap();
        assert_eq!(d.train.len(), 3);
        assert_eq!(d.queries.len(), 2);
        assert_eq!(d.train[0].vector.len(), 128);
        assert_eq!(d.train[1].vector[0], 128.0); // i=1,d=0 -> 128
        assert_eq!(d.queries[1][5], 5.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truncated_and_bad_header_fail() {
        let dir = std::env::temp_dir().join("omendb-sift-tests-bad");
        std::fs::create_dir_all(&dir).unwrap();
        // bad dim
        std::fs::write(dir.join("b1.bin"), [0u8; 24]).unwrap();
        assert!(load(&dir.join("b1.bin")).is_err());
        // too short
        std::fs::write(dir.join("b2.bin"), b"1234").unwrap();
        assert!(load(&dir.join("b2.bin")).is_err());
        // dim=64 header, no body
        let mut b3 = Vec::new();
        b3.extend_from_slice(&1u64.to_le_bytes());
        b3.extend_from_slice(&64u64.to_le_bytes());
        std::fs::write(dir.join("b3.bin"), &b3).unwrap();
        assert!(load(&dir.join("b3.bin")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
