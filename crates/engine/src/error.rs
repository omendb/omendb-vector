//! Engine error types.

use std::fmt;

/// Errors produced by the engine: I/O, binary codec, schema validation,
/// WAL corruption, and segment corruption.
#[derive(Debug)]
pub enum EngineError {
    /// Underlying filesystem or I/O failure.
    Io(std::io::Error),
    /// A value failed to encode or decode (malformed or truncated bytes).
    Codec(String),
    /// A record or query violated the collection schema (dimension,
    /// finiteness, id, metric mismatch).
    Schema(String),
    /// The WAL is structurally valid to scan but contains a frame whose
    /// contents fail verification. Treated as corruption: fail loud,
    /// never silently truncate committed data.
    Wal(String),
    /// An on-disk segment failed magic/version/checksum verification.
    /// Segments are canonical storage after a checkpoint, so corruption
    /// is data loss, not something to rebuild from.
    Segment(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Io(e) => write!(f, "io error: {e}"),
            EngineError::Codec(m) => write!(f, "codec error: {m}"),
            EngineError::Schema(m) => write!(f, "schema error: {m}"),
            EngineError::Wal(m) => write!(f, "wal error: {m}"),
            EngineError::Segment(m) => write!(f, "segment error: {m}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EngineError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Io(e)
    }
}

/// Convenience alias used across the engine.
pub type EngineResult<T> = Result<T, EngineError>;
