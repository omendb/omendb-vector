//! Filesystem durability helpers shared by the WAL and segment layers.

use crate::error::EngineResult;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// fsync a directory so rename/create operations inside it are durable.
pub(crate) fn fsync_dir(path: &Path) -> EngineResult<()> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => Path::new(".").to_path_buf(),
    };
    let dir = File::open(parent)?;
    dir.sync_all()?;
    Ok(())
}

/// Atomically replace `dir/final_name` with `bytes`: write the temp
/// file, fsync it, rename over the target, fsync the directory. A
/// crash leaves either the old or the new content, never a partial
/// mix — rename is atomic on the POSIX systems v0 targets.
///
/// A leftover `.tmp` file from a crashed publish is harmless: the next
/// publish truncates and overwrites it.
pub(crate) fn atomic_write(dir: &Path, final_name: &str, bytes: &[u8]) -> EngineResult<()> {
    let tmp = dir.join(format!("{final_name}.tmp"));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, dir.join(final_name))?;
    fsync_dir(dir)
}
