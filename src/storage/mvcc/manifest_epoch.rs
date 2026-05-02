// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Cross-process manifest epoch published as `<db>/volumes/epoch`.
//!
//! Reader processes need to detect when the writer has produced a new
//! checkpoint without re-opening the database. The writer bumps a
//! single u64 counter at the end of every successful
//! `checkpoint_cycle_inner` after all per-table manifests are durable.
//! Readers poll the file at refresh time and reload manifests on
//! advance. The poll is the cheap path that gates the (more expensive)
//! per-table reload — comparing two u64s before any I/O work.
//!
//! ## File format
//!
//! Plain 8-byte little-endian u64. Writer rewrites via tmp+rename+fsync
//! so reader sees either the old value or the new value, never a torn
//! intermediate. Atomic rename + parent-dir fsync covers durability.
//!
//! ## Lifecycle
//!
//! - Missing file = epoch 0 (no checkpoint has run yet, or this is a
//!   fresh database).
//! - Bumped only on successful checkpoint with all manifests persisted.
//! - Never decremented. Writer crash + recovery re-reads disk state and
//!   continues from the persisted value.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::core::{Error, Result};

/// Subdirectory under the database path that holds the epoch file.
/// Matches the existing `volumes/` layout where per-table manifests
/// live (`<db>/volumes/<table>/manifest.bin`).
pub const VOLUMES_DIR: &str = "volumes";

/// Filename of the cross-process manifest epoch counter inside `volumes/`.
pub const EPOCH_FILE: &str = "epoch";

/// Compute the absolute path to the epoch file for a database directory.
pub fn epoch_path(db_path: &Path) -> PathBuf {
    db_path.join(VOLUMES_DIR).join(EPOCH_FILE)
}

/// Read the current epoch. Returns 0 if the file does not exist (fresh
/// database, no checkpoint yet). Treats short reads / unexpected size as
/// 0 too — the writer always rewrites the full 8 bytes via tmp+rename, so
/// any non-8-byte read indicates a corrupted file from an older format
/// or a foreign process; in v1 we silently rebuild from 0 next checkpoint.
pub fn read_epoch(db_path: &Path) -> Result<u64> {
    let path = epoch_path(db_path);
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(Error::internal(format!(
                "failed to open epoch file '{}': {}",
                path.display(),
                e
            )))
        }
    };
    let mut buf = [0u8; 8];
    match file.read_exact(&mut buf) {
        Ok(()) => Ok(u64::from_le_bytes(buf)),
        Err(_) => Ok(0),
    }
}

/// Atomically write `value` to the epoch file. Tmp+rename+fsync of both
/// the tmp file and the parent directory.
///
/// Caller must ensure the `<db>/volumes/` directory exists (the engine
/// creates it during open). If it is missing we attempt `create_dir_all`
/// as a best-effort.
pub fn write_epoch(db_path: &Path, value: u64) -> Result<()> {
    let dir = db_path.join(VOLUMES_DIR);
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| {
            Error::internal(format!(
                "failed to create volumes dir '{}': {}",
                dir.display(),
                e
            ))
        })?;
    }
    let final_path = dir.join(EPOCH_FILE);
    let tmp_path = dir.join(format!("{}.tmp", EPOCH_FILE));

    // Open + write + fsync the tmp file BEFORE rename. fsync is required
    // for crash safety; if we crash after rename but before fsync of the
    // file's contents, the rename is durable but the data may not be.
    {
        let mut tmp = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)
            .map_err(|e| {
                Error::internal(format!(
                    "failed to create epoch tmp '{}': {}",
                    tmp_path.display(),
                    e
                ))
            })?;
        tmp.write_all(&value.to_le_bytes()).map_err(|e| {
            Error::internal(format!(
                "failed to write epoch tmp '{}': {}",
                tmp_path.display(),
                e
            ))
        })?;
        tmp.sync_all().map_err(|e| {
            Error::internal(format!(
                "failed to fsync epoch tmp '{}': {}",
                tmp_path.display(),
                e
            ))
        })?;
    }

    fs::rename(&tmp_path, &final_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        Error::internal(format!(
            "failed to rename epoch tmp -> '{}': {}",
            final_path.display(),
            e
        ))
    })?;

    // fsync the directory so the rename is durable across crashes.
    if let Ok(d) = File::open(&dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// Read the current epoch and write back `current + 1` atomically.
/// Returns the new value. Single-writer semantics — caller (the engine
/// during checkpoint) is the only producer, no inter-process CAS needed.
pub fn bump_epoch(db_path: &Path) -> Result<u64> {
    let next = read_epoch(db_path)?.saturating_add(1);
    write_epoch(db_path, next)?;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs as stdfs;

    fn tmp_db_with_volumes() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        stdfs::create_dir_all(tmp.path().join(VOLUMES_DIR)).unwrap();
        tmp
    }

    #[test]
    fn read_epoch_missing_file_returns_zero() {
        let tmp = tempfile::tempdir().unwrap();
        // No volumes/ dir at all.
        assert_eq!(read_epoch(tmp.path()).unwrap(), 0);
        // Empty volumes/ but no epoch file.
        stdfs::create_dir_all(tmp.path().join(VOLUMES_DIR)).unwrap();
        assert_eq!(read_epoch(tmp.path()).unwrap(), 0);
    }

    #[test]
    fn write_epoch_then_read_roundtrips() {
        let tmp = tmp_db_with_volumes();
        write_epoch(tmp.path(), 7).unwrap();
        assert_eq!(read_epoch(tmp.path()).unwrap(), 7);
        write_epoch(tmp.path(), u64::MAX - 1).unwrap();
        assert_eq!(read_epoch(tmp.path()).unwrap(), u64::MAX - 1);
    }

    #[test]
    fn bump_epoch_advances_monotonically() {
        let tmp = tmp_db_with_volumes();
        assert_eq!(bump_epoch(tmp.path()).unwrap(), 1);
        assert_eq!(bump_epoch(tmp.path()).unwrap(), 2);
        assert_eq!(bump_epoch(tmp.path()).unwrap(), 3);
        assert_eq!(read_epoch(tmp.path()).unwrap(), 3);
    }

    #[test]
    fn write_epoch_creates_volumes_dir_if_missing() {
        let tmp = tempfile::tempdir().unwrap();
        // Don't pre-create volumes/ — write_epoch must materialize it.
        write_epoch(tmp.path(), 42).unwrap();
        assert!(tmp.path().join(VOLUMES_DIR).is_dir());
        assert_eq!(read_epoch(tmp.path()).unwrap(), 42);
    }

    #[test]
    fn read_epoch_returns_zero_on_truncated_file() {
        // A short/corrupt epoch file must read as 0 (caller treats this as
        // "no checkpoint" and the next checkpoint rewrites the full 8 bytes).
        let tmp = tmp_db_with_volumes();
        let p = epoch_path(tmp.path());
        stdfs::write(&p, b"abc").unwrap(); // 3 bytes, not 8
        assert_eq!(read_epoch(tmp.path()).unwrap(), 0);
    }

    #[test]
    fn write_epoch_leaves_no_tmp_file() {
        let tmp = tmp_db_with_volumes();
        write_epoch(tmp.path(), 100).unwrap();
        let tmp_path = tmp.path().join(VOLUMES_DIR).join("epoch.tmp");
        assert!(
            !tmp_path.exists(),
            "tmp file must be renamed away, not left behind"
        );
    }
}
