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

//! File-based database locking to prevent concurrent access from multiple processes.
//!
//! This module provides OS-level file locking to ensure only one process can
//! access a database directory at a time. It uses:
//! - `flock()` on Unix systems (Linux, macOS)
//! - `LockFileEx()` on Windows
//!

use std::fs::{self, File, OpenOptions};
#[cfg(not(target_os = "wasi"))]
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::core::{Error, Result};

/// Lock acquisition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// Exclusive lock — only one holder at a time. Used by writable engines
    /// to ensure single-writer semantics.
    Exclusive,
    /// Shared lock — multiple holders allowed concurrently, but blocks
    /// any exclusive-lock acquisition. Used by read-only engines so
    /// multiple readers can coexist while still preventing a writer
    /// from corrupting the data they observe.
    Shared,
}

/// Represents a lock on a database directory.
///
/// The lock is automatically released when this struct is dropped. The
/// lock mode (exclusive vs shared) is recorded for diagnostics.
///
/// `file` is `None` only on the "lockless shared" fallback: `Shared` mode
/// on a read-only mount where `db.lock` is missing and cannot be created.
/// A genuinely read-only mount cannot have racing writers, so skipping
/// `flock` there is safe — this supports packaged databases shipped to
/// read-only filesystems without a pre-created lock artifact.
#[derive(Debug)]
pub struct FileLock {
    /// The lock file handle (kept open to maintain the lock). `None` only
    /// for the lockless shared fallback described above.
    #[allow(dead_code)]
    file: Option<File>,
    /// Path to the lock file
    path: PathBuf,
    /// Mode the lock was acquired in
    mode: LockMode,
}

impl FileLock {
    /// Acquire an exclusive lock on the database directory.
    ///
    /// Creates a `db.lock` file in the database directory and locks it using
    /// OS-level file locking. Returns an error if the lock cannot be acquired
    /// (typically because another process has it).
    ///
    /// # Arguments
    /// * `db_path` - Path to the database directory
    ///
    /// # Returns
    /// * `Ok(FileLock)` - Lock was acquired successfully
    /// * `Err` - Lock could not be acquired (database is in use by another process)
    ///
    /// # Example
    /// ```ignore
    /// let lock = FileLock::acquire("/path/to/db")?;
    /// // ... use database ...
    /// // Lock is released when `lock` is dropped
    /// ```
    pub fn acquire(db_path: impl AsRef<Path>) -> Result<Self> {
        Self::acquire_with_mode(db_path, LockMode::Exclusive)
    }

    /// Acquire a shared (read-only) lock on the database directory.
    ///
    /// Multiple shared locks may be held concurrently (multiple readers),
    /// but a shared lock blocks any subsequent exclusive-lock acquisition
    /// (and vice versa). Used by read-only engines.
    pub fn acquire_shared(db_path: impl AsRef<Path>) -> Result<Self> {
        Self::acquire_with_mode(db_path, LockMode::Shared)
    }

    /// Acquire a lock with the specified mode.
    pub fn acquire_with_mode(db_path: impl AsRef<Path>, mode: LockMode) -> Result<Self> {
        let db_path = db_path.as_ref();

        // For Exclusive locks (writable opens) the directory must exist —
        // create it if missing, hard-error otherwise. For Shared locks
        // (read-only opens) attempt creation but ignore failures: on a
        // genuinely read-only mount the dir already exists (the caller —
        // `open_read_only` — checked above us), and `create_dir_all` would
        // fail with EROFS or EACCES even though there is nothing to do.
        match mode {
            LockMode::Exclusive => {
                fs::create_dir_all(db_path).map_err(|e| {
                    Error::internal(format!("failed to create database directory: {}", e))
                })?;
            }
            LockMode::Shared => {
                let _ = fs::create_dir_all(db_path);
            }
        }

        // Lock file path
        let lock_file_path = db_path.join("db.lock");

        // Choose how to open the lock file.
        //
        // Exclusive: must be writable (we record our PID into it). Created
        // if missing.
        //
        // Shared: prefer a read-only open so that read-only directories /
        // mounts (chmod -w on the dir or db.lock) still work — that is a
        // core deployment scenario for `?read_only=true` /
        // `open_read_only`. Fall back to create+write+read only when the
        // file is missing AND we can create it. If both fail because the
        // FILESYSTEM is mounted read-only at the kernel level, drop into
        // the lockless shared fallback — no process on this mount can
        // ever take a writer lock, so skipping `flock` is sound. We do
        // NOT take the lockless path on a chmod-only EACCES (writable
        // mount, restricted dir): permissions can be lifted at any time
        // and another process / user could acquire a writer lock,
        // breaking the read-only contract for any reader that opted out
        // of `flock`. On Unix `flock` requires a file descriptor
        // regardless of access mode, so a read-only fd is sufficient to
        // acquire `LOCK_SH`.
        let file: Option<File> = match mode {
            LockMode::Exclusive => Some(
                OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(&lock_file_path)
                    .map_err(|e| Error::internal(format!("failed to open lock file: {}", e)))?,
            ),
            LockMode::Shared => match OpenOptions::new().read(true).open(&lock_file_path) {
                Ok(f) => Some(f),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // File missing — try to create it. On a read-only mount
                    // the create will fail with EACCES / EROFS; check the
                    // mount-level read-only flag to decide whether the
                    // lockless shared fallback is safe.
                    match OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .read(true)
                        .write(true)
                        .open(&lock_file_path)
                    {
                        Ok(f) => Some(f),
                        Err(create_err)
                            if (create_err.kind() == std::io::ErrorKind::PermissionDenied
                                || create_err.raw_os_error() == Some(libc_erofs()))
                                && is_path_on_readonly_mount(db_path) =>
                        {
                            None
                        }
                        Err(create_err)
                            if create_err.kind() == std::io::ErrorKind::PermissionDenied =>
                        {
                            return Err(Error::internal(format!(
                                "failed to create lock file on a writable filesystem (the \
                                 directory is not writable, but the mount is not read-only at \
                                 the kernel level either): {}. Either ship a `db.lock` file \
                                 with the database, mount the filesystem read-only, or open \
                                 the database from a writable directory.",
                                create_err
                            )));
                        }
                        Err(create_err) => {
                            return Err(Error::internal(format!(
                                "failed to open lock file: {}",
                                create_err
                            )));
                        }
                    }
                }
                Err(e) => {
                    return Err(Error::internal(format!("failed to open lock file: {}", e)));
                }
            },
        };

        // Try to acquire the lock (platform-specific). Skip when we have
        // no file to lock — lockless shared fallback, see struct doc.
        // This must happen BEFORE any file content modification.
        if let Some(ref f) = file {
            acquire_lock(f, mode)?;
        }

        // Only write our PID for exclusive locks. Shared locks may have many
        // holders, so a single PID would be misleading; leave the existing
        // file content (which may show the original exclusive holder).
        // Also: on a Shared open we may have only a read-only fd (or no fd
        // at all in the lockless fallback), so writes would fail anyway.
        #[cfg(not(target_os = "wasi"))]
        if mode == LockMode::Exclusive {
            if let Some(ref f) = file {
                #[allow(unused_mut)]
                let mut f = f;
                f.set_len(0)
                    .map_err(|e| Error::internal(format!("failed to truncate lock file: {}", e)))?;
                let pid = std::process::id();
                write!(f, "{}", pid).ok();
                f.sync_all().ok();
            }
        }

        Ok(Self {
            file,
            path: lock_file_path,
            mode,
        })
    }

    /// Get the path to the lock file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the mode this lock was acquired in.
    pub fn mode(&self) -> LockMode {
        self.mode
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Do NOT delete db.lock here. On Unix, flock protects the inode, not
        // the path. Deleting while holding the lock lets another process create
        // a new db.lock (different inode) and acquire its own flock, admitting
        // two writers. The lock file is harmless on disk — acquire_lock handles
        // stale files by simply re-flocking the existing inode.
        // The OS flock is released automatically when the File handle is dropped.
    }
}

// ============================================================================
// Read-only mount detection
// ============================================================================
//
// Used by the `Shared` lock-acquire path to decide whether the lockless
// fallback is safe. A truly read-only mount cannot have racing writers
// from any process (the kernel rejects writes regardless of UID, regardless
// of dir mode bits), so skipping `flock` there is sound. A directory whose
// permissions merely restrict the current process (chmod -w, EACCES on a
// writable mount) does NOT qualify: another process / a perm change can
// reintroduce a writer, breaking the read-only contract for any reader who
// took the lockless path.

/// Returns `true` only when `path` lives on a kernel-level read-only mount.
/// Any other failure (statvfs error, unsupported platform) returns `false`
/// — the caller treats that as "lockless fallback NOT safe" and errors out
/// with a permission diagnostic, which is the conservative behaviour.
#[cfg(unix)]
fn is_path_on_readonly_mount(path: &Path) -> bool {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let Ok(c_path) = CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: c_path is a valid NUL-terminated C string; we pass an
    // initialized statvfs struct via &mut. statvfs writes the result on
    // success.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return false;
    }
    // libc::ST_RDONLY and stat.f_flag underlying types vary across Unix
    // platforms (u32 on macOS, u64 on Linux). Cast both to u64 so the AND
    // works uniformly. Clippy may flag one cast as redundant on whichever
    // platform you build on; the cast is required for the *other* one.
    #[allow(clippy::unnecessary_cast)]
    let f_flag = stat.f_flag as u64;
    #[allow(clippy::unnecessary_cast)]
    let rdonly = libc::ST_RDONLY as u64;
    (f_flag & rdonly) != 0
}

#[cfg(not(unix))]
fn is_path_on_readonly_mount(_path: &Path) -> bool {
    // Conservative default: no detection on non-Unix targets. Callers must
    // ship a writable directory or a pre-created `db.lock` with the
    // packaged database.
    false
}

/// EROFS errno — surfaced by `OpenOptions::open` as `io::Error::raw_os_error()`
/// on a read-only filesystem. `Err.kind()` is currently `PermissionDenied` on
/// Linux for this errno, but we match on the raw errno too in case a libc
/// returns a different `ErrorKind` mapping.
#[cfg(unix)]
fn libc_erofs() -> i32 {
    libc::EROFS
}

#[cfg(not(unix))]
fn libc_erofs() -> i32 {
    -1
}

// ============================================================================
// Unix implementation (Linux, macOS, etc.)
// ============================================================================

#[cfg(unix)]
fn acquire_lock(file: &File, mode: LockMode) -> Result<()> {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();

    let lock_flag = match mode {
        LockMode::Exclusive => libc::LOCK_EX,
        LockMode::Shared => libc::LOCK_SH,
    };

    // SAFETY: fd is a valid file descriptor from AsRawFd on an open File.
    // libc::flock is safe to call with valid fd and standard flock flags.
    // LOCK_NB = non-blocking; we want fail-fast on contention.
    let result = unsafe { libc::flock(fd, lock_flag | libc::LOCK_NB) };

    if result != 0 {
        let errno = std::io::Error::last_os_error();
        if errno.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Err(Error::DatabaseLocked);
        }
        return Err(Error::internal(format!(
            "failed to acquire lock: {}",
            errno
        )));
    }

    Ok(())
}

// ============================================================================
// Windows implementation
// ============================================================================

#[cfg(windows)]
fn acquire_lock(file: &File, mode: LockMode) -> Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{ERROR_LOCK_VIOLATION, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let handle = file.as_raw_handle() as HANDLE;

    let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };

    // For shared locks, omit LOCKFILE_EXCLUSIVE_LOCK; LockFileEx then
    // acquires a shared lock by default.
    let mut flags = LOCKFILE_FAIL_IMMEDIATELY;
    if mode == LockMode::Exclusive {
        flags |= LOCKFILE_EXCLUSIVE_LOCK;
    }

    let result = unsafe {
        LockFileEx(
            handle,
            flags,
            0,
            1, // Lock 1 byte
            0,
            &mut overlapped,
        )
    };

    if result == 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
            return Err(Error::DatabaseLocked);
        }
        return Err(Error::internal(format!(
            "failed to acquire lock: {}",
            error
        )));
    }

    Ok(())
}

// ============================================================================
// Fallback for other platforms (no-op, just a warning)
// ============================================================================

#[cfg(not(any(unix, windows)))]
fn acquire_lock(_file: &File, _mode: LockMode) -> Result<()> {
    // On unsupported platforms, we can't guarantee exclusive or shared access.
    // Log a warning but allow operation to continue.
    eprintln!("Warning: File locking not supported on this platform");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_acquire_lock() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Should be able to acquire lock
        let lock = FileLock::acquire(&db_path).unwrap();

        // Lock file should exist
        assert!(db_path.join("db.lock").exists());

        // Lock file should contain our PID
        // Note: On Windows, we can't read the file while it's exclusively locked,
        // so we only verify the contents on Unix systems
        #[cfg(unix)]
        {
            let contents = fs::read_to_string(db_path.join("db.lock")).unwrap();
            assert_eq!(contents, std::process::id().to_string());
        }

        drop(lock);
    }

    #[test]
    fn test_lock_prevents_second_acquisition() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Acquire first lock
        let _lock1 = FileLock::acquire(&db_path).unwrap();

        // Second lock should fail
        let result = FileLock::acquire(&db_path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("locked by another process"));
    }

    #[test]
    fn test_lock_released_on_drop() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Acquire and release lock
        {
            let _lock = FileLock::acquire(&db_path).unwrap();
        }

        // Should be able to acquire again after drop
        let _lock2 = FileLock::acquire(&db_path).unwrap();
    }

    #[test]
    fn test_shared_lock_acquires() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let lock = FileLock::acquire_shared(&db_path).unwrap();
        assert_eq!(lock.mode(), LockMode::Shared);
    }

    #[test]
    fn test_two_shared_locks_coexist() {
        // Multiple readers must be allowed simultaneously.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let _lock1 = FileLock::acquire_shared(&db_path).unwrap();
        let _lock2 = FileLock::acquire_shared(&db_path).unwrap();
        // Both succeed; both released on drop.
    }

    #[test]
    fn test_shared_lock_blocks_exclusive() {
        // A reader prevents a writer from acquiring.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let _shared = FileLock::acquire_shared(&db_path).unwrap();
        let result = FileLock::acquire(&db_path);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::DatabaseLocked)));
    }

    #[test]
    fn test_exclusive_lock_blocks_shared() {
        // A writer prevents readers from acquiring.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let _excl = FileLock::acquire(&db_path).unwrap();
        let result = FileLock::acquire_shared(&db_path);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::DatabaseLocked)));
    }

    #[test]
    fn test_shared_lock_released_on_drop() {
        // After all shared locks drop, an exclusive can acquire.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        {
            let _l1 = FileLock::acquire_shared(&db_path).unwrap();
            let _l2 = FileLock::acquire_shared(&db_path).unwrap();
        }
        let _excl = FileLock::acquire(&db_path).unwrap();
    }
}
