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
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::core::{Error, Result};

/// Represents an exclusive lock on a database directory.
///
/// The lock is automatically released when this struct is dropped.
#[derive(Debug)]
pub struct FileLock {
    /// The lock file handle (kept open to maintain the lock)
    #[allow(dead_code)]
    file: File,
    /// Path to the lock file
    path: PathBuf,
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
        let db_path = db_path.as_ref();

        // Ensure the directory exists
        fs::create_dir_all(db_path)
            .map_err(|e| Error::internal(format!("failed to create database directory: {}", e)))?;

        // Lock file path
        let lock_file_path = db_path.join("db.lock");

        // Open the lock file (create if it doesn't exist)
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&lock_file_path)
            .map_err(|e| Error::internal(format!("failed to open lock file: {}", e)))?;

        // Try to acquire an exclusive lock (platform-specific)
        acquire_lock(&file)?;

        // Write the current process ID to the lock file for debugging
        let pid = std::process::id();
        write!(file, "{}", pid).ok();
        file.sync_all().ok();

        Ok(Self {
            file,
            path: lock_file_path,
        })
    }

    /// Get the path to the lock file
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Lock is automatically released when the file is closed
        // We don't remove the lock file as it will be reused on next open
        #[cfg(windows)]
        {
            // Windows file handles may take a moment to be fully released
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

// ============================================================================
// Unix implementation (Linux, macOS, etc.)
// ============================================================================

#[cfg(unix)]
fn acquire_lock(file: &File) -> Result<()> {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();

    // LOCK_EX = exclusive lock, LOCK_NB = non-blocking
    let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };

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
fn acquire_lock(file: &File) -> Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{ERROR_LOCK_VIOLATION, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    let handle = file.as_raw_handle() as HANDLE;

    let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };

    let result = unsafe {
        LockFileEx(
            handle,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
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
fn acquire_lock(_file: &File) -> Result<()> {
    // On unsupported platforms, we can't guarantee exclusive access
    // Log a warning but allow operation to continue
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
}
