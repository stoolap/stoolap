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
use std::path::{Path, PathBuf};

use crate::core::{Error, Result};

/// Per-database "startup gate" lock filename. The writer holds this
/// EX from before `db.lock` EX through `mark_ready`; readers take
/// it SH to PROVE no writer is currently in startup before trusting
/// `db.shm`. See `await_writer_startup_quiescent` for the protocol.
pub(crate) const STARTUP_LOCK_FILENAME: &str = "db.startup.lock";

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

    /// Acquire a "shared" lock on the database directory.
    ///
    /// **SWMR v1 semantics**: this does NOT take a kernel `flock`.
    /// Multiple shared "locks" coexist trivially because nothing
    /// touches the kernel lock table. A shared lock also does NOT
    /// block (and is not blocked by) a concurrent Exclusive lock —
    /// SWMR readers and a writer can run side-by-side. The only
    /// purpose of this call is to materialize a `FileLock` value of
    /// `LockMode::Shared` so the caller can record the read-only
    /// intent.
    ///
    /// Reader presence for the writer's GC deferral is signaled via
    /// per-PID lease files in `<db>/readers/`, not via this lock —
    /// see [`crate::storage::mvcc::lease::LeaseManager`].
    ///
    /// On a kernel-level read-only mount where even creating
    /// `db.lock` fails, the returned `FileLock` is a fully-lockless
    /// fallback (`file: None`). See the struct doc.
    pub fn acquire_shared(db_path: impl AsRef<Path>) -> Result<Self> {
        Self::acquire_with_mode(db_path, LockMode::Shared)
    }

    /// Internal dispatch for [`Self::acquire`] and [`Self::acquire_shared`].
    /// Callers should use the typed wrappers; the mode parameter changes
    /// behavior in non-obvious ways (Exclusive takes a kernel `flock`,
    /// Shared takes none under SWMR v1) and the typed entry points
    /// document each contract separately.
    pub(crate) fn acquire_with_mode(db_path: impl AsRef<Path>, mode: LockMode) -> Result<Self> {
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

        // Try to acquire the lock (platform-specific).
        //
        // For Exclusive mode we always take a real kernel lock.
        //
        // For Shared mode the behavior is platform-conditional:
        //
        // - Unix (SWMR v1/v2): SKIP `flock`. Reader presence is
        //   signaled via lease files in `<db>/readers/` and
        //   coordination with the writer flows through `db.shm` /
        //   the startup gate (see `await_writer_startup_quiescent`).
        //   Skipping `flock` lets a writer holding `LOCK_EX`
        //   coexist with one or more reader processes — the
        //   writer's destructive cleanup defers while live leases
        //   exist (`MVCCEngine::defer_for_live_readers`).
        //
        // - Non-Unix (Windows): TAKE a real shared `LockFileEx`.
        //   `db.shm` is Unix-only, so cross-process SWMR
        //   coexistence has no coordination surface here. The
        //   shared lock must be held for the engine's full
        //   lifetime — the pre-open handshake's SH guard is
        //   dropped right after the engine opens, so a writer
        //   could otherwise start mid-read and race the reader.
        //   With a kernel SH held by every reader, a would-be
        //   writer's `LOCK_EX` blocks until all readers close.
        //
        // Also skipped entirely when we have no file to lock —
        // the lockless RO-mount fallback (see struct doc).
        //
        // Must run BEFORE any file content modification.
        let should_take_kernel_lock = match mode {
            LockMode::Exclusive => true,
            LockMode::Shared => cfg!(not(unix)),
        };
        if should_take_kernel_lock {
            if let Some(ref f) = file {
                acquire_lock(f, mode)?;
            }
        }

        // Stamp our PID into `db.lock` for diagnostic purposes
        // only. The reader's startup handshake no longer derives
        // identity from this content (it gates on the dedicated
        // `db.startup.lock` instead), so a stale or partially-
        // written PID can't be mistaken for the live writer.
        // Failure to publish IS fatal though: the rest of the
        // engine assumes `db.lock` always carries the current
        // holder's PID for `lsof`-style debugging, and an
        // unreported write failure would silently mask other
        // problems with the lock file (e.g. ENOSPC on the lock
        // partition).
        // Unix gate: `pwrite` via `FileExt::write_all_at` is a
        // Unix extension. On Windows the LockFileEx lock holds
        // the file exclusively, so the diagnostic PID is not
        // observable to other processes anyway and the cost of
        // skipping the stamp is purely diagnostic.
        #[cfg(unix)]
        if mode == LockMode::Exclusive {
            if let Some(ref f) = file {
                use std::os::unix::fs::FileExt as _;
                let pid = std::process::id();
                let pid_str = pid.to_string();
                f.write_all_at(pid_str.as_bytes(), 0).map_err(|e| {
                    Error::internal(format!("failed to write PID to lock file: {}", e))
                })?;
                #[allow(unused_mut)]
                let mut f = f;
                f.set_len(pid_str.len() as u64)
                    .map_err(|e| Error::internal(format!("failed to truncate lock file: {}", e)))?;
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

    /// Acquire the per-database "startup gate" lock EXCLUSIVELY.
    ///
    /// The writer holds this lock from BEFORE it acquires
    /// `db.lock` EX through `mark_ready` on `db.shm`. While held,
    /// any reader's `await_writer_startup_quiescent` blocks at the
    /// startup-gate SH step, guaranteeing the reader cannot
    /// classify a stale READY shm as "trustworthy live writer"
    /// during the new writer's create_writer / WAL recovery /
    /// publish window.
    ///
    /// Returns `Ok(Some(guard))` when the gate is held by this
    /// process. The guard MUST be kept alive across the writer's
    /// startup-publish work and dropped explicitly only after
    /// `mark_ready` has flipped `init_done`.
    ///
    /// Returns `Ok(None)` only when the lock file cannot be created
    /// or opened on a positively-verified read-only mount: there
    /// can be no concurrent reader interpreting our shm anyway,
    /// and the writer path is not reachable on a read-only mount
    /// either, so the no-op is sound. Any other failure is
    /// propagated as `Err` so the caller can refuse to bring the
    /// writer up without a working gate.
    #[cfg(unix)]
    pub fn acquire_startup_exclusive(db_path: &Path) -> Result<Option<StartupLockGuard>> {
        use std::io::ErrorKind;
        use std::os::unix::io::AsRawFd;
        let lock_path = db_path.join(STARTUP_LOCK_FILENAME);
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e)
                if (matches!(e.kind(), ErrorKind::PermissionDenied)
                    || e.raw_os_error() == Some(libc::EROFS))
                    && is_path_on_readonly_mount(db_path) =>
            {
                return Ok(None);
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "failed to open startup lock '{}': {}",
                    lock_path.display(),
                    e
                )));
            }
        };
        let fd = file.as_raw_fd();
        // Non-blocking: a contending writer would fail
        // `acquire(db_path)` with `DatabaseLocked` shortly after
        // anyway. If startup.lock is held but db.lock is free,
        // some other party has misused the file: refuse rather
        // than wait indefinitely.
        // SAFETY: fd is a valid descriptor from an open File above.
        let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            let errno = std::io::Error::last_os_error();
            if errno.raw_os_error() == Some(libc::EWOULDBLOCK) {
                return Err(Error::DatabaseLocked);
            }
            return Err(Error::internal(format!(
                "failed to acquire startup lock '{}': {}",
                lock_path.display(),
                errno
            )));
        }
        Ok(Some(StartupLockGuard { _file: file }))
    }

    /// Stub for non-Unix platforms — no-op, returns None.
    #[cfg(not(unix))]
    pub fn acquire_startup_exclusive(_db_path: &Path) -> Result<Option<StartupLockGuard>> {
        Ok(None)
    }

    /// SWMR v2 attach handshake: classify the writer state at
    /// attach time, returning a [`HandshakeOutcome`] the caller
    /// uses to decide whether to trust `db.shm`.
    ///
    /// - `NoWriter(SharedLockGuard)` — db.lock SH was acquired
    ///   directly. No writer holds LOCK_EX. Caller does uncapped
    ///   WAL recovery and KEEPS the SH alive across it: a new
    ///   writer's LOCK_EX is blocked while we hold SH.
    ///
    /// - `LiveWriter(StartupLockGuard)` — a writer holds db.lock
    ///   EX AND we proved (by holding `db.startup.lock` SH) that
    ///   the writer is past `mark_ready`. Caller MUST keep the
    ///   startup guard alive across opening `db.shm`, sampling
    ///   its header, AND a final liveness recheck via
    ///   [`Self::recheck_writer_still_holds_lock`]. If the
    ///   recheck shows the writer disappeared during the sample,
    ///   the sample MUST be discarded and treated as no-writer.
    ///
    /// - `ReadOnlyMount` — positively-verified RO mount
    ///   (the `db.lock` create returned EROFS / EACCES with the
    ///   mount flag set). No writer can ever exist on this
    ///   filesystem. Caller may treat any on-disk shm as
    ///   trustworthy without further proof, since there is no
    ///   crash-and-restart race to defend against.
    ///
    /// `Err(...)` is returned for transient or unexpected
    /// failures (EINTR loop exhausted, ENFILE, restrictive perms
    /// on a writable mount, ...). Caller MUST treat this as
    /// "handshake unavailable" and refuse to proceed. Silently
    /// falling back to uncapped replay would race a writer.
    #[cfg(unix)]
    pub fn await_writer_startup_quiescent(db_path: &Path) -> Result<HandshakeOutcome> {
        use std::io::ErrorKind;
        use std::os::unix::io::AsRawFd;
        let lock_path = db_path.join("db.lock");
        // Try to open db.lock for read+write+create.
        //
        // Error classification:
        //   - EROFS / PermissionDenied AND `is_path_on_readonly_mount`
        //     returns true → POSITIVELY-verified read-only mount;
        //     no writer can ever start here, return `Ok(None)`.
        //   - EROFS / PermissionDenied on a WRITABLE mount (e.g.
        //     chmod-restricted directory or db.lock file): NOT
        //     proof of read-only. Another process / user can still
        //     start a writer. Try a read-only open as a fallback;
        //     if that succeeds we can still take LOCK_SH against
        //     the existing inode. If it fails, propagate Err
        //     rather than silently downgrade to "no handshake".
        //   - Any other error: transient, propagate Err.
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e)
                if (matches!(e.kind(), ErrorKind::PermissionDenied)
                    || e.raw_os_error() == Some(libc::EROFS))
                    && is_path_on_readonly_mount(db_path) =>
            {
                // Verified RO mount: no writer can start.
                return Ok(HandshakeOutcome::ReadOnlyMount);
            }
            Err(e)
                if matches!(e.kind(), ErrorKind::PermissionDenied)
                    || e.raw_os_error() == Some(libc::EROFS) =>
            {
                // Restrictive perms on a WRITABLE mount: a writer
                // could still start with different credentials.
                // Try a read-only open against the existing inode.
                match OpenOptions::new().read(true).open(&lock_path) {
                    Ok(f) => f,
                    Err(open_err) => {
                        return Err(Error::internal(format!(
                            "failed to open db.lock for SWMR attach handshake at '{}': \
                             create failed ({}); read-only fallback also failed ({}); \
                             the directory looks writable but another process could \
                             still start a writer — refusing to skip the handshake",
                            lock_path.display(),
                            e,
                            open_err
                        )));
                    }
                }
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "failed to open db.lock for SWMR attach handshake at '{}': {} \
                     (handshake required to prevent uncapped WAL replay racing a \
                     writer's startup)",
                    lock_path.display(),
                    e
                )));
            }
        };
        let fd = file.as_raw_fd();
        // Step 1: try `db.lock` LOCK_SH non-blocking. If we get
        // it, no process holds LOCK_EX → no live writer → caller
        // does uncapped WAL recovery while keeping our SH guard
        // alive (it blocks any new writer's LOCK_EX from
        // sneaking in mid-recovery).
        const MAX_EINTR_RETRIES: u32 = 32;
        let mut eintr_attempts = 0;
        let initial_errno = loop {
            // SAFETY: fd from AsRawFd on an open File above; standard flock flags.
            let result = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) };
            if result == 0 {
                return Ok(HandshakeOutcome::NoWriter(SharedLockGuard { _file: file }));
            }
            let errno = std::io::Error::last_os_error();
            match errno.raw_os_error() {
                Some(code) if code == libc::EINTR && eintr_attempts < MAX_EINTR_RETRIES => {
                    eintr_attempts += 1;
                    continue;
                }
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => break errno,
                _ => {
                    return Err(Error::internal(format!(
                        "failed to acquire LOCK_SH on db.lock at '{}': {} \
                         (handshake required to prevent uncapped WAL replay)",
                        lock_path.display(),
                        errno
                    )));
                }
            }
        };
        let _ = initial_errno;
        // Step 2: a writer holds LOCK_EX. Take `db.startup.lock`
        // SH, polling within a budget. The writer's
        // `acquire_startup_exclusive` holds startup.lock EX from
        // BEFORE its `db.lock` EX through `mark_ready`, so:
        //   - As long as we cannot get startup.lock SH, the
        //     writer is in its startup-publish window and any
        //     READY shm we observe might still belong to the
        //     prior incarnation (the new writer's create_writer
        //     has not yet wiped init_done, or the wipe has not
        //     yet been observed by readers).
        //   - Once startup.lock SH succeeds, no writer is in the
        //     startup window. Either there is no writer at all
        //     (we'll prove that with the db.lock retry) or the
        //     writer that holds db.lock EX has already passed
        //     `mark_ready`, so its db.shm is the live frontier.
        //
        // The poll budget bounds how long we wait for a slow
        // writer's recovery. 30s is generous: real WAL replay
        // even on a multi-GB log finishes well inside that.
        // Beyond the budget we refuse the attach rather than
        // fall back to uncapped replay against an unknown
        // writer state.
        let startup_path = db_path.join(STARTUP_LOCK_FILENAME);
        let startup_file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&startup_path)
        {
            Ok(f) => f,
            Err(e)
                if (matches!(e.kind(), ErrorKind::PermissionDenied)
                    || e.raw_os_error() == Some(libc::EROFS))
                    && is_path_on_readonly_mount(db_path) =>
            {
                // Verified RO mount: no writer can be running.
                // Treat as "shm trustworthy" (the caller would
                // have failed at db.lock-create step otherwise),
                // matching the existing RO-mount short circuit.
                return Ok(HandshakeOutcome::ReadOnlyMount);
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "failed to open startup lock '{}': {} \
                     (handshake required to prevent uncapped WAL replay)",
                    startup_path.display(),
                    e
                )));
            }
        };
        let startup_fd = startup_file.as_raw_fd();
        const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(5);
        const STARTUP_POLL_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);
        let start = std::time::Instant::now();
        let mut startup_eintr = 0;
        let startup_guard = loop {
            // SAFETY: startup_fd from AsRawFd on an open File above.
            let result = unsafe { libc::flock(startup_fd, libc::LOCK_SH | libc::LOCK_NB) };
            if result == 0 {
                break SharedLockGuard {
                    _file: startup_file,
                };
            }
            let errno = std::io::Error::last_os_error();
            match errno.raw_os_error() {
                Some(code) if code == libc::EINTR && startup_eintr < MAX_EINTR_RETRIES => {
                    startup_eintr += 1;
                    continue;
                }
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => {
                    if start.elapsed() >= STARTUP_POLL_BUDGET {
                        return Err(Error::internal(format!(
                            "writer at '{}' did not release startup gate \
                             '{}' after {}s (writer is stuck in recovery, \
                             or the gate has been leaked); refusing to \
                             fall back to uncapped WAL replay",
                            db_path.display(),
                            startup_path.display(),
                            STARTUP_POLL_BUDGET.as_secs()
                        )));
                    }
                    std::thread::sleep(POLL_INTERVAL);
                    continue;
                }
                _ => {
                    return Err(Error::internal(format!(
                        "failed to acquire LOCK_SH on startup lock '{}': {} \
                         (handshake required to prevent uncapped WAL replay)",
                        startup_path.display(),
                        errno
                    )));
                }
            }
        };
        // Step 3: with startup.lock SH held, retry db.lock SH.
        //   - Success → the writer disappeared between step 1
        //     and now (rare but legal). We hold db.lock SH; a
        //     new writer's LOCK_EX is blocked. Caller does
        //     uncapped recovery and any pre-existing READY shm
        //     is treated as stale (see caller's
        //     `shm_is_stale_leftover`).
        //   - EWOULDBLOCK → a writer still holds db.lock EX
        //     AND we hold startup.lock SH (so no writer is in
        //     startup right now). Therefore the live writer is
        //     past `mark_ready`; return LiveWriter and HAND THE
        //     STARTUP GUARD TO THE CALLER so it stays held
        //     across opening + sampling db.shm. Caller calls
        //     `recheck_writer_still_holds_lock` after sampling
        //     to confirm the writer didn't exit mid-sample.
        let mut retry_eintr = 0;
        loop {
            // SAFETY: fd from AsRawFd on an open File at the top of this fn.
            let result = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) };
            if result == 0 {
                drop(startup_guard);
                return Ok(HandshakeOutcome::NoWriter(SharedLockGuard { _file: file }));
            }
            let errno = std::io::Error::last_os_error();
            match errno.raw_os_error() {
                Some(code) if code == libc::EINTR && retry_eintr < MAX_EINTR_RETRIES => {
                    retry_eintr += 1;
                    continue;
                }
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => {
                    return Ok(HandshakeOutcome::LiveWriter(StartupLockGuard {
                        _file: startup_guard._file,
                    }));
                }
                _ => {
                    return Err(Error::internal(format!(
                        "failed to retry LOCK_SH on db.lock at '{}': {} \
                         (handshake required to prevent uncapped WAL replay)",
                        lock_path.display(),
                        errno
                    )));
                }
            }
        }
    }

    /// Liveness recheck used by readers AFTER they've taken the
    /// snapshot of `db.shm` while holding the [`StartupLockGuard`]
    /// returned by `await_writer_startup_quiescent` in the
    /// `LiveWriter` variant.
    ///
    /// Returns `Ok(Some(SharedLockGuard))` when the writer
    /// disappeared during the sample window: db.lock SH succeeded,
    /// so no process holds LOCK_EX anymore. The sample MUST be
    /// discarded; caller treats this as the no-writer path,
    /// keeping the returned SH guard alive across uncapped WAL
    /// recovery (it blocks any new writer's LOCK_EX).
    ///
    /// Returns `Ok(None)` when the writer is still alive:
    /// db.lock SH still EWOULDBLOCKs. The earlier shm sample
    /// reflects a writer that was past `mark_ready` for the
    /// entire sample window, so its `visible_commit_lsn` is
    /// authoritative.
    ///
    /// `Err(...)` is returned for transient or unexpected flock
    /// failures. Caller MUST treat `Err` as "liveness unknown"
    /// and refuse to proceed — same contract as the original
    /// handshake.
    #[cfg(unix)]
    pub fn recheck_writer_still_holds_lock(db_path: &Path) -> Result<Option<SharedLockGuard>> {
        use std::io::ErrorKind;
        use std::os::unix::io::AsRawFd;
        let lock_path = db_path.join("db.lock");
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e)
                if (matches!(e.kind(), ErrorKind::PermissionDenied)
                    || e.raw_os_error() == Some(libc::EROFS))
                    && is_path_on_readonly_mount(db_path) =>
            {
                // Verified RO mount: writer can't exist; recheck
                // is moot. Treat as "still alive" (i.e., trust
                // shm) — the existing RO-mount short circuit.
                return Ok(None);
            }
            Err(e)
                if matches!(e.kind(), ErrorKind::PermissionDenied)
                    || e.raw_os_error() == Some(libc::EROFS) =>
            {
                match OpenOptions::new().read(true).open(&lock_path) {
                    Ok(f) => f,
                    Err(open_err) => {
                        return Err(Error::internal(format!(
                            "liveness recheck open of db.lock at '{}' failed: \
                             create failed ({}); read-only fallback also \
                             failed ({})",
                            lock_path.display(),
                            e,
                            open_err
                        )));
                    }
                }
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "liveness recheck open of db.lock at '{}' failed: {}",
                    lock_path.display(),
                    e
                )));
            }
        };
        let fd = file.as_raw_fd();
        const MAX_EINTR_RETRIES: u32 = 32;
        let mut eintr_attempts = 0;
        loop {
            // SAFETY: fd from AsRawFd on an open File above; standard flock flags.
            let result = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) };
            if result == 0 {
                return Ok(Some(SharedLockGuard { _file: file }));
            }
            let errno = std::io::Error::last_os_error();
            match errno.raw_os_error() {
                Some(code) if code == libc::EINTR && eintr_attempts < MAX_EINTR_RETRIES => {
                    eintr_attempts += 1;
                    continue;
                }
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN => {
                    return Ok(None);
                }
                _ => {
                    return Err(Error::internal(format!(
                        "liveness recheck flock on db.lock at '{}' failed: {}",
                        lock_path.display(),
                        errno
                    )));
                }
            }
        }
    }

    /// Non-Unix (Windows) handshake.
    ///
    /// `db.shm` is currently Unix-only — Windows has no
    /// `ShmHandle::create_writer` or `open_reader`, so the SWMR
    /// startup-gate / shm protocol cannot run. Without shm, the
    /// only safe coordination is a real shared `LockFileEx` on
    /// `db.lock`:
    ///
    ///   - Successful SH lock → no writer holds EX → safe to
    ///     do uncapped WAL recovery while we keep the guard.
    ///   - `LOCK_VIOLATION` → a writer holds EX, but we have no
    ///     shm to negotiate visibility through. Refuse the
    ///     attach rather than silently fall back to uncapped
    ///     replay against a live writer.
    ///   - Verified RO mount short-circuit (EROFS / EACCES with
    ///     mount flag set) → no writer can ever exist; trust
    ///     whatever's on disk.
    #[cfg(not(unix))]
    pub fn await_writer_startup_quiescent(db_path: &Path) -> Result<HandshakeOutcome> {
        use std::io::ErrorKind;
        let lock_path = db_path.join("db.lock");
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e)
                if matches!(e.kind(), ErrorKind::PermissionDenied)
                    && is_path_on_readonly_mount(db_path) =>
            {
                return Ok(HandshakeOutcome::ReadOnlyMount);
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "failed to open db.lock for SWMR attach handshake at '{}': {} \
                     (handshake required to prevent uncapped WAL replay racing a \
                     writer's startup)",
                    lock_path.display(),
                    e
                )));
            }
        };
        match acquire_lock(&file, LockMode::Shared) {
            Ok(()) => Ok(HandshakeOutcome::NoWriter(SharedLockGuard { _file: file })),
            Err(Error::DatabaseLocked) => Err(Error::internal(format!(
                "SWMR attach refused at '{}': a writer holds the exclusive \
                 lock on db.lock and this platform has no db.shm support, \
                 so live reader/writer coexistence is unavailable. Close \
                 the writer or retry the open.",
                db_path.display()
            ))),
            Err(e) => Err(e),
        }
    }

    /// Non-Unix recheck: equivalent to a fresh handshake but
    /// returns `Option<SharedLockGuard>` (`None` = writer still
    /// alive). Reused by callers that took the `LiveWriter` path
    /// — but on non-Unix we never return `LiveWriter` (no shm),
    /// so this should normally not be called. Provided for API
    /// symmetry; behaves the same way.
    #[cfg(not(unix))]
    pub fn recheck_writer_still_holds_lock(db_path: &Path) -> Result<Option<SharedLockGuard>> {
        use std::io::ErrorKind;
        let lock_path = db_path.join("db.lock");
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e)
                if matches!(e.kind(), ErrorKind::PermissionDenied)
                    && is_path_on_readonly_mount(db_path) =>
            {
                return Ok(None);
            }
            Err(e) => {
                return Err(Error::internal(format!(
                    "liveness recheck open of db.lock at '{}' failed: {}",
                    lock_path.display(),
                    e
                )));
            }
        };
        match acquire_lock(&file, LockMode::Shared) {
            Ok(()) => Ok(Some(SharedLockGuard { _file: file })),
            Err(Error::DatabaseLocked) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// Outcome of [`FileLock::await_writer_startup_quiescent`].
///
/// Encodes whether the caller may trust an on-disk `db.shm`
/// snapshot and what guard it must hold across follow-up work.
#[derive(Debug)]
pub enum HandshakeOutcome {
    /// No process holds `db.lock` LOCK_EX. Caller is free to do
    /// uncapped WAL recovery, but MUST keep the returned guard
    /// alive across that recovery: a new writer's LOCK_EX is
    /// blocked while we hold SH, so no writer can sneak in and
    /// race the recovery.
    NoWriter(SharedLockGuard),
    /// A writer holds `db.lock` LOCK_EX and is past
    /// `mark_ready` AT THE MOMENT of return. Caller MUST keep
    /// the embedded [`StartupLockGuard`] alive across opening
    /// `db.shm`, sampling its header, and then a final liveness
    /// recheck via [`FileLock::recheck_writer_still_holds_lock`].
    /// If that recheck shows the writer disappeared mid-sample,
    /// the sample MUST be discarded.
    LiveWriter(StartupLockGuard),
    /// Positively-verified read-only mount: no writer can ever
    /// run on this filesystem. Caller may trust any on-disk
    /// `db.shm` snapshot without further proof.
    ReadOnlyMount,
}

/// RAII guard for a shared `flock` on `db.lock` taken by
/// `FileLock::await_writer_startup_quiescent`. Drop releases the
/// kernel lock automatically by closing the underlying File.
#[derive(Debug)]
pub struct SharedLockGuard {
    _file: File,
}

/// RAII guard for the writer-side EX lock on `db.startup.lock`.
/// Drop releases the kernel lock by closing the underlying File.
/// The writer must keep this alive across `create_writer` /
/// recovery / publish / `mark_ready`, then drop it explicitly so
/// readers can complete their startup-quiescence handshake.
#[derive(Debug)]
pub struct StartupLockGuard {
    _file: File,
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
/// Public wrapper for `is_path_on_readonly_mount` so callers
/// outside this module (e.g. `Database::pre_acquire`) can
/// distinguish a verified read-only mount from a writable mount
/// with restrictive permissions when classifying registration
/// failures.
pub fn is_path_on_readonly_mount_pub(path: &Path) -> bool {
    is_path_on_readonly_mount(path)
}

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

    #[cfg(unix)]
    #[test]
    fn test_shared_does_not_block_exclusive_under_swmr() {
        // v1/v2 SWMR contract (Unix only): Shared takes no kernel lock — reader
        // presence is tracked via lease files, not flock, and coordination
        // flows through `db.shm`. So a reader holding a Shared FileLock does
        // NOT block a subsequent Exclusive acquire from another would-be
        // writer. This is what makes live cross-process reader-with-writer
        // possible. On non-Unix (Windows) Shared takes a real LockFileEx SH
        // — see `acquire_with_mode` — so this contract does not apply there.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let _shared = FileLock::acquire_shared(&db_path).unwrap();
        let excl = FileLock::acquire(&db_path);
        assert!(
            excl.is_ok(),
            "Shared must not block Exclusive under SWMR (got {:?})",
            excl.err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_exclusive_does_not_block_shared_under_swmr() {
        // v1/v2 SWMR contract (Unix only): a writer holding Exclusive does
        // NOT block a reader from opening Shared, because Shared no longer
        // takes a kernel lock. Reader sees consistent on-disk state via the
        // tmp+rename atomicity of manifests + the lease-deferred unlink of
        // volumes + `db.shm` cross-process visibility. On non-Unix Shared
        // takes a real LockFileEx SH — see `acquire_with_mode` — so this
        // contract does not apply there.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let _excl = FileLock::acquire(&db_path).unwrap();
        let shared = FileLock::acquire_shared(&db_path);
        assert!(
            shared.is_ok(),
            "Exclusive must not block Shared under SWMR (got {:?})",
            shared.err()
        );
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
