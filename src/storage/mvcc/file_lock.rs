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

//! OS-level file locking on `db.lock` (flock on Unix, LockFileEx on Windows).

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use crate::core::{Error, Result};

/// Startup-gate lock file. Writer holds EX from before `db.lock` EX through
/// `mark_ready`; readers take SH to prove the writer is past startup before
/// trusting `db.shm`. See `await_writer_startup_quiescent`.
pub(crate) const STARTUP_LOCK_FILENAME: &str = "db.startup.lock";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    Exclusive,
    Shared,
}

/// RAII lock on a database directory. Released on drop.
///
/// `file` is `None` only on the lockless shared fallback (Shared mode on a
/// read-only mount where `db.lock` is missing and cannot be created).
#[derive(Debug)]
pub struct FileLock {
    #[allow(dead_code)]
    file: Option<File>,
    path: PathBuf,
    mode: LockMode,
}

impl FileLock {
    /// Take an exclusive lock on the database directory. Returns
    /// `Error::DatabaseLocked` if another process holds it.
    pub fn acquire(db_path: impl AsRef<Path>) -> Result<Self> {
        Self::acquire_with_mode(db_path, LockMode::Exclusive)
    }

    /// Take a shared lock. On Unix (SWMR) this takes no kernel flock —
    /// reader presence is signaled via lease files in `<db>/readers/`. On
    /// non-Unix it takes a real LockFileEx SH. Falls back to lockless
    /// (`file: None`) on a verified read-only mount where `db.lock` is
    /// missing and cannot be created.
    pub fn acquire_shared(db_path: impl AsRef<Path>) -> Result<Self> {
        Self::acquire_with_mode(db_path, LockMode::Shared)
    }

    pub(crate) fn acquire_with_mode(db_path: impl AsRef<Path>, mode: LockMode) -> Result<Self> {
        let db_path = db_path.as_ref();

        // Exclusive requires a writable dir; Shared tolerates read-only mounts.
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

        let lock_file_path = db_path.join("db.lock");

        // Shared prefers a read-only open so chmod-restricted dirs work; falls
        // back to create+rw, then to the lockless path only on verified RO
        // mounts (EACCES on a writable mount could later allow a writer).
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

        // Exclusive always takes a kernel lock. Shared takes one only on
        // non-Unix (Windows lacks db.shm coordination); on Unix the lockless
        // RO-mount fallback (file is None) also short-circuits.
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

        // Stamp our PID for `lsof`-style diagnostics. Identity is not
        // derived from this content (the startup handshake uses
        // db.startup.lock instead), but ENOSPC etc. should still surface.
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

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn mode(&self) -> LockMode {
        self.mode
    }

    /// Take EX on `db.startup.lock`. Writer holds from before `db.lock` EX
    /// through `mark_ready`; readers gate on its SH-availability to avoid
    /// trusting a stale READY shm during writer startup. Guard must outlive
    /// `mark_ready`. Returns `Ok(None)` only on verified RO mounts.
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
        // SAFETY: fd is a valid descriptor from an open File above.
        // Non-blocking: contending writer fails with DatabaseLocked.
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

    #[cfg(not(unix))]
    pub fn acquire_startup_exclusive(_db_path: &Path) -> Result<Option<StartupLockGuard>> {
        Ok(None)
    }

    /// SWMR attach handshake. Returns one of three outcomes:
    ///
    ///   db.lock SH? --yes--> NoWriter(sh_guard)         [keep across recovery]
    ///        |
    ///        no (EWOULDBLOCK)
    ///        v
    ///   take startup.lock SH (poll up to 30s)
    ///        |
    ///        v
    ///   retry db.lock SH? --yes--> NoWriter             [writer exited]
    ///        |
    ///        no --> LiveWriter(startup_guard)           [past mark_ready;
    ///                                                    caller keeps guard
    ///                                                    across shm sample +
    ///                                                    recheck]
    ///
    /// `ReadOnlyMount` short-circuits when db.lock create fails with
    /// EROFS/EACCES on a verified RO mount. `Err` means handshake
    /// unavailable; caller MUST refuse to proceed.
    #[cfg(unix)]
    pub fn await_writer_startup_quiescent(db_path: &Path) -> Result<HandshakeOutcome> {
        use std::io::ErrorKind;
        use std::os::unix::io::AsRawFd;
        let lock_path = db_path.join("db.lock");
        // EROFS/EACCES on verified RO mount → ReadOnlyMount.
        // EROFS/EACCES on writable mount → fall back to read-only open.
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
                return Ok(HandshakeOutcome::ReadOnlyMount);
            }
            Err(e)
                if matches!(e.kind(), ErrorKind::PermissionDenied)
                    || e.raw_os_error() == Some(libc::EROFS) =>
            {
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
        // Step 1: try db.lock LOCK_SH non-blocking. Success → NoWriter.
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
        // Step 2: writer holds db.lock EX. Poll startup.lock SH (30s budget).
        // SH success proves no writer is in its startup window.
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
        // Step 3: retry db.lock SH while holding startup.lock SH.
        // Success → writer exited; EWOULDBLOCK → live writer past mark_ready.
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

    /// Post-sample liveness recheck for the `LiveWriter` path.
    /// `Ok(Some(sh_guard))` → writer disappeared mid-sample; discard sample
    /// and treat as no-writer (keep guard across WAL recovery).
    /// `Ok(None)` → writer still alive; sample is authoritative.
    /// `Err` → liveness unknown; caller MUST refuse to proceed.
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
                // RO mount: no writer can exist; treat sample as authoritative.
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

    /// Non-Unix handshake. db.shm is Unix-only, so coordination relies on
    /// LockFileEx SH: success → NoWriter; LOCK_VIOLATION → refuse attach.
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

    /// Non-Unix recheck. Provided for API symmetry; non-Unix never returns
    /// `LiveWriter`, so this should normally not be called.
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
#[derive(Debug)]
pub enum HandshakeOutcome {
    /// No writer holds db.lock EX. Keep guard across WAL recovery to
    /// block a new writer's EX from racing the recovery.
    NoWriter(SharedLockGuard),
    /// Writer holds db.lock EX and is past `mark_ready`. Keep guard
    /// across shm sample + final `recheck_writer_still_holds_lock`.
    LiveWriter(StartupLockGuard),
    /// Verified read-only mount; no writer can exist. Trust on-disk shm.
    ReadOnlyMount,
}

/// RAII shared-flock guard on `db.lock`. Drop releases the lock.
#[derive(Debug)]
pub struct SharedLockGuard {
    _file: File,
}

/// RAII writer-side EX guard on `db.startup.lock`. Held across
/// create_writer / recovery / publish / mark_ready, then dropped.
#[derive(Debug)]
pub struct StartupLockGuard {
    _file: File,
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Do NOT delete db.lock: flock protects the inode, not the path.
        // Deleting while locked would let another process create a new inode
        // and lock it, admitting two writers. OS flock releases on file drop.
    }
}

// Read-only mount detection. Only a kernel-level RO mount qualifies for the
// lockless shared fallback; chmod-restricted dirs on a writable mount don't
// (another process could still acquire a writer lock).

/// Public wrapper so callers (e.g. `Database::pre_acquire`) can distinguish
/// a verified RO mount from a writable mount with restrictive permissions.
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
    // f_flag width varies across Unix (u32 on macOS, u64 on Linux); cast both
    // to u64 for a uniform AND. One cast is redundant per-platform.
    #[allow(clippy::unnecessary_cast)]
    let f_flag = stat.f_flag as u64;
    #[allow(clippy::unnecessary_cast)]
    let rdonly = libc::ST_RDONLY as u64;
    (f_flag & rdonly) != 0
}

#[cfg(not(unix))]
fn is_path_on_readonly_mount(_path: &Path) -> bool {
    false
}

/// EROFS errno. We match on the raw errno because Linux maps it to
/// `ErrorKind::PermissionDenied` rather than a dedicated variant.
#[cfg(unix)]
fn libc_erofs() -> i32 {
    libc::EROFS
}

#[cfg(not(unix))]
fn libc_erofs() -> i32 {
    -1
}

// Unix flock implementation.

#[cfg(unix)]
fn acquire_lock(file: &File, mode: LockMode) -> Result<()> {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();

    let lock_flag = match mode {
        LockMode::Exclusive => libc::LOCK_EX,
        LockMode::Shared => libc::LOCK_SH,
    };

    // SAFETY: fd from AsRawFd on an open File; standard flock flags.
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

// Windows LockFileEx implementation.

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

    // Omit LOCKFILE_EXCLUSIVE_LOCK for SH; LockFileEx defaults to shared.
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

#[cfg(not(any(unix, windows)))]
fn acquire_lock(_file: &File, _mode: LockMode) -> Result<()> {
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

        let lock = FileLock::acquire(&db_path).unwrap();

        assert!(db_path.join("db.lock").exists());

        // PID is only readable on Unix; Windows holds the file exclusively.
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

        let _lock1 = FileLock::acquire(&db_path).unwrap();

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

        {
            let _lock = FileLock::acquire(&db_path).unwrap();
        }

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
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        let _lock1 = FileLock::acquire_shared(&db_path).unwrap();
        let _lock2 = FileLock::acquire_shared(&db_path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn test_shared_does_not_block_exclusive_under_swmr() {
        // SWMR contract (Unix): Shared takes no kernel flock, so a reader
        // does not block a writer's Exclusive acquire.
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
        // SWMR contract (Unix): Exclusive does not block Shared.
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
