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

//! Cross-process reader presence + WAL pin via `<db>/readers/<pid>.lease`.
//!
//! Each lease is an 8-byte LE `pinned_lsn` payload. Writer reads two signals:
//! presence (mtime within `max_age`, defers destructive ops) and WAL pin
//! (payload constrains `truncate_wal` floor; payload `0` = presence only).
//! Stale leases (mtime older than `max_age`) are reaped. No flock; PID reuse
//! is benign because the new process re-asserts presence.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use crate::core::{Error, Result};

/// Subdirectory under the database path that holds reader lease files.
pub const READERS_DIR: &str = "readers";

/// Per-path refcount: same-process double-open shares one `<pid>.lease`,
/// and only the last drop unlinks. Without this, dropping one handle would
/// remove the lease while a sibling still needs it.
fn lease_refcount() -> &'static Mutex<HashMap<PathBuf, usize>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Per-(lease_path, handle_id) WAL pin contributions. On-disk pin must be
/// MIN across handles so a higher floor never overwrites a lagging sibling's
/// lower floor.
type PinContributions = HashMap<PathBuf, std::collections::BTreeMap<u64, u64>>;
fn lease_pin_contributions() -> &'static Mutex<PinContributions> {
    static REGISTRY: OnceLock<Mutex<PinContributions>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Monotonic process-wide handle id allocator. Each contributing handle
/// uses its id as the key into `lease_pin_contributions`.
pub fn next_handle_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// A handle that asserts this process is reading the database. Drop unlinks
/// the lease file (best-effort). The writer's cleanup paths skip destructive
/// operations while any live lease exists.
#[derive(Debug)]
pub struct LeaseManager {
    /// The lease file path: `<db>/readers/<pid>.lease`.
    lease_path: PathBuf,
}

impl LeaseManager {
    /// Create a lease for this process. Stale lease for the same PID is
    /// overwritten (same as if the prior instance had cleaned up).
    pub fn register(db_path: &Path) -> Result<Self> {
        let dir = db_path.join(READERS_DIR);
        fs::create_dir_all(&dir).map_err(|e| {
            Error::internal(format!(
                "failed to create lease dir '{}': {}",
                dir.display(),
                e
            ))
        })?;

        // Canonicalize so aliased opens (symlinks, `..`) share one registry key.
        // Falls back to the joined path on rare canonicalize failure.
        let dir = fs::canonicalize(&dir).unwrap_or(dir);

        let pid = std::process::id();
        let lease_path = dir.join(format!("{}.lease", pid));

        // Truncate only when no sibling already holds this lease; otherwise
        // we'd zero the active pinned_lsn payload.
        let should_truncate = {
            let mut reg = lease_refcount().lock().unwrap_or_else(|p| p.into_inner());
            let existing = reg.get(&lease_path).copied().unwrap_or(0);
            *reg.entry(lease_path.clone()).or_insert(0) += 1;
            existing == 0
        };

        let mut opts = OpenOptions::new();
        opts.create(true).write(true);
        if should_truncate {
            opts.truncate(true);
        }
        let open_result = opts.open(&lease_path).and_then(|mut f| {
            if should_truncate {
                use std::io::Write;
                f.write_all(&0u64.to_le_bytes())?;
                f.sync_all()?;
            }
            Ok(())
        });
        if let Err(e) = open_result {
            // Roll back refcount on failure so a future register() can retry.
            let mut reg = lease_refcount().lock().unwrap_or_else(|p| p.into_inner());
            if let Some(count) = reg.get_mut(&lease_path) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    reg.remove(&lease_path);
                }
            }
            return Err(Error::internal(format!(
                "failed to create lease '{}': {}",
                lease_path.display(),
                e
            )));
        }

        // First in-process registration for this PID's lease: clear any
        // <pid>.<*>.epoch leftovers from a prior crashed incarnation
        // (PID reuse). Otherwise the writer's per-PID MIN would include
        // the stale low epoch and pin reaping for our lifetime.
        if should_truncate {
            let pid_prefix = format!("{}.", pid);
            if let Ok(it) = fs::read_dir(&dir) {
                for entry in it.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|s| s.to_str()) != Some("epoch") {
                        continue;
                    }
                    if p.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.starts_with(&pid_prefix))
                        .unwrap_or(false)
                    {
                        let _ = fs::remove_file(&p);
                    }
                }
            }
        }

        // Refresh mtime so live_leases sees us immediately. On failure roll
        // back the refcount: returning Err here without rollback would
        // strand the entry at >0, making future same-process register()s
        // skip truncate and inherit stale content forever.
        if let Err(e) = touch_path(&lease_path) {
            let mut reg = lease_refcount().lock().unwrap_or_else(|p| p.into_inner());
            if let Some(count) = reg.get_mut(&lease_path) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    reg.remove(&lease_path);
                }
            }
            return Err(Error::internal(format!(
                "failed to touch lease '{}' after create: {}",
                lease_path.display(),
                e
            )));
        }

        Ok(Self { lease_path })
    }

    /// Bump the lease's mtime to the current time. Called from query /
    /// execute / refresh entry points.
    ///
    /// Self-heals like `EpochFile::write`: if the writer reaped this PID's
    /// lease while the process slept past `max_age`, rewrite the current
    /// MIN from the in-process registry via `set_pinned_lsn` (which uses
    /// `.create(true)`). A plain mtime touch on a reaped file returns
    /// `NotFound`, the lease stays absent, and the writer is free to
    /// truncate WAL ranges this reader still references.
    ///
    /// Fast path: try the cheap `touch_path` (`open + set_modified`) first;
    /// only fall through to the heavier `set_pinned_lsn` self-heal when
    /// the file is missing. Steady-state heartbeats (the common case) pay
    /// one syscall instead of three.
    pub fn touch(&self) -> Result<()> {
        match touch_path(&self.lease_path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == ErrorKind::NotFound => {
                let reg = lease_pin_contributions()
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                let pin = reg
                    .get(&self.lease_path)
                    .and_then(|m| m.values().copied().min())
                    .unwrap_or(0);
                let result = self.set_pinned_lsn(pin);
                drop(reg);
                result
            }
            Err(e) => Err(Error::internal(format!(
                "failed to touch lease '{}': {}",
                self.lease_path.display(),
                e
            ))),
        }
    }

    /// Overwrite the 8-byte `pinned_lsn` payload in place and bump mtime.
    /// Pass `db.shm.visible_commit_lsn` to pin everything published so far;
    /// pass `0` to release the pin while keeping presence.
    pub fn set_pinned_lsn(&self, lsn: u64) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&self.lease_path)
            .map_err(|e| {
                Error::internal(format!(
                    "failed to open lease '{}' to set pinned_lsn: {}",
                    self.lease_path.display(),
                    e
                ))
            })?;
        use std::io::{Seek, SeekFrom, Write};
        // Leases are always 8 bytes (init in register); overwrite in place.
        f.seek(SeekFrom::Start(0)).map_err(|e| {
            Error::internal(format!(
                "failed to seek lease '{}' to 0 for pinned_lsn write: {}",
                self.lease_path.display(),
                e
            ))
        })?;
        f.write_all(&lsn.to_le_bytes()).map_err(|e| {
            Error::internal(format!(
                "failed to write pinned_lsn to lease '{}': {}",
                self.lease_path.display(),
                e
            ))
        })?;
        // mtime failure is a warning, not an error: the pin VALUE is already
        // on disk. Returning Err would let set_handle_pin roll back its
        // registry contribution while the file still advertises the floor,
        // a strictly worse outcome than just relying on the next touch.
        if let Err(e) = f.set_modified(SystemTime::now()) {
            eprintln!(
                "Warning: failed to set mtime after pinned_lsn write on '{}': {} \
                 (pin value persisted; lease keep-alive may rely on next touch)",
                self.lease_path.display(),
                e
            );
        }
        Ok(())
    }

    /// Contribute this handle's `pinned_lsn` and rewrite the on-disk lease
    /// to the MIN across all live contributions. Returns the value written.
    /// Caller MUST call `remove_handle_pin(handle_id)` on drop.
    pub fn set_handle_pin(&self, handle_id: u64, pin: u64) -> Result<u64> {
        // Hold the registry lock across the file write so concurrent siblings
        // can't reorder the on-disk value behind the registry's current MIN.
        let mut reg = lease_pin_contributions()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let entry = reg.entry(self.lease_path.clone()).or_default();
        // Capture prior so we can roll back on write failure (a stale
        // contribution would inflate later sibling MIN computations).
        let prior = entry.insert(handle_id, pin);
        let new_min = entry.values().copied().min().unwrap_or(pin);
        if let Err(e) = self.set_pinned_lsn(new_min) {
            let entry = reg.entry(self.lease_path.clone()).or_default();
            match prior {
                Some(prev) => {
                    entry.insert(handle_id, prev);
                }
                None => {
                    entry.remove(&handle_id);
                    if entry.is_empty() {
                        reg.remove(&self.lease_path);
                    }
                }
            }
            return Err(e);
        }
        Ok(new_min)
    }

    /// Remove this handle's contribution and rewrite the lease to the new MIN.
    /// When the LAST contribution drops, write `0` (release sentinel) so the
    /// writer no longer pins WAL on this PID's behalf even if a non-overlay
    /// handle keeps the LeaseManager alive via the refcount.
    pub fn remove_handle_pin(&self, handle_id: u64) {
        // Lock held across the file write, same ordering invariant as set.
        let mut reg = lease_pin_contributions()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let outcome = if let Some(entry) = reg.get_mut(&self.lease_path) {
            entry.remove(&handle_id);
            if entry.is_empty() {
                reg.remove(&self.lease_path);
                Some(0u64)
            } else {
                Some(entry.values().copied().min().unwrap())
            }
        } else {
            None
        };
        if let Some(pin) = outcome {
            // Best-effort: a write failure leaves the on-disk pin at its prior
            // (conservative for the writer) value; next set_handle_pin refreshes.
            let _ = self.set_pinned_lsn(pin);
        }
    }

    /// The lease file path, for diagnostics and tests.
    pub fn path(&self) -> &Path {
        &self.lease_path
    }
}

impl Drop for LeaseManager {
    fn drop(&mut self) {
        // Unlink only when the last in-process handle for this lease drops.
        // Drop is best-effort (SIGKILL skips it); reap_stale_leases is the
        // backstop for crashes.
        let mut reg = lease_refcount().lock().unwrap_or_else(|p| p.into_inner());
        let should_unlink = match reg.get_mut(&self.lease_path) {
            Some(count) => {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    reg.remove(&self.lease_path);
                    true
                } else {
                    false
                }
            }
            // No registry entry shouldn't happen; be defensive and unlink.
            None => true,
        };
        drop(reg);
        if should_unlink {
            let _ = fs::remove_file(&self.lease_path);
        }
    }
}

/// Per-handle progress file at `<db>/readers/<pid>.<handle_id>.epoch`.
///
/// Holds the last `manifest_epoch` this handle successfully reloaded.
/// Distinct from the lease (which is heartbeat + WAL pin); a stale or
/// missing epoch file does NOT release the lease, only signals "this
/// handle has not proven a refresh past epoch N." Writer-side reaper
/// uses MIN across live PIDs' epoch files to gate retire-sidecar
/// unlinks. Drop unlinks; SIGKILL leaves stale files for the writer
/// to age out via lease-mtime correlation.
pub struct EpochFile {
    path: PathBuf,
}

impl EpochFile {
    /// Create / truncate `<readers_dir>/<pid>.<handle_id>.epoch` and
    /// write `initial_epoch` (the attach-loaded epoch). Truncating
    /// means a PID-reused-after-crash collision overwrites cleanly.
    /// Pass the same directory the lease lives in (typically derived
    /// from `LeaseManager::path().parent()`).
    pub fn create(readers_dir: &Path, handle_id: u64, initial_epoch: u64) -> Result<Self> {
        fs::create_dir_all(readers_dir).map_err(|e| {
            Error::internal(format!(
                "failed to create readers dir '{}': {}",
                readers_dir.display(),
                e
            ))
        })?;
        let pid = std::process::id();
        let path = readers_dir.join(format!("{}.{}.epoch", pid, handle_id));
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| {
                Error::internal(format!(
                    "failed to open epoch file '{}': {}",
                    path.display(),
                    e
                ))
            })?;
        use std::io::Write;
        f.write_all(&initial_epoch.to_le_bytes()).map_err(|e| {
            Error::internal(format!(
                "failed to write initial epoch to '{}': {}",
                path.display(),
                e
            ))
        })?;
        // sync_all is overkill on a non-durable signal; rely on page
        // cache (a writer that reads stale value just defers reaping
        // one cycle).
        Ok(Self { path })
    }

    /// Overwrite the 8-byte epoch payload. Cheap (1 syscall + mtime).
    /// Caller is responsible for monotonicity (writer's MIN logic
    /// tolerates non-monotonic stores but visibility regressions
    /// would defer reaping).
    pub fn write(&self, epoch: u64) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&self.path)
            .map_err(|e| {
                Error::internal(format!(
                    "failed to open epoch file '{}': {}",
                    self.path.display(),
                    e
                ))
            })?;
        use std::io::{Seek, SeekFrom, Write};
        f.seek(SeekFrom::Start(0)).map_err(|e| {
            Error::internal(format!(
                "failed to seek epoch file '{}': {}",
                self.path.display(),
                e
            ))
        })?;
        f.write_all(&epoch.to_le_bytes()).map_err(|e| {
            Error::internal(format!(
                "failed to write epoch to '{}': {}",
                self.path.display(),
                e
            ))
        })?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for EpochFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Read an epoch file's 8-byte payload. `None` if missing / wrong size.
pub fn read_epoch_file(path: &Path) -> Option<u64> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() != 8 {
        return None;
    }
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

/// Min `manifest_epoch` across all live readers' per-handle epoch
/// files. Writer's retire-sidecar reaper uses this to gate unlinks.
///
/// Algorithm:
///   1. Build set of live PIDs from `<pid>.lease` files (mtime within
///      `max_age`).
///   2. For each live PID, scan `<pid>.<*>.epoch` files. The PID's
///      effective epoch is the MIN across its handles (slowest sibling
///      gates).
///   3. Live PID with NO `.epoch` files = epoch 0 (legacy reader, or
///      attach window before the first epoch file lands).
///   4. Stale `.epoch` files for non-live PIDs are ignored.
///   5. Global result is the MIN across PIDs.
///
/// Returns `Ok(None)` when there are no live leases (no constraint).
/// Returns `Ok(Some(0))` when at least one live PID has no epoch file
/// or has a handle that hasn't refreshed (defer reaping).
pub fn min_reader_handle_epoch(dir: &Path, max_age: Duration) -> Result<Option<u64>> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(Error::internal(format!(
                "failed to read readers dir '{}': {}",
                dir.display(),
                e
            )))
        }
    };
    let now = SystemTime::now();
    let mut live_pids: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
    let mut epoch_files_by_pid: rustc_hash::FxHashMap<u32, Vec<PathBuf>> =
        rustc_hash::FxHashMap::default();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        match ext {
            "lease" => {
                if !lease_is_live_conservative(now, &entry, max_age) {
                    continue;
                }
                if let Ok(pid) = stem.parse::<u32>() {
                    live_pids.insert(pid);
                }
            }
            "epoch" => {
                // Stem is `<pid>.<handle_id>`; PID is the part before the dot.
                let pid_str = match stem.split_once('.') {
                    Some((p, _)) => p,
                    None => continue,
                };
                if let Ok(pid) = pid_str.parse::<u32>() {
                    epoch_files_by_pid.entry(pid).or_default().push(path);
                }
            }
            _ => {}
        }
    }
    if live_pids.is_empty() {
        return Ok(None);
    }
    let mut global_min = u64::MAX;
    for pid in &live_pids {
        let pid_min = match epoch_files_by_pid.get(pid) {
            None => 0, // legacy reader / pre-epoch-file attach window
            Some(paths) => paths
                .iter()
                .map(|p| read_epoch_file(p).unwrap_or(0))
                .min()
                .unwrap_or(0),
        };
        if pid_min < global_min {
            global_min = pid_min;
        }
    }
    Ok(Some(global_min))
}

/// Update the mtime of `path` to now (open w/o truncate, set_modified).
/// Returns the raw `io::Error` so callers can detect `NotFound` and route
/// through a self-heal path; wrapping in our error type erases that kind.
fn touch_path(path: &Path) -> std::io::Result<()> {
    let f = OpenOptions::new().write(true).open(path)?;
    f.set_modified(SystemTime::now())
}

/// Age of a lease relative to `now`. Future mtimes clamp to zero so a
/// reader is never neither-live-nor-stale (which would let the writer
/// unlink volumes out from under it).
fn lease_age_clamped(now: SystemTime, mtime: SystemTime) -> Duration {
    now.duration_since(mtime).unwrap_or(Duration::ZERO)
}

/// Conservative liveness: future mtimes and metadata-read failures both
/// clamp to live so the writer never destroys data behind an active reader.
fn lease_is_live_conservative(now: SystemTime, entry: &fs::DirEntry, max_age: Duration) -> bool {
    match entry.metadata().and_then(|m| m.modified()) {
        Ok(mtime) => lease_age_clamped(now, mtime) <= max_age,
        Err(_) => true,
    }
}

/// Return the lease files in `dir` whose mtime is within `max_age` of the
/// current time. Used by the writer's cleanup paths to decide whether
/// destructive operations (volume unlink, WAL truncation) must be deferred.
///
/// If `dir` does not exist, returns an empty list (no readers ever attached).
pub fn live_leases(dir: &Path, max_age: Duration) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(Error::internal(format!(
                "failed to read lease dir '{}': {}",
                dir.display(),
                e
            )))
        }
    };

    let now = SystemTime::now();
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("lease") {
            continue;
        }
        if lease_is_live_conservative(now, &entry, max_age) {
            out.push(path);
        }
    }
    Ok(out)
}

/// Read the 8-byte `pinned_lsn` payload. `None` for unreadable, missing,
/// or non-8-byte files (treated as no-pin so writer progress isn't blocked).
pub fn read_pinned_lsn(path: &Path) -> Option<u64> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() != 8 {
        return None;
    }
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

/// Minimum `pinned_lsn` across live leases. `Ok(None)` = no constraint,
/// writer may truncate up to `checkpoint_lsn`. `Ok(Some(lsn))` = keep WAL
/// entries with LSN >= `lsn`. A reader pin of `0` means presence-only and
/// is excluded so it doesn't pin everything to LSN 0.
pub fn min_pinned_lsn(dir: &Path, max_age: Duration) -> Result<Option<u64>> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(Error::internal(format!(
                "failed to read lease dir '{}': {}",
                dir.display(),
                e
            )))
        }
    };

    let now = SystemTime::now();
    let mut min: Option<u64> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("lease") {
            continue;
        }
        // Stale leases don't constrain truncation (reaped next pass).
        if !lease_is_live_conservative(now, &entry, max_age) {
            continue;
        }
        // 0 = released, None = unreadable; both excluded from the min.
        let lsn = match read_pinned_lsn(&path) {
            Some(0) => continue,
            Some(lsn) => lsn,
            None => continue,
        };
        min = Some(min.map_or(lsn, |m| m.min(lsn)));
    }
    Ok(min)
}

/// Remove lease files in `dir` whose mtime is older than `max_age`. Returns
/// the number of leases reaped. Called from the writer's cleanup paths
/// before computing whether destructive ops are safe.
pub fn reap_stale_leases(dir: &Path, max_age: Duration) -> Result<usize> {
    // Two-pass: collect live PIDs from .lease files, then reap any
    // file whose owning PID has no live lease (.lease, .epoch).
    let entries: Vec<_> = match fs::read_dir(dir) {
        Ok(e) => e.flatten().collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => {
            return Err(Error::internal(format!(
                "failed to read lease dir '{}': {}",
                dir.display(),
                e
            )))
        }
    };
    let now = SystemTime::now();
    let mut live_pids: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
    for entry in &entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("lease") {
            continue;
        }
        if !lease_is_live_conservative(now, entry, max_age) {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if let Ok(pid) = stem.parse::<u32>() {
                live_pids.insert(pid);
            }
        }
    }
    let mut reaped = 0;
    for entry in &entries {
        let path = entry.path();
        let pid = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<u32>().ok());
        let owning_pid_alive = pid.map(|p| live_pids.contains(&p)).unwrap_or(false);
        if owning_pid_alive {
            continue;
        }
        if fs::remove_file(&path).is_ok() {
            reaped += 1;
        }
    }
    Ok(reaped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::thread;

    fn tmp_db() -> tempfile::TempDir {
        tempfile::tempdir().expect("create tempdir")
    }

    #[test]
    fn register_creates_dir_and_file() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        assert!(lease.path().exists(), "lease file must exist");
        // register canonicalizes readers dir; on macOS that turns /var into
        // /private/var so dir.path() is no longer a prefix.
        let canonical_readers = fs::canonicalize(dir.path().join(READERS_DIR))
            .expect("readers dir should exist after register");
        assert!(
            lease.path().starts_with(&canonical_readers),
            "lease must be under canonical readers/ subdir (lease={}, readers={})",
            lease.path().display(),
            canonical_readers.display()
        );
        let pid_file = format!("{}.lease", std::process::id());
        assert_eq!(
            lease.path().file_name().and_then(|s| s.to_str()),
            Some(pid_file.as_str())
        );
    }

    #[test]
    fn drop_unlinks_lease() {
        let dir = tmp_db();
        let path = {
            let lease = LeaseManager::register(dir.path()).unwrap();
            lease.path().to_path_buf()
        };
        assert!(
            !path.exists(),
            "lease must be removed when LeaseManager drops"
        );
    }

    #[test]
    fn touch_advances_mtime() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        let mtime1 = fs::metadata(lease.path()).unwrap().modified().unwrap();

        // 50ms covers HFS+ (1s) and APFS (~1ns) without slowing the test.
        thread::sleep(Duration::from_millis(50));
        lease.touch().unwrap();
        let mtime2 = fs::metadata(lease.path()).unwrap().modified().unwrap();
        assert!(
            mtime2 > mtime1,
            "touch must advance mtime (was {:?}, now {:?})",
            mtime1,
            mtime2
        );
    }

    #[test]
    fn live_leases_returns_only_lease_files() {
        let dir = tmp_db();
        let _lease = LeaseManager::register(dir.path()).unwrap();

        // Drop a non-lease file in the readers dir; live_leases must ignore.
        let other = dir.path().join(READERS_DIR).join("not-a-lease.txt");
        File::create(&other).unwrap();

        let live = live_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(live.len(), 1, "must only count .lease files");
        assert_eq!(live[0].extension().and_then(|s| s.to_str()), Some("lease"));
    }

    #[test]
    fn live_leases_skips_stale_and_returns_fresh() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();

        // Backdate the lease's mtime to well past `max_age`.
        let old = SystemTime::now() - Duration::from_secs(3600);
        let f = OpenOptions::new().write(true).open(lease.path()).unwrap();
        f.set_modified(old).unwrap();
        drop(f);

        let live = live_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert!(
            live.is_empty(),
            "stale lease (>1h old) must not appear in live_leases(60s)"
        );

        // Touch brings it back to live.
        lease.touch().unwrap();
        let live = live_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(live.len(), 1, "after touch, lease must be live again");
    }

    #[test]
    fn live_leases_treats_future_mtime_as_live() {
        // Clock-step-backward / fs clock skew can put mtime in the future.
        // Such leases must count as live so a reader isn't invisible.
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        let future = SystemTime::now() + Duration::from_secs(3600);
        let f = OpenOptions::new().write(true).open(lease.path()).unwrap();
        f.set_modified(future).unwrap();
        drop(f);

        let live = live_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(live.len(), 1, "future-mtime lease must count as live");

        // And reap_stale_leases must NOT remove it.
        let reaped =
            reap_stale_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(reaped, 0, "future-mtime lease must NOT be reaped");
        assert!(lease.path().exists());
    }

    #[test]
    fn min_pinned_lsn_treats_future_mtime_as_live() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        lease.set_pinned_lsn(42).unwrap();
        let future = SystemTime::now() + Duration::from_secs(3600);
        let f = OpenOptions::new().write(true).open(lease.path()).unwrap();
        f.set_modified(future).unwrap();
        drop(f);

        let min = min_pinned_lsn(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(
            min,
            Some(42),
            "future-mtime lease's pinned_lsn must constrain WAL truncation"
        );
    }

    #[test]
    fn live_leases_on_missing_dir_returns_empty() {
        let dir = tmp_db();
        let live = live_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert!(
            live.is_empty(),
            "missing readers/ dir must yield empty list"
        );
    }

    #[test]
    fn reap_stale_leases_removes_old_keeps_fresh() {
        let dir = tmp_db();
        let fresh = LeaseManager::register(dir.path()).unwrap();

        let stale_path = dir.path().join(READERS_DIR).join("99999.lease");
        File::create(&stale_path).unwrap();
        let old = SystemTime::now() - Duration::from_secs(3600);
        let f = OpenOptions::new().write(true).open(&stale_path).unwrap();
        f.set_modified(old).unwrap();
        drop(f);

        let reaped =
            reap_stale_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(reaped, 1, "must reap exactly the stale lease");
        assert!(!stale_path.exists(), "stale lease must be gone");
        assert!(fresh.path().exists(), "fresh lease must remain");
    }

    #[test]
    fn reap_stale_leases_on_missing_dir_returns_zero() {
        let dir = tmp_db();
        let reaped =
            reap_stale_leases(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(reaped, 0);
    }

    // -----------------------------------------------------------------
    // pinned_lsn coverage
    // -----------------------------------------------------------------

    #[test]
    fn set_pinned_lsn_writes_eight_bytes_le_and_advances_mtime() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        let old = SystemTime::now() - Duration::from_secs(60);
        let f = OpenOptions::new().write(true).open(lease.path()).unwrap();
        f.set_modified(old).unwrap();
        drop(f);

        lease.set_pinned_lsn(0xDEAD_BEEF_CAFE_F00D).unwrap();

        let bytes = std::fs::read(lease.path()).unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(
            u64::from_le_bytes(bytes.try_into().unwrap()),
            0xDEAD_BEEF_CAFE_F00D,
            "pinned_lsn must be readable as little-endian u64"
        );

        let mtime = std::fs::metadata(lease.path()).unwrap().modified().unwrap();
        assert!(
            mtime > old,
            "set_pinned_lsn must advance mtime past backdated value"
        );
    }

    #[test]
    fn register_initializes_lease_with_zero_pinned_lsn() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        // 8 zero bytes from register avoids a zero-byte window that
        // min_pinned_lsn would treat as corrupt.
        assert_eq!(std::fs::metadata(lease.path()).unwrap().len(), 8);
        assert_eq!(
            read_pinned_lsn(lease.path()),
            Some(0),
            "fresh lease must read back as pinned_lsn = 0 (presence only)"
        );
    }

    #[test]
    fn read_pinned_lsn_returns_some_after_set() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        lease.set_pinned_lsn(42).unwrap();
        assert_eq!(read_pinned_lsn(lease.path()), Some(42));
    }

    #[test]
    fn read_pinned_lsn_returns_none_for_wrong_size_file() {
        let dir = tmp_db();
        let path = dir.path().join(READERS_DIR).join("99999.lease");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, [1u8, 2, 3, 4]).unwrap();
        assert_eq!(
            read_pinned_lsn(&path),
            None,
            "non-8-byte lease must read as None (caller treats as no-pin)"
        );
    }

    #[test]
    fn epoch_write_recreates_missing_file_and_restores_min() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        let readers_dir = lease.path().parent().unwrap().to_path_buf();
        let epoch = EpochFile::create(&readers_dir, next_handle_id(), 7).unwrap();
        let path = epoch.path().to_path_buf();

        assert_eq!(
            min_reader_handle_epoch(&readers_dir, Duration::from_secs(60)).unwrap(),
            Some(7)
        );

        fs::remove_file(&path).unwrap();
        assert_eq!(
            min_reader_handle_epoch(&readers_dir, Duration::from_secs(60)).unwrap(),
            Some(0),
            "a live reader with a missing epoch file must fail closed"
        );

        epoch.write(42).unwrap();

        assert_eq!(
            read_epoch_file(&path),
            Some(42),
            "epoch write must self-heal when a stale reaper removed the file"
        );
        assert_eq!(
            min_reader_handle_epoch(&readers_dir, Duration::from_secs(60)).unwrap(),
            Some(42)
        );
    }

    #[test]
    fn touch_recreates_missing_lease_and_restores_pin() {
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();
        let handle_id = next_handle_id();
        lease.set_handle_pin(handle_id, 1234).unwrap();
        let path = lease.path().to_path_buf();
        assert_eq!(read_pinned_lsn(&path), Some(1234));

        fs::remove_file(&path).unwrap();
        assert!(!path.exists());

        lease.touch().unwrap();

        assert!(path.exists(), "touch must self-heal a reaped lease file");
        assert_eq!(
            read_pinned_lsn(&path),
            Some(1234),
            "touch must restore the registry MIN pin"
        );
        lease.remove_handle_pin(handle_id);
    }

    #[test]
    fn min_pinned_lsn_returns_none_when_only_zero_pin_readers() {
        let dir = tmp_db();
        // Fresh register writes pinned_lsn=0 (presence only); excluded from min.
        let _r = LeaseManager::register(dir.path()).unwrap();
        let m = min_pinned_lsn(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(
            m, None,
            "presence-only readers (pinned_lsn=0) must NOT constrain WAL truncation"
        );
    }

    #[test]
    fn min_pinned_lsn_returns_lowest_lsn_across_live_readers() {
        let dir = tmp_db();
        let r1 = LeaseManager::register(dir.path()).unwrap();
        r1.set_pinned_lsn(100).unwrap();

        let lower = dir.path().join(READERS_DIR).join("99998.lease");
        std::fs::write(&lower, 50u64.to_le_bytes()).unwrap();

        let higher = dir.path().join(READERS_DIR).join("99999.lease");
        std::fs::write(&higher, 200u64.to_le_bytes()).unwrap();

        let m = min_pinned_lsn(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(m, Some(50), "min across {{50, 100, 200}} must be 50");
    }

    #[test]
    fn min_pinned_lsn_skips_stale_leases() {
        let dir = tmp_db();
        let live_lease = LeaseManager::register(dir.path()).unwrap();
        live_lease.set_pinned_lsn(500).unwrap();

        let stale = dir.path().join(READERS_DIR).join("99000.lease");
        std::fs::write(&stale, 1u64.to_le_bytes()).unwrap();
        let old = SystemTime::now() - Duration::from_secs(3600);
        let f = OpenOptions::new().write(true).open(&stale).unwrap();
        f.set_modified(old).unwrap();
        drop(f);

        let m = min_pinned_lsn(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(
            m,
            Some(500),
            "stale lease at LSN 1 must NOT constrain min (live lease wins)"
        );
    }

    #[test]
    fn min_pinned_lsn_excludes_pin_lsn_zero_release() {
        let dir = tmp_db();
        let pinned = LeaseManager::register(dir.path()).unwrap();
        pinned.set_pinned_lsn(123).unwrap();

        // lsn=0 with live mtime = released, excluded from min.
        let released = dir.path().join(READERS_DIR).join("99000.lease");
        std::fs::write(&released, 0u64.to_le_bytes()).unwrap();

        let m = min_pinned_lsn(&dir.path().join(READERS_DIR), Duration::from_secs(60)).unwrap();
        assert_eq!(
            m,
            Some(123),
            "lsn=0 released lease must NOT pull min down to 0"
        );
    }

    #[test]
    fn min_pinned_lsn_returns_none_on_missing_dir() {
        let dir = tmp_db();
        let m = min_pinned_lsn(&dir.path().join("nonexistent"), Duration::from_secs(60)).unwrap();
        assert_eq!(m, None);
    }

    #[test]
    fn set_handle_pin_writes_min_across_in_process_handles() {
        // Two handles share one <pid>.lease; on-disk value must be the MIN.
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();

        let id_a = next_handle_id();
        let id_b = next_handle_id();
        assert_ne!(id_a, id_b);

        let m = lease.set_handle_pin(id_a, 500).unwrap();
        assert_eq!(m, 500, "only one contribution → MIN == A's pin");
        assert_eq!(read_pinned_lsn(lease.path()), Some(500));

        let m = lease.set_handle_pin(id_b, 100).unwrap();
        assert_eq!(m, 100, "MIN must drop to B's lower pin");
        assert_eq!(read_pinned_lsn(lease.path()), Some(100));

        // A's higher pin must not overwrite B's lower floor.
        let m = lease.set_handle_pin(id_a, 900).unwrap();
        assert_eq!(m, 100, "A's higher pin must NOT overwrite B's lower floor");
        assert_eq!(read_pinned_lsn(lease.path()), Some(100));

        lease.remove_handle_pin(id_b);
        assert_eq!(
            read_pinned_lsn(lease.path()),
            Some(900),
            "after B releases, on-disk pin advances to A's value"
        );

        // Last contribution dropped → file rewritten to 0 (release sentinel).
        lease.remove_handle_pin(id_a);
        assert_eq!(
            read_pinned_lsn(lease.path()),
            Some(0),
            "after last contribution drops, on-disk pin must be reset \
             to the 0 release sentinel so the writer no longer pins \
             WAL on this PID's behalf"
        );
    }

    #[test]
    fn drop_does_not_unlink_while_another_in_process_handle_holds_lease() {
        // Two LeaseManagers on the same db share one <pid>.lease;
        // dropping one must not unlink while the other is alive.
        let dir = tmp_db();
        let lease1 = LeaseManager::register(dir.path()).unwrap();
        let path1 = lease1.path().to_path_buf();
        let lease2 = LeaseManager::register(dir.path()).unwrap();
        assert_eq!(
            lease1.path(),
            lease2.path(),
            "same PID + same db path → same lease file"
        );

        // Drop the first handle; the file must still exist.
        drop(lease1);
        assert!(
            path1.exists(),
            "lease file must survive while another in-process handle holds it"
        );

        // Drop the second handle; now the file should be gone.
        drop(lease2);
        assert!(
            !path1.exists(),
            "lease file must be removed once the LAST in-process handle drops"
        );
    }

    #[test]
    fn re_register_overwrites_existing_lease_for_same_pid() {
        // Models a process that crashed and restarted reusing its PID before reap.
        let dir = tmp_db();
        let _first = LeaseManager::register(dir.path()).unwrap();
        let second = LeaseManager::register(dir.path()).unwrap();
        assert!(second.path().exists());
    }
}
