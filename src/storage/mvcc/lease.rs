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

//! Cross-process reader presence + WAL pin via
//! `<db>/readers/<pid>.lease` files.
//!
//! Each reader keeps an 8-byte little-endian `pinned_lsn` payload in
//! its lease file. The writer combines two signals from the lease:
//!
//! 1. **Presence (mtime).** Readers bump the file's mtime on open
//!    and on every query/refresh. The writer's destructive paths
//!    (volume unlink, WAL truncation) defer while any lease is
//!    within `max_age` (default `2 * checkpoint_interval`) of now;
//!    older leases are stale (the reader process died without
//!    cleaning up) and get reaped.
//! 2. **WAL pin (payload).** The 8-byte payload is the WAL LSN the
//!    reader still needs. The writer's `truncate_wal` floor is
//!    `min(checkpoint_lsn, min_pinned_lsn - 1)` so a reader pinned
//!    at LSN P keeps entries `P..` available. A payload of `0`
//!    means "presence only, no WAL constraint" — a freshly opened
//!    reader that hasn't yet attached uses this value, and the
//!    `0` is excluded from the min so it doesn't pin everything.
//!
//! No `flock` is used. File existence + mtime + payload is the only
//! signal. PID reuse is bounded: a reaped stale lease can be re-
//! created by a fresh process with the same PID, and that's fine —
//! the new process re-asserts presence. There's no cross-PID
//! identity check because the contract (defer-while-live, reap-
//! when-stale) doesn't need one.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use crate::core::{Error, Result};

/// Subdirectory under the database path that holds reader lease files.
pub const READERS_DIR: &str = "readers";

/// Process-wide ref count of lease handles per path. Two
/// `ReadOnlyDatabase` instances opened against the same DSN in one
/// process share the SAME `<pid>.lease` file (the file path is keyed
/// by PID, not handle); each `LeaseManager::register` bumps the
/// count and each `Drop` decrements. Only when the count reaches
/// zero do we unlink the file. Without this, dropping the first
/// of two concurrent in-process readers would unlink the only
/// lease while the other reader is still active, letting the writer
/// stop deferring volume cleanup / WAL truncation.
fn lease_refcount() -> &'static Mutex<HashMap<PathBuf, usize>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Per-(lease_path, handle_id) WAL pin contributions.
/// All in-process read-only handles writing to the SAME `<pid>.lease`
/// must publish a pinned_lsn that's the MIN of every live handle's
/// contribution — otherwise a handle with a higher floor would
/// silently overwrite a lagging handle's lower floor and let the
/// writer truncate WAL the lagging handle still needs.
///
/// Handles register a unique `handle_id` (from `next_handle_id`),
/// call `set_handle_pin(id, pin)` to update their contribution, and
/// `remove_handle_pin(id)` on drop. The on-disk lease always
/// reflects the current MIN.
type PinContributions = HashMap<PathBuf, std::collections::BTreeMap<u64, u64>>;
fn lease_pin_contributions() -> &'static Mutex<PinContributions> {
    static REGISTRY: OnceLock<Mutex<PinContributions>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Monotonic process-wide handle id allocator. Each
/// `ReadOnlyDatabase` (or any future handle that contributes to a
/// lease pin) takes a fresh id via `next_handle_id` and uses it as
/// the key into `lease_pin_contributions`.
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
    /// Create a lease for this process. Creates the `readers/` directory if
    /// missing. If a stale lease for this PID already exists (process
    /// crashed previously, same PID happened to be reused), we overwrite it
    /// — same effect as if the previous instance had cleaned up.
    pub fn register(db_path: &Path) -> Result<Self> {
        let dir = db_path.join(READERS_DIR);
        fs::create_dir_all(&dir).map_err(|e| {
            Error::internal(format!(
                "failed to create lease dir '{}': {}",
                dir.display(),
                e
            ))
        })?;

        // Canonicalize the readers/ directory BEFORE deriving the
        // lease_path. Two same-process read-only opens of the same
        // DB through path aliases (symlink, `..` components,
        // double slashes, ...) point at the same physical
        // `<db>/readers/<pid>.lease` file but would otherwise
        // produce different `PathBuf` registry keys. The second
        // `register` would then see refcount == 0 in
        // `lease_refcount`, truncate the active lease (wiping
        // the first handle's pinned_lsn), and the first handle's
        // Drop could later unlink the file while the alias handle
        // is still alive — letting writer WAL truncation and
        // volume cleanup ignore an active reader. Canonicalizing
        // once here makes both registries (`lease_refcount` and
        // `lease_pin_contributions`) and the on-disk path agree
        // on the same key per filesystem location.
        //
        // Falls back to the joined path on canonicalize failure
        // (rare — `create_dir_all` just succeeded). The
        // pre-canonicalization behavior is preserved in that case
        // so a fs/platform that can't canonicalize doesn't fail
        // the open outright.
        let dir = fs::canonicalize(&dir).unwrap_or(dir);

        let pid = std::process::id();
        let lease_path = dir.join(format!("{}.lease", pid));

        // Bump the in-process refcount FIRST and decide whether
        // to truncate based on whether another in-process handle
        // is already holding this lease. If it is
        // (existing_count > 0), open WITHOUT truncate so we
        // preserve the 8-byte `pinned_lsn` the first handle has
        // already written. Truncating would zero the payload and
        // `min_pinned_lsn` would treat the file as corrupt
        // (non-8-byte length), allowing the writer to truncate
        // WAL the first handle still depends on.
        //
        // For the no-other-handle case (count == 0) we DO
        // truncate AND initialize 8 zero bytes (`pinned_lsn = 0`,
        // meaning "presence only, no WAL constraint"). Any
        // existing file is from a stale prior process with our
        // PID and overwriting it is safe.
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
            // Roll back the refcount bump on open failure so a
            // future register() can still try to truncate the stale
            // file.
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

        // Set mtime to now so live_leases sees us immediately.
        // Same touch on both paths: a co-existing handle's pin
        // payload is preserved by skipping the truncate, but
        // mtime still gets refreshed.
        touch_path(&lease_path)?;

        Ok(Self { lease_path })
    }

    /// Bump the lease's mtime to the current time. Cheap (one syscall on
    /// Unix, equivalent on Windows). Called from query/execute/refresh
    /// entry points on read-only handles.
    pub fn touch(&self) -> Result<()> {
        touch_path(&self.lease_path)
    }

    /// Write `lsn` as the reader's WAL pin into the lease file
    /// (8 bytes little-endian, replacing any prior content), then
    /// bump mtime. The writer's `truncate_wal` floor is constrained
    /// by the minimum pinned_lsn across live leases. Pass the
    /// current `db.shm.visible_commit_lsn` to pin to "everything
    /// published so far"; pass `0` to release the pin while keeping
    /// presence.
    ///
    /// Atomicity: opens without truncate and overwrites the 8 bytes
    /// in place from offset 0. The file is initialized to 8 zero
    /// bytes by `register()`, so its size never changes after that
    /// — a concurrent writer scan always sees a valid 8-byte
    /// snapshot (kernel buffer-cache updates of <= page-size writes
    /// are observed atomically on supported platforms). Without the
    /// register-side init, this method's first call would race a
    /// concurrent scan that observed the file at size 0 and
    /// silently excluded the reader.
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
        // Always write from offset 0 so we overwrite the prior 8
        // bytes in place. Leases are always exactly 8 bytes (init
        // happens in register()) and `read_pinned_lsn` rejects any
        // other length.
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
        // mtime advance keeps us live for `max_age` regardless of the
        // FS's atime/ctime semantics. Demoted to a warning: the
        // 8 bytes are already on disk, so the pin VALUE is
        // correct from the writer's perspective. If we returned
        // Err here, callers like `set_handle_pin` would roll back
        // their registry contribution while the on-disk file
        // already advertised the higher floor — the writer
        // would scan the higher disk value, the reader would
        // think the lower registry value was active, and the
        // writer could truncate WAL the reader still needs.
        // mtime miss only risks earlier reaping of this lease
        // (caller's next `touch` will refresh it).
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

    /// Contribute this in-process handle's
    /// `pinned_lsn` to the shared lease file. The on-disk file
    /// is updated to the MIN across every live handle's
    /// contribution, so a high pin from one handle never
    /// silently overwrites a lower pin from a lagging sibling.
    ///
    /// Caller must remove its contribution via
    /// `remove_handle_pin(handle_id)` on drop, otherwise the
    /// pin will keep constraining writer truncation until the
    /// process exits.
    ///
    /// Returns the value actually written to disk (the MIN). If
    /// the MIN didn't change since the last write, the file
    /// write is skipped.
    pub fn set_handle_pin(&self, handle_id: u64, pin: u64) -> Result<u64> {
        // Hold the registry lock ACROSS the file
        // write. Prior version computed `min` under lock, dropped
        // the lock, then wrote to disk. Two interleaving handles
        // could then publish their MIN-at-compute-time in arbitrary
        // order, leaving the on-disk value as whichever wrote last
        // — even if the registry's current MIN was lower. With the
        // lock held, file writes serialize and the LAST-written
        // value always matches the registry's CURRENT MIN.
        //
        // The lock spans an 8-byte file write; under typical
        // workloads (one pin update per refresh per handle), the
        // serialization cost is negligible.
        let mut reg = lease_pin_contributions()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let entry = reg.entry(self.lease_path.clone()).or_default();
        // Capture the prior contribution (if any) so we can roll
        // back on a file-write failure. Callers use `is_ok()` to
        // decide whether the pin is active; leaving a stale
        // contribution in the registry after a failed write would
        // make later sibling `set_handle_pin` calls compute a MIN
        // that includes our phantom value, pinning WAL we don't
        // actually need.
        let prior = entry.insert(handle_id, pin);
        let new_min = entry.values().copied().min().unwrap_or(pin);
        if let Err(e) = self.set_pinned_lsn(new_min) {
            // Roll back the contribution to its prior state. If
            // there was no prior entry, remove ours; if there was,
            // restore it.
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

    /// Remove this handle's pin contribution and
    /// rewrite the on-disk lease to reflect the new MIN across
    /// remaining handles. Called from the owning handle's
    /// `Drop`. If no handles remain for this lease path, the
    /// registry entry is removed and the on-disk lease is left
    /// untouched (the `LeaseManager`'s own `Drop` may then
    /// unlink the file via the refcount path).
    pub fn remove_handle_pin(&self, handle_id: u64) {
        // Same ordering invariant as
        // set_handle_pin — hold the registry lock across the
        // file write so the on-disk value can't be reordered
        // behind a concurrent set/remove from a sibling handle.
        let mut reg = lease_pin_contributions()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let outcome = if let Some(entry) = reg.get_mut(&self.lease_path) {
            entry.remove(&handle_id);
            if entry.is_empty() {
                // The LAST contribution just
                // dropped. Don't leave the file at its old MIN —
                // that would keep the writer pinning WAL on
                // behalf of a reader who's gone, especially if
                // another non-overlay handle (e.g. a
                // `Database::open(?read_only=true)` reader that
                // doesn't tail) is keeping the LeaseManager
                // alive via the refcount. Write `0` (the
                // "released" sentinel: `min_pinned_lsn`'s scan
                // skips it) so the writer's truncate floor is no
                // longer constrained by this PID.
                reg.remove(&self.lease_path);
                Some(0u64)
            } else {
                Some(entry.values().copied().min().unwrap())
            }
        } else {
            None
        };
        if let Some(pin) = outcome {
            // Best-effort: a write failure here just means the
            // on-disk pin stays at the prior (possibly higher)
            // value, which is conservative for the writer
            // (truncate floor is still bounded by the file's
            // value). The remaining handles' next set_handle_pin
            // call will refresh it.
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
        // Decrement the per-path refcount and unlink ONLY when the
        // last in-process handle for this lease file drops. Without
        // this, opening two `ReadOnlyDatabase` instances against the
        // same DSN, then dropping one, would unlink the only lease
        // while the other is still active — letting the writer
        // prematurely stop deferring volume cleanup / WAL truncation.
        //
        // OS-level guarantees: as before, Drop is best-effort
        // (SIGKILL / abort skip it). The writer's
        // `reap_stale_leases` removes the file once it goes stale
        // past `max_age`. The refcount only protects clean-shutdown
        // behaviour for in-process multi-handle cases.
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
            // No registry entry: shouldn't happen, but be defensive
            // and unlink (single-handle behaviour).
            None => true,
        };
        drop(reg);
        if should_unlink {
            let _ = fs::remove_file(&self.lease_path);
        }
    }
}

/// Update the mtime of `path` to now. Opens the file with write access (no
/// truncation) and calls `File::set_modified`.
fn touch_path(path: &Path) -> Result<()> {
    let f = OpenOptions::new().write(true).open(path).map_err(|e| {
        Error::internal(format!(
            "failed to open lease '{}' for touch: {}",
            path.display(),
            e
        ))
    })?;
    f.set_modified(SystemTime::now()).map_err(|e| {
        Error::internal(format!(
            "failed to set mtime on lease '{}': {}",
            path.display(),
            e
        ))
    })?;
    Ok(())
}

/// Age of a lease relative to `now`. A future mtime (clock stepped
/// backward, or filesystem clock skew) is treated as age zero, NOT
/// dropped. Dropping it would let `live_leases` / `min_pinned_lsn`
/// silently omit an active reader while `reap_stale_leases` (which
/// also rejects future mtimes) leaves the file in place — the lease
/// would be neither live nor stale, and the writer could unlink
/// volumes / truncate WAL out from under the reader.
fn lease_age_clamped(now: SystemTime, mtime: SystemTime) -> Duration {
    now.duration_since(mtime).unwrap_or(Duration::ZERO)
}

/// Liveness verdict for a lease entry whose mtime read may have
/// failed. Returns `true` when the lease should be treated as live —
/// i.e. when the writer must defer destructive ops AND must NOT
/// reap. Future mtimes clamp to age zero. Metadata / mtime read
/// failures (EACCES, EIO, raced unlink) clamp to live as well so a
/// transient stat error cannot create the same neither-live-nor-
/// stale window as a future mtime did.
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

/// Read the 8-byte `pinned_lsn` payload from a lease file. Returns
/// `Some(lsn)` for a well-formed lease, `None` for unreadable files
/// (removed between the directory scan and this read — common race
/// that we silently ignore so the writer's `min_pinned_lsn` keeps
/// making progress) or files whose length is not exactly 8 bytes
/// (corrupt / partially written).
pub fn read_pinned_lsn(path: &Path) -> Option<u64> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() != 8 {
        return None;
    }
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

/// Minimum `pinned_lsn` across live leases in `dir`. Returns
/// `Ok(None)` if no live reader is present (the writer has no WAL
/// pinning constraint and may truncate up to `checkpoint_lsn`).
///
/// `Ok(Some(lsn))` when at least one live reader has a non-zero pin;
/// the writer must keep WAL entries with LSN >= `lsn` (i.e. truncate
/// up to `lsn - 1` at most). A `pinned_lsn` of `0` from a reader
/// means "presence only, no WAL entries needed" and is excluded from
/// the min so it doesn't pin everything to LSN 0.
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
        // Liveness gate first — stale leases don't constrain
        // truncation (they'll get reaped on the next pass
        // anyway). Future mtimes clamp to age zero, and
        // metadata / mtime read failures clamp to live, so a
        // transient stat error can't drop a real reader's WAL
        // pin contribution.
        if !lease_is_live_conservative(now, &entry, max_age) {
            continue;
        }
        // Read the pinned_lsn payload. A `0` value means the reader
        // has no active pin — skip so the min isn't pulled down to 0.
        // None covers unreadable / partially written / corrupt files
        // (no constraint we can derive).
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
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
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
    let mut reaped = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("lease") {
            continue;
        }
        // Future mtimes (clamped to age zero) are NEVER stale,
        // and metadata / mtime read failures are also NEVER
        // stale. Both cases match `live_leases` so a single
        // lease can never be both not-live AND not-stale and
        // silently linger.
        let stale = !lease_is_live_conservative(now, &entry, max_age);
        if stale && fs::remove_file(&path).is_ok() {
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
        // `LeaseManager::register` now canonicalizes the readers
        // dir before deriving lease_path (so two aliased opens
        // share the same registry key). On macOS tempdirs that
        // means `/var/folders/...` becomes `/private/var/folders/...`,
        // so the literal `dir.path()` is no longer a prefix of
        // `lease.path()`. Compare against the canonical readers
        // dir instead.
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

        // Sleep enough that the filesystem records a different mtime. macOS
        // HFS+ has 1s mtime resolution; APFS is ~1ns. Use 50ms to be safe
        // on most filesystems while keeping the test fast.
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
        // Clock-step-backward / fs clock skew: a lease's mtime can
        // be in the future relative to `now`. live_leases used to
        // drop those (Err from duration_since) while reap_stale
        // also left them in place, so a real reader could be
        // invisible to liveness checks.
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
        // Don't create any lease — readers/ dir doesn't exist.
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

        // Manually create a stale lease for a different "pid".
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
        // Backdate to verify mtime advances.
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
        // register() always writes 8 zero bytes so the file is
        // immediately a valid lease (avoids a zero-byte window
        // where min_pinned_lsn would treat it as corrupt).
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
        // Write 4 bytes — not a valid lease payload.
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
    fn min_pinned_lsn_returns_none_when_only_zero_pin_readers() {
        let dir = tmp_db();
        // Fresh register() writes pinned_lsn = 0 (presence only).
        // Such readers must NOT pin WAL — min_pinned_lsn excludes
        // them.
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

        // Simulate a sibling reader by creating a second lease file
        // for a different "pid" with a lower LSN.
        let lower = dir.path().join(READERS_DIR).join("99998.lease");
        std::fs::write(&lower, 50u64.to_le_bytes()).unwrap();

        // And a higher one to verify min, not max.
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

        // Stale lease at very low LSN — must be ignored.
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

        // Sibling reader released its pin (lsn=0) but still has a live
        // mtime. Per contract: lsn=0 means "no pin" and is excluded.
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
        // two in-process handles share the same
        // <pid>.lease. Each contributes a pin via set_handle_pin
        // with its own handle_id. The on-disk value must be the
        // MIN across both, NOT whichever handle wrote last.
        let dir = tmp_db();
        let lease = LeaseManager::register(dir.path()).unwrap();

        let id_a = next_handle_id();
        let id_b = next_handle_id();
        assert_ne!(id_a, id_b);

        // Handle A contributes pin = 500.
        let m = lease.set_handle_pin(id_a, 500).unwrap();
        assert_eq!(m, 500, "only one contribution → MIN == A's pin");
        assert_eq!(read_pinned_lsn(lease.path()), Some(500));

        // Handle B contributes a LOWER pin = 100. MIN drops to 100.
        let m = lease.set_handle_pin(id_b, 100).unwrap();
        assert_eq!(m, 100, "MIN must drop to B's lower pin");
        assert_eq!(read_pinned_lsn(lease.path()), Some(100));

        // Handle A advances to pin = 900 (higher). MIN stays at 100
        // because B is still at 100 — A's higher pin does NOT
        // overwrite B's floor.
        let m = lease.set_handle_pin(id_a, 900).unwrap();
        assert_eq!(m, 100, "A's higher pin must NOT overwrite B's lower floor");
        assert_eq!(read_pinned_lsn(lease.path()), Some(100));

        // Handle B drops out. Now the MIN is A's 900.
        lease.remove_handle_pin(id_b);
        assert_eq!(
            read_pinned_lsn(lease.path()),
            Some(900),
            "after B releases, on-disk pin advances to A's value"
        );

        // A drops out too. the registry entry's
        // last contribution is removed → the lease file is
        // explicitly written with `0` (release sentinel) so
        // min_pinned_lsn skips it. Without this, the file
        // would retain A's stale 900 and the writer would
        // keep honoring it for the lifetime of the
        // refcount-keeping handle.
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
        // Repro for the same-process double-handle bug: two
        // LeaseManager registrations on the same db path point at
        // the same `<pid>.lease` file. Dropping one must NOT
        // unlink — the other is still alive.
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
        // Same PID, second register: must succeed (this models a process
        // that crashed and restarted reusing its own PID before reap).
        let dir = tmp_db();
        let _first = LeaseManager::register(dir.path()).unwrap();
        let second = LeaseManager::register(dir.path()).unwrap();
        assert!(second.path().exists());
    }
}
