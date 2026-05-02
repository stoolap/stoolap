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

//! Cross-process shared header for SWMR v2 at `<db>/db.shm`.
//!
//! The writer `mmap`s this file read-write; readers `mmap` it read-only.
//! Both use `MAP_SHARED` so writes propagate to readers via the kernel
//! page cache. Fields are naturally-aligned `AtomicU64`s so
//! hardware-atomic loads/stores on x86_64 and aarch64 work cross-process
//! without tearing.
//!
//! ## Memory ordering (cross-process caveat)
//!
//! Rust's `AtomicU64` has well-defined semantics for threads within a
//! single process; the abstract machine does NOT guarantee cross-process
//! ordering. In practice, on the targets stoolap supports (Linux +
//! macOS, x86_64 and aarch64), naturally-aligned 8-byte loads/stores
//! are atomic at the hardware level and writes to `MAP_SHARED` memory
//! are visible to other mappers via cache coherence after the writer's
//! next syscall boundary. The SWMR v2 publish ordering requires the
//! writer to advance `visible_commit_lsn` only after the corresponding
//! WAL bytes have reached the WAL file. `SyncMode::Normal` / `Full`
//! add their usual fsync guarantees; `SyncMode::None` may lag
//! visibility until its buffered WAL bytes are flushed to the file.
//!
//! ## Torn-init detection
//!
//! The writer initializes `magic` and `version` first, zeros all fields,
//! then writes `init_done = MAGIC_READY` as the LAST operation. Readers
//! that observe `init_done != MAGIC_READY` refuse the attach. This
//! covers the case where the writer crashes mid-init or a foreign
//! process leaves a stale file.
//!
//! ## Windows
//!
//! Windows support is not yet implemented. `ShmHandle::create_writer`
//! and `open_reader` return `NotImplemented` on non-Unix targets. SWMR
//! v2 callers must check platform before opening.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::{Error, Result};

/// Sentinel written last by the writer during init; readers refuse any
/// `ShmHandle::open_reader` whose `init_done` field doesn't match.
pub const SHM_INIT_DONE_MAGIC: u64 = 0x5245414459305057; // ASCII "READY0PW" marker
/// Magic identifying a stoolap SWMR v2 `db.shm`.
pub const SHM_MAGIC: u32 = 0x535A4D32; // "SZM2" LE — stoolap shm v2
/// Current header version. Any future format change bumps this.
pub const SHM_VERSION: u32 = 1;

/// Total size of `db.shm`. One page on all supported platforms. Most of
/// the space is reserved for fields v3+ may add without another file
/// layout change.
pub const SHM_SIZE: usize = 4096;

/// Filename inside the database directory.
pub const SHM_FILENAME: &str = "db.shm";

/// Shared memory header. Layout is stable across stoolap versions with
/// matching `SHM_VERSION`.
///
/// All multi-byte fields are little-endian on disk (the only supported
/// platforms are LE). `AtomicU64` is used directly so callers can
/// `load(Acquire)` / `store(Release)` against the mmap'd region.
#[repr(C, align(8))]
pub struct ShmHeader {
    /// Magic number (`SHM_MAGIC`). Not atomic — written once by writer
    /// at create time; readers validate before any atomic access.
    pub magic: u32,
    /// Format version (`SHM_VERSION`).
    pub version: u32,
    /// Set to `SHM_INIT_DONE_MAGIC` by the writer as its LAST
    /// initialization step. Readers refuse to attach when this is
    /// anything else — that indicates a half-initialized or foreign
    /// file.
    pub init_done: AtomicU64,
    /// Bumped by the writer on every engine startup. Readers snapshot
    /// this at open and compare on every refresh; a change means the
    /// writer crashed and recovered, and the reader's pinned state is
    /// invalid (must hard-reopen).
    pub writer_generation: AtomicU64,
    /// The commit_seq + LSN watermark: rows whose commit_seq <= this
    /// are visible to readers. Published by the writer AFTER WAL
    /// fsync of the commit marker AND `registry.complete_commit`
    /// (SWMR v2 Phase C).
    pub visible_commit_lsn: AtomicU64,
    /// Bumped per successful checkpoint cycle after all manifests
    /// persist. Reader auto-refresh polls this to decide whether to
    /// reload manifests.
    pub manifest_epoch: AtomicU64,
    /// Bumped per DDL event (CREATE INDEX, CREATE VIEW, etc.) — a
    /// superset of manifest_epoch for pure-DDL operations that don't
    /// rewrite manifests.
    pub catalog_epoch: AtomicU64,
    /// Monotonically increases when any column rename / drop / add
    /// changes the schema version for any table. Reader uses this to
    /// invalidate prepared-statement cache entries.
    pub schema_generation: AtomicU64,
    /// First live WAL file id. Readers whose pinned_lsn falls below
    /// this must return `Error::SnapshotExpired` — the bytes are gone.
    pub wal_chain_head: AtomicU64,
    /// Current (latest) WAL file id. Reader's WAL-tail iterates
    /// `wal_chain_head..=wal_current`.
    pub wal_current: AtomicU64,
    /// Minimum `pinned_lsn` across live reader leases. Writer's
    /// `truncate_wal` floor is `min(checkpoint_lsn, min_pinned_lsn)`.
    pub min_pinned_lsn: AtomicU64,
    /// SWMR v2 P2 perf fix: lowest LSN of the first DML entry of any
    /// currently-active user transaction (`txn_id > 0`). `u64::MAX`
    /// means no active user txns, `0` means writer hasn't published
    /// yet (treat as "scan from 0"). Reader's WAL-tail uses this as
    /// the Phase 2 entry scan floor: any entry with LSN below this
    /// watermark is guaranteed to belong to a transaction that
    /// committed (and was applied) BEFORE this watermark was last
    /// snapshotted. Without it, Phase 2 has to scan all live WAL
    /// entries from LSN 0 on every refresh; with it, scans become
    /// O(delta) for the steady-state no-long-running-txn case.
    pub oldest_active_txn_lsn: AtomicU64,
    /// Seqlock for coherent (visible_commit_lsn,
    /// oldest_active_txn_lsn) snapshots. Standard "bump-odd /
    /// bump-even" shape:
    ///   - Writer: bump to ODD (Acquire+Release fence) BEFORE
    ///     mutating either field, store both fields, then bump to
    ///     EVEN AFTER both stores complete.
    ///   - Reader: load `seq_before`; if ODD, a publish is in
    ///     progress — retry. Otherwise read both fields, then
    ///     load `seq_after`; accept the pair iff
    ///     `seq_before == seq_after` (which implicitly requires
    ///     `seq_after` is also even). Initial value is 0 (even,
    ///     no publish in progress).
    ///
    /// A "bump-after-only" counter is INSUFFICIENT: a reader can
    /// load `seq_before = N`, observe the writer's new oldest
    /// before the writer stores the new visible (or before the
    /// seq bump), then load `seq_after = N` and accept an
    /// old-visible / future-oldest pair.
    pub publish_seq: AtomicU64,
    /// PID of the process that last successfully completed
    /// `create_writer` + `mark_ready`. Reader's pre-acquire
    /// handshake compares this against `db.lock`'s PID to prove
    /// the READY shm belongs to the CURRENT writer incarnation —
    /// not the prior writer's leftover values that a new writer's
    /// LOCK_EX-acquire-without-yet-clearing window would otherwise
    /// expose. PID stored as `u64` to share the natural `AtomicU64`
    /// alignment of the rest of the header. `0` until set by
    /// `mark_ready`, so any reader observing
    /// `init_done = READY && writer_pid = 0` rejects (init must
    /// have crashed before mark_ready).
    pub writer_pid: AtomicU64,
    /// Reserved for future additions without a format bump. Writers
    /// zero this at init; readers ignore.
    pub _reserved: [u8; SHM_SIZE - 104],
}

const _: () = {
    // Compile-time size assertion. If this fails, someone added/removed
    // a field above and the _reserved size must be adjusted.
    assert!(std::mem::size_of::<ShmHeader>() == SHM_SIZE);
};

impl ShmHeader {
    /// Maximum spins before we give up and return the conservative
    /// fallback. 8 covers any reasonable contention pattern (the
    /// writer's publish path is ~3 atomic stores). Chosen small
    /// enough that a runaway writer can't stall us indefinitely.
    const SAMPLE_MAX_RETRIES: u32 = 8;

    /// Coherent snapshot of `(visible_commit_lsn, oldest_active_txn_lsn)`.
    ///
    /// Returns `Some((visible, oldest))` when the writer was
    /// quiescent across the sample (no in-progress publish, no
    /// publish completed between the two seq reads). Returns
    /// `None` when the writer kept publishing during every retry
    /// — caller should use a conservative fallback (`oldest = 0`
    /// = "scan everything").
    ///
    /// Standard seqlock read:
    ///   1. `seq_before = seq.load(Acquire)` — must be EVEN. If
    ///      odd, a writer publish is in progress (writer bumped to
    ///      odd before the field stores) — retry.
    ///   2. Read both fields with Acquire (paired with the
    ///      writer's Release stores).
    ///   3. `seq_after = seq.load(Acquire)` — must equal
    ///      `seq_before`. If different, a publish completed
    ///      between our two seq reads — our field reads may
    ///      straddle the publish — retry.
    pub fn sample_visibility_pair(&self) -> Option<(u64, u64)> {
        for _ in 0..Self::SAMPLE_MAX_RETRIES {
            let seq_before = self.publish_seq.load(Ordering::Acquire);
            if seq_before & 1 == 1 {
                // Publish in progress — fields are mid-mutation.
                std::hint::spin_loop();
                continue;
            }
            let visible = self.visible_commit_lsn.load(Ordering::Acquire);
            let oldest = self.oldest_active_txn_lsn.load(Ordering::Acquire);
            let seq_after = self.publish_seq.load(Ordering::Acquire);
            if seq_before == seq_after {
                return Some((visible, oldest));
            }
            std::hint::spin_loop();
        }
        None
    }

    /// Stable-header attach snapshot. Returns
    /// `Some((writer_generation, visible_commit_lsn,
    /// oldest_active_txn_lsn))` only when all three were sampled
    /// against the SAME writer incarnation AND `init_done` was
    /// `MAGIC_READY` both before and after the sample. Returns
    /// `None` if the writer is mid-reincarnation across all
    /// retries (caller must refuse the attach rather than seed
    /// state from a half-initialized header).
    ///
    /// Closes the fresh-attach race that `open_reader`'s one-shot
    /// `init_done` check alone cannot:
    ///   - `open_reader` sees `MAGIC_READY` from the prior writer
    ///     incarnation.
    ///   - The prior writer crashes; the new writer clears
    ///     `init_done`, bumps `writer_generation`, wipes
    ///     visibility (per `create_writer`'s ordered reinit).
    ///   - A naive single-load snapshot would pair the new
    ///     `writer_generation` with `visible_commit_lsn = 0`,
    ///     seed `expected_writer_gen` to the new value, and
    ///     never surface `SwmrWriterReincarnated` afterwards.
    ///
    /// Protocol per iteration:
    ///   1. Verify `init_done == MAGIC_READY`. If not, the writer
    ///      is mid-init or hasn't completed recovery — retry.
    ///   2. Load `writer_generation` (gen_before).
    ///   3. `sample_visibility_pair()` for coherent (visible,
    ///      oldest).
    ///   4. Re-load `writer_generation` (gen_after) and re-verify
    ///      `init_done == MAGIC_READY`.
    ///   5. Accept iff `gen_before == gen_after` AND init_done
    ///      stayed READY across the sample.
    pub fn sample_attach_snapshot(&self) -> Option<(u64, u64, u64)> {
        for _ in 0..Self::SAMPLE_MAX_RETRIES {
            if self.init_done.load(Ordering::Acquire) != SHM_INIT_DONE_MAGIC {
                std::hint::spin_loop();
                continue;
            }
            let gen_before = self.writer_generation.load(Ordering::Acquire);
            let (visible, oldest) = match self.sample_visibility_pair() {
                Some(p) => p,
                None => {
                    std::hint::spin_loop();
                    continue;
                }
            };
            let gen_after = self.writer_generation.load(Ordering::Acquire);
            if gen_before != gen_after {
                std::hint::spin_loop();
                continue;
            }
            if self.init_done.load(Ordering::Acquire) != SHM_INIT_DONE_MAGIC {
                std::hint::spin_loop();
                continue;
            }
            return Some((gen_before, visible, oldest));
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Platform-specific handle
// ---------------------------------------------------------------------------

#[cfg(unix)]
pub use unix::ShmHandle;

#[cfg(not(unix))]
pub use stub::ShmHandle;

// ---------------------------------------------------------------------------
// Unix: real mmap-backed implementation
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod unix {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::os::fd::AsRawFd;

    /// Owns the memory mapping backing `db.shm`. Drop unmaps.
    pub struct ShmHandle {
        /// Kept open so the mapping stays valid. Not directly used
        /// after `mmap` succeeds.
        _fd: File,
        ptr: *mut u8,
        writable: bool,
    }

    // SAFETY: `ShmHandle` wraps an `mmap` region. The region itself is
    // `Send`/`Sync` because all access is via atomic ops on naturally
    // aligned fields; the raw pointer is just a base for those ops.
    unsafe impl Send for ShmHandle {}
    unsafe impl Sync for ShmHandle {}

    impl ShmHandle {
        /// Create (or truncate) `<db>/db.shm`, `mmap` it read-write,
        /// zero it, write magic + version, then publish
        /// `SHM_INIT_DONE_MAGIC` as the final atomic store so readers
        /// can safely attach.
        ///
        /// Must be called by the writer only, ideally while holding
        /// `writer.lock` so two writers don't race the init.
        pub fn create_writer(db_path: &Path) -> Result<Self> {
            std::fs::create_dir_all(db_path).map_err(|e| {
                Error::internal(format!(
                    "failed to create db dir for shm '{}': {}",
                    db_path.display(),
                    e
                ))
            })?;
            let path = db_path.join(SHM_FILENAME);

            // SWMR v2 Phase I: writer_generation must monotonically
            // advance across writer incarnations so reader processes
            // can detect a writer crash + restart via a generation
            // bump (Error::SwmrWriterReincarnated). Read the prior
            // value from the existing file BEFORE truncating, so the
            // bumped value below picks up where the previous writer
            // left off rather than resetting to 1.
            //
            // Pre-read failures (file missing, too small, bad magic)
            // are silent — those are first-open or corrupted cases
            // and starting at 0 is correct.
            let prior_gen = std::fs::File::open(&path)
                .ok()
                .and_then(|f| {
                    use std::io::{Read as _, Seek as _, SeekFrom};
                    let mut f = f;
                    // writer_generation lives at offset 16 in the
                    // header (after magic[4] + version[4] + init_done[8]).
                    let mut buf = [0u8; 8];
                    f.seek(SeekFrom::Start(16)).ok()?;
                    f.read_exact(&mut buf).ok()?;
                    Some(u64::from_le_bytes(buf))
                })
                .unwrap_or(0);

            // Do NOT use truncate(true). Existing
            // reader processes may still hold an mmap of this file;
            // shrinking it (even momentarily, before set_len
            // re-extends) makes any access in the truncated region
            // SIGBUS instead of cleanly surfacing
            // SwmrWriterReincarnated. Open without truncate, then
            // grow ONLY when the file is smaller than SHM_SIZE.
            // We zero the contents in-place via the mmap below; the
            // file size never shrinks, so live reader mmaps stay
            // valid. Their writer_generation poll picks up the
            // bumped value and triggers SwmrWriterReincarnated as
            // designed.
            let fd = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|e| {
                    Error::internal(format!("failed to open shm '{}': {}", path.display(), e))
                })?;
            let current_size = fd.metadata().map(|m| m.len()).unwrap_or(0);
            if current_size < SHM_SIZE as u64 {
                fd.set_len(SHM_SIZE as u64).map_err(|e| {
                    Error::internal(format!(
                        "failed to size shm '{}' to {}: {}",
                        path.display(),
                        SHM_SIZE,
                        e
                    ))
                })?;
            }
            // (We deliberately do NOT shrink if current_size > SHM_SIZE.
            // That shouldn't happen for a stoolap-managed db.shm; if
            // it does — e.g., a future SHM_VERSION bump that grows
            // the layout was once written here — zero-init will only
            // touch the first SHM_SIZE bytes, the trailing bytes are
            // ignored by readers, and existing mmaps stay safe.)
            // SAFETY: `fd` is a valid file descriptor of length `SHM_SIZE`
            // set just above. We request `PROT_READ | PROT_WRITE` and
            // `MAP_SHARED` so writes propagate to other mmappers. `ptr`
            // is stored in the returned struct and unmapped on Drop.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    SHM_SIZE,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(Error::internal(format!(
                    "mmap failed for shm '{}': {}",
                    path.display(),
                    std::io::Error::last_os_error()
                )));
            }
            let handle = Self {
                _fd: fd,
                ptr: ptr as *mut u8,
                writable: true,
            };
            // SWMR v2 Phase I: three-step ordered reinit. The order
            // is critical for both new and existing readers:
            //
            //   STEP 1: clear init_done. This blocks ALL new
            //   open_reader attaches: they validate init_done ==
            //   MAGIC_READY before snapshotting any field. Without
            //   this first, a fresh attach could see prior-
            //   incarnation MAGIC_READY, snapshot the about-to-be-
            //   bumped writer_generation as `expected`, then
            //   observe wiped visibility on its first refresh —
            //   the gen recheck would still match (we just stored
            //   it), so the reader would silently use 0.
            //
            //   STEP 2: store new writer_generation. Existing
            //   readers' next refresh observes mismatch and fires
            //   SwmrWriterReincarnated. Done in a SINGLE atomic
            //   store (not restore-then-bump) so the field never
            //   transits through 0 or `prior_gen`.
            //
            //   STEP 3: wipe everything AFTER writer_generation
            //   (offset 24..end). Magic, version (offset 0..8) are
            //   NOT in the wiped range — they were already correct
            //   for a reincarnation; for a fresh file they're 0
            //   from mmap zero-init and re-set unconditionally
            //   below. init_done (offset 8..16) was cleared in
            //   step 1; writer_generation (offset 16..24) holds
            //   the bumped value from step 2.
            //
            // Crash safety across the steps:
            //   - Crash before step 1: prior state intact; next
            //     writer reads `prior_gen` from disk and bumps to
            //     `prior_gen + 1`. Existing readers fire on
            //     mismatch.
            //   - Crash between steps 1 and 2: init_done = 0 on
            //     disk; new attaches refuse. Existing readers
            //     still see prior gen (matches) but visibility
            //     state is intact (only init_done was cleared) —
            //     reads remain consistent until next refresh,
            //     which the next writer will trigger via its own
            //     gen bump.
            //   - Crash after step 2: writer_generation on disk =
            //     `prior_gen + 1`. Next writer's prior_gen is now
            //     `prior_gen + 1`, so it bumps to `prior_gen + 2`
            //     — strictly above any prior reader's expected.
            handle.header().init_done.store(0, Ordering::Release);
            handle
                .header()
                .writer_generation
                .store(prior_gen.saturating_add(1), Ordering::Release);
            // Wipe offset 24..end (everything after writer_generation).
            // Static asserts pin the layout so a future field
            // reorder fails the build instead of silently wiping
            // generation or init_done.
            const POST_GEN_OFFSET: usize = std::mem::offset_of!(ShmHeader, visible_commit_lsn);
            const _: () = assert!(POST_GEN_OFFSET == 24);
            const _: () = assert!(std::mem::offset_of!(ShmHeader, init_done) == 8);
            const _: () = assert!(std::mem::offset_of!(ShmHeader, writer_generation) == 16);
            // SAFETY: handle.ptr points to SHM_SIZE writable bytes
            // we own. POST_GEN_OFFSET == 24 (asserted above), so the
            // wipe spans from end-of-writer_generation to end-of-
            // region without touching init_done or writer_generation.
            unsafe {
                std::ptr::write_bytes(
                    handle.ptr.add(POST_GEN_OFFSET),
                    0,
                    SHM_SIZE - POST_GEN_OFFSET,
                );
            }
            // Write magic + version. For a fresh file (mmap zero-
            // init) these were 0 and need setting; for a
            // reincarnation they were already correct and we
            // deliberately did NOT wipe offset 0..8. Writing them
            // unconditionally handles both cases.
            // SAFETY: ptr is aligned to u64, so the u32 writes at
            // the start of the struct are aligned too.
            unsafe {
                let hdr = handle.ptr as *mut ShmHeader;
                std::ptr::addr_of_mut!((*hdr).magic).write(SHM_MAGIC);
                std::ptr::addr_of_mut!((*hdr).version).write(SHM_VERSION);
            }
            // Do NOT publish `init_done = MAGIC_READY` here. WAL
            // recovery + post-recovery seal still have to run before
            // the writer's `visible_commit_lsn` reflects the recovered
            // frontier; if we marked the shm ready now, a reader
            // attaching in that window would see `visible_commit_lsn = 0`
            // and either (a) cap WAL replay to 0 and skip everything
            // recovery is about to apply, or (b) tail with a stale
            // attach snapshot. The engine calls `mark_ready()` AFTER
            // recovery; until then `open_reader` errors and the reader
            // falls back to v1 mtime-only snapshot mode, which is
            // independent of the writer's recovery state.
            Ok(handle)
        }

        /// Publish `init_done = MAGIC_READY` so reader processes are
        /// allowed to attach via `open_reader`. Called by the engine
        /// AFTER WAL recovery + post-recovery seal complete and
        /// `visible_commit_lsn` reflects the recovered frontier.
        ///
        /// Release-ordered so any reader that observes MAGIC_READY
        /// also observes every prior atomic store the writer made
        /// during recovery (visible_commit_lsn, oldest_active_txn_lsn,
        /// writer_generation).
        pub fn mark_ready(&self) {
            // Stamp this process's PID BEFORE flipping init_done
            // to MAGIC_READY. Reader's pre-acquire handshake reads
            // db.lock's PID and shm.writer_pid; if they match, the
            // READY shm belongs to the current LOCK_EX holder, not
            // a prior writer's leftover state. Storing the PID
            // first means any reader that observes init_done =
            // READY also observes the matching writer_pid via the
            // Acquire/Release pairing on init_done.
            self.header()
                .writer_pid
                .store(std::process::id() as u64, Ordering::Release);
            self.header()
                .init_done
                .store(SHM_INIT_DONE_MAGIC, Ordering::Release);
        }

        /// Attach read-only to an existing `<db>/db.shm`. Fails if
        /// the file is missing, too small, or `init_done` doesn't
        /// match — all signal that no initialized writer has
        /// published a header yet (either never opened writable in
        /// this directory, or the writer crashed mid-init).
        pub fn open_reader(db_path: &Path) -> Result<Self> {
            let path = db_path.join(SHM_FILENAME);
            let fd = OpenOptions::new().read(true).open(&path).map_err(|e| {
                Error::internal(format!(
                    "failed to open shm '{}' as reader: {}",
                    path.display(),
                    e
                ))
            })?;
            let md = fd.metadata().map_err(|e| {
                Error::internal(format!("failed to stat shm '{}': {}", path.display(), e))
            })?;
            if (md.len() as usize) < SHM_SIZE {
                return Err(Error::internal(format!(
                    "shm '{}' is {} bytes; expected >= {}",
                    path.display(),
                    md.len(),
                    SHM_SIZE
                )));
            }
            // SAFETY: fd is open read-only, file size >= SHM_SIZE,
            // `PROT_READ | MAP_SHARED` so we can observe the writer's
            // writes via the page cache.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    SHM_SIZE,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(Error::internal(format!(
                    "mmap read failed for shm '{}': {}",
                    path.display(),
                    std::io::Error::last_os_error()
                )));
            }
            let handle = Self {
                _fd: fd,
                ptr: ptr as *mut u8,
                writable: false,
            };
            // Validate magic + version (non-atomic reads are fine —
            // writer wrote them once before publishing init_done).
            // SAFETY: mmap region is readable for SHM_SIZE bytes.
            let (magic, version) = unsafe {
                let hdr = handle.ptr as *const ShmHeader;
                (
                    std::ptr::addr_of!((*hdr).magic).read(),
                    std::ptr::addr_of!((*hdr).version).read(),
                )
            };
            if magic != SHM_MAGIC {
                return Err(Error::internal(format!(
                    "shm '{}' bad magic: 0x{:08x} (expected 0x{:08x})",
                    path.display(),
                    magic,
                    SHM_MAGIC
                )));
            }
            if version != SHM_VERSION {
                return Err(Error::internal(format!(
                    "shm '{}' unsupported version {} (this build: {})",
                    path.display(),
                    version,
                    SHM_VERSION
                )));
            }
            // init_done is the last write by the writer; Acquire load
            // pairs with the writer's Release store to guarantee
            // visibility of earlier fields.
            let init_done = handle.header().init_done.load(Ordering::Acquire);
            if init_done != SHM_INIT_DONE_MAGIC {
                return Err(Error::internal(format!(
                    "shm '{}' init_done is 0x{:016x} (expected 0x{:016x}); writer \
                     may have crashed mid-init, or this is a stale file from an \
                     earlier version",
                    path.display(),
                    init_done,
                    SHM_INIT_DONE_MAGIC
                )));
            }
            Ok(handle)
        }

        /// Get an immutable reference to the header. Callers use the
        /// `AtomicU64` fields directly with their preferred ordering.
        pub fn header(&self) -> &ShmHeader {
            // SAFETY: ptr is SHM_SIZE bytes, aligned to at least 8,
            // and backs a live ShmHeader (written by the writer at
            // create time).
            unsafe { &*(self.ptr as *const ShmHeader) }
        }

        /// Whether this handle can write. Currently only used for tests
        /// and diagnostics; atomic stores against a read-only mapping
        /// would SIGBUS on most kernels, which is intentional.
        pub fn is_writable(&self) -> bool {
            self.writable
        }
    }

    impl Drop for ShmHandle {
        fn drop(&mut self) {
            // SAFETY: ptr came from libc::mmap with SHM_SIZE; munmap
            // is called exactly once (Drop runs once). Errors are
            // ignored — failing here means the kernel refused to
            // unmap, which doesn't happen on well-formed input.
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, SHM_SIZE);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Non-Unix: stub that errors — SWMR v2 requires Unix-native mmap.
// ---------------------------------------------------------------------------

#[cfg(not(unix))]
mod stub {
    use super::*;

    pub struct ShmHandle {
        _never: std::marker::PhantomData<()>,
    }

    impl ShmHandle {
        pub fn create_writer(_db_path: &Path) -> Result<Self> {
            Err(Error::internal(
                "SWMR v2 db.shm is Unix-only in this build; file a feature request \
                 for Windows support.",
            ))
        }

        pub fn open_reader(_db_path: &Path) -> Result<Self> {
            Err(Error::internal(
                "SWMR v2 db.shm is Unix-only in this build; file a feature request \
                 for Windows support.",
            ))
        }

        pub fn header(&self) -> &ShmHeader {
            unreachable!()
        }

        pub fn is_writable(&self) -> bool {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn tmp_db() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn create_writer_initializes_header_and_bumps_generation() {
        let dir = tmp_db();
        let h = ShmHandle::create_writer(dir.path()).unwrap();
        assert_eq!(h.header().magic, SHM_MAGIC);
        assert_eq!(h.header().version, SHM_VERSION);
        // init_done must stay 0 until the engine calls mark_ready();
        // open_reader uses this gate to refuse attaching while the
        // writer is still in WAL recovery.
        assert_eq!(h.header().init_done.load(Ordering::Acquire), 0);
        assert_eq!(h.header().visible_commit_lsn.load(Ordering::Acquire), 0);
        assert_eq!(h.header().manifest_epoch.load(Ordering::Acquire), 0);
        // writer_generation is bumped to `prior_gen + 1` IN this
        // call (single store, no restore-then-bump window). On a
        // fresh DB prior_gen = 0, so the published value is 1.
        assert_eq!(h.header().writer_generation.load(Ordering::Acquire), 1);
        assert!(h.is_writable());
        // After mark_ready, init_done flips to MAGIC.
        h.mark_ready();
        assert_eq!(
            h.header().init_done.load(Ordering::Acquire),
            SHM_INIT_DONE_MAGIC
        );
    }

    #[test]
    fn open_reader_blocks_until_writer_marks_ready() {
        let dir = tmp_db();
        let _w = ShmHandle::create_writer(dir.path()).unwrap();
        // Writer hasn't called mark_ready yet — reader must refuse.
        let err = ShmHandle::open_reader(dir.path());
        assert!(
            err.is_err(),
            "open_reader must refuse attach until mark_ready is called"
        );
        _w.mark_ready();
        let r = ShmHandle::open_reader(dir.path());
        assert!(r.is_ok(), "open_reader succeeds once writer marks ready");
    }

    #[test]
    fn writer_store_is_visible_to_reader_in_same_process() {
        let dir = tmp_db();
        let w = ShmHandle::create_writer(dir.path()).unwrap();
        w.header().visible_commit_lsn.store(42, Ordering::Release);
        w.header().writer_generation.store(7, Ordering::Release);
        w.mark_ready();

        let r = ShmHandle::open_reader(dir.path()).unwrap();
        assert_eq!(r.header().visible_commit_lsn.load(Ordering::Acquire), 42);
        assert_eq!(r.header().writer_generation.load(Ordering::Acquire), 7);
        assert!(!r.is_writable());
    }

    #[test]
    fn open_reader_fails_when_file_missing() {
        let dir = tmp_db();
        // No writer created yet — shm file missing.
        let err = ShmHandle::open_reader(dir.path());
        assert!(err.is_err(), "must error when db.shm missing");
    }

    #[test]
    fn open_reader_fails_when_file_too_small() {
        let dir = tmp_db();
        let path = dir.path().join(SHM_FILENAME);
        std::fs::write(&path, vec![0u8; 16]).unwrap();
        let err = ShmHandle::open_reader(dir.path());
        assert!(
            err.is_err(),
            "must error when db.shm is smaller than SHM_SIZE"
        );
    }

    #[test]
    fn open_reader_fails_when_init_done_missing() {
        // Simulate a writer that crashed before writing init_done:
        // create a file with magic + version but init_done = 0.
        let dir = tmp_db();
        let path = dir.path().join(SHM_FILENAME);
        let mut buf = vec![0u8; SHM_SIZE];
        buf[0..4].copy_from_slice(&SHM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&SHM_VERSION.to_le_bytes());
        // init_done bytes at offset 8 stay 0.
        std::fs::write(&path, buf).unwrap();
        let err = ShmHandle::open_reader(dir.path());
        assert!(
            err.is_err(),
            "must error when init_done != SHM_INIT_DONE_MAGIC"
        );
    }

    #[test]
    fn open_reader_fails_on_bad_magic() {
        let dir = tmp_db();
        let path = dir.path().join(SHM_FILENAME);
        let mut buf = vec![0u8; SHM_SIZE];
        buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        buf[4..8].copy_from_slice(&SHM_VERSION.to_le_bytes());
        buf[8..16].copy_from_slice(&SHM_INIT_DONE_MAGIC.to_le_bytes());
        std::fs::write(&path, buf).unwrap();
        let err = ShmHandle::open_reader(dir.path());
        assert!(err.is_err(), "must error on bad magic");
    }

    #[test]
    fn open_reader_fails_on_version_mismatch() {
        let dir = tmp_db();
        let path = dir.path().join(SHM_FILENAME);
        let mut buf = vec![0u8; SHM_SIZE];
        buf[0..4].copy_from_slice(&SHM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&(SHM_VERSION + 99).to_le_bytes());
        buf[8..16].copy_from_slice(&SHM_INIT_DONE_MAGIC.to_le_bytes());
        std::fs::write(&path, buf).unwrap();
        let err = ShmHandle::open_reader(dir.path());
        assert!(err.is_err(), "must error on version mismatch");
    }

    #[test]
    fn header_size_is_exactly_one_page() {
        assert_eq!(std::mem::size_of::<ShmHeader>(), SHM_SIZE);
    }

    #[test]
    fn oldest_active_txn_lsn_field_persists_across_reader_open() {
        // The watermark field must round-trip
        // through the mmap so reader processes can observe what
        // the writer stored. This is the cross-process equivalent
        // of "we wrote it here, did the reader see it there?"
        let dir = tmp_db();
        let w = ShmHandle::create_writer(dir.path()).unwrap();
        // Initially u64::MAX would be the "no active user txns"
        // sentinel; new shm starts at 0 (zeroed by create_writer).
        assert_eq!(
            w.header().oldest_active_txn_lsn.load(Ordering::Acquire),
            0,
            "fresh shm zeroes oldest_active_txn_lsn"
        );
        // Writer stores a low watermark (some long-running txn).
        w.header()
            .oldest_active_txn_lsn
            .store(12345, Ordering::Release);
        w.mark_ready();

        let r = ShmHandle::open_reader(dir.path()).unwrap();
        assert_eq!(
            r.header().oldest_active_txn_lsn.load(Ordering::Acquire),
            12345,
            "reader must observe writer's stored watermark"
        );
    }

    #[test]
    fn publish_order_release_acquire_pair_includes_watermark_before_visible() {
        // In the SHM layout, both
        // `oldest_active_txn_lsn` and `visible_commit_lsn` are
        // independent atomics. The writer's publish_visible_commit_lsn
        // stores them in this order: watermark FIRST (Release),
        // then visible_commit_lsn (Release/AcqRel). The reader's
        // refresh path reads visible_commit_lsn FIRST (Acquire),
        // then watermark (Acquire). This test simulates that
        // ordering at the per-store level: when we set both with
        // the writer's order, a reader that observes the new
        // visible_commit_lsn always observes the matching (or
        // lower) watermark — never a stale higher one.
        let dir = tmp_db();
        let w = ShmHandle::create_writer(dir.path()).unwrap();

        // Initial state: both 0.
        w.header().oldest_active_txn_lsn.store(0, Ordering::Release);
        w.header().visible_commit_lsn.store(100, Ordering::Release);

        // Writer about to publish a new commit. New watermark
        // reflects an in-flight txn at LSN 50 (lower than the
        // PREVIOUS visible LSN of 100). The new visible LSN is
        // 200. Per the publish contract, watermark goes FIRST.
        w.header()
            .oldest_active_txn_lsn
            .store(50, Ordering::Release);
        w.header().visible_commit_lsn.store(200, Ordering::Release);
        w.mark_ready();

        // Reader sees the new visible LSN. Acquire-Release
        // pairing: any Acquire load of visible_commit_lsn that
        // returns 200 happens-after the writer's Release store of
        // visible_commit_lsn, which happens-after the prior Release
        // store of oldest_active_txn_lsn. So a subsequent
        // Acquire load of oldest_active_txn_lsn must return 50,
        // never the stale 0.
        let r = ShmHandle::open_reader(dir.path()).unwrap();
        let visible = r.header().visible_commit_lsn.load(Ordering::Acquire);
        assert_eq!(visible, 200);
        let watermark = r.header().oldest_active_txn_lsn.load(Ordering::Acquire);
        assert_eq!(
            watermark, 50,
            "reader observing new visible LSN MUST observe matching watermark, \
             not stale 0"
        );
    }
}
