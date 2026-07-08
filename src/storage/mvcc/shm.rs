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

//! Cross-process shared header for SWMR coordination at `<db>/db.shm`.
//!
//! Writer `mmap`s read-write, readers read-only; both use `MAP_SHARED`.
//! Naturally-aligned `AtomicU64` fields give hardware-atomic loads/stores
//! on x86_64/aarch64. `init_done = MAGIC_READY` is published LAST by the
//! writer so half-init / stale files are detectable. Unix only.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::{Error, Result};

/// Sentinel written last by the writer during init; readers refuse any
/// attach whose `init_done` doesn't match.
pub const SHM_INIT_DONE_MAGIC: u64 = 0x5245414459305057; // "READY0PW"
/// Magic identifying a stoolap `db.shm`.
pub const SHM_MAGIC: u32 = 0x535A4D32; // "SZM2" LE
/// Current header version.
pub const SHM_VERSION: u32 = 1;

/// Total size of `db.shm`. One page; trailing bytes reserved for future fields.
pub const SHM_SIZE: usize = 4096;

/// Filename inside the database directory.
pub const SHM_FILENAME: &str = "db.shm";

/// Shared memory header. All multi-byte fields are little-endian.
#[repr(C, align(8))]
pub struct ShmHeader {
    /// Written once by writer at create time; readers validate before atomic access.
    pub magic: u32,
    /// Format version (`SHM_VERSION`).
    pub version: u32,
    /// Set to `SHM_INIT_DONE_MAGIC` by the writer as its LAST init step.
    pub init_done: AtomicU64,
    /// Bumped on every writer startup; reader mismatch means writer crashed and recovered.
    pub writer_generation: AtomicU64,
    /// Visibility watermark: rows whose commit_seq <= this are visible.
    /// Published AFTER WAL fsync and `registry.complete_commit`.
    pub visible_commit_lsn: AtomicU64,
    /// Bumped after each checkpoint completes; readers poll to invalidate manifest cache.
    pub manifest_epoch: AtomicU64,
    /// Highest LSN of any appended DDL WAL entry. SWMR readers skip the
    /// tail scan when this hasn't moved past their last_applied_lsn.
    pub catalog_epoch: AtomicU64,
    /// Bumps on column rename/drop/add; readers use it to invalidate prepared-statement cache.
    pub schema_generation: AtomicU64,
    /// First live WAL file id. Readers below this get `SnapshotExpired`.
    pub wal_chain_head: AtomicU64,
    /// Current (latest) WAL file id.
    pub wal_current: AtomicU64,
    /// Minimum `pinned_lsn` across live reader leases.
    pub min_pinned_lsn: AtomicU64,
    /// LSN floor for reader WAL-tail scans: lowest LSN of any active user txn's
    /// first DML. `u64::MAX` = no active user txns, `0` = unpublished.
    pub oldest_active_txn_lsn: AtomicU64,
    /// Seqlock for coherent (visible_commit_lsn, oldest_active_txn_lsn) snapshots.
    /// Writer bumps odd before stores, even after; reader retries while odd or
    /// when seq_before != seq_after. Bump-after-only is INSUFFICIENT.
    pub publish_seq: AtomicU64,
    /// PID of the writer that last completed `mark_ready`. Reader handshake
    /// compares against `db.lock`'s PID to detect prior-incarnation leftovers.
    /// Stored when init_done flips to READY; `0` rejects the attach.
    pub writer_pid: AtomicU64,
    /// Reserved for future additions; writers zero at init, readers ignore.
    pub _reserved: [u8; SHM_SIZE - 104],
}

const _: () = {
    assert!(std::mem::size_of::<ShmHeader>() == SHM_SIZE);
};

impl ShmHeader {
    /// Spin retry cap; small enough that a runaway writer can't stall us.
    const SAMPLE_MAX_RETRIES: u32 = 8;

    /// Coherent seqlock snapshot of `(visible_commit_lsn, oldest_active_txn_lsn)`.
    /// Returns `None` when the writer kept publishing across all retries;
    /// caller should fall back to `oldest = 0`.
    pub fn sample_visibility_pair(&self) -> Option<(u64, u64)> {
        for _ in 0..Self::SAMPLE_MAX_RETRIES {
            let seq_before = self.publish_seq.load(Ordering::Acquire);
            if seq_before & 1 == 1 {
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

    /// Stable attach snapshot of `(writer_generation, visible, oldest)` sampled
    /// against the SAME writer incarnation with `init_done == MAGIC_READY`
    /// before and after. Returns `None` if mid-reincarnation across all retries,
    /// caller must refuse the attach rather than seed from a half-init header.
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
        /// Kept open so the mapping stays valid.
        _fd: File,
        ptr: *mut u8,
        writable: bool,
    }

    // SAFETY: all access is via atomic ops on naturally-aligned fields;
    // the raw pointer is just a base for those ops.
    unsafe impl Send for ShmHandle {}
    unsafe impl Sync for ShmHandle {}

    impl ShmHandle {
        /// Create or open `<db>/db.shm`, mmap rw, perform ordered reinit
        /// (clear init_done, bump writer_generation, wipe post-gen fields,
        /// set magic+version). Does NOT mark ready; the engine calls
        /// `mark_ready()` after WAL recovery completes. Caller should hold
        /// `writer.lock` to serialize against other writers.
        pub fn create_writer(db_path: &Path) -> Result<Self> {
            std::fs::create_dir_all(db_path).map_err(|e| {
                Error::internal(format!(
                    "failed to create db dir for shm '{}': {}",
                    db_path.display(),
                    e
                ))
            })?;
            let path = db_path.join(SHM_FILENAME);

            // Read prior writer_generation from offset 16 BEFORE truncating, so
            // the bumped value advances monotonically across writer incarnations
            // (readers detect crash+restart via mismatch).
            let prior_gen = std::fs::File::open(&path)
                .ok()
                .and_then(|f| {
                    use std::io::{Read as _, Seek as _, SeekFrom};
                    let mut f = f;
                    let mut buf = [0u8; 8];
                    f.seek(SeekFrom::Start(16)).ok()?;
                    f.read_exact(&mut buf).ok()?;
                    Some(u64::from_le_bytes(buf))
                })
                .unwrap_or(0);

            // Never truncate: shrinking the file SIGBUSes existing reader mmaps.
            // Grow only when smaller than SHM_SIZE; zero in-place via mmap below.
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
            // SAFETY: fd is a valid file descriptor of length >= SHM_SIZE.
            // PROT_READ|PROT_WRITE + MAP_SHARED so writes propagate.
            // ptr is unmapped on Drop.
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
            // Ordered reinit: clear init_done, bump writer_generation in a single
            // store, wipe post-gen fields. Order blocks fresh attaches from
            // pairing a new gen with a stale visibility, and crashes between
            // steps remain recoverable (init_done=0 refuses new attaches).
            handle.header().init_done.store(0, Ordering::Release);
            handle
                .header()
                .writer_generation
                .store(prior_gen.saturating_add(1), Ordering::Release);
            // Static asserts pin layout so a field reorder fails the build.
            const POST_GEN_OFFSET: usize = std::mem::offset_of!(ShmHeader, visible_commit_lsn);
            const _: () = assert!(POST_GEN_OFFSET == 24);
            const _: () = assert!(std::mem::offset_of!(ShmHeader, init_done) == 8);
            const _: () = assert!(std::mem::offset_of!(ShmHeader, writer_generation) == 16);
            // SAFETY: handle.ptr owns SHM_SIZE writable bytes; the wipe spans
            // post-gen..end and does not touch init_done or writer_generation.
            unsafe {
                std::ptr::write_bytes(
                    handle.ptr.add(POST_GEN_OFFSET),
                    0,
                    SHM_SIZE - POST_GEN_OFFSET,
                );
            }
            // SAFETY: ptr is u64-aligned, so the leading u32 writes are aligned.
            unsafe {
                let hdr = handle.ptr as *mut ShmHeader;
                std::ptr::addr_of_mut!((*hdr).magic).write(SHM_MAGIC);
                std::ptr::addr_of_mut!((*hdr).version).write(SHM_VERSION);
            }
            // init_done stays 0 here. The engine calls mark_ready() after WAL
            // recovery so attaching readers don't see visible_commit_lsn=0.
            Ok(handle)
        }

        /// Publish `init_done = MAGIC_READY` so readers may attach.
        /// Called by the engine after WAL recovery completes. PID is
        /// stamped first so the Acquire-pair on init_done guarantees
        /// readers see a matching writer_pid.
        pub fn mark_ready(&self) {
            self.header()
                .writer_pid
                .store(std::process::id() as u64, Ordering::Release);
            self.header()
                .init_done
                .store(SHM_INIT_DONE_MAGIC, Ordering::Release);
        }

        /// Attach read-only. Fails when missing, too small, or init_done
        /// doesn't match (no ready writer in this directory).
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
            // PROT_READ | MAP_SHARED so writer stores are visible.
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
            // Acquire pairs with the writer's Release on init_done, a successful
            // load means earlier field stores are visible too.
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

        /// Get an immutable reference to the header.
        pub fn header(&self) -> &ShmHeader {
            // SAFETY: ptr is SHM_SIZE bytes, 8-aligned, backed by a live ShmHeader.
            unsafe { &*(self.ptr as *const ShmHeader) }
        }

        /// Whether this handle can write. Stores against a read-only mapping SIGBUS.
        pub fn is_writable(&self) -> bool {
            self.writable
        }
    }

    impl Drop for ShmHandle {
        fn drop(&mut self) {
            // SAFETY: ptr came from libc::mmap with SHM_SIZE; munmap runs once.
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, SHM_SIZE);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Non-Unix: stub that errors, SWMR `db.shm` requires Unix-native mmap.
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
                "SWMR db.shm is Unix-only in this build; file a feature request \
                 for Windows support.",
            ))
        }

        pub fn open_reader(_db_path: &Path) -> Result<Self> {
            Err(Error::internal(
                "SWMR db.shm is Unix-only in this build; file a feature request \
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
        // init_done stays 0 until mark_ready; gates open_reader during recovery.
        assert_eq!(h.header().init_done.load(Ordering::Acquire), 0);
        assert_eq!(h.header().visible_commit_lsn.load(Ordering::Acquire), 0);
        assert_eq!(h.header().manifest_epoch.load(Ordering::Acquire), 0);
        // Fresh DB: prior_gen=0, single-store bump publishes 1.
        assert_eq!(h.header().writer_generation.load(Ordering::Acquire), 1);
        assert!(h.is_writable());
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
        let err = ShmHandle::open_reader(dir.path());
        assert!(err.is_err(), "must refuse attach until mark_ready");
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
        // Simulates writer that crashed before writing init_done.
        let dir = tmp_db();
        let path = dir.path().join(SHM_FILENAME);
        let mut buf = vec![0u8; SHM_SIZE];
        buf[0..4].copy_from_slice(&SHM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&SHM_VERSION.to_le_bytes());
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
        let dir = tmp_db();
        let w = ShmHandle::create_writer(dir.path()).unwrap();
        assert_eq!(
            w.header().oldest_active_txn_lsn.load(Ordering::Acquire),
            0,
            "fresh shm zeroes oldest_active_txn_lsn"
        );
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
        // Writer stores watermark FIRST then visible_commit_lsn (both Release).
        // Reader's Acquire load of visible_commit_lsn that observes the new value
        // must observe the matching (or lower) watermark, never a stale higher one.
        let dir = tmp_db();
        let w = ShmHandle::create_writer(dir.path()).unwrap();

        w.header().oldest_active_txn_lsn.store(0, Ordering::Release);
        w.header().visible_commit_lsn.store(100, Ordering::Release);

        w.header()
            .oldest_active_txn_lsn
            .store(50, Ordering::Release);
        w.header().visible_commit_lsn.store(200, Ordering::Release);
        w.mark_ready();

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
