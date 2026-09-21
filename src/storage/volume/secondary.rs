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

//! A paged secondary index over one integer column of a sealed volume.
//!
//! The side file holds, per indexed column, the distinct keys in order and
//! for each key the ascending row positions that hold it, in pages bounded
//! in bytes: a key page carries keys with the position index each key ends
//! at, a position page carries positions. A small directory names the pages
//! and stays resident with the volume; pages are read on demand into a
//! process-wide cache with a byte budget.
//!
//! Every allocation the index makes is charged to one ledger before it is
//! made, at the capacity it takes, and released when its last owner lets
//! go: a directory for its file's life, a page for as long as any holder
//! keeps it, the raw bytes of a page or a directory while they are parsed,
//! a build's sort run, page buffers, output buffer and merge readers, and a
//! cursor's window. The ledger is accounting with best-effort eviction: a
//! reservation that finds nothing unheld to evict is still granted and
//! counted as over budget. Admission, the rule that refuses or delays an
//! allocation the budget cannot take, is not here; it is a prerequisite of
//! the query integration and is decided there.
//!
//! Layout of `vol_<id>.sidx`:
//!
//! ```text
//! [magic "STSX"][version u32][generation u64]        16 bytes
//! pages, each [n u32][entries][crc32 u32]
//! directory: [column count u32] then per column
//!   [col u32][key tag u8][n_keys u64][n_positions u64][key pages u32][pos pages u32]
//!   key pages: [first_key i64][last_key i64][offset u64][len u32][key_start u64][pos_start u64]
//!   pos pages: [offset u64][len u32][pos_start u64]
//! footer: [directory offset u64][directory len u32][directory crc32 u32][magic]
//! ```
//!
//! The generation in the header is immutable for the file; a page read
//! checks it, and a cache key carries it, so a file replaced under a
//! reader never mixes the reader's directory with the new file's pages.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use super::writer::VolumeFile;
const MAGIC: [u8; 4] = *b"STSX";
const VERSION: u32 = 4;
const KEY_I64: u8 = 1;
const HEADER_LEN: u64 = 16;
const FOOTER_LEN: u64 = 20;
const KEY_DIR_ENTRY: usize = 44;
const POS_DIR_ENTRY: usize = 20;
const COLUMN_ENTRY: usize = 37;
pub const SIDE_EXT: &str = "sidx";

/// The byte bound of one page, header and checksum included.
pub const PAGE_BYTES: usize = 32 * 1024;
const PAGE_OVERHEAD: usize = 8;
/// A key and the position index it ends at, which fits u32 since a
/// volume's positions do
const KEY_ENTRY: usize = 12;
const POS_ENTRY: usize = 4;
/// Keys one key page holds at most, and positions one position page holds
pub const KEYS_PER_PAGE: usize = (PAGE_BYTES - PAGE_OVERHEAD) / KEY_ENTRY;
pub const POSITIONS_PER_PAGE: usize = (PAGE_BYTES - PAGE_OVERHEAD) / POS_ENTRY;

/// The side file next to a volume file.
pub fn side_path(volume_path: &Path) -> PathBuf {
    volume_path.with_extension(SIDE_EXT)
}

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(0);

/// A generation no earlier side file of this process carries, and none of
/// an earlier process with a probability the clock gives.
pub fn next_generation() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    NEXT_GENERATION
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |last| {
            Some(now.max(last + 1))
        })
        .map(|last| now.max(last + 1))
        .unwrap_or(now)
}

fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("side index: {msg}"),
    )
}

/// The error a refused reservation gives; `is_refused` names it. A caller
/// answers it with its fallback, not as a failure.
fn refused(what: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::WouldBlock,
        format!("side index: the budget refused {what}"),
    )
}

/// Whether an error is a budget's refusal
pub fn is_refused(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::WouldBlock
}

// =============================================================================
// Accounting: three ledgers, pages, builds and directories
// =============================================================================

/// A process-wide ledger of bytes: atomics only, charged before the bytes
/// are allocated and released when their owner drops.
pub struct Ledger {
    charged: AtomicUsize,
    budget: AtomicU64,
    /// The most bytes charged at once since the last reset
    peak: AtomicUsize,
    refused: AtomicU64,
}

/// A ledger's state, for `PRAGMA MEMORY_STATS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerStats {
    pub budget_bytes: u64,
    pub charged_bytes: usize,
    pub peak_bytes: usize,
    /// Charges refused by `try_charge`
    pub refused: u64,
}

impl Ledger {
    const fn new(budget: u64) -> Self {
        Self {
            charged: AtomicUsize::new(0),
            budget: AtomicU64::new(budget),
            peak: AtomicUsize::new(0),
            refused: AtomicU64::new(0),
        }
    }

    pub fn set_budget_bytes(&self, bytes: u64) {
        self.budget.store(bytes, Ordering::Release);
    }

    pub fn stats(&self) -> LedgerStats {
        LedgerStats {
            budget_bytes: self.budget.load(Ordering::Acquire),
            charged_bytes: self.charged.load(Ordering::Acquire),
            peak_bytes: self.peak.load(Ordering::Acquire),
            refused: self.refused.load(Ordering::Relaxed),
        }
    }

    /// Counts a refusal decided outside `try_charge`, by a share of the
    /// budget a caller keeps to
    pub fn count_refused(&self) {
        self.refused.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset_peak(&self) {
        self.peak
            .store(self.charged.load(Ordering::Acquire), Ordering::Release);
    }

    /// Charges `bytes` whatever the budget: the ledger reports them
    pub fn charge(&'static self, bytes: usize) -> Held {
        let after = self.charged.fetch_add(bytes, Ordering::AcqRel) + bytes;
        self.peak.fetch_max(after, Ordering::AcqRel);
        Held {
            ledger: self,
            bytes,
        }
    }

    /// Charges `bytes` only if the budget takes them, else counts the
    /// refusal; lowering the budget later never revokes what was granted
    pub fn try_charge(&'static self, bytes: usize) -> Option<Held> {
        let budget = self.budget.load(Ordering::Acquire);
        let mut current = self.charged.load(Ordering::Acquire);
        loop {
            if (current + bytes) as u64 > budget {
                self.refused.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            match self.charged.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(current + bytes, Ordering::AcqRel);
                    return Some(Held {
                        ledger: self,
                        bytes,
                    });
                }
                Err(now) => current = now,
            }
        }
    }
}

/// Bytes charged to a ledger, released on drop.
pub struct Held {
    ledger: &'static Ledger,
    bytes: usize,
}

impl Held {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.ledger.charged.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// The workspaces of the builds in flight: a build is admitted only when
/// its whole workspace fits beside the others', else refused
pub static INDEX_BUILDS: Ledger = Ledger::new(DEFAULT_BUILD_BUDGET_BYTES);

/// The resident directories of every open side file: reported, never
/// refused, since a directory is what makes a file usable at all
pub static INDEX_DIRECTORIES: Ledger = Ledger::new(0);

/// The process-wide ledger of page bytes and the page cache over it. The
/// ledger is atomics; the one lock guards the cache's map and order, and
/// nothing is dropped or reserved while it is held.
pub struct IndexPages {
    charged: AtomicUsize,
    over_budget: AtomicU64,
    budget: AtomicU64,
    cache: Mutex<PageCache>,
    loads: AtomicU64,
    hits: AtomicU64,
    evictions: AtomicU64,
    /// The most bytes charged at once since the last reset: a high-water
    /// mark taken at every charge, not a sample between calls
    peak: AtomicUsize,
    refused: AtomicU64,
}

#[derive(Default)]
struct PageCache {
    pages: HashMap<PageKey, Arc<Page>>,
    order: VecDeque<PageKey>,
}

const DEFAULT_BUDGET_BYTES: u64 = 16 * 1024 * 1024;
const DEFAULT_BUILD_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

pub static INDEX_PAGES: LazyLock<IndexPages> = LazyLock::new(|| IndexPages {
    charged: AtomicUsize::new(0),
    over_budget: AtomicU64::new(0),
    budget: AtomicU64::new(DEFAULT_BUDGET_BYTES),
    cache: Mutex::new(PageCache::default()),
    loads: AtomicU64::new(0),
    hits: AtomicU64::new(0),
    evictions: AtomicU64::new(0),
    peak: AtomicUsize::new(0),
    refused: AtomicU64::new(0),
});

/// A reservation of bytes in the ledger, released on drop. It grows when
/// its owner's allocation grows.
pub struct Reservation {
    bytes: usize,
}

impl Reservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Reserves `more` bytes on top, for an owner whose allocation grew
    pub fn grow(&mut self, more: usize) {
        if more == 0 {
            return;
        }
        INDEX_PAGES.charge(more);
        self.bytes += more;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        INDEX_PAGES.charged.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// The ledger's state, for `PRAGMA MEMORY_STATS` and the measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexStats {
    pub budget_bytes: u64,
    /// Every byte reserved right now: pages held or cached, directories,
    /// workspaces, raw bytes being parsed
    pub charged_bytes: usize,
    /// Bytes of pages the cache holds (a held page counts here too)
    pub cached_bytes: usize,
    pub cached_pages: usize,
    pub loads: u64,
    pub hits: u64,
    pub evictions: u64,
    /// Reservations granted while the ledger was over budget with nothing
    /// unheld left to evict
    pub over_budget: u64,
    /// The most bytes charged at once since `reset_peak`
    pub peak_bytes: usize,
    /// Reservations refused by `try_reserve`: the budget could not take
    /// them after every unheld page was evicted
    pub refused: u64,
}

impl IndexPages {
    pub fn set_budget_bytes(&self, bytes: u64) {
        self.budget.store(bytes, Ordering::Release);
        self.make_room(0);
    }

    pub fn stats(&self) -> IndexStats {
        let (cached_bytes, cached_pages) = {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            (
                cache.pages.values().map(|p| p.bytes).sum(),
                cache.pages.len(),
            )
        };
        IndexStats {
            budget_bytes: self.budget.load(Ordering::Acquire),
            charged_bytes: self.charged.load(Ordering::Acquire),
            cached_bytes,
            cached_pages,
            loads: self.loads.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            over_budget: self.over_budget.load(Ordering::Relaxed),
            peak_bytes: self.peak.load(Ordering::Acquire),
            refused: self.refused.load(Ordering::Relaxed),
        }
    }

    /// Starts the high-water mark again from what is charged now
    pub fn reset_peak(&self) {
        self.peak
            .store(self.charged.load(Ordering::Acquire), Ordering::Release);
    }

    /// Drops every cached page nobody holds; held pages stay charged.
    pub fn clear(&self) {
        let dropped: Vec<Arc<Page>> = {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            let PageCache { pages, order } = &mut *cache;
            let mut dropped = Vec::new();
            pages.retain(|_, page| {
                if Arc::strong_count(page) > 1 {
                    true
                } else {
                    dropped.push(Arc::clone(page));
                    false
                }
            });
            order.retain(|k| pages.contains_key(k));
            dropped
        };
        self.evictions
            .fetch_add(dropped.len() as u64, Ordering::Relaxed);
        // Dropped outside the lock: a page's drop releases its reservation
        drop(dropped);
    }

    /// Reserves `bytes` before they are allocated: unheld cached pages are
    /// evicted first while the ledger would go over budget; when nothing
    /// unheld is left the reservation is still granted and counted as over
    /// budget. This is accounting, not admission: nothing is refused here.
    pub fn reserve(&self, bytes: usize) -> Reservation {
        self.make_room(bytes);
        self.charge(bytes);
        Reservation { bytes }
    }

    /// Reserves `bytes` only if the budget takes them after every unheld
    /// page was evicted, else counts the refusal. The read paths reserve
    /// this way: a refused page or window sends the reader to its
    /// fallback, and lowering the budget never revokes a holder.
    pub fn try_reserve(&self, bytes: usize) -> Option<Reservation> {
        self.make_room(bytes);
        let budget = self.budget.load(Ordering::Acquire);
        let mut current = self.charged.load(Ordering::Acquire);
        loop {
            if (current + bytes) as u64 > budget {
                self.refused.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            match self.charged.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(current + bytes, Ordering::AcqRel);
                    return Some(Reservation { bytes });
                }
                Err(now) => current = now,
            }
        }
    }

    fn charge(&self, bytes: usize) {
        let after = self.charged.fetch_add(bytes, Ordering::AcqRel) + bytes;
        self.peak.fetch_max(after, Ordering::AcqRel);
        if after as u64 > self.budget.load(Ordering::Acquire) {
            self.over_budget.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether the pages named are cached, or would fit beside what is
    /// charged and a reservation of `reserve` bytes without an eviction:
    /// what a probe of them would cost the cache, before any read. A
    /// missing page counts at its parsed size, and the largest raw page
    /// on top, since a read holds the raw bytes while it parses them
    pub fn admits(
        &self,
        file: &IndexFile,
        column: usize,
        pages: impl Iterator<Item = (PageKind, usize)>,
        reserve: usize,
    ) -> std::io::Result<bool> {
        let mut parsed = 0usize;
        let mut raw = 0usize;
        let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        for (kind, number) in pages {
            let key = PageKey {
                file_id: file.file_id,
                generation: file.directory.generation,
                column: column as u32,
                kind,
                number: number as u32,
            };
            if !cache.pages.contains_key(&key) {
                let len = file.directory.page_location(column, kind, number)?.1 as usize;
                let entries = len.saturating_sub(PAGE_OVERHEAD);
                parsed += match kind {
                    PageKind::Keys => {
                        entries / KEY_ENTRY
                            * (std::mem::size_of::<i64>() + std::mem::size_of::<u64>())
                    }
                    PageKind::Positions => entries / POS_ENTRY * std::mem::size_of::<u32>(),
                };
                raw = raw.max(len);
            }
        }
        drop(cache);
        Ok(
            (self.charged.load(Ordering::Acquire) + parsed + raw + reserve) as u64
                <= self.budget.load(Ordering::Acquire),
        )
    }

    fn make_room(&self, incoming: usize) {
        let budget = self.budget.load(Ordering::Acquire);
        loop {
            if (self.charged.load(Ordering::Acquire) + incoming) as u64 <= budget {
                return;
            }
            // The oldest page nobody holds, dropped outside the lock
            let victim = {
                let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                let at = cache.order.iter().position(|k| {
                    cache
                        .pages
                        .get(k)
                        .is_some_and(|p| Arc::strong_count(p) == 1)
                });
                let Some(at) = at else {
                    return;
                };
                let key = cache
                    .order
                    .remove(at)
                    .expect("position came from the deque");
                cache.pages.remove(&key)
            };
            self.evictions.fetch_add(1, Ordering::Relaxed);
            drop(victim);
        }
    }

    /// The page `kind`/`number` of `column` in `file`, from the cache or
    /// read through the file's handle and checked against its generation.
    pub fn load(
        &self,
        file: &IndexFile,
        column: usize,
        kind: PageKind,
        number: usize,
    ) -> std::io::Result<Arc<Page>> {
        let key = PageKey {
            file_id: file.file_id,
            generation: file.directory.generation,
            column: column as u32,
            kind,
            number: number as u32,
        };
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(page) = cache.pages.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                let page = Arc::clone(page);
                if let Some(at) = cache.order.iter().position(|k| *k == key) {
                    cache.order.remove(at);
                    cache.order.push_back(key);
                }
                return Ok(page);
            }
        }
        let (offset, len) = file.directory.page_location(column, kind, number)?;
        let page = Arc::new(Page::read(file, kind, offset, len)?);
        self.loads.fetch_add(1, Ordering::Relaxed);
        let replaced = {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(existing) = cache.pages.get(&key) {
                // Another reader loaded it meanwhile; ours goes, and its
                // reservation with it, outside the lock
                Err(Arc::clone(existing))
            } else {
                cache.pages.insert(key, Arc::clone(&page));
                cache.order.push_back(key);
                Ok(())
            }
        };
        match replaced {
            Ok(()) => Ok(page),
            Err(existing) => {
                drop(page);
                Ok(existing)
            }
        }
    }
}

// =============================================================================
// Pages
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageKind {
    Keys,
    Positions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PageKey {
    file_id: u64,
    generation: u64,
    column: u32,
    kind: PageKind,
    number: u32,
}

/// One page as parsed from the file. The reservation it carries is the
/// capacity of its vectors, released when the last holder drops it.
pub struct Page {
    content: PageContent,
    bytes: usize,
    _reservation: Reservation,
}

pub enum PageContent {
    /// Keys in order, and for each the position index it ends at
    Keys {
        keys: Vec<i64>,
        ends: Vec<u64>,
    },
    Positions(Vec<u32>),
}

/// A page's entry count and entries once its length and checksum hold
fn verified_entries(raw: &[u8]) -> std::io::Result<(usize, &[u8])> {
    if raw.len() < PAGE_OVERHEAD {
        return Err(invalid("page shorter than its header"));
    }
    let body = &raw[..raw.len() - 4];
    let stored = u32::from_le_bytes(raw[raw.len() - 4..].try_into().expect("4 bytes"));
    if crc32fast::hash(body) != stored {
        return Err(invalid("page checksum mismatch"));
    }
    let n = u32::from_le_bytes(body[..4].try_into().expect("4 bytes")) as usize;
    Ok((n, &body[4..]))
}

impl Page {
    /// Reads and parses one page: the raw bytes are reserved for the read
    /// and released after the parse; the parsed vectors are reserved at
    /// their capacity before they are allocated and stay with the page.
    fn read(file: &IndexFile, kind: PageKind, offset: u64, len: u32) -> std::io::Result<Self> {
        let raw_reservation = INDEX_PAGES
            .try_reserve(len as usize)
            .ok_or_else(|| refused("a page's raw bytes"))?;
        let raw = file.read_page(offset, len)?;
        let (n, entries) = verified_entries(&raw)?;
        let (content, bytes, reservation) = match kind {
            PageKind::Keys => {
                if entries.len() != n * KEY_ENTRY {
                    return Err(invalid("key page length does not match its count"));
                }
                let bytes = n * (std::mem::size_of::<i64>() + std::mem::size_of::<u64>());
                let reservation = INDEX_PAGES
                    .try_reserve(bytes)
                    .ok_or_else(|| refused("a key page"))?;
                let mut keys = Vec::with_capacity(n);
                let mut ends = Vec::with_capacity(n);
                for entry in entries.as_chunks::<KEY_ENTRY>().0 {
                    keys.push(i64::from_le_bytes(entry[..8].try_into().expect("8 bytes")));
                    ends.push(u32::from_le_bytes(entry[8..].try_into().expect("4 bytes")) as u64);
                }
                if keys.windows(2).any(|w| w[0] >= w[1]) || ends.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(invalid("key page is not in order"));
                }
                (PageContent::Keys { keys, ends }, bytes, reservation)
            }
            PageKind::Positions => {
                if entries.len() != n * POS_ENTRY {
                    return Err(invalid("position page length does not match its count"));
                }
                let bytes = n * std::mem::size_of::<u32>();
                let reservation = INDEX_PAGES
                    .try_reserve(bytes)
                    .ok_or_else(|| refused("a position page"))?;
                let mut positions = Vec::with_capacity(n);
                positions.extend(
                    entries
                        .as_chunks::<POS_ENTRY>()
                        .0
                        .iter()
                        .map(|e| u32::from_le_bytes(*e)),
                );
                (PageContent::Positions(positions), bytes, reservation)
            }
        };
        drop(raw);
        drop(raw_reservation);
        Ok(Self {
            content,
            bytes,
            _reservation: reservation,
        })
    }

    pub fn content(&self) -> &PageContent {
        &self.content
    }

    /// The capacity the page's vectors hold, as charged
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

// =============================================================================
// Directory
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyPageMeta {
    pub first_key: i64,
    pub last_key: i64,
    offset: u64,
    len: u32,
    /// Index of the page's first key among the column's keys
    pub key_start: u64,
    /// Position index the page's first key starts at
    pub pos_start: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PosPageMeta {
    offset: u64,
    len: u32,
    /// Position index of the page's first entry
    pub pos_start: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDirectory {
    pub column: u32,
    /// The identity of the index definition the column was built for:
    /// the LSN of the catalog record that created the index. Zero is no
    /// identity, and such a column is never served
    pub identity: u64,
    pub n_keys: u64,
    pub n_positions: u64,
    pub key_pages: Vec<KeyPageMeta>,
    pub pos_pages: Vec<PosPageMeta>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Directory {
    pub generation: u64,
    pub columns: Vec<ColumnDirectory>,
}

impl Directory {
    pub fn column(&self, column: usize) -> Option<&ColumnDirectory> {
        self.columns.iter().find(|c| c.column as usize == column)
    }

    fn page_location(
        &self,
        column: usize,
        kind: PageKind,
        number: usize,
    ) -> std::io::Result<(u64, u32)> {
        let col = self
            .column(column)
            .ok_or_else(|| invalid("column has no index"))?;
        match kind {
            PageKind::Keys => col.key_pages.get(number).map(|p| (p.offset, p.len)),
            PageKind::Positions => col.pos_pages.get(number).map(|p| (p.offset, p.len)),
        }
        .ok_or_else(|| invalid("page number out of range"))
    }

    /// Bytes the directory's vectors take resident
    pub fn bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|c| {
                c.key_pages.capacity() * std::mem::size_of::<KeyPageMeta>()
                    + c.pos_pages.capacity() * std::mem::size_of::<PosPageMeta>()
            })
            .sum::<usize>()
            + self.columns.capacity() * std::mem::size_of::<ColumnDirectory>()
    }

    fn encoded_len(&self) -> usize {
        4 + self
            .columns
            .iter()
            .map(|c| {
                COLUMN_ENTRY + c.key_pages.len() * KEY_DIR_ENTRY + c.pos_pages.len() * POS_DIR_ENTRY
            })
            .sum::<usize>()
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        for c in &self.columns {
            out.extend_from_slice(&c.column.to_le_bytes());
            out.push(KEY_I64);
            out.extend_from_slice(&c.identity.to_le_bytes());
            out.extend_from_slice(&c.n_keys.to_le_bytes());
            out.extend_from_slice(&c.n_positions.to_le_bytes());
            out.extend_from_slice(&(c.key_pages.len() as u32).to_le_bytes());
            out.extend_from_slice(&(c.pos_pages.len() as u32).to_le_bytes());
            for p in &c.key_pages {
                out.extend_from_slice(&p.first_key.to_le_bytes());
                out.extend_from_slice(&p.last_key.to_le_bytes());
                out.extend_from_slice(&p.offset.to_le_bytes());
                out.extend_from_slice(&p.len.to_le_bytes());
                out.extend_from_slice(&p.key_start.to_le_bytes());
                out.extend_from_slice(&p.pos_start.to_le_bytes());
            }
            for p in &c.pos_pages {
                out.extend_from_slice(&p.offset.to_le_bytes());
                out.extend_from_slice(&p.len.to_le_bytes());
                out.extend_from_slice(&p.pos_start.to_le_bytes());
            }
        }
    }

    /// Decodes the directory, charging each column's vectors at their
    /// capacity to the directories ledger before they are allocated; the
    /// charges come back with it.
    fn decode(generation: u64, data: &[u8], file_len: u64) -> std::io::Result<(Self, Vec<Held>)> {
        let mut pos = 0usize;
        let count = read_u32(data, &mut pos)? as usize;
        if count > 4096 {
            return Err(invalid("too many columns"));
        }
        let mut charges = Vec::with_capacity(count + 1);
        charges.push(INDEX_DIRECTORIES.charge(count * std::mem::size_of::<ColumnDirectory>()));
        let mut columns = Vec::with_capacity(count);
        for _ in 0..count {
            if data.len() < pos + COLUMN_ENTRY {
                return Err(invalid("directory truncated"));
            }
            let column = read_u32(data, &mut pos)?;
            let tag = data[pos];
            pos += 1;
            if tag != KEY_I64 {
                return Err(invalid("key type unsupported"));
            }
            let identity = read_u64(data, &mut pos)?;
            let n_keys = read_u64(data, &mut pos)?;
            let n_positions = read_u64(data, &mut pos)?;
            let key_pages = read_u32(data, &mut pos)? as usize;
            let pos_pages = read_u32(data, &mut pos)? as usize;
            if data.len() < pos + key_pages * KEY_DIR_ENTRY + pos_pages * POS_DIR_ENTRY {
                return Err(invalid("directory shorter than its page counts"));
            }
            charges.push(INDEX_DIRECTORIES.charge(
                key_pages * std::mem::size_of::<KeyPageMeta>()
                    + pos_pages * std::mem::size_of::<PosPageMeta>(),
            ));
            let mut kp = Vec::with_capacity(key_pages);
            for _ in 0..key_pages {
                let first_key = read_u64(data, &mut pos)? as i64;
                let last_key = read_u64(data, &mut pos)? as i64;
                let offset = read_u64(data, &mut pos)?;
                let len = read_u32(data, &mut pos)?;
                let key_start = read_u64(data, &mut pos)?;
                let pos_start = read_u64(data, &mut pos)?;
                if first_key > last_key
                    || offset + len as u64 > file_len
                    || len as usize > PAGE_BYTES
                {
                    return Err(invalid("key page entry out of bounds"));
                }
                kp.push(KeyPageMeta {
                    first_key,
                    last_key,
                    offset,
                    len,
                    key_start,
                    pos_start,
                });
            }
            let mut pp = Vec::with_capacity(pos_pages);
            for _ in 0..pos_pages {
                let offset = read_u64(data, &mut pos)?;
                let len = read_u32(data, &mut pos)?;
                let pos_start = read_u64(data, &mut pos)?;
                if offset + len as u64 > file_len || len as usize > PAGE_BYTES {
                    return Err(invalid("position page entry out of bounds"));
                }
                pp.push(PosPageMeta {
                    offset,
                    len,
                    pos_start,
                });
            }
            if kp
                .windows(2)
                .any(|w| w[0].last_key >= w[1].first_key || w[0].pos_start > w[1].pos_start)
                || pp.windows(2).any(|w| w[0].pos_start >= w[1].pos_start)
            {
                return Err(invalid("directory pages are not in order"));
            }
            columns.push(ColumnDirectory {
                column,
                identity,
                n_keys,
                n_positions,
                key_pages: kp,
                pos_pages: pp,
            });
        }
        if pos != data.len() {
            return Err(invalid("directory has trailing bytes"));
        }
        Ok((
            Self {
                generation,
                columns,
            },
            charges,
        ))
    }
}

fn read_u64(data: &[u8], pos: &mut usize) -> std::io::Result<u64> {
    let end = *pos + 8;
    let bytes = data.get(*pos..end).ok_or_else(|| invalid("truncated"))?;
    *pos = end;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
}

fn read_u32(data: &[u8], pos: &mut usize) -> std::io::Result<u32> {
    let end = *pos + 4;
    let bytes = data.get(*pos..end).ok_or_else(|| invalid("truncated"))?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
}

// =============================================================================
// The open file: directory resident, pages on demand
// =============================================================================

/// A side file opened at one generation: its directory, resident and
/// charged to the directories ledger for the file's life, and the handle
/// its pages are read through. No descriptor is held between reads; every
/// page read checks the header's generation against the one the directory
/// was read at.
pub struct IndexFile {
    file: Arc<VolumeFile>,
    /// The bytes of this generation, open for the file's lifetime: a
    /// build that replaces the path leaves a holder's pages as they were
    descriptor: std::fs::File,
    file_id: u64,
    directory: Directory,
    _charges: Vec<Held>,
}

/// An error a caller answers by reopening the file.
pub fn is_generation_changed(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::InvalidData
        && error.to_string().contains("generation changed")
}

impl IndexFile {
    /// Opens `path` through the process's handle of it. `file_id`
    /// distinguishes files in the page cache; the segment id is the
    /// natural choice.
    pub fn open(path: &Path, file_id: u64) -> std::io::Result<Self> {
        Self::open_through(&VolumeFile::shared(path), file_id)
    }

    /// Opens the file through its handle, whose open holds the path lock
    /// so a rename cannot land between the two, and reads the header,
    /// footer and directory, checking their consistency. The file keeps
    /// the handle: a page read opens through it, and the file on disk
    /// goes with the last holder once retired.
    pub fn open_through(handle: &Arc<VolumeFile>, file_id: u64) -> std::io::Result<Self> {
        let mut file = handle.open()?;
        let file_len = file.metadata()?.len();
        if file_len < HEADER_LEN + FOOTER_LEN {
            return Err(invalid("file shorter than header and footer"));
        }
        let mut header = [0u8; HEADER_LEN as usize];
        file.read_exact(&mut header)?;
        if header[..4] != MAGIC {
            return Err(invalid("bad magic"));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().expect("4 bytes"));
        if version != VERSION {
            return Err(invalid("version unsupported"));
        }
        let generation = u64::from_le_bytes(header[8..16].try_into().expect("8 bytes"));
        let mut footer = [0u8; FOOTER_LEN as usize];
        read_exact_at(&file, &mut footer, file_len - FOOTER_LEN)?;
        if footer[16..] != MAGIC {
            return Err(invalid("bad footer magic"));
        }
        let dir_offset = u64::from_le_bytes(footer[..8].try_into().expect("8 bytes"));
        let dir_len = u32::from_le_bytes(footer[8..12].try_into().expect("4 bytes")) as u64;
        let dir_crc = u32::from_le_bytes(footer[12..16].try_into().expect("4 bytes"));
        if dir_offset < HEADER_LEN || dir_offset + dir_len != file_len - FOOTER_LEN {
            return Err(invalid("directory location out of bounds"));
        }
        // The raw directory is charged while it is read and parsed
        let raw_charge = INDEX_DIRECTORIES.charge(dir_len as usize);
        let mut dir = vec![0u8; dir_len as usize];
        read_exact_at(&file, &mut dir, dir_offset)?;
        if crc32fast::hash(&dir) != dir_crc {
            return Err(invalid("directory checksum mismatch"));
        }
        let (directory, charges) = Directory::decode(generation, &dir, dir_offset)?;
        drop(dir);
        drop(raw_charge);
        Ok(Self {
            file: Arc::clone(handle),
            descriptor: file,
            file_id,
            directory,
            _charges: charges,
        })
    }

    pub fn directory(&self) -> &Directory {
        &self.directory
    }

    pub fn generation(&self) -> u64 {
        self.directory.generation
    }

    /// The handle the pages are read through
    pub fn handle(&self) -> &Arc<VolumeFile> {
        &self.file
    }

    /// Whether the file indexes `column` for the index definition with
    /// `identity`; a column without an identity covers nothing
    pub fn covers(&self, column: usize, identity: u64) -> bool {
        identity != 0
            && self
                .directory
                .column(column)
                .is_some_and(|c| c.identity == identity)
    }

    /// Reads one page's raw bytes into `buf`, whose capacity the caller
    /// reserved
    fn read_page_into(&self, offset: u64, len: u32, buf: &mut Vec<u8>) -> std::io::Result<()> {
        let file = self.file.open()?;
        let mut header = [0u8; HEADER_LEN as usize];
        read_exact_at(&file, &mut header, 0)?;
        if header[..4] != MAGIC {
            return Err(invalid("bad magic"));
        }
        let generation = u64::from_le_bytes(header[8..16].try_into().expect("8 bytes"));
        if generation != self.directory.generation {
            return Err(invalid("generation changed"));
        }
        buf.clear();
        buf.resize(len as usize, 0);
        read_exact_at(&file, buf, offset)
    }

    fn read_page(&self, offset: u64, len: u32) -> std::io::Result<Vec<u8>> {
        let mut raw = vec![0u8; len as usize];
        read_exact_at(&self.descriptor, &mut raw, offset)?;
        Ok(raw)
    }

    /// The position index range `[start, end)` of `key` in `column`, or
    /// None when the key is absent; reads at most one key page, through
    /// the cache alone: a page the cache refuses is this call's error.
    pub fn equal(&self, column: usize, key: i64) -> std::io::Result<Option<(u64, u64)>> {
        equal_with(&Cached { file: self, column }, key)
    }

    /// The upper bound the directory alone gives on the positions of keys
    /// in `[low, high]`: every position of every key page the range
    /// touches, boundary pages whole.
    pub fn candidate_bound(&self, column: usize, low: i64, high: i64) -> std::io::Result<u64> {
        let col = self
            .directory
            .column(column)
            .ok_or_else(|| invalid("column has no index"))?;
        if low > high {
            return Ok(0);
        }
        let first = col.key_pages.partition_point(|p| p.last_key < low);
        let last = col.key_pages.partition_point(|p| p.first_key <= high);
        if first >= last {
            return Ok(0);
        }
        let start = col.key_pages[first].pos_start;
        let end = col
            .key_pages
            .get(last)
            .map_or(col.n_positions, |p| p.pos_start);
        Ok(end - start)
    }

    /// What the directory alone says of the positions of keys in
    /// `[low, high]`, without reading a page: None when no key page touches
    /// the range; else the positions of the pages inside the range exactly
    /// and of the boundary pages by key interpolation, one at least
    pub fn candidate_estimate(
        &self,
        column: usize,
        low: i64,
        high: i64,
    ) -> std::io::Result<Option<u64>> {
        let col = self
            .directory
            .column(column)
            .ok_or_else(|| invalid("column has no index"))?;
        if low > high {
            return Ok(None);
        }
        let first = col.key_pages.partition_point(|p| p.last_key < low);
        let last = col.key_pages.partition_point(|p| p.first_key <= high);
        if first >= last {
            return Ok(None);
        }
        let positions_of = |page: usize| -> u64 {
            let start = col.key_pages[page].pos_start;
            col.key_pages
                .get(page + 1)
                .map_or(col.n_positions, |p| p.pos_start)
                - start
        };
        let share_of = |page: usize| -> u64 {
            let p = &col.key_pages[page];
            let span = (p.last_key as i128 - p.first_key as i128 + 1) as u128;
            let overlap = (high.min(p.last_key) as i128 - low.max(p.first_key) as i128 + 1) as u128;
            (positions_of(page) as u128 * overlap / span) as u64
        };
        let estimate = if last - first == 1 {
            share_of(first)
        } else {
            let interior = col.key_pages[last - 1].pos_start - col.key_pages[first + 1].pos_start;
            share_of(first) + interior + share_of(last - 1)
        };
        Ok(Some(estimate.max(1)))
    }

    /// Whether the pages a probe of `[low, high]` touches (the key pages
    /// the range spans and the position pages under them) are cached or
    /// would fit in the cache beside a reader's reservation of `reserve`
    /// bytes without an eviction
    pub fn pages_admissible(
        &self,
        column: usize,
        low: i64,
        high: i64,
        reserve: usize,
    ) -> std::io::Result<bool> {
        let col = self
            .directory
            .column(column)
            .ok_or_else(|| invalid("column has no index"))?;
        let first = col.key_pages.partition_point(|p| p.last_key < low);
        let last = col.key_pages.partition_point(|p| p.first_key <= high);
        if first >= last {
            return Ok(true);
        }
        let start = col.key_pages[first].pos_start;
        let end = col
            .key_pages
            .get(last)
            .map_or(col.n_positions, |p| p.pos_start);
        let pos_first = col
            .pos_pages
            .partition_point(|p| p.pos_start <= start)
            .saturating_sub(1);
        let pos_last = col.pos_pages.partition_point(|p| p.pos_start < end);
        let pages = (first..last)
            .map(|n| (PageKind::Keys, n))
            .chain((pos_first..pos_last).map(|n| (PageKind::Positions, n)));
        INDEX_PAGES.admits(self, column, pages, reserve)
    }

    /// The exact position index range of keys in `[low, high]`; reads the
    /// boundary key pages through the cache alone.
    pub fn range(&self, column: usize, low: i64, high: i64) -> std::io::Result<(u64, u64)> {
        range_with(&Cached { file: self, column }, low, high)
    }

    /// A cursor over the positions at index range `[start, end)` of
    /// `column`, yielding them in windows of at most `window` positions,
    /// each window sorted ascending; the window is the cursor's whole
    /// workspace, reserved before it is allocated, for the cursor's life.
    pub fn cursor(
        &self,
        column: usize,
        range: (u64, u64),
        window: usize,
    ) -> std::io::Result<Cursor<'_>> {
        let window = window.max(1);
        let reservation = INDEX_PAGES
            .try_reserve(window * POS_ENTRY)
            .ok_or_else(|| refused("a cursor window"))?;
        Ok(Cursor {
            pages: Cached { file: self, column },
            next: range.0,
            end: range.1,
            window,
            buffer: Vec::with_capacity(window),
            _reservation: reservation,
        })
    }

    /// A reader of `column` with one working reservation for its whole
    /// walk: a window of `window` positions, a raw page buffer and a
    /// parsed page of each kind, taken before its first row. Refused, the
    /// caller serves the volume through the scan (`is_refused`). Admitted,
    /// the reader always progresses: a page comes from the cache when the
    /// cache admits it and through the reader's own buffers when it does
    /// not, so a cache that admits nothing or a budget lowered afterwards
    /// never stops it.
    pub fn reader(self: &Arc<Self>, column: usize, window: usize) -> std::io::Result<Reader> {
        let window = window.max(1);
        let reservation = INDEX_PAGES
            .try_reserve(reader_bytes(window))
            .ok_or_else(|| refused("a reader's working reservation"))?;
        Ok(Reader {
            file: Arc::clone(self),
            column,
            next: 0,
            end: 0,
            window,
            buffer: Vec::with_capacity(window),
            own: std::cell::RefCell::new(OwnPages {
                raw: Vec::with_capacity(PAGE_BYTES),
                keys: PageContent::Keys {
                    keys: Vec::with_capacity(KEYS_PER_PAGE),
                    ends: Vec::with_capacity(KEYS_PER_PAGE),
                },
                positions: PageContent::Positions(Vec::with_capacity(POSITIONS_PER_PAGE)),
                pages: 0,
            }),
            _reservation: reservation,
        })
    }
}

/// A reader with its own working space: the cursor window and one page
/// of each kind, so an admitted walk completes whatever the cache admits.
/// It holds its file, so a scanner can carry it across its rows.
pub struct Reader {
    file: Arc<IndexFile>,
    column: usize,
    next: u64,
    end: u64,
    window: usize,
    buffer: Vec<u32>,
    own: std::cell::RefCell<OwnPages>,
    _reservation: Reservation,
}

/// The reader's own page buffers, filled when the cache refuses a page
struct OwnPages {
    raw: Vec<u8>,
    keys: PageContent,
    positions: PageContent,
    /// Pages read through these buffers rather than the cache
    pages: u64,
}

impl Reader {
    /// Pages this reader read through its own buffers
    pub fn own_pages(&self) -> u64 {
        self.own.borrow().pages
    }

    /// The position index range `[start, end)` of `key`, or None when the
    /// key is absent; reads at most one key page.
    pub fn equal(&self, key: i64) -> std::io::Result<Option<(u64, u64)>> {
        equal_with(self, key)
    }

    /// The exact position index range of keys in `[low, high]`; reads the
    /// boundary key pages.
    pub fn range(&self, low: i64, high: i64) -> std::io::Result<(u64, u64)> {
        range_with(self, low, high)
    }

    /// The upper bound the directory alone gives on the positions of keys
    /// in `[low, high]`, without reading a page
    pub fn candidate_bound(&self, low: i64, high: i64) -> std::io::Result<u64> {
        self.file.candidate_bound(self.column, low, high)
    }

    /// Starts a walk over the position index range `[start, end)`
    pub fn walk(&mut self, range: (u64, u64)) {
        self.next = range.0;
        self.end = range.1;
    }

    /// Positions left to yield, including the current window
    pub fn remaining(&self) -> u64 {
        self.end.saturating_sub(self.next)
    }

    /// The window last returned by `next_window`, in the reader's own
    /// reserved space
    pub fn window(&self) -> &[u32] {
        &self.buffer
    }

    /// The next window of the walk, sorted ascending, or None at the end.
    /// A page that fails to read leaves the walk where the window began.
    pub fn next_window(&mut self) -> std::io::Result<Option<&[u32]>> {
        let mut buffer = std::mem::take(&mut self.buffer);
        let filled = next_window_with(self, self.next, self.end, self.window, &mut buffer);
        if filled.is_err() {
            // A failed window is no window: the walk stays where it began
            buffer.clear();
        }
        self.buffer = buffer;
        match filled? {
            Some(advanced) => {
                self.next = advanced;
                Ok(Some(&self.buffer))
            }
            None => Ok(None),
        }
    }
}

impl Pages for Reader {
    fn file(&self) -> &IndexFile {
        &self.file
    }

    fn column(&self) -> usize {
        self.column
    }

    /// From the cache when it admits the page, else read and parsed into
    /// the reader's own buffers. A real read or checksum error is the
    /// walk's error.
    fn with_page<R>(
        &self,
        kind: PageKind,
        number: usize,
        f: impl FnOnce(&PageContent) -> R,
    ) -> std::io::Result<R> {
        match INDEX_PAGES.load(&self.file, self.column, kind, number) {
            Ok(page) => return Ok(f(page.content())),
            Err(error) if is_refused(&error) => {}
            Err(error) => return Err(error),
        }
        let (offset, len) = self
            .file
            .directory
            .page_location(self.column, kind, number)?;
        let mut own = self.own.borrow_mut();
        let own = &mut *own;
        self.file.read_page_into(offset, len, &mut own.raw)?;
        let (n, entries) = verified_entries(&own.raw)?;
        let content = match kind {
            PageKind::Keys => {
                if entries.len() != n * KEY_ENTRY {
                    return Err(invalid("key page length does not match its count"));
                }
                let PageContent::Keys { keys, ends } = &mut own.keys else {
                    return Err(invalid("expected a key page"));
                };
                keys.clear();
                ends.clear();
                for entry in entries.as_chunks::<KEY_ENTRY>().0 {
                    keys.push(i64::from_le_bytes(entry[..8].try_into().expect("8 bytes")));
                    ends.push(u32::from_le_bytes(entry[8..].try_into().expect("4 bytes")) as u64);
                }
                if keys.windows(2).any(|w| w[0] >= w[1]) || ends.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(invalid("key page is not in order"));
                }
                &own.keys
            }
            PageKind::Positions => {
                if entries.len() != n * POS_ENTRY {
                    return Err(invalid("position page length does not match its count"));
                }
                let PageContent::Positions(positions) = &mut own.positions else {
                    return Err(invalid("expected a position page"));
                };
                positions.clear();
                positions.extend(
                    entries
                        .as_chunks::<POS_ENTRY>()
                        .0
                        .iter()
                        .map(|e| u32::from_le_bytes(*e)),
                );
                &own.positions
            }
        };
        own.pages += 1;
        Ok(f(content))
    }
}

/// Where a search or a walk gets its pages: the file, the column, and
/// one page at a time, however it is held. The search and the walk are
/// written once over this.
trait Pages {
    fn file(&self) -> &IndexFile;
    fn column(&self) -> usize;
    fn with_page<R>(
        &self,
        kind: PageKind,
        number: usize,
        f: impl FnOnce(&PageContent) -> R,
    ) -> std::io::Result<R>;
}

/// Pages through the cache alone: a page the cache refuses is an error
struct Cached<'a> {
    file: &'a IndexFile,
    column: usize,
}

impl Pages for Cached<'_> {
    fn file(&self) -> &IndexFile {
        self.file
    }

    fn column(&self) -> usize {
        self.column
    }

    fn with_page<R>(
        &self,
        kind: PageKind,
        number: usize,
        f: impl FnOnce(&PageContent) -> R,
    ) -> std::io::Result<R> {
        let page = INDEX_PAGES.load(self.file, self.column, kind, number)?;
        Ok(f(page.content()))
    }
}

fn column_of(pages: &impl Pages) -> std::io::Result<&ColumnDirectory> {
    pages
        .file()
        .directory
        .column(pages.column())
        .ok_or_else(|| invalid("column has no index"))
}

/// The position index range `[start, end)` of `key`, or None when the
/// key is absent; reads at most one key page.
fn equal_with(pages: &impl Pages, key: i64) -> std::io::Result<Option<(u64, u64)>> {
    let col = column_of(pages)?;
    let page_no = col.key_pages.partition_point(|p| p.last_key < key);
    let Some(meta) = col.key_pages.get(page_no) else {
        return Ok(None);
    };
    if key < meta.first_key {
        return Ok(None);
    }
    let pos_start = meta.pos_start;
    pages.with_page(PageKind::Keys, page_no, |content| {
        let PageContent::Keys { keys, ends } = content else {
            return Err(invalid("expected a key page"));
        };
        let Ok(i) = keys.binary_search(&key) else {
            return Ok(None);
        };
        let start = if i == 0 { pos_start } else { ends[i - 1] };
        Ok(Some((start, ends[i])))
    })?
}

/// The exact position index range of keys in `[low, high]`; reads the
/// boundary key pages.
fn range_with(pages: &impl Pages, low: i64, high: i64) -> std::io::Result<(u64, u64)> {
    let col = column_of(pages)?;
    if low > high {
        return Ok((0, 0));
    }
    let first = col.key_pages.partition_point(|p| p.last_key < low);
    let last = col.key_pages.partition_point(|p| p.first_key <= high);
    if first >= last {
        return Ok((0, 0));
    }
    let first_meta = col.key_pages[first].clone();
    let last_meta = col.key_pages[last - 1].clone();
    let after_last = col
        .key_pages
        .get(last)
        .map_or(col.n_positions, |p| p.pos_start);
    let start = if low <= first_meta.first_key {
        first_meta.pos_start
    } else {
        pages.with_page(PageKind::Keys, first, |content| {
            let PageContent::Keys { keys, ends } = content else {
                return Err(invalid("expected a key page"));
            };
            let i = keys.partition_point(|&k| k < low);
            Ok(if i == 0 {
                first_meta.pos_start
            } else {
                ends[i - 1]
            })
        })??
    };
    let end = if high >= last_meta.last_key {
        after_last
    } else {
        pages.with_page(PageKind::Keys, last - 1, |content| {
            let PageContent::Keys { keys, ends } = content else {
                return Err(invalid("expected a key page"));
            };
            let i = keys.partition_point(|&k| k <= high);
            Ok(if i == 0 {
                last_meta.pos_start
            } else {
                ends[i - 1]
            })
        })??
    };
    Ok((start, end.max(start)))
}

/// Fills `buffer` with the next window of at most `window` positions of
/// the walk at `next` towards `end`, sorted ascending, and returns where
/// the walk stands after it; None at the end. Progress is reported only
/// for a window filled whole: a page that failed leaves the walk where
/// the window began.
fn next_window_with(
    pages: &impl Pages,
    next: u64,
    end: u64,
    window: usize,
    buffer: &mut Vec<u32>,
) -> std::io::Result<Option<u64>> {
    if next >= end {
        return Ok(None);
    }
    buffer.clear();
    let stop = (next + window as u64).min(end);
    let col = column_of(pages)?;
    let mut cursor = next;
    while cursor < stop {
        let page_no = col
            .pos_pages
            .partition_point(|p| p.pos_start <= cursor)
            .checked_sub(1)
            .ok_or_else(|| invalid("position index before the first page"))?;
        let page_start = col.pos_pages[page_no].pos_start;
        cursor = pages.with_page(PageKind::Positions, page_no, |content| {
            let PageContent::Positions(positions) = content else {
                return Err(invalid("expected a position page"));
            };
            let from = (cursor - page_start) as usize;
            let to = ((stop - page_start) as usize).min(positions.len());
            if from >= to {
                return Err(invalid("position page does not cover its index"));
            }
            buffer.extend_from_slice(&positions[from..to]);
            Ok(page_start + to as u64)
        })??;
    }
    buffer.sort_unstable();
    Ok(Some(cursor))
}

fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "side index file ends inside a page",
                ));
            }
            done += n;
        }
        Ok(())
    }
}

/// Positions of one key or one range, a bounded window at a time, the
/// pages through the cache alone.
pub struct Cursor<'a> {
    pages: Cached<'a>,
    next: u64,
    end: u64,
    window: usize,
    buffer: Vec<u32>,
    _reservation: Reservation,
}

impl Cursor<'_> {
    /// Positions left to yield, including the current window
    pub fn remaining(&self) -> u64 {
        self.end - self.next
    }

    /// The next window, sorted ascending, or None at the end. Pages the
    /// window spans are loaded one at a time and released as the window
    /// moves on.
    pub fn next_window(&mut self) -> std::io::Result<Option<&[u32]>> {
        let mut buffer = std::mem::take(&mut self.buffer);
        let (next, end, window) = (self.next, self.end, self.window);
        let filled = next_window_with(&self.pages, next, end, window, &mut buffer);
        if filled.is_err() {
            buffer.clear();
        }
        self.buffer = buffer;
        match filled? {
            Some(advanced) => {
                self.next = advanced;
                Ok(Some(&self.buffer))
            }
            None => Ok(None),
        }
    }
}

// =============================================================================
// Building
// =============================================================================

/// What a build did, for the measurements.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BuildReport {
    pub keys: u64,
    pub positions: u64,
    pub key_pages: usize,
    pub pos_pages: usize,
    pub file_bytes: u64,
    /// Sorted runs spilled to disk because the pairs exceeded the workspace
    pub runs_spilled: usize,
    /// Merges of runs into runs, before the merge into pages
    pub merge_passes: usize,
    /// The most runs merged at once
    pub max_fan_in: usize,
    /// The most run files alive at once
    pub max_live_runs: usize,
    /// The most bytes this build held at once in its own ledger
    pub workspace_peak: usize,
    pub build_ns: u64,
}

/// A column to index: its physical index and its `(position, key)` pairs,
/// null rows left out, in any order. An iterator without an upper size
/// bound is accepted, with a quarter of the remaining workspace kept for
/// its page metadata; a column whose metadata outgrows that fails the
/// build, and a hint avoids the failure.
pub struct ColumnInput<'a> {
    pub column: u32,
    /// The identity of the index definition the column is built for
    pub identity: u64,
    pub pairs: Box<dyn Iterator<Item = (u32, i64)> + 'a>,
}

const PAIR_BYTES: usize = std::mem::size_of::<(i64, u32)>();
/// Each merge input reads through a buffer of this size
pub const MERGE_READ_BYTES: usize = 16 * 1024;
/// Runs merged at once at most, whatever the workspace: a ceiling on open
/// descriptors
pub const MAX_FAN_IN: usize = 32;
const MIN_FAN_IN: usize = 2;
const MIN_RUN_PAIRS: usize = 1024;
/// The output buffer of the side file and of a spilled run
const OUT_BUFFER_BYTES: usize = 64 * 1024;
/// What the page writer holds whatever the input: a key page, a position
/// page and the encoded body of the page being written
const WRITER_BYTES: usize = KEYS_PER_PAGE * std::mem::size_of::<(i64, u64)>()
    + POSITIONS_PER_PAGE * std::mem::size_of::<u32>()
    + PAGE_BYTES;
/// One entry of the merge heap
const MERGE_HEAP_ENTRY: usize = std::mem::size_of::<std::cmp::Reverse<((i64, u32), usize)>>();
/// What one merge input costs beside its read buffer: its reader, its heap
/// entry and its path
const MERGE_INPUT_BYTES: usize = std::mem::size_of::<RunReader>()
    + MERGE_HEAP_ENTRY
    + std::mem::size_of::<PathBuf>()
    + RUN_NAME_BYTES;
/// A run's path fits this
const RUN_NAME_BYTES: usize = 256;
/// The least workspace a build accepts: the writer's buffers, the output
/// buffer, the buffer of a run being spilled or merged into, two merge
/// inputs with their read buffers, and a run of `MIN_RUN_PAIRS`; the page
/// metadata an input needs comes on top and is checked per input
pub const MIN_WORKSPACE_BYTES: usize = WRITER_BYTES
    + 2 * OUT_BUFFER_BYTES
    + MIN_FAN_IN * (MERGE_READ_BYTES + MERGE_INPUT_BYTES)
    + MIN_RUN_PAIRS * PAIR_BYTES
    + COLUMN_SLOT_BYTES;

/// What one input column costs the build beside its pages: its directory
/// entry and the charge of its page metadata; the minimum covers one, and
/// every further column adds its own
pub const COLUMN_SLOT_BYTES: usize =
    std::mem::size_of::<ColumnDirectory>() + std::mem::size_of::<Charge<'static>>();

/// The page metadata `rows` rows produce at most, at the doubling
/// capacities the writer's vectors grow by
pub fn metadata_allowance(rows: usize) -> usize {
    let key_pages = rows
        .div_ceil(KEYS_PER_PAGE)
        .max(1)
        .next_power_of_two()
        .max(4);
    let pos_pages = rows
        .div_ceil(POSITIONS_PER_PAGE)
        .max(1)
        .next_power_of_two()
        .max(4);
    key_pages * std::mem::size_of::<KeyPageMeta>() + pos_pages * std::mem::size_of::<PosPageMeta>()
}

fn workspace_exceeded(what: &str, bytes: usize, used: usize, workspace: usize) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "side index build: {bytes} bytes for {what} would take the build to {} bytes, over its workspace of {workspace}",
            used + bytes
        ),
    )
}

/// The build's own ledger over its workspace: every byte the build holds
/// is taken here before it is allocated, and a take the workspace cannot
/// hold fails the build. The workspace itself is admitted whole to the
/// builds ledger before the build starts.
struct Budget {
    workspace: usize,
    used: std::cell::Cell<usize>,
    peak: std::cell::Cell<usize>,
}

impl Budget {
    fn new(workspace: usize) -> Self {
        Self {
            workspace,
            used: std::cell::Cell::new(0),
            peak: std::cell::Cell::new(0),
        }
    }

    fn left(&self) -> usize {
        self.workspace - self.used.get()
    }

    fn take(&self, bytes: usize, what: &str) -> std::io::Result<Charge<'_>> {
        let used = self.used.get();
        if used + bytes > self.workspace {
            return Err(workspace_exceeded(what, bytes, used, self.workspace));
        }
        self.used.set(used + bytes);
        self.peak.set(self.peak.get().max(used + bytes));
        Ok(Charge {
            bytes,
            budget: self,
        })
    }
}

/// Bytes taken from a build's budget, given back on drop.
struct Charge<'b> {
    bytes: usize,
    budget: &'b Budget,
}

impl Charge<'_> {
    fn grow(&mut self, more: usize, what: &str) -> std::io::Result<()> {
        let used = self.budget.used.get();
        if used + more > self.budget.workspace {
            return Err(workspace_exceeded(what, more, used, self.budget.workspace));
        }
        self.budget.used.set(used + more);
        self.budget
            .peak
            .set(self.budget.peak.get().max(used + more));
        self.bytes += more;
        Ok(())
    }
}

impl Drop for Charge<'_> {
    fn drop(&mut self) {
        self.budget
            .used
            .set(self.budget.used.get().saturating_sub(self.bytes));
    }
}

/// A build's own directory beside its target, `<name>.build-<pid>-<generation>`,
/// created by the build and refused if it already exists, so that every
/// file in it is the build's: the temporary output and the runs. On a
/// failure the directory goes with everything in it; on success the
/// output is renamed out of it and the empty directory is removed. Nothing
/// outside it is ever touched, and a relative target without a parent
/// builds beside itself in the current directory.
struct TempFiles {
    build_dir: PathBuf,
    keep: bool,
}

impl TempFiles {
    fn new(path: &Path, generation: u64) -> std::io::Result<Self> {
        let dir = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| invalid("side file path has no name"))?;
        let build_dir = dir.join(format!(
            "{name}.build-{}-{generation:x}",
            std::process::id()
        ));
        std::fs::create_dir(&build_dir)?;
        Ok(Self {
            build_dir,
            keep: false,
        })
    }

    fn tmp_path(&self) -> PathBuf {
        self.build_dir.join("out")
    }

    fn run_path(&self, column: usize, level: usize, number: usize) -> PathBuf {
        self.build_dir
            .join(format!("run{column}-L{level}-{number}"))
    }

    /// The output was renamed out; the directory is empty and goes
    fn finished(mut self) {
        self.keep = true;
        let _ = std::fs::remove_dir(&self.build_dir);
    }
}

impl Drop for TempFiles {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.build_dir);
        }
    }
}

/// The sorted runs of one column on disk: at most `fan_in - 1` queued at
/// each level, a level's runs merged into one run of the next level as
/// soon as there are `fan_in` of them, so the files alive and the names
/// held are bounded by the fan-in and the levels, not by the input. A run
/// is named by its level and a number the level hands out in order.
struct Runs<'t> {
    temps: &'t TempFiles,
    column: usize,
    fan_in: usize,
    levels: Vec<Level>,
    live: usize,
}

#[derive(Default)]
struct Level {
    /// The number the level's next run takes
    next: usize,
    /// The runs of the level not yet merged, oldest first
    queued: VecDeque<usize>,
}

impl<'t> Runs<'t> {
    fn new(temps: &'t TempFiles, column: usize, fan_in: usize) -> Self {
        Self {
            temps,
            column,
            fan_in,
            levels: Vec::new(),
            live: 0,
        }
    }

    fn path(&self, level: usize, number: usize) -> PathBuf {
        self.temps.run_path(self.column, level, number)
    }

    /// The path the next run of `level` takes; `added` records it once
    /// it is written
    fn next_path(&mut self, level: usize) -> PathBuf {
        if self.levels.len() <= level {
            self.levels.resize_with(level + 1, Level::default);
        }
        self.path(level, self.levels[level].next)
    }

    /// The run at `next_path(level)` was written; merges the level
    /// upward while it is full.
    fn added(
        &mut self,
        level: usize,
        budget: &Budget,
        report: &mut BuildReport,
    ) -> std::io::Result<()> {
        let mut level = level;
        loop {
            let number = self.levels[level].next;
            self.levels[level].next += 1;
            self.levels[level].queued.push_back(number);
            self.live += 1;
            report.max_live_runs = report.max_live_runs.max(self.live);
            if self.levels[level].queued.len() < self.fan_in {
                return Ok(());
            }
            let inputs: Vec<PathBuf> = self.levels[level]
                .queued
                .iter()
                .map(|&n| self.path(level, n))
                .collect();
            let target = self.next_path(level + 1);
            self.merge_into(&inputs, &target, budget, report)?;
            self.levels[level].queued.clear();
            self.live -= inputs.len();
            level += 1;
        }
    }

    fn merge_into(
        &mut self,
        inputs: &[PathBuf],
        target: &Path,
        budget: &Budget,
        report: &mut BuildReport,
    ) -> std::io::Result<()> {
        let names = budget.take(
            inputs.iter().map(|p| p.capacity()).sum::<usize>() + std::mem::size_of_val(inputs),
            "run names",
        )?;
        let sink_buffer = budget.take(OUT_BUFFER_BYTES, "merge output buffer")?;
        let mut sink = RunSink::create(target)?;
        merge_runs(inputs, budget, report, |key, pos| sink.push(key, pos))?;
        sink.finish()?;
        drop(sink_buffer);
        drop(names);
        for input in inputs {
            std::fs::remove_file(input)?;
        }
        report.merge_passes += 1;
        Ok(())
    }

    /// Every run alive as `(level, number)`, oldest level first
    fn all(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (level, l) in self.levels.iter().enumerate() {
            for &n in &l.queued {
                out.push((level, n));
            }
        }
        out
    }

    /// Merges runs together until at most `fan_in` are left, and returns
    /// their paths
    fn settle(
        &mut self,
        budget: &Budget,
        report: &mut BuildReport,
    ) -> std::io::Result<Vec<PathBuf>> {
        loop {
            let all = self.all();
            if all.len() <= self.fan_in {
                return Ok(all.iter().map(|&(l, n)| self.path(l, n)).collect());
            }
            // The oldest `fan_in` runs go into a new run at the top level
            let taken: Vec<(usize, usize)> = all.into_iter().take(self.fan_in).collect();
            let inputs: Vec<PathBuf> = taken.iter().map(|&(l, n)| self.path(l, n)).collect();
            let top = self.levels.len();
            let target = self.next_path(top);
            self.merge_into(&inputs, &target, budget, report)?;
            for (l, n) in taken {
                self.levels[l].queued.retain(|&q| q != n);
            }
            self.live -= inputs.len();
            self.added(top, budget, report)?;
        }
    }
}

/// Writes the side file at `path` for `columns`, under `workspace_bytes`
/// of memory for everything the build holds at once: the sort run, the
/// page writer, the output buffer, the spilled runs' names, merge readers
/// and heap, and the page metadata of every column built so far. Pairs
/// are sorted in runs that fit, runs are spilled next to the file and
/// merged upward as they accumulate, in a fan-in the workspace and a
/// descriptor ceiling bound. Written whole to a temporary file and renamed
/// into place, with its generation in the header; a failure at any step
/// removes the files the build created and keeps what was published. The
/// workspace is admitted whole to the builds ledger first, and a build
/// the ledger refuses (`is_refused`) creates nothing.
pub fn build_side_file(
    path: &Path,
    generation: u64,
    columns: Vec<ColumnInput<'_>>,
    workspace_bytes: usize,
) -> std::io::Result<BuildReport> {
    let (report, temps) = build_side_file_staged(path, generation, columns, workspace_bytes)?;
    std::fs::rename(temps.tmp_path(), path)?;
    temps.finished();
    Ok(report)
}

/// `build_side_file` up to the rename: the output stays in its build
/// directory beside `path`, to be published by the caller or dropped
fn build_side_file_staged(
    path: &Path,
    generation: u64,
    columns: Vec<ColumnInput<'_>>,
    workspace_bytes: usize,
) -> std::io::Result<(BuildReport, TempFiles)> {
    if workspace_bytes < MIN_WORKSPACE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "side index build needs at least {MIN_WORKSPACE_BYTES} bytes of workspace, {workspace_bytes} given"
            ),
        ));
    }
    let _admitted = INDEX_BUILDS
        .try_charge(workspace_bytes)
        .ok_or_else(|| refused("a build workspace"))?;
    let started = std::time::Instant::now();
    let budget = Budget::new(workspace_bytes);
    let mut report = BuildReport::default();
    // The output buffer and the writer's page buffers live for the build,
    // and so do the directory's columns and their metadata charges, one
    // slot per input column, taken before either vector is allocated
    let fixed = budget.take(OUT_BUFFER_BYTES + WRITER_BYTES, "output and page buffers")?;
    let per_column = budget.take(columns.len() * COLUMN_SLOT_BYTES, "the directory's columns")?;
    let temps = TempFiles::new(path, generation)?;
    let tmp = temps.tmp_path();
    let temps = temps;
    let mut out = BufWriter::with_capacity(OUT_BUFFER_BYTES, std::fs::File::create(&tmp)?);
    out.write_all(&MAGIC)?;
    out.write_all(&VERSION.to_le_bytes())?;
    out.write_all(&generation.to_le_bytes())?;
    let mut offset = HEADER_LEN;
    let mut directory = Directory {
        generation,
        columns: Vec::with_capacity(columns.len()),
    };
    // The reservations of the columns' page metadata, kept while the
    // directory is
    let mut meta_charges: Vec<Charge<'_>> = Vec::with_capacity(columns.len());
    for (ci, input) in columns.into_iter().enumerate() {
        // What this column may spend: the workspace less what earlier
        // columns still hold, less a merge's fixed costs, less the page
        // metadata this column will need (a quarter of the rest when the
        // input gives no bound)
        let hint = input.pairs.size_hint().1;
        let left = budget.left();
        let after_sink = left.checked_sub(OUT_BUFFER_BYTES).ok_or_else(|| {
            workspace_exceeded(
                "a merge output buffer",
                OUT_BUFFER_BYTES,
                budget.used.get(),
                workspace_bytes,
            )
        })?;
        let allowance = match hint {
            Some(rows) => metadata_allowance(rows),
            None => after_sink / 4,
        };
        let for_runs = after_sink.checked_sub(allowance).ok_or_else(|| {
            workspace_exceeded(
                "page metadata",
                allowance,
                budget.used.get(),
                workspace_bytes,
            )
        })?;
        // The sort run and a cascade merge's inputs are alive together, so
        // the merge inputs take at most half of what is left and the run
        // the rest
        let per_input = MERGE_READ_BYTES + MERGE_INPUT_BYTES;
        let fan_in = ((for_runs / 2) / per_input).clamp(MIN_FAN_IN, MAX_FAN_IN);
        let free = for_runs.saturating_sub(fan_in * per_input);
        let run_capacity = free / PAIR_BYTES;
        if run_capacity < MIN_RUN_PAIRS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "side index build: {left} bytes of workspace left for column {} cannot hold {fan_in} merge inputs and a sort run of {MIN_RUN_PAIRS} pairs beside its page metadata of {allowance} bytes",
                    input.column
                ),
            ));
        }
        // A run holds what the workspace allows, or fewer when the input
        // says it is smaller, so a small column reserves what it needs
        let run_len = run_capacity.min(hint.unwrap_or(run_capacity).max(MIN_RUN_PAIRS));
        let run_charge = budget.take(run_len * PAIR_BYTES, "a sort run")?;
        let mut pairs: Vec<(i64, u32)> = Vec::with_capacity(run_len);
        let mut runs = Runs::new(&temps, ci, fan_in);
        for (pos, key) in input.pairs {
            if pairs.len() == run_len {
                pairs.sort_unstable();
                let run = runs.next_path(0);
                spill(&run, &pairs, &budget)?;
                report.runs_spilled += 1;
                runs.added(0, &budget, &mut report)?;
                pairs.clear();
            }
            pairs.push((key, pos));
        }
        pairs.sort_unstable();
        let mut writer = PageWriter::new(&mut out, offset, input.column, input.identity, &budget);
        if runs.live == 0 {
            for &(key, pos) in &pairs {
                writer.push(key, pos)?;
            }
            drop(pairs);
            drop(run_charge);
        } else {
            let run = runs.next_path(0);
            spill(&run, &pairs, &budget)?;
            report.runs_spilled += 1;
            runs.added(0, &budget, &mut report)?;
            drop(pairs);
            drop(run_charge);
            let inputs = runs.settle(&budget, &mut report)?;
            let names = budget.take(
                inputs.iter().map(|p| p.capacity()).sum::<usize>()
                    + inputs.capacity() * std::mem::size_of::<PathBuf>(),
                "run names",
            )?;
            merge_runs(&inputs, &budget, &mut report, |key, pos| {
                writer.push(key, pos)
            })?;
            drop(names);
            for input in &inputs {
                std::fs::remove_file(input)?;
            }
        }
        let (column_dir, next_offset, metas) = writer.finish()?;
        meta_charges.push(metas);
        report.keys += column_dir.n_keys;
        report.positions += column_dir.n_positions;
        report.key_pages += column_dir.key_pages.len();
        report.pos_pages += column_dir.pos_pages.len();
        directory.columns.push(column_dir);
        offset = next_offset;
    }
    let dir_len = directory.encoded_len();
    let dir_charge = budget.take(dir_len, "the encoded directory")?;
    let mut dir = Vec::with_capacity(dir_len);
    directory.encode_into(&mut dir);
    out.write_all(&dir)?;
    out.write_all(&offset.to_le_bytes())?;
    out.write_all(&(dir.len() as u32).to_le_bytes())?;
    out.write_all(&crc32fast::hash(&dir).to_le_bytes())?;
    out.write_all(&MAGIC)?;
    out.flush()?;
    let file = out.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(file);
    // Each owner before the charge that backs it
    drop(dir);
    drop(dir_charge);
    drop(directory);
    drop(meta_charges);
    drop(per_column);
    drop(fixed);
    report.file_bytes = offset + dir_len as u64 + FOOTER_LEN;
    report.workspace_peak = budget.peak.get();
    report.build_ns = started.elapsed().as_nanos() as u64;
    Ok((report, temps))
}

#[cfg(test)]
thread_local! {
    /// A test fails the spill with this number (from one) on its thread
    static FAIL_SPILL: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static SPILLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn spill(run: &Path, pairs: &[(i64, u32)], budget: &Budget) -> std::io::Result<()> {
    let _buffer = budget.take(OUT_BUFFER_BYTES, "a spill buffer")?;
    #[cfg(test)]
    {
        let number = SPILLS.with(|s| {
            s.set(s.get() + 1);
            s.get()
        });
        if FAIL_SPILL.with(|f| f.get()) == Some(number) {
            return Err(std::io::Error::other("spill failed by the test"));
        }
    }
    let mut sink = RunSink::create(run)?;
    for &(key, pos) in pairs {
        sink.push(key, pos)?;
    }
    sink.finish()
}

/// Merges `runs` in key order into `emit`, one reader per run; the
/// readers, their buffers and the heap are taken from the budget for the
/// merge's life.
fn merge_runs(
    runs: &[PathBuf],
    budget: &Budget,
    report: &mut BuildReport,
    mut emit: impl FnMut(i64, u32) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let _inputs = budget.take(
        runs.len() * (MERGE_READ_BYTES + std::mem::size_of::<RunReader>() + MERGE_HEAP_ENTRY),
        "merge inputs",
    )?;
    report.max_fan_in = report.max_fan_in.max(runs.len());
    let mut readers: Vec<RunReader> = Vec::with_capacity(runs.len());
    for run in runs {
        readers.push(RunReader::open(run)?);
    }
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<((i64, u32), usize)>> =
        std::collections::BinaryHeap::with_capacity(runs.len());
    for (i, reader) in readers.iter_mut().enumerate() {
        if let Some(pair) = reader.next_pair()? {
            heap.push(std::cmp::Reverse((pair, i)));
        }
    }
    while let Some(std::cmp::Reverse(((key, pos), i))) = heap.pop() {
        emit(key, pos)?;
        if let Some(pair) = readers[i].next_pair()? {
            heap.push(std::cmp::Reverse((pair, i)));
        }
    }
    Ok(())
}

struct RunSink {
    out: BufWriter<std::fs::File>,
}

impl RunSink {
    fn create(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            out: BufWriter::with_capacity(OUT_BUFFER_BYTES, std::fs::File::create(path)?),
        })
    }

    fn push(&mut self, key: i64, pos: u32) -> std::io::Result<()> {
        self.out.write_all(&key.to_le_bytes())?;
        self.out.write_all(&pos.to_le_bytes())
    }

    fn finish(mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

struct RunReader {
    reader: std::io::BufReader<std::fs::File>,
}

impl RunReader {
    fn open(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            reader: std::io::BufReader::with_capacity(MERGE_READ_BYTES, std::fs::File::open(path)?),
        })
    }

    fn next_pair(&mut self) -> std::io::Result<Option<(i64, u32)>> {
        if self.reader.fill_buf()?.is_empty() {
            return Ok(None);
        }
        let mut key = [0u8; 8];
        let mut pos = [0u8; 4];
        self.reader.read_exact(&mut key)?;
        self.reader.read_exact(&mut pos)?;
        Ok(Some((i64::from_le_bytes(key), u32::from_le_bytes(pos))))
    }
}

/// Streams `(key, position)` pairs in order into key and position pages.
/// Its page buffers are part of the build's fixed charge; the page
/// metadata it collects is taken from the budget as it grows.
struct PageWriter<'a, 'b, W: Write> {
    out: &'a mut W,
    offset: u64,
    column: u32,
    identity: u64,
    key_page: Vec<(i64, u64)>,
    key_page_first_key_index: u64,
    key_page_pos_start: u64,
    pos_page: Vec<u32>,
    pos_page_start: u64,
    body: Vec<u8>,
    key_pages: Vec<KeyPageMeta>,
    pos_pages: Vec<PosPageMeta>,
    metas: Charge<'b>,
    current: Option<i64>,
    n_keys: u64,
    n_positions: u64,
}

impl<'a, 'b, W: Write> PageWriter<'a, 'b, W> {
    fn new(out: &'a mut W, offset: u64, column: u32, identity: u64, budget: &'b Budget) -> Self {
        Self {
            out,
            offset,
            column,
            identity,
            key_page: Vec::with_capacity(KEYS_PER_PAGE),
            key_page_first_key_index: 0,
            key_page_pos_start: 0,
            pos_page: Vec::with_capacity(POSITIONS_PER_PAGE),
            pos_page_start: 0,
            body: Vec::with_capacity(PAGE_BYTES),
            key_pages: Vec::new(),
            pos_pages: Vec::new(),
            metas: Charge { bytes: 0, budget },
            current: None,
            n_keys: 0,
            n_positions: 0,
        }
    }

    fn push(&mut self, key: i64, pos: u32) -> std::io::Result<()> {
        match self.current {
            Some(k) if k == key => {}
            Some(k) => {
                if k > key {
                    return Err(invalid("pairs out of order"));
                }
                self.close_key(k)?;
                self.current = Some(key);
            }
            None => self.current = Some(key),
        }
        if self.pos_page.len() == POSITIONS_PER_PAGE {
            self.flush_pos_page()?;
        }
        self.pos_page.push(pos);
        self.n_positions += 1;
        Ok(())
    }

    fn close_key(&mut self, key: i64) -> std::io::Result<()> {
        if self.n_positions > u32::MAX as u64 {
            return Err(invalid("more positions than a volume can hold"));
        }
        if self.key_page.len() == KEYS_PER_PAGE {
            self.flush_key_page()?;
        }
        self.key_page.push((key, self.n_positions));
        self.n_keys += 1;
        Ok(())
    }

    fn write_page(&mut self) -> std::io::Result<(u64, u32)> {
        let crc = crc32fast::hash(&self.body);
        self.out.write_all(&self.body)?;
        self.out.write_all(&crc.to_le_bytes())?;
        let at = self.offset;
        let len = (self.body.len() + 4) as u32;
        self.offset += len as u64;
        Ok((at, len))
    }

    /// Takes the growth of a metadata vector from the budget before it
    /// is pushed to
    fn reserve_meta<T>(metas: &mut Charge<'_>, vec: &mut Vec<T>) -> std::io::Result<()> {
        if vec.len() == vec.capacity() {
            let grown = vec.capacity().max(4) * 2;
            metas.grow(
                (grown - vec.capacity()) * std::mem::size_of::<T>(),
                "page metadata",
            )?;
            vec.reserve_exact(grown - vec.len());
        }
        Ok(())
    }

    fn flush_key_page(&mut self) -> std::io::Result<()> {
        if self.key_page.is_empty() {
            return Ok(());
        }
        self.body.clear();
        self.body
            .extend_from_slice(&(self.key_page.len() as u32).to_le_bytes());
        for (key, end) in &self.key_page {
            self.body.extend_from_slice(&key.to_le_bytes());
            self.body.extend_from_slice(&(*end as u32).to_le_bytes());
        }
        let (offset, len) = self.write_page()?;
        let first_key = self.key_page[0].0;
        let last = self.key_page[self.key_page.len() - 1];
        Self::reserve_meta(&mut self.metas, &mut self.key_pages)?;
        self.key_pages.push(KeyPageMeta {
            first_key,
            last_key: last.0,
            offset,
            len,
            key_start: self.key_page_first_key_index,
            pos_start: self.key_page_pos_start,
        });
        self.key_page_first_key_index += self.key_page.len() as u64;
        self.key_page_pos_start = last.1;
        self.key_page.clear();
        Ok(())
    }

    fn flush_pos_page(&mut self) -> std::io::Result<()> {
        if self.pos_page.is_empty() {
            return Ok(());
        }
        self.body.clear();
        self.body
            .extend_from_slice(&(self.pos_page.len() as u32).to_le_bytes());
        for pos in &self.pos_page {
            self.body.extend_from_slice(&pos.to_le_bytes());
        }
        let (offset, len) = self.write_page()?;
        Self::reserve_meta(&mut self.metas, &mut self.pos_pages)?;
        self.pos_pages.push(PosPageMeta {
            offset,
            len,
            pos_start: self.pos_page_start,
        });
        self.pos_page_start += self.pos_page.len() as u64;
        self.pos_page.clear();
        Ok(())
    }

    /// The column's directory, the offset after its pages, and the charge
    /// of the directory's vectors, which the caller keeps for as long as
    /// it keeps them
    fn finish(mut self) -> std::io::Result<(ColumnDirectory, u64, Charge<'b>)> {
        if let Some(key) = self.current.take() {
            self.close_key(key)?;
        }
        self.flush_pos_page()?;
        self.flush_key_page()?;
        let column = ColumnDirectory {
            column: self.column,
            identity: self.identity,
            n_keys: self.n_keys,
            n_positions: self.n_positions,
            key_pages: std::mem::take(&mut self.key_pages),
            pos_pages: std::mem::take(&mut self.pos_pages),
        };
        let budget = self.metas.budget;
        let metas = std::mem::replace(&mut self.metas, Charge { bytes: 0, budget });
        Ok((column, self.offset, metas))
    }
}

// =============================================================================
// The engine's side files: built with a volume, opened with it, retired
// with it
// =============================================================================

/// Builds that failed with an I/O error, apart from those the budget refused
pub static BUILDS_FAILED: AtomicU64 = AtomicU64::new(0);
/// Side files built and then discarded because the index definitions
/// changed before the volume was published
pub static SIDES_DISCARDED: AtomicU64 = AtomicU64::new(0);

/// The share of the builds budget one build of the engine takes at most
const BUILD_SHARE_BYTES: usize = 16 * 1024 * 1024;

/// The workspace one build of the engine takes: its share of the builds
/// budget less the `input` decode admitted beside it, and at least what
/// `rows` rows of `columns` columns need
pub fn workspace_for(rows: usize, columns: usize, input: usize) -> usize {
    let share = (INDEX_BUILDS.stats().budget_bytes as usize).min(BUILD_SHARE_BYTES);
    share
        .saturating_sub(input)
        .max(least_workspace(rows, columns))
}

/// The least workspace a build of `rows` rows of `columns` columns needs
pub fn least_workspace(rows: usize, columns: usize) -> usize {
    MIN_WORKSPACE_BYTES + columns * (metadata_allowance(rows) + COLUMN_SLOT_BYTES)
}

fn integer_keys(data: &super::column::ColumnData) -> Option<(&[i64], &[bool])> {
    match data {
        super::column::ColumnData::Int64 { values, nulls }
        | super::column::ColumnData::TimestampNanos { values, nulls } => Some((values, nulls)),
        _ => None,
    }
}

/// The `(position, key)` pairs of an integer or timestamp column of
/// `volume`, null rows left out: a column already decoded is read in
/// place, else one row group at a time through the group cache. A decode
/// or type error ends the pairs and is left in `failure`, so the build's
/// caller can tell a short input from a complete one.
fn volume_pairs<'a>(
    volume: &'a super::writer::FrozenVolume,
    column: usize,
    failure: &'a std::cell::Cell<Option<std::io::Error>>,
) -> impl Iterator<Item = (u32, i64)> + 'a {
    let rows = volume.meta.row_count;
    let fail = move |what: &str| {
        failure.set(Some(invalid(what)));
        None
    };
    let pairs: Box<dyn Iterator<Item = (u32, i64)> + 'a> = match volume.columns.resident(column) {
        Some(data) => match integer_keys(data) {
            Some((values, nulls)) => Box::new(
                (0..rows.min(values.len()))
                    .filter(move |&i| !nulls.get(i).copied().unwrap_or(false))
                    .map(move |i| (i as u32, values[i])),
            ),
            None => Box::new(std::iter::from_fn(move || fail("column is not indexable"))),
        },
        None => match volume.columns.compressed_store() {
            None => Box::new(std::iter::from_fn(move || {
                fail("column data is not loaded")
            })),
            Some(store) => Box::new(GroupPairs {
                store,
                column,
                rows,
                group_size: store.group_size().max(1),
                current: None,
                next: 0,
                failure,
            }),
        },
    };
    Bounded { pairs, rows }
}

/// The pairs of a column still in its compressed form, holding one
/// decoded group at a time
struct GroupPairs<'a> {
    store: &'a super::writer::CompressedBlockStore,
    column: usize,
    rows: usize,
    group_size: usize,
    current: Option<(usize, Arc<super::column::ColumnData>)>,
    next: usize,
    failure: &'a std::cell::Cell<Option<std::io::Error>>,
}

impl Iterator for GroupPairs<'_> {
    type Item = (u32, i64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.rows {
            let group = self.next / self.group_size;
            if self.current.as_ref().is_none_or(|(g, _)| *g != group) {
                // The group done with goes before the next is decoded, so
                // one group's decode is what the input holds at most
                self.current = None;
                match self.store.group_column(self.column, group) {
                    Ok(decoded) => self.current = Some((group, decoded)),
                    Err(error) => {
                        self.failure.set(Some(error));
                        self.next = self.rows;
                        return None;
                    }
                }
            }
            let position = self.next;
            self.next += 1;
            let decoded = &self.current.as_ref()?.1;
            let Some((values, nulls)) = integer_keys(decoded) else {
                self.failure.set(Some(invalid("column is not indexable")));
                self.next = self.rows;
                return None;
            };
            let i = position - group * self.group_size;
            if nulls.get(i).copied().unwrap_or(false) {
                continue;
            }
            let Some(&key) = values.get(i) else {
                self.failure
                    .set(Some(invalid("row group shorter than its rows")));
                self.next = self.rows;
                return None;
            };
            return Some((position as u32, key));
        }
        self.current = None;
        None
    }
}

/// What decoding one row group of `column` allocates at most, beyond the
/// build's own workspace: the compressed block read, its decompressed
/// bytes and the decoded column, one group at a time. Nothing for a
/// column already decoded.
fn decode_allowance(volume: &super::writer::FrozenVolume, column: usize) -> usize {
    if volume.columns.resident(column).is_some() {
        return 0;
    }
    let Some(store) = volume.columns.compressed_store() else {
        return 0;
    };
    let decompressed = store
        .decompressed_lens()
        .get(column)
        .and_then(|groups| groups.iter().max().copied())
        .unwrap_or(0);
    // The compressed block is at most the LZ4 bound of its bytes
    let compressed = decompressed + decompressed / 255 + 16;
    compressed
        + decompressed
        + store.group_size() * (std::mem::size_of::<i64>() + std::mem::size_of::<bool>())
}

/// An iterator with the volume's row count as its upper size bound, so a
/// build reserves the page metadata it needs rather than a quarter of its
/// workspace
struct Bounded<'a> {
    pairs: Box<dyn Iterator<Item = (u32, i64)> + 'a>,
    rows: usize,
}

impl Iterator for Bounded<'_> {
    type Item = (u32, i64);
    fn next(&mut self) -> Option<Self::Item> {
        self.pairs.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.rows))
    }
}

/// Builds the side file of the volume at `volume_path` for `identities`,
/// each a physical column and the identity of the index it is built for, and
/// opens it. The build's workspace and the decode of its input are
/// admitted together against the builds budget. `Ok(None)` leaves the
/// volume uncovered: no identities, an admission the budget refused
/// (counted by the ledger), or a side file that could not be written or
/// read back (counted in `BUILDS_FAILED`); each is logged and leaves no
/// file. An error reading the volume itself is the caller's failure, not
/// the side file's, and comes back as `Err`.
pub fn build_side_for(
    volume: &super::writer::FrozenVolume,
    volume_path: &Path,
    file_id: u64,
    identities: &[(usize, u64)],
) -> std::io::Result<Option<Arc<IndexFile>>> {
    match stage_side_for(volume, volume_path, file_id, identities, None)? {
        SideBuild::Built(staged) => match staged.publish(&side_path(volume_path), file_id) {
            Ok(side) => Ok(Some(side)),
            Err(error) => {
                BUILDS_FAILED.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "Warning: side index {:?} unreadable after its build: {error}",
                    side_path(volume_path)
                );
                retire_side_of(volume_path);
                Ok(None)
            }
        },
        SideBuild::Refused | SideBuild::Failed => Ok(None),
    }
}

/// What a build came to: the staged file, or the budget's refusal, or a
/// failure that was logged and counted
pub enum SideBuild {
    Built(StagedSide),
    Refused,
    Failed,
}

/// A side file built and checked but not yet at its volume's path: it
/// lives in its build directory beside the volume until `publish` renames
/// it over the path, and goes with the directory when dropped before
pub struct StagedSide {
    temps: TempFiles,
    file_id: u64,
}

impl StagedSide {
    /// The staged file's directory, wherever the volume's directory is
    /// now: a table rename moved it along
    fn built_at(&self, volume_path: &Path) -> PathBuf {
        let own = self.temps.tmp_path();
        if own.exists() {
            return own;
        }
        match (volume_path.parent(), self.temps.build_dir.file_name()) {
            (Some(dir), Some(name)) => dir.join(name).join("out"),
            _ => own,
        }
    }

    /// Reads the staged file, to check what it covers before it is published
    pub fn open(&self, volume_path: &Path) -> std::io::Result<IndexFile> {
        let path = self.built_at(volume_path);
        IndexFile::open(&path, self.file_id)
    }

    /// Renames the staged file over `side` and opens it through the
    /// path's shared handle: the holders of the generation there before
    /// keep their own bytes
    pub fn publish(self, side: &Path, file_id: u64) -> std::io::Result<Arc<IndexFile>> {
        let volume_path = side.with_extension("vol");
        let built = self.built_at(&volume_path);
        std::fs::rename(&built, side)?;
        if let Some(dir) = built.parent() {
            let _ = std::fs::remove_dir(dir);
        }
        let mut temps = self.temps;
        temps.keep = true;
        drop(temps);
        let handle = VolumeFile::shared(side);
        IndexFile::open_through(&handle, file_id).map(Arc::new)
    }
}

/// Builds the side file of `volume` beside it, staged: admitted through the
/// builds budget as the seal's build is, and when `cap` is given, sized to
/// fit under it (a backfill takes at most its share of the budget, and a
/// build that does not fit even at its least is refused)
pub fn stage_side_for(
    volume: &super::writer::FrozenVolume,
    volume_path: &Path,
    file_id: u64,
    identities: &[(usize, u64)],
    cap: Option<usize>,
) -> std::io::Result<SideBuild> {
    if identities.is_empty() {
        return Ok(SideBuild::Refused);
    }
    let side = side_path(volume_path);
    let input = identities
        .iter()
        .map(|&(column, _)| decode_allowance(volume, column))
        .max()
        .unwrap_or(0);
    let mut workspace = workspace_for(volume.meta.row_count, identities.len(), input);
    if let Some(cap) = cap {
        let least = least_workspace(volume.meta.row_count, identities.len());
        if input + least > cap {
            INDEX_BUILDS.count_refused();
            return Ok(SideBuild::Refused);
        }
        workspace = workspace.min(cap - input).max(least);
    }
    let Some(_input_admitted) = INDEX_BUILDS.try_charge(input) else {
        eprintln!(
            "Warning: side index {:?} not built: the budget refused its input's decode",
            side
        );
        return Ok(SideBuild::Refused);
    };
    let failure = std::cell::Cell::new(None);
    let columns = identities
        .iter()
        .map(|&(column, identity)| ColumnInput {
            column: column as u32,
            identity,
            pairs: Box::new(volume_pairs(volume, column, &failure)),
        })
        .collect();
    let built = build_side_file_staged(&side, next_generation(), columns, workspace);
    if let Some(error) = failure.take() {
        // The volume could not be read: whatever was written goes
        return Err(error);
    }
    match built {
        Ok((_, temps)) => Ok(SideBuild::Built(StagedSide { temps, file_id })),
        Err(error) if is_refused(&error) => {
            eprintln!("Warning: side index {:?} not built: {error}", side);
            Ok(SideBuild::Refused)
        }
        Err(error) => {
            BUILDS_FAILED.fetch_add(1, Ordering::Relaxed);
            eprintln!("Warning: side index {:?} failed: {error}", side);
            Ok(SideBuild::Failed)
        }
    }
}

/// The side file beside `volume_path` at reopen, if there is one it can
/// read. One it cannot read stays where it is, logged: a failed open
/// does not show the file is bad, and the volume is uncovered until the
/// next compaction rewrites it
pub fn open_side_for(volume_path: &Path, file_id: u64) -> Option<Arc<IndexFile>> {
    let side = side_path(volume_path);
    if !side.exists() {
        return None;
    }
    let handle = VolumeFile::shared(&side);
    match IndexFile::open_through(&handle, file_id) {
        Ok(file) => Some(Arc::new(file)),
        Err(error) => {
            eprintln!(
                "Warning: side index {:?} unavailable, the volume is uncovered: {error}",
                side
            );
            None
        }
    }
}

/// Retires the side file beside `volume_path`, if any: it goes once its
/// last holder lets go
pub fn retire_side_of(volume_path: &Path) {
    let side = side_path(volume_path);
    match VolumeFile::retire_path(&side) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => eprintln!("Warning: side index {:?} not retired: {error}", side),
    }
}

/// Whether any column of `side` still stands for an index the catalog
/// has now, `current` being the columns and identities it covers
pub fn still_covers(side: &IndexFile, current: &[(usize, u64)]) -> bool {
    current
        .iter()
        .any(|&(column, identity)| side.covers(column, identity))
}

/// Whether `side` covers every column and identity in `wanted`: what a
/// backfill asks of a volume's file before leaving it alone
pub fn covers_all(side: &IndexFile, wanted: &[(usize, u64)]) -> bool {
    wanted
        .iter()
        .all(|&(column, identity)| side.covers(column, identity))
}

// =============================================================================
// The query path's side of the contract: what a reader counts, and the
// walk a scanner carries
// =============================================================================

/// Counters of the reads through side files, for `PRAGMA INDEX_READ_STATS`
/// and the tests; process-wide.
pub struct ReadCounters {
    /// Volumes whose side file answered a probe (an empty answer included)
    pub probes: AtomicU64,
    /// Probes whose answer was empty, so the volume was not read at all
    pub misses: AtomicU64,
    /// Candidate positions the probes named
    pub candidates: AtomicU64,
    /// Rows produced from candidates, after every visibility rule
    pub rows: AtomicU64,
    /// Windows walked
    pub windows: AtomicU64,
    /// Volumes served by the scan because the working reservation was refused
    pub refused: AtomicU64,
    /// Volumes served by the scan because no side file column stands for
    /// the index (none attached, or another identity)
    pub ineligible: AtomicU64,
    /// Volumes served by the scan because the candidates were too many
    pub cost_scans: AtomicU64,
    /// Small volumes served by the scan because their pages are not
    /// resident and the cache has no room for them without an eviction
    pub page_scans: AtomicU64,
    /// Metadata-only volumes reloaded because a probe had candidates
    pub reloads: AtomicU64,
}

pub static READS: ReadCounters = ReadCounters {
    probes: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    candidates: AtomicU64::new(0),
    rows: AtomicU64::new(0),
    windows: AtomicU64::new(0),
    refused: AtomicU64::new(0),
    ineligible: AtomicU64::new(0),
    cost_scans: AtomicU64::new(0),
    page_scans: AtomicU64::new(0),
    reloads: AtomicU64::new(0),
};

impl ReadCounters {
    pub fn count(&self, counter: &AtomicU64, by: u64) {
        counter.fetch_add(by, Ordering::Relaxed);
    }

    /// The counters by name, in a fixed order
    pub fn snapshot(&self) -> [(&'static str, u64); 10] {
        [
            ("probes", self.probes.load(Ordering::Relaxed)),
            ("misses", self.misses.load(Ordering::Relaxed)),
            ("candidates", self.candidates.load(Ordering::Relaxed)),
            ("rows", self.rows.load(Ordering::Relaxed)),
            ("windows", self.windows.load(Ordering::Relaxed)),
            ("refused", self.refused.load(Ordering::Relaxed)),
            ("ineligible", self.ineligible.load(Ordering::Relaxed)),
            ("cost_scans", self.cost_scans.load(Ordering::Relaxed)),
            ("page_scans", self.page_scans.load(Ordering::Relaxed)),
            ("reloads", self.reloads.load(Ordering::Relaxed)),
        ]
    }
}

/// Positions a side file walk yields per window
pub const SIDE_WINDOW: usize = 4096;

/// A reader's working reservation for a window: the window, a raw page and
/// the parsed key and position pages
pub fn reader_bytes(window: usize) -> usize {
    window.max(1) * POS_ENTRY
        + PAGE_BYTES
        + KEYS_PER_PAGE * (std::mem::size_of::<i64>() + std::mem::size_of::<u64>())
        + POSITIONS_PER_PAGE * std::mem::size_of::<u32>()
}

/// A walk decided for a volume: the side file, the physical column and the
/// position index range the probe found. The reader and its working
/// reservation are taken when the walk starts (`walk`), not when the
/// decision is made, so volumes waiting their turn hold nothing.
pub struct SidePlan {
    file: Arc<IndexFile>,
    column: usize,
    range: (u64, u64),
}

impl SidePlan {
    pub fn new(file: Arc<IndexFile>, column: usize, range: (u64, u64)) -> Self {
        Self {
            file,
            column,
            range,
        }
    }

    /// Candidates the walk will name
    pub fn candidates(&self) -> u64 {
        self.range.1 - self.range.0
    }

    /// Starts the walk: takes the reader's working reservation now. Refused
    /// (`is_refused`), the volume goes through the scan.
    pub fn walk(&self, window: usize) -> std::io::Result<SideWalk> {
        let mut reader = self.file.reader(self.column, window)?;
        reader.walk(self.range);
        Ok(SideWalk { reader, at: 0 })
    }
}

/// The positions a reader's walk names, taken one at a time by a scanner
/// or a collect loop from the reader's own window: windows are pulled as
/// they are consumed, so a LIMIT that is satisfied early leaves the later
/// windows unread, and nothing is copied out of the reservation.
pub struct SideWalk {
    reader: Reader,
    at: usize,
}

impl SideWalk {
    /// The next candidate position, None at the end; a page that fails to
    /// read is the walk's error and the walk stays where it was
    pub fn next_position(&mut self) -> std::io::Result<Option<usize>> {
        if self.at >= self.reader.window().len() {
            match self.reader.next_window() {
                Ok(Some(window)) if !window.is_empty() => {
                    self.at = 0;
                    READS.count(&READS.windows, 1);
                }
                Ok(_) => return Ok(None),
                Err(error) => {
                    // The reader's window is empty now; so is this one
                    self.at = 0;
                    return Err(error);
                }
            }
        }
        let position = self.reader.window()[self.at] as usize;
        self.at += 1;
        Ok(Some(position))
    }

    /// Candidates left, including the current window's rest
    pub fn remaining(&self) -> u64 {
        self.reader.remaining() + (self.reader.window().len() - self.at) as u64
    }
}

/// Discards a side file built for a volume whose index definitions
/// changed before it was published: the volume is uncovered
pub fn discard_side(side: Arc<IndexFile>) {
    SIDES_DISCARDED.fetch_add(1, Ordering::Relaxed);
    side.handle().retire();
    drop(side);
}

#[cfg(test)]
mod tests {
    use super::*;

    static SERIAL: Mutex<()> = Mutex::new(());

    /// A counting allocator for the tests of this binary: live and peak
    /// bytes per thread, so a build's or a lookup's allocations are
    /// measured on the thread that makes them, whatever other tests do
    #[cfg(not(feature = "mimalloc"))]
    mod counting {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        pub struct Counting;

        thread_local! {
            static LIVE: Cell<usize> = const { Cell::new(0) };
            static PEAK: Cell<usize> = const { Cell::new(0) };
        }

        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = LIVE.try_with(|live| {
                    let now = live.get() + layout.size();
                    live.set(now);
                    let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
                });
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                let _ = LIVE.try_with(|live| live.set(live.get().saturating_sub(layout.size())));
                unsafe { System.dealloc(ptr, layout) }
            }
        }

        #[global_allocator]
        static COUNTING: Counting = Counting;

        /// Starts the thread's peak from its live bytes now
        pub fn mark() -> usize {
            LIVE.with(|live| {
                let now = live.get();
                PEAK.with(|peak| peak.set(now));
                now
            })
        }

        /// The most bytes live on this thread since `mark`, above it
        pub fn peak_since(mark: usize) -> usize {
            PEAK.with(|peak| peak.get().saturating_sub(mark))
        }
    }

    /// Bytes a build or a lookup may allocate beyond its ledger: the
    /// strings of file names and open calls, and the iterator's box
    #[cfg(not(feature = "mimalloc"))]
    const ALLOC_SLACK: usize = 32 * 1024;

    fn build(path: &Path, pairs: Vec<(u32, i64)>, workspace: usize) -> BuildReport {
        build_side_file(
            path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(pairs.into_iter()),
            }],
            workspace,
        )
        .unwrap()
    }

    fn all_positions(file: &IndexFile, range: (u64, u64), window: usize) -> Vec<u32> {
        let mut cursor = file.cursor(1, range, window).unwrap();
        let mut out = Vec::new();
        while let Some(w) = cursor.next_window().unwrap() {
            assert!(w.windows(2).all(|p| p[0] <= p[1]), "a window is sorted");
            assert!(w.len() <= window);
            out.extend_from_slice(w);
        }
        out
    }

    fn leftovers(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".build-") || n.ends_with(".tmp"))
            .collect();
        names.sort();
        names
    }

    /// Positions `0..rows` with scattered keys, as a streaming iterator
    fn scattered(rows: u32, modulus: u64) -> impl Iterator<Item = (u32, i64)> {
        (0..rows).map(move |p| (p, ((p as u64 * 104_729) % modulus) as i64))
    }

    /// The same, without an upper size bound
    struct Unbounded<I: Iterator>(I);

    impl<I: Iterator> Iterator for Unbounded<I> {
        type Item = I::Item;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (0, None)
        }
    }

    const WORKSPACE: usize = 64 * 1024 * 1024;

    #[test]
    fn keys_map_to_their_positions_through_pages() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        // 50,000 rows, key = row % 977 so every key has about 51 positions
        let pairs: Vec<(u32, i64)> = (0..50_000u32).map(|p| (p, (p % 977) as i64)).collect();
        let report = build(&path, pairs.clone(), WORKSPACE);
        assert_eq!(report.keys, 977);
        assert_eq!(report.positions, 50_000);
        assert_eq!(report.runs_spilled, 0);
        assert!(report.pos_pages >= 50_000 / POSITIONS_PER_PAGE);
        let file = IndexFile::open(&path, 1).unwrap();
        for key in [0i64, 1, 500, 976] {
            let range = file.equal(1, key).unwrap().unwrap();
            let want: Vec<u32> = pairs
                .iter()
                .filter(|(_, k)| *k == key)
                .map(|(p, _)| *p)
                .collect();
            assert_eq!(all_positions(&file, range, 7), want, "key {key}");
        }
        assert_eq!(file.equal(1, 977).unwrap(), None);
        assert_eq!(file.equal(1, -1).unwrap(), None);
        let range = file.range(1, 100, 103).unwrap();
        let mut want: Vec<u32> = pairs
            .iter()
            .filter(|(_, k)| (100..=103).contains(k))
            .map(|(p, _)| *p)
            .collect();
        want.sort_unstable();
        let mut got = all_positions(&file, range, 1000);
        got.sort_unstable();
        assert_eq!(got, want);
        assert!(file.candidate_bound(1, 100, 103).unwrap() >= want.len() as u64);
        assert_eq!(file.range(1, 2000, 3000).unwrap(), (0, 0));
        assert_eq!(file.candidate_bound(1, 2000, 3000).unwrap(), 0);
    }

    #[test]
    fn a_key_with_a_million_positions_spans_pages_and_a_window_holds_a_bounded_slice() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let n = 1_000_000u32;
        let pairs: Vec<(u32, i64)> = (0..n).map(|p| (p, 42)).chain([(n, 43)]).collect();
        let report = build(&path, pairs, WORKSPACE);
        assert_eq!(report.keys, 2);
        assert!(
            report.pos_pages > 100,
            "{} position pages",
            report.pos_pages
        );
        INDEX_PAGES.clear();
        let baseline = INDEX_PAGES.stats().charged_bytes;
        let file = IndexFile::open(&path, 2).unwrap();
        let range = file.equal(1, 42).unwrap().unwrap();
        assert_eq!(range, (0, n as u64));
        // Eight pages of budget over the directory: the walk over 4 MB of
        // positions must fit in it, holding one page and one window at a
        // time; the high-water mark sees every charge, not the state
        // between calls
        let budget = INDEX_PAGES.stats().charged_bytes + 8 * PAGE_BYTES;
        INDEX_PAGES.set_budget_bytes(budget as u64);
        INDEX_PAGES.reset_peak();
        let over_before = INDEX_PAGES.stats().over_budget;
        let mut cursor = file.cursor(1, range, 4096).unwrap();
        let mut seen = 0u64;
        while let Some(w) = cursor.next_window().unwrap() {
            assert_eq!(w[0] as u64, seen);
            seen += w.len() as u64;
        }
        assert_eq!(seen, n as u64);
        let peak = INDEX_PAGES.stats().peak_bytes;
        assert!(peak <= budget, "peak charged {peak} within budget {budget}");
        assert_eq!(
            INDEX_PAGES.stats().over_budget,
            over_before,
            "no load went over budget"
        );
        assert_eq!(file.equal(1, 43).unwrap(), Some((n as u64, n as u64 + 1)));
        drop(cursor);
        drop(file);
        INDEX_PAGES.clear();
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
        assert_eq!(INDEX_PAGES.stats().charged_bytes, baseline);
    }

    #[test]
    fn a_page_is_charged_at_the_capacity_of_its_vectors_and_its_raw_bytes_only_while_parsed() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let pairs: Vec<(u32, i64)> = (0..100_000u32).map(|p| (p, p as i64)).collect();
        build(&path, pairs, WORKSPACE);
        INDEX_PAGES.clear();
        let baseline = INDEX_PAGES.stats().charged_bytes;
        let directories = INDEX_DIRECTORIES.stats().charged_bytes;
        let file = IndexFile::open(&path, 7).unwrap();
        assert_eq!(
            INDEX_DIRECTORIES.stats().charged_bytes - directories,
            file.directory().bytes(),
            "the directory is charged to its own ledger"
        );
        assert_eq!(INDEX_PAGES.stats().charged_bytes, baseline);
        INDEX_PAGES.reset_peak();
        let key_page = INDEX_PAGES.load(&file, 1, PageKind::Keys, 0).unwrap();
        let PageContent::Keys { keys, ends } = key_page.content() else {
            panic!("key page")
        };
        assert_eq!(keys.len(), KEYS_PER_PAGE, "a full key page");
        assert_eq!(
            key_page.bytes(),
            keys.capacity() * 8 + ends.capacity() * 8,
            "charged at the vectors' capacity, not the encoded length"
        );
        assert_eq!(
            INDEX_PAGES.stats().charged_bytes,
            baseline + key_page.bytes(),
            "the raw bytes were released after the parse"
        );
        assert_eq!(
            INDEX_PAGES.stats().peak_bytes,
            baseline + key_page.bytes() + PAGE_BYTES,
            "the raw page was charged while it was parsed"
        );
        let pos_page = INDEX_PAGES.load(&file, 1, PageKind::Positions, 0).unwrap();
        let PageContent::Positions(positions) = pos_page.content() else {
            panic!("position page")
        };
        assert_eq!(pos_page.bytes(), positions.capacity() * 4);
        drop(key_page);
        drop(pos_page);
        drop(file);
        INDEX_PAGES.clear();
        assert_eq!(INDEX_PAGES.stats().charged_bytes, baseline);
    }

    #[test]
    fn a_build_beyond_its_workspace_spills_runs_merges_them_upward_and_answers_the_same() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let small = dir.path().join("small.sidx");
        let large = dir.path().join("large.sidx");
        let pairs: Vec<(u32, i64)> = scattered(400_000, 20_011).collect();
        // The minimum plus this input's page metadata
        let workspace = MIN_WORKSPACE_BYTES + metadata_allowance(400_000);
        let a = build(&small, pairs.clone(), workspace);
        let b = build(&large, pairs, WORKSPACE);
        assert!(a.runs_spilled > MAX_FAN_IN, "{} runs", a.runs_spilled);
        assert!(a.merge_passes >= 2, "{} passes", a.merge_passes);
        assert!(a.max_fan_in <= MAX_FAN_IN);
        assert!(
            a.max_live_runs <= a.max_fan_in * 12,
            "{} runs alive at once with a fan-in of {}",
            a.max_live_runs,
            a.max_fan_in
        );
        assert!(
            a.workspace_peak <= workspace,
            "peak {} within the workspace {}",
            a.workspace_peak,
            workspace
        );
        assert_eq!(b.runs_spilled, 0);
        assert_eq!(b.merge_passes, 0);
        assert_eq!(
            std::fs::read(&small).unwrap()[HEADER_LEN as usize..],
            std::fs::read(&large).unwrap()[HEADER_LEN as usize..]
        );
        assert!(leftovers(dir.path()).is_empty());
        let err = build_side_file(
            &dir.path().join("tiny.sidx"),
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(std::iter::empty()),
            }],
            MIN_WORKSPACE_BYTES - 1,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    /// The three shapes the review measured over the workspace, held to it
    /// by the build's own ledger and checked against the allocator.
    #[test]
    fn a_workspace_bounds_the_whole_build_for_every_input_shape() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();

        // 1. One column of 1.2 million unique keys under the minimum: more
        //    than a thousand runs, merged upward as they come
        let path = dir.path().join("many-runs.sidx");
        let workspace = MIN_WORKSPACE_BYTES + metadata_allowance(1_200_000);
        #[cfg(not(feature = "mimalloc"))]
        let mark = counting::mark();
        let report = build_side_file(
            &path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(scattered(1_200_000, 1_200_001)),
            }],
            workspace,
        )
        .unwrap();
        #[cfg(not(feature = "mimalloc"))]
        {
            let peak = counting::peak_since(mark);
            assert!(
                peak <= workspace + ALLOC_SLACK,
                "many runs: allocator peak {peak} within {workspace} plus slack"
            );
        }
        assert!(report.runs_spilled > 1000, "{} runs", report.runs_spilled);
        assert!(report.workspace_peak <= workspace);
        assert!(
            report.max_live_runs <= report.max_fan_in * 12,
            "{} runs alive with fan-in {}",
            report.max_live_runs,
            report.max_fan_in
        );
        let file = IndexFile::open(&path, 10).unwrap();
        assert_eq!(
            file.equal(1, ((777u64 * 104_729) % 1_200_001) as i64)
                .unwrap()
                .map(|(s, e)| e - s),
            Some(1)
        );

        // 2. Three columns of 400,000 rows under 512 KiB: earlier columns'
        //    metadata counts against the later ones
        let path = dir.path().join("three.sidx");
        let workspace = 512 * 1024;
        #[cfg(not(feature = "mimalloc"))]
        let mark = counting::mark();
        let report = build_side_file(
            &path,
            next_generation(),
            (1..=3)
                .map(|c| ColumnInput {
                    column: c,
                    identity: 0,
                    pairs: Box::new(scattered(400_000, 65_521 + c as u64)),
                })
                .collect(),
            workspace,
        )
        .unwrap();
        #[cfg(not(feature = "mimalloc"))]
        {
            let peak = counting::peak_since(mark);
            assert!(
                peak <= workspace + ALLOC_SLACK,
                "three columns: allocator peak {peak} within {workspace} plus slack"
            );
        }
        assert!(
            report.workspace_peak <= workspace,
            "{}",
            report.workspace_peak
        );
        let file = IndexFile::open(&path, 11).unwrap();
        for c in 1..=3usize {
            assert!(file
                .equal(c, ((5u64 * 104_729) % (65_521 + c as u64)) as i64)
                .unwrap()
                .is_some());
        }

        // 3. An input without a size bound: a quarter of the workspace is
        //    kept for its metadata; the build either fits or refuses, and
        //    never allocates past the workspace
        let path = dir.path().join("unbounded.sidx");
        #[cfg(not(feature = "mimalloc"))]
        let mark = counting::mark();
        let result = build_side_file(
            &path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(Unbounded(scattered(4_194_304, 4_194_301))),
            }],
            MIN_WORKSPACE_BYTES,
        );
        #[cfg(not(feature = "mimalloc"))]
        {
            let peak = counting::peak_since(mark);
            assert!(
                peak <= MIN_WORKSPACE_BYTES + ALLOC_SLACK,
                "unbounded: allocator peak {peak} within {MIN_WORKSPACE_BYTES} plus slack"
            );
        }
        match result {
            Ok(report) => assert!(report.workspace_peak <= MIN_WORKSPACE_BYTES),
            Err(err) => {
                assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
                assert!(!path.exists(), "nothing was published");
            }
        }
        assert!(
            leftovers(dir.path()).is_empty(),
            "{:?}",
            leftovers(dir.path())
        );
        // The same input with its bound given builds under a workspace that
        // holds its metadata
        let hinted = build_side_file(
            &path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(scattered(4_194_304, 4_194_301)),
            }],
            MIN_WORKSPACE_BYTES + metadata_allowance(4_194_304),
        )
        .unwrap();
        assert!(hinted.workspace_peak <= MIN_WORKSPACE_BYTES + metadata_allowance(4_194_304));
    }

    #[test]
    fn a_failed_build_touches_nothing_but_its_own_directory() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alpha.sidx");
        // A published file whose name looks like a build's leftovers, and
        // a stray file with the target's name as a prefix
        let lookalike = dir.path().join("alpha.sidx.run-backup.sidx");
        build(
            &lookalike,
            (0..500u32).map(|p| (p, p as i64)).collect(),
            WORKSPACE,
        );
        let stray = dir.path().join("alpha.sidx.tmp");
        std::fs::write(&stray, b"not ours").unwrap();
        SPILLS.with(|c| c.set(0));
        FAIL_SPILL.with(|f| f.set(Some(1)));
        let result = build_side_file(
            &path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(scattered(100_000, 100_003)),
            }],
            MIN_WORKSPACE_BYTES + metadata_allowance(100_000),
        );
        FAIL_SPILL.with(|f| f.set(None));
        assert!(result.is_err());
        assert!(!path.exists());
        assert!(
            IndexFile::open(&lookalike, 12)
                .unwrap()
                .equal(1, 7)
                .unwrap()
                .is_some(),
            "the published lookalike still answers"
        );
        assert_eq!(std::fs::read(&stray).unwrap(), b"not ours");
        assert!(
            std::fs::read_dir(dir.path()).unwrap().all(|e| !e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".build-")),
            "the build's own directory is gone"
        );
        // A directory already standing where the build's would be is not
        // the build's: the build refuses and leaves it
        let generation = next_generation();
        let taken = dir.path().join(format!(
            "alpha.sidx.build-{}-{generation:x}",
            std::process::id()
        ));
        std::fs::create_dir(&taken).unwrap();
        std::fs::write(taken.join("out"), b"someone else's").unwrap();
        let err = build_side_file(
            &path,
            generation,
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(std::iter::empty()),
            }],
            WORKSPACE,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(taken.join("out")).unwrap(), b"someone else's");
    }

    #[test]
    fn a_relative_target_without_a_parent_builds_beside_itself_and_cleans_up() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        // A bare name resolves to the current directory; the build is
        // refused after its directory was created, and nothing stays
        let name = format!("relative-{}.sidx", std::process::id());
        let path = Path::new(&name);
        let result = build_side_file(
            path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(Unbounded(scattered(4_194_304, 4_194_301))),
            }],
            MIN_WORKSPACE_BYTES,
        );
        assert!(result.is_err(), "refused for its metadata");
        let stale: Vec<String> = std::fs::read_dir(".")
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with(&name))
            .collect();
        assert!(stale.is_empty(), "{stale:?}");
        // And a bare name builds and is found where it was asked for
        build(
            path,
            (0..100u32).map(|p| (p, p as i64)).collect(),
            WORKSPACE,
        );
        assert!(IndexFile::open(path, 13)
            .unwrap()
            .equal(1, 7)
            .unwrap()
            .is_some());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn the_directory_s_columns_are_taken_from_the_budget_before_they_are_allocated() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("many-columns.sidx");
        let empty = || {
            (1..=2000u32)
                .map(|c| ColumnInput {
                    column: c,
                    identity: 0,
                    pairs: Box::new(std::iter::empty()),
                })
                .collect::<Vec<_>>()
        };
        // Under the minimum the 2,000 slots do not fit beside the buffers:
        // refused before anything is allocated or created
        #[cfg(not(feature = "mimalloc"))]
        let mark = counting::mark();
        let err =
            build_side_file(&path, next_generation(), empty(), MIN_WORKSPACE_BYTES).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
        #[cfg(not(feature = "mimalloc"))]
        assert!(
            counting::peak_since(mark)
                <= 2000 * std::mem::size_of::<ColumnInput<'_>>() + ALLOC_SLACK,
            "nothing beyond the inputs was allocated"
        );
        assert!(leftovers(dir.path()).is_empty());
        // With room for the slots they build within the budget
        let slots = 2000 * COLUMN_SLOT_BYTES;
        let workspace = MIN_WORKSPACE_BYTES + slots + 2000 * metadata_allowance(0);
        #[cfg(not(feature = "mimalloc"))]
        let mark = counting::mark();
        let report = build_side_file(&path, next_generation(), empty(), workspace).unwrap();
        #[cfg(not(feature = "mimalloc"))]
        {
            let peak = counting::peak_since(mark);
            assert!(
                peak <= workspace + 2000 * std::mem::size_of::<ColumnInput<'_>>() + ALLOC_SLACK,
                "many columns: allocator peak {peak} within {workspace} plus the inputs and slack"
            );
        }
        assert!(report.workspace_peak <= workspace);
        let file = IndexFile::open(&path, 14).unwrap();
        assert_eq!(file.directory().columns.len(), 2000);
    }

    #[test]
    fn a_fixed_workspace_holds_while_the_input_grows() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let workspace = 512 * 1024;
        for rows in [50_000u32, 200_000, 800_000] {
            let path = dir.path().join(format!("{rows}.sidx"));
            let report = build(&path, scattered(rows, 65_521).collect(), workspace);
            assert!(
                report.workspace_peak <= workspace,
                "{rows} rows: peak {} within {workspace}",
                report.workspace_peak
            );
            assert!(report.max_fan_in <= MAX_FAN_IN);
            let file = IndexFile::open(&path, 8).unwrap();
            let probe_key = (12_345u64 * 104_729) % 65_521;
            let (start, end) = file.equal(1, probe_key as i64).unwrap().unwrap();
            assert_eq!(
                (end - start) as usize,
                (0..rows)
                    .filter(|p| (*p as u64 * 104_729) % 65_521 == probe_key)
                    .count()
            );
        }
        assert!(leftovers(dir.path()).is_empty());
    }

    #[test]
    fn a_failed_spill_and_a_failed_finish_leave_no_temporary_file() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        build(
            &path,
            (0..1000u32).map(|p| (p, p as i64)).collect(),
            WORKSPACE,
        );
        let published = std::fs::read(&path).unwrap();
        // The second spill fails after the first succeeded
        SPILLS.with(|c| c.set(0));
        FAIL_SPILL.with(|f| f.set(Some(2)));
        let pairs: Vec<(u32, i64)> = (0..300_000u32).map(|p| (p, (p % 1000) as i64)).collect();
        let result = build_side_file(
            &path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(pairs.clone().into_iter()),
            }],
            MIN_WORKSPACE_BYTES + metadata_allowance(300_000),
        );
        FAIL_SPILL.with(|f| f.set(None));
        assert!(result.is_err(), "the second spill fails");
        assert!(
            leftovers(dir.path()).is_empty(),
            "{:?}",
            leftovers(dir.path())
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            published,
            "the published file is untouched"
        );

        // The final rename fails: the destination is a directory
        let blocked = dir.path().join("blocked.sidx");
        std::fs::create_dir(&blocked).unwrap();
        let result = build_side_file(
            &blocked,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(pairs.into_iter()),
            }],
            WORKSPACE,
        );
        assert!(result.is_err());
        assert!(
            leftovers(dir.path()).is_empty(),
            "{:?}",
            leftovers(dir.path())
        );
        INDEX_PAGES.clear();
    }

    #[test]
    fn a_corrupt_page_fails_alone_and_a_corrupt_directory_fails_the_open() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let pairs: Vec<(u32, i64)> = (0..30_000u32).map(|p| (p, (p % 5000) as i64)).collect();
        build(&path, pairs, WORKSPACE);
        let good = std::fs::read(&path).unwrap();
        let file = IndexFile::open(&path, 3).unwrap();
        let second_pos_page = file.directory().column(1).unwrap().pos_pages[1].offset as usize + 10;
        let mut bad = good.clone();
        bad[second_pos_page] ^= 0xff;
        std::fs::write(&path, &bad).unwrap();
        INDEX_PAGES.clear();
        // Key 1 lives in the first position page, key 1366 in the second
        let range = file.equal(1, 1).unwrap().unwrap();
        assert_eq!(
            all_positions(&file, range, 100).len(),
            6,
            "the first page is intact"
        );
        let range = file.equal(1, 1366).unwrap().unwrap();
        let mut cursor = file.cursor(1, range, 100).unwrap();
        let first = cursor.next_window().map(|w| w.map(|w| w.len()));
        let err = first.unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
        let mut bad_dir = good.clone();
        let len = bad_dir.len();
        bad_dir[len - FOOTER_LEN as usize - 3] ^= 0x01;
        std::fs::write(&path, &bad_dir).unwrap();
        let err = match IndexFile::open(&path, 3) {
            Err(err) => err,
            Ok(_) => panic!("a corrupt directory opened"),
        };
        assert!(err.to_string().contains("directory checksum"), "{err}");
        std::fs::write(&path, &good[..good.len() - 5]).unwrap();
        assert!(IndexFile::open(&path, 3).is_err());
    }

    /// A window that fails to read is no window: the walk stays where it
    /// began, reports the same remainder, and once the file is repaired
    /// the same walk yields every position once
    #[test]
    fn a_failed_window_is_retried_without_repeating_positions() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        build(
            &path,
            (0..20_000u32).map(|p| (p, 1i64)).collect(),
            WORKSPACE,
        );
        let file = Arc::new(IndexFile::open(&path, 3).unwrap());
        let good = std::fs::read(&path).unwrap();
        let mut bad = good.clone();
        let second_page = file.directory().column(1).unwrap().pos_pages[1].offset as usize + 10;
        bad[second_page] ^= 0xff;
        std::fs::write(&path, &bad).unwrap();
        INDEX_PAGES.clear();
        let mut walk = SidePlan::new(Arc::clone(&file), 1, (4096, 10_000))
            .walk(4096)
            .unwrap();
        // The pages come through the reader's own buffers, not the cache
        let charged = INDEX_PAGES.stats().charged_bytes;
        INDEX_PAGES.set_budget_bytes(charged as u64);
        let err = walk.next_position().unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
        assert_eq!(walk.remaining(), 5904, "the walk stayed where it began");
        std::fs::write(&path, &good).unwrap();
        let mut got = Vec::new();
        while let Some(position) = walk.next_position().unwrap() {
            got.push(position);
        }
        drop(walk);
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
        INDEX_PAGES.clear();
        assert_eq!(got.len(), 5904);
        let unique: std::collections::BTreeSet<usize> = got.iter().copied().collect();
        assert_eq!(unique.len(), 5904, "every position once");
        assert_eq!(got[0], 4096);
    }

    /// A refill that fails after a window was served: the walk reports the
    /// remainder from the window's end and, repaired, yields the rest once
    #[test]
    fn a_failed_refill_after_a_served_window_keeps_the_remainder() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        build(
            &path,
            (0..20_000u32).map(|p| (p, 1i64)).collect(),
            WORKSPACE,
        );
        let file = Arc::new(IndexFile::open(&path, 3).unwrap());
        let good = std::fs::read(&path).unwrap();
        let mut bad = good.clone();
        let second_page = file.directory().column(1).unwrap().pos_pages[1].offset as usize + 10;
        bad[second_page] ^= 0xff;
        INDEX_PAGES.clear();
        let mut walk = SidePlan::new(Arc::clone(&file), 1, (0, 20_000))
            .walk(4096)
            .unwrap();
        let charged = INDEX_PAGES.stats().charged_bytes;
        INDEX_PAGES.set_budget_bytes(charged as u64);
        let mut got = Vec::new();
        for _ in 0..4096 {
            got.push(walk.next_position().unwrap().unwrap());
        }
        assert_eq!(walk.remaining(), 15_904);
        std::fs::write(&path, &bad).unwrap();
        let err = walk.next_position().unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
        assert_eq!(
            walk.remaining(),
            15_904,
            "the failed refill changed nothing"
        );
        std::fs::write(&path, &good).unwrap();
        while let Some(position) = walk.next_position().unwrap() {
            got.push(position);
        }
        drop(walk);
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
        INDEX_PAGES.clear();
        assert_eq!(got.len(), 20_000);
        let unique: std::collections::BTreeSet<usize> = got.iter().copied().collect();
        assert_eq!(unique.len(), 20_000, "every position once");
    }

    #[test]
    fn a_holder_keeps_its_generation_after_the_file_was_replaced() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        build(
            &path,
            (0..1000u32).map(|p| (p, p as i64)).collect(),
            WORKSPACE,
        );
        let old = IndexFile::open(&path, 4).unwrap();
        build(
            &path,
            (0..1000u32).map(|p| (p, (p * 2) as i64)).collect(),
            WORKSPACE,
        );
        INDEX_PAGES.clear();
        // The holder of the old generation keeps its own bytes
        assert_eq!(old.equal(1, 5).unwrap(), Some((5, 6)));
        let new = IndexFile::open(&path, 4).unwrap();
        assert_ne!(new.generation(), old.generation());
        assert_eq!(new.equal(1, 5).unwrap(), None);
        assert!(new.equal(1, 10).unwrap().is_some());
        assert_eq!(old.equal(1, 5).unwrap(), Some((5, 6)));
    }

    #[test]
    fn a_held_page_stays_charged_after_its_eviction_and_the_budget_is_honoured_otherwise() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let pairs: Vec<(u32, i64)> = (0..200_000u32).map(|p| (p, (p % 40_000) as i64)).collect();
        build(&path, pairs, WORKSPACE);
        INDEX_PAGES.clear();
        let baseline = INDEX_PAGES.stats();
        let file = IndexFile::open(&path, 5).unwrap();
        // A budget of three pages above the baseline
        INDEX_PAGES.set_budget_bytes((baseline.charged_bytes + 3 * PAGE_BYTES) as u64);
        let held = INDEX_PAGES.load(&file, 1, PageKind::Positions, 0).unwrap();
        let held_bytes = held.bytes();
        let before = INDEX_PAGES.stats();
        for page in 1..12 {
            INDEX_PAGES
                .load(&file, 1, PageKind::Positions, page)
                .unwrap();
        }
        let after = INDEX_PAGES.stats();
        assert!(
            after.evictions > before.evictions,
            "pages were evicted to stay in budget"
        );
        assert!(
            after.charged_bytes as u64 <= after.budget_bytes,
            "charged {} within budget {}",
            after.charged_bytes,
            after.budget_bytes
        );
        assert!(
            after.charged_bytes >= baseline.charged_bytes + held_bytes,
            "the held page's bytes are still charged"
        );
        // Lowering the budget under the held page revokes nothing; a load
        // that cannot fit beside it is refused and counted, and answered
        // again once the budget allows
        let refused_before = after.refused;
        INDEX_PAGES.set_budget_bytes(baseline.charged_bytes as u64);
        let stats = INDEX_PAGES.stats();
        assert!(
            stats.charged_bytes >= baseline.charged_bytes + held_bytes,
            "a held page cannot be evicted"
        );
        let err = match INDEX_PAGES.load(&file, 1, PageKind::Positions, 20) {
            Err(err) => err,
            Ok(_) => panic!("a load over the budget with nothing to evict was granted"),
        };
        assert!(is_refused(&err), "{err}");
        assert_eq!(INDEX_PAGES.stats().refused, refused_before + 1);
        assert_eq!(
            INDEX_PAGES.stats().charged_bytes,
            stats.charged_bytes,
            "a refused load charged nothing"
        );
        INDEX_PAGES.set_budget_bytes((baseline.charged_bytes + 3 * PAGE_BYTES) as u64);
        INDEX_PAGES.load(&file, 1, PageKind::Positions, 20).unwrap();
        drop(held);
        drop(file);
        INDEX_PAGES.clear();
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
        assert_eq!(
            INDEX_PAGES.stats().charged_bytes,
            baseline.charged_bytes,
            "everything released"
        );
    }

    #[test]
    fn statistics_reservations_and_evictions_do_not_wait_on_each_other() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let pairs: Vec<(u32, i64)> = (0..200_000u32).map(|p| (p, (p % 40_000) as i64)).collect();
        build(&path, pairs, WORKSPACE);
        INDEX_PAGES.clear();
        let file = Arc::new(IndexFile::open(&path, 9).unwrap());
        INDEX_PAGES.set_budget_bytes((INDEX_PAGES.stats().charged_bytes + 2 * PAGE_BYTES) as u64);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut workers = Vec::new();
        for w in 0..4usize {
            let file = Arc::clone(&file);
            let done = Arc::clone(&done);
            workers.push(std::thread::spawn(move || {
                let mut i = 0usize;
                while !done.load(Ordering::Relaxed) {
                    match w % 2 {
                        0 => {
                            let _ = INDEX_PAGES.stats();
                            let _ = INDEX_PAGES.reserve(1);
                        }
                        _ => {
                            let page = (i * 7 + w) % 12;
                            // Refused under the two-page budget at times
                            let _ = INDEX_PAGES.load(&file, 1, PageKind::Positions, page);
                            if i.is_multiple_of(50) {
                                INDEX_PAGES.clear();
                            }
                        }
                    }
                    i += 1;
                }
            }));
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        done.store(true, Ordering::Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        for worker in workers {
            while !worker.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "a worker is still blocked after the stop: statistics, reservations and evictions wait on each other"
                );
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            worker.join().unwrap();
        }
        drop(file);
        INDEX_PAGES.clear();
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
    }

    #[test]
    fn a_build_is_refused_while_the_others_hold_the_builds_budget() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let baseline = INDEX_BUILDS.stats();
        INDEX_BUILDS.set_budget_bytes((baseline.charged_bytes + WORKSPACE) as u64);
        // Another build holds one byte of the budget: this one does not fit
        // whole, creates nothing and is counted
        let other = INDEX_BUILDS.try_charge(1).unwrap();
        let err = build_side_file(
            &path,
            next_generation(),
            vec![ColumnInput {
                column: 1,
                identity: 0,
                pairs: Box::new(scattered(1000, 97)),
            }],
            WORKSPACE,
        )
        .unwrap_err();
        assert!(is_refused(&err), "{err}");
        assert!(!path.exists());
        assert!(leftovers(dir.path()).is_empty());
        assert_eq!(INDEX_BUILDS.stats().refused, baseline.refused + 1);
        drop(other);
        // With the budget free it builds, holding its whole workspace
        // meanwhile and releasing it after
        INDEX_BUILDS.reset_peak();
        build(&path, scattered(1000, 97).collect(), WORKSPACE);
        assert_eq!(
            INDEX_BUILDS.stats().peak_bytes,
            baseline.charged_bytes + WORKSPACE
        );
        assert_eq!(INDEX_BUILDS.stats().charged_bytes, baseline.charged_bytes);
        INDEX_BUILDS.set_budget_bytes(DEFAULT_BUILD_BUDGET_BYTES);
    }

    #[test]
    fn a_column_records_the_identity_it_was_built_for() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        build_side_file(
            &path,
            next_generation(),
            vec![
                ColumnInput {
                    column: 1,
                    identity: 41,
                    pairs: Box::new(scattered(1000, 97)),
                },
                ColumnInput {
                    column: 2,
                    identity: 0,
                    pairs: Box::new(scattered(1000, 89)),
                },
            ],
            WORKSPACE,
        )
        .unwrap();
        let file = IndexFile::open(&path, 15).unwrap();
        assert!(file.covers(1, 41));
        assert!(!file.covers(1, 42), "another index's identity");
        assert!(!file.covers(3, 41), "a column the file has not");
        assert!(
            !file.covers(2, 0),
            "a column without an identity covers nothing"
        );
        // Every open of the same path reads through the one handle
        let again = IndexFile::open(&path, 15).unwrap();
        assert!(Arc::ptr_eq(file.handle(), again.handle()));
        // A file of the previous format is not opened: its columns carry
        // no identity, and the volume is uncovered until rewritten
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = match IndexFile::open(&path, 15) {
            Err(err) => err,
            Ok(_) => panic!("a VERSION 3 side file opened"),
        };
        assert!(err.to_string().contains("version"), "{err}");
    }

    /// A file-backed volume of `rows` integer rows in two columns, its
    /// blocks compressed on disk, so the side build decodes its input
    fn file_backed(dir: &Path, rows: usize) -> (super::super::writer::FrozenVolume, PathBuf) {
        use crate::core::{DataType, SchemaBuilder};
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("k", DataType::Integer, false, false)
            .build();
        let ids: Vec<i64> = (0..rows as i64).collect();
        let keys: Vec<i64> = ids.iter().map(|i| (i * 104_729) % 65_521).collect();
        let nulls = vec![false; rows];
        let mut writer =
            super::super::output::VolumeFileWriter::new(dir, "t", 1, &schema, rows, true).unwrap();
        writer
            .append_typed(
                &ids,
                &[
                    super::super::writer::TypedCells::Int64 {
                        values: &ids,
                        nulls: &nulls,
                    },
                    super::super::writer::TypedCells::Int64 {
                        values: &keys,
                        nulls: &nulls,
                    },
                ],
            )
            .unwrap();
        let (volume, path) = writer.finish().unwrap();
        assert!(
            volume.columns.resident(1).is_none(),
            "the column is on disk"
        );
        (volume, path)
    }

    #[test]
    fn the_input_s_decode_is_admitted_with_the_build_and_the_allocation_stays_within_it() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let (volume, path) = file_backed(dir.path(), 131_072);
        let identity = 41u64;
        // No group cache: every decode is the build's own allocation. The
        // cache is process global; the guard holds its lock and puts the
        // default back, on a panic too
        let _cache = super::super::group_cache::test_budget::hold(0);
        let baseline = INDEX_BUILDS.stats();
        // A budget under the input's decode: refused before anything is
        // decoded or written
        let input = decode_allowance(&volume, 1);
        assert!(input > 131_072 * 8, "{input} bytes for a group's decode");
        INDEX_BUILDS.set_budget_bytes((baseline.charged_bytes + input / 2) as u64);
        let refused = build_side_for(&volume, &path, 1, &[(1, identity)]).unwrap();
        assert!(refused.is_none());
        assert_eq!(INDEX_BUILDS.stats().refused, baseline.refused + 1);
        assert!(!side_path(&path).exists());
        // A budget of exactly the input's decode and the least workspace:
        // admitted whole, and the build's allocation stays within it
        let share = input + MIN_WORKSPACE_BYTES + metadata_allowance(131_072) + COLUMN_SLOT_BYTES;
        INDEX_BUILDS.set_budget_bytes((baseline.charged_bytes + share) as u64);
        INDEX_BUILDS.reset_peak();
        #[cfg(not(feature = "mimalloc"))]
        let mark = counting::mark();
        let side = build_side_for(&volume, &path, 1, &[(1, identity)])
            .unwrap()
            .expect("built");
        let admitted = INDEX_BUILDS.stats().peak_bytes - baseline.charged_bytes;
        #[cfg(not(feature = "mimalloc"))]
        {
            let peak = counting::peak_since(mark);
            assert!(
                peak <= admitted + side.directory().bytes() + 2 * ALLOC_SLACK,
                "allocator peak {peak} within the admitted {admitted} plus the directory and slack"
            );
        }
        assert_eq!(
            admitted, share,
            "the input's decode and the workspace share the admission"
        );
        assert_eq!(
            side.equal(1, (7 * 104_729) % 65_521)
                .unwrap()
                .map(|(s, e)| e - s),
            Some(3)
        );
        assert_eq!(
            INDEX_BUILDS.stats().charged_bytes,
            baseline.charged_bytes,
            "workspace and input released"
        );
        drop(side);
        INDEX_BUILDS.set_budget_bytes(DEFAULT_BUILD_BUDGET_BYTES);
    }

    #[test]
    fn a_volume_that_cannot_be_read_fails_its_side_build_as_the_caller_s_error() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let (volume, path) = file_backed(dir.path(), 1_000);
        let identity = 41u64;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        let failed_before = BUILDS_FAILED.load(Ordering::Relaxed);
        let err = match build_side_for(&volume, &path, 1, &[(1, identity)]) {
            Err(err) => err,
            Ok(_) => panic!("a volume that cannot be read built a side file"),
        };
        assert!(!is_refused(&err), "{err}");
        assert!(!side_path(&path).exists(), "no short side file is left");
        assert_eq!(
            BUILDS_FAILED.load(Ordering::Relaxed),
            failed_before,
            "the volume's error is not a side file failure"
        );
        assert_eq!(INDEX_BUILDS.stats().charged_bytes, 0);
    }

    #[test]
    fn a_side_file_that_does_not_open_at_reopen_is_left_in_place() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let volume_path = dir.path().join("vol_0000000000000001.vol");
        let side = side_path(&volume_path);
        build(
            &side,
            (0..100u32).map(|p| (p, p as i64)).collect(),
            WORKSPACE,
        );
        let bytes = std::fs::read(&side).unwrap();
        std::fs::write(&side, &bytes[..bytes.len() - 3]).unwrap();
        assert!(open_side_for(&volume_path, 1).is_none());
        assert!(side.exists(), "the file stays for whoever can read it");
        std::fs::write(&side, &bytes).unwrap();
        assert!(open_side_for(&volume_path, 1).is_some());
    }

    #[test]
    fn a_reader_refused_its_working_reservation_is_refused_before_any_row() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        build(
            &path,
            (0..1000u32).map(|p| (p, p as i64)).collect(),
            WORKSPACE,
        );
        INDEX_PAGES.clear();
        let file = Arc::new(IndexFile::open(&path, 16).unwrap());
        let refused_before = INDEX_PAGES.stats().refused;
        INDEX_PAGES.set_budget_bytes(INDEX_PAGES.stats().charged_bytes as u64);
        let err = match file.reader(1, 64) {
            Err(err) => err,
            Ok(_) => panic!("a reader was admitted with no budget"),
        };
        assert!(is_refused(&err), "{err}");
        assert_eq!(INDEX_PAGES.stats().refused, refused_before + 1);
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
    }

    #[test]
    fn an_admitted_reader_completes_through_its_own_buffers_when_the_cache_admits_nothing() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let pairs: Vec<(u32, i64)> = (0..200_000u32).map(|p| (p, (p % 40_000) as i64)).collect();
        build(&path, pairs.clone(), WORKSPACE);
        INDEX_PAGES.clear();
        let file = Arc::new(IndexFile::open(&path, 17).unwrap());
        // What the cache path answers, with the budget open
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
        let range = file.range(1, 100, 2_100).unwrap();
        let want = all_positions(&file, range, 512);
        let want_equal = file.equal(1, 7).unwrap();
        INDEX_PAGES.clear();
        // The reader is admitted, then the budget drops to what is held:
        // every cache load is refused from here on
        let mut reader = file.reader(1, 512).unwrap();
        INDEX_PAGES.set_budget_bytes(INDEX_PAGES.stats().charged_bytes as u64);
        let refused_before = INDEX_PAGES.stats().refused;
        assert_eq!(reader.equal(7).unwrap(), want_equal);
        assert_eq!(reader.range(100, 2_100).unwrap(), range);
        reader.walk(range);
        let mut got = Vec::new();
        while let Some(w) = reader.next_window().unwrap() {
            assert!(w.len() <= 512);
            got.extend_from_slice(w);
        }
        assert_eq!(got, want, "the same positions as the cache path");
        assert!(
            reader.own_pages() > 0,
            "pages were read through the reader's own buffers"
        );
        assert!(
            INDEX_PAGES.stats().refused > refused_before,
            "the cache refused meanwhile"
        );
        drop(reader);
        drop(file);
        INDEX_PAGES.clear();
        INDEX_PAGES.set_budget_bytes(DEFAULT_BUDGET_BYTES);
    }

    #[test]
    fn a_checksum_failure_mid_walk_is_the_reader_s_error_and_it_does_not_advance_past_it() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let pairs: Vec<(u32, i64)> = (0..30_000u32).map(|p| (p, (p % 5000) as i64)).collect();
        build(&path, pairs, WORKSPACE);
        let good = std::fs::read(&path).unwrap();
        let file = Arc::new(IndexFile::open(&path, 18).unwrap());
        let second = file.directory().column(1).unwrap().pos_pages[1].offset as usize + 10;
        let mut bad = good.clone();
        bad[second] ^= 0xff;
        std::fs::write(&path, &bad).unwrap();
        INDEX_PAGES.clear();
        // Key 1366 lives in the second position page; a small window puts
        // the failing page at the head of the walk
        let mut reader = file.reader(1, 4).unwrap();
        let range = reader.equal(1366).unwrap().unwrap();
        reader.walk(range);
        let remaining = reader.remaining();
        let err = match reader.next_window() {
            Err(err) => err,
            Ok(_) => panic!("a corrupt page was served"),
        };
        assert!(err.to_string().contains("checksum"), "{err}");
        assert!(!is_refused(&err));
        assert_eq!(reader.remaining(), remaining, "not advanced past the page");
        std::fs::write(&path, &good).unwrap();
        INDEX_PAGES.clear();
        assert_eq!(reader.next_window().unwrap().map(|w| w.len()), Some(4));
    }

    #[test]
    fn a_window_that_fails_on_its_second_page_returns_nothing_and_keeps_its_place() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        // One key over three position pages
        build(&path, (0..20_000u32).map(|p| (p, 42)).collect(), WORKSPACE);
        let good = std::fs::read(&path).unwrap();
        let file = Arc::new(IndexFile::open(&path, 19).unwrap());
        let second = file.directory().column(1).unwrap().pos_pages[1].offset as usize + 10;
        let mut bad = good.clone();
        bad[second] ^= 0xff;
        std::fs::write(&path, &bad).unwrap();
        INDEX_PAGES.clear();
        // A window wider than a page: its first page reads, its second
        // fails, and the walk reports nothing and stays where it began
        let mut reader = file.reader(1, 8_192).unwrap();
        let range = reader.equal(42).unwrap().unwrap();
        reader.walk(range);
        assert_eq!(reader.remaining(), 20_000);
        assert!(reader.next_window().is_err());
        assert_eq!(
            reader.remaining(),
            20_000,
            "no progress for a window not returned"
        );
        // Repaired, the same window comes whole from the start
        std::fs::write(&path, &good).unwrap();
        INDEX_PAGES.clear();
        let window = reader.next_window().unwrap().unwrap();
        assert_eq!(window.len(), 8_192);
        assert_eq!(window[0], 0);
        assert_eq!(reader.remaining(), 20_000 - 8_192);
        // The cache-only cursor keeps its place the same way
        std::fs::write(&path, &bad).unwrap();
        INDEX_PAGES.clear();
        let mut cursor = file.cursor(1, range, 8_192).unwrap();
        assert!(cursor.next_window().is_err());
        assert_eq!(cursor.remaining(), 20_000);
    }

    #[test]
    fn a_range_bound_from_the_directory_is_never_below_the_exact_count() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.sidx");
        let pairs: Vec<(u32, i64)> = (0..100_000u32).map(|p| (p, (p % 30_000) as i64)).collect();
        build(&path, pairs.clone(), WORKSPACE);
        let file = IndexFile::open(&path, 6).unwrap();
        for (low, high) in [(0, 0), (10, 2050), (2040, 2050), (29_990, 40_000), (-5, 3)] {
            let (start, end) = file.range(1, low, high).unwrap();
            let exact = pairs
                .iter()
                .filter(|(_, k)| (low..=high).contains(k))
                .count() as u64;
            assert_eq!(end - start, exact, "range {low}..={high}");
            assert!(file.candidate_bound(1, low, high).unwrap() >= exact);
        }
    }
}
