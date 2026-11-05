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

//! A byte-budgeted cache of decoded row-group columns.
//!
//! A warm volume keeps its columns compressed; every scan of a row group
//! decoded the columns it needed and dropped them again. The cache keeps the
//! decoded columns, keyed by (block store, column, group), so a group read by
//! query after query is decoded once. Concurrent readers of the same group
//! share one decode: the first takes the slot's cell, the others wait on it,
//! and the map lock is never held while decoding. The budget is a process
//! total; the least recently used entries leave when a new one crosses it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use super::column::ColumnData;

/// Default budget: 64 MB of decoded columns
pub const DEFAULT_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// (block store id, column, group)
pub type GroupKey = (usize, u32, u32);

struct Slot {
    cell: OnceLock<std::io::Result<Arc<ColumnData>>>,
    /// The decoder charges the entry's bytes once
    charged: AtomicBool,
}

struct Entry {
    slot: Arc<Slot>,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<GroupKey, Entry>,
    bytes: usize,
    tick: u64,
}

pub struct DecodedGroupCache {
    inner: Mutex<Inner>,
    budget: AtomicUsize,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Snapshot of the cache for PRAGMA reporting
pub struct CacheStats {
    pub budget_bytes: usize,
    pub bytes: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
}

pub static DECODED_GROUPS: LazyLock<DecodedGroupCache> =
    LazyLock::new(|| DecodedGroupCache::new(DEFAULT_BUDGET_BYTES));

impl DecodedGroupCache {
    fn new(budget_bytes: usize) -> Self {
        DecodedGroupCache {
            inner: Mutex::new(Inner::default()),
            budget: AtomicUsize::new(budget_bytes),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// The decoded column for `key`, decoding it with `decode` when absent.
    /// A budget of zero bypasses the cache.
    pub fn get_or_decode(
        &self,
        key: GroupKey,
        decode: impl FnOnce() -> std::io::Result<ColumnData>,
    ) -> std::io::Result<Arc<ColumnData>> {
        let budget = self.budget.load(Ordering::Relaxed);
        if budget == 0 {
            return decode().map(Arc::new);
        }
        let slot = {
            let mut inner = self.lock();
            inner.tick += 1;
            let tick = inner.tick;
            match inner.entries.get_mut(&key) {
                Some(entry) => {
                    entry.last_used = tick;
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    Arc::clone(&entry.slot)
                }
                None => {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    let slot = Arc::new(Slot {
                        cell: OnceLock::new(),
                        charged: AtomicBool::new(false),
                    });
                    inner.entries.insert(
                        key,
                        Entry {
                            slot: Arc::clone(&slot),
                            bytes: 0,
                            last_used: tick,
                        },
                    );
                    slot
                }
            }
        };
        let outcome = slot.cell.get_or_init(|| decode().map(Arc::new));
        match outcome {
            Ok(column) => {
                if !slot.charged.swap(true, Ordering::AcqRel) {
                    self.charge(key, &slot, column.cache_size());
                }
                Ok(Arc::clone(column))
            }
            Err(error) => {
                // A block that does not decode is not kept; the next reader tries again
                let mut inner = self.lock();
                if inner
                    .entries
                    .get(&key)
                    .is_some_and(|entry| Arc::ptr_eq(&entry.slot, &slot))
                {
                    inner.entries.remove(&key);
                }
                drop(inner);
                Err(std::io::Error::new(error.kind(), error.to_string()))
            }
        }
    }

    /// Account a decoded entry and make room for it. The entry may have been
    /// evicted while it decoded, the budget may have moved, or the column may
    /// not fit at all: in each case the caller keeps its column and the cache
    /// keeps nothing.
    fn charge(&self, key: GroupKey, slot: &Arc<Slot>, bytes: usize) {
        // The budget is read under the same lock that set_budget_bytes holds
        // while it stores it, so a change cannot slip in between
        let mut inner = self.lock();
        let budget = self.budget.load(Ordering::Relaxed);
        let still_ours = inner
            .entries
            .get(&key)
            .is_some_and(|entry| Arc::ptr_eq(&entry.slot, slot));
        if !still_ours {
            return;
        }
        if budget == 0 || bytes > budget {
            inner.entries.remove(&key);
            return;
        }
        if let Some(entry) = inner.entries.get_mut(&key) {
            entry.bytes = bytes;
        }
        inner.bytes += bytes;
        while inner.bytes > budget {
            let victim = inner
                .entries
                .iter()
                .filter(|(k, _)| **k != key)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k);
            match victim {
                Some(victim) => {
                    if let Some(entry) = inner.entries.remove(&victim) {
                        inner.bytes -= entry.bytes;
                    }
                }
                None => break,
            }
        }
    }

    /// Drop every entry of a block store that went away
    pub fn remove_store(&self, store_id: usize) {
        let mut inner = self.lock();
        let gone: Vec<GroupKey> = inner
            .entries
            .keys()
            .filter(|k| k.0 == store_id)
            .copied()
            .collect();
        for key in gone {
            if let Some(entry) = inner.entries.remove(&key) {
                inner.bytes -= entry.bytes;
            }
        }
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget.load(Ordering::Relaxed)
    }

    /// Set the budget; entries beyond it leave at once
    pub fn set_budget_bytes(&self, budget: usize) {
        let mut inner = self.lock();
        self.budget.store(budget, Ordering::Relaxed);
        while inner.bytes > budget && !inner.entries.is_empty() {
            let victim = inner
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k);
            match victim {
                Some(victim) => {
                    if let Some(entry) = inner.entries.remove(&victim) {
                        inner.bytes -= entry.bytes;
                    }
                }
                None => break,
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.lock();
        CacheStats {
            budget_bytes: self.budget.load(Ordering::Relaxed),
            bytes: inner.bytes,
            entries: inner.entries.len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The cache is process global: a test that sets its budget holds the
/// one lock through `hold`, and the default comes back when the guard
/// drops, on a panic too.
#[cfg(test)]
pub(crate) mod test_budget {
    static CACHE_BUDGET: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) struct BudgetHeld {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    pub(crate) fn hold(bytes: usize) -> BudgetHeld {
        let lock = CACHE_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
        super::DECODED_GROUPS.set_budget_bytes(0);
        super::DECODED_GROUPS.set_budget_bytes(bytes);
        BudgetHeld { _lock: lock }
    }

    impl Drop for BudgetHeld {
        fn drop(&mut self) {
            super::DECODED_GROUPS.set_budget_bytes(0);
            super::DECODED_GROUPS.set_budget_bytes(super::DEFAULT_BUDGET_BYTES);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DecodedGroupCache;
    use crate::core::DataType;
    use crate::storage::volume::column::ColumnData;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Barrier};
    use std::time::Duration;

    fn ints(rows: usize) -> ColumnData {
        ColumnData::Int64 {
            values: vec![1; rows],
            nulls: vec![false; rows],
        }
    }

    #[test]
    fn concurrent_readers_share_one_decode() {
        let cache = DecodedGroupCache::new(1024);
        let starts = Barrier::new(16);
        let decodes = AtomicUsize::new(0);
        let columns: Vec<_> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        starts.wait();
                        cache
                            .get_or_decode((600, 0, 0), || {
                                decodes.fetch_add(1, Ordering::SeqCst);
                                Ok(ints(1))
                            })
                            .unwrap()
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
        assert!(columns.iter().all(|c| Arc::ptr_eq(&columns[0], c)));
        cache.remove_store(600);
        assert_eq!(cache.stats().bytes, 0);
    }

    #[test]
    fn failed_decode_preserves_error_kind_and_can_retry() {
        let cache = DecodedGroupCache::new(1024);
        let error = cache
            .get_or_decode((700, 0, 0), || {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid group payload",
                ))
            })
            .err()
            .unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "invalid group payload");
        let stats = cache.stats();
        assert_eq!((stats.bytes, stats.entries), (0, 0));
        let column = cache.get_or_decode((700, 0, 0), || Ok(ints(2))).unwrap();
        assert_eq!(column.len(), 2);
    }

    #[test]
    fn failed_decode_does_not_remove_a_replacement_entry() {
        let cache = DecodedGroupCache::new(1024);
        let mut replacement = None;
        let result = cache.get_or_decode((800, 0, 0), || {
            cache.remove_store(800);
            replacement = Some(cache.get_or_decode((800, 0, 0), || Ok(ints(2))).unwrap());
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "obsolete decode failed",
            ))
        });
        assert!(result.is_err());
        let replacement = replacement.unwrap();
        let stats = cache.stats();
        assert_eq!((stats.bytes, stats.entries), (replacement.cache_size(), 1));
        let cached = cache
            .get_or_decode((800, 0, 0), || panic!("replacement was evicted"))
            .unwrap();
        assert!(Arc::ptr_eq(&cached, &replacement));
        cache.remove_store(800);
        let stats = cache.stats();
        assert_eq!((stats.bytes, stats.entries), (0, 0));
    }

    #[test]
    fn an_entry_evicted_while_decoding_leaves_no_phantom_bytes() {
        let cache = DecodedGroupCache::new(18);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let cache = &cache;
            let worker = scope.spawn(move || {
                cache
                    .get_or_decode((100, 0, 0), || {
                        entered_tx.send(()).unwrap();
                        resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                        Ok(ints(1))
                    })
                    .unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            cache.get_or_decode((200, 0, 0), || Ok(ints(2))).unwrap();
            cache.get_or_decode((300, 0, 0), || Ok(ints(2))).unwrap();
            resume_tx.send(()).unwrap();
            worker.join().unwrap();
        });
        for store in [100, 200, 300] {
            cache.remove_store(store);
        }
        let stats = cache.stats();
        assert_eq!((stats.bytes, stats.entries), (0, 0));
    }

    #[test]
    fn a_budget_of_zero_set_during_a_decode_leaves_the_cache_empty() {
        let cache = DecodedGroupCache::new(18);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let cache = &cache;
            let worker = scope.spawn(move || {
                cache
                    .get_or_decode((400, 0, 0), || {
                        entered_tx.send(()).unwrap();
                        resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                        Ok(ints(1))
                    })
                    .unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            cache.set_budget_bytes(0);
            resume_tx.send(()).unwrap();
            worker.join().unwrap();
        });
        let stats = cache.stats();
        assert_eq!((stats.budget_bytes, stats.bytes, stats.entries), (0, 0, 0));
    }

    #[test]
    fn a_column_larger_than_the_budget_is_not_kept() {
        let cache = DecodedGroupCache::new(1024 * 1024);
        cache
            .get_or_decode((500, 0, 0), || {
                Ok(ColumnData::Bytes {
                    data: vec![b'x'; 8 * 1024 * 1024],
                    offsets: vec![(0, 8 * 1024 * 1024)],
                    ext_type: DataType::Json,
                    nulls: vec![false],
                })
            })
            .unwrap();
        let stats = cache.stats();
        assert!(
            stats.bytes <= stats.budget_bytes,
            "kept {} bytes",
            stats.bytes
        );
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn a_zero_budget_racing_with_a_charge_leaves_nothing_behind() {
        const READERS: usize = 8;
        const ROUNDS: usize = 20_000;
        let cache = DecodedGroupCache::new(0);
        let barrier = Barrier::new(READERS + 1);
        let mut failure = None;
        std::thread::scope(|scope| {
            for worker in 0..READERS {
                let (cache, barrier) = (&cache, &barrier);
                scope.spawn(move || {
                    for _ in 0..ROUNDS {
                        barrier.wait();
                        cache
                            .get_or_decode((10_000 + worker, 0, 0), || Ok(ints(1)))
                            .unwrap();
                        barrier.wait();
                    }
                });
            }
            for round in 0..ROUNDS {
                cache.set_budget_bytes(1024);
                barrier.wait();
                cache.set_budget_bytes(0);
                barrier.wait();
                let stats = cache.stats();
                if stats.bytes != 0 || stats.entries != 0 {
                    failure.get_or_insert((round, stats.bytes, stats.entries));
                }
            }
        });
        assert_eq!(
            failure, None,
            "a zero budget left data behind (round, bytes, entries)"
        );
    }
}
