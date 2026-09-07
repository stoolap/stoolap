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
    cell: OnceLock<Result<Arc<ColumnData>, String>>,
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

pub static DECODED_GROUPS: LazyLock<DecodedGroupCache> = LazyLock::new(|| DecodedGroupCache {
    inner: Mutex::new(Inner::default()),
    budget: AtomicUsize::new(DEFAULT_BUDGET_BYTES),
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
});

impl DecodedGroupCache {
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
        let outcome = slot
            .cell
            .get_or_init(|| decode().map(Arc::new).map_err(|e| e.to_string()));
        match outcome {
            Ok(column) => {
                if !slot.charged.swap(true, Ordering::AcqRel) {
                    self.charge(key, column.memory_size(), budget);
                }
                Ok(Arc::clone(column))
            }
            Err(message) => {
                // A block that does not decode is not kept; the next reader tries again
                self.lock().entries.remove(&key);
                Err(std::io::Error::other(message.clone()))
            }
        }
    }

    /// Account a decoded entry and make room for it
    fn charge(&self, key: GroupKey, bytes: usize, budget: usize) {
        let mut inner = self.lock();
        if let Some(entry) = inner.entries.get_mut(&key) {
            entry.bytes = bytes;
        }
        inner.bytes += bytes;
        while inner.bytes > budget && inner.entries.len() > 1 {
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
        self.budget.store(budget, Ordering::Relaxed);
        let mut inner = self.lock();
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
