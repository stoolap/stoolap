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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::storage::index::IndexMemory;

static HOT_METADATA_BYTES: RetainedBytes = RetainedBytes::new();

pub(crate) fn hot_metadata_bytes() -> usize {
    HOT_METADATA_BYTES.get()
}

pub(crate) fn hash_table_bytes<K, V>(capacity_high_water: usize) -> u128 {
    if capacity_high_water == 0 {
        return 0;
    }
    // Bound Swiss-table buckets, control bytes and alignment from capacity.
    2 * capacity_high_water as u128 * (std::mem::size_of::<(K, V)>() + 1) as u128
        + 2 * std::mem::align_of::<(K, V)>().max(16) as u128
}

pub(crate) fn name_bytes(name: &crate::common::SmartString) -> u128 {
    if name.is_heap() {
        (arc_allocation_bytes::<String>() + name.heap_capacity()) as u128
    } else {
        0
    }
}

pub(crate) fn smallvec_bytes<A: smallvec::Array>(values: &smallvec::SmallVec<A>) -> usize {
    if values.spilled() {
        values.capacity() * std::mem::size_of::<A::Item>()
    } else {
        0
    }
}

pub(crate) const fn arc_allocation_bytes<T>() -> usize {
    let header = 2 * std::mem::size_of::<usize>();
    let alignment = std::mem::align_of::<T>();
    let value_offset = header.div_ceil(alignment) * alignment;
    let allocation_alignment = if alignment > std::mem::align_of::<usize>() {
        alignment
    } else {
        std::mem::align_of::<usize>()
    };
    (value_offset + std::mem::size_of::<T>()).div_ceil(allocation_alignment) * allocation_alignment
}

#[derive(Debug, Default)]
pub(crate) struct HotMetadataCharge(u128);

impl HotMetadataCharge {
    pub fn new(bytes: u128) -> Self {
        HOT_METADATA_BYTES.add(bytes);
        Self(bytes)
    }

    pub fn resize(&mut self, bytes: u128) {
        HOT_METADATA_BYTES.resize(self.0, bytes);
        self.0 = bytes;
    }

    #[cfg(test)]
    pub fn bytes(&self) -> u128 {
        self.0
    }
}

impl Drop for HotMetadataCharge {
    fn drop(&mut self) {
        HOT_METADATA_BYTES.remove(self.0);
    }
}

pub(crate) struct HotObjectCharge<T>(std::marker::PhantomData<fn() -> T>);

pub(crate) struct NamedMap<V> {
    entries: rustc_hash::FxHashMap<String, V>,
    key_bytes: u128,
    capacity_high_water: usize,
    memory: HotMetadataCharge,
}

impl<V> Default for NamedMap<V> {
    fn default() -> Self {
        Self {
            entries: rustc_hash::FxHashMap::default(),
            key_bytes: 0,
            capacity_high_water: 0,
            memory: HotMetadataCharge::new(
                arc_allocation_bytes::<std::sync::RwLock<Self>>() as u128
            ),
        }
    }
}

impl<V> NamedMap<V> {
    pub fn insert(&mut self, key: String, value: V) -> Option<V> {
        use std::collections::hash_map::Entry;
        let previous = match self.entries.entry(key) {
            Entry::Occupied(mut entry) => Some(entry.insert(value)),
            Entry::Vacant(entry) => {
                self.key_bytes += entry.key().capacity() as u128;
                entry.insert(value);
                None
            }
        };
        self.capacity_high_water = self.capacity_high_water.max(self.entries.capacity());
        self.refresh_memory();
        previous
    }

    pub fn remove(&mut self, name: &str) -> Option<V> {
        let (key, value) = self.entries.remove_entry(name)?;
        self.key_bytes = self.key_bytes.saturating_sub(key.capacity() as u128);
        drop(key);
        self.refresh_memory();
        Some(value)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.key_bytes = 0;
        self.refresh_memory();
    }

    fn refresh_memory(&mut self) {
        self.memory.resize(
            arc_allocation_bytes::<std::sync::RwLock<Self>>() as u128
                + self.key_bytes
                + hash_table_bytes::<String, V>(self.capacity_high_water),
        );
    }
}

impl<V> std::ops::Deref for NamedMap<V> {
    type Target = rustc_hash::FxHashMap<String, V>;

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl<'a, V> IntoIterator for &'a NamedMap<V> {
    type Item = (&'a String, &'a V);
    type IntoIter = std::collections::hash_map::Iter<'a, String, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl<T> Default for HotObjectCharge<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HotObjectCharge<T> {
    pub fn new() -> Self {
        HOT_METADATA_BYTES.add(arc_allocation_bytes::<T>() as u128);
        Self(std::marker::PhantomData)
    }
}

impl<T> Drop for HotObjectCharge<T> {
    fn drop(&mut self) {
        HOT_METADATA_BYTES.remove(arc_allocation_bytes::<T>() as u128);
    }
}

pub(crate) struct ChargedWeak<T> {
    weak: Weak<T>,
    _memory: HotObjectCharge<T>,
}

impl<T> ChargedWeak<T> {
    pub fn new(owner: &Arc<T>) -> Self {
        Self {
            weak: Arc::downgrade(owner),
            _memory: HotObjectCharge::new(),
        }
    }
}

impl<T> std::ops::Deref for ChargedWeak<T> {
    type Target = Weak<T>;

    fn deref(&self) -> &Self::Target {
        &self.weak
    }
}

struct WeakAccounts<T> {
    entries: Vec<ChargedWeak<T>>,
    memory: HotMetadataCharge,
}

impl<T> Default for WeakAccounts<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            memory: HotMetadataCharge::default(),
        }
    }
}

impl<T> WeakAccounts<T> {
    fn register(&mut self, owner: &Arc<T>) {
        self.entries.retain(|account| account.strong_count() != 0);
        self.entries.push(ChargedWeak::new(owner));
        self.memory
            .resize((self.entries.capacity() * std::mem::size_of::<ChargedWeak<T>>()) as u128);
    }
}

pub(crate) struct ChargedSmallVec<A: smallvec::Array> {
    values: smallvec::SmallVec<A>,
    memory: HotMetadataCharge,
}

impl<A: smallvec::Array> Default for ChargedSmallVec<A> {
    fn default() -> Self {
        Self {
            values: smallvec::SmallVec::new(),
            memory: HotMetadataCharge::default(),
        }
    }
}

impl<A: smallvec::Array> ChargedSmallVec<A> {
    pub fn push(&mut self, value: A::Item) {
        self.values.push(value);
        self.memory.resize(smallvec_bytes(&self.values) as u128);
    }
}

impl<A: smallvec::Array> std::ops::Deref for ChargedSmallVec<A> {
    type Target = smallvec::SmallVec<A>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

#[derive(Default)]
pub(crate) struct TableMemory {
    pub version_payloads: AtomicUsize,
    pub version_tree: AtomicUsize,
    pub pinned_versions: Mutex<PinnedVersionMemory>,
    pub arena_payloads: AtomicUsize,
    pub retired_arena_payloads: Mutex<u128>,
    pub arena_capacity: RetainedBytes,
    pub transaction_versions: RetainedBytes,
    pub transaction_undo: RetainedBytes,
    pub row_claims: AtomicUsize,
    indexes: Mutex<WeakAccounts<IndexMemory>>,
    _object: HotObjectCharge<Self>,
}

#[derive(Default)]
pub(crate) struct PinnedVersionMemory {
    pub payloads: u128,
    pub tree: u128,
}

#[derive(Default)]
pub(crate) struct TableMemoryUsage {
    pub version_payloads: usize,
    pub pinned_version_payloads: usize,
    pub version_tree: usize,
    pub pinned_version_tree: usize,
    pub arena_payloads: usize,
    pub retired_arena_payloads: usize,
    pub arena_capacity: usize,
    pub transaction_versions: usize,
    pub transaction_undo: usize,
    pub row_claims: usize,
    pub index_requested: usize,
    pub index_estimated: usize,
}

impl TableMemory {
    pub fn register_index(&self, account: &Arc<IndexMemory>) {
        self.indexes.lock().register(account);
    }

    pub fn usage(&self) -> TableMemoryUsage {
        let version_payloads = self.version_payloads.load(Ordering::Acquire);
        let version_tree = self.version_tree.load(Ordering::Acquire);
        let (pinned_version_payloads, pinned_version_tree) = {
            let pinned = self.pinned_versions.lock();
            (
                pinned.payloads.min(usize::MAX as u128) as usize,
                pinned.tree.min(usize::MAX as u128) as usize,
            )
        };
        let arena_payloads = self.arena_payloads.load(Ordering::Acquire);
        let retired_arena_payloads =
            (*self.retired_arena_payloads.lock()).min(usize::MAX as u128) as usize;
        let mut index_requested = 0usize;
        let mut index_estimated = 0usize;
        self.indexes.lock().entries.retain(|index| {
            let Some(index) = index.upgrade() else {
                return false;
            };
            index_requested = index_requested.saturating_add(index.requested_bytes());
            index_estimated = index_estimated.saturating_add(index.estimated_bytes());
            true
        });
        TableMemoryUsage {
            version_payloads,
            pinned_version_payloads,
            version_tree,
            pinned_version_tree,
            arena_payloads,
            retired_arena_payloads,
            arena_capacity: self.arena_capacity.get(),
            transaction_versions: self.transaction_versions.get(),
            transaction_undo: self.transaction_undo.get(),
            row_claims: self.row_claims.load(Ordering::Acquire),
            index_requested,
            index_estimated,
        }
    }
}

#[derive(Default)]
pub(crate) struct RetainedBytes {
    // usize::MAX permanently routes this account through the widened counter.
    bytes: AtomicUsize,
    wide: std::sync::OnceLock<Box<Mutex<u128>>>,
}

impl RetainedBytes {
    pub const fn new() -> Self {
        Self {
            bytes: AtomicUsize::new(0),
            wide: std::sync::OnceLock::new(),
        }
    }

    pub fn resize(&self, before: u128, after: u128) {
        if after > before {
            self.add(after - before);
        } else if before > after {
            self.remove(before - after);
        }
    }

    fn wide(&self) -> &Mutex<u128> {
        self.wide.get_or_init(|| Box::new(Mutex::new(0)))
    }

    pub fn add(&self, bytes: u128) {
        if bytes == 0 {
            return;
        }
        let mut current = self.bytes.load(Ordering::Acquire);
        loop {
            if current == usize::MAX {
                *self.wide().lock() += bytes;
                return;
            }
            let total = current as u128 + bytes;
            if total < usize::MAX as u128 {
                match self.bytes.compare_exchange_weak(
                    current,
                    total as usize,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return,
                    Err(observed) => current = observed,
                }
            } else {
                let mut wide = self.wide().lock();
                match self.bytes.compare_exchange(
                    current,
                    usize::MAX,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        *wide = total;
                        return;
                    }
                    Err(observed) => current = observed,
                }
            }
        }
    }

    pub fn remove(&self, bytes: u128) {
        if bytes == 0 {
            return;
        }
        let mut current = self.bytes.load(Ordering::Acquire);
        loop {
            if current == usize::MAX {
                let mut wide = self.wide().lock();
                *wide = wide.saturating_sub(bytes);
                return;
            }
            match self.bytes.compare_exchange_weak(
                current,
                (current as u128).saturating_sub(bytes) as usize,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    pub fn get(&self) -> usize {
        match self.bytes.load(Ordering::Acquire) {
            usize::MAX => (*self.wide().lock()).min(usize::MAX as u128) as usize,
            bytes => bytes,
        }
    }

    pub fn get_wide(&self) -> u128 {
        match self.bytes.load(Ordering::Acquire) {
            usize::MAX => *self.wide().lock(),
            bytes => bytes as u128,
        }
    }
}

#[derive(Default)]
pub(crate) struct HotMemoryRegistry {
    accounts: Mutex<WeakAccounts<TableMemory>>,
    _object: HotObjectCharge<Self>,
}

impl HotMemoryRegistry {
    pub fn register(&self, account: &Arc<TableMemory>) {
        self.accounts.lock().register(account);
    }

    pub fn total(&self) -> TableMemoryUsage {
        let mut total = TableMemoryUsage::default();
        self.accounts.lock().entries.retain(|account| {
            let Some(account) = account.upgrade() else {
                return false;
            };
            let usage = account.usage();
            total.version_payloads = total
                .version_payloads
                .saturating_add(usage.version_payloads);
            total.pinned_version_payloads = total
                .pinned_version_payloads
                .saturating_add(usage.pinned_version_payloads);
            total.version_tree = total.version_tree.saturating_add(usage.version_tree);
            total.pinned_version_tree = total
                .pinned_version_tree
                .saturating_add(usage.pinned_version_tree);
            total.arena_payloads = total.arena_payloads.saturating_add(usage.arena_payloads);
            total.retired_arena_payloads = total
                .retired_arena_payloads
                .saturating_add(usage.retired_arena_payloads);
            total.arena_capacity = total.arena_capacity.saturating_add(usage.arena_capacity);
            total.transaction_versions = total
                .transaction_versions
                .saturating_add(usage.transaction_versions);
            total.transaction_undo = total
                .transaction_undo
                .saturating_add(usage.transaction_undo);
            total.row_claims = total.row_claims.saturating_add(usage.row_claims);
            total.index_requested = total.index_requested.saturating_add(usage.index_requested);
            total.index_estimated = total.index_estimated.saturating_add(usage.index_estimated);
            true
        });
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_maps_track_stored_keys_and_keep_bucket_capacity() {
        let mut map = NamedMap::default();
        let mut key = String::with_capacity(8192);
        key.push_str("table");
        map.insert(key, 1u64);
        let before = map.memory.bytes();
        assert!(before >= 8192);
        map.insert("table".to_string(), 2);
        assert_eq!(
            map.key_bytes, 8192,
            "replacement keeps the original key allocation"
        );
        assert_eq!(map.memory.bytes(), before);
        for index in 0..128 {
            map.insert(format!("table_{index}"), index);
        }
        let capacity = map.capacity_high_water;
        let keys = map.key_bytes;
        let full = map.memory.bytes();
        map.clear();
        assert_eq!(map.memory.bytes(), full - keys);
        assert_eq!(map.capacity_high_water, capacity);
        map.insert("x".to_string(), 3);
        let full = map.memory.bytes();
        assert_eq!(map.remove("x"), Some(3));
        assert_eq!(map.memory.bytes(), full - 1);
    }

    #[test]
    fn account_lists_charge_retained_capacity_and_smallvec_spills() {
        let owners: Vec<_> = (0..32).map(|_| Arc::new(0u64)).collect();
        let mut accounts = WeakAccounts::default();
        for owner in &owners {
            accounts.register(owner);
        }
        let bytes = accounts.memory.bytes();
        assert_eq!(
            bytes,
            (accounts.entries.capacity() * std::mem::size_of::<Weak<u64>>()) as u128
        );
        assert!(bytes >= 32 * std::mem::size_of::<Weak<u64>>() as u128);
        drop(owners);
        accounts
            .entries
            .retain(|account| account.strong_count() != 0);
        assert_eq!(accounts.memory.bytes(), bytes);

        let mut values = ChargedSmallVec::<[u64; 2]>::default();
        values.push(1);
        values.push(2);
        assert_eq!(values.memory.bytes(), 0);
        values.push(3);
        assert_eq!(
            values.memory.bytes(),
            (values.capacity() * std::mem::size_of::<u64>()) as u128
        );
        assert!(values.memory.bytes() >= 3 * std::mem::size_of::<u64>() as u128);
    }

    #[test]
    fn transaction_counter_preserves_excess_through_release() {
        let memory = RetainedBytes::default();
        assert!(memory.wide.get().is_none());
        memory.add(48);
        memory.add(32);
        assert_eq!(memory.get(), 80);
        memory.remove(80);
        assert_eq!(memory.bytes.load(Ordering::Relaxed), 0);
        assert!(memory.wide.get().is_none());

        memory.add(usize::MAX as u128 - 8);
        memory.add(64);
        assert_eq!(memory.get(), usize::MAX);
        memory.remove(64);
        assert_eq!(memory.get(), usize::MAX - 8);
        memory.remove(usize::MAX as u128 - 8);
        memory.add(48);
        assert_eq!(memory.get(), 48);
        assert_eq!(memory.bytes.load(Ordering::Relaxed), usize::MAX);
        memory.remove(48);
        assert_eq!(memory.get(), 0);
    }

    #[test]
    fn counter_release_saturates_before_narrowing() {
        let memory = RetainedBytes::default();
        memory.add(48);
        memory.remove(usize::MAX as u128 + 32);
        assert_eq!(memory.get(), 0);
        memory.add(usize::MAX as u128 + 32);
        memory.remove(usize::MAX as u128 + 48);
        assert_eq!(memory.get_wide(), 0);
        memory.add(64);
        assert_eq!(memory.get_wide(), 64);
    }

    #[test]
    fn transaction_counter_serializes_overflow_transition() {
        let memory = RetainedBytes::default();
        memory.add(usize::MAX as u128 - 1);
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    memory.add(32);
                    barrier.wait();
                    assert_eq!(memory.get(), usize::MAX);
                    barrier.wait();
                    memory.remove(32);
                });
            }
        });
        assert_eq!(memory.get(), usize::MAX - 1);
        memory.remove(usize::MAX as u128 - 1);
        assert_eq!(memory.get(), 0);
    }
}
