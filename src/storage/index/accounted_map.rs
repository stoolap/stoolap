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

//! Incremental ownership accounting under each index's existing write lock.

use crate::common::{CompactArc, CompactArcDrop, CompactVec, MemoryAccount, MemoryCharge};
use crate::core::Value;
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{BuildHasher, Hash};
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Default)]
pub(crate) struct HeapBytes {
    pub retained: usize,
    pub conservative: usize,
}
impl HeapBytes {
    fn plus(self, other: Self) -> Self {
        Self {
            retained: self.retained + other.retained,
            conservative: self.conservative + other.conservative,
        }
    }
}

/// Shared allocations carry their own charge. This counts only private buffers
/// nested inside an entry, without adding a sidecar to every key or posting.
pub(crate) trait PrivateHeap {
    const HAS_HEAP: bool = true;
    const MUTATION_BOUND_SUPPLIED: bool = false;
    fn private_heap(&self) -> HeapBytes;
}
macro_rules! no_heap {
    ($($t:ty),*) => {$(impl PrivateHeap for $t {
        const HAS_HEAP: bool = false;
        fn private_heap(&self) -> HeapBytes { HeapBytes::default() }
    })*};
}
no_heap!(i64, u64, u32, usize, Value);
impl<T: ?Sized> PrivateHeap for std::sync::Arc<T> {
    const HAS_HEAP: bool = false;
    fn private_heap(&self) -> HeapBytes {
        HeapBytes::default()
    }
}
impl PrivateHeap for String {
    fn private_heap(&self) -> HeapBytes {
        HeapBytes {
            retained: self.capacity(),
            conservative: 0,
        }
    }
}
impl<T: CompactArcDrop + ?Sized> PrivateHeap for CompactArc<T> {
    const HAS_HEAP: bool = false;
    fn private_heap(&self) -> HeapBytes {
        HeapBytes::default()
    }
}
impl<T: PrivateHeap> PrivateHeap for Vec<T> {
    fn private_heap(&self) -> HeapBytes {
        let mut bytes = HeapBytes {
            retained: self.capacity() * std::mem::size_of::<T>(),
            conservative: 0,
        };
        if T::HAS_HEAP {
            for value in self {
                bytes = bytes.plus(value.private_heap());
            }
        }
        bytes
    }
}
impl<T: PrivateHeap> PrivateHeap for CompactVec<T> {
    fn private_heap(&self) -> HeapBytes {
        let mut bytes = HeapBytes {
            retained: self.capacity() * std::mem::size_of::<T>(),
            conservative: 0,
        };
        if T::HAS_HEAP {
            for value in self {
                bytes = bytes.plus(value.private_heap());
            }
        }
        bytes
    }
}
impl<A: PrivateHeap, B: PrivateHeap> PrivateHeap for (A, B) {
    const HAS_HEAP: bool = A::HAS_HEAP || B::HAS_HEAP;
    fn private_heap(&self) -> HeapBytes {
        self.0.private_heap().plus(self.1.private_heap())
    }
}

/// Conservative node bound for the supported std BTree layout (B=6, 11 keys,
/// 12 edges, >=5 keys per nonroot node). Charge every possible node as internal
/// and include padding. An empty allocated root is retained until explicit clear.
pub(crate) fn btree_bound<K, V>(len: usize, initialized: bool) -> usize {
    if !initialized {
        return 0;
    }
    let alignment = std::mem::align_of::<K>()
        .max(std::mem::align_of::<V>())
        .max(std::mem::align_of::<usize>());
    let node = 16
        + 11 * (std::mem::size_of::<K>() + std::mem::size_of::<V>())
        + 12 * std::mem::size_of::<usize>()
        + 4 * alignment;
    (1 + len.saturating_sub(1) / 5) * node
}
impl<T: Ord> PrivateHeap for BTreeSet<T> {
    fn private_heap(&self) -> HeapBytes {
        HeapBytes {
            retained: 0,
            conservative: btree_bound::<T, ()>(self.len(), true),
        }
    }
}

#[derive(Default)]
struct NestedCharge {
    bytes: HeapBytes,
    retained: Option<MemoryCharge>,
    conservative: Option<MemoryCharge>,
}
impl NestedCharge {
    fn attach(&mut self, account: &MemoryAccount) {
        self.retained = Some(MemoryCharge::new(account, self.bytes.retained));
        self.conservative = Some(MemoryCharge::conservative(account, self.bytes.conservative));
    }
    fn account(&self) -> Option<&MemoryAccount> {
        self.retained.as_ref().map(MemoryCharge::account)
    }
    fn change(&mut self, before: HeapBytes, after: HeapBytes) {
        self.bytes.retained = self.bytes.retained - before.retained + after.retained;
        self.bytes.conservative =
            self.bytes.conservative - before.conservative + after.conservative;
        if let Some(charge) = &mut self.retained {
            if charge.bytes() != self.bytes.retained {
                charge.resize(self.bytes.retained);
            }
        }
        if let Some(charge) = &mut self.conservative {
            if charge.bytes() != self.bytes.conservative {
                charge.resize(self.bytes.conservative);
            }
        }
    }
}

pub(crate) struct ValueMut<'a, V: PrivateHeap> {
    value: &'a mut V,
    before: HeapBytes,
    charge: &'a mut NestedCharge,
}
impl<V: PrivateHeap> Deref for ValueMut<'_, V> {
    type Target = V;
    fn deref(&self) -> &V {
        self.value
    }
}
impl<V: PrivateHeap> DerefMut for ValueMut<'_, V> {
    fn deref_mut(&mut self) -> &mut V {
        self.value
    }
}
impl<V: PrivateHeap> Drop for ValueMut<'_, V> {
    fn drop(&mut self) {
        let after = self.value.private_heap();
        if !V::MUTATION_BOUND_SUPPLIED && after.retained > self.before.retained {
            // The private collection controls realloc internally. Record a
            // conservative old+new overlap, rather than calling its capacity
            // delta an exact physical peak. Only growth touches these counters.
            if let Some(account) = self.charge.account() {
                drop(MemoryCharge::conservative(account, after.retained));
            }
        }
        self.charge.change(self.before, after);
    }
}

pub(crate) struct TrackedBTree<K: Ord + PrivateHeap, V: PrivateHeap> {
    data: BTreeMap<K, V>,
    initialized: bool,
    nested: NestedCharge,
    nodes: Option<MemoryCharge>,
}
impl<K: Ord + PrivateHeap, V: PrivateHeap> Default for TrackedBTree<K, V> {
    fn default() -> Self {
        Self {
            data: BTreeMap::new(),
            initialized: false,
            nested: NestedCharge::default(),
            nodes: None,
        }
    }
}
impl<K: Ord + PrivateHeap, V: PrivateHeap> Deref for TrackedBTree<K, V> {
    type Target = BTreeMap<K, V>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl<K: Ord + PrivateHeap, V: PrivateHeap> TrackedBTree<K, V> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn attach(&mut self, account: &MemoryAccount) {
        // Private capacities are maintained before attachment too. A populated
        // startup index therefore needs no second structural allocation scan.
        self.nested.attach(account);
        self.nodes = Some(MemoryCharge::conservative(
            account,
            btree_bound::<K, V>(self.data.len(), self.initialized),
        ));
    }
    fn refresh_nodes(&mut self, len: usize) {
        if let Some(nodes) = &mut self.nodes {
            let bytes = btree_bound::<K, V>(len, self.initialized);
            if nodes.bytes() != bytes {
                nodes.resize(bytes);
            }
        }
    }
    pub fn mutate_remove_if<Q: Ord + ?Sized>(
        &mut self,
        key: &Q,
        mutate: impl FnOnce(&mut V) -> bool,
    ) -> bool
    where
        K: Borrow<Q>,
    {
        let remove = match self.get_mut(key) {
            Some(mut value) => mutate(&mut value),
            None => false,
        };
        if remove {
            self.remove(key);
        }
        remove
    }
    pub fn get_mut<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<ValueMut<'_, V>>
    where
        K: Borrow<Q>,
    {
        let value = self.data.get_mut(key)?;
        Some(ValueMut {
            before: value.private_heap(),
            value,
            charge: &mut self.nested,
        })
    }
    pub fn entry(&mut self, key: K) -> BTreeEntry<'_, K, V> {
        let future_len = self.data.len() + 1;
        BTreeEntry {
            entry: self.data.entry(key),
            nested: &mut self.nested,
            nodes: &mut self.nodes,
            initialized: &mut self.initialized,
            future_len,
        }
    }
    pub fn insert(&mut self, key: K, value: V) {
        let future_len = self.data.len() + usize::from(!self.data.contains_key(&key));
        self.initialized = true;
        self.refresh_nodes(future_len);
        let key_bytes = key.private_heap();
        let new_bytes = value.private_heap();
        // Existing BTree keys are retained on replacement.
        let existing = self.data.contains_key(&key);
        self.nested.change(
            HeapBytes::default(),
            new_bytes.plus(if existing {
                HeapBytes::default()
            } else {
                key_bytes
            }),
        );
        if let Some(old) = self.data.insert(key, value) {
            let old_bytes = old.private_heap();
            drop(old);
            self.nested.change(old_bytes, HeapBytes::default());
        }
    }
    pub fn remove<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<()>
    where
        K: Borrow<Q>,
    {
        let (key, value) = self.data.remove_entry(key)?;
        let bytes = key.private_heap().plus(value.private_heap());
        drop((key, value));
        self.nested.change(bytes, HeapBytes::default());
        self.refresh_nodes(self.data.len());
        Some(())
    }
    pub fn clear(&mut self) {
        self.data = BTreeMap::new();
        self.initialized = false;
        self.nested.change(self.nested.bytes, HeapBytes::default());
        self.refresh_nodes(0);
    }
}
pub(crate) struct BTreeEntry<'a, K: Ord + PrivateHeap, V: PrivateHeap> {
    entry: std::collections::btree_map::Entry<'a, K, V>,
    nested: &'a mut NestedCharge,
    nodes: &'a mut Option<MemoryCharge>,
    initialized: &'a mut bool,
    future_len: usize,
}
impl<'a, K: Ord + PrivateHeap, V: PrivateHeap + Default> BTreeEntry<'a, K, V> {
    pub fn or_default(self) -> ValueMut<'a, V> {
        let value = match self.entry {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                *self.initialized = true;
                if let Some(nodes) = self.nodes {
                    let bound = btree_bound::<K, V>(self.future_len, true);
                    if nodes.bytes() != bound {
                        nodes.resize(bound);
                    }
                }
                let value = V::default();
                self.nested.change(
                    HeapBytes::default(),
                    entry.key().private_heap().plus(value.private_heap()),
                );
                entry.insert(value)
            }
        };
        ValueMut {
            before: value.private_heap(),
            value,
            charge: self.nested,
        }
    }
}

pub(crate) struct TrackedHash<K: Eq + Hash + PrivateHeap, V: PrivateHeap, S: BuildHasher + Clone> {
    data: hashbrown::HashMap<K, V, S>,
    nested: NestedCharge,
    buffer: Option<MemoryCharge>,
}
impl<K: Eq + Hash + PrivateHeap, V: PrivateHeap, S: BuildHasher + Clone + Default> Default
    for TrackedHash<K, V, S>
{
    fn default() -> Self {
        Self::with_capacity_and_hasher(0, S::default())
    }
}
impl<K: Eq + Hash + PrivateHeap, V: PrivateHeap, S: BuildHasher + Clone> Deref
    for TrackedHash<K, V, S>
{
    type Target = hashbrown::HashMap<K, V, S>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl<K: Eq + Hash + PrivateHeap, V: PrivateHeap, S: BuildHasher + Clone> TrackedHash<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            data: hashbrown::HashMap::with_capacity_and_hasher(capacity, hasher),
            nested: NestedCharge::default(),
            buffer: None,
        }
    }
    pub fn attach(&mut self, account: &MemoryAccount) {
        // Private capacities are maintained before attachment too. A populated
        // startup index therefore needs no second structural allocation scan.
        self.nested.attach(account);
        self.buffer = Some(MemoryCharge::new(account, self.data.allocation_size()));
    }
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.data.len() + additional;
        if needed <= self.data.capacity() {
            return;
        }
        let capacity = needed.max(self.data.capacity().saturating_mul(2));
        let mut replacement =
            hashbrown::HashMap::with_capacity_and_hasher(capacity, self.data.hasher().clone());
        let replacement_charge = self
            .buffer
            .as_ref()
            .map(|charge| MemoryCharge::new(charge.account(), replacement.allocation_size()));
        for (key, value) in self.data.drain() {
            replacement.insert(key, value);
        }
        let old = std::mem::replace(&mut self.data, replacement);
        let old_charge = std::mem::replace(&mut self.buffer, replacement_charge);
        drop(old);
        drop(old_charge);
    }
    fn room_for(&mut self, key: &K) {
        if self.data.len() == self.data.capacity() && !self.data.contains_key(key) {
            self.reserve(1);
        }
    }
    pub fn mutate_remove_if<Q: Eq + Hash + ?Sized>(
        &mut self,
        key: &Q,
        mutate: impl FnOnce(&mut V) -> bool,
    ) -> bool
    where
        K: Borrow<Q>,
    {
        let remove = match self.get_mut(key) {
            Some(mut value) => mutate(&mut value),
            None => false,
        };
        if remove {
            self.remove(key);
        }
        remove
    }
    pub fn get_mut<Q: Eq + Hash + ?Sized>(&mut self, key: &Q) -> Option<ValueMut<'_, V>>
    where
        K: Borrow<Q>,
    {
        let value = self.data.get_mut(key)?;
        Some(ValueMut {
            before: value.private_heap(),
            value,
            charge: &mut self.nested,
        })
    }
    pub fn entry(&mut self, key: K) -> HashEntry<'_, K, V, S> {
        self.room_for(&key);
        match self.data.entry(key) {
            hashbrown::hash_map::Entry::Occupied(entry) => HashEntry::Occupied(HashOccupiedEntry {
                entry,
                nested: &mut self.nested,
            }),
            hashbrown::hash_map::Entry::Vacant(entry) => HashEntry::Vacant(HashVacantEntry {
                entry,
                nested: &mut self.nested,
            }),
        }
    }
    pub fn insert(&mut self, key: K, value: V) {
        self.room_for(&key);
        match self.data.entry(key) {
            hashbrown::hash_map::Entry::Occupied(mut entry) => {
                let before = entry.get().private_heap();
                self.nested
                    .change(HeapBytes::default(), value.private_heap());
                drop(entry.insert(value));
                self.nested.change(before, HeapBytes::default());
            }
            hashbrown::hash_map::Entry::Vacant(entry) => {
                self.nested.change(
                    HeapBytes::default(),
                    entry.key().private_heap().plus(value.private_heap()),
                );
                entry.insert(value);
            }
        }
    }
    pub fn remove<Q: Eq + Hash + ?Sized>(&mut self, key: &Q) -> Option<()>
    where
        K: Borrow<Q>,
    {
        let (key, value) = self.data.remove_entry(key)?;
        let before = key.private_heap().plus(value.private_heap());
        drop((key, value));
        self.nested.change(before, HeapBytes::default());
        Some(())
    }
    /// Removal may return only a value whose allocations own independent charges.
    /// Private Vec/String buffers must instead remain in an owned charge guard.
    pub fn remove_value<Q: Eq + Hash + ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        assert!(
            !V::HAS_HEAP,
            "private buffers require an owned removal charge"
        );
        let (key, value) = self.data.remove_entry(key)?;
        let before = key.private_heap();
        drop(key);
        self.nested.change(before, HeapBytes::default());
        Some(value)
    }
    pub fn clear(&mut self) {
        self.data.clear();
        self.nested.change(self.nested.bytes, HeapBytes::default());
    }
}
pub(crate) enum HashEntry<'a, K: Eq + Hash + PrivateHeap, V: PrivateHeap, S: BuildHasher + Clone> {
    Occupied(HashOccupiedEntry<'a, K, V, S>),
    Vacant(HashVacantEntry<'a, K, V, S>),
}
pub(crate) struct HashOccupiedEntry<
    'a,
    K: Eq + Hash + PrivateHeap,
    V: PrivateHeap,
    S: BuildHasher + Clone,
> {
    entry: hashbrown::hash_map::OccupiedEntry<'a, K, V, S>,
    nested: &'a mut NestedCharge,
}
pub(crate) struct HashVacantEntry<
    'a,
    K: Eq + Hash + PrivateHeap,
    V: PrivateHeap,
    S: BuildHasher + Clone,
> {
    entry: hashbrown::hash_map::VacantEntry<'a, K, V, S>,
    nested: &'a mut NestedCharge,
}
impl<'a, K: Eq + Hash + PrivateHeap, V: PrivateHeap, S: BuildHasher + Clone>
    HashOccupiedEntry<'a, K, V, S>
{
    pub fn get(&self) -> &V {
        self.entry.get()
    }
    pub fn into_mut(self) -> ValueMut<'a, V> {
        let value = self.entry.into_mut();
        ValueMut {
            before: value.private_heap(),
            value,
            charge: self.nested,
        }
    }
}
impl<'a, K: Eq + Hash + PrivateHeap, V: PrivateHeap, S: BuildHasher + Clone>
    HashVacantEntry<'a, K, V, S>
{
    pub fn insert(self, value: V) -> ValueMut<'a, V> {
        self.nested.change(
            HeapBytes::default(),
            self.entry.key().private_heap().plus(value.private_heap()),
        );
        let value = self.entry.insert(value);
        ValueMut {
            before: value.private_heap(),
            value,
            charge: self.nested,
        }
    }
}
impl<'a, K: Eq + Hash + PrivateHeap, V: PrivateHeap + Default, S: BuildHasher + Clone>
    HashEntry<'a, K, V, S>
{
    pub fn or_default(self) -> ValueMut<'a, V> {
        match self {
            Self::Occupied(entry) => entry.into_mut(),
            Self::Vacant(entry) => entry.insert(V::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Rows = CompactVec<i64>;
    type Map = TrackedHash<u64, Rows, std::collections::hash_map::RandomState>;
    #[test]
    fn tracked_hash_only_charges_live_private_capacities_and_final_buffer() {
        let root = MemoryAccount::new();
        let mut map = Map::default();
        map.attach(&root);
        for key in 0..20 {
            for row in 0..20 {
                map.entry(key).or_default().push(row);
            }
        }
        let rows = map
            .data
            .values()
            .map(|rows| rows.capacity() * std::mem::size_of::<i64>())
            .sum::<usize>();
        assert_eq!(
            root.snapshot().retained_bytes,
            rows + map.data.allocation_size()
        );
        for key in 0..10 {
            map.remove(&key);
        }
        assert_eq!(
            root.snapshot().retained_bytes,
            rows / 2 + map.data.allocation_size()
        );
        map.clear();
        assert_eq!(root.snapshot().retained_bytes, map.data.allocation_size());
        drop(map);
        assert_eq!(root.snapshot().retained_bytes, 0);
    }
    #[test]
    fn tracked_btree_separates_exact_posting_capacity_and_opaque_nodes() {
        let root = MemoryAccount::new();
        let base = root.snapshot().conservative_bytes;
        let mut map = TrackedBTree::<i64, Rows>::new();
        map.attach(&root);
        for key in 0..100 {
            for row in 0..20 {
                map.entry(key).or_default().push(row);
            }
        }
        let exact = map
            .data
            .values()
            .map(|rows| rows.capacity() * std::mem::size_of::<i64>())
            .sum::<usize>();
        assert_eq!(root.snapshot().retained_bytes, exact);
        assert_eq!(
            root.snapshot().conservative_bytes,
            base + btree_bound::<i64, Rows>(100, true)
        );
        for key in 0..100 {
            map.remove(&key);
        }
        assert_eq!(root.snapshot().retained_bytes, 0);
        assert_eq!(
            root.snapshot().conservative_bytes,
            base + btree_bound::<i64, Rows>(0, true)
        );
        map.clear();
        assert_eq!(root.snapshot().conservative_bytes, base);
    }
    #[derive(Clone, Default)]
    struct CollidingHasher;
    impl std::hash::Hasher for CollidingHasher {
        fn write(&mut self, _: &[u8]) {}
        fn finish(&self) -> u64 {
            0
        }
    }

    #[test]
    fn hash_tombstones_keep_physical_bucket_charge() {
        let account = MemoryAccount::new();
        let mut map =
            TrackedHash::<u64, Vec<i64>, std::hash::BuildHasherDefault<CollidingHasher>>::default();
        map.attach(&account);
        map.reserve(112);
        for key in 0..112 {
            map.insert(key, vec![key as i64]);
        }
        let physical = map.allocation_size();
        for key in 0..70 {
            map.remove(&key);
        }
        assert_eq!(map.allocation_size(), physical);
        assert_eq!(account.snapshot().retained_bytes, physical + 42 * 8);
        for key in 112..400 {
            map.insert(key, vec![key as i64]);
        }
        assert_eq!(
            account.snapshot().retained_bytes,
            map.allocation_size() + 330 * 8
        );
        map.clear();
        assert_eq!(account.snapshot().retained_bytes, map.allocation_size());
        drop(map);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }
}
