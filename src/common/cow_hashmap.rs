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

//! Copy-on-Write Hash Map for i64 keys
//!
//! Design goals:
//! - Blazing fast reads: O(1) average, same hash as I64Map
//! - Low memory: 8 bytes key + sizeof(V) per slot, adaptive load factor
//! - O(1) snapshot cloning for lock-free iteration
//! - COW semantics: writers clone only when shared
//! - Auto-shrink after deletes to reclaim memory
//!
//! Memory layout (single allocation):
//! ```text
//! [ Header (12 bytes) | Slots (capacity * (8 + sizeof(V))) ]
//! ```
//!
//! Uses same optimizations as I64Map:
//! - i64::MIN as empty sentinel (no metadata array)
//! - FxHash with pre-mixing for 0 sequential collisions
//! - Backward-shift deletion (no tombstones)
//!
//! Thread safety:
//! - Wrap in `Mutex<CowHashMap<V>>` for concurrent access (preferred for short operations)
//! - `RwLock<CowHashMap<V>>` works but Mutex is often faster due to short critical sections
//! - `map.lock().clone()` gives O(1) snapshot for lock-free iteration
//! - Writers automatically COW if the map is shared
//!
//! SAFETY: i64::MIN CANNOT be used as a key (reserved as empty sentinel).

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU32, Ordering};

/// Empty sentinel - same as I64Map
const EMPTY: i64 = i64::MIN;

/// Minimum capacity (must be power of 2)
const MIN_CAPACITY: usize = 8;

/// Shrink threshold: shrink when len < capacity / SHRINK_DIVISOR
/// Only shrink if capacity > MIN_SHRINK_CAPACITY to avoid thrashing
const SHRINK_DIVISOR: usize = 4;
const MIN_SHRINK_CAPACITY: usize = 64;

/// Adaptive load factor based on capacity to reduce memory waste.
/// Smaller maps use lower load factor (faster), larger maps use higher (less memory waste).
#[inline]
fn load_factor_for_capacity(capacity: usize) -> (usize, usize) {
    if capacity <= 1024 {
        (3, 4) // 75% - fast for small maps
    } else if capacity <= 65536 {
        (7, 8) // 87.5% - balanced
    } else {
        (15, 16) // 93.75% - memory efficient for large maps
    }
}

/// Calculate capacity needed for n entries with adaptive load factor
#[inline]
fn capacity_for_entries(n: usize) -> usize {
    if n == 0 {
        return MIN_CAPACITY;
    }

    // Start with a guess and refine
    let mut cap = n.next_power_of_two();

    // Check if this capacity works with its load factor
    loop {
        let (num, den) = load_factor_for_capacity(cap);
        let max_entries = cap * num / den;
        if max_entries >= n {
            return cap.max(MIN_CAPACITY);
        }
        cap *= 2;
    }
}

/// Slot with key and value. key == EMPTY means slot is empty.
#[repr(C)]
struct Slot<V> {
    key: i64,
    value: MaybeUninit<V>,
}

/// Header for the hash map allocation
///
/// Layout: 12 bytes total (optimized from 24 bytes)
/// [ capacity: 4 | len: 4 | drop_count: 4 ]
///
/// Using u32 for capacity/len limits us to ~4B entries, which is plenty.
/// This saves 12 bytes per map vs u64 and improves cache efficiency.
#[repr(C)]
struct Header {
    /// Number of slots (always power of 2, max ~4B)
    capacity: u32,
    /// Number of occupied entries
    len: u32,
    /// Reference count for COW (like CowBTree's drop_count)
    drop_count: AtomicU32,
}

impl Header {
    #[inline]
    fn get_capacity(&self) -> usize {
        self.capacity as usize
    }
}

/// A Copy-on-Write hash map optimized for i64 keys.
///
/// Uses same algorithm as I64Map but with COW semantics:
/// - O(1) snapshot cloning (just increment reference count)
/// - Writers automatically deep-clone if map is shared
///
/// SAFETY: i64::MIN cannot be used as a key (reserved as empty sentinel).
pub struct CowHashMap<V: Clone> {
    /// Pointer to the allocation (Header + slots)
    ptr: NonNull<u8>,
    _marker: PhantomData<V>,
}

// SAFETY: CowHashMap is Send if V is Send (data is owned, no thread-local state)
unsafe impl<V: Clone + Send> Send for CowHashMap<V> {}

// SAFETY: CowHashMap is Sync if V is Sync (reads are safe, writes require &mut)
unsafe impl<V: Clone + Sync> Sync for CowHashMap<V> {}

impl<V: Clone> Clone for CowHashMap<V> {
    /// O(1) clone - just increments reference count
    #[inline]
    fn clone(&self) -> Self {
        // Increment drop_count atomically
        self.header().drop_count.fetch_add(1, Ordering::Relaxed);

        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<V: Clone> Drop for CowHashMap<V> {
    fn drop(&mut self) {
        let header = self.header();
        let old_count = header.drop_count.fetch_sub(1, Ordering::AcqRel);

        if old_count != 1 {
            // Not the last reference - don't drop contents
            return;
        }

        // We're the last reference - drop contents and deallocate
        let capacity = header.get_capacity();

        if mem::needs_drop::<V>() {
            let slots = self.slots();
            for slot in slots.iter() {
                if slot.key != EMPTY {
                    // SAFETY: We are the last reference (ref_count == 1 checked above).
                    // The slot is occupied (key != EMPTY), so value is initialized.
                    unsafe {
                        ptr::drop_in_place(slot.value.as_ptr() as *mut V);
                    }
                }
            }
        }

        // Deallocate
        let (layout, _) = Self::layout_for_capacity::<V>(capacity);
        // SAFETY: ptr was allocated with this layout, and we're the last reference
        unsafe {
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

impl<V: Clone> Default for CowHashMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Panics if key is the reserved EMPTY sentinel (i64::MIN).
#[cold]
#[inline(never)]
fn assert_valid_key(key: i64) {
    if key == EMPTY {
        panic!("i64::MIN cannot be used as a key in CowHashMap (reserved as empty sentinel)");
    }
}

/// Checks if key is valid. Branch predictor should always predict true.
#[inline(always)]
fn check_key(key: i64) {
    if key == EMPTY {
        assert_valid_key(key);
    }
}

impl<V: Clone> CowHashMap<V> {
    /// Create a new empty hash map
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new hash map with at least the specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity_for_entries(capacity);

        if cap > u32::MAX as usize {
            panic!("CowHashMap capacity overflow: {} > u32::MAX", cap);
        }

        let (layout, slots_offset) = Self::layout_for_capacity::<V>(cap);

        // SAFETY: Layout is valid (non-zero size, proper alignment)
        let ptr = unsafe { alloc(layout) };
        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };

        // Initialize header
        // SAFETY: ptr points to valid memory of correct size
        unsafe {
            let header = ptr.as_ptr() as *mut Header;
            (*header).capacity = cap as u32;
            (*header).len = 0;
            (*header).drop_count = AtomicU32::new(1);
        }

        // Initialize all slots to empty
        // SAFETY: slots area is valid, uninitialized memory
        unsafe {
            let slots_ptr = ptr.as_ptr().add(slots_offset) as *mut Slot<V>;
            for i in 0..cap {
                (*slots_ptr.add(i)).key = EMPTY;
            }
        }

        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Calculate layout for a given capacity
    /// Returns (layout, slots_offset)
    fn layout_for_capacity<T>(capacity: usize) -> (Layout, usize) {
        let header_layout = Layout::new::<Header>();
        let slot_layout = Layout::new::<Slot<T>>();

        // Header + (Slot * capacity)
        let (layout, offset) = header_layout
            .extend(
                Layout::from_size_align(slot_layout.size() * capacity, slot_layout.align())
                    .unwrap(),
            )
            .unwrap();

        (layout, offset)
    }

    #[inline]
    fn slots_offset<T>() -> usize {
        let header_layout = Layout::new::<Header>();
        let slot_layout = Layout::new::<Slot<T>>();
        let (_, offset) = header_layout.extend(slot_layout).unwrap();
        offset
    }

    #[inline]
    fn header(&self) -> &Header {
        // SAFETY: ptr points to a valid Header
        unsafe { &*(self.ptr.as_ptr() as *const Header) }
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        // SAFETY: ptr points to a valid Header, we have &mut self
        unsafe { &mut *(self.ptr.as_ptr() as *mut Header) }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.header().capacity as usize
    }

    #[inline]
    fn mask(&self) -> usize {
        self.capacity() - 1
    }

    /// Returns the number of elements in the map
    #[inline]
    pub fn len(&self) -> usize {
        self.header().len as usize
    }

    /// Returns true if the map is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn slots(&self) -> &[Slot<V>] {
        let capacity = self.capacity();
        let offset = Self::slots_offset::<V>();
        // SAFETY: slots area is valid, capacity slots available
        unsafe {
            let slots_ptr = self.ptr.as_ptr().add(offset) as *const Slot<V>;
            std::slice::from_raw_parts(slots_ptr, capacity)
        }
    }

    #[inline]
    fn slots_mut(&mut self) -> &mut [Slot<V>] {
        let capacity = self.capacity();
        let offset = Self::slots_offset::<V>();
        // SAFETY: slots area is valid, we have &mut self
        unsafe {
            let slots_ptr = self.ptr.as_ptr().add(offset) as *mut Slot<V>;
            std::slice::from_raw_parts_mut(slots_ptr, capacity)
        }
    }

    /// FxHash with pre-mixing - same as I64Map
    /// XOR the key with its shifted self before multiplication to break stride patterns.
    /// Maintains 0 collisions for sequential keys.
    #[inline(always)]
    fn hash(key: i64) -> usize {
        let k = key as u64;
        let k = k ^ (k >> 16); // Pre-mix to break stride patterns (bijective)
        k.wrapping_mul(0x517cc1b727220a95) as usize
    }

    /// Check if we need to ensure unique ownership (COW)
    #[inline]
    fn ensure_unique(&mut self) {
        let drop_count = self.header().drop_count.load(Ordering::Acquire);
        if drop_count > 1 {
            *self = self.deep_clone();
        }
    }

    /// Create a deep clone of this map
    fn deep_clone(&self) -> Self {
        let capacity = self.capacity();
        let len = self.len();

        // Create new map with same capacity
        let mut new_map = Self::with_capacity(len);

        // If new capacity is different (due to load factor), we need to rehash
        if new_map.capacity() != capacity {
            // Rehash into new map
            let slots = self.slots();
            for slot in slots.iter() {
                if slot.key != EMPTY {
                    // SAFETY: slot is occupied, value is initialized
                    let value = unsafe { (*slot.value.as_ptr()).clone() };
                    new_map.insert_internal(slot.key, value);
                }
            }
        } else {
            // Same capacity - direct slot copy (faster than rehashing)
            // new_map already has correct capacity and drop_count=1 from with_capacity
            // Just need to set len and copy slots
            new_map.header_mut().len = len as u32;

            // Clone values (not just copy - need V::clone for non-Copy types)
            let src_slots = self.slots();
            let dst_slots = new_map.slots_mut();

            for i in 0..capacity {
                // EXCEPTION SAFETY:
                // We must clone the value BEFORE writing the key to the destination.
                // If clone() panics, the destination slot remains EMPTY (from with_capacity),
                // so Drop won't try to drop uninitialized memory.
                if src_slots[i].key != EMPTY {
                    // SAFETY: src is valid initialized V, dst is uninitialized
                    unsafe {
                        let value = (*src_slots[i].value.as_ptr()).clone();
                        dst_slots[i].value = MaybeUninit::new(value);
                        // Only set key after value is successfully initialized
                        dst_slots[i].key = src_slots[i].key;
                    }
                }
                // Empty slots already initialized to EMPTY by with_capacity
            }
        }

        new_map
    }

    /// Internal insert without COW check (used during deep_clone and grow)
    fn insert_internal(&mut self, key: i64, value: V) -> Option<V> {
        let mask = self.mask();
        let mut idx = Self::hash(key) & mask;
        let slots = self.slots_mut();

        loop {
            if slots[idx].key == EMPTY {
                // Found empty slot
                slots[idx].key = key;
                slots[idx].value = MaybeUninit::new(value);
                self.header_mut().len += 1;
                return None;
            }
            if slots[idx].key == key {
                // Key exists - replace
                // SAFETY: slot is occupied, value is initialized
                let old = unsafe { ptr::read(slots[idx].value.as_ptr()) };
                slots[idx].value = MaybeUninit::new(value);
                return Some(old);
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Grow the table when load factor exceeded
    fn grow(&mut self) {
        let old_capacity = self.capacity();
        let mut new_map = Self::with_capacity(self.len() * 2);

        // Rehash all entries
        for slot in self.slots_mut().iter_mut().take(old_capacity) {
            if slot.key != EMPTY {
                let key = slot.key;
                // SAFETY: We're moving the value, old slot will be marked empty
                let value = unsafe { ptr::read(slot.value.as_ptr()) };
                slot.key = EMPTY; // Mark as empty so Drop won't try to drop it
                new_map.insert_internal(key, value);
            }
        }

        // Set old len to 0 so Drop won't try to drop moved values
        self.header_mut().len = 0;

        // Replace self with new map
        *self = new_map;
    }

    /// Get a value by key. O(1) average.
    #[inline]
    pub fn get(&self, key: i64) -> Option<&V> {
        check_key(key);

        let mask = self.mask();
        let mut idx = Self::hash(key) & mask;
        let slots = self.slots();

        loop {
            if slots[idx].key == EMPTY {
                return None;
            }
            if slots[idx].key == key {
                // SAFETY: slot is occupied, value is initialized
                return Some(unsafe { &*slots[idx].value.as_ptr() });
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Check if key exists. O(1) average.
    #[inline]
    pub fn contains_key(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    /// Insert a key-value pair. Returns old value if key existed.
    /// Check if we need to grow based on adaptive load factor
    #[inline]
    fn needs_grow(&self) -> bool {
        let cap = self.capacity();
        let (num, den) = load_factor_for_capacity(cap);
        self.len() * den >= cap * num
    }

    /// Check if we should shrink: len < capacity / SHRINK_DIVISOR
    /// Only shrink if capacity > MIN_SHRINK_CAPACITY to avoid thrashing
    #[inline]
    fn should_shrink(&self) -> bool {
        let cap = self.capacity();
        cap > MIN_SHRINK_CAPACITY && self.len() < cap / SHRINK_DIVISOR
    }

    /// Shrink the table to fit current entries
    fn shrink(&mut self) {
        let new_cap = capacity_for_entries(self.len());
        if new_cap >= self.capacity() {
            return; // No need to shrink
        }

        let mut new_map = Self::with_capacity(self.len());

        // Rehash all entries into the new smaller map
        for slot in self.slots_mut().iter_mut() {
            if slot.key != EMPTY {
                let key = slot.key;
                // SAFETY: We're moving the value, old slot will be marked empty
                let value = unsafe { ptr::read(slot.value.as_ptr()) };
                slot.key = EMPTY; // Mark as empty so Drop won't try to drop it
                new_map.insert_internal(key, value);
            }
        }

        // Set old len to 0 so Drop won't try to drop moved values
        self.header_mut().len = 0;

        // Replace self with new map
        *self = new_map;
    }

    /// Shrink the map to fit its current contents, releasing excess memory.
    ///
    /// Call this after removing many entries to reclaim memory.
    pub fn shrink_to_fit(&mut self) {
        self.ensure_unique();
        self.shrink();
    }

    pub fn insert(&mut self, key: i64, value: V) -> Option<V> {
        check_key(key);
        self.ensure_unique();

        // Check if we need to grow
        if self.needs_grow() {
            self.grow();
        }

        self.insert_internal(key, value)
    }

    /// Remove a key from the map. Returns the value if it existed.
    ///
    /// Uses backward-shift deletion (same as I64Map) - no tombstones.
    pub fn remove(&mut self, key: i64) -> Option<V> {
        check_key(key);
        self.ensure_unique();

        // Cache capacity before getting mutable slots borrow
        let capacity = self.capacity();
        let mask = capacity - 1;
        let mut idx = Self::hash(key) & mask;

        // Find the key
        {
            let slots = self.slots();
            loop {
                if slots[idx].key == EMPTY {
                    return None;
                }
                if slots[idx].key == key {
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        // Found it - decrement length and take the value
        self.header_mut().len -= 1;

        let slots = self.slots_mut();

        // SAFETY: slot is occupied, value is initialized
        let value = unsafe { ptr::read(slots[idx].value.as_ptr()) };

        // Backward-shift deletion: shift subsequent entries back
        let mut empty_idx = idx;
        loop {
            let next_idx = (empty_idx + 1) & mask;

            if slots[next_idx].key == EMPTY {
                // Hit empty slot - done
                slots[empty_idx].key = EMPTY;
                break;
            }

            // Check if next entry should be shifted back
            let next_natural = Self::hash(slots[next_idx].key) & mask;

            // Calculate wrapped distances
            let dist_to_empty = if empty_idx >= next_natural {
                empty_idx - next_natural
            } else {
                capacity - next_natural + empty_idx
            };
            let dist_to_next = if next_idx >= next_natural {
                next_idx - next_natural
            } else {
                capacity - next_natural + next_idx
            };

            if dist_to_empty <= dist_to_next {
                // Shift this entry back
                slots[empty_idx].key = slots[next_idx].key;
                // SAFETY: Moving initialized value
                unsafe {
                    ptr::copy_nonoverlapping(
                        slots[next_idx].value.as_ptr(),
                        slots[empty_idx].value.as_mut_ptr(),
                        1,
                    );
                }
                empty_idx = next_idx;
            } else {
                // Can't shift further - mark current as empty
                slots[empty_idx].key = EMPTY;
                break;
            }
        }

        // Check if we should shrink after removal
        if self.should_shrink() {
            self.shrink();
        }

        Some(value)
    }

    /// Get a mutable reference to a value. Triggers COW if shared.
    #[inline]
    pub fn get_mut(&mut self, key: i64) -> Option<&mut V> {
        check_key(key);
        self.ensure_unique();

        let mask = self.mask();
        let mut idx = Self::hash(key) & mask;
        let slots = self.slots_mut();

        loop {
            if slots[idx].key == EMPTY {
                return None;
            }
            if slots[idx].key == key {
                // SAFETY: slot is occupied, value is initialized
                return Some(unsafe { &mut *slots[idx].value.as_mut_ptr() });
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Iterate over all key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (i64, &V)> {
        let slots = self.slots();
        slots.iter().filter_map(|slot| {
            if slot.key != EMPTY {
                // SAFETY: slot is occupied, value is initialized
                Some((slot.key, unsafe { &*slot.value.as_ptr() }))
            } else {
                None
            }
        })
    }

    /// Iterate over all keys
    pub fn keys_iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over all values
    pub fn values_iter(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.ensure_unique();

        for slot in self.slots_mut() {
            if slot.key != EMPTY {
                if mem::needs_drop::<V>() {
                    // SAFETY: slot is occupied, we're dropping it
                    unsafe {
                        ptr::drop_in_place(slot.value.as_mut_ptr());
                    }
                }
                slot.key = EMPTY;
            }
        }

        self.header_mut().len = 0;
    }

    /// Entry API - get or insert
    pub fn entry(&mut self, key: i64) -> Entry<'_, V> {
        check_key(key);
        self.ensure_unique();

        let mask = self.mask();
        let mut idx = Self::hash(key) & mask;
        let slots = self.slots();

        loop {
            if slots[idx].key == EMPTY {
                return Entry::Vacant(VacantEntry { map: self, key });
            }
            if slots[idx].key == key {
                return Entry::Occupied(OccupiedEntry { map: self, idx });
            }
            idx = (idx + 1) & mask;
        }
    }
}

/// Entry API for CowHashMap
pub enum Entry<'a, V: Clone> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, V>),
}

pub struct OccupiedEntry<'a, V: Clone> {
    map: &'a mut CowHashMap<V>,
    idx: usize,
}

pub struct VacantEntry<'a, V: Clone> {
    map: &'a mut CowHashMap<V>,
    key: i64,
}

impl<'a, V: Clone> Entry<'a, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default()),
        }
    }
}

impl<'a, V: Clone> OccupiedEntry<'a, V> {
    pub fn get(&self) -> &V {
        // SAFETY: slot is occupied
        unsafe { &*self.map.slots()[self.idx].value.as_ptr() }
    }

    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: slot is occupied
        unsafe { &mut *self.map.slots_mut()[self.idx].value.as_mut_ptr() }
    }

    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: slot is occupied
        unsafe { &mut *self.map.slots_mut()[self.idx].value.as_mut_ptr() }
    }
}

impl<'a, V: Clone> VacantEntry<'a, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        // May need to grow
        if self.map.needs_grow() {
            self.map.grow();
        }

        let mask = self.map.mask();
        let mut idx = CowHashMap::<V>::hash(self.key) & mask;

        // Find empty slot (linear probe), then insert
        {
            let slots = self.map.slots_mut();
            while slots[idx].key != EMPTY {
                idx = (idx + 1) & mask;
            }
            slots[idx].key = self.key;
            slots[idx].value = MaybeUninit::new(value);
        }

        // Update header after releasing slots borrow
        self.map.header_mut().len += 1;

        // SAFETY: we just initialized this slot at idx
        unsafe { &mut *self.map.slots_mut()[idx].value.as_mut_ptr() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_operations() {
        let mut map: CowHashMap<String> = CowHashMap::new();

        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        // Insert
        assert_eq!(map.insert(1, "one".to_string()), None);
        assert_eq!(map.insert(2, "two".to_string()), None);
        assert_eq!(map.insert(3, "three".to_string()), None);

        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());

        // Get
        assert_eq!(map.get(1), Some(&"one".to_string()));
        assert_eq!(map.get(2), Some(&"two".to_string()));
        assert_eq!(map.get(3), Some(&"three".to_string()));
        assert_eq!(map.get(4), None);

        // Contains
        assert!(map.contains_key(1));
        assert!(!map.contains_key(4));

        // Update
        assert_eq!(map.insert(1, "ONE".to_string()), Some("one".to_string()));
        assert_eq!(map.get(1), Some(&"ONE".to_string()));

        // Remove
        assert_eq!(map.remove(2), Some("two".to_string()));
        assert_eq!(map.get(2), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_cow_semantics() {
        let mut map: CowHashMap<i64> = CowHashMap::new();
        map.insert(1, 100);
        map.insert(2, 200);

        // Clone (O(1))
        let snapshot = map.clone();

        // Verify snapshot has same data
        assert_eq!(snapshot.get(1), Some(&100));
        assert_eq!(snapshot.get(2), Some(&200));

        // Modify original (triggers COW)
        map.insert(3, 300);
        map.insert(1, 111);

        // Original is modified
        assert_eq!(map.get(1), Some(&111));
        assert_eq!(map.get(3), Some(&300));

        // Snapshot is unchanged
        assert_eq!(snapshot.get(1), Some(&100));
        assert_eq!(snapshot.get(3), None);
    }

    #[test]
    fn test_growth() {
        let mut map: CowHashMap<i64> = CowHashMap::new();

        // Insert many entries to trigger growth
        for i in 0..1000 {
            map.insert(i, i * 2);
        }

        assert_eq!(map.len(), 1000);

        // Verify all entries
        for i in 0..1000 {
            assert_eq!(map.get(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_negative_keys() {
        let mut map: CowHashMap<String> = CowHashMap::new();

        map.insert(-1, "negative one".to_string());
        map.insert(0, "zero".to_string());
        // Note: i64::MIN is reserved, use i64::MIN + 1
        map.insert(i64::MIN + 1, "near_min".to_string());
        map.insert(i64::MAX, "max".to_string());

        assert_eq!(map.get(-1), Some(&"negative one".to_string()));
        assert_eq!(map.get(0), Some(&"zero".to_string()));
        assert_eq!(map.get(i64::MIN + 1), Some(&"near_min".to_string()));
        assert_eq!(map.get(i64::MAX), Some(&"max".to_string()));
    }

    #[test]
    fn test_concurrent_snapshots() {
        let map: Arc<RwLock<CowHashMap<i64>>> = Arc::new(RwLock::new(CowHashMap::new()));

        // Insert initial data
        {
            let mut guard = map.write();
            for i in 0..100 {
                guard.insert(i, i * 10);
            }
        }

        // Spawn reader threads that take snapshots
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    for _ in 0..100 {
                        // O(1) snapshot
                        let snapshot = map.read().clone();

                        // Lock-free iteration
                        let sum: i64 = snapshot.iter().map(|(_, v)| *v).sum();
                        assert!(sum >= 0); // Just verify it doesn't crash
                    }
                })
            })
            .collect();

        // Spawn writer thread
        let writer_map = Arc::clone(&map);
        let writer = thread::spawn(move || {
            for i in 100..200 {
                writer_map.write().insert(i, i * 10);
            }
        });

        writer.join().unwrap();
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(map.read().len(), 200);
    }

    #[test]
    fn test_clear() {
        let mut map: CowHashMap<String> = CowHashMap::new();

        for i in 0..100 {
            map.insert(i, format!("value_{}", i));
        }

        assert_eq!(map.len(), 100);

        map.clear();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.get(0), None);
    }

    #[test]
    fn test_backward_shift_delete() {
        let mut map: CowHashMap<i64> = CowHashMap::new();

        // Insert entries that might cluster
        for i in 0..50 {
            map.insert(i, i);
        }

        // Remove some entries
        for i in (0..50).step_by(2) {
            map.remove(i);
        }

        assert_eq!(map.len(), 25);

        // Verify remaining entries
        for i in (1..50).step_by(2) {
            assert_eq!(map.get(i), Some(&i));
        }

        // Verify removed entries
        for i in (0..50).step_by(2) {
            assert_eq!(map.get(i), None);
        }
    }

    #[test]
    fn test_iter() {
        let mut map: CowHashMap<i64> = CowHashMap::new();

        for i in 0..10 {
            map.insert(i, i * i);
        }

        let mut pairs: Vec<_> = map.iter().map(|(k, v)| (k, *v)).collect();
        pairs.sort_by_key(|(k, _)| *k);

        assert_eq!(pairs.len(), 10);
        for (i, (k, v)) in pairs.iter().enumerate() {
            assert_eq!(*k, i as i64);
            assert_eq!(*v, (i * i) as i64);
        }
    }

    #[test]
    fn test_entry_api() {
        let mut map: CowHashMap<i64> = CowHashMap::new();

        // Insert via entry
        *map.entry(1).or_insert(0) += 10;
        assert_eq!(map.get(1), Some(&10));

        // Update via entry
        *map.entry(1).or_insert(0) += 5;
        assert_eq!(map.get(1), Some(&15));

        // or_insert_with
        map.entry(2).or_insert_with(|| 100);
        assert_eq!(map.get(2), Some(&100));
    }

    #[test]
    #[should_panic(expected = "i64::MIN cannot be used")]
    fn test_min_key_panics() {
        let mut map: CowHashMap<i64> = CowHashMap::new();
        map.insert(i64::MIN, 0); // Should panic
    }

    #[test]
    fn test_shrink_after_delete() {
        let mut map: CowHashMap<i64> = CowHashMap::new();

        // Insert many entries to grow the map
        for i in 0..1000 {
            map.insert(i, i * 2);
        }

        let capacity_after_insert = map.capacity();
        assert!(capacity_after_insert >= 1000);

        // Remove most entries (keep only 10)
        for i in 10..1000 {
            map.remove(i);
        }

        assert_eq!(map.len(), 10);

        // Capacity should have shrunk (automatic shrink after remove)
        let capacity_after_remove = map.capacity();
        assert!(
            capacity_after_remove < capacity_after_insert,
            "capacity should shrink: {} < {}",
            capacity_after_remove,
            capacity_after_insert
        );

        // Verify remaining entries still work
        for i in 0..10 {
            assert_eq!(map.get(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut map: CowHashMap<i64> = CowHashMap::with_capacity(1000);

        // Insert only a few entries
        for i in 0..10 {
            map.insert(i, i);
        }

        let initial_capacity = map.capacity();
        assert!(initial_capacity >= 1000);

        // Shrink to fit
        map.shrink_to_fit();

        let after_shrink = map.capacity();
        assert!(
            after_shrink < initial_capacity,
            "capacity should shrink: {} < {}",
            after_shrink,
            initial_capacity
        );

        // Verify entries still work
        for i in 0..10 {
            assert_eq!(map.get(i), Some(&i));
        }
    }
}
