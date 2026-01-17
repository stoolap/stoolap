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

//! High-performance i64 HashMap
// - Uses i64::MIN as empty sentinel (row IDs and txn IDs are always >= 0)
// - Direct key storage (no XOR transform)
// - FxHash with pre-mixing (XOR>>16 before multiply) - 0 sequential collisions,
//   65% reduction in strided key collisions
// - Backward-shift deletion (no tombstones)
//
// SAFETY: i64::MIN CANNOT be used as a key - it is reserved as the empty sentinel.
// Inserting i64::MIN will cause silent data corruption. This is enforced via
// assertions in debug and release builds.

use std::mem::MaybeUninit;

const MIN_CAPACITY: usize = 8;
const LOAD_FACTOR_NUM: usize = 3;
const LOAD_FACTOR_DEN: usize = 4;

// Empty sentinel - i64::MIN is never used as row ID or transaction ID
const EMPTY: i64 = i64::MIN;

/// Panics if key is the reserved EMPTY sentinel (i64::MIN).
/// Cold-path annotated to minimize branch prediction overhead.
#[cold]
#[inline(never)]
fn assert_valid_key(key: i64) {
    if key == EMPTY {
        panic!("i64::MIN cannot be used as a key in I64Map (reserved as empty sentinel)");
    }
}

/// Checks if key is valid. Branch predictor should always predict true.
#[inline(always)]
fn check_key(key: i64) {
    if key == EMPTY {
        assert_valid_key(key);
    }
}

/// Slot with key and value. key == EMPTY means slot is empty.
#[repr(C)]
struct Slot<V> {
    key: i64,
    value: MaybeUninit<V>,
}

/// High-performance HashMap for i64 keys.
/// Note: i64::MIN cannot be used as a key (reserved as empty sentinel).
pub struct I64Map<V> {
    slots: Box<[Slot<V>]>,
    len: usize,
    mask: usize,
}

impl<V: Clone> Clone for I64Map<V> {
    fn clone(&self) -> Self {
        let mut new_map = Self::with_capacity(self.len);
        for (key, value) in self.iter() {
            new_map.insert(key, value.clone());
        }
        new_map
    }
}

impl<V> Default for I64Map<V> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<V> I64Map<V> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let cap = if capacity == 0 {
            MIN_CAPACITY
        } else {
            capacity
                .saturating_mul(LOAD_FACTOR_DEN)
                .saturating_div(LOAD_FACTOR_NUM)
                .next_power_of_two()
                .max(MIN_CAPACITY)
        };

        let slots: Vec<Slot<V>> = (0..cap)
            .map(|_| Slot {
                key: EMPTY,
                value: MaybeUninit::uninit(),
            })
            .collect();

        Self {
            slots: slots.into_boxed_slice(),
            len: 0,
            mask: cap - 1,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// FxHash with pre-mixing - XOR the key with its shifted self before
    /// multiplication to break stride patterns while preserving bijectivity.
    /// This maintains 0 collisions for sequential keys while reducing strided
    /// key collisions by ~65% (e.g., stride=1024 goes from 99,872 to 34,464).
    #[inline(always)]
    fn hash(key: i64) -> usize {
        let k = key as u64;
        let k = k ^ (k >> 16); // Pre-mix to break stride patterns (bijective)
        k.wrapping_mul(0x517cc1b727220a95) as usize
    }

    #[inline(always)]
    pub fn insert(&mut self, key: i64, value: V) -> Option<V> {
        check_key(key);

        if self.len * LOAD_FACTOR_DEN >= self.slots.len() * LOAD_FACTOR_NUM {
            self.grow();
        }

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        loop {
            // SAFETY: idx is always in bounds due to masking with (capacity - 1).
            let slot = unsafe { self.slots.get_unchecked_mut(idx) };

            if slot.key == EMPTY {
                // Empty slot - insert here
                slot.key = key;
                slot.value.write(value);
                self.len += 1;
                return None;
            }

            if slot.key == key {
                // Key exists - replace value
                // SAFETY: slot.key == key means this slot is occupied with initialized value.
                let old = unsafe { slot.value.as_ptr().read() };
                slot.value.write(value);
                return Some(old);
            }

            idx = (idx + 1) & mask;
        }
    }

    #[inline(always)]
    pub fn get(&self, key: i64) -> Option<&V> {
        check_key(key);

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        loop {
            // SAFETY: idx is always in bounds due to masking with (capacity - 1).
            let slot = unsafe { self.slots.get_unchecked(idx) };

            if slot.key == EMPTY {
                return None;
            }

            if slot.key == key {
                // SAFETY: slot.key == key means this slot is occupied with initialized value.
                return Some(unsafe { slot.value.assume_init_ref() });
            }

            idx = (idx + 1) & mask;
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, key: i64) -> Option<&mut V> {
        check_key(key);

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        // Find index first to avoid borrow issues
        let found_idx = loop {
            // SAFETY: idx is always in bounds due to masking with (capacity - 1).
            let slot = unsafe { self.slots.get_unchecked(idx) };

            if slot.key == EMPTY {
                return None;
            }

            if slot.key == key {
                break idx;
            }

            idx = (idx + 1) & mask;
        };

        // SAFETY: found_idx is valid and the slot at that index is occupied (key matched).
        Some(unsafe {
            self.slots
                .get_unchecked_mut(found_idx)
                .value
                .assume_init_mut()
        })
    }

    #[inline(always)]
    pub fn contains_key(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    #[inline(always)]
    pub fn remove(&mut self, key: i64) -> Option<V> {
        check_key(key);

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        // Find the key
        loop {
            // SAFETY: idx is always in bounds due to masking with (capacity - 1).
            let slot = unsafe { self.slots.get_unchecked(idx) };

            if slot.key == EMPTY {
                return None;
            }

            if slot.key == key {
                break;
            }

            idx = (idx + 1) & mask;
        }

        // Found - extract value
        // SAFETY: idx is valid and slot is occupied (we just found the key).
        let value = unsafe { self.slots.get_unchecked(idx).value.as_ptr().read() };
        self.len -= 1;

        // Backward shift deletion
        let mut empty_idx = idx;
        let mut next_idx = (idx + 1) & mask;

        loop {
            // SAFETY: next_idx is always in bounds due to masking.
            let next_slot = unsafe { self.slots.get_unchecked(next_idx) };

            if next_slot.key == EMPTY {
                break;
            }

            let next_home = Self::hash(next_slot.key) & mask;

            // Check if empty_idx is between next_home and next_idx (considering wrap)
            let can_move = if next_home <= next_idx {
                empty_idx >= next_home && empty_idx < next_idx
            } else {
                empty_idx >= next_home || empty_idx < next_idx
            };

            if can_move {
                // Move entry back
                // SAFETY: Both indices are in bounds, src slot is occupied, dst slot is empty.
                unsafe {
                    let src = self.slots.as_ptr().add(next_idx);
                    let dst = self.slots.as_mut_ptr().add(empty_idx);
                    (*dst).key = (*src).key;
                    std::ptr::copy_nonoverlapping(
                        (*src).value.as_ptr(),
                        (*dst).value.as_mut_ptr(),
                        1,
                    );
                }
                empty_idx = next_idx;
            }

            next_idx = (next_idx + 1) & mask;
        }

        // SAFETY: empty_idx is in bounds and we're marking the now-empty slot.
        unsafe {
            self.slots.get_unchecked_mut(empty_idx).key = EMPTY;
        }

        Some(value)
    }

    fn grow(&mut self) {
        let new_cap = (self.slots.len() * 2).max(MIN_CAPACITY);
        let new_mask = new_cap - 1;

        let new_slots: Vec<Slot<V>> = (0..new_cap)
            .map(|_| Slot {
                key: EMPTY,
                value: MaybeUninit::uninit(),
            })
            .collect();

        let old_slots = std::mem::replace(&mut self.slots, new_slots.into_boxed_slice());
        let old_len = self.len;
        self.len = 0;
        self.mask = new_mask;

        for slot in Vec::from(old_slots) {
            if slot.key != EMPTY {
                // SAFETY: slot.key != EMPTY means the value is initialized.
                let value = unsafe { slot.value.assume_init() };
                self.insert(slot.key, value);
            }
        }

        debug_assert_eq!(self.len, old_len);
    }

    pub fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            if slot.key != EMPTY {
                // SAFETY: slot.key != EMPTY means the value is initialized.
                unsafe {
                    std::ptr::drop_in_place(slot.value.as_mut_ptr());
                }
                slot.key = EMPTY;
            }
        }
        self.len = 0;
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (i64, &V)> {
        self.slots.iter().filter_map(|slot| {
            if slot.key != EMPTY {
                // SAFETY: slot.key != EMPTY means the value is initialized.
                Some((slot.key, unsafe { slot.value.assume_init_ref() }))
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = i64> + '_ {
        self.slots
            .iter()
            .filter_map(|s| if s.key != EMPTY { Some(s.key) } else { None })
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.slots.iter().filter_map(|slot| {
            if slot.key != EMPTY {
                // SAFETY: slot.key != EMPTY means the value is initialized.
                Some(unsafe { slot.value.assume_init_ref() })
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (i64, &mut V)> {
        self.slots.iter_mut().filter_map(|slot| {
            if slot.key != EMPTY {
                // SAFETY: slot.key != EMPTY means the value is initialized.
                Some((slot.key, unsafe { slot.value.assume_init_mut() }))
            } else {
                None
            }
        })
    }

    /// Drains all entries from the map, returning an iterator over them
    #[inline]
    pub fn drain(&mut self) -> Drain<V> {
        // Take slots and replace with fresh minimum-capacity slots
        let old_slots = std::mem::replace(
            &mut self.slots,
            (0..MIN_CAPACITY)
                .map(|_| Slot {
                    key: EMPTY,
                    value: MaybeUninit::uninit(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        self.len = 0;
        self.mask = MIN_CAPACITY - 1;
        Drain {
            slots: old_slots,
            pos: 0,
        }
    }

    #[inline(always)]
    pub fn entry(&mut self, key: i64) -> Entry<'_, V> {
        check_key(key);

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        // First, check if key exists WITHOUT growing
        loop {
            // SAFETY: idx is always in bounds due to masking with (capacity - 1).
            let slot = unsafe { self.slots.get_unchecked(idx) };

            if slot.key == EMPTY {
                // Key not found - now check if we need to grow before insertion
                if self.len * LOAD_FACTOR_DEN >= self.slots.len() * LOAD_FACTOR_NUM {
                    self.grow();
                    // After grow, need to find the slot again
                    let new_mask = self.mask;
                    let mut new_idx = Self::hash(key) & new_mask;
                    loop {
                        // SAFETY: new_idx is always in bounds due to masking.
                        let slot = unsafe { self.slots.get_unchecked(new_idx) };
                        if slot.key == EMPTY {
                            return Entry::Vacant(VacantEntry {
                                map: self,
                                key,
                                idx: new_idx,
                            });
                        }
                        new_idx = (new_idx + 1) & new_mask;
                    }
                }
                return Entry::Vacant(VacantEntry {
                    map: self,
                    key,
                    idx,
                });
            }

            if slot.key == key {
                return Entry::Occupied(OccupiedEntry { map: self, idx });
            }

            idx = (idx + 1) & mask;
        }
    }
}

impl<V> Drop for I64Map<V> {
    fn drop(&mut self) {
        for slot in self.slots.iter_mut() {
            if slot.key != EMPTY {
                // SAFETY: slot.key != EMPTY means the value is initialized.
                unsafe {
                    std::ptr::drop_in_place(slot.value.as_mut_ptr());
                }
            }
        }
    }
}

pub enum Entry<'a, V> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, V>),
}

impl<'a, V> Entry<'a, V> {
    #[inline(always)]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    #[inline(always)]
    pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(f()),
        }
    }

    #[inline(always)]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(V::default()),
        }
    }

    #[inline(always)]
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            Entry::Occupied(mut e) => {
                f(e.get_mut());
                Entry::Occupied(e)
            }
            Entry::Vacant(e) => Entry::Vacant(e),
        }
    }
}

pub struct OccupiedEntry<'a, V> {
    map: &'a mut I64Map<V>,
    idx: usize,
}

impl<'a, V> OccupiedEntry<'a, V> {
    #[inline(always)]
    pub fn get(&self) -> &V {
        // SAFETY: OccupiedEntry is only created for occupied slots, so idx is
        // valid and the value at that index is initialized.
        unsafe {
            self.map
                .slots
                .get_unchecked(self.idx)
                .value
                .assume_init_ref()
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: OccupiedEntry is only created for occupied slots, so idx is
        // valid and the value at that index is initialized.
        unsafe {
            self.map
                .slots
                .get_unchecked_mut(self.idx)
                .value
                .assume_init_mut()
        }
    }

    #[inline(always)]
    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: OccupiedEntry is only created for occupied slots, so idx is
        // valid and the value at that index is initialized.
        unsafe {
            self.map
                .slots
                .get_unchecked_mut(self.idx)
                .value
                .assume_init_mut()
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, value: V) -> V {
        // SAFETY: OccupiedEntry is only created for occupied slots.
        let slot = unsafe { self.map.slots.get_unchecked_mut(self.idx) };
        // SAFETY: The slot is occupied, so value is initialized.
        let old = unsafe { slot.value.as_ptr().read() };
        slot.value.write(value);
        old
    }

    #[inline(always)]
    pub fn remove(self) -> V {
        // Extract value directly - we already have the index
        // SAFETY: OccupiedEntry is only created for occupied slots.
        let key = unsafe { self.map.slots.get_unchecked(self.idx).key };
        // SAFETY: The slot is occupied, so value is initialized.
        let value = unsafe { self.map.slots.get_unchecked(self.idx).value.as_ptr().read() };
        self.map.len -= 1;

        // Backward shift deletion at known index
        let mask = self.map.mask;
        let mut empty_idx = self.idx;
        let mut next_idx = (self.idx + 1) & mask;

        loop {
            // SAFETY: next_idx is always in bounds due to masking.
            let next_slot = unsafe { self.map.slots.get_unchecked(next_idx) };

            if next_slot.key == EMPTY {
                break;
            }

            let next_home = I64Map::<V>::hash(next_slot.key) & mask;

            // Check if empty_idx is between next_home and next_idx (considering wrap)
            let can_move = if next_home <= next_idx {
                empty_idx >= next_home && empty_idx < next_idx
            } else {
                empty_idx >= next_home || empty_idx < next_idx
            };

            if can_move {
                // Move entry back
                // SAFETY: Both indices are in bounds, src slot is occupied, dst slot is empty.
                unsafe {
                    let src = self.map.slots.as_ptr().add(next_idx);
                    let dst = self.map.slots.as_mut_ptr().add(empty_idx);
                    (*dst).key = (*src).key;
                    std::ptr::copy_nonoverlapping(
                        (*src).value.as_ptr(),
                        (*dst).value.as_mut_ptr(),
                        1,
                    );
                }
                empty_idx = next_idx;
            }

            next_idx = (next_idx + 1) & mask;
        }

        // SAFETY: empty_idx is in bounds and we're marking the now-empty slot.
        unsafe {
            self.map.slots.get_unchecked_mut(empty_idx).key = EMPTY;
        }

        // Suppress unused variable warning
        let _ = key;

        value
    }
}

pub struct VacantEntry<'a, V> {
    map: &'a mut I64Map<V>,
    key: i64,
    idx: usize,
}

impl<'a, V> VacantEntry<'a, V> {
    #[inline(always)]
    pub fn key(&self) -> i64 {
        self.key
    }

    #[inline(always)]
    pub fn insert(self, value: V) -> &'a mut V {
        // Direct insert at pre-computed index - NO re-lookup needed
        // SAFETY: VacantEntry stores a valid idx that was found during entry() lookup.
        let slot = unsafe { self.map.slots.get_unchecked_mut(self.idx) };
        slot.key = self.key;
        slot.value.write(value);
        self.map.len += 1;
        // SAFETY: We just wrote the value, so it's initialized.
        unsafe { slot.value.assume_init_mut() }
    }
}

/// Owning iterator over the entries of an I64Map
pub struct IntoIter<V> {
    slots: Box<[Slot<V>]>,
    pos: usize,
}

impl<V> Iterator for IntoIter<V> {
    type Item = (i64, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.slots.len() {
            let slot = &mut self.slots[self.pos];
            self.pos += 1;

            if slot.key != EMPTY {
                let key = slot.key;
                // SAFETY: slot.key != EMPTY means the value is initialized.
                let value = unsafe { slot.value.as_ptr().read() };
                slot.key = EMPTY; // Mark as consumed to prevent double-drop
                return Some((key, value));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len() - self.pos))
    }
}

impl<V> Drop for IntoIter<V> {
    fn drop(&mut self) {
        // Drop remaining unconsumed elements
        while self.pos < self.slots.len() {
            let slot = &mut self.slots[self.pos];
            self.pos += 1;

            if slot.key != EMPTY {
                // SAFETY: slot.key != EMPTY means the value is initialized.
                unsafe {
                    std::ptr::drop_in_place(slot.value.as_mut_ptr());
                }
                slot.key = EMPTY; // Mark as dropped
            }
        }
    }
}

impl<V> IntoIterator for I64Map<V> {
    type Item = (i64, V);
    type IntoIter = IntoIter<V>;

    fn into_iter(mut self) -> Self::IntoIter {
        let slots = std::mem::take(&mut self.slots);
        self.len = 0; // Prevent drop from cleaning up values we're moving out
        IntoIter { slots, pos: 0 }
    }
}

/// Draining iterator over the entries of an I64Map
pub struct Drain<V> {
    slots: Box<[Slot<V>]>,
    pos: usize,
}

impl<V> Iterator for Drain<V> {
    type Item = (i64, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.slots.len() {
            let slot = &mut self.slots[self.pos];
            self.pos += 1;

            if slot.key != EMPTY {
                let key = slot.key;
                // SAFETY: slot.key != EMPTY means the value is initialized.
                let value = unsafe { slot.value.as_ptr().read() };
                slot.key = EMPTY; // Mark as consumed to prevent double-drop
                return Some((key, value));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.slots.len() - self.pos))
    }
}

impl<V> Drop for Drain<V> {
    fn drop(&mut self) {
        // Consume remaining elements to ensure they're dropped
        for _ in self.by_ref() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Helper struct to track drops
    struct DropTracker {
        count: Rc<RefCell<usize>>,
    }

    impl DropTracker {
        fn new(count: Rc<RefCell<usize>>) -> Self {
            Self { count }
        }
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            *self.count.borrow_mut() += 1;
        }
    }

    #[test]
    fn test_into_iter_partial_consume_drops_remaining() {
        let drop_count = Rc::new(RefCell::new(0));

        let mut map = I64Map::new();
        map.insert(1, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(2, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(3, DropTracker::new(Rc::clone(&drop_count)));

        // Only consume one element
        let mut iter = map.into_iter();
        let _ = iter.next(); // Consume 1 item

        // Drop count should be 1 (the consumed item)
        assert_eq!(*drop_count.borrow(), 1);

        // Drop the iterator without consuming remaining elements
        drop(iter);

        // All 3 items should now be dropped
        assert_eq!(
            *drop_count.borrow(),
            3,
            "Memory leak detected! Only {} items dropped",
            *drop_count.borrow()
        );
    }

    #[test]
    fn test_into_iter_no_consume_drops_all() {
        let drop_count = Rc::new(RefCell::new(0));

        let mut map = I64Map::new();
        map.insert(1, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(2, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(3, DropTracker::new(Rc::clone(&drop_count)));

        // Don't consume any elements
        let iter = map.into_iter();
        drop(iter);

        // All 3 items should be dropped
        assert_eq!(
            *drop_count.borrow(),
            3,
            "Memory leak detected! Only {} items dropped",
            *drop_count.borrow()
        );
    }

    #[test]
    fn test_into_iter_full_consume() {
        let drop_count = Rc::new(RefCell::new(0));

        let mut map = I64Map::new();
        map.insert(1, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(2, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(3, DropTracker::new(Rc::clone(&drop_count)));

        // Fully consume the iterator
        for _ in map.into_iter() {}

        // All 3 items should be dropped
        assert_eq!(*drop_count.borrow(), 3);
    }

    #[test]
    fn test_drain_partial_consume_drops_remaining() {
        let drop_count = Rc::new(RefCell::new(0));

        let mut map = I64Map::new();
        map.insert(1, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(2, DropTracker::new(Rc::clone(&drop_count)));
        map.insert(3, DropTracker::new(Rc::clone(&drop_count)));

        // Only consume one element from drain
        let mut drain = map.drain();
        let _ = drain.next();

        assert_eq!(*drop_count.borrow(), 1);

        // Drop drain
        drop(drain);

        // All 3 should be dropped
        assert_eq!(*drop_count.borrow(), 3);
    }

    #[test]
    fn test_basic_operations() {
        let mut map = I64Map::new();

        assert!(map.insert(1, "one").is_none());
        assert!(map.insert(2, "two").is_none());
        assert!(map.insert(3, "three").is_none());
        assert_eq!(map.len(), 3);

        assert_eq!(map.get(1), Some(&"one"));
        assert_eq!(map.get(2), Some(&"two"));
        assert_eq!(map.get(3), Some(&"three"));
        assert_eq!(map.get(4), None);

        assert_eq!(map.insert(2, "TWO"), Some("two"));
        assert_eq!(map.get(2), Some(&"TWO"));

        assert_eq!(map.remove(2), Some("TWO"));
        assert_eq!(map.get(2), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_entry_api() {
        let mut map = I64Map::new();

        *map.entry(1).or_insert(10) += 5;
        assert_eq!(map.get(1), Some(&15));

        *map.entry(1).or_insert(100) += 5;
        assert_eq!(map.get(1), Some(&20));

        map.entry(2).or_insert_with(|| 42);
        assert_eq!(map.get(2), Some(&42));

        let v: &mut i32 = map.entry(3).or_default();
        *v = 99;
        assert_eq!(map.get(3), Some(&99));
    }

    #[test]
    fn test_grow() {
        let mut map = I64Map::new();

        for i in 0..1000 {
            map.insert(i, i * 2);
        }

        assert_eq!(map.len(), 1000);

        for i in 0..1000 {
            assert_eq!(map.get(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_edge_values() {
        let mut map = I64Map::new();

        // i64::MIN is reserved as EMPTY sentinel, so we test other edge values
        map.insert(i64::MIN + 1, "near_min");
        map.insert(i64::MAX, "max");
        map.insert(0, "zero");
        map.insert(-1, "neg one");
        map.insert(1, "one");

        assert_eq!(map.get(i64::MIN + 1), Some(&"near_min"));
        assert_eq!(map.get(i64::MAX), Some(&"max"));
        assert_eq!(map.get(0), Some(&"zero"));
        assert_eq!(map.get(-1), Some(&"neg one"));
        assert_eq!(map.get(1), Some(&"one"));
    }

    #[test]
    fn test_deletion() {
        let mut map = I64Map::with_capacity(16);

        for i in 0..10 {
            map.insert(i, i);
        }

        map.remove(5);
        assert!(!map.contains_key(5));

        for i in 0..10 {
            if i != 5 {
                assert_eq!(map.get(i), Some(&i));
            }
        }

        map.insert(5, 55);
        assert_eq!(map.get(5), Some(&55));
    }

    #[test]
    fn test_clear() {
        let mut map = I64Map::new();

        for i in 0..100 {
            map.insert(i, i);
        }

        map.clear();
        assert!(map.is_empty());

        for i in 0..100 {
            assert!(!map.contains_key(i));
        }
    }

    #[test]
    fn test_iterators() {
        let mut map = I64Map::new();

        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        let mut keys: Vec<_> = map.keys().collect();
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3]);

        let mut values: Vec<_> = map.values().copied().collect();
        values.sort();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn test_drain() {
        let mut map = I64Map::new();

        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        let mut drained: Vec<_> = map.drain().collect();
        drained.sort_by_key(|(k, _)| *k);

        assert_eq!(drained, vec![(1, 10), (2, 20), (3, 30)]);
        assert!(map.is_empty());

        // Map should still be usable after drain
        map.insert(4, 40);
        assert_eq!(map.get(4), Some(&40));
    }

    #[test]
    fn test_strided_keys_no_collision_catastrophe() {
        // This test verifies that strided keys (e.g., multiples of 1024)
        // don't cause catastrophic collisions that would result in O(N^2) behavior.
        // With the old low-bit masking hash, this would timeout or be very slow.
        let mut map = I64Map::with_capacity(10000);
        let stride = 1024;

        // Insert 10000 keys with stride of 1024
        for i in 0..10000i64 {
            map.insert(i * stride, i);
        }

        // Verify all keys are present and correct
        assert_eq!(map.len(), 10000);
        for i in 0..10000i64 {
            assert_eq!(map.get(i * stride), Some(&i), "Missing key {}", i * stride);
        }

        // Remove half and verify
        for i in (0..10000i64).step_by(2) {
            assert_eq!(map.remove(i * stride), Some(i));
        }
        assert_eq!(map.len(), 5000);

        // Verify remaining half
        for i in (1..10000i64).step_by(2) {
            assert_eq!(map.get(i * stride), Some(&i));
        }
    }

    #[test]
    #[should_panic(expected = "i64::MIN cannot be used as a key")]
    fn test_i64_min_panics_on_insert() {
        let mut map = I64Map::<i64>::new();
        map.insert(i64::MIN, 42);
    }

    #[test]
    #[should_panic(expected = "i64::MIN cannot be used as a key")]
    fn test_i64_min_panics_on_get() {
        let map = I64Map::<i64>::new();
        let _ = map.get(i64::MIN);
    }

    #[test]
    #[should_panic(expected = "i64::MIN cannot be used as a key")]
    fn test_i64_min_panics_on_entry() {
        let mut map = I64Map::<i64>::new();
        let _ = map.entry(i64::MIN);
    }
}
