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
// - Uses i64::MIN as the empty marker inside the slot array; the entry
//   for that key lives in a side slot, so the full i64 range is storable
// - Direct key storage (no XOR transform)
// - FxHash with pre-mixing (XOR>>16 before multiply) - 0 sequential collisions,
//   65% reduction in strided key collisions
// - Backward-shift deletion (no tombstones)

use std::mem::MaybeUninit;

const MIN_CAPACITY: usize = 8;
const LOAD_FACTOR_NUM: usize = 3;
const LOAD_FACTOR_DEN: usize = 4;

/// Shrink threshold: shrink when len < capacity / SHRINK_DIVISOR
/// Only shrink if capacity > MIN_SHRINK_CAPACITY to avoid thrashing
const SHRINK_DIVISOR: usize = 4;
const MIN_SHRINK_CAPACITY: usize = 64;

// Empty marker for slots. An entry whose key equals this marker is held
// in the map's side slot instead of the array, so callers may use the
// whole i64 range.
const EMPTY: i64 = i64::MIN;

/// Slot with key and value. key == EMPTY means slot is empty.
#[repr(C)]
struct Slot<V> {
    key: i64,
    value: MaybeUninit<V>,
}

/// High-performance HashMap for i64 keys, covering the full i64 range.
pub struct I64Map<V> {
    slots: Box<[Slot<V>]>,
    len: usize,
    mask: usize,
    /// i64::MIN is the slot array's empty marker, so its entry lives
    /// beside the table. Callers hand this map keys derived from user
    /// data, so the full i64 range must be storable.
    min_slot: Option<V>,
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
    // The side slot is touched by one key out of 2^64, so every access
    // to it is a cold, never-inlined call: the probe loops stay exactly
    // as tight as they were when this branch was a panic.
    #[cold]
    #[inline(never)]
    fn sentinel_get(&self) -> Option<&V> {
        self.min_slot.as_ref()
    }

    #[cold]
    #[inline(never)]
    fn sentinel_get_mut(&mut self) -> Option<&mut V> {
        self.min_slot.as_mut()
    }

    #[cold]
    #[inline(never)]
    fn sentinel_insert(&mut self, value: V) -> Option<V> {
        self.min_slot.replace(value)
    }

    #[cold]
    #[inline(never)]
    fn sentinel_remove(&mut self) -> Option<V> {
        self.min_slot.take()
    }

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
            min_slot: None,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len + usize::from(self.min_slot.is_some())
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0 && self.min_slot.is_none()
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the map. The collection may reserve more space to avoid frequent reallocations.
    pub fn reserve(&mut self, additional: usize) {
        let target_len = self.len + additional;
        let target_cap = if target_len == 0 {
            MIN_CAPACITY
        } else {
            target_len
                .saturating_mul(LOAD_FACTOR_DEN)
                .saturating_div(LOAD_FACTOR_NUM)
                .next_power_of_two()
                .max(MIN_CAPACITY)
        };

        if target_cap <= self.slots.len() {
            return;
        }

        let new_cap = target_cap;
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

        for slot in old_slots.iter() {
            if slot.key != EMPTY {
                // SAFETY: We are moving valid initialized values
                let value = unsafe { slot.value.as_ptr().read() };
                self.insert(slot.key, value);
            }
        }

        debug_assert_eq!(self.len, old_len);
    }

    /// FxHash with pre-mixing - XOR the key with its shifted self before
    /// multiplication to break stride patterns while preserving bijectivity.
    /// This maintains 0 collisions for sequential keys while reducing strided
    /// key collisions by ~65% (e.g., stride=1024 goes from 99,872 to 34,464).
    ///
    /// The bucket is the low bits of the returned value, and multiplication
    /// only carries entropy upwards, so only the low bits of the key and the
    /// 16 above them can reach it. That is what keeps dense keys collision
    /// free, and it is deliberate: row ids, transaction ids and every other
    /// counter key their maps this way. The price is that a key carrying its
    /// entropy higher up has to be mixed before it arrives here, which is
    /// what [`key_from_f64`] is for.
    #[inline(always)]
    fn hash(key: i64) -> usize {
        let k = key as u64;
        let k = k ^ (k >> 16); // Pre-mix to break stride patterns (bijective)
        k.wrapping_mul(0x517cc1b727220a95) as usize
    }

    #[inline(always)]
    pub fn insert(&mut self, key: i64, value: V) -> Option<V> {
        if key == EMPTY {
            return self.sentinel_insert(value);
        }

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
        if key == EMPTY {
            return self.sentinel_get();
        }

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
        if key == EMPTY {
            return self.sentinel_get_mut();
        }

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
    /// Backward-shift deletion: closes the probe chain through `idx` after
    /// its value has been moved out, so no tombstone is needed.
    fn shift_back(&mut self, idx: usize) {
        let mask = self.mask;
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
                // Derive both pointers from a single as_mut_ptr() call to avoid
                // Stacked Borrows invalidation (as_ptr then as_mut_ptr conflicts).
                unsafe {
                    let base = self.slots.as_mut_ptr();
                    let src = base.add(next_idx);
                    let dst = base.add(empty_idx);
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
    }

    pub fn remove(&mut self, key: i64) -> Option<V> {
        if key == EMPTY {
            return self.sentinel_remove();
        }

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

        self.shift_back(idx);

        // Check if we should shrink after removal
        if self.should_shrink() {
            self.shrink();
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

    /// Check if we should shrink: len < capacity / SHRINK_DIVISOR
    /// Only shrink if capacity > MIN_SHRINK_CAPACITY to avoid thrashing
    #[inline]
    fn should_shrink(&self) -> bool {
        let cap = self.slots.len();
        cap > MIN_SHRINK_CAPACITY && self.len < cap / SHRINK_DIVISOR
    }

    /// Shrink the table to fit current entries
    fn shrink(&mut self) {
        // Calculate new capacity needed for current entries
        let new_cap = if self.len == 0 {
            MIN_CAPACITY
        } else {
            self.len
                .saturating_mul(LOAD_FACTOR_DEN)
                .saturating_div(LOAD_FACTOR_NUM)
                .next_power_of_two()
                .max(MIN_CAPACITY)
        };

        if new_cap >= self.slots.len() {
            return; // No need to shrink
        }

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

    /// Shrink the map to fit its current contents, releasing excess memory.
    ///
    /// Call this after removing many entries to reclaim memory.
    pub fn shrink_to_fit(&mut self) {
        self.shrink();
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
        self.min_slot = None;
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (i64, &V)> {
        self.slots
            .iter()
            .filter_map(|slot| {
                if slot.key != EMPTY {
                    // SAFETY: slot.key != EMPTY means the value is initialized.
                    Some((slot.key, unsafe { slot.value.assume_init_ref() }))
                } else {
                    None
                }
            })
            .chain(self.min_slot.as_ref().map(|v| (EMPTY, v)))
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = i64> + '_ {
        self.slots
            .iter()
            .filter_map(|s| if s.key != EMPTY { Some(s.key) } else { None })
            .chain(self.min_slot.as_ref().map(|_| EMPTY))
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.slots
            .iter()
            .filter_map(|slot| {
                if slot.key != EMPTY {
                    // SAFETY: slot.key != EMPTY means the value is initialized.
                    Some(unsafe { slot.value.assume_init_ref() })
                } else {
                    None
                }
            })
            .chain(self.min_slot.as_ref())
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (i64, &mut V)> {
        self.slots
            .iter_mut()
            .filter_map(|slot| {
                if slot.key != EMPTY {
                    // SAFETY: slot.key != EMPTY means the value is initialized.
                    Some((slot.key, unsafe { slot.value.assume_init_mut() }))
                } else {
                    None
                }
            })
            .chain(self.min_slot.as_mut().map(|v| (EMPTY, v)))
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all entries `(k, v)` where `f(k, &mut v)` returns `false`.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(i64, &mut V) -> bool,
    {
        // Collect keys to remove (can't remove while iterating due to backward-shift)
        let keys_to_remove: Vec<i64> = self
            .slots
            .iter_mut()
            .filter_map(|slot| {
                if slot.key != EMPTY {
                    // SAFETY: slot.key != EMPTY means the value is initialized.
                    let value = unsafe { slot.value.assume_init_mut() };
                    if f(slot.key, value) {
                        None // Keep this entry
                    } else {
                        Some(slot.key) // Mark for removal
                    }
                } else {
                    None
                }
            })
            .collect();

        for key in keys_to_remove {
            self.remove(key);
        }

        if let Some(value) = self.min_slot.as_mut() {
            if !f(EMPTY, value) {
                self.min_slot = None;
            }
        }
    }

    /// Drains all entries from the map, returning an iterator over them
    #[inline]
    /// Move every entry out, leaving the map empty with its table intact.
    ///
    /// The iterator borrows the map and empties slots as it yields them, so
    /// the map's capacity survives the drain. Swapping in a fresh 8-slot
    /// table, as this used to, meant every pooled transaction map came back
    /// at minimum size and regrew from scratch on the next transaction. An
    /// entry the iterator has not reached yet stays in the map, so a leaked
    /// drain leaves a consistent, merely undrained map behind.
    pub fn drain(&mut self) -> Drain<'_, V> {
        let start_len = self.len;
        Drain {
            map: self,
            pos: 0,
            start_len,
        }
    }

    #[inline(always)]
    pub fn entry(&mut self, key: i64) -> Entry<'_, V> {
        if key == EMPTY {
            return if self.min_slot.is_some() {
                Entry::Occupied(OccupiedEntry {
                    target: EntryTarget::Sentinel(&mut self.min_slot),
                })
            } else {
                Entry::Vacant(VacantEntry {
                    target: VacantTarget::Sentinel(&mut self.min_slot),
                    key,
                })
            };
        }

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
                                target: VacantTarget::Slot {
                                    map: self,
                                    idx: new_idx,
                                },
                                key,
                            });
                        }
                        new_idx = (new_idx + 1) & new_mask;
                    }
                }
                return Entry::Vacant(VacantEntry {
                    target: VacantTarget::Slot { map: self, idx },
                    key,
                });
            }

            if slot.key == key {
                return Entry::Occupied(OccupiedEntry {
                    target: EntryTarget::Slot { map: self, idx },
                });
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

/// Where an entry points: a table slot, or the out-of-band slot that
/// holds the i64::MIN key
enum EntryTarget<'a, V> {
    Slot { map: &'a mut I64Map<V>, idx: usize },
    Sentinel(&'a mut Option<V>),
}

enum VacantTarget<'a, V> {
    Slot { map: &'a mut I64Map<V>, idx: usize },
    Sentinel(&'a mut Option<V>),
}

/// Encode an `f64` as an [`I64Map`] or [`I64Set`] key.
///
/// The map takes its bucket from the low bits of the key, so a key whose
/// entropy sits above them lands in a handful of buckets. Raw `f64::to_bits()`
/// is exactly that shape: an IEEE-754 double holding a round value carries its
/// significant bits at the top of the mantissa and leaves the low mantissa
/// bits zero. Keying a map directly with the bit pattern puts every group of a
/// `GROUP BY <double column>` into one bucket and turns the scan quadratic, on
/// data as ordinary as whole numbers or money amounts.
///
/// The bit pattern is passed through the SplitMix64 finalizer, whose steps are
/// each invertible, so the encoding is a bijection: distinct doubles keep
/// distinct keys and grouping stays exact. [`f64_from_key`] recovers the
/// original value.
///
/// The encoding preserves the bit-level identity of the value, so `0.0` and
/// `-0.0` remain distinct keys and NaN payloads stay distinguished, exactly as
/// the raw bit pattern did.
#[inline]
pub fn key_from_f64(value: f64) -> i64 {
    let mut x = value.to_bits();
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x as i64
}

/// Recover the `f64` that [`key_from_f64`] encoded.
///
/// Each step undoes one step of the finalizer in reverse order. Undoing
/// `x ^= x >> s` takes the shift applied at `s`, then `2s`, until it reaches
/// past the width, and the multiplications are undone with the multiplicative
/// inverses of their constants modulo 2^64.
#[inline]
pub fn f64_from_key(key: i64) -> f64 {
    let mut x = key as u64;
    x ^= x >> 31;
    x ^= x >> 62;
    x = x.wrapping_mul(0x3196_42b2_d24d_8ec3);
    x ^= x >> 27;
    x ^= x >> 54;
    x = x.wrapping_mul(0x96de_1b17_3f11_9089);
    x ^= x >> 30;
    x ^= x >> 60;
    f64::from_bits(x)
}

pub struct OccupiedEntry<'a, V> {
    target: EntryTarget<'a, V>,
}

impl<'a, V> OccupiedEntry<'a, V> {
    #[inline(always)]
    pub fn get(&self) -> &V {
        match &self.target {
            // SAFETY: an OccupiedEntry is only built for an occupied slot,
            // so idx is valid and its value is initialized.
            EntryTarget::Slot { map, idx } => unsafe {
                map.slots.get_unchecked(*idx).value.assume_init_ref()
            },
            EntryTarget::Sentinel(slot) => slot.as_ref().expect("occupied sentinel"),
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut V {
        match &mut self.target {
            // SAFETY: see get().
            EntryTarget::Slot { map, idx } => unsafe {
                map.slots.get_unchecked_mut(*idx).value.assume_init_mut()
            },
            EntryTarget::Sentinel(slot) => slot.as_mut().expect("occupied sentinel"),
        }
    }

    #[inline(always)]
    pub fn into_mut(self) -> &'a mut V {
        match self.target {
            // SAFETY: see get().
            EntryTarget::Slot { map, idx } => unsafe {
                map.slots.get_unchecked_mut(idx).value.assume_init_mut()
            },
            EntryTarget::Sentinel(slot) => slot.as_mut().expect("occupied sentinel"),
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, value: V) -> V {
        match &mut self.target {
            EntryTarget::Slot { map, idx } => {
                // SAFETY: OccupiedEntry is only created for occupied slots.
                let slot = unsafe { map.slots.get_unchecked_mut(*idx) };
                // SAFETY: The slot is occupied, so value is initialized.
                let old = unsafe { slot.value.as_ptr().read() };
                slot.value.write(value);
                old
            }
            EntryTarget::Sentinel(slot) => slot.replace(value).expect("occupied sentinel"),
        }
    }

    pub fn remove(self) -> V {
        match self.target {
            EntryTarget::Slot { map, idx } => {
                // Extract value directly - we already have the index
                // SAFETY: OccupiedEntry is only created for occupied slots.
                let key = unsafe { map.slots.get_unchecked(idx).key };
                // SAFETY: The slot is occupied, so value is initialized.
                let value = unsafe { map.slots.get_unchecked(idx).value.as_ptr().read() };
                map.len -= 1;

                // Backward shift deletion at known index
                let mask = map.mask;
                let mut empty_idx = idx;
                let mut next_idx = (idx + 1) & mask;

                loop {
                    // SAFETY: next_idx is always in bounds due to masking.
                    let next_slot = unsafe { map.slots.get_unchecked(next_idx) };

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
                        // Derive both pointers from a single as_mut_ptr() call to avoid
                        // Stacked Borrows invalidation (as_ptr then as_mut_ptr conflicts).
                        unsafe {
                            let base = map.slots.as_mut_ptr();
                            let src = base.add(next_idx);
                            let dst = base.add(empty_idx);
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
                    map.slots.get_unchecked_mut(empty_idx).key = EMPTY;
                }

                // Suppress unused variable warning
                let _ = key;

                value
            }
            EntryTarget::Sentinel(slot) => slot.take().expect("occupied sentinel"),
        }
    }
}

pub struct VacantEntry<'a, V> {
    target: VacantTarget<'a, V>,
    key: i64,
}

impl<'a, V> VacantEntry<'a, V> {
    #[inline(always)]
    pub fn key(&self) -> i64 {
        self.key
    }

    #[inline(always)]
    pub fn insert(self, value: V) -> &'a mut V {
        match self.target {
            VacantTarget::Slot { map, idx } => {
                // Direct insert at pre-computed index - NO re-lookup needed
                // SAFETY: VacantEntry stores a valid idx that was found during entry() lookup.
                let slot = unsafe { map.slots.get_unchecked_mut(idx) };
                slot.key = self.key;
                slot.value.write(value);
                map.len += 1;
                // SAFETY: We just wrote the value, so it's initialized.
                unsafe { slot.value.assume_init_mut() }
            }
            VacantTarget::Sentinel(slot) => slot.insert(value),
        }
    }
}

/// Owning iterator over the entries of an I64Map
pub struct IntoIter<V> {
    slots: Box<[Slot<V>]>,
    pos: usize,
    min_slot: Option<V>,
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
        // The sentinel entry lives beside the table and is yielded last
        self.min_slot.take().map(|v| (EMPTY, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let pending = usize::from(self.min_slot.is_some());
        (pending, Some(self.slots.len() - self.pos + pending))
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
        let min_slot = self.min_slot.take();
        self.len = 0; // Prevent drop from cleaning up values we're moving out
        IntoIter {
            slots,
            pos: 0,
            min_slot,
        }
    }
}

/// Draining iterator over the entries of an I64Map
pub struct Drain<'a, V> {
    map: &'a mut I64Map<V>,
    pos: usize,
    start_len: usize,
}

// =============================================================================
// I64Set - High-performance HashSet for i64 keys
// =============================================================================

/// High-performance HashSet for i64 keys.
///
/// Uses the same optimizations as I64Map:
/// - i64::MIN marks empty slots; that member is held beside the array
/// - FxHash with pre-mixing (XOR>>16 before multiply) - 0 sequential collisions
/// - Backward-shift deletion (no tombstones)
pub struct I64Set {
    slots: Box<[i64]>,
    len: usize,
    mask: usize,
    /// i64::MIN is the slot array's empty sentinel, so it is stored out
    /// of band. Callers pass user data (join keys, IN lists), so the set
    /// must accept the full i64 range instead of panicking.
    has_min: bool,
}

impl Clone for I64Set {
    fn clone(&self) -> Self {
        let mut new_set = Self::with_capacity(self.len);
        for key in self.iter() {
            new_set.insert(key);
        }
        new_set
    }
}

impl std::fmt::Debug for I64Set {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl Default for I64Set {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl I64Set {
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

        let slots: Vec<i64> = vec![EMPTY; cap];

        Self {
            slots: slots.into_boxed_slice(),
            len: 0,
            mask: cap - 1,
            has_min: false,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len + usize::from(self.has_min)
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0 && !self.has_min
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the set. The collection may reserve more space to avoid frequent reallocations.
    pub fn reserve(&mut self, additional: usize) {
        let target_len = self.len + additional;
        let target_cap = if target_len == 0 {
            MIN_CAPACITY
        } else {
            target_len
                .saturating_mul(LOAD_FACTOR_DEN)
                .saturating_div(LOAD_FACTOR_NUM)
                .next_power_of_two()
                .max(MIN_CAPACITY)
        };

        if target_cap <= self.slots.len() {
            return;
        }

        let new_cap = target_cap;
        let new_mask = new_cap - 1;

        let new_slots: Vec<i64> = vec![EMPTY; new_cap];
        let old_slots = std::mem::replace(&mut self.slots, new_slots.into_boxed_slice());
        let old_len = self.len;
        self.len = 0;
        self.mask = new_mask;

        for slot in old_slots.iter() {
            if *slot != EMPTY {
                self.insert(*slot);
            }
        }

        debug_assert_eq!(self.len, old_len);
    }

    /// FxHash with pre-mixing - same as I64Map
    #[inline(always)]
    fn hash(key: i64) -> usize {
        let k = key as u64;
        let k = k ^ (k >> 16);
        k.wrapping_mul(0x517cc1b727220a95) as usize
    }

    /// Insert a value into the set. Returns true if the value was newly inserted.
    #[inline(always)]
    pub fn insert(&mut self, key: i64) -> bool {
        if key == EMPTY {
            let inserted = !self.has_min;
            self.has_min = true;
            return inserted;
        }

        if self.len * LOAD_FACTOR_DEN >= self.slots.len() * LOAD_FACTOR_NUM {
            self.grow();
        }

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        loop {
            // SAFETY: idx is always (hash & mask), where mask = slots.len() - 1.
            // Since slots.len() is a power of 2, idx is always in bounds.
            let slot = unsafe { *self.slots.get_unchecked(idx) };

            if slot == EMPTY {
                // SAFETY: Same bounds reasoning as above - idx is always valid.
                unsafe { *self.slots.get_unchecked_mut(idx) = key };
                self.len += 1;
                return true;
            }

            if slot == key {
                return false; // Already exists
            }

            idx = (idx + 1) & mask;
        }
    }

    #[inline(always)]
    pub fn contains(&self, key: i64) -> bool {
        if key == EMPTY {
            return self.has_min;
        }

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        loop {
            // SAFETY: idx is always (hash & mask), where mask = slots.len() - 1.
            // Since slots.len() is a power of 2, idx is always in bounds.
            let slot = unsafe { *self.slots.get_unchecked(idx) };

            if slot == EMPTY {
                return false;
            }

            if slot == key {
                return true;
            }

            idx = (idx + 1) & mask;
        }
    }

    #[inline(always)]
    pub fn remove(&mut self, key: i64) -> bool {
        if key == EMPTY {
            let removed = self.has_min;
            self.has_min = false;
            return removed;
        }

        let mask = self.mask;
        let mut idx = Self::hash(key) & mask;

        // Find the key
        loop {
            // SAFETY: idx is always (hash & mask), where mask = slots.len() - 1.
            // Since slots.len() is a power of 2, idx is always in bounds.
            let slot = unsafe { *self.slots.get_unchecked(idx) };

            if slot == EMPTY {
                return false;
            }

            if slot == key {
                break;
            }

            idx = (idx + 1) & mask;
        }

        self.len -= 1;

        // Backward shift deletion
        let mut empty_idx = idx;
        let mut next_idx = (idx + 1) & mask;

        loop {
            // SAFETY: next_idx is always (some_value & mask), where mask = slots.len() - 1.
            // Since slots.len() is a power of 2, next_idx is always in bounds.
            let next_slot = unsafe { *self.slots.get_unchecked(next_idx) };

            if next_slot == EMPTY {
                break;
            }

            let next_home = Self::hash(next_slot) & mask;

            let can_move = if next_home <= next_idx {
                empty_idx >= next_home && empty_idx < next_idx
            } else {
                empty_idx >= next_home || empty_idx < next_idx
            };

            if can_move {
                // SAFETY: empty_idx was either the original idx (valid) or a previous
                // next_idx (also valid by the same mask reasoning).
                unsafe {
                    *self.slots.get_unchecked_mut(empty_idx) = next_slot;
                }
                empty_idx = next_idx;
            }

            next_idx = (next_idx + 1) & mask;
        }

        // SAFETY: empty_idx is always a valid index (same mask reasoning as above).
        unsafe {
            *self.slots.get_unchecked_mut(empty_idx) = EMPTY;
        }

        // Check if we should shrink after removal
        if self.should_shrink() {
            self.shrink();
        }

        true
    }

    fn grow(&mut self) {
        let new_cap = (self.slots.len() * 2).max(MIN_CAPACITY);
        let new_mask = new_cap - 1;

        let new_slots: Vec<i64> = vec![EMPTY; new_cap];
        let old_slots = std::mem::replace(&mut self.slots, new_slots.into_boxed_slice());
        let old_len = self.len;
        self.len = 0;
        self.mask = new_mask;

        for slot in old_slots.iter() {
            if *slot != EMPTY {
                self.insert(*slot);
            }
        }

        debug_assert_eq!(self.len, old_len);
    }

    /// Check if we should shrink: len < capacity / SHRINK_DIVISOR
    /// Only shrink if capacity > MIN_SHRINK_CAPACITY to avoid thrashing
    #[inline]
    fn should_shrink(&self) -> bool {
        let cap = self.slots.len();
        cap > MIN_SHRINK_CAPACITY && self.len < cap / SHRINK_DIVISOR
    }

    /// Shrink the set to fit current entries
    fn shrink(&mut self) {
        // Calculate new capacity needed for current entries
        let new_cap = if self.len == 0 {
            MIN_CAPACITY
        } else {
            self.len
                .saturating_mul(LOAD_FACTOR_DEN)
                .saturating_div(LOAD_FACTOR_NUM)
                .next_power_of_two()
                .max(MIN_CAPACITY)
        };

        if new_cap >= self.slots.len() {
            return; // No need to shrink
        }

        let new_mask = new_cap - 1;

        let new_slots: Vec<i64> = vec![EMPTY; new_cap];
        let old_slots = std::mem::replace(&mut self.slots, new_slots.into_boxed_slice());
        let old_len = self.len;
        self.len = 0;
        self.mask = new_mask;

        for slot in old_slots.iter() {
            if *slot != EMPTY {
                self.insert(*slot);
            }
        }

        debug_assert_eq!(self.len, old_len);
    }

    /// Shrink the set to fit its current contents, releasing excess memory.
    ///
    /// Call this after removing many entries to reclaim memory.
    pub fn shrink_to_fit(&mut self) {
        self.shrink();
    }

    pub fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            *slot = EMPTY;
        }
        self.len = 0;
        self.has_min = false;
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.slots
            .iter()
            .filter_map(|&slot| if slot != EMPTY { Some(slot) } else { None })
            .chain(if self.has_min { Some(EMPTY) } else { None })
    }

    /// Drains all values from the set, returning an iterator over them
    #[inline]
    pub fn drain(&mut self) -> impl Iterator<Item = i64> + '_ {
        // Hand the storage to the iterator instead of clearing lazily:
        // a drain dropped half-way used to leave the slots populated
        // while len said the set was empty, so contains() still matched
        let old_slots = std::mem::replace(
            &mut self.slots,
            vec![EMPTY; MIN_CAPACITY].into_boxed_slice(),
        );
        let had_min = std::mem::replace(&mut self.has_min, false);
        self.len = 0;
        self.mask = MIN_CAPACITY - 1;
        old_slots
            .into_vec()
            .into_iter()
            .filter(|&slot| slot != EMPTY)
            .chain(if had_min { Some(EMPTY) } else { None })
    }
}

impl IntoIterator for I64Set {
    type Item = i64;
    type IntoIter = I64SetIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        I64SetIntoIter {
            slots: self.slots,
            pos: 0,
            has_min: self.has_min,
        }
    }
}

/// Owning iterator over the values of an I64Set
pub struct I64SetIntoIter {
    slots: Box<[i64]>,
    pos: usize,
    has_min: bool,
}

impl Iterator for I64SetIntoIter {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.slots.len() {
            let slot = self.slots[self.pos];
            self.pos += 1;

            if slot != EMPTY {
                return Some(slot);
            }
        }
        if self.has_min {
            self.has_min = false;
            return Some(EMPTY);
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // The out-of-band sentinel is still pending after the slots are
        // exhausted, so it counts toward the upper bound
        // The out-of-band sentinel is still pending after the slots are
        // exhausted, so it counts toward both bounds
        (
            usize::from(self.has_min),
            Some(self.slots.len() - self.pos + usize::from(self.has_min)),
        )
    }
}

impl std::iter::FromIterator<i64> for I64Set {
    fn from_iter<T: IntoIterator<Item = i64>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut set = I64Set::with_capacity(lower);
        for key in iter {
            set.insert(key);
        }
        set
    }
}

impl Extend<i64> for I64Set {
    fn extend<T: IntoIterator<Item = i64>>(&mut self, iter: T) {
        for key in iter {
            self.insert(key);
        }
    }
}

impl<V> Iterator for Drain<'_, V> {
    type Item = (i64, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.map.slots.len() {
            let slot = &self.map.slots[self.pos];
            if slot.key != EMPTY {
                let key = slot.key;
                // SAFETY: slot.key != EMPTY means the value is initialized; the
                // shift below empties this slot or refills it from its chain,
                // so the moved-out value is neither read again nor dropped.
                let value = unsafe { slot.value.as_ptr().read() };
                self.map.len -= 1;
                // Closing the chain on every yield keeps the map valid if the
                // drain is leaked or its Drop unwinds. The cursor stays put:
                // a later entry may have shifted into this slot, and every
                // slot before it is already empty.
                self.map.shift_back(self.pos);
                return Some((key, value));
            }
            self.pos += 1;
        }
        // The side-slot entry drains last
        self.map.min_slot.take().map(|v| (EMPTY, v))
    }

    /// Exact: the map knows how many entries it still holds.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.map.len + usize::from(self.map.min_slot.is_some());
        (remaining, Some(remaining))
    }
}

impl<V> ExactSizeIterator for Drain<'_, V> {}

impl<V> Drop for Drain<'_, V> {
    fn drop(&mut self) {
        // Consume remaining elements so they are dropped and the map is empty
        for _ in self.by_ref() {}
        // A table far larger than the batch it just held would tax every
        // later drain with its empty slots, so it is released here, sized
        // for that batch.
        let cap = self.map.slots.len();
        if cap > MIN_SHRINK_CAPACITY && self.start_len < cap / SHRINK_DIVISOR {
            *self.map = I64Map::with_capacity(self.start_len);
        }
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

    /// Count how many distinct buckets a set of keys lands in.
    fn bucket_occupancy(keys: &[i64], cap: usize) -> usize {
        let mask = cap - 1;
        let mut seen = vec![false; cap];
        for &key in keys {
            seen[I64Map::<u64>::hash(key) & mask] = true;
        }
        seen.iter().filter(|&&s| s).count()
    }

    /// Capacities filled to 3/4, which is the load factor the map grows at.
    /// One capacity proves nothing on its own: the bucket index is masked, so
    /// how a key shape behaves depends on the mask width and on how full the
    /// table is.
    fn sweep() -> impl Iterator<Item = (usize, usize)> {
        [1024usize, 4096, 65536]
            .into_iter()
            .map(|cap| (cap, cap * 3 / 4))
    }

    /// The encoding must be exactly reversible, or grouping would report the
    /// wrong value for every float group.
    #[test]
    fn test_float_key_round_trips() {
        let mut values = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            -0.1,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(1),                  // smallest subnormal
            f64::from_bits(0x7ff8000000000001), // a NaN payload
        ];
        // A spread of ordinary values plus a deterministic pseudo-random walk
        // over raw bit patterns, so exponents and mantissas of every shape are
        // covered rather than just the friendly ones.
        for i in 1..2000i64 {
            values.push(i as f64);
            values.push(i as f64 * 0.01);
            values.push(1.0 / i as f64);
        }
        let mut x = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..20_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            values.push(f64::from_bits(x));
        }

        for value in values {
            let key = key_from_f64(value);
            let back = f64_from_key(key);
            assert_eq!(
                back.to_bits(),
                value.to_bits(),
                "round trip changed the bit pattern of {value:?}"
            );
        }
    }

    /// Distinct doubles must keep distinct keys, or two groups would merge.
    #[test]
    fn test_float_key_is_injective() {
        let mut keys = std::collections::HashSet::new();
        for i in 0..50_000i64 {
            assert!(
                keys.insert(key_from_f64(i as f64 * 0.5)),
                "two doubles collapsed onto one key"
            );
        }
    }

    /// Float keys must spread across buckets.
    ///
    /// A double holding a round value keeps its significant bits at the top of
    /// the mantissa and its low mantissa bits at zero, and the map's bucket
    /// comes from the low bits of the key, so feeding the raw bit pattern in
    /// puts every group into one bucket. The bar here is half the keys landing
    /// in distinct buckets; an ideal hash reaches about 0.7 * n at this load.
    #[test]
    fn test_float_keys_spread_across_buckets() {
        for (cap, n) in sweep() {
            type FloatFamily = (&'static str, fn(usize) -> f64);
            let families: [FloatFamily; 5] = [
                ("whole", |i| i as f64),
                ("halves", |i| i as f64 * 0.5),
                ("money", |i| i as f64 * 0.01),
                ("irregular", |i| i as f64 * 1.000_000_123_4),
                ("tiny", |i| i as f64 * 1e-9),
            ];
            for (name, f) in families {
                let keys: Vec<i64> = (1..=n).map(|i| key_from_f64(f(i))).collect();
                let occupancy = bucket_occupancy(&keys, cap);
                assert!(
                    occupancy > n / 2,
                    "{n} {name} float keys landed in {occupancy} of {cap} buckets"
                );
            }
        }
    }

    /// A dense key range inside one 2^16 block must never collide.
    ///
    /// Row ids, transaction ids and every other counter key their maps this
    /// way, so this is the property the hot path is built on, and it is
    /// provable rather than measured: inside such a block `k >> 16` is
    /// constant, so the pre-mix is `k ^ const`, and both that and the odd
    /// multiply are permutations of the low bits.
    ///
    /// It stops being provable once a range straddles a block boundary, since
    /// the two halves get different constants. See the test below for what
    /// actually happens there.
    #[test]
    fn test_hash_keeps_dense_keys_collision_free() {
        // 65536-aligned, so every window below stays inside one block.
        const HIGH_BLOCK: i64 = 65536 * 15258;

        for (cap, n) in sweep() {
            let shapes: [(&str, Vec<i64>); 3] = [
                ("from zero", (0..n as i64).collect()),
                ("negative", (-(n as i64)..0).collect()),
                ("high block", (HIGH_BLOCK..HIGH_BLOCK + n as i64).collect()),
            ];
            for (name, keys) in &shapes {
                let occupancy = bucket_occupancy(keys, cap);
                assert_eq!(
                    occupancy, n,
                    "hash collided on {n} dense keys ({name}) in {cap} slots"
                );
            }
        }
    }

    /// A dense range that straddles a 2^16 block boundary is not collision
    /// free, and this pins down how far it degrades.
    ///
    /// The two halves of such a range are pre-mixed with different constants,
    /// so their images can overlap. Sliding a 3/4-full window across several
    /// boundaries, the worst case found is this one: half the keys share a
    /// bucket with another key, and no key needs more than 6 probes. This is a
    /// baseline guard rather than an invariant, so that a change to the hash
    /// cannot quietly make the boundary worse.
    #[test]
    fn test_hash_bounds_dense_keys_across_a_block_boundary() {
        let cap = 1024usize;
        let n = 768usize;
        let start = 33_554_048i64; // ends 384 keys past a 2^16 boundary
        let keys: Vec<i64> = (start..start + n as i64).collect();

        let occupancy = bucket_occupancy(&keys, cap);
        assert!(
            occupancy * 2 >= n,
            "{n} boundary-straddling dense keys landed in {occupancy} of {cap} buckets"
        );
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

    /// Draining must not throw the table away: the transaction map pools
    /// hand a drained map back to the next transaction, and a map that came
    /// back at 8 slots regrew from scratch every time.
    /// A drain leaked part-way, or cut short by a panicking destructor,
    /// must leave a valid table: an entry behind the yielded one on the
    /// same probe chain stays reachable.
    #[test]
    fn test_partial_drain_leak_keeps_probe_chains() {
        let mut map: I64Map<&str> = I64Map::with_capacity(0);
        assert_eq!(map.capacity(), 8);
        assert_eq!(I64Map::<&str>::hash(0) & 7, I64Map::<&str>::hash(8) & 7);
        map.insert(0, "a");
        map.insert(8, "b");

        let mut drain = map.drain();
        let (first, _) = drain.next().unwrap();
        std::mem::forget(drain);

        let other = if first == 0 { 8 } else { 0 };
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(first), None);
        assert_eq!(
            map.get(other).copied(),
            Some(if other == 0 { "a" } else { "b" })
        );
        map.insert(first, "c");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(first).copied(), Some("c"));
    }

    /// One large batch must not leave every later small drain scanning a
    /// table sized for it: the table is released once the batch it held
    /// is far below its capacity.
    #[test]
    fn test_drain_releases_an_oversized_table() {
        let mut map: I64Map<u64> = I64Map::new();
        for i in 0..2000 {
            map.insert(i, i as u64);
        }
        assert_eq!(map.drain().count(), 2000);
        let grown = map.capacity();
        assert!(grown > MIN_SHRINK_CAPACITY);

        map.insert(1, 1);
        assert_eq!(map.drain().count(), 1);
        assert_eq!(map.capacity(), MIN_CAPACITY);
        assert!(map.is_empty());
        map.insert(2, 2);
        assert_eq!(map.get(2), Some(&2));
    }

    #[test]
    fn test_drain_keeps_capacity() {
        let mut map: I64Map<u64> = I64Map::new();
        for i in 0..1000 {
            map.insert(i, i as u64);
        }
        let before = map.capacity();
        assert!(before >= 1000);
        let drained = map.drain().count();
        assert_eq!(drained, 1000);
        assert!(map.is_empty());
        assert_eq!(map.capacity(), before, "drain must keep the table");

        // and the map is fully usable afterwards, without growing
        for i in 0..1000 {
            map.insert(i, i as u64);
        }
        assert_eq!(map.capacity(), before);
        assert_eq!(map.len(), 1000);
    }

    /// `collect()` sizes its allocation from the lower bound, so a drain that
    /// reported 0 forced a Vec to grow through every doubling.
    #[test]
    fn test_drain_size_hint_is_exact() {
        let mut map: I64Map<u64> = I64Map::new();
        for i in 0..100 {
            map.insert(i, i as u64);
        }
        map.insert(i64::MIN, 7);
        let mut drain = map.drain();
        assert_eq!(drain.size_hint(), (101, Some(101)));
        assert_eq!(drain.len(), 101);
        drain.next();
        assert_eq!(drain.size_hint(), (100, Some(100)));
        let rest: Vec<(i64, u64)> = drain.collect();
        assert_eq!(rest.len(), 100);
        assert!(map.is_empty());
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
    fn test_shrink_after_delete() {
        let mut map = I64Map::new();

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
        let mut map: I64Map<i64> = I64Map::with_capacity(1000);

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
    fn test_i64_min_is_a_regular_key() {
        // The empty marker lives in the slot array only; the entry for
        // that key is held beside it, so the whole i64 range works
        let mut map = I64Map::<i64>::new();
        assert!(map.get(i64::MIN).is_none());
        assert!(map.is_empty());

        assert_eq!(map.insert(i64::MIN, 42), None);
        assert_eq!(map.get(i64::MIN), Some(&42));
        assert_eq!(map.insert(i64::MIN, 7), Some(42));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        assert!(map.contains_key(i64::MIN));

        map.insert(3, 30);
        map.insert(-1, -10);
        assert_eq!(map.len(), 3);
        let mut pairs: Vec<(i64, i64)> = map.iter().map(|(k, v)| (k, *v)).collect();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![(i64::MIN, 7), (-1, -10), (3, 30)]);

        *map.get_mut(i64::MIN).unwrap() += 1;
        assert_eq!(map.get(i64::MIN), Some(&8));

        let mut owned: Vec<(i64, i64)> = map.clone().into_iter().collect();
        owned.sort_unstable();
        assert_eq!(owned, vec![(i64::MIN, 8), (-1, -10), (3, 30)]);

        assert_eq!(map.remove(i64::MIN), Some(8));
        assert_eq!(map.remove(i64::MIN), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_i64set_partial_drain_leaves_no_stale_keys() {
        let mut set = I64Set::new();
        set.insert(5);
        set.insert(6);
        set.insert(7);
        {
            let mut d = set.drain();
            let _ = d.next();
        }
        assert!(set.is_empty(), "drain must empty the set");
        for k in [5, 6, 7] {
            assert!(!set.contains(k), "stale key {k} still visible after drain");
        }
    }

    #[test]
    fn test_i64_min_drain() {
        let mut map = I64Map::<i64>::new();
        map.insert(i64::MIN, 42);
        map.insert(9, 90);

        let mut drained: Vec<(i64, i64)> = map.drain().collect();
        drained.sort_unstable();
        assert_eq!(drained, vec![(i64::MIN, 42), (9, 90)]);
        assert!(map.is_empty(), "drain must leave the map empty");
        assert!(map.get(i64::MIN).is_none());

        // A dropped, partially consumed drain must not leave the entry behind
        map.insert(i64::MIN, 7);
        map.insert(1, 10);
        {
            let mut d = map.drain();
            let _ = d.next();
        }
        assert!(map.is_empty());
        assert!(map.get(i64::MIN).is_none());
    }

    #[test]
    fn test_i64_min_entry_api() {
        let mut map = I64Map::<i64>::new();
        *map.entry(i64::MIN).or_insert(1) += 5;
        assert_eq!(map.get(i64::MIN), Some(&6));

        match map.entry(i64::MIN) {
            Entry::Occupied(mut e) => {
                assert_eq!(*e.get(), 6);
                assert_eq!(e.insert(9), 6);
                assert_eq!(e.remove(), 9);
            }
            Entry::Vacant(_) => panic!("sentinel entry should be occupied"),
        }
        assert!(map.get(i64::MIN).is_none());

        match map.entry(i64::MIN) {
            Entry::Vacant(e) => {
                assert_eq!(e.key(), i64::MIN);
                *e.insert(4) += 1;
            }
            Entry::Occupied(_) => panic!("sentinel entry should be vacant"),
        }
        assert_eq!(map.get(i64::MIN), Some(&5));

        map.retain(|k, _| k != i64::MIN);
        assert!(map.get(i64::MIN).is_none());

        map.insert(i64::MIN, 1);
        map.clear();
        assert!(map.is_empty());
        assert!(map.get(i64::MIN).is_none());
    }

    // =========================================================================
    // I64Set Tests
    // =========================================================================

    #[test]
    fn test_i64set_basic_operations() {
        let mut set = I64Set::new();

        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(set.insert(3));
        assert_eq!(set.len(), 3);

        assert!(set.contains(1));
        assert!(set.contains(2));
        assert!(set.contains(3));
        assert!(!set.contains(4));

        // Duplicate insert returns false
        assert!(!set.insert(2));
        assert_eq!(set.len(), 3);

        // Remove
        assert!(set.remove(2));
        assert!(!set.contains(2));
        assert_eq!(set.len(), 2);

        // Remove non-existent returns false
        assert!(!set.remove(2));
    }

    #[test]
    fn test_i64set_grow() {
        let mut set = I64Set::new();

        for i in 0..1000 {
            set.insert(i);
        }

        assert_eq!(set.len(), 1000);

        for i in 0..1000 {
            assert!(set.contains(i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_i64set_edge_values() {
        let mut set = I64Set::new();

        set.insert(i64::MIN + 1);
        set.insert(i64::MAX);
        set.insert(0);
        set.insert(-1);
        set.insert(1);

        assert!(set.contains(i64::MIN + 1));
        assert!(set.contains(i64::MAX));
        assert!(set.contains(0));
        assert!(set.contains(-1));
        assert!(set.contains(1));
    }

    #[test]
    fn test_i64set_into_iter() {
        let mut set = I64Set::new();
        set.insert(1);
        set.insert(2);
        set.insert(3);

        let mut values: Vec<i64> = set.into_iter().collect();
        values.sort();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_i64set_from_iter() {
        let set: I64Set = vec![1, 2, 3, 2, 1].into_iter().collect();
        assert_eq!(set.len(), 3);
        assert!(set.contains(1));
        assert!(set.contains(2));
        assert!(set.contains(3));
    }

    #[test]
    fn test_i64set_shrink_after_delete() {
        let mut set = I64Set::new();

        // Insert many entries to grow the set
        for i in 0..1000 {
            set.insert(i);
        }

        let capacity_after_insert = set.capacity();
        assert!(capacity_after_insert >= 1000);

        // Remove most entries (keep only 10)
        for i in 10..1000 {
            set.remove(i);
        }

        assert_eq!(set.len(), 10);

        // Capacity should have shrunk (automatic shrink after remove)
        let capacity_after_remove = set.capacity();
        assert!(
            capacity_after_remove < capacity_after_insert,
            "capacity should shrink: {} < {}",
            capacity_after_remove,
            capacity_after_insert
        );

        // Verify remaining entries still work
        for i in 0..10 {
            assert!(set.contains(i));
        }
    }

    #[test]
    fn test_i64set_shrink_to_fit() {
        let mut set = I64Set::with_capacity(1000);

        // Insert only a few entries
        for i in 0..10 {
            set.insert(i);
        }

        let initial_capacity = set.capacity();
        assert!(initial_capacity >= 1000);

        // Shrink to fit
        set.shrink_to_fit();

        let after_shrink = set.capacity();
        assert!(
            after_shrink < initial_capacity,
            "capacity should shrink: {} < {}",
            after_shrink,
            initial_capacity
        );

        // Verify entries still work
        for i in 0..10 {
            assert!(set.contains(i));
        }
    }

    #[test]
    fn test_i64set_strided_keys() {
        let mut set = I64Set::with_capacity(10000);
        let stride = 1024;

        for i in 0..10000i64 {
            set.insert(i * stride);
        }

        assert_eq!(set.len(), 10000);
        for i in 0..10000i64 {
            assert!(set.contains(i * stride), "Missing key {}", i * stride);
        }
    }

    #[test]
    fn test_i64set_into_iter_size_hint_covers_sentinel() {
        // The buggy state is "slots exhausted, sentinel still pending",
        // which occurs whenever the last occupied slot is the final one.
        // Construct it directly so the assertion cannot depend on where
        // the hash happened to place a key.
        let mut set = I64Set::new();
        set.insert(i64::MIN);
        let mut iter = set.into_iter();
        iter.pos = iter.slots.len();

        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 1, "a pending sentinel is a guaranteed item");
        assert_eq!(
            upper,
            Some(1),
            "upper bound must still admit the pending sentinel"
        );
        assert_eq!(iter.next(), Some(i64::MIN));
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_i64set_handles_min_out_of_band() {
        // I64Set stores i64::MIN beside the slot array, so callers can
        // hand it user data without a sentinel dance
        let mut set = I64Set::new();
        assert!(!set.contains(i64::MIN));
        assert!(set.insert(i64::MIN));
        assert!(!set.insert(i64::MIN));
        assert!(set.contains(i64::MIN));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());

        set.insert(7);
        set.insert(-3);
        let mut collected: Vec<i64> = set.iter().collect();
        collected.sort_unstable();
        assert_eq!(collected, vec![i64::MIN, -3, 7]);
        assert_eq!(set.len(), 3);

        let mut drained: Vec<i64> = set.clone().into_iter().collect();
        drained.sort_unstable();
        assert_eq!(drained, vec![i64::MIN, -3, 7]);

        assert!(set.remove(i64::MIN));
        assert!(!set.remove(i64::MIN));
        assert!(!set.contains(i64::MIN));
        assert_eq!(set.len(), 2);

        set.insert(i64::MIN);
        set.clear();
        assert!(set.is_empty());
        assert!(!set.contains(i64::MIN));
    }
}
