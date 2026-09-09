use alloc::collections::btree_map::{BTreeMap, Entry};
use core::iter;
use core::ops::RangeBounds;

use crate::IntegerTooSmall;
use crate::RoaringBitmap;
use crate::RoaringTreemap;

use super::util;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

impl RoaringTreemap {
    /// Creates an empty `RoaringTreemap`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    /// let rb = RoaringTreemap::new();
    /// ```
    pub fn new() -> RoaringTreemap {
        RoaringTreemap { map: BTreeMap::new() }
    }

    /// Creates a full `RoaringTreemap`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use roaring::RoaringTreemap;
    /// let rb = RoaringTreemap::full();
    /// ```
    pub fn full() -> RoaringTreemap {
        RoaringTreemap { map: (0..=u32::MAX).zip(iter::repeat(RoaringBitmap::full())).collect() }
    }

    /// Adds a value to the set. Returns `true` if the value was not already present in the set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert_eq!(rb.insert(3), true);
    /// assert_eq!(rb.insert(3), false);
    /// assert_eq!(rb.contains(3), true);
    /// ```
    pub fn insert(&mut self, value: u64) -> bool {
        let (hi, lo) = util::split(value);
        self.map.entry(hi).or_default().insert(lo)
    }

    /// Inserts a range of values.
    ///
    /// Returns the number of inserted values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// rb.insert_range(2..4);
    /// assert!(rb.contains(2));
    /// assert!(rb.contains(3));
    /// assert!(!rb.contains(4));
    /// ```
    pub fn insert_range<R: RangeBounds<u64>>(&mut self, range: R) -> u64 {
        let (start, end) = match util::convert_range_to_inclusive(range) {
            Some(range) => (*range.start(), *range.end()),
            None => return 0,
        };

        let (start_hi, start_lo) = util::split(start);
        let (end_hi, end_lo) = util::split(end);

        let mut counter = 0u64;

        // Split the input range by the leading 32 bits
        for hi in start_hi..=end_hi {
            let entry = self.map.entry(hi);

            // Calculate the sub-range from the lower 32 bits
            counter += if hi == end_hi && hi == start_hi {
                entry.or_default().insert_range(start_lo..=end_lo)
            } else if hi == start_hi {
                entry.or_default().insert_range(start_lo..=u32::MAX)
            } else if hi == end_hi {
                entry.or_default().insert_range(0..=end_lo)
            } else {
                // We insert a full bitmap if it doesn't already exist and return the size of it.
                // But if the bitmap already exists at this spot we replace it with a full bitmap
                // and specify that we didn't inserted the integers from the previous bitmap.
                let full_bitmap = RoaringBitmap::full();
                match entry {
                    Entry::Vacant(entry) => entry.insert(full_bitmap).len(),
                    Entry::Occupied(mut entry) => {
                        full_bitmap.len() - entry.insert(full_bitmap).len()
                    }
                }
            };
        }

        counter
    }

    /// Pushes `value` in the treemap only if it is greater than the current maximum value.
    ///
    /// Returns whether the value was inserted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert!(rb.push(1));
    /// assert!(rb.push(3));
    /// assert_eq!(rb.push(3), false);
    /// assert!(rb.push(5));
    ///
    /// assert_eq!(rb.iter().collect::<Vec<u64>>(), vec![1, 3, 5]);
    /// ```
    #[deprecated(since = "0.11.0", note = "use `try_push` instead")]
    pub fn push(&mut self, value: u64) -> bool {
        let (hi, lo) = util::split(value);
        self.map.entry(hi).or_default().try_push(lo).is_ok()
    }

    /// Pushes `value` in the treemap only if it is greater than the current maximum value.
    ///
    /// Returns an error if the value is not greater than the current maximum value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::{RoaringTreemap, IntegerTooSmall};
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert!(rb.try_push(1).is_ok());
    /// assert!(rb.try_push(3).is_ok());
    /// assert_eq!(rb.try_push(3), Err(IntegerTooSmall));
    /// assert!(rb.try_push(5).is_ok());
    ///
    /// assert_eq!(rb.iter().collect::<Vec<u64>>(), vec![1, 3, 5]);
    /// ```
    pub fn try_push(&mut self, value: u64) -> Result<(), IntegerTooSmall> {
        let (hi, lo) = util::split(value);
        self.map.entry(hi).or_default().try_push(lo)
    }

    /// Pushes `value` in the treemap only if it is greater than the current maximum value.
    /// It is up to the caller to have validated index > self.max()
    ///
    /// # Panics
    ///
    /// If debug_assertions enabled and index is > self.max()
    pub(crate) fn push_unchecked(&mut self, value: u64) {
        let (hi, lo) = util::split(value);
        // BTreeMap last_mut not stabilized see https://github.com/rust-lang/rust/issues/62924
        match self.map.iter_mut().next_back() {
            Some((&key, bitmap)) if key == hi => bitmap.push_unchecked(lo),
            Some((&key, _)) if cfg!(debug_assertions) && key > hi => {
                panic!("last bitmap key > key of value")
            }
            _otherwise => {
                // The tree is empty
                let mut rb = RoaringBitmap::new();
                rb.push_unchecked(lo);
                self.map.insert(hi, rb);
            }
        }
    }

    /// Removes a value from the set. Returns `true` if the value was present in the set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// rb.insert(3);
    /// assert_eq!(rb.remove(3), true);
    /// assert_eq!(rb.remove(3), false);
    /// assert_eq!(rb.contains(3), false);
    /// ```
    pub fn remove(&mut self, value: u64) -> bool {
        let (hi, lo) = util::split(value);
        match self.map.entry(hi) {
            Entry::Vacant(_) => false,
            Entry::Occupied(mut ent) => {
                if ent.get_mut().remove(lo) {
                    if ent.get().is_empty() {
                        ent.remove();
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Physical capacity affected by one point insert/remove of `value`.
    /// Returns `(high-key map entries, bitmap directory bytes, target store
    /// bytes, additional allocation bound)`. The private BTreeMap's opaque node
    /// storage is excluded; callers can apply a separate conservative bound.
    /// Taking this snapshot before and after a point mutation yields retained
    /// allocation deltas without scanning unrelated bitmaps or containers.
    /// Complexity is O(log high-key groups + log containers in the target group).
    pub fn allocation_for_value(&self, value: u64, inserting: bool) -> (usize, usize, usize, usize) {
        let (hi, lo) = util::split(value);
        let (directory, store, bound) = match self.map.get(&hi) {
            Some(bitmap) => bitmap.allocation_for_value(lo, inserting),
            None => RoaringBitmap::new().allocation_for_value(lo, inserting),
        };
        (self.map.len(), directory, store, bound)
    }

    /// Removes a range of values.
    /// Returns the number of removed values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// rb.insert(2);
    /// rb.insert(3);
    /// assert_eq!(rb.remove_range(2..4), 2);
    /// ```
    pub fn remove_range<R>(&mut self, range: R) -> u64
    where
        R: RangeBounds<u64>,
    {
        let (start, end) = match util::convert_range_to_inclusive(range) {
            Some(range) => (*range.start(), *range.end()),
            None => return 0,
        };

        let (start_container_key, start_index) = util::split(start);
        let (end_container_key, end_index) = util::split(end);

        let mut keys_to_remove = Vec::new();
        let mut removed = 0;

        for (&key, rb) in &mut self.map {
            if key >= start_container_key && key <= end_container_key {
                let a = if key == start_container_key { start_index } else { 0 };
                let b = if key == end_container_key { end_index } else { u32::MAX };
                removed += rb.remove_range(a..=b);
                if rb.is_empty() {
                    keys_to_remove.push(key);
                }
            }
        }

        for key in keys_to_remove {
            self.map.remove(&key);
        }

        removed
    }

    /// Returns `true` if this set contains the specified integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// rb.insert(1);
    /// assert_eq!(rb.contains(0), false);
    /// assert_eq!(rb.contains(1), true);
    /// assert_eq!(rb.contains(100), false);
    /// ```
    pub fn contains(&self, value: u64) -> bool {
        let (hi, lo) = util::split(value);
        match self.map.get(&hi) {
            None => false,
            Some(r) => r.contains(lo),
        }
    }

    /// Clears all integers in this set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// rb.insert(1);
    /// assert_eq!(rb.contains(1), true);
    /// rb.clear();
    /// assert_eq!(rb.contains(1), false);
    /// ```
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Returns `true` if there are no integers in this set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert_eq!(rb.is_empty(), true);
    ///
    /// rb.insert(3);
    /// assert_eq!(rb.is_empty(), false);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.map.values().all(RoaringBitmap::is_empty)
    }

    /// Returns `true` if there are every possible integers in this set.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::full();
    /// assert!(!rb.is_empty());
    /// assert!(rb.is_full());
    /// ```
    pub fn is_full(&self) -> bool {
        self.map.len() == (u32::MAX as usize + 1) && self.map.values().all(RoaringBitmap::is_full)
    }

    /// Returns the number of distinct integers added to the set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert_eq!(rb.len(), 0);
    ///
    /// rb.insert(3);
    /// assert_eq!(rb.len(), 1);
    ///
    /// rb.insert(3);
    /// rb.insert(4);
    /// assert_eq!(rb.len(), 2);
    /// ```
    pub fn len(&self) -> u64 {
        self.map.values().map(RoaringBitmap::len).sum()
    }

    /// Returns the minimum value in the set (if the set is non-empty).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert_eq!(rb.min(), None);
    ///
    /// rb.insert(3);
    /// rb.insert(4);
    /// assert_eq!(rb.min(), Some(3));
    /// ```
    pub fn min(&self) -> Option<u64> {
        self.map
            .iter()
            .find(|&(_, rb)| rb.min().is_some())
            .map(|(k, rb)| util::join(*k, rb.min().unwrap()))
    }

    /// Returns the maximum value in the set (if the set is non-empty).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert_eq!(rb.max(), None);
    ///
    /// rb.insert(3);
    /// rb.insert(4);
    /// assert_eq!(rb.max(), Some(4));
    /// ```
    pub fn max(&self) -> Option<u64> {
        self.map
            .iter()
            .rev()
            .find(|&(_, rb)| rb.max().is_some())
            .map(|(k, rb)| util::join(*k, rb.max().unwrap()))
    }

    /// Returns the number of integers that are <= value. rank(u64::MAX) == len()
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert_eq!(rb.rank(0), 0);
    ///
    /// rb.insert(3);
    /// rb.insert(4);
    /// assert_eq!(rb.rank(3), 1);
    /// assert_eq!(rb.rank(10), 2)
    /// ```
    pub fn rank(&self, value: u64) -> u64 {
        // if len becomes cached for RoaringTreemap: return len if len > value

        let (hi, lo) = util::split(value);
        let mut iter = self.map.range(..=hi).rev();

        iter.next()
            .map(|(&k, bitmap)| if k == hi { bitmap.rank(lo) } else { bitmap.len() })
            .unwrap_or(0)
            + iter.map(|(_, bitmap)| bitmap.len()).sum::<u64>()
    }

    /// Returns the `n`th integer in the set or `None` if `n >= len()`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringTreemap;
    ///
    /// let mut rb = RoaringTreemap::new();
    /// assert_eq!(rb.select(0), None);
    ///
    /// rb.append(vec![0, 10, 100]);
    ///
    /// assert_eq!(rb.select(0), Some(0));
    /// assert_eq!(rb.select(1), Some(10));
    /// assert_eq!(rb.select(2), Some(100));
    /// assert_eq!(rb.select(3), None);
    /// ```
    pub fn select(&self, mut n: u64) -> Option<u64> {
        for (&key, bitmap) in &self.map {
            let len = bitmap.len();
            if len > n {
                return Some(((key as u64) << 32) | bitmap.select(n as u32).unwrap() as u64);
            }
            n -= len;
        }

        None
    }
}

impl Default for RoaringTreemap {
    fn default() -> RoaringTreemap {
        RoaringTreemap::new()
    }
}

impl Clone for RoaringTreemap {
    fn clone(&self) -> Self {
        RoaringTreemap { map: self.map.clone() }
    }

    fn clone_from(&mut self, other: &Self) {
        self.map.clone_from(&other.map);
    }
}


#[cfg(test)]
mod allocation_tests {
    use crate::RoaringTreemap;

    #[test]
    fn point_capacity_tracks_sparse_dense_conversion_and_group_release() {
        let mut bitmap = RoaringTreemap::new();
        let empty = bitmap.allocation_for_value(1, true);
        assert_eq!((empty.0, empty.1, empty.2), (0, 0, 0));
        assert!(empty.3 < 1024);
        bitmap.insert(1);
        let sparse = bitmap.allocation_for_value(1, true);
        assert_eq!(sparse.0, 1);
        assert!(sparse.1 > 0 && sparse.2 > 0 && sparse.2 < 64);
        for value in 2..=4096 { bitmap.insert(value); }
        let before = bitmap.allocation_for_value(4097, true);
        bitmap.insert(4097);
        let dense = bitmap.allocation_for_value(4097, false);
        assert_eq!(dense.2, 8192);
        assert!(before.3 >= dense.2);
        bitmap.remove(4097);
        assert!(bitmap.allocation_for_value(1, false).2 <= 8192);
        bitmap.insert(1 << 48);
        assert_eq!(bitmap.allocation_for_value(1, false).0, 2);
        bitmap.remove(1 << 48);
        let removed_group = bitmap.allocation_for_value(1 << 48, false);
        assert_eq!((removed_group.0, removed_group.1, removed_group.2), (1, 0, 0));
        for value in 1..=4096 { bitmap.remove(value); }
        assert_eq!(bitmap.allocation_for_value(1, false), (0, 0, 0, 0));
    }
}
