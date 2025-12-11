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

//! Fast hash maps for integer keys
//!
//! # Implementation Notes
//!
//! This module provides optimized hash maps for integer keys:
//! - `FxHashMap` from rustc-hash for single-threaded access (fast hash, no cryptographic guarantees)
//! - `DashMap` for concurrent access (sharded, lock-free reads)

use dashmap::DashMap;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::collections::BTreeMap;
use std::hash::BuildHasherDefault;
use std::sync::RwLock;

/// Type alias for FxHash's BuildHasher
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// Fast single-threaded hash map for i64 keys
///
/// Uses FxHash (the hash function used by rustc) which is optimized for
/// integer keys and provides excellent performance for non-cryptographic uses.
///
/// # Example
/// ```
/// use stoolap::common::Int64Map;
///
/// let mut map: Int64Map<String> = Int64Map::default();
/// map.insert(42, "hello".to_string());
/// assert_eq!(map.get(&42), Some(&"hello".to_string()));
/// ```
pub type Int64Map<V> = FxHashMap<i64, V>;

/// Fast single-threaded hash map for u64 keys
///
/// Same as Int64Map but for unsigned integers.
pub type UInt64Map<V> = FxHashMap<u64, V>;

/// Fast single-threaded hash map for usize keys
///
/// Useful for array indices and similar use cases.
pub type UsizeMap<V> = FxHashMap<usize, V>;

/// Fast single-threaded hash set for i64 keys
pub type Int64Set = FxHashSet<i64>;

/// Fast single-threaded hash set for u64 keys
pub type UInt64Set = FxHashSet<u64>;

/// Fast single-threaded hash set for usize keys
pub type UsizeSet = FxHashSet<usize>;

/// Concurrent hash map for i64 keys
///
/// Uses DashMap with FxHash for fast concurrent access.
/// Provides sharded, lock-free reads and fine-grained locking for writes.
///
/// # Example
/// ```
/// use stoolap::common::ConcurrentInt64Map;
///
/// let map: ConcurrentInt64Map<String> = ConcurrentInt64Map::default();
/// map.insert(42, "hello".to_string());
/// assert_eq!(map.get(&42).map(|v| v.clone()), Some("hello".to_string()));
/// ```
pub type ConcurrentInt64Map<V> = DashMap<i64, V, FxBuildHasher>;

/// Concurrent hash map for u64 keys
pub type ConcurrentUInt64Map<V> = DashMap<u64, V, FxBuildHasher>;

/// Concurrent hash map for usize keys
pub type ConcurrentUsizeMap<V> = DashMap<usize, V, FxBuildHasher>;

/// Ordered concurrent map for i64 keys using RwLock<BTreeMap>
///
/// Provides ordered iteration without sorting overhead. Best for workloads
/// that need ordered scans but have single-writer semantics (like MVCC).
///
/// Performance characteristics:
/// - Ordered iteration: O(n) - no sorting needed
/// - Point lookup: O(log n)
/// - Insert: O(log n) under write lock
///
/// # Example
/// ```
/// use stoolap::common::OrderedInt64Map;
/// use std::collections::BTreeMap;
/// use std::sync::RwLock;
///
/// let map: OrderedInt64Map<String> = RwLock::new(BTreeMap::new());
/// map.write().unwrap().insert(42, "hello".to_string());
/// assert_eq!(map.read().unwrap().get(&42), Some(&"hello".to_string()));
/// ```
pub type OrderedInt64Map<V> = RwLock<BTreeMap<i64, V>>;

/// Create a new Int64Map with default capacity
pub fn new_int64_map<V>() -> Int64Map<V> {
    FxHashMap::default()
}

/// Create a new Int64Map with specified capacity
pub fn new_int64_map_with_capacity<V>(capacity: usize) -> Int64Map<V> {
    FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new UInt64Map with default capacity
pub fn new_uint64_map<V>() -> UInt64Map<V> {
    FxHashMap::default()
}

/// Create a new UInt64Map with specified capacity
pub fn new_uint64_map_with_capacity<V>(capacity: usize) -> UInt64Map<V> {
    FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new UsizeMap with default capacity
pub fn new_usize_map<V>() -> UsizeMap<V> {
    FxHashMap::default()
}

/// Create a new UsizeMap with specified capacity
pub fn new_usize_map_with_capacity<V>(capacity: usize) -> UsizeMap<V> {
    FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new ConcurrentInt64Map with default capacity
pub fn new_concurrent_int64_map<V>() -> ConcurrentInt64Map<V> {
    DashMap::with_hasher(FxBuildHasher::default())
}

/// Create a new ConcurrentInt64Map with specified capacity
pub fn new_concurrent_int64_map_with_capacity<V>(capacity: usize) -> ConcurrentInt64Map<V> {
    DashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new ConcurrentUInt64Map with default capacity
pub fn new_concurrent_uint64_map<V>() -> ConcurrentUInt64Map<V> {
    DashMap::with_hasher(FxBuildHasher::default())
}

/// Create a new ConcurrentUInt64Map with specified capacity
pub fn new_concurrent_uint64_map_with_capacity<V>(capacity: usize) -> ConcurrentUInt64Map<V> {
    DashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new ConcurrentUsizeMap with default capacity
pub fn new_concurrent_usize_map<V>() -> ConcurrentUsizeMap<V> {
    DashMap::with_hasher(FxBuildHasher::default())
}

/// Create a new ConcurrentUsizeMap with specified capacity
pub fn new_concurrent_usize_map_with_capacity<V>(capacity: usize) -> ConcurrentUsizeMap<V> {
    DashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new OrderedInt64Map
pub fn new_ordered_int64_map<V>() -> OrderedInt64Map<V> {
    RwLock::new(BTreeMap::new())
}

/// Create a new Int64Set with default capacity
pub fn new_int64_set() -> Int64Set {
    FxHashSet::default()
}

/// Create a new Int64Set with specified capacity
pub fn new_int64_set_with_capacity(capacity: usize) -> Int64Set {
    FxHashSet::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new UInt64Set with default capacity
pub fn new_uint64_set() -> UInt64Set {
    FxHashSet::default()
}

/// Create a new UInt64Set with specified capacity
pub fn new_uint64_set_with_capacity(capacity: usize) -> UInt64Set {
    FxHashSet::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new UsizeSet with default capacity
pub fn new_usize_set() -> UsizeSet {
    FxHashSet::default()
}

/// Create a new UsizeSet with specified capacity
pub fn new_usize_set_with_capacity(capacity: usize) -> UsizeSet {
    FxHashSet::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Segment for SegmentInt64Map - holds a portion of the map with its own lock
struct Segment<V> {
    data: std::sync::RwLock<Int64Map<V>>,
}

/// Fast segmented concurrent map for i64 keys
///
/// Uses sharding to reduce lock contention while maintaining the performance
/// benefits of FxHashMap. Each segment has its own RwLock for fine-grained locking.
///
/// Performance characteristics:
/// - 16 segments by default (configurable)
/// - Each segment uses FxHashMap internally
/// - Very low overhead for non-contended access
/// - Better cache locality than DashMap
pub struct SegmentInt64Map<V> {
    segments: Box<[Segment<V>]>,
    segment_mask: usize,
}

impl<V> SegmentInt64Map<V> {
    /// Creates a new SegmentInt64Map with default 16 segments
    pub fn new() -> Self {
        Self::with_segments(16, 64)
    }

    /// Creates a new SegmentInt64Map with specified segment count and capacity per segment
    pub fn with_segments(segment_count: usize, capacity_per_segment: usize) -> Self {
        // Ensure segment count is power of 2
        let segment_count = segment_count.next_power_of_two();
        let segment_mask = segment_count - 1;

        let segments: Vec<Segment<V>> = (0..segment_count)
            .map(|_| Segment {
                data: std::sync::RwLock::new(FxHashMap::with_capacity_and_hasher(
                    capacity_per_segment,
                    FxBuildHasher::default(),
                )),
            })
            .collect();

        Self {
            segments: segments.into_boxed_slice(),
            segment_mask,
        }
    }

    /// Hash function to distribute keys across segments
    #[inline]
    fn segment_index(&self, key: i64) -> usize {
        // Use multiplicative hashing for good distribution
        let h = (key as u64).wrapping_mul(0x9E3779B97F4A7C15);
        ((h >> 32) as usize) & self.segment_mask
    }

    /// Get a value by key
    #[inline]
    pub fn get(&self, key: &i64) -> Option<V>
    where
        V: Clone,
    {
        let idx = self.segment_index(*key);
        let guard = self.segments[idx].data.read().unwrap();
        guard.get(key).cloned()
    }

    /// Check if key exists
    #[inline]
    pub fn contains_key(&self, key: &i64) -> bool {
        let idx = self.segment_index(*key);
        let guard = self.segments[idx].data.read().unwrap();
        guard.contains_key(key)
    }

    /// Insert a key-value pair
    #[inline]
    pub fn insert(&self, key: i64, value: V) {
        let idx = self.segment_index(key);
        let mut guard = self.segments[idx].data.write().unwrap();
        guard.insert(key, value);
    }

    /// Insert if not exists, returns (existing_value, was_inserted)
    #[inline]
    pub fn insert_if_not_exists(&self, key: i64, value: V) -> (Option<V>, bool)
    where
        V: Clone,
    {
        let idx = self.segment_index(key);
        let mut guard = self.segments[idx].data.write().unwrap();
        if let Some(existing) = guard.get(&key) {
            (Some(existing.clone()), false)
        } else {
            guard.insert(key, value);
            (None, true)
        }
    }

    /// Remove a key
    #[inline]
    pub fn remove(&self, key: &i64) -> Option<V> {
        let idx = self.segment_index(*key);
        let mut guard = self.segments[idx].data.write().unwrap();
        guard.remove(key)
    }

    /// Get total count across all segments
    pub fn len(&self) -> usize {
        self.segments
            .iter()
            .map(|s| s.data.read().unwrap().len())
            .sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.segments
            .iter()
            .all(|s| s.data.read().unwrap().is_empty())
    }

    /// Clear all segments
    pub fn clear(&self) {
        for segment in self.segments.iter() {
            segment.data.write().unwrap().clear();
        }
    }

    /// Iterate over all key-value pairs (collects to avoid holding locks)
    pub fn iter_collect(&self) -> Vec<(i64, V)>
    where
        V: Clone,
    {
        let mut result = Vec::new();
        for segment in self.segments.iter() {
            let guard = segment.data.read().unwrap();
            for (k, v) in guard.iter() {
                result.push((*k, v.clone()));
            }
        }
        result
    }
}

impl<V> Default for SegmentInt64Map<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a new SegmentInt64Map with default settings
pub fn new_segment_int64_map<V>() -> SegmentInt64Map<V> {
    SegmentInt64Map::new()
}

/// Create a new SegmentInt64Map with specified segment count
pub fn new_segment_int64_map_with_segments<V>(
    segment_count: usize,
    capacity_per_segment: usize,
) -> SegmentInt64Map<V> {
    SegmentInt64Map::with_segments(segment_count, capacity_per_segment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_int64_map_basic() {
        let mut map: Int64Map<String> = new_int64_map();

        // Insert
        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(3, "three".to_string());

        // Get
        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.get(&2), Some(&"two".to_string()));
        assert_eq!(map.get(&3), Some(&"three".to_string()));
        assert_eq!(map.get(&4), None);

        // Remove
        assert_eq!(map.remove(&2), Some("two".to_string()));
        assert_eq!(map.get(&2), None);

        // Length
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_int64_map_with_capacity() {
        let map: Int64Map<i32> = new_int64_map_with_capacity(100);
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn test_int64_map_negative_keys() {
        let mut map: Int64Map<String> = new_int64_map();

        map.insert(-1, "negative one".to_string());
        map.insert(i64::MIN, "min".to_string());
        map.insert(i64::MAX, "max".to_string());

        assert_eq!(map.get(&-1), Some(&"negative one".to_string()));
        assert_eq!(map.get(&i64::MIN), Some(&"min".to_string()));
        assert_eq!(map.get(&i64::MAX), Some(&"max".to_string()));
    }

    #[test]
    fn test_uint64_map_basic() {
        let mut map: UInt64Map<String> = new_uint64_map();

        map.insert(0, "zero".to_string());
        map.insert(u64::MAX, "max".to_string());

        assert_eq!(map.get(&0), Some(&"zero".to_string()));
        assert_eq!(map.get(&u64::MAX), Some(&"max".to_string()));
    }

    #[test]
    fn test_usize_map_basic() {
        let mut map: UsizeMap<String> = new_usize_map();

        map.insert(0, "zero".to_string());
        map.insert(100, "hundred".to_string());

        assert_eq!(map.get(&0), Some(&"zero".to_string()));
        assert_eq!(map.get(&100), Some(&"hundred".to_string()));
    }

    #[test]
    fn test_int64_set_basic() {
        let mut set = new_int64_set();

        set.insert(1);
        set.insert(2);
        set.insert(3);

        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(!set.contains(&4));

        set.remove(&2);
        assert!(!set.contains(&2));
    }

    #[test]
    fn test_concurrent_int64_map_basic() {
        let map: ConcurrentInt64Map<String> = new_concurrent_int64_map();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());

        assert_eq!(map.get(&1).map(|v| v.clone()), Some("one".to_string()));
        assert_eq!(map.get(&2).map(|v| v.clone()), Some("two".to_string()));
        assert!(map.get(&3).is_none());

        map.remove(&1);
        assert!(map.get(&1).is_none());
    }

    #[test]
    fn test_concurrent_int64_map_multithreaded() {
        let map: Arc<ConcurrentInt64Map<i64>> = Arc::new(new_concurrent_int64_map());
        let num_threads: i64 = 8;
        let ops_per_thread: i64 = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let map: Arc<ConcurrentInt64Map<i64>> = Arc::clone(&map);
                thread::spawn(move || {
                    let base = t * ops_per_thread;
                    // Insert
                    for i in 0..ops_per_thread {
                        map.insert(base + i, (base + i) * 2);
                    }
                    // Read
                    for i in 0..ops_per_thread {
                        let key = base + i;
                        let expected = key * 2;
                        assert_eq!(map.get(&key).map(|v| *v), Some(expected));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all values
        assert_eq!(map.len(), (num_threads * ops_per_thread) as usize);
    }

    #[test]
    fn test_concurrent_int64_map_with_capacity() {
        let map: ConcurrentInt64Map<i32> = new_concurrent_int64_map_with_capacity(100);
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn test_concurrent_map_update() {
        let map: ConcurrentInt64Map<i64> = new_concurrent_int64_map();

        map.insert(1, 100);
        assert_eq!(*map.get(&1).unwrap(), 100);

        // Update existing key
        map.insert(1, 200);
        assert_eq!(*map.get(&1).unwrap(), 200);

        // Use entry API
        map.entry(2).or_insert(300);
        assert_eq!(*map.get(&2).unwrap(), 300);
    }

    #[test]
    fn test_concurrent_map_iter() {
        let map: ConcurrentInt64Map<i64> = new_concurrent_int64_map();

        for i in 0..10 {
            map.insert(i, i * 10);
        }

        let sum: i64 = map.iter().map(|r| *r.value()).sum();
        assert_eq!(sum, 450); // 0 + 10 + 20 + ... + 90 = 450
    }

    #[test]
    fn test_map_type_inference() {
        // Test that types work with various value types
        let mut int_map: Int64Map<i32> = new_int64_map();
        int_map.insert(1, 42);

        let mut string_map: Int64Map<String> = new_int64_map();
        string_map.insert(1, "hello".to_string());

        let mut vec_map: Int64Map<Vec<u8>> = new_int64_map();
        vec_map.insert(1, vec![1, 2, 3]);

        assert_eq!(int_map.get(&1), Some(&42));
        assert_eq!(string_map.get(&1), Some(&"hello".to_string()));
        assert_eq!(vec_map.get(&1), Some(&vec![1, 2, 3]));
    }

    #[test]
    fn test_set_operations() {
        let mut set1 = new_int64_set();
        let mut set2 = new_int64_set();

        for i in 0..5 {
            set1.insert(i);
        }
        for i in 3..8 {
            set2.insert(i);
        }

        // Intersection
        let intersection: Int64Set = set1.intersection(&set2).copied().collect();
        assert_eq!(intersection.len(), 2); // 3, 4

        // Union
        let union: Int64Set = set1.union(&set2).copied().collect();
        assert_eq!(union.len(), 8); // 0, 1, 2, 3, 4, 5, 6, 7
    }
}
