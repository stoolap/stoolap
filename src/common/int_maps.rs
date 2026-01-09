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
//! This module provides optimized hash maps for integer keys:
//! - `FxHashMap` from rustc-hash for single-threaded access
//! - `DashMap` for concurrent access (sharded, lock-free reads)
//! - `BTreeMap` with `RwLock` for ordered iteration

use dashmap::DashMap;
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::collections::BTreeMap;
use std::hash::BuildHasherDefault;

/// Type alias for FxHash's BuildHasher
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// Fast single-threaded hash map for i64 keys
///
/// Uses FxHash (the hash function used by rustc) which is optimized for
/// integer keys and provides excellent performance for non-cryptographic uses.
pub type Int64Map<V> = FxHashMap<i64, V>;

/// Fast single-threaded hash set for i64 keys
pub type Int64Set = FxHashSet<i64>;

/// Concurrent hash map for i64 keys
///
/// Uses DashMap with FxHash for fast concurrent access.
/// Provides sharded, lock-free reads and fine-grained locking for writes.
pub type ConcurrentInt64Map<V> = DashMap<i64, V, FxBuildHasher>;

/// BTreeMap-based map for i64 keys using parking_lot::RwLock
///
/// Uses BTreeMap for ordered iteration and O(log n) lookups.
/// Better for large datasets (1M+ rows) due to memory efficiency.
pub type BTreeInt64Map<V> = RwLock<BTreeMap<i64, V>>;

/// Create a new Int64Map with default capacity
#[inline]
pub fn new_int64_map<V>() -> Int64Map<V> {
    FxHashMap::default()
}

/// Create a new Int64Map with specified capacity
#[inline]
pub fn new_int64_map_with_capacity<V>(capacity: usize) -> Int64Map<V> {
    FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new ConcurrentInt64Map with default capacity
#[inline]
pub fn new_concurrent_int64_map<V>() -> ConcurrentInt64Map<V> {
    DashMap::with_hasher(FxBuildHasher::default())
}

/// Create a new BTreeInt64Map
#[inline]
pub fn new_btree_int64_map<V>() -> BTreeInt64Map<V> {
    RwLock::new(BTreeMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_int64_map_basic() {
        let mut map: Int64Map<String> = new_int64_map();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(-1, "negative".to_string());
        map.insert(i64::MIN, "min".to_string());
        map.insert(i64::MAX, "max".to_string());

        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.get(&-1), Some(&"negative".to_string()));
        assert_eq!(map.get(&i64::MIN), Some(&"min".to_string()));
        assert_eq!(map.get(&i64::MAX), Some(&"max".to_string()));

        map.remove(&2);
        assert!(!map.contains_key(&2));
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn test_int64_map_with_capacity() {
        let map: Int64Map<i32> = new_int64_map_with_capacity(100);
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn test_int64_set() {
        let mut set = Int64Set::default();

        set.insert(1);
        set.insert(2);
        set.insert(3);

        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&4));

        set.remove(&2);
        assert!(!set.contains(&2));
    }

    #[test]
    fn test_concurrent_int64_map() {
        let map: ConcurrentInt64Map<String> = new_concurrent_int64_map();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());

        assert_eq!(map.get(&1).map(|v| v.clone()), Some("one".to_string()));
        assert!(map.get(&3).is_none());

        map.remove(&1);
        assert!(map.get(&1).is_none());
    }

    #[test]
    fn test_concurrent_int64_map_multithreaded() {
        let map: Arc<ConcurrentInt64Map<i64>> = Arc::new(new_concurrent_int64_map());
        let num_threads: i64 = 4;
        let ops_per_thread: i64 = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    let base = t * ops_per_thread;
                    for i in 0..ops_per_thread {
                        map.insert(base + i, (base + i) * 2);
                    }
                    for i in 0..ops_per_thread {
                        let key = base + i;
                        assert_eq!(map.get(&key).map(|v| *v), Some(key * 2));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(map.len(), (num_threads * ops_per_thread) as usize);
    }

    #[test]
    fn test_btree_int64_map() {
        let map: BTreeInt64Map<String> = new_btree_int64_map();

        map.write().insert(3, "three".to_string());
        map.write().insert(1, "one".to_string());
        map.write().insert(2, "two".to_string());

        assert_eq!(map.read().get(&1), Some(&"one".to_string()));

        // Verify ordered iteration
        let keys: Vec<i64> = map.read().keys().copied().collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }
}
