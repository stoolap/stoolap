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

//! Fast hash maps for integer and string keys
//!
//! This module provides optimized hash maps:
//! - `I64Map`/`I64Set` for i64 keys (custom pre-mixing hash)
//! - `StringMap`/`StringSet` for String keys (AHash)
//! - `ConcurrentI64Map` for concurrent access (DashMap)
//! - `CowBTreeMap` for ordered iteration with lock-free reads

use ahash::{AHashMap, AHashSet};
use dashmap::DashMap;
use parking_lot::RwLock;
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;

/// Type alias for FxHash's BuildHasher
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// Fast single-threaded hash map for i64 keys
///
/// Uses custom I64Map with key transformation for 45% faster lookups
/// compared to FxHashMap while maintaining similar insert performance.
pub type I64Map<V> = crate::common::i64_map::I64Map<V>;

/// Fast single-threaded hash set for i64 keys
///
/// Uses custom I64Set with pre-mixing hash for 0 sequential key collisions.
/// Note: i64::MIN cannot be used (reserved as empty sentinel).
pub type I64Set = crate::common::i64_map::I64Set;

/// Fast hash map for String keys
///
/// Uses AHash which provides ~10% faster lookups than FxHash for String keys
/// due to AES-NI instructions handling complex types well.
pub type StringMap<V> = AHashMap<String, V>;

/// Fast hash set for String keys
///
/// Uses AHash which provides ~10% faster lookups than FxHash for String keys.
pub type StringSet = AHashSet<String>;

/// Concurrent hash map for i64 keys
///
/// Uses DashMap with FxHash for fast concurrent access.
/// Provides sharded, lock-free reads and fine-grained locking for writes.
pub type ConcurrentI64Map<V> = DashMap<i64, V, FxBuildHasher>;

/// Copy-on-Write B-tree map for i64 keys with lock-free reads
///
/// Key advantage: `.read().clone()` is O(1) for creating snapshots.
///
/// Usage pattern for lock-free reads:
/// ```ignore
/// // Brief lock, O(1) clone, then lock-free iteration
/// let snapshot = map.read().clone();
/// for (k, v) in snapshot.iter() { ... }  // No lock held!
/// ```
pub type CowBTreeMap<V> = RwLock<crate::common::CowBTree<V>>;

/// Create a new I64Map with default capacity
#[inline]
pub fn new_i64_map<V>() -> I64Map<V> {
    crate::common::i64_map::I64Map::new()
}

/// Create a new I64Map with specified capacity
#[inline]
pub fn new_i64_map_with_capacity<V>(capacity: usize) -> I64Map<V> {
    crate::common::i64_map::I64Map::with_capacity(capacity)
}

/// Create a new ConcurrentI64Map with default capacity
#[inline]
pub fn new_concurrent_i64_map<V>() -> ConcurrentI64Map<V> {
    DashMap::with_hasher(FxBuildHasher::default())
}

/// Create a new CowBTreeMap (lock-free reads via snapshots)
#[inline]
pub fn new_cow_btree_map<V: Clone>() -> CowBTreeMap<V> {
    RwLock::new(crate::common::CowBTree::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_i64_map_basic() {
        let mut map: I64Map<String> = new_i64_map();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());
        map.insert(-1, "negative".to_string());
        // Note: i64::MIN is reserved as empty sentinel, so we use i64::MIN + 1
        map.insert(i64::MIN + 1, "near_min".to_string());
        map.insert(i64::MAX, "max".to_string());

        assert_eq!(map.get(1), Some(&"one".to_string()));
        assert_eq!(map.get(-1), Some(&"negative".to_string()));
        assert_eq!(map.get(i64::MIN + 1), Some(&"near_min".to_string()));
        assert_eq!(map.get(i64::MAX), Some(&"max".to_string()));

        map.remove(2);
        assert!(!map.contains_key(2));
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn test_i64_map_with_capacity() {
        let map: I64Map<i32> = new_i64_map_with_capacity(100);
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn test_i64_set() {
        let mut set = I64Set::default();

        set.insert(1);
        set.insert(2);
        set.insert(3);

        assert!(set.contains(1));
        assert!(set.contains(2));
        assert!(!set.contains(4));

        set.remove(2);
        assert!(!set.contains(2));
    }

    #[test]
    fn test_concurrent_i64_map() {
        let map: ConcurrentI64Map<String> = new_concurrent_i64_map();

        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());

        assert_eq!(map.get(&1).map(|v| v.clone()), Some("one".to_string()));
        assert!(map.get(&3).is_none());

        map.remove(&1);
        assert!(map.get(&1).is_none());
    }

    #[test]
    fn test_concurrent_i64_map_multithreaded() {
        let map: Arc<ConcurrentI64Map<i64>> = Arc::new(new_concurrent_i64_map());
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
    fn test_cow_btree_map() {
        let map: CowBTreeMap<String> = new_cow_btree_map();

        map.write().insert(3, "three".to_string());
        map.write().insert(1, "one".to_string());
        map.write().insert(2, "two".to_string());

        // Read via read lock
        assert_eq!(map.read().get(1), Some(&"one".to_string()));
        assert_eq!(map.read().len(), 3);

        // O(1) snapshot for lock-free iteration
        let snapshot = map.read().clone(); // This is O(1) for CowBTree!
        let keys: Vec<i64> = snapshot.keys().collect();
        assert_eq!(keys, vec![1, 2, 3]);

        // Snapshot is independent - writer can modify while reader iterates
        map.write().insert(4, "four".to_string());
        assert_eq!(snapshot.len(), 3); // Snapshot unchanged
        assert_eq!(map.read().len(), 4); // Map updated
    }

    #[test]
    fn test_cow_btree_map_multithreaded() {
        let map: Arc<CowBTreeMap<i64>> = Arc::new(new_cow_btree_map());
        let num_threads: i64 = 4;
        let ops_per_thread: i64 = 100;

        // Writer threads
        let write_handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    let base = t * ops_per_thread;
                    for i in 0..ops_per_thread {
                        map.write().insert(base + i, (base + i) * 2);
                    }
                })
            })
            .collect();

        // Reader threads using O(1) snapshots (lock-free after clone)
        let read_handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let snapshot = map.read().clone(); // O(1) clone!
                        let _count = snapshot.iter().count(); // Lock-free iteration
                    }
                })
            })
            .collect();

        for handle in write_handles {
            handle.join().unwrap();
        }
        for handle in read_handles {
            handle.join().unwrap();
        }

        assert_eq!(map.read().len(), (num_threads * ops_per_thread) as usize);
    }
}
