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

//! Common utilities for Stoolap
//!
//! This module contains shared utilities used throughout the database:
//!
//! - [`version`] - Version information and constants
//! - [`buffer_pool`] - Self-tuning buffer pool for efficient memory reuse
//! - [`int_maps`] - Fast hash maps for integer keys
//! - [`compact_vec`] - 16-byte vector optimized for Row storage
//! - [`compact_arc`] - Memory-efficient Arc without weak reference support

pub mod buffer_pool;
pub mod compact_arc;
pub mod compact_vec;
pub mod int_maps;
pub mod version;

// Re-export main types for convenience
pub use buffer_pool::{BufferPool, PoolStats};
pub use compact_arc::CompactArc;
pub use compact_vec::CompactVec;
pub use int_maps::{
    new_btree_int64_map, new_concurrent_int64_map, new_int64_map, new_int64_map_with_capacity,
    BTreeInt64Map, ConcurrentInt64Map, Int64Map, Int64Set,
};
pub use version::{version, version_info, SemVer, BUILD_TIME, GIT_COMMIT, MAJOR, MINOR, PATCH};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_buffer_pool_with_int64_map() {
        // Test buffer pool and int64 map working together
        let pool = BufferPool::new(1024, 4096, "test");
        let map: Int64Map<Vec<u8>> = new_int64_map();

        // Get buffer, use it, store in map
        let mut buf = pool.get();
        buf.extend_from_slice(b"test data");

        let mut map = map;
        map.insert(1, buf);

        assert!(map.contains_key(&1));
        assert_eq!(map.get(&1).unwrap(), b"test data");
    }

    #[test]
    fn test_version_constants() {
        // Ensure version is accessible
        let v = version();
        assert!(!v.is_empty());

        let info = version_info();
        assert!(info.contains("stoolap"));
    }

    #[test]
    fn test_concurrent_map_with_buffer_pool() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(BufferPool::new(1024, 4096, "test"));
        let map: Arc<ConcurrentInt64Map<Vec<u8>>> = Arc::new(new_concurrent_int64_map());

        let handles: Vec<_> = (0i64..4)
            .map(|i| {
                let pool = Arc::clone(&pool);
                let map: Arc<ConcurrentInt64Map<Vec<u8>>> = Arc::clone(&map);
                thread::spawn(move || {
                    for j in 0i64..10 {
                        let mut buf = pool.get();
                        buf.extend_from_slice(format!("thread {} item {}", i, j).as_bytes());
                        map.insert(i * 100 + j, buf);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(map.len(), 40);
    }
}
