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
//! - [`maps`] - Fast hash maps for integer keys
//! - [`compact_vec`] - 16-byte vector optimized for Row storage
//! - [`compact_arc`] - Memory-efficient Arc without weak reference support

pub mod buffer_pool;
pub mod compact_arc;
pub mod compact_vec;
pub mod cow_btree;
pub mod cow_hashmap;
pub mod i64_map;
pub mod maps;
pub mod smart_string;
pub mod version;

// Re-export main types for convenience
pub use buffer_pool::{BufferPool, PoolStats};
pub use compact_arc::CompactArc;
pub use compact_vec::CompactVec;
pub use cow_btree::CowBTree;
pub use cow_hashmap::CowHashMap;
pub use i64_map::{I64Map, I64Set};
pub use maps::{
    new_cow_btree_map, new_i64_map, new_i64_map_with_capacity, CowBTreeMap, StringMap, StringSet,
};
pub use smart_string::SmartString;
pub use version::{version, version_info, SemVer, BUILD_TIME, GIT_COMMIT, MAJOR, MINOR, PATCH};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_buffer_pool_with_i64_map() {
        // Test buffer pool and i64 map working together
        let pool = BufferPool::new(1024, 4096, "test");
        let map: I64Map<Vec<u8>> = new_i64_map();

        // Get buffer, use it, store in map
        let mut buf = pool.get();
        buf.extend_from_slice(b"test data");

        let mut map = map;
        map.insert(1, buf);

        assert!(map.contains_key(1));
        assert_eq!(map.get(1).unwrap(), b"test data");
    }

    #[test]
    fn test_version_constants() {
        // Ensure version is accessible
        let v = version();
        assert!(!v.is_empty());

        let info = version_info();
        assert!(info.contains("stoolap"));
    }
}
