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

//! Index implementations for Stoolap
//!
//! This module provides B-tree based index structures for efficient
//! key-value lookups and range scans.
//!
//! # Index Types
//!
//! - [`BTree`] - Generic B-tree for any comparable key type
//! - [`Int64BTree`] - Optimized B-tree for int64 keys

pub mod btree;
pub mod int64_btree;

// Re-export main types
pub use btree::BTree;
pub use int64_btree::Int64BTree;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_basic() {
        let mut tree: BTree<i64, String> = BTree::new();

        tree.insert(5, "five".to_string());
        tree.insert(3, "three".to_string());
        tree.insert(7, "seven".to_string());

        assert_eq!(tree.size(), 3);
        assert_eq!(tree.search(&5), Some(&"five".to_string()));
        assert_eq!(tree.search(&3), Some(&"three".to_string()));
        assert_eq!(tree.search(&7), Some(&"seven".to_string()));
        assert_eq!(tree.search(&1), None);
    }

    #[test]
    fn test_int64_btree_basic() {
        let mut tree: Int64BTree<String> = Int64BTree::new();

        tree.insert(5, "five".to_string());
        tree.insert(3, "three".to_string());
        tree.insert(7, "seven".to_string());

        assert_eq!(tree.size(), 3);
        assert_eq!(tree.search(5), Some(&"five".to_string()));
        assert_eq!(tree.search(3), Some(&"three".to_string()));
        assert_eq!(tree.search(7), Some(&"seven".to_string()));
        assert_eq!(tree.search(1), None);
    }
}
