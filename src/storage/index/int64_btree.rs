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

//! Optimized B-tree for int64 keys
//!
//!
//! Key optimizations:
//! - Fixed-size arrays for cache-friendly memory layout
//! - Specialized int64 comparison (no trait overhead)
//! - Linear search for small nodes, binary search for larger ones
//! - Efficient range search

use std::fmt::Debug;

/// B-tree degree (max children per node)
const BTREE_DEGREE: usize = 32;

/// Maximum keys per node
const BTREE_MAX_KEYS: usize = BTREE_DEGREE - 1;

/// Minimum keys per node (except root)
const BTREE_MIN_KEYS: usize = BTREE_MAX_KEYS / 2;

/// Array size with room for temporary overflow
const BTREE_ARRAY_SIZE: usize = BTREE_MAX_KEYS + 1;

/// A node in the Int64BTree
///
/// Uses fixed-size arrays for cache-friendly memory layout.
#[derive(Clone)]
struct BTreeNode<V: Clone> {
    /// Keys stored in this node (sorted)
    keys: [i64; BTREE_ARRAY_SIZE],
    /// Values corresponding to keys
    values: Vec<V>,
    /// Child pointers
    children: Vec<BTreeNode<V>>,
    /// Number of keys actually stored
    key_count: usize,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl<V: Clone + Debug> Debug for BTreeNode<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BTreeNode")
            .field("keys", &&self.keys[..self.key_count])
            .field("values", &self.values)
            .field("key_count", &self.key_count)
            .field("is_leaf", &self.is_leaf)
            .field("children_count", &self.children.len())
            .finish()
    }
}

impl<V: Clone> BTreeNode<V> {
    /// Create a new leaf node
    fn new_leaf() -> Self {
        Self {
            keys: [0; BTREE_ARRAY_SIZE],
            values: Vec::with_capacity(BTREE_ARRAY_SIZE),
            children: Vec::new(),
            key_count: 0,
            is_leaf: true,
        }
    }

    /// Create a new internal node
    fn new_internal() -> Self {
        Self {
            keys: [0; BTREE_ARRAY_SIZE],
            values: Vec::with_capacity(BTREE_ARRAY_SIZE),
            children: Vec::with_capacity(BTREE_ARRAY_SIZE + 1),
            key_count: 0,
            is_leaf: false,
        }
    }

    /// Binary search for key position
    /// Returns the index of the first key >= target
    #[inline]
    fn binary_search(&self, key: i64) -> usize {
        // Fast path for small nodes - linear search
        if self.key_count <= 8 {
            for i in 0..self.key_count {
                if self.keys[i] >= key {
                    return i;
                }
            }
            return self.key_count;
        }

        // Binary search for larger nodes
        let mut left = 0;
        let mut right = self.key_count;
        while left < right {
            let mid = left + (right - left) / 2;
            if self.keys[mid] < key {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left
    }

    /// Check if node is full
    #[inline]
    fn is_full(&self) -> bool {
        self.key_count >= BTREE_MAX_KEYS
    }

    /// Check if node has minimum keys
    #[inline]
    fn has_min_keys(&self) -> bool {
        self.key_count <= BTREE_MIN_KEYS
    }

    /// Search for a key in this subtree
    fn search(&self, key: i64) -> Option<&V> {
        let i = self.binary_search(key);

        if i < self.key_count && self.keys[i] == key {
            return Some(&self.values[i]);
        }

        if self.is_leaf {
            return None;
        }

        self.children[i].search(key)
    }

    /// Insert a key-value pair into this subtree
    /// Returns true if a new key was inserted
    fn insert(&mut self, key: i64, value: V) -> bool {
        let i = self.binary_search(key);

        if i < self.key_count && self.keys[i] == key {
            self.values[i] = value;
            return false;
        }

        if self.is_leaf {
            // Shift elements to make room
            for j in (i..self.key_count).rev() {
                self.keys[j + 1] = self.keys[j];
            }
            self.values.insert(i, value);
            self.keys[i] = key;
            self.key_count += 1;
            return true;
        }

        // Check if child is full
        if self.children[i].is_full() {
            self.split_child(i);
            if key == self.keys[i] {
                self.values[i] = value;
                return false;
            } else if key > self.keys[i] {
                return self.children[i + 1].insert(key, value);
            }
        }
        self.children[i].insert(key, value)
    }

    /// Split a full child
    fn split_child(&mut self, child_index: usize) {
        let child = &mut self.children[child_index];

        // Create new node for right half
        let mut new_child = BTreeNode {
            keys: [0; BTREE_ARRAY_SIZE],
            values: Vec::with_capacity(BTREE_ARRAY_SIZE),
            children: Vec::new(),
            key_count: BTREE_MIN_KEYS,
            is_leaf: child.is_leaf,
        };

        // Copy right half of keys to new child
        for j in 0..BTREE_MIN_KEYS {
            new_child.keys[j] = child.keys[j + BTREE_MIN_KEYS + 1];
        }

        // Copy right half of values
        new_child.values = child.values.split_off(BTREE_MIN_KEYS + 1);

        // If not leaf, copy right half of children
        if !child.is_leaf {
            new_child.children = child.children.split_off(BTREE_MIN_KEYS + 1);
        }

        // Get median key and value
        let median_key = child.keys[BTREE_MIN_KEYS];
        let median_value = child.values.pop().unwrap();
        child.key_count = BTREE_MIN_KEYS;

        // Shift parent's children to make room
        self.children.insert(child_index + 1, new_child);

        // Shift parent's keys to make room
        for j in (child_index..self.key_count).rev() {
            self.keys[j + 1] = self.keys[j];
        }

        // Insert median into parent
        self.keys[child_index] = median_key;
        self.values.insert(child_index, median_value);
        self.key_count += 1;
    }

    /// Delete a key from this subtree
    /// Returns true if the key was deleted
    fn delete(&mut self, key: i64) -> bool {
        let i = self.binary_search(key);
        let key_found = i < self.key_count && self.keys[i] == key;

        if key_found {
            if self.is_leaf {
                self.remove_from_leaf(i);
                true
            } else {
                self.remove_from_non_leaf(i)
            }
        } else if self.is_leaf {
            false
        } else {
            if self.children[i].has_min_keys() {
                self.fill_child(i);
            }

            let child_idx = if i > self.key_count {
                self.key_count
            } else {
                i
            };
            self.children[child_idx].delete(key)
        }
    }

    /// Remove key from leaf node
    fn remove_from_leaf(&mut self, index: usize) {
        for j in index + 1..self.key_count {
            self.keys[j - 1] = self.keys[j];
        }
        self.values.remove(index);
        self.key_count -= 1;
    }

    /// Remove key from internal node
    fn remove_from_non_leaf(&mut self, index: usize) -> bool {
        let key = self.keys[index];

        // Try predecessor from left child
        if !self.children[index].has_min_keys() {
            let (pred_key, pred_value) = self.children[index].get_last_key();
            self.keys[index] = pred_key;
            self.values[index] = pred_value;
            return self.children[index].delete(pred_key);
        }

        // Try successor from right child
        if !self.children[index + 1].has_min_keys() {
            let (succ_key, succ_value) = self.children[index + 1].get_first_key();
            self.keys[index] = succ_key;
            self.values[index] = succ_value;
            return self.children[index + 1].delete(succ_key);
        }

        // Merge children
        self.merge_children(index);
        self.children[index].delete(key)
    }

    /// Get the last key in this subtree
    fn get_last_key(&self) -> (i64, V) {
        if self.is_leaf {
            let last = self.key_count - 1;
            (self.keys[last], self.values[last].clone())
        } else {
            self.children[self.children.len() - 1].get_last_key()
        }
    }

    /// Get the first key in this subtree
    fn get_first_key(&self) -> (i64, V) {
        if self.is_leaf {
            (self.keys[0], self.values[0].clone())
        } else {
            self.children[0].get_first_key()
        }
    }

    /// Ensure child has enough keys
    fn fill_child(&mut self, index: usize) {
        // Try borrowing from left sibling
        if index > 0 && !self.children[index - 1].has_min_keys() {
            self.borrow_from_prev(index);
            return;
        }

        // Try borrowing from right sibling
        if index < self.key_count && !self.children[index + 1].has_min_keys() {
            self.borrow_from_next(index);
            return;
        }

        // Merge with sibling
        if index < self.key_count {
            self.merge_children(index);
        } else {
            self.merge_children(index - 1);
        }
    }

    /// Borrow from left sibling
    fn borrow_from_prev(&mut self, index: usize) {
        let parent_key = self.keys[index - 1];
        let parent_value = self.values[index - 1].clone();

        let left = &mut self.children[index - 1];
        let last_key = left.keys[left.key_count - 1];
        let last_value = left.values.pop().unwrap();
        let last_child = if !left.is_leaf {
            Some(left.children.pop().unwrap())
        } else {
            None
        };
        left.key_count -= 1;

        self.keys[index - 1] = last_key;
        self.values[index - 1] = last_value;

        let child = &mut self.children[index];
        for j in (0..child.key_count).rev() {
            child.keys[j + 1] = child.keys[j];
        }
        child.keys[0] = parent_key;
        child.values.insert(0, parent_value);
        if let Some(c) = last_child {
            child.children.insert(0, c);
        }
        child.key_count += 1;
    }

    /// Borrow from right sibling
    fn borrow_from_next(&mut self, index: usize) {
        let parent_key = self.keys[index];
        let parent_value = self.values[index].clone();

        let right = &mut self.children[index + 1];
        let first_key = right.keys[0];
        let first_value = right.values.remove(0);
        let first_child = if !right.is_leaf {
            Some(right.children.remove(0))
        } else {
            None
        };

        for j in 1..right.key_count {
            right.keys[j - 1] = right.keys[j];
        }
        right.key_count -= 1;

        self.keys[index] = first_key;
        self.values[index] = first_value;

        let child = &mut self.children[index];
        child.keys[child.key_count] = parent_key;
        child.values.push(parent_value);
        if let Some(c) = first_child {
            child.children.push(c);
        }
        child.key_count += 1;
    }

    /// Merge two children
    fn merge_children(&mut self, index: usize) {
        let right = self.children.remove(index + 1);
        let parent_key = self.keys[index];
        let parent_value = self.values.remove(index);

        for j in index + 1..self.key_count {
            self.keys[j - 1] = self.keys[j];
        }
        self.key_count -= 1;

        let left = &mut self.children[index];
        left.keys[left.key_count] = parent_key;
        left.values.push(parent_value);
        left.key_count += 1;

        for j in 0..right.key_count {
            left.keys[left.key_count] = right.keys[j];
            left.key_count += 1;
        }
        left.values.extend(right.values);
        left.children.extend(right.children);
    }

    /// Range search in this subtree
    fn range_search<'a>(&'a self, start: i64, end: i64, results: &mut Vec<&'a V>) {
        if self.is_leaf {
            for i in 0..self.key_count {
                if self.keys[i] >= start && self.keys[i] <= end {
                    results.push(&self.values[i]);
                }
            }
            return;
        }

        // Find first position >= start
        let mut start_pos = 0;
        while start_pos < self.key_count && self.keys[start_pos] < start {
            start_pos += 1;
        }

        // Search left child that might contain start
        self.children[start_pos].range_search(start, end, results);

        // Process keys and their right children
        for i in start_pos..self.key_count {
            if self.keys[i] > end {
                break;
            }

            if self.keys[i] >= start {
                results.push(&self.values[i]);
            }

            self.children[i + 1].range_search(start, end, results);
        }
    }

    /// Iterate over all key-value pairs in order
    fn for_each<F>(&self, callback: &mut F) -> bool
    where
        F: FnMut(i64, &V) -> bool,
    {
        for i in 0..self.key_count {
            if !self.is_leaf && !self.children[i].for_each(callback) {
                return false;
            }

            if !callback(self.keys[i], &self.values[i]) {
                return false;
            }
        }

        if !self.is_leaf
            && self.children.len() > self.key_count
            && !self.children[self.key_count].for_each(callback)
        {
            return false;
        }

        true
    }
}

/// An optimized B-tree for int64 keys
pub struct Int64BTree<V: Clone> {
    /// Root node
    root: Box<BTreeNode<V>>,
    /// Number of key-value pairs
    size: usize,
}

impl<V: Clone + Debug> Debug for Int64BTree<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Int64BTree")
            .field("size", &self.size)
            .field("root", &self.root)
            .finish()
    }
}

impl<V: Clone + Debug> Int64BTree<V> {
    /// Create a new empty Int64BTree
    pub fn new() -> Self {
        Self {
            root: Box::new(BTreeNode::new_leaf()),
            size: 0,
        }
    }

    /// Get the number of key-value pairs in the tree
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Search for a key in the tree
    ///
    /// Returns a reference to the value if found, None otherwise.
    pub fn search(&self, key: i64) -> Option<&V> {
        self.root.search(key)
    }

    /// Insert a key-value pair
    ///
    /// Returns true if a new key was inserted, false if updated.
    pub fn insert(&mut self, key: i64, value: V) -> bool {
        // Handle root overflow
        if self.root.is_full() {
            let mut new_root = Box::new(BTreeNode::new_internal());
            let old_root = std::mem::replace(&mut self.root, Box::new(BTreeNode::new_leaf()));
            new_root.children.push(*old_root);
            new_root.split_child(0);
            self.root = new_root;
        }

        let inserted = self.root.insert(key, value);
        if inserted {
            self.size += 1;
        }
        inserted
    }

    /// Delete a key from the tree
    ///
    /// Returns true if the key was found and deleted.
    pub fn delete(&mut self, key: i64) -> bool {
        let deleted = self.root.delete(key);
        if deleted {
            self.size -= 1;
            // If root is empty and has a child, make it the new root
            if self.root.key_count == 0 && !self.root.is_leaf {
                *self.root = self.root.children.remove(0);
            }
        }
        deleted
    }

    /// Range search: find all values with keys in [start, end]
    pub fn range_search(&self, start: i64, end: i64) -> Vec<&V> {
        if start > end {
            return Vec::new();
        }

        let mut results = Vec::new();
        self.root.range_search(start, end, &mut results);
        results
    }

    /// Iterate over all key-value pairs in order
    pub fn for_each<F>(&self, mut callback: F)
    where
        F: FnMut(i64, &V) -> bool,
    {
        self.root.for_each(&mut callback);
    }

    /// Batch insert multiple key-value pairs
    ///
    /// Keys are sorted before insertion for better performance.
    pub fn batch_insert(&mut self, mut pairs: Vec<(i64, V)>) {
        if pairs.is_empty() {
            return;
        }

        // Sort by key
        pairs.sort_by_key(|(k, _)| *k);

        // Insert in batches
        for (key, value) in pairs {
            self.insert(key, value);
        }
    }

    /// Get an iterator over all key-value pairs
    pub fn iter(&self) -> Int64BTreeIterator<'_, V> {
        Int64BTreeIterator::new(self)
    }
}

impl<V: Clone + Debug> Default for Int64BTree<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over Int64BTree
pub struct Int64BTreeIterator<'a, V: Clone> {
    /// Stack of (node, index) pairs
    stack: Vec<(&'a BTreeNode<V>, usize)>,
}

impl<'a, V: Clone + Debug> Int64BTreeIterator<'a, V> {
    /// Create a new iterator
    fn new(tree: &'a Int64BTree<V>) -> Self {
        let mut iter = Self { stack: Vec::new() };
        iter.push_left_edge(&tree.root);
        iter
    }

    /// Push all left-edge nodes onto the stack
    fn push_left_edge(&mut self, mut node: &'a BTreeNode<V>) {
        loop {
            self.stack.push((node, 0));
            if node.is_leaf || node.children.is_empty() {
                break;
            }
            node = &node.children[0];
        }
    }
}

impl<'a, V: Clone + Debug> Iterator for Int64BTreeIterator<'a, V> {
    type Item = (i64, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, idx)) = self.stack.pop() {
            if idx < node.key_count {
                let result = (node.keys[idx], &node.values[idx]);

                self.stack.push((node, idx + 1));

                if !node.is_leaf && idx + 1 < node.children.len() {
                    self.push_left_edge(&node.children[idx + 1]);
                }

                return Some(result);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree() {
        let tree: Int64BTree<i32> = Int64BTree::new();
        assert_eq!(tree.size(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let mut tree = Int64BTree::new();

        tree.insert(5, "five");
        tree.insert(3, "three");
        tree.insert(7, "seven");
        tree.insert(1, "one");
        tree.insert(9, "nine");

        assert_eq!(tree.size(), 5);
        assert_eq!(tree.search(5), Some(&"five"));
        assert_eq!(tree.search(3), Some(&"three"));
        assert_eq!(tree.search(7), Some(&"seven"));
        assert_eq!(tree.search(1), Some(&"one"));
        assert_eq!(tree.search(9), Some(&"nine"));
        assert_eq!(tree.search(0), None);
        assert_eq!(tree.search(10), None);
    }

    #[test]
    fn test_update_existing() {
        let mut tree = Int64BTree::new();

        assert!(tree.insert(5, "five"));
        assert!(!tree.insert(5, "FIVE"));

        assert_eq!(tree.size(), 1);
        assert_eq!(tree.search(5), Some(&"FIVE"));
    }

    #[test]
    fn test_delete() {
        let mut tree = Int64BTree::new();

        for i in 1..=10 {
            tree.insert(i, i * 10);
        }

        assert_eq!(tree.size(), 10);

        assert!(tree.delete(5));
        assert_eq!(tree.size(), 9);
        assert_eq!(tree.search(5), None);

        assert!(tree.delete(1));
        assert!(tree.delete(10));
        assert_eq!(tree.size(), 7);

        assert!(!tree.delete(100));
        assert_eq!(tree.size(), 7);

        assert_eq!(tree.search(3), Some(&30));
        assert_eq!(tree.search(7), Some(&70));
    }

    #[test]
    fn test_many_insertions() {
        let mut tree = Int64BTree::new();

        for i in 0..1000 {
            tree.insert(i, i * 2);
        }

        assert_eq!(tree.size(), 1000);

        for i in 0..1000 {
            assert_eq!(tree.search(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_range_search() {
        let mut tree = Int64BTree::new();

        for i in 0..100 {
            tree.insert(i, i);
        }

        let results = tree.range_search(25, 75);
        assert_eq!(results.len(), 51); // 25 to 75 inclusive

        // Verify all values are in range
        for v in results {
            assert!(*v >= 25 && *v <= 75);
        }
    }

    #[test]
    fn test_range_search_empty() {
        let mut tree = Int64BTree::new();

        for i in 0..100 {
            tree.insert(i, i);
        }

        // Range with no matches
        let results = tree.range_search(200, 300);
        assert!(results.is_empty());

        // Invalid range
        let results = tree.range_search(50, 25);
        assert!(results.is_empty());
    }

    #[test]
    fn test_for_each() {
        let mut tree = Int64BTree::new();

        tree.insert(5, 50);
        tree.insert(3, 30);
        tree.insert(7, 70);
        tree.insert(1, 10);
        tree.insert(9, 90);

        let mut keys = Vec::new();
        tree.for_each(|k, _v| {
            keys.push(k);
            true
        });

        assert_eq!(keys, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn test_for_each_early_exit() {
        let mut tree = Int64BTree::new();

        for i in 1..=10 {
            tree.insert(i, i);
        }

        let mut count = 0;
        tree.for_each(|_k, _v| {
            count += 1;
            count < 5
        });

        assert_eq!(count, 5);
    }

    #[test]
    fn test_iterator() {
        let mut tree = Int64BTree::new();

        tree.insert(5, "e");
        tree.insert(3, "c");
        tree.insert(7, "g");
        tree.insert(1, "a");
        tree.insert(9, "i");

        let pairs: Vec<_> = tree.iter().collect();
        assert_eq!(pairs.len(), 5);
        assert_eq!(pairs[0], (1, &"a"));
        assert_eq!(pairs[1], (3, &"c"));
        assert_eq!(pairs[2], (5, &"e"));
        assert_eq!(pairs[3], (7, &"g"));
        assert_eq!(pairs[4], (9, &"i"));
    }

    #[test]
    fn test_batch_insert() {
        let mut tree = Int64BTree::new();

        let pairs: Vec<_> = (0..100).map(|i| (i, i * 2)).collect();
        tree.batch_insert(pairs);

        assert_eq!(tree.size(), 100);

        for i in 0..100 {
            assert_eq!(tree.search(i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_reverse_insertion_order() {
        let mut tree = Int64BTree::new();

        for i in (0..100).rev() {
            tree.insert(i, i);
        }

        assert_eq!(tree.size(), 100);

        let mut prev = -1i64;
        tree.for_each(|k, _| {
            assert!(k > prev);
            prev = k;
            true
        });
    }

    #[test]
    fn test_delete_all() {
        let mut tree = Int64BTree::new();

        for i in 0..50 {
            tree.insert(i, i);
        }

        for i in 0..50 {
            assert!(tree.delete(i));
        }

        assert_eq!(tree.size(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_negative_keys() {
        let mut tree = Int64BTree::new();

        tree.insert(-10, "neg10");
        tree.insert(-5, "neg5");
        tree.insert(0, "zero");
        tree.insert(5, "pos5");
        tree.insert(10, "pos10");

        assert_eq!(tree.search(-10), Some(&"neg10"));
        assert_eq!(tree.search(-5), Some(&"neg5"));
        assert_eq!(tree.search(0), Some(&"zero"));
        assert_eq!(tree.search(5), Some(&"pos5"));
        assert_eq!(tree.search(10), Some(&"pos10"));

        let results = tree.range_search(-7, 7);
        assert_eq!(results.len(), 3); // -5, 0, 5
    }

    #[test]
    fn test_node_splits() {
        let mut tree = Int64BTree::new();

        // Insert enough to trigger multiple splits
        for i in 0..200 {
            tree.insert(i, i);
        }

        assert_eq!(tree.size(), 200);

        // Verify all searchable
        for i in 0..200 {
            assert_eq!(tree.search(i), Some(&i), "Failed to find key {}", i);
        }
    }
}
