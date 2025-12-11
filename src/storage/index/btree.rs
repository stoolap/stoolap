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

//! Generic B-tree implementation for Stoolap
//!

use std::cmp::Ordering;
use std::fmt::Debug;

/// Maximum number of keys per node
const MAX_KEYS: usize = 31;

/// Minimum number of keys per node (except root)
const MIN_KEYS: usize = MAX_KEYS / 2;

/// A B-tree node
#[derive(Debug, Clone)]
struct BTreeNode<K, V> {
    /// Keys stored in this node (sorted)
    keys: Vec<K>,
    /// Values corresponding to keys
    values: Vec<V>,
    /// Child pointers (len = keys.len() + 1 for internal nodes, 0 for leaves)
    children: Vec<BTreeNode<K, V>>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl<K: Ord + Clone, V: Clone> BTreeNode<K, V> {
    /// Create a new empty leaf node
    fn new_leaf() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: true,
        }
    }

    /// Create a new internal node
    fn new_internal() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: false,
        }
    }

    /// Find the position for a key using binary search
    /// Returns the index of the first key >= target
    fn find_position(&self, key: &K) -> usize {
        match self.keys.binary_search(key) {
            Ok(i) => i,
            Err(i) => i,
        }
    }

    /// Check if the node is full
    fn is_full(&self) -> bool {
        self.keys.len() >= MAX_KEYS
    }

    /// Check if the node has minimum keys
    fn has_min_keys(&self) -> bool {
        self.keys.len() <= MIN_KEYS
    }

    /// Search for a key in this subtree
    fn search(&self, key: &K) -> Option<&V> {
        let i = self.find_position(key);

        if i < self.keys.len() && self.keys[i].cmp(key) == Ordering::Equal {
            return Some(&self.values[i]);
        }

        if self.is_leaf {
            return None;
        }

        self.children[i].search(key)
    }

    /// Insert a key-value pair into this subtree
    /// Returns true if a new key was inserted
    fn insert(&mut self, key: K, value: V) -> bool {
        let i = self.find_position(&key);

        if i < self.keys.len() && self.keys[i].cmp(&key) == Ordering::Equal {
            self.values[i] = value;
            return false;
        }

        if self.is_leaf {
            self.keys.insert(i, key);
            self.values.insert(i, value);
            return true;
        }

        // Check if child is full
        if self.children[i].is_full() {
            self.split_child(i);
            // Decide which child to follow
            let cmp = key.cmp(&self.keys[i]);
            if cmp == Ordering::Equal {
                self.values[i] = value;
                return false;
            } else if cmp == Ordering::Greater {
                return self.children[i + 1].insert(key, value);
            }
        }
        self.children[i].insert(key, value)
    }

    /// Split a full child
    fn split_child(&mut self, child_index: usize) {
        let child = &mut self.children[child_index];
        let mid = child.keys.len() / 2;

        // Create the right node with the right half
        let right_node = BTreeNode {
            keys: child.keys.split_off(mid + 1),
            values: child.values.split_off(mid + 1),
            children: if child.is_leaf {
                Vec::new()
            } else {
                child.children.split_off(mid + 1)
            },
            is_leaf: child.is_leaf,
        };

        // Get the median key/value to promote
        let median_key = child.keys.pop().unwrap();
        let median_value = child.values.pop().unwrap();

        // Insert the median into this node
        self.keys.insert(child_index, median_key);
        self.values.insert(child_index, median_value);
        self.children.insert(child_index + 1, right_node);
    }

    /// Delete a key from this subtree
    /// Returns true if the key was deleted
    fn delete(&mut self, key: &K) -> bool {
        let i = self.find_position(key);
        let key_found = i < self.keys.len() && self.keys[i].cmp(key) == Ordering::Equal;

        if key_found {
            if self.is_leaf {
                self.keys.remove(i);
                self.values.remove(i);
                true
            } else {
                self.delete_from_internal(i)
            }
        } else if self.is_leaf {
            false
        } else {
            if self.children[i].has_min_keys() {
                self.ensure_child_has_enough_keys(i);
            }

            let child_idx = if i > self.keys.len() {
                self.keys.len()
            } else {
                i
            };
            self.children[child_idx].delete(key)
        }
    }

    /// Delete from an internal node
    fn delete_from_internal(&mut self, key_idx: usize) -> bool {
        // Try to get predecessor from left child
        if !self.children[key_idx].has_min_keys() {
            let (pred_key, pred_value) = self.children[key_idx].find_rightmost();
            let pred_key_clone = pred_key.clone();
            self.keys[key_idx] = pred_key;
            self.values[key_idx] = pred_value;
            return self.children[key_idx].delete(&pred_key_clone);
        }

        // Try to get successor from right child
        if !self.children[key_idx + 1].has_min_keys() {
            let (succ_key, succ_value) = self.children[key_idx + 1].find_leftmost();
            let succ_key_clone = succ_key.clone();
            self.keys[key_idx] = succ_key;
            self.values[key_idx] = succ_value;
            return self.children[key_idx + 1].delete(&succ_key_clone);
        }

        // Both children have minimum keys, merge them
        let merge_key = self.keys[key_idx].clone();
        self.merge_children(key_idx);
        self.children[key_idx].delete(&merge_key)
    }

    /// Find the rightmost key in this subtree
    fn find_rightmost(&self) -> (K, V) {
        if self.is_leaf {
            let last = self.keys.len() - 1;
            (self.keys[last].clone(), self.values[last].clone())
        } else {
            self.children[self.children.len() - 1].find_rightmost()
        }
    }

    /// Find the leftmost key in this subtree
    fn find_leftmost(&self) -> (K, V) {
        if self.is_leaf {
            (self.keys[0].clone(), self.values[0].clone())
        } else {
            self.children[0].find_leftmost()
        }
    }

    /// Ensure a child has enough keys for deletion
    fn ensure_child_has_enough_keys(&mut self, child_idx: usize) {
        // Try to borrow from left sibling
        if child_idx > 0 && !self.children[child_idx - 1].has_min_keys() {
            self.borrow_from_left(child_idx);
            return;
        }

        // Try to borrow from right sibling
        if child_idx < self.children.len() - 1 && !self.children[child_idx + 1].has_min_keys() {
            self.borrow_from_right(child_idx);
            return;
        }

        // Merge with a sibling
        if child_idx > 0 {
            self.merge_children(child_idx - 1);
        } else {
            self.merge_children(child_idx);
        }
    }

    /// Borrow a key from the left sibling
    fn borrow_from_left(&mut self, child_idx: usize) {
        let key_from_parent = self.keys[child_idx - 1].clone();
        let value_from_parent = self.values[child_idx - 1].clone();

        let left_sibling = &mut self.children[child_idx - 1];
        let key_from_left = left_sibling.keys.pop().unwrap();
        let value_from_left = left_sibling.values.pop().unwrap();
        let child_from_left = if !left_sibling.is_leaf {
            Some(left_sibling.children.pop().unwrap())
        } else {
            None
        };

        self.keys[child_idx - 1] = key_from_left;
        self.values[child_idx - 1] = value_from_left;

        let child = &mut self.children[child_idx];
        child.keys.insert(0, key_from_parent);
        child.values.insert(0, value_from_parent);
        if let Some(c) = child_from_left {
            child.children.insert(0, c);
        }
    }

    /// Borrow a key from the right sibling
    fn borrow_from_right(&mut self, child_idx: usize) {
        let key_from_parent = self.keys[child_idx].clone();
        let value_from_parent = self.values[child_idx].clone();

        let right_sibling = &mut self.children[child_idx + 1];
        let key_from_right = right_sibling.keys.remove(0);
        let value_from_right = right_sibling.values.remove(0);
        let child_from_right = if !right_sibling.is_leaf {
            Some(right_sibling.children.remove(0))
        } else {
            None
        };

        self.keys[child_idx] = key_from_right;
        self.values[child_idx] = value_from_right;

        let child = &mut self.children[child_idx];
        child.keys.push(key_from_parent);
        child.values.push(value_from_parent);
        if let Some(c) = child_from_right {
            child.children.push(c);
        }
    }

    /// Merge two child nodes
    fn merge_children(&mut self, key_idx: usize) {
        let key_from_parent = self.keys.remove(key_idx);
        let value_from_parent = self.values.remove(key_idx);
        let mut right_child = self.children.remove(key_idx + 1);

        let left_child = &mut self.children[key_idx];
        left_child.keys.push(key_from_parent);
        left_child.values.push(value_from_parent);
        left_child.keys.append(&mut right_child.keys);
        left_child.values.append(&mut right_child.values);
        left_child.children.append(&mut right_child.children);
    }

    /// Iterate over all key-value pairs in order
    fn for_each<F>(&self, callback: &mut F) -> bool
    where
        F: FnMut(&K, &V) -> bool,
    {
        for i in 0..self.keys.len() {
            if !self.is_leaf && i < self.children.len() && !self.children[i].for_each(callback) {
                return false;
            }

            if !callback(&self.keys[i], &self.values[i]) {
                return false;
            }
        }

        if !self.is_leaf
            && self.children.len() > self.keys.len()
            && !self.children[self.keys.len()].for_each(callback)
        {
            return false;
        }

        true
    }
}

/// A generic B-tree implementation
///

#[derive(Debug)]
pub struct BTree<K, V> {
    /// Root node
    root: Option<Box<BTreeNode<K, V>>>,
    /// Number of key-value pairs
    size: usize,
}

impl<K: Ord + Clone + Debug, V: Clone + Debug> BTree<K, V> {
    /// Create a new empty B-tree
    pub fn new() -> Self {
        Self {
            root: Some(Box::new(BTreeNode::new_leaf())),
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
    pub fn search(&self, key: &K) -> Option<&V> {
        self.root.as_ref().and_then(|root| root.search(key))
    }

    /// Insert a key-value pair into the tree
    ///
    /// If the key already exists, the value is updated.
    /// Returns true if a new key was inserted, false if an existing key was updated.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        // Handle root split if needed
        if let Some(ref root) = self.root {
            if root.is_full() {
                let mut new_root = Box::new(BTreeNode::new_internal());
                let old_root = *self.root.take().unwrap();
                new_root.children.push(old_root);
                new_root.split_child(0);
                self.root = Some(new_root);
            }
        }

        // Insert into the non-full root
        if let Some(ref mut root) = self.root {
            let inserted = root.insert(key, value);
            if inserted {
                self.size += 1;
            }
            inserted
        } else {
            false
        }
    }

    /// Delete a key from the tree
    ///
    /// Returns true if the key was found and deleted, false otherwise.
    pub fn delete(&mut self, key: &K) -> bool {
        let deleted = if let Some(ref mut root) = self.root {
            root.delete(key)
        } else {
            false
        };

        if deleted {
            self.size -= 1;

            // If root is empty and has a child, make it the new root
            if let Some(ref root) = self.root {
                if root.keys.is_empty() && !root.is_leaf {
                    let mut old_root = self.root.take().unwrap();
                    self.root = Some(Box::new(old_root.children.remove(0)));
                }
            }
        }

        deleted
    }

    /// Iterate over all key-value pairs in order
    pub fn for_each<F>(&self, mut callback: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        if let Some(ref root) = self.root {
            root.for_each(&mut callback);
        }
    }

    /// Get an iterator over all key-value pairs
    pub fn iter(&self) -> BTreeIterator<'_, K, V> {
        BTreeIterator::new(self)
    }

    /// Seek to the first key >= target
    pub fn seek_ge(&self, target: &K) -> BTreeIterator<'_, K, V> {
        BTreeIterator::seek_ge(self, target)
    }
}

impl<K: Ord + Clone + Debug, V: Clone + Debug> Default for BTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over a B-tree
pub struct BTreeIterator<'a, K, V> {
    /// Stack of (node, index) pairs for traversal
    stack: Vec<(&'a BTreeNode<K, V>, usize)>,
}

impl<'a, K: Ord + Clone + Debug, V: Clone + Debug> BTreeIterator<'a, K, V> {
    /// Create a new iterator positioned at the first element
    fn new(tree: &'a BTree<K, V>) -> Self {
        let mut iter = Self { stack: Vec::new() };
        if let Some(ref root) = tree.root {
            iter.push_left_edge(root);
        }
        iter
    }

    /// Create an iterator positioned at the first key >= target
    fn seek_ge(tree: &'a BTree<K, V>, target: &K) -> Self {
        let mut iter = Self { stack: Vec::new() };
        if let Some(ref root) = tree.root {
            iter.seek_ge_node(root, target);
        }
        iter
    }

    /// Push all left-edge nodes onto the stack
    fn push_left_edge(&mut self, mut node: &'a BTreeNode<K, V>) {
        loop {
            self.stack.push((node, 0));
            if node.is_leaf || node.children.is_empty() {
                break;
            }
            node = &node.children[0];
        }
    }

    /// Seek to first key >= target
    fn seek_ge_node(&mut self, node: &'a BTreeNode<K, V>, target: &K) {
        let i = node.find_position(target);

        self.stack.push((node, i));

        if !node.is_leaf && i < node.children.len() {
            if i < node.keys.len() && node.keys[i].cmp(target) == Ordering::Equal {
                return;
            }
            self.seek_ge_node(&node.children[i], target);
        }
    }
}

impl<'a, K: Ord + Clone + Debug, V: Clone + Debug> Iterator for BTreeIterator<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, idx)) = self.stack.pop() {
            if idx < node.keys.len() {
                let result = (&node.keys[idx], &node.values[idx]);

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
        let tree: BTree<i32, i32> = BTree::new();
        assert_eq!(tree.size(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let mut tree = BTree::new();

        tree.insert(5, "five");
        tree.insert(3, "three");
        tree.insert(7, "seven");
        tree.insert(1, "one");
        tree.insert(9, "nine");

        assert_eq!(tree.size(), 5);
        assert_eq!(tree.search(&5), Some(&"five"));
        assert_eq!(tree.search(&3), Some(&"three"));
        assert_eq!(tree.search(&7), Some(&"seven"));
        assert_eq!(tree.search(&1), Some(&"one"));
        assert_eq!(tree.search(&9), Some(&"nine"));
        assert_eq!(tree.search(&0), None);
        assert_eq!(tree.search(&10), None);
    }

    #[test]
    fn test_update_existing() {
        let mut tree = BTree::new();

        assert!(tree.insert(5, "five"));
        assert!(!tree.insert(5, "FIVE"));

        assert_eq!(tree.size(), 1);
        assert_eq!(tree.search(&5), Some(&"FIVE"));
    }

    #[test]
    fn test_delete() {
        let mut tree = BTree::new();

        for i in 1..=10 {
            tree.insert(i, i * 10);
        }

        assert_eq!(tree.size(), 10);

        assert!(tree.delete(&5));
        assert_eq!(tree.size(), 9);
        assert_eq!(tree.search(&5), None);

        assert!(tree.delete(&1));
        assert!(tree.delete(&10));
        assert_eq!(tree.size(), 7);

        assert!(!tree.delete(&100));
        assert_eq!(tree.size(), 7);

        assert_eq!(tree.search(&3), Some(&30));
        assert_eq!(tree.search(&7), Some(&70));
    }

    #[test]
    fn test_many_insertions() {
        let mut tree = BTree::new();

        for i in 0..1000 {
            tree.insert(i, i * 2);
        }

        assert_eq!(tree.size(), 1000);

        for i in 0..1000 {
            assert_eq!(tree.search(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_for_each() {
        let mut tree = BTree::new();

        tree.insert(5, 50);
        tree.insert(3, 30);
        tree.insert(7, 70);
        tree.insert(1, 10);
        tree.insert(9, 90);

        let mut keys = Vec::new();
        tree.for_each(|k, _v| {
            keys.push(*k);
            true
        });

        assert_eq!(keys, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn test_for_each_early_exit() {
        let mut tree = BTree::new();

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
        let mut tree = BTree::new();

        tree.insert(5, "e");
        tree.insert(3, "c");
        tree.insert(7, "g");
        tree.insert(1, "a");
        tree.insert(9, "i");

        let pairs: Vec<_> = tree.iter().collect();
        assert_eq!(pairs.len(), 5);
        assert_eq!(pairs[0], (&1, &"a"));
        assert_eq!(pairs[1], (&3, &"c"));
        assert_eq!(pairs[2], (&5, &"e"));
        assert_eq!(pairs[3], (&7, &"g"));
        assert_eq!(pairs[4], (&9, &"i"));
    }

    #[test]
    fn test_seek_ge() {
        let mut tree = BTree::new();

        for i in (0..100).step_by(10) {
            tree.insert(i, i);
        }

        let pairs: Vec<_> = tree.seek_ge(&50).collect();
        assert!(pairs.len() >= 5);
        assert_eq!(pairs[0], (&50, &50));

        let pairs: Vec<_> = tree.seek_ge(&55).collect();
        assert!(pairs.len() >= 4);
        assert_eq!(pairs[0], (&60, &60));

        let pairs: Vec<_> = tree.seek_ge(&1000).collect();
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_string_keys() {
        let mut tree = BTree::new();

        tree.insert("banana".to_string(), 2);
        tree.insert("apple".to_string(), 1);
        tree.insert("cherry".to_string(), 3);

        assert_eq!(tree.search(&"apple".to_string()), Some(&1));
        assert_eq!(tree.search(&"banana".to_string()), Some(&2));
        assert_eq!(tree.search(&"cherry".to_string()), Some(&3));
        assert_eq!(tree.search(&"date".to_string()), None);

        let keys: Vec<_> = tree.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_reverse_insertion_order() {
        let mut tree = BTree::new();

        for i in (0..100).rev() {
            tree.insert(i, i);
        }

        assert_eq!(tree.size(), 100);

        let mut prev = -1i32;
        tree.for_each(|k, _| {
            assert!(*k > prev);
            prev = *k;
            true
        });
    }

    #[test]
    fn test_delete_all() {
        let mut tree = BTree::new();

        for i in 0..50 {
            tree.insert(i, i);
        }

        for i in 0..50 {
            assert!(tree.delete(&i));
        }

        assert_eq!(tree.size(), 0);
        assert!(tree.is_empty());
    }
}
