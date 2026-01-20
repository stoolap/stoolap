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

//! Copy-on-Write B-tree for i64 keys
//!
//! Design:
//! - Each node is wrapped in CompactArc (8 bytes) for reference counting
//! - Readers clone the root Arc (atomic, ~1ns) and traverse without locks
//! - Writers use CompactArc::make_mut() which clones only if shared
//! - Structural sharing: unmodified subtrees are shared between versions
//!
//! Performance characteristics:
//! - Reads: O(log n), completely lock-free
//! - Writes: O(log n) + O(B) per modified node for COW
//! - Clone: O(1) - just increments root's reference count
//! - Memory: Shared nodes between snapshots

use super::{CompactArc, CompactVec};
use std::ops::Bound;

/// Maximum keys per node. Smaller = more nodes but faster COW.
/// 64 gives good balance for database workloads.
const MAX_KEYS: usize = 128;

/// Minimum keys per node (except root).
const MIN_KEYS: usize = MAX_KEYS / 2;

/// A B+ tree node (values only in leaves)
struct Node<V: Clone> {
    /// Keys stored in this node
    keys: CompactVec<i64>,
    /// Values (only for leaf nodes, same length as keys)
    values: CompactVec<V>,
    /// Child pointers (only for internal nodes, len = keys.len() + 1)
    children: CompactVec<CompactArc<Node<V>>>,
}

impl<V: Clone> Node<V> {
    fn new_leaf() -> Self {
        Self {
            keys: CompactVec::with_capacity(8),
            values: CompactVec::with_capacity(8),
            children: CompactVec::new(),
        }
    }

    fn new_internal() -> Self {
        Self {
            keys: CompactVec::with_capacity(8),
            values: CompactVec::new(),
            children: CompactVec::with_capacity(8),
        }
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Binary search for key position
    #[inline]
    fn search(&self, key: i64) -> Result<usize, usize> {
        self.keys.binary_search(&key)
    }
}

impl<V: Clone> Clone for Node<V> {
    fn clone(&self) -> Self {
        // Optimization: reserve MAX_KEYS + 1 capacity to avoid immediate reallocation upon insert.
        // COW operations typically pattern: clone node -> insert key.
        // If we only allocate exactly len(), the subsequent insert triggers a grow() + realloc.
        // By reserving full capacity (plus 1 for overflow before split), we avoid this second allocation.

        let mut keys = CompactVec::with_capacity(MAX_KEYS + 1);
        keys.extend_copy(&self.keys);

        let mut values = CompactVec::new();
        if !self.values.is_empty() || self.is_leaf() {
            // Leaves need value capacity. Internal nodes don't use values.
            values.reserve(MAX_KEYS + 1);
            values.extend_clone(&self.values);
        }

        let mut children = CompactVec::new();
        if !self.children.is_empty() {
            // Internal nodes need children capacity (MAX_KEYS + 2)
            children.reserve(MAX_KEYS + 2);
            children.extend_clone(&self.children);
        }

        Self {
            keys,
            values,
            children,
        }
    }
}

/// Copy-on-Write B+ tree for i64 keys
///
/// Provides lock-free reads through structural sharing.
/// Clone is O(1) - just increments root's reference count.
pub struct CowBTree<V: Clone> {
    root: Option<CompactArc<Node<V>>>,
    /// Cached maximum key in the tree. Valid if root.is_some().
    max_key: i64,
}

impl<V: Clone> Default for CowBTree<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone> Clone for CowBTree<V> {
    /// O(1) clone - just increments root's reference count
    #[inline]
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            max_key: self.max_key,
        }
    }
}

impl<V: Clone> CowBTree<V> {
    #[inline]
    pub fn new() -> Self {
        Self {
            root: None,
            max_key: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.root.as_ref().map(CompactArc::meta).unwrap_or(0)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get a value by key. Lock-free, O(log n).
    #[inline]
    pub fn get(&self, key: i64) -> Option<&V> {
        let mut node = self.root.as_ref()?;
        loop {
            match node.search(key) {
                Ok(i) => {
                    if node.is_leaf() {
                        return Some(&node.values[i]);
                    }
                    // In B+ tree, keys in internal nodes are separators
                    // Go to right child
                    node = &node.children[i + 1];
                }
                Err(i) => {
                    if node.is_leaf() {
                        return None;
                    }
                    node = &node.children[i];
                }
            }
        }
    }

    /// Get a mutable reference to a value. Triggers COW if needed.
    #[inline]
    pub fn get_mut(&mut self, key: i64) -> Option<&mut V> {
        // Navigate to the leaf, applying COW along the path
        let mut path: CompactVec<usize> = CompactVec::new();
        {
            let mut node = self.root.as_ref()?;
            loop {
                match node.search(key) {
                    Ok(i) => {
                        if node.is_leaf() {
                            path.push(i);
                            break;
                        }
                        path.push(i + 1);
                        node = &node.children[i + 1];
                    }
                    Err(i) => {
                        if node.is_leaf() {
                            return None; // Key not found
                        }
                        path.push(i);
                        node = &node.children[i];
                    }
                }
            }
        }

        // Now traverse with COW
        let root = self.root.as_mut()?;
        let mut node = CompactArc::make_mut(root);

        for (depth, &idx) in path.iter().enumerate() {
            if depth == path.len() - 1 {
                // Last level - this is the key index in the leaf
                return Some(&mut node.values[idx]);
            }
            // Apply COW to child
            let child = &mut node.children[idx];
            node = CompactArc::make_mut(child);
        }

        None
    }

    /// Check if key exists. Lock-free, O(log n).
    #[inline]
    pub fn contains_key(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    /// Insert a key-value pair. Returns old value if key existed.
    pub fn insert(&mut self, key: i64, value: V) -> Option<V> {
        if self.root.is_none() {
            let mut node = Node::new_leaf();
            node.keys.push(key);
            node.values.push(value);
            // Count stored in Arc meta
            self.root = Some(CompactArc::new_with_meta(node, 1));
            self.max_key = key;
            return None;
        }

        // Fast path for sequential inserts: if key > max key, append to rightmost leaf
        if self.is_key_greater_than_max(key) {
            let root = self.root.as_mut().unwrap();
            let result = Self::insert_rightmost(root, key, value);

            // Update max_key since we know we are appending to the end
            self.max_key = key;

            return match result {
                InsertResult::Done(old) => {
                    // Update count in root
                    let root_arc = self.root.as_mut().unwrap();
                    let count = CompactArc::meta(root_arc);
                    // We must ensure we have a unique root to update meta
                    // insert_rightmost already called make_mut on root, so it IS unique.
                    // We can safely update meta.
                    unsafe { CompactArc::set_meta(root_arc, count + 1) };
                    old
                }
                InsertResult::Split(median, right) => {
                    let old_root = self.root.take().unwrap();
                    let old_len = CompactArc::meta(&old_root);
                    let mut new_root = Node::new_internal();
                    new_root.keys.push(median);
                    new_root.children.push(old_root);
                    new_root.children.push(right);
                    self.root = Some(CompactArc::new_with_meta(new_root, old_len + 1));
                    None
                }
            };
        }

        let root = self.root.as_mut().unwrap();
        let result = Self::insert_recursive(root, key, value);

        // Update max_key if necessary
        if key > self.max_key {
            self.max_key = key;
        }

        match result {
            InsertResult::Done(old) => {
                if old.is_none() {
                    // Update count in root
                    let root_arc = self.root.as_mut().unwrap();
                    let count = CompactArc::meta(root_arc);
                    // insert_recursive ensures root is unique (make_mut)
                    unsafe { CompactArc::set_meta(root_arc, count + 1) };
                }
                old
            }
            InsertResult::Split(median, right) => {
                // Root was split, create new root
                let old_root = self.root.take().unwrap();
                let old_len = CompactArc::meta(&old_root);
                let mut new_root = Node::new_internal();
                new_root.keys.push(median);
                new_root.children.push(old_root);
                new_root.children.push(right);
                self.root = Some(CompactArc::new_with_meta(new_root, old_len + 1));
                None
            }
        }
    }

    // ...

    /// Check if key is greater than maximum key in tree (O(1) with caching)
    #[inline]
    fn is_key_greater_than_max(&self, key: i64) -> bool {
        // If root is None, this function shouldn't be called or implies tree empty
        // In insert, we check root.is_none() first.
        // So here self.max_key is valid.
        key > self.max_key
    }

    /// Fast path: insert into rightmost leaf (for sequential inserts)
    fn insert_rightmost(node_arc: &mut CompactArc<Node<V>>, key: i64, value: V) -> InsertResult<V> {
        let node = CompactArc::make_mut(node_arc);

        if node.is_leaf() {
            // Append to end (O(1) push instead of O(n) insert)
            node.keys.push(key);
            node.values.push(value);

            if node.keys.len() > MAX_KEYS {
                let (median, right) = Self::split_leaf(node);
                InsertResult::Split(median, CompactArc::new_with_meta(right, 0))
            // New node len is unknown/0 internal
            } else {
                InsertResult::Done(None)
            }
        } else {
            // Go to rightmost child
            let last_idx = node.children.len() - 1;
            let result = Self::insert_rightmost(&mut node.children[last_idx], key, value);

            match result {
                InsertResult::Done(old) => InsertResult::Done(old),
                InsertResult::Split(median, right) => {
                    // Child was split, append to end
                    node.keys.push(median);
                    node.children.push(right);

                    if node.keys.len() > MAX_KEYS {
                        let (m, r) = Self::split_internal(node);
                        InsertResult::Split(m, CompactArc::new_with_meta(r, 0))
                    } else {
                        InsertResult::Done(None)
                    }
                }
            }
        }
    }

    fn insert_recursive(node_arc: &mut CompactArc<Node<V>>, key: i64, value: V) -> InsertResult<V> {
        // COW: clone node if shared
        let node = CompactArc::make_mut(node_arc);

        if node.is_leaf() {
            match node.search(key) {
                Ok(i) => {
                    // Key exists, replace value
                    let old = std::mem::replace(&mut node.values[i], value);
                    InsertResult::Done(Some(old))
                }
                Err(i) => {
                    // Insert at position i
                    node.keys.insert(i, key);
                    node.values.insert(i, value);

                    // Check if split needed
                    if node.keys.len() > MAX_KEYS {
                        let (median, right) = Self::split_leaf(node);
                        InsertResult::Split(median, CompactArc::new_with_meta(right, 0))
                    } else {
                        InsertResult::Done(None)
                    }
                }
            }
        } else {
            // Internal node - find child to recurse into
            let i = match node.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };

            let result = Self::insert_recursive(&mut node.children[i], key, value);

            match result {
                InsertResult::Done(old) => InsertResult::Done(old),
                InsertResult::Split(median, right) => {
                    // Child was split, insert median into this node
                    node.keys.insert(i, median);
                    node.children.insert(i + 1, right);

                    // Check if this node needs splitting
                    if node.keys.len() > MAX_KEYS {
                        let (m, r) = Self::split_internal(node);
                        InsertResult::Split(m, CompactArc::new_with_meta(r, 0))
                    } else {
                        InsertResult::Done(None)
                    }
                }
            }
        }
    }

    /// Split a leaf node. Returns (median_key, right_node).
    fn split_leaf(node: &mut Node<V>) -> (i64, Node<V>) {
        let mid = node.keys.len() / 2;

        let right_keys: CompactVec<i64> = node.keys.drain(mid..).collect();
        let right_values: CompactVec<V> = node.values.drain(mid..).collect();

        let median = right_keys[0];
        let right = Node {
            keys: right_keys,
            values: right_values,
            children: CompactVec::new(),
        };

        (median, right)
    }

    fn split_internal(node: &mut Node<V>) -> (i64, Node<V>) {
        let mid = node.keys.len() / 2;

        let right_keys: CompactVec<i64> = node.keys.drain(mid + 1..).collect();
        let median = node.keys.pop().unwrap();
        let right_children: CompactVec<CompactArc<Node<V>>> =
            node.children.drain(mid + 1..).collect();

        let right = Node {
            keys: right_keys,
            values: CompactVec::new(),
            children: right_children,
        };

        (median, right)
    }

    /// Remove a key. Returns the value if it existed.
    pub fn remove(&mut self, key: i64) -> Option<V> {
        let root = self.root.as_mut()?;
        let result = Self::remove_recursive(root, key);

        if result.is_some() {
            // Update len in root
            let root_arc = self.root.as_mut().unwrap();
            let count = CompactArc::meta(root_arc);
            // remove_recursive ensures root is unique (make_mut)
            let new_count = count - 1;
            unsafe { CompactArc::set_meta(root_arc, new_count) };

            if new_count == 0 {
                self.root = None;
                self.max_key = 0; // reset
            } else if self.max_key == key {
                // We removed the max key, need to find the new max
                self.refresh_max_key();
            }

            // If root is empty internal node, make its only child the new root
            if let Some(ref root) = self.root {
                if !root.is_leaf() && root.keys.is_empty() {
                    if root.children.len() == 1 {
                        let child_node = root.children[0].clone();
                        // Propagate count from old root to new root
                        self.root = Some(child_node);
                        let new_root_arc = self.root.as_mut().unwrap();
                        unsafe { CompactArc::set_meta(new_root_arc, new_count) };
                    }
                } else if root.is_leaf() && root.keys.is_empty() {
                    // Should be covered by new_count == 0 check above, but safe
                    self.root = None;
                    self.max_key = 0;
                }
            }
        }

        result
    }

    fn refresh_max_key(&mut self) {
        if let Some(root) = &self.root {
            let mut node = root.as_ref();
            loop {
                if node.is_leaf() {
                    self.max_key = node.keys.last().copied().unwrap_or(0);
                    break;
                }
                node = node.children.last().unwrap().as_ref();
            }
        } else {
            self.max_key = 0;
        }
    }

    fn remove_recursive(node_arc: &mut CompactArc<Node<V>>, key: i64) -> Option<V> {
        let node = CompactArc::make_mut(node_arc);

        if node.is_leaf() {
            match node.search(key) {
                Ok(i) => {
                    node.keys.remove(i);
                    Some(node.values.remove(i))
                }
                Err(_) => None,
            }
        } else {
            let i = match node.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };

            // Ensure child has enough keys before recursing
            if node.children[i].keys.len() <= MIN_KEYS {
                Self::ensure_child_can_lose_key(node, i);
            }

            // Recalculate index after rebalancing
            let new_i = match node.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };

            let i = new_i.min(node.children.len() - 1);
            Self::remove_recursive(&mut node.children[i], key)
        }
    }

    fn ensure_child_can_lose_key(node: &mut Node<V>, i: usize) {
        let can_borrow_left = i > 0 && node.children[i - 1].keys.len() > MIN_KEYS;
        let can_borrow_right =
            i < node.children.len() - 1 && node.children[i + 1].keys.len() > MIN_KEYS;

        if can_borrow_left {
            Self::borrow_from_left(node, i);
        } else if can_borrow_right {
            Self::borrow_from_right(node, i);
        } else if i > 0 {
            Self::merge_with_left(node, i);
        } else if i < node.children.len() - 1 {
            Self::merge_with_right(node, i);
        }
    }

    fn borrow_from_left(node: &mut Node<V>, i: usize) {
        // First, extract what we need from left sibling
        let is_leaf = node.children[i - 1].is_leaf();
        let (borrowed_key, borrowed_val, borrowed_child) = {
            let left = CompactArc::make_mut(&mut node.children[i - 1]);
            let key = left.keys.pop().unwrap();
            let val = if is_leaf {
                Some(left.values.pop().unwrap())
            } else {
                None
            };
            let child = if !is_leaf {
                Some(left.children.pop().unwrap())
            } else {
                None
            };
            (key, val, child)
        };

        // Now modify the current child
        let child = CompactArc::make_mut(&mut node.children[i]);
        if is_leaf {
            child.keys.insert(0, borrowed_key);
            child.values.insert(0, borrowed_val.unwrap());
            // Update separator to be the new first key of child
            node.keys[i - 1] = child.keys[0];
        } else {
            let separator = std::mem::replace(&mut node.keys[i - 1], borrowed_key);
            child.keys.insert(0, separator);
            child.children.insert(0, borrowed_child.unwrap());
        }
    }

    fn borrow_from_right(node: &mut Node<V>, i: usize) {
        // First, extract what we need from right sibling
        let is_leaf = node.children[i + 1].is_leaf();
        let (borrowed_key, borrowed_val, borrowed_child, new_separator) = {
            let right = CompactArc::make_mut(&mut node.children[i + 1]);
            let key = right.keys.remove(0);
            let val = if is_leaf {
                Some(right.values.remove(0))
            } else {
                None
            };
            let child = if !is_leaf {
                Some(right.children.remove(0))
            } else {
                None
            };
            let new_sep = if is_leaf { right.keys[0] } else { key };
            (key, val, child, new_sep)
        };

        // Now modify the current child
        let child = CompactArc::make_mut(&mut node.children[i]);
        if is_leaf {
            child.keys.push(borrowed_key);
            child.values.push(borrowed_val.unwrap());
            // Update separator to be the new first key of right
            node.keys[i] = new_separator;
        } else {
            let separator = std::mem::replace(&mut node.keys[i], new_separator);
            child.keys.push(separator);
            child.children.push(borrowed_child.unwrap());
        }
    }

    fn merge_with_left(node: &mut Node<V>, i: usize) {
        let separator = node.keys.remove(i - 1);
        let right = node.children.remove(i);

        let left = CompactArc::make_mut(&mut node.children[i - 1]);

        if !left.is_leaf() {
            left.keys.push(separator);
        }

        // Move all from right to left
        for k in right.keys.iter() {
            left.keys.push(*k);
        }
        for v in right.values.iter() {
            left.values.push(v.clone());
        }
        for c in right.children.iter() {
            left.children.push(c.clone());
        }
    }

    fn merge_with_right(node: &mut Node<V>, i: usize) {
        let separator = node.keys.remove(i);
        let right = node.children.remove(i + 1);

        let left = CompactArc::make_mut(&mut node.children[i]);

        if !left.is_leaf() {
            left.keys.push(separator);
        }

        for k in right.keys.iter() {
            left.keys.push(*k);
        }
        for v in right.values.iter() {
            left.values.push(v.clone());
        }
        for c in right.children.iter() {
            left.children.push(c.clone());
        }
    }

    /// Iterate over chunks of keys and values (O(1) amortized traversal)
    /// Yields `(&[i64], &[V])` slices directly from leaf nodes.
    /// This is ~100x faster for iteration overhead as it traverses the tree once per 128 items.
    pub fn iter_chunks(&self) -> impl Iterator<Item = (&[i64], &[V])> {
        CowBTreeChunkIter::new(self.root.as_ref())
    }

    /// Iterate over all key-value pairs in sorted order
    pub fn iter(&self) -> impl Iterator<Item = (&i64, &V)> {
        // Optimized to use chunk iteration internally
        self.iter_chunks()
            .flat_map(|(keys, values)| keys.iter().zip(values.iter()))
    }

    /// Iterate over keys in sorted order
    pub fn keys(&self) -> impl Iterator<Item = i64> + '_ {
        self.iter().map(|(k, _)| *k)
    }

    /// Iterate over values in sorted order
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Yields chunks of keys and values within the range.
    /// Each chunk is a slice from a single leaf node.
    pub fn range_chunks<R>(&self, range: R) -> impl Iterator<Item = (&[i64], &[V])>
    where
        R: std::ops::RangeBounds<i64>,
    {
        CowBTreeRangeChunkIter::new(self.root.as_ref(), range)
    }

    /// Iterator over a sub-range of elements in the B-tree.
    pub fn range<R>(&self, range: R) -> impl Iterator<Item = (&i64, &V)>
    where
        R: std::ops::RangeBounds<i64>,
    {
        self.range_chunks(range)
            .flat_map(|(keys, values)| keys.iter().zip(values.iter()))
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.root = None;
        self.max_key = 0;
    }

    /// Entry API for in-place updates
    pub fn entry(&mut self, key: i64) -> Entry<'_, V> {
        if self.contains_key(key) {
            Entry::Occupied(OccupiedEntry { tree: self, key })
        } else {
            Entry::Vacant(VacantEntry { tree: self, key })
        }
    }
}

/// Entry API for CowBTree
pub enum Entry<'a, V: Clone> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, V>),
}

/// An occupied entry in the CowBTree
pub struct OccupiedEntry<'a, V: Clone> {
    tree: &'a mut CowBTree<V>,
    key: i64,
}

impl<'a, V: Clone> OccupiedEntry<'a, V> {
    #[inline]
    pub fn key(&self) -> i64 {
        self.key
    }

    #[inline]
    pub fn get(&self) -> &V {
        // Safe because we verified key exists in entry()
        self.tree.get(self.key).unwrap()
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        // Safe because we verified key exists in entry()
        self.tree.get_mut(self.key).unwrap()
    }

    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        // Safe because we verified key exists in entry()
        self.tree.get_mut(self.key).unwrap()
    }

    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        // Replace existing value, returns old value
        self.tree.insert(self.key, value).unwrap()
    }
}

/// A vacant entry in the CowBTree
pub struct VacantEntry<'a, V: Clone> {
    tree: &'a mut CowBTree<V>,
    key: i64,
}

impl<'a, V: Clone> VacantEntry<'a, V> {
    #[inline]
    pub fn key(&self) -> i64 {
        self.key
    }

    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        self.tree.insert(self.key, value);
        // Safe because we just inserted
        self.tree.get_mut(self.key).unwrap()
    }
}

enum InsertResult<V: Clone> {
    Done(Option<V>),
    Split(i64, CompactArc<Node<V>>),
}

/// Iterator over chunks of a CowBTree (leaf slices)
struct CowBTreeChunkIter<'a, V: Clone> {
    /// Stack of (node, next_child_index) for traversal
    /// Only holds internal nodes.
    stack: Vec<(&'a Node<V>, usize)>,
    /// Current leaf node being yielded (if any)
    current_leaf: Option<&'a Node<V>>,
}

impl<'a, V: Clone> CowBTreeChunkIter<'a, V> {
    fn new(root: Option<&'a CompactArc<Node<V>>>) -> Self {
        let mut iter = Self {
            stack: Vec::new(),
            current_leaf: None,
        };
        if let Some(root) = root {
            iter.descend_to_leftmost(root.as_ref());
        }
        iter
    }

    /// Descend to the leftmost leaf, pushing internal nodes onto the stack
    fn descend_to_leftmost(&mut self, mut node: &'a Node<V>) {
        while !node.is_leaf() {
            self.stack.push((node, 1)); // Start at child 1 (we're descending to child 0)
            node = node.children[0].as_ref();
        }
        // Found a leaf - this is our first chunk
        self.current_leaf = Some(node);
    }
}

impl<'a, V: Clone> Iterator for CowBTreeChunkIter<'a, V> {
    type Item = (&'a [i64], &'a [V]);

    fn next(&mut self) -> Option<Self::Item> {
        // If we have a current leaf, yield it and consume it
        if let Some(leaf) = self.current_leaf.take() {
            return Some((&leaf.keys, &leaf.values));
        }

        // No current leaf, find next leaf from stack
        loop {
            let (node, idx) = self.stack.last_mut()?;

            if *idx < node.children.len() {
                let child_idx = *idx;
                *idx += 1;
                let child = node.children[child_idx].as_ref();
                self.descend_to_leftmost(child);

                // descend_to_leftmost sets current_leaf to the found leaf
                if let Some(leaf) = self.current_leaf.take() {
                    return Some((&leaf.keys, &leaf.values));
                }
            } else {
                // Done with all children of this internal node
                self.stack.pop();
            }
        }
    }
}

/// Range iterator over a CowBTree yielding chunks
/// Optimized: Seeks directly to start bound and yields slices
struct CowBTreeRangeChunkIter<'a, V: Clone, R> {
    stack: Vec<(&'a Node<V>, usize)>,
    range: R,
    current_leaf: Option<&'a Node<V>>,
    current_idx: usize,
    finished: bool,
}

impl<'a, V: Clone, R: std::ops::RangeBounds<i64>> CowBTreeRangeChunkIter<'a, V, R> {
    fn new(root: Option<&'a CompactArc<Node<V>>>, range: R) -> Self {
        let mut iter = Self {
            stack: Vec::new(),
            range,
            current_leaf: None,
            current_idx: 0,
            finished: false,
        };
        if let Some(root) = root {
            iter.seek_to_start(root.as_ref());
        } else {
            iter.finished = true;
        }
        iter
    }

    fn seek_to_start(&mut self, mut node: &'a Node<V>) {
        let start_key = match self.range.start_bound() {
            Bound::Included(&k) => Some(k),
            Bound::Excluded(&k) => Some(k),
            Bound::Unbounded => None,
        };

        loop {
            if node.is_leaf() {
                let mut idx = if let Some(k) = start_key {
                    match node.keys.binary_search(&k) {
                        Ok(i) => i,
                        Err(i) => i,
                    }
                } else {
                    0
                };

                if let Bound::Excluded(&k) = self.range.start_bound() {
                    if idx < node.keys.len() && node.keys[idx] == k {
                        idx += 1;
                    }
                }

                self.current_leaf = Some(node);
                self.current_idx = idx;
                break;
            } else {
                let idx = if let Some(k) = start_key {
                    match node.search(k) {
                        Ok(i) => i + 1,
                        Err(i) => i,
                    }
                } else {
                    0
                };

                self.stack.push((node, idx + 1));
                node = node.children[idx].as_ref();
            }
        }
    }
}

impl<'a, V: Clone, R: std::ops::RangeBounds<i64>> Iterator for CowBTreeRangeChunkIter<'a, V, R> {
    type Item = (&'a [i64], &'a [V]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            // Check if we have a current leaf to yield from
            if let Some(leaf) = self.current_leaf {
                if self.current_idx < leaf.keys.len() {
                    let start = self.current_idx;
                    // Determine end of chunk via bounds check
                    let end = match self.range.end_bound() {
                        Bound::Unbounded => leaf.keys.len(),
                        Bound::Included(&k) => {
                            if leaf.keys.last().unwrap() <= &k {
                                leaf.keys.len()
                            } else {
                                let pos = leaf.keys[start..].partition_point(|&x| x <= k);
                                self.finished = true;
                                start + pos
                            }
                        }
                        Bound::Excluded(&k) => {
                            if leaf.keys.last().unwrap() < &k {
                                leaf.keys.len()
                            } else {
                                let pos = leaf.keys[start..].partition_point(|&x| x < k);
                                self.finished = true;
                                start + pos
                            }
                        }
                    };

                    if start >= end {
                        self.finished = true;
                        self.current_leaf = None;
                        return None;
                    }

                    self.current_idx = end;
                    let result = (&leaf.keys[start..end], &leaf.values[start..end]);

                    if end == leaf.keys.len() && !self.finished {
                        self.current_leaf = None;
                    } else {
                        self.finished = true;
                        self.current_leaf = None;
                    }

                    return Some(result);
                } else {
                    self.current_leaf = None;
                }
            }

            if self.finished {
                return None;
            }

            // Find next leaf from stack
            if let Some((node, idx)) = self.stack.last_mut() {
                if *idx < node.children.len() {
                    let child_idx = *idx;
                    *idx += 1;
                    let mut child = node.children[child_idx].as_ref();

                    loop {
                        if child.is_leaf() {
                            self.current_leaf = Some(child);
                            self.current_idx = 0;
                            break;
                        } else {
                            self.stack.push((child, 1));
                            child = child.children[0].as_ref();
                        }
                    }
                } else {
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.finished = true;
                        return None;
                    }
                }
            } else {
                self.finished = true;
                return None;
            }
        }
    }
}

impl<V: Clone + std::fmt::Debug> std::fmt::Debug for CowBTree<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut tree: CowBTree<String> = CowBTree::new();

        assert!(tree.insert(5, "five".to_string()).is_none());
        assert!(tree.insert(3, "three".to_string()).is_none());
        assert!(tree.insert(7, "seven".to_string()).is_none());

        assert_eq!(tree.get(5), Some(&"five".to_string()));
        assert_eq!(tree.get(3), Some(&"three".to_string()));
        assert_eq!(tree.get(7), Some(&"seven".to_string()));
        assert_eq!(tree.get(1), None);

        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_update() {
        let mut tree: CowBTree<String> = CowBTree::new();

        assert!(tree.insert(5, "five".to_string()).is_none());
        assert_eq!(tree.insert(5, "FIVE".to_string()), Some("five".to_string()));
        assert_eq!(tree.get(5), Some(&"FIVE".to_string()));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_cow_semantics() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i, i * 10);
        }

        // Clone is O(1)
        let snapshot = tree.clone();

        // Modify original
        tree.insert(50, 999);
        tree.insert(200, 2000);

        // Snapshot still has old values
        assert_eq!(snapshot.get(50), Some(&500));
        assert_eq!(snapshot.get(200), None);

        // Original has new values
        assert_eq!(tree.get(50), Some(&999));
        assert_eq!(tree.get(200), Some(&2000));
    }

    #[test]
    fn test_many_inserts_sequential() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..10_000 {
            tree.insert(i, i * 2);
        }

        assert_eq!(tree.len(), 10_000);

        for i in 0..10_000 {
            assert_eq!(tree.get(i), Some(&(i * 2)), "Failed at key {}", i);
        }
    }

    #[test]
    fn test_random_inserts() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Pseudo-random inserts
        let keys: Vec<i64> = (0..1000).map(|i| (i * 7919 + 13) % 10000).collect();

        for &k in &keys {
            tree.insert(k, k * 2);
        }

        for &k in &keys {
            assert_eq!(tree.get(k), Some(&(k * 2)), "Failed at key {}", k);
        }
    }

    #[test]
    fn test_iteration() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in [5, 2, 8, 1, 9, 3, 7, 4, 6, 0] {
            tree.insert(i, i * 10);
        }

        let keys: Vec<i64> = tree.keys().collect();
        assert_eq!(keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_range() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i, i);
        }

        let range: Vec<i64> = tree.range(25..75).map(|(k, _)| *k).collect();
        assert_eq!(range.len(), 50);
        assert_eq!(range[0], 25);
        assert_eq!(range[49], 74);
    }

    #[test]
    fn test_remove_leaf() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20 {
            tree.insert(i, i * 10);
        }

        assert_eq!(tree.remove(10), Some(100));
        assert_eq!(tree.get(10), None);
        assert_eq!(tree.len(), 19);

        assert_eq!(tree.remove(10), None);
    }

    #[test]
    fn test_remove_many() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..1000 {
            tree.insert(i, i);
        }

        // Remove every other
        for i in (0..1000).step_by(2) {
            assert_eq!(tree.remove(i), Some(i));
        }

        assert_eq!(tree.len(), 500);

        // Check remaining
        for i in 0..1000 {
            if i % 2 == 0 {
                assert_eq!(tree.get(i), None);
            } else {
                assert_eq!(tree.get(i), Some(&i));
            }
        }
    }

    #[test]
    fn test_get_mut() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        tree.insert(5, 50);
        tree.insert(3, 30);
        tree.insert(7, 70);

        if let Some(v) = tree.get_mut(5) {
            *v = 500;
        }

        assert_eq!(tree.get(5), Some(&500));
        assert_eq!(tree.get(3), Some(&30));
    }

    #[test]
    fn test_cow_with_get_mut() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i, i);
        }

        let snapshot = tree.clone();

        // Modify via get_mut
        if let Some(v) = tree.get_mut(50) {
            *v = 999;
        }

        // Snapshot unchanged
        assert_eq!(snapshot.get(50), Some(&50));
        // Original changed
        assert_eq!(tree.get(50), Some(&999));
    }

    #[test]
    fn test_memory_size() {
        // CowBTree should be small (root: 8 + max_key: 8 = 16 bytes)
        assert!(std::mem::size_of::<CowBTree<i64>>() <= 16);
    }
}
