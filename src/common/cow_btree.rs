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
//! - Each node is wrapped in CompactArc (8 bytes) for memory management
//! - Nodes maintain their own `drop_count` (AtomicU32) for thread-safe V drops
//! - Readers clone the root (atomic, ~1ns) and traverse without locks
//! - Writers check `drop_count` and deep-clone only if shared (COW)
//! - Structural sharing: unmodified subtrees are shared between versions
//!
//! Thread safety:
//! - `drop_count` uses atomic fetch_sub to coordinate drops across threads
//! - Exactly one thread (whoever sees old_count=1) drops V values
//! - This avoids race conditions that could cause memory leaks
//!
//! Performance characteristics:
//! - Reads: O(log n), completely lock-free
//! - Writes: O(log n) + O(B) per modified node for COW
//! - Clone: O(1) - just increments reference counts
//! - Memory: Shared nodes between snapshots

use super::CompactArc;
use std::marker::PhantomData;
use std::mem;
use std::ops::Bound;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum keys per node. Smaller = more nodes but faster COW.
/// 128 gives good balance for database workloads.
///
/// CONSTRAINT: MAX_KEYS <= 255 because NodePath uses u8 for child indices.
/// Internal nodes have MAX_KEYS + 1 children, so index can be at most MAX_KEYS.
const MAX_KEYS: usize = 128;

// Compile-time assertion: MAX_KEYS must fit in u8 (NodePath uses [u8; MAX_TREE_DEPTH])
const _: () = assert!(
    MAX_KEYS <= 255,
    "MAX_KEYS must be <= 255 (NodePath uses u8 indices)"
);

/// Minimum keys per node (except root).
const MIN_KEYS: usize = MAX_KEYS / 2;

/// Header: 8 bytes
/// [ len (u16) | is_leaf (u8) | pad (1) | drop_count (4) ]
///
/// `drop_count` is a separate refcount that coordinates dropping V values.
/// This fixes a race condition where multiple threads could skip dropping
/// V values when relying solely on CompactArc's is_unique() check.
#[repr(C)]
struct NodeHeader {
    len: u16,
    is_leaf: u8,
    _pad1: u8,
    /// Refcount for coordinating V/children drops.
    /// Decremented atomically in NodePtr::drop; whoever sees old_count=1 drops contents.
    /// Using U32 to support up to ~4 billion concurrent snapshots.
    drop_count: AtomicU32,
}

/// A Ref-counted pointer to a Byte-Packed Node.
/// Memory Layout: [ Header | Keys (MAX_KEYS) | Values/Children ]
/// One contiguous allocation.
pub struct NodePtr<V: Clone> {
    ptr: CompactArc<[u8]>,
    _marker: PhantomData<V>,
}

impl<V: Clone> Clone for NodePtr<V> {
    fn clone(&self) -> Self {
        // Clone CompactArc first (keeps memory alive), then increment drop_count.
        // This ordering ensures drop_count is only incremented after the clone succeeds.
        let new_ptr = self.ptr.clone();

        // SAFETY: ptr points to a valid CompactArc allocation that starts with NodeHeader.
        // The header is read-only here (only accessing drop_count atomically).
        let header = unsafe { &*(self.ptr.as_ptr() as *const NodeHeader) };
        header.drop_count.fetch_add(1, Ordering::Relaxed);

        Self {
            ptr: new_ptr,
            _marker: PhantomData,
        }
    }
}

impl<V: Clone> Drop for NodePtr<V> {
    fn drop(&mut self) {
        // Atomically decrement drop_count and check if we're the last reference.
        // This fixes a race condition where multiple threads using is_unique()
        // could all see refcount > 1 and skip dropping V values.
        //
        // With atomic fetch_sub, exactly one thread will see old_count == 1
        // and that thread is responsible for dropping the contents.
        // SAFETY: ptr points to a valid CompactArc allocation that starts with NodeHeader.
        let header = unsafe { &*(self.ptr.as_ptr() as *const NodeHeader) };
        let old_count = header.drop_count.fetch_sub(1, Ordering::AcqRel);

        if old_count != 1 {
            // Not the last reference - don't drop contents
            return;
        }

        // We're the last reference - drop contents
        let len = self.len();
        if self.is_leaf() {
            // Drop values in valid range
            // SAFETY: We are the last reference (old_count == 1). The values at indices
            // 0..len are valid initialized V instances. After dropping, we don't access them.
            unsafe {
                let v_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
                for i in 0..len {
                    ptr::drop_in_place(v_ptr.add(i));
                }
            }
        } else {
            // Drop children (keys count + 1)
            // SAFETY: We are the last reference (old_count == 1). The children at indices
            // 0..=len are valid NodePtr instances. After dropping, we don't access them.
            unsafe {
                let c_ptr =
                    (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
                for i in 0..=len {
                    ptr::drop_in_place(c_ptr.add(i));
                }
            }
        }
    }
}

impl<V: Clone> NodePtr<V> {
    /// Keys start immediately after the header
    const fn keys_offset() -> usize {
        mem::size_of::<NodeHeader>()
    }

    /// Offset to values array (leaf nodes only).
    /// Same as `children_offset` - this is union-style memory sharing:
    /// leaf nodes store values here, internal nodes store children here.
    /// A node is either leaf OR internal, never both.
    const fn values_offset() -> usize {
        Self::keys_offset() + ((MAX_KEYS + 1) * 8)
    }

    /// Offset to children array (internal nodes only).
    /// Same as `values_offset` - see comment above.
    const fn children_offset() -> usize {
        Self::keys_offset() + ((MAX_KEYS + 1) * 8)
    }

    fn new_leaf() -> Self {
        let size = Self::values_offset() + ((MAX_KEYS + 1) * mem::size_of::<V>());
        let vec = vec![0u8; size];

        let mut ptr = NodePtr {
            ptr: CompactArc::from_vec(vec),
            _marker: PhantomData,
        };

        let header = ptr.header_mut();
        header.len = 0;
        header.is_leaf = 1;
        header.drop_count = AtomicU32::new(1);

        ptr
    }

    fn new_internal() -> Self {
        let size = Self::children_offset() + ((MAX_KEYS + 2) * mem::size_of::<NodePtr<V>>());
        let vec = vec![0u8; size];

        let mut ptr = NodePtr {
            ptr: CompactArc::from_vec(vec),
            _marker: PhantomData,
        };

        let header = ptr.header_mut();
        header.len = 0;
        header.is_leaf = 0;
        header.drop_count = AtomicU32::new(1);

        ptr
    }

    fn make_mut(&mut self) -> &mut Self {
        // Check if this node is shared using our own drop_count.
        // If drop_count > 1, we need to deep clone to maintain COW semantics.
        // SAFETY: ptr points to a valid CompactArc allocation that starts with NodeHeader.
        let header = unsafe { &*(self.ptr.as_ptr() as *const NodeHeader) };
        if header.drop_count.load(Ordering::Acquire) != 1 {
            // Shared - need proper deep clone (not just byte copy!)
            // Byte copy would cause double-free for non-Copy value types
            *self = self.deep_clone();
        }
        self
    }

    /// Create a deep clone of this node.
    /// For leaf nodes: clones all values using V::clone()
    /// For internal nodes: clones all children (incrementing their refcounts)
    fn deep_clone(&self) -> Self {
        let len = self.len();

        if self.is_leaf() {
            let mut new_node = NodePtr::new_leaf();
            // Do NOT set len yet to ensure exception safety!
            // If V::clone() panics, new_node.drop() will only drop 0 elements.

            // Copy keys (i64 is Copy, byte copy is fine)
            // SAFETY: Both src and dst are valid pointers within their respective node allocations.
            // Keys are i64 (Copy type), so byte copy is safe. len <= MAX_KEYS.
            unsafe {
                let k_src = self.ptr.as_ptr().add(Self::keys_offset()) as *const i64;
                let k_dst = (new_node.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
                ptr::copy_nonoverlapping(k_src, k_dst, len);
            }

            // Clone values using V::clone() - critical for non-Copy types!
            // SAFETY: v_src points to len valid V instances. v_dst points to uninitialized memory.
            // We use ptr::write to initialize each slot, and increment len after each successful
            // clone to maintain exception safety (if clone panics, only initialized values are dropped).
            unsafe {
                let v_src = self.ptr.as_ptr().add(Self::values_offset()) as *const V;
                let v_dst = (new_node.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
                for i in 0..len {
                    ptr::write(v_dst.add(i), (*v_src.add(i)).clone());
                    // Increment len after successful write/clone
                    // If clone panics on next iteration, this node will correctly drop 'i' elements
                    new_node.set_len(i + 1);
                }
            }

            new_node
        } else {
            let mut new_node = NodePtr::new_internal();
            new_node.set_len(len);

            // Copy keys (i64 is Copy)
            // SAFETY: Both src and dst are valid pointers within their respective node allocations.
            // Keys are i64 (Copy type), so byte copy is safe. len <= MAX_KEYS.
            unsafe {
                let k_src = self.ptr.as_ptr().add(Self::keys_offset()) as *const i64;
                let k_dst = (new_node.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
                ptr::copy_nonoverlapping(k_src, k_dst, len);
            }

            // Clone children (NodePtr::clone increments CompactArc refcount)
            // NodePtr::clone cannot panic, so strict exception safety sequence not needed here,
            // but setting len upfront is fine.
            // SAFETY: c_src points to len+1 valid NodePtr instances. c_dst points to uninitialized memory.
            // NodePtr::clone only does atomic operations and cannot panic.
            unsafe {
                let c_src = self.ptr.as_ptr().add(Self::children_offset()) as *const NodePtr<V>;
                let c_dst = (new_node.ptr.as_ptr() as *mut u8).add(Self::children_offset())
                    as *mut NodePtr<V>;
                for i in 0..=len {
                    // children count = keys count + 1
                    ptr::write(c_dst.add(i), (*c_src.add(i)).clone());
                }
            }

            new_node
        }
    }

    fn header(&self) -> &NodeHeader {
        // SAFETY: ptr points to a valid CompactArc allocation that starts with NodeHeader.
        unsafe { &*(self.ptr.as_ptr() as *const NodeHeader) }
    }

    fn header_mut(&mut self) -> &mut NodeHeader {
        // Assumes we have unique access (called make_mut)
        // SAFETY: ptr points to a valid CompactArc allocation. Caller guarantees unique access.
        unsafe { &mut *(self.ptr.as_ptr() as *mut NodeHeader) }
    }

    fn is_leaf(&self) -> bool {
        self.header().is_leaf == 1
    }

    fn len(&self) -> usize {
        self.header().len as usize
    }

    fn set_len(&mut self, len: usize) {
        self.header_mut().len = len as u16;
    }

    fn keys(&self) -> &[i64] {
        let len = self.len();
        // SAFETY: ptr + keys_offset points to an array of len initialized i64 keys.
        // The slice lifetime is tied to &self, ensuring validity.
        unsafe {
            let ptr = self.ptr.as_ptr().add(Self::keys_offset()) as *const i64;
            std::slice::from_raw_parts(ptr, len)
        }
    }

    fn values(&self) -> &[V] {
        assert!(self.is_leaf());
        let len = self.len();
        // SAFETY: This is a leaf node (asserted). ptr + values_offset points to
        // an array of len initialized V values. The slice lifetime is tied to &self.
        unsafe {
            let ptr = self.ptr.as_ptr().add(Self::values_offset()) as *const V;
            std::slice::from_raw_parts(ptr, len)
        }
    }

    fn values_mut_slice(&mut self) -> &mut [V] {
        assert!(self.is_leaf());
        let len = self.len();
        // SAFETY: This is a leaf node (asserted). ptr + values_offset points to
        // an array of len initialized V values. Caller has &mut self, ensuring unique access.
        unsafe {
            let ptr = (self.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    fn children(&self) -> &[NodePtr<V>] {
        assert!(!self.is_leaf());
        let len = self.len() + 1; // Children = keys + 1
                                  // SAFETY: This is an internal node (asserted). ptr + children_offset points to
                                  // an array of len initialized NodePtr children. The slice lifetime is tied to &self.
        unsafe {
            let ptr = self.ptr.as_ptr().add(Self::children_offset()) as *const NodePtr<V>;
            std::slice::from_raw_parts(ptr, len)
        }
    }

    fn children_mut_slice(&mut self) -> &mut [NodePtr<V>] {
        assert!(!self.is_leaf());
        let len = self.len() + 1;
        // SAFETY: This is an internal node (asserted). ptr + children_offset points to
        // an array of len initialized NodePtr children. Caller has &mut self, ensuring unique access.
        unsafe {
            let ptr =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    fn child(&self, index: usize) -> &NodePtr<V> {
        &self.children()[index]
    }

    fn child_mut(&mut self, index: usize) -> &mut NodePtr<V> {
        &mut self.children_mut_slice()[index]
    }

    fn search(&self, key: i64) -> Result<usize, usize> {
        self.keys().binary_search(&key)
    }

    fn push_leaf(&mut self, key: i64, value: V) {
        assert!(self.is_leaf());
        let len = self.len();
        assert!(len <= MAX_KEYS); // Allow temporary overflow

        // SAFETY: This is a leaf node (asserted). len <= MAX_KEYS, so index len is within
        // the allocated array bounds (size MAX_KEYS + 1). We write to uninitialized slots.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::write(k_ptr.add(len), key);

            let v_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
            ptr::write(v_ptr.add(len), value);
        }
        self.set_len(len + 1);
    }

    fn remove_leaf(&mut self, index: usize) -> V {
        assert!(self.is_leaf());
        let len = self.len();
        assert!(index < len);

        // SAFETY: This is a leaf node (asserted). index < len, so all accesses are in bounds.
        // We read the value at index (moving it out), then shift remaining elements left.
        // The last position becomes logically invalid and is excluded by set_len.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::copy(k_ptr.add(index + 1), k_ptr.add(index), len - index - 1);

            let v_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
            let val = ptr::read(v_ptr.add(index));
            ptr::copy(v_ptr.add(index + 1), v_ptr.add(index), len - index - 1);

            self.set_len(len - 1);
            val
        }
    }

    fn insert_leaf(&mut self, index: usize, key: i64, value: V) {
        assert!(self.is_leaf());
        let len = self.len();
        assert!(len <= MAX_KEYS);
        assert!(
            index <= len,
            "insert_leaf: index {} > len {} for key {}",
            index,
            len,
            key
        );

        // SAFETY: This is a leaf node (asserted). len <= MAX_KEYS, so we have room for one more.
        // We shift elements at [index..len] to [index+1..len+1], then write the new key/value at index.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            let p_key = k_ptr.add(index);
            ptr::copy(p_key, p_key.add(1), len - index);
            ptr::write(p_key, key);

            let v_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
            let p_val = v_ptr.add(index);
            ptr::copy(p_val, p_val.add(1), len - index);
            ptr::write(p_val, value);
        }
        self.set_len(len + 1);
    }

    fn push_internal(&mut self, key: i64, child: NodePtr<V>) {
        assert!(!self.is_leaf());
        let len = self.len();
        assert!(len <= MAX_KEYS);

        // SAFETY: This is an internal node (asserted). len <= MAX_KEYS, so indices len (for key)
        // and len+1 (for child) are within allocated bounds. We write to uninitialized slots.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::write(k_ptr.add(len), key);

            let c_ptr =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            ptr::write(c_ptr.add(len + 1), child);
        }
        self.set_len(len + 1);
    }

    fn split_internal(&mut self) -> (i64, NodePtr<V>) {
        let len = self.len();
        let mid = len / 2;
        let med_key = self.keys()[mid];
        let mut right = NodePtr::new_internal();
        let right_keys_count = len - mid - 1;
        let right_children_count = right_keys_count + 1;

        // SAFETY: Both self and right are internal nodes. We copy keys from indices [mid+1..len)
        // and children from indices [mid+1..len+1) of self into the start of right. The source
        // and destination do not overlap (different allocations). Keys are Copy (i64). Children
        // (NodePtr<V>) are bitwise copied - this is safe because we set self.len = mid afterwards,
        // so those slots become logically uninitialized (won't be dropped when self is dropped).
        unsafe {
            let k_src = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            let k_dst = (right.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::copy_nonoverlapping(k_src.add(mid + 1), k_dst, right_keys_count);

            let c_src =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            let c_dst =
                (right.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            ptr::copy_nonoverlapping(c_src.add(mid + 1), c_dst, right_children_count);
        }

        right.set_len(right_keys_count);
        self.set_len(mid);
        (med_key, right)
    }

    /// Optimized split for rightmost (sequential) inserts on internal nodes.
    /// Keeps MAX_KEYS keys in left node, moves only the last child to right.
    /// The median key (last key) goes to parent. Right node has 0 keys, 1 child.
    fn split_internal_rightmost(&mut self) -> (i64, NodePtr<V>) {
        let len = self.len();
        debug_assert!(
            len == MAX_KEYS + 1,
            "split_internal_rightmost expects overflow node"
        );

        // Median is the last key - it goes to parent
        let med_key = self.keys()[len - 1];

        let mut right = NodePtr::new_internal();

        // SAFETY: self is an internal node with MAX_KEYS + 1 keys (MAX_KEYS + 2 children).
        // We move only the last child to right. Children are moved via ptr::read then ptr::write.
        // After set_len(len-1), the source slots are logically uninitialized.
        unsafe {
            let c_src =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            let c_dst =
                (right.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;

            // Move last child (index len, since we have len+1 children)
            ptr::write(c_dst, ptr::read(c_src.add(len)));
        }

        right.set_len(0); // 0 keys, but 1 child
        self.set_len(len - 1); // MAX_KEYS keys, MAX_KEYS + 1 children

        (med_key, right)
    }

    fn borrow_from_left(&mut self, index: usize) {
        assert!(!self.is_leaf());
        let is_left_leaf = self.child(index - 1).is_leaf();

        if is_left_leaf {
            let (key, val) = {
                let left = self.child_mut(index - 1).make_mut();
                let left_len = left.len();
                let key = left.keys()[left_len - 1];
                // SAFETY: left is a leaf node (verified by is_left_leaf). left_len > 0 because
                // we only borrow from siblings with spare keys. Index left_len - 1 is valid.
                // We move the value out via ptr::read, then set_len(left_len - 1) marks it
                // as logically removed so it won't be double-dropped.
                let val = unsafe {
                    let val_ptr =
                        (left.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
                    ptr::read(val_ptr.add(left_len - 1))
                };
                left.set_len(left_len - 1);
                (key, val)
            };

            let current = self.child_mut(index).make_mut();
            current.insert_leaf(0, key, val);

            // SAFETY: self is an internal node (asserted). index > 0 (we're borrowing from left).
            // index - 1 is a valid key index in self (parent of left and current).
            // Writing i64 which is Copy, overwriting existing separator key.
            unsafe {
                let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
                ptr::write(k_ptr.add(index - 1), key);
            }
        } else {
            let (child, key) = {
                let left = self.child_mut(index - 1).make_mut();
                let left_len = left.len();
                let key = left.keys()[left_len - 1];
                // SAFETY: left is an internal node (is_left_leaf is false). left_len > 0.
                // Index left_len is valid for children array (internal nodes have len+1 children).
                // We move the child out via ptr::read, then set_len(left_len - 1) ensures it
                // won't be double-dropped.
                let child = unsafe {
                    let c_ptr = (left.ptr.as_ptr() as *mut u8).add(Self::children_offset())
                        as *mut NodePtr<V>;
                    ptr::read(c_ptr.add(left_len))
                };
                left.set_len(left_len - 1);
                (child, key)
            };

            let separator_idx = index - 1;
            let separator = self.keys()[separator_idx];

            let current = self.child_mut(index).make_mut();
            current.insert_internal_at_start(separator, child);

            // SAFETY: self is an internal node (asserted). separator_idx is valid key index.
            // Writing i64 which is Copy, overwriting existing separator key.
            unsafe {
                let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
                ptr::write(k_ptr.add(separator_idx), key);
            }
        }
    }

    fn insert_internal(&mut self, index: usize, key: i64, child: NodePtr<V>) {
        assert!(!self.is_leaf());
        let len = self.len();
        assert!(len <= MAX_KEYS);

        // SAFETY: This is an internal node (asserted). len <= MAX_KEYS, so there's space.
        // index <= len. We shift keys [index..len) right by 1 to make room, then write key
        // at index. We shift children [index+1..len+1) right by 1, then write new child at
        // index+1. ptr::copy handles overlapping regions correctly. Keys are Copy (i64).
        // The new child is moved in (not cloned) via ptr::write.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::copy(k_ptr.add(index), k_ptr.add(index + 1), len - index);
            ptr::write(k_ptr.add(index), key);

            let c_ptr =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            ptr::copy(c_ptr.add(index + 1), c_ptr.add(index + 2), len - index);
            ptr::write(c_ptr.add(index + 1), child);
        }
        self.set_len(len + 1);
    }

    fn insert_internal_at_start(&mut self, key: i64, child: NodePtr<V>) {
        let len = self.len();
        // SAFETY: This is an internal node (only called from internal node contexts).
        // We shift all keys [0..len) right by 1 and write new key at index 0.
        // We shift all children [0..len+1) right by 1 and write new child at index 0.
        // ptr::copy handles overlapping regions correctly. Keys are Copy (i64).
        // The new child is moved in via ptr::write. len + 1 <= MAX_KEYS + 1 by B-tree invariants.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::copy(k_ptr, k_ptr.add(1), len);
            ptr::write(k_ptr, key);

            let c_ptr =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            ptr::copy(c_ptr, c_ptr.add(1), len + 1);
            ptr::write(c_ptr, child);
        }
        self.set_len(len + 1);
    }

    fn borrow_from_right(&mut self, index: usize) {
        assert!(!self.is_leaf());
        let is_right_leaf = self.child(index + 1).is_leaf();

        if is_right_leaf {
            let (key, val) = {
                let right = self.child_mut(index + 1).make_mut();
                let key = right.keys()[0];
                let val = right.remove_leaf(0);
                (key, val)
            };

            let current = self.child_mut(index).make_mut();
            current.push_leaf(key, val);

            let right = self.child(index + 1);
            let new_sep = right.keys()[0];
            // SAFETY: self is an internal node (asserted). index is valid key index in self
            // (it's the separator between children at index and index+1).
            // Writing i64 which is Copy, overwriting existing separator key.
            unsafe {
                let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
                ptr::write(k_ptr.add(index), new_sep);
            }
        } else {
            let (child, key) = {
                let right = self.child_mut(index + 1).make_mut();
                let key = right.keys()[0];
                // SAFETY: right is an internal node (is_right_leaf is false). right.len() > 0.
                // Index 0 is valid for children array. We move the first child out via ptr::read,
                // then remove_internal_at_start() shifts remaining elements left and decrements len.
                let child = unsafe {
                    let c_ptr = (right.ptr.as_ptr() as *mut u8).add(Self::children_offset())
                        as *mut NodePtr<V>;
                    ptr::read(c_ptr)
                };
                right.remove_internal_at_start();
                (child, key)
            };

            let separator = self.keys()[index];

            let current = self.child_mut(index).make_mut();
            current.push_internal(separator, child);

            // SAFETY: self is an internal node (asserted). index is valid key index.
            // Writing i64 which is Copy, overwriting existing separator key.
            unsafe {
                let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
                ptr::write(k_ptr.add(index), key);
            }
        }
    }

    fn remove_internal_at_start(&mut self) {
        let len = self.len();
        // SAFETY: This is an internal node (only called from internal node contexts). len > 0.
        // We shift keys [1..len) left by 1 (overwriting key at index 0).
        // We shift children [1..len+1) left by 1 (overwriting child at index 0, which was
        // already moved out by caller via ptr::read). Keys are Copy (i64). Children are
        // bitwise copied (ptr::copy handles overlap correctly). The child at the end becomes
        // logically uninitialized after set_len decrements the count.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::copy(k_ptr.add(1), k_ptr, len - 1);

            let c_ptr =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            ptr::copy(c_ptr.add(1), c_ptr, len);
        }
        self.set_len(len - 1);
    }

    fn merge_with_left(&mut self, index: usize) {
        let separator = self.keys()[index - 1];

        // Make right uniquely owned so we can move data out of it
        let right_raw = self.child_mut(index) as *mut NodePtr<V>;
        // SAFETY: right_raw points to a valid NodePtr<V> in self's children array at index.
        // We need a raw pointer here to avoid borrow checker issues: we'll access left sibling
        // later, but Rust sees child_mut(index) and child_mut(index-1) as conflicting borrows.
        // The raw pointer lets us work around this while maintaining actual safety since we
        // only access right through right_raw, and later access left through child_mut.
        let right = unsafe { (*right_raw).make_mut() };
        let is_leaf = right.is_leaf();
        let r_len = right.len();

        if is_leaf {
            // Move keys and values out of right using ptr::read (no clone!)
            let mut keys_vals: Vec<(i64, V)> = Vec::with_capacity(r_len);
            // SAFETY: right is a leaf node (verified by is_leaf). We read all keys and values
            // from indices [0..r_len). Keys are Copy (i64). Values are moved via ptr::read.
            // We set_len(0) immediately after to mark them as logically removed, preventing
            // double-free when right's NodePtr is eventually dropped.
            unsafe {
                let k_ptr = right.ptr.as_ptr().add(Self::keys_offset()) as *const i64;
                let v_ptr = right.ptr.as_ptr().add(Self::values_offset()) as *mut V;
                for i in 0..r_len {
                    let key = *k_ptr.add(i);
                    let val = ptr::read(v_ptr.add(i)); // Move, not clone!
                    keys_vals.push((key, val));
                }
            }
            // Set len=0 BEFORE any other operation to prevent double-free when right is dropped
            right.set_len(0);

            // Now we can safely mutably borrow left
            let left = self.child_mut(index - 1).make_mut();
            for (k, v) in keys_vals {
                left.push_leaf(k, v);
            }
        } else {
            // Move keys and children out of right using ptr::read (not clone!)
            // After make_mut(), we have unique ownership of the node structure,
            // so we can safely move the children out.
            let keys: Vec<i64> = right.keys().to_vec();
            // SAFETY: right is an internal node. We read all children from indices [0..r_len+1).
            // Children are moved via ptr::read.
            let children: Vec<NodePtr<V>> = unsafe {
                let c_ptr = right.ptr.as_ptr().add(Self::children_offset()) as *const NodePtr<V>;
                (0..=r_len).map(|i| ptr::read(c_ptr.add(i))).collect()
            };

            // CRITICAL: Set drop_count to 2 to prevent NodePtr::drop from dropping
            // the children again (they've been moved to the children Vec).
            // For internal nodes, Drop iterates 0..=len, so with len=0 it would still
            // try to drop child 0. By setting drop_count > 1, Drop thinks this isn't
            // the last reference and skips content dropping entirely.
            // SAFETY: We have unique access to right via make_mut(). Setting drop_count
            // to 2 is safe because we're about to drop this node in remove_key_and_child,
            // and we want Drop to skip the children (they've been moved out).
            unsafe {
                let header = &*(right.ptr.as_ptr() as *const NodeHeader);
                header.drop_count.store(2, Ordering::Release);
            }
            right.set_len(0);

            let left = self.child_mut(index - 1).make_mut();

            // Move children to left using into_iter (no cloning needed)
            let mut children_iter = children.into_iter();
            left.push_internal(separator, children_iter.next().unwrap());
            for (key, child) in keys.into_iter().zip(children_iter) {
                left.push_internal(key, child);
            }
        }

        self.remove_key_and_child(index - 1, index);
    }

    fn merge_with_right(&mut self, index: usize) {
        self.merge_with_left(index + 1)
    }

    fn remove_key_and_child(&mut self, key_idx: usize, child_idx: usize) {
        let len = self.len();
        // SAFETY: This is an internal node (only called from internal node contexts).
        // key_idx < len and child_idx <= len (valid indices). We shift keys [key_idx+1..len)
        // left by 1 to fill the gap. For children, we first drop_in_place the child being
        // removed (it's a NodePtr that needs cleanup), then shift children [child_idx+1..len+1)
        // left by 1. ptr::copy handles overlapping regions. Keys are Copy (i64). The slots
        // at the end become logically uninitialized after set_len decrements the count.
        unsafe {
            let k_ptr = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::copy(
                k_ptr.add(key_idx + 1),
                k_ptr.add(key_idx),
                len - key_idx - 1,
            );

            let c_ptr =
                (self.ptr.as_ptr() as *mut u8).add(Self::children_offset()) as *mut NodePtr<V>;
            // Drop the child being removed before overwriting it
            ptr::drop_in_place(c_ptr.add(child_idx));
            ptr::copy(
                c_ptr.add(child_idx + 1),
                c_ptr.add(child_idx),
                len - child_idx,
            );
        }
        self.set_len(len - 1);
    }

    fn split_leaf(&mut self) -> (i64, NodePtr<V>) {
        let mid = self.len() / 2;
        let right_count = self.len() - mid;

        let mut right = NodePtr::new_leaf();

        // SAFETY: Both self and right are leaf nodes. We copy keys from indices [mid..len)
        // and values from indices [mid..len) of self into the start of right. The source
        // and destination do not overlap (different allocations). Keys are Copy (i64). Values
        // are bitwise copied - this is safe because we set self.len = mid afterwards, so those
        // slots become logically uninitialized (won't be dropped when self is dropped).
        unsafe {
            let k_src = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            let k_dst = (right.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;
            ptr::copy_nonoverlapping(k_src.add(mid), k_dst, right_count);

            let v_src = (self.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
            let v_dst = (right.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
            ptr::copy_nonoverlapping(v_src.add(mid), v_dst, right_count);
        }

        right.set_len(right_count);
        self.set_len(mid);

        let median = right.keys()[0];
        (median, right)
    }

    /// Optimized split for rightmost (sequential) inserts.
    /// Keeps MAX_KEYS in left node, moves only 1 key to right.
    /// This achieves ~100% fill factor for sequential insert workloads
    /// instead of the standard 50% from midpoint splits.
    fn split_leaf_rightmost(&mut self) -> (i64, NodePtr<V>) {
        let len = self.len();
        debug_assert!(
            len == MAX_KEYS + 1,
            "split_leaf_rightmost expects overflow node"
        );

        let mut right = NodePtr::new_leaf();

        // SAFETY: self is a leaf with MAX_KEYS + 1 elements. We move only the last
        // key/value to right. The key is Copy (i64). The value is moved via ptr::read
        // then ptr::write. After set_len(len-1), the source slot is logically uninitialized.
        unsafe {
            let k_src = (self.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *const i64;
            let k_dst = (right.ptr.as_ptr() as *mut u8).add(Self::keys_offset()) as *mut i64;

            let v_src = (self.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;
            let v_dst = (right.ptr.as_ptr() as *mut u8).add(Self::values_offset()) as *mut V;

            // Copy last key
            ptr::write(k_dst, *k_src.add(len - 1));

            // Move last value (ptr::read moves ownership out)
            ptr::write(v_dst, ptr::read(v_src.add(len - 1)));
        }

        right.set_len(1);
        self.set_len(len - 1); // MAX_KEYS

        let median = right.keys()[0];
        (median, right)
    }
}

/// Copy-on-Write B+ tree for i64 keys
///
/// Provides lock-free reads through structural sharing.
/// Clone is O(1) - just increments root's reference count.
///
/// # Thread Safety
///
/// - **Readers**: Multiple readers can safely access the tree concurrently via cloned
///   snapshots. Each snapshot is immutable from the reader's perspective.
/// - **Writers**: Write operations (`insert`, `remove`, `get_mut`, `entry`) require
///   exclusive access (`&mut self`). External synchronization (e.g., `Mutex`, `RwLock`)
///   is required if multiple threads need write access.
/// - **Pattern**: Clone the tree to create a snapshot, then readers use the snapshot
///   while writers mutate the original. This is the standard MVCC pattern.
const MAX_TREE_DEPTH: usize = 16;

/// Stack-based path to avoid allocations
#[derive(Clone, Copy)]
pub struct NodePath {
    indices: [u8; MAX_TREE_DEPTH],
    len: u8,
}

impl Default for NodePath {
    fn default() -> Self {
        Self {
            indices: [0; MAX_TREE_DEPTH],
            len: 0,
        }
    }
}

impl NodePath {
    pub fn new() -> Self {
        Self::default()
    }

    #[cold]
    #[inline(never)]
    fn depth_overflow() -> ! {
        panic!("B-Tree depth exceeded maximum")
    }

    pub fn push(&mut self, idx: usize) {
        if (self.len as usize) < MAX_TREE_DEPTH {
            self.indices[self.len as usize] = idx as u8;
            self.len += 1;
        } else {
            Self::depth_overflow();
        }
    }

    fn get(&self, depth: usize) -> usize {
        self.indices[depth] as usize
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len as usize).map(|i| self.indices[i] as usize)
    }
}

pub struct CowBTree<V: Clone> {
    root: Option<NodePtr<V>>,
    /// Cached maximum key in the tree. Valid if root.is_some().
    max_key: i64,
    /// Number of elements in the tree
    len: usize,
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
            len: self.len,
        }
    }
}

impl<V: Clone> CowBTree<V> {
    #[inline]
    pub fn new() -> Self {
        assert!(
            mem::align_of::<V>() <= 8,
            "CowBTree value type alignment must be <= 8"
        );
        Self {
            root: None,
            max_key: 0,
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
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
                        return Some(&node.values()[i]);
                    }
                    node = node.child(i + 1);
                }
                Err(i) => {
                    if node.is_leaf() {
                        return None;
                    }
                    node = node.child(i);
                }
            }
        }
    }

    /// Get a mutable reference to a value. Triggers COW if needed.
    #[inline]
    pub fn get_mut(&mut self, key: i64) -> Option<&mut V> {
        let (path, leaf_idx) = self.search_path(key);
        let leaf_idx = leaf_idx.ok()?;

        self.get_mut_with_path(key, &path, leaf_idx)
    }

    fn get_mut_with_path(&mut self, _key: i64, path: &NodePath, leaf_idx: usize) -> Option<&mut V> {
        let root = self.root.as_mut()?;
        let mut node = root.make_mut();

        for idx in path.iter() {
            let child = node.child_mut(idx);
            node = child.make_mut();
        }

        if leaf_idx < node.len() {
            Some(&mut node.values_mut_slice()[leaf_idx])
        } else {
            None
        }
    }

    /// Check if key exists. Lock-free, O(log n).
    #[inline]
    pub fn contains_key(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    /// Insert a key-value pair. Returns old value if key existed.
    pub fn insert(&mut self, key: i64, value: V) -> Option<V> {
        if self.root.is_none() {
            let mut node = NodePtr::new_leaf();
            node.push_leaf(key, value);
            self.root = Some(node);
            self.max_key = key;
            self.len = 1;
            return None;
        }

        // Fast path for sequential inserts: if key > max key, append to rightmost leaf
        if self.is_key_greater_than_max(key) {
            let root = self.root.as_mut().unwrap();
            let result = Self::insert_rightmost(root, key, value);
            self.max_key = key;

            return match result {
                InsertResult::Done(old) => {
                    if old.is_none() {
                        self.len += 1;
                    }
                    old
                }
                InsertResult::Split(median, right) => {
                    let old_root = self.root.take().unwrap();
                    let mut new_root = NodePtr::new_internal();

                    // SAFETY: new_root is a freshly created internal node with len=0.
                    // We write old_root to children[0]. Internal nodes have len+1 children,
                    // so with len=0 we have space for 1 child at index 0. old_root is moved
                    // (not cloned) into the slot. push_internal will add children[1] and set len=1.
                    unsafe {
                        let c_ptr = (new_root.ptr.as_ptr() as *mut u8)
                            .add(NodePtr::<V>::children_offset())
                            as *mut NodePtr<V>;
                        ptr::write(c_ptr, old_root);
                    }

                    new_root.push_internal(median, right);
                    self.root = Some(new_root);
                    self.len += 1;
                    None
                }
            };
        }

        let root = self.root.as_mut().unwrap();
        let result = Self::insert_recursive(root, key, value);

        if key > self.max_key {
            self.max_key = key;
        }

        match result {
            InsertResult::Done(old) => {
                if old.is_none() {
                    self.len += 1;
                }
                old
            }
            InsertResult::Split(median, right) => {
                let old_root = self.root.take().unwrap();
                let mut new_root = NodePtr::new_internal();
                // SAFETY: new_root is a freshly created internal node with len=0.
                // We write old_root to children[0]. Internal nodes have len+1 children,
                // so with len=0 we have space for 1 child at index 0. old_root is moved
                // (not cloned) into the slot. push_internal will add children[1] and set len=1.
                unsafe {
                    let c_ptr = (new_root.ptr.as_ptr() as *mut u8)
                        .add(NodePtr::<V>::children_offset())
                        as *mut NodePtr<V>;
                    ptr::write(c_ptr, old_root);
                }
                new_root.push_internal(median, right);
                self.root = Some(new_root);
                self.len += 1;
                None
            }
        }
    }

    /// Check if key is greater than maximum key in tree (O(1) with caching)
    #[inline]
    fn is_key_greater_than_max(&self, key: i64) -> bool {
        self.root.is_some() && key > self.max_key
    }

    /// Fast path: insert into rightmost leaf (for sequential inserts)
    fn insert_rightmost(node: &mut NodePtr<V>, key: i64, value: V) -> InsertResult<V> {
        let (res, _) = Self::insert_rightmost_return_ptr(node, key, value);
        res
    }

    fn insert_rightmost_return_ptr(
        node: &mut NodePtr<V>,
        key: i64,
        value: V,
    ) -> (InsertResult<V>, *mut V) {
        let node = node.make_mut();

        if node.is_leaf() {
            node.push_leaf(key, value);

            // SAFETY: node is a leaf (verified above). We just pushed a value, so len >= 1.
            // We compute a pointer to the newly inserted value (at index len - 1).
            // If the node overflows (len > MAX_KEYS), we use rightmost split which moves
            // only the last element to the new right node, so the pointer is at right[0].
            // All pointer arithmetic stays within allocated bounds.
            unsafe {
                let len = node.len();
                let v_ptr =
                    (node.ptr.as_ptr() as *mut u8).add(NodePtr::<V>::values_offset()) as *mut V;
                let ptr = v_ptr.add(len - 1);

                if node.len() > MAX_KEYS {
                    // Use rightmost split: keeps MAX_KEYS in left, moves 1 to right
                    let (median, right) = node.split_leaf_rightmost();
                    // After rightmost split, the inserted value is at right[0]
                    let v_ptr_new = (right.ptr.as_ptr() as *mut u8)
                        .add(NodePtr::<V>::values_offset())
                        as *mut V;
                    (InsertResult::Split(median, right), v_ptr_new)
                } else {
                    (InsertResult::Done(None), ptr)
                }
            }
        } else {
            let last_idx = node.len();
            let child = node.child_mut(last_idx);
            let (result, ptr) = Self::insert_rightmost_return_ptr(child, key, value);

            match result {
                InsertResult::Done(old) => (InsertResult::Done(old), ptr),
                InsertResult::Split(median, right) => {
                    node.push_internal(median, right);

                    if node.len() > MAX_KEYS {
                        // Use rightmost split for internal nodes too
                        let (m, r) = node.split_internal_rightmost();
                        (InsertResult::Split(m, r), ptr)
                    } else {
                        (InsertResult::Done(None), ptr)
                    }
                }
            }
        }
    }

    fn insert_recursive(node: &mut NodePtr<V>, key: i64, value: V) -> InsertResult<V> {
        let node = node.make_mut();

        if node.is_leaf() {
            match node.search(key) {
                // SAFETY: node is a leaf (verified above). search returned Ok(i), meaning
                // key exists at index i (where i < node.len()). We read the old value via
                // ptr::read and write the new value via ptr::write. This is a replacement
                // of an existing value, so no len change is needed.
                Ok(i) => unsafe {
                    let v_ptr =
                        (node.ptr.as_ptr() as *mut u8).add(NodePtr::<V>::values_offset()) as *mut V;
                    let old = ptr::read(v_ptr.add(i));
                    ptr::write(v_ptr.add(i), value);
                    InsertResult::Done(Some(old))
                },
                Err(i) => {
                    node.insert_leaf(i, key, value);

                    if node.len() > MAX_KEYS {
                        // Optimization: if we inserted at the end, use rightmost split
                        // This handles interleaved sequential inserts that miss the global fast path
                        let (median, right) = if i == node.len() - 1 {
                            node.split_leaf_rightmost()
                        } else {
                            node.split_leaf()
                        };
                        InsertResult::Split(median, right)
                    } else {
                        InsertResult::Done(None)
                    }
                }
            }
        } else {
            let i = match node.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };

            let result = Self::insert_recursive(node.child_mut(i), key, value);

            match result {
                InsertResult::Done(old) => InsertResult::Done(old),
                InsertResult::Split(median, right) => {
                    node.insert_internal(i, median, right);

                    if node.len() > MAX_KEYS {
                        // Optimization: if we inserted at the end, use rightmost split
                        let (m, r) = if i == node.len() - 1 {
                            node.split_internal_rightmost()
                        } else {
                            node.split_internal()
                        };
                        InsertResult::Split(m, r)
                    } else {
                        InsertResult::Done(None)
                    }
                }
            }
        }
    }

    /// Remove a key. Returns the value if it existed.
    pub fn remove(&mut self, key: i64) -> Option<V> {
        let root = self.root.as_mut()?;
        let result = Self::remove_recursive(root, key);

        if result.is_some() {
            self.len -= 1;

            if self.len == 0 {
                self.root = None;
                self.max_key = 0;
            } else if self.max_key == key {
                self.refresh_max_key();
            }

            if let Some(ref mut root) = self.root {
                let root = root.make_mut();
                if !root.is_leaf() && root.len() == 0 {
                    let child_node = root.child_mut(0).clone();
                    self.root = Some(child_node);
                } else if root.is_leaf() && root.len() == 0 {
                    self.root = None;
                    self.max_key = 0;
                }
            }
        }

        result
    }

    /// Search and return path to the leaf.
    /// Returns (path_indices, leaf_search_result)
    /// path_indices: indices of children taken to reach the leaf
    fn search_path(&self, key: i64) -> (NodePath, Result<usize, usize>) {
        let mut path = NodePath::new();
        if self.root.is_none() {
            return (path, Err(0));
        }

        let mut node = self.root.as_ref().unwrap();
        loop {
            match node.search(key) {
                Ok(i) => {
                    if node.is_leaf() {
                        return (path, Ok(i));
                    }
                    path.push(i + 1);
                    node = node.child(i + 1);
                }
                Err(i) => {
                    if node.is_leaf() {
                        return (path, Err(i));
                    }
                    path.push(i);
                    node = node.child(i);
                }
            }
        }
    }

    /// Insert using pre-computed path and leaf index (avoids redundant search).
    /// `leaf_idx` is the index in the leaf where the key should be inserted.
    fn insert_with_path(
        node_ptr: &mut NodePtr<V>,
        key: i64,
        value: V,
        path: &NodePath,
        depth: usize,
        leaf_idx: usize,
    ) -> (InsertResult<V>, *mut V) {
        let node = node_ptr.make_mut();

        if node.is_leaf() {
            // Use pre-computed leaf_idx directly - no search needed
            let i = leaf_idx;
            node.insert_leaf(i, key, value);

            if node.len() > MAX_KEYS {
                let (median, right_node) = node.split_leaf();
                // Must match split_leaf's mid calculation: self.len() / 2
                // Before split, len was MAX_KEYS + 1, so mid = (MAX_KEYS + 1) / 2 = MAX_KEYS.div_ceil(2)
                let mid = MAX_KEYS.div_ceil(2);

                let ptr = if i < mid {
                    // SAFETY: After split, i < mid means the inserted value stayed in node.
                    // node is a valid leaf with i < node.len() after the split. We compute a
                    // pointer to the inserted value at index i.
                    unsafe {
                        let v_ptr = (node.ptr.as_ptr() as *mut u8)
                            .add(NodePtr::<V>::values_offset())
                            as *mut V;
                        v_ptr.add(i)
                    }
                } else {
                    let right_idx = i - mid;
                    // SAFETY: After split, i >= mid means the inserted value moved to right_node.
                    // right_node is a valid leaf with right_idx < right_node.len() after the split.
                    // We compute a pointer to the inserted value at index right_idx.
                    unsafe {
                        let v_ptr = (right_node.ptr.as_ptr() as *mut u8)
                            .add(NodePtr::<V>::values_offset())
                            as *mut V;
                        v_ptr.add(right_idx)
                    }
                };

                (InsertResult::Split(median, right_node), ptr)
            } else {
                // SAFETY: node is a leaf, we just inserted at index i (where i < node.len()).
                // We compute a pointer to the newly inserted value.
                unsafe {
                    let v_ptr =
                        (node.ptr.as_ptr() as *mut u8).add(NodePtr::<V>::values_offset()) as *mut V;
                    let ptr = v_ptr.add(i);
                    (InsertResult::Done(None), ptr)
                }
            }
        } else {
            let i = path.get(depth);

            let (result, ptr) =
                Self::insert_with_path(node.child_mut(i), key, value, path, depth + 1, leaf_idx);

            match result {
                InsertResult::Done(old) => (InsertResult::Done(old), ptr),
                InsertResult::Split(median, right) => {
                    node.insert_internal(i, median, right);

                    if node.len() > MAX_KEYS {
                        // Optimization: if we inserted at the end, use rightmost split
                        let (m, r) = if i == node.len() - 1 {
                            node.split_internal_rightmost()
                        } else {
                            node.split_internal()
                        };
                        (InsertResult::Split(m, r), ptr)
                    } else {
                        (InsertResult::Done(None), ptr)
                    }
                }
            }
        }
    }

    fn refresh_max_key(&mut self) {
        if let Some(root) = &self.root {
            let mut node = root;
            loop {
                if node.is_leaf() {
                    self.max_key = node.keys().last().copied().unwrap_or(0);
                    break;
                }
                node = node.children().last().unwrap();
            }
        } else {
            self.max_key = 0;
        }
    }

    fn remove_recursive(node: &mut NodePtr<V>, key: i64) -> Option<V> {
        let node = node.make_mut();

        if node.is_leaf() {
            match node.search(key) {
                Ok(i) => Some(node.remove_leaf(i)),
                Err(_) => None,
            }
        } else {
            let i = match node.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };

            if node.child(i).len() <= MIN_KEYS {
                Self::ensure_child_can_lose_key(node, i);
            }

            let new_i = match node.search(key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };

            let i = new_i.min(node.len()); // Children len is len+1. Max index len.
            Self::remove_recursive(node.child_mut(i), key)
        }
    }

    fn ensure_child_can_lose_key(node: &mut NodePtr<V>, i: usize) {
        let can_borrow_left = i > 0 && node.child(i - 1).len() > MIN_KEYS;
        let can_borrow_right = i < node.len() && node.child(i + 1).len() > MIN_KEYS;

        if can_borrow_left {
            node.borrow_from_left(i);
        } else if can_borrow_right {
            node.borrow_from_right(i);
        } else if i > 0 {
            node.merge_with_left(i);
        } else if i < node.len() {
            node.merge_with_right(i);
        }
    }

    /// Iterate over chunks of keys and values (O(1) amortized traversal)
    /// Yields `(&[i64], &[V])` slices directly from leaf nodes.
    pub fn iter_chunks(&self) -> impl Iterator<Item = (&[i64], &[V])> {
        CowBTreeChunkIter::new(self.root.as_ref())
    }

    /// Iterate over all key-value pairs in sorted order
    pub fn iter(&self) -> impl Iterator<Item = (&i64, &V)> {
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

    /// Iterate over chunks in reverse order (rightmost leaf to leftmost)
    pub fn iter_rev_chunks(&self) -> impl Iterator<Item = (&[i64], &[V])> {
        CowBTreeRevChunkIter::new(self.root.as_ref())
    }

    /// Iterate over all key-value pairs in reverse sorted order (largest to smallest)
    pub fn iter_rev(&self) -> impl Iterator<Item = (&i64, &V)> {
        self.iter_rev_chunks()
            .flat_map(|(keys, values)| keys.iter().zip(values.iter()).rev())
    }

    /// Yields chunks within the range in reverse order (rightmost matching leaf first).
    /// Each chunk is a slice from a single leaf node, keys in ascending order within the chunk.
    pub fn range_rev_chunks<R>(&self, range: R) -> impl Iterator<Item = (&[i64], &[V])>
    where
        R: std::ops::RangeBounds<i64>,
    {
        CowBTreeRevRangeChunkIter::new(self.root.as_ref(), range)
    }

    /// Iterator over a sub-range in reverse order (largest to smallest within range).
    pub fn range_rev<R>(&self, range: R) -> impl Iterator<Item = (&i64, &V)>
    where
        R: std::ops::RangeBounds<i64>,
    {
        self.range_rev_chunks(range)
            .flat_map(|(keys, values)| keys.iter().zip(values.iter()).rev())
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.root = None;
        self.max_key = 0;
        self.len = 0;
    }

    /// Returns the cached maximum key, or None if the tree is empty.
    #[inline]
    pub fn max_key(&self) -> Option<i64> {
        if self.root.is_some() {
            Some(self.max_key)
        } else {
            None
        }
    }

    fn insert_rightmost_entry(&mut self, key: i64, value: V) -> *mut V {
        let root = self.root.as_mut().unwrap();
        let (result, ptr) = Self::insert_rightmost_return_ptr(root, key, value);
        self.max_key = key;

        match result {
            InsertResult::Done(old) => {
                if old.is_none() {
                    self.len += 1;
                }
            }
            InsertResult::Split(median, right) => {
                let old_root = self.root.take().unwrap();
                let mut new_root = NodePtr::new_internal();
                // SAFETY: new_root is a freshly created internal node with len=0.
                // We write old_root to children[0]. Internal nodes have len+1 children,
                // so with len=0 we have space for 1 child at index 0. old_root is moved
                // (not cloned) into the slot. push_internal will add children[1] and set len=1.
                unsafe {
                    let c_ptr = (new_root.ptr.as_ptr() as *mut u8)
                        .add(NodePtr::<V>::children_offset())
                        as *mut NodePtr<V>;
                    ptr::write(c_ptr, old_root);
                }
                new_root.push_internal(median, right);
                self.root = Some(new_root);
                self.len += 1;
            }
        }

        ptr
    }

    /// Entry API for in-place updates.
    /// Optimized: single traversal, O(1) get().
    /// - entry(): 1 traversal
    /// - OccupiedEntry::get(): O(1) via cached pointer
    /// - OccupiedEntry::get_mut()/insert(): 1 traversal with COW
    pub fn entry(&mut self, key: i64) -> Entry<'_, V> {
        // Fast path for sequential append
        if self.is_key_greater_than_max(key) {
            return Entry::Vacant(VacantEntry {
                tree: self,
                key,
                path: NodePath::new(), // Dummy, not used for rightmost
                leaf_idx: 0,           // Dummy, not used for rightmost
                is_rightmost: true,
            });
        }

        let (path, leaf_result) = self.search_path(key);
        match leaf_result {
            Ok(idx) => Entry::Occupied(OccupiedEntry {
                tree: self,
                key,
                path,
                leaf_idx: idx,
            }),
            Err(idx) => Entry::Vacant(VacantEntry {
                tree: self,
                key,
                path,
                leaf_idx: idx,
                is_rightmost: false,
            }),
        }
    }

    fn insert_using_path(
        &mut self,
        key: i64,
        value: V,
        path: &NodePath,
        leaf_idx: usize,
    ) -> *mut V {
        if self.root.is_none() {
            self.insert(key, value);
            return self.get_mut(key).unwrap() as *mut V;
        }

        let root = self.root.as_mut().unwrap();
        let (result, ptr) = Self::insert_with_path(root, key, value, path, 0, leaf_idx);

        if key > self.max_key {
            self.max_key = key;
        }

        match result {
            InsertResult::Done(old) => {
                if old.is_none() {
                    self.len += 1;
                }
            }
            InsertResult::Split(median, right) => {
                let old_root = self.root.take().unwrap();
                let mut new_root = NodePtr::new_internal();
                // SAFETY: new_root is a freshly created internal node with len=0.
                // We write old_root to children[0]. Internal nodes have len+1 children,
                // so with len=0 we have space for 1 child at index 0. old_root is moved
                // (not cloned) into the slot. push_internal will add children[1] and set len=1.
                unsafe {
                    let c_ptr = (new_root.ptr.as_ptr() as *mut u8)
                        .add(NodePtr::<V>::children_offset())
                        as *mut NodePtr<V>;
                    ptr::write(c_ptr, old_root);
                }
                new_root.push_internal(median, right);
                self.root = Some(new_root);
                self.len += 1;
            }
        }

        ptr
    }
}

/// Entry API for CowBTree
pub enum Entry<'a, V: Clone> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, V>),
}

impl<'a, V: Clone> Entry<'a, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }

    pub fn key(&self) -> i64 {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }
}

/// An occupied entry in the CowBTree
pub struct OccupiedEntry<'a, V: Clone> {
    tree: &'a mut CowBTree<V>,
    key: i64,
    path: NodePath,
    leaf_idx: usize,
}

impl<'a, V: Clone> OccupiedEntry<'a, V> {
    #[inline]
    pub fn key(&self) -> i64 {
        self.key
    }

    /// O(log n) - traverses the cached path to reach the leaf
    #[inline]
    pub fn get(&self) -> &V {
        let mut node = self.tree.root.as_ref().unwrap();
        for idx in self.path.iter() {
            node = node.child(idx);
        }
        &node.values()[self.leaf_idx]
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.tree
            .get_mut_with_path(self.key, &self.path, self.leaf_idx)
            .unwrap()
    }

    pub fn into_mut(self) -> &'a mut V {
        self.tree
            .get_mut_with_path(self.key, &self.path, self.leaf_idx)
            .unwrap()
    }

    pub fn insert(&mut self, value: V) -> V {
        let node = self.tree.root.as_mut().unwrap();
        let mut node = node.make_mut();

        for idx in self.path.iter() {
            let child = node.child_mut(idx);
            node = child.make_mut();
        }

        // SAFETY: node is a leaf (we followed the path to a leaf). self.leaf_idx is the
        // index where the key was found during entry lookup, so it's valid (< node.len()).
        // We read the old value via ptr::read and write the new value via ptr::write.
        // This is a replacement of an existing value, so no len change is needed.
        unsafe {
            let v_ptr = (node.ptr.as_ptr() as *mut u8).add(NodePtr::<V>::values_offset()) as *mut V;
            let ptr = v_ptr.add(self.leaf_idx);
            let old = ptr::read(ptr);
            ptr::write(ptr, value);
            old
        }
    }
}

/// A vacant entry in the CowBTree
pub struct VacantEntry<'a, V: Clone> {
    tree: &'a mut CowBTree<V>,
    key: i64,
    path: NodePath,
    /// Index in the leaf node where the key should be inserted
    leaf_idx: usize,
    /// Optimization: true if this entry represents a sequential insert (key > max_key)
    is_rightmost: bool,
}

impl<'a, V: Clone> VacantEntry<'a, V> {
    #[inline]
    pub fn key(&self) -> i64 {
        self.key
    }

    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        if self.is_rightmost {
            let ptr = self.tree.insert_rightmost_entry(self.key, value);
            // SAFETY: insert_rightmost_entry returns a valid pointer to the newly inserted
            // value. The pointer remains valid for the lifetime 'a because we have exclusive
            // access to the tree (&'a mut). The insert functions guarantee the pointer points
            // to initialized, properly aligned memory within a leaf node.
            unsafe { &mut *ptr }
        } else {
            let ptr = self
                .tree
                .insert_using_path(self.key, value, &self.path, self.leaf_idx);
            // SAFETY: insert_using_path returns a valid pointer to the newly inserted value.
            // The pointer remains valid for the lifetime 'a because we have exclusive access
            // to the tree (&'a mut). The insert functions guarantee the pointer points to
            // initialized, properly aligned memory within a leaf node.
            unsafe { &mut *ptr }
        }
    }
}

enum InsertResult<V: Clone> {
    Done(Option<V>),
    Split(i64, NodePtr<V>),
}

/// Iterator over chunks of a CowBTree (leaf slices)
struct CowBTreeChunkIter<'a, V: Clone> {
    /// Stack of (node, next_child_index) for traversal
    /// Only holds internal nodes.
    stack: Vec<(&'a NodePtr<V>, usize)>,
    /// Current leaf node being yielded (if any)
    current_leaf: Option<&'a NodePtr<V>>,
}

impl<'a, V: Clone> CowBTreeChunkIter<'a, V> {
    fn new(root: Option<&'a NodePtr<V>>) -> Self {
        let mut iter = Self {
            stack: Vec::new(),
            current_leaf: None,
        };
        if let Some(root) = root {
            iter.descend_to_leftmost(root);
        }
        iter
    }

    /// Descend to the leftmost leaf, pushing internal nodes onto the stack
    fn descend_to_leftmost(&mut self, mut node: &'a NodePtr<V>) {
        while !node.is_leaf() {
            self.stack.push((node, 1));
            node = node.child(0);
        }
        self.current_leaf = Some(node);
    }
}

impl<'a, V: Clone> Iterator for CowBTreeChunkIter<'a, V> {
    type Item = (&'a [i64], &'a [V]);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(leaf) = self.current_leaf.take() {
            return Some((leaf.keys(), leaf.values()));
        }

        loop {
            let (node, idx) = self.stack.last_mut()?;

            if *idx < node.len() + 1 {
                let child_idx = *idx;
                *idx += 1;
                let child = node.child(child_idx);
                self.descend_to_leftmost(child);

                if let Some(leaf) = self.current_leaf.take() {
                    return Some((leaf.keys(), leaf.values()));
                }
            } else {
                self.stack.pop();
            }
        }
    }
}

/// Range iterator over a CowBTree yielding chunks
/// Optimized: Seeks directly to start bound and yields slices
struct CowBTreeRangeChunkIter<'a, V: Clone, R> {
    stack: Vec<(&'a NodePtr<V>, usize)>,
    range: R,
    current_leaf: Option<&'a NodePtr<V>>,
    current_idx: usize,
    finished: bool,
}

impl<'a, V: Clone, R: std::ops::RangeBounds<i64>> CowBTreeRangeChunkIter<'a, V, R> {
    fn new(root: Option<&'a NodePtr<V>>, range: R) -> Self {
        let mut iter = Self {
            stack: Vec::new(),
            range,
            current_leaf: None,
            current_idx: 0,
            finished: false,
        };
        if let Some(root) = root {
            iter.seek_to_start(root);
        } else {
            iter.finished = true;
        }
        iter
    }

    fn seek_to_start(&mut self, mut node: &'a NodePtr<V>) {
        let start_key = match self.range.start_bound() {
            Bound::Included(&k) => Some(k),
            Bound::Excluded(&k) => Some(k),
            Bound::Unbounded => None,
        };

        loop {
            if node.is_leaf() {
                let keys = node.keys();
                let mut idx = if let Some(k) = start_key {
                    match keys.binary_search(&k) {
                        Ok(i) => i,
                        Err(i) => i,
                    }
                } else {
                    0
                };

                if let Bound::Excluded(&k) = self.range.start_bound() {
                    if idx < keys.len() && keys[idx] == k {
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
                node = node.child(idx);
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
            if let Some(leaf) = self.current_leaf {
                let keys = leaf.keys();
                let values = leaf.values();
                if self.current_idx < keys.len() {
                    let start = self.current_idx;
                    let end = match self.range.end_bound() {
                        Bound::Unbounded => keys.len(),
                        Bound::Included(&k) => {
                            if keys.last().unwrap() <= &k {
                                keys.len()
                            } else {
                                let pos = keys[start..].partition_point(|&x| x <= k);
                                self.finished = true;
                                start + pos
                            }
                        }
                        Bound::Excluded(&k) => {
                            if keys.last().unwrap() < &k {
                                keys.len()
                            } else {
                                let pos = keys[start..].partition_point(|&x| x < k);
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
                    let result = (&keys[start..end], &values[start..end]);

                    if end == keys.len() && !self.finished {
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

            if let Some((node, idx)) = self.stack.last_mut() {
                if *idx < node.len() + 1 {
                    let child_idx = *idx;
                    *idx += 1;
                    let mut child = node.child(child_idx);

                    loop {
                        if child.is_leaf() {
                            self.current_leaf = Some(child);
                            self.current_idx = 0;
                            break;
                        } else {
                            self.stack.push((child, 1));
                            child = child.child(0);
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

/// Reverse iterator over a CowBTree yielding chunks from rightmost leaf to leftmost
struct CowBTreeRevChunkIter<'a, V: Clone> {
    /// Stack of (node, next_child_plus_one) for reverse traversal.
    /// next_child_plus_one == 0 means no more children to visit at this level.
    stack: Vec<(&'a NodePtr<V>, usize)>,
    /// Current leaf node being yielded
    current_leaf: Option<&'a NodePtr<V>>,
}

impl<'a, V: Clone> CowBTreeRevChunkIter<'a, V> {
    fn new(root: Option<&'a NodePtr<V>>) -> Self {
        let mut iter = Self {
            stack: Vec::new(),
            current_leaf: None,
        };
        if let Some(root) = root {
            iter.descend_to_rightmost(root);
        }
        iter
    }

    /// Descend to the rightmost leaf, pushing internal nodes onto the stack
    fn descend_to_rightmost(&mut self, mut node: &'a NodePtr<V>) {
        while !node.is_leaf() {
            let last_child = node.len(); // children indices: 0..=len
                                         // After visiting child(last_child), next to visit going left is child(last_child-1)
                                         // Store last_child as next_child_plus_one
            self.stack.push((node, last_child));
            node = node.child(last_child);
        }
        self.current_leaf = Some(node);
    }
}

impl<'a, V: Clone> Iterator for CowBTreeRevChunkIter<'a, V> {
    type Item = (&'a [i64], &'a [V]);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(leaf) = self.current_leaf.take() {
            return Some((leaf.keys(), leaf.values()));
        }

        loop {
            let (node, next_plus_one) = self.stack.last_mut()?;

            if *next_plus_one > 0 {
                let child_idx = *next_plus_one - 1;
                *next_plus_one = child_idx;
                let child = node.child(child_idx);
                self.descend_to_rightmost(child);

                if let Some(leaf) = self.current_leaf.take() {
                    return Some((leaf.keys(), leaf.values()));
                }
            } else {
                self.stack.pop();
            }
        }
    }
}

/// Reverse range iterator over a CowBTree yielding chunks from end bound to start bound.
/// Each chunk contains keys in ascending order; the consumer should reverse within each chunk.
struct CowBTreeRevRangeChunkIter<'a, V: Clone, R> {
    stack: Vec<(&'a NodePtr<V>, usize)>,
    range: R,
    current_leaf: Option<&'a NodePtr<V>>,
    /// End index (exclusive) within current leaf
    current_end_idx: usize,
    finished: bool,
}

impl<'a, V: Clone, R: std::ops::RangeBounds<i64>> CowBTreeRevRangeChunkIter<'a, V, R> {
    fn new(root: Option<&'a NodePtr<V>>, range: R) -> Self {
        let mut iter = Self {
            stack: Vec::new(),
            range,
            current_leaf: None,
            current_end_idx: 0,
            finished: false,
        };
        if let Some(root) = root {
            iter.seek_to_end(root);
        } else {
            iter.finished = true;
        }
        iter
    }

    /// Descend to the rightmost leaf, setting current_end_idx to the full leaf length
    fn descend_to_rightmost(&mut self, mut node: &'a NodePtr<V>) {
        while !node.is_leaf() {
            let last_child = node.len();
            self.stack.push((node, last_child));
            node = node.child(last_child);
        }
        self.current_leaf = Some(node);
        self.current_end_idx = node.len();
    }

    /// Seek to the end bound of the range (the starting point for reverse iteration)
    fn seek_to_end(&mut self, mut node: &'a NodePtr<V>) {
        let end_key = match self.range.end_bound() {
            Bound::Included(&k) | Bound::Excluded(&k) => Some(k),
            Bound::Unbounded => None,
        };

        if end_key.is_none() {
            // Unbounded upper: start from rightmost leaf
            self.descend_to_rightmost(node);
            return;
        }

        let k = end_key.unwrap();

        loop {
            if node.is_leaf() {
                let keys = node.keys();
                let idx = match keys.binary_search(&k) {
                    Ok(i) => match self.range.end_bound() {
                        Bound::Included(_) => i + 1,
                        _ => i,
                    },
                    Err(i) => i,
                };

                if idx > 0 {
                    self.current_leaf = Some(node);
                    self.current_end_idx = idx;
                }
                // If idx == 0, no valid entries in this leaf for our range.
                // The first next() call will navigate to the previous leaf.
                break;
            } else {
                let child_idx = match node.search(k) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                };
                // Store child_idx as next_child_plus_one: children before child_idx
                self.stack.push((node, child_idx));
                node = node.child(child_idx);
            }
        }
    }
}

impl<'a, V: Clone, R: std::ops::RangeBounds<i64>> Iterator for CowBTreeRevRangeChunkIter<'a, V, R> {
    type Item = (&'a [i64], &'a [V]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            if let Some(leaf) = self.current_leaf {
                let keys = leaf.keys();
                let values = leaf.values();
                let end = self.current_end_idx;

                // Check start bound (lower bound) - trim from the left
                let start = if end == 0 {
                    end // Empty chunk, will be skipped
                } else {
                    match self.range.start_bound() {
                        Bound::Unbounded => 0,
                        Bound::Included(&k) => {
                            if keys[0] >= k {
                                0 // Entire chunk is within range
                            } else {
                                self.finished = true;
                                keys[..end].partition_point(|&x| x < k)
                            }
                        }
                        Bound::Excluded(&k) => {
                            if keys[0] > k {
                                0
                            } else {
                                self.finished = true;
                                keys[..end].partition_point(|&x| x <= k)
                            }
                        }
                    }
                };

                self.current_leaf = None;

                if start < end {
                    return Some((&keys[start..end], &values[start..end]));
                }

                if self.finished {
                    return None;
                }
            }

            // Navigate to previous leaf
            let (node, next_plus_one) = self.stack.last_mut()?;

            if *next_plus_one > 0 {
                let child_idx = *next_plus_one - 1;
                *next_plus_one = child_idx;
                let child = node.child(child_idx);
                self.descend_to_rightmost(child);
            } else {
                self.stack.pop();
            }
        }
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

        let snapshot = tree.clone();

        tree.insert(50, 999);
        tree.insert(200, 2000);

        assert_eq!(snapshot.get(50), Some(&500));
        assert_eq!(snapshot.get(200), None);

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

        for i in (0..1000).step_by(2) {
            assert_eq!(tree.remove(i), Some(i));
        }

        assert_eq!(tree.len(), 500);

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

        if let Some(v) = tree.get_mut(50) {
            *v = 999;
        }

        assert_eq!(snapshot.get(50), Some(&50));
        assert_eq!(tree.get(50), Some(&999));
    }

    #[test]
    fn test_memory_size() {
        assert!(std::mem::size_of::<CowBTree<i64>>() <= 24);
    }

    #[test]
    fn test_entry_api() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        let v = tree.entry(1).or_insert(10);
        assert_eq!(*v, 10);
        assert_eq!(tree.get(1), Some(&10));

        let v = tree.entry(1).or_insert(20);
        assert_eq!(*v, 10);

        tree.entry(1).and_modify(|v| *v += 5);
        assert_eq!(tree.get(1), Some(&15));

        for i in 2..100 {
            tree.entry(i).or_insert(i * 10);
        }
        assert_eq!(tree.len(), 99);
        assert_eq!(tree.get(50), Some(&500));

        tree.entry(50).and_modify(|v| *v = 999);
        assert_eq!(tree.get(50), Some(&999));
    }

    #[test]
    fn test_entry_vacant() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        match tree.entry(42) {
            Entry::Occupied(_) => panic!("Expected vacant"),
            Entry::Vacant(v) => {
                assert_eq!(v.key(), 42);
                let val = v.insert(420);
                assert_eq!(*val, 420);
            }
        }
        assert_eq!(tree.get(42), Some(&420));
        assert_eq!(tree.len(), 1);

        tree.insert(10, 100);
        tree.insert(50, 500);

        match tree.entry(30) {
            Entry::Occupied(_) => panic!("Expected vacant"),
            Entry::Vacant(v) => {
                assert_eq!(v.key(), 30);
                v.insert(300);
            }
        }
        assert_eq!(tree.get(30), Some(&300));
        assert_eq!(tree.len(), 4);

        match tree.entry(5) {
            Entry::Occupied(_) => panic!("Expected vacant"),
            Entry::Vacant(v) => {
                v.insert(50);
            }
        }
        match tree.entry(100) {
            Entry::Occupied(_) => panic!("Expected vacant"),
            Entry::Vacant(v) => {
                v.insert(1000);
            }
        }
        assert_eq!(tree.get(5), Some(&50));
        assert_eq!(tree.get(100), Some(&1000));
    }

    #[test]
    fn test_entry_occupied() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i, i * 10);
        }

        match tree.entry(50) {
            Entry::Occupied(o) => {
                assert_eq!(o.key(), 50);
            }
            Entry::Vacant(_) => panic!("Expected occupied"),
        }

        match tree.entry(50) {
            Entry::Occupied(o) => {
                assert_eq!(*o.get(), 500);
            }
            Entry::Vacant(_) => panic!("Expected occupied"),
        }

        match tree.entry(50) {
            Entry::Occupied(mut o) => {
                *o.get_mut() = 5000;
            }
            Entry::Vacant(_) => panic!("Expected occupied"),
        }
        assert_eq!(tree.get(50), Some(&5000));

        match tree.entry(50) {
            Entry::Occupied(mut o) => {
                let old = o.insert(50000);
                assert_eq!(old, 5000);
            }
            Entry::Vacant(_) => panic!("Expected occupied"),
        }
        assert_eq!(tree.get(50), Some(&50000));

        let val_ref = match tree.entry(50) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(_) => panic!("Expected occupied"),
        };
        *val_ref = 500000;
        assert_eq!(tree.get(50), Some(&500000));
    }

    #[test]
    fn test_entry_cow_semantics() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i, i);
        }

        let snapshot = tree.clone();

        match tree.entry(50) {
            Entry::Occupied(mut o) => {
                o.insert(999);
            }
            Entry::Vacant(_) => panic!("Expected occupied"),
        }

        match tree.entry(200) {
            Entry::Occupied(_) => panic!("Expected vacant"),
            Entry::Vacant(v) => {
                v.insert(2000);
            }
        }

        assert_eq!(snapshot.get(50), Some(&50));
        assert_eq!(snapshot.get(200), None);
        assert_eq!(snapshot.len(), 100);

        assert_eq!(tree.get(50), Some(&999));
        assert_eq!(tree.get(200), Some(&2000));
        assert_eq!(tree.len(), 101);
    }

    #[test]
    fn test_entry_many_inserts() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..1000 {
            match tree.entry(i) {
                Entry::Occupied(_) => panic!("Should be vacant"),
                Entry::Vacant(v) => {
                    v.insert(i * 10);
                }
            }
        }

        assert_eq!(tree.len(), 1000);

        for i in 0..1000 {
            match tree.entry(i) {
                Entry::Occupied(mut o) => {
                    assert_eq!(*o.get(), i * 10);
                    o.insert(i * 20);
                }
                Entry::Vacant(_) => panic!("Should be occupied"),
            }
        }

        for i in 0..1000 {
            assert_eq!(tree.get(i), Some(&(i * 20)));
        }
    }

    #[test]
    fn test_node_path_limit() {
        let mut path = NodePath::new();
        for i in 0..MAX_TREE_DEPTH {
            path.push(i);
        }
        assert_eq!(path.len, MAX_TREE_DEPTH as u8);
    }

    #[test]
    fn test_entry_sequential_optimization() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        tree.insert(0, 0);

        for i in 1..1000 {
            tree.entry(i).or_insert(i * 10);
        }

        assert_eq!(tree.len(), 1000);
        assert_eq!(tree.get(999), Some(&9990));
        assert_eq!(tree.max_key, 999);
    }

    #[test]
    fn test_drop_is_called() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone)]
        #[allow(dead_code)]
        struct DropCounter(i64);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        {
            let mut tree: CowBTree<DropCounter> = CowBTree::new();
            for i in 0..100 {
                tree.insert(i, DropCounter(i));
            }
            assert_eq!(tree.len(), 100);
        }

        let drops = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(drops, 100, "Expected 100 drops, got {}", drops);
    }

    #[test]
    fn test_drop_with_splits() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone)]
        #[allow(dead_code)]
        struct DropCounter(i64);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        {
            let mut tree: CowBTree<DropCounter> = CowBTree::new();
            for i in 0..500 {
                tree.insert(i, DropCounter(i));
            }
            assert_eq!(tree.len(), 500);
        }

        let drops = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(drops, 500, "Expected 500 drops, got {}", drops);
    }

    #[test]
    fn test_drop_with_clone() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone)]
        #[allow(dead_code)]
        struct DropCounter(i64);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        {
            let mut tree1: CowBTree<DropCounter> = CowBTree::new();
            for i in 0..100 {
                tree1.insert(i, DropCounter(i));
            }

            let tree2 = tree1.clone();
            assert_eq!(tree1.len(), 100);
            assert_eq!(tree2.len(), 100);

            drop(tree1);
            assert_eq!(
                DROP_COUNT.load(Ordering::SeqCst),
                0,
                "Data should not be dropped while clone exists"
            );
            drop(tree2);
        }

        let drops = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            drops, 100,
            "Expected 100 drops after both trees dropped, got {}",
            drops
        );
    }

    #[test]
    fn test_rightmost_split_optimization() {
        // Test that sequential inserts use rightmost split optimization
        // which achieves ~100% fill factor instead of 50%
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert enough to cause multiple splits
        // With MAX_KEYS=128, after 1000 inserts we should have ~8 leaf nodes
        // With standard 50% split: ~16 leaf nodes at 50% fill
        // With rightmost split: ~8 leaf nodes at ~100% fill
        for i in 0..1000 {
            tree.insert(i, i * 10);
        }

        assert_eq!(tree.len(), 1000);

        // Verify all values are correct
        for i in 0..1000 {
            assert_eq!(tree.get(i), Some(&(i * 10)), "Failed at key {}", i);
        }

        // Count leaf nodes and check fill factor
        let mut leaf_count = 0;
        let mut total_keys = 0;
        for (keys, _) in tree.iter_chunks() {
            leaf_count += 1;
            total_keys += keys.len();

            // All but the last (rightmost) leaf should be nearly full (MAX_KEYS)
            // The rightmost leaf can have 1..=MAX_KEYS keys
            // With rightmost split, non-rightmost leaves have exactly MAX_KEYS
        }

        assert_eq!(total_keys, 1000);

        // With rightmost split optimization:
        // - 1000 keys, MAX_KEYS=128
        // - First 7 leaves should have 128 keys each = 896 keys
        // - Last leaf should have 104 keys
        // - Total: 8 leaves
        //
        // Without optimization (50% split):
        // - Would have ~16 leaves at ~64 keys each

        // We expect 8 leaves (1000 / 128 = 7.8, rounded up)
        // Allow some variance, but should be much less than 16
        assert!(
            leaf_count <= 10,
            "Expected ~8 leaf nodes with rightmost split, got {} (suggests 50% split is being used)",
            leaf_count
        );

        // Verify fill factor: total_keys / (leaf_count * MAX_KEYS) should be > 75%
        let fill_factor = (total_keys as f64) / (leaf_count as f64 * MAX_KEYS as f64);
        assert!(
            fill_factor > 0.75,
            "Expected fill factor > 75%, got {:.1}%",
            fill_factor * 100.0
        );
    }

    #[test]
    fn test_rightmost_split_with_entry_api() {
        // Test that entry API also benefits from rightmost split
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..1000 {
            tree.entry(i).or_insert(i * 10);
        }

        assert_eq!(tree.len(), 1000);

        // Count leaves - should match the insert test
        let leaf_count = tree.iter_chunks().count();
        assert!(
            leaf_count <= 10,
            "Expected ~8 leaf nodes with rightmost split via entry API, got {}",
            leaf_count
        );
    }

    #[test]
    fn test_clear() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 100);
        assert!(!tree.is_empty());

        tree.clear();
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert_eq!(tree.get(50), None);

        // Can insert after clear
        tree.insert(1, 10);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.get(1), Some(&10));
    }

    #[test]
    fn test_values_iterator() {
        let mut tree: CowBTree<String> = CowBTree::new();

        tree.insert(3, "three".to_string());
        tree.insert(1, "one".to_string());
        tree.insert(2, "two".to_string());

        let values: Vec<&String> = tree.values().collect();
        // Values should be in key-sorted order
        assert_eq!(values, vec!["one", "two", "three"]);
    }

    #[test]
    fn test_range_unbounded() {
        use std::ops::Bound;

        let mut tree: CowBTree<i64> = CowBTree::new();
        for i in 0..100 {
            tree.insert(i, i);
        }

        // Unbounded start: ..50
        let range: Vec<i64> = tree.range(..50).map(|(k, _)| *k).collect();
        assert_eq!(range.len(), 50);
        assert_eq!(range[0], 0);
        assert_eq!(range[49], 49);

        // Unbounded end: 50..
        let range: Vec<i64> = tree.range(50..).map(|(k, _)| *k).collect();
        assert_eq!(range.len(), 50);
        assert_eq!(range[0], 50);
        assert_eq!(range[49], 99);

        // Fully unbounded: ..
        let range: Vec<i64> = tree.range(..).map(|(k, _)| *k).collect();
        assert_eq!(range.len(), 100);

        // RangeInclusive: 25..=75
        let range: Vec<i64> = tree.range(25..=75).map(|(k, _)| *k).collect();
        assert_eq!(range.len(), 51);
        assert_eq!(range[0], 25);
        assert_eq!(range[50], 75);

        // Custom bounds with Excluded start
        let range: Vec<i64> = tree
            .range((Bound::Excluded(25), Bound::Included(30)))
            .map(|(k, _)| *k)
            .collect();
        assert_eq!(range, vec![26, 27, 28, 29, 30]);
    }

    #[test]
    fn test_deletion_triggers_borrow_from_left() {
        // To trigger borrow_from_left, we need:
        // 1. A leaf node that falls below MIN_KEYS after deletion
        // 2. A left sibling with spare keys (more than MIN_KEYS)
        //
        // Strategy: Insert keys to create specific tree structure,
        // then delete from the right leaf to trigger borrowing from left.
        let mut tree: CowBTree<i64> = CowBTree::new();

        // With MAX_KEYS=128, MIN_KEYS=64
        // Insert 200 keys sequentially to get 2 leaves (128 + 72)
        for i in 0..200 {
            tree.insert(i, i);
        }

        // Now delete keys from the second leaf (keys 128-199) until it has < 64 keys
        // We need to delete more than 72 - 64 = 8 keys from the right leaf
        // to trigger underflow. Delete keys 190-199 (10 keys from end).
        for i in 190..200 {
            tree.remove(i);
        }

        // Should still have 190 keys
        assert_eq!(tree.len(), 190);

        // Continue deleting to trigger rebalancing
        // Delete more keys from the second leaf
        for i in 170..190 {
            tree.remove(i);
        }

        // Should have 170 keys
        assert_eq!(tree.len(), 170);

        // Verify tree integrity - all remaining keys should be accessible
        for i in 0..170 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_deletion_triggers_borrow_from_right() {
        // To trigger borrow_from_right, we need:
        // 1. A leaf node that falls below MIN_KEYS after deletion
        // 2. A right sibling with spare keys
        //
        // Strategy: Create tree, then delete from the LEFT leaf
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert 200 keys to get 2 leaves
        for i in 0..200 {
            tree.insert(i, i);
        }

        // Delete keys from the first leaf (keys 0-127) to trigger underflow
        // Delete keys 0-70 to make first leaf have < 64 keys
        for i in 0..71 {
            tree.remove(i);
        }

        // Should have 129 keys (200 - 71)
        assert_eq!(tree.len(), 129);

        // Verify tree integrity
        for i in 71..200 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_deletion_triggers_merge() {
        // To trigger merge, we need:
        // 1. A leaf node below MIN_KEYS
        // 2. A sibling also at or below MIN_KEYS (no spare keys to borrow)
        //
        // Strategy: Create minimal tree, delete until both siblings are at MIN_KEYS,
        // then delete one more to trigger merge.
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert 150 keys to create 2 leaves with ~75 each after split
        // Actually with rightmost split: first leaf has 128, second has 22
        // Let's insert more to have a more balanced scenario
        for i in 0..300 {
            tree.insert(i, i);
        }

        // Delete many keys to force merges
        // Delete all even keys first
        for i in (0..300).step_by(2) {
            tree.remove(i);
        }

        // Should have 150 keys
        assert_eq!(tree.len(), 150);

        // Delete more to trigger merges
        for i in (1..300).step_by(4) {
            tree.remove(i);
        }

        // Verify remaining keys
        let remaining: Vec<i64> = tree.keys().collect();
        for key in &remaining {
            assert_eq!(tree.get(*key), Some(key));
        }
    }

    #[test]
    fn test_heavy_deletion_rebalancing() {
        // Stress test for deletion rebalancing
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert 1000 keys
        for i in 0..1000 {
            tree.insert(i, i);
        }

        // Delete in a pattern that forces various rebalancing scenarios
        // Delete every 3rd key starting from 0
        for i in (0..1000).step_by(3) {
            tree.remove(i);
        }

        // Delete every 3rd key starting from 1
        for i in (1..1000).step_by(3) {
            tree.remove(i);
        }

        // Only keys where i % 3 == 2 remain (333 keys)
        let expected_count = (0..1000).filter(|i| i % 3 == 2).count();
        assert_eq!(tree.len(), expected_count);

        // Verify all remaining keys
        for i in 0..1000 {
            if i % 3 == 2 {
                assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
            } else {
                assert_eq!(tree.get(i), None, "Unexpected key {}", i);
            }
        }
    }

    #[test]
    fn test_delete_all_keys() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..500 {
            tree.insert(i, i);
        }

        // Delete all keys
        for i in 0..500 {
            assert_eq!(tree.remove(i), Some(i), "Failed to remove key {}", i);
        }

        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        // Can reuse the tree
        tree.insert(1, 1);
        assert_eq!(tree.get(1), Some(&1));
    }

    #[test]
    fn test_remove_from_internal_node() {
        // When removing a key that exists in an internal node (as separator),
        // the algorithm must replace it with predecessor/successor
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert enough to create internal nodes
        for i in 0..500 {
            tree.insert(i, i);
        }

        // Delete keys in the middle that might be internal node separators
        for i in 100..200 {
            assert_eq!(tree.remove(i), Some(i));
        }

        // Verify tree integrity
        for i in 0..100 {
            assert_eq!(tree.get(i), Some(&i));
        }
        for i in 100..200 {
            assert_eq!(tree.get(i), None);
        }
        for i in 200..500 {
            assert_eq!(tree.get(i), Some(&i));
        }
    }

    #[test]
    fn test_cow_with_deletion() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i, i);
        }

        let snapshot = tree.clone();

        // Delete from original
        for i in 0..50 {
            tree.remove(i);
        }

        // Snapshot should still have all keys
        assert_eq!(snapshot.len(), 100);
        for i in 0..100 {
            assert_eq!(snapshot.get(i), Some(&i));
        }

        // Original should have only keys 50-99
        assert_eq!(tree.len(), 50);
        for i in 0..50 {
            assert_eq!(tree.get(i), None);
        }
        for i in 50..100 {
            assert_eq!(tree.get(i), Some(&i));
        }
    }

    #[test]
    fn test_entry_key_method() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        tree.insert(10, 100);
        tree.insert(20, 200);

        // Test Entry::key() on occupied entry
        let key = tree.entry(10).key();
        assert_eq!(key, 10);

        // Test Entry::key() on vacant entry
        let key = tree.entry(15).key();
        assert_eq!(key, 15);

        // Verify tree wasn't modified by key() calls
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_and_modify_on_vacant() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        tree.insert(10, 100);

        // and_modify on vacant entry should return Vacant unchanged
        let entry = tree.entry(20).and_modify(|v| *v += 1);
        match entry {
            Entry::Vacant(v) => {
                assert_eq!(v.key(), 20);
                v.insert(200);
            }
            Entry::Occupied(_) => panic!("Expected Vacant after and_modify on non-existent key"),
        }

        assert_eq!(tree.get(20), Some(&200));

        // and_modify on occupied entry should modify the value
        tree.entry(10).and_modify(|v| *v += 1);
        assert_eq!(tree.get(10), Some(&101));
    }

    #[test]
    fn test_entry_api_root_split() {
        // Test that entry API correctly handles root splits
        // This happens when inserting via entry() causes the tree to grow a new level
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert enough keys to cause multiple splits via entry API
        // MAX_KEYS = 128, so after 128 inserts we get first split
        for i in 0..500 {
            tree.entry(i).or_insert(i * 10);
        }

        assert_eq!(tree.len(), 500);

        // Verify all values
        for i in 0..500 {
            assert_eq!(tree.get(i), Some(&(i * 10)));
        }
    }

    #[test]
    fn test_deep_tree_internal_node_split() {
        // Create a tree with 3+ levels to trigger internal node splitting
        // With MAX_KEYS=128, we need more than 128*128 = 16384 keys for 3 levels
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert 20000 keys to ensure 3-level tree
        for i in 0..20_000 {
            tree.insert(i, i);
        }

        assert_eq!(tree.len(), 20_000);

        // Verify random samples
        for i in (0..20_000).step_by(100) {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_deep_tree_internal_node_rebalancing() {
        // Create a deep tree and then delete keys to trigger internal node rebalancing
        // This tests borrow_from_left/right and merge operations on internal nodes
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert 20000 keys to create 3-level tree
        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Delete keys to trigger internal node underflow and rebalancing
        // Delete from the middle to maximize rebalancing
        for i in 5000..15000 {
            tree.remove(i);
        }

        assert_eq!(tree.len(), 10_000);

        // Verify remaining keys
        for i in 0..5000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
        for i in 5000..15000 {
            assert_eq!(tree.get(i), None, "Key {} should be deleted", i);
        }
        for i in 15000..20000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_deep_tree_random_deletion() {
        // Random deletion pattern to exercise various internal node operations
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Delete in a pattern that causes internal node rebalancing
        // Delete every 3rd key
        for i in (0..20_000).step_by(3) {
            tree.remove(i);
        }

        // Delete every 5th key of remaining
        for i in (1..20_000).step_by(5) {
            tree.remove(i);
        }

        // Verify integrity - remaining keys should still be accessible
        let remaining: Vec<i64> = tree.keys().collect();
        for key in &remaining {
            assert_eq!(tree.get(*key), Some(key));
        }
    }

    #[test]
    fn test_deep_tree_cow_semantics() {
        // Test COW semantics with deep tree
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20_000 {
            tree.insert(i, i);
        }

        let snapshot = tree.clone();

        // Modify original
        for i in 0..5000 {
            tree.remove(i);
        }
        tree.insert(25000, 25000);

        // Snapshot should be unchanged
        assert_eq!(snapshot.len(), 20_000);
        assert_eq!(snapshot.get(0), Some(&0));
        assert_eq!(snapshot.get(25000), None);

        // Original should have modifications
        assert_eq!(tree.len(), 15_001);
        assert_eq!(tree.get(0), None);
        assert_eq!(tree.get(25000), Some(&25000));
    }

    #[test]
    fn test_default_trait() {
        let tree: CowBTree<i64> = CowBTree::default();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_empty_tree_range_iteration() {
        let tree: CowBTree<i64> = CowBTree::new();

        // Range on empty tree should return empty iterator
        let range: Vec<_> = tree.range(..).collect();
        assert!(range.is_empty());

        let range: Vec<_> = tree.range(0..100).collect();
        assert!(range.is_empty());

        let range: Vec<_> = tree.range(..50).collect();
        assert!(range.is_empty());

        // iter_chunks on empty tree
        let chunks: Vec<_> = tree.iter_chunks().collect();
        assert!(chunks.is_empty());

        // range_chunks on empty tree
        let chunks: Vec<_> = tree.range_chunks(0..100).collect();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_random_order_entry_api() {
        // Random order inserts via entry API triggers split_internal (non-rightmost path)
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Use a pseudo-random pattern to insert keys out of order
        let keys: Vec<i64> = (0..1000).map(|i| (i * 7919 + 13) % 10000).collect();

        for &k in &keys {
            tree.entry(k).or_insert(k * 10);
        }

        // Verify all inserted values
        for &k in &keys {
            assert_eq!(tree.get(k), Some(&(k * 10)), "Missing key {}", k);
        }
    }

    #[test]
    fn test_random_order_entry_api_large() {
        // Larger random order inserts to ensure internal node splits
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Generate keys in random order
        let keys: Vec<i64> = (0..5000).map(|i| (i * 7919 + 13) % 50000).collect();

        for &k in &keys {
            tree.entry(k).or_insert(k);
        }

        // Verify all keys are present
        for &k in &keys {
            assert!(tree.contains_key(k), "Missing key {}", k);
        }
    }

    #[test]
    fn test_range_on_deep_tree() {
        // Range iteration on a deep tree exercises seek_to_start internal node path
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Range in the middle of the tree
        let range: Vec<i64> = tree.range(5000..5100).map(|(k, _)| *k).collect();
        assert_eq!(range.len(), 100);
        assert_eq!(range[0], 5000);
        assert_eq!(range[99], 5099);

        // Range with excluded start bound
        use std::ops::Bound;
        let range: Vec<i64> = tree
            .range((Bound::Excluded(10000), Bound::Included(10010)))
            .map(|(k, _)| *k)
            .collect();
        assert_eq!(range.len(), 10);
        assert_eq!(range[0], 10001);
        assert_eq!(range[9], 10010);

        // Unbounded range on deep tree
        let count = tree.range(..).count();
        assert_eq!(count, 20_000);
    }

    #[test]
    fn test_range_chunks_on_deep_tree() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Range chunks in the middle
        let mut total = 0;
        for (keys, values) in tree.range_chunks(5000..6000) {
            assert_eq!(keys.len(), values.len());
            total += keys.len();
        }
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_deep_tree_delete_from_left_internal() {
        // Delete pattern that triggers borrow_from_right on internal nodes
        // We need to delete from the LEFT side of a deep tree to make left internal
        // nodes underflow while right internal nodes have spare keys
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert 20K keys for 3-level tree
        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Delete the first 8000 keys - this should cause internal node rebalancing
        // where left internal nodes need to borrow from right siblings
        for i in 0..8000 {
            tree.remove(i);
        }

        assert_eq!(tree.len(), 12_000);

        // Verify remaining keys
        for i in 8000..20_000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_deep_tree_delete_alternating_pattern() {
        // Alternating delete pattern to exercise more internal node paths
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Delete in an alternating pattern across the tree
        // First delete from beginning
        for i in 0..3000 {
            tree.remove(i);
        }

        // Then delete from end
        for i in 17000..20_000 {
            tree.remove(i);
        }

        // Then delete from middle
        for i in 8000..12000 {
            tree.remove(i);
        }

        // Verify deleted keys are gone
        for i in 0..3000 {
            assert_eq!(tree.get(i), None, "Key {} should be deleted", i);
        }
        for i in 17000..20_000 {
            assert_eq!(tree.get(i), None, "Key {} should be deleted", i);
        }
        for i in 8000..12000 {
            assert_eq!(tree.get(i), None, "Key {} should be deleted", i);
        }

        // Verify remaining keys exist
        for i in 3000..8000 {
            assert_eq!(tree.get(i), Some(&i), "Key {} should exist", i);
        }
        for i in 12000..17000 {
            assert_eq!(tree.get(i), Some(&i), "Key {} should exist", i);
        }
    }

    #[test]
    fn test_entry_on_deep_tree_random() {
        // Entry API with random keys on deep tree
        let mut tree: CowBTree<i64> = CowBTree::new();

        // First build a deep tree sequentially
        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Now use entry API with random access patterns
        for i in (0..20_000).step_by(7) {
            tree.entry(i).and_modify(|v| *v *= 2);
        }

        // Verify modifications
        for i in 0..20_000 {
            if i % 7 == 0 {
                assert_eq!(tree.get(i), Some(&(i * 2)));
            } else {
                assert_eq!(tree.get(i), Some(&i));
            }
        }
    }

    #[test]
    fn test_iter_on_deep_tree() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // iter() should return all keys in order
        let keys: Vec<i64> = tree.keys().collect();
        assert_eq!(keys.len(), 20_000);
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(*k, i as i64);
        }

        // values() should return all values in order
        let values: Vec<&i64> = tree.values().collect();
        assert_eq!(values.len(), 20_000);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(**v, i as i64);
        }
    }

    #[test]
    fn test_get_mut_on_deep_tree() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // get_mut at various positions
        if let Some(v) = tree.get_mut(0) {
            *v = 999;
        }
        if let Some(v) = tree.get_mut(10000) {
            *v = 888;
        }
        if let Some(v) = tree.get_mut(19999) {
            *v = 777;
        }

        assert_eq!(tree.get(0), Some(&999));
        assert_eq!(tree.get(10000), Some(&888));
        assert_eq!(tree.get(19999), Some(&777));
    }

    #[test]
    fn test_contains_key() {
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..100 {
            tree.insert(i * 2, i);
        }

        // Even keys should exist
        for i in 0..100 {
            assert!(tree.contains_key(i * 2));
        }

        // Odd keys should not exist
        for i in 0..100 {
            assert!(!tree.contains_key(i * 2 + 1));
        }
    }

    #[test]
    fn test_reverse_order_inserts() {
        // Reverse order inserts force standard split_internal (not rightmost)
        // because we're always inserting at the leftmost position
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert in reverse order to trigger split_internal (midpoint split)
        for i in (0..5000).rev() {
            tree.insert(i, i);
        }

        assert_eq!(tree.len(), 5000);

        // Verify all keys
        for i in 0..5000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }

        // Verify iteration order
        let keys: Vec<i64> = tree.keys().collect();
        for (idx, &k) in keys.iter().enumerate() {
            assert_eq!(k, idx as i64);
        }
    }

    #[test]
    fn test_shuffled_inserts_large() {
        // Shuffled inserts to trigger standard split_internal paths
        // Using a shuffle pattern that avoids sequential rightmost inserts
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Generate shuffled keys - interleave from both ends
        let mut keys: Vec<i64> = Vec::with_capacity(10000);
        for i in 0..5000 {
            keys.push(i * 2); // Even: 0, 2, 4, ...
            keys.push(i * 2 + 1); // Odd: 1, 3, 5, ...
        }
        // Reverse half to make it more shuffled
        for i in 0..5000 {
            keys.swap(i, 9999 - i);
        }

        for &k in &keys {
            tree.insert(k, k);
        }

        assert_eq!(tree.len(), 10000);

        for i in 0..10000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_internal_node_borrow_from_right() {
        // This test triggers internal node borrow_from_right (the else branch)
        // by creating a 3-level tree and causing internal node underflow
        // where the right sibling is also an internal node (not a leaf)
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Use a shuffle pattern to ensure we don't get the rightmost-optimized splits
        // This creates a more balanced tree with internal nodes that can borrow from each other
        let mut keys: Vec<i64> = (0..50_000).collect();
        // Shuffle using a deterministic pattern
        for i in 0..keys.len() {
            let j = (i * 31337 + 17) % keys.len();
            keys.swap(i, j);
        }

        for &k in &keys {
            tree.insert(k, k);
        }

        assert_eq!(tree.len(), 50_000);

        // Delete a large contiguous region from the left side
        // This should cause leaf merges in that region, which cascade up
        // to cause internal node underflow and borrow/merge operations
        for i in 0..20_000 {
            tree.remove(i);
        }

        assert_eq!(tree.len(), 30_000);

        // Verify remaining keys
        for i in 0..20_000 {
            assert_eq!(tree.get(i), None, "Key {} should be deleted", i);
        }
        for i in 20_000..50_000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_internal_node_merge_cascade() {
        // Trigger internal node merge by deleting enough to cause cascade
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Use reverse interleaving for balanced structure
        for i in 0..30_000 {
            let key = if i % 2 == 0 {
                i / 2
            } else {
                30_000 - 1 - i / 2
            };
            tree.insert(key, key);
        }

        // Delete from multiple regions to trigger internal node merges
        // Delete first third
        for i in 0..10_000 {
            tree.remove(i);
        }

        // Delete last third
        for i in 20_000..30_000 {
            tree.remove(i);
        }

        // Verify remaining keys in middle third
        for i in 10_000..20_000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_massive_deletion_internal_rebalance() {
        // Create large tree and delete 90% of keys to heavily exercise internal node rebalancing
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Interleaved insert pattern
        for i in 0..60_000 {
            let key = (i * 7919) % 60_000;
            tree.insert(key, key);
        }

        // Delete 90% - keep only every 10th key
        for i in 0..60_000 {
            if i % 10 != 0 {
                tree.remove(i);
            }
        }

        assert_eq!(tree.len(), 6000);

        // Verify remaining keys
        for i in (0..60_000).step_by(10) {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_left_heavy_deletion() {
        // Delete heavily from left side to trigger internal node borrow from right
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Build tree with reverse order to create left-biased structure
        for i in (0..22_000).rev() {
            tree.insert(i, i);
        }

        // Delete 80% from left side
        for i in 0..17_600 {
            tree.remove(i);
        }

        assert_eq!(tree.len(), 4400);

        // Verify
        for i in 17_600..22_000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_right_heavy_deletion() {
        // Delete heavily from right side to trigger borrow from left
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Sequential insert for rightmost structure
        for i in 0..30_000 {
            tree.insert(i, i);
        }

        // Delete 80% from right side
        for i in 6000..30_000 {
            tree.remove(i);
        }

        assert_eq!(tree.len(), 6000);

        // Verify
        for i in 0..6000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_alternating_sides_deletion() {
        // Alternate between deleting from left and right to exercise both borrow directions
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..30_000 {
            tree.insert(i, i);
        }

        // Delete from ends alternating
        let mut left = 0;
        let mut right = 29_999;
        for _ in 0..20_000 {
            if left < right {
                tree.remove(left);
                left += 1;
            }
            if left < right {
                tree.remove(right);
                right -= 1;
            }
        }

        // Verify middle section
        for i in left..=right {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_entry_api_reverse_order() {
        // Entry API with reverse order to trigger split_internal (not rightmost)
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in (0..3000).rev() {
            tree.entry(i).or_insert(i * 10);
        }

        assert_eq!(tree.len(), 3000);

        for i in 0..3000 {
            assert_eq!(tree.get(i), Some(&(i * 10)));
        }
    }

    #[test]
    fn test_cow_during_internal_rebalancing() {
        // Test COW semantics are maintained during internal node rebalancing
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Create deep tree
        for i in 0..30_000 {
            tree.insert(i, i);
        }

        // Clone before deletions
        let snapshot = tree.clone();

        // Delete heavily to trigger internal rebalancing
        for i in 0..20_000 {
            tree.remove(i);
        }

        // Snapshot must be intact
        assert_eq!(snapshot.len(), 30_000);
        for i in 0..30_000 {
            assert_eq!(snapshot.get(i), Some(&i), "Snapshot missing key {}", i);
        }

        // Modified tree
        assert_eq!(tree.len(), 10_000);
        for i in 0..20_000 {
            assert_eq!(tree.get(i), None);
        }
        for i in 20_000..30_000 {
            assert_eq!(tree.get(i), Some(&i));
        }
    }

    #[test]
    fn test_zigzag_insert_pattern() {
        // Insert in zigzag pattern: 0, 9999, 1, 9998, 2, 9997, ...
        // This forces many internal node splits at non-rightmost positions
        let mut tree: CowBTree<i64> = CowBTree::new();

        for i in 0..5000 {
            tree.insert(i, i);
            tree.insert(9999 - i, 9999 - i);
        }

        assert_eq!(tree.len(), 10000);

        for i in 0..10000 {
            assert_eq!(tree.get(i), Some(&i), "Missing key {}", i);
        }
    }

    #[test]
    fn test_middle_insert_split() {
        // Insert at middle positions to force non-rightmost splits
        let mut tree: CowBTree<i64> = CowBTree::new();

        // First insert boundaries
        for i in (0..5000).step_by(100) {
            tree.insert(i, i);
        }

        // Then fill in midpoints repeatedly
        for gap in [50i64, 25, 12, 6, 3, 1] {
            let mut i = gap;
            while i < 5000 {
                if tree.get(i).is_none() {
                    tree.insert(i, i);
                }
                i += gap * 2;
            }
        }

        // Fill any remaining gaps
        for i in 0..5000 {
            if tree.get(i).is_none() {
                tree.insert(i, i);
            }
        }

        assert_eq!(tree.len(), 5000);

        for i in 0..5000 {
            assert_eq!(tree.get(i), Some(&i));
        }
    }

    #[test]
    fn test_update_separator_keys() {
        // This test triggers the Ok(i) => i + 1 path in insert_recursive
        // by updating keys that are separator keys in internal nodes.
        // Separator keys are promoted during splits.
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert in reverse order to use insert_recursive (not rightmost optimization)
        // This ensures splits go through split_internal, not split_internal_rightmost
        for i in (0..1000).rev() {
            tree.insert(i, i);
        }

        // Now update ALL keys - some of them are separator keys in internal nodes
        // When we update a separator key, search returns Ok(i) and we go to child i+1
        for i in 0..1000 {
            let old = tree.insert(i, i * 100);
            assert_eq!(old, Some(i), "Key {} should have existed", i);
        }

        // Verify updates
        for i in 0..1000 {
            assert_eq!(tree.get(i), Some(&(i * 100)));
        }
    }

    #[test]
    fn test_reverse_insert_internal_node_split() {
        // This test ensures split_internal() (not split_internal_rightmost) is called
        // by inserting in reverse order with enough keys to overflow internal nodes.
        // With MAX_KEYS=128, we need 128*128 = 16384+ keys for 3 levels.
        let mut tree: CowBTree<i64> = CowBTree::new();

        // Insert 20K keys in reverse order
        // All inserts go through insert_recursive since key < max_key
        for i in (0..20_000).rev() {
            tree.insert(i, i);
        }

        assert_eq!(tree.len(), 20_000);

        // Verify all keys
        for i in (0..20_000).step_by(100) {
            assert_eq!(tree.get(i), Some(&i));
        }
    }

    #[test]
    fn test_iter_rev() {
        let mut tree: CowBTree<i64> = CowBTree::new();
        for i in 0..500 {
            tree.insert(i, i * 10);
        }

        // iter_rev should yield all entries in descending key order
        let rev: Vec<(i64, i64)> = tree.iter_rev().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(rev.len(), 500);
        for (i, &(k, v)) in rev.iter().enumerate() {
            assert_eq!(k, 499 - i as i64);
            assert_eq!(v, k * 10);
        }

        // iter_rev on empty tree
        let empty: CowBTree<i64> = CowBTree::new();
        assert_eq!(empty.iter_rev().count(), 0);
    }

    #[test]
    fn test_range_rev() {
        let mut tree: CowBTree<i64> = CowBTree::new();
        for i in 0..500 {
            tree.insert(i, i * 10);
        }

        // range_rev with Excluded start, Unbounded end (keyset pagination pattern)
        let result: Vec<(i64, i64)> = tree
            .range_rev((std::ops::Bound::Excluded(100), std::ops::Bound::Unbounded))
            .map(|(&k, &v)| (k, v))
            .collect();
        assert_eq!(result.len(), 399); // 101..499 inclusive
        assert_eq!(result[0], (499, 4990)); // Largest first
        assert_eq!(result[398], (101, 1010)); // Smallest last

        // range_rev with Included start, Unbounded end
        let result: Vec<(i64, i64)> = tree
            .range_rev((std::ops::Bound::Included(100), std::ops::Bound::Unbounded))
            .map(|(&k, &v)| (k, v))
            .collect();
        assert_eq!(result.len(), 400); // 100..499 inclusive
        assert_eq!(result[0], (499, 4990));
        assert_eq!(result[399], (100, 1000));

        // range_rev with both bounds
        let result: Vec<(i64, i64)> = tree
            .range_rev((
                std::ops::Bound::Included(200),
                std::ops::Bound::Excluded(300),
            ))
            .map(|(&k, &v)| (k, v))
            .collect();
        assert_eq!(result.len(), 100); // 200..299 inclusive
        assert_eq!(result[0], (299, 2990));
        assert_eq!(result[99], (200, 2000));

        // range_rev with Included end
        let result: Vec<(i64, i64)> = tree
            .range_rev((
                std::ops::Bound::Included(200),
                std::ops::Bound::Included(300),
            ))
            .map(|(&k, &v)| (k, v))
            .collect();
        assert_eq!(result.len(), 101); // 200..=300 inclusive
        assert_eq!(result[0], (300, 3000));
        assert_eq!(result[100], (200, 2000));

        // range_rev unbounded both sides = iter_rev
        let all_rev: Vec<i64> = tree
            .range_rev((
                std::ops::Bound::Unbounded,
                std::ops::Bound::Unbounded::<i64>,
            ))
            .map(|(&k, _)| k)
            .collect();
        let iter_rev: Vec<i64> = tree.iter_rev().map(|(&k, _)| k).collect();
        assert_eq!(all_rev, iter_rev);
    }

    #[test]
    fn test_iter_rev_on_deep_tree() {
        let mut tree: CowBTree<i64> = CowBTree::new();
        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // iter_rev should return all keys in reverse order
        let rev_keys: Vec<i64> = tree.iter_rev().map(|(&k, _)| k).collect();
        assert_eq!(rev_keys.len(), 20_000);
        for (i, &k) in rev_keys.iter().enumerate() {
            assert_eq!(k, 19_999 - i as i64);
        }

        // Taking a small prefix is O(limit), not O(n)
        let first_5: Vec<i64> = tree.iter_rev().map(|(&k, _)| k).take(5).collect();
        assert_eq!(first_5, vec![19_999, 19_998, 19_997, 19_996, 19_995]);
    }

    #[test]
    fn test_range_rev_on_deep_tree() {
        let mut tree: CowBTree<i64> = CowBTree::new();
        for i in 0..20_000 {
            tree.insert(i, i);
        }

        // Keyset pagination pattern: WHERE id > 15000 ORDER BY id DESC LIMIT 5
        let page: Vec<i64> = tree
            .range_rev((
                std::ops::Bound::Excluded(15_000),
                std::ops::Bound::Unbounded,
            ))
            .map(|(&k, _)| k)
            .take(5)
            .collect();
        assert_eq!(page, vec![19_999, 19_998, 19_997, 19_996, 19_995]);

        // Keyset pagination: WHERE id > 5000 ORDER BY id DESC LIMIT 5
        let page: Vec<i64> = tree
            .range_rev((std::ops::Bound::Excluded(5_000), std::ops::Bound::Unbounded))
            .map(|(&k, _)| k)
            .take(5)
            .collect();
        assert_eq!(page, vec![19_999, 19_998, 19_997, 19_996, 19_995]);

        // Bounded range in reverse
        let result: Vec<i64> = tree
            .range_rev((
                std::ops::Bound::Included(10_000),
                std::ops::Bound::Excluded(10_010),
            ))
            .map(|(&k, _)| k)
            .collect();
        assert_eq!(
            result,
            vec![10_009, 10_008, 10_007, 10_006, 10_005, 10_004, 10_003, 10_002, 10_001, 10_000]
        );
    }
}
