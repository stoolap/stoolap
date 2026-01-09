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

//! CompactVec - A 16-byte vector optimized for Row storage
//!
//! Standard Vec<T> is 24 bytes (ptr + len + cap as usize).
//! CompactVec<T> is 16 bytes (ptr + packed len/cap as u32).
//!
//! Benefits:
//! - 33% smaller than Vec (16 vs 24 bytes)
//! - O(1) len() access (unlike ThinVec which requires dereference)
//! - Faster moves due to smaller size
//! - Supports up to 4 billion elements (u32::MAX)

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::fmt;
use std::iter::FromIterator;
use std::mem::{self, ManuallyDrop};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::{self, NonNull};
use std::slice;

/// A compact vector that uses 16 bytes instead of Vec's 24 bytes.
/// Stores length and capacity as u32 (max 4 billion elements).
pub struct CompactVec<T> {
    ptr: NonNull<T>,
    /// Packed length (low 32 bits) and capacity (high 32 bits)
    len_cap: u64,
}

// Safety: CompactVec has the same thread-safety as Vec
unsafe impl<T: Send> Send for CompactVec<T> {}
unsafe impl<T: Sync> Sync for CompactVec<T> {}

impl<T> CompactVec<T> {
    /// Pack length and capacity into a single u64
    #[inline(always)]
    const fn pack(len: u32, cap: u32) -> u64 {
        (len as u64) | ((cap as u64) << 32)
    }

    /// Unpack length from len_cap
    #[inline(always)]
    const fn unpack_len(len_cap: u64) -> u32 {
        len_cap as u32
    }

    /// Unpack capacity from len_cap
    #[inline(always)]
    const fn unpack_cap(len_cap: u64) -> u32 {
        (len_cap >> 32) as u32
    }

    /// Creates an empty CompactVec.
    #[inline]
    pub const fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len_cap: 0,
        }
    }

    /// Creates a CompactVec with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }

        let cap = capacity.min(u32::MAX as usize) as u32;

        // Allocate memory
        let layout = Layout::array::<T>(cap as usize).unwrap();
        let ptr = unsafe { alloc(layout) as *mut T };

        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len_cap: Self::pack(0, cap),
        }
    }

    /// Returns the number of elements in the vector.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        Self::unpack_len(self.len_cap) as usize
    }

    /// Returns true if the vector contains no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        Self::unpack_len(self.len_cap) == 0
    }

    /// Returns the capacity of the vector.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        Self::unpack_cap(self.len_cap) as usize
    }

    /// Set the length (unsafe - caller must ensure elements are initialized)
    #[inline(always)]
    unsafe fn set_len(&mut self, new_len: usize) {
        let cap = Self::unpack_cap(self.len_cap);
        self.len_cap = Self::pack(new_len as u32, cap);
    }

    /// Returns a raw pointer to the vector's buffer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the vector's buffer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Appends an element to the back of the vector.
    #[inline]
    pub fn push(&mut self, value: T) {
        // Unpack once instead of calling len() and capacity() separately
        let len = Self::unpack_len(self.len_cap) as usize;
        let cap = Self::unpack_cap(self.len_cap) as usize;

        if len == cap {
            self.grow();
        }

        unsafe {
            ptr::write(self.ptr.as_ptr().add(len), value);
            self.set_len(len + 1);
        }
    }

    /// Removes the last element from the vector and returns it.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 {
            return None;
        }

        unsafe {
            self.set_len(len - 1);
            Some(ptr::read(self.ptr.as_ptr().add(len - 1)))
        }
    }

    /// Clears the vector, removing all values.
    #[inline]
    pub fn clear(&mut self) {
        let len = self.len();
        if len == 0 {
            return;
        }

        // Drop all elements
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), len));
            self.set_len(0);
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        // Unpack once
        let len = Self::unpack_len(self.len_cap) as usize;
        let cap = Self::unpack_cap(self.len_cap) as usize;
        let required = len.saturating_add(additional);

        if required > cap {
            // Calculate new capacity directly, avoid second capacity() call
            let new_cap = required.min(u32::MAX as usize);
            self.realloc(new_cap);
        }
    }

    /// Grow the vector (double capacity or start at 4)
    fn grow(&mut self) {
        let cap = self.capacity();
        let new_cap = if cap == 0 {
            4
        } else {
            cap.saturating_mul(2).min(u32::MAX as usize)
        };

        self.realloc(new_cap);
    }

    /// Reallocate to new capacity
    fn realloc(&mut self, new_cap: usize) {
        let len = self.len();
        let old_cap = self.capacity();
        let new_cap = new_cap as u32;

        if mem::size_of::<T>() == 0 {
            // ZST - no actual allocation needed
            self.len_cap = Self::pack(len as u32, new_cap);
            return;
        }

        let new_layout = Layout::array::<T>(new_cap as usize).unwrap();

        let new_ptr = if old_cap == 0 {
            // Fresh allocation
            unsafe { alloc(new_layout) as *mut T }
        } else {
            // Realloc existing
            let old_layout = Layout::array::<T>(old_cap).unwrap();
            unsafe {
                realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size()) as *mut T
            }
        };

        if new_ptr.is_null() {
            std::alloc::handle_alloc_error(new_layout);
        }

        self.ptr = unsafe { NonNull::new_unchecked(new_ptr) };
        self.len_cap = Self::pack(len as u32, new_cap);
    }

    /// Truncates the vector, keeping the first `len` elements.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        let current_len = self.len();
        if len >= current_len {
            return;
        }

        // Drop elements beyond new length
        unsafe {
            let remaining = current_len - len;
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr().add(len),
                remaining,
            ));
            self.set_len(len);
        }
    }

    /// Removes and returns the element at position `index`, shifting all elements after it.
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len, "removal index out of bounds");

        unsafe {
            let ptr = self.ptr.as_ptr().add(index);
            let value = ptr::read(ptr);

            // Shift elements down
            ptr::copy(ptr.add(1), ptr, len - index - 1);
            self.set_len(len - 1);

            value
        }
    }

    /// Removes an element from the vector and returns it, replacing it with the last element.
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(index < len, "swap_remove index out of bounds");

        unsafe {
            let ptr = self.ptr.as_ptr();
            let value = ptr::read(ptr.add(index));

            // Copy last element to the removed position (if not removing last)
            if index < len - 1 {
                ptr::copy_nonoverlapping(ptr.add(len - 1), ptr.add(index), 1);
            }

            self.set_len(len - 1);
            value
        }
    }

    /// Extends the vector with elements from an iterator.
    #[inline]
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();

        // If exact size is known, write directly without per-element bounds checks
        if Some(lower) == upper && lower > 0 {
            self.reserve(lower);
            let mut len = self.len();
            // SAFETY:
            // - reserve(lower) guarantees capacity for `lower` more elements
            // - ExactSizeIterator (lower == upper) guarantees exactly `lower` items
            // - We write each item and increment len, then set_len at the end
            unsafe {
                let ptr = self.ptr.as_ptr();
                for item in iter {
                    ptr::write(ptr.add(len), item);
                    len += 1;
                }
                self.set_len(len);
            }
        } else {
            // Fallback for unknown size
            self.reserve(lower);
            for item in iter {
                self.push(item);
            }
        }
    }

    /// Returns a slice containing all elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    /// Returns a mutable slice containing all elements.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: ptr is valid and aligned, len elements are initialized
        unsafe { &mut *ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }

    /// Returns an iterator over the vector.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the vector.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Gets a reference to an element.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Gets a mutable reference to an element.
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Converts the vector into a standard Vec.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        let len = self.len();
        let cap = self.capacity();

        if cap == 0 {
            mem::forget(self);
            return Vec::new();
        }

        let ptr = self.ptr.as_ptr();
        mem::forget(self);

        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    /// Converts the vector into a boxed slice.
    #[inline]
    pub fn into_boxed_slice(self) -> Box<[T]> {
        self.into_vec().into_boxed_slice()
    }

    /// Creates a CompactVec from a standard Vec.
    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = vec.len().min(u32::MAX as usize) as u32;
        let cap = vec.capacity().min(u32::MAX as usize) as u32;

        if cap == 0 {
            mem::forget(vec);
            return Self::new();
        }

        let mut vec = ManuallyDrop::new(vec);
        let ptr = vec.as_mut_ptr();

        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len_cap: Self::pack(len, cap),
        }
    }
}

impl<T: Clone> CompactVec<T> {
    /// Resizes the vector to `new_len`, filling with clones of `value`.
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.reserve(new_len - len);
            for _ in len..new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }
}

impl<T> Drop for CompactVec<T> {
    fn drop(&mut self) {
        if self.capacity() == 0 {
            return;
        }

        // Drop all elements
        let len = self.len();
        if len > 0 {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), len));
            }
        }

        // Deallocate memory
        if mem::size_of::<T>() > 0 {
            let layout = Layout::array::<T>(self.capacity()).unwrap();
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T: Clone> Clone for CompactVec<T> {
    fn clone(&self) -> Self {
        let len = self.len();
        if len == 0 {
            return Self::new();
        }

        let mut new_vec = Self::with_capacity(len);
        // SAFETY:
        // - with_capacity(len) guarantees capacity >= len
        // - We write exactly len elements to indices 0..len
        // - set_len(len) matches the number of initialized elements
        // - If clone() panics mid-way, some elements leak (not UB, just a leak)
        unsafe {
            let src = self.ptr.as_ptr();
            let dst = new_vec.ptr.as_ptr();
            for i in 0..len {
                ptr::write(dst.add(i), (*src.add(i)).clone());
            }
            new_vec.set_len(len);
        }
        new_vec
    }
}

impl<T> Default for CompactVec<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for CompactVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Deref for CompactVec<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for CompactVec<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Index<usize> for CompactVec<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        &self.as_slice()[index]
    }
}

impl<T> IndexMut<usize> for CompactVec<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.as_mut_slice()[index]
    }
}

impl<T> FromIterator<T> for CompactVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();

        // If exact size is known, write directly without per-element bounds checks
        if Some(lower) == upper && lower > 0 {
            let mut vec = Self::with_capacity(lower);
            // SAFETY:
            // - with_capacity(lower) guarantees capacity >= lower
            // - ExactSizeIterator guarantees exactly `lower` items
            // - We write each item sequentially and set_len at the end
            unsafe {
                let ptr = vec.ptr.as_ptr();
                let mut len = 0;
                for item in iter {
                    ptr::write(ptr.add(len), item);
                    len += 1;
                }
                vec.set_len(len);
            }
            vec
        } else {
            // Fallback for unknown size - uses push with bounds checks
            let mut vec = Self::with_capacity(lower);
            for item in iter {
                vec.push(item);
            }
            vec
        }
    }
}

impl<T> IntoIterator for CompactVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter::new(self)
    }
}

impl<'a, T> IntoIterator for &'a CompactVec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut CompactVec<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T: PartialEq> PartialEq for CompactVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for CompactVec<T> {}

impl<T> From<Vec<T>> for CompactVec<T> {
    fn from(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
}

impl<T> From<CompactVec<T>> for Vec<T> {
    fn from(vec: CompactVec<T>) -> Self {
        vec.into_vec()
    }
}

/// Owning iterator for CompactVec.
pub struct IntoIter<T> {
    vec: CompactVec<T>,
    index: usize,
}

impl<T> IntoIter<T> {
    fn new(vec: CompactVec<T>) -> Self {
        Self { vec, index: 0 }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.index < self.vec.len() {
            let item = unsafe { ptr::read(self.vec.ptr.as_ptr().add(self.index)) };
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.vec.len() - self.index
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        let len = self.vec.len();
        if self.index < len {
            // SAFETY:
            // - Elements [index..len] have not been read/moved out yet
            // - They are valid initialized elements that need dropping
            // - After drop_in_place, we set len=0 to prevent double-drop
            unsafe {
                let remaining = len - self.index;
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.vec.ptr.as_ptr().add(self.index),
                    remaining,
                ));
            }
        }

        // SAFETY: All elements are now dropped, set len=0 to prevent double-drop
        unsafe {
            self.vec.set_len(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<CompactVec<u8>>(), 16);
        assert_eq!(std::mem::size_of::<CompactVec<u64>>(), 16);
        assert_eq!(std::mem::size_of::<Vec<u8>>(), 24);
    }

    #[test]
    fn test_basic_operations() {
        let mut vec = CompactVec::new();
        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);

        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);

        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_with_capacity() {
        let vec: CompactVec<i32> = CompactVec::with_capacity(100);
        assert!(vec.is_empty());
        assert!(vec.capacity() >= 100);
    }

    #[test]
    fn test_clone() {
        let mut vec = CompactVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let cloned = vec.clone();
        assert_eq!(vec.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_iteration() {
        let mut vec = CompactVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let sum: i32 = vec.iter().sum();
        assert_eq!(sum, 6);

        let collected: Vec<i32> = vec.into_iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_from_iterator() {
        let vec: CompactVec<i32> = (0..5).collect();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_extend() {
        let mut vec = CompactVec::new();
        vec.push(1);
        vec.extend(vec![2, 3, 4]);
        assert_eq!(vec.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_truncate() {
        let mut vec: CompactVec<i32> = (0..10).collect();
        vec.truncate(5);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_clear() {
        let mut vec: CompactVec<i32> = (0..10).collect();
        let cap = vec.capacity();
        vec.clear();
        assert!(vec.is_empty());
        assert_eq!(vec.capacity(), cap); // Capacity preserved
    }

    #[test]
    fn test_swap_remove() {
        let mut vec = CompactVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.swap_remove(0), 1);
        assert_eq!(vec.as_slice(), &[3, 2]);
    }

    #[test]
    fn test_into_vec_and_back() {
        let compact: CompactVec<i32> = (0..5).collect();
        let std_vec: Vec<i32> = compact.into_vec();
        assert_eq!(std_vec, vec![0, 1, 2, 3, 4]);

        let compact_again = CompactVec::from_vec(std_vec);
        assert_eq!(compact_again.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_with_strings() {
        let mut vec = CompactVec::new();
        vec.push(String::from("hello"));
        vec.push(String::from("world"));

        assert_eq!(vec[0], "hello");
        assert_eq!(vec[1], "world");

        let cloned = vec.clone();
        assert_eq!(cloned[0], "hello");
    }
}
