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

// One pointer plus two u32s, and the pointer's niche keeps Option free.
const _: () = assert!(
    mem::size_of::<CompactVec<u8>>() == mem::size_of::<usize>() + 2 * mem::size_of::<u32>()
        && mem::size_of::<Option<CompactVec<u8>>>() == mem::size_of::<CompactVec<u8>>()
);

/// A compact vector: one pointer plus two u32s, 16 bytes on 64-bit targets
/// against Vec's 24. Holds at most u32::MAX elements.
pub struct CompactVec<T> {
    ptr: NonNull<T>,
    /// Packed length (low 32 bits) and capacity (high 32 bits)
    len_cap: u64,
}

// SAFETY: CompactVec has the same thread-safety as Vec - it can be sent
// between threads if T can be sent, as it owns its data exclusively.
unsafe impl<T: Send> Send for CompactVec<T> {}
// SAFETY: CompactVec can be shared between threads if T can be shared,
// as it only provides shared access to its elements through &self methods.
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

        if mem::size_of::<T>() == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len_cap: Self::pack(0, cap),
            };
        }

        // Allocate memory
        let layout = Layout::array::<T>(cap as usize).unwrap();
        // SAFETY: Layout is valid (non-zero size, proper alignment for T).
        let ptr = unsafe { alloc(layout) as *mut T };

        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Self {
            // SAFETY: We just checked that ptr is not null above.
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

    /// Set the length.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to `capacity()`
    /// - The elements at `old_len..new_len` must be initialized (if growing)
    /// - The elements at `new_len..old_len` will NOT be dropped (if shrinking)
    #[inline(always)]
    pub unsafe fn set_len(&mut self, new_len: usize) {
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

        // SAFETY: After grow(), we have capacity > len, so ptr.add(len) is valid.
        // The memory at that location is uninitialized but allocated for T.
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

        // SAFETY: len > 0, so len - 1 is a valid index. The element at that
        // position is initialized. After read, we decrement len so the moved
        // element won't be dropped again.
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

        // SAFETY: All elements [0..len] are initialized. After dropping them,
        // we set len to 0 so they won't be dropped again.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), len));
            self.set_len(0);
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// Grows geometrically like `Vec::reserve`, so a sequence of small
    /// reservations costs O(log n) reallocations rather than one per call.
    /// Use [`CompactVec::reserve_exact`] when the final size is known.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        // Unpack once
        let len = Self::unpack_len(self.len_cap) as usize;
        let cap = Self::unpack_cap(self.len_cap) as usize;
        let required = len
            .checked_add(additional)
            .filter(|&required| required <= u32::MAX as usize)
            .expect("CompactVec capacity overflow");

        if required > cap {
            let new_cap = required.max(cap.saturating_mul(2)).min(u32::MAX as usize);
            self.realloc(new_cap);
        }
    }

    /// Reserves capacity for exactly `additional` more elements.
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        let len = Self::unpack_len(self.len_cap) as usize;
        let cap = Self::unpack_cap(self.len_cap) as usize;
        let required = len
            .checked_add(additional)
            .filter(|&required| required <= u32::MAX as usize)
            .expect("CompactVec capacity overflow");

        if required > cap {
            self.realloc(required);
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
        assert!(new_cap > cap, "CompactVec capacity overflow");

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
            // SAFETY: new_layout is valid (non-zero size, proper alignment).
            unsafe { alloc(new_layout) as *mut T }
        } else {
            // Realloc existing
            let old_layout = Layout::array::<T>(old_cap).unwrap();
            // SAFETY: self.ptr was allocated with old_layout, and new_layout.size()
            // is valid. The allocator will copy existing data to new location.
            unsafe {
                realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size()) as *mut T
            }
        };

        if new_ptr.is_null() {
            std::alloc::handle_alloc_error(new_layout);
        }

        // SAFETY: We just checked that new_ptr is not null above.
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

        // SAFETY: Elements [len..current_len] are initialized. After dropping,
        // we set length to `len` so they won't be dropped again.
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

        // SAFETY: index < len is asserted above, so the element is valid.
        // After reading, we shift remaining elements and decrement length.
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

        // SAFETY: index < len is asserted above. We read the element at index,
        // then copy the last element to fill the gap, and decrement length.
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

    /// Inserts an element at position `index`, shifting all elements after it to the right.
    ///
    /// # Panics
    /// Panics if `index > len`.
    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        let len = self.len();
        assert!(index <= len, "insertion index out of bounds");

        // Ensure we have capacity for one more element
        if len == self.capacity() {
            self.grow();
        }

        // SAFETY: index <= len is asserted above. After grow(), capacity > len.
        // We shift elements right to make room, then write the new element.
        unsafe {
            let ptr = self.ptr.as_ptr().add(index);

            // Shift elements to the right
            if index < len {
                ptr::copy(ptr, ptr.add(1), len - index);
            }

            // Write the new element
            ptr::write(ptr, element);
            self.set_len(len + 1);
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let original_len = self.len();

        // The vector disowns its elements while they are being shifted; the
        // guard hands back exactly the surviving ones, even if `f` panics.
        // SAFETY: the guard below restores a correct length in every exit path.
        unsafe {
            self.set_len(0);
        }

        struct RetainGuard<'a, T> {
            vec: &'a mut CompactVec<T>,
            read_idx: usize,
            write_idx: usize,
            original_len: usize,
        }

        impl<T> Drop for RetainGuard<'_, T> {
            fn drop(&mut self) {
                let tail = self.original_len - self.read_idx;
                if tail > 0 && self.write_idx != self.read_idx {
                    // SAFETY: [read_idx..original_len] is still initialized and
                    // write_idx < read_idx, so the move is backwards in place.
                    unsafe {
                        let ptr = self.vec.ptr.as_ptr();
                        ptr::copy(ptr.add(self.read_idx), ptr.add(self.write_idx), tail);
                    }
                }
                // SAFETY: exactly write_idx + tail elements are initialized.
                unsafe {
                    self.vec.set_len(self.write_idx + tail);
                }
            }
        }

        let mut guard = RetainGuard {
            vec: self,
            read_idx: 0,
            write_idx: 0,
            original_len,
        };

        while guard.read_idx < original_len {
            let ptr = guard.vec.ptr.as_ptr();
            // SAFETY: read_idx < original_len, so the element is initialized.
            let keep = unsafe { f(&*ptr.add(guard.read_idx)) };
            if keep {
                if guard.write_idx != guard.read_idx {
                    // SAFETY: write_idx < read_idx, so the two slots are distinct.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            ptr.add(guard.read_idx),
                            ptr.add(guard.write_idx),
                            1,
                        );
                    }
                }
                guard.write_idx += 1;
                guard.read_idx += 1;
            } else {
                // Advance past the slot before dropping it: if `T::drop`
                // panics, the guard must not count this element as untouched
                // tail and hand the dropped value back to the vector.
                let idx = guard.read_idx;
                guard.read_idx += 1;
                // SAFETY: the element is initialized and is not moved anywhere.
                unsafe {
                    ptr::drop_in_place(ptr.add(idx));
                }
            }
        }
    }

    /// Extends the vector with elements from an iterator.
    #[inline]
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();

        // If exact size is known, write directly with bounds protection
        if Some(lower) == upper && lower > 0 {
            self.reserve(lower);
            let original_len = self.len();
            let cap = self.capacity();

            // RAII guard for panic safety: if iter.next() panics, we set the length
            // to the number of successfully written elements so Drop can clean up.
            struct ExtendGuard<'a, T> {
                vec: &'a mut CompactVec<T>,
                written_count: usize,
            }

            impl<T> Drop for ExtendGuard<'_, T> {
                fn drop(&mut self) {
                    // SAFETY: written_count tracks how many elements were successfully
                    // written before a panic. Setting len to this value ensures Drop
                    // will clean up exactly those elements. This is only reached on panic
                    // (normal path uses mem::forget).
                    unsafe {
                        self.vec.set_len(self.written_count);
                    }
                }
            }

            let mut guard = ExtendGuard {
                vec: self,
                written_count: original_len,
            };

            // SAFETY:
            // - reserve(lower) guarantees capacity for `lower` more elements
            // - We add bounds check (len < cap) to protect against malicious iterators
            //   that yield more elements than size_hint promised
            // - If iter.next() panics, guard drops and sets len, ensuring cleanup
            unsafe {
                let ptr = guard.vec.ptr.as_ptr();
                for item in iter {
                    if guard.written_count >= cap {
                        // Iterator lied about size - fall back to safe push
                        // Guard already has correct written_count, push updates len
                        let count = guard.written_count;
                        guard.vec.set_len(count);
                        guard.vec.push(item);
                        guard.written_count = guard.vec.len();
                        continue;
                    }
                    ptr::write(ptr.add(guard.written_count), item);
                    guard.written_count += 1;
                }
            }

            // Success! Set final length and forget the guard
            let final_len = guard.written_count;
            mem::forget(guard);
            // SAFETY: final_len equals the number of elements written by the loop.
            // reserve() ensured capacity, and the loop wrote exactly final_len elements.
            unsafe {
                self.set_len(final_len);
            }
        } else {
            // Fallback for unknown size - push is already panic-safe
            self.reserve(lower);
            for item in iter {
                self.push(item);
            }
        }
    }

    /// Extend from a slice, cloning each element.
    ///
    /// This is faster than `extend(slice.iter().cloned())` because it avoids
    /// the `Cloned` iterator adapter overhead. Profile shows the adapter adds
    /// 7x overhead vs actual clone cost.
    ///
    /// OPTIMIZATION: Uses pointer increment instead of indexed access to reduce
    /// per-element overhead from enumerate() + ptr.add(i).
    #[inline]
    pub fn extend_clone(&mut self, slice: &[T])
    where
        T: Clone,
    {
        let slice_len = slice.len();
        if slice_len == 0 {
            return;
        }
        self.reserve(slice_len);
        let original_len = self.len();

        // RAII guard for panic safety: if clone() panics, we set the length
        // to the number of successfully cloned elements so Drop can clean up.
        struct ExtendCloneGuard<'a, T> {
            vec: &'a mut CompactVec<T>,
            written_count: usize,
        }

        impl<T> Drop for ExtendCloneGuard<'_, T> {
            fn drop(&mut self) {
                // SAFETY: written_count tracks how many elements were successfully
                // cloned before a panic. Setting len to this value ensures Drop
                // will clean up exactly those elements. This is only reached on panic
                // (normal path uses mem::forget).
                unsafe {
                    self.vec.set_len(self.written_count);
                }
            }
        }

        let mut guard = ExtendCloneGuard {
            vec: self,
            written_count: original_len,
        };

        // SAFETY: reserve() guarantees capacity for slice_len more elements
        // If clone() panics, guard drops and sets len, ensuring cleanup
        unsafe {
            let mut dst = guard.vec.ptr.as_ptr().add(original_len);
            for item in slice {
                ptr::write(dst, item.clone());
                guard.written_count += 1;
                dst = dst.add(1);
            }
        }

        // Success! Set final length and forget the guard
        let final_len = guard.written_count;
        mem::forget(guard);
        // SAFETY: final_len equals original_len + slice_len. reserve() ensured capacity,
        // and the loop successfully cloned all slice_len elements.
        unsafe {
            self.set_len(final_len);
        }
    }

    /// Extends the vector by copying elements from a slice (optimized for Copy types).
    ///
    /// This uses `ptr::copy_nonoverlapping` (memcpy) which is significantly faster
    /// than iterating and cloning for simple types like i64.
    #[inline]
    pub fn extend_copy(&mut self, slice: &[T])
    where
        T: Copy,
    {
        let slice_len = slice.len();
        if slice_len == 0 {
            return;
        }
        self.reserve(slice_len);
        let len = self.len();

        // SAFETY:
        // - reserve() guarantees capacity
        // - T is Copy, so no panic safety issues during copy
        // - ptrs are valid
        unsafe {
            let dst = self.ptr.as_ptr().add(len);
            ptr::copy_nonoverlapping(slice.as_ptr(), dst, slice_len);
            self.set_len(len + slice_len);
        }
    }

    /// Returns a slice containing all elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr is valid and aligned, and len() elements are initialized.
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

    /// Drains elements from `start` to end, returning an iterator.
    /// Elements are removed from the vector.
    #[inline]
    pub fn drain(&mut self, range: std::ops::RangeFrom<usize>) -> Drain<'_, T> {
        let start = range.start;
        let len = self.len();
        assert!(start <= len, "drain start index out of bounds");

        // Shorten the vector before yielding anything: a Drain that is leaked
        // after moving elements out must not leave them owned by the vector.
        // SAFETY: start <= len, and [0..start] stays initialized.
        unsafe {
            self.set_len(start);
        }

        Drain {
            vec: self,
            current: start,
            end: len,
        }
    }

    /// Gets a reference to an element.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            // SAFETY: index < len is checked above, so the element is valid.
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Gets a mutable reference to an element.
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            // SAFETY: index < len is checked above, so the element is valid.
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

        // SAFETY: ptr was allocated by the global allocator with the given capacity,
        // len elements are initialized, and we've forgotten self to prevent double-free.
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
            // SAFETY: Vec always has a non-null pointer when capacity > 0.
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
            // SAFETY: All len elements are initialized and valid for dropping.
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), len));
            }
        }

        // Deallocate memory
        if mem::size_of::<T>() > 0 {
            let layout = Layout::array::<T>(self.capacity()).unwrap();
            // SAFETY: ptr was allocated with this layout and capacity > 0.
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

        // RAII guard for panic safety: if clone() panics, we set the length
        // to the number of successfully cloned elements so Drop can clean up.
        struct CloneGuard<'a, T> {
            vec: &'a mut CompactVec<T>,
            cloned_count: usize,
        }

        impl<T> Drop for CloneGuard<'_, T> {
            fn drop(&mut self) {
                // SAFETY: cloned_count tracks how many elements were successfully
                // cloned before a panic. Setting len to this value ensures Drop
                // will clean up exactly those elements. This is only reached on panic
                // (normal path uses mem::forget).
                unsafe {
                    self.vec.set_len(self.cloned_count);
                }
            }
        }

        let mut guard = CloneGuard {
            vec: &mut new_vec,
            cloned_count: 0,
        };

        // SAFETY:
        // - with_capacity(len) guarantees capacity >= len
        // - We write elements to indices 0..len
        // - If clone() panics, guard drops and sets len to cloned_count,
        //   ensuring proper cleanup of partially cloned elements
        unsafe {
            let src = self.ptr.as_ptr();
            let dst = guard.vec.ptr.as_ptr();
            for i in 0..len {
                ptr::write(dst.add(i), (*src.add(i)).clone());
                guard.cloned_count += 1;
            }
        }

        // Success! Set final length and forget the guard (prevent double-set)
        let cloned = guard.cloned_count;
        mem::forget(guard);
        // SAFETY: cloned equals len (the number of elements in self).
        // with_capacity(len) ensured capacity, and all len elements were cloned.
        unsafe {
            new_vec.set_len(cloned);
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

impl<T> Index<std::ops::Range<usize>> for CompactVec<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, range: std::ops::Range<usize>) -> &[T] {
        &self.as_slice()[range]
    }
}

impl<T> Index<std::ops::RangeFrom<usize>> for CompactVec<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, range: std::ops::RangeFrom<usize>) -> &[T] {
        &self.as_slice()[range]
    }
}

impl<T> Index<std::ops::RangeTo<usize>> for CompactVec<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, range: std::ops::RangeTo<usize>) -> &[T] {
        &self.as_slice()[range]
    }
}

impl<T> Index<std::ops::RangeFull> for CompactVec<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, _range: std::ops::RangeFull) -> &[T] {
        self.as_slice()
    }
}

impl<T> FromIterator<T> for CompactVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();

        // If exact size is known, write directly with bounds protection
        if Some(lower) == upper && lower > 0 {
            let mut vec = Self::with_capacity(lower);
            let cap = vec.capacity();

            // RAII guard for panic safety: if iter.next() panics, we set the length
            // to the number of successfully written elements so Drop can clean up.
            struct FromIterGuard<'a, T> {
                vec: &'a mut CompactVec<T>,
                written_count: usize,
            }

            impl<T> Drop for FromIterGuard<'_, T> {
                fn drop(&mut self) {
                    // SAFETY: written_count tracks how many elements were successfully
                    // written before a panic. Setting len to this value ensures Drop
                    // will clean up exactly those elements. This is only reached on panic
                    // (normal path uses mem::forget).
                    unsafe {
                        self.vec.set_len(self.written_count);
                    }
                }
            }

            let mut guard = FromIterGuard {
                vec: &mut vec,
                written_count: 0,
            };

            // SAFETY:
            // - with_capacity(lower) guarantees capacity >= lower
            // - We add bounds check (len < cap) to protect against malicious iterators
            //   that yield more elements than size_hint promised
            // - If iter.next() panics, guard drops and sets len, ensuring cleanup
            unsafe {
                let ptr = guard.vec.ptr.as_ptr();
                for item in iter {
                    if guard.written_count >= cap {
                        // Iterator lied about size - fall back to safe push
                        let count = guard.written_count;
                        guard.vec.set_len(count);
                        guard.vec.push(item);
                        guard.written_count = guard.vec.len();
                        continue;
                    }
                    ptr::write(ptr.add(guard.written_count), item);
                    guard.written_count += 1;
                }
            }

            // Success! Set final length and forget the guard
            let final_len = guard.written_count;
            mem::forget(guard);
            // SAFETY: final_len equals the number of elements written by the loop.
            // with_capacity() ensured capacity, and the loop wrote exactly final_len elements.
            unsafe {
                vec.set_len(final_len);
            }
            vec
        } else {
            // Fallback for unknown size - push is already panic-safe
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

/// Drain iterator for CompactVec.
/// Removes elements from the vector as they are iterated.
pub struct Drain<'a, T> {
    vec: &'a mut CompactVec<T>,
    current: usize,
    end: usize,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.current < self.end {
            // SAFETY: current < end <= original len, so the element is valid.
            // After reading, we increment current so it won't be read again.
            let item = unsafe { ptr::read(self.vec.ptr.as_ptr().add(self.current)) };
            self.current += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {
    fn len(&self) -> usize {
        self.end - self.current
    }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        let remaining = self.end - self.current;
        if remaining > 0 {
            let first = self.current;
            self.current = self.end;
            // Drop the rest as a slice: slice drop glue keeps dropping the
            // remaining elements when one destructor panics, where an
            // element-by-element loop would strand them. The vector no longer
            // owns this range, so nothing else would ever free them.
            // SAFETY: [first..end] are initialized and owned by this drain.
            unsafe {
                let start = self.vec.ptr.as_ptr().add(first);
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(start, remaining));
            }
        }
        // The length was already lowered to `start` by `drain()`, so there is
        // nothing to restore here.
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
            // SAFETY: index < len, so the element is valid. After reading,
            // we increment index so it won't be read again.
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

/// Creates a [`CompactVec`] containing the arguments.
///
/// `compact_vec!` allows creating a `CompactVec` with the same syntax as `vec![]`:
///
/// ```ignore
/// let v = compact_vec![1, 2, 3];
/// assert_eq!(v.as_slice(), &[1, 2, 3]);
/// ```
#[macro_export]
macro_rules! compact_vec {
    () => {
        $crate::common::CompactVec::new()
    };
    ($($elem:expr),+ $(,)?) => {{
        // Use array to get count at compile time, then collect
        let arr = [$($elem),+];
        let mut vec = $crate::common::CompactVec::with_capacity(arr.len());
        for elem in arr {
            vec.push(elem);
        }
        vec
    }};
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

    /// A retain predicate that panics must not leave the vector holding
    /// duplicated elements: every element is dropped exactly once.
    #[test]
    fn test_retain_panic_does_not_double_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct Tracked(usize);

        impl Drop for Tracked {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vec: CompactVec<Tracked> = CompactVec::new();
            for i in 0..5 {
                vec.push(Tracked(i));
            }
            vec.retain(|item| {
                if item.0 == 3 {
                    panic!("intentional panic in retain predicate");
                }
                item.0 != 2
            });
        }));

        assert!(result.is_err(), "the predicate panic must propagate");
        assert_eq!(
            DROP_COUNT.load(Ordering::SeqCst),
            5,
            "each of the 5 elements must be dropped exactly once"
        );
    }

    /// A destructor that panics while `retain` is dropping a rejected element
    /// must not hand that element back to the vector.
    #[test]
    fn test_retain_destructor_panic_does_not_double_drop() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        static PANICKED: AtomicBool = AtomicBool::new(false);

        struct PanicOnDrop(usize);

        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
                // Panic once, so a second drop of the same slot is observable
                // in the counter instead of aborting the process.
                if self.0 == 2 && !PANICKED.swap(true, Ordering::SeqCst) {
                    panic!("intentional panic in destructor");
                }
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        PANICKED.store(false, Ordering::SeqCst);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vec: CompactVec<PanicOnDrop> = CompactVec::new();
            for i in 0..5 {
                vec.push(PanicOnDrop(i));
            }
            vec.retain(|item| item.0 != 2);
        }));

        assert!(result.is_err(), "the destructor panic must propagate");
        assert_eq!(
            DROP_COUNT.load(Ordering::SeqCst),
            5,
            "each of the 5 elements must be dropped exactly once"
        );
    }

    /// A destructor that panics while a Drain is cleaning up must not strand
    /// the elements behind it: the vector no longer owns them.
    #[test]
    fn test_drain_drop_finishes_after_destructor_panic() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        static PANICKED: AtomicBool = AtomicBool::new(false);

        struct PanicOnDrop(usize);

        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
                if self.0 == 2 && !PANICKED.swap(true, Ordering::SeqCst) {
                    panic!("intentional panic in destructor");
                }
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        PANICKED.store(false, Ordering::SeqCst);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vec: CompactVec<PanicOnDrop> = CompactVec::new();
            for i in 0..5 {
                vec.push(PanicOnDrop(i));
            }
            drop(vec.drain(1..));
        }));

        assert!(result.is_err(), "the destructor panic must propagate");
        assert_eq!(
            DROP_COUNT.load(Ordering::SeqCst),
            5,
            "the drain must drop 1..5 and the vector must drop 0, none stranded"
        );
    }

    /// A partially consumed Drain that is forgotten must leave the vector at
    /// the drain start, so the consumed elements are not dropped a second time.
    #[test]
    fn test_forgotten_drain_does_not_double_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct Tracked;

        impl Drop for Tracked {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        let mut vec: CompactVec<Tracked> = CompactVec::new();
        for _ in 0..4 {
            vec.push(Tracked);
        }

        let mut drain = vec.drain(2..);
        let taken = drain.next();
        assert!(taken.is_some());
        drop(taken);
        std::mem::forget(drain);

        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(
            vec.len(),
            2,
            "drain must lower the length before yielding elements"
        );

        drop(vec);

        // 0 and 1 are dropped with the vector, 2 was dropped above, 3 is leaked
        // by mem::forget. Nothing is dropped twice.
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    }

    /// Repeated small reservations must grow geometrically, not once per call.
    /// A zero-sized element type never touches the allocator, with or
    /// without a requested capacity.
    #[test]
    fn test_zero_sized_elements_with_capacity() {
        let mut v: CompactVec<()> = CompactVec::with_capacity(8);
        assert_eq!(v.capacity(), 8);
        for _ in 0..20 {
            v.push(());
        }
        assert_eq!(v.len(), 20);
        assert_eq!(v.pop(), Some(()));
        v.clear();
        assert!(v.is_empty());
    }

    /// The length and capacity share one u32 each; asking for more is an
    /// error, not a silent cap that a later push would write past.
    #[test]
    #[should_panic(expected = "CompactVec capacity overflow")]
    fn test_reserve_past_u32_panics() {
        let mut v: CompactVec<u8> = CompactVec::new();
        v.push(1);
        v.reserve(u32::MAX as usize);
    }

    #[test]
    fn test_reserve_amortizes_and_reserve_exact_does_not() {
        let mut v: CompactVec<u64> = CompactVec::new();
        let mut reallocations = 0;
        let mut last_cap = v.capacity();
        for i in 0..1000u64 {
            v.reserve(1);
            v.push(i);
            if v.capacity() != last_cap {
                reallocations += 1;
                last_cap = v.capacity();
            }
        }
        assert!(
            reallocations <= 12,
            "reserve(1) x1000 reallocated {reallocations} times"
        );

        let mut e: CompactVec<u64> = CompactVec::new();
        e.reserve_exact(37);
        assert_eq!(e.capacity(), 37);
        e.reserve_exact(10);
        assert_eq!(e.capacity(), 37, "already has room, must not grow");
    }

    /// Test panic safety in Clone implementation
    #[test]
    fn test_clone_panic_safety() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        static CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct PanicOnThirdClone(usize);

        impl Clone for PanicOnThirdClone {
            fn clone(&self) -> Self {
                let count = CLONE_COUNT.fetch_add(1, Ordering::SeqCst);
                if count == 2 {
                    panic!("Intentional panic on third clone");
                }
                PanicOnThirdClone(self.0)
            }
        }

        impl Drop for PanicOnThirdClone {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        // Reset counters
        DROP_COUNT.store(0, Ordering::SeqCst);
        CLONE_COUNT.store(0, Ordering::SeqCst);

        let mut vec = CompactVec::new();
        vec.push(PanicOnThirdClone(1));
        vec.push(PanicOnThirdClone(2));
        vec.push(PanicOnThirdClone(3));
        vec.push(PanicOnThirdClone(4));

        // Try to clone - should panic on third element
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _cloned = vec.clone();
        }));

        assert!(result.is_err(), "Should have panicked");

        // Verify panic safety: 2 successfully cloned elements should be dropped
        // by the CloneGuard when the panic unwinds
        let drops_from_cleanup = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            drops_from_cleanup, 2,
            "Should have dropped 2 successfully cloned elements, got {}",
            drops_from_cleanup
        );

        // Now drop the original vec (4 elements)
        drop(vec);
        let total_drops = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            total_drops, 6,
            "Total drops should be 6 (2 from cleanup + 4 from original), got {}",
            total_drops
        );
    }

    /// Test panic safety in extend_clone implementation
    #[test]
    fn test_extend_clone_panic_safety() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        static CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct PanicOnThirdClone(usize);

        impl Clone for PanicOnThirdClone {
            fn clone(&self) -> Self {
                let count = CLONE_COUNT.fetch_add(1, Ordering::SeqCst);
                if count == 2 {
                    panic!("Intentional panic on third clone");
                }
                PanicOnThirdClone(self.0)
            }
        }

        impl Drop for PanicOnThirdClone {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        // Reset counters
        DROP_COUNT.store(0, Ordering::SeqCst);
        CLONE_COUNT.store(0, Ordering::SeqCst);

        // Create source slice (won't be dropped, just borrowed)
        let source = [
            PanicOnThirdClone(1),
            PanicOnThirdClone(2),
            PanicOnThirdClone(3),
            PanicOnThirdClone(4),
        ];

        let mut vec: CompactVec<PanicOnThirdClone> = CompactVec::new();

        // Try to extend_clone - should panic on third element
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            vec.extend_clone(&source);
        }));

        assert!(result.is_err(), "Should have panicked");

        // The guard has set vec's length to 2 (the successfully cloned elements)
        // Now when we drop vec, those 2 elements should be properly dropped
        assert_eq!(
            vec.len(),
            2,
            "Guard should have set length to 2 (elements cloned before panic)"
        );

        // Drop vec - this should drop the 2 elements the guard accounted for
        drop(vec);
        let drops_after_vec_drop = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            drops_after_vec_drop, 2,
            "Should have dropped 2 successfully cloned elements when vec dropped, got {}",
            drops_after_vec_drop
        );

        // Drop source (4 elements)
        drop(source);
        let total_drops = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            total_drops, 6,
            "Total drops should be 6 (2 from vec + 4 from source), got {}",
            total_drops
        );
    }

    /// Test panic safety in from_iter implementation
    #[test]
    fn test_from_iter_panic_safety() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        #[allow(dead_code)] // Field used to give struct a non-zero size
        struct PanicOnThirdNext(usize);

        impl Drop for PanicOnThirdNext {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        // An iterator that panics on the third next() call
        struct PanickingIter {
            current: usize,
            max: usize,
        }

        impl Iterator for PanickingIter {
            type Item = PanicOnThirdNext;

            fn next(&mut self) -> Option<Self::Item> {
                if self.current >= self.max {
                    return None;
                }
                self.current += 1;
                if self.current == 3 {
                    panic!("Intentional panic on third next()");
                }
                Some(PanicOnThirdNext(self.current))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let remaining = self.max - self.current;
                (remaining, Some(remaining))
            }
        }

        impl ExactSizeIterator for PanickingIter {}

        // Reset counter
        DROP_COUNT.store(0, Ordering::SeqCst);

        // Try to collect - should panic on third element
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _vec: CompactVec<PanicOnThirdNext> = PanickingIter { current: 0, max: 5 }.collect();
        }));

        assert!(result.is_err(), "Should have panicked");

        // Verify panic safety: 2 successfully written elements should be dropped
        let drops_after_panic = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            drops_after_panic, 2,
            "Should have dropped 2 successfully written elements, got {}",
            drops_after_panic
        );
    }
}
