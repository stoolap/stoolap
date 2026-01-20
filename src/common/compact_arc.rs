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

//! CompactArc - An Arc variant with thin pointers for DSTs
//!
//! This module provides `CompactArc<T>`, a thread-safe reference-counted pointer
//! optimized for dynamically-sized types (DSTs) like `str` and `[T]`.
//!
//! ## When to Use CompactArc
//!
//! **Use for DSTs** (`str`, `[T]`) when you have many clones sharing one allocation:
//! - Stack pointer: 8 bytes (thin) vs std::Arc's 16 bytes (fat)
//! - Heap header: 24 bytes vs std::Arc's 16 bytes
//! - Net savings: 8 bytes per clone minus 8 bytes per allocation
//! - Break-even: 2+ clones per allocation
//!
//! **Avoid for sized types** (`i64`, `String`, structs):
//! - Stack pointer: 8 bytes (same as std::Arc)
//! - Heap header: 24 bytes vs std::Arc's 16 bytes
//! - Net cost: 8 bytes MORE per allocation
//!
//! ## Memory Layout
//!
//! All types use a unified header with a stored dropper function:
//!
//! ```text
//! Stack:  [ptr: 8 bytes] ──────────────────┐
//!                                          ▼
//! Heap:   [refcount: 8][dropper: 8][len: 8][data...]
//! ```
//!
//! The dropper function pointer captures type-specific drop and dealloc logic,
//! enabling a single generic Drop impl for all types including DSTs.
//!
//! ## Pointer Sizes (All Thin!)
//!
//! | Type | CompactArc | std::Arc |
//! |------|------------|----------|
//! | `CompactArc<i64>` | 8 bytes | 8 bytes |
//! | `CompactArc<str>` | 8 bytes | 16 bytes |
//! | `CompactArc<[T]>` | 8 bytes | 16 bytes |

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
use std::ops::Deref;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

// ============================================================================
// Unified Header - The Magic Trick!
// ============================================================================

/// Unified header for all CompactArc allocations.
/// The dropper function pointer captures type-specific behavior,
/// allowing a single Drop impl to work for all types.
#[repr(C)]
struct Header {
    count: AtomicUsize,
    /// Type-erased dropper that handles both element drops and deallocation.
    /// This is the key trick that enables thin pointers with a single Drop impl!
    dropper: unsafe fn(*mut Header),
    /// Length of data. For sized types: 0. For str: byte length. For [T]: element count.
    len: usize,
    // Data follows immediately after, aligned appropriately
}

/// Returns the byte offset from the header to the data for type T.
/// For `CompactArc<[T]>`, pass the element type T, not the slice type.
#[inline]
const fn data_offset_for<T>() -> usize {
    let header_size = mem::size_of::<Header>();
    let align = mem::align_of::<T>();
    (header_size + align - 1) & !(align - 1)
}

// ============================================================================
// CompactArc - Unified type with thin pointers!
// ============================================================================

/// A thread-safe reference-counted pointer without weak reference support.
///
/// `CompactArc<T>` provides shared ownership of a value of type `T`, allocated
/// on the heap. It saves memory compared to `std::sync::Arc`:
/// - 8 bytes less per allocation (no weak count)
/// - Thin pointers for DSTs (8 bytes instead of 16 for `str` and `[T]`)
///
/// # Pointer Sizes
///
/// | Type | Size |
/// |------|------|
/// | `CompactArc<i64>` | 8 bytes |
/// | `CompactArc<str>` | 8 bytes (thin!) |
/// | `CompactArc<[T]>` | 8 bytes (thin!) |
pub struct CompactArc<T: ?Sized> {
    /// Thin pointer to Header (always 8 bytes, even for DSTs!)
    ptr: NonNull<Header>,
    _marker: PhantomData<T>,
}

// SAFETY: CompactArc can be sent between threads if T can be sent and shared.
unsafe impl<T: ?Sized + Send + Sync> Send for CompactArc<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for CompactArc<T> {}

// ============================================================================
// THE SINGLE DROP IMPL - Works for ALL types!
// ============================================================================

impl<T: ?Sized> Drop for CompactArc<T> {
    #[inline]
    fn drop(&mut self) {
        let header = self.ptr.as_ptr();
        let old_count = unsafe { (*header).count.fetch_sub(1, AtomicOrdering::Release) };

        if old_count == 1 {
            std::sync::atomic::fence(AtomicOrdering::Acquire);
            // Call the stored dropper - it knows how to handle this specific type!
            unsafe {
                let dropper = (*header).dropper;
                dropper(header);
            }
        }
    }
}

// ============================================================================
// THE SINGLE CLONE IMPL - Works for ALL types!
// ============================================================================

impl<T: ?Sized> Clone for CompactArc<T> {
    #[inline]
    fn clone(&self) -> Self {
        let header = self.ptr.as_ptr();
        let old_count = unsafe { (*header).count.fetch_add(1, AtomicOrdering::Relaxed) };

        if old_count > isize::MAX as usize {
            std::process::abort();
        }

        CompactArc {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// Common methods
// ============================================================================

impl<T: ?Sized> CompactArc<T> {
    /// Returns `true` if the two `CompactArc`s point to the same allocation.
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::addr_eq(this.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns the number of strong references to this allocation.
    ///
    /// Note: This uses `Relaxed` ordering and should only be used for
    /// debugging/logging purposes, not for synchronization decisions.
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        unsafe { (*this.ptr.as_ptr()).count.load(AtomicOrdering::Relaxed) }
    }

    /// Returns `true` if this is the only reference to the allocation.
    ///
    /// Uses `Acquire` ordering to synchronize with `Release` in `drop`,
    /// ensuring visibility of all modifications made by other threads
    /// before they dropped their references.
    #[inline]
    fn is_unique(this: &Self) -> bool {
        unsafe { (*this.ptr.as_ptr()).count.load(AtomicOrdering::Acquire) == 1 }
    }
}

// ============================================================================
// Sized type implementations
// ============================================================================

/// Dropper for sized types - drops data and deallocates
unsafe fn drop_sized<T>(header: *mut Header) {
    let data_offset = data_offset_for::<T>();
    let align = mem::align_of::<T>().max(mem::align_of::<Header>());

    // Drop the data
    let data_ptr = (header as *mut u8).add(data_offset) as *mut T;
    ptr::drop_in_place(data_ptr);

    // Deallocate
    let layout = Layout::from_size_align_unchecked(data_offset + mem::size_of::<T>(), align);
    dealloc(header as *mut u8, layout);
}

impl<T> CompactArc<T> {
    /// Creates a new `CompactArc<T>` containing the given value.
    #[inline]
    #[must_use]
    pub fn new(data: T) -> Self {
        let data_offset = data_offset_for::<T>();
        let align = mem::align_of::<T>().max(mem::align_of::<Header>());
        let total_size = data_offset + mem::size_of::<T>();
        let layout = Layout::from_size_align(total_size, align).expect("layout overflow");

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            // Write header
            let header = ptr as *mut Header;
            ptr::write(
                header,
                Header {
                    count: AtomicUsize::new(1),
                    dropper: drop_sized::<T>, // Store type-specific dropper!
                    len: 0,                   // Sized types use 0
                },
            );

            // Write data
            let data_ptr = ptr.add(data_offset) as *mut T;
            ptr::write(data_ptr, data);

            CompactArc {
                ptr: NonNull::new_unchecked(header),
                _marker: PhantomData,
            }
        }
    }

    /// Attempts to unwrap the `CompactArc`, returning the inner value if this
    /// is the only reference.
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        let header = this.ptr.as_ptr();

        if unsafe {
            (*header)
                .count
                .compare_exchange(1, 0, AtomicOrdering::Acquire, AtomicOrdering::Relaxed)
                .is_ok()
        } {
            let _ = ManuallyDrop::new(this);

            unsafe {
                // Read data
                let data_offset = data_offset_for::<T>();
                let data_ptr = (header as *const u8).add(data_offset) as *const T;
                let data = ptr::read(data_ptr);

                // Deallocate (without calling dropper since we took the data)
                let align = mem::align_of::<T>().max(mem::align_of::<Header>());
                let layout =
                    Layout::from_size_align_unchecked(data_offset + mem::size_of::<T>(), align);
                dealloc(header as *mut u8, layout);

                Ok(data)
            }
        } else {
            Err(this)
        }
    }

    /// Gets a mutable reference to the inner value, if there are no other references.
    ///
    /// Uses `Acquire` ordering to synchronize with other threads that may have
    /// dropped their references, ensuring all their modifications are visible.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Self::is_unique(this) {
            unsafe {
                let data_ptr = (this.ptr.as_ptr() as *mut u8).add(data_offset_for::<T>()) as *mut T;
                Some(&mut *data_ptr)
            }
        } else {
            None
        }
    }

    /// Makes a mutable reference to the inner value (clone-on-write).
    ///
    /// If there are other references, clones the data into a new allocation.
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        // Check if we're the only reference (uses Acquire ordering)
        if !Self::is_unique(this) {
            // Clone the data since there are other references
            *this = CompactArc::new((**this).clone());
        }
        // SAFETY: After the above, we're guaranteed to be the only reference
        Self::get_mut(this).unwrap()
    }

    /// Returns a raw pointer to the contained data.
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        &**this as *const T
    }

    /// Converts a `CompactArc<T>` into a raw pointer.
    #[inline]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = &*this as *const T;
        mem::forget(this);
        ptr
    }

    /// Constructs a `CompactArc<T>` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The raw pointer must have been previously returned by `CompactArc::into_raw`.
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        let header = (ptr as *const u8).sub(data_offset_for::<T>()) as *mut Header;
        CompactArc {
            ptr: NonNull::new_unchecked(header),
            _marker: PhantomData,
        }
    }
}

impl<T> Deref for CompactArc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe {
            let data_ptr = (self.ptr.as_ptr() as *const u8).add(data_offset_for::<T>()) as *const T;
            &*data_ptr
        }
    }
}

impl<T: Default> Default for CompactArc<T> {
    #[inline]
    fn default() -> Self {
        CompactArc::new(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for CompactArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display> fmt::Display for CompactArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for CompactArc<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if CompactArc::ptr_eq(self, other) {
            return true;
        }
        **self == **other
    }
}

impl<T: Eq> Eq for CompactArc<T> {}

impl<T: PartialOrd> PartialOrd for CompactArc<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: Ord> Ord for CompactArc<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: Hash> Hash for CompactArc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T> Borrow<T> for CompactArc<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T> AsRef<T> for CompactArc<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T> From<T> for CompactArc<T> {
    #[inline]
    fn from(value: T) -> Self {
        CompactArc::new(value)
    }
}

// ============================================================================
// DST Support: str (Thin Pointer!)
// ============================================================================

/// Dropper for str - just deallocates (no element drops needed)
unsafe fn drop_str(header: *mut Header) {
    let len = (*header).len;
    let data_offset = data_offset_for::<u8>(); // str has align 1
    let total_size = data_offset + len;
    let layout = Layout::from_size_align_unchecked(total_size, mem::align_of::<Header>());
    dealloc(header as *mut u8, layout);
}

impl CompactArc<str> {
    /// Creates a new `CompactArc<str>` from a string slice.
    ///
    /// The pointer is only 8 bytes (thin), with length stored in heap header.
    #[must_use]
    pub fn from_str_slice(s: &str) -> Self {
        let len = s.len();
        let data_offset = data_offset_for::<u8>(); // str has align 1
        let total_size = data_offset + len;
        let layout = Layout::from_size_align(total_size, mem::align_of::<Header>())
            .expect("layout overflow");

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            // Write header
            let header = ptr as *mut Header;
            ptr::write(
                header,
                Header {
                    count: AtomicUsize::new(1),
                    dropper: drop_str, // Store str-specific dropper!
                    len,
                },
            );

            // Write string bytes
            let data_ptr = ptr.add(data_offset);
            ptr::copy_nonoverlapping(s.as_ptr(), data_ptr, len);

            CompactArc {
                ptr: NonNull::new_unchecked(header),
                _marker: PhantomData,
            }
        }
    }

    /// Returns the length of the string in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        unsafe { (*self.ptr.as_ptr()).len }
    }

    /// Returns true if the string is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Deref for CompactArc<str> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        unsafe {
            let header = self.ptr.as_ptr();
            let len = (*header).len;
            let data_offset = data_offset_for::<u8>(); // str has align 1
            let data_ptr = (header as *const u8).add(data_offset);
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(data_ptr, len))
        }
    }
}

impl From<&str> for CompactArc<str> {
    #[inline]
    fn from(s: &str) -> Self {
        CompactArc::from_str_slice(s)
    }
}

impl From<String> for CompactArc<str> {
    #[inline]
    fn from(s: String) -> Self {
        CompactArc::from_str_slice(&s)
    }
}

impl fmt::Debug for CompactArc<str> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl fmt::Display for CompactArc<str> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl PartialEq for CompactArc<str> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if CompactArc::ptr_eq(self, other) {
            return true;
        }
        **self == **other
    }
}

impl Eq for CompactArc<str> {}

impl PartialOrd for CompactArc<str> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompactArc<str> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl Hash for CompactArc<str> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl Borrow<str> for CompactArc<str> {
    fn borrow(&self) -> &str {
        self
    }
}

impl AsRef<str> for CompactArc<str> {
    fn as_ref(&self) -> &str {
        self
    }
}

// ============================================================================
// DST Support: [T] (Thin Pointer!)
// ============================================================================

/// Dropper for slices - drops each element then deallocates
#[allow(clippy::manual_slice_size_calculation)] // We don't have a slice ref, only length from header
unsafe fn drop_slice<T>(header: *mut Header) {
    let len = (*header).len;
    let data_offset = data_offset_for::<T>();
    let align = mem::align_of::<T>().max(mem::align_of::<Header>());

    // Drop elements
    let data_ptr = (header as *mut u8).add(data_offset) as *mut T;
    ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(data_ptr, len));

    // Deallocate
    let layout = Layout::from_size_align_unchecked(data_offset + mem::size_of::<T>() * len, align);
    dealloc(header as *mut u8, layout);
}

impl<T> CompactArc<[T]> {
    /// Creates a new `CompactArc<[T]>` by moving elements from a Vec.
    ///
    /// This is more efficient than `from_slice` as it moves elements instead of cloning.
    #[must_use]
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let len = vec.len();
        let data_offset = data_offset_for::<T>();
        let align = mem::align_of::<T>().max(mem::align_of::<Header>());
        let data_size = mem::size_of::<T>() * len;
        let layout =
            Layout::from_size_align(data_offset + data_size, align).expect("layout overflow");

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            // Write header
            let header = ptr as *mut Header;
            ptr::write(
                header,
                Header {
                    count: AtomicUsize::new(1),
                    dropper: drop_slice::<T>,
                    len,
                },
            );

            // Move elements from Vec (copy bytes, then prevent Vec from dropping them)
            let data_ptr = ptr.add(data_offset) as *mut T;
            ptr::copy_nonoverlapping(vec.as_ptr(), data_ptr, len);

            // Prevent Vec from dropping the moved elements (buffer will still be freed)
            vec.set_len(0);

            CompactArc {
                ptr: NonNull::new_unchecked(header),
                _marker: PhantomData,
            }
        }
    }
}

impl<T: Clone> CompactArc<[T]> {
    /// Creates a new `CompactArc<[T]>` from a slice by cloning elements.
    ///
    /// The pointer is only 8 bytes (thin), with length stored in heap header.
    ///
    /// # Panic Safety
    ///
    /// If `T::clone()` panics, all successfully cloned elements are dropped
    /// and the allocation is freed. No memory is leaked.
    #[must_use]
    pub fn from_slice(slice: &[T]) -> Self {
        let len = slice.len();
        let data_offset = data_offset_for::<T>();
        let align = mem::align_of::<T>().max(mem::align_of::<Header>());
        let layout = Layout::from_size_align(data_offset + mem::size_of_val(slice), align)
            .expect("layout overflow");

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }

            // Write header
            let header = ptr as *mut Header;
            ptr::write(
                header,
                Header {
                    count: AtomicUsize::new(1),
                    dropper: drop_slice::<T>,
                    len,
                },
            );

            let data_ptr = ptr.add(data_offset) as *mut T;

            // RAII guard for panic safety: if clone() panics, this cleans up
            struct CloneGuard<T> {
                data_ptr: *mut T,
                alloc_ptr: *mut u8,
                layout: Layout,
                written: usize,
            }

            impl<T> Drop for CloneGuard<T> {
                fn drop(&mut self) {
                    unsafe {
                        // Drop all successfully written elements
                        let slice = ptr::slice_from_raw_parts_mut(self.data_ptr, self.written);
                        ptr::drop_in_place(slice);
                        // Deallocate the memory
                        dealloc(self.alloc_ptr, self.layout);
                    }
                }
            }

            let mut guard = CloneGuard {
                data_ptr,
                alloc_ptr: ptr,
                layout,
                written: 0,
            };

            // Clone elements - if this panics, guard cleans up
            for (i, item) in slice.iter().enumerate() {
                ptr::write(data_ptr.add(i), item.clone());
                guard.written += 1;
            }

            // Success! Prevent guard from cleaning up
            mem::forget(guard);

            CompactArc {
                ptr: NonNull::new_unchecked(header),
                _marker: PhantomData,
            }
        }
    }
}

impl<T> CompactArc<[T]> {
    /// Returns the number of elements in the slice.
    #[inline]
    pub fn len(&self) -> usize {
        unsafe { (*self.ptr.as_ptr()).len }
    }

    /// Returns true if the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Deref for CompactArc<[T]> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        unsafe {
            let header = self.ptr.as_ptr();
            let len = (*header).len;
            let data_ptr = (header as *const u8).add(data_offset_for::<T>()) as *const T;
            std::slice::from_raw_parts(data_ptr, len)
        }
    }
}

impl<T: Clone> From<&[T]> for CompactArc<[T]> {
    #[inline]
    fn from(slice: &[T]) -> Self {
        CompactArc::from_slice(slice)
    }
}

impl<T> From<Vec<T>> for CompactArc<[T]> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        CompactArc::from_vec(vec)
    }
}

impl<T: fmt::Debug> fmt::Debug for CompactArc<[T]> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: PartialEq> PartialEq for CompactArc<[T]> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if CompactArc::ptr_eq(self, other) {
            return true;
        }
        **self == **other
    }
}

impl<T: Eq> Eq for CompactArc<[T]> {}

impl<T: PartialOrd> PartialOrd for CompactArc<[T]> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: Ord> Ord for CompactArc<[T]> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: Hash> Hash for CompactArc<[T]> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T> Borrow<[T]> for CompactArc<[T]> {
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T> AsRef<[T]> for CompactArc<[T]> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_deref() {
        let arc = CompactArc::new(42);
        assert_eq!(*arc, 42);
    }

    #[test]
    fn test_clone_and_count() {
        let arc = CompactArc::new(42);
        assert_eq!(CompactArc::strong_count(&arc), 1);

        let arc2 = arc.clone();
        assert_eq!(CompactArc::strong_count(&arc), 2);
        assert_eq!(CompactArc::strong_count(&arc2), 2);

        drop(arc2);
        assert_eq!(CompactArc::strong_count(&arc), 1);
    }

    #[test]
    fn test_ptr_eq() {
        let arc1 = CompactArc::new(42);
        let arc2 = arc1.clone();
        let arc3 = CompactArc::new(42);

        assert!(CompactArc::ptr_eq(&arc1, &arc2));
        assert!(!CompactArc::ptr_eq(&arc1, &arc3));
    }

    #[test]
    fn test_try_unwrap_success() {
        let arc = CompactArc::new(42);
        let value = CompactArc::try_unwrap(arc).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_try_unwrap_failure() {
        let arc = CompactArc::new(42);
        let _arc2 = arc.clone();
        let result = CompactArc::try_unwrap(arc);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_mut() {
        let mut arc = CompactArc::new(42);
        *CompactArc::get_mut(&mut arc).unwrap() = 100;
        assert_eq!(*arc, 100);

        let _arc2 = arc.clone();
        assert!(CompactArc::get_mut(&mut arc).is_none());
    }

    #[test]
    fn test_make_mut() {
        let mut arc = CompactArc::new(42);
        *CompactArc::make_mut(&mut arc) = 100;
        assert_eq!(*arc, 100);

        let arc2 = arc.clone();
        *CompactArc::make_mut(&mut arc) = 200;
        assert_eq!(*arc, 200);
        assert_eq!(*arc2, 100);
    }

    #[test]
    fn test_into_raw_from_raw() {
        let arc = CompactArc::new(42);
        let ptr = CompactArc::into_raw(arc);

        let arc2 = unsafe { CompactArc::from_raw(ptr) };
        assert_eq!(*arc2, 42);
    }

    #[test]
    fn test_debug_display() {
        let arc = CompactArc::new(42);
        assert_eq!(format!("{:?}", arc), "42");
        assert_eq!(format!("{}", arc), "42");
    }

    #[test]
    fn test_equality() {
        let arc1 = CompactArc::new(42);
        let arc2 = CompactArc::new(42);
        let arc3 = CompactArc::new(100);

        assert_eq!(arc1, arc2);
        assert_ne!(arc1, arc3);
    }

    #[test]
    fn test_ordering() {
        let arc1 = CompactArc::new(1);
        let arc2 = CompactArc::new(2);

        assert!(arc1 < arc2);
        assert!(arc2 > arc1);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashMap;

        let arc = CompactArc::new(42);
        let mut map = HashMap::new();
        map.insert(arc.clone(), "value");

        assert_eq!(map.get(&arc), Some(&"value"));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<CompactArc<i32>>();
        assert_sync::<CompactArc<i32>>();
    }

    #[test]
    fn test_sized_pointer_size() {
        // CompactArc<T> should be 8 bytes (thin pointer)
        assert_eq!(std::mem::size_of::<CompactArc<i32>>(), 8);
        assert_eq!(std::mem::size_of::<CompactArc<i64>>(), 8);
        assert_eq!(std::mem::size_of::<CompactArc<String>>(), 8);
    }

    #[test]
    fn test_high_alignment_type() {
        #[repr(align(64))]
        #[derive(Debug, Clone, PartialEq)]
        struct Aligned64 {
            value: u64,
        }

        let arc = CompactArc::new(Aligned64 { value: 42 });
        assert_eq!(arc.value, 42);

        // Verify data pointer is properly aligned
        let data_ptr = CompactArc::as_ptr(&arc);
        assert_eq!(data_ptr as usize % 64, 0, "Data should be 64-byte aligned");

        // Test clone and drop
        let arc2 = arc.clone();
        assert_eq!(arc2.value, 42);
        assert_eq!(CompactArc::strong_count(&arc), 2);

        drop(arc);
        assert_eq!(arc2.value, 42);

        // Test try_unwrap
        let value = CompactArc::try_unwrap(arc2).unwrap();
        assert_eq!(value.value, 42);
    }

    #[test]
    fn test_high_alignment_slice() {
        #[repr(align(32))]
        #[derive(Debug, Clone, PartialEq)]
        struct Aligned32(u32);

        let arr: CompactArc<[Aligned32]> =
            CompactArc::from_slice(&[Aligned32(1), Aligned32(2), Aligned32(3)]);

        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], Aligned32(1));
        assert_eq!(arr[1], Aligned32(2));
        assert_eq!(arr[2], Aligned32(3));

        // Verify first element is properly aligned
        let first_ptr = &arr[0] as *const Aligned32;
        assert_eq!(
            first_ptr as usize % 32,
            0,
            "Elements should be 32-byte aligned"
        );
    }

    #[test]
    fn test_drop_complex_type() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct DropCounter;
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        {
            let arc = CompactArc::new(DropCounter);
            let _arc2 = arc.clone();
            let _arc3 = arc.clone();
        }

        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let arc = CompactArc::new(0);
        let mut handles = vec![];

        for _ in 0..10 {
            let arc_clone = arc.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let _ = *arc_clone;
                    let _another = arc_clone.clone();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        drop(arc);
    }

    // ========================================================================
    // DST Tests: str
    // ========================================================================

    #[test]
    fn test_str_basic() {
        let s: CompactArc<str> = CompactArc::from_str_slice("hello world");
        assert_eq!(&*s, "hello world");
        assert_eq!(s.len(), 11);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_str_empty() {
        let s: CompactArc<str> = CompactArc::from_str_slice("");
        assert_eq!(&*s, "");
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn test_str_clone() {
        let s1: CompactArc<str> = CompactArc::from_str_slice("hello");
        let s2 = s1.clone();

        assert_eq!(&*s1, "hello");
        assert_eq!(&*s2, "hello");
        assert!(CompactArc::ptr_eq(&s1, &s2));
        assert_eq!(CompactArc::strong_count(&s1), 2);
    }

    #[test]
    fn test_str_drop() {
        let s1: CompactArc<str> = CompactArc::from_str_slice("test string");
        let s2 = s1.clone();
        assert_eq!(CompactArc::strong_count(&s1), 2);

        drop(s1);
        assert_eq!(CompactArc::strong_count(&s2), 1);
        assert_eq!(&*s2, "test string");
    }

    #[test]
    fn test_str_unicode() {
        let s: CompactArc<str> = CompactArc::from_str_slice("こんにちは世界");
        assert_eq!(&*s, "こんにちは世界");
        assert_eq!(s.len(), 21);
    }

    #[test]
    fn test_str_from_impls() {
        let s1: CompactArc<str> = CompactArc::from("hello");
        let s2: CompactArc<str> = CompactArc::from(String::from("world"));

        assert_eq!(&*s1, "hello");
        assert_eq!(&*s2, "world");
    }

    #[test]
    fn test_str_thin_pointer() {
        // KEY TEST: CompactArc<str> should be 8 bytes (thin pointer!)
        assert_eq!(std::mem::size_of::<CompactArc<str>>(), 8);
    }

    #[test]
    fn test_str_equality() {
        let s1: CompactArc<str> = CompactArc::from_str_slice("hello");
        let s2: CompactArc<str> = CompactArc::from_str_slice("hello");
        let s3: CompactArc<str> = CompactArc::from_str_slice("world");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_str_hash() {
        use std::collections::HashMap;

        let s: CompactArc<str> = CompactArc::from_str_slice("key");
        let mut map = HashMap::new();
        map.insert(s.clone(), "value");

        assert_eq!(map.get(&s), Some(&"value"));
    }

    // ========================================================================
    // DST Tests: [T]
    // ========================================================================

    #[test]
    fn test_slice_basic() {
        let arr: CompactArc<[i32]> = CompactArc::from_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(&*arr, &[1, 2, 3, 4, 5]);
        assert_eq!(arr.len(), 5);
        assert!(!arr.is_empty());
    }

    #[test]
    fn test_slice_empty() {
        let empty: &[i32] = &[];
        let arr: CompactArc<[i32]> = CompactArc::from_slice(empty);
        assert_eq!(&*arr, empty);
        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_slice_clone() {
        let arr1: CompactArc<[i32]> = CompactArc::from_slice(&[1, 2, 3]);
        let arr2 = arr1.clone();

        assert_eq!(&*arr1, &[1, 2, 3]);
        assert_eq!(&*arr2, &[1, 2, 3]);
        assert!(CompactArc::ptr_eq(&arr1, &arr2));
        assert_eq!(CompactArc::strong_count(&arr1), 2);
    }

    #[test]
    fn test_slice_thin_pointer() {
        // KEY TEST: CompactArc<[T]> should be 8 bytes (thin pointer!)
        assert_eq!(std::mem::size_of::<CompactArc<[i32]>>(), 8);
        assert_eq!(std::mem::size_of::<CompactArc<[String]>>(), 8);
    }

    #[test]
    fn test_slice_from_vec() {
        let arr: CompactArc<[String]> = CompactArc::from(vec![
            String::from("a"),
            String::from("b"),
            String::from("c"),
        ]);
        assert_eq!(arr.len(), 3);
        assert_eq!(&arr[0], "a");
        assert_eq!(&arr[1], "b");
        assert_eq!(&arr[2], "c");
    }

    #[test]
    fn test_slice_drop_elements() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone)]
        struct DropCounter;
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);

        {
            let arr: CompactArc<[DropCounter]> =
                CompactArc::from_slice(&[DropCounter, DropCounter, DropCounter]);
            let _arr2 = arr.clone();
        }

        // 3 from original slice + 3 from arc = 6
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn test_from_slice_panic_safety() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        static CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct PanicOnThird(u32);

        impl Clone for PanicOnThird {
            fn clone(&self) -> Self {
                let count = CLONE_COUNT.fetch_add(1, Ordering::SeqCst);
                if count == 2 {
                    panic!("Panic on third clone!");
                }
                PanicOnThird(self.0)
            }
        }

        impl Drop for PanicOnThird {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        CLONE_COUNT.store(0, Ordering::SeqCst);

        let slice = &[
            PanicOnThird(1),
            PanicOnThird(2),
            PanicOnThird(3),
            PanicOnThird(4),
        ];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: CompactArc<[PanicOnThird]> = CompactArc::from_slice(slice);
        }));

        assert!(result.is_err(), "Should have panicked");

        // Verify panic safety: 2 successfully cloned elements should be dropped
        // (the 3rd clone panicked before being written)
        let drops_from_cleanup = DROP_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            drops_from_cleanup, 2,
            "Should have dropped 2 successfully cloned elements"
        );
    }

    #[test]
    fn test_from_vec_moves_elements() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
        static CLONE_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug, PartialEq)]
        struct MoveTracker(u32);

        impl Clone for MoveTracker {
            fn clone(&self) -> Self {
                CLONE_COUNT.fetch_add(1, Ordering::SeqCst);
                MoveTracker(self.0)
            }
        }

        impl Drop for MoveTracker {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        CLONE_COUNT.store(0, Ordering::SeqCst);

        {
            let vec = vec![MoveTracker(1), MoveTracker(2), MoveTracker(3)];
            let arr: CompactArc<[MoveTracker]> = CompactArc::from_vec(vec);

            // Verify elements are accessible (compare inner values to avoid creating temporaries)
            assert_eq!(arr[0].0, 1);
            assert_eq!(arr[1].0, 2);
            assert_eq!(arr[2].0, 3);

            // No clones should have happened (elements were moved)
            assert_eq!(
                CLONE_COUNT.load(Ordering::SeqCst),
                0,
                "from_vec should move, not clone"
            );
        }

        // Only 3 drops: the elements in the CompactArc
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_dst_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<CompactArc<str>>();
        assert_sync::<CompactArc<str>>();
        assert_send::<CompactArc<[i32]>>();
        assert_sync::<CompactArc<[i32]>>();
    }

    #[test]
    fn test_str_thread_safety() {
        use std::thread;

        let s: CompactArc<str> = CompactArc::from_str_slice("shared string");
        let mut handles = vec![];

        for _ in 0..10 {
            let s_clone = s.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let _ = s_clone.len();
                    let _another = s_clone.clone();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        drop(s);
    }
}
