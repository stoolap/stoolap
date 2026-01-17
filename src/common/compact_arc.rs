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

//! CompactArc - A memory-efficient Arc without weak reference support
//!
//! This module provides `CompactArc<T>`, a thread-safe reference-counted pointer
//! that saves 8 bytes per allocation compared to `std::sync::Arc` by not supporting
//! weak references.
//!
//! ## Memory Layout Comparison
//!
//! | Type | Pointer | Header | Total Overhead |
//! |------|---------|--------|----------------|
//! | `std::sync::Arc<T>` | 8 bytes | 16 bytes (strong + weak) | 24 bytes |
//! | `CompactArc<T>` | 8 bytes | 8 bytes (count only) | 16 bytes |
//!
//! ## When to Use
//!
//! Use `CompactArc` when:
//! - You have many small allocations (like `Value` in database rows)
//! - You don't need weak references
//! - Memory efficiency is important
//!
//! Use `std::sync::Arc` when:
//! - You need `Weak` references (e.g., breaking reference cycles)
//! - You need `Arc::downgrade()` functionality

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

/// Inner structure for CompactArc - just count + data (no weak count)
#[repr(C)]
struct CompactArcInner<T> {
    /// Reference count (strong only, no weak)
    count: AtomicUsize,
    /// The actual data
    data: T,
}

/// A thread-safe reference-counted pointer without weak reference support.
///
/// `CompactArc<T>` provides shared ownership of a value of type `T`, allocated
/// on the heap. It saves 8 bytes per allocation compared to `std::sync::Arc`
/// by not supporting weak references.
///
/// # Thread Safety
///
/// `CompactArc<T>` is `Send` and `Sync` when `T` is `Send` and `Sync`.
/// The reference count is managed with atomic operations.
///
/// # Example
///
/// ```ignore
/// use stoolap::common::CompactArc;
///
/// let value = CompactArc::new(42);
/// let clone = CompactArc::clone(&value);
///
/// assert_eq!(*value, 42);
/// assert_eq!(*clone, 42);
/// ```
pub struct CompactArc<T> {
    ptr: NonNull<CompactArcInner<T>>,
    _marker: PhantomData<CompactArcInner<T>>,
}

// SAFETY: CompactArc can be sent between threads if T can be sent and shared.
// The reference count is managed with atomic operations, ensuring thread-safe
// access to the count. The data T is only accessed through shared references
// (via Deref) unless we have exclusive access (count == 1).
unsafe impl<T: Send + Sync> Send for CompactArc<T> {}
// SAFETY: CompactArc can be shared between threads if T can be shared.
// All accesses to the reference count use atomic operations, and the data
// is accessed through an immutable reference, which is safe when T: Sync.
unsafe impl<T: Send + Sync> Sync for CompactArc<T> {}

impl<T> CompactArc<T> {
    /// Creates a new `CompactArc<T>` containing the given value.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use stoolap::common::CompactArc;
    /// let five = CompactArc::new(5);
    /// ```
    #[inline]
    pub fn new(data: T) -> Self {
        let layout = Layout::new::<CompactArcInner<T>>();

        // Safety: We're allocating a properly aligned, sized block
        let ptr = unsafe { alloc(layout) as *mut CompactArcInner<T> };

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        // Safety: ptr is valid and properly aligned
        unsafe {
            ptr::write(
                ptr,
                CompactArcInner {
                    count: AtomicUsize::new(1),
                    data,
                },
            );
        }

        CompactArc {
            // Safety: We just checked ptr is not null
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            _marker: PhantomData,
        }
    }

    /// Returns the number of strong references to this allocation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use stoolap::common::CompactArc;
    /// let five = CompactArc::new(5);
    /// let _clone = CompactArc::clone(&five);
    /// assert_eq!(CompactArc::strong_count(&five), 2);
    /// ```
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        // SAFETY: The pointer is valid because CompactArc maintains the invariant that
        // ptr always points to a valid, aligned CompactArcInner allocation as long as
        // the CompactArc exists. Relaxed ordering is sufficient for a non-synchronized read.
        unsafe { this.ptr.as_ref().count.load(AtomicOrdering::Relaxed) }
    }

    /// Returns `true` if the two `CompactArc`s point to the same allocation.
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr == other.ptr
    }

    /// Attempts to unwrap the `CompactArc`, returning the inner value if this
    /// is the only reference.
    ///
    /// If there are other references, returns `Err(self)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use stoolap::common::CompactArc;
    /// let x = CompactArc::new(3);
    /// assert_eq!(CompactArc::try_unwrap(x), Ok(3));
    ///
    /// let x = CompactArc::new(4);
    /// let _y = CompactArc::clone(&x);
    /// assert!(CompactArc::try_unwrap(x).is_err());
    /// ```
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // Try to set count from 1 to 0
        // SAFETY: The pointer is valid because CompactArc maintains the invariant that
        // ptr always points to a valid, aligned CompactArcInner allocation.
        if unsafe {
            this.ptr
                .as_ref()
                .count
                .compare_exchange(1, 0, AtomicOrdering::Acquire, AtomicOrdering::Relaxed)
                .is_ok()
        } {
            // We're the only reference, take ownership
            // Don't run Drop since we're manually cleaning up
            let this = ManuallyDrop::new(this);

            // Safety: We just confirmed we're the only reference
            unsafe {
                let data = ptr::read(&this.ptr.as_ref().data);
                let layout = Layout::new::<CompactArcInner<T>>();
                dealloc(this.ptr.as_ptr() as *mut u8, layout);
                Ok(data)
            }
        } else {
            Err(this)
        }
    }

    /// Gets a mutable reference to the inner value, if there are no other
    /// `CompactArc` pointers to the same allocation.
    ///
    /// Returns `None` if there are other references.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Self::strong_count(this) == 1 {
            // Safety: We're the only reference
            unsafe { Some(&mut this.ptr.as_mut().data) }
        } else {
            None
        }
    }

    /// Makes a mutable reference to the inner value.
    ///
    /// If there are other `CompactArc` pointers to the same allocation,
    /// the inner value is cloned to a new allocation.
    ///
    /// This is also known as clone-on-write.
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        // Fast path: we're the only reference
        if Self::strong_count(this) == 1 {
            // Safety: We're the only reference
            unsafe {
                return &mut this.ptr.as_mut().data;
            }
        }

        // Slow path: clone the data
        let new_arc = CompactArc::new((**this).clone());
        *this = new_arc;
        // Safety: We just created a new allocation with count 1
        unsafe { &mut this.ptr.as_mut().data }
    }

    /// Returns a raw pointer to the contained data.
    ///
    /// This does not consume the `CompactArc` and the pointer is valid
    /// as long as the `CompactArc` exists.
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        &**this as *const T
    }

    /// Converts a `CompactArc<T>` into a raw pointer.
    ///
    /// The pointer can be converted back into a `CompactArc` using
    /// `CompactArc::from_raw`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the raw pointer is eventually converted
    /// back into a `CompactArc` to avoid memory leaks.
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
    /// The raw pointer must have been previously returned by a call to
    /// `CompactArc::into_raw`.
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // The pointer is to the data field, we need to get to the CompactArcInner
        let offset = mem::offset_of!(CompactArcInner<T>, data);
        let inner_ptr = (ptr as *const u8).sub(offset) as *mut CompactArcInner<T>;

        CompactArc {
            ptr: NonNull::new_unchecked(inner_ptr),
            _marker: PhantomData,
        }
    }
}

impl<T> Deref for CompactArc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // Safety: The pointer is valid as long as the CompactArc exists
        unsafe { &self.ptr.as_ref().data }
    }
}

impl<T> Clone for CompactArc<T> {
    #[inline]
    fn clone(&self) -> Self {
        // Increment the reference count
        // Using Relaxed is fine here because we're not synchronizing with anything
        // The synchronization happens in Drop when we decrement
        // SAFETY: The pointer is valid because CompactArc maintains the invariant that
        // ptr always points to a valid, aligned CompactArcInner allocation.
        let old_count = unsafe {
            self.ptr
                .as_ref()
                .count
                .fetch_add(1, AtomicOrdering::Relaxed)
        };

        // Check for overflow (very unlikely but good to be safe)
        if old_count > isize::MAX as usize {
            std::process::abort();
        }

        CompactArc {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for CompactArc<T> {
    #[inline]
    fn drop(&mut self) {
        // Decrement the reference count
        // We need Release ordering to ensure all writes before drop are visible
        // to the thread that will deallocate
        // SAFETY: The pointer is valid because CompactArc maintains the invariant that
        // ptr always points to a valid, aligned CompactArcInner allocation.
        let old_count = unsafe {
            self.ptr
                .as_ref()
                .count
                .fetch_sub(1, AtomicOrdering::Release)
        };

        if old_count == 1 {
            // We were the last reference, deallocate
            // Need Acquire fence to synchronize with Release in fetch_sub
            std::sync::atomic::fence(AtomicOrdering::Acquire);

            // Safety: We're the last reference
            unsafe {
                // Drop the data
                ptr::drop_in_place(&mut self.ptr.as_mut().data);
                // Deallocate
                let layout = Layout::new::<CompactArcInner<T>>();
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
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
        // Fast path: same pointer
        if Self::ptr_eq(self, other) {
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
        assert_eq!(*arc2, 100); // Original unchanged
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
    fn test_size() {
        // CompactArc should be pointer-sized
        assert_eq!(std::mem::size_of::<CompactArc<i32>>(), 8);

        // Inner should be count (8) + data
        assert_eq!(std::mem::size_of::<CompactArcInner<i32>>(), 16); // 8 + 4 + padding
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

        // After all threads done, only original arc remains
        drop(arc);
    }
}
