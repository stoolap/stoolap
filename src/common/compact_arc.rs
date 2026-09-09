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
//! - Heap header: 24 bytes, including an optional allocation account
//! - Net savings: 8 bytes per clone (thin pointer)
//!
//! **Avoid for sized types** (`i64`, `String`, structs):
//! - Stack pointer: 8 bytes (same as std::Arc)
//! - Heap header: 24 bytes, including an optional allocation account
//! - No advantage over std::Arc
//!
//! ## Memory Layout
//!
//! All types use a compact 24-byte header on 64-bit targets. Type-specific drop logic is resolved
//! at compile time via monomorphization (no stored function pointer needed):
//!
//! ```text
//! Stack:  [ptr: 8 bytes] ──────────────────┐
//!                                          ▼
//! Heap:   [refcount: 8][len: 8][account: 8][data...]
//! ```
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

use super::memory::AccountSlot;
use super::{CompactVec, MemoryAccount, MemoryAdoption, MemoryCharge};

// ============================================================================
// Unified Header - three pointer words (no stored function pointer!)
// ============================================================================

/// Unified header for all CompactArc allocations.
/// Drop logic is resolved at compile time via the CompactArcDrop trait,
/// so no function pointer needs to be stored.
#[repr(C)]
struct Header {
    count: AtomicUsize,
    /// Length of data. For sized types: 0 (or metadata). For str: byte length. For [T]: element count.
    len: usize,
    account: AccountSlot,
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

fn slice_layout<T>(len: usize) -> Layout {
    match Layout::array::<T>(len).and_then(|data| Layout::new::<Header>().extend(data)) {
        Ok((layout, offset)) => {
            debug_assert_eq!(offset, data_offset_for::<T>());
            layout
        }
        Err(_) => panic!("layout overflow"),
    }
}

// ============================================================================
// CompactArcDrop - Compile-time drop dispatch (replaces stored fn pointer)
// ============================================================================

/// Trait for type-specific drop and deallocation logic.
///
/// This trait enables compile-time dispatch for dropping CompactArc contents,
/// replacing the previous runtime function pointer approach. Since Rust
/// monomorphizes generics, the compiler resolves the correct implementation
/// at compile time - making this both smaller (no stored pointer) and faster
/// (direct call instead of indirect).
///
/// # Safety
///
/// Implementations must correctly drop T's data and deallocate the
/// header+data allocation using the correct Layout. This trait is
/// auto-implemented for all types used with CompactArc.
pub unsafe trait CompactArcDrop {
    /// Layout of the live header and data allocation.
    ///
    /// # Safety
    /// `ptr` must be a live allocation for this exact CompactArc type.
    unsafe fn allocation_layout(ptr: *mut u8) -> Layout;

    /// Drop data and deallocate its exclusively owned header allocation.
    ///
    /// # Safety
    /// `ptr` must be a valid final-owned CompactArc allocation of this type.
    unsafe fn drop_and_dealloc(ptr: *mut u8);
}

// Retain the charge while destructors run, releasing it only after the backing
// allocation is freed. This guard also runs during a destructor unwind.
struct AllocationGuard {
    ptr: *mut u8,
    layout: Layout,
    _charge: Option<MemoryCharge>,
}

impl AllocationGuard {
    /// # Safety
    /// `ptr` must be a live, exclusively owned header allocated with `layout`.
    unsafe fn new(ptr: *mut u8, layout: Layout) -> Self {
        let charge = (*(ptr as *mut Header)).account.take(layout.size());
        Self {
            ptr,
            layout,
            _charge: charge,
        }
    }
}

impl Drop for AllocationGuard {
    fn drop(&mut self) {
        // SAFETY: this guard owns the original layout; _charge drops after deallocation.
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

// Array construction owns only its initialized prefix until publication. Field
// drop releases the backing even if an initialized element's destructor unwinds.
struct SliceInitGuard<T> {
    data_ptr: *mut T,
    written: usize,
    ptr: *mut u8,
    layout: Layout,
}

impl<T> Drop for SliceInitGuard<T> {
    fn drop(&mut self) {
        // SAFETY: failed construction exclusively owns the live header and initialized prefix.
        let _allocation = unsafe { AllocationGuard::new(self.ptr, self.layout) };
        // SAFETY: written is advanced only after each successful ptr::write.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.data_ptr, self.written));
        }
    }
}

// SAFETY: the layout matches the sized constructor and data is dropped once.
unsafe impl<T> CompactArcDrop for T {
    #[inline]
    unsafe fn allocation_layout(_ptr: *mut u8) -> Layout {
        Layout::from_size_align_unchecked(
            data_offset_for::<T>() + mem::size_of::<T>(),
            mem::align_of::<T>().max(mem::align_of::<Header>()),
        )
    }

    #[inline]
    unsafe fn drop_and_dealloc(ptr: *mut u8) {
        let _allocation = AllocationGuard::new(ptr, Self::allocation_layout(ptr));
        ptr::drop_in_place(ptr.add(data_offset_for::<T>()) as *mut T);
    }
}

// SAFETY: immutable length and byte alignment match the str constructor.
unsafe impl CompactArcDrop for str {
    #[inline]
    unsafe fn allocation_layout(ptr: *mut u8) -> Layout {
        Layout::from_size_align_unchecked(
            data_offset_for::<u8>() + (*(ptr as *mut Header)).len,
            mem::align_of::<Header>(),
        )
    }

    #[inline]
    unsafe fn drop_and_dealloc(ptr: *mut u8) {
        let _allocation = AllocationGuard::new(ptr, Self::allocation_layout(ptr));
    }
}

// SAFETY: constructors checked the length/layout; initialized elements are
// dropped through a slice, with allocation cleanup protected against unwind.
unsafe impl<T> CompactArcDrop for [T] {
    #[inline]
    unsafe fn allocation_layout(ptr: *mut u8) -> Layout {
        let len = (*(ptr as *mut Header)).len;
        Layout::from_size_align_unchecked(
            data_offset_for::<T>() + mem::size_of::<T>() * len,
            mem::align_of::<T>().max(mem::align_of::<Header>()),
        )
    }

    #[inline]
    unsafe fn drop_and_dealloc(ptr: *mut u8) {
        let len = (*(ptr as *mut Header)).len;
        let _allocation = AllocationGuard::new(ptr, Self::allocation_layout(ptr));
        ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
            ptr.add(data_offset_for::<T>()) as *mut T,
            len,
        ));
    }
}

// ============================================================================
// CompactArc - Unified type with thin pointers!
// ============================================================================

/// A thread-safe reference-counted pointer without weak reference support.
///
/// `CompactArc<T>` provides shared ownership of a value of type `T`, allocated
/// on the heap. It saves memory compared to `std::sync::Arc`:
/// - One optional allocation account without growing the pointer handle
/// - Thin pointers for DSTs (8 bytes instead of 16 for `str` and `[T]`)
///
/// # Pointer Sizes
///
/// | Type | Size |
/// |------|------|
/// | `CompactArc<i64>` | 8 bytes |
/// | `CompactArc<str>` | 8 bytes (thin!) |
/// | `CompactArc<[T]>` | 8 bytes (thin!) |
pub struct CompactArc<T: ?Sized + CompactArcDrop> {
    /// Thin pointer to Header (always 8 bytes, even for DSTs!)
    ptr: NonNull<Header>,
    _marker: PhantomData<T>,
}

// SAFETY: CompactArc can be sent between threads if T can be sent and shared.
// The refcount is atomic (AtomicUsize) ensuring thread-safe increment/decrement.
// T: Send + Sync ensures the data itself can be safely shared across threads.
unsafe impl<T: ?Sized + CompactArcDrop + Send + Sync> Send for CompactArc<T> {}
// SAFETY: Same reasoning as Send - atomic refcount and T: Send + Sync.
unsafe impl<T: ?Sized + CompactArcDrop + Send + Sync> Sync for CompactArc<T> {}

// ============================================================================
// THE SINGLE DROP IMPL - Compile-time dispatch via CompactArcDrop!
// ============================================================================

impl<T: ?Sized + CompactArcDrop> Drop for CompactArc<T> {
    #[inline]
    fn drop(&mut self) {
        let header = self.ptr.as_ptr();
        // SAFETY: self.ptr is always valid (NonNull) and points to a properly initialized
        // Header. The atomic decrement is safe for concurrent access. Release ordering
        // ensures our writes are visible to whoever sees the decremented count.
        let old_count = unsafe { (*header).count.fetch_sub(1, AtomicOrdering::Release) };

        if old_count == 1 {
            std::sync::atomic::fence(AtomicOrdering::Acquire);
            // SAFETY: old_count == 1 means we had the last reference. The Acquire fence
            // synchronizes with Release in other drops, ensuring we see all their writes.
            // T::drop_and_dealloc is resolved at compile time via monomorphization.
            unsafe {
                T::drop_and_dealloc(header as *mut u8);
            }
        }
    }
}

// ============================================================================
// THE SINGLE CLONE IMPL - Works for ALL types!
// ============================================================================

impl<T: ?Sized + CompactArcDrop> Clone for CompactArc<T> {
    #[inline]
    fn clone(&self) -> Self {
        let header = self.ptr.as_ptr();
        // SAFETY: self.ptr is always valid (NonNull) and points to a properly initialized
        // Header. The atomic increment is safe for concurrent access. Relaxed ordering
        // is sufficient since we don't need to synchronize any data with this operation.
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

impl<T: ?Sized + CompactArcDrop> CompactArc<T> {
    // Only constructors call this, before their fresh allocation is exposed.
    fn account_new_allocation(&mut self, account: &MemoryAccount) {
        let bytes = self.allocation_size();
        // SAFETY: every caller exclusively owns a fresh, untracked header.
        unsafe {
            (*self.ptr.as_ptr()).account.install_new(account, bytes);
        }
    }

    /// Requested bytes of this allocation only, excluding heaps owned by T.
    pub fn allocation_size(&self) -> usize {
        // SAFETY: self keeps this correctly typed allocation alive.
        unsafe { T::allocation_layout(self.ptr.as_ptr() as *mut u8).size() }
    }

    pub fn memory_account(&self) -> Option<MemoryAccount> {
        // SAFETY: self retains the allocation and its account owner.
        unsafe { (*self.ptr.as_ptr()).account.get() }
    }

    pub fn belongs_to(&self, account: &MemoryAccount) -> bool {
        // SAFETY: self retains the header and its installed account during inspection.
        unsafe { (*self.ptr.as_ptr()).account.belongs_to(account) }
    }

    /// Adopt this allocation only; this does not certify ownership of T's heaps.
    pub fn try_adopt_shallow(&self, account: &MemoryAccount) -> MemoryAdoption {
        // SAFETY: self retains the allocation throughout account publication.
        unsafe {
            (*self.ptr.as_ptr())
                .account
                .adopt(account, self.allocation_size())
        }
    }

    pub(crate) fn is_fully_accounted(&self) -> bool {
        // SAFETY: self retains the header during the atomic account-tag read.
        unsafe { (*self.ptr.as_ptr()).account.is_fully_accounted() }
    }

    /// Call only after the container's ingress accounted for all nested owners.
    pub(crate) fn mark_fully_accounted(&self) {
        // SAFETY: self retains the header; ingress has accounted for every nested owner.
        unsafe {
            (*self.ptr.as_ptr()).account.mark_fully_accounted();
        }
    }

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
        // SAFETY: this.ptr is always valid (NonNull) and points to a properly initialized
        // Header. Reading the atomic count with Relaxed ordering is always safe.
        unsafe { (*this.ptr.as_ptr()).count.load(AtomicOrdering::Relaxed) }
    }

    /// Returns `true` if this is the only reference to the allocation.
    ///
    /// Uses `Acquire` ordering to synchronize with `Release` in `drop`,
    /// ensuring visibility of all modifications made by other threads
    /// before they dropped their references.
    #[inline]
    fn is_unique(this: &Self) -> bool {
        // SAFETY: this.ptr is always valid (NonNull) and points to a properly initialized
        // Header. Acquire ordering synchronizes with Release in drop.
        unsafe { (*this.ptr.as_ptr()).count.load(AtomicOrdering::Acquire) == 1 }
    }

    /// Returns the metadata stored in the header (used for count).
    #[inline]
    pub fn meta(this: &Self) -> usize {
        // SAFETY: this.ptr is always valid (NonNull) and points to a properly initialized
        // Header. The len field is immutable after construction (for DSTs) or can be
        // safely read (for sized types using it as metadata).
        unsafe { (*this.ptr.as_ptr()).len }
    }
}

// ============================================================================
// Sized type implementations
// ============================================================================

impl<T: CompactArcDrop> CompactArc<T> {
    /// Creates a new `CompactArc<T>` containing the given value.
    #[inline]
    #[must_use]
    pub fn new(data: T) -> Self {
        Self::new_with_meta(data, 0)
    }

    /// Charge this allocation before returning; nested T ownership is separate.
    pub fn new_in(data: T, account: &MemoryAccount) -> Self {
        let mut value = Self::new(data);
        value.account_new_allocation(account);
        value
    }

    /// Creates a new `CompactArc<T>` containing the given value and metadata.
    /// The metadata is stored in the header's `len` field, which is unused for Sized types.
    #[inline]
    #[must_use]
    pub fn new_with_meta(data: T, meta: usize) -> Self {
        let data_offset = data_offset_for::<T>();
        let align = mem::align_of::<T>().max(mem::align_of::<Header>());
        let total_size = data_offset + mem::size_of::<T>();
        let layout = Layout::from_size_align(total_size, align).expect("layout overflow");

        // SAFETY: We allocate memory with the correct layout for Header + T.
        // We initialize all fields before returning. The allocation is guaranteed
        // to be non-null (we call handle_alloc_error on failure). data_offset_for<T>()
        // ensures proper alignment for T after the header.
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
                    account: AccountSlot::new(),
                    len: meta,
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

        // SAFETY: this.ptr is valid. compare_exchange atomically checks if count == 1
        // and sets it to 0. Acquire ordering on success synchronizes with Release in
        // other drops, ensuring we see all their writes.
        if unsafe {
            (*header)
                .count
                .compare_exchange(1, 0, AtomicOrdering::Acquire, AtomicOrdering::Relaxed)
                .is_ok()
        } {
            let _ = ManuallyDrop::new(this);

            // SAFETY: compare_exchange succeeded, so we had the only reference and now
            // own the data exclusively. We read the data out (moving it), then deallocate
            // the memory without calling the dropper (since we took ownership of the data).
            unsafe {
                // Read data
                let data_offset = data_offset_for::<T>();
                let data_ptr = (header as *const u8).add(data_offset) as *const T;
                let data = ptr::read(data_ptr);

                // Deallocate (without dropping since we took the data)
                let align = mem::align_of::<T>().max(mem::align_of::<Header>());
                let layout =
                    Layout::from_size_align_unchecked(data_offset + mem::size_of::<T>(), align);
                drop(AllocationGuard::new(header as *mut u8, layout));

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
            // SAFETY: is_unique() returned true with Acquire ordering, meaning we have
            // exclusive access. this.ptr is valid and data_offset_for<T>() gives the
            // correct offset to the properly aligned T.
            unsafe {
                (*this.ptr.as_ptr()).account.clear_fully_accounted();
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
            let meta = Self::meta(this);
            let account = this.memory_account();
            let mut replacement = CompactArc::new_with_meta((**this).clone(), meta);
            if let Some(account) = &account {
                replacement.account_new_allocation(account);
            }
            // Cloned nested heaps remain uncertified until the container's ingress checks them.
            *this = replacement;
        }
        // SAFETY: After the above, we're guaranteed to be the only reference
        Self::get_mut(this).unwrap()
    }

    /// Returns a raw pointer to the contained data.
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        // Derive from the header pointer via raw pointer arithmetic to avoid
        // creating a shared reference (&T) that would restrict the borrow stack.
        // Going through Deref (&*this) creates a SharedReadOnly tag that
        // invalidates later writes to the header (e.g., refcount decrement).
        let header = this.ptr.as_ptr();
        unsafe { (header as *const u8).add(data_offset_for::<T>()) as *const T }
    }

    /// Converts a `CompactArc<T>` into a raw pointer.
    #[inline]
    pub fn into_raw(this: Self) -> *const T {
        // Use raw pointer arithmetic instead of &*this to avoid Stacked Borrows
        // violation: a SharedReadOnly retag from &*this would conflict with
        // the SharedReadWrite needed by Drop to decrement the refcount.
        let header = this.ptr.as_ptr();
        let ptr = unsafe { (header as *const u8).add(data_offset_for::<T>()) as *const T };
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

impl<T: CompactArcDrop> Deref for CompactArc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: self.ptr is always valid (NonNull) and points to a properly initialized
        // allocation. data_offset_for<T>() gives the correct offset to the properly aligned
        // T data. The data was initialized in new() or new_with_meta().
        unsafe {
            let data_ptr = (self.ptr.as_ptr() as *const u8).add(data_offset_for::<T>()) as *const T;
            &*data_ptr
        }
    }
}

impl<T: CompactArcDrop + Default> Default for CompactArc<T> {
    #[inline]
    fn default() -> Self {
        CompactArc::new(T::default())
    }
}

impl<T: CompactArcDrop + fmt::Debug> fmt::Debug for CompactArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: CompactArcDrop + fmt::Display> fmt::Display for CompactArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: CompactArcDrop + PartialEq> PartialEq for CompactArc<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if CompactArc::ptr_eq(self, other) {
            return true;
        }
        **self == **other
    }
}

impl<T: CompactArcDrop + Eq> Eq for CompactArc<T> {}

impl<T: CompactArcDrop + PartialOrd> PartialOrd for CompactArc<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: CompactArcDrop + Ord> Ord for CompactArc<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: CompactArcDrop + Hash> Hash for CompactArc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: CompactArcDrop> Borrow<T> for CompactArc<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: CompactArcDrop> AsRef<T> for CompactArc<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: CompactArcDrop> From<T> for CompactArc<T> {
    #[inline]
    fn from(value: T) -> Self {
        CompactArc::new(value)
    }
}

// ============================================================================
// DST Support: str (Thin Pointer!)
// ============================================================================

impl CompactArc<str> {
    /// Creates a new `CompactArc<str>` from a string slice.
    ///
    /// The pointer is only 8 bytes (thin), with length stored in heap header.
    #[must_use]
    pub fn from_str_slice(s: &str) -> Self {
        let len = s.len();
        let data_offset = data_offset_for::<u8>(); // str has align 1
        let layout = slice_layout::<u8>(len);

        // SAFETY: We allocate memory with the correct layout for Header + str bytes.
        // We initialize all fields before returning. The source string s is valid UTF-8,
        // and we copy its bytes verbatim, preserving UTF-8 validity.
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
                    account: AccountSlot::new(),
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
        // SAFETY: self.ptr is valid and points to an initialized Header.
        // The len field contains the string length set during construction.
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
        // SAFETY: self.ptr is valid and points to an initialized allocation.
        // The len field contains the correct string length. The bytes were copied
        // from a valid UTF-8 string in from_str_slice(), so they are valid UTF-8.
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

impl<T> CompactArc<[T]> {
    /// Build one charged array without an intermediate Vec. ExactSizeIterator
    /// supplies a capacity hint, not an unsafe promise: both directions of a
    /// dishonest length panic with initialized-prefix cleanup.
    pub(crate) fn from_exact_iter_in<I>(mut values: I, account: &MemoryAccount) -> Self
    where
        I: ExactSizeIterator<Item = T>,
    {
        let len = values.len();
        let data_offset = data_offset_for::<T>();
        let layout = slice_layout::<T>(len);
        // SAFETY: the checked layout fits len values; publication follows complete initialization.
        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            let header = ptr as *mut Header;
            ptr::write(
                header,
                Header {
                    count: AtomicUsize::new(1),
                    len,
                    account: AccountSlot::new(),
                },
            );
            let data_ptr = ptr.add(data_offset) as *mut T;
            let mut guard = SliceInitGuard {
                data_ptr,
                written: 0,
                ptr,
                layout,
            };
            // Charge the allocated array before iterator callbacks can observe the account.
            (*header).account.install_new(account, layout.size());
            for index in 0..len {
                let Some(value) = values.next() else {
                    panic!("exact iterator returned too few values");
                };
                ptr::write(data_ptr.add(index), value);
                guard.written += 1;
            }
            assert!(
                values.next().is_none(),
                "exact iterator returned too many values"
            );
            drop(values);
            mem::forget(guard);
            CompactArc {
                ptr: NonNull::new_unchecked(header),
                _marker: PhantomData,
            }
        }
    }

    pub fn from_vec_in(values: Vec<T>, account: &MemoryAccount) -> Self {
        let mut values = Self::from_vec(values);
        values.account_new_allocation(account);
        values
    }

    pub fn from_compact_vec_in(values: CompactVec<T>, account: &MemoryAccount) -> Self {
        let mut values = Self::from_compact_vec(values);
        values.account_new_allocation(account);
        values
    }

    /// Creates a new `CompactArc<[T]>` by moving elements from a Vec.
    ///
    /// This is more efficient than `from_slice` as it moves elements instead of cloning.
    #[must_use]
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let len = vec.len();
        let data_offset = data_offset_for::<T>();
        let layout = slice_layout::<T>(len);

        // SAFETY: We allocate memory with the correct layout for Header + [T].
        // We copy (move) the elements from vec into the allocation, then set vec's len to 0
        // to prevent double-free. The vec's buffer is still freed when vec drops, but
        // its elements have been moved to our allocation.
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
                    account: AccountSlot::new(),
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

    /// Creates a new `CompactArc<[T]>` by moving elements from a CompactVec.
    ///
    /// This is more efficient than `from_slice` as it moves elements instead of cloning.
    /// Avoids the intermediate Vec conversion compared to `from_vec`.
    #[must_use]
    pub fn from_compact_vec(mut vec: CompactVec<T>) -> Self {
        let len = vec.len();
        let data_offset = data_offset_for::<T>();
        let layout = slice_layout::<T>(len);

        // SAFETY: We allocate memory with the correct layout for Header + [T].
        // We copy (move) the elements from vec into the allocation, then set vec's len to 0
        // to prevent double-free. The vec's buffer is still freed when vec drops, but
        // its elements have been moved to our allocation.
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
                    account: AccountSlot::new(),
                    len,
                },
            );

            // Move elements from CompactVec (copy bytes, then prevent CompactVec from dropping them)
            let data_ptr = ptr.add(data_offset) as *mut T;
            ptr::copy_nonoverlapping(vec.as_ptr(), data_ptr, len);

            // Prevent CompactVec from dropping the moved elements (buffer will still be freed)
            vec.set_len(0);

            CompactArc {
                ptr: NonNull::new_unchecked(header),
                _marker: PhantomData,
            }
        }
    }
}

impl<T: Clone> CompactArc<[T]> {
    pub fn from_slice_in(values: &[T], account: &MemoryAccount) -> Self {
        Self::from_exact_iter_in(values.iter().cloned(), account)
    }

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
        let layout = slice_layout::<T>(len);

        // SAFETY: We allocate memory with the correct layout for Header + [T].
        // We use a CloneGuard for panic safety - if any clone() panics, the guard
        // drops all successfully cloned elements and frees the allocation.
        // On success, we forget the guard and return the initialized CompactArc.
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
                    account: AccountSlot::new(),
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
                    // SAFETY: data_ptr points to an array where the first `written` elements
                    // are initialized. We drop those elements, then deallocate the memory
                    // using the stored layout. This is only called on panic during clone.
                    unsafe {
                        let _allocation = AllocationGuard::new(self.alloc_ptr, self.layout);
                        // Drop all successfully written elements
                        let slice = ptr::slice_from_raw_parts_mut(self.data_ptr, self.written);
                        ptr::drop_in_place(slice);
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
        // SAFETY: self.ptr is valid and points to an initialized Header.
        // The len field contains the slice length set during construction.
        unsafe { (*self.ptr.as_ptr()).len }
    }

    /// Returns true if the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a raw mutable pointer to the first element of the data region.
    ///
    /// Derives the pointer from the header via raw pointer arithmetic, bypassing
    /// `Deref` (which would create a SharedReadOnly borrow tag under Stacked
    /// Borrows and prevent subsequent mutable access to the same allocation).
    #[inline]
    pub(crate) fn data_ptr_mut(&self) -> *mut T {
        unsafe { (self.ptr.as_ptr() as *mut u8).add(data_offset_for::<T>()) as *mut T }
    }
}

impl<T> Deref for CompactArc<[T]> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: self.ptr is valid and points to an initialized allocation.
        // The len field contains the correct element count. data_offset_for<T>()
        // gives the correct offset to the properly aligned [T] data. All len
        // elements were initialized in from_slice() or from_vec().
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
    fn from_slice_in_accounts_clone_callbacks_and_peak_overlap() {
        use std::cell::Cell;

        struct Probe<'a> {
            account: &'a MemoryAccount,
            clone_retained: &'a Cell<usize>,
        }
        impl Clone for Probe<'_> {
            fn clone(&self) -> Self {
                self.clone_retained
                    .set(self.account.snapshot().retained_bytes);
                // A nested allocation can disappear before construction ends;
                // its peak must still overlap the destination array's charge.
                let temporary = MemoryCharge::new(self.account, 4096);
                drop(temporary);
                Self {
                    account: self.account,
                    clone_retained: self.clone_retained,
                }
            }
        }

        let account = MemoryAccount::new();
        let clone_retained = Cell::new(0);
        let source = [Probe {
            account: &account,
            clone_retained: &clone_retained,
        }];
        let values = CompactArc::from_slice_in(&source, &account);
        let destination_bytes = values.allocation_size();
        let snapshot = account.snapshot();
        assert_eq!(clone_retained.get(), destination_bytes);
        assert_eq!(snapshot.retained_bytes, destination_bytes);
        assert_eq!(
            snapshot.peak_accounted_bytes,
            snapshot.conservative_bytes + destination_bytes + 4096
        );
        drop(values);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn from_slice_in_keeps_charge_through_clone_panic_cleanup() {
        use std::cell::Cell;

        struct Probe<'a> {
            account: &'a MemoryAccount,
            clones: &'a Cell<usize>,
            drops: &'a Cell<usize>,
            drop_retained: &'a Cell<usize>,
            is_clone: bool,
        }
        impl Clone for Probe<'_> {
            fn clone(&self) -> Self {
                self.clones.set(self.clones.get() + 1);
                assert_ne!(self.clones.get(), 2, "second clone fails");
                Self {
                    account: self.account,
                    clones: self.clones,
                    drops: self.drops,
                    drop_retained: self.drop_retained,
                    is_clone: true,
                }
            }
        }
        impl Drop for Probe<'_> {
            fn drop(&mut self) {
                if self.is_clone {
                    self.drops.set(self.drops.get() + 1);
                    // Record instead of asserting while unwinding: the test
                    // must report missing accounting without a double panic.
                    self.drop_retained
                        .set(self.account.snapshot().retained_bytes);
                }
            }
        }

        let account = MemoryAccount::new();
        let clones = Cell::new(0);
        let drops = Cell::new(0);
        let drop_retained = Cell::new(0);
        let source = std::array::from_fn::<_, 3, _>(|_| Probe {
            account: &account,
            clones: &clones,
            drops: &drops,
            drop_retained: &drop_retained,
            is_clone: false,
        });
        let destination_bytes = data_offset_for::<Probe<'_>>() + mem::size_of_val(&source);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CompactArc::from_slice_in(&source, &account)
        }));
        assert!(result.is_err());
        assert_eq!(clones.get(), 2);
        assert_eq!(drops.get(), 1);
        assert_eq!(drop_retained.get(), destination_bytes);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn exact_iterator_has_one_array_and_checks_dishonest_lengths() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;
        struct Item(Arc<AtomicUsize>);
        impl Drop for Item {
            fn drop(&mut self) {
                self.0.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        struct Liar {
            values: std::vec::IntoIter<Item>,
            advertised: usize,
            panic_after: Option<usize>,
            seen: usize,
        }
        impl Iterator for Liar {
            type Item = Item;
            fn next(&mut self) -> Option<Item> {
                if self.panic_after == Some(self.seen) {
                    panic!("iterator next panic");
                }
                self.seen += 1;
                self.values.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.advertised, Some(self.advertised))
            }
        }
        impl ExactSizeIterator for Liar {}
        let account = MemoryAccount::new();
        for (actual, advertised, panic_after) in [(2, 4, None), (4, 2, None), (3, 3, Some(1))] {
            let dropped = Arc::new(AtomicUsize::new(0));
            let values = (0..actual)
                .map(|_| Item(dropped.clone()))
                .collect::<Vec<_>>();
            let iterator = Liar {
                values: values.into_iter(),
                advertised,
                panic_after,
                seen: 0,
            };
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                CompactArc::from_exact_iter_in(iterator, &account);
            }))
            .is_err());
            assert_eq!(dropped.load(AtomicOrdering::Relaxed), actual);
            assert_eq!(account.snapshot().retained_bytes, 0);
        }
        let values = CompactArc::from_exact_iter_in([1usize, 2, 3].into_iter(), &account);
        assert_eq!(&*values, &[1, 2, 3]);
        assert!(!values.is_fully_accounted());
        assert_eq!(account.snapshot().retained_bytes, values.allocation_size());
        drop(values);
        let empty = CompactArc::<[usize]>::from_exact_iter_in([].into_iter(), &account);
        assert!(empty.is_empty());
        drop(empty);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn exact_iterator_accounts_callbacks_and_destructor_unwind() {
        use std::sync::Arc;
        let account = MemoryAccount::new();
        let values = [1usize, 2, 3].into_iter().inspect(|_| {
            assert!(account.snapshot().retained_bytes > 0);
        });
        let array = CompactArc::from_exact_iter_in(values, &account);
        assert_eq!(account.snapshot().retained_bytes, array.allocation_size());
        drop(array);

        struct Item(MemoryAccount, Arc<AtomicUsize>);
        impl Drop for Item {
            fn drop(&mut self) {
                assert!(self.0.snapshot().retained_bytes > 0);
                self.1.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        struct PanicDrop(std::vec::IntoIter<Item>);
        impl Iterator for PanicDrop {
            type Item = Item;
            fn next(&mut self) -> Option<Item> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }
        impl ExactSizeIterator for PanicDrop {}
        impl Drop for PanicDrop {
            fn drop(&mut self) {
                panic!("iterator drop panic");
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let values = (0..3)
            .map(|_| Item(account.clone(), dropped.clone()))
            .collect::<Vec<_>>();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CompactArc::from_exact_iter_in(PanicDrop(values.into_iter()), &account);
        }))
        .is_err());
        assert_eq!(dropped.load(AtomicOrdering::Relaxed), 3);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn charged_str_and_high_alignment_raw_owners_release_exactly() {
        #[repr(align(128))]
        #[derive(Clone, Debug, PartialEq)]
        struct Aligned([u8; 129]);
        let account = MemoryAccount::new();
        let value = CompactArc::new_in(Aligned([7; 129]), &account);
        let bytes = value.allocation_size();
        let raw = CompactArc::into_raw(value);
        assert_eq!(raw.addr() % 128, 0);
        let restored = unsafe { CompactArc::from_raw(raw) };
        assert_eq!(account.snapshot().retained_bytes, bytes);
        assert_eq!(CompactArc::try_unwrap(restored).unwrap(), Aligned([7; 129]));
        assert_eq!(account.snapshot().retained_bytes, 0);
        let mut text = CompactArc::from_str_slice("charged UTF-8: İstanbul");
        text.account_new_allocation(&account);
        let bytes = text.allocation_size();
        let other = text.clone();
        drop(text);
        assert_eq!(account.snapshot().retained_bytes, bytes);
        assert_eq!(&*other, "charged UTF-8: İstanbul");
        drop(other);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn charged_cow_and_unwrap_keep_nested_heap_aliases() {
        #[derive(Clone, Debug)]
        struct Container {
            child: super::super::SmartString,
        }
        let account = MemoryAccount::new();
        let child = super::super::SmartString::new("nested child survives COW and unwrap")
            .into_hot(&account);
        let child_bytes = account.snapshot().retained_bytes;
        let mut value = CompactArc::new_in(Container { child }, &account);
        let outer_bytes = value.allocation_size();
        value.mark_fully_accounted();
        let previous = value.clone();
        assert_eq!(
            CompactArc::make_mut(&mut value).child.as_str(),
            "nested child survives COW and unwrap"
        );
        assert!(!value.is_fully_accounted());
        assert_eq!(
            account.snapshot().retained_bytes,
            child_bytes + 2 * outer_bytes
        );
        let owned = CompactArc::try_unwrap(value).unwrap();
        drop(previous);
        assert_eq!(account.snapshot().retained_bytes, child_bytes);
        assert_eq!(owned.child.as_str(), "nested child survives COW and unwrap");
        drop(owned);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn accounted_aliases_release_only_final_allocation() {
        let account = MemoryAccount::new();
        let values = CompactArc::from_vec_in(vec![1u64, 2, 3], &account);
        let bytes = values.allocation_size();
        assert_eq!(account.snapshot().retained_bytes, bytes);
        let alias = values.clone();
        drop(values);
        assert_eq!(account.snapshot().retained_bytes, bytes);
        assert_eq!(&*alias, &[1, 2, 3]);
        drop(alias);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn raw_round_trip_unwrap_and_mutable_escape_keep_correct_charge() {
        let account = MemoryAccount::new();
        let mut value = CompactArc::new_in(42usize, &account);
        let bytes = value.allocation_size();
        value.mark_fully_accounted();
        assert!(value.is_fully_accounted());
        *CompactArc::get_mut(&mut value).unwrap() = 43;
        assert!(!value.is_fully_accounted());
        value.mark_fully_accounted();
        let raw = CompactArc::into_raw(value);
        let mut value = unsafe { CompactArc::from_raw(raw) };
        assert!(value.is_fully_accounted());
        let previous = value.clone();
        *CompactArc::make_mut(&mut value) = 44;
        assert!(!value.is_fully_accounted());
        assert!(previous.is_fully_accounted());
        assert_eq!(account.snapshot().retained_bytes, 2 * bytes);
        assert_eq!(CompactArc::try_unwrap(value).unwrap(), 44);
        assert_eq!(account.snapshot().retained_bytes, bytes);
        drop(previous);
        assert_eq!(account.snapshot().retained_bytes, 0);
        let mut unique = CompactArc::new_in(1usize, &account);
        unique.mark_fully_accounted();
        CompactArc::make_mut(&mut unique);
        assert!(!unique.is_fully_accounted());
    }

    #[test]
    fn panicking_destructor_still_frees_header_and_charge() {
        struct Panicking;
        impl Drop for Panicking {
            fn drop(&mut self) {
                panic!("expected destructor panic");
            }
        }
        let account = MemoryAccount::new();
        let value = CompactArc::new_in(Panicking, &account);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(value))).is_err());
        assert_eq!(account.snapshot().retained_bytes, 0);
        let values = CompactArc::from_vec_in(vec![Panicking], &account);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(values))).is_err());
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

    #[test]
    fn nested_string_alias_outlives_parent_array() {
        let account = MemoryAccount::new();
        let text = super::super::SmartString::new("nested heap backing remains retained")
            .into_hot(&account);
        let string_bytes = account.snapshot().retained_bytes;
        let alias = text.clone();
        let values = CompactArc::from_vec_in(vec![text], &account);
        assert!(!values.is_fully_accounted());
        values.mark_fully_accounted();
        assert!(values.is_fully_accounted());
        drop(values);
        assert_eq!(account.snapshot().retained_bytes, string_bytes);
        drop(alias);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }

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

        // SAFETY: ptr was obtained from into_raw and has not been used since
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
    fn test_header_size() {
        // Three pointer words: refcount, length, optional account.
        assert_eq!(
            std::mem::size_of::<Header>(),
            3 * std::mem::size_of::<usize>()
        );
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
    fn test_slice_from_compact_vec() {
        let mut compact_vec = CompactVec::new();
        compact_vec.push(String::from("x"));
        compact_vec.push(String::from("y"));
        compact_vec.push(String::from("z"));

        let arr: CompactArc<[String]> = CompactArc::from_compact_vec(compact_vec);
        assert_eq!(arr.len(), 3);
        assert_eq!(&arr[0], "x");
        assert_eq!(&arr[1], "y");
        assert_eq!(&arr[2], "z");
    }

    #[test]
    fn test_from_compact_vec_moves_elements() {
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
            let mut compact_vec = CompactVec::new();
            compact_vec.push(MoveTracker(1));
            compact_vec.push(MoveTracker(2));
            compact_vec.push(MoveTracker(3));

            let arr: CompactArc<[MoveTracker]> = CompactArc::from_compact_vec(compact_vec);

            // Verify elements are accessible
            assert_eq!(arr[0].0, 1);
            assert_eq!(arr[1].0, 2);
            assert_eq!(arr[2].0, 3);

            // No clones should have happened (elements were moved)
            assert_eq!(
                CLONE_COUNT.load(Ordering::SeqCst),
                0,
                "from_compact_vec should move, not clone"
            );
        }

        // Only 3 drops: the elements in the CompactArc
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
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
