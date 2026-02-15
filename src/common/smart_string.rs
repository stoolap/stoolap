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

//! SmartString - A 16-byte string type with inline small string optimization
//!
//! This is a compact string representation that enables 16-byte Value enums
//! through niche optimization. The design:
//!
//! - **Inline**: strings ≤15 bytes stored inline (no heap allocation)
//! - **Heap**: strings >15 bytes stored as Arc<String> for O(1) clone
//!
//! ## Memory Layout (16 bytes, 8-byte aligned)
//!
//! ```text
//! Inline (≤15 bytes):
//!   [tag: 1 byte] [data: 15 bytes]
//!   tag = 0-15 (encodes length)
//!
//! Heap (>15 bytes) on 64-bit:
//!   [tag: 1 byte] [pad: 7 bytes] [Arc<String>: 8 bytes]
//!   tag = 16 (heap marker), pointer at data[7..15]
//!
//! Heap (>15 bytes) on 32-bit:
//!   [tag: 1 byte] [pad: 11 bytes] [Arc<String>: 4 bytes]
//!   tag = 16 (heap marker), pointer at data[11..15]
//! ```
//!
//! ## Niche Optimization
//!
//! StringTag only uses values 0-16, leaving values 17-255 as "niches".
//! The Rust compiler uses these niches to store the Value enum's discriminant,
//! enabling a 16-byte Value enum without explicit discriminant storage.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

/// Maximum inline string length (15 bytes)
pub const MAX_INLINE_LEN: usize = 15;

/// Platform-specific pointer layout for heap strings.
/// On 64-bit: 8-byte pointer stored at data[7..15]
/// On 32-bit: 4-byte pointer stored at data[11..15]
#[cfg(target_pointer_width = "64")]
mod ptr_layout {
    /// Offset into data array where pointer starts
    pub const OFFSET: usize = 7;
    /// Pointer size in bytes
    pub const SIZE: usize = 8;
    /// Type alias for pointer bytes array
    pub type Bytes = [u8; 8];
}

#[cfg(target_pointer_width = "32")]
mod ptr_layout {
    /// Offset into data array where pointer starts
    pub const OFFSET: usize = 11;
    /// Pointer size in bytes
    pub const SIZE: usize = 4;
    /// Type alias for pointer bytes array
    pub type Bytes = [u8; 4];
}

/// String tag that encodes storage mode and inline length.
///
/// Values 0-16 are valid, 17-255 are niches for Value enum discriminant.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringTag {
    // Inline lengths 0-15
    Inline0 = 0,
    Inline1 = 1,
    Inline2 = 2,
    Inline3 = 3,
    Inline4 = 4,
    Inline5 = 5,
    Inline6 = 6,
    Inline7 = 7,
    Inline8 = 8,
    Inline9 = 9,
    Inline10 = 10,
    Inline11 = 11,
    Inline12 = 12,
    Inline13 = 13,
    Inline14 = 14,
    Inline15 = 15,
    // Heap storage
    Heap = 16,
    // Values 17-255: NICHES for Value enum discriminant (239 niches)
}

impl StringTag {
    /// Returns true if the string is stored inline
    #[inline]
    pub const fn is_inline(self) -> bool {
        (self as u8) <= 15
    }

    /// Returns true if the string is stored on the heap
    #[inline]
    pub const fn is_heap(self) -> bool {
        (self as u8) == 16
    }

    /// Get the inline length if this is an inline tag
    #[inline]
    pub const fn inline_len(self) -> Option<usize> {
        let v = self as u8;
        if v <= 15 {
            Some(v as usize)
        } else {
            None
        }
    }

    /// Create an inline tag for the given length (0-15)
    #[inline]
    const fn inline(len: usize) -> Self {
        debug_assert!(len <= 15);
        // SAFETY: len is 0-15, which maps to valid Inline variants
        unsafe { std::mem::transmute(len as u8) }
    }
}

/// A 16-byte string with inline SSO (Small String Optimization).
///
/// This compact representation enables 16-byte Value enums through niche optimization.
/// The StringTag provides 239 niche values (17-255) that Rust uses to store Value's
/// discriminant without additional memory.
#[repr(C, align(8))]
pub struct SmartString {
    /// Tag byte: encodes storage mode (inline length or heap)
    tag: StringTag,
    /// Data bytes: inline string data OR (padding + Arc pointer)
    /// - 64-bit: 7 bytes padding + 8-byte pointer at data[7..15]
    /// - 32-bit: 11 bytes padding + 4-byte pointer at data[11..15]
    data: [u8; 15],
}

impl SmartString {
    /// Create a new SmartString from a string slice.
    #[inline]
    pub fn new(s: &str) -> Self {
        let len = s.len();
        if len <= MAX_INLINE_LEN {
            let mut data = [0u8; 15];
            data[..len].copy_from_slice(s.as_bytes());
            SmartString {
                tag: StringTag::inline(len),
                data,
            }
        } else {
            Self::new_heap(s.to_string())
        }
    }

    /// Create a SmartString from an owned String (heap case)
    #[inline]
    fn new_heap(s: String) -> Self {
        let arc = Arc::new(s);
        let ptr = Arc::into_raw(arc) as usize;
        let mut data = [0u8; 15];
        let end = ptr_layout::OFFSET + ptr_layout::SIZE;
        data[ptr_layout::OFFSET..end].copy_from_slice(&ptr.to_ne_bytes());
        SmartString {
            tag: StringTag::Heap,
            data,
        }
    }

    /// Create from a string slice - for API compatibility
    #[inline]
    pub fn const_new(s: &str) -> Self {
        Self::new(s)
    }

    /// Alias for new() - for API compatibility
    #[inline]
    pub fn new_text(s: &str) -> Self {
        Self::new(s)
    }

    #[inline]
    pub fn with_capacity(_capacity: usize) -> Self {
        Self::default()
    }

    /// Create from owned String
    #[inline]
    pub fn from_string(s: String) -> Self {
        let len = s.len();
        if len <= MAX_INLINE_LEN {
            let mut data = [0u8; 15];
            data[..len].copy_from_slice(s.as_bytes());
            SmartString {
                tag: StringTag::inline(len),
                data,
            }
        } else {
            Self::new_heap(s)
        }
    }

    /// Create from owned String - uses Arc for heap (for values that will be cloned)
    #[inline]
    pub fn from_string_shared(s: String) -> Self {
        // This is the same as from_string since we always use Arc for heap
        Self::from_string(s)
    }

    /// Returns a string slice of the contents.
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        if let Some(len) = self.tag.inline_len() {
            // Inline case: data contains the string bytes
            // SAFETY: SmartString only stores valid UTF-8
            unsafe { std::str::from_utf8_unchecked(&self.data[..len]) }
        } else {
            // Heap case: Arc<String> pointer at platform-specific offset
            let end = ptr_layout::OFFSET + ptr_layout::SIZE;
            let ptr_bytes: ptr_layout::Bytes =
                self.data[ptr_layout::OFFSET..end].try_into().unwrap();
            let ptr = usize::from_ne_bytes(ptr_bytes) as *const String;
            // SAFETY: We only store valid Arc<String> pointers
            unsafe { (*ptr).as_str() }
        }
    }

    /// Returns the length of the string in bytes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        if let Some(len) = self.tag.inline_len() {
            len
        } else {
            self.as_str().len()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_inline(&self) -> bool {
        self.tag.is_inline()
    }

    #[inline]
    pub fn is_heap(&self) -> bool {
        self.tag.is_heap()
    }

    /// Get the raw Arc<String> pointer for heap strings
    #[inline]
    fn get_arc_ptr(&self) -> *const String {
        debug_assert!(self.tag.is_heap());
        let end = ptr_layout::OFFSET + ptr_layout::SIZE;
        let ptr_bytes: ptr_layout::Bytes = self.data[ptr_layout::OFFSET..end].try_into().unwrap();
        usize::from_ne_bytes(ptr_bytes) as *const String
    }

    /// Convert to a version that will use Arc for future clones (no-op now, kept for API compat)
    #[inline]
    pub fn into_shared(self) -> Self {
        self
    }

    /// Convert self to shared in place (no-op now, kept for API compat)
    #[inline]
    pub fn make_shared(&mut self) {
        // No-op: we always use Arc for heap strings now
    }

    /// Appends a character to the string.
    ///
    /// # Performance Warning
    ///
    /// This operation may require reallocation. For string building,
    /// use `String` first then convert with `SmartString::from_string()`.
    #[inline]
    pub fn push(&mut self, ch: char) {
        let ch_len = ch.len_utf8();

        if let Some(current_len) = self.tag.inline_len() {
            if current_len + ch_len <= MAX_INLINE_LEN {
                // Still fits inline
                ch.encode_utf8(&mut self.data[current_len..]);
                self.tag = StringTag::inline(current_len + ch_len);
                return;
            }
        }

        // Need to go to heap or already on heap
        let mut s = self.as_str().to_string();
        s.push(ch);
        *self = Self::from_string(s);
    }

    /// Appends a string slice to the string.
    #[inline]
    pub fn push_str(&mut self, string: &str) {
        if string.is_empty() {
            return;
        }

        let add_len = string.len();

        if let Some(current_len) = self.tag.inline_len() {
            if current_len + add_len <= MAX_INLINE_LEN {
                // Still fits inline
                self.data[current_len..current_len + add_len].copy_from_slice(string.as_bytes());
                self.tag = StringTag::inline(current_len + add_len);
                return;
            }
        }

        // Need to go to heap or already on heap
        let mut s = self.as_str().to_string();
        s.push_str(string);
        *self = Self::from_string(s);
    }

    #[inline]
    pub fn to_lowercase(&self) -> SmartString {
        SmartString::from_string(self.as_str().to_lowercase())
    }

    #[inline]
    pub fn to_uppercase(&self) -> SmartString {
        SmartString::from_string(self.as_str().to_uppercase())
    }

    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.as_str().is_ascii()
    }

    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        if let Some(len) = self.tag.inline_len() {
            self.data[..len].make_ascii_uppercase();
        } else {
            let mut s = self.as_str().to_string();
            s.make_ascii_uppercase();
            *self = Self::from_string(s);
        }
    }

    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        if let Some(len) = self.tag.inline_len() {
            self.data[..len].make_ascii_lowercase();
        } else {
            let mut s = self.as_str().to_string();
            s.make_ascii_lowercase();
            *self = Self::from_string(s);
        }
    }

    #[inline]
    pub fn into_string(self) -> String {
        if self.tag.is_inline() {
            self.as_str().to_owned()
        } else {
            // Take ownership of the Arc without incrementing refcount
            let ptr = self.get_arc_ptr();
            std::mem::forget(self); // Prevent Drop from decrementing
                                    // SAFETY: ptr is a valid Arc<String> pointer, and we've prevented
                                    // self's Drop from running, so we're taking over its ownership
            let arc = unsafe { Arc::from_raw(ptr) };
            match Arc::try_unwrap(arc) {
                Ok(s) => s,
                Err(arc) => (*arc).clone(),
            }
        }
    }

    #[inline]
    pub fn concat(a: &str, b: &str) -> SmartString {
        let total_len = a.len() + b.len();
        if total_len <= MAX_INLINE_LEN {
            let mut data = [0u8; 15];
            data[..a.len()].copy_from_slice(a.as_bytes());
            data[a.len()..total_len].copy_from_slice(b.as_bytes());
            SmartString {
                tag: StringTag::inline(total_len),
                data,
            }
        } else {
            let mut s = String::with_capacity(total_len);
            s.push_str(a);
            s.push_str(b);
            SmartString::from_string(s)
        }
    }

    #[inline]
    pub fn concat_many(strings: &[&str]) -> SmartString {
        let total_len: usize = strings.iter().map(|s| s.len()).sum();
        if total_len <= MAX_INLINE_LEN {
            let mut data = [0u8; 15];
            let mut pos = 0;
            for s in strings {
                data[pos..pos + s.len()].copy_from_slice(s.as_bytes());
                pos += s.len();
            }
            SmartString {
                tag: StringTag::inline(total_len),
                data,
            }
        } else {
            let mut result = String::with_capacity(total_len);
            for s in strings {
                result.push_str(s);
            }
            SmartString::from_string(result)
        }
    }

    #[inline]
    pub fn from_string_heap(s: String) -> Self {
        debug_assert!(s.len() > MAX_INLINE_LEN);
        Self::new_heap(s)
    }

    /// Build an inline SmartString by writing bytes directly.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the bytes written by `builder` form valid UTF-8.
    #[inline]
    pub unsafe fn build_inline<F>(total_len: usize, builder: F) -> Option<SmartString>
    where
        F: FnOnce(&mut [u8]),
    {
        if total_len <= MAX_INLINE_LEN {
            let mut data = [0u8; 15];
            builder(&mut data[..total_len]);
            Some(SmartString {
                tag: StringTag::inline(total_len),
                data,
            })
        } else {
            None
        }
    }
}

impl Clone for SmartString {
    #[inline]
    fn clone(&self) -> Self {
        if self.tag.is_heap() {
            // Heap: increment Arc refcount to balance the new owner's Drop
            // SAFETY: get_arc_ptr() returns a valid Arc<String> pointer
            unsafe {
                Arc::increment_strong_count(self.get_arc_ptr());
            }
        }
        // Data bytes already contain the correct content (inline bytes or pointer)
        SmartString {
            tag: self.tag,
            data: self.data,
        }
    }
}

impl Drop for SmartString {
    #[inline]
    fn drop(&mut self) {
        if self.tag.is_heap() {
            // Heap: decrement Arc refcount
            let ptr = self.get_arc_ptr();
            // SAFETY: We only store valid Arc<String> pointers
            unsafe {
                Arc::from_raw(ptr);
            }
        }
    }
}

impl Default for SmartString {
    #[inline]
    fn default() -> Self {
        SmartString {
            tag: StringTag::Inline0,
            data: [0u8; 15],
        }
    }
}

impl Deref for SmartString {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for SmartString {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for SmartString {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for SmartString {
    #[inline]
    fn from(s: &str) -> Self {
        SmartString::new(s)
    }
}

impl From<String> for SmartString {
    #[inline]
    fn from(s: String) -> Self {
        SmartString::from_string(s)
    }
}

impl From<Arc<str>> for SmartString {
    #[inline]
    fn from(arc: Arc<str>) -> Self {
        SmartString::new(&arc)
    }
}

impl fmt::Debug for SmartString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for SmartString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl PartialEq for SmartString {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for SmartString {}

impl PartialOrd for SmartString {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SmartString {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for SmartString {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // IMPORTANT: Must delegate to str::hash to maintain Hash-Borrow contract.
        // SmartString implements Borrow<str>, so its hash must match str's hash
        // for the same content.
        self.as_str().hash(state);
    }
}

impl PartialEq<str> for SmartString {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for SmartString {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for SmartString {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<SmartString> for str {
    #[inline]
    fn eq(&self, other: &SmartString) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<SmartString> for &str {
    #[inline]
    fn eq(&self, other: &SmartString) -> bool {
        *self == other.as_str()
    }
}

impl From<SmartString> for String {
    #[inline]
    fn from(s: SmartString) -> Self {
        s.into_string()
    }
}

impl PartialEq<SmartString> for String {
    #[inline]
    fn eq(&self, other: &SmartString) -> bool {
        self.as_str() == other.as_str()
    }
}

impl std::iter::FromIterator<char> for SmartString {
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> Self {
        let s: String = iter.into_iter().collect();
        SmartString::from_string(s)
    }
}

impl fmt::Write for SmartString {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.push(c);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::mem::size_of;

    #[test]
    fn test_size() {
        assert_eq!(size_of::<SmartString>(), 16);
        assert_eq!(size_of::<Option<SmartString>>(), 16);
    }

    #[test]
    fn test_inline() {
        let s = SmartString::new("hello");
        assert!(s.is_inline());
        assert!(!s.is_heap());
        assert_eq!(s.as_str(), "hello");
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_inline_max() {
        // 15 bytes is the max inline length
        let s = SmartString::new("123456789012345");
        assert!(s.is_inline());
        assert_eq!(s.len(), 15);
        assert_eq!(s.as_str(), "123456789012345");
    }

    #[test]
    fn test_heap() {
        // 16 bytes exceeds inline capacity
        let s = SmartString::new("1234567890123456");
        assert!(s.is_heap());
        assert!(!s.is_inline());
        assert_eq!(s.len(), 16);
        assert_eq!(s.as_str(), "1234567890123456");
    }

    #[test]
    fn test_clone_inline() {
        let s1 = SmartString::new("hello");
        let s2 = s1.clone();
        assert_eq!(s1.as_str(), s2.as_str());
        assert!(s1.is_inline());
        assert!(s2.is_inline());
    }

    #[test]
    fn test_clone_heap() {
        let s1 = SmartString::new("this is a longer string that goes on the heap");
        let s2 = s1.clone();
        assert_eq!(s1.as_str(), s2.as_str());
        assert!(s1.is_heap());
        assert!(s2.is_heap());
    }

    #[test]
    fn test_push() {
        let mut s = SmartString::new("hello");
        s.push(' ');
        s.push('w');
        assert_eq!(s.as_str(), "hello w");
        assert!(s.is_inline());
    }

    #[test]
    fn test_push_str() {
        let mut s = SmartString::new("hello");
        s.push_str(" world");
        assert_eq!(s.as_str(), "hello world");
        assert!(s.is_inline());
    }

    #[test]
    fn test_push_str_to_heap() {
        let mut s = SmartString::new("hello");
        s.push_str(" world, this is a long string");
        assert_eq!(s.as_str(), "hello world, this is a long string");
        assert!(s.is_heap());
    }

    #[test]
    fn test_from_string() {
        let s = SmartString::from_string("hello".to_string());
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "hello");

        let s = SmartString::from_string("this is a longer string".to_string());
        assert!(s.is_heap());
        assert_eq!(s.as_str(), "this is a longer string");
    }

    #[test]
    fn test_into_string() {
        let s = SmartString::new("hello");
        let string: String = s.into_string();
        assert_eq!(string, "hello");

        let s = SmartString::new("this is a longer string");
        let string: String = s.into_string();
        assert_eq!(string, "this is a longer string");
    }

    #[test]
    fn test_eq() {
        let s1 = SmartString::new("hello");
        let s2 = SmartString::new("hello");
        let s3 = SmartString::new("world");
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_ord() {
        let s1 = SmartString::new("apple");
        let s2 = SmartString::new("banana");
        assert!(s1 < s2);
    }

    #[test]
    fn test_hash() {
        let mut map = HashMap::new();
        map.insert(SmartString::new("key"), "value");
        assert_eq!(map.get(&SmartString::new("key")), Some(&"value"));
    }

    #[test]
    fn test_borrow_lookup() {
        use std::collections::HashMap;
        let mut map: HashMap<SmartString, i32> = HashMap::new();
        map.insert(SmartString::new("hello"), 42);

        // Can look up with &str due to Borrow<str> implementation
        assert_eq!(map.get("hello"), Some(&42));
    }

    #[test]
    fn test_concat() {
        let s = SmartString::concat("hello", " world");
        assert_eq!(s.as_str(), "hello world");
        assert!(s.is_inline());

        let s = SmartString::concat("hello", " world, this is a very long string");
        assert_eq!(s.as_str(), "hello world, this is a very long string");
        assert!(s.is_heap());
    }

    #[test]
    fn test_concat_many() {
        let s = SmartString::concat_many(&["a", "b", "c"]);
        assert_eq!(s.as_str(), "abc");
        assert!(s.is_inline());
    }

    #[test]
    fn test_empty() {
        let s = SmartString::new("");
        assert!(s.is_empty());
        assert!(s.is_inline());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_unicode() {
        let s = SmartString::new("こんにちは"); // 15 bytes in UTF-8
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "こんにちは");

        let s = SmartString::new("こんにちは!"); // 16 bytes
        assert!(s.is_heap());
        assert_eq!(s.as_str(), "こんにちは!");
    }

    #[test]
    fn test_case_conversion() {
        let s = SmartString::new("Hello World");
        assert_eq!(s.to_lowercase().as_str(), "hello world");
        assert_eq!(s.to_uppercase().as_str(), "HELLO WORLD");
    }

    #[test]
    fn test_ascii_case() {
        let mut s = SmartString::new("Hello");
        s.make_ascii_uppercase();
        assert_eq!(s.as_str(), "HELLO");

        s.make_ascii_lowercase();
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_default() {
        let s = SmartString::default();
        assert!(s.is_empty());
        assert!(s.is_inline());
    }

    #[test]
    fn test_from_arc_str() {
        let arc: Arc<str> = Arc::from("hello");
        let s = SmartString::from(arc);
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_build_inline() {
        // SAFETY: Callback writes exactly 5 valid UTF-8 bytes ("hello")
        let s = unsafe {
            SmartString::build_inline(5, |buf| {
                buf.copy_from_slice(b"hello");
            })
        };
        assert!(s.is_some());
        assert_eq!(s.unwrap().as_str(), "hello");

        // SAFETY: Length 20 exceeds inline capacity, callback won't be invoked
        let s = unsafe {
            SmartString::build_inline(20, |_| {
                // Too long, won't be called
            })
        };
        assert!(s.is_none());
    }
}
