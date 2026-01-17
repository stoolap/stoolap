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

//! SmartString - A string type with inline small string optimization
//!
//! Uses a hybrid approach:
//! - Inline: strings ≤22 bytes stored inline (no heap allocation)
//! - Owned: String for newly created strings (avoids realloc from shrink_to_fit)
//! - Shared: Arc<str> for strings that benefit from O(1) clone

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

const MAX_INLINE_LEN: usize = 22;

/// A string with inline SSO, Box for owned, Arc for shared - best of both worlds
pub enum SmartString {
    Inline {
        len: u8,
        data: [u8; MAX_INLINE_LEN],
    },
    /// Owned heap string - O(N) clone
    Owned(Box<str>),
    /// Shared heap string - O(1) clone via Arc
    Shared(Arc<str>),
}

impl Clone for SmartString {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            SmartString::Inline { len, data } => SmartString::Inline {
                len: *len,
                data: *data,
            },
            SmartString::Owned(boxed) => SmartString::Owned(boxed.clone()),
            SmartString::Shared(arc) => SmartString::Shared(arc.clone()),
        }
    }
}

impl SmartString {
    /// Create from a string slice.
    ///
    /// **Note**: Despite the name, this is NOT a const fn. Use `new()` instead.
    /// This function exists for API compatibility and may be deprecated.
    #[inline]
    pub fn const_new(s: &str) -> Self {
        SmartString::new(s)
    }

    #[inline]
    pub fn with_capacity(_capacity: usize) -> Self {
        SmartString::default()
    }

    /// Create from a string slice - uses Shared for heap (typical for storage reads)
    #[inline]
    pub fn new(s: &str) -> Self {
        let len = s.len();
        if len <= MAX_INLINE_LEN {
            let mut data = [0u8; MAX_INLINE_LEN];
            data[..len].copy_from_slice(s.as_bytes());
            SmartString::Inline {
                len: len as u8,
                data,
            }
        } else {
            SmartString::Shared(Arc::from(s))
        }
    }

    /// Create from owned String - uses Owned for heap (zero-copy, ideal for computed values)
    #[inline]
    pub fn from_string(s: String) -> Self {
        let len = s.len();
        if len <= MAX_INLINE_LEN {
            let mut data = [0u8; MAX_INLINE_LEN];
            data[..len].copy_from_slice(s.as_bytes());
            SmartString::Inline {
                len: len as u8,
                data,
            }
        } else {
            // Zero-copy: String -> Box<str> reuses the allocation
            SmartString::Owned(s.into_boxed_str())
        }
    }

    /// Create from owned String as Shared - for values that will be cloned (e.g., GROUP BY keys)
    #[inline]
    pub fn from_string_shared(s: String) -> Self {
        let len = s.len();
        if len <= MAX_INLINE_LEN {
            let mut data = [0u8; MAX_INLINE_LEN];
            data[..len].copy_from_slice(s.as_bytes());
            SmartString::Inline {
                len: len as u8,
                data,
            }
        } else {
            SmartString::Shared(Arc::from(s))
        }
    }

    /// Returns a string slice of the contents.
    ///
    /// This is the hot path for string access. The match is ordered with
    /// Inline first as it's the most common case in database workloads.
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        match self {
            // Most common: short strings stored inline (no heap access)
            SmartString::Inline { len, data } => {
                // SAFETY: SmartString only stores valid UTF-8
                unsafe { std::str::from_utf8_unchecked(&data[..*len as usize]) }
            }
            // Heap cases - dereference pointer
            SmartString::Owned(boxed) => boxed,
            SmartString::Shared(arc) => arc,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            SmartString::Inline { len, .. } => *len as usize,
            SmartString::Owned(boxed) => boxed.len(),
            SmartString::Shared(arc) => arc.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self, SmartString::Inline { .. })
    }

    #[inline]
    pub fn is_heap(&self) -> bool {
        matches!(self, SmartString::Owned(_) | SmartString::Shared(_))
    }

    /// Convert to Shared variant for O(1) future clones
    #[inline]
    pub fn into_shared(self) -> Self {
        match self {
            SmartString::Inline { .. } => self,
            SmartString::Owned(boxed) => SmartString::Shared(Arc::from(boxed)),
            SmartString::Shared(_) => self,
        }
    }

    /// Convert self to Shared variant in place
    #[inline]
    pub fn make_shared(&mut self) {
        if let SmartString::Owned(boxed) = self {
            *self = SmartString::Shared(Arc::from(std::mem::take(boxed)));
        }
    }

    /// Appends a character to the string.
    ///
    /// # Performance Warning
    ///
    /// For `Owned` and `Shared` variants, this operation requires a full reallocation
    /// and copy on every call, resulting in O(N) per push. Building a string character
    /// by character will have O(N²) complexity. For string building, use `String` first
    /// then convert with `SmartString::from_string()`.
    #[inline]
    pub fn push(&mut self, ch: char) {
        let ch_len = ch.len_utf8();
        match self {
            SmartString::Inline { len, data } => {
                let current_len = *len as usize;
                if current_len + ch_len <= MAX_INLINE_LEN {
                    ch.encode_utf8(&mut data[current_len..]);
                    *len = (current_len + ch_len) as u8;
                } else {
                    let mut s = String::with_capacity(current_len + ch_len);
                    // SAFETY: Inline data is always valid UTF-8 - it's created from valid
                    // strings via as_bytes() and only modified by UTF-8 safe operations.
                    s.push_str(unsafe { std::str::from_utf8_unchecked(&data[..current_len]) });
                    s.push(ch);
                    *self = SmartString::Owned(s.into_boxed_str());
                }
            }
            SmartString::Owned(boxed) => {
                let mut s = String::with_capacity(boxed.len() + ch_len);
                s.push_str(boxed);
                s.push(ch);
                *self = SmartString::Owned(s.into_boxed_str());
            }
            SmartString::Shared(arc) => {
                let mut s = String::with_capacity(arc.len() + ch_len);
                s.push_str(arc);
                s.push(ch);
                *self = SmartString::Owned(s.into_boxed_str());
            }
        }
    }

    /// Appends a string slice to the string.
    ///
    /// # Performance Warning
    ///
    /// For `Owned` and `Shared` variants, this operation requires a full reallocation
    /// and copy on every call, resulting in O(N) per push. Repeated calls will have
    /// O(N²) complexity. For string building, use `String` first then convert with
    /// `SmartString::from_string()`.
    #[inline]
    pub fn push_str(&mut self, string: &str) {
        let add_len = string.len();
        if add_len == 0 {
            return;
        }
        match self {
            SmartString::Inline { len, data } => {
                let current_len = *len as usize;
                if current_len + add_len <= MAX_INLINE_LEN {
                    data[current_len..current_len + add_len].copy_from_slice(string.as_bytes());
                    *len = (current_len + add_len) as u8;
                } else {
                    let mut s = String::with_capacity(current_len + add_len);
                    // SAFETY: Inline data is always valid UTF-8 - it's created from valid
                    // strings via as_bytes() and only modified by UTF-8 safe operations.
                    s.push_str(unsafe { std::str::from_utf8_unchecked(&data[..current_len]) });
                    s.push_str(string);
                    *self = SmartString::Owned(s.into_boxed_str());
                }
            }
            SmartString::Owned(boxed) => {
                let mut s = String::with_capacity(boxed.len() + add_len);
                s.push_str(boxed);
                s.push_str(string);
                *self = SmartString::Owned(s.into_boxed_str());
            }
            SmartString::Shared(arc) => {
                let mut s = String::with_capacity(arc.len() + add_len);
                s.push_str(arc);
                s.push_str(string);
                *self = SmartString::Owned(s.into_boxed_str());
            }
        }
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
        match self {
            SmartString::Inline { len, data } => {
                data[..*len as usize].make_ascii_uppercase();
            }
            SmartString::Owned(boxed) => {
                let mut s = std::mem::take(boxed).into_string();
                s.make_ascii_uppercase();
                *self = SmartString::Owned(s.into_boxed_str());
            }
            SmartString::Shared(arc) => {
                let mut s = arc.to_string();
                s.make_ascii_uppercase();
                *self = SmartString::Owned(s.into_boxed_str());
            }
        }
    }

    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        match self {
            SmartString::Inline { len, data } => {
                data[..*len as usize].make_ascii_lowercase();
            }
            SmartString::Owned(boxed) => {
                let mut s = std::mem::take(boxed).into_string();
                s.make_ascii_lowercase();
                *self = SmartString::Owned(s.into_boxed_str());
            }
            SmartString::Shared(arc) => {
                let mut s = arc.to_string();
                s.make_ascii_lowercase();
                *self = SmartString::Owned(s.into_boxed_str());
            }
        }
    }

    #[inline]
    pub fn into_string(self) -> String {
        match self {
            SmartString::Inline { len, data } => {
                // SAFETY: Inline data is always valid UTF-8 - it's created from valid
                // strings via as_bytes() and only modified by UTF-8 safe operations.
                unsafe { std::str::from_utf8_unchecked(&data[..len as usize]) }.to_owned()
            }
            SmartString::Owned(boxed) => boxed.into_string(),
            SmartString::Shared(arc) => arc.to_string(),
        }
    }

    #[inline]
    pub fn concat(a: &str, b: &str) -> SmartString {
        let total_len = a.len() + b.len();
        if total_len <= MAX_INLINE_LEN {
            let mut data = [0u8; MAX_INLINE_LEN];
            data[..a.len()].copy_from_slice(a.as_bytes());
            data[a.len()..total_len].copy_from_slice(b.as_bytes());
            SmartString::Inline {
                len: total_len as u8,
                data,
            }
        } else {
            let mut s = String::with_capacity(total_len);
            s.push_str(a);
            s.push_str(b);
            SmartString::Owned(s.into_boxed_str())
        }
    }

    #[inline]
    pub fn concat_many(strings: &[&str]) -> SmartString {
        let total_len: usize = strings.iter().map(|s| s.len()).sum();
        if total_len <= MAX_INLINE_LEN {
            let mut data = [0u8; MAX_INLINE_LEN];
            let mut pos = 0;
            for s in strings {
                data[pos..pos + s.len()].copy_from_slice(s.as_bytes());
                pos += s.len();
            }
            SmartString::Inline {
                len: total_len as u8,
                data,
            }
        } else {
            let mut result = String::with_capacity(total_len);
            for s in strings {
                result.push_str(s);
            }
            SmartString::Owned(result.into_boxed_str())
        }
    }

    #[inline]
    pub fn from_string_heap(s: String) -> Self {
        debug_assert!(s.len() > MAX_INLINE_LEN);
        SmartString::Owned(s.into_boxed_str())
    }

    /// Build an inline SmartString by writing bytes directly.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the bytes written by `builder` form valid UTF-8.
    /// Writing invalid UTF-8 is undefined behavior.
    ///
    /// # Returns
    ///
    /// `Some(SmartString)` if `total_len <= 22`, `None` otherwise.
    #[inline]
    pub unsafe fn build_inline<F>(total_len: usize, builder: F) -> Option<SmartString>
    where
        F: FnOnce(&mut [u8]),
    {
        if total_len <= MAX_INLINE_LEN {
            let mut data = [0u8; MAX_INLINE_LEN];
            builder(&mut data[..total_len]);
            Some(SmartString::Inline {
                len: total_len as u8,
                data,
            })
        } else {
            None
        }
    }
}

impl Default for SmartString {
    #[inline]
    fn default() -> Self {
        SmartString::Inline {
            len: 0,
            data: [0u8; MAX_INLINE_LEN],
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
        if arc.len() <= MAX_INLINE_LEN {
            SmartString::new(&arc)
        } else {
            SmartString::Shared(arc)
        }
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
        match (self, other) {
            (
                SmartString::Inline { len: l1, data: d1 },
                SmartString::Inline { len: l2, data: d2 },
            ) => l1 == l2 && d1[..*l1 as usize] == d2[..*l1 as usize],
            (SmartString::Shared(a), SmartString::Shared(b)) => Arc::ptr_eq(a, b) || **a == **b,
            _ => self.as_str() == other.as_str(),
        }
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
    use std::mem::size_of;

    #[test]
    fn test_size() {
        assert_eq!(size_of::<SmartString>(), 24);
    }

    #[test]
    fn test_inline() {
        let s = SmartString::new("hello");
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_heap() {
        let s = SmartString::new("12345678901234567890123");
        assert!(s.is_heap());
        assert_eq!(s.len(), 23);
    }

    #[test]
    fn test_clone_shared_shares() {
        let s1 = SmartString::new("this is a longer string on heap");
        let s2 = s1.clone();
        if let (SmartString::Shared(a), SmartString::Shared(b)) = (&s1, &s2) {
            assert!(Arc::ptr_eq(a, b));
        }
    }

    #[test]
    fn test_from_string_uses_owned() {
        let s = SmartString::from_string("this is a longer string from owned String".to_string());
        assert!(matches!(s, SmartString::Owned(_)));
    }

    #[test]
    fn test_from_string_shared_uses_arc() {
        let s = SmartString::from_string_shared("this is a longer string for sharing".to_string());
        assert!(matches!(s, SmartString::Shared(_)));
    }

    #[test]
    fn test_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(SmartString::new("key"), "value");
        assert_eq!(map.get(&SmartString::new("key")), Some(&"value"));
    }
}
