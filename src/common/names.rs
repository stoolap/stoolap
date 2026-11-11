// Copyright 2026 Stoolap Contributors
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

//! How names ignore case: two names are one when `str::to_lowercase` gives
//! each the same text, final sigma included

use std::borrow::Cow;

/// `s` lowercased as `str::to_lowercase` does, borrowed when it is ASCII
/// with no uppercase letter
#[inline]
pub(crate) fn to_lowercase_cow(s: &str) -> Cow<'_, str> {
    if s.bytes().all(|b| b.is_ascii() && !b.is_ascii_uppercase()) {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(s.to_lowercase())
    }
}

/// `f` over the characters of `s` lowercased as `str::to_lowercase` gives
/// them; a copy is made only for a name holding `Σ`, the one letter it
/// lowercases by its context
fn with_lowercase<R>(s: &str, f: impl FnOnce(&mut dyn Iterator<Item = char>) -> R) -> R {
    if s.contains('Σ') {
        f(&mut s.to_lowercase().chars())
    } else {
        f(&mut s.chars().flat_map(char::to_lowercase))
    }
}

/// Case-insensitive comparison of a column name against an already
/// lowercased name, lowercasing the column as `str::to_lowercase` does,
/// final sigma included
#[inline]
pub(crate) fn eq_folded(column: &str, lower: &str) -> bool {
    if column.is_ascii() {
        lower.is_ascii() && column.eq_ignore_ascii_case(lower)
    } else {
        with_lowercase(column, |column| column.eq(lower.chars()))
    }
}

/// Whether two names are one ignoring case, each lowercased on its own as
/// `str::to_lowercase` does
pub(crate) fn eq_names(a: &str, b: &str) -> bool {
    if a.is_ascii() && b.is_ascii() {
        a.eq_ignore_ascii_case(b)
    } else {
        with_lowercase(a, |a| with_lowercase(b, |b| a.eq(b)))
    }
}
