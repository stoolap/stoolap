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

/// `s` lowercased as `str::to_lowercase` does, borrowed when that changes
/// none of its characters
#[inline]
pub(crate) fn to_lowercase_cow(s: &str) -> Cow<'_, str> {
    let unchanged = match s.is_ascii() {
        true => !s.bytes().any(|b| b.is_ascii_uppercase()),
        false => !s.chars().any(changes_lowercased),
    };
    match unchanged {
        true => Cow::Borrowed(s),
        false => Cow::Owned(s.to_lowercase()),
    }
}

/// Whether lowercasing `c` gives anything but `c`; `Σ`, the one letter
/// lowered by its context, always changes
pub(crate) fn changes_lowercased(c: char) -> bool {
    let mut lower = c.to_lowercase();
    lower.next() != Some(c) || lower.next().is_some()
}

/// `f` over the characters of `s` lowercased as `str::to_lowercase` gives
/// them; a copy is made only for a name holding `Σ`, the one letter it
/// lowercases by its context
pub(crate) fn with_lowercase<R>(s: &str, f: impl FnOnce(&mut dyn Iterator<Item = char>) -> R) -> R {
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
        same_lowercase(column, lower)
    }
}

/// Whether two names are one ignoring case, each lowercased on its own as
/// `str::to_lowercase` does
pub(crate) fn eq_names(a: &str, b: &str) -> bool {
    if a.is_ascii() && b.is_ascii() {
        a.eq_ignore_ascii_case(b)
    } else {
        same_lowercase(a, b)
    }
}

/// Whether `a` and `b` lowercase to one text. Equal names also agree
/// letter by letter with both sigmas taken as one, so only a pair that
/// passes that and holds `Σ` is copied to settle its final sigma
fn same_lowercase(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    fn letters(s: &str) -> impl Iterator<Item = char> + '_ {
        s.chars().flat_map(char::to_lowercase)
    }
    if !a.contains('Σ') && !b.contains('Σ') {
        return letters(a).eq(letters(b));
    }
    let sigma = |c: char| if c == 'ς' { 'σ' } else { c };
    letters(a).map(sigma).eq(letters(b).map(sigma))
        && with_lowercase(a, |a| with_lowercase(b, |b| a.eq(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: &[&str] = &[
        "a",
        "A",
        "ab",
        "AB",
        "id",
        "äb",
        "ÄB",
        "Äb",
        "aÄ",
        "Σx",
        "σx",
        "ςx",
        "xΣ",
        "xς",
        "xσ",
        "σοφός",
        "ΣΟΦΌΣ",
        "σοφόσ",
        "ΣΟΦΌς",
        "ΣΑΣ",
        "σας",
        "σασ",
        "İx",
        "i\u{307}x",
        "ix",
        "ß",
        "SS",
        "ss",
        "straße",
        "STRASSE",
        "ǅ",
        "ǆ",
        "Ǆ",
        "K",
        "\u{212A}",
        "k",
        "",
    ];

    #[test]
    fn names_compare_as_their_lowercase_texts() {
        for a in NAMES {
            for b in NAMES {
                let same = a.to_lowercase() == b.to_lowercase();
                assert_eq!(eq_names(a, b), same, "eq_names({a:?}, {b:?})");
                assert_eq!(
                    eq_folded(a, &b.to_lowercase()),
                    same,
                    "eq_folded({a:?}, {b:?})"
                );
            }
            let lower = to_lowercase_cow(a);
            assert_eq!(lower, a.to_lowercase(), "to_lowercase_cow({a:?})");
            assert_eq!(
                matches!(lower, Cow::Borrowed(_)),
                a.to_lowercase() == *a,
                "{a:?} is borrowed exactly when lowercasing leaves it"
            );
        }
    }
}
