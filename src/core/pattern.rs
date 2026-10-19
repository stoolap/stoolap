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

//! The one LIKE, ILIKE and GLOB matcher. Every path that matches a pattern,
//! the expression VM and the storage filters, compiles it here, and reads
//! the matched value as text by [`crate::core::Value::text_form`].

use std::cell::RefCell;
use std::num::NonZeroUsize;

use lru::LruCache;
use memchr::memmem;

use crate::core::Value;

/// Regexes kept per thread, as storage filters compile their pattern again
/// for every statement
const REGEX_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(128) {
    Some(size) => size,
    None => NonZeroUsize::MIN,
};

thread_local! {
    static REGEX_CACHE: RefCell<LruCache<String, regex::Regex>> =
        RefCell::new(LruCache::new(REGEX_CACHE_SIZE));
}

/// Drops the regexes this thread keeps, to release their memory
pub fn clear_regex_cache() {
    REGEX_CACHE.with(|cache| cache.borrow_mut().clear());
}

/// A compiled LIKE, ILIKE or GLOB pattern
#[derive(Debug, Clone)]
pub struct Pattern {
    shape: Shape,
    /// Whether text is matched with its case folded; the pattern is folded
    /// when compiled
    case_insensitive: bool,
}

#[derive(Debug, Clone)]
enum Shape {
    /// No wildcard
    Exact(String),
    /// `abc%`
    Prefix(String),
    /// `%abc`
    Suffix(String),
    /// `%abc%`, with its finder built once for the case-sensitive search,
    /// boxed so a filter holding the pattern stays small
    Contains(String, Box<memmem::Finder<'static>>),
    /// `abc%xyz`
    PrefixSuffix(String, String),
    /// `_` wildcards and literal characters, with an optional trailing `%`;
    /// case-sensitive only
    Template {
        chars: Vec<Option<char>>,
        trailing_any: bool,
    },
    /// `%` alone
    Any,
    /// `_` alone
    OneChar,
    /// Anything else
    Regex(regex::Regex),
}

impl Pattern {
    /// A LIKE pattern, or an ILIKE one when `case_insensitive`: `%` is any
    /// run of characters, `_` one character, and `\` makes the next of `%`,
    /// `_` and `\` literal
    pub fn like(pattern: &str, case_insensitive: bool) -> Self {
        let pat = if case_insensitive {
            pattern.to_lowercase()
        } else {
            pattern.to_string()
        };
        let shape = Self::simple_like(&pat, case_insensitive)
            .unwrap_or_else(|| Self::regex(&like_to_regex(&pat), case_insensitive, &pat));
        Pattern {
            shape,
            case_insensitive,
        }
    }

    /// A GLOB pattern: `*` is any run of characters, `?` one character,
    /// `[...]` a character class, and `\` makes the next of `*`, `?` and `\`
    /// literal; always case-sensitive
    pub fn glob(pattern: &str) -> Self {
        let pat = pattern.to_string();
        let shape = if !pat.contains(['*', '?', '[', '\\']) {
            Shape::Exact(pat)
        } else if pat == "*" {
            Shape::Any
        } else if pat == "?" {
            Shape::OneChar
        } else {
            match Self::wildcard_split(&pat, '*', &['?', '[', '\\']) {
                Some(shape) => shape,
                None => Self::regex(&glob_to_regex(&pat), false, &pat),
            }
        };
        Pattern {
            shape,
            case_insensitive: false,
        }
    }

    /// The LIKE shapes that need no regex
    fn simple_like(pat: &str, case_insensitive: bool) -> Option<Shape> {
        if pat.contains('\\') {
            return None;
        }
        if !pat.contains(['%', '_']) {
            return Some(Shape::Exact(pat.to_string()));
        }
        if pat == "%" {
            return Some(Shape::Any);
        }
        if pat == "_" {
            return Some(Shape::OneChar);
        }
        if let Some(shape) = Self::wildcard_split(pat, '%', &['_']) {
            return Some(shape);
        }
        let body = pat.strip_suffix('%').unwrap_or(pat);
        if !case_insensitive && !body.contains('%') {
            return Some(Shape::Template {
                chars: body.chars().map(|c| (c != '_').then_some(c)).collect(),
                trailing_any: body.len() != pat.len(),
            });
        }
        None
    }

    /// Exact, prefix, suffix, contains and prefix-suffix shapes of a pattern
    /// whose only wildcard is `any`, and none of `others`
    fn wildcard_split(pat: &str, any: char, others: &[char]) -> Option<Shape> {
        if pat.contains(others) {
            return None;
        }
        let parts: Vec<&str> = pat.split(any).collect();
        match parts.as_slice() {
            [whole] => Some(Shape::Exact(whole.to_string())),
            [prefix, ""] => Some(Shape::Prefix(prefix.to_string())),
            ["", suffix] => Some(Shape::Suffix(suffix.to_string())),
            ["", middle, ""] if !middle.is_empty() => Some(Shape::Contains(
                middle.to_string(),
                Box::new(memmem::Finder::new(middle.as_bytes()).into_owned()),
            )),
            [prefix, suffix] => Some(Shape::PrefixSuffix(prefix.to_string(), suffix.to_string())),
            _ => None,
        }
    }

    fn regex(body: &str, case_insensitive: bool, pat: &str) -> Shape {
        let flags = if case_insensitive { "(?i)" } else { "" };
        let source = format!("{flags}^{body}$");
        let cached = REGEX_CACHE.with(|cache| cache.borrow_mut().get(&source).cloned());
        if let Some(re) = cached {
            return Shape::Regex(re);
        }
        match regex::Regex::new(&source) {
            Ok(re) => {
                REGEX_CACHE.with(|cache| cache.borrow_mut().put(source, re.clone()));
                Shape::Regex(re)
            }
            Err(_) => Shape::Exact(pat.to_string()),
        }
    }

    /// Whether `value`, read as its [`Value::text_form`], matches the
    /// pattern; None for a NULL, which is UNKNOWN
    #[inline(always)]
    pub fn matches_value(&self, value: &Value) -> Option<bool> {
        match value {
            Value::Text(text) => Some(self.matches(text)),
            other => self.matches_text_form(other),
        }
    }

    #[cold]
    #[inline(never)]
    fn matches_text_form(&self, value: &Value) -> Option<bool> {
        value.text_form().map(|text| self.matches(&text))
    }

    /// Whether `text` matches the pattern
    #[inline(always)]
    pub fn matches(&self, text: &str) -> bool {
        if !self.case_insensitive {
            return match &self.shape {
                Shape::Exact(p) => text == p,
                Shape::Prefix(p) => text.starts_with(p.as_str()),
                Shape::Suffix(p) => text.ends_with(p.as_str()),
                Shape::Contains(_, finder) => finder.find(text.as_bytes()).is_some(),
                Shape::PrefixSuffix(prefix, suffix) => {
                    text.len() >= prefix.len() + suffix.len()
                        && text.starts_with(prefix.as_str())
                        && text.ends_with(suffix.as_str())
                }
                Shape::Template {
                    chars,
                    trailing_any,
                } => {
                    let mut text = text.chars();
                    chars.iter().all(|want| match (text.next(), want) {
                        (Some(got), Some(want)) => got == *want,
                        (Some(_), None) => true,
                        (None, _) => false,
                    }) && (*trailing_any || text.next().is_none())
                }
                Shape::Regex(re) => re.is_match(text),
                Shape::Any => true,
                Shape::OneChar => text.chars().count() == 1,
            };
        }
        self.matches_folded(text)
    }

    /// `matches` for an ILIKE pattern, out of line so a LIKE match stays small
    /// where it is inlined
    #[inline(never)]
    fn matches_folded(&self, text: &str) -> bool {
        // The pattern was folded when compiled; ASCII text is compared byte
        // by byte, any other is folded first
        let ascii = |p: &str| text.is_ascii() && p.is_ascii();
        match &self.shape {
            Shape::Exact(p) if ascii(p) => text.eq_ignore_ascii_case(p),
            Shape::Exact(p) => text.to_lowercase() == *p,
            Shape::Prefix(p) if ascii(p) => {
                text.len() >= p.len() && text[..p.len()].eq_ignore_ascii_case(p)
            }
            Shape::Prefix(p) => text.to_lowercase().starts_with(p.as_str()),
            Shape::Suffix(p) if ascii(p) => {
                text.len() >= p.len() && text[text.len() - p.len()..].eq_ignore_ascii_case(p)
            }
            Shape::Suffix(p) => text.to_lowercase().ends_with(p.as_str()),
            Shape::Contains(p, _) if ascii(p) => {
                p.len() <= text.len()
                    && text
                        .as_bytes()
                        .windows(p.len())
                        .any(|window| window.eq_ignore_ascii_case(p.as_bytes()))
            }
            Shape::Contains(p, _) => text.to_lowercase().contains(p.as_str()),
            Shape::PrefixSuffix(prefix, suffix) if ascii(prefix) && ascii(suffix) => {
                text.len() >= prefix.len() + suffix.len()
                    && text[..prefix.len()].eq_ignore_ascii_case(prefix)
                    && text[text.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
            }
            Shape::PrefixSuffix(prefix, suffix) => {
                let folded = text.to_lowercase();
                folded.len() >= prefix.len() + suffix.len()
                    && folded.starts_with(prefix.as_str())
                    && folded.ends_with(suffix.as_str())
            }
            // Compiled only for a case-sensitive pattern
            Shape::Template { .. } => false,
            Shape::Regex(re) => re.is_match(text),
            Shape::Any => true,
            Shape::OneChar => text.chars().count() == 1,
        }
    }

    /// The literal text every match starts with, for an index range: empty
    /// when the pattern starts with a wildcard or is case-insensitive
    pub fn literal_prefix(pattern: &str) -> &str {
        let end = pattern.find(['%', '_', '\\']).unwrap_or(pattern.len());
        &pattern[..end]
    }

    /// The least text above every text that starts with `prefix`, the end
    /// of its index range: `prefix` with its last character raised by one.
    /// None for an empty prefix or one that ends in the highest character
    pub fn prefix_range_end(prefix: &str) -> Option<String> {
        let last = prefix.chars().last()?;
        let next = (last as u32 + 1..=char::MAX as u32).find_map(char::from_u32)?;
        let mut end = prefix[..prefix.len() - last.len_utf8()].to_string();
        end.push(next);
        Some(end)
    }
}

/// A LIKE pattern as a regex body
fn like_to_regex(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len() * 2);
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '%' => result.push_str(".*"),
            '_' => result.push('.'),
            '\\' => match chars.peek() {
                Some(&next @ ('%' | '_' | '\\')) => {
                    chars.next();
                    result.push_str(&regex::escape(&next.to_string()));
                }
                _ => result.push_str(&regex::escape("\\")),
            },
            _ => result.push_str(&regex::escape(&c.to_string())),
        }
    }
    result
}

/// A GLOB pattern as a regex body
fn glob_to_regex(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len() * 2);
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => result.push_str(".*"),
            '?' => result.push('.'),
            '\\' => match chars.peek() {
                Some(&next @ ('*' | '?' | '\\')) => {
                    chars.next();
                    result.push_str(&regex::escape(&next.to_string()));
                }
                Some(_) => result.push_str(&regex::escape("\\")),
                None => {}
            },
            '[' => {
                result.push('[');
                for c in chars.by_ref() {
                    result.push(c);
                    if c == ']' {
                        break;
                    }
                }
            }
            _ => result.push_str(&regex::escape(&c.to_string())),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_like_shapes() {
        for (pattern, text, want) in [
            ("abc", "abc", true),
            ("abc", "abcd", false),
            ("ab%", "abz", true),
            ("%bc", "abc", true),
            ("%b%", "abc", true),
            ("a%c", "abbc", true),
            ("a%c", "ac", true),
            ("a%c", "c", false),
            ("User_1%", "User_12", true),
            ("User_1%", "User12", false),
            ("a_b", "axb", true),
            ("a_b", "axxb", false),
            ("a\\%b", "a%b", true),
            ("a\\%b", "axb", false),
            ("%", "", true),
            ("_", "é", true),
            ("a%b%c", "aXbYc", true),
            ("hello", "Hello", false),
            ("hello", "hello world", false),
            ("%", "with spaces and 123", true),
            ("_", "", false),
            ("_", "ab", false),
            ("hello%", "hello", true),
            ("hello%", "Hello", false),
            ("hello%", "say hello", false),
            ("%world", "world", true),
            ("%world", "World", false),
            ("%world", "world!", false),
            ("%ello%", "yellow", true),
            ("%ello%", "HELLO", false),
            ("hello%world", "helloworld", true),
            ("hello%world", "hello beautiful world", true),
            ("hello%world", "hello", false),
            ("abc%xyz", "abcxy", false),
            ("abc%xyz", "abcxyz", true),
            ("a%b%c", "ac", false),
            ("a_c", "aXc", true),
            ("a_c", "ac", false),
            ("a_c", "abbc", false),
            ("a_%b", "aXYZb", true),
            ("a_%b", "ab", false),
        ] {
            assert_eq!(
                Pattern::like(pattern, false).matches(text),
                want,
                "{text} LIKE {pattern}"
            );
        }
    }

    #[test]
    fn test_ilike_folds_case_beyond_ascii() {
        for (pattern, text, want) in [
            ("JOHN", "john", true),
            ("é%", "Émile", true),
            ("%MILE", "Émile", true),
            ("user_1%", "USER_12", true),
            ("%É%", "xéy", true),
            ("Hello", "HeLLo", true),
            ("Hello%", "HELLO WORLD", true),
            ("Hello%", "say hello", false),
            ("%World", "HELLO WORLD", true),
            ("%World", "world!", false),
            ("%ELLO%", "YELLOW", true),
            ("%ELLO%", "hi", false),
            ("Hello%World", "HELLO WORLD", true),
            ("Hello%World", "hello", false),
            ("ab%ba", "aba", false),
        ] {
            assert_eq!(
                Pattern::like(pattern, true).matches(text),
                want,
                "{text} ILIKE {pattern}"
            );
        }
    }

    #[test]
    fn test_glob_shapes() {
        for (pattern, text, want) in [
            ("a*", "abc", true),
            ("A*", "abc", false),
            ("*.txt", "a.txt", true),
            ("?", "x", true),
            ("[ab]c", "bc", true),
            ("a\\*", "a*", true),
            ("a\\*", "ab", false),
            ("hello", "Hello", false),
            ("*", "anything", true),
            ("?", "ab", false),
            ("hello*", "hello world", true),
            ("*world", "hello world", true),
            ("*ello*", "yellow", true),
            ("hello*world", "hello beautiful world", true),
            ("a*b*c", "aXbYc", true),
            ("a?c", "ac", false),
            ("a[bc]d", "acd", true),
            ("a[bc]*", "abz", true),
            ("*[bc]*", "xcx", true),
            ("a\\*b", "a*b", true),
            ("a\\*b", "axb", false),
            ("*a*b*", "XaYbZ", true),
            ("*a*b*", "ab", true),
        ] {
            assert_eq!(
                Pattern::glob(pattern).matches(text),
                want,
                "{text} GLOB {pattern}"
            );
        }
    }

    #[test]
    fn test_literal_prefix_stops_at_a_wildcard_or_an_escape() {
        assert_eq!(Pattern::literal_prefix("abc%"), "abc");
        assert_eq!(Pattern::literal_prefix("a\\%b"), "a");
        assert_eq!(Pattern::literal_prefix("%a"), "");
        assert_eq!(Pattern::literal_prefix("ab_c"), "ab");
    }

    #[test]
    fn test_prefix_range_end_bounds_every_text_with_the_prefix() {
        assert_eq!(Pattern::prefix_range_end("abc").as_deref(), Some("abd"));
        assert_eq!(
            Pattern::prefix_range_end("a\u{D7FF}").as_deref(),
            Some("a\u{E000}")
        );
        assert_eq!(Pattern::prefix_range_end(""), None);
        assert_eq!(Pattern::prefix_range_end("a\u{10FFFF}"), None);
        let end = Pattern::prefix_range_end("ab").unwrap();
        for text in ["ab", "abz", "ab\u{FFFF}", "ab\u{1F600}x"] {
            assert!(text >= "ab" && *text < *end, "{text:?} in [ab, {end:?})");
        }
        assert!("ac" >= end.as_str());
    }
}
