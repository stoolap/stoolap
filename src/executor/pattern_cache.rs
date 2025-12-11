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

//! Compiled Pattern Cache for LIKE expressions
//!
//! This module provides a high-performance cache for compiled LIKE patterns.
//! Instead of converting SQL LIKE patterns to regex and compiling them on every
//! row evaluation, we:
//!
//! 1. Recognize simple patterns and use optimized string operations
//! 2. Cache compiled regex patterns for complex patterns
//!
//! This optimization is inspired by PostgreSQL's internal pattern matching.
//!
//! ## Pattern Types
//!
//! - **Exact**: `'hello'` - Direct string equality
//! - **Prefix**: `'hello%'` - starts_with check
//! - **Suffix**: `'%hello'` - ends_with check
//! - **Contains**: `'%hello%'` - contains check
//! - **Complex**: `'h_llo%'` - Compiled regex (cached)

use regex::Regex;
use rustc_hash::FxHashMap;
use std::sync::RwLock;

/// Maximum number of patterns to cache (LRU eviction when exceeded)
const MAX_CACHE_SIZE: usize = 10_000;

/// OPTIMIZATION: Case-insensitive substring search without allocation
/// Uses sliding window comparison instead of to_lowercase()
#[inline]
fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }

    // Sliding window comparison
    let needle_bytes = needle.as_bytes();
    let haystack_bytes = haystack.as_bytes();

    'outer: for i in 0..=(haystack_bytes.len() - needle_bytes.len()) {
        for j in 0..needle_bytes.len() {
            if !haystack_bytes[i + j].eq_ignore_ascii_case(&needle_bytes[j]) {
                continue 'outer;
            }
        }
        return true;
    }
    false
}

/// Compiled pattern types for fast matching
#[derive(Debug, Clone)]
pub enum CompiledPattern {
    /// Exact match: `'hello'`
    Exact(String),
    /// Prefix match: `'hello%'`
    Prefix(String),
    /// Suffix match: `'%hello'`
    Suffix(String),
    /// Contains match: `'%hello%'`
    Contains(String),
    /// Prefix + Suffix: `'hello%world'`
    PrefixSuffix(String, String),
    /// Complex pattern requiring regex
    Regex(Regex),
    /// Match anything: `'%'`
    MatchAll,
    /// Match single char: `'_'`
    SingleChar,
}

impl CompiledPattern {
    /// Match the pattern against a string
    #[inline]
    pub fn matches(&self, text: &str) -> bool {
        match self {
            CompiledPattern::MatchAll => true,
            CompiledPattern::SingleChar => text.len() == 1,
            CompiledPattern::Exact(s) => text == s,
            CompiledPattern::Prefix(p) => text.starts_with(p),
            CompiledPattern::Suffix(s) => text.ends_with(s),
            CompiledPattern::Contains(c) => text.contains(c),
            CompiledPattern::PrefixSuffix(p, s) => {
                text.starts_with(p) && text.ends_with(s) && text.len() >= p.len() + s.len()
            }
            CompiledPattern::Regex(re) => re.is_match(text),
        }
    }

    /// Match case-insensitively
    #[inline]
    pub fn matches_insensitive(&self, text: &str) -> bool {
        match self {
            CompiledPattern::MatchAll => true,
            CompiledPattern::SingleChar => text.len() == 1,
            CompiledPattern::Exact(s) => text.eq_ignore_ascii_case(s),
            CompiledPattern::Prefix(p) => {
                text.len() >= p.len() && text[..p.len()].eq_ignore_ascii_case(p)
            }
            CompiledPattern::Suffix(s) => {
                text.len() >= s.len() && text[text.len() - s.len()..].eq_ignore_ascii_case(s)
            }
            CompiledPattern::Contains(c) => contains_case_insensitive(text, c),
            CompiledPattern::PrefixSuffix(p, s) => {
                if text.len() < p.len() + s.len() {
                    return false;
                }
                text[..p.len()].eq_ignore_ascii_case(p)
                    && text[text.len() - s.len()..].eq_ignore_ascii_case(s)
            }
            CompiledPattern::Regex(re) => {
                // For regex, we compile case-insensitive version separately
                re.is_match(text)
            }
        }
    }
}

/// Global pattern cache entry
struct CacheEntry {
    pattern: CompiledPattern,
    case_insensitive_pattern: Option<CompiledPattern>,
}

/// Thread-safe global cache for compiled LIKE patterns
pub struct PatternCache {
    cache: RwLock<FxHashMap<String, CacheEntry>>,
}

impl PatternCache {
    /// Create a new pattern cache
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(FxHashMap::default()),
        }
    }

    /// Get or compile a pattern for case-sensitive matching
    pub fn get_or_compile(&self, pattern: &str) -> CompiledPattern {
        // Fast path: check cache
        if let Ok(cache) = self.cache.read() {
            if let Some(entry) = cache.get(pattern) {
                return entry.pattern.clone();
            }
        }

        // Compile and cache
        let compiled = compile_pattern(pattern, false);

        if let Ok(mut cache) = self.cache.write() {
            // Evict if cache is too large
            if cache.len() >= MAX_CACHE_SIZE {
                // Simple eviction: clear half the cache
                let keys: Vec<_> = cache.keys().take(MAX_CACHE_SIZE / 2).cloned().collect();
                for key in keys {
                    cache.remove(&key);
                }
            }

            cache.insert(
                pattern.to_string(),
                CacheEntry {
                    pattern: compiled.clone(),
                    case_insensitive_pattern: None,
                },
            );
        }

        compiled
    }

    /// Get or compile a pattern for case-insensitive matching
    pub fn get_or_compile_insensitive(&self, pattern: &str) -> CompiledPattern {
        // Fast path: check cache
        if let Ok(cache) = self.cache.read() {
            if let Some(entry) = cache.get(pattern) {
                if let Some(ref ci_pattern) = entry.case_insensitive_pattern {
                    return ci_pattern.clone();
                }
            }
        }

        // Compile case-insensitive version
        let compiled = compile_pattern(pattern, true);

        if let Ok(mut cache) = self.cache.write() {
            if let Some(entry) = cache.get_mut(pattern) {
                entry.case_insensitive_pattern = Some(compiled.clone());
            } else {
                // Evict if needed
                if cache.len() >= MAX_CACHE_SIZE {
                    let keys: Vec<_> = cache.keys().take(MAX_CACHE_SIZE / 2).cloned().collect();
                    for key in keys {
                        cache.remove(&key);
                    }
                }

                cache.insert(
                    pattern.to_string(),
                    CacheEntry {
                        pattern: compile_pattern(pattern, false),
                        case_insensitive_pattern: Some(compiled.clone()),
                    },
                );
            }
        }

        compiled
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn size(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }
}

impl Default for PatternCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Global pattern cache instance
static GLOBAL_CACHE: std::sync::OnceLock<PatternCache> = std::sync::OnceLock::new();

/// Get the global pattern cache
pub fn global_pattern_cache() -> &'static PatternCache {
    GLOBAL_CACHE.get_or_init(PatternCache::new)
}

/// Compile a SQL LIKE pattern to an optimized CompiledPattern
fn compile_pattern(pattern: &str, case_insensitive: bool) -> CompiledPattern {
    // Handle special cases
    if pattern.is_empty() {
        return CompiledPattern::Exact(String::new());
    }
    if pattern == "%" {
        return CompiledPattern::MatchAll;
    }
    if pattern == "_" {
        return CompiledPattern::SingleChar;
    }

    // Check if pattern contains wildcards
    let has_percent = pattern.contains('%');
    let has_underscore = pattern.contains('_');

    // No wildcards = exact match
    if !has_percent && !has_underscore {
        return CompiledPattern::Exact(pattern.to_string());
    }

    // Try to optimize simple patterns
    if !has_underscore {
        // Only % wildcards - check for simple cases
        let parts: Vec<&str> = pattern.split('%').collect();

        match parts.as_slice() {
            // "%suffix"
            ["", suffix] if !suffix.is_empty() => {
                return CompiledPattern::Suffix(suffix.to_string());
            }
            // "prefix%"
            [prefix, ""] if !prefix.is_empty() => {
                return CompiledPattern::Prefix(prefix.to_string());
            }
            // "%contains%"
            ["", contains, ""] if !contains.is_empty() => {
                return CompiledPattern::Contains(contains.to_string());
            }
            // "prefix%suffix"
            [prefix, suffix] if !prefix.is_empty() && !suffix.is_empty() => {
                return CompiledPattern::PrefixSuffix(prefix.to_string(), suffix.to_string());
            }
            _ => {}
        }
    }

    // Complex pattern - compile to regex
    let regex_pattern = like_to_regex(pattern, case_insensitive);
    match Regex::new(&regex_pattern) {
        Ok(re) => CompiledPattern::Regex(re),
        Err(_) => {
            // Fallback to exact match on regex error
            CompiledPattern::Exact(pattern.to_string())
        }
    }
}

/// Convert SQL LIKE pattern to regex
fn like_to_regex(pattern: &str, case_insensitive: bool) -> String {
    let mut regex = String::with_capacity(pattern.len() * 2 + 4);

    if case_insensitive {
        regex.push_str("(?i)");
    }
    regex.push('^');

    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '%' => regex.push_str(".*"),
            '_' => regex.push('.'),
            '\\' => {
                // Escape sequence
                if let Some(next) = chars.next() {
                    regex.push('\\');
                    regex.push(next);
                }
            }
            // Escape regex special characters
            '.' | '^' | '$' | '*' | '+' | '?' | '{' | '}' | '[' | ']' | '(' | ')' | '|' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }

    regex.push('$');
    regex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let pattern = compile_pattern("hello", false);
        assert!(pattern.matches("hello"));
        assert!(!pattern.matches("Hello"));
        assert!(!pattern.matches("hello world"));
    }

    #[test]
    fn test_prefix_match() {
        let pattern = compile_pattern("hello%", false);
        assert!(pattern.matches("hello"));
        assert!(pattern.matches("hello world"));
        assert!(!pattern.matches("say hello"));
    }

    #[test]
    fn test_suffix_match() {
        let pattern = compile_pattern("%world", false);
        assert!(pattern.matches("world"));
        assert!(pattern.matches("hello world"));
        assert!(!pattern.matches("world hello"));
    }

    #[test]
    fn test_contains_match() {
        let pattern = compile_pattern("%ell%", false);
        assert!(pattern.matches("hello"));
        assert!(pattern.matches("yell"));
        assert!(pattern.matches("well done"));
        assert!(!pattern.matches("hallo"));
    }

    #[test]
    fn test_prefix_suffix_match() {
        let pattern = compile_pattern("hello%world", false);
        assert!(pattern.matches("helloworld"));
        assert!(pattern.matches("hello big world"));
        assert!(!pattern.matches("hello"));
        assert!(!pattern.matches("world"));
    }

    #[test]
    fn test_match_all() {
        let pattern = compile_pattern("%", false);
        assert!(pattern.matches(""));
        assert!(pattern.matches("anything"));
    }

    #[test]
    fn test_single_char() {
        let pattern = compile_pattern("_", false);
        assert!(pattern.matches("a"));
        assert!(pattern.matches("Z"));
        assert!(!pattern.matches(""));
        assert!(!pattern.matches("ab"));
    }

    #[test]
    fn test_complex_pattern() {
        let pattern = compile_pattern("h_llo%", false);
        assert!(pattern.matches("hello"));
        assert!(pattern.matches("hallo world"));
        assert!(!pattern.matches("hllo"));
    }

    #[test]
    fn test_case_insensitive() {
        let pattern = compile_pattern("hello%", false);
        assert!(pattern.matches_insensitive("Hello World"));
        assert!(pattern.matches_insensitive("HELLO"));
        assert!(!pattern.matches_insensitive("say hello"));
    }

    #[test]
    fn test_global_cache() {
        let cache = global_pattern_cache();

        // First call compiles
        let p1 = cache.get_or_compile("test%");
        assert!(p1.matches("testing"));

        // Second call should hit cache
        let p2 = cache.get_or_compile("test%");
        assert!(p2.matches("testing"));

        // Cache should have at least one entry
        assert!(cache.size() >= 1);
    }

    #[test]
    fn test_cache_insensitive() {
        let cache = global_pattern_cache();

        let p = cache.get_or_compile_insensitive("%Test%");
        assert!(p.matches_insensitive("testing"));
        assert!(p.matches_insensitive("TEST"));
    }
}
