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

//! Query Cache for parsed SQL statements
//!
//! This module provides a cache for previously parsed SQL queries,
//! storing the parse tree to avoid the overhead of parsing the same
//! query multiple times.
//!
//! # Example
//!
//! ```ignore
//! let cache = QueryCache::new(1000);
//!
//! // First query - will be parsed and cached
//! if let Some(plan) = cache.get("SELECT * FROM users") {
//!     // Use cached plan
//! } else {
//!     // Parse and cache
//!     let stmt = parse(sql);
//!     cache.put(sql, stmt, false, 0);
//! }
//!
//! // Second identical query - retrieved from cache
//! let plan = cache.get("SELECT * FROM users").unwrap();
//! ```

use std::borrow::Cow;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use compact_str::CompactString;
use rustc_hash::FxHashMap;

use crate::common::CompactArc;
use crate::core::Schema;
use crate::parser::ast::Statement;

/// Convert to lowercase without allocation if already lowercase.
#[inline]
fn to_lowercase_cow(s: &str) -> Cow<'_, str> {
    if s.bytes().all(|b| !b.is_ascii_uppercase()) {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(s.to_lowercase())
    }
}

/// How to extract the PK value for a compiled PK lookup
#[derive(Debug, Clone)]
pub enum PkValueSource {
    /// Value comes from a parameter (0-indexed)
    Parameter(usize),
    /// Value is a literal integer
    Literal(i64),
}

/// Pre-compiled state for PK lookup queries (SELECT * WHERE pk = value)
#[derive(Debug, Clone)]
pub struct CompiledPkLookup {
    /// Table name (already lowercased)
    pub table_name: CompactString,
    /// Cached schema
    pub schema: Arc<Schema>,
    /// Pre-computed column names for result (CompactArc<Vec<String>> for zero-copy O(1) clone on execution)
    pub column_names: CompactArc<Vec<String>>,
    /// How to extract the PK value
    pub pk_value_source: PkValueSource,
    /// Schema epoch at compilation time (for fast cache invalidation)
    pub cached_epoch: u64,
}

/// Pre-compiled update column assignment
#[derive(Debug, Clone)]
pub struct CompiledUpdateColumn {
    /// Column index in the schema
    pub column_idx: usize,
    /// Column data type for coercion
    pub column_type: crate::core::DataType,
    /// How to get the new value (literal or parameter)
    pub value_source: UpdateValueSource,
}

/// How to get the update value
#[derive(Debug, Clone)]
pub enum UpdateValueSource {
    /// Value is a literal
    Literal(crate::core::Value),
    /// Value comes from a parameter (0-indexed)
    Parameter(usize),
}

/// Pre-compiled state for PK-based UPDATE (UPDATE table SET col = val WHERE pk = value)
#[derive(Debug, Clone)]
pub struct CompiledPkUpdate {
    /// Table name (already lowercased)
    pub table_name: CompactString,
    /// Cached schema
    pub schema: Arc<Schema>,
    /// Cached PK column name
    pub pk_column_name: CompactString,
    /// How to extract the PK value
    pub pk_value_source: PkValueSource,
    /// Pre-compiled column assignments
    pub updates: Vec<CompiledUpdateColumn>,
    /// Schema epoch at compilation time (for fast cache invalidation)
    pub cached_epoch: u64,
}

/// Pre-compiled state for PK-based DELETE (DELETE FROM table WHERE pk = value)
#[derive(Debug, Clone)]
pub struct CompiledPkDelete {
    /// Table name (already lowercased)
    pub table_name: CompactString,
    /// Cached schema
    pub schema: Arc<Schema>,
    /// Cached PK column name
    pub pk_column_name: CompactString,
    /// How to extract the PK value
    pub pk_value_source: PkValueSource,
    /// Schema epoch at compilation time (for fast cache invalidation)
    pub cached_epoch: u64,
}

/// Pre-compiled state for INSERT statements
/// Caches schema-derived information to avoid recomputation on every INSERT execution
#[derive(Debug, Clone)]
pub struct CompiledInsert {
    /// Table name (already lowercased, CompactString for inline storage)
    pub table_name: CompactString,
    /// Column indices for INSERT (which schema columns to populate)
    pub column_indices: Arc<Vec<usize>>,
    /// Column types for the INSERT columns (for type coercion)
    pub column_types: Arc<Vec<crate::core::DataType>>,
    /// Column names for error messages (Arc for zero-copy sharing)
    pub column_names: Arc<Vec<CompactString>>,
    /// All column types in the schema (for default value evaluation)
    pub all_column_types: Arc<Vec<crate::core::DataType>>,
    /// Pre-evaluated default values for all columns (avoids re-evaluation per row)
    /// Each element is either the default Value or null_unknown if no default.
    pub default_row_template: Arc<Vec<crate::core::Value>>,
    /// CHECK constraint expressions: (column_idx, column_name, check_expr)
    pub check_exprs: Arc<Vec<(usize, CompactString, CompactString)>>,
    /// Schema epoch at compilation time (for fast cache invalidation)
    pub cached_epoch: u64,
}

/// Pre-compiled execution state for fast paths
#[derive(Debug, Clone, Default)]
pub enum CompiledExecution {
    /// Not analyzed yet
    #[default]
    Unknown,
    /// Analyzed but doesn't qualify for any fast path
    NotOptimizable,
    /// PK lookup fast path (SELECT)
    PkLookup(CompiledPkLookup),
    /// PK-based UPDATE fast path
    PkUpdate(CompiledPkUpdate),
    /// PK-based DELETE fast path
    PkDelete(CompiledPkDelete),
    /// Cached INSERT compilation (schema-derived info)
    Insert(CompiledInsert),
}

/// Default cache size (number of cached plans)
pub const DEFAULT_CACHE_SIZE: usize = 1000;

/// Lightweight reference to a cached plan for query execution.
/// Contains only what's needed to execute: the immutable statement and param info.
#[derive(Debug, Clone)]
pub struct CachedPlanRef {
    /// The parsed AST (cheap Arc clone)
    pub statement: Arc<Statement>,
    /// Whether this query has parameter placeholders
    pub has_params: bool,
    /// Number of parameters required
    pub param_count: usize,
    /// Shared reference to compiled execution state (lazily populated)
    pub compiled: Arc<RwLock<CompiledExecution>>,
}

/// Represents a parsed and prepared statement stored in the cache
#[derive(Debug, Clone)]
pub struct CachedQueryPlan {
    /// The parsed AST (wrapped in Arc for cheap cloning - statements are immutable)
    pub statement: Arc<Statement>,
    /// Original query text
    pub query_text: CompactString,
    /// Last time this plan was used (monotonic)
    pub last_used: Instant,
    /// Number of times this plan has been used
    pub usage_count: u64,
    /// Whether this query has parameter placeholders
    pub has_params: bool,
    /// Number of parameters required
    pub param_count: usize,
    /// Normalized query text (cache key)
    pub normalized_query: CompactString,
    /// Compiled execution state (lazily populated on first execution)
    pub compiled: Arc<RwLock<CompiledExecution>>,
}

impl CachedQueryPlan {
    /// Create a new cached query plan
    pub fn new(
        statement: Arc<Statement>,
        query_text: CompactString,
        has_params: bool,
        param_count: usize,
        normalized_query: CompactString,
    ) -> Self {
        Self {
            statement,
            query_text,
            last_used: Instant::now(),
            usage_count: 1,
            has_params,
            param_count,
            normalized_query,
            compiled: Arc::new(RwLock::new(CompiledExecution::Unknown)),
        }
    }
}

/// Query cache for parsed SQL statements
///
/// Provides thread-safe caching of parsed SQL queries to avoid
/// the overhead of parsing the same query multiple times.
pub struct QueryCache {
    /// Cached plans indexed by normalized query text (FxHash for fast string hashing)
    plans: RwLock<FxHashMap<CompactString, CachedQueryPlan>>,
    /// Maximum number of cached plans
    max_size: usize,
    /// Factor to determine how many plans to prune when cache is full (0.0-1.0)
    prune_factor: f64,
}

impl QueryCache {
    /// Create a new query cache with the given maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            plans: RwLock::new(FxHashMap::default()),
            max_size,
            prune_factor: 0.2, // Prune 20% of entries when cache is full
        }
    }

    /// Create a new query cache with default size
    pub fn default_sized() -> Self {
        Self::new(DEFAULT_CACHE_SIZE)
    }

    /// Get a cached plan for a query if available
    ///
    /// Returns a cheap Arc clone of the cached statement and metadata.
    /// The Statement is immutable and shared via Arc.
    ///
    /// OPTIMIZATION: Only uses read lock for cache hits (no write lock for stats).
    /// Stats are only updated on cache misses via put().
    pub fn get(&self, query: &str) -> Option<CachedPlanRef> {
        let normalized = normalize_query(query);

        // Only use read lock - skip stats update for performance
        let plans = self.plans.read().ok()?;
        // Use the Cow<str> as a key lookup without allocating CompactString
        let plan = plans.get(normalized.as_ref())?;

        // Only clone the Arc (cheap) and copy the small fields
        Some(CachedPlanRef {
            statement: plan.statement.clone(),
            has_params: plan.has_params,
            param_count: plan.param_count,
            compiled: plan.compiled.clone(), // Share compiled state
        })
    }

    /// Add a plan to the cache
    ///
    /// Returns a lightweight reference to the cached plan (CachedPlanRef).
    /// This avoids cloning CompactStrings since callers only need the statement
    /// and compiled execution state.
    pub fn put(
        &self,
        query: &str,
        statement: Arc<Statement>,
        has_params: bool,
        param_count: usize,
    ) -> CachedPlanRef {
        let normalized = normalize_query(query);
        // Convert Cow to CompactString for storage
        let normalized_key: CompactString = match normalized {
            Cow::Borrowed(s) => CompactString::new(s),
            Cow::Owned(s) => CompactString::new(&s),
        };

        // Create the compiled state upfront - shared between stored plan and returned ref
        let compiled = Arc::new(RwLock::new(CompiledExecution::Unknown));

        if let Ok(mut plans) = self.plans.write() {
            // Check if we need to prune the cache
            if plans.len() >= self.max_size {
                self.prune_cache(&mut plans);
            }

            // Insert plan into map - use normalized_key for both key and field
            // Only clone normalized_key for the map key; move it into the plan struct
            let key_for_insert = normalized_key.clone();
            plans.insert(
                key_for_insert,
                CachedQueryPlan {
                    statement: statement.clone(),
                    query_text: CompactString::new(query),
                    last_used: Instant::now(),
                    usage_count: 1,
                    has_params,
                    param_count,
                    normalized_query: normalized_key, // moved, not cloned
                    compiled: compiled.clone(),       // Arc clone - cheap
                },
            );
        }

        // Return lightweight reference - only Arc clones, no CompactString clones
        CachedPlanRef {
            statement,
            has_params,
            param_count,
            compiled,
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut plans) = self.plans.write() {
            plans.clear();
        }
    }

    /// Invalidate all cached plans that reference a specific table
    /// Called after DDL operations (ALTER TABLE, DROP TABLE, etc.)
    pub fn invalidate_table(&self, table_name: &str) {
        let table_lower = to_lowercase_cow(table_name);
        if let Ok(mut plans) = self.plans.write() {
            // Remove plans that reference this table
            // Check both the compiled PK lookup table name and query text
            plans.retain(|_key, plan| {
                // Check if compiled lookup references this table
                if let Ok(compiled) = plan.compiled.read() {
                    if let CompiledExecution::PkLookup(lookup) = &*compiled {
                        if lookup.table_name == *table_lower {
                            return false; // Remove this plan
                        }
                    }
                }
                // Also check query text for table reference (simple heuristic)
                let query_lower = to_lowercase_cow(&plan.query_text);
                !query_lower.contains(&format!(" {} ", &*table_lower))
                    && !query_lower.contains(&format!(" {}\n", &*table_lower))
                    && !query_lower.contains(&format!(" {};", &*table_lower))
                    && !query_lower.contains(&format!("from {}", &*table_lower))
                    && !query_lower.contains(&format!("join {}", &*table_lower))
                    && !query_lower.contains(&format!("into {}", &*table_lower))
                    && !query_lower.contains(&format!("update {}", &*table_lower))
            });
        }
    }

    /// Get the number of plans in the cache
    pub fn size(&self) -> usize {
        self.plans.read().map(|p| p.len()).unwrap_or(0)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let plans = match self.plans.read() {
            Ok(p) => p,
            Err(_) => {
                return CacheStats {
                    size: 0,
                    max_size: self.max_size,
                    total_usage: 0,
                    avg_usage: 0.0,
                }
            }
        };

        let size = plans.len();
        let total_usage: u64 = plans.values().map(|p| p.usage_count).sum();
        let avg_usage = if size > 0 {
            total_usage as f64 / size as f64
        } else {
            0.0
        };

        CacheStats {
            size,
            max_size: self.max_size,
            total_usage,
            avg_usage,
        }
    }

    /// Prune the least recently used entries when the cache is full
    fn prune_cache(&self, plans: &mut FxHashMap<CompactString, CachedQueryPlan>) {
        // Calculate how many entries to remove
        let num_to_remove = ((self.max_size as f64) * self.prune_factor).ceil() as usize;
        let num_to_remove = num_to_remove.max(1);

        if plans.len() <= num_to_remove {
            return;
        }

        // Build a list of references sorted by last used time and usage count
        // Use references to avoid cloning all keys
        let mut entries: Vec<(&CompactString, Instant, u64)> = plans
            .iter()
            .map(|(k, p)| (k, p.last_used, p.usage_count))
            .collect();

        // Sort by last used (oldest first), then by usage count (least used first)
        entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

        // Collect only the keys to remove (clone only what we need)
        let keys_to_remove: Vec<CompactString> = entries
            .into_iter()
            .take(num_to_remove)
            .map(|(k, _, _)| k.clone())
            .collect();

        // Remove the oldest/least used entries
        for key in keys_to_remove {
            plans.remove(&key);
        }
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::default_sized()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of cached plans
    pub size: usize,
    /// Maximum cache size
    pub max_size: usize,
    /// Total usage count across all cached plans
    pub total_usage: u64,
    /// Average usage per cached plan
    pub avg_usage: f64,
}

/// Normalize a query for caching
///
/// This handles irrelevant whitespace differences to improve cache hits.
/// OPTIMIZATION: Single-pass check + Cow to avoid allocation when already normalized.
#[inline]
fn normalize_query(query: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;

    let bytes = query.as_bytes();
    let len = bytes.len();

    // Find start (skip leading whitespace)
    let start = bytes
        .iter()
        .position(|&b| !b.is_ascii_whitespace())
        .unwrap_or(len);
    if start == len {
        return Cow::Borrowed("");
    }

    // Find end (skip trailing whitespace)
    let end = bytes
        .iter()
        .rposition(|&b| !b.is_ascii_whitespace())
        .map(|p| p + 1)
        .unwrap_or(start);

    // Single pass: check if normalization needed (consecutive whitespace or non-space whitespace)
    let trimmed = &bytes[start..end];
    let mut prev_ws = false;
    let mut needs_normalization = false;

    for &b in trimmed {
        let is_ws = b.is_ascii_whitespace();
        if is_ws {
            // Check for consecutive whitespace or non-space whitespace chars
            if prev_ws || b != b' ' {
                needs_normalization = true;
                break;
            }
        }
        prev_ws = is_ws;
    }

    if !needs_normalization {
        // SAFETY: trimmed is valid UTF-8 since it's a slice of a UTF-8 string
        return Cow::Borrowed(unsafe { std::str::from_utf8_unchecked(trimmed) });
    }

    // Slow path: normalize whitespace (rarely hit for well-formed queries)
    // SAFETY: trimmed is valid UTF-8 since it's a slice of a UTF-8 string
    // at ASCII whitespace boundaries (ASCII chars are single-byte in UTF-8)
    let trimmed_str = unsafe { std::str::from_utf8_unchecked(trimmed) };
    let mut result = String::with_capacity(trimmed.len());
    let mut last_was_space = false;

    for c in trimmed_str.chars() {
        if c.is_ascii_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(c);
            last_was_space = false;
        }
    }

    Cow::Owned(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{Expression, GroupByClause, SelectStatement, StarExpression};
    use crate::parser::token::{Position, Token, TokenType};

    fn dummy_token() -> Token {
        Token::new(TokenType::Keyword, "SELECT", Position::new(0, 1, 1))
    }

    fn star_token() -> Token {
        Token::new(TokenType::Operator, "*", Position::new(0, 1, 1))
    }

    fn create_test_statement() -> Arc<Statement> {
        Arc::new(Statement::Select(SelectStatement {
            token: dummy_token(),
            with: None,
            distinct: false,
            columns: vec![Expression::Star(StarExpression {
                token: star_token(),
            })],
            table_expr: None,
            where_clause: None,
            group_by: GroupByClause::default(),
            having: None,
            window_defs: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            set_operations: vec![],
        }))
    }

    #[test]
    fn test_cache_put_get() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        // Put in cache
        cache.put("SELECT * FROM users", stmt.clone(), false, 0);
        assert_eq!(cache.size(), 1);

        // Get from cache
        let plan = cache.get("SELECT * FROM users");
        assert!(plan.is_some());

        let plan = plan.unwrap();
        assert!(!plan.has_params);
        assert_eq!(plan.param_count, 0);
    }

    #[test]
    fn test_cache_miss() {
        let cache = QueryCache::new(100);

        let plan = cache.get("SELECT * FROM users");
        assert!(plan.is_none());
    }

    #[test]
    fn test_cache_usage_count() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT * FROM users", stmt, false, 0);

        // Get multiple times - OPTIMIZATION: get() no longer updates stats
        // to avoid write lock contention
        for _ in 0..5 {
            cache.get("SELECT * FROM users");
        }

        // Usage count is only incremented on put(), not get() (for performance)
        let stats = cache.stats();
        assert_eq!(stats.total_usage, 1); // Only initial put = 1
    }

    #[test]
    fn test_cache_clear() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT * FROM users", stmt, false, 0);
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_pruning() {
        let cache = QueryCache::new(5);
        let stmt = create_test_statement();

        // Fill the cache
        for i in 0..10 {
            let query = format!("SELECT * FROM table{}", i);
            cache.put(&query, stmt.clone(), false, 0);
        }

        // Cache should have pruned some entries
        assert!(cache.size() <= 5);
    }

    #[test]
    fn test_normalize_query() {
        assert_eq!(
            normalize_query("  SELECT  *  FROM  users  "),
            "SELECT * FROM users"
        );
        assert_eq!(
            normalize_query("SELECT\n*\nFROM\nusers"),
            "SELECT * FROM users"
        );
        assert_eq!(
            normalize_query("SELECT\t*\t\tFROM users"),
            "SELECT * FROM users"
        );
    }

    #[test]
    fn test_normalize_query_utf8() {
        // UTF-8 characters should be preserved in fast path (no normalization needed)
        assert_eq!(
            normalize_query("SELECT * FROM t WHERE name = '日本語'"),
            "SELECT * FROM t WHERE name = '日本語'"
        );

        // UTF-8 characters should be preserved in slow path (normalization needed)
        assert_eq!(
            normalize_query("SELECT  *  FROM t WHERE name = '日本語'"),
            "SELECT * FROM t WHERE name = '日本語'"
        );

        // Mixed ASCII and UTF-8 with tabs/newlines
        assert_eq!(
            normalize_query("SELECT\t*\tFROM t WHERE city = '東京' AND country = '中国'"),
            "SELECT * FROM t WHERE city = '東京' AND country = '中国'"
        );

        // Emoji should also be preserved
        assert_eq!(
            normalize_query("SELECT  *  FROM t WHERE emoji = '🎉'"),
            "SELECT * FROM t WHERE emoji = '🎉'"
        );
    }

    #[test]
    fn test_normalized_cache_hit() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        // Put with one formatting
        cache.put("SELECT * FROM users", stmt, false, 0);

        // Get with different formatting should still hit
        let plan = cache.get("  SELECT  *  FROM  users  ");
        assert!(plan.is_some());
    }

    #[test]
    fn test_parameterized_query() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT * FROM users WHERE id = $1", stmt, true, 1);

        let plan = cache.get("SELECT * FROM users WHERE id = $1").unwrap();
        assert!(plan.has_params);
        assert_eq!(plan.param_count, 1);
    }

    #[test]
    fn test_cache_stats() {
        let cache = QueryCache::new(100);
        let stmt = create_test_statement();

        cache.put("SELECT 1", stmt.clone(), false, 0);
        cache.put("SELECT 2", stmt.clone(), false, 0);

        // Access first query more - get() no longer updates stats (OPTIMIZATION)
        for _ in 0..5 {
            cache.get("SELECT 1");
        }

        let stats = cache.stats();
        assert_eq!(stats.size, 2);
        assert_eq!(stats.max_size, 100);
        assert_eq!(stats.total_usage, 2); // Only 2 puts (gets don't update stats)
    }

    #[test]
    fn test_cache_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(QueryCache::new(1000));
        let stmt = create_test_statement();

        // Pre-populate
        cache.put("SELECT * FROM users", stmt.clone(), false, 0);

        let mut handles = vec![];

        // Spawn multiple reader threads
        for _ in 0..10 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    cache.get("SELECT * FROM users");
                }
            }));
        }

        // Spawn writer threads
        for i in 0..5 {
            let cache = Arc::clone(&cache);
            let stmt = stmt.clone();
            handles.push(thread::spawn(move || {
                for j in 0..20 {
                    let query = format!("SELECT * FROM table{}_{}", i, j);
                    cache.put(&query, stmt.clone(), false, 0);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should still be functional
        assert!(cache.get("SELECT * FROM users").is_some());
    }
}
