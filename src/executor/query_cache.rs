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

use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::parser::ast::Statement;

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
}

/// Represents a parsed and prepared statement stored in the cache
#[derive(Debug, Clone)]
pub struct CachedQueryPlan {
    /// The parsed AST (wrapped in Arc for cheap cloning - statements are immutable)
    pub statement: Arc<Statement>,
    /// Original query text
    pub query_text: String,
    /// Last time this plan was used (monotonic)
    pub last_used: Instant,
    /// Number of times this plan has been used
    pub usage_count: u64,
    /// Whether this query has parameter placeholders
    pub has_params: bool,
    /// Number of parameters required
    pub param_count: usize,
    /// Normalized query text (cache key)
    pub normalized_query: String,
}

impl CachedQueryPlan {
    /// Create a new cached query plan
    pub fn new(
        statement: Arc<Statement>,
        query_text: String,
        has_params: bool,
        param_count: usize,
        normalized_query: String,
    ) -> Self {
        Self {
            statement,
            query_text,
            last_used: Instant::now(),
            usage_count: 1,
            has_params,
            param_count,
            normalized_query,
        }
    }
}

/// Query cache for parsed SQL statements
///
/// Provides thread-safe caching of parsed SQL queries to avoid
/// the overhead of parsing the same query multiple times.
pub struct QueryCache {
    /// Cached plans indexed by normalized query text
    plans: RwLock<std::collections::HashMap<String, CachedQueryPlan>>,
    /// Maximum number of cached plans
    max_size: usize,
    /// Factor to determine how many plans to prune when cache is full (0.0-1.0)
    prune_factor: f64,
}

impl QueryCache {
    /// Create a new query cache with the given maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            plans: RwLock::new(std::collections::HashMap::new()),
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
        let plan = plans.get(normalized.as_ref())?;

        // Only clone the Arc (cheap) and copy the small fields
        Some(CachedPlanRef {
            statement: plan.statement.clone(),
            has_params: plan.has_params,
            param_count: plan.param_count,
        })
    }

    /// Add a plan to the cache
    ///
    /// Returns the cached plan for convenience.
    pub fn put(
        &self,
        query: &str,
        statement: Arc<Statement>,
        has_params: bool,
        param_count: usize,
    ) -> CachedQueryPlan {
        let normalized = normalize_query(query);
        // Convert Cow to String for storage (only allocates if not already owned)
        let normalized_key = normalized.into_owned();

        let plan = CachedQueryPlan::new(
            statement,
            query.to_string(),
            has_params,
            param_count,
            normalized_key.clone(),
        );

        if let Ok(mut plans) = self.plans.write() {
            // Check if we need to prune the cache
            if plans.len() >= self.max_size {
                self.prune_cache(&mut plans);
            }

            plans.insert(normalized_key, plan.clone());
        }

        plan
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut plans) = self.plans.write() {
            plans.clear();
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
    fn prune_cache(&self, plans: &mut std::collections::HashMap<String, CachedQueryPlan>) {
        // Calculate how many entries to remove
        let num_to_remove = ((self.max_size as f64) * self.prune_factor).ceil() as usize;
        let num_to_remove = num_to_remove.max(1);

        if plans.len() <= num_to_remove {
            return;
        }

        // Build a list of all plans sorted by last used time and usage count
        let mut entries: Vec<(String, Instant, u64)> = plans
            .iter()
            .map(|(k, p)| (k.clone(), p.last_used, p.usage_count))
            .collect();

        // Sort by last used (oldest first), then by usage count (least used first)
        entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

        // Remove the oldest/least used entries
        for (key, _, _) in entries.into_iter().take(num_to_remove) {
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
/// OPTIMIZATION: Returns Cow to avoid allocation when query is already normalized.
fn normalize_query(query: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;

    let trimmed = query.trim();

    // Fast path: check if already normalized (no consecutive whitespace)
    let needs_normalization = trimmed
        .as_bytes()
        .windows(2)
        .any(|w| w[0].is_ascii_whitespace() && w[1].is_ascii_whitespace())
        || trimmed
            .bytes()
            .any(|b| b == b'\n' || b == b'\t' || b == b'\r');

    if !needs_normalization {
        return Cow::Borrowed(trimmed);
    }

    // Slow path: normalize whitespace
    let mut result = String::with_capacity(trimmed.len());
    let mut last_was_space = false;

    for c in trimmed.chars() {
        if c.is_whitespace() {
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
