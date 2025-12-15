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

//! Cardinality Feedback for Query Optimization
//!
//! This module implements a learning system that improves cardinality estimates
//! by tracking the difference between estimated and actual row counts during
//! query execution. When similar predicates are seen again, the correction
//! factors are applied to produce more accurate estimates.
//!
//! ## How It Works
//!
//! 1. During EXPLAIN ANALYZE, we record estimated vs actual row counts
//! 2. A fingerprint is computed for each predicate pattern (structure, not values)
//! 3. Correction factors are stored: `correction = actual / estimated`
//! 4. Future queries with similar patterns use the correction factor
//!
//! ## Example
//!
//! ```text
//! -- First query: estimated 100, actual 1000 → correction = 10.0
//! EXPLAIN ANALYZE SELECT * FROM users WHERE status = 'active';
//!
//! -- Later query: base estimate 50, corrected estimate 500
//! SELECT * FROM users WHERE status = 'pending';
//! ```

use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::parser::ast::Expression;

/// Default decay factor for exponential moving average
pub const DEFAULT_DECAY_FACTOR: f64 = 0.3;

/// Minimum sample count before applying feedback
pub const MIN_SAMPLE_COUNT: u64 = 2;

/// Maximum correction factor to prevent extreme adjustments
pub const MAX_CORRECTION_FACTOR: f64 = 100.0;

/// Minimum correction factor
pub const MIN_CORRECTION_FACTOR: f64 = 0.01;

/// Cardinality feedback entry for a predicate pattern
#[derive(Debug, Clone)]
pub struct CardinalityFeedback {
    /// Hash of the predicate structure (not values)
    pub predicate_hash: u64,
    /// Table name this feedback applies to
    pub table_name: String,
    /// Column name (if applicable)
    pub column_name: Option<String>,
    /// Estimated row count from cost model
    pub estimated_rows: u64,
    /// Actual row count from execution
    pub actual_rows: u64,
    /// Correction factor (actual/estimated), smoothed over samples
    pub correction_factor: f64,
    /// Number of samples used to compute the correction
    pub sample_count: u64,
    /// Last update timestamp (nanoseconds since epoch)
    pub last_updated: i64,
}

impl CardinalityFeedback {
    /// Create a new feedback entry from estimated and actual row counts
    pub fn new(
        predicate_hash: u64,
        table_name: impl Into<String>,
        column_name: Option<String>,
        estimated_rows: u64,
        actual_rows: u64,
    ) -> Self {
        let correction = if estimated_rows > 0 {
            (actual_rows as f64 / estimated_rows as f64)
                .clamp(MIN_CORRECTION_FACTOR, MAX_CORRECTION_FACTOR)
        } else {
            1.0
        };

        Self {
            predicate_hash,
            table_name: table_name.into(),
            column_name,
            estimated_rows,
            actual_rows,
            correction_factor: correction,
            sample_count: 1,
            last_updated: get_current_timestamp(),
        }
    }

    /// Update feedback with a new sample using exponential moving average
    pub fn update(&mut self, estimated_rows: u64, actual_rows: u64, decay_factor: f64) {
        let new_correction = if estimated_rows > 0 {
            (actual_rows as f64 / estimated_rows as f64)
                .clamp(MIN_CORRECTION_FACTOR, MAX_CORRECTION_FACTOR)
        } else {
            1.0
        };

        // Exponential moving average: new = decay * new_sample + (1-decay) * old
        self.correction_factor =
            decay_factor * new_correction + (1.0 - decay_factor) * self.correction_factor;

        // Clamp the correction factor
        self.correction_factor = self
            .correction_factor
            .clamp(MIN_CORRECTION_FACTOR, MAX_CORRECTION_FACTOR);

        self.estimated_rows = estimated_rows;
        self.actual_rows = actual_rows;
        self.sample_count += 1;
        self.last_updated = get_current_timestamp();
    }

    /// Check if this feedback has enough samples to be reliable
    pub fn is_reliable(&self) -> bool {
        self.sample_count >= MIN_SAMPLE_COUNT
    }

    /// Apply correction to an estimate
    pub fn apply_correction(&self, base_estimate: u64) -> u64 {
        if !self.is_reliable() {
            return base_estimate;
        }
        ((base_estimate as f64 * self.correction_factor).round() as u64).max(1)
    }
}

/// Cache for cardinality feedback entries
#[derive(Debug)]
pub struct FeedbackCache {
    /// Feedback entries keyed by (table_name, predicate_hash)
    entries: RwLock<FxHashMap<(String, u64), CardinalityFeedback>>,
    /// Decay factor for EMA smoothing
    decay_factor: f64,
    /// Maximum number of entries to keep
    max_entries: usize,
}

impl Default for FeedbackCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackCache {
    /// Create a new feedback cache with default settings
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(FxHashMap::default()),
            decay_factor: DEFAULT_DECAY_FACTOR,
            max_entries: 10000,
        }
    }

    /// Create a cache with custom settings
    pub fn with_settings(decay_factor: f64, max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(FxHashMap::default()),
            decay_factor,
            max_entries,
        }
    }

    /// Record cardinality feedback for a predicate
    pub fn record_feedback(
        &self,
        table_name: &str,
        predicate_hash: u64,
        column_name: Option<String>,
        estimated_rows: u64,
        actual_rows: u64,
    ) {
        let key = (table_name.to_string(), predicate_hash);

        let mut entries = self.entries.write().unwrap();

        if let Some(existing) = entries.get_mut(&key) {
            existing.update(estimated_rows, actual_rows, self.decay_factor);
        } else {
            // Evict oldest entries if at capacity
            if entries.len() >= self.max_entries {
                self.evict_oldest(&mut entries);
            }

            let feedback = CardinalityFeedback::new(
                predicate_hash,
                table_name,
                column_name,
                estimated_rows,
                actual_rows,
            );
            entries.insert(key, feedback);
        }
    }

    /// Look up feedback for a predicate pattern
    pub fn lookup(&self, table_name: &str, predicate_hash: u64) -> Option<CardinalityFeedback> {
        let key = (table_name.to_string(), predicate_hash);
        let entries = self.entries.read().unwrap();
        entries.get(&key).cloned()
    }

    /// Get correction factor for a predicate, returns 1.0 if no feedback
    pub fn get_correction(&self, table_name: &str, predicate_hash: u64) -> f64 {
        match self.lookup(table_name, predicate_hash) {
            Some(feedback) if feedback.is_reliable() => feedback.correction_factor,
            _ => 1.0,
        }
    }

    /// Apply correction to an estimate
    pub fn apply_correction(&self, table_name: &str, predicate_hash: u64, estimate: u64) -> u64 {
        let correction = self.get_correction(table_name, predicate_hash);
        ((estimate as f64 * correction).round() as u64).max(1)
    }

    /// Clear all feedback entries
    pub fn clear(&self) {
        self.entries.write().unwrap().clear();
    }

    /// Get the number of feedback entries
    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
    }

    /// Evict oldest entries to make room for new ones
    fn evict_oldest(&self, entries: &mut FxHashMap<(String, u64), CardinalityFeedback>) {
        // Find the oldest 10% of entries
        let evict_count = self.max_entries / 10;

        let mut timestamps: Vec<_> = entries
            .iter()
            .map(|(k, v)| (k.clone(), v.last_updated))
            .collect();

        timestamps.sort_by_key(|(_, ts)| *ts);

        for (key, _) in timestamps.into_iter().take(evict_count) {
            entries.remove(&key);
        }
    }

    /// Get all feedback entries for a table
    pub fn get_table_feedback(&self, table_name: &str) -> Vec<CardinalityFeedback> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .filter(|((name, _), _)| name == table_name)
            .map(|(_, fb)| fb.clone())
            .collect()
    }
}

/// Compute a fingerprint for a predicate expression
///
/// The fingerprint captures the structure of the predicate but not the literal
/// values. This allows the same correction factor to be applied to:
/// - `status = 'active'` and `status = 'pending'` (same structure)
///
/// Different structures produce different fingerprints:
/// - `status = 'active'` vs `status = 'active' AND age > 30`
pub fn fingerprint_predicate(table_name: &str, expr: &Expression) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Include table name
    table_name.hash(&mut hasher);

    // Hash the expression structure (recursive)
    hash_expression_structure(expr, &mut hasher);

    hasher.finish()
}

/// Hash the structure of an expression (not literal values)
fn hash_expression_structure(expr: &Expression, hasher: &mut DefaultHasher) {
    // Hash the expression type discriminant
    std::mem::discriminant(expr).hash(hasher);

    match expr {
        Expression::Identifier(id) => {
            // Column names are part of structure
            id.value.hash(hasher);
        }
        Expression::QualifiedIdentifier(qid) => {
            // Table.column is structural
            qid.qualifier.value.hash(hasher);
            qid.name.value.hash(hasher);
        }
        Expression::Infix(infix) => {
            // Operator type is structural
            infix.op_type.hash(hasher);
            hash_expression_structure(&infix.left, hasher);
            hash_expression_structure(&infix.right, hasher);
        }
        Expression::Prefix(prefix) => {
            prefix.op_type.hash(hasher);
            hash_expression_structure(&prefix.right, hasher);
        }
        Expression::Between(between) => {
            "BETWEEN".hash(hasher);
            between.not.hash(hasher);
            hash_expression_structure(&between.expr, hasher);
            // Lower and upper structure matters
            hash_expression_structure(&between.lower, hasher);
            hash_expression_structure(&between.upper, hasher);
        }
        Expression::In(in_expr) => {
            "IN".hash(hasher);
            in_expr.not.hash(hasher);
            hash_expression_structure(&in_expr.left, hasher);
            // Hash the right side structure (list or subquery)
            hash_expression_structure(&in_expr.right, hasher);
        }
        Expression::Like(like) => {
            "LIKE".hash(hasher);
            like.operator.hash(hasher);
            hash_expression_structure(&like.left, hasher);
            // Pattern structure (prefix vs contains vs suffix) could matter
            // but for simplicity we just note it's a LIKE
        }
        Expression::FunctionCall(func) => {
            "FUNCTION".hash(hasher);
            func.function.hash(hasher);
            func.arguments.len().hash(hasher);
            for arg in &func.arguments {
                hash_expression_structure(arg, hasher);
            }
        }
        Expression::Case(case) => {
            "CASE".hash(hasher);
            case.when_clauses.len().hash(hasher);
            case.else_value.is_some().hash(hasher);
        }
        Expression::Cast(cast) => {
            "CAST".hash(hasher);
            cast.type_name.hash(hasher);
            hash_expression_structure(&cast.expr, hasher);
        }
        Expression::ScalarSubquery(_) => {
            "SUBQUERY".hash(hasher);
            // Subqueries are complex, just mark as present
        }
        Expression::Exists(_) => {
            "EXISTS".hash(hasher);
        }
        // Literals - don't hash the value, just the type
        Expression::IntegerLiteral(_) => {
            "INTEGER_LITERAL".hash(hasher);
        }
        Expression::FloatLiteral(_) => {
            "FLOAT_LITERAL".hash(hasher);
        }
        Expression::StringLiteral(_) => {
            "STRING_LITERAL".hash(hasher);
        }
        Expression::BooleanLiteral(_) => {
            "BOOLEAN_LITERAL".hash(hasher);
        }
        Expression::NullLiteral(_) => {
            "NULL_LITERAL".hash(hasher);
        }
        // Other expressions just mark as present
        Expression::List(list) => {
            "LIST".hash(hasher);
            list.elements.len().hash(hasher);
        }
        Expression::Star(_) => {
            "STAR".hash(hasher);
        }
        _ => {
            // For other types, just use the discriminant
            "OTHER".hash(hasher);
        }
    }
}

/// Extract the column name from a simple predicate (col = value)
pub fn extract_column_from_predicate(expr: &Expression) -> Option<String> {
    match expr {
        Expression::Infix(infix) => {
            // Check if left side is a column
            if let Expression::Identifier(id) = &*infix.left {
                return Some(id.value.clone());
            }
            if let Expression::QualifiedIdentifier(qid) = &*infix.left {
                return Some(qid.name.value.clone());
            }
            // Check if right side is a column (for reversed comparisons)
            if let Expression::Identifier(id) = &*infix.right {
                return Some(id.value.clone());
            }
            if let Expression::QualifiedIdentifier(qid) = &*infix.right {
                return Some(qid.name.value.clone());
            }
            None
        }
        Expression::Between(between) => {
            if let Expression::Identifier(id) = &*between.expr {
                return Some(id.value.clone());
            }
            if let Expression::QualifiedIdentifier(qid) = &*between.expr {
                return Some(qid.name.value.clone());
            }
            None
        }
        Expression::In(in_expr) => {
            if let Expression::Identifier(id) = &*in_expr.left {
                return Some(id.value.clone());
            }
            if let Expression::QualifiedIdentifier(qid) = &*in_expr.left {
                return Some(qid.name.value.clone());
            }
            None
        }
        Expression::Like(like) => {
            if let Expression::Identifier(id) = &*like.left {
                return Some(id.value.clone());
            }
            if let Expression::QualifiedIdentifier(qid) = &*like.left {
                return Some(qid.name.value.clone());
            }
            None
        }
        // IS NULL is expressed via Infix expression with Is/IsNot operator
        // The column extraction from Infix already handles this case
        _ => None,
    }
}

/// Get current timestamp in nanoseconds
fn get_current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// Global feedback cache (singleton pattern for easy access)
static FEEDBACK_CACHE: std::sync::OnceLock<FeedbackCache> = std::sync::OnceLock::new();

/// Get the global feedback cache
pub fn global_feedback_cache() -> &'static FeedbackCache {
    FEEDBACK_CACHE.get_or_init(FeedbackCache::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{InfixExpression, InfixOperator};
    use crate::parser::{Identifier, IntegerLiteral, Position, Token, TokenType};

    fn make_token(literal: &str) -> Token {
        Token::new(TokenType::Identifier, literal, Position::new(0, 1, 1))
    }

    fn make_identifier(name: &str) -> Expression {
        Expression::Identifier(Identifier::new(make_token(name), name.to_string()))
    }

    fn make_literal_int(val: i64) -> Expression {
        Expression::IntegerLiteral(IntegerLiteral {
            token: Token::new(TokenType::Integer, val.to_string(), Position::new(0, 1, 1)),
            value: val,
        })
    }

    fn make_equality(col: &str, val: i64) -> Expression {
        Expression::Infix(InfixExpression {
            token: Token::new(TokenType::Operator, "=", Position::new(0, 1, 1)),
            left: Box::new(make_identifier(col)),
            operator: "=".to_string(),
            op_type: InfixOperator::Equal,
            right: Box::new(make_literal_int(val)),
        })
    }

    #[test]
    fn test_feedback_entry_creation() {
        let fb = CardinalityFeedback::new(12345, "users", None, 100, 1000);
        assert_eq!(fb.correction_factor, 10.0);
        assert_eq!(fb.sample_count, 1);
        assert!(!fb.is_reliable()); // Need MIN_SAMPLE_COUNT samples
    }

    #[test]
    fn test_feedback_update_ema() {
        let mut fb = CardinalityFeedback::new(12345, "users", None, 100, 1000);
        // correction = 10.0

        // Second sample: estimated 100, actual 100 → new_correction = 1.0
        fb.update(100, 100, DEFAULT_DECAY_FACTOR);
        // EMA: 0.3 * 1.0 + 0.7 * 10.0 = 7.3
        assert!((fb.correction_factor - 7.3).abs() < 0.001);
        assert_eq!(fb.sample_count, 2);
        assert!(fb.is_reliable());
    }

    #[test]
    fn test_feedback_cache() {
        let cache = FeedbackCache::new();

        // Record feedback
        cache.record_feedback("users", 12345, Some("status".to_string()), 100, 1000);

        // First sample not reliable yet
        assert_eq!(cache.get_correction("users", 12345), 1.0);

        // Add second sample
        cache.record_feedback("users", 12345, Some("status".to_string()), 100, 1000);

        // Now should have correction
        let correction = cache.get_correction("users", 12345);
        assert!(correction > 1.0);
    }

    #[test]
    fn test_fingerprint_same_structure() {
        // Two predicates with same structure but different values should have same hash
        let pred1 = make_equality("status", 1);
        let pred2 = make_equality("status", 2);

        let hash1 = fingerprint_predicate("users", &pred1);
        let hash2 = fingerprint_predicate("users", &pred2);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_fingerprint_different_columns() {
        // Different columns should have different hashes
        let pred1 = make_equality("status", 1);
        let pred2 = make_equality("role", 1);

        let hash1 = fingerprint_predicate("users", &pred1);
        let hash2 = fingerprint_predicate("users", &pred2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_fingerprint_different_tables() {
        // Same predicate on different tables should have different hashes
        let pred = make_equality("status", 1);

        let hash1 = fingerprint_predicate("users", &pred);
        let hash2 = fingerprint_predicate("orders", &pred);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_extract_column() {
        let pred = make_equality("status", 1);
        let col = extract_column_from_predicate(&pred);
        assert_eq!(col, Some("status".to_string()));
    }

    #[test]
    fn test_apply_correction() {
        let cache = FeedbackCache::new();

        // Record multiple samples to make it reliable
        cache.record_feedback("users", 12345, None, 100, 500);
        cache.record_feedback("users", 12345, None, 100, 500);

        // Apply correction to a new estimate
        let corrected = cache.apply_correction("users", 12345, 200);

        // Should be > 200 because correction factor > 1
        assert!(corrected > 200);
    }
}
