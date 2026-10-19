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

//! LIKE expression for SQL pattern matching
//!
//! Supports SQL LIKE and ILIKE (case-insensitive) pattern matching.
//! - `%` matches any sequence of characters (including empty)
//! - `_` matches any single character

use std::any::Any;
use std::fmt;

use rustc_hash::FxHashMap;

use super::{find_column_index, resolve_alias, Expression};
use crate::core::pattern::Pattern;
use crate::core::{Result, Row, Schema, Value};

/// LIKE expression for SQL pattern matching
///
/// Matches a column value against a SQL LIKE pattern.
/// - `%` matches any sequence of characters (including empty)
/// - `_` matches any single character
///
/// # Examples
///
/// ```text
/// name LIKE 'John%'     -- starts with "John"
/// name LIKE '%son'      -- ends with "son"
/// name LIKE '%oh%'      -- contains "oh"
/// name LIKE 'J_n'       -- matches "Jon", "Jan", etc.
/// name ILIKE 'JOHN%'    -- case-insensitive match
/// ```
pub struct LikeExpr {
    /// Column name to match
    column: String,
    /// SQL LIKE pattern
    pattern: String,
    /// Whether the match is case-insensitive (ILIKE)
    case_insensitive: bool,
    /// Negated (NOT LIKE)
    negated: bool,
    /// Pre-computed column index for fast evaluation
    col_index: Option<usize>,
    /// The pattern as the shared matcher compiled it
    compiled: Pattern,
    /// Whether preparation has been attempted
    prepared: bool,
}

impl LikeExpr {
    /// Create a new LIKE expression
    pub fn new(column: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self::with_case(column.into(), pattern.into(), false)
    }

    /// Create a new ILIKE expression (case-insensitive)
    pub fn new_ilike(column: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self::with_case(column.into(), pattern.into(), true)
    }

    fn with_case(column: String, pattern: String, case_insensitive: bool) -> Self {
        let compiled = Pattern::like(&pattern, case_insensitive);
        Self {
            column,
            pattern,
            case_insensitive,
            negated: false,
            col_index: None,
            compiled,
            prepared: false,
        }
    }

    /// Create a NOT LIKE expression
    pub fn not_like(column: impl Into<String>, pattern: impl Into<String>) -> Self {
        let mut expr = Self::new(column, pattern);
        expr.negated = true;
        expr
    }

    /// Create a NOT ILIKE expression
    pub fn not_ilike(column: impl Into<String>, pattern: impl Into<String>) -> Self {
        let mut expr = Self::new_ilike(column, pattern);
        expr.negated = true;
        expr
    }

    /// Whether the row's value, read as its text form, matches, the negation
    /// applied; None when the value is NULL or missing, which is UNKNOWN
    fn holds(&self, row: &Row) -> Option<bool> {
        let matched = self.compiled.matches_value(row.get(self.col_index?)?)?;
        Some(matched != self.negated)
    }

    /// The compiled pattern, for a filter built from this expression
    pub(crate) fn compiled(&self) -> &Pattern {
        &self.compiled
    }

    /// Get the pattern (for expression compilation)
    pub fn get_pattern(&self) -> &str {
        &self.pattern
    }

    /// Check if case insensitive (for expression compilation)
    pub fn is_case_insensitive(&self) -> bool {
        self.case_insensitive
    }

    /// Check if negated (for expression compilation)
    pub fn is_negated(&self) -> bool {
        self.negated
    }
}

impl fmt::Debug for LikeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negated {
            if self.case_insensitive {
                write!(f, "{} NOT ILIKE '{}'", self.column, self.pattern)
            } else {
                write!(f, "{} NOT LIKE '{}'", self.column, self.pattern)
            }
        } else if self.case_insensitive {
            write!(f, "{} ILIKE '{}'", self.column, self.pattern)
        } else {
            write!(f, "{} LIKE '{}'", self.column, self.pattern)
        }
    }
}

impl Expression for LikeExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        Ok(self.holds(row).unwrap_or(false))
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        self.holds(row).unwrap_or(false)
    }

    fn is_unknown_due_to_null(&self, row: &Row) -> bool {
        self.col_index
            .and_then(|idx| row.get(idx))
            .is_some_and(Value::is_null)
    }

    fn with_aliases(&self, aliases: &FxHashMap<String, String>) -> Box<dyn Expression> {
        let resolved = resolve_alias(&self.column, aliases);
        Box::new(LikeExpr {
            column: resolved.to_string(),
            pattern: self.pattern.clone(),
            case_insensitive: self.case_insensitive,
            negated: self.negated,
            col_index: None,
            compiled: self.compiled.clone(),
            prepared: false,
        })
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        self.col_index = find_column_index(schema, &self.column);
        self.prepared = true;
    }

    fn collect_column_indices(&self, out: &mut Vec<usize>) -> bool {
        if let Some(idx) = self.col_index {
            out.push(idx);
            true
        } else {
            false
        }
    }

    fn is_prepared(&self) -> bool {
        self.prepared
    }

    fn get_column_name(&self) -> Option<&str> {
        Some(&self.column)
    }

    fn can_use_index(&self) -> bool {
        // LIKE can use index for prefix patterns (no leading %)
        !self.pattern.starts_with('%')
    }

    fn get_like_prefix_info(&self) -> Option<(&str, String, bool)> {
        // Only optimize non-negated, case-sensitive LIKE with prefix pattern
        if self.case_insensitive || self.pattern.starts_with('%') {
            return None;
        }

        // The literal text before the first wildcard or escape
        let prefix = Pattern::literal_prefix(&self.pattern).to_string();

        // Need at least one character of prefix to be useful
        if prefix.is_empty() {
            return None;
        }

        Some((&self.column, prefix, self.negated))
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(LikeExpr {
            column: self.column.clone(),
            pattern: self.pattern.clone(),
            case_insensitive: self.case_insensitive,
            negated: self.negated,
            col_index: self.col_index,
            compiled: self.compiled.clone(),
            prepared: self.prepared,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder, Value};

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add_nullable("email", DataType::Text)
            .build()
    }

    #[test]
    fn test_like_starts_with() {
        let schema = test_schema();
        let mut expr = LikeExpr::new("name", "John%");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("John"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("Johnny"),
            Value::null_unknown(),
        ]);
        let row3 = Row::from(vec![
            Value::Integer(3),
            Value::text("Jane"),
            Value::null_unknown(),
        ]);

        assert!(expr.evaluate(&row1).unwrap());
        assert!(expr.evaluate(&row2).unwrap());
        assert!(!expr.evaluate(&row3).unwrap());
    }

    #[test]
    fn test_like_ends_with() {
        let schema = test_schema();
        let mut expr = LikeExpr::new("name", "%son");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("Johnson"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("Jason"),
            Value::null_unknown(),
        ]);
        let row3 = Row::from(vec![
            Value::Integer(3),
            Value::text("John"),
            Value::null_unknown(),
        ]);

        assert!(expr.evaluate(&row1).unwrap());
        assert!(expr.evaluate(&row2).unwrap());
        assert!(!expr.evaluate(&row3).unwrap());
    }

    #[test]
    fn test_like_contains() {
        let schema = test_schema();
        let mut expr = LikeExpr::new("name", "%oh%");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("John"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("Mohawk"),
            Value::null_unknown(),
        ]);
        let row3 = Row::from(vec![
            Value::Integer(3),
            Value::text("Jane"),
            Value::null_unknown(),
        ]);

        assert!(expr.evaluate(&row1).unwrap());
        assert!(expr.evaluate(&row2).unwrap());
        assert!(!expr.evaluate(&row3).unwrap());
    }

    #[test]
    fn test_like_single_char() {
        let schema = test_schema();
        let mut expr = LikeExpr::new("name", "J_n");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("Jon"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("Jan"),
            Value::null_unknown(),
        ]);
        let row3 = Row::from(vec![
            Value::Integer(3),
            Value::text("John"),
            Value::null_unknown(),
        ]);

        assert!(expr.evaluate(&row1).unwrap());
        assert!(expr.evaluate(&row2).unwrap());
        assert!(!expr.evaluate(&row3).unwrap()); // 4 chars, not 3
    }

    #[test]
    fn test_ilike_case_insensitive() {
        let schema = test_schema();
        let mut expr = LikeExpr::new_ilike("name", "JOHN%");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("john"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("JOHN"),
            Value::null_unknown(),
        ]);
        let row3 = Row::from(vec![
            Value::Integer(3),
            Value::text("JoHn"),
            Value::null_unknown(),
        ]);

        assert!(expr.evaluate(&row1).unwrap());
        assert!(expr.evaluate(&row2).unwrap());
        assert!(expr.evaluate(&row3).unwrap());
    }

    #[test]
    fn test_not_like() {
        let schema = test_schema();
        let mut expr = LikeExpr::not_like("name", "John%");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("John"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("Jane"),
            Value::null_unknown(),
        ]);

        assert!(!expr.evaluate(&row1).unwrap());
        assert!(expr.evaluate(&row2).unwrap());
    }

    #[test]
    fn test_like_null() {
        let schema = test_schema();
        let mut expr = LikeExpr::new("name", "John%");
        expr.prepare_for_schema(&schema);

        let row = Row::from(vec![
            Value::Integer(1),
            Value::null_unknown(),
            Value::null_unknown(),
        ]);
        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_like_exact_match() {
        let schema = test_schema();
        let mut expr = LikeExpr::new("name", "John");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("John"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("Johnny"),
            Value::null_unknown(),
        ]);

        assert!(expr.evaluate(&row1).unwrap());
        assert!(!expr.evaluate(&row2).unwrap());
    }

    #[test]
    fn test_like_special_chars() {
        let schema = test_schema();
        // Pattern with regex special characters that should be escaped
        let mut expr = LikeExpr::new("name", "test.name%");
        expr.prepare_for_schema(&schema);

        let row1 = Row::from(vec![
            Value::Integer(1),
            Value::text("test.name123"),
            Value::null_unknown(),
        ]);
        let row2 = Row::from(vec![
            Value::Integer(2),
            Value::text("testXname123"),
            Value::null_unknown(),
        ]);

        assert!(expr.evaluate(&row1).unwrap());
        assert!(!expr.evaluate(&row2).unwrap()); // . should be literal, not regex wildcard
    }

    #[test]
    fn test_can_use_index() {
        // Prefix pattern can use index
        let expr1 = LikeExpr::new("name", "John%");
        assert!(expr1.can_use_index());

        // Leading wildcard cannot use index
        let expr2 = LikeExpr::new("name", "%John");
        assert!(!expr2.can_use_index());

        let expr3 = LikeExpr::new("name", "%John%");
        assert!(!expr3.can_use_index());
    }
}
