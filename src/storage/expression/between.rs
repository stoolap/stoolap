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

//! BETWEEN expression for Stoolap
//!

use std::any::Any;
use std::collections::HashMap;

use chrono::{DateTime, Utc};

use super::{find_column_index, resolve_alias, Expression};
use crate::core::{Result, Row, Schema, Value};

/// BETWEEN expression (column BETWEEN low AND high)
///
/// By default, BETWEEN is inclusive (>= low AND <= high).
#[derive(Debug, Clone)]
pub struct BetweenExpr {
    /// Column name
    column: String,
    /// Lower bound
    lower_bound: Value,
    /// Upper bound
    upper_bound: Value,
    /// Whether bounds are inclusive (true for standard BETWEEN)
    inclusive: bool,
    /// Whether this is a NOT BETWEEN expression
    /// When true and the value is NULL, returns false (SQL standard: NOT NULL = NULL = false in WHERE)
    not: bool,

    /// Pre-computed column index
    col_index: Option<usize>,

    /// Column aliases
    aliases: HashMap<String, String>,
    /// Original column name if using alias
    original_column: Option<String>,
}

impl BetweenExpr {
    /// Create a new BETWEEN expression (inclusive by default)
    pub fn new(column: impl Into<String>, lower: Value, upper: Value) -> Self {
        Self {
            column: column.into(),
            lower_bound: lower,
            upper_bound: upper,
            inclusive: true,
            not: false,
            col_index: None,
            aliases: HashMap::new(),
            original_column: None,
        }
    }

    /// Create a NOT BETWEEN expression
    pub fn not_between(column: impl Into<String>, lower: Value, upper: Value) -> Self {
        Self {
            column: column.into(),
            lower_bound: lower,
            upper_bound: upper,
            inclusive: true,
            not: true,
            col_index: None,
            aliases: HashMap::new(),
            original_column: None,
        }
    }

    /// Create a BETWEEN expression with custom inclusivity
    pub fn with_inclusivity(
        column: impl Into<String>,
        lower: Value,
        upper: Value,
        inclusive: bool,
    ) -> Self {
        Self {
            column: column.into(),
            lower_bound: lower,
            upper_bound: upper,
            inclusive,
            not: false,
            col_index: None,
            aliases: HashMap::new(),
            original_column: None,
        }
    }

    /// Check if inclusive
    pub fn is_inclusive(&self) -> bool {
        self.inclusive
    }

    /// Get the bounds (for expression compilation)
    pub fn get_bounds(&self) -> (&Value, &Value) {
        (&self.lower_bound, &self.upper_bound)
    }

    /// Check if this is a NOT BETWEEN expression
    pub fn is_negated(&self) -> bool {
        self.not
    }

    /// Compare integers with bounds
    #[inline]
    fn check_integer(&self, val: i64, lower: i64, upper: i64) -> bool {
        if self.inclusive {
            val >= lower && val <= upper
        } else {
            val > lower && val < upper
        }
    }

    /// Compare floats with bounds
    #[inline]
    fn check_float(&self, val: f64, lower: f64, upper: f64) -> bool {
        if self.inclusive {
            val >= lower && val <= upper
        } else {
            val > lower && val < upper
        }
    }

    /// Compare strings with bounds
    #[inline]
    fn check_string(&self, val: &str, lower: &str, upper: &str) -> bool {
        if self.inclusive {
            val >= lower && val <= upper
        } else {
            val > lower && val < upper
        }
    }

    /// Compare timestamps with bounds
    #[inline]
    fn check_timestamp(
        &self,
        val: DateTime<Utc>,
        lower: DateTime<Utc>,
        upper: DateTime<Utc>,
    ) -> bool {
        if self.inclusive {
            val >= lower && val <= upper
        } else {
            val > lower && val < upper
        }
    }
}

impl Expression for BetweenExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return Ok(false),
        };

        let col_value = &row[col_idx];

        // NULL BETWEEN ... is always false (NULL in WHERE context is false)
        // NOT BETWEEN with NULL is also false (NOT NULL = NULL = false in WHERE)
        if col_value.is_null() {
            return Ok(false);
        }

        // Type-specific comparisons
        let in_range =
            match col_value {
                Value::Integer(val) => {
                    let lower = self.lower_bound.as_int64().ok_or_else(|| {
                        crate::core::Error::type_conversion("lower bound", "integer")
                    })?;
                    let upper = self.upper_bound.as_int64().ok_or_else(|| {
                        crate::core::Error::type_conversion("upper bound", "integer")
                    })?;
                    self.check_integer(*val, lower, upper)
                }

                Value::Float(val) => {
                    let lower = self.lower_bound.as_float64().ok_or_else(|| {
                        crate::core::Error::type_conversion("lower bound", "float")
                    })?;
                    let upper = self.upper_bound.as_float64().ok_or_else(|| {
                        crate::core::Error::type_conversion("upper bound", "float")
                    })?;
                    self.check_float(*val, lower, upper)
                }

                Value::Text(val) => {
                    let lower = self.lower_bound.as_string().ok_or_else(|| {
                        crate::core::Error::type_conversion("lower bound", "string")
                    })?;
                    let upper = self.upper_bound.as_string().ok_or_else(|| {
                        crate::core::Error::type_conversion("upper bound", "string")
                    })?;
                    self.check_string(val, &lower, &upper)
                }

                Value::Timestamp(val) => {
                    let lower = self.lower_bound.as_timestamp().ok_or_else(|| {
                        crate::core::Error::type_conversion("lower bound", "timestamp")
                    })?;
                    let upper = self.upper_bound.as_timestamp().ok_or_else(|| {
                        crate::core::Error::type_conversion("upper bound", "timestamp")
                    })?;
                    self.check_timestamp(*val, lower, upper)
                }

                _ => false,
            };

        // Apply NOT if this is a NOT BETWEEN expression
        Ok(if self.not { !in_range } else { in_range })
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return false,
        };

        let col_value = &row[col_idx];

        // NULL BETWEEN ... is always false (NULL in WHERE context is false)
        // NOT BETWEEN with NULL is also false (NOT NULL = NULL = false in WHERE)
        if col_value.is_null() {
            return false;
        }

        let in_range = match col_value {
            Value::Integer(val) => {
                if let (Some(lower), Some(upper)) =
                    (self.lower_bound.as_int64(), self.upper_bound.as_int64())
                {
                    self.check_integer(*val, lower, upper)
                } else {
                    return false;
                }
            }

            Value::Float(val) => {
                if let (Some(lower), Some(upper)) =
                    (self.lower_bound.as_float64(), self.upper_bound.as_float64())
                {
                    self.check_float(*val, lower, upper)
                } else {
                    return false;
                }
            }

            Value::Text(val) => {
                if let (Some(lower), Some(upper)) =
                    (self.lower_bound.as_string(), self.upper_bound.as_string())
                {
                    self.check_string(val, &lower, &upper)
                } else {
                    return false;
                }
            }

            Value::Timestamp(val) => {
                if let (Some(lower), Some(upper)) = (
                    self.lower_bound.as_timestamp(),
                    self.upper_bound.as_timestamp(),
                ) {
                    self.check_timestamp(*val, lower, upper)
                } else {
                    return false;
                }
            }

            _ => return false,
        };

        // Apply NOT if this is a NOT BETWEEN expression
        if self.not {
            !in_range
        } else {
            in_range
        }
    }

    fn with_aliases(&self, aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        let resolved = resolve_alias(&self.column, aliases);
        let mut expr = self.clone();

        if resolved != self.column {
            expr.original_column = Some(self.column.clone());
            expr.column = resolved.to_string();
        }

        expr.aliases = aliases.clone();
        expr.col_index = None;
        Box::new(expr)
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        if self.col_index.is_some() {
            return;
        }
        self.col_index = find_column_index(schema, &self.column);
    }

    fn is_prepared(&self) -> bool {
        self.col_index.is_some()
    }

    fn get_column_name(&self) -> Option<&str> {
        Some(&self.column)
    }

    fn can_use_index(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, SchemaBuilder};

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("score", DataType::Float)
            .add("name", DataType::Text)
            .build()
    }

    #[test]
    fn test_integer_between() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(5),
            Value::float(75.0),
            Value::text("Alice"),
        ]);

        // 5 BETWEEN 1 AND 10
        let mut expr = BetweenExpr::new("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));

        // 5 BETWEEN 1 AND 4 (out of range)
        let mut expr = BetweenExpr::new("id", Value::integer(1), Value::integer(4));
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_inclusive_bounds() {
        let schema = test_schema();
        let row = Row::from_values(vec![Value::integer(1), Value::float(0.0), Value::text("a")]);

        // 1 BETWEEN 1 AND 10 (inclusive, on lower bound)
        let mut expr = BetweenExpr::new("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&row).unwrap());

        let row = Row::from_values(vec![
            Value::integer(10),
            Value::float(0.0),
            Value::text("a"),
        ]);

        // 10 BETWEEN 1 AND 10 (inclusive, on upper bound)
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_exclusive_bounds() {
        let schema = test_schema();
        let row = Row::from_values(vec![Value::integer(1), Value::float(0.0), Value::text("a")]);

        // 1 BETWEEN 1 AND 10 (exclusive - should fail)
        let mut expr =
            BetweenExpr::with_inclusivity("id", Value::integer(1), Value::integer(10), false);
        expr.prepare_for_schema(&schema);
        assert!(!expr.evaluate(&row).unwrap());

        let row = Row::from_values(vec![Value::integer(5), Value::float(0.0), Value::text("a")]);
        // 5 BETWEEN 1 AND 10 (exclusive - should pass)
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_float_between() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::float(75.0),
            Value::text("Alice"),
        ]);

        // 75.0 BETWEEN 0.0 AND 100.0
        let mut expr = BetweenExpr::new("score", Value::float(0.0), Value::float(100.0));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_string_between() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::float(0.0),
            Value::text("Bob"),
        ]);

        // "Bob" BETWEEN "Alice" AND "Charlie"
        let mut expr = BetweenExpr::new("name", Value::text("Alice"), Value::text("Charlie"));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());

        let row = Row::from_values(vec![
            Value::integer(1),
            Value::float(0.0),
            Value::text("Zack"),
        ]);

        // "Zack" BETWEEN "Alice" AND "Charlie" (out of range)
        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_null_in_between() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::null(DataType::Integer),
            Value::float(0.0),
            Value::text("Alice"),
        ]);

        // NULL BETWEEN 1 AND 10 is always false
        let mut expr = BetweenExpr::new("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_unprepared() {
        let row = Row::from_values(vec![Value::integer(5)]);
        let expr = BetweenExpr::new("id", Value::integer(1), Value::integer(10));

        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));
    }

    #[test]
    fn test_with_aliases() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(5),
            Value::float(0.0),
            Value::text("Alice"),
        ]);

        let mut aliases = HashMap::new();
        aliases.insert("i".to_string(), "id".to_string());

        let expr = BetweenExpr::new("i", Value::integer(1), Value::integer(10));
        let mut aliased = expr.with_aliases(&aliases);
        aliased.prepare_for_schema(&schema);

        assert!(aliased.evaluate(&row).unwrap());
    }
}
