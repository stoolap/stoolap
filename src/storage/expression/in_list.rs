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

//! IN list expression for Stoolap
//!
//!
//! ## Optimization: HashSet for O(1) Lookup
//!
//! Instead of linear search through the IN list for each row (O(n)),
//! we pre-compute HashSets at prepare time for O(1) lookup.
//! This is the same optimization PostgreSQL uses for "hashed IN lists".

use std::any::Any;
use std::collections::HashMap;

use rustc_hash::FxHashSet;

use super::{find_column_index, resolve_alias, Expression};
use crate::core::{Result, Row, Schema, Value};

/// Pre-computed hash sets for O(1) IN list lookup
#[derive(Debug, Clone)]
enum HashedValues {
    /// Not yet computed
    None,
    /// Integer hash set for O(1) lookup
    Integers(FxHashSet<i64>),
    /// String hash set for O(1) lookup
    Strings(FxHashSet<String>),
    /// Boolean set (only 2 possible values)
    Booleans { has_true: bool, has_false: bool },
    /// Mixed types - fall back to linear search
    Mixed,
}

/// IN list expression (column IN (v1, v2, ...))
///
/// ## Performance
///
/// Uses O(1) HashSet lookup instead of O(n) linear search for each row.
/// Hash sets are pre-computed at `prepare_for_schema` time.
#[derive(Debug, Clone)]
pub struct InListExpr {
    /// Column name
    column: String,
    /// List of values to check against
    values: Vec<Value>,
    /// True for NOT IN
    not: bool,

    /// Pre-computed column index
    col_index: Option<usize>,

    /// Pre-computed hash sets for O(1) lookup
    hashed: HashedValues,

    /// Pre-computed: whether the values list contains NULL
    /// Used for SQL three-valued logic: x NOT IN (a, NULL) returns UNKNOWN if x != a
    has_null: bool,

    /// Column aliases
    aliases: HashMap<String, String>,
    /// Original column name if using alias
    original_column: Option<String>,
}

impl InListExpr {
    /// Create a new IN expression
    pub fn new(column: impl Into<String>, values: Vec<Value>) -> Self {
        let has_null = values.iter().any(|v| v.is_null());
        Self {
            column: column.into(),
            values,
            not: false,
            col_index: None,
            hashed: HashedValues::None,
            has_null,
            aliases: HashMap::new(),
            original_column: None,
        }
    }

    /// Create a NOT IN expression
    pub fn not_in(column: impl Into<String>, values: Vec<Value>) -> Self {
        let has_null = values.iter().any(|v| v.is_null());
        Self {
            column: column.into(),
            values,
            not: true,
            col_index: None,
            hashed: HashedValues::None,
            has_null,
            aliases: HashMap::new(),
            original_column: None,
        }
    }

    /// Check if this is a NOT IN expression
    pub fn is_not(&self) -> bool {
        self.not
    }

    /// Get the values
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Get the values (alias for values(), for expression compilation)
    pub fn get_values(&self) -> &[Value] {
        &self.values
    }

    /// Build hash sets for O(1) lookup
    fn build_hash_sets(&mut self) {
        if self.values.is_empty() {
            self.hashed = HashedValues::None;
            return;
        }

        // Detect the type from the first non-null value
        let first_type = self.values.iter().find_map(|v| match v {
            Value::Integer(_) => Some("int"),
            Value::Float(_) => Some("float"),
            Value::Text(_) => Some("text"),
            Value::Boolean(_) => Some("bool"),
            Value::Null(_) => None,
            _ => Some("other"),
        });

        match first_type {
            Some("int") => {
                // Check if all values are integers (or convertible floats)
                let mut set = FxHashSet::default();
                let mut all_int = true;
                for v in &self.values {
                    match v {
                        Value::Integer(i) => {
                            set.insert(*i);
                        }
                        Value::Float(f) => {
                            // Only include if it's a whole number
                            if f.fract() == 0.0 {
                                set.insert(*f as i64);
                            } else {
                                all_int = false;
                                break;
                            }
                        }
                        Value::Null(_) => {} // Skip nulls
                        _ => {
                            all_int = false;
                            break;
                        }
                    }
                }
                if all_int {
                    self.hashed = HashedValues::Integers(set);
                } else {
                    self.hashed = HashedValues::Mixed;
                }
            }
            Some("text") => {
                let mut set = FxHashSet::default();
                let mut all_text = true;
                for v in &self.values {
                    match v {
                        Value::Text(s) => {
                            set.insert(s.to_string());
                        }
                        Value::Null(_) => {}
                        _ => {
                            all_text = false;
                            break;
                        }
                    }
                }
                if all_text {
                    self.hashed = HashedValues::Strings(set);
                } else {
                    self.hashed = HashedValues::Mixed;
                }
            }
            Some("bool") => {
                let mut has_true = false;
                let mut has_false = false;
                for v in &self.values {
                    match v {
                        Value::Boolean(true) => has_true = true,
                        Value::Boolean(false) => has_false = true,
                        Value::Null(_) => {}
                        _ => {}
                    }
                }
                self.hashed = HashedValues::Booleans {
                    has_true,
                    has_false,
                };
            }
            _ => {
                self.hashed = HashedValues::Mixed;
            }
        }
    }

    /// Check if integer is in list - O(1) with hash set, O(n) fallback
    #[inline]
    fn check_integer(&self, val: i64) -> bool {
        match &self.hashed {
            HashedValues::Integers(set) => set.contains(&val),
            _ => {
                // Fallback to linear search
                for v in &self.values {
                    if let Some(list_val) = v.as_int64() {
                        if val == list_val {
                            return true;
                        }
                    } else if let Some(list_val) = v.as_float64() {
                        if val as f64 == list_val {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    /// Check if float is in list
    #[inline]
    fn check_float(&self, val: f64) -> bool {
        // Floats use linear search due to precision issues with hashing
        for v in &self.values {
            if let Some(list_val) = v.as_float64() {
                if val == list_val {
                    return true;
                }
            } else if let Some(list_val) = v.as_int64() {
                if val == list_val as f64 {
                    return true;
                }
            }
        }
        false
    }

    /// Check if string is in list - O(1) with hash set, O(n) fallback
    #[inline]
    fn check_string(&self, val: &str) -> bool {
        match &self.hashed {
            HashedValues::Strings(set) => set.contains(val),
            _ => {
                // Fallback to linear search
                for v in &self.values {
                    if let Some(list_val) = v.as_string() {
                        if val == list_val {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    /// Check if boolean is in list - O(1)
    #[inline]
    fn check_boolean(&self, val: bool) -> bool {
        match &self.hashed {
            HashedValues::Booleans {
                has_true,
                has_false,
            } => {
                if val {
                    *has_true
                } else {
                    *has_false
                }
            }
            _ => {
                // Fallback
                self.values.iter().any(|v| v.as_boolean() == Some(val))
            }
        }
    }
}

impl Expression for InListExpr {
    /// Evaluate IN/NOT IN expression with proper SQL NULL semantics
    ///
    /// SQL Standard three-valued logic:
    /// - `x IN (a, b, NULL)`: TRUE if x matches, UNKNOWN (treated as false for filtering) if not
    /// - `x NOT IN (a, b, NULL)`: FALSE if x matches, UNKNOWN (treated as false for filtering) if not
    ///
    /// For storage-layer filtering, UNKNOWN is treated as false (don't return the row)
    fn evaluate(&self, row: &Row) -> Result<bool> {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return Ok(false),
        };

        let col_value = &row[col_idx];

        // NULL IN (...) is UNKNOWN (false for filtering)
        // NULL NOT IN (...) is UNKNOWN (false for filtering)
        if col_value.is_null() {
            return Ok(false);
        }

        // O(1) lookup using pre-computed hash sets!
        let found = match col_value {
            Value::Integer(val) => self.check_integer(*val),
            Value::Float(val) => self.check_float(*val),
            Value::Text(val) => self.check_string(val),
            Value::Boolean(val) => self.check_boolean(*val),
            _ => false,
        };

        if found {
            // Found a match
            Ok(!self.not) // IN returns true, NOT IN returns false
        } else if self.has_null {
            // No match found, but list contains NULL (pre-computed)
            // Result is UNKNOWN, which for filtering purposes means false
            Ok(false)
        } else {
            // No match, no NULL in list - definitive answer
            Ok(self.not) // IN returns false, NOT IN returns true
        }
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return false, // Unknown column, return false for safety
        };

        let col_value = &row[col_idx];

        // NULL comparisons result in UNKNOWN (false for filtering)
        if col_value.is_null() {
            return false;
        }

        // O(1) lookup using pre-computed hash sets!
        let found = match col_value {
            Value::Integer(val) => self.check_integer(*val),
            Value::Float(val) => self.check_float(*val),
            Value::Text(val) => self.check_string(val),
            Value::Boolean(val) => self.check_boolean(*val),
            _ => false,
        };

        if found {
            !self.not // IN returns true, NOT IN returns false
        } else if self.has_null {
            // No match but list has NULL (pre-computed) - result is UNKNOWN (false for filtering)
            false
        } else {
            self.not // IN returns false, NOT IN returns true
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
        expr.hashed = HashedValues::None; // Reset hash sets
        Box::new(expr)
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        if self.col_index.is_some() {
            return;
        }
        self.col_index = find_column_index(schema, &self.column);
        // Build hash sets for O(1) lookup
        self.build_hash_sets();
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
            .add("name", DataType::Text)
            .add("status", DataType::Text)
            .build()
    }

    #[test]
    fn test_integer_in() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(2),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // 2 IN (1, 2, 3)
        let mut expr = InListExpr::new(
            "id",
            vec![Value::integer(1), Value::integer(2), Value::integer(3)],
        );
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));

        // 2 IN (5, 6, 7)
        let mut expr = InListExpr::new(
            "id",
            vec![Value::integer(5), Value::integer(6), Value::integer(7)],
        );
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_string_in() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // "active" IN ("active", "inactive", "pending")
        let mut expr = InListExpr::new(
            "status",
            vec![
                Value::text("active"),
                Value::text("inactive"),
                Value::text("pending"),
            ],
        );
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_not_in() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(4),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // 4 NOT IN (1, 2, 3)
        let mut expr = InListExpr::not_in(
            "id",
            vec![Value::integer(1), Value::integer(2), Value::integer(3)],
        );
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());

        // 4 NOT IN (4, 5, 6)
        let mut expr = InListExpr::not_in(
            "id",
            vec![Value::integer(4), Value::integer(5), Value::integer(6)],
        );
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_null_in() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::null(DataType::Integer),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // NULL IN (1, 2, 3) is false
        let mut expr = InListExpr::new(
            "id",
            vec![Value::integer(1), Value::integer(2), Value::integer(3)],
        );
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());

        // NULL NOT IN (1, 2, 3) is also false
        let mut expr = InListExpr::not_in(
            "id",
            vec![Value::integer(1), Value::integer(2), Value::integer(3)],
        );
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_empty_list() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // 1 IN () is always false
        let mut expr = InListExpr::new("id", vec![]);
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());

        // 1 NOT IN () is always true
        let mut expr = InListExpr::not_in("id", vec![]);
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_mixed_numeric_types() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(2),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // 2 IN (1.0, 2.0, 3.0) - mixed int/float
        let mut expr = InListExpr::new(
            "id",
            vec![Value::float(1.0), Value::float(2.0), Value::float(3.0)],
        );
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_with_aliases() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        let mut aliases = HashMap::new();
        aliases.insert("i".to_string(), "id".to_string());

        let expr = InListExpr::new("i", vec![Value::integer(1), Value::integer(2)]);
        let mut aliased = expr.with_aliases(&aliases);
        aliased.prepare_for_schema(&schema);

        assert!(aliased.evaluate(&row).unwrap());
    }

    #[test]
    fn test_is_not() {
        let expr = InListExpr::new("id", vec![Value::integer(1)]);
        assert!(!expr.is_not());

        let expr = InListExpr::not_in("id", vec![Value::integer(1)]);
        assert!(expr.is_not());
    }

    #[test]
    fn test_not_in_with_null_in_list() {
        // SQL Standard: x NOT IN (a, NULL) returns UNKNOWN if x != a
        // For storage-layer filtering, UNKNOWN is treated as false
        let schema = test_schema();

        // id = 1, check NOT IN (2, NULL)
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // 1 NOT IN (2, NULL) - 1 != 2 is TRUE, but 1 != NULL is UNKNOWN
        // TRUE AND UNKNOWN = UNKNOWN, so for filtering this returns false
        let mut expr = InListExpr::not_in(
            "id",
            vec![Value::integer(2), Value::null(DataType::Integer)],
        );
        expr.prepare_for_schema(&schema);

        // For storage-layer filtering, UNKNOWN means false (exclude row)
        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));

        // 2 NOT IN (2, NULL) - 2 == 2 means FALSE (value is in list)
        let row2 = Row::from_values(vec![
            Value::integer(2),
            Value::text("Bob"),
            Value::text("active"),
        ]);
        assert!(!expr.evaluate(&row2).unwrap());
        assert!(!expr.evaluate_fast(&row2));
    }

    #[test]
    fn test_in_with_null_in_list() {
        // SQL Standard: x IN (a, NULL) returns TRUE if x = a, UNKNOWN otherwise
        let schema = test_schema();

        // id = 1, check IN (2, NULL)
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // 1 IN (2, NULL) - 1 != 2 and 1 = NULL is UNKNOWN
        // FALSE OR UNKNOWN = UNKNOWN
        let mut expr = InListExpr::new(
            "id",
            vec![Value::integer(2), Value::null(DataType::Integer)],
        );
        expr.prepare_for_schema(&schema);

        // For storage-layer filtering, UNKNOWN means false
        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));

        // 2 IN (2, NULL) - 2 == 2 means TRUE (value is in list)
        let row2 = Row::from_values(vec![
            Value::integer(2),
            Value::text("Bob"),
            Value::text("active"),
        ]);
        assert!(expr.evaluate(&row2).unwrap());
        assert!(expr.evaluate_fast(&row2));
    }

    #[test]
    fn test_not_in_without_null() {
        // Without NULL in list, NOT IN works normally
        let schema = test_schema();

        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::text("active"),
        ]);

        // 1 NOT IN (2, 3) - 1 != 2 AND 1 != 3 = TRUE
        let mut expr = InListExpr::not_in("id", vec![Value::integer(2), Value::integer(3)]);
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }
}
