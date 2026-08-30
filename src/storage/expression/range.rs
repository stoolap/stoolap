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

//! Range expression for Stoolap
//!

use chrono::{DateTime, Utc};
use rustc_hash::FxHashMap;

use super::{find_column_index, resolve_alias, Expression};
use crate::core::{DataType, Result, Row, Schema, Value};

/// Pre-computed bounds for different types
#[derive(Debug, Clone)]
pub struct RangeBounds {
    /// Per-bound types: integer cells compare each bound in the bound's
    /// own type, exactly (converting either side through f64 rounds
    /// above 2^53)
    pub min_is_float: bool,
    pub max_is_float: bool,

    /// Integer bounds
    pub int_min: i64,
    pub int_max: i64,

    /// Float bounds
    pub float_min: f64,
    pub float_max: f64,

    /// String bounds
    pub string_min: String,
    pub string_max: String,

    /// Timestamp bounds
    pub time_min: Option<DateTime<Utc>>,
    pub time_max: Option<DateTime<Utc>>,

    /// Detected data type
    pub data_type: DataType,
}

impl Default for RangeBounds {
    fn default() -> Self {
        Self {
            min_is_float: false,
            max_is_float: false,
            int_min: 0,
            int_max: 0,
            float_min: 0.0,
            float_max: 0.0,
            string_min: String::new(),
            string_max: String::new(),
            time_min: None,
            time_max: None,
            data_type: DataType::Null,
        }
    }
}

/// Range expression for custom inclusivity patterns
///
/// This is used for optimizing patterns like:
/// "column > min AND column <= max" into a single expression
#[derive(Debug, Clone)]
pub struct RangeExpr {
    /// Column name
    column: String,
    /// Minimum (lower) bound
    min_value: Value,
    /// Maximum (upper) bound
    max_value: Value,
    /// Whether to include the minimum bound (>= vs >)
    include_min: bool,
    /// Whether to include the maximum bound (<= vs <)
    include_max: bool,

    /// Pre-computed bounds for fast evaluation
    bounds: RangeBounds,

    /// Pre-computed column index
    col_index: Option<usize>,

    /// Column aliases
    aliases: FxHashMap<String, String>,
    /// Original column name if using alias
    original_column: Option<String>,
}

impl RangeExpr {
    /// Create a new range expression with custom inclusivity flags
    pub fn new(
        column: impl Into<String>,
        min_value: Value,
        max_value: Value,
        include_min: bool,
        include_max: bool,
    ) -> Self {
        let mut expr = Self {
            column: column.into(),
            min_value,
            max_value,
            include_min,
            include_max,
            bounds: RangeBounds::default(),
            col_index: None,
            aliases: FxHashMap::default(),
            original_column: None,
        };
        expr.compute_typed_bounds();
        expr
    }

    /// Create an inclusive range (>= min AND <= max)
    pub fn inclusive(column: impl Into<String>, min_value: Value, max_value: Value) -> Self {
        Self::new(column, min_value, max_value, true, true)
    }

    /// Create an exclusive range (> min AND < max)
    pub fn exclusive(column: impl Into<String>, min_value: Value, max_value: Value) -> Self {
        Self::new(column, min_value, max_value, false, false)
    }

    /// Create a half-open range (>= min AND < max)
    pub fn half_open(column: impl Into<String>, min_value: Value, max_value: Value) -> Self {
        Self::new(column, min_value, max_value, true, false)
    }

    /// Get whether min is included
    pub fn includes_min(&self) -> bool {
        self.include_min
    }

    /// Get whether max is included
    pub fn includes_max(&self) -> bool {
        self.include_max
    }

    /// Pre-compute type-specific bounds for faster evaluation
    fn compute_typed_bounds(&mut self) {
        // Process minimum bound
        match &self.min_value {
            Value::Integer(v) => {
                self.bounds.int_min = *v;
                self.bounds.float_min = *v as f64;
                self.bounds.data_type = DataType::Integer;
            }
            Value::Float(v) => {
                let f = *v;
                self.bounds.float_min = f;
                self.bounds.int_min = f as i64;
                self.bounds.min_is_float = true;
                self.bounds.data_type = DataType::Float;
            }
            Value::Text(v) => {
                self.bounds.string_min = v.to_string();
                self.bounds.data_type = DataType::Text;

                // Try to convert to number
                if let Ok(int_val) = v.parse::<i64>() {
                    self.bounds.int_min = int_val;
                    self.bounds.float_min = int_val as f64;
                } else if let Ok(float_val) = v.parse::<f64>() {
                    self.bounds.float_min = float_val;
                    self.bounds.int_min = float_val as i64;
                }

                // Try to parse as timestamp
                if let Ok(ts) = v.parse::<DateTime<Utc>>() {
                    self.bounds.time_min = Some(ts);
                    self.bounds.data_type = DataType::Timestamp;
                }
            }
            Value::Timestamp(v) => {
                self.bounds.time_min = Some(*v);
                self.bounds.data_type = DataType::Timestamp;
            }
            _ => {}
        }

        // Process maximum bound
        match &self.max_value {
            Value::Integer(v) => {
                self.bounds.int_max = *v;
                self.bounds.float_max = *v as f64;
                if self.bounds.data_type == DataType::Null {
                    self.bounds.data_type = DataType::Integer;
                }
            }
            Value::Float(v) => {
                let f = *v;
                self.bounds.float_max = f;
                self.bounds.int_max = f as i64;
                self.bounds.max_is_float = true;
                if self.bounds.data_type == DataType::Null {
                    self.bounds.data_type = DataType::Float;
                }
            }
            Value::Text(v) => {
                self.bounds.string_max = v.to_string();
                if self.bounds.data_type == DataType::Null {
                    self.bounds.data_type = DataType::Text;
                }

                // Try to convert to number
                if let Ok(int_val) = v.parse::<i64>() {
                    self.bounds.int_max = int_val;
                    self.bounds.float_max = int_val as f64;
                } else if let Ok(float_val) = v.parse::<f64>() {
                    self.bounds.float_max = float_val;
                    self.bounds.int_max = float_val as i64;
                }

                // Try to parse as timestamp
                if let Ok(ts) = v.parse::<DateTime<Utc>>() {
                    self.bounds.time_max = Some(ts);
                    if self.bounds.data_type == DataType::Null {
                        self.bounds.data_type = DataType::Timestamp;
                    }
                }
            }
            Value::Timestamp(v) => {
                self.bounds.time_max = Some(*v);
                if self.bounds.data_type == DataType::Null {
                    self.bounds.data_type = DataType::Timestamp;
                }
            }
            _ => {}
        }
    }

    /// Check integer bounds, each bound in its own type: a float bound
    /// must not truncate (5.1..5.9 is not [5, 5]) and an integer bound
    /// must not round through f64 (Integer(i64::MAX) is not 2^63).
    #[inline]
    fn check_integer(&self, val: i64) -> bool {
        use crate::core::value::{i64_ge_f64, i64_gt_f64, i64_le_f64, i64_lt_f64};

        let lower_ok = if self.bounds.min_is_float {
            if self.include_min {
                i64_ge_f64(val, self.bounds.float_min)
            } else {
                i64_gt_f64(val, self.bounds.float_min)
            }
        } else if self.include_min {
            val >= self.bounds.int_min
        } else {
            val > self.bounds.int_min
        };
        if !lower_ok {
            return false;
        }

        if self.bounds.max_is_float {
            if self.include_max {
                i64_le_f64(val, self.bounds.float_max)
            } else {
                i64_lt_f64(val, self.bounds.float_max)
            }
        } else if self.include_max {
            val <= self.bounds.int_max
        } else {
            val < self.bounds.int_max
        }
    }

    /// Check float bounds, each bound in its own type
    #[inline]
    fn check_float(&self, val: f64) -> bool {
        use crate::core::value::{i64_ge_f64, i64_gt_f64, i64_le_f64, i64_lt_f64};

        let lower_ok = if self.bounds.min_is_float {
            if self.include_min {
                val >= self.bounds.float_min
            } else {
                val > self.bounds.float_min
            }
        } else if self.include_min {
            i64_le_f64(self.bounds.int_min, val)
        } else {
            i64_lt_f64(self.bounds.int_min, val)
        };
        if !lower_ok {
            return false;
        }

        if self.bounds.max_is_float {
            if self.include_max {
                val <= self.bounds.float_max
            } else {
                val < self.bounds.float_max
            }
        } else if self.include_max {
            i64_ge_f64(self.bounds.int_max, val)
        } else {
            i64_gt_f64(self.bounds.int_max, val)
        }
    }

    /// Check string bounds
    #[inline]
    fn check_string(&self, val: &str) -> bool {
        // Check minimum
        if self.include_min {
            if val < self.bounds.string_min.as_str() {
                return false;
            }
        } else if val <= self.bounds.string_min.as_str() {
            return false;
        }

        // Check maximum
        if self.include_max {
            if val > self.bounds.string_max.as_str() {
                return false;
            }
        } else if val >= self.bounds.string_max.as_str() {
            return false;
        }

        true
    }

    /// Check timestamp bounds
    #[inline]
    fn check_timestamp(&self, val: DateTime<Utc>) -> bool {
        // Check minimum
        if let Some(min) = self.bounds.time_min {
            if self.include_min {
                if val < min {
                    return false;
                }
            } else if val <= min {
                return false;
            }
        }

        // Check maximum
        if let Some(max) = self.bounds.time_max {
            if self.include_max {
                if val > max {
                    return false;
                }
            } else if val >= max {
                return false;
            }
        }

        true
    }
}

impl Expression for RangeExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return Ok(false),
        };

        let col_value = &row[col_idx];

        // NULL in range check is always false
        if col_value.is_null() {
            return Ok(false);
        }

        let result = match col_value {
            Value::Integer(val) => self.check_integer(*val),
            Value::Float(val) => self.check_float(*val),
            Value::Text(val) => self.check_string(val),
            Value::Timestamp(val) => self.check_timestamp(*val),
            _ => false,
        };

        Ok(result)
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return false,
        };

        let col_value = &row[col_idx];

        if col_value.is_null() {
            return false;
        }

        match col_value {
            Value::Integer(val) => self.check_integer(*val),
            Value::Float(val) => self.check_float(*val),
            Value::Text(val) => self.check_string(val),
            Value::Timestamp(val) => self.check_timestamp(*val),
            _ => false,
        }
    }

    fn with_aliases(&self, aliases: &FxHashMap<String, String>) -> Box<dyn Expression> {
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

    fn collect_column_indices(&self, out: &mut Vec<usize>) -> bool {
        if let Some(idx) = self.col_index {
            out.push(idx);
            true
        } else {
            false
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SchemaBuilder;

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("score", DataType::Float)
            .add("name", DataType::Text)
            .build()
    }

    #[test]
    fn test_inclusive_range() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(5),
            Value::float(75.0),
            Value::text("Alice"),
        ]);

        // 5 >= 1 AND 5 <= 10 (inclusive)
        let mut expr = RangeExpr::inclusive("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_inclusive_on_boundary() {
        let schema = test_schema();

        // Test on lower boundary
        let row = Row::from_values(vec![Value::integer(1), Value::float(0.0), Value::text("a")]);

        let mut expr = RangeExpr::inclusive("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&row).unwrap());

        // Test on upper boundary
        let row = Row::from_values(vec![
            Value::integer(10),
            Value::float(0.0),
            Value::text("a"),
        ]);
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_exclusive_range() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(5),
            Value::float(75.0),
            Value::text("Alice"),
        ]);

        // 5 > 1 AND 5 < 10 (exclusive)
        let mut expr = RangeExpr::exclusive("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_exclusive_on_boundary() {
        let schema = test_schema();

        // Test on lower boundary (should fail)
        let row = Row::from_values(vec![Value::integer(1), Value::float(0.0), Value::text("a")]);

        let mut expr = RangeExpr::exclusive("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);
        assert!(!expr.evaluate(&row).unwrap());

        // Test on upper boundary (should fail)
        let row = Row::from_values(vec![
            Value::integer(10),
            Value::float(0.0),
            Value::text("a"),
        ]);
        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_half_open_range() {
        let schema = test_schema();

        // >= 1 AND < 10 (half-open)
        let mut expr = RangeExpr::half_open("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);

        // On lower boundary (should pass)
        let row = Row::from_values(vec![Value::integer(1), Value::float(0.0), Value::text("a")]);
        assert!(expr.evaluate(&row).unwrap());

        // On upper boundary (should fail)
        let row = Row::from_values(vec![
            Value::integer(10),
            Value::float(0.0),
            Value::text("a"),
        ]);
        assert!(!expr.evaluate(&row).unwrap());

        // Inside range (should pass)
        let row = Row::from_values(vec![Value::integer(5), Value::float(0.0), Value::text("a")]);
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_out_of_range() {
        let schema = test_schema();

        let mut expr = RangeExpr::inclusive("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);

        // Below range
        let row = Row::from_values(vec![Value::integer(0), Value::float(0.0), Value::text("a")]);
        assert!(!expr.evaluate(&row).unwrap());

        // Above range
        let row = Row::from_values(vec![
            Value::integer(11),
            Value::float(0.0),
            Value::text("a"),
        ]);
        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_float_range() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::float(75.0),
            Value::text("Alice"),
        ]);

        // 75.0 >= 0.0 AND 75.0 <= 100.0
        let mut expr = RangeExpr::inclusive("score", Value::float(0.0), Value::float(100.0));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_string_range() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::float(0.0),
            Value::text("Bob"),
        ]);

        // "Bob" >= "Alice" AND "Bob" <= "Charlie"
        let mut expr = RangeExpr::inclusive("name", Value::text("Alice"), Value::text("Charlie"));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_null_in_range() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::null(DataType::Integer),
            Value::float(0.0),
            Value::text("Alice"),
        ]);

        let mut expr = RangeExpr::inclusive("id", Value::integer(1), Value::integer(10));
        expr.prepare_for_schema(&schema);

        // NULL in range check is always false
        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_unprepared() {
        let row = Row::from_values(vec![Value::integer(5)]);
        let expr = RangeExpr::inclusive("id", Value::integer(1), Value::integer(10));

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

        let mut aliases = FxHashMap::default();
        aliases.insert("i".to_string(), "id".to_string());

        let expr = RangeExpr::inclusive("i", Value::integer(1), Value::integer(10));
        let mut aliased = expr.with_aliases(&aliases);
        aliased.prepare_for_schema(&schema);

        assert!(aliased.evaluate(&row).unwrap());
    }

    #[test]
    fn test_custom_inclusivity() {
        let schema = test_schema();
        let row = Row::from_values(vec![Value::integer(5), Value::float(0.0), Value::text("a")]);

        // > 1 AND <= 10 (custom: min exclusive, max inclusive)
        let mut expr = RangeExpr::new(
            "id",
            Value::integer(1),
            Value::integer(10),
            false, // exclude min
            true,  // include max
        );
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(!expr.includes_min());
        assert!(expr.includes_max());
    }

    #[test]
    fn test_can_use_index() {
        let expr = RangeExpr::inclusive("id", Value::integer(1), Value::integer(10));
        assert!(expr.can_use_index());
    }

    #[test]
    fn test_get_column_name() {
        let expr = RangeExpr::inclusive("id", Value::integer(1), Value::integer(10));
        assert_eq!(expr.get_column_name(), Some("id"));
    }

    #[test]
    fn mixed_bounds_are_exact_at_extremes() {
        let schema = test_schema();

        // [Integer(i64::MAX), Float(2^63)]: i64::MAX - 1 is below the
        // integer lower bound; rounding it through f64 would accept it
        let mut expr = RangeExpr::inclusive(
            "id",
            Value::integer(i64::MAX),
            Value::float(9_223_372_036_854_775_808.0),
        );
        expr.prepare_for_schema(&schema);

        let below = Row::from_values(vec![
            Value::integer(i64::MAX - 1),
            Value::float(0.0),
            Value::text("x"),
        ]);
        assert!(!expr.evaluate(&below).unwrap());
        assert!(!expr.evaluate_fast(&below));

        let at_max = Row::from_values(vec![
            Value::integer(i64::MAX),
            Value::float(0.0),
            Value::text("x"),
        ]);
        assert!(expr.evaluate(&at_max).unwrap());
        assert!(expr.evaluate_fast(&at_max));

        // Float cell 2^63 against an Integer(i64::MAX) upper bound:
        // above the bound, must reject
        let mut expr = RangeExpr::inclusive("score", Value::integer(0), Value::integer(i64::MAX));
        expr.prepare_for_schema(&schema);
        let high = Row::from_values(vec![
            Value::integer(1),
            Value::float(9_223_372_036_854_775_808.0),
            Value::text("x"),
        ]);
        assert!(!expr.evaluate(&high).unwrap());
        assert!(!expr.evaluate_fast(&high));

        // Fractional float bounds stay empty for integers
        let mut expr = RangeExpr::inclusive("id", Value::float(5.1), Value::float(5.9));
        expr.prepare_for_schema(&schema);
        let five = Row::from_values(vec![Value::integer(5), Value::float(0.0), Value::text("x")]);
        assert!(!expr.evaluate(&five).unwrap());
        assert!(!expr.evaluate_fast(&five));
    }
}
