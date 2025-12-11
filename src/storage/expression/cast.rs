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

//! CAST expression for Stoolap
//!

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};

use super::{find_column_index, resolve_alias, Expression};
use crate::core::{DataType, Operator, Result, Row, Schema, Value};

/// CAST expression (CAST(column AS type))
///

#[derive(Debug, Clone)]
pub struct CastExpr {
    /// Column name to cast
    column: String,
    /// Target data type
    target_type: DataType,

    /// Pre-computed column index
    col_index: Option<usize>,

    /// Column aliases
    aliases: HashMap<String, String>,
    /// Original column name if using alias
    original_column: Option<String>,
}

impl CastExpr {
    /// Create a new CAST expression
    pub fn new(column: impl Into<String>, target_type: DataType) -> Self {
        Self {
            column: column.into(),
            target_type,
            col_index: None,
            aliases: HashMap::new(),
            original_column: None,
        }
    }

    /// Get the target type
    pub fn target_type(&self) -> DataType {
        self.target_type
    }

    /// Perform the cast operation on a value
    pub fn perform_cast(&self, value: &Value) -> Result<Value> {
        if value.is_null() {
            return Ok(Value::null(self.target_type));
        }

        match self.target_type {
            DataType::Integer => cast_to_integer(value),
            DataType::Float => cast_to_float(value),
            DataType::Text => cast_to_string(value),
            DataType::Boolean => cast_to_boolean(value),
            DataType::Timestamp => cast_to_timestamp(value),
            DataType::Json => cast_to_json(value),
            DataType::Null => Ok(Value::null(DataType::Null)),
        }
    }
}

impl Expression for CastExpr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn evaluate(&self, row: &Row) -> Result<bool> {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return Ok(false),
        };

        let col_value = &row[col_idx];

        if col_value.is_null() {
            return Ok(false);
        }

        // Perform the cast - if it succeeds, return true
        // (CAST by itself doesn't filter, parent expression handles comparison)
        match self.perform_cast(col_value) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
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

        self.perform_cast(col_value).is_ok()
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

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }
}

/// Compound expression for CAST with comparison
///
/// This handles WHERE clauses like: WHERE CAST(column AS INTEGER) > 100
#[derive(Debug, Clone)]
pub struct CompoundExpr {
    /// The CAST expression
    cast_expr: CastExpr,
    /// The comparison operator
    operator: Operator,
    /// The value to compare against
    value: Value,

    /// Whether prepared for schema
    is_optimized: bool,
}

impl CompoundExpr {
    /// Create a new compound expression
    pub fn new(cast_expr: CastExpr, operator: Operator, value: Value) -> Self {
        Self {
            cast_expr,
            operator,
            value,
            is_optimized: false,
        }
    }

    /// Get the operator
    pub fn operator(&self) -> Operator {
        self.operator
    }

    /// Get the comparison value
    pub fn comparison_value(&self) -> &Value {
        &self.value
    }
}

impl Expression for CompoundExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        let col_idx = match self.cast_expr.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return Ok(false),
        };

        let col_value = &row[col_idx];

        if col_value.is_null() {
            return Ok(false);
        }

        // Cast the column value
        let casted = self.cast_expr.perform_cast(col_value)?;

        // Convert comparison value to target type if needed
        let comp_value = self.cast_expr.perform_cast(&self.value)?;

        // Compare the values
        let cmp = compare_values(&casted, &comp_value);

        let result = match self.operator {
            Operator::Eq => cmp == 0,
            Operator::Ne => cmp != 0,
            Operator::Gt => cmp > 0,
            Operator::Gte => cmp >= 0,
            Operator::Lt => cmp < 0,
            Operator::Lte => cmp <= 0,
            _ => false,
        };

        Ok(result)
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        let col_idx = match self.cast_expr.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return false,
        };

        let col_value = &row[col_idx];

        if col_value.is_null() {
            return false;
        }

        // Fast path based on target type
        match self.cast_expr.target_type {
            DataType::Integer => {
                let col_int = match col_value {
                    Value::Integer(v) => *v,
                    Value::Float(v) => *v as i64,
                    Value::Boolean(b) => {
                        if *b {
                            1
                        } else {
                            0
                        }
                    }
                    Value::Text(s) => {
                        if let Ok(i) = s.parse::<i64>() {
                            i
                        } else if let Ok(f) = s.parse::<f64>() {
                            f as i64
                        } else {
                            return false;
                        }
                    }
                    _ => return false,
                };

                let comp_int = match &self.value {
                    Value::Integer(v) => *v,
                    Value::Float(v) => *v as i64,
                    _ => return false,
                };

                match self.operator {
                    Operator::Eq => col_int == comp_int,
                    Operator::Ne => col_int != comp_int,
                    Operator::Gt => col_int > comp_int,
                    Operator::Gte => col_int >= comp_int,
                    Operator::Lt => col_int < comp_int,
                    Operator::Lte => col_int <= comp_int,
                    _ => false,
                }
            }
            DataType::Float => {
                let col_float = match col_value {
                    Value::Integer(v) => *v as f64,
                    Value::Float(v) => *v,
                    Value::Boolean(b) => {
                        if *b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => return false,
                };

                let comp_float = match &self.value {
                    Value::Integer(v) => *v as f64,
                    Value::Float(v) => *v,
                    _ => return false,
                };

                match self.operator {
                    Operator::Eq => col_float == comp_float,
                    Operator::Ne => col_float != comp_float,
                    Operator::Gt => col_float > comp_float,
                    Operator::Gte => col_float >= comp_float,
                    Operator::Lt => col_float < comp_float,
                    Operator::Lte => col_float <= comp_float,
                    _ => false,
                }
            }
            DataType::Text => {
                let col_str = col_value.as_string();
                let col_str = match col_str {
                    Some(s) => s,
                    None => return false,
                };

                let comp_str = match &self.value {
                    Value::Text(s) => &**s,
                    _ => return false,
                };

                match self.operator {
                    Operator::Eq => col_str == comp_str,
                    Operator::Ne => col_str != comp_str,
                    Operator::Gt => col_str.as_str() > comp_str,
                    Operator::Gte => col_str.as_str() >= comp_str,
                    Operator::Lt => col_str.as_str() < comp_str,
                    Operator::Lte => col_str.as_str() <= comp_str,
                    _ => false,
                }
            }
            DataType::Boolean => {
                let col_bool = match col_value {
                    Value::Integer(v) => *v != 0,
                    Value::Float(v) => *v != 0.0,
                    Value::Boolean(b) => *b,
                    _ => return false,
                };

                let comp_bool = match &self.value {
                    Value::Boolean(b) => *b,
                    Value::Integer(v) => *v != 0,
                    _ => return false,
                };

                match self.operator {
                    Operator::Eq => col_bool == comp_bool,
                    Operator::Ne => col_bool != comp_bool,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn with_aliases(&self, aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        let aliased_cast = self.cast_expr.with_aliases(aliases);
        let cast_expr = if let Some(cast) = aliased_cast.as_any().downcast_ref::<CastExpr>() {
            cast.clone()
        } else {
            self.cast_expr.clone()
        };

        Box::new(CompoundExpr {
            cast_expr,
            operator: self.operator,
            value: self.value.clone(),
            is_optimized: false,
        })
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        if self.is_optimized {
            return;
        }
        self.cast_expr.prepare_for_schema(schema);
        self.is_optimized = true;
    }

    fn is_prepared(&self) -> bool {
        self.is_optimized
    }

    fn get_column_name(&self) -> Option<&str> {
        self.cast_expr.get_column_name()
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Cast helper functions

fn cast_to_integer(value: &Value) -> Result<Value> {
    match value {
        Value::Integer(v) => Ok(Value::Integer(*v)),
        Value::Float(v) => Ok(Value::Integer(*v as i64)),
        Value::Text(s) => {
            if let Ok(i) = s.parse::<i64>() {
                Ok(Value::Integer(i))
            } else if let Ok(f) = s.parse::<f64>() {
                Ok(Value::Integer(f as i64))
            } else {
                Ok(Value::Integer(0))
            }
        }
        Value::Boolean(b) => Ok(Value::Integer(if *b { 1 } else { 0 })),
        Value::Timestamp(t) => Ok(Value::Integer(t.timestamp())),
        Value::Null(_) => Ok(Value::null(DataType::Integer)),
        _ => Ok(Value::Integer(0)),
    }
}

fn cast_to_float(value: &Value) -> Result<Value> {
    match value {
        Value::Integer(v) => Ok(Value::float(*v as f64)),
        Value::Float(v) => Ok(Value::float(*v)),
        Value::Text(s) => {
            if let Ok(f) = s.parse::<f64>() {
                Ok(Value::float(f))
            } else {
                Ok(Value::float(0.0))
            }
        }
        Value::Boolean(b) => Ok(Value::float(if *b { 1.0 } else { 0.0 })),
        Value::Timestamp(t) => Ok(Value::float(t.timestamp() as f64)),
        Value::Null(_) => Ok(Value::null(DataType::Float)),
        _ => Ok(Value::float(0.0)),
    }
}

fn cast_to_string(value: &Value) -> Result<Value> {
    match value {
        Value::Integer(v) => Ok(Value::Text(Arc::from(v.to_string().as_str()))),
        Value::Float(v) => Ok(Value::Text(Arc::from(v.to_string().as_str()))),
        Value::Text(s) => Ok(Value::Text(s.clone())),
        Value::Boolean(b) => Ok(Value::Text(Arc::from(if *b { "true" } else { "false" }))),
        Value::Timestamp(t) => Ok(Value::Text(Arc::from(t.to_rfc3339().as_str()))),
        Value::Json(j) => Ok(Value::Text(j.clone())),
        Value::Null(_) => Ok(Value::null(DataType::Text)),
    }
}

fn cast_to_boolean(value: &Value) -> Result<Value> {
    match value {
        Value::Integer(v) => Ok(Value::Boolean(*v != 0)),
        Value::Float(v) => Ok(Value::Boolean(*v != 0.0)),
        Value::Text(s) => {
            let lower = s.to_lowercase();
            let b = matches!(lower.as_str(), "true" | "1" | "t" | "yes" | "y");
            Ok(Value::Boolean(b))
        }
        Value::Boolean(b) => Ok(Value::Boolean(*b)),
        Value::Null(_) => Ok(Value::null(DataType::Boolean)),
        _ => Ok(Value::Boolean(false)),
    }
}

fn cast_to_timestamp(value: &Value) -> Result<Value> {
    match value {
        Value::Integer(v) => Ok(Value::Timestamp(Utc.timestamp_opt(*v, 0).unwrap())),
        Value::Float(v) => Ok(Value::Timestamp(Utc.timestamp_opt(*v as i64, 0).unwrap())),
        Value::Timestamp(t) => Ok(Value::Timestamp(*t)),
        Value::Text(s) => {
            // Try various timestamp formats
            if let Ok(ts) = s.parse::<DateTime<Utc>>() {
                Ok(Value::Timestamp(ts))
            } else {
                // Default to current time if parsing fails
                Ok(Value::Timestamp(Utc::now()))
            }
        }
        Value::Null(_) => Ok(Value::null(DataType::Timestamp)),
        _ => Ok(Value::Timestamp(Utc::now())),
    }
}

fn cast_to_json(value: &Value) -> Result<Value> {
    match value {
        Value::Json(j) => Ok(Value::Json(j.clone())),
        Value::Text(s) => Ok(Value::Json(s.clone())),
        Value::Integer(v) => Ok(Value::Json(Arc::from(v.to_string().as_str()))),
        Value::Float(v) => Ok(Value::Json(Arc::from(v.to_string().as_str()))),
        Value::Boolean(b) => Ok(Value::Json(Arc::from(if *b { "true" } else { "false" }))),
        Value::Null(_) => Ok(Value::Json(Arc::from("null"))),
        _ => Ok(Value::Json(Arc::from("null"))),
    }
}

/// Compare two values
/// Returns: -1 if a < b, 0 if a == b, 1 if a > b
fn compare_values(a: &Value, b: &Value) -> i32 {
    // Handle NULL values
    if a.is_null() && b.is_null() {
        return 0;
    }
    if a.is_null() {
        return -1;
    }
    if b.is_null() {
        return 1;
    }

    // Same type comparison
    match (a, b) {
        (Value::Integer(av), Value::Integer(bv)) => {
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        (Value::Float(av), Value::Float(bv)) => {
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        (Value::Text(av), Value::Text(bv)) => {
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        (Value::Boolean(av), Value::Boolean(bv)) => match (*av, *bv) {
            (false, true) => -1,
            (true, false) => 1,
            _ => 0,
        },
        (Value::Timestamp(av), Value::Timestamp(bv)) => {
            if av < bv {
                -1
            } else if av > bv {
                1
            } else {
                0
            }
        }
        // Mixed numeric types
        (Value::Integer(av), Value::Float(bv)) => {
            let af = *av as f64;
            if af < *bv {
                -1
            } else if af > *bv {
                1
            } else {
                0
            }
        }
        (Value::Float(av), Value::Integer(bv)) => {
            let bf = *bv as f64;
            if *av < bf {
                -1
            } else if *av > bf {
                1
            } else {
                0
            }
        }
        // Fallback to string comparison
        _ => {
            let as_str = a.as_string().unwrap_or_default();
            let bs_str = b.as_string().unwrap_or_default();
            if as_str < bs_str {
                -1
            } else if as_str > bs_str {
                1
            } else {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SchemaBuilder;

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("value", DataType::Text)
            .add("score", DataType::Float)
            .build()
    }

    #[test]
    fn test_cast_to_integer() {
        let result = cast_to_integer(&Value::text("42")).unwrap();
        assert_eq!(result, Value::integer(42));

        let result = cast_to_integer(&Value::float(3.5)).unwrap();
        assert_eq!(result, Value::integer(3));

        let result = cast_to_integer(&Value::Boolean(true)).unwrap();
        assert_eq!(result, Value::integer(1));
    }

    #[test]
    fn test_cast_to_float() {
        let result = cast_to_float(&Value::text("3.5")).unwrap();
        assert_eq!(result, Value::float(3.5));

        let result = cast_to_float(&Value::integer(42)).unwrap();
        assert_eq!(result, Value::float(42.0));
    }

    #[test]
    fn test_cast_to_string() {
        let result = cast_to_string(&Value::integer(42)).unwrap();
        assert_eq!(result, Value::text("42"));

        let result = cast_to_string(&Value::Boolean(true)).unwrap();
        assert_eq!(result, Value::text("true"));
    }

    #[test]
    fn test_cast_to_boolean() {
        let result = cast_to_boolean(&Value::text("true")).unwrap();
        assert_eq!(result, Value::Boolean(true));

        let result = cast_to_boolean(&Value::text("yes")).unwrap();
        assert_eq!(result, Value::Boolean(true));

        let result = cast_to_boolean(&Value::integer(0)).unwrap();
        assert_eq!(result, Value::Boolean(false));

        let result = cast_to_boolean(&Value::integer(1)).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_cast_expr_evaluate() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("42"),
            Value::float(3.5),
        ]);

        let mut expr = CastExpr::new("value", DataType::Integer);
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_compound_expr_integer_comparison() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("42"),
            Value::float(3.5),
        ]);

        // CAST(value AS INTEGER) > 40
        let cast = CastExpr::new("value", DataType::Integer);
        let mut expr = CompoundExpr::new(cast, Operator::Gt, Value::integer(40));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));

        // CAST(value AS INTEGER) < 40
        let cast = CastExpr::new("value", DataType::Integer);
        let mut expr = CompoundExpr::new(cast, Operator::Lt, Value::integer(40));
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_compound_expr_float_comparison() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("3.14"),
            Value::float(3.5),
        ]);

        // CAST(value AS FLOAT) >= 3.0
        let cast = CastExpr::new("value", DataType::Float);
        let mut expr = CompoundExpr::new(cast, Operator::Gte, Value::float(3.0));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_compound_expr_string_comparison() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(42),
            Value::text("hello"),
            Value::float(3.5),
        ]);

        // CAST(id AS TEXT) = '42'
        let cast = CastExpr::new("id", DataType::Text);
        let mut expr = CompoundExpr::new(cast, Operator::Eq, Value::text("42"));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_null_cast() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::null(DataType::Text),
            Value::float(3.5),
        ]);

        let mut expr = CastExpr::new("value", DataType::Integer);
        expr.prepare_for_schema(&schema);

        // NULL values should return false
        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));
    }

    #[test]
    fn test_with_aliases() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("42"),
            Value::float(3.5),
        ]);

        let mut aliases = HashMap::new();
        aliases.insert("v".to_string(), "value".to_string());

        let expr = CastExpr::new("v", DataType::Integer);
        let mut aliased = expr.with_aliases(&aliases);
        aliased.prepare_for_schema(&schema);

        assert!(aliased.evaluate(&row).unwrap());
    }

    #[test]
    fn test_compare_values() {
        assert_eq!(compare_values(&Value::integer(1), &Value::integer(2)), -1);
        assert_eq!(compare_values(&Value::integer(2), &Value::integer(2)), 0);
        assert_eq!(compare_values(&Value::integer(3), &Value::integer(2)), 1);

        assert_eq!(compare_values(&Value::float(1.0), &Value::float(2.0)), -1);
        assert_eq!(compare_values(&Value::text("a"), &Value::text("b")), -1);
    }

    #[test]
    fn test_get_column_name() {
        let expr = CastExpr::new("id", DataType::Integer);
        assert_eq!(expr.get_column_name(), Some("id"));
    }

    #[test]
    fn test_target_type() {
        let expr = CastExpr::new("id", DataType::Integer);
        assert_eq!(expr.target_type(), DataType::Integer);
    }

    #[test]
    fn test_cast_invalid_string_to_integer() {
        let result = cast_to_integer(&Value::text("not_a_number")).unwrap();
        assert_eq!(result, Value::integer(0)); // Invalid strings default to 0
    }

    #[test]
    fn test_cast_float_string_to_integer() {
        let result = cast_to_integer(&Value::text("3.7")).unwrap();
        assert_eq!(result, Value::integer(3)); // Truncates to integer
    }
}
