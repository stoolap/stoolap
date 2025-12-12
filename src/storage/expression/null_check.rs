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

//! NULL check expression for Stoolap
//!

use std::any::Any;
use std::collections::HashMap;

use super::{find_column_index, resolve_alias, Expression};
use crate::core::{Result, Row, Schema};

/// NULL check expression (IS NULL / IS NOT NULL)
///

#[derive(Debug, Clone)]
pub struct NullCheckExpr {
    /// Column name to check
    column: String,
    /// True for IS NULL, false for IS NOT NULL
    is_null: bool,

    /// Pre-computed column index
    col_index: Option<usize>,

    /// Column aliases
    aliases: HashMap<String, String>,
    /// Original column name if using alias
    original_column: Option<String>,
}

impl NullCheckExpr {
    /// Create a new NULL check expression
    pub fn new(column: impl Into<String>, is_null: bool) -> Self {
        Self {
            column: column.into(),
            is_null,
            col_index: None,
            aliases: HashMap::new(),
            original_column: None,
        }
    }

    /// Create an IS NULL expression
    pub fn is_null(column: impl Into<String>) -> Self {
        Self::new(column, true)
    }

    /// Create an IS NOT NULL expression
    pub fn is_not_null(column: impl Into<String>) -> Self {
        Self::new(column, false)
    }

    /// Check if this is an IS NULL check (vs IS NOT NULL)
    pub fn checks_for_null(&self) -> bool {
        self.is_null
    }

    /// Check if this is an IS NULL check (alias for checks_for_null, for expression compilation)
    pub fn is_null_check(&self) -> bool {
        self.is_null
    }
}

impl Expression for NullCheckExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => {
                // When not prepared, IS NULL returns true, IS NOT NULL returns false
                return Ok(self.is_null);
            }
        };

        let col_value = &row[col_idx];
        let value_is_null = col_value.is_null();

        // IS NULL returns true if value is NULL
        // IS NOT NULL returns true if value is NOT NULL
        Ok(self.is_null == value_is_null)
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        let col_idx = match self.col_index {
            Some(idx) if idx < row.len() => idx,
            _ => return self.is_null,
        };

        let col_value = &row[col_idx];
        let value_is_null = col_value.is_null();

        self.is_null == value_is_null
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
    use crate::core::{DataType, SchemaBuilder, Value};

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add_nullable("name", DataType::Text)
            .build()
    }

    #[test]
    fn test_is_null_with_null_value() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::null(DataType::Text), // name is NULL
        ]);

        let mut expr = NullCheckExpr::is_null("name");
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_is_null_with_non_null_value() {
        let schema = test_schema();
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"), // name is not NULL
        ]);

        let mut expr = NullCheckExpr::is_null("name");
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));
    }

    #[test]
    fn test_is_not_null_with_null_value() {
        let schema = test_schema();
        let row = Row::from_values(vec![Value::integer(1), Value::null(DataType::Text)]);

        let mut expr = NullCheckExpr::is_not_null("name");
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));
    }

    #[test]
    fn test_is_not_null_with_non_null_value() {
        let schema = test_schema();
        let row = Row::from_values(vec![Value::integer(1), Value::text("Alice")]);

        let mut expr = NullCheckExpr::is_not_null("name");
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_unprepared_behavior() {
        let row = Row::from_values(vec![Value::integer(1)]);

        // Unprepared IS NULL should return true
        let expr = NullCheckExpr::is_null("whatever");
        assert!(expr.evaluate(&row).unwrap());

        // Unprepared IS NOT NULL should return false
        let expr = NullCheckExpr::is_not_null("whatever");
        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_with_aliases() {
        let schema = test_schema();
        let row = Row::from_values(vec![Value::integer(1), Value::null(DataType::Text)]);

        let mut aliases = HashMap::new();
        aliases.insert("n".to_string(), "name".to_string());

        let expr = NullCheckExpr::is_null("n");
        let mut aliased = expr.with_aliases(&aliases);
        aliased.prepare_for_schema(&schema);

        assert!(aliased.evaluate(&row).unwrap());
    }

    #[test]
    fn test_checks_for_null() {
        let expr = NullCheckExpr::is_null("col");
        assert!(expr.checks_for_null());

        let expr = NullCheckExpr::is_not_null("col");
        assert!(!expr.checks_for_null());
    }
}
