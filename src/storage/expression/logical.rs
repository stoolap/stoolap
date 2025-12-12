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

//! Logical expressions (AND, OR, NOT) for Stoolap
//!

use std::any::Any;
use std::collections::HashMap;

use super::Expression;
use crate::core::{Result, Row, Schema};

/// AND expression - evaluates to true if all child expressions are true
///
/// # Short-circuit evaluation
/// Returns false immediately when the first child evaluates to false.
#[derive(Debug, Clone)]
pub struct AndExpr {
    /// Child expressions
    expressions: Vec<Box<dyn Expression>>,
    /// Whether schema preparation has been done
    is_prepared: bool,
}

impl AndExpr {
    /// Create a new AND expression from a list of expressions
    pub fn new(expressions: Vec<Box<dyn Expression>>) -> Self {
        Self {
            expressions,
            is_prepared: false,
        }
    }

    /// Create an AND expression from two expressions
    pub fn and(left: Box<dyn Expression>, right: Box<dyn Expression>) -> Self {
        Self::new(vec![left, right])
    }

    /// Add an expression to the AND
    pub fn push(&mut self, expr: Box<dyn Expression>) {
        self.expressions.push(expr);
        self.is_prepared = false;
    }

    /// Get the number of child expressions
    pub fn len(&self) -> usize {
        self.expressions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.expressions.is_empty()
    }
}

impl Expression for AndExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        for expr in &self.expressions {
            if !expr.evaluate(row)? {
                return Ok(false); // Short-circuit on first false
            }
        }
        Ok(true) // All expressions were true
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        for expr in &self.expressions {
            if !expr.evaluate_fast(row) {
                return false; // Short-circuit on first false
            }
        }
        true // All expressions were true
    }

    fn with_aliases(&self, aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        let aliased_exprs: Vec<Box<dyn Expression>> = self
            .expressions
            .iter()
            .map(|e| e.with_aliases(aliases))
            .collect();
        Box::new(AndExpr::new(aliased_exprs))
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        if self.is_prepared {
            return;
        }
        for expr in &mut self.expressions {
            expr.prepare_for_schema(schema);
        }
        self.is_prepared = true;
    }

    fn is_prepared(&self) -> bool {
        self.is_prepared
    }

    fn get_and_operands(&self) -> Option<&[Box<dyn Expression>]> {
        Some(&self.expressions)
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// OR expression - evaluates to true if any child expression is true
///
/// # Short-circuit evaluation
/// Returns true immediately when the first child evaluates to true.
#[derive(Debug, Clone)]
pub struct OrExpr {
    /// Child expressions
    expressions: Vec<Box<dyn Expression>>,
    /// Whether schema preparation has been done
    is_prepared: bool,
}

impl OrExpr {
    /// Create a new OR expression from a list of expressions
    pub fn new(expressions: Vec<Box<dyn Expression>>) -> Self {
        Self {
            expressions,
            is_prepared: false,
        }
    }

    /// Create an OR expression from two expressions
    pub fn or(left: Box<dyn Expression>, right: Box<dyn Expression>) -> Self {
        Self::new(vec![left, right])
    }

    /// Add an expression to the OR
    pub fn push(&mut self, expr: Box<dyn Expression>) {
        self.expressions.push(expr);
        self.is_prepared = false;
    }

    /// Get the number of child expressions
    pub fn len(&self) -> usize {
        self.expressions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.expressions.is_empty()
    }
}

impl Expression for OrExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        for expr in &self.expressions {
            // Note: Errors in OR are skipped to allow short-circuit evaluation
            match expr.evaluate(row) {
                Ok(true) => return Ok(true), // Short-circuit on first true
                Ok(false) => continue,
                Err(_) => continue, // Skip errors in OR
            }
        }
        Ok(false) // No expression was true
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        for expr in &self.expressions {
            if expr.evaluate_fast(row) {
                return true; // Short-circuit on first true
            }
        }
        false // No expression was true
    }

    fn with_aliases(&self, aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        let aliased_exprs: Vec<Box<dyn Expression>> = self
            .expressions
            .iter()
            .map(|e| e.with_aliases(aliases))
            .collect();
        Box::new(OrExpr::new(aliased_exprs))
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        if self.is_prepared {
            return;
        }
        for expr in &mut self.expressions {
            expr.prepare_for_schema(schema);
        }
        self.is_prepared = true;
    }

    fn is_prepared(&self) -> bool {
        self.is_prepared
    }

    fn get_or_operands(&self) -> Option<&[Box<dyn Expression>]> {
        Some(&self.expressions)
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// NOT expression - negates the result of the inner expression
///

#[derive(Debug, Clone)]
pub struct NotExpr {
    /// Inner expression to negate
    inner: Box<dyn Expression>,
    /// Whether schema preparation has been done
    is_prepared: bool,
}

impl NotExpr {
    /// Create a new NOT expression
    pub fn new(inner: Box<dyn Expression>) -> Self {
        Self {
            inner,
            is_prepared: false,
        }
    }

    /// Create a NOT expression (convenience)
    pub fn not(expr: Box<dyn Expression>) -> Self {
        Self::new(expr)
    }

    /// Get the inner expression (for expression compilation)
    pub fn get_inner(&self) -> Option<&dyn Expression> {
        Some(self.inner.as_ref())
    }
}

impl Expression for NotExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        // Three-valued logic: NOT(UNKNOWN) = UNKNOWN
        // If the inner expression is UNKNOWN due to NULL, NOT should also return false (UNKNOWN)
        if self.inner.is_unknown_due_to_null(row) {
            return Ok(false);
        }
        Ok(!self.inner.evaluate(row)?)
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        // Three-valued logic: NOT(UNKNOWN) = UNKNOWN
        // If the inner expression is UNKNOWN due to NULL, NOT should also return false (UNKNOWN)
        if self.inner.is_unknown_due_to_null(row) {
            return false;
        }
        !self.inner.evaluate_fast(row)
    }

    fn with_aliases(&self, aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        Box::new(NotExpr::new(self.inner.with_aliases(aliases)))
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        if self.is_prepared {
            return;
        }
        self.inner.prepare_for_schema(schema);
        self.is_prepared = true;
    }

    fn is_prepared(&self) -> bool {
        self.is_prepared
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }

    fn is_unknown_due_to_null(&self, row: &Row) -> bool {
        // NOT(UNKNOWN) = UNKNOWN, so propagate from inner expression
        self.inner.is_unknown_due_to_null(row)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Constant boolean expression - always returns true or false
///
/// Used for expressions that have been pre-evaluated (e.g., EXISTS subqueries)
#[derive(Debug, Clone)]
pub struct ConstBoolExpr {
    /// The constant value
    value: bool,
}

impl ConstBoolExpr {
    /// Create a new constant boolean expression
    pub fn new(value: bool) -> Self {
        Self { value }
    }

    /// Create a TRUE expression
    pub fn true_expr() -> Self {
        Self::new(true)
    }

    /// Create a FALSE expression
    pub fn false_expr() -> Self {
        Self::new(false)
    }

    /// Get the constant boolean value (for expression compilation)
    pub fn value(&self) -> bool {
        self.value
    }
}

impl Expression for ConstBoolExpr {
    fn evaluate(&self, _row: &Row) -> Result<bool> {
        Ok(self.value)
    }

    fn evaluate_fast(&self, _row: &Row) -> bool {
        self.value
    }

    fn with_aliases(&self, _aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        Box::new(self.clone())
    }

    fn prepare_for_schema(&mut self, _schema: &Schema) {
        // Nothing to prepare for constant expression
    }

    fn is_prepared(&self) -> bool {
        true // Always prepared
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
    use crate::storage::expression::ComparisonExpr;

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add("age", DataType::Integer)
            .build()
    }

    fn test_row() -> Row {
        Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::integer(30),
        ])
    }

    #[test]
    fn test_and_all_true() {
        let schema = test_schema();
        let row = test_row();

        let mut expr = AndExpr::new(vec![
            Box::new(ComparisonExpr::eq("id", Value::integer(1))),
            Box::new(ComparisonExpr::eq("name", Value::text("Alice"))),
            Box::new(ComparisonExpr::eq("age", Value::integer(30))),
        ]);
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_and_one_false() {
        let schema = test_schema();
        let row = test_row();

        let mut expr = AndExpr::new(vec![
            Box::new(ComparisonExpr::eq("id", Value::integer(1))),
            Box::new(ComparisonExpr::eq("name", Value::text("Bob"))), // false
            Box::new(ComparisonExpr::eq("age", Value::integer(30))),
        ]);
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));
    }

    #[test]
    fn test_and_short_circuit() {
        let schema = test_schema();
        let row = test_row();

        // First expression is false, second should not be evaluated
        let mut expr = AndExpr::new(vec![
            Box::new(ComparisonExpr::eq("id", Value::integer(999))), // false
            Box::new(ComparisonExpr::eq("name", Value::text("Alice"))),
        ]);
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_or_one_true() {
        let schema = test_schema();
        let row = test_row();

        let mut expr = OrExpr::new(vec![
            Box::new(ComparisonExpr::eq("id", Value::integer(999))), // false
            Box::new(ComparisonExpr::eq("name", Value::text("Alice"))), // true
        ]);
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_or_all_false() {
        let schema = test_schema();
        let row = test_row();

        let mut expr = OrExpr::new(vec![
            Box::new(ComparisonExpr::eq("id", Value::integer(999))),
            Box::new(ComparisonExpr::eq("name", Value::text("Bob"))),
        ]);
        expr.prepare_for_schema(&schema);

        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));
    }

    #[test]
    fn test_or_short_circuit() {
        let schema = test_schema();
        let row = test_row();

        // First expression is true, second should not be evaluated
        let mut expr = OrExpr::new(vec![
            Box::new(ComparisonExpr::eq("id", Value::integer(1))), // true
            Box::new(ComparisonExpr::eq("name", Value::text("Bob"))),
        ]);
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_not_true() {
        let schema = test_schema();
        let row = test_row();

        let mut expr = NotExpr::new(Box::new(ComparisonExpr::eq("id", Value::integer(999))));
        expr.prepare_for_schema(&schema);

        // NOT(false) = true
        assert!(expr.evaluate(&row).unwrap());
        assert!(expr.evaluate_fast(&row));
    }

    #[test]
    fn test_not_false() {
        let schema = test_schema();
        let row = test_row();

        let mut expr = NotExpr::new(Box::new(ComparisonExpr::eq("id", Value::integer(1))));
        expr.prepare_for_schema(&schema);

        // NOT(true) = false
        assert!(!expr.evaluate(&row).unwrap());
        assert!(!expr.evaluate_fast(&row));
    }

    #[test]
    fn test_complex_expression() {
        let schema = test_schema();
        let row = test_row(); // id=1, name="Alice", age=30

        // (id = 1 AND name = 'Alice') OR age > 40
        let mut expr = OrExpr::new(vec![
            Box::new(AndExpr::new(vec![
                Box::new(ComparisonExpr::eq("id", Value::integer(1))),
                Box::new(ComparisonExpr::eq("name", Value::text("Alice"))),
            ])),
            Box::new(ComparisonExpr::gt("age", Value::integer(40))),
        ]);
        expr.prepare_for_schema(&schema);

        // First AND is true, so OR is true
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_nested_not() {
        let schema = test_schema();
        let row = test_row();

        // NOT(NOT(id = 1)) = id = 1
        let mut expr = NotExpr::new(Box::new(NotExpr::new(Box::new(ComparisonExpr::eq(
            "id",
            Value::integer(1),
        )))));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_empty_and() {
        let row = test_row();
        let expr = AndExpr::new(vec![]);
        // Empty AND is vacuously true
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_empty_or() {
        let row = test_row();
        let expr = OrExpr::new(vec![]);
        // Empty OR is false (no true found)
        assert!(!expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_and_push() {
        let schema = test_schema();
        let row = test_row();

        let mut expr = AndExpr::new(vec![Box::new(ComparisonExpr::eq("id", Value::integer(1)))]);
        expr.push(Box::new(ComparisonExpr::eq("name", Value::text("Alice"))));
        expr.prepare_for_schema(&schema);

        assert!(expr.evaluate(&row).unwrap());
        assert_eq!(expr.len(), 2);
    }

    #[test]
    fn test_with_aliases() {
        let schema = test_schema();
        let row = test_row();

        let mut aliases = HashMap::new();
        aliases.insert("i".to_string(), "id".to_string());
        aliases.insert("n".to_string(), "name".to_string());

        let expr = AndExpr::new(vec![
            Box::new(ComparisonExpr::eq("i", Value::integer(1))),
            Box::new(ComparisonExpr::eq("n", Value::text("Alice"))),
        ]);

        let mut aliased = expr.with_aliases(&aliases);
        aliased.prepare_for_schema(&schema);

        assert!(aliased.evaluate(&row).unwrap());
    }
}
