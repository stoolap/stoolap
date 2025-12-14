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

//! Predicate Pushdown Framework
//!
//! This module provides a clean, extensible architecture for predicate pushdown.
//! Each pushdown rule is self-contained and implements the `PushdownRule` trait.
//!
//! ## Adding a New Pushdown Rule
//!
//! 1. Create a new struct implementing `PushdownRule`
//! 2. Register it in `PushdownRegistry::new()`
//! 3. Done!
//!
//! ## Example
//!
//! ```ignore
//! pub struct MyCustomRule;
//!
//! impl PushdownRule for MyCustomRule {
//!     fn name(&self) -> &'static str { "my_custom_rule" }
//!
//!     fn try_convert(
//!         &self,
//!         expr: &ast::Expression,
//!         ctx: &PushdownContext<'_>,
//!     ) -> PushdownResult {
//!         // Check if this rule applies and convert
//!     }
//! }
//! ```

mod rules;

use crate::core::{Schema, Value};
use crate::executor::context::ExecutionContext;
use crate::parser::ast::{self as ast};
use crate::storage::expression::Expression as StorageExpr;

pub use rules::*;

/// Result of a pushdown attempt
#[derive(Debug)]
pub enum PushdownResult {
    /// Successfully converted to storage expression (fully pushed)
    Converted(Box<dyn StorageExpr>),
    /// Partially converted - some parts pushed, but still needs memory filter
    Partial(Box<dyn StorageExpr>),
    /// This rule doesn't apply to this expression (try next rule)
    NotApplicable,
    /// Expression cannot be pushed down (needs memory filter)
    CannotPush,
}

/// Context for pushdown operations
pub struct PushdownContext<'a> {
    /// Schema for type coercion and column index lookup
    pub schema: &'a Schema,
    /// Execution context for parameter resolution
    pub exec_ctx: Option<&'a ExecutionContext>,
}

impl<'a> PushdownContext<'a> {
    pub fn new(schema: &'a Schema, exec_ctx: Option<&'a ExecutionContext>) -> Self {
        Self { schema, exec_ctx }
    }

    /// Get column data type by name
    pub fn column_type(&self, name: &str) -> Option<crate::core::DataType> {
        self.schema
            .column_index_map()
            .get(name)
            .and_then(|&idx| self.schema.columns.get(idx).map(|c| c.data_type))
    }

    /// Coerce value to column type if known
    pub fn coerce_to_column_type(&self, column: &str, value: Value) -> Value {
        if let Some(col_type) = self.column_type(column) {
            value.into_coerce_to_type(col_type)
        } else {
            value
        }
    }
}

/// Trait for pushdown rules
///
/// Each rule is responsible for:
/// 1. Checking if it can handle an expression
/// 2. Converting the expression to a storage expression
pub trait PushdownRule: Send + Sync {
    /// Rule name for debugging and logging
    fn name(&self) -> &'static str;

    /// Try to convert an expression to a storage expression.
    ///
    /// Returns:
    /// - `Converted(expr)` if successfully converted
    /// - `NotApplicable` if this rule doesn't handle this expression type
    /// - `CannotPush` if the expression matches but cannot be pushed down
    fn try_convert(&self, expr: &ast::Expression, ctx: &PushdownContext<'_>) -> PushdownResult;
}

/// Registry of all pushdown rules
///
/// The registry tries rules in order and returns the first successful conversion.
/// Rules are ordered from most specific to most general.
pub struct PushdownRegistry {
    rules: Vec<Box<dyn PushdownRule>>,
}

impl Default for PushdownRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PushdownRegistry {
    /// Create a new registry with all built-in rules
    pub fn new() -> Self {
        let mut registry = Self { rules: vec![] };

        // Register rules in priority order (most specific first)
        // Logical operators (handle compound expressions)
        registry.register(Box::new(LogicalAndRule));
        registry.register(Box::new(LogicalOrRule));
        registry.register(Box::new(LogicalNotRule));
        registry.register(Box::new(LogicalXorRule));

        // Specific expression types
        registry.register(Box::new(BetweenRule));
        registry.register(Box::new(InListRule));
        registry.register(Box::new(LikeRule));
        registry.register(Box::new(NullCheckRule));
        registry.register(Box::new(BooleanCheckRule));

        // Comparison operators (most common, should be fast)
        registry.register(Box::new(ComparisonRule));

        // Boolean literals (constant expressions)
        registry.register(Box::new(BooleanLiteralRule));

        registry
    }

    /// Register a custom pushdown rule
    pub fn register(&mut self, rule: Box<dyn PushdownRule>) {
        self.rules.push(rule);
    }

    /// Try to push down an expression to storage layer
    ///
    /// Returns (Option<StorageExpr>, needs_memory_filter)
    pub fn try_pushdown(
        &self,
        expr: &ast::Expression,
        schema: &Schema,
        exec_ctx: Option<&ExecutionContext>,
    ) -> (Option<Box<dyn StorageExpr>>, bool) {
        let ctx = PushdownContext::new(schema, exec_ctx);
        self.try_pushdown_with_ctx(expr, &ctx)
    }

    /// Internal pushdown with pre-built context (for recursive calls)
    pub(crate) fn try_pushdown_with_ctx(
        &self,
        expr: &ast::Expression,
        ctx: &PushdownContext<'_>,
    ) -> (Option<Box<dyn StorageExpr>>, bool) {
        for rule in &self.rules {
            match rule.try_convert(expr, ctx) {
                PushdownResult::Converted(storage_expr) => {
                    return (Some(storage_expr), false);
                }
                PushdownResult::Partial(storage_expr) => {
                    // Partially pushed - have storage expr but also need memory filter
                    return (Some(storage_expr), true);
                }
                PushdownResult::CannotPush => {
                    // Expression matched but can't be pushed - need memory filter
                    return (None, true);
                }
                PushdownResult::NotApplicable => {
                    // Try next rule
                    continue;
                }
            }
        }

        // No rule matched - need memory filter
        (None, true)
    }

    /// Try to convert an expression, returning only the storage expression
    /// (for internal use in compound rules)
    pub(crate) fn convert_expr(
        &self,
        expr: &ast::Expression,
        ctx: &PushdownContext<'_>,
    ) -> Option<Box<dyn StorageExpr>> {
        let (storage_expr, _) = self.try_pushdown_with_ctx(expr, ctx);
        storage_expr
    }
}

// Global registry instance (lazily initialized)
use std::sync::OnceLock;

static REGISTRY: OnceLock<PushdownRegistry> = OnceLock::new();

/// Get the global pushdown registry
pub fn registry() -> &'static PushdownRegistry {
    REGISTRY.get_or_init(PushdownRegistry::new)
}

/// Convenience function to try pushdown using the global registry
pub fn try_pushdown(
    expr: &ast::Expression,
    schema: &Schema,
    exec_ctx: Option<&ExecutionContext>,
) -> (Option<Box<dyn StorageExpr>>, bool) {
    registry().try_pushdown(expr, schema, exec_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder};
    use crate::parser::token::{Position, Token, TokenType};

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add("age", DataType::Integer)
            .add_nullable("email", DataType::Text)
            .add("active", DataType::Boolean)
            .add("price", DataType::Float)
            .build()
    }

    fn test_row() -> Row {
        Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::integer(30),
            Value::text("alice@example.com"),
            Value::Boolean(true),
            Value::Float(99.99),
        ])
    }

    fn dummy_token() -> Token {
        Token::new(TokenType::Error, "", Position::new(0, 1, 1))
    }

    fn make_ident(name: &str) -> ast::Expression {
        ast::Expression::Identifier(ast::Identifier::new(dummy_token(), name.to_string()))
    }

    fn make_int(value: i64) -> ast::Expression {
        ast::Expression::IntegerLiteral(ast::IntegerLiteral {
            token: dummy_token(),
            value,
        })
    }

    fn make_str(value: &str) -> ast::Expression {
        ast::Expression::StringLiteral(ast::StringLiteral {
            token: dummy_token(),
            value: value.to_string(),
            type_hint: None,
        })
    }

    fn make_infix(left: ast::Expression, op: &str, right: ast::Expression) -> ast::Expression {
        ast::Expression::Infix(ast::InfixExpression::new(
            dummy_token(),
            Box::new(left),
            op.to_string(),
            Box::new(right),
        ))
    }

    #[test]
    fn test_simple_equality() {
        let schema = test_schema();
        let expr = make_infix(make_ident("id"), "=", make_int(1));

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_some());
        assert!(!needs_mem);

        let mut expr = storage_expr.unwrap();
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&test_row()).unwrap());
    }

    #[test]
    fn test_and_expression() {
        let schema = test_schema();
        let left = make_infix(make_ident("id"), "=", make_int(1));
        let right = make_infix(make_ident("age"), ">", make_int(20));
        let expr = make_infix(left, "AND", right);

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_some());
        assert!(!needs_mem);

        let mut expr = storage_expr.unwrap();
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&test_row()).unwrap());
    }

    #[test]
    fn test_function_not_pushable() {
        let schema = test_schema();
        let func = ast::Expression::FunctionCall(ast::FunctionCall {
            token: dummy_token(),
            function: "LENGTH".to_string(),
            arguments: vec![make_ident("name")],
            is_distinct: false,
            order_by: vec![],
            filter: None,
        });
        let expr = make_infix(func, ">", make_int(5));

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_none());
        assert!(needs_mem);
    }

    #[test]
    fn test_partial_pushdown() {
        let schema = test_schema();
        // id = 1 AND LENGTH(name) > 5
        let pushable = make_infix(make_ident("id"), "=", make_int(1));
        let func = ast::Expression::FunctionCall(ast::FunctionCall {
            token: dummy_token(),
            function: "LENGTH".to_string(),
            arguments: vec![make_ident("name")],
            is_distinct: false,
            order_by: vec![],
            filter: None,
        });
        let not_pushable = make_infix(func, ">", make_int(5));
        let expr = make_infix(pushable, "AND", not_pushable);

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        // Should push the id = 1 part
        assert!(storage_expr.is_some());
        // But still need memory filter for LENGTH() part
        assert!(needs_mem);

        let mut expr = storage_expr.unwrap();
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&test_row()).unwrap());
    }

    #[test]
    fn test_between() {
        let schema = test_schema();
        let expr = ast::Expression::Between(ast::BetweenExpression {
            token: dummy_token(),
            expr: Box::new(make_ident("age")),
            lower: Box::new(make_int(25)),
            upper: Box::new(make_int(35)),
            not: false,
        });

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_some());
        assert!(!needs_mem);

        let mut expr = storage_expr.unwrap();
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&test_row()).unwrap()); // age = 30
    }

    #[test]
    fn test_in_list() {
        let schema = test_schema();
        let expr = ast::Expression::In(ast::InExpression {
            token: dummy_token(),
            left: Box::new(make_ident("id")),
            right: Box::new(ast::Expression::ExpressionList(ast::ExpressionList {
                token: dummy_token(),
                expressions: vec![make_int(1), make_int(2), make_int(3)],
            })),
            not: false,
        });

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_some());
        assert!(!needs_mem);

        let mut expr = storage_expr.unwrap();
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&test_row()).unwrap()); // id = 1
    }

    #[test]
    fn test_like() {
        let schema = test_schema();
        let expr = ast::Expression::Like(ast::LikeExpression {
            token: dummy_token(),
            left: Box::new(make_ident("name")),
            operator: "LIKE".to_string(),
            pattern: Box::new(make_str("Ali%")),
            escape: None,
        });

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_some());
        assert!(!needs_mem);

        let mut expr = storage_expr.unwrap();
        expr.prepare_for_schema(&schema);
        assert!(expr.evaluate(&test_row()).unwrap()); // name = "Alice"
    }

    #[test]
    fn test_is_null() {
        let schema = test_schema();
        let expr = make_infix(
            make_ident("email"),
            "IS",
            ast::Expression::NullLiteral(ast::NullLiteral {
                token: dummy_token(),
            }),
        );

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_some());
        assert!(!needs_mem);

        let mut expr = storage_expr.unwrap();
        expr.prepare_for_schema(&schema);
        // email is not null in test_row
        assert!(!expr.evaluate(&test_row()).unwrap());
    }

    #[test]
    fn test_or_fully_pushable() {
        let schema = test_schema();
        let left = make_infix(make_ident("id"), "=", make_int(1));
        let right = make_infix(make_ident("id"), "=", make_int(2));
        let expr = make_infix(left, "OR", right);

        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_some());
        assert!(!needs_mem);
    }

    #[test]
    fn test_or_not_pushable() {
        let schema = test_schema();
        let left = make_infix(make_ident("id"), "=", make_int(1));
        let func = ast::Expression::FunctionCall(ast::FunctionCall {
            token: dummy_token(),
            function: "LENGTH".to_string(),
            arguments: vec![make_ident("name")],
            is_distinct: false,
            order_by: vec![],
            filter: None,
        });
        let right = make_infix(func, ">", make_int(5));
        let expr = make_infix(left, "OR", right);

        // OR with non-pushable part cannot be pushed (would change semantics)
        let (storage_expr, needs_mem) = try_pushdown(&expr, &schema, None);
        assert!(storage_expr.is_none());
        assert!(needs_mem);
    }
}
