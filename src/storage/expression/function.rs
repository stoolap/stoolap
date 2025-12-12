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

//! Function expression for evaluating scalar functions in WHERE clauses
//!
//! This module provides `FunctionExpr` which wraps a scalar function call
//! and evaluates it as a boolean expression. This enables queries like:
//!
//! ```sql
//! SELECT * FROM users WHERE UPPER(name) = 'ALICE'
//! SELECT * FROM products WHERE LENGTH(description) > 100
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::core::{Operator, Result, Row, Schema, Value};
use crate::functions::ScalarFunction;

use super::{find_column_index, resolve_alias, Expression};

/// Expression that evaluates a scalar function and compares the result
///
/// This expression wraps a scalar function call with its arguments,
/// evaluates the function for each row, and compares the result to
/// a target value using a comparison operator.
///
/// # Example
///
/// For `UPPER(name) = 'ALICE'`:
/// - function: UPPER
/// - arguments: [ColumnArg("name")]
/// - operator: Eq
/// - compare_value: Text("ALICE")
pub struct FunctionExpr {
    /// The scalar function to call
    function: Arc<dyn ScalarFunction>,
    /// Arguments to pass to the function
    arguments: Vec<FunctionArg>,
    /// Comparison operator
    operator: Operator,
    /// Value to compare the function result against
    compare_value: Value,
    /// Pre-computed column indices for arguments
    arg_indices: Vec<Option<usize>>,
    /// Whether this expression has been prepared
    prepared: bool,
}

/// Argument to a function in a FunctionExpr
#[derive(Clone)]
pub enum FunctionArg {
    /// A column reference
    Column(String),
    /// A literal value
    Literal(Value),
    /// A nested function call (for function composition)
    Function {
        function: Arc<dyn ScalarFunction>,
        arguments: Vec<FunctionArg>,
    },
}

impl Debug for FunctionArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionArg::Column(col) => f.debug_tuple("Column").field(col).finish(),
            FunctionArg::Literal(val) => f.debug_tuple("Literal").field(val).finish(),
            FunctionArg::Function {
                function,
                arguments,
            } => f
                .debug_struct("Function")
                .field("name", &function.name())
                .field("arguments", arguments)
                .finish(),
        }
    }
}

impl FunctionExpr {
    /// Create a new function expression
    ///
    /// # Arguments
    /// * `function` - The scalar function to call
    /// * `arguments` - Arguments to pass to the function
    /// * `operator` - Comparison operator
    /// * `compare_value` - Value to compare the function result against
    pub fn new(
        function: Arc<dyn ScalarFunction>,
        arguments: Vec<FunctionArg>,
        operator: Operator,
        compare_value: Value,
    ) -> Self {
        let arg_count = arguments.len();
        Self {
            function,
            arguments,
            operator,
            compare_value,
            arg_indices: vec![None; arg_count],
            prepared: false,
        }
    }

    /// Create a function expression for equality comparison
    pub fn eq(
        function: Arc<dyn ScalarFunction>,
        arguments: Vec<FunctionArg>,
        compare_value: Value,
    ) -> Self {
        Self::new(function, arguments, Operator::Eq, compare_value)
    }

    /// Create a function expression that evaluates to boolean (no comparison)
    ///
    /// This is for functions that return boolean directly, like custom predicates
    pub fn boolean(function: Arc<dyn ScalarFunction>, arguments: Vec<FunctionArg>) -> Self {
        Self::new(function, arguments, Operator::Eq, Value::Boolean(true))
    }

    /// Get the function name
    pub fn function_name(&self) -> &str {
        self.function.name()
    }

    /// Get the arguments
    pub fn get_arguments(&self) -> &[FunctionArg] {
        &self.arguments
    }

    /// Get the operator
    pub fn get_operator(&self) -> Operator {
        self.operator
    }

    /// Get the compare value
    pub fn get_compare_value(&self) -> &Value {
        &self.compare_value
    }

    /// Evaluate a function argument to get its value
    #[allow(clippy::only_used_in_recursion)]
    fn evaluate_arg(
        &self,
        arg: &FunctionArg,
        arg_index: Option<usize>,
        row: &Row,
    ) -> Result<Value> {
        match arg {
            FunctionArg::Column(col_name) => {
                if let Some(idx) = arg_index {
                    Ok(row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                } else {
                    // Fallback: try to find column by name (shouldn't happen if prepared)
                    Err(crate::core::Error::ColumnNotFoundNamed(col_name.clone()))
                }
            }
            FunctionArg::Literal(value) => Ok(value.clone()),
            FunctionArg::Function {
                function,
                arguments,
            } => {
                // Recursively evaluate nested function
                let args: Result<Vec<Value>> = arguments
                    .iter()
                    .map(|a| self.evaluate_arg(a, None, row))
                    .collect();
                function.evaluate(&args?)
            }
        }
    }

    /// Compare two values using the configured operator
    fn compare(&self, result: &Value, target: &Value) -> bool {
        match self.operator {
            Operator::Eq => result == target,
            Operator::Ne => result != target,
            Operator::Lt => result < target,
            Operator::Lte => result <= target,
            Operator::Gt => result > target,
            Operator::Gte => result >= target,
            // For other operators, fall back to equality check or false
            _ => false,
        }
    }
}

impl Debug for FunctionExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionExpr")
            .field("function", &self.function.name())
            .field("arguments", &self.arguments)
            .field("operator", &self.operator)
            .field("compare_value", &self.compare_value)
            .field("prepared", &self.prepared)
            .finish()
    }
}

impl Expression for FunctionExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        // Evaluate all arguments
        let mut arg_values = Vec::with_capacity(self.arguments.len());
        for (i, arg) in self.arguments.iter().enumerate() {
            let value = self.evaluate_arg(arg, self.arg_indices.get(i).copied().flatten(), row)?;
            arg_values.push(value);
        }

        // Call the function
        let result = self.function.evaluate(&arg_values)?;

        // Compare with target value
        Ok(self.compare(&result, &self.compare_value))
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        self.evaluate(row).unwrap_or(false)
    }

    fn with_aliases(&self, aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        let new_arguments: Vec<FunctionArg> = self
            .arguments
            .iter()
            .map(|arg| match arg {
                FunctionArg::Column(col) => {
                    FunctionArg::Column(resolve_alias(col, aliases).to_string())
                }
                FunctionArg::Literal(v) => FunctionArg::Literal(v.clone()),
                FunctionArg::Function {
                    function,
                    arguments,
                } => {
                    // Recursively resolve aliases in nested functions
                    FunctionArg::Function {
                        function: Arc::clone(function),
                        arguments: arguments
                            .iter()
                            .map(|a| match a {
                                FunctionArg::Column(c) => {
                                    FunctionArg::Column(resolve_alias(c, aliases).to_string())
                                }
                                other => other.clone(),
                            })
                            .collect(),
                    }
                }
            })
            .collect();

        Box::new(FunctionExpr {
            function: Arc::clone(&self.function),
            arguments: new_arguments,
            operator: self.operator,
            compare_value: self.compare_value.clone(),
            arg_indices: vec![None; self.arguments.len()],
            prepared: false,
        })
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        self.arg_indices = self
            .arguments
            .iter()
            .map(|arg| match arg {
                FunctionArg::Column(col) => find_column_index(schema, col),
                _ => None,
            })
            .collect();
        self.prepared = true;
    }

    fn is_prepared(&self) -> bool {
        self.prepared
    }

    fn can_use_index(&self) -> bool {
        // Function expressions generally can't use indexes
        // unless we have function-based indexes (future optimization)
        false
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(FunctionExpr {
            function: Arc::clone(&self.function),
            arguments: self.arguments.clone(),
            operator: self.operator,
            compare_value: self.compare_value.clone(),
            arg_indices: self.arg_indices.clone(),
            prepared: self.prepared,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Expression that wraps a closure for dynamic evaluation
///
/// This is useful for testing or when you need a custom predicate
/// that doesn't fit the standard expression types.
pub struct EvalExpr {
    /// The evaluation function
    eval_fn: Box<dyn Fn(&Row) -> bool + Send + Sync>,
}

impl EvalExpr {
    /// Create a new eval expression from a closure
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Row) -> bool + Send + Sync + 'static,
    {
        Self {
            eval_fn: Box::new(f),
        }
    }
}

impl Debug for EvalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalExpr").finish()
    }
}

impl Expression for EvalExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        Ok((self.eval_fn)(row))
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        (self.eval_fn)(row)
    }

    fn with_aliases(&self, _aliases: &HashMap<String, String>) -> Box<dyn Expression> {
        // Can't modify the closure, return a clone
        panic!("EvalExpr does not support alias resolution")
    }

    fn prepare_for_schema(&mut self, _schema: &Schema) {
        // Nothing to prepare for closures
    }

    fn is_prepared(&self) -> bool {
        true // Always "prepared"
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        panic!("EvalExpr cannot be cloned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, SchemaBuilder};
    use crate::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, UpperFunction,
    };

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add("age", DataType::Integer)
            .build()
    }

    // Helper function for testing - LENGTH function
    struct TestLengthFn;

    impl ScalarFunction for TestLengthFn {
        fn name(&self) -> &str {
            "LENGTH"
        }

        fn info(&self) -> FunctionInfo {
            FunctionInfo::new(
                "LENGTH",
                FunctionType::Scalar,
                "Returns the length of a string",
                FunctionSignature::new(
                    FunctionDataType::Integer,
                    vec![FunctionDataType::String],
                    1,
                    1,
                ),
            )
        }

        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            match args.first() {
                Some(Value::Text(s)) => Ok(Value::Integer(s.len() as i64)),
                Some(Value::Null(_)) => Ok(Value::null_unknown()),
                _ => Ok(Value::Integer(0)),
            }
        }

        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(TestLengthFn)
        }
    }

    #[test]
    fn test_function_expr_upper() {
        let schema = test_schema();
        let upper_fn = Arc::new(UpperFunction);

        let mut expr = FunctionExpr::eq(
            upper_fn,
            vec![FunctionArg::Column("name".to_string())],
            Value::text("ALICE"),
        );
        expr.prepare_for_schema(&schema);

        // Row with name = "alice" should match UPPER(name) = 'ALICE'
        let row1 = Row::from_values(vec![
            Value::Integer(1),
            Value::text("alice"),
            Value::Integer(30),
        ]);
        assert!(expr.evaluate(&row1).unwrap());

        // Row with name = "Alice" should also match
        let row2 = Row::from_values(vec![
            Value::Integer(2),
            Value::text("Alice"),
            Value::Integer(25),
        ]);
        assert!(expr.evaluate(&row2).unwrap());

        // Row with name = "bob" should not match
        let row3 = Row::from_values(vec![
            Value::Integer(3),
            Value::text("bob"),
            Value::Integer(35),
        ]);
        assert!(!expr.evaluate(&row3).unwrap());
    }

    #[test]
    fn test_function_expr_with_literal() {
        let schema = test_schema();
        let upper_fn = Arc::new(UpperFunction);

        let mut expr = FunctionExpr::eq(
            upper_fn,
            vec![FunctionArg::Literal(Value::text("hello"))],
            Value::text("HELLO"),
        );
        expr.prepare_for_schema(&schema);

        // Should always match since it's comparing literals
        let row = Row::from_values(vec![
            Value::Integer(1),
            Value::text("anything"),
            Value::Integer(30),
        ]);
        assert!(expr.evaluate(&row).unwrap());
    }

    #[test]
    fn test_function_expr_operators() {
        let schema = test_schema();

        let length_fn = Arc::new(TestLengthFn);

        // Test LENGTH(name) > 3
        let mut expr = FunctionExpr::new(
            length_fn,
            vec![FunctionArg::Column("name".to_string())],
            Operator::Gt,
            Value::Integer(3),
        );
        expr.prepare_for_schema(&schema);

        // "alice" has length 5 > 3
        let row1 = Row::from_values(vec![
            Value::Integer(1),
            Value::text("alice"),
            Value::Integer(30),
        ]);
        assert!(expr.evaluate(&row1).unwrap());

        // "bob" has length 3, not > 3
        let row2 = Row::from_values(vec![
            Value::Integer(2),
            Value::text("bob"),
            Value::Integer(25),
        ]);
        assert!(!expr.evaluate(&row2).unwrap());
    }

    #[test]
    fn test_eval_expr() {
        let expr = EvalExpr::new(|row| {
            // Return true if first column is Integer > 5
            match row.get(0) {
                Some(Value::Integer(n)) => *n > 5,
                _ => false,
            }
        });

        let row1 = Row::from_values(vec![Value::Integer(10)]);
        assert!(expr.evaluate(&row1).unwrap());

        let row2 = Row::from_values(vec![Value::Integer(3)]);
        assert!(!expr.evaluate(&row2).unwrap());
    }

    #[test]
    fn test_function_expr_clone() {
        let upper_fn = Arc::new(UpperFunction);

        let expr = FunctionExpr::eq(
            upper_fn,
            vec![FunctionArg::Column("name".to_string())],
            Value::text("ALICE"),
        );

        let cloned = expr.clone_box();
        assert!(format!("{:?}", cloned).contains("FunctionExpr"));
    }
}
