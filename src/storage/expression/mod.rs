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

//! Expression system for Stoolap
//!
//! This module provides boolean expressions used for filtering rows in queries.
//!
//! # Expression Types
//!
//! - [`ComparisonExpr`] - Simple comparison (column op value)
//! - [`AndExpr`], [`OrExpr`], [`NotExpr`] - Logical operators
//! - [`BetweenExpr`] - Range check (column BETWEEN low AND high)
//! - [`InListExpr`] - List membership (column IN (v1, v2, ...))
//! - [`NullCheckExpr`] - NULL check (column IS NULL / IS NOT NULL)
//! - [`RangeExpr`] - Optimized range check with custom inclusivity
//! - [`CastExpr`] - Type cast expression
//! - [`LikeExpr`] - Pattern matching (LIKE/ILIKE with % and _ wildcards)
//! - [`FunctionExpr`] - Scalar function evaluation (e.g., UPPER(col) = 'X')

pub mod between;
pub mod cast;
pub mod comparison;
pub mod compiled;
pub mod function;
pub mod in_list;
pub mod like;
pub mod logical;
pub mod null_check;
pub mod range;

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::core::{Operator, Result, Row, Schema, Value};

// Re-export expression types
pub use between::BetweenExpr;
pub use cast::{CastExpr, CompoundExpr};
pub use comparison::ComparisonExpr;
pub use compiled::{CompiledFilter, CompiledPattern};
pub use function::{EvalExpr, FunctionArg, FunctionExpr};
pub use in_list::InListExpr;
pub use like::LikeExpr;
pub use logical::{AndExpr, ConstBoolExpr, NotExpr, OrExpr};
pub use null_check::NullCheckExpr;
pub use range::RangeExpr;

/// Expression trait for boolean expressions used in WHERE clauses
///
/// All expressions evaluate a row and return true/false to indicate
/// whether the row matches the condition.
pub trait Expression: Send + Sync + Debug {
    /// Evaluate the expression against a row
    ///
    /// Returns `Ok(true)` if the row matches, `Ok(false)` if it doesn't,
    /// or an error if evaluation fails.
    fn evaluate(&self, row: &Row) -> Result<bool>;

    /// Fast evaluation without detailed error handling
    ///
    /// This is optimized for the hot path in query processing.
    /// Returns `false` on any error condition.
    fn evaluate_fast(&self, row: &Row) -> bool;

    /// Create a copy of this expression with column aliases resolved
    ///
    /// The aliases map maps alias names to original column names.
    /// If a column in the expression matches an alias, it will be
    /// replaced with the original name in the returned expression.
    fn with_aliases(&self, aliases: &HashMap<String, String>) -> Box<dyn Expression>;

    /// Prepare the expression for a specific schema
    ///
    /// This pre-computes column indices for fast row access during evaluation.
    /// Should be called before evaluating many rows with the same schema.
    fn prepare_for_schema(&mut self, schema: &Schema);

    /// Check if this expression has been prepared for a schema
    fn is_prepared(&self) -> bool;

    /// Get the column name this expression operates on (if single column)
    fn get_column_name(&self) -> Option<&str> {
        None
    }

    /// Check if this expression can potentially use an index
    fn can_use_index(&self) -> bool {
        false
    }

    /// Extract equality comparison info for index lookups
    ///
    /// Returns (column_name, operator, value) if this is a simple comparison expression.
    /// This is used for primary key lookups and index lookups without requiring downcasting.
    fn get_comparison_info(&self) -> Option<(&str, Operator, &Value)> {
        None
    }

    /// Get child expressions if this is an AND expression
    ///
    /// Returns Some with a slice of child expressions for AND expressions,
    /// None for other expression types. Used for expression pushdown optimization.
    fn get_and_operands(&self) -> Option<&[Box<dyn Expression>]> {
        None
    }

    /// Get child expressions if this is an OR expression
    ///
    /// Returns Some with a slice of child expressions for OR expressions,
    /// None for other expression types. Used for OR index union optimization.
    fn get_or_operands(&self) -> Option<&[Box<dyn Expression>]> {
        None
    }

    /// Get LIKE prefix info for index range scanning
    ///
    /// For LIKE expressions with prefix patterns (e.g., 'John%'), returns:
    /// - column_name: The column being matched
    /// - prefix: The prefix before the first wildcard (e.g., "John")
    /// - negated: Whether this is NOT LIKE
    ///
    /// Returns None for patterns with leading wildcards or non-LIKE expressions.
    fn get_like_prefix_info(&self) -> Option<(&str, String, bool)> {
        None
    }

    /// Collect all simple comparisons from this expression tree
    ///
    /// For AND expressions, recursively collects comparisons from all branches.
    /// For comparison expressions, returns itself.
    /// Used for index pushdown optimization.
    fn collect_comparisons(&self) -> Vec<(&str, Operator, &Value)> {
        if let Some(info) = self.get_comparison_info() {
            vec![info]
        } else if let Some(children) = self.get_and_operands() {
            let mut result = Vec::new();
            for child in children {
                result.extend(child.collect_comparisons());
            }
            result
        } else {
            vec![]
        }
    }

    /// Clone the expression into a boxed trait object
    fn clone_box(&self) -> Box<dyn Expression>;

    /// Check if the expression result would be UNKNOWN (NULL) for this row
    ///
    /// In SQL's three-valued logic, comparisons with NULL return UNKNOWN.
    /// For filtering purposes, UNKNOWN is treated as false.
    /// However, NOT(UNKNOWN) should remain UNKNOWN, not become true.
    ///
    /// This method helps detect when a false result is actually UNKNOWN due to NULL,
    /// so that NOT expressions can handle three-valued logic correctly.
    ///
    /// Default implementation returns false (expression is never unknown due to NULL).
    fn is_unknown_due_to_null(&self, _row: &Row) -> bool {
        false
    }

    /// Get a reference to the expression as Any for downcasting
    fn as_any(&self) -> &dyn Any {
        // Default implementation that returns self
        // Implementations should override if they need to be downcast
        panic!("as_any not implemented for this expression type")
    }
}

impl Clone for Box<dyn Expression> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Helper to find column index in schema
/// OPTIMIZATION: Uses Schema's cached column index map for O(1) lookup
pub(crate) fn find_column_index(schema: &Schema, column: &str) -> Option<usize> {
    // Use cached lowercase column index map from Schema for O(1) lookup
    // The map uses lowercase keys, so we need to lowercase the input
    schema
        .column_index_map()
        .get(&column.to_lowercase())
        .copied()
}

/// Helper to resolve column name through aliases
pub(crate) fn resolve_alias<'a>(column: &'a str, aliases: &'a HashMap<String, String>) -> &'a str {
    aliases.get(column).map(|s| s.as_str()).unwrap_or(column)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, SchemaBuilder};

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add("age", DataType::Integer)
            .add_nullable("email", DataType::Text)
            .build()
    }

    #[test]
    fn test_find_column_index() {
        let schema = test_schema();

        assert_eq!(find_column_index(&schema, "id"), Some(0));
        assert_eq!(find_column_index(&schema, "name"), Some(1));
        assert_eq!(find_column_index(&schema, "age"), Some(2));
        assert_eq!(find_column_index(&schema, "email"), Some(3));
        assert_eq!(find_column_index(&schema, "nonexistent"), None);

        // Case insensitive
        assert_eq!(find_column_index(&schema, "ID"), Some(0));
        assert_eq!(find_column_index(&schema, "NAME"), Some(1));
    }

    #[test]
    fn test_resolve_alias() {
        let mut aliases = HashMap::new();
        aliases.insert("n".to_string(), "name".to_string());
        aliases.insert("a".to_string(), "age".to_string());

        assert_eq!(resolve_alias("n", &aliases), "name");
        assert_eq!(resolve_alias("a", &aliases), "age");
        assert_eq!(resolve_alias("id", &aliases), "id"); // Not an alias
    }
}
