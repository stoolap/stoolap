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

//! Query Classification Cache
//!
//! This module caches pre-computed characteristics of SELECT statements to avoid
//! repeated AST traversals. For example, determining if a query has aggregation
//! requires walking the entire SELECT column list - we cache this result.
//!
//! # Performance Impact
//!
//! Before: has_aggregation() called 3-5 times per query, each traversing all columns
//! After: Single traversal on first access, O(1) lookup thereafter

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use crate::parser::ast::{Expression, SelectStatement};

/// Maximum number of cached query classifications (LRU eviction)
const CLASSIFICATION_CACHE_SIZE: usize = 512;

/// Global cache for query classifications
static CLASSIFICATION_CACHE: Mutex<Option<LruCache<u64, Arc<QueryClassification>>>> =
    Mutex::new(None);

/// Pre-computed characteristics of a SELECT statement
/// Some fields are computed for future optimizations but not yet used.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QueryClassification {
    // === Basic query structure ===
    /// Whether the query has aggregate functions (COUNT, SUM, etc.)
    pub has_aggregation: bool,
    /// Whether the query has window functions (ROW_NUMBER, etc.)
    pub has_window_functions: bool,
    /// Whether the query has GROUP BY clause
    pub has_group_by: bool,
    /// Number of columns in GROUP BY
    pub group_by_column_count: usize,
    /// Whether the query has ORDER BY clause
    pub has_order_by: bool,
    /// Whether the query has LIMIT clause
    pub has_limit: bool,
    /// Whether the query has OFFSET clause
    pub has_offset: bool,
    /// Whether the query has DISTINCT
    pub has_distinct: bool,
    /// Whether the query has CTEs (WITH clause)
    pub has_cte: bool,
    /// Whether the query has set operations (UNION, etc.)
    pub has_set_operations: bool,
    /// Whether the query has HAVING clause
    pub has_having: bool,

    // === SELECT clause analysis ===
    /// Whether the SELECT is `*` (all columns)
    pub is_select_star: bool,
    /// Whether the SELECT has qualified star (table.*)
    pub has_qualified_star: bool,
    /// Number of columns in SELECT
    pub select_column_count: usize,
    /// Whether SELECT has scalar subqueries
    pub select_has_scalar_subqueries: bool,

    // === JOIN analysis ===
    /// Whether the query has any joins
    pub has_joins: bool,
    /// Whether the query has outer joins (LEFT, RIGHT, FULL)
    pub has_outer_joins: bool,
    /// Number of joins in the query
    pub join_count: usize,

    // === WHERE clause analysis ===
    /// Whether query has a WHERE clause
    pub has_where: bool,
    /// Whether WHERE clause has parameters ($1, $2, etc.)
    pub where_has_parameters: bool,
    /// Whether WHERE clause has any subqueries
    pub where_has_subqueries: bool,
    /// Whether WHERE has EXISTS subqueries
    pub where_has_exists: bool,
    /// Whether WHERE has IN with subquery
    pub where_has_in_subquery: bool,
    /// Whether WHERE has scalar subqueries
    pub where_has_scalar_subquery: bool,
    /// Whether WHERE has ALL/ANY subqueries
    pub where_has_all_any: bool,
    /// Whether WHERE has correlated subqueries (references outer columns)
    pub where_has_correlated_subqueries: bool,

    // === SELECT clause correlated subqueries ===
    /// Whether any SELECT column has correlated subqueries
    pub select_has_correlated_subqueries: bool,

    // === ORDER BY analysis ===
    /// Whether ORDER BY has correlated subqueries
    pub order_by_has_correlated_subqueries: bool,

    // === Table source analysis ===
    /// Whether query has derived tables (subqueries in FROM)
    pub has_derived_tables: bool,

    // === Compound classifications ===
    /// Simple scan: no joins, no subqueries, no aggregation, no window functions
    pub is_simple_scan: bool,
}

impl QueryClassification {
    /// Classify a SELECT statement, computing all characteristics in a single pass
    pub fn classify(stmt: &SelectStatement) -> Self {
        // Basic query structure
        let has_aggregation = Self::check_has_aggregation(stmt);
        let has_window_functions = Self::check_has_window_functions(stmt);
        let has_group_by = !stmt.group_by.columns.is_empty();
        let group_by_column_count = stmt.group_by.columns.len();
        let has_order_by = !stmt.order_by.is_empty();
        let has_limit = stmt.limit.is_some();
        let has_offset = stmt.offset.is_some();
        let has_distinct = stmt.distinct;
        let has_cte = stmt.with.is_some();
        let has_set_operations = !stmt.set_operations.is_empty();
        let has_having = stmt.having.is_some();

        // SELECT clause analysis
        let is_select_star =
            stmt.columns.len() == 1 && matches!(stmt.columns.first(), Some(Expression::Star(_)));
        let has_qualified_star = stmt
            .columns
            .iter()
            .any(|c| matches!(c, Expression::QualifiedStar(_)));
        let select_column_count = stmt.columns.len();
        let select_has_scalar_subqueries = stmt
            .columns
            .iter()
            .any(Self::expression_has_scalar_subquery);

        // JOIN analysis
        let (has_joins, has_outer_joins, join_count, has_derived_tables) =
            Self::analyze_table_source(&stmt.table_expr);

        // WHERE clause analysis
        let has_where = stmt.where_clause.is_some();
        let (
            where_has_parameters,
            where_has_subqueries,
            where_has_exists,
            where_has_in_subquery,
            where_has_scalar_subquery,
            where_has_all_any,
            where_has_correlated_subqueries,
        ) = if let Some(ref where_clause) = stmt.where_clause {
            Self::analyze_where_clause(where_clause)
        } else {
            (false, false, false, false, false, false, false)
        };

        // SELECT column correlated subquery analysis
        let select_has_correlated_subqueries = stmt
            .columns
            .iter()
            .any(Self::expression_has_correlated_subqueries);

        // ORDER BY correlated subquery analysis
        let order_by_has_correlated_subqueries = stmt
            .order_by
            .iter()
            .any(|ob| Self::expression_has_correlated_subqueries(&ob.expression));

        // Compound classification
        let is_simple_scan = !has_joins
            && !has_aggregation
            && !has_window_functions
            && !where_has_subqueries
            && !select_has_scalar_subqueries
            && !has_derived_tables
            && !has_cte
            && !has_set_operations;

        QueryClassification {
            has_aggregation,
            has_window_functions,
            has_group_by,
            group_by_column_count,
            has_order_by,
            has_limit,
            has_offset,
            has_distinct,
            has_cte,
            has_set_operations,
            has_having,
            is_select_star,
            has_qualified_star,
            select_column_count,
            select_has_scalar_subqueries,
            has_joins,
            has_outer_joins,
            join_count,
            has_where,
            where_has_parameters,
            where_has_subqueries,
            where_has_exists,
            where_has_in_subquery,
            where_has_scalar_subquery,
            where_has_all_any,
            where_has_correlated_subqueries,
            select_has_correlated_subqueries,
            order_by_has_correlated_subqueries,
            has_derived_tables,
            is_simple_scan,
        }
    }

    /// Analyze table source for joins and derived tables
    fn analyze_table_source(table_expr: &Option<Box<Expression>>) -> (bool, bool, usize, bool) {
        let mut has_joins = false;
        let mut has_outer_joins = false;
        let mut join_count = 0;
        let mut has_derived_tables = false;

        if let Some(ref expr) = table_expr {
            Self::analyze_table_expr_recursive(
                expr,
                &mut has_joins,
                &mut has_outer_joins,
                &mut join_count,
                &mut has_derived_tables,
            );
        }

        (has_joins, has_outer_joins, join_count, has_derived_tables)
    }

    /// Recursively analyze table expression for joins and derived tables
    fn analyze_table_expr_recursive(
        expr: &Expression,
        has_joins: &mut bool,
        has_outer_joins: &mut bool,
        join_count: &mut usize,
        has_derived_tables: &mut bool,
    ) {
        match expr {
            Expression::JoinSource(join) => {
                *has_joins = true;
                *join_count += 1;

                // Check for outer join types
                let join_type = join.join_type.to_uppercase();
                if join_type.contains("LEFT")
                    || join_type.contains("RIGHT")
                    || join_type.contains("FULL")
                {
                    *has_outer_joins = true;
                }

                // Recurse into left and right
                Self::analyze_table_expr_recursive(
                    &join.left,
                    has_joins,
                    has_outer_joins,
                    join_count,
                    has_derived_tables,
                );
                Self::analyze_table_expr_recursive(
                    &join.right,
                    has_joins,
                    has_outer_joins,
                    join_count,
                    has_derived_tables,
                );
            }
            Expression::SubquerySource(_) | Expression::ScalarSubquery(_) => {
                *has_derived_tables = true;
            }
            Expression::Aliased(aliased) => {
                Self::analyze_table_expr_recursive(
                    &aliased.expression,
                    has_joins,
                    has_outer_joins,
                    join_count,
                    has_derived_tables,
                );
            }
            _ => {}
        }
    }

    /// Analyze WHERE clause for various subquery types
    fn analyze_where_clause(expr: &Expression) -> (bool, bool, bool, bool, bool, bool, bool) {
        let has_parameters = Self::expression_has_parameters(expr);
        let mut has_exists = false;
        let mut has_in_subquery = false;
        let mut has_scalar_subquery = false;
        let mut has_all_any = false;

        Self::analyze_where_expr_recursive(
            expr,
            &mut has_exists,
            &mut has_in_subquery,
            &mut has_scalar_subquery,
            &mut has_all_any,
        );

        let has_subqueries = has_exists || has_in_subquery || has_scalar_subquery || has_all_any;

        // Check for correlated subqueries (expensive AST traversal - cached here)
        let has_correlated = Self::expression_has_correlated_subqueries(expr);

        (
            has_parameters,
            has_subqueries,
            has_exists,
            has_in_subquery,
            has_scalar_subquery,
            has_all_any,
            has_correlated,
        )
    }

    /// Recursively analyze WHERE expression for subquery types
    fn analyze_where_expr_recursive(
        expr: &Expression,
        has_exists: &mut bool,
        has_in_subquery: &mut bool,
        has_scalar_subquery: &mut bool,
        has_all_any: &mut bool,
    ) {
        match expr {
            Expression::Exists(_) => {
                *has_exists = true;
            }
            Expression::AllAny(_) => {
                *has_all_any = true;
            }
            Expression::ScalarSubquery(_) => {
                *has_scalar_subquery = true;
            }
            Expression::In(in_expr) => {
                if matches!(
                    in_expr.right.as_ref(),
                    Expression::ScalarSubquery(_) | Expression::SubquerySource(_)
                ) {
                    *has_in_subquery = true;
                }
                Self::analyze_where_expr_recursive(
                    &in_expr.left,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
                Self::analyze_where_expr_recursive(
                    &in_expr.right,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
            }
            Expression::Infix(infix) => {
                Self::analyze_where_expr_recursive(
                    &infix.left,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
                Self::analyze_where_expr_recursive(
                    &infix.right,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
            }
            Expression::Prefix(prefix) => {
                Self::analyze_where_expr_recursive(
                    &prefix.right,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
            }
            Expression::FunctionCall(func) => {
                for arg in &func.arguments {
                    Self::analyze_where_expr_recursive(
                        arg,
                        has_exists,
                        has_in_subquery,
                        has_scalar_subquery,
                        has_all_any,
                    );
                }
            }
            Expression::Case(case) => {
                for wc in &case.when_clauses {
                    Self::analyze_where_expr_recursive(
                        &wc.condition,
                        has_exists,
                        has_in_subquery,
                        has_scalar_subquery,
                        has_all_any,
                    );
                    Self::analyze_where_expr_recursive(
                        &wc.then_result,
                        has_exists,
                        has_in_subquery,
                        has_scalar_subquery,
                        has_all_any,
                    );
                }
                if let Some(ref else_val) = case.else_value {
                    Self::analyze_where_expr_recursive(
                        else_val,
                        has_exists,
                        has_in_subquery,
                        has_scalar_subquery,
                        has_all_any,
                    );
                }
            }
            Expression::Between(between) => {
                Self::analyze_where_expr_recursive(
                    &between.expr,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
                Self::analyze_where_expr_recursive(
                    &between.lower,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
                Self::analyze_where_expr_recursive(
                    &between.upper,
                    has_exists,
                    has_in_subquery,
                    has_scalar_subquery,
                    has_all_any,
                );
            }
            _ => {}
        }
    }

    /// Check if expression contains scalar subqueries
    fn expression_has_scalar_subquery(expr: &Expression) -> bool {
        match expr {
            Expression::ScalarSubquery(_) => true,
            Expression::Aliased(aliased) => {
                Self::expression_has_scalar_subquery(&aliased.expression)
            }
            Expression::Infix(infix) => {
                Self::expression_has_scalar_subquery(&infix.left)
                    || Self::expression_has_scalar_subquery(&infix.right)
            }
            Expression::Prefix(prefix) => Self::expression_has_scalar_subquery(&prefix.right),
            Expression::FunctionCall(func) => func
                .arguments
                .iter()
                .any(Self::expression_has_scalar_subquery),
            Expression::Case(case) => {
                case.when_clauses.iter().any(|w| {
                    Self::expression_has_scalar_subquery(&w.condition)
                        || Self::expression_has_scalar_subquery(&w.then_result)
                }) || case
                    .else_value
                    .as_ref()
                    .is_some_and(|e| Self::expression_has_scalar_subquery(e))
            }
            _ => false,
        }
    }

    /// Check if any column expression contains aggregate functions
    fn check_has_aggregation(stmt: &SelectStatement) -> bool {
        for col_expr in &stmt.columns {
            if Self::expression_has_aggregation(col_expr) {
                return true;
            }
        }
        !stmt.group_by.columns.is_empty()
    }

    /// Check if an expression contains aggregate functions
    fn expression_has_aggregation(expr: &Expression) -> bool {
        match expr {
            Expression::FunctionCall(func) => {
                if is_aggregate_function(&func.function) {
                    return true;
                }
                func.arguments.iter().any(Self::expression_has_aggregation)
            }
            Expression::Aliased(aliased) => Self::expression_has_aggregation(&aliased.expression),
            Expression::Infix(infix) => {
                Self::expression_has_aggregation(&infix.left)
                    || Self::expression_has_aggregation(&infix.right)
            }
            Expression::Prefix(prefix) => Self::expression_has_aggregation(&prefix.right),
            Expression::Cast(cast) => Self::expression_has_aggregation(&cast.expr),
            Expression::Case(case) => {
                case.when_clauses.iter().any(|w| {
                    Self::expression_has_aggregation(&w.condition)
                        || Self::expression_has_aggregation(&w.then_result)
                }) || case
                    .else_value
                    .as_ref()
                    .is_some_and(|e| Self::expression_has_aggregation(e))
            }
            Expression::ScalarSubquery(_) | Expression::SubquerySource(_) => false, // Subquery aggregates are handled separately
            _ => false,
        }
    }

    /// Check if any column expression contains window functions
    fn check_has_window_functions(stmt: &SelectStatement) -> bool {
        stmt.columns
            .iter()
            .any(Self::expression_has_window_function)
    }

    /// Check if an expression contains window functions
    fn expression_has_window_function(expr: &Expression) -> bool {
        match expr {
            Expression::Window(_) => true,
            Expression::Aliased(aliased) => {
                Self::expression_has_window_function(&aliased.expression)
            }
            Expression::Infix(infix) => {
                Self::expression_has_window_function(&infix.left)
                    || Self::expression_has_window_function(&infix.right)
            }
            Expression::Prefix(prefix) => Self::expression_has_window_function(&prefix.right),
            Expression::Cast(cast) => Self::expression_has_window_function(&cast.expr),
            Expression::Case(case) => {
                case.when_clauses.iter().any(|w| {
                    Self::expression_has_window_function(&w.condition)
                        || Self::expression_has_window_function(&w.then_result)
                }) || case
                    .else_value
                    .as_ref()
                    .is_some_and(|e| Self::expression_has_window_function(e))
            }
            _ => false,
        }
    }

    /// Check if an expression contains parameter placeholders ($1, $2, etc.)
    /// Mirrors the implementation in utils.rs to ensure consistent behavior
    fn expression_has_parameters(expr: &Expression) -> bool {
        match expr {
            Expression::Parameter(_) => true,
            Expression::Prefix(prefix) => Self::expression_has_parameters(&prefix.right),
            Expression::Infix(infix) => {
                Self::expression_has_parameters(&infix.left)
                    || Self::expression_has_parameters(&infix.right)
            }
            Expression::In(in_expr) => {
                Self::expression_has_parameters(&in_expr.left)
                    || match in_expr.right.as_ref() {
                        Expression::List(list) => {
                            list.elements.iter().any(Self::expression_has_parameters)
                        }
                        Expression::ExpressionList(list) => {
                            list.expressions.iter().any(Self::expression_has_parameters)
                        }
                        other => Self::expression_has_parameters(other),
                    }
            }
            Expression::Between(between) => {
                Self::expression_has_parameters(&between.expr)
                    || Self::expression_has_parameters(&between.lower)
                    || Self::expression_has_parameters(&between.upper)
            }
            Expression::Like(like) => {
                Self::expression_has_parameters(&like.left)
                    || Self::expression_has_parameters(&like.pattern)
            }
            Expression::Case(case) => {
                case.value
                    .as_ref()
                    .is_some_and(|e| Self::expression_has_parameters(e))
                    || case.when_clauses.iter().any(|w| {
                        Self::expression_has_parameters(&w.condition)
                            || Self::expression_has_parameters(&w.then_result)
                    })
                    || case
                        .else_value
                        .as_ref()
                        .is_some_and(|e| Self::expression_has_parameters(e))
            }
            Expression::FunctionCall(func) => {
                func.arguments.iter().any(Self::expression_has_parameters)
            }
            Expression::Aliased(aliased) => Self::expression_has_parameters(&aliased.expression),
            Expression::Cast(cast) => Self::expression_has_parameters(&cast.expr),
            _ => false,
        }
    }

    /// Check if an expression contains correlated subqueries (references outer columns)
    /// This is an expensive check as it must examine each subquery's WHERE clause
    fn expression_has_correlated_subqueries(expr: &Expression) -> bool {
        match expr {
            Expression::Exists(exists) => Self::is_subquery_correlated(&exists.subquery),
            Expression::ScalarSubquery(subquery) => {
                Self::is_subquery_correlated(&subquery.subquery)
            }
            Expression::AllAny(all_any) => Self::is_subquery_correlated(&all_any.subquery),
            Expression::Prefix(prefix) => {
                // Handle NOT EXISTS
                if let Expression::Exists(exists) = prefix.right.as_ref() {
                    return Self::is_subquery_correlated(&exists.subquery);
                }
                Self::expression_has_correlated_subqueries(&prefix.right)
            }
            Expression::Infix(infix) => {
                Self::expression_has_correlated_subqueries(&infix.left)
                    || Self::expression_has_correlated_subqueries(&infix.right)
            }
            Expression::In(in_expr) => {
                if let Expression::ScalarSubquery(subquery) = in_expr.right.as_ref() {
                    return Self::is_subquery_correlated(&subquery.subquery);
                }
                Self::expression_has_correlated_subqueries(&in_expr.left)
            }
            Expression::Between(between) => {
                Self::expression_has_correlated_subqueries(&between.expr)
                    || Self::expression_has_correlated_subqueries(&between.lower)
                    || Self::expression_has_correlated_subqueries(&between.upper)
            }
            Expression::Aliased(aliased) => {
                Self::expression_has_correlated_subqueries(&aliased.expression)
            }
            Expression::FunctionCall(func) => func
                .arguments
                .iter()
                .any(Self::expression_has_correlated_subqueries),
            Expression::Case(case) => {
                case.when_clauses.iter().any(|w| {
                    Self::expression_has_correlated_subqueries(&w.condition)
                        || Self::expression_has_correlated_subqueries(&w.then_result)
                }) || case
                    .else_value
                    .as_ref()
                    .is_some_and(|e| Self::expression_has_correlated_subqueries(e))
            }
            _ => false,
        }
    }

    /// Check if a subquery is correlated (references outer table columns)
    fn is_subquery_correlated(subquery: &SelectStatement) -> bool {
        // Collect table names/aliases from the subquery's FROM clause
        let subquery_tables = Self::collect_subquery_tables(&subquery.table_expr);

        // Check if WHERE clause references columns not in subquery's own tables
        if let Some(ref where_clause) = subquery.where_clause {
            if Self::has_outer_column_reference(where_clause, &subquery_tables) {
                return true;
            }
        }

        // Check SELECT columns too (for scalar subqueries)
        for col in &subquery.columns {
            if Self::has_outer_column_reference(col, &subquery_tables) {
                return true;
            }
        }

        false
    }

    /// Collect table names and aliases from a subquery's FROM clause
    fn collect_subquery_tables(table_expr: &Option<Box<Expression>>) -> Vec<String> {
        let mut tables = Vec::new();
        if let Some(ref expr) = table_expr {
            Self::collect_tables_recursive(expr, &mut tables);
        }
        tables
    }

    /// Recursively collect table names and aliases
    fn collect_tables_recursive(expr: &Expression, tables: &mut Vec<String>) {
        match expr {
            Expression::Identifier(ident) => {
                tables.push(ident.value_lower.clone());
            }
            Expression::Aliased(aliased) => {
                // Add alias (use value_lower for case-insensitive matching)
                tables.push(aliased.alias.value_lower.clone());
                // Also collect from inner expression
                Self::collect_tables_recursive(&aliased.expression, tables);
            }
            Expression::JoinSource(join) => {
                Self::collect_tables_recursive(&join.left, tables);
                Self::collect_tables_recursive(&join.right, tables);
            }
            Expression::SubquerySource(subquery) => {
                // Subquery has an alias, collect it
                if let Some(ref alias) = subquery.alias {
                    tables.push(alias.value_lower.clone());
                }
            }
            Expression::TableSource(table) => {
                // Add table name and alias if present
                tables.push(table.name.value_lower.clone());
                if let Some(ref alias) = table.alias {
                    tables.push(alias.value_lower.clone());
                }
            }
            _ => {}
        }
    }

    /// Check if an expression references columns from tables NOT in the given list
    fn has_outer_column_reference(expr: &Expression, inner_tables: &[String]) -> bool {
        match expr {
            Expression::QualifiedIdentifier(qi) => {
                // Has table qualifier - check if it's NOT in inner tables
                let table_ref = &qi.qualifier.value_lower;
                // If table reference is NOT in inner tables, it's an outer reference
                !inner_tables.iter().any(|t| t == table_ref)
            }
            Expression::Infix(infix) => {
                Self::has_outer_column_reference(&infix.left, inner_tables)
                    || Self::has_outer_column_reference(&infix.right, inner_tables)
            }
            Expression::Prefix(prefix) => {
                Self::has_outer_column_reference(&prefix.right, inner_tables)
            }
            Expression::FunctionCall(func) => func
                .arguments
                .iter()
                .any(|a| Self::has_outer_column_reference(a, inner_tables)),
            Expression::Case(case) => {
                case.when_clauses.iter().any(|w| {
                    Self::has_outer_column_reference(&w.condition, inner_tables)
                        || Self::has_outer_column_reference(&w.then_result, inner_tables)
                }) || case
                    .else_value
                    .as_ref()
                    .is_some_and(|e| Self::has_outer_column_reference(e, inner_tables))
            }
            Expression::In(in_expr) => {
                Self::has_outer_column_reference(&in_expr.left, inner_tables)
                    || Self::has_outer_column_reference(&in_expr.right, inner_tables)
            }
            Expression::Between(between) => {
                Self::has_outer_column_reference(&between.expr, inner_tables)
                    || Self::has_outer_column_reference(&between.lower, inner_tables)
                    || Self::has_outer_column_reference(&between.upper, inner_tables)
            }
            Expression::Aliased(aliased) => {
                Self::has_outer_column_reference(&aliased.expression, inner_tables)
            }
            Expression::List(list) => list
                .elements
                .iter()
                .any(|e| Self::has_outer_column_reference(e, inner_tables)),
            Expression::ExpressionList(list) => list
                .expressions
                .iter()
                .any(|e| Self::has_outer_column_reference(e, inner_tables)),
            _ => false,
        }
    }
}

/// Check if a function name is an aggregate function
fn is_aggregate_function(name: &str) -> bool {
    matches!(
        name.to_uppercase().as_str(),
        "COUNT"
            | "SUM"
            | "AVG"
            | "MIN"
            | "MAX"
            | "GROUP_CONCAT"
            | "STRING_AGG"
            | "ARRAY_AGG"
            | "STDDEV"
            | "STDDEV_POP"
            | "STDDEV_SAMP"
            | "VARIANCE"
            | "VAR_POP"
            | "VAR_SAMP"
            | "PERCENTILE"
            | "PERCENTILE_CONT"
            | "PERCENTILE_DISC"
            | "MEDIAN"
            | "MODE"
            | "BOOL_AND"
            | "BOOL_OR"
            | "BIT_AND"
            | "BIT_OR"
            | "BIT_XOR"
            | "FIRST"
            | "LAST"
            | "ANY_VALUE"
    )
}

/// Compute a cache key for a SELECT statement
/// Only hashes structural elements that affect classification (not literal values)
fn compute_classification_key(stmt: &SelectStatement) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash structural properties
    stmt.distinct.hash(&mut hasher);
    stmt.columns.len().hash(&mut hasher);
    stmt.group_by.columns.len().hash(&mut hasher);
    stmt.order_by.len().hash(&mut hasher);
    stmt.limit.is_some().hash(&mut hasher);
    stmt.offset.is_some().hash(&mut hasher);
    stmt.having.is_some().hash(&mut hasher);
    stmt.with.is_some().hash(&mut hasher);
    stmt.set_operations.len().hash(&mut hasher);

    // Hash column expression types (not values)
    for col in &stmt.columns {
        hash_expression_structure(col, &mut hasher);
    }

    // Hash WHERE clause structure if present
    if let Some(ref where_clause) = stmt.where_clause {
        hash_expression_structure(where_clause, &mut hasher);
    }

    // Hash ORDER BY expressions (critical for correlated subquery detection)
    // Without this, queries with same ORDER BY count but different expressions
    // would incorrectly share classification (e.g., "ORDER BY 1" vs "ORDER BY (SELECT ...)")
    for ob in &stmt.order_by {
        hash_expression_structure(&ob.expression, &mut hasher);
        ob.ascending.hash(&mut hasher);
        ob.nulls_first.hash(&mut hasher);
    }

    // Hash GROUP BY expressions
    for gb in &stmt.group_by.columns {
        hash_expression_structure(gb, &mut hasher);
    }

    // Hash HAVING clause if present
    if let Some(ref having) = stmt.having {
        hash_expression_structure(having, &mut hasher);
    }

    hasher.finish()
}

/// Hash the structural elements of an expression (discriminants, not values)
fn hash_expression_structure(expr: &Expression, hasher: &mut DefaultHasher) {
    std::mem::discriminant(expr).hash(hasher);

    match expr {
        Expression::FunctionCall(func) => {
            // Hash function name case-insensitively without allocating
            for c in func.function.bytes() {
                c.to_ascii_uppercase().hash(hasher);
            }
            func.arguments.len().hash(hasher);
            for arg in &func.arguments {
                hash_expression_structure(arg, hasher);
            }
        }
        Expression::Window(wf) => {
            // Hash window function name case-insensitively without allocating
            for c in wf.function.function.bytes() {
                c.to_ascii_uppercase().hash(hasher);
            }
        }
        Expression::Aliased(aliased) => {
            hash_expression_structure(&aliased.expression, hasher);
        }
        Expression::Infix(infix) => {
            infix.operator.hash(hasher);
            hash_expression_structure(&infix.left, hasher);
            hash_expression_structure(&infix.right, hasher);
        }
        Expression::Prefix(prefix) => {
            hash_expression_structure(&prefix.right, hasher);
        }
        Expression::Cast(cast) => {
            cast.type_name.hash(hasher);
            hash_expression_structure(&cast.expr, hasher);
        }
        Expression::Case(case) => {
            case.when_clauses.len().hash(hasher);
            for wc in &case.when_clauses {
                hash_expression_structure(&wc.condition, hasher);
                hash_expression_structure(&wc.then_result, hasher);
            }
            if let Some(ref else_val) = case.else_value {
                hash_expression_structure(else_val, hasher);
            }
        }
        Expression::In(in_expr) => {
            in_expr.not.hash(hasher);
            hash_expression_structure(&in_expr.left, hasher);
            hash_expression_structure(&in_expr.right, hasher);
        }
        Expression::Between(between) => {
            between.not.hash(hasher);
            hash_expression_structure(&between.expr, hasher);
            hash_expression_structure(&between.lower, hasher);
            hash_expression_structure(&between.upper, hasher);
        }
        Expression::List(list) => {
            list.elements.len().hash(hasher);
        }
        Expression::ScalarSubquery(_)
        | Expression::SubquerySource(_)
        | Expression::Exists(_)
        | Expression::AllAny(_) => {
            // Mark as having subquery without deep hashing
            "SUBQUERY".hash(hasher);
        }
        Expression::Parameter(param) => {
            param.index.hash(hasher);
        }
        _ => {
            // For literals and identifiers, just use discriminant
        }
    }
}

/// Get or compute the classification for a SELECT statement
pub fn get_classification(stmt: &SelectStatement) -> Arc<QueryClassification> {
    let cache_key = compute_classification_key(stmt);

    // Single lock to avoid TOCTOU race condition where another thread
    // could compute and insert the same classification between our check and insert
    let mut guard = CLASSIFICATION_CACHE.lock();
    let cache = guard.get_or_insert_with(|| {
        LruCache::new(NonZeroUsize::new(CLASSIFICATION_CACHE_SIZE).unwrap())
    });

    // Return cached if available, otherwise compute and cache
    if let Some(classification) = cache.get(&cache_key) {
        return classification.clone();
    }

    // Cache miss - compute classification while holding lock
    // This is acceptable because classification is fast (single AST traversal)
    let classification = Arc::new(QueryClassification::classify(stmt));
    cache.put(cache_key, classification.clone());
    classification
}

/// Clear the classification cache (for testing)
#[cfg(test)]
pub fn clear_cache() {
    let mut guard = CLASSIFICATION_CACHE.lock();
    if let Some(cache) = guard.as_mut() {
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{GroupByClause, StarExpression};
    use crate::parser::token::{Position, Token, TokenType};

    fn dummy_token() -> Token {
        Token::new(TokenType::Keyword, "SELECT", Position::new(0, 1, 1))
    }

    fn create_select_star() -> SelectStatement {
        SelectStatement {
            token: dummy_token(),
            with: None,
            distinct: false,
            columns: vec![Expression::Star(StarExpression {
                token: dummy_token(),
            })],
            table_expr: None,
            where_clause: None,
            group_by: GroupByClause::default(),
            having: None,
            window_defs: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            set_operations: vec![],
        }
    }

    #[test]
    fn test_select_star_classification() {
        let stmt = create_select_star();
        let classification = QueryClassification::classify(&stmt);

        // Basic flags
        assert!(classification.is_select_star);
        assert!(!classification.has_aggregation);
        assert!(!classification.has_window_functions);
        assert!(!classification.has_group_by);
        assert!(!classification.has_order_by);
        assert!(!classification.has_limit);
        assert!(!classification.has_distinct);

        // New fields
        assert!(!classification.has_joins);
        assert!(!classification.has_outer_joins);
        assert_eq!(classification.join_count, 0);
        assert!(!classification.has_where);
        assert!(!classification.where_has_subqueries);
        assert!(!classification.has_derived_tables);
        assert!(classification.is_simple_scan); // SELECT * is a simple scan
    }

    #[test]
    fn test_classification_caching() {
        clear_cache();

        let stmt = create_select_star();

        // First call - should compute
        let class1 = get_classification(&stmt);
        assert!(class1.is_select_star);

        // Second call - should hit cache (same Arc)
        let class2 = get_classification(&stmt);
        assert!(Arc::ptr_eq(&class1, &class2));
    }
}
