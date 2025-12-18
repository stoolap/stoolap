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

//! Aggregation and GROUP BY Execution
//!
//! This module implements aggregation with GROUP BY and HAVING clauses:
//!
//! - Global aggregation (without GROUP BY): `SELECT COUNT(*) FROM table`
//! - Grouped aggregation: `SELECT category, SUM(amount) FROM sales GROUP BY category`
//! - HAVING clause: `SELECT category, SUM(amount) FROM sales GROUP BY category HAVING SUM(amount) > 100`

use std::hash::Hasher;

use ahash::AHasher;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::{Result, Row, Value};
use crate::functions::aggregate::CompiledAggregate;
use crate::functions::AggregateFunction;
use crate::parser::ast::*;
use crate::storage::traits::QueryResult;

use super::context::ExecutionContext;
#[allow(deprecated)]
use super::expression::CompiledEvaluator;
use super::expression::{ExpressionEval, RowFilter};
use super::join::build_column_index_map;
use super::result::ExecutorMemoryResult;
use super::utils::hash_value_into;
use super::Executor;

// Re-export for backward compatibility
pub use super::utils::{expression_contains_aggregate, is_aggregate_function};

/// Represents a grouping set for ROLLUP/CUBE operations
/// Each grouping set specifies which columns are active (included in grouping)
/// For ROLLUP(a, b), we get: [true, true], [true, false], [false, false]
#[derive(Clone, Debug)]
struct GroupingSet {
    /// For each GROUP BY column, whether it's included in this grouping level
    /// If false, the column value will be NULL in the output (rolled up)
    active_columns: Vec<bool>,
}

/// Generate a canonical key for an expression for semantic matching.
/// This ensures consistent matching regardless of token positions or formatting.
/// All string-based keys are lowercased for case-insensitive matching.
fn expression_canonical_key(expr: &Expression) -> String {
    match expr {
        Expression::Identifier(id) => id.value.to_lowercase(),
        Expression::QualifiedIdentifier(qid) => {
            format!("{}.{}", qid.qualifier.value, qid.name.value).to_lowercase()
        }
        Expression::IntegerLiteral(lit) => format!("$pos:{}", lit.value),
        Expression::FloatLiteral(lit) => format!("$float:{}", lit.value),
        Expression::StringLiteral(lit) => format!("$str:{}", lit.value.to_lowercase()),
        Expression::BooleanLiteral(lit) => format!("$bool:{}", lit.value),
        Expression::FunctionCall(func) => {
            // For function calls, build a canonical form
            let args: Vec<String> = func
                .arguments
                .iter()
                .map(expression_canonical_key)
                .collect();
            format!("{}({})", func.function.to_lowercase(), args.join(","))
        }
        Expression::Infix(bin) => {
            // For infix/binary operations, build a canonical form
            format!(
                "({} {} {})",
                expression_canonical_key(&bin.left),
                bin.operator.to_lowercase(),
                expression_canonical_key(&bin.right)
            )
        }
        Expression::Prefix(un) => {
            // For prefix/unary operations
            format!(
                "({}{})",
                un.operator.to_lowercase(),
                expression_canonical_key(&un.right)
            )
        }
        Expression::Aliased(aliased) => {
            // For aliased expressions, use the underlying expression
            expression_canonical_key(&aliased.expression)
        }
        // For other complex expressions, use Display but lowercase for consistency
        _ => format!("{}", expr).to_lowercase(),
    }
}

/// Generate a canonical key for a GroupByItem for semantic matching.
fn group_by_item_canonical_key(item: &GroupByItem) -> String {
    match item {
        GroupByItem::Column(name) => name.to_lowercase(),
        GroupByItem::Position(pos) => format!("$pos:{}", pos),
        GroupByItem::Expression { expr, .. } => expression_canonical_key(expr),
    }
}

/// Represents a GROUP BY item - either a column reference or an expression
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum GroupByItem {
    /// Simple column reference by name
    Column(String),
    /// Positional reference like GROUP BY 1
    Position(usize),
    /// Complex expression that needs to be evaluated
    Expression {
        /// The expression to evaluate
        expr: Expression,
        /// Display name for the result column (from alias if available)
        display_name: String,
    },
}

/// Represents the source of a column in post-aggregation processing
#[derive(Clone, Debug)]
enum ColumnSource {
    /// Column comes directly from aggregation result
    AggColumn(String),
    /// Column needs to be evaluated from an expression (boxed to reduce enum size)
    Expression(Box<Expression>),
    /// Correlated subquery expression that needs per-row evaluation with outer row context
    CorrelatedExpression(Box<Expression>),
    /// GROUPING() function - index is the GROUP BY column position (0-based)
    GroupingFlag(usize),
}

/// Compute a hash for a group key (slice of Values)
/// This avoids allocating Vec<Value> for each row
/// OPTIMIZATION: Use AHasher for optimal hashing of Value types (strings, floats, JSON, etc.)
/// Empirically tested to perform better than FxHasher for GROUP BY workloads
/// Called on every row in GROUP BY, so performance is critical
#[inline]
fn hash_group_key(values: &[Value]) -> u64 {
    let mut hasher = AHasher::default();
    for v in values {
        hash_value_into(v, &mut hasher);
    }
    hasher.finish()
}

/// Group entry storing the key values and row indices
struct GroupEntry {
    /// The actual key values (stored once per group)
    key_values: Vec<Value>,
    /// Indices of rows belonging to this group
    row_indices: Vec<usize>,
}

/// Represents an aggregate function call in a SELECT list
#[derive(Clone, Debug)]
pub struct SqlAggregateFunction {
    /// Function name (COUNT, SUM, AVG, MIN, MAX, etc.)
    pub name: String,
    /// Column name the function operates on (* for COUNT(*))
    pub column: String,
    /// Pre-computed lowercase column name for index lookups
    pub column_lower: String,
    /// Alias for the result column
    pub alias: Option<String>,
    /// Whether DISTINCT is specified
    pub distinct: bool,
    /// Extra arguments (e.g., separator for STRING_AGG)
    pub extra_args: Vec<Value>,
    /// The expression to evaluate for each row (for SUM(val * 2), AVG(a + b), etc.)
    /// If None, use column directly; if Some, evaluate expression first
    pub expression: Option<Expression>,
    /// ORDER BY clause for ordered-set aggregates like STRING_AGG
    pub order_by: Vec<crate::parser::ast::OrderByExpression>,
    /// FILTER clause condition - only accumulate rows where this is true
    pub filter: Option<Expression>,
    /// Whether this aggregate is hidden (only used for ORDER BY, not in SELECT)
    pub hidden: bool,
}

impl SqlAggregateFunction {
    /// Get the result column name
    pub fn get_column_name(&self) -> String {
        if let Some(ref alias) = self.alias {
            alias.clone()
        } else if self.column == "*" {
            format!("{}(*)", self.name)
        } else if self.extra_args.is_empty() {
            format!("{}({})", self.name, self.column)
        } else {
            // Include extra arguments in column name (e.g., STRING_AGG(name, ' | '))
            let args_str: Vec<String> = std::iter::once(self.column.clone())
                .chain(self.extra_args.iter().map(|v| match v {
                    Value::Text(s) => format!("'{}'", s),
                    other => other.to_string(),
                }))
                .collect();
            format!("{}({})", self.name, args_str.join(", "))
        }
    }

    /// Get the expression name (without alias) for HAVING clause matching
    /// This returns `SUM(price)` even if there's an alias like `AS total`
    pub fn get_expression_name(&self) -> String {
        if self.column == "*" {
            format!("{}(*)", self.name)
        } else {
            format!("{}({})", self.name, self.column)
        }
    }
}

impl Executor {
    /// Execute SELECT with aggregation (GROUP BY support)
    pub(crate) fn execute_select_with_aggregation(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: Vec<Row>,
        base_columns: &[String],
    ) -> Result<Box<dyn QueryResult>> {
        // Parse aggregations and group by columns
        let (aggregations, _non_agg_columns) = self.parse_aggregations(stmt)?;
        let group_by_columns = self.parse_group_by(stmt, base_columns)?;

        // Create column index map for fast lookup (FxHashMap for speed)
        let col_index_map = build_column_index_map(base_columns);

        // Determine if we can push LIMIT to aggregation for early termination
        // This is safe when:
        // 1. There's a LIMIT but no ORDER BY (order doesn't matter)
        // 2. No HAVING clause (filtering might reduce groups below limit)
        // 3. No DISTINCT (deduplication might reduce results)
        let can_push_limit = stmt.limit.is_some()
            && stmt.order_by.is_empty()
            && stmt.having.is_none()
            && !stmt.distinct;

        let aggregation_limit = if can_push_limit {
            stmt.limit.as_ref().and_then(|limit_expr| {
                let mut eval = ExpressionEval::compile(limit_expr, &[])
                    .ok()?
                    .with_context(ctx);
                eval.eval_slice(&[]).ok().and_then(|v| match v {
                    crate::core::Value::Integer(n) if n >= 0 => Some(n as usize),
                    _ => None,
                })
            })
        } else {
            None
        };

        // Build result
        let (result_columns, result_rows) = if group_by_columns.is_empty() {
            // Global aggregation (no GROUP BY)
            self.execute_global_aggregation(
                &aggregations,
                &base_rows,
                base_columns,
                &col_index_map,
                ctx,
            )?
        } else if stmt.group_by.modifier != GroupByModifier::None {
            // ROLLUP or CUBE aggregation
            self.execute_rollup_aggregation(
                &aggregations,
                &group_by_columns,
                &base_rows,
                base_columns,
                &col_index_map,
                stmt,
                ctx,
            )?
        } else {
            // Regular grouped aggregation - pass limit for early termination
            self.execute_grouped_aggregation(
                &aggregations,
                &group_by_columns,
                &base_rows,
                base_columns,
                &col_index_map,
                stmt,
                ctx,
                aggregation_limit,
            )?
        };

        // Apply HAVING clause BEFORE projection (HAVING may reference aggregates not in SELECT)
        let (having_columns, having_rows) = if let Some(ref having) = stmt.having {
            // Pre-process scalar subqueries in HAVING clause
            // This executes subqueries like (SELECT AVG(a) FROM t) and replaces them with values
            let processed_having = self.process_where_subqueries(having, ctx)?;

            // Build aggregate expression aliases for HAVING clause
            // IMPORTANT: Include ALL aggregates, not just aliased ones,
            // because CompiledEvaluator needs expression_aliases to match FunctionCall expressions
            let group_by_count = group_by_columns.len();
            let agg_aliases: Vec<(String, usize)> = aggregations
                .iter()
                .enumerate()
                .map(|(i, agg)| (agg.get_expression_name(), group_by_count + i))
                .collect();

            // Build GROUP BY expression aliases for HAVING clause
            // This maps expressions like "x + y" to their GROUP BY column indices
            let expr_aliases: Vec<(String, usize)> = group_by_columns
                .iter()
                .enumerate()
                .filter_map(|(i, item)| {
                    if let GroupByItem::Expression { expr, .. } = item {
                        Some((self.expression_to_string(expr), i))
                    } else {
                        None
                    }
                })
                .collect();

            let tmp_result = Box::new(ExecutorMemoryResult::new(
                result_columns.clone(),
                result_rows,
            ));
            let mut having_result = self.apply_having(
                tmp_result,
                &processed_having,
                &result_columns,
                &agg_aliases,
                &expr_aliases,
                ctx,
            )?;

            // Collect rows after HAVING filter
            let mut filtered_rows = Vec::new();
            while having_result.next() {
                filtered_rows.push(having_result.take_row());
            }
            (result_columns, filtered_rows)
        } else {
            (result_columns, result_rows)
        };

        // Apply post-aggregation expression evaluation and column projection
        let (mut final_columns, mut final_rows) = self.apply_post_aggregation_expressions(
            stmt,
            ctx,
            having_columns.clone(),
            having_rows.clone(),
        )?;

        // Append hidden aggregates (ORDER BY only) to the result for ORDER BY to use
        // These will be removed after sorting by the ProjectedResult wrapper
        let group_by_count = group_by_columns.len();
        let hidden_aggs: Vec<(usize, &SqlAggregateFunction)> = aggregations
            .iter()
            .enumerate()
            .filter(|(_, agg)| agg.hidden)
            .collect();

        if !hidden_aggs.is_empty() {
            // Build index map for having_columns (aggregation result columns)
            let having_col_index_map = build_column_index_map(&having_columns);

            for (agg_idx, agg) in &hidden_aggs {
                // Get the column name for this aggregate
                let col_name = agg.get_column_name();
                final_columns.push(col_name.clone());

                // Find the index in having_columns (group_by_count + aggregate index)
                let having_idx = group_by_count + agg_idx;

                // Append the value from each row
                for (row_idx, row) in final_rows.iter_mut().enumerate() {
                    if let Some(having_row) = having_rows.get(row_idx) {
                        if let Some(val) = having_row.get(having_idx) {
                            row.push(val.clone());
                        } else {
                            row.push(Value::null_unknown());
                        }
                    } else {
                        row.push(Value::null_unknown());
                    }
                }
            }

            // Also try to find by column name in case index doesn't match
            // (This handles cases where aggregate was deduplicated but still marked hidden)
            for (_, agg) in &hidden_aggs {
                let col_name = agg.get_column_name();
                let col_lower = col_name.to_lowercase();
                if let Some(&idx) = having_col_index_map.get(&col_lower) {
                    // Only add if not already added by index
                    if !final_columns
                        .iter()
                        .any(|c| c.eq_ignore_ascii_case(&col_name))
                    {
                        final_columns.push(col_name);
                        for (row_idx, row) in final_rows.iter_mut().enumerate() {
                            if let Some(having_row) = having_rows.get(row_idx) {
                                if let Some(val) = having_row.get(idx) {
                                    row.push(val.clone());
                                } else {
                                    row.push(Value::null_unknown());
                                }
                            } else {
                                row.push(Value::null_unknown());
                            }
                        }
                    }
                }
            }
        }

        let result: Box<dyn QueryResult> =
            Box::new(ExecutorMemoryResult::new(final_columns, final_rows));

        Ok(result)
    }

    /// Apply post-aggregation expressions to the result
    /// This handles expressions like `CASE WHEN SUM(x) > 100 THEN 'big' ELSE 'small' END`
    fn apply_post_aggregation_expressions(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        agg_columns: Vec<String>,
        agg_rows: Vec<Row>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // Parse GROUP BY items for GROUPING() function support
        let group_by_columns = self.parse_group_by(stmt, &agg_columns)?;

        // Check which original columns have correlated subqueries
        // These need per-row evaluation with outer row context, not pre-processing
        let correlated_flags: Vec<bool> = stmt
            .columns
            .iter()
            .map(Self::has_correlated_subqueries)
            .collect();
        let has_any_correlated = correlated_flags.iter().any(|&f| f);

        // Pre-process scalar subqueries in SELECT columns (only non-correlated ones)
        // This executes subqueries like (SELECT SUM(amount) FROM sales) and replaces them
        // with their literal values before we process the column sources
        let processed_columns = if has_any_correlated {
            // Don't pre-process if we have correlated subqueries - handle them per-row
            None
        } else {
            self.try_process_select_subqueries(&stmt.columns, ctx)?
        };
        let columns_to_use = processed_columns.as_ref().unwrap_or(&stmt.columns);

        // Build column index map for aggregate result columns
        let mut agg_col_index_map = build_column_index_map(&agg_columns);

        // Build additional mappings for aggregate expressions to handle deduplication
        // When aggregates are deduplicated (e.g., SUM(value) and SUM(value) AS total),
        // we need to map the expression "sum(value)" to the index even if the column
        // is named by its alias "total"
        for col_expr in &stmt.columns {
            if let Expression::Aliased(aliased) = col_expr {
                if let Expression::FunctionCall(func) = aliased.expression.as_ref() {
                    if is_aggregate_function(&func.function) {
                        let expr_name = self.get_aggregate_column_name(func).to_lowercase();
                        let alias_lower = aliased.alias.value.to_lowercase();
                        // If the alias exists in the map but the expression doesn't, add the expression
                        if let Some(&idx) = agg_col_index_map.get(&alias_lower) {
                            agg_col_index_map.entry(expr_name).or_insert(idx);
                        }
                        // If the expression exists in the map but the alias doesn't, add the alias
                        if let Some(&idx) = agg_col_index_map
                            .get(&self.get_aggregate_column_name(func).to_lowercase())
                        {
                            agg_col_index_map.entry(alias_lower).or_insert(idx);
                        }
                    }
                }
            } else if let Expression::FunctionCall(func) = col_expr {
                if is_aggregate_function(&func.function) {
                    let expr_name = self.get_aggregate_column_name(func).to_lowercase();
                    // Check if any aliased version exists for this expression
                    for other_col in &stmt.columns {
                        if let Expression::Aliased(other_aliased) = other_col {
                            if let Expression::FunctionCall(other_func) =
                                other_aliased.expression.as_ref()
                            {
                                if is_aggregate_function(&other_func.function) {
                                    let other_expr =
                                        self.get_aggregate_column_name(other_func).to_lowercase();
                                    if other_expr == expr_name {
                                        let alias_lower = other_aliased.alias.value.to_lowercase();
                                        if let Some(&idx) = agg_col_index_map.get(&alias_lower) {
                                            agg_col_index_map
                                                .entry(expr_name.clone())
                                                .or_insert(idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check if we have any expressions that need post-processing
        // This includes CASE, Prefix (-SUM(x)), Infix (SUM(x) + 1), and non-aggregate
        // functions wrapping aggregates like COALESCE(SUM(x), 0)
        let has_post_agg_exprs = columns_to_use.iter().any(|col| match col {
            Expression::Case(_) | Expression::Prefix(_) | Expression::Infix(_) => true,
            Expression::FunctionCall(func) => {
                // Non-aggregate function wrapping aggregate (e.g., COALESCE(SUM(val), 0))
                !is_aggregate_function(&func.function)
                    && func.arguments.iter().any(expression_contains_aggregate)
            }
            Expression::Aliased(a) => match a.expression.as_ref() {
                Expression::Case(_) | Expression::Prefix(_) | Expression::Infix(_) => true,
                Expression::FunctionCall(func) => {
                    // Non-aggregate function wrapping aggregate (e.g., COALESCE(SUM(val), 0) AS x)
                    !is_aggregate_function(&func.function)
                        && func.arguments.iter().any(expression_contains_aggregate)
                }
                _ => false,
            },
            _ => false,
        });

        // Check if SELECT columns match the aggregation columns (group by + aggregates)
        // If they differ, we need to project the result to match SELECT order
        let select_col_count = columns_to_use.len();
        let agg_col_count = agg_columns.len();
        let needs_projection = select_col_count != agg_col_count || has_post_agg_exprs;

        // Can't return early if we have correlated subqueries - they need per-row evaluation
        if !needs_projection && !has_any_correlated {
            // Check if columns are in the same order
            let mut columns_match = true;
            for (i, col_expr) in columns_to_use.iter().enumerate() {
                let expected_name = match col_expr {
                    Expression::Identifier(id) => id.value.to_lowercase(),
                    Expression::QualifiedIdentifier(qid) => {
                        format!("{}.{}", qid.qualifier.value, qid.name.value).to_lowercase()
                    }
                    Expression::Aliased(a) => a.alias.value.to_lowercase(),
                    Expression::FunctionCall(func) if is_aggregate_function(&func.function) => {
                        self.get_aggregate_column_name(func).to_lowercase()
                    }
                    _ => continue, // Can't easily compare, assume mismatch
                };
                if i >= agg_columns.len() || agg_columns[i].to_lowercase() != expected_name {
                    columns_match = false;
                    break;
                }
            }
            if columns_match {
                // Columns match exactly, return as-is
                return Ok((agg_columns, agg_rows));
            }
        }

        // Build new result with all SELECT columns in order
        let mut final_columns = Vec::new();
        let mut column_sources: Vec<ColumnSource> = Vec::new();

        // Helper to create the right ColumnSource based on whether expression has correlated subquery
        let make_source = |expr: &Expression, is_correlated: bool| -> ColumnSource {
            if is_correlated {
                ColumnSource::CorrelatedExpression(Box::new(expr.clone()))
            } else {
                ColumnSource::Expression(Box::new(expr.clone()))
            }
        };

        for (i, col_expr) in columns_to_use.iter().enumerate() {
            // Check if this column has a correlated subquery (use original column)
            let is_correlated = correlated_flags.get(i).copied().unwrap_or(false);
            // If correlated, we need to use the original expression
            let original_expr = &stmt.columns[i];

            match col_expr {
                Expression::Identifier(id) => {
                    final_columns.push(id.value.clone());
                    column_sources.push(ColumnSource::AggColumn(id.value.to_lowercase()));
                }
                Expression::QualifiedIdentifier(qid) => {
                    let name = format!("{}.{}", qid.qualifier.value, qid.name.value);
                    final_columns.push(name.clone());
                    // Try qualified name first, then fall back to unqualified column name
                    let qualified_lower = name.to_lowercase();
                    let unqualified_lower = qid.name.value.to_lowercase();
                    if agg_col_index_map.contains_key(&qualified_lower) {
                        column_sources.push(ColumnSource::AggColumn(qualified_lower));
                    } else {
                        column_sources.push(ColumnSource::AggColumn(unqualified_lower));
                    }
                }
                Expression::FunctionCall(func) => {
                    if is_aggregate_function(&func.function) {
                        let col_name = self.get_aggregate_column_name(func);
                        final_columns.push(col_name.clone());
                        column_sources.push(ColumnSource::AggColumn(col_name.to_lowercase()));
                    } else if func.function.eq_ignore_ascii_case("GROUPING") {
                        // GROUPING() function - map to the appropriate grouping flag column
                        final_columns.push(self.get_aggregate_column_name(func));
                        if let Some(idx) =
                            self.find_grouping_column_index(func, &group_by_columns, &agg_columns)
                        {
                            column_sources.push(ColumnSource::GroupingFlag(idx));
                        } else {
                            // If column not found, return 0 (treated as regularly grouped)
                            column_sources.push(ColumnSource::Expression(Box::new(
                                Expression::IntegerLiteral(crate::parser::ast::IntegerLiteral {
                                    token: crate::parser::token::Token::new(
                                        crate::parser::token::TokenType::Integer,
                                        "0",
                                        crate::parser::token::Position::new(0, 0, 0),
                                    ),
                                    value: 0,
                                }),
                            )));
                        }
                    } else {
                        // Non-aggregate function - evaluate it
                        final_columns.push(format!("{}(...)", func.function));
                        column_sources.push(make_source(col_expr, is_correlated));
                    }
                }
                Expression::Aliased(aliased) => {
                    final_columns.push(aliased.alias.value.clone());
                    let alias_lower = aliased.alias.value.to_lowercase();

                    // First check if the alias matches an aggregation column name
                    // This handles GROUP BY expression columns like UPPER(name) AS upper_name
                    if agg_col_index_map.contains_key(&alias_lower) && !is_correlated {
                        column_sources.push(ColumnSource::AggColumn(alias_lower));
                    } else if is_correlated {
                        // For correlated subqueries, use the original expression for per-row eval
                        column_sources.push(ColumnSource::CorrelatedExpression(Box::new(
                            original_expr.clone(),
                        )));
                    } else {
                        match aliased.expression.as_ref() {
                            Expression::FunctionCall(func)
                                if is_aggregate_function(&func.function) =>
                            {
                                // Aliased aggregate - look up by expression name (e.g., "SUM(value)")
                                // not by alias, since aggregates may be deduplicated
                                let agg_name = self.get_aggregate_column_name(func).to_lowercase();
                                if agg_col_index_map.contains_key(&agg_name) {
                                    column_sources.push(ColumnSource::AggColumn(agg_name));
                                } else if agg_col_index_map.contains_key(&alias_lower) {
                                    // Fall back to alias if expression name not found
                                    column_sources.push(ColumnSource::AggColumn(alias_lower));
                                } else {
                                    // Last resort: evaluate the expression
                                    column_sources.push(ColumnSource::Expression(Box::new(
                                        aliased.expression.as_ref().clone(),
                                    )));
                                }
                            }
                            Expression::FunctionCall(func)
                                if func.function.eq_ignore_ascii_case("GROUPING") =>
                            {
                                // Aliased GROUPING() function
                                if let Some(idx) = self.find_grouping_column_index(
                                    func,
                                    &group_by_columns,
                                    &agg_columns,
                                ) {
                                    column_sources.push(ColumnSource::GroupingFlag(idx));
                                } else {
                                    column_sources.push(ColumnSource::Expression(Box::new(
                                        Expression::IntegerLiteral(
                                            crate::parser::ast::IntegerLiteral {
                                                token: crate::parser::token::Token::new(
                                                    crate::parser::token::TokenType::Integer,
                                                    "0",
                                                    crate::parser::token::Position::new(0, 0, 0),
                                                ),
                                                value: 0,
                                            },
                                        ),
                                    )));
                                }
                            }
                            Expression::Case(_) => {
                                // CASE with aggregates - needs evaluation
                                column_sources.push(ColumnSource::Expression(Box::new(
                                    aliased.expression.as_ref().clone(),
                                )));
                            }
                            _ => {
                                // Try to find it in agg columns by expression string,
                                // otherwise evaluate
                                let expr_str =
                                    self.expression_to_string(aliased.expression.as_ref());
                                let expr_lower = expr_str.to_lowercase();
                                if agg_col_index_map.contains_key(&expr_lower) {
                                    column_sources.push(ColumnSource::AggColumn(expr_lower));
                                } else {
                                    column_sources.push(ColumnSource::Expression(Box::new(
                                        aliased.expression.as_ref().clone(),
                                    )));
                                }
                            }
                        }
                    }
                }
                Expression::Case(_) => {
                    // Unnamed CASE - check if expression string matches an agg column
                    let expr_str = self.expression_to_string(col_expr);
                    let expr_lower = expr_str.to_lowercase();
                    final_columns.push(expr_str);
                    if agg_col_index_map.contains_key(&expr_lower) && !is_correlated {
                        // CASE is a GROUP BY column - use existing value
                        column_sources.push(ColumnSource::AggColumn(expr_lower));
                    } else {
                        // CASE with aggregates or correlated - needs evaluation
                        column_sources.push(make_source(col_expr, is_correlated));
                    }
                }
                _ => {
                    // Other expressions
                    final_columns.push(self.expression_to_string(col_expr));
                    column_sources.push(make_source(col_expr, is_correlated));
                }
            }
        }

        // Check if we have any correlated expressions that need special handling
        let has_correlated_sources = column_sources
            .iter()
            .any(|s| matches!(s, ColumnSource::CorrelatedExpression(_)));

        // Evaluate each row
        let mut final_rows = Vec::with_capacity(agg_rows.len());
        let mut evaluator = CompiledEvaluator::new(crate::functions::registry::global_registry());
        evaluator.init_columns(&agg_columns);

        // Add aggregate expression aliases so COALESCE(SUM(val), 0) can find the "sum(val)" column
        // when the aggregate is named by its alias (e.g., "raw" from SUM(val) AS raw)
        let agg_aliases: Vec<(String, usize)> = agg_col_index_map
            .iter()
            .map(|(name, &idx)| (name.clone(), idx))
            .collect();
        evaluator.add_aggregate_aliases(&agg_aliases);

        // Pre-compute outer row column names if we have correlated expressions
        let outer_col_names: Option<std::sync::Arc<Vec<String>>> = if has_correlated_sources {
            Some(std::sync::Arc::new(agg_columns.clone()))
        } else {
            None
        };

        // Extract table alias from FROM clause for qualified column names in correlated subqueries
        let table_alias: Option<String> = if has_correlated_sources {
            if let Some(ref table_expr) = stmt.table_expr {
                match table_expr.as_ref() {
                    Expression::TableSource(source) => {
                        if let Some(ref alias) = source.alias {
                            Some(alias.value.to_lowercase())
                        } else {
                            Some(source.name.value.to_lowercase())
                        }
                    }
                    Expression::Aliased(aliased) => Some(aliased.alias.value.to_lowercase()),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };

        // OPTIMIZATION: Pre-compute lowercase and qualified column names for correlated expressions
        // This avoids repeated to_lowercase() and format!() allocations per row
        let correlated_col_names: Option<Vec<(String, Option<String>)>> = if has_correlated_sources
        {
            Some(
                agg_columns
                    .iter()
                    .map(|col_name| {
                        let col_lower = col_name.to_lowercase();
                        let qualified = table_alias
                            .as_ref()
                            .map(|alias| format!("{}.{}", alias, col_lower));
                        (col_lower, qualified)
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Reusable map for correlated expressions
        let estimated_entries = agg_columns.len() * 2;
        let mut outer_row_map: FxHashMap<String, Value> =
            FxHashMap::with_capacity_and_hasher(estimated_entries, Default::default());

        for row in agg_rows {
            let mut new_values = Vec::with_capacity(column_sources.len());
            evaluator.set_row_array(&row);

            for source in &column_sources {
                let value = match source {
                    ColumnSource::AggColumn(col_name) => {
                        if let Some(&idx) = agg_col_index_map.get(col_name) {
                            row.get(idx).cloned().unwrap_or(Value::null_unknown())
                        } else {
                            Value::null_unknown()
                        }
                    }
                    ColumnSource::Expression(expr) => {
                        // Evaluate the expression using the aggregated row as context
                        evaluator.evaluate(expr).unwrap_or(Value::null_unknown())
                    }
                    ColumnSource::CorrelatedExpression(expr) => {
                        // Build outer row context using pre-computed column names
                        outer_row_map.clear();
                        if let Some(ref col_names) = correlated_col_names {
                            for (idx, (col_lower, qualified)) in col_names.iter().enumerate() {
                                let val = row.get(idx).cloned().unwrap_or(Value::null_unknown());
                                outer_row_map.insert(col_lower.clone(), val.clone());
                                if let Some(q) = qualified {
                                    outer_row_map.insert(q.clone(), val);
                                }
                            }
                        }

                        // Create context with outer row (move map, take it back after)
                        let mut correlated_ctx = ctx.with_outer_row(
                            std::mem::take(&mut outer_row_map),
                            outer_col_names.clone().unwrap(),
                        );

                        // Process the correlated expression with the outer row context
                        let result = match self.process_correlated_expression(expr, &correlated_ctx)
                        {
                            Ok(processed_expr) => {
                                // Evaluate the processed expression
                                let mut corr_eval = CompiledEvaluator::new(&self.function_registry)
                                    .with_context(&correlated_ctx);
                                corr_eval.init_columns(&agg_columns);
                                corr_eval.set_row_array(&row);
                                corr_eval
                                    .evaluate(&processed_expr)
                                    .unwrap_or(Value::null_unknown())
                            }
                            Err(_) => Value::null_unknown(),
                        };

                        // Take back map for reuse
                        outer_row_map = correlated_ctx.outer_row.take().unwrap_or_default();
                        result
                    }
                    ColumnSource::GroupingFlag(idx) => {
                        // Look up the grouping flag from the hidden __grouping_N__ columns
                        // These columns are at the end of the row, after aggregate columns
                        let grouping_col_name = format!("__grouping_{}__", idx);
                        if let Some(&col_idx) = agg_col_index_map.get(&grouping_col_name) {
                            row.get(col_idx).cloned().unwrap_or(Value::Integer(0))
                        } else {
                            // Fallback: column is grouped normally
                            Value::Integer(0)
                        }
                    }
                };
                new_values.push(value);
            }

            final_rows.push(Row::from_values(new_values));
        }

        Ok((final_columns, final_rows))
    }

    /// Get the column name for an aggregate function
    fn get_aggregate_column_name(&self, func: &crate::parser::ast::FunctionCall) -> String {
        let args_str: Vec<String> = func
            .arguments
            .iter()
            .map(|a| self.expression_to_string(a))
            .collect();
        format!("{}({})", func.function, args_str.join(", "))
    }

    /// Find the GROUP BY column index for a GROUPING() function call
    /// Returns the index (0-based) of the GROUP BY column that matches the GROUPING() argument
    fn find_grouping_column_index(
        &self,
        func: &crate::parser::ast::FunctionCall,
        group_by_columns: &[GroupByItem],
        columns: &[String],
    ) -> Option<usize> {
        // GROUPING() takes one argument - the column name
        if func.arguments.is_empty() {
            return None;
        }

        let arg = &func.arguments[0];
        let arg_name = match arg {
            Expression::Identifier(id) => id.value.to_lowercase(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.to_lowercase(),
            _ => return None,
        };

        // Find the matching GROUP BY column
        for (idx, item) in group_by_columns.iter().enumerate() {
            let matches = match item {
                GroupByItem::Column(col_name) => col_name.to_lowercase() == arg_name,
                GroupByItem::Position(pos) => {
                    // Position is 1-indexed, convert to 0-indexed
                    let col_idx = pos.saturating_sub(1);
                    if col_idx < columns.len() {
                        columns[col_idx].to_lowercase() == arg_name
                    } else {
                        false
                    }
                }
                GroupByItem::Expression { display_name, .. } => {
                    display_name.to_lowercase() == arg_name
                }
            };
            if matches {
                return Some(idx);
            }
        }

        None
    }

    /// Execute GROUP BY aggregation and return raw columns/rows for window function processing
    /// This is used when both GROUP BY and window functions are present in the query.
    /// Window functions operate on the aggregated result.
    pub(crate) fn execute_aggregation_for_window(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: Vec<Row>,
        base_columns: &[String],
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // Parse aggregations and group by columns
        let (aggregations, _non_agg_columns) = self.parse_aggregations(stmt)?;
        let group_by_columns = self.parse_group_by(stmt, base_columns)?;

        // Create column index map for fast lookup
        let col_index_map = build_column_index_map(base_columns);

        // Build result
        // Note: No limit pushdown here because window functions need all rows
        let (result_columns, mut result_rows) = if group_by_columns.is_empty() {
            // Global aggregation (no GROUP BY)
            self.execute_global_aggregation(
                &aggregations,
                &base_rows,
                base_columns,
                &col_index_map,
                ctx,
            )?
        } else {
            // Grouped aggregation - no limit since window functions need all groups
            self.execute_grouped_aggregation(
                &aggregations,
                &group_by_columns,
                &base_rows,
                base_columns,
                &col_index_map,
                stmt,
                ctx,
                None, // Window functions need all groups
            )?
        };

        // Apply HAVING clause filter (in-place)
        if let Some(ref having) = stmt.having {
            // Build aggregate expression aliases for HAVING clause
            // This maps "SUM(price)" to its column index even if aliased as "total"
            // IMPORTANT: Include ALL aggregates, not just aliased ones,
            // because the evaluator needs expression_aliases to match FunctionCall expressions
            let group_by_count = group_by_columns.len();
            let mut all_aliases: Vec<(String, usize)> = aggregations
                .iter()
                .enumerate()
                .map(|(i, agg)| (agg.get_expression_name(), group_by_count + i))
                .collect();

            // Build GROUP BY expression aliases for HAVING clause
            // This maps expressions like "x + y" to their GROUP BY column indices
            // allowing HAVING x + y > 20 to work when GROUP BY x + y
            for (i, item) in group_by_columns.iter().enumerate() {
                if let GroupByItem::Expression { expr, .. } = item {
                    all_aliases.push((self.expression_to_string(expr), i));
                }
            }

            // Create RowFilter with all aliases and context
            let having_filter =
                RowFilter::with_aliases(having, &result_columns, &all_aliases)?.with_context(ctx);

            // Filter rows using the pre-compiled filter
            result_rows.retain(|row| having_filter.matches(row));
        }

        Ok((result_columns, result_rows))
    }

    /// Parse aggregate functions from SELECT list
    /// Returns: (aggregations, non_agg_columns, post_agg_expressions)
    fn parse_aggregations(
        &self,
        stmt: &SelectStatement,
    ) -> Result<(Vec<SqlAggregateFunction>, Vec<String>)> {
        let mut aggregations = Vec::new();
        let mut non_agg_columns = Vec::new();

        for col_expr in &stmt.columns {
            self.extract_aggregates_from_expr(col_expr, &mut aggregations, &mut non_agg_columns)?;
        }

        // Also extract aggregates from HAVING clause
        // These need to be computed even if they're not in SELECT
        if let Some(ref having) = stmt.having {
            self.extract_aggregates_from_expr(having, &mut aggregations, &mut non_agg_columns)?;
        }

        // Also extract aggregates from ORDER BY clause
        // These need to be computed even if they're not in SELECT (marked as hidden)
        // Record the count before adding ORDER BY aggregates
        let visible_count = aggregations.len();
        for order_expr in &stmt.order_by {
            self.extract_aggregates_from_expr(
                &order_expr.expression,
                &mut aggregations,
                &mut non_agg_columns,
            )?;
        }
        // Mark any new aggregates (from ORDER BY) as hidden, but only if they're truly new
        // Helper to create a signature string for an aggregate including its filter
        let make_sig = |agg: &SqlAggregateFunction| -> (String, String, bool, String) {
            let filter_sig = agg
                .filter
                .as_ref()
                .map(|f| format!("{:?}", f))
                .unwrap_or_default();
            (
                agg.name.to_uppercase(),
                agg.column.to_lowercase(),
                agg.distinct,
                filter_sig,
            )
        };

        // Check by comparing the expression signature to avoid duplicates
        for i in visible_count..aggregations.len() {
            // Check if this aggregate already exists in the visible portion
            let new_sig = make_sig(&aggregations[i]);
            let already_exists = aggregations[..visible_count]
                .iter()
                .any(|existing| make_sig(existing) == new_sig);
            if !already_exists {
                aggregations[i].hidden = true;
            }
        }
        // Remove duplicates (aggregates that exist in both SELECT and ORDER BY)
        // Include the filter in the signature so aggregates with different filters are kept
        let mut seen: std::collections::HashSet<(String, String, bool, String)> =
            std::collections::HashSet::new();
        aggregations.retain(|agg| seen.insert(make_sig(agg)));

        Ok((aggregations, non_agg_columns))
    }

    /// Extract aggregate functions from an expression (recursively)
    fn extract_aggregates_from_expr(
        &self,
        expr: &Expression,
        aggregations: &mut Vec<SqlAggregateFunction>,
        non_agg_columns: &mut Vec<String>,
    ) -> Result<()> {
        match expr {
            Expression::FunctionCall(func) => {
                if is_aggregate_function(&func.function) {
                    // Check for nested aggregates - this is invalid SQL
                    // e.g., SUM(COUNT(*)) or AVG(SUM(x)) should return an error
                    for arg in &func.arguments {
                        if expression_contains_aggregate(arg) {
                            return Err(crate::core::Error::InvalidArgumentMessage(format!(
                                "aggregate function calls cannot be nested: {}",
                                func.function
                            )));
                        }
                    }

                    let (column, distinct, extra_args, expression) =
                        self.extract_agg_column(&func.arguments)?;
                    let column_lower = column.to_lowercase();
                    aggregations.push(SqlAggregateFunction {
                        name: func.function.clone(),
                        column,
                        column_lower,
                        alias: None,
                        distinct: distinct || func.is_distinct,
                        extra_args,
                        expression,
                        order_by: func.order_by.clone(),
                        filter: func.filter.as_ref().map(|f| (**f).clone()),
                        hidden: false,
                    });
                } else {
                    // Non-aggregate function: recursively check arguments for nested aggregates
                    // e.g., COALESCE(SUM(val), 0), ABS(SUM(val)), etc.
                    for arg in &func.arguments {
                        self.extract_aggregates_from_expr(arg, aggregations, non_agg_columns)?;
                    }
                }
            }
            Expression::Aliased(aliased) => {
                // For aliased expressions, extract aggregates from the inner expression
                self.extract_aggregates_from_aliased(aliased, aggregations, non_agg_columns)?;
            }
            Expression::Identifier(id) => {
                non_agg_columns.push(id.value.clone());
            }
            Expression::Case(case) => {
                // Extract aggregates from CASE expression
                for when_clause in &case.when_clauses {
                    self.extract_aggregates_from_expr(
                        &when_clause.condition,
                        aggregations,
                        non_agg_columns,
                    )?;
                    self.extract_aggregates_from_expr(
                        &when_clause.then_result,
                        aggregations,
                        non_agg_columns,
                    )?;
                }
                if let Some(ref else_val) = case.else_value {
                    self.extract_aggregates_from_expr(else_val, aggregations, non_agg_columns)?;
                }
            }
            Expression::Infix(infix) => {
                self.extract_aggregates_from_expr(&infix.left, aggregations, non_agg_columns)?;
                self.extract_aggregates_from_expr(&infix.right, aggregations, non_agg_columns)?;
            }
            Expression::Prefix(prefix) => {
                self.extract_aggregates_from_expr(&prefix.right, aggregations, non_agg_columns)?;
            }
            Expression::Cast(cast) => {
                self.extract_aggregates_from_expr(&cast.expr, aggregations, non_agg_columns)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Extract aggregates from an aliased expression
    fn extract_aggregates_from_aliased(
        &self,
        aliased: &crate::parser::ast::AliasedExpression,
        aggregations: &mut Vec<SqlAggregateFunction>,
        non_agg_columns: &mut Vec<String>,
    ) -> Result<()> {
        match aliased.expression.as_ref() {
            Expression::FunctionCall(func) => {
                if is_aggregate_function(&func.function) {
                    // Check for nested aggregates - this is invalid SQL
                    // e.g., SUM(COUNT(*)) AS total should return an error
                    for arg in &func.arguments {
                        if expression_contains_aggregate(arg) {
                            return Err(crate::core::Error::InvalidArgumentMessage(format!(
                                "aggregate function calls cannot be nested: {}",
                                func.function
                            )));
                        }
                    }

                    let (column, distinct, extra_args, expression) =
                        self.extract_agg_column(&func.arguments)?;
                    let column_lower = column.to_lowercase();
                    aggregations.push(SqlAggregateFunction {
                        name: func.function.clone(),
                        column,
                        column_lower,
                        alias: Some(aliased.alias.value.clone()),
                        distinct: distinct || func.is_distinct,
                        extra_args,
                        expression,
                        order_by: func.order_by.clone(),
                        filter: func.filter.as_ref().map(|f| (**f).clone()),
                        hidden: false,
                    });
                } else {
                    // Non-aggregate function: recursively check arguments for nested aggregates
                    // e.g., COALESCE(SUM(val), 0) AS total, ABS(SUM(val)) AS abs_sum
                    for arg in &func.arguments {
                        self.extract_aggregates_from_expr(arg, aggregations, non_agg_columns)?;
                    }
                }
            }
            Expression::Case(case) => {
                // Extract aggregates from CASE, but keep the alias as the column name
                for when_clause in &case.when_clauses {
                    self.extract_aggregates_from_expr(
                        &when_clause.condition,
                        aggregations,
                        non_agg_columns,
                    )?;
                    self.extract_aggregates_from_expr(
                        &when_clause.then_result,
                        aggregations,
                        non_agg_columns,
                    )?;
                }
                if let Some(ref else_val) = case.else_value {
                    self.extract_aggregates_from_expr(else_val, aggregations, non_agg_columns)?;
                }
            }
            Expression::Cast(cast) => {
                // Extract aggregates from CAST expression (e.g., CAST(SUM(val) AS TEXT) AS sum_text)
                self.extract_aggregates_from_expr(&cast.expr, aggregations, non_agg_columns)?;
            }
            _ => {
                self.extract_aggregates_from_expr(
                    &aliased.expression,
                    aggregations,
                    non_agg_columns,
                )?;
            }
        }
        Ok(())
    }

    /// Extract column name, expression, and extra arguments from aggregate function arguments
    ///
    /// Returns: (column_name, distinct, extra_args, expression)
    /// - column_name: The column or expression string the aggregate operates on (* for COUNT(*))
    /// - distinct: Whether DISTINCT was found in arguments
    /// - extra_args: Additional arguments (e.g., separator for STRING_AGG)
    /// - expression: The expression to evaluate (Some for complex expressions like val * 2)
    fn extract_agg_column(
        &self,
        args: &[Expression],
    ) -> Result<(String, bool, Vec<Value>, Option<Expression>)> {
        if args.is_empty() {
            return Ok(("*".to_string(), false, Vec::new(), None));
        }

        let (column, distinct, expression) = match &args[0] {
            Expression::Star(_) => ("*".to_string(), false, None),
            Expression::Identifier(id) => (id.value.clone(), false, None),
            Expression::QualifiedIdentifier(qid) => {
                // Use full qualified name (e.g., "p.price" instead of just "price")
                // This is needed for JOIN queries where columns are qualified with table aliases
                let qualified_name = format!("{}.{}", qid.qualifier.value, qid.name.value);
                (qualified_name, false, None)
            }
            // For expressions like val * 2, a + b, etc. - store the expression
            expr => {
                let expr_str = self.expression_to_string(expr);
                (expr_str, false, Some(expr.clone()))
            }
        };

        // Extract extra arguments (starting from index 1)
        let mut extra_args = Vec::new();
        for arg in args.iter().skip(1) {
            match arg {
                Expression::StringLiteral(lit) => {
                    extra_args.push(Value::text(&lit.value));
                }
                Expression::IntegerLiteral(lit) => {
                    extra_args.push(Value::Integer(lit.value));
                }
                Expression::FloatLiteral(lit) => {
                    extra_args.push(Value::Float(lit.value));
                }
                Expression::BooleanLiteral(b) => {
                    extra_args.push(Value::Boolean(b.value));
                }
                Expression::NullLiteral(_) => {
                    extra_args.push(Value::null_unknown());
                }
                // For identifiers, we'd need to evaluate them at runtime
                // For now, skip non-literal extra arguments
                _ => {}
            }
        }

        Ok((column, distinct, extra_args, expression))
    }

    /// Parse GROUP BY clause
    fn parse_group_by(
        &self,
        stmt: &SelectStatement,
        _base_columns: &[String],
    ) -> Result<Vec<GroupByItem>> {
        let mut group_items = Vec::new();

        // Build a map of aliases to their expressions from SELECT clause
        let alias_map: FxHashMap<String, Expression> = stmt
            .columns
            .iter()
            .filter_map(|col| {
                if let Expression::Aliased(aliased) = col {
                    Some((
                        aliased.alias.value.to_lowercase(),
                        (*aliased.expression).clone(),
                    ))
                } else {
                    None
                }
            })
            .collect();

        // For GROUPING SETS, extract all unique columns from all sets
        let columns_to_parse: Vec<&Expression> =
            if let GroupByModifier::GroupingSets(ref sets) = stmt.group_by.modifier {
                // Collect all unique columns from all grouping sets
                // Use canonical key for uniqueness (handles case-insensitivity and structural matching)
                let mut seen = std::collections::HashSet::new();
                let mut unique_cols = Vec::new();
                for set in sets {
                    for expr in set {
                        let key = expression_canonical_key(expr);
                        if seen.insert(key) {
                            unique_cols.push(expr);
                        }
                    }
                }
                unique_cols
            } else {
                // Regular GROUP BY, ROLLUP, or CUBE - use columns directly
                stmt.group_by.columns.iter().collect()
            };

        for expr in columns_to_parse {
            match expr {
                Expression::Identifier(id) => {
                    // Check if this identifier is an alias defined in SELECT
                    if let Some(aliased_expr) = alias_map.get(&id.value.to_lowercase()) {
                        // Use the aliased expression, with the alias as the display name
                        group_items.push(GroupByItem::Expression {
                            expr: aliased_expr.clone(),
                            display_name: id.value.clone(),
                        });
                    } else {
                        // Regular column reference
                        group_items.push(GroupByItem::Column(id.value.clone()));
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    // Use full qualified name (e.g., "c.name" instead of just "name")
                    let qualified_name = format!("{}.{}", qid.qualifier.value, qid.name.value);
                    group_items.push(GroupByItem::Column(qualified_name));
                }
                Expression::IntegerLiteral(lit) => {
                    // GROUP BY 1 refers to first SELECT column (1-indexed)
                    let pos = lit.value as usize;
                    if pos > 0 && pos <= stmt.columns.len() {
                        // Convert position to the actual SELECT column expression
                        let select_col = &stmt.columns[pos - 1];
                        match select_col {
                            Expression::Identifier(id) => {
                                // Simple column reference - use the column name
                                group_items.push(GroupByItem::Column(id.value.clone()));
                            }
                            Expression::Aliased(aliased) => {
                                // Aliased expression - extract the underlying expression
                                match aliased.expression.as_ref() {
                                    Expression::Identifier(id) => {
                                        // Aliased column reference
                                        group_items.push(GroupByItem::Column(id.value.clone()));
                                    }
                                    expr => {
                                        // Complex expression with alias
                                        group_items.push(GroupByItem::Expression {
                                            expr: expr.clone(),
                                            display_name: aliased.alias.value.clone(),
                                        });
                                    }
                                }
                            }
                            expr => {
                                // Other expressions (e.g., function calls)
                                let display_name = self.find_expression_alias(stmt, expr);
                                group_items.push(GroupByItem::Expression {
                                    expr: expr.clone(),
                                    display_name,
                                });
                            }
                        }
                    } else {
                        // Invalid position, fall back to storing position
                        group_items.push(GroupByItem::Position(pos));
                    }
                }
                Expression::FunctionCall(_) => {
                    // For function expressions, find a matching alias in SELECT
                    let display_name = self.find_expression_alias(stmt, expr);
                    group_items.push(GroupByItem::Expression {
                        expr: expr.clone(),
                        display_name,
                    });
                }
                _ => {
                    // Try to handle other expressions generically
                    let display_name = self.find_expression_alias(stmt, expr);
                    group_items.push(GroupByItem::Expression {
                        expr: expr.clone(),
                        display_name,
                    });
                }
            }
        }

        Ok(group_items)
    }

    /// Find the alias for an expression in the SELECT list
    fn find_expression_alias(&self, stmt: &SelectStatement, target_expr: &Expression) -> String {
        // Check if this expression has an alias in SELECT
        for col_expr in &stmt.columns {
            if let Expression::Aliased(aliased) = col_expr {
                // Compare expressions by converting to canonical string representation
                let aliased_str = self.expression_to_string(&aliased.expression);
                let target_str = self.expression_to_string(target_expr);
                if aliased_str == target_str {
                    return aliased.alias.value.clone();
                }
            }
        }
        // No alias found, generate a name from the expression
        self.expression_to_string(target_expr)
    }

    /// Convert an expression to a display string
    #[allow(clippy::only_used_in_recursion)]
    fn expression_to_string(&self, expr: &Expression) -> String {
        match expr {
            Expression::FunctionCall(func) => {
                let args: Vec<String> = func
                    .arguments
                    .iter()
                    .map(|a| self.expression_to_string(a))
                    .collect();
                format!("{}({})", func.function, args.join(", "))
            }
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => {
                format!("{}.{}", qid.qualifier.value, qid.name.value)
            }
            Expression::StringLiteral(lit) => format!("'{}'", lit.value),
            Expression::IntegerLiteral(lit) => lit.value.to_string(),
            Expression::FloatLiteral(lit) => lit.value.to_string(),
            Expression::BooleanLiteral(lit) => lit.value.to_string(),
            Expression::Case(case) => {
                // Use the Display implementation for CaseExpression
                format!("{}", case)
            }
            Expression::Infix(infix) => {
                format!(
                    "{} {} {}",
                    self.expression_to_string(&infix.left),
                    infix.operator,
                    self.expression_to_string(&infix.right)
                )
            }
            Expression::Prefix(prefix) => {
                format!(
                    "{}{}",
                    prefix.operator,
                    self.expression_to_string(&prefix.right)
                )
            }
            Expression::Cast(cast) => {
                format!(
                    "CAST({} AS {})",
                    self.expression_to_string(&cast.expr),
                    cast.type_name
                )
            }
            // For any other expression type, use the Display trait if implemented
            _ => format!("{}", expr),
        }
    }

    /// Execute global aggregation (no GROUP BY)
    ///
    /// Optimized with parallel processing for large datasets:
    /// - Partitions data into chunks
    /// - Processes chunks in parallel
    /// - Merges partial results
    fn execute_global_aggregation(
        &self,
        aggregations: &[SqlAggregateFunction],
        rows: &[Row],
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // Check if any aggregation has an expression (e.g., SUM(val * 2)), ORDER BY, or FILTER
        let has_expression = aggregations
            .iter()
            .any(|a| a.expression.is_some() || !a.order_by.is_empty() || a.filter.is_some());

        // FAST PATH: Single COUNT(*) - just return row count directly
        // Only use fast path when no expressions/filters are involved
        if !has_expression
            && aggregations.len() == 1
            && aggregations[0].name == "COUNT"
            && aggregations[0].column == "*"
            && !aggregations[0].distinct
            && aggregations[0].filter.is_none()
        {
            let result_columns: Vec<String> =
                aggregations.iter().map(|a| a.get_column_name()).collect();
            return Ok((
                result_columns,
                vec![Row::from_values(vec![Value::Integer(rows.len() as i64)])],
            ));
        }

        // Pre-compute column indices for faster access
        // OPTIMIZATION: Use pre-computed column_lower instead of calling to_lowercase() each time
        // Handle both qualified (e.g., "o.amount") and unqualified column names
        let agg_col_indices: Vec<Option<usize>> = aggregations
            .iter()
            .map(|agg| {
                if agg.column == "*" || agg.expression.is_some() {
                    None // Don't use column index for expressions
                } else {
                    Self::lookup_column_index(&agg.column_lower, col_index_map)
                }
            })
            .collect();

        // FAST PATH: Single SUM on integer column without DISTINCT (no expressions)
        if !has_expression
            && aggregations.len() == 1
            && aggregations[0].name == "SUM"
            && !aggregations[0].distinct
        {
            if let Some(col_idx) = agg_col_indices[0] {
                let result = self.fast_sum_column(rows, col_idx);
                let result_columns: Vec<String> =
                    aggregations.iter().map(|a| a.get_column_name()).collect();
                return Ok((result_columns, vec![Row::from_values(vec![result])]));
            }
        }

        // FAST PATH: Single AVG on column without DISTINCT (no expressions)
        if !has_expression
            && aggregations.len() == 1
            && aggregations[0].name == "AVG"
            && !aggregations[0].distinct
        {
            if let Some(col_idx) = agg_col_indices[0] {
                let result = self.fast_avg_column(rows, col_idx);
                let result_columns: Vec<String> =
                    aggregations.iter().map(|a| a.get_column_name()).collect();
                return Ok((result_columns, vec![Row::from_values(vec![result])]));
            }
        }

        // Check if any aggregation uses DISTINCT (can't parallelize easily)
        let has_distinct = aggregations.iter().any(|a| a.distinct);

        // Pre-compile filter, expression, and ORDER BY programs for VM-based evaluation
        // CRITICAL: Propagate errors instead of silently ignoring compilation failures
        use super::expression::{compile_expression, ExecuteContext, ExprVM, SharedProgram};
        let compiled_filters: Vec<Option<SharedProgram>> = if has_expression {
            aggregations
                .iter()
                .map(|agg| {
                    agg.filter
                        .as_ref()
                        .map(|f| compile_expression(f, columns))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; aggregations.len()]
        };
        let compiled_agg_expressions: Vec<Option<SharedProgram>> = if has_expression {
            aggregations
                .iter()
                .map(|agg| {
                    agg.expression
                        .as_ref()
                        .map(|e| compile_expression(e, columns))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; aggregations.len()]
        };
        // Pre-compile ORDER BY expressions for each aggregation
        // CRITICAL: Propagate errors instead of silently skipping failed compilations
        let compiled_order_by: Vec<Vec<SharedProgram>> = if has_expression {
            aggregations
                .iter()
                .map(|agg| {
                    agg.order_by
                        .iter()
                        .map(|o| compile_expression(&o.expression, columns))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![Vec::new(); aggregations.len()]
        };
        let mut expr_vm = if has_expression {
            Some(ExprVM::new())
        } else {
            None
        };

        // Use parallel processing for large datasets without DISTINCT
        let use_parallel = rows.len() >= 100_000 && !has_distinct && !has_expression;

        let result_values: Vec<Value> = if use_parallel {
            // PARALLEL: Split into chunks and process in parallel
            let chunk_size = (rows.len() / rayon::current_num_threads()).max(1000);
            let function_registry = &self.function_registry;

            // Process chunks in parallel, each producing partial aggregates
            let partial_results: Vec<Vec<Value>> = rows
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut agg_funcs: Vec<Option<Box<dyn AggregateFunction>>> = aggregations
                        .iter()
                        .map(|agg| function_registry.get_aggregate(&agg.name))
                        .collect();

                    // Configure aggregate functions with extra arguments (e.g., separator for STRING_AGG)
                    for (i, agg) in aggregations.iter().enumerate() {
                        if !agg.extra_args.is_empty() {
                            if let Some(ref mut func) = agg_funcs[i] {
                                func.configure(&agg.extra_args);
                            }
                        }
                    }

                    // Pre-create static Value for COUNT(*)
                    let count_star_value = Value::Integer(1);
                    for row in chunk {
                        for (i, _agg) in aggregations.iter().enumerate() {
                            if let Some(ref mut func) = agg_funcs[i] {
                                // OPTIMIZATION: Avoid cloning by using reference directly
                                let value_ref = if let Some(col_idx) = agg_col_indices[i] {
                                    row.get(col_idx)
                                } else {
                                    Some(&count_star_value) // COUNT(*)
                                };
                                if let Some(v) = value_ref {
                                    func.accumulate(v, false);
                                }
                                // Skip if None (missing column) - this is effectively null
                            }
                        }
                    }

                    // Return partial results
                    agg_funcs
                        .iter()
                        .map(|f| {
                            f.as_ref()
                                .map(|func| func.result())
                                .unwrap_or_else(Value::null_unknown)
                        })
                        .collect()
                })
                .collect();

            // Merge partial results
            self.merge_partial_aggregates(aggregations, partial_results)
        } else {
            // SEQUENTIAL: Check if we can use the fast compiled path
            // Fast path: no expressions, no filters, no order by on any aggregate
            // NOTE: This is conservative - it could potentially be extended to handle:
            // - Simple column expressions (not computed expressions)
            // - STRING_AGG with extra_args (separator) by passing to CompiledAggregate
            // For now, we keep it strict to ensure correctness.
            let can_use_compiled = aggregations.iter().all(|agg| {
                agg.expression.is_none()
                    && agg.filter.is_none()
                    && agg.order_by.is_empty()
                    && agg.extra_args.is_empty()
            });

            if can_use_compiled {
                // FAST PATH: Use CompiledAggregate for zero virtual dispatch
                let mut compiled_aggs: Vec<CompiledAggregate> = aggregations
                    .iter()
                    .map(|agg| {
                        let is_count_star = agg.name == "COUNT" && agg.column == "*";
                        CompiledAggregate::compile(
                            &agg.name,
                            is_count_star,
                            agg.distinct,
                            self.function_registry.get_aggregate(&agg.name),
                        )
                        .unwrap_or_else(|| {
                            // Fallback for unknown aggregates
                            CompiledAggregate::dynamic(
                                self.function_registry
                                    .get_aggregate(&agg.name)
                                    .unwrap_or_else(|| {
                                        Box::new(
                                            crate::functions::aggregate::CountFunction::default(),
                                        )
                                    }),
                            )
                        })
                    })
                    .collect();

                // Pre-create static Value for COUNT(*)
                let count_star_value = Value::Integer(1);

                // Hot loop with compiled aggregates - zero virtual dispatch
                for row in rows {
                    for i in 0..compiled_aggs.len() {
                        let value = if let Some(col_idx) = agg_col_indices[i] {
                            row.get(col_idx)
                        } else {
                            Some(&count_star_value) // COUNT(*)
                        };

                        if let Some(v) = value {
                            compiled_aggs[i].accumulate(v);
                        }
                    }
                }

                // Collect results
                compiled_aggs.iter().map(|agg| agg.result()).collect()
            } else {
                // SLOW PATH: Original algorithm with dynamic dispatch for complex aggregates
                let mut agg_funcs: Vec<Option<Box<dyn AggregateFunction>>> = aggregations
                    .iter()
                    .map(|agg| self.function_registry.get_aggregate(&agg.name))
                    .collect();

                // Configure aggregate functions with extra arguments (e.g., separator for STRING_AGG)
                for (i, agg) in aggregations.iter().enumerate() {
                    if !agg.extra_args.is_empty() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            func.configure(&agg.extra_args);
                        }
                    }
                }

                // Configure ORDER BY for ordered-set aggregates (ARRAY_AGG, STRING_AGG, etc.)
                for (i, agg) in aggregations.iter().enumerate() {
                    if !agg.order_by.is_empty() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            let directions: Vec<bool> = agg
                                .order_by
                                .iter()
                                .map(|o| o.ascending) // true = ASC, false = DESC
                                .collect();
                            func.set_order_by(directions);
                        }
                    }
                }

                // Pre-create static Value for COUNT(*)
                let count_star_value = Value::Integer(1);

                // Buffer for evaluated expression values (to avoid repeated allocation)
                let mut expr_values: Vec<Value> = vec![Value::null_unknown(); aggregations.len()];

                for row in rows {
                    // Create execution context for this row
                    // CRITICAL: Include params for parameterized queries
                    let row_data = row.as_slice();
                    let exec_ctx = ExecuteContext::new(row_data)
                        .with_params(ctx.params())
                        .with_named_params(ctx.named_params());

                    for (i, agg) in aggregations.iter().enumerate() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            // Check FILTER clause first - skip row if filter is false
                            if let Some(ref filter_program) = compiled_filters[i] {
                                if let Some(ref mut vm) = expr_vm {
                                    match vm.execute(filter_program, &exec_ctx) {
                                        Ok(Value::Boolean(true)) => {} // Continue with accumulation
                                        Ok(Value::Boolean(false)) | Ok(Value::Null(_)) => continue, // Skip this row
                                        Ok(_) => continue, // Non-boolean treated as false
                                        Err(_) => continue, // Error treated as false
                                    }
                                } else {
                                    // Can't evaluate filter without VM - skip
                                    continue;
                                }
                            }

                            // Get the value to accumulate
                            let value = if let Some(ref expr_program) = compiled_agg_expressions[i]
                            {
                                // Evaluate the expression for this row using VM
                                if let Some(ref mut vm) = expr_vm {
                                    match vm.execute(expr_program, &exec_ctx) {
                                        Ok(val) => {
                                            expr_values[i] = val;
                                            Some(&expr_values[i])
                                        }
                                        Err(e) => {
                                            // Expression evaluation failed - skip row
                                            #[cfg(debug_assertions)]
                                            eprintln!(
                                                "Warning: aggregate expression evaluation failed: {}",
                                                e
                                            );
                                            let _ = e;
                                            None
                                        }
                                    }
                                } else {
                                    None
                                }
                            } else {
                                // Simple column reference or COUNT(*)
                                if let Some(col_idx) = agg_col_indices[i] {
                                    row.get(col_idx)
                                } else {
                                    Some(&count_star_value)
                                }
                            };

                            if let Some(v) = value {
                                // Check if this aggregate has ORDER BY and supports it
                                if !compiled_order_by[i].is_empty() && func.supports_order_by() {
                                    // Evaluate ORDER BY expressions to get sort keys using pre-compiled programs
                                    if let Some(ref mut vm) = expr_vm {
                                        let mut sort_keys =
                                            Vec::with_capacity(compiled_order_by[i].len());
                                        let mut all_ok = true;
                                        for order_program in &compiled_order_by[i] {
                                            match vm.execute(order_program, &exec_ctx) {
                                                Ok(key) => sort_keys.push(key),
                                                Err(_) => {
                                                    all_ok = false;
                                                    break;
                                                }
                                            }
                                        }
                                        if all_ok {
                                            func.accumulate_with_sort_key(
                                                v,
                                                sort_keys,
                                                agg.distinct,
                                            );
                                        }
                                    } else {
                                        // No VM - fall back to regular accumulate
                                        func.accumulate(v, agg.distinct);
                                    }
                                } else {
                                    func.accumulate(v, agg.distinct);
                                }
                            }
                        }
                    }
                }

                aggregations
                    .iter()
                    .enumerate()
                    .map(|(i, agg)| {
                        if let Some(ref func) = agg_funcs[i] {
                            func.result()
                        } else if agg.name == "COUNT" && agg.column == "*" {
                            Value::Integer(rows.len() as i64)
                        } else {
                            Value::null_unknown()
                        }
                    })
                    .collect()
            }
        };

        // Build result columns
        let result_columns: Vec<String> =
            aggregations.iter().map(|a| a.get_column_name()).collect();

        Ok((result_columns, vec![Row::from_values(result_values)]))
    }

    /// Merge partial aggregate results from parallel processing
    fn merge_partial_aggregates(
        &self,
        aggregations: &[SqlAggregateFunction],
        partial_results: Vec<Vec<Value>>,
    ) -> Vec<Value> {
        if partial_results.is_empty() {
            return aggregations.iter().map(|_| Value::null_unknown()).collect();
        }

        aggregations
            .iter()
            .enumerate()
            .map(|(i, agg)| {
                let partials: Vec<&Value> = partial_results.iter().map(|r| &r[i]).collect();
                self.merge_single_aggregate(&agg.name, partials)
            })
            .collect()
    }

    /// Merge partial results for a single aggregate function
    fn merge_single_aggregate(&self, func_name: &str, partials: Vec<&Value>) -> Value {
        // OPTIMIZATION: func_name comes from SqlAggregateFunction.name which is already uppercase
        match func_name {
            "COUNT" | "SUM" => {
                // Sum all partial results
                let mut total: i64 = 0;
                let mut has_float = false;
                let mut total_float: f64 = 0.0;

                for val in partials {
                    match val {
                        Value::Integer(n) => {
                            if has_float {
                                total_float += *n as f64;
                            } else {
                                total += n;
                            }
                        }
                        Value::Float(f) => {
                            if !has_float {
                                has_float = true;
                                total_float = total as f64;
                            }
                            total_float += f;
                        }
                        _ => {}
                    }
                }

                if has_float {
                    Value::Float(total_float)
                } else {
                    Value::Integer(total)
                }
            }
            "AVG" => {
                // For AVG, we get partial AVGs, but we need SUM/COUNT
                // This is approximate - for exact results, we'd need to track count separately
                // For now, just average the partial averages (less accurate for uneven chunks)
                let mut sum: f64 = 0.0;
                let mut count = 0;

                for val in &partials {
                    match val {
                        Value::Float(f) => {
                            sum += f;
                            count += 1;
                        }
                        Value::Integer(n) => {
                            sum += *n as f64;
                            count += 1;
                        }
                        _ => {}
                    }
                }

                if count > 0 {
                    Value::Float(sum / count as f64)
                } else {
                    Value::null_unknown()
                }
            }
            "MIN" => {
                // Take minimum of all partials
                let mut min_val: Option<Value> = None;

                // OPTIMIZATION: Only clone when value actually changes
                for val in partials {
                    if matches!(val, Value::Null(_)) {
                        continue;
                    }
                    match &min_val {
                        None => min_val = Some(val.clone()),
                        Some(current) if val < current => min_val = Some(val.clone()),
                        _ => {} // Keep current, no clone needed
                    }
                }

                min_val.unwrap_or_else(Value::null_unknown)
            }
            "MAX" => {
                // Take maximum of all partials
                let mut max_val: Option<Value> = None;

                // OPTIMIZATION: Only clone when value actually changes
                for val in partials {
                    if matches!(val, Value::Null(_)) {
                        continue;
                    }
                    match &max_val {
                        None => max_val = Some(val.clone()),
                        Some(current) if val > current => max_val = Some(val.clone()),
                        _ => {} // Keep current, no clone needed
                    }
                }

                max_val.unwrap_or_else(Value::null_unknown)
            }
            _ => {
                // For unknown functions, just take the first non-null
                partials
                    .into_iter()
                    .find(|v| !matches!(v, Value::Null(_)))
                    .cloned()
                    .unwrap_or_else(Value::null_unknown)
            }
        }
    }

    /// Try to use fast single-pass aggregation for simple cases
    ///
    /// Returns Some((columns, rows)) if fast path was used, None otherwise.
    /// Fast path is used when:
    /// - All GROUP BY items are simple column references
    /// - All aggregates are SUM or COUNT (no DISTINCT, no FILTER, no ORDER BY, no expression)
    ///
    /// When `limit` is provided and there's no ORDER BY, enables early termination:
    /// once we have `limit` complete groups, we stop creating new groups.
    fn try_fast_aggregation(
        &self,
        aggregations: &[SqlAggregateFunction],
        group_by_items: &[GroupByItem],
        rows: &[Row],
        _columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        limit: Option<usize>,
    ) -> Result<Option<(Vec<String>, Vec<Row>)>> {
        // Check if all GROUP BY items are simple column references
        let group_by_indices: Vec<usize> = group_by_items
            .iter()
            .filter_map(|item| match item {
                GroupByItem::Column(col_name) => {
                    Self::lookup_column_index(&col_name.to_lowercase(), col_index_map)
                }
                _ => None,
            })
            .collect();

        // All GROUP BY items must be resolved to column indices
        if group_by_indices.len() != group_by_items.len() {
            return Ok(None);
        }

        // Check if all aggregates are simple (SUM or COUNT without DISTINCT/FILTER/ORDER BY/expression)
        #[derive(Clone)]
        enum SimpleAgg {
            Count,      // COUNT(*) or COUNT(col)
            Sum(usize), // SUM(col) - stores column index
        }

        let simple_aggs: Vec<Option<SimpleAgg>> = aggregations
            .iter()
            .map(|agg| {
                // Must not have DISTINCT, FILTER, ORDER BY, or expression
                if agg.distinct
                    || agg.filter.is_some()
                    || !agg.order_by.is_empty()
                    || agg.expression.is_some()
                {
                    return None;
                }

                match agg.name.to_uppercase().as_str() {
                    "COUNT" => Some(SimpleAgg::Count),
                    "SUM" => {
                        if agg.column == "*" {
                            None // SUM(*) is not valid
                        } else {
                            Self::lookup_column_index(&agg.column_lower, col_index_map)
                                .map(SimpleAgg::Sum)
                        }
                    }
                    _ => None, // Other aggregates not supported in fast path
                }
            })
            .collect();

        // All aggregates must be resolved for fast path
        if simple_aggs.iter().any(|a| a.is_none()) {
            return Ok(None);
        }

        let simple_aggs: Vec<SimpleAgg> = simple_aggs.into_iter().map(|a| a.unwrap()).collect();

        // Fast path: single-pass streaming aggregation
        // Store aggregate state directly in hash map instead of row indices
        struct FastGroupState {
            key_values: Vec<Value>,
            agg_values: Vec<f64>,     // Running sums stored as f64
            agg_has_value: Vec<bool>, // Track if any non-NULL value was seen (for SUM)
            counts: Vec<i64>,         // For COUNT
        }

        // Pre-allocate hash map with estimated capacity to reduce resizing.
        // Estimate: for high-cardinality groupings, assume ~1/3 of rows are unique groups.
        // For low-cardinality, this over-allocates but that's fine.
        // CRITICAL: Use Vec to handle hash collisions (multiple groups per hash)
        let estimated_groups = (rows.len() / 3).max(64);
        let mut groups: FxHashMap<u64, Vec<FastGroupState>> =
            FxHashMap::with_capacity_and_hasher(estimated_groups, Default::default());
        let num_aggs = simple_aggs.len();

        // Track for early termination optimization
        let group_limit = limit.unwrap_or(usize::MAX);
        let has_limit = limit.is_some();
        let mut current_group_count: usize = 0; // Track actual group count for O(1) LIMIT checks

        for row in rows {
            // Build key values for this row (needed for collision detection)
            let key_values: Vec<Value> = group_by_indices
                .iter()
                .map(|&idx| row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                .collect();

            // Hash the group key with AHasher (optimal for Value types)
            let mut hasher = AHasher::default();
            for value in &key_values {
                hash_value_into(value, &mut hasher);
            }
            let hash = hasher.finish();

            // Early termination: check if this key exists before checking limit
            // CRITICAL: Must check actual key equality, not just hash (handle collisions)
            let key_exists = groups
                .get(&hash)
                .map(|bucket| bucket.iter().any(|state| state.key_values == key_values))
                .unwrap_or(false);

            if has_limit && !key_exists && current_group_count >= group_limit {
                // Would create a new group, but we're at limit - skip this row
                continue;
            }

            // Get or create bucket for this hash
            let bucket = groups.entry(hash).or_default();

            // Find existing group or create new one (handles hash collisions)
            let state = if let Some(state) = bucket.iter_mut().find(|s| s.key_values == key_values)
            {
                // Existing group - reuse it
                state
            } else {
                // New group - create and add to bucket
                bucket.push(FastGroupState {
                    key_values,
                    agg_values: vec![0.0; num_aggs],
                    agg_has_value: vec![false; num_aggs],
                    counts: vec![0; num_aggs],
                });
                current_group_count += 1; // Track for O(1) LIMIT checks
                bucket.last_mut().unwrap()
            };

            // Accumulate aggregates
            for (i, agg) in simple_aggs.iter().enumerate() {
                match agg {
                    SimpleAgg::Count => {
                        state.counts[i] += 1;
                    }
                    SimpleAgg::Sum(col_idx) => {
                        if let Some(value) = row.get(*col_idx) {
                            match value {
                                Value::Integer(v) => {
                                    state.agg_values[i] += *v as f64;
                                    state.agg_has_value[i] = true;
                                }
                                Value::Float(v) => {
                                    state.agg_values[i] += v;
                                    state.agg_has_value[i] = true;
                                }
                                _ => {} // Skip non-numeric and NULL
                            }
                        }
                    }
                }
            }
        }

        // Build result columns
        let mut result_columns = Vec::with_capacity(group_by_items.len() + aggregations.len());

        // Add GROUP BY column names (use column index to get actual name)
        for (i, item) in group_by_items.iter().enumerate() {
            let name = match item {
                GroupByItem::Column(col_name) => col_name.clone(),
                _ => format!("col{}", i),
            };
            result_columns.push(name);
        }

        // Add aggregate column names
        for agg in aggregations {
            let col_name = if let Some(ref alias) = agg.alias {
                alias.clone()
            } else {
                agg.get_expression_name()
            };
            result_columns.push(col_name);
        }

        // Build result rows - flatten buckets (each bucket may have multiple groups due to collisions)
        let result_rows: Vec<Row> = groups
            .into_values()
            .flatten() // Flatten Vec<FastGroupState> from each bucket
            .map(|state| {
                let mut values = Vec::with_capacity(group_by_indices.len() + simple_aggs.len());
                values.extend(state.key_values);

                for (i, agg) in simple_aggs.iter().enumerate() {
                    let value = match agg {
                        SimpleAgg::Count => Value::Integer(state.counts[i]),
                        SimpleAgg::Sum(_) => {
                            // SUM returns NULL if no non-NULL values were seen
                            if state.agg_has_value[i] {
                                Value::Float(state.agg_values[i])
                            } else {
                                Value::null_unknown()
                            }
                        }
                    };
                    values.push(value);
                }

                Row::from_values(values)
            })
            .collect();

        Ok(Some((result_columns, result_rows)))
    }

    /// Execute grouped aggregation (with GROUP BY)
    ///
    /// Optimized version that:
    /// 1. Uses hash-based grouping instead of Vec<Value> keys
    /// 2. Pre-allocates aggregate functions once, resets per group
    /// 3. Pre-computes column indices for aggregate columns
    /// 4. Supports complex expressions in GROUP BY (e.g., function calls)
    ///
    /// When `limit` is provided (and there's no ORDER BY), enables early termination
    /// for streaming aggregation - stops creating new groups after limit is reached.
    #[allow(clippy::too_many_arguments)]
    fn execute_grouped_aggregation(
        &self,
        aggregations: &[SqlAggregateFunction],
        group_by_items: &[GroupByItem],
        rows: &[Row],
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        _stmt: &SelectStatement,
        ctx: &ExecutionContext,
        limit: Option<usize>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // FAST PATH: For simple aggregates (SUM, COUNT without DISTINCT/FILTER/ORDER BY/expression),
        // use single-pass streaming aggregation that accumulates values directly
        if let Some(result) = self.try_fast_aggregation(
            aggregations,
            group_by_items,
            rows,
            columns,
            col_index_map,
            limit,
        )? {
            return Ok(result);
        }

        // Check if any aggregation has an expression (e.g., SUM(val * 2)), ORDER BY, or FILTER
        let has_agg_expression = aggregations
            .iter()
            .any(|a| a.expression.is_some() || !a.order_by.is_empty() || a.filter.is_some());

        // Pre-compute aggregate column indices (once, not per row)
        // OPTIMIZATION: Use pre-computed column_lower instead of calling to_lowercase() each time
        // Handle both qualified (e.g., "o.amount") and unqualified column names
        let agg_col_indices: Vec<Option<usize>> = aggregations
            .iter()
            .map(|agg| {
                if agg.column == "*" || agg.expression.is_some() {
                    None // COUNT(*) and expressions don't use column index
                } else {
                    Self::lookup_column_index(&agg.column_lower, col_index_map)
                }
            })
            .collect();

        // Use hash-based grouping with collision handling: u64 hash -> Vec<GroupEntry>
        // Each hash bucket can contain multiple groups (handles hash collisions correctly)
        // Uses u64 keys for performance (8 bytes vs hundreds of bytes for Vec<Value>)
        // FxHashMap is optimized for trusted keys in embedded database context
        let mut groups: FxHashMap<u64, Vec<GroupEntry>> = FxHashMap::default();

        // Temporary buffer for computing group key hash (reused across rows)
        let mut key_buffer: Vec<Value> = Vec::with_capacity(group_by_items.len());

        // OPTIMIZATION: Pre-compute column indices for GROUP BY items to avoid to_lowercase() per row
        enum PrecomputedGroupBy<'a> {
            ColumnIndex(usize),
            Position(usize),
            Expression(&'a Expression),
            NotFound,
        }

        let precomputed_group_by: Vec<PrecomputedGroupBy> = group_by_items
            .iter()
            .map(|item| match item {
                GroupByItem::Column(col_name) => {
                    // Use lookup_column_index to handle qualified names (e.g., "t.dept" -> "dept")
                    if let Some(idx) =
                        Self::lookup_column_index(&col_name.to_lowercase(), col_index_map)
                    {
                        PrecomputedGroupBy::ColumnIndex(idx)
                    } else {
                        PrecomputedGroupBy::NotFound
                    }
                }
                GroupByItem::Position(pos) => PrecomputedGroupBy::Position(pos.saturating_sub(1)),
                GroupByItem::Expression { expr, .. } => PrecomputedGroupBy::Expression(expr),
            })
            .collect();

        // OPTIMIZATION: Check if we have any Expression GROUP BY items
        // If so, pre-compile expressions and use VM for evaluation
        let has_expr_group_by = precomputed_group_by
            .iter()
            .any(|item| matches!(item, PrecomputedGroupBy::Expression(_)));

        // Pre-compile GROUP BY expressions for VM-based evaluation
        // CRITICAL: Propagate errors instead of silently ignoring compilation failures
        use super::expression::{compile_expression, ExecuteContext, ExprVM, SharedProgram};
        let compiled_group_by_exprs: Vec<Option<SharedProgram>> = precomputed_group_by
            .iter()
            .map(|item| match item {
                PrecomputedGroupBy::Expression(expr) => compile_expression(expr, columns).map(Some),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        let mut expr_vm = if has_expr_group_by || has_agg_expression {
            Some(ExprVM::new())
        } else {
            None
        };

        // OPTIMIZATION: Check if all GROUP BY items are simple column indices
        // In this case, we can hash directly from the row without cloning
        let all_simple_columns = precomputed_group_by.iter().all(|item| {
            matches!(
                item,
                PrecomputedGroupBy::ColumnIndex(_) | PrecomputedGroupBy::Position(_)
            )
        });

        // Track for early termination optimization
        let group_limit = limit.unwrap_or(usize::MAX);
        let has_limit = limit.is_some();
        let mut current_group_count: usize = 0; // Track actual group count for LIMIT optimization

        if all_simple_columns && expr_vm.is_none() {
            // Fast path: extract key values directly from row columns
            let column_indices: Vec<usize> = precomputed_group_by
                .iter()
                .map(|item| match item {
                    PrecomputedGroupBy::ColumnIndex(idx) => *idx,
                    PrecomputedGroupBy::Position(idx) => *idx,
                    _ => unreachable!(),
                })
                .collect();

            // OPTIMIZATION: Single-column GROUP BY uses direct hash map (no Vec<Value> overhead)
            if column_indices.len() == 1 {
                let col_idx = column_indices[0];
                // Use value hash directly as key, store (key_value, row_indices)
                let mut single_col_groups: FxHashMap<u64, (Value, Vec<usize>)> =
                    FxHashMap::default();

                for (row_idx, row) in rows.iter().enumerate() {
                    let key_value = row
                        .get(col_idx)
                        .cloned()
                        .unwrap_or_else(Value::null_unknown);
                    let hash = {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = rustc_hash::FxHasher::default();
                        key_value.hash(&mut hasher);
                        hasher.finish()
                    };

                    // For single column, hash collisions are rare - use simple entry API
                    single_col_groups
                        .entry(hash)
                        .and_modify(|e| e.1.push(row_idx))
                        .or_insert_with(|| (key_value, vec![row_idx]));
                }

                // Convert to GroupEntry format for downstream processing
                for (_hash, (key_value, row_indices)) in single_col_groups {
                    groups
                        .entry(0) // Use dummy hash, we'll flatten anyway
                        .or_default()
                        .push(GroupEntry {
                            key_values: vec![key_value],
                            row_indices,
                        });
                }
            } else {
                for (row_idx, row) in rows.iter().enumerate() {
                    // Build key values for this row
                    key_buffer.clear();
                    for &idx in &column_indices {
                        key_buffer.push(row.get(idx).cloned().unwrap_or_else(Value::null_unknown));
                    }

                    // Compute hash once (8-byte key instead of cloning entire Vec<Value>)
                    let hash = hash_group_key(&key_buffer);

                    // Early termination: if we've reached the limit, check before adding new groups
                    // CRITICAL: Must check if key exists in bucket to distinguish new group vs existing group
                    // Count actual groups, not buckets (hash collisions create multiple groups per bucket)
                    if has_limit {
                        let key_exists_in_bucket = groups
                            .get(&hash)
                            .map(|bucket| bucket.iter().any(|entry| entry.key_values == key_buffer))
                            .unwrap_or(false);

                        if !key_exists_in_bucket && current_group_count >= group_limit {
                            // This would create a new group, but we're at limit - skip
                            continue;
                        }
                    }

                    // Handle bucket with proper collision detection
                    match groups.entry(hash) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            let bucket = e.get_mut();
                            // Search bucket for matching group (handles hash collisions)
                            if let Some(entry) = bucket
                                .iter_mut()
                                .find(|entry| entry.key_values == key_buffer)
                            {
                                // Existing group - just add this row to it
                                entry.row_indices.push(row_idx);
                            } else {
                                // Hash collision: different key with same hash - add new group
                                // (limit already checked above)
                                bucket.push(GroupEntry {
                                    key_values: key_buffer.clone(),
                                    row_indices: vec![row_idx],
                                });
                                current_group_count += 1; // Track new group for LIMIT optimization
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            // First entry for this hash (limit already checked above)
                            e.insert(vec![GroupEntry {
                                key_values: key_buffer.clone(),
                                row_indices: vec![row_idx],
                            }]);
                            current_group_count += 1; // Track new group for LIMIT optimization
                        }
                    }
                }
            }
        } else {
            // Slow path: need to evaluate expressions, use buffer
            for (row_idx, row) in rows.iter().enumerate() {
                key_buffer.clear();

                // Create execution context for this row
                // CRITICAL: Include params for parameterized queries
                let row_data = row.as_slice();
                let exec_ctx = ExecuteContext::new(row_data)
                    .with_params(ctx.params())
                    .with_named_params(ctx.named_params());

                for (i, item) in precomputed_group_by.iter().enumerate() {
                    let value = match item {
                        PrecomputedGroupBy::ColumnIndex(idx) => {
                            row.get(*idx).cloned().unwrap_or_else(Value::null_unknown)
                        }
                        PrecomputedGroupBy::Position(idx) => {
                            row.get(*idx).cloned().unwrap_or_else(Value::null_unknown)
                        }
                        PrecomputedGroupBy::Expression(_) => {
                            // Use pre-compiled expression with VM
                            if let (Some(ref mut vm), Some(ref program)) =
                                (&mut expr_vm, &compiled_group_by_exprs[i])
                            {
                                vm.execute(program, &exec_ctx)
                                    .unwrap_or_else(|_| Value::null_unknown())
                            } else {
                                Value::null_unknown()
                            }
                        }
                        PrecomputedGroupBy::NotFound => Value::null_unknown(),
                    };
                    key_buffer.push(value);
                }

                // Compute hash of key (8-byte key for fast lookups)
                let hash = hash_group_key(&key_buffer);

                // Early termination: if we've reached the limit, check before adding new groups
                // CRITICAL: Must check if key exists in bucket to distinguish new group vs existing group
                // Count actual groups, not buckets (hash collisions create multiple groups per bucket)
                if has_limit {
                    let key_exists_in_bucket = groups
                        .get(&hash)
                        .map(|bucket| bucket.iter().any(|entry| entry.key_values == key_buffer))
                        .unwrap_or(false);

                    if !key_exists_in_bucket && current_group_count >= group_limit {
                        // This would create a new group, but we're at limit - skip
                        continue;
                    }
                }

                // Handle bucket with proper collision detection
                match groups.entry(hash) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        let bucket = e.get_mut();
                        // Search bucket for matching group (handles hash collisions)
                        if let Some(entry) = bucket
                            .iter_mut()
                            .find(|entry| entry.key_values == key_buffer)
                        {
                            // Existing group - just add this row to it
                            entry.row_indices.push(row_idx);
                        } else {
                            // Hash collision: different key with same hash - add new group
                            // (limit already checked above)
                            bucket.push(GroupEntry {
                                key_values: key_buffer.clone(),
                                row_indices: vec![row_idx],
                            });
                            current_group_count += 1; // Track new group for LIMIT optimization
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        // First entry for this hash (limit already checked above)
                        e.insert(vec![GroupEntry {
                            key_values: key_buffer.clone(),
                            row_indices: vec![row_idx],
                        }]);
                        current_group_count += 1; // Track new group for LIMIT optimization
                    }
                }
            }
        }

        // Convert groups to Vec for parallel processing
        // Flatten buckets: each bucket may contain multiple groups (hash collisions)
        let groups_vec: Vec<GroupEntry> = groups
            .into_iter()
            .flat_map(|(_hash, bucket)| bucket)
            .collect();

        // Pre-compile aggregate filter and expression programs for VM-based evaluation
        // CRITICAL: Propagate errors instead of silently ignoring compilation failures
        let compiled_agg_filters: Vec<Option<SharedProgram>> = if has_agg_expression {
            aggregations
                .iter()
                .map(|agg| {
                    agg.filter
                        .as_ref()
                        .map(|f| compile_expression(f, columns))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; aggregations.len()]
        };
        // CRITICAL: Propagate errors instead of silently ignoring compilation failures
        let compiled_agg_expressions: Vec<Option<SharedProgram>> = if has_agg_expression {
            aggregations
                .iter()
                .map(|agg| {
                    agg.expression
                        .as_ref()
                        .map(|e| compile_expression(e, columns))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; aggregations.len()]
        };
        // Pre-compile ORDER BY expressions for each aggregation
        // CRITICAL: Propagate errors instead of silently ignoring compilation failures
        let compiled_agg_order_by: Vec<Vec<SharedProgram>> = if has_agg_expression {
            aggregations
                .iter()
                .map(|agg| {
                    agg.order_by
                        .iter()
                        .map(|o| compile_expression(&o.expression, columns))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![Vec::new(); aggregations.len()]
        };

        // Determine if parallel processing is beneficial
        // Don't use parallel processing when expressions are involved (harder to handle)
        // Key insight: parallel creates aggregate functions PER GROUP, so for many small groups
        // (e.g., 10k groups with 3 rows each), the allocation overhead dominates.
        // Only parallelize when groups are large enough to amortize the allocation cost.
        let total_rows: usize = groups_vec.iter().map(|g| g.row_indices.len()).sum();
        let avg_rows_per_group = total_rows / groups_vec.len().max(1);
        let use_parallel = groups_vec.len() >= 4
            && total_rows >= 10_000
            && avg_rows_per_group >= 50
            && !has_agg_expression;

        // Process groups (parallel or sequential based on data size)
        let result_rows: Vec<Row> = if use_parallel {
            // PARALLEL: Process each group independently using Rayon
            let function_registry = &self.function_registry;

            groups_vec
                .into_par_iter()
                .map(|group| {
                    // Each thread creates its own aggregate functions
                    let mut agg_funcs: Vec<Option<Box<dyn AggregateFunction>>> = aggregations
                        .iter()
                        .map(|agg| function_registry.get_aggregate(&agg.name))
                        .collect();

                    // Configure aggregate functions with extra arguments (e.g., separator for STRING_AGG)
                    for (i, agg) in aggregations.iter().enumerate() {
                        if !agg.extra_args.is_empty() {
                            if let Some(ref mut func) = agg_funcs[i] {
                                func.configure(&agg.extra_args);
                            }
                        }
                    }

                    // Accumulate values for this group
                    // Pre-create static Value for COUNT(*)
                    let count_star_value = Value::Integer(1);
                    for &row_idx in &group.row_indices {
                        let row = &rows[row_idx];
                        for (i, agg) in aggregations.iter().enumerate() {
                            if let Some(ref mut func) = agg_funcs[i] {
                                // OPTIMIZATION: Avoid cloning by using reference directly
                                let value_ref = if let Some(col_idx) = agg_col_indices[i] {
                                    row.get(col_idx)
                                } else {
                                    Some(&count_star_value)
                                };
                                if let Some(v) = value_ref {
                                    func.accumulate(v, agg.distinct);
                                }
                            }
                        }
                    }

                    // Build result row
                    let mut row_values =
                        Vec::with_capacity(group_by_items.len() + aggregations.len());
                    row_values.extend(group.key_values);

                    for (i, agg) in aggregations.iter().enumerate() {
                        let value = if let Some(ref func) = agg_funcs[i] {
                            func.result()
                        } else if agg.name == "COUNT" && agg.column == "*" {
                            Value::Integer(group.row_indices.len() as i64)
                        } else {
                            Value::null_unknown()
                        };
                        row_values.push(value);
                    }

                    Row::from_values(row_values)
                })
                .collect()
        } else {
            // SEQUENTIAL: For small datasets, avoid parallel overhead, or when expressions are involved
            let mut agg_funcs: Vec<Option<Box<dyn AggregateFunction>>> = aggregations
                .iter()
                .map(|agg| self.function_registry.get_aggregate(&agg.name))
                .collect();

            // Configure aggregate functions with extra arguments (e.g., separator for STRING_AGG)
            // This is done once, not per group, as configuration persists across resets
            for (i, agg) in aggregations.iter().enumerate() {
                if !agg.extra_args.is_empty() {
                    if let Some(ref mut func) = agg_funcs[i] {
                        func.configure(&agg.extra_args);
                    }
                }
            }

            // Configure ORDER BY for ordered-set aggregates (ARRAY_AGG, STRING_AGG, etc.)
            for (i, agg) in aggregations.iter().enumerate() {
                if !agg.order_by.is_empty() {
                    if let Some(ref mut func) = agg_funcs[i] {
                        let directions: Vec<bool> = agg
                            .order_by
                            .iter()
                            .map(|o| o.ascending) // true = ASC, false = DESC
                            .collect();
                        func.set_order_by(directions);
                    }
                }
            }

            // Buffer for evaluated expression values (to avoid repeated allocation)
            let mut expr_values: Vec<Value> = vec![Value::null_unknown(); aggregations.len()];

            let mut result_rows_seq = Vec::with_capacity(groups_vec.len());
            for group in groups_vec {
                // Reset aggregate functions for this group
                for f in agg_funcs.iter_mut().flatten() {
                    f.reset();
                }

                // Accumulate values for this group
                // Pre-create static Value for COUNT(*)
                let count_star_value = Value::Integer(1);
                for &row_idx in &group.row_indices {
                    let row = &rows[row_idx];

                    // Create execution context for this row
                    // CRITICAL: Include params for parameterized queries
                    let row_data = row.as_slice();
                    let exec_ctx = ExecuteContext::new(row_data)
                        .with_params(ctx.params())
                        .with_named_params(ctx.named_params());

                    for (i, agg) in aggregations.iter().enumerate() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            // Check FILTER clause first - skip row if filter is false
                            if let Some(ref filter_program) = compiled_agg_filters[i] {
                                if let Some(ref mut vm) = expr_vm {
                                    match vm.execute(filter_program, &exec_ctx) {
                                        Ok(Value::Boolean(true)) => {} // Continue with accumulation
                                        Ok(Value::Boolean(false)) | Ok(Value::Null(_)) => continue, // Skip this row
                                        Ok(_) => continue, // Non-boolean treated as false
                                        Err(_) => continue, // Error treated as false
                                    }
                                } else {
                                    // Can't evaluate filter without VM - skip
                                    continue;
                                }
                            }

                            // Check if this aggregate has an expression to evaluate
                            let value = if let Some(ref expr_program) = compiled_agg_expressions[i]
                            {
                                // Evaluate the expression for this row using VM
                                if let Some(ref mut vm) = expr_vm {
                                    match vm.execute(expr_program, &exec_ctx) {
                                        Ok(val) => {
                                            expr_values[i] = val;
                                            Some(&expr_values[i])
                                        }
                                        Err(e) => {
                                            // Expression evaluation failed - see comment above
                                            #[cfg(debug_assertions)]
                                            eprintln!(
                                                "Warning: aggregate expression evaluation failed: {}",
                                                e
                                            );
                                            let _ = e;
                                            None
                                        }
                                    }
                                } else {
                                    None
                                }
                            } else {
                                // Simple column reference or COUNT(*)
                                if let Some(col_idx) = agg_col_indices[i] {
                                    row.get(col_idx)
                                } else {
                                    Some(&count_star_value)
                                }
                            };

                            if let Some(v) = value {
                                // Check if this aggregate has ORDER BY and supports it
                                if !compiled_agg_order_by[i].is_empty() && func.supports_order_by()
                                {
                                    // Evaluate ORDER BY expressions to get sort keys using pre-compiled programs
                                    if let Some(ref mut vm) = expr_vm {
                                        let mut sort_keys =
                                            Vec::with_capacity(compiled_agg_order_by[i].len());
                                        let mut all_ok = true;
                                        for order_program in &compiled_agg_order_by[i] {
                                            match vm.execute(order_program, &exec_ctx) {
                                                Ok(key) => sort_keys.push(key),
                                                Err(_) => {
                                                    all_ok = false;
                                                    break;
                                                }
                                            }
                                        }
                                        if all_ok {
                                            func.accumulate_with_sort_key(
                                                v,
                                                sort_keys,
                                                agg.distinct,
                                            );
                                        }
                                    } else {
                                        // No VM - fall back to regular accumulate
                                        func.accumulate(v, agg.distinct);
                                    }
                                } else {
                                    func.accumulate(v, agg.distinct);
                                }
                            }
                        }
                    }
                }

                // Build result row
                let mut row_values = Vec::with_capacity(group_by_items.len() + aggregations.len());
                row_values.extend(group.key_values);

                for (i, agg) in aggregations.iter().enumerate() {
                    let value = if let Some(ref func) = agg_funcs[i] {
                        func.result()
                    } else if agg.name == "COUNT" && agg.column == "*" {
                        Value::Integer(group.row_indices.len() as i64)
                    } else {
                        Value::null_unknown()
                    };
                    row_values.push(value);
                }

                result_rows_seq.push(Row::from_values(row_values));
            }
            result_rows_seq
        };

        // Build result columns
        let mut result_columns: Vec<String> =
            self.resolve_group_by_column_names_new(group_by_items, columns, col_index_map);
        result_columns.extend(aggregations.iter().map(|a| a.get_column_name()));

        Ok((result_columns, result_rows))
    }

    /// Execute ROLLUP/CUBE aggregation
    ///
    /// ROLLUP(a, b, c) generates grouping sets:
    ///   (a, b, c) - most detailed
    ///   (a, b)    - subtotal for a, b
    ///   (a)       - subtotal for a
    ///   ()        - grand total
    ///
    /// CUBE(a, b) generates all combinations:
    ///   (a, b), (a), (b), ()
    #[allow(clippy::too_many_arguments)]
    fn execute_rollup_aggregation(
        &self,
        aggregations: &[SqlAggregateFunction],
        group_by_items: &[GroupByItem],
        rows: &[Row],
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // Generate grouping sets based on modifier type
        let grouping_sets = match &stmt.group_by.modifier {
            GroupByModifier::Rollup => Self::generate_rollup_sets(group_by_items.len()),
            GroupByModifier::Cube => Self::generate_cube_sets(group_by_items.len()),
            GroupByModifier::GroupingSets(sets) => {
                Self::generate_explicit_grouping_sets(sets, group_by_items)
            }
            GroupByModifier::None => {
                // Shouldn't happen, but handle it
                vec![GroupingSet {
                    active_columns: vec![true; group_by_items.len()],
                }]
            }
        };

        // Check if any aggregation has an expression (e.g., SUM(val * 2)) or ORDER BY
        let has_agg_expression = aggregations
            .iter()
            .any(|a| a.expression.is_some() || !a.order_by.is_empty());

        // Pre-compute aggregate column indices
        let agg_col_indices: Vec<Option<usize>> = aggregations
            .iter()
            .map(|agg| {
                if agg.column == "*" || agg.expression.is_some() {
                    None
                } else {
                    Self::lookup_column_index(&agg.column_lower, col_index_map)
                }
            })
            .collect();

        // Pre-compute GROUP BY column indices
        enum PrecomputedGroupBy<'a> {
            ColumnIndex(usize),
            Position(usize),
            Expression(&'a Expression),
            NotFound,
        }

        let precomputed_group_by: Vec<PrecomputedGroupBy> = group_by_items
            .iter()
            .map(|item| match item {
                GroupByItem::Column(col_name) => {
                    if let Some(idx) =
                        Self::lookup_column_index(&col_name.to_lowercase(), col_index_map)
                    {
                        PrecomputedGroupBy::ColumnIndex(idx)
                    } else {
                        PrecomputedGroupBy::NotFound
                    }
                }
                GroupByItem::Position(pos) => PrecomputedGroupBy::Position(pos.saturating_sub(1)),
                GroupByItem::Expression { expr, .. } => PrecomputedGroupBy::Expression(expr),
            })
            .collect();

        // Pre-compile GROUP BY and aggregate expressions for VM-based evaluation
        let has_expr_group_by = precomputed_group_by
            .iter()
            .any(|item| matches!(item, PrecomputedGroupBy::Expression(_)));

        // Pre-compile GROUP BY expressions
        // CRITICAL: Propagate errors instead of silently ignoring compilation failures
        use super::expression::{compile_expression, ExecuteContext, ExprVM, SharedProgram};
        let compiled_group_by_exprs: Vec<Option<SharedProgram>> = precomputed_group_by
            .iter()
            .map(|item| match item {
                PrecomputedGroupBy::Expression(expr) => compile_expression(expr, columns).map(Some),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;

        // Pre-compile aggregate expressions
        // CRITICAL: Propagate errors instead of silently ignoring compilation failures
        let compiled_agg_expressions: Vec<Option<SharedProgram>> = if has_agg_expression {
            aggregations
                .iter()
                .map(|agg| {
                    agg.expression
                        .as_ref()
                        .map(|e| compile_expression(e, columns))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![None; aggregations.len()]
        };

        let mut expr_vm = if has_expr_group_by || has_agg_expression {
            Some(ExprVM::new())
        } else {
            None
        };

        // Collect all results from all grouping sets
        let mut all_result_rows: Vec<Row> = Vec::new();

        for grouping_set in &grouping_sets {
            // Count active columns in this grouping set
            let active_count = grouping_set.active_columns.iter().filter(|&&x| x).count();

            if active_count == 0 {
                // Grand total: aggregate all rows without grouping
                let mut agg_funcs: Vec<Option<Box<dyn AggregateFunction>>> = aggregations
                    .iter()
                    .map(|agg| self.function_registry.get_aggregate(&agg.name))
                    .collect();

                // Configure aggregate functions
                for (i, agg) in aggregations.iter().enumerate() {
                    if !agg.extra_args.is_empty() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            func.configure(&agg.extra_args);
                        }
                    }
                }

                let count_star_value = Value::Integer(1);
                let mut expr_values: Vec<Value> = vec![Value::null_unknown(); aggregations.len()];

                for row in rows {
                    // Create execution context for this row
                    // CRITICAL: Include params for parameterized queries
                    let row_data = row.as_slice();
                    let exec_ctx = ExecuteContext::new(row_data)
                        .with_params(ctx.params())
                        .with_named_params(ctx.named_params());

                    for (i, agg) in aggregations.iter().enumerate() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            if let Some(ref expr_program) = compiled_agg_expressions[i] {
                                if let Some(ref mut vm) = expr_vm {
                                    if let Ok(val) = vm.execute(expr_program, &exec_ctx) {
                                        expr_values[i] = val;
                                        func.accumulate(&expr_values[i], agg.distinct);
                                    }
                                }
                            } else {
                                let value_ref = if let Some(col_idx) = agg_col_indices[i] {
                                    row.get(col_idx)
                                } else {
                                    Some(&count_star_value)
                                };
                                if let Some(v) = value_ref {
                                    func.accumulate(v, agg.distinct);
                                }
                            }
                        }
                    }
                }

                // Build result row: all GROUP BY columns are NULL, then aggregates, then grouping flags
                let mut row_values = Vec::with_capacity(
                    group_by_items.len() + aggregations.len() + group_by_items.len(),
                );
                for _ in group_by_items {
                    row_values.push(Value::null_unknown());
                }
                for (i, agg) in aggregations.iter().enumerate() {
                    let value = if let Some(ref func) = agg_funcs[i] {
                        func.result()
                    } else if agg.name == "COUNT" && agg.column == "*" {
                        Value::Integer(rows.len() as i64)
                    } else {
                        Value::null_unknown()
                    };
                    row_values.push(value);
                }
                // Add GROUPING flags: all columns are rolled up in grand total (GROUPING = 1)
                for _ in group_by_items {
                    row_values.push(Value::Integer(1));
                }
                all_result_rows.push(Row::from_values(row_values));
            } else {
                // Partial grouping: group by active columns only
                // Use hash-based grouping with collision handling: u64 hash -> Vec<GroupEntry>
                // Each hash bucket can contain multiple groups (handles hash collisions correctly)
                // FxHashMap is optimized for trusted keys in embedded database context
                let mut groups: FxHashMap<u64, Vec<GroupEntry>> = FxHashMap::default();
                let mut key_buffer: Vec<Value> = Vec::with_capacity(active_count);

                for (row_idx, row) in rows.iter().enumerate() {
                    key_buffer.clear();

                    // Create execution context for this row
                    // CRITICAL: Include params for parameterized queries
                    let row_data = row.as_slice();
                    let exec_ctx = ExecuteContext::new(row_data)
                        .with_params(ctx.params())
                        .with_named_params(ctx.named_params());

                    // Only include active columns in the key
                    for (col_idx, &is_active) in grouping_set.active_columns.iter().enumerate() {
                        if is_active {
                            let value = match &precomputed_group_by[col_idx] {
                                PrecomputedGroupBy::ColumnIndex(idx) => {
                                    row.get(*idx).cloned().unwrap_or_else(Value::null_unknown)
                                }
                                PrecomputedGroupBy::Position(idx) => {
                                    row.get(*idx).cloned().unwrap_or_else(Value::null_unknown)
                                }
                                PrecomputedGroupBy::Expression(_) => {
                                    // Use pre-compiled expression with VM
                                    if let (Some(ref mut vm), Some(ref program)) =
                                        (&mut expr_vm, &compiled_group_by_exprs[col_idx])
                                    {
                                        vm.execute(program, &exec_ctx)
                                            .unwrap_or_else(|_| Value::null_unknown())
                                    } else {
                                        Value::null_unknown()
                                    }
                                }
                                PrecomputedGroupBy::NotFound => Value::null_unknown(),
                            };
                            key_buffer.push(value);
                        }
                    }

                    // Compute hash and handle bucket with proper collision detection
                    let hash = hash_group_key(&key_buffer);
                    match groups.entry(hash) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            let bucket = e.get_mut();
                            // Search bucket for matching group (handles hash collisions)
                            if let Some(entry) = bucket
                                .iter_mut()
                                .find(|entry| entry.key_values == key_buffer)
                            {
                                entry.row_indices.push(row_idx);
                            } else {
                                // Hash collision: different key with same hash
                                bucket.push(GroupEntry {
                                    key_values: key_buffer.clone(),
                                    row_indices: vec![row_idx],
                                });
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            // First entry for this hash
                            e.insert(vec![GroupEntry {
                                key_values: key_buffer.clone(),
                                row_indices: vec![row_idx],
                            }]);
                        }
                    }
                }

                // Process each group
                let mut agg_funcs: Vec<Option<Box<dyn AggregateFunction>>> = aggregations
                    .iter()
                    .map(|agg| self.function_registry.get_aggregate(&agg.name))
                    .collect();

                // Configure aggregate functions
                for (i, agg) in aggregations.iter().enumerate() {
                    if !agg.extra_args.is_empty() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            func.configure(&agg.extra_args);
                        }
                    }
                }

                let mut expr_values: Vec<Value> = vec![Value::null_unknown(); aggregations.len()];

                // Note: We don't sort groups here for performance.
                // SQL does not guarantee result order without ORDER BY.
                // Users who need ordered results should add ORDER BY to their query.
                // Flatten buckets: each bucket may contain multiple groups (hash collisions)
                for (_hash, bucket) in groups {
                    for group in bucket {
                        // Reset aggregate functions
                        for f in agg_funcs.iter_mut().flatten() {
                            f.reset();
                        }

                        let count_star_value = Value::Integer(1);
                        for &row_idx in &group.row_indices {
                            let row = &rows[row_idx];

                            // Create execution context for this row
                            // CRITICAL: Include params for parameterized queries
                            let row_data = row.as_slice();
                            let exec_ctx = ExecuteContext::new(row_data)
                                .with_params(ctx.params())
                                .with_named_params(ctx.named_params());

                            for (i, agg) in aggregations.iter().enumerate() {
                                if let Some(ref mut func) = agg_funcs[i] {
                                    if let Some(ref expr_program) = compiled_agg_expressions[i] {
                                        if let Some(ref mut vm) = expr_vm {
                                            if let Ok(val) = vm.execute(expr_program, &exec_ctx) {
                                                expr_values[i] = val;
                                                func.accumulate(&expr_values[i], agg.distinct);
                                            }
                                        }
                                    } else {
                                        let value_ref = if let Some(col_idx) = agg_col_indices[i] {
                                            row.get(col_idx)
                                        } else {
                                            Some(&count_star_value)
                                        };
                                        if let Some(v) = value_ref {
                                            func.accumulate(v, agg.distinct);
                                        }
                                    }
                                }
                            }
                        }

                        // Build result row
                        // For GROUP BY columns: use key value if active, NULL if rolled up
                        let mut row_values = Vec::with_capacity(
                            group_by_items.len() + aggregations.len() + group_by_items.len(),
                        );

                        let mut key_idx = 0;
                        for &is_active in &grouping_set.active_columns {
                            if is_active {
                                row_values.push(group.key_values[key_idx].clone());
                                key_idx += 1;
                            } else {
                                row_values.push(Value::null_unknown());
                            }
                        }

                        for (i, agg) in aggregations.iter().enumerate() {
                            let value = if let Some(ref func) = agg_funcs[i] {
                                func.result()
                            } else if agg.name == "COUNT" && agg.column == "*" {
                                Value::Integer(group.row_indices.len() as i64)
                            } else {
                                Value::null_unknown()
                            };
                            row_values.push(value);
                        }

                        // Add GROUPING flags: 0 if column is active (grouped), 1 if rolled up
                        for &is_active in &grouping_set.active_columns {
                            row_values.push(Value::Integer(if is_active { 0 } else { 1 }));
                        }

                        all_result_rows.push(Row::from_values(row_values));
                    } // end for group in bucket
                } // end for bucket in groups
            }
        }

        // Build result columns
        let mut result_columns: Vec<String> =
            self.resolve_group_by_column_names_new(group_by_items, columns, col_index_map);
        result_columns.extend(aggregations.iter().map(|a| a.get_column_name()));

        // Add hidden grouping flag columns for GROUPING() function support
        // These will be used by the projection phase and stripped from final output
        for i in 0..group_by_items.len() {
            result_columns.push(format!("__grouping_{}__", i));
        }

        Ok((result_columns, all_result_rows))
    }

    /// Generate grouping sets for ROLLUP
    /// ROLLUP(a, b, c) generates: (a,b,c), (a,b), (a), ()
    fn generate_rollup_sets(num_columns: usize) -> Vec<GroupingSet> {
        let mut sets = Vec::with_capacity(num_columns + 1);

        // From most specific to least specific (grand total)
        for active_count in (0..=num_columns).rev() {
            let mut active_columns = vec![false; num_columns];
            for item in active_columns.iter_mut().take(active_count) {
                *item = true;
            }
            sets.push(GroupingSet { active_columns });
        }

        sets
    }

    /// Generate grouping sets for CUBE
    /// CUBE(a, b) generates: (a,b), (a), (b), ()
    fn generate_cube_sets(num_columns: usize) -> Vec<GroupingSet> {
        let mut sets = Vec::with_capacity(1 << num_columns);

        // Generate all 2^n combinations
        for mask in (0..(1 << num_columns)).rev() {
            let mut active_columns = vec![false; num_columns];
            for (i, item) in active_columns.iter_mut().enumerate() {
                if mask & (1 << (num_columns - 1 - i)) != 0 {
                    *item = true;
                }
            }
            sets.push(GroupingSet { active_columns });
        }

        sets
    }

    /// Generate grouping sets from explicit GROUPING SETS clause
    /// GROUPING SETS ((a, b), (a), ()) generates exactly those three sets
    fn generate_explicit_grouping_sets(
        sets: &[Vec<Expression>],
        group_by_items: &[GroupByItem],
    ) -> Vec<GroupingSet> {
        let num_columns = group_by_items.len();
        let mut result = Vec::with_capacity(sets.len());

        // Build a lookup from canonical key to group_by_items index
        // Uses the same canonical key function for consistent matching
        let item_to_index: FxHashMap<String, usize> = group_by_items
            .iter()
            .enumerate()
            .map(|(i, item)| (group_by_item_canonical_key(item), i))
            .collect();

        for set in sets {
            let mut active_columns = vec![false; num_columns];

            for expr in set {
                // Get the canonical key for this expression
                let key = expression_canonical_key(expr);

                // Find the index in group_by_items
                if let Some(&idx) = item_to_index.get(&key) {
                    active_columns[idx] = true;
                }
            }

            result.push(GroupingSet { active_columns });
        }

        result
    }

    /// Resolve GROUP BY column names for result (new version supporting GroupByItem)
    fn resolve_group_by_column_names_new(
        &self,
        group_by_items: &[GroupByItem],
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
    ) -> Vec<String> {
        let mut names = Vec::new();
        for item in group_by_items {
            match item {
                GroupByItem::Column(col_name) => {
                    // Use lookup_column_index to handle qualified names (e.g., "t.dept" -> "dept")
                    if let Some(idx) =
                        Self::lookup_column_index(&col_name.to_lowercase(), col_index_map)
                    {
                        if idx < columns.len() {
                            names.push(columns[idx].clone());
                        } else {
                            names.push(col_name.clone());
                        }
                    } else {
                        names.push(col_name.clone());
                    }
                }
                GroupByItem::Position(pos) => {
                    // Position is 1-indexed
                    let idx = pos.saturating_sub(1);
                    if idx < columns.len() {
                        names.push(columns[idx].clone());
                    } else {
                        names.push(format!("${}", pos));
                    }
                }
                GroupByItem::Expression { display_name, .. } => {
                    // Use the display name (alias or generated name)
                    names.push(display_name.clone());
                }
            }
        }
        names
    }

    /// Look up column index, handling both qualified (e.g., "o.amount") and unqualified names.
    /// If a qualified name lookup fails, tries the unqualified part (after the dot).
    fn lookup_column_index(
        column_lower: &str,
        col_index_map: &FxHashMap<String, usize>,
    ) -> Option<usize> {
        // First try exact match
        if let Some(&idx) = col_index_map.get(column_lower) {
            return Some(idx);
        }

        // If it's a qualified name (contains a dot), try the unqualified part
        if let Some(dot_pos) = column_lower.rfind('.') {
            let unqualified = &column_lower[dot_pos + 1..];
            if let Some(&idx) = col_index_map.get(unqualified) {
                return Some(idx);
            }
        }

        None
    }

    /// Get column value from row
    #[allow(dead_code)]
    fn get_column_value(
        &self,
        row: &Row,
        column: &str,
        columns: &[String],
        col_index_map: &FxHashMap<String, usize>,
    ) -> Value {
        if column == "*" {
            // For COUNT(*), return a non-null value
            Value::Integer(1)
        } else if let Some(idx) = Self::lookup_column_index(&column.to_lowercase(), col_index_map) {
            row.get(idx).cloned().unwrap_or_else(Value::null_unknown)
        } else {
            // Try to find by case-insensitive match
            for (i, col) in columns.iter().enumerate() {
                if col.eq_ignore_ascii_case(column) {
                    return row.get(i).cloned().unwrap_or_else(Value::null_unknown);
                }
            }
            Value::null_unknown()
        }
    }

    /// Apply HAVING clause to aggregated results
    fn apply_having(
        &self,
        result: Box<dyn QueryResult>,
        having: &Expression,
        columns: &[String],
        agg_aliases: &[(String, usize)],
        expr_aliases: &[(String, usize)],
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Materialize the result
        let mut rows = Vec::new();
        let mut result = result;
        while result.next() {
            rows.push(result.take_row());
        }

        // Combine all aliases for HAVING clause evaluation
        let mut all_aliases: Vec<(String, usize)> = agg_aliases.to_vec();
        all_aliases.extend_from_slice(expr_aliases);

        // Create RowFilter with all aliases and context
        let having_filter =
            RowFilter::with_aliases(having, columns, &all_aliases)?.with_context(ctx);

        // Filter rows using the pre-compiled filter
        let filtered_rows: Vec<Row> = rows
            .into_iter()
            .filter(|row| having_filter.matches(row))
            .collect();

        Ok(Box::new(ExecutorMemoryResult::new(
            columns.to_vec(),
            filtered_rows,
        )))
    }

    /// Fast SUM implementation that bypasses the generic aggregate function
    /// Uses loop unrolling for better performance
    #[inline]
    fn fast_sum_column(&self, rows: &[Row], col_idx: usize) -> Value {
        if rows.is_empty() {
            return Value::null_unknown();
        }

        // Use parallel processing for large datasets
        if rows.len() >= 10_000 {
            return self.fast_sum_column_parallel(rows, col_idx);
        }

        let mut sum_int: i64 = 0;
        let mut sum_float: f64 = 0.0;
        let mut has_float = false;
        let mut has_value = false;

        // Unroll loop by 4 for better CPU pipelining
        let chunks = rows.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            for row in chunk {
                if let Some(val) = row.get(col_idx) {
                    match val {
                        Value::Integer(i) => {
                            has_value = true;
                            if has_float {
                                sum_float += *i as f64;
                            } else {
                                sum_int += i;
                            }
                        }
                        Value::Float(f) => {
                            has_value = true;
                            if !has_float {
                                has_float = true;
                                sum_float = sum_int as f64;
                            }
                            sum_float += f;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Handle remainder
        for row in remainder {
            if let Some(val) = row.get(col_idx) {
                match val {
                    Value::Integer(i) => {
                        has_value = true;
                        if has_float {
                            sum_float += *i as f64;
                        } else {
                            sum_int += i;
                        }
                    }
                    Value::Float(f) => {
                        has_value = true;
                        if !has_float {
                            has_float = true;
                            sum_float = sum_int as f64;
                        }
                        sum_float += f;
                    }
                    _ => {}
                }
            }
        }

        if !has_value {
            Value::null_unknown()
        } else if has_float {
            Value::Float(sum_float)
        } else {
            Value::Integer(sum_int)
        }
    }

    /// Parallel SUM implementation using Rayon
    #[inline]
    fn fast_sum_column_parallel(&self, rows: &[Row], col_idx: usize) -> Value {
        let chunk_size = (rows.len() / rayon::current_num_threads()).max(1000);

        // Process in parallel, collecting (sum_int, sum_float, has_float, has_value)
        let results: Vec<(i64, f64, bool, bool)> = rows
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut sum_int: i64 = 0;
                let mut sum_float: f64 = 0.0;
                let mut has_float = false;
                let mut has_value = false;

                for row in chunk {
                    if let Some(val) = row.get(col_idx) {
                        match val {
                            Value::Integer(i) => {
                                has_value = true;
                                if has_float {
                                    sum_float += *i as f64;
                                } else {
                                    sum_int += i;
                                }
                            }
                            Value::Float(f) => {
                                has_value = true;
                                if !has_float {
                                    has_float = true;
                                    sum_float = sum_int as f64;
                                }
                                sum_float += f;
                            }
                            _ => {}
                        }
                    }
                }

                (sum_int, sum_float, has_float, has_value)
            })
            .collect();

        // Merge results
        let mut total_int: i64 = 0;
        let mut total_float: f64 = 0.0;
        let mut any_float = false;
        let mut any_value = false;

        for (si, sf, hf, hv) in results {
            if hv {
                any_value = true;
                if hf || any_float {
                    any_float = true;
                    if hf {
                        total_float += sf;
                    } else {
                        total_float += si as f64;
                    }
                } else {
                    total_int += si;
                }
            }
        }

        // If we switched to float mid-way, add the integer total
        if any_float && total_int != 0 {
            total_float += total_int as f64;
        }

        if !any_value {
            Value::null_unknown()
        } else if any_float {
            Value::Float(total_float)
        } else {
            Value::Integer(total_int)
        }
    }

    /// Fast AVG implementation
    #[inline]
    fn fast_avg_column(&self, rows: &[Row], col_idx: usize) -> Value {
        if rows.is_empty() {
            return Value::null_unknown();
        }

        // Use parallel processing for large datasets
        if rows.len() >= 10_000 {
            return self.fast_avg_column_parallel(rows, col_idx);
        }

        let mut sum: f64 = 0.0;
        let mut count: i64 = 0;

        for row in rows {
            if let Some(val) = row.get(col_idx) {
                match val {
                    Value::Integer(i) => {
                        sum += *i as f64;
                        count += 1;
                    }
                    Value::Float(f) => {
                        sum += f;
                        count += 1;
                    }
                    _ => {}
                }
            }
        }

        if count == 0 {
            Value::null_unknown()
        } else {
            Value::Float(sum / count as f64)
        }
    }

    /// Parallel AVG implementation
    #[inline]
    fn fast_avg_column_parallel(&self, rows: &[Row], col_idx: usize) -> Value {
        let chunk_size = (rows.len() / rayon::current_num_threads()).max(1000);

        // Process in parallel, collecting (sum, count)
        let results: Vec<(f64, i64)> = rows
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut sum: f64 = 0.0;
                let mut count: i64 = 0;

                for row in chunk {
                    if let Some(val) = row.get(col_idx) {
                        match val {
                            Value::Integer(i) => {
                                sum += *i as f64;
                                count += 1;
                            }
                            Value::Float(f) => {
                                sum += f;
                                count += 1;
                            }
                            _ => {}
                        }
                    }
                }

                (sum, count)
            })
            .collect();

        // Merge results
        let mut total_sum: f64 = 0.0;
        let mut total_count: i64 = 0;

        for (s, c) in results {
            total_sum += s;
            total_count += c;
        }

        if total_count == 0 {
            Value::null_unknown()
        } else {
            Value::Float(total_sum / total_count as f64)
        }
    }

    /// Check if a SELECT statement has aggregation
    pub(crate) fn has_aggregation(&self, stmt: &SelectStatement) -> bool {
        // Check if there are aggregate functions in SELECT
        for col_expr in &stmt.columns {
            if self.expression_has_aggregation(col_expr) {
                return true;
            }
        }

        // Check if there's a GROUP BY clause
        !stmt.group_by.columns.is_empty()
    }

    /// Check if an expression contains aggregate functions (recursively)
    #[allow(clippy::only_used_in_recursion)]
    fn expression_has_aggregation(&self, expr: &Expression) -> bool {
        match expr {
            Expression::FunctionCall(func) => {
                // Check if this function is an aggregate
                if is_aggregate_function(&func.function) {
                    return true;
                }
                // Check arguments recursively (e.g., COALESCE(SUM(val), 0))
                for arg in &func.arguments {
                    if self.expression_has_aggregation(arg) {
                        return true;
                    }
                }
                false
            }
            Expression::Aliased(aliased) => self.expression_has_aggregation(&aliased.expression),
            Expression::Infix(infix) => {
                self.expression_has_aggregation(&infix.left)
                    || self.expression_has_aggregation(&infix.right)
            }
            Expression::Prefix(prefix) => self.expression_has_aggregation(&prefix.right),
            Expression::Cast(cast) => self.expression_has_aggregation(&cast.expr),
            Expression::Case(case) => {
                for when_clause in &case.when_clauses {
                    if self.expression_has_aggregation(&when_clause.condition)
                        || self.expression_has_aggregation(&when_clause.then_result)
                    {
                        return true;
                    }
                }
                if let Some(ref else_val) = case.else_value {
                    if self.expression_has_aggregation(else_val) {
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mvcc::engine::MVCCEngine;
    use std::sync::Arc;

    fn create_test_executor() -> Executor {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        Executor::new(Arc::new(engine))
    }

    fn setup_test_data(executor: &Executor) {
        executor
            .execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER)")
            .unwrap();
        executor
            .execute("INSERT INTO sales VALUES (1, 'electronics', 100)")
            .unwrap();
        executor
            .execute("INSERT INTO sales VALUES (2, 'electronics', 200)")
            .unwrap();
        executor
            .execute("INSERT INTO sales VALUES (3, 'clothing', 50)")
            .unwrap();
        executor
            .execute("INSERT INTO sales VALUES (4, 'clothing', 75)")
            .unwrap();
    }

    #[test]
    fn test_count_star() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor.execute("SELECT COUNT(*) FROM sales").unwrap();
        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(4)));
    }

    #[test]
    fn test_sum_column() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor.execute("SELECT SUM(amount) FROM sales").unwrap();
        assert!(result.next());
        let row = result.row();
        // SUM(100 + 200 + 50 + 75) = 425
        let value = row.get(0).unwrap();
        match value {
            Value::Integer(n) => assert_eq!(*n, 425),
            Value::Float(f) => assert!((f - 425.0).abs() < 0.01),
            _ => panic!("Expected numeric value, got {:?}", value),
        }
    }

    #[test]
    fn test_avg_column() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor.execute("SELECT AVG(amount) FROM sales").unwrap();
        assert!(result.next());
        let row = result.row();
        // AVG = 425 / 4 = 106.25
        let value = row.get(0).unwrap();
        match value {
            Value::Float(f) => assert!((f - 106.25).abs() < 0.01),
            Value::Integer(n) => assert_eq!(*n, 106), // Might truncate
            _ => panic!("Expected numeric value, got {:?}", value),
        }
    }

    #[test]
    fn test_min_max() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT MIN(amount), MAX(amount) FROM sales")
            .unwrap();
        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(50)));
        assert_eq!(row.get(1), Some(&Value::Integer(200)));
    }

    #[test]
    fn test_group_by_simple() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT category, SUM(amount) FROM sales GROUP BY category")
            .unwrap();

        let mut found = std::collections::HashMap::new();
        while result.next() {
            let row = result.row();
            if let Some(Value::Text(cat)) = row.get(0) {
                let sum = row.get(1).cloned().unwrap();
                found.insert(cat.clone(), sum);
            }
        }

        // electronics: 100 + 200 = 300
        // clothing: 50 + 75 = 125
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_aggregate_with_alias() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT COUNT(*) AS cnt FROM sales")
            .unwrap();
        let columns = result.columns();
        assert!(columns.contains(&"cnt".to_string()));

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(4)));
    }

    #[test]
    fn test_multiple_aggregates() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute(
                "SELECT COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM sales",
            )
            .unwrap();

        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(4))); // COUNT
    }

    #[test]
    fn test_is_aggregate_function() {
        assert!(is_aggregate_function("COUNT"));
        assert!(is_aggregate_function("count"));
        assert!(is_aggregate_function("SUM"));
        assert!(is_aggregate_function("AVG"));
        assert!(is_aggregate_function("MIN"));
        assert!(is_aggregate_function("MAX"));
        assert!(!is_aggregate_function("UPPER"));
        assert!(!is_aggregate_function("CONCAT"));
    }

    #[test]
    fn test_min_max_with_index_optimization() {
        // Test that MIN/MAX queries can use index optimization
        let executor = create_test_executor();

        // Create table with an index
        executor
            .execute("CREATE TABLE indexed_values (id INTEGER PRIMARY KEY, value INTEGER)")
            .unwrap();

        // Create index on the value column
        executor
            .execute("CREATE INDEX idx_value ON indexed_values (value)")
            .unwrap();

        // Insert some data
        executor
            .execute("INSERT INTO indexed_values VALUES (1, 100)")
            .unwrap();
        executor
            .execute("INSERT INTO indexed_values VALUES (2, 50)")
            .unwrap();
        executor
            .execute("INSERT INTO indexed_values VALUES (3, 200)")
            .unwrap();
        executor
            .execute("INSERT INTO indexed_values VALUES (4, 75)")
            .unwrap();
        executor
            .execute("INSERT INTO indexed_values VALUES (5, 150)")
            .unwrap();

        // Test MIN with index
        let mut result = executor
            .execute("SELECT MIN(value) FROM indexed_values")
            .unwrap();
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(50)));

        // Test MAX with index
        let mut result = executor
            .execute("SELECT MAX(value) FROM indexed_values")
            .unwrap();
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(200)));

        // Test MIN with alias
        let mut result = executor
            .execute("SELECT MIN(value) AS min_val FROM indexed_values")
            .unwrap();
        let columns = result.columns();
        assert!(columns.contains(&"min_val".to_string()));
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(50)));

        // Test MAX with alias
        let mut result = executor
            .execute("SELECT MAX(value) AS max_val FROM indexed_values")
            .unwrap();
        let columns = result.columns();
        assert!(columns.contains(&"max_val".to_string()));
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(200)));
    }
}
