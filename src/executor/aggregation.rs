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

use std::borrow::Cow;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::RwLock;

use ahash::AHasher;
use hashbrown::hash_map::RawEntryMut;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
// SmallVec removed - Vec is faster due to spilled() check overhead in hot loops

use crate::common::{CompactArc, CompactVec, I64Map, StringMap};
use crate::core::{Result, Row, RowVec, SumAccumulator, Value};
use crate::functions::aggregate::CompiledAggregate;
use crate::functions::AggregateFunction;
use crate::parser::ast::*;
use crate::storage::traits::transaction::StatementTable;
use crate::storage::traits::{Engine, QueryResult};

use super::context::ExecutionContext;
use super::expr_key::ExprKey;
use super::expression::{ExprVM, Op};
use super::query_cache::{CompiledCountDistinct, CompiledExecution};
use super::query_classification::QueryClassification;
use super::result::ExecutorResult;
use super::utils::{build_column_index_map, hash_value_into};
use super::Executor;
use crate::core::value::grouping_float;

// Re-export for backward compatibility
pub use super::utils::{expression_contains_aggregate, is_aggregate_function};

/// Every path that builds an aggregate result. Each ends in
/// `finish_aggregation`, which records the path for the equivalence tests
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AggregatePath {
    General,
    Pushdown,
    FilteredPushdown,
    StreamingGlobal,
    StreamingGlobalScan,
    StorageGrouped,
    StreamingGroupBy,
    StreamingDerivedTable,
}

#[cfg(any(test, feature = "test-failpoints"))]
impl AggregatePath {
    pub(crate) const ALL: [AggregatePath; 8] = [
        AggregatePath::General,
        AggregatePath::Pushdown,
        AggregatePath::FilteredPushdown,
        AggregatePath::StreamingGlobal,
        AggregatePath::StreamingGlobalScan,
        AggregatePath::StorageGrouped,
        AggregatePath::StreamingGroupBy,
        AggregatePath::StreamingDerivedTable,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            AggregatePath::General => "general",
            AggregatePath::Pushdown => "pushdown",
            AggregatePath::FilteredPushdown => "filtered_pushdown",
            AggregatePath::StreamingGlobal => "streaming_global",
            AggregatePath::StreamingGlobalScan => "streaming_global_scan",
            AggregatePath::StorageGrouped => "storage_grouped",
            AggregatePath::StreamingGroupBy => "streaming_group_by",
            AggregatePath::StreamingDerivedTable => "streaming_derived_table",
        }
    }
}

/// What the slot rows of an aggregate result stand for
#[derive(Clone, Copy)]
pub(crate) struct AggregateSlots<'a> {
    pub stmt: &'a SelectStatement,
    /// The statement as written, when lifting subqueries rewrote `stmt`
    pub written_stmt: &'a SelectStatement,
    pub aggregations: &'a [SqlAggregateFunction],
    pub group_by_count: usize,
}

/// Single condition in a HAVING clause
#[derive(Clone, Debug)]
struct HavingCondition {
    /// Index of the aggregate in the aggregations array
    agg_index: usize,
    /// The column-constant comparison the general path compiles this
    /// condition to, so both answer alike for every aggregate type
    op: Op,
}

/// Simple HAVING filter for inline application during fast aggregation
/// Supports: SUM(col) op value, COUNT(*) op value, COUNT(col) op value
/// Also supports AND combinations: COUNT(*) > 10 AND SUM(x) > 100
#[derive(Clone, Debug)]
pub(crate) struct SimpleHavingFilter {
    /// All conditions that must pass (AND semantics)
    conditions: Vec<HavingCondition>,
}

#[derive(Clone, Copy, Debug)]
enum ComparisonOp {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
    Neq,
}

impl HavingCondition {
    /// Check if a value passes this condition
    fn matches(&self, value: &Value) -> bool {
        ExprVM::fused_cell(&self.op, Some(value)) == Some(true)
    }
}

impl SimpleHavingFilter {
    /// Create a filter with a single condition
    fn single(agg_index: usize, op: ComparisonOp, threshold: Value) -> Self {
        let op = match op {
            ComparisonOp::Gt => Op::GtColumnConst(0, threshold),
            ComparisonOp::Gte => Op::GeColumnConst(0, threshold),
            ComparisonOp::Lt => Op::LtColumnConst(0, threshold),
            ComparisonOp::Lte => Op::LeColumnConst(0, threshold),
            ComparisonOp::Eq => Op::EqColumnConst(0, threshold),
            ComparisonOp::Neq => Op::NeColumnConst(0, threshold),
        };
        Self {
            conditions: vec![HavingCondition { agg_index, op }],
        }
    }

    /// Combine two filters with AND semantics
    fn and(mut self, other: Self) -> Self {
        self.conditions.extend(other.conditions);
        self
    }

    /// Whether a group whose aggregate slots hold `aggregates` passes
    pub(crate) fn passes(&self, aggregates: &[Value]) -> bool {
        self.conditions.iter().all(|cond| {
            aggregates
                .get(cond.agg_index)
                .is_some_and(|value| cond.matches(value))
        })
    }
}

/// Simple aggregate type for fast aggregation path
/// Supports COUNT, SUM, AVG, MIN, MAX (no DISTINCT, FILTER, ORDER BY, or expressions)
#[derive(Clone)]
enum SimpleAgg {
    Count,              // COUNT(*)
    CountColumn(usize), // COUNT(col) - counts only non-NULL values
    Sum(usize),         // SUM(col) - stores column index
    Avg(usize),         // AVG(col) - stores column index
    Min(usize),         // MIN(col) - stores column index
    Max(usize),         // MAX(col) - stores column index
}

/// Try to parse a simple HAVING clause for inline filtering
/// Returns None if the HAVING is too complex for inline optimization
/// Supports: single conditions and AND combinations
pub(crate) fn try_parse_simple_having(
    having: &Expression,
    aggregations: &[SqlAggregateFunction],
) -> Option<SimpleHavingFilter> {
    // Handle AND expressions: parse both sides and combine
    if let Expression::Infix(binop) = having {
        if binop.operator.eq_ignore_ascii_case("AND") {
            let left = try_parse_simple_having(&binop.left, aggregations)?;
            let right = try_parse_simple_having(&binop.right, aggregations)?;
            return Some(left.and(right));
        }
    }

    // Handle single comparison: AGG(col) op value
    try_parse_single_having_condition(having, aggregations)
        .map(|(agg_index, op, threshold)| SimpleHavingFilter::single(agg_index, op, threshold))
}

/// Parse a single HAVING condition (not AND/OR)
fn try_parse_single_having_condition(
    having: &Expression,
    aggregations: &[SqlAggregateFunction],
) -> Option<(usize, ComparisonOp, Value)> {
    // Handle comparison: AGG(col) op value
    if let Expression::Infix(binop) = having {
        let (op, threshold) = match binop.operator.as_str() {
            ">" => (ComparisonOp::Gt, extract_numeric_value(&binop.right)?),
            ">=" => (ComparisonOp::Gte, extract_numeric_value(&binop.right)?),
            "<" => (ComparisonOp::Lt, extract_numeric_value(&binop.right)?),
            "<=" => (ComparisonOp::Lte, extract_numeric_value(&binop.right)?),
            "=" => (ComparisonOp::Eq, extract_numeric_value(&binop.right)?),
            "!=" | "<>" => (ComparisonOp::Neq, extract_numeric_value(&binop.right)?),
            _ => return None,
        };

        // Left side should be an aggregate function
        if let Expression::FunctionCall(func) = &*binop.left {
            let func_upper = func.function.to_uppercase();
            if matches!(func_upper.as_str(), "SUM" | "COUNT" | "AVG" | "MIN" | "MAX") {
                // Find matching aggregate
                for (i, agg) in aggregations.iter().enumerate() {
                    if agg.name.to_uppercase() == func_upper && !agg.distinct {
                        // Check if column matches (for non-COUNT(*))
                        let col_matches = if func_upper == "COUNT" {
                            // COUNT(*) must bind to a COUNT(*) aggregate;
                            // matching any COUNT would read a COUNT(col)
                            // slot, which no longer counts NULL rows.
                            match func.arguments.first() {
                                None => agg.column == "*",
                                Some(Expression::Star(_)) => agg.column == "*",
                                Some(Expression::Identifier(id)) => {
                                    id.value_lower == agg.column_lower
                                }
                                Some(_) => false,
                            }
                        } else {
                            // SUM, AVG, etc. - check column
                            func.arguments.first().is_some_and(|arg| match arg {
                                Expression::Identifier(id) => id.value_lower == agg.column_lower,
                                _ => false,
                            })
                        };

                        if col_matches {
                            return Some((i, op, threshold));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Keeps the first evaluation error of a per-row loop for after it, and
/// stands a NULL in for the value; out of line so the loop stays lean
#[cold]
#[inline(never)]
fn keep_first_error(slot: &mut Option<crate::core::Error>, e: crate::core::Error) -> Value {
    slot.get_or_insert(e);
    Value::null_unknown()
}

/// A program over aggregate slot rows: `aliases` read a group item or
/// aggregate from its slot by identity, `outer_columns` the outer row by name
fn compile_over_slots(
    expr: &Expression,
    columns: &[String],
    registry: &crate::functions::FunctionRegistry,
    aliases: Option<&FxHashMap<ExprKey, usize>>,
    outer_columns: Option<&[String]>,
) -> Result<super::expression::SharedProgram> {
    let mut compile = super::expression::CompileContext::new(columns, registry);
    if let Some(aliases) = aliases {
        compile = compile.with_expression_aliases(
            aliases
                .iter()
                .map(|(&key, &slot)| (key, slot as u16))
                .collect(),
        );
    }
    if let Some(outer) = outer_columns {
        compile = compile.with_outer_columns(outer);
    }
    super::expression::ExprCompiler::new(&compile)
        .compile(expr)
        .map(CompactArc::new)
        .map_err(|e| crate::core::Error::internal(format!("Compile error: {}", e)))
}

/// The VM context for one slot row of `ctx`'s statement
fn slot_context<'a>(
    row: &'a Row,
    ctx: &'a ExecutionContext,
) -> super::expression::ExecuteContext<'a> {
    let exec = super::expression::ExecuteContext::with_common_params(
        row,
        ctx.params(),
        Some(ctx.named_params()),
        ctx.transaction_id(),
    );
    match ctx.outer_row() {
        Some(outer) => exec.with_outer_row(outer),
        None => exec,
    }
}

/// Extract numeric value from expression, in its own type
fn extract_numeric_value(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
        Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
        Expression::Prefix(unary) if unary.operator == "-" => {
            match extract_numeric_value(&unary.right)? {
                Value::Integer(i) => i.checked_neg().map(Value::Integer),
                Value::Float(f) => Some(Value::Float(-f)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Represents a grouping set for ROLLUP/CUBE operations
/// Each grouping set specifies which columns are active (included in grouping)
/// For ROLLUP(a, b), we get: [true, true], [true, false], [false, false]
#[derive(Clone, Debug)]
struct GroupingSet {
    /// For each GROUP BY column, whether it's included in this grouping level
    /// If false, the column value will be NULL in the output (rolled up)
    active_columns: Vec<bool>,
}

/// The identity of a GROUP BY item
fn group_by_item_key(item: &GroupByItem) -> ExprKey {
    match item {
        GroupByItem::Column(name) => ExprKey::column(name),
        GroupByItem::Position(pos) => ExprKey::position(*pos),
        GroupByItem::Expression { expr, .. } => ExprKey::of(expr),
    }
}

/// The slot of each group item and aggregate in an aggregated row, which
/// holds the group items first and the aggregates after them; a SELECT
/// alias names its expression's slot, unless a group column has that name
pub(crate) fn aggregate_slot_keys(
    group_items: &[GroupByItem],
    aggregations: &[SqlAggregateFunction],
    select: &[Expression],
) -> FxHashMap<ExprKey, usize> {
    let mut slots = FxHashMap::default();
    for (i, item) in group_items.iter().enumerate() {
        slots.entry(group_by_item_key(item)).or_insert(i);
    }
    for (i, agg) in aggregations.iter().enumerate() {
        slots.entry(agg.key).or_insert(group_items.len() + i);
    }
    for col in select {
        if let Expression::Aliased(aliased) = col {
            if let Some(&slot) = slots.get(&ExprKey::of(&aliased.expression)) {
                slots
                    .entry(ExprKey::column(&aliased.alias.value))
                    .or_insert(slot);
            }
        }
    }
    slots
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
    /// The aggregated row's slot holding this group item or aggregate
    AggSlot(usize),
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
    /// The identity of the call, shared by every place that writes it alike
    pub(crate) key: ExprKey,
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
        base_rows: RowVec,
        base_columns: &[String],
    ) -> Result<Box<dyn QueryResult>> {
        let written_stmt = stmt;
        // A subquery inside an aggregate is resolved per input row and read
        // back as a column, since an aggregate cannot resolve one itself
        let lifted_stmt;
        let lifted_columns;
        let (stmt, base_rows, base_columns) = match Self::aggregates_hold_subquery(stmt)
            .then(|| Self::lift_subqueries_in_aggregates(stmt, base_columns))
            .flatten()
        {
            Some((rewritten, lifted, names)) => {
                let qualifier = stmt
                    .table_expr
                    .as_deref()
                    .and_then(super::utils::get_table_alias_from_expr);
                let (rows, all_columns) = self.materialize_lifted_subqueries(
                    &lifted,
                    &names,
                    ctx,
                    &base_rows,
                    base_columns,
                    qualifier.as_deref(),
                )?;
                lifted_stmt = rewritten;
                lifted_columns = all_columns;
                (&lifted_stmt, rows, lifted_columns.as_slice())
            }
            None => (stmt, base_rows, base_columns),
        };

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

        // OFFSET is applied after aggregation, so the groups it skips must be
        // built too: the pushed limit is LIMIT + OFFSET
        let aggregation_limit = if can_push_limit {
            let eval_usize = |expr: &Expression| {
                let registry = crate::functions::registry::global_registry();
                let program = compile_over_slots(expr, &[], registry, None, None).ok()?;
                let row = Row::new();
                match ExprVM::new()
                    .execute(&program, &slot_context(&row, ctx))
                    .ok()?
                {
                    crate::core::Value::Integer(n) if n >= 0 => Some(n as usize),
                    _ => None,
                }
            };
            let limit = stmt.limit.as_deref().and_then(eval_usize);
            match stmt.offset.as_deref() {
                None => limit,
                Some(offset_expr) => {
                    limit.and_then(|l| eval_usize(offset_expr).map(|o| l.saturating_add(o)))
                }
            }
        } else {
            None
        };

        // Build result
        // having_applied_inline tracks if HAVING was already applied during fast aggregation
        let (result_columns, result_rows, having_applied_inline) = if group_by_columns.is_empty() {
            // Global aggregation (no GROUP BY)
            let (cols, rows) = self.execute_global_aggregation(
                &aggregations,
                &base_rows,
                base_columns,
                &col_index_map,
                ctx,
            )?;
            (cols, rows, false)
        } else if stmt.group_by.modifier != GroupByModifier::None {
            // ROLLUP or CUBE aggregation
            let (cols, rows) = self.execute_rollup_aggregation(
                &aggregations,
                &group_by_columns,
                &base_rows,
                base_columns,
                &col_index_map,
                stmt,
                ctx,
            )?;
            (cols, rows, false)
        } else {
            // Regular grouped aggregation - pass limit for early termination
            // May apply HAVING inline for simple cases (returns having_applied flag)
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
        // Skip if HAVING was already applied inline during fast aggregation
        let (having_columns, having_rows) = if let Some(ref having) = stmt.having {
            if having_applied_inline {
                // HAVING already applied inline, skip separate filtering
                (result_columns, result_rows)
            } else {
                // Pre-process scalar subqueries in HAVING clause
                // This executes subqueries like (SELECT AVG(a) FROM t) and replaces them with values.
                // One that reads the group is left as written; apply_having
                // runs it once per group with that group as the outer row
                let has_correlated_having = Self::has_correlated_subqueries(having);
                let processed_having = if has_correlated_having {
                    (**having).clone()
                } else {
                    self.process_where_subqueries(having, ctx)?
                };
                let outer_table: Option<String> =
                    stmt.table_expr.as_ref().and_then(|te| match te.as_ref() {
                        Expression::TableSource(source) => Some(
                            source
                                .alias
                                .as_ref()
                                .map(|a| a.value_lower.to_string())
                                .unwrap_or_else(|| source.name.value_lower.to_string()),
                        ),
                        Expression::Aliased(aliased) => Some(aliased.alias.value_lower.to_string()),
                        _ => None,
                    });

                // HAVING reads every group item and aggregate from its slot
                let slots = aggregate_slot_keys(&group_by_columns, &aggregations, &stmt.columns);
                let tmp_result = Box::new(ExecutorResult::new(result_columns.clone(), result_rows));
                let mut having_result = self.apply_having(
                    tmp_result,
                    &processed_having,
                    &result_columns,
                    &slots,
                    outer_table.as_deref(),
                    ctx,
                )?;

                // Collect rows after HAVING filter
                let mut filtered_rows = RowVec::new();
                let mut row_id = 0i64;
                while having_result.next() {
                    filtered_rows.push((row_id, having_result.take_row()));
                    row_id += 1;
                }
                (result_columns, filtered_rows)
            }
        } else {
            (result_columns, result_rows)
        };

        self.finish_aggregation(
            AggregatePath::General,
            AggregateSlots {
                stmt,
                written_stmt,
                aggregations: &aggregations,
                group_by_count: group_by_columns.len(),
            },
            ctx,
            having_columns,
            having_rows,
        )
    }

    /// The one end of every aggregate path: `slot_rows` hold the group
    /// items, then one value per distinct aggregate; the result projects them
    /// onto the SELECT list and carries what ORDER BY alone names
    pub(crate) fn finish_aggregation(
        &self,
        path: AggregatePath,
        slots: AggregateSlots<'_>,
        ctx: &ExecutionContext,
        having_columns: Vec<String>,
        having_rows: RowVec,
    ) -> Result<Box<dyn QueryResult>> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::aggregate_path_taken(path as usize);
        let _ = path;
        let AggregateSlots {
            stmt,
            written_stmt,
            aggregations,
            group_by_count,
        } = slots;

        // ORDER BY may name a grouped column the SELECT list drops. Capture
        // those columns before projection consumes the aggregate rows; they
        // are appended below and trimmed again after the sort.
        let hidden_order_by_columns =
            Self::collect_hidden_order_by_columns(stmt, &having_columns, &having_rows);

        // Check for hidden aggregates (ORDER BY only) BEFORE cloning
        // These will be removed after sorting by the ProjectedResult wrapper
        let hidden_aggs: Vec<(usize, &SqlAggregateFunction)> = aggregations
            .iter()
            .enumerate()
            .filter(|(_, agg)| agg.hidden)
            .collect();

        // Apply post-aggregation expression evaluation and column projection
        // Only clone if we need the original data for hidden_aggs processing
        let (final_columns, final_rows) = if hidden_aggs.is_empty() {
            // No hidden aggregates - move data directly (no clone)
            self.apply_post_aggregation_expressions(
                stmt,
                ctx,
                aggregations,
                having_columns,
                having_rows,
            )?
        } else {
            // Hidden aggregates exist: extract only the needed columns
            // up front, then move the aggregated rows instead of cloning
            // the whole result set
            // Lifting rewrote the aggregates; the sort looks for them as written
            let written_keys: Option<Vec<ExprKey>> = if std::ptr::eq(stmt, written_stmt) {
                None
            } else {
                let (written, _) = self.parse_aggregations(written_stmt)?;
                (written.len() == aggregations.len())
                    .then(|| written.iter().map(|agg| agg.key).collect())
            };
            let by_index: Vec<(String, Vec<Value>)> = hidden_aggs
                .iter()
                .map(|(agg_idx, agg)| {
                    let having_idx = group_by_count + agg_idx;
                    let values = having_rows
                        .iter()
                        .map(|(_, row)| {
                            row.get(having_idx)
                                .cloned()
                                .unwrap_or_else(Value::null_unknown)
                        })
                        .collect();
                    let key = written_keys.as_ref().map_or(agg.key, |keys| keys[*agg_idx]);
                    (key.hidden_column_name(), values)
                })
                .collect();

            let (mut cols, mut rows) = self.apply_post_aggregation_expressions(
                stmt,
                ctx,
                aggregations,
                having_columns,
                having_rows,
            )?;

            // Append hidden aggregates to the result for ORDER BY to use
            for (col_name, values) in by_index {
                cols.push(col_name);
                for (row_idx, (_, row)) in rows.iter_mut().enumerate() {
                    row.push(
                        values
                            .get(row_idx)
                            .cloned()
                            .unwrap_or_else(Value::null_unknown),
                    );
                }
            }

            (cols, rows)
        };

        // ORDER BY may reference a grouped column that the SELECT list does
        // not project. The sort runs after this function returns, so carry
        // those columns along; execute_select trims them back off once the
        // rows are ordered (the same contract the non-aggregate path uses).
        let (final_columns, final_rows) = Self::append_hidden_order_by_columns(
            final_columns,
            final_rows,
            hidden_order_by_columns,
        );

        let result: Box<dyn QueryResult> = Box::new(ExecutorResult::new(final_columns, final_rows));

        Ok(result)
    }

    /// Append grouped columns that ORDER BY needs but SELECT does not project.
    /// Values come from the pre-projection aggregate rows, matched by position.
    /// Grouped columns that ORDER BY names but the SELECT list does not
    /// project, paired with their per-group values. A grouped column may be
    /// stored qualified ("t.v") or bare, and ORDER BY resolves the qualified
    /// form first, so both spellings are tried in that order.
    /// Resolve one ORDER BY item against the pre-projection columns.
    /// Returns the name to publish the carried column under (the spelling
    /// the sorter will look for) and its index in `source_columns`.
    ///
    /// The two spellings do not have to agree: a query may group by `t.v`
    /// and order by `v`, or the reverse, so a bare name also matches a
    /// unique `<qualifier>.<name>` column and a qualified name falls back
    /// to its bare form.
    pub(crate) fn resolve_hidden_order_by_column(
        ob: &crate::parser::ast::OrderByExpression,
        source_columns: &[String],
    ) -> Option<(String, usize)> {
        let (publish_as, candidates) = match &ob.expression {
            Expression::Identifier(id) => (id.value.to_string(), vec![id.value.to_string()]),
            Expression::QualifiedIdentifier(qid) => {
                let full = format!("{}.{}", qid.qualifier.value, qid.name.value);
                (full.clone(), vec![full, qid.name.value.to_string()])
            }
            _ => return None,
        };

        for cand in &candidates {
            if let Some(idx) = source_columns
                .iter()
                .position(|c| c.eq_ignore_ascii_case(cand))
            {
                return Some((publish_as, idx));
            }
        }

        // Bare name against a qualified column: only when unambiguous
        let bare = match &ob.expression {
            Expression::Identifier(id) => id.value.to_string(),
            Expression::QualifiedIdentifier(qid) => qid.name.value.to_string(),
            _ => return None,
        };
        let suffix = format!(".{}", bare);
        let mut matches = source_columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.to_lowercase().ends_with(&suffix.to_lowercase()));
        let (idx, _) = matches.next()?;
        if matches.next().is_some() {
            return None; // ambiguous across qualifiers
        }
        Some((publish_as, idx))
    }

    /// Grouped columns that ORDER BY names but the SELECT list does not
    /// project, paired with their per-group values.
    pub(crate) fn collect_hidden_order_by_columns(
        stmt: &SelectStatement,
        source_columns: &[String],
        source_rows: &RowVec,
    ) -> Vec<(String, Vec<Value>)> {
        stmt.order_by
            .iter()
            .filter_map(|ob| {
                let (publish_as, idx) = Self::resolve_hidden_order_by_column(ob, source_columns)?;
                let values = source_rows
                    .iter()
                    .map(|(_, row)| row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                    .collect();
                Some((publish_as, values))
            })
            .collect()
    }

    fn append_hidden_order_by_columns(
        mut columns: Vec<String>,
        mut rows: RowVec,
        hidden: Vec<(String, Vec<Value>)>,
    ) -> (Vec<String>, RowVec) {
        for (name, values) in hidden {
            if columns.iter().any(|c| c.eq_ignore_ascii_case(&name)) {
                continue; // already projected, nothing to carry
            }
            columns.push(name);
            for (row_idx, (_, row)) in rows.iter_mut().enumerate() {
                row.push(
                    values
                        .get(row_idx)
                        .cloned()
                        .unwrap_or_else(Value::null_unknown),
                );
            }
        }
        (columns, rows)
    }

    /// Apply post-aggregation expressions to the result
    /// This handles expressions like `CASE WHEN SUM(x) > 100 THEN 'big' ELSE 'small' END`
    pub(crate) fn apply_post_aggregation_expressions(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        aggregations: &[SqlAggregateFunction],
        agg_columns: Vec<String>,
        agg_rows: RowVec,
    ) -> Result<(Vec<String>, RowVec)> {
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

        // Names find the result columns that bare names and aliases point at
        let agg_col_index_map = build_column_index_map(&agg_columns);
        // Identities find the slot of each group item and aggregate, however it is named
        let slot_keys = aggregate_slot_keys(&group_by_columns, aggregations, &stmt.columns);
        let slot_of = |expr: &Expression| slot_keys.get(&ExprKey::of(expr)).copied();

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
            // Each SELECT column already sits in its slot, in order, under the name it shows
            let columns_match = columns_to_use.iter().enumerate().all(|(i, col_expr)| {
                let Some(name) = agg_columns.get(i) else {
                    return false;
                };
                match col_expr {
                    Expression::Identifier(id) => name.to_lowercase() == id.value_lower.as_str(),
                    Expression::QualifiedIdentifier(qid) => {
                        name.to_lowercase()
                            == format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower)
                    }
                    Expression::Aliased(a) => name.to_lowercase() == a.alias.value_lower.as_str(),
                    Expression::FunctionCall(func) if is_aggregate_function(&func.function) => {
                        slot_of(col_expr) == Some(i)
                            && self.get_aggregate_column_name(func) == *name
                    }
                    _ => {
                        slot_of(col_expr) == Some(i) && self.expression_to_string(col_expr) == *name
                    }
                }
            });
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
                    final_columns.push(id.value.to_string());
                    // A name no result column carries may be another alias of a shared slot
                    match slot_of(col_expr)
                        .filter(|_| !agg_col_index_map.contains_key(id.value_lower.as_str()))
                    {
                        Some(slot) => column_sources.push(ColumnSource::AggSlot(slot)),
                        None => {
                            column_sources.push(ColumnSource::AggColumn(id.value_lower.to_string()))
                        }
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    // Use only the base column name (strip table alias prefix)
                    // e.g., u.username -> username
                    final_columns.push(qid.name.value.to_string());
                    // Try qualified name first, then fall back to unqualified column name
                    let qualified_lower =
                        format!("{}.{}", qid.qualifier.value_lower, qid.name.value_lower);
                    let unqualified_lower: String = qid.name.value_lower.to_string();
                    if agg_col_index_map.contains_key(&qualified_lower) {
                        column_sources.push(ColumnSource::AggColumn(qualified_lower));
                    } else {
                        column_sources.push(ColumnSource::AggColumn(unqualified_lower));
                    }
                }
                Expression::FunctionCall(func) => {
                    if is_aggregate_function(&func.function) {
                        final_columns.push(self.get_aggregate_column_name(func));
                        column_sources.push(match slot_of(col_expr) {
                            Some(slot) => ColumnSource::AggSlot(slot),
                            None => make_source(col_expr, is_correlated),
                        });
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
                        // Non-aggregate function: a GROUP BY expression column is
                        // read as it is, anything else is evaluated
                        final_columns.push(format!("{}(...)", func.function));
                        match slot_of(col_expr).filter(|_| !is_correlated) {
                            Some(slot) => column_sources.push(ColumnSource::AggSlot(slot)),
                            None => column_sources.push(make_source(col_expr, is_correlated)),
                        }
                    }
                }
                Expression::Aliased(aliased) => {
                    final_columns.push(aliased.alias.value.to_string());
                    let alias_lower: String = aliased.alias.value_lower.to_string();

                    // The item's own identity first: an alias may repeat the
                    // name another slot carries, as in `g AS n, COUNT(*) AS n`
                    let slot = slot_of(&aliased.expression).filter(|_| !is_correlated);
                    if let Some(slot) = slot {
                        column_sources.push(ColumnSource::AggSlot(slot));
                    } else if agg_col_index_map.contains_key(&alias_lower) && !is_correlated {
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
                                // Read by identity, not alias: a deduplicated aggregate keeps one name
                                if let Some(slot) = slot_of(&aliased.expression) {
                                    column_sources.push(ColumnSource::AggSlot(slot));
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
                                // A grouped expression is read from its slot, anything else evaluated
                                if let Some(slot) = slot_of(&aliased.expression) {
                                    column_sources.push(ColumnSource::AggSlot(slot));
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
                    final_columns.push(self.expression_to_string(col_expr));
                    if let Some(slot) = slot_of(col_expr).filter(|_| !is_correlated) {
                        // CASE is a GROUP BY column - use existing value
                        column_sources.push(ColumnSource::AggSlot(slot));
                    } else {
                        // CASE with aggregates or correlated - needs evaluation
                        column_sources.push(make_source(col_expr, is_correlated));
                    }
                }
                _ => {
                    // A grouped expression is read from its slot, anything else evaluated
                    final_columns.push(self.expression_to_string(col_expr));
                    match slot_of(col_expr).filter(|_| !is_correlated) {
                        Some(slot) => column_sources.push(ColumnSource::AggSlot(slot)),
                        None => column_sources.push(make_source(col_expr, is_correlated)),
                    }
                }
            }
        }

        // Check if we have any correlated expressions that need special handling
        let has_correlated_sources = column_sources
            .iter()
            .any(|s| matches!(s, ColumnSource::CorrelatedExpression(_)));

        // Evaluate each row
        let mut final_rows = RowVec::with_capacity(agg_rows.len());
        let mut vm = super::expression::ExprVM::new();
        // A statement inside a correlated subquery reads the outer row by name
        let outer_columns: Option<Vec<String>> = ctx
            .outer_row()
            .map(|outer| outer.keys().map(|k| k.to_string()).collect());

        // Pre-compute outer row column names if we have correlated expressions
        let outer_col_names: Option<CompactArc<Vec<String>>> = if has_correlated_sources {
            Some(CompactArc::new(agg_columns.clone()))
        } else {
            None
        };

        // Extract table alias from FROM clause for qualified column names in correlated subqueries
        let table_alias: Option<String> = if has_correlated_sources {
            if let Some(ref table_expr) = stmt.table_expr {
                match table_expr.as_ref() {
                    Expression::TableSource(source) => {
                        if let Some(ref alias) = source.alias {
                            Some(alias.value_lower.to_string())
                        } else {
                            Some(source.name.value_lower.to_string())
                        }
                    }
                    Expression::Aliased(aliased) => Some(aliased.alias.value_lower.to_string()),
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
        // Uses CompactArc<str> for zero-cost cloning in the per-row loop
        #[allow(clippy::type_complexity)]
        let correlated_col_names: Option<Vec<(CompactArc<str>, Option<CompactArc<str>>)>> =
            if has_correlated_sources {
                Some(
                    agg_columns
                        .iter()
                        .map(|col_name| {
                            let col_lower: CompactArc<str> =
                                CompactArc::from(col_name.to_lowercase().as_str());
                            let qualified = table_alias.as_ref().map(|alias| {
                                CompactArc::from(format!("{}.{}", alias, col_lower).as_str())
                            });
                            (col_lower, qualified)
                        })
                        .collect(),
                )
            } else {
                None
            };

        // Reusable map for correlated expressions
        // Uses CompactArc<str> keys for zero-cost cloning
        let estimated_entries = agg_columns.len() * 2;
        let mut outer_row_map: FxHashMap<CompactArc<str>, Value> =
            FxHashMap::with_capacity_and_hasher(estimated_entries, Default::default());

        // Plain expression sources compile once for the row loop; a group
        // item or aggregate inside one reads its slot, whatever names it
        let source_programs: Vec<Option<super::expression::SharedProgram>> = column_sources
            .iter()
            .map(|s| match s {
                ColumnSource::Expression(expr) => compile_over_slots(
                    expr,
                    &agg_columns,
                    crate::functions::registry::global_registry(),
                    Some(&slot_keys),
                    outer_columns.as_deref(),
                )
                .map(Some),
                _ => Ok(None),
            })
            .collect::<Result<_>>()?;

        for (id, row) in agg_rows {
            // Use CompactVec directly to avoid Vec→CompactVec conversion
            let mut new_values: CompactVec<Value> = CompactVec::with_capacity(column_sources.len());

            for (src_idx, source) in column_sources.iter().enumerate() {
                let value = match source {
                    ColumnSource::AggColumn(col_name) => {
                        if let Some(&idx) = agg_col_index_map.get(col_name) {
                            row.get(idx).cloned().unwrap_or(Value::null_unknown())
                        } else {
                            Value::null_unknown()
                        }
                    }
                    ColumnSource::AggSlot(idx) => {
                        row.get(*idx).cloned().unwrap_or(Value::null_unknown())
                    }
                    ColumnSource::Expression(_) => match &source_programs[src_idx] {
                        Some(p) => vm.execute_cow(p, &slot_context(&row, ctx))?,
                        None => Value::null_unknown(),
                    },
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
                        let result = self
                            .process_correlated_expression(expr, &correlated_ctx)
                            .and_then(|processed_expr| {
                                let outer: Option<Vec<String>> = correlated_ctx
                                    .outer_row()
                                    .map(|outer| outer.keys().map(|k| k.to_string()).collect());
                                let program = compile_over_slots(
                                    &processed_expr,
                                    &agg_columns,
                                    &self.function_registry,
                                    None,
                                    outer.as_deref(),
                                )?;
                                vm.execute_cow(&program, &slot_context(&row, &correlated_ctx))
                            });

                        // Take back map for reuse
                        outer_row_map = correlated_ctx.outer_row.take().unwrap_or_default();
                        result?
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

            final_rows.push((id, Row::from_compact_vec(new_values)));
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
        let arg_name: &str = match arg {
            Expression::Identifier(id) => id.value_lower.as_str(),
            Expression::QualifiedIdentifier(qid) => qid.name.value_lower.as_str(),
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

    /// The ORDER BY of an aggregate that orders its own input, with the
    /// NULLS placement the query asked for
    pub(crate) fn aggregate_order_keys(
        order_by: &[crate::parser::ast::OrderByExpression],
    ) -> Vec<crate::functions::AggregateOrder> {
        order_by
            .iter()
            .map(|order| crate::functions::AggregateOrder {
                ascending: order.ascending,
                nulls_first: order.nulls_first,
            })
            .collect()
    }

    /// Aggregate, then run the window functions over the aggregate's output.
    /// A subquery inside an aggregate is resolved per input row first, and
    /// both stages read the same rewritten statement
    pub(crate) fn execute_aggregation_then_windows(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
    ) -> Result<Box<dyn QueryResult>> {
        let lifted_stmt;
        let lifted_rows;
        let lifted_columns;
        let (stmt, base_rows, base_columns) = match Self::aggregates_hold_subquery(stmt)
            .then(|| Self::lift_subqueries_in_aggregates(stmt, base_columns))
            .flatten()
        {
            Some((rewritten, lifted, names)) => {
                let qualifier = stmt
                    .table_expr
                    .as_deref()
                    .and_then(super::utils::get_table_alias_from_expr);
                let (rows, all_columns) = self.materialize_lifted_subqueries(
                    &lifted,
                    &names,
                    ctx,
                    base_rows,
                    base_columns,
                    qualifier.as_deref(),
                )?;
                lifted_stmt = rewritten;
                lifted_rows = rows;
                lifted_columns = all_columns;
                (
                    &lifted_stmt,
                    lifted_rows.as_slice(),
                    lifted_columns.as_slice(),
                )
            }
            None => (stmt, base_rows, base_columns),
        };
        let (agg_columns, agg_rows, slots) =
            self.execute_aggregation_for_window(stmt, ctx, base_rows, base_columns)?;
        self.execute_select_with_window_functions_over_groups(
            stmt,
            ctx,
            &agg_rows,
            &agg_columns,
            &slots,
        )
    }

    /// Execute GROUP BY aggregation and return raw columns/rows for window function processing
    /// This is used when both GROUP BY and window functions are present in the query.
    /// Window functions operate on the aggregated result.
    fn execute_aggregation_for_window(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
    ) -> Result<(Vec<String>, RowVec, FxHashMap<ExprKey, usize>)> {
        // Parse aggregations and group by columns
        let (aggregations, _non_agg_columns) = self.parse_aggregations(stmt)?;
        let group_by_columns = self.parse_group_by(stmt, base_columns)?;

        // Create column index map for fast lookup
        let col_index_map = build_column_index_map(base_columns);

        // Build result
        // Note: No limit pushdown here because window functions need all rows
        let (result_columns, result_rows) = if group_by_columns.is_empty() {
            // Global aggregation (no GROUP BY)
            self.execute_global_aggregation(
                &aggregations,
                base_rows,
                base_columns,
                &col_index_map,
                ctx,
            )?
        } else {
            // Grouped aggregation - no limit since window functions need all groups
            // Discard the having_applied flag - window functions apply HAVING separately
            let (cols, rows, _having_applied) = self.execute_grouped_aggregation(
                &aggregations,
                &group_by_columns,
                base_rows,
                base_columns,
                &col_index_map,
                stmt,
                ctx,
                None, // Window functions need all groups
            )?;
            (cols, rows)
        };

        // Apply HAVING clause filter (in-place)
        let slots = aggregate_slot_keys(&group_by_columns, &aggregations, &stmt.columns);
        let mut result_rows_with_ids = RowVec::with_capacity(result_rows.len());
        if let Some(ref having) = stmt.having {
            let having_filter = compile_over_slots(
                having,
                &result_columns,
                crate::functions::registry::global_registry(),
                Some(&slots),
                None,
            )?;
            let mut vm = ExprVM::new();
            for (id, row) in result_rows {
                if vm.execute_bool_checked(&having_filter, &slot_context(&row, ctx))? {
                    result_rows_with_ids.push((id, row));
                }
            }
        } else {
            for (id, row) in result_rows {
                result_rows_with_ids.push((id, row));
            }
        }

        Ok((result_columns, result_rows_with_ids, slots))
    }

    /// Parse aggregate functions from SELECT list
    /// Returns: (aggregations, non_agg_columns, post_agg_expressions)
    pub(crate) fn parse_aggregations(
        &self,
        stmt: &SelectStatement,
    ) -> Result<(Vec<SqlAggregateFunction>, Vec<String>)> {
        let mut aggregations = Vec::new();
        let mut non_agg_columns = Vec::new();

        for col_expr in &stmt.columns {
            self.extract_aggregates_from_expr(col_expr, &mut aggregations, &mut non_agg_columns)?;
        }
        let select_count = aggregations.len();

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
        // A named window a function refers to with OVER w may partition or
        // order by an aggregate as well; it is computed like ORDER BY's
        for window in &stmt.window_defs {
            for expr in window
                .partition_by
                .iter()
                .chain(window.order_by.iter().map(|o| &o.expression))
            {
                self.extract_aggregates_from_expr(expr, &mut aggregations, &mut non_agg_columns)?;
            }
        }
        // An ORDER BY aggregate the SELECT list does not compute is hidden
        for i in visible_count..aggregations.len() {
            let already_exists = aggregations[..visible_count]
                .iter()
                .any(|existing| existing.key == aggregations[i].key);
            if !already_exists {
                aggregations[i].hidden = true;
            }
        }
        // A HAVING aggregate the sort reads too is carried to it, unless SELECT holds it
        for i in select_count..visible_count {
            let key = aggregations[i].key;
            if aggregations[visible_count..]
                .iter()
                .any(|agg| agg.key == key)
                && !aggregations[..select_count]
                    .iter()
                    .any(|agg| agg.key == key)
            {
                aggregations[i].hidden = true;
            }
        }
        // One slot per distinct call, whichever clause wrote it first
        let mut seen: FxHashSet<ExprKey> = FxHashSet::default();
        aggregations.retain(|agg| seen.insert(agg.key));

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
                            return Err(crate::core::Error::InvalidArgument(format!(
                                "aggregate function calls cannot be nested: {}",
                                func.function
                            )));
                        }
                    }

                    let (column, distinct, extra_args, expression) =
                        self.extract_agg_column(&func.arguments)?;
                    let column_lower = column.to_lowercase();
                    aggregations.push(SqlAggregateFunction {
                        name: func.function.to_string(),
                        column,
                        column_lower,
                        alias: None,
                        distinct: distinct || func.is_distinct,
                        extra_args,
                        expression,
                        order_by: func.order_by.clone(),
                        filter: func.filter.as_ref().map(|f| (**f).clone()),
                        hidden: false,
                        key: ExprKey::of_call(func),
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
                non_agg_columns.push(id.value.to_string());
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
            Expression::QuantifiedSet(set) => {
                self.extract_aggregates_from_expr(&set.left, aggregations, non_agg_columns)?;
            }
            // An aggregate a window function takes as its argument, or
            // partitions or orders by, is computed like any other
            Expression::Window(window) => {
                for expr in window
                    .function
                    .arguments
                    .iter()
                    .chain(window.partition_by.iter())
                    .chain(window.order_by.iter().map(|o| &o.expression))
                {
                    self.extract_aggregates_from_expr(expr, aggregations, non_agg_columns)?;
                }
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
                            return Err(crate::core::Error::InvalidArgument(format!(
                                "aggregate function calls cannot be nested: {}",
                                func.function
                            )));
                        }
                    }

                    let (column, distinct, extra_args, expression) =
                        self.extract_agg_column(&func.arguments)?;
                    let column_lower = column.to_lowercase();
                    aggregations.push(SqlAggregateFunction {
                        name: func.function.to_string(),
                        column,
                        column_lower,
                        alias: Some(aliased.alias.value.to_string()),
                        distinct: distinct || func.is_distinct,
                        extra_args,
                        expression,
                        order_by: func.order_by.clone(),
                        filter: func.filter.as_ref().map(|f| (**f).clone()),
                        hidden: false,
                        key: ExprKey::of_call(func),
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
            Expression::Identifier(id) => (id.value.to_string(), false, None),
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
                    extra_args.push(Value::text(lit.value.as_str()));
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
                Expression::Identifier(id) if id.quoted => {
                    extra_args.push(Value::text(id.value.as_str()));
                }
                _ => {}
            }
        }

        Ok((column, distinct, extra_args, expression))
    }

    /// Parse GROUP BY clause
    pub(crate) fn parse_group_by(
        &self,
        stmt: &SelectStatement,
        _base_columns: &[String],
    ) -> Result<Vec<GroupByItem>> {
        let mut group_items = Vec::new();

        // Build a map of aliases to their expressions from SELECT clause; an
        // aggregate cannot be grouped, so its alias leaves the name to the source column
        let alias_map: FxHashMap<String, Expression> = stmt
            .columns
            .iter()
            .filter_map(|col| {
                if let Expression::Aliased(aliased) = col {
                    if expression_contains_aggregate(&aliased.expression) {
                        return None;
                    }
                    Some((
                        aliased.alias.value_lower.to_string(),
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
                let mut seen = FxHashSet::default();
                let mut unique_cols = Vec::new();
                for set in sets {
                    for expr in set {
                        if seen.insert(ExprKey::of(expr)) {
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
                    let id_lower: &str = id.value_lower.as_str();
                    if let Some(aliased_expr) = alias_map.get(id_lower) {
                        // Use the aliased expression, with the alias as the display name
                        group_items.push(GroupByItem::Expression {
                            expr: aliased_expr.clone(),
                            display_name: id.value.to_string(),
                        });
                    } else {
                        // Regular column reference
                        group_items.push(GroupByItem::Column(id.value.to_string()));
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
                                group_items.push(GroupByItem::Column(id.value.to_string()));
                            }
                            Expression::Aliased(aliased) => {
                                // Aliased expression - extract the underlying expression
                                match aliased.expression.as_ref() {
                                    Expression::Identifier(id) => {
                                        // Aliased column reference
                                        group_items.push(GroupByItem::Column(id.value.to_string()));
                                    }
                                    expr => {
                                        // Complex expression with alias
                                        group_items.push(GroupByItem::Expression {
                                            expr: expr.clone(),
                                            display_name: aliased.alias.value.to_string(),
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

    /// The source column a single named GROUP BY item groups by, bound by
    /// `parse_group_by` as the shared projection binds it, and whether the
    /// item carried a qualifier
    pub(crate) fn single_group_column(
        &self,
        stmt: &SelectStatement,
    ) -> Result<Option<(String, bool)>> {
        match stmt.group_by.columns.as_slice() {
            [Expression::Identifier(_)] => {}
            [Expression::QualifiedIdentifier(qid)] => {
                return Ok(Some((qid.name.value_lower.to_string(), true)));
            }
            _ => return Ok(None),
        }
        Ok(match self.parse_group_by(stmt, &[])?.as_slice() {
            [GroupByItem::Column(name)] => Some((name.to_lowercase(), false)),
            [GroupByItem::Expression {
                expr: Expression::Identifier(id),
                ..
            }] => Some((id.value_lower.to_string(), false)),
            _ => None,
        })
    }

    /// Find the alias for an expression in the SELECT list
    fn find_expression_alias(&self, stmt: &SelectStatement, target_expr: &Expression) -> String {
        let target = ExprKey::of(target_expr);
        for col_expr in &stmt.columns {
            if let Expression::Aliased(aliased) = col_expr {
                if ExprKey::of(&aliased.expression) == target {
                    return aliased.alias.value.to_string();
                }
            }
        }
        // No alias found, generate a name from the expression
        self.expression_to_string(target_expr)
    }

    /// Convert an expression to a display string: the one stringifier, so
    /// two aggregates over different expressions never share a name
    fn expression_to_string(&self, expr: &Expression) -> String {
        super::utils::expression_to_string(expr)
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
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, RowVec)> {
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
            let mut result_rows = RowVec::with_capacity(1);
            result_rows.push((0, Row::from_values(vec![Value::Integer(rows.len() as i64)])));
            return Ok((result_columns, result_rows));
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

        // FAST PATH: a single SUM or AVG over a column, without DISTINCT
        if !has_expression
            && aggregations.len() == 1
            && matches!(aggregations[0].name.as_str(), "SUM" | "AVG")
            && !aggregations[0].distinct
        {
            if let Some(col_idx) = agg_col_indices[0] {
                let sum = Self::sum_column_rows(rows, col_idx);
                let result = if aggregations[0].name == "SUM" {
                    sum.sum()?
                } else {
                    sum.avg()
                };
                let result_columns: Vec<String> =
                    aggregations.iter().map(|a| a.get_column_name()).collect();
                let mut result_rows = RowVec::with_capacity(1);
                result_rows.push((0, Row::from_values(vec![result])));
                return Ok((result_columns, result_rows));
            }
        }

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

        // Fast path: no expressions, no filters, no order by on any aggregate
        let can_use_compiled = aggregations.iter().all(|agg| {
            agg.expression.is_none()
                && agg.filter.is_none()
                && agg.order_by.is_empty()
                && agg.extra_args.is_empty()
        });

        let result_values: Vec<Value> = if can_use_compiled {
            // FAST PATH: Use CompiledAggregate for zero virtual dispatch
            let compile_all = || -> Vec<CompiledAggregate> {
                aggregations
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
                    .collect()
            };

            // Hot loop with compiled aggregates - zero virtual dispatch
            let accumulate_rows = |aggs: &mut [CompiledAggregate], rows: &[(i64, Row)]| {
                // Pre-create static Value for COUNT(*)
                let count_star_value = Value::Integer(1);
                for (_, row) in rows {
                    for (i, agg) in aggs.iter_mut().enumerate() {
                        let value = if let Some(col_idx) = agg_col_indices[i] {
                            row.get(col_idx)
                        } else {
                            Some(&count_star_value) // COUNT(*)
                        };

                        if let Some(v) = value {
                            agg.accumulate(v);
                        }
                    }
                }
            };

            let mut compiled_aggs = compile_all();
            // Large inputs split into chunks when every aggregate merges
            // its chunk runs into the whole
            #[cfg(feature = "parallel")]
            if rows.len() >= 100_000 && compiled_aggs.iter().all(CompiledAggregate::can_merge) {
                let chunk_size = (rows.len() / rayon::current_num_threads()).max(1000);
                let chunks: Vec<Vec<CompiledAggregate>> = rows
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut aggs = compile_all();
                        accumulate_rows(&mut aggs, chunk);
                        aggs
                    })
                    .collect();
                for chunk in &chunks {
                    for (agg, run) in compiled_aggs.iter_mut().zip(chunk) {
                        agg.merge(run);
                    }
                }
            } else {
                accumulate_rows(&mut compiled_aggs, rows);
            }
            #[cfg(not(feature = "parallel"))]
            accumulate_rows(&mut compiled_aggs, rows);

            compiled_aggs
                .iter()
                .map(|agg| agg.result())
                .collect::<Result<_>>()?
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
                        func.set_order_by(Self::aggregate_order_keys(&agg.order_by));
                    }
                }
            }

            // Pre-create static Value for COUNT(*)
            let count_star_value = Value::Integer(1);

            // Buffer for evaluated expression values (to avoid repeated allocation)
            let mut expr_values: Vec<Value> = vec![Value::null_unknown(); aggregations.len()];

            for (_, row) in rows {
                // Create execution context for this row
                // CRITICAL: Include params for parameterized queries
                let exec_ctx = ExecuteContext::new(row)
                    .with_params(ctx.params())
                    .with_named_params(ctx.named_params());

                for (i, agg) in aggregations.iter().enumerate() {
                    if let Some(ref mut func) = agg_funcs[i] {
                        // Check FILTER clause first - skip row if filter is false
                        if let Some(ref filter_program) = compiled_filters[i] {
                            if let Some(ref mut vm) = expr_vm {
                                match vm.execute_cow(filter_program, &exec_ctx) {
                                    Ok(Value::Boolean(true)) => {} // Continue with accumulation
                                    Ok(Value::Boolean(false)) | Ok(Value::Null(_)) => continue, // Skip this row
                                    Ok(_) => continue, // Non-boolean treated as false
                                    Err(e) => return Err(e),
                                }
                            } else {
                                // Can't evaluate filter without VM - skip
                                continue;
                            }
                        }

                        // Get the value to accumulate
                        let value = if let Some(ref expr_program) = compiled_agg_expressions[i] {
                            // Evaluate the expression for this row using VM
                            if let Some(ref mut vm) = expr_vm {
                                match vm.execute_cow(expr_program, &exec_ctx) {
                                    Ok(val) => {
                                        expr_values[i] = val;
                                        Some(&expr_values[i])
                                    }
                                    Err(e) => {
                                        return Err(crate::core::Error::expression_evaluation(
                                            format!("{}({}): {}", agg.name, agg.column, e),
                                        ));
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
                                    for order_program in &compiled_order_by[i] {
                                        sort_keys.push(vm.execute_cow(order_program, &exec_ctx)?);
                                    }
                                    func.accumulate_with_sort_key(v, sort_keys, agg.distinct);
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
                        Ok(Value::Integer(rows.len() as i64))
                    } else {
                        Ok(Value::null_unknown())
                    }
                })
                .collect::<Result<_>>()?
        };

        // Build result columns
        let result_columns: Vec<String> =
            aggregations.iter().map(|a| a.get_column_name()).collect();

        let mut result_rows = RowVec::with_capacity(1);
        result_rows.push((0, Row::from_values(result_values)));
        Ok((result_columns, result_rows))
    }

    /// Try to use fast single-pass aggregation for simple cases
    ///
    /// Returns Some((columns, rows)) if fast path was used, None otherwise.
    /// Fast path is used when:
    /// - All GROUP BY items are simple column references
    /// - All aggregates are COUNT, SUM, AVG, MIN, or MAX (no DISTINCT, FILTER, ORDER BY, or expression)
    ///
    /// When `limit` is provided and there's no ORDER BY, enables early termination:
    /// once we have `limit` complete groups, we stop creating new groups.
    ///
    /// When `having_filter` is provided, applies HAVING inline during row generation,
    /// avoiding a separate filtering pass.
    #[allow(clippy::too_many_arguments)]
    fn try_fast_aggregation(
        &self,
        aggregations: &[SqlAggregateFunction],
        group_by_items: &[GroupByItem],
        rows: &[(i64, Row)],
        _columns: &[String],
        col_index_map: &StringMap<usize>,
        limit: Option<usize>,
        having_filter: Option<&SimpleHavingFilter>,
    ) -> Result<Option<(Vec<String>, RowVec)>> {
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

        // Check if all aggregates are simple (COUNT/SUM/AVG/MIN/MAX without DISTINCT/FILTER/ORDER BY/expression)
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
                    "COUNT" => {
                        if agg.column == "*" {
                            Some(SimpleAgg::Count)
                        } else {
                            Self::lookup_column_index(&agg.column_lower, col_index_map)
                                .map(SimpleAgg::CountColumn)
                        }
                    }
                    "SUM" => {
                        if agg.column == "*" {
                            None // SUM(*) is not valid
                        } else {
                            Self::lookup_column_index(&agg.column_lower, col_index_map)
                                .map(SimpleAgg::Sum)
                        }
                    }
                    "AVG" => {
                        if agg.column == "*" {
                            None // AVG(*) is not valid
                        } else {
                            Self::lookup_column_index(&agg.column_lower, col_index_map)
                                .map(SimpleAgg::Avg)
                        }
                    }
                    "MIN" => {
                        if agg.column == "*" {
                            None // MIN(*) is not valid
                        } else {
                            Self::lookup_column_index(&agg.column_lower, col_index_map)
                                .map(SimpleAgg::Min)
                        }
                    }
                    "MAX" => {
                        if agg.column == "*" {
                            None // MAX(*) is not valid
                        } else {
                            Self::lookup_column_index(&agg.column_lower, col_index_map)
                                .map(SimpleAgg::Max)
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

        // OPTIMIZATION: Single-column GROUP BY uses direct Value storage (no Vec allocation per row)
        if group_by_indices.len() == 1 {
            return self.try_fast_aggregation_single_column(
                &group_by_indices[0],
                &simple_aggs,
                aggregations,
                group_by_items,
                rows,
                limit,
                having_filter,
            );
        }

        // Fast path: single-pass streaming aggregation for multi-column GROUP BY
        // Store aggregate state directly in hash map instead of row indices
        // SmallVec for inline storage when ≤4 aggregations (common case)
        use smallvec::SmallVec;
        type AggVec<T> = SmallVec<[T; 4]>;

        struct FastGroupState {
            sums: AggVec<SumAccumulator>,      // For SUM and AVG
            counts: AggVec<i64>,               // For COUNT
            min_values: AggVec<Option<Value>>, // For MIN
            max_values: AggVec<Option<Value>>, // For MAX
        }

        // Pre-allocate hash map with estimated capacity to reduce resizing.
        // Estimate: for high-cardinality groupings, assume ~1/3 of rows are unique groups.
        // OPTIMIZATION: Use hashbrown::HashMap with Vec<Value> key directly - HashMap handles
        // collisions efficiently with open addressing, avoiding our manual Vec-based collision chaining.
        // Using raw_entry_mut API for O(1) lookup without cloning keys.
        // NOTE: Uses FxHash here because raw_entry_mut().from_hash() requires compatible hasher.
        // Value::hash() is simple (optimized for AHash), but FxHash still works correctly.
        // Start small - HashMap grows efficiently, over-allocation wastes memory
        let estimated_groups = (rows.len() / 32).clamp(16, 256);
        type FxBuildHasher = BuildHasherDefault<FxHasher>;
        let mut groups: hashbrown::HashMap<Vec<Value>, FastGroupState, FxBuildHasher> =
            hashbrown::HashMap::with_capacity_and_hasher(
                estimated_groups,
                FxBuildHasher::default(),
            );
        let num_aggs = simple_aggs.len();

        // Track for early termination optimization
        let group_limit = limit.unwrap_or(usize::MAX);
        let has_limit = limit.is_some();
        let mut current_group_count: usize = 0;

        for (_, row) in rows {
            // OPTIMIZATION: Hash directly from row references (no clone for hashing)
            // FxHasher has zero initialization cost unlike AHash.
            // The manual hash MUST equal the map hasher's hash_one(&Vec<Value>),
            // which length-prefixes the slice: on resize hashbrown rehashes
            // every entry with the map's own hasher, and a mismatch scatters
            // the table and splits groups
            let mut hasher = FxHasher::default();
            hasher.write_usize(group_by_indices.len());
            for &idx in &group_by_indices {
                if let Some(value) = row.get(idx) {
                    value.hash(&mut hasher);
                } else {
                    Value::null_unknown().hash(&mut hasher);
                }
            }
            let hash = hasher.finish();

            // OPTIMIZATION: Use raw_entry_mut for O(1) lookup without cloning
            // - Compute hash from row references (already done above)
            // - Compare stored keys against row references (no clone for lookup)
            // - Only clone when inserting a new group
            // OPTIMIZATION: Unrolled comparison for common cases (2-3 columns)
            // Avoids loop overhead and enables better branch prediction
            let num_group_cols = group_by_indices.len();
            let entry = groups.raw_entry_mut().from_hash(hash, |stored_key| {
                if stored_key.len() != num_group_cols {
                    return false;
                }
                // Inline helper for value comparison
                #[inline(always)]
                fn val_eq(stored: &Value, row_val: Option<&Value>) -> bool {
                    match row_val {
                        Some(rv) => stored == rv,
                        None => matches!(stored, Value::Null(_)),
                    }
                }
                match num_group_cols {
                    2 => {
                        // Unrolled 2-column comparison (most common multi-column case)
                        val_eq(&stored_key[0], row.get(group_by_indices[0]))
                            && val_eq(&stored_key[1], row.get(group_by_indices[1]))
                    }
                    3 => {
                        // Unrolled 3-column comparison
                        val_eq(&stored_key[0], row.get(group_by_indices[0]))
                            && val_eq(&stored_key[1], row.get(group_by_indices[1]))
                            && val_eq(&stored_key[2], row.get(group_by_indices[2]))
                    }
                    _ => {
                        // Generic loop for 4+ columns
                        for i in 0..num_group_cols {
                            if !val_eq(&stored_key[i], row.get(group_by_indices[i])) {
                                return false;
                            }
                        }
                        true
                    }
                }
            });

            let state = match entry {
                RawEntryMut::Occupied(occupied) => occupied.into_mut(),
                RawEntryMut::Vacant(vacant) => {
                    // New group - check limit before creating
                    if has_limit && current_group_count >= group_limit {
                        continue;
                    }
                    // Only clone values when creating a new group
                    let key_values: Vec<Value> = group_by_indices
                        .iter()
                        .map(|&idx| row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                        .collect();
                    current_group_count += 1;
                    let (_, state) = vacant.insert_hashed_nocheck(
                        hash,
                        key_values,
                        FastGroupState {
                            sums: smallvec::smallvec![SumAccumulator::default(); num_aggs],
                            counts: smallvec::smallvec![0; num_aggs],
                            min_values: smallvec::smallvec![None; num_aggs],
                            max_values: smallvec::smallvec![None; num_aggs],
                        },
                    );
                    state
                }
            };

            // Accumulate aggregates
            for (i, agg) in simple_aggs.iter().enumerate() {
                match agg {
                    SimpleAgg::Count => {
                        state.counts[i] += 1;
                    }
                    SimpleAgg::CountColumn(col_idx) => {
                        if let Some(value) = row.get(*col_idx) {
                            if !value.is_null() {
                                state.counts[i] += 1;
                            }
                        }
                    }
                    SimpleAgg::Sum(col_idx) | SimpleAgg::Avg(col_idx) => {
                        if let Some(value) = row.get(*col_idx) {
                            state.sums[i].add(value);
                        }
                    }
                    SimpleAgg::Min(col_idx) => {
                        if let Some(value) = row.get(*col_idx) {
                            if !value.is_null() {
                                match &state.min_values[i] {
                                    None => state.min_values[i] = Some(value.clone()),
                                    Some(current) if value < current => {
                                        state.min_values[i] = Some(value.clone())
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    SimpleAgg::Max(col_idx) => {
                        if let Some(value) = row.get(*col_idx) {
                            if !value.is_null() {
                                match &state.max_values[i] {
                                    None => state.max_values[i] = Some(value.clone()),
                                    Some(current) if value > current => {
                                        state.max_values[i] = Some(value.clone())
                                    }
                                    _ => {}
                                }
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

        // Build result rows from HashMap entries
        // OPTIMIZATION: Apply HAVING filter inline if provided, avoiding separate filtering pass
        let mut result_rows = RowVec::new();
        let mut row_id = 0i64;
        for (key_values, mut state) in groups.into_iter() {
            // Apply inline HAVING filter if provided (supports AND combinations)
            if let Some(filter) = having_filter {
                // All conditions must pass (AND semantics)
                let mut passes = true;
                for cond in &filter.conditions {
                    let agg_value = match &simple_aggs[cond.agg_index] {
                        SimpleAgg::Count | SimpleAgg::CountColumn(_) => {
                            Some(Cow::Owned(Value::Integer(state.counts[cond.agg_index])))
                        }
                        SimpleAgg::Sum(_) => Some(state.sums[cond.agg_index].sum()?)
                            .filter(|v| !v.is_null())
                            .map(Cow::Owned),
                        SimpleAgg::Avg(_) => Some(state.sums[cond.agg_index].avg())
                            .filter(|v| !v.is_null())
                            .map(Cow::Owned),
                        SimpleAgg::Min(_) => {
                            state.min_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                        }
                        SimpleAgg::Max(_) => {
                            state.max_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                        }
                    };
                    match agg_value {
                        Some(val) => {
                            if !cond.matches(&val) {
                                passes = false;
                                break;
                            }
                        }
                        None => {
                            passes = false;
                            break;
                        }
                    }
                }
                if !passes {
                    continue;
                }
            }

            // Use CompactVec directly to avoid Vec→CompactVec conversion
            let mut values: CompactVec<Value> =
                CompactVec::with_capacity(key_values.len() + simple_aggs.len());
            values.extend(key_values);

            for (i, agg) in simple_aggs.iter().enumerate() {
                let value = match agg {
                    SimpleAgg::Count | SimpleAgg::CountColumn(_) => Value::Integer(state.counts[i]),
                    SimpleAgg::Sum(_) => state.sums[i].sum()?,
                    SimpleAgg::Avg(_) => state.sums[i].avg(),
                    SimpleAgg::Min(_) => state.min_values[i]
                        .take()
                        .unwrap_or_else(Value::null_unknown),
                    SimpleAgg::Max(_) => state.max_values[i]
                        .take()
                        .unwrap_or_else(Value::null_unknown),
                };
                values.push(value);
            }

            result_rows.push((row_id, Row::from_compact_vec(values)));
            row_id += 1;
        }

        Ok(Some((result_columns, result_rows)))
    }

    /// Single-column GROUP BY fast aggregation - avoids Vec<Value> allocation per row
    ///
    /// For single-column GROUP BY (the most common case), we can store the group key
    /// as a single Value instead of Vec<Value>, eliminating allocation overhead.
    #[allow(clippy::too_many_arguments)]
    fn try_fast_aggregation_single_column(
        &self,
        group_col_idx: &usize,
        simple_aggs: &[SimpleAgg],
        aggregations: &[SqlAggregateFunction],
        group_by_items: &[GroupByItem],
        rows: &[(i64, Row)],
        limit: Option<usize>,
        having_filter: Option<&SimpleHavingFilter>,
    ) -> Result<Option<(Vec<String>, RowVec)>> {
        // State for single-column GROUP BY - stores single Value instead of Vec<Value>
        // SmallVec for inline storage when ≤4 aggregations (common case)
        use smallvec::SmallVec;
        type AggVec<T> = SmallVec<[T; 4]>;

        struct SingleColGroupState {
            key_value: Value,
            sums: AggVec<SumAccumulator>,
            counts: AggVec<i64>,
            min_values: AggVec<Option<Value>>,
            max_values: AggVec<Option<Value>>,
        }

        // Start small - HashMap grows efficiently, over-allocation wastes memory
        let estimated_groups = (rows.len() / 32).clamp(16, 256);
        let num_aggs = simple_aggs.len();
        let col_idx = *group_col_idx;

        // Track for early termination optimization
        let group_limit = limit.unwrap_or(usize::MAX);
        let has_limit = limit.is_some();

        // OPTIMIZATION: Check if we can use Integer or String fast path
        // Sample multiple rows to detect column type (first non-NULL value determines type)
        // This handles cases where the first row might have NULL values.
        let sampled_type = rows
            .iter()
            .take(16) // Sample up to 16 rows for efficiency
            .filter_map(|(_, r)| r.get(col_idx))
            .find(|v| !matches!(v, Value::Null(_)));

        let use_integer_fast_path = sampled_type
            .map(|v| matches!(v, Value::Integer(_)))
            .unwrap_or(false);
        let use_string_fast_path = sampled_type
            .map(|v| matches!(v, Value::Text(_)))
            .unwrap_or(false);

        if use_integer_fast_path {
            // Ultra-fast path for Integer GROUP BY: no Value cloning, no hashing overhead
            // SmallVec for inline storage when ≤4 aggregations (common case)
            // Avoids heap allocation on clone for new groups
            use smallvec::SmallVec;
            type AggVec<T> = SmallVec<[T; 4]>;

            #[derive(Clone)]
            struct IntGroupState {
                sums: AggVec<SumAccumulator>,
                counts: AggVec<i64>,
                min_values: AggVec<Option<Value>>,
                max_values: AggVec<Option<Value>>,
            }

            let mut groups: I64Map<IntGroupState> = I64Map::with_capacity(estimated_groups);
            // Separate tracking for NULL group (SQL: all NULLs group together)
            let mut null_group: Option<IntGroupState> = None;
            let mut current_group_count: usize = 0;

            // OPTIMIZATION: Pre-allocate template state - clone is faster than separate smallvec! calls
            let state_template = IntGroupState {
                sums: smallvec::smallvec![SumAccumulator::default(); num_aggs],
                counts: smallvec::smallvec![0; num_aggs],
                min_values: smallvec::smallvec![None; num_aggs],
                max_values: smallvec::smallvec![None; num_aggs],
            };

            for (_, row) in rows {
                // Extract integer key directly - no clone, no hash
                // Handle NULL values separately (they form their own group)
                let key_opt = match row.get(col_idx) {
                    Some(Value::Integer(v)) => Some(*v),
                    Some(Value::Null(_)) => None, // NULL goes to null_group (inline pattern)
                    None => None,                 // Missing value treated as NULL
                    _ => continue,                // Skip non-integer, non-NULL values
                };

                let state = if let Some(key) = key_opt {
                    // OPTIMIZATION: Single hash lookup using entry API instead of
                    // contains_key + entry (was doing 2 hash lookups)
                    use crate::common::i64_map::Entry;
                    match groups.entry(key) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(e) => {
                            // Early termination check for new groups
                            if has_limit && current_group_count >= group_limit {
                                continue;
                            }
                            current_group_count += 1;
                            e.insert(state_template.clone())
                        }
                    }
                } else {
                    // NULL group
                    if null_group.is_none() {
                        if has_limit && current_group_count >= group_limit {
                            continue;
                        }
                        current_group_count += 1;
                        null_group = Some(state_template.clone());
                    }
                    null_group.as_mut().unwrap()
                };

                // Accumulate aggregates
                for (i, agg) in simple_aggs.iter().enumerate() {
                    match agg {
                        SimpleAgg::Count => {
                            state.counts[i] += 1;
                        }
                        SimpleAgg::CountColumn(col_idx) => {
                            if let Some(value) = row.get(*col_idx) {
                                if !value.is_null() {
                                    state.counts[i] += 1;
                                }
                            }
                        }
                        SimpleAgg::Sum(sum_col_idx) | SimpleAgg::Avg(sum_col_idx) => {
                            if let Some(value) = row.get(*sum_col_idx) {
                                state.sums[i].add(value);
                            }
                        }
                        SimpleAgg::Min(min_col_idx) => {
                            if let Some(value) = row.get(*min_col_idx) {
                                if !value.is_null() {
                                    match &state.min_values[i] {
                                        None => state.min_values[i] = Some(value.clone()),
                                        Some(current) if value < current => {
                                            state.min_values[i] = Some(value.clone())
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        SimpleAgg::Max(max_col_idx) => {
                            if let Some(value) = row.get(*max_col_idx) {
                                if !value.is_null() {
                                    match &state.max_values[i] {
                                        None => state.max_values[i] = Some(value.clone()),
                                        Some(current) if value > current => {
                                            state.max_values[i] = Some(value.clone())
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Build result columns
            let mut result_columns = Vec::with_capacity(1 + aggregations.len());
            let group_col_name = match &group_by_items[0] {
                GroupByItem::Column(col_name) => col_name.clone(),
                _ => "col0".to_string(),
            };
            result_columns.push(group_col_name);
            for agg in aggregations {
                let col_name = if let Some(ref alias) = agg.alias {
                    alias.clone()
                } else {
                    agg.get_expression_name()
                };
                result_columns.push(col_name);
            }

            // Helper to check HAVING filter
            let passes_having = |state: &IntGroupState| -> Result<bool> {
                if let Some(filter) = having_filter {
                    for cond in &filter.conditions {
                        let agg_value = match &simple_aggs[cond.agg_index] {
                            SimpleAgg::Count | SimpleAgg::CountColumn(_) => {
                                Some(Cow::Owned(Value::Integer(state.counts[cond.agg_index])))
                            }
                            SimpleAgg::Sum(_) => Some(state.sums[cond.agg_index].sum()?)
                                .filter(|v| !v.is_null())
                                .map(Cow::Owned),
                            SimpleAgg::Avg(_) => Some(state.sums[cond.agg_index].avg())
                                .filter(|v| !v.is_null())
                                .map(Cow::Owned),
                            SimpleAgg::Min(_) => {
                                state.min_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                            }
                            SimpleAgg::Max(_) => {
                                state.max_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                            }
                        };
                        match agg_value {
                            Some(val) => {
                                if !cond.matches(&val) {
                                    return Ok(false);
                                }
                            }
                            None => return Ok(false),
                        }
                    }
                }
                Ok(true)
            };

            // Helper to build row from state
            // Use CompactVec directly to avoid Vec→CompactVec conversion
            let build_row = |key_value: Value, mut state: IntGroupState| -> Result<Row> {
                let mut values: CompactVec<Value> =
                    CompactVec::with_capacity(1 + simple_aggs.len());
                values.push(key_value);
                for (i, agg) in simple_aggs.iter().enumerate() {
                    let value = match agg {
                        SimpleAgg::Count | SimpleAgg::CountColumn(_) => {
                            Value::Integer(state.counts[i])
                        }
                        SimpleAgg::Sum(_) => state.sums[i].sum()?,
                        SimpleAgg::Avg(_) => state.sums[i].avg(),
                        SimpleAgg::Min(_) => state.min_values[i]
                            .take()
                            .unwrap_or_else(Value::null_unknown),
                        SimpleAgg::Max(_) => state.max_values[i]
                            .take()
                            .unwrap_or_else(Value::null_unknown),
                    };
                    values.push(value);
                }
                Ok(Row::from_compact_vec(values))
            };

            // Build result rows with inline HAVING filter (supports AND combinations)
            let mut result_rows = RowVec::new();
            let mut row_id = 0i64;
            for (key, state) in groups.into_iter() {
                if passes_having(&state)? {
                    result_rows.push((row_id, build_row(Value::Integer(key), state)?));
                    row_id += 1;
                }
            }

            // Add NULL group if it exists and passes HAVING
            if let Some(ng) = null_group {
                if passes_having(&ng)? {
                    result_rows.push((row_id, build_row(Value::null_unknown(), ng)?));
                }
            }

            return Ok(Some((result_columns, result_rows)));
        }

        // String fast path for Text GROUP BY: direct SmartString key, no Value::eq overhead
        // SmallVec for inline storage when ≤4 aggregations (common case)
        if use_string_fast_path {
            use smallvec::SmallVec;
            type AggVec<T> = SmallVec<[T; 4]>;

            #[derive(Clone)]
            struct StringGroupState {
                sums: AggVec<SumAccumulator>,
                counts: AggVec<i64>,
                min_values: AggVec<Option<Value>>,
                max_values: AggVec<Option<Value>>,
            }

            let mut groups: FxHashMap<crate::common::SmartString, StringGroupState> =
                FxHashMap::with_capacity_and_hasher(estimated_groups, Default::default());
            // Separate tracking for NULL group (SQL: all NULLs group together)
            let mut null_group: Option<StringGroupState> = None;
            let mut current_group_count: usize = 0;

            // OPTIMIZATION: Pre-allocate template state - clone is faster than separate smallvec! calls
            let state_template = StringGroupState {
                sums: smallvec::smallvec![SumAccumulator::default(); num_aggs],
                counts: smallvec::smallvec![0; num_aggs],
                min_values: smallvec::smallvec![None; num_aggs],
                max_values: smallvec::smallvec![None; num_aggs],
            };

            for (_, row) in rows {
                // Extract string key directly - only clone when creating new group
                // Handle NULL values separately (they form their own group)
                let key_str_opt = match row.get(col_idx) {
                    Some(Value::Text(s)) => Some(s),
                    Some(Value::Null(_)) => None, // NULL goes to null_group (inline pattern)
                    None => None,                 // Missing value treated as NULL
                    _ => continue,                // Skip non-text, non-NULL values
                };

                let state = if let Some(key_str) = key_str_opt {
                    // OPTIMIZATION: get_mut first (no clone for existing groups)
                    // For aggregation with many rows but few groups, most rows hit existing groups
                    // This avoids cloning the key on every row - major perf win
                    match groups.get_mut(key_str) {
                        Some(existing) => existing,
                        None => {
                            // Early termination check for new groups
                            if has_limit && current_group_count >= group_limit {
                                continue;
                            }
                            current_group_count += 1;
                            // Clone key only when creating new group
                            groups.insert(key_str.clone(), state_template.clone());
                            // SAFETY: we just inserted, so key exists
                            groups.get_mut(key_str).unwrap()
                        }
                    }
                } else {
                    // NULL group
                    if null_group.is_none() {
                        if has_limit && current_group_count >= group_limit {
                            continue;
                        }
                        current_group_count += 1;
                        null_group = Some(state_template.clone());
                    }
                    null_group.as_mut().unwrap()
                };

                // Accumulate aggregates
                for (i, agg) in simple_aggs.iter().enumerate() {
                    match agg {
                        SimpleAgg::Count => {
                            state.counts[i] += 1;
                        }
                        SimpleAgg::CountColumn(col_idx) => {
                            if let Some(value) = row.get(*col_idx) {
                                if !value.is_null() {
                                    state.counts[i] += 1;
                                }
                            }
                        }
                        SimpleAgg::Sum(sum_col_idx) | SimpleAgg::Avg(sum_col_idx) => {
                            if let Some(value) = row.get(*sum_col_idx) {
                                state.sums[i].add(value);
                            }
                        }
                        SimpleAgg::Min(min_col_idx) => {
                            if let Some(value) = row.get(*min_col_idx) {
                                if !value.is_null() {
                                    match &state.min_values[i] {
                                        None => state.min_values[i] = Some(value.clone()),
                                        Some(current) if value < current => {
                                            state.min_values[i] = Some(value.clone())
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        SimpleAgg::Max(max_col_idx) => {
                            if let Some(value) = row.get(*max_col_idx) {
                                if !value.is_null() {
                                    match &state.max_values[i] {
                                        None => state.max_values[i] = Some(value.clone()),
                                        Some(current) if value > current => {
                                            state.max_values[i] = Some(value.clone())
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Build result columns
            let mut result_columns = Vec::with_capacity(1 + aggregations.len());
            let group_col_name = match &group_by_items[0] {
                GroupByItem::Column(col_name) => col_name.clone(),
                _ => "col0".to_string(),
            };
            result_columns.push(group_col_name);
            for agg in aggregations {
                let col_name = if let Some(ref alias) = agg.alias {
                    alias.clone()
                } else {
                    agg.get_expression_name()
                };
                result_columns.push(col_name);
            }

            // Helper to check HAVING filter
            let passes_having = |state: &StringGroupState| -> Result<bool> {
                if let Some(filter) = having_filter {
                    for cond in &filter.conditions {
                        let agg_value = match &simple_aggs[cond.agg_index] {
                            SimpleAgg::Count | SimpleAgg::CountColumn(_) => {
                                Some(Cow::Owned(Value::Integer(state.counts[cond.agg_index])))
                            }
                            SimpleAgg::Sum(_) => Some(state.sums[cond.agg_index].sum()?)
                                .filter(|v| !v.is_null())
                                .map(Cow::Owned),
                            SimpleAgg::Avg(_) => Some(state.sums[cond.agg_index].avg())
                                .filter(|v| !v.is_null())
                                .map(Cow::Owned),
                            SimpleAgg::Min(_) => {
                                state.min_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                            }
                            SimpleAgg::Max(_) => {
                                state.max_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                            }
                        };
                        match agg_value {
                            Some(val) => {
                                if !cond.matches(&val) {
                                    return Ok(false);
                                }
                            }
                            None => return Ok(false),
                        }
                    }
                }
                Ok(true)
            };

            // Helper to build row from state
            // Use CompactVec directly to avoid Vec→CompactVec conversion
            let build_row = |key_value: Value, mut state: StringGroupState| -> Result<Row> {
                let mut values: CompactVec<Value> =
                    CompactVec::with_capacity(1 + simple_aggs.len());
                values.push(key_value);
                for (i, agg) in simple_aggs.iter().enumerate() {
                    let value = match agg {
                        SimpleAgg::Count | SimpleAgg::CountColumn(_) => {
                            Value::Integer(state.counts[i])
                        }
                        SimpleAgg::Sum(_) => state.sums[i].sum()?,
                        SimpleAgg::Avg(_) => state.sums[i].avg(),
                        SimpleAgg::Min(_) => state.min_values[i]
                            .take()
                            .unwrap_or_else(Value::null_unknown),
                        SimpleAgg::Max(_) => state.max_values[i]
                            .take()
                            .unwrap_or_else(Value::null_unknown),
                    };
                    values.push(value);
                }
                Ok(Row::from_compact_vec(values))
            };

            // Build result rows with inline HAVING filter
            let mut result_rows = RowVec::new();
            let mut row_id = 0i64;
            for (key, state) in groups.into_iter() {
                if passes_having(&state)? {
                    result_rows.push((row_id, build_row(Value::Text(key), state)?));
                    row_id += 1;
                }
            }

            // Add NULL group if it exists and passes HAVING
            if let Some(ng) = null_group {
                if passes_having(&ng)? {
                    result_rows.push((row_id, build_row(Value::null_unknown(), ng)?));
                }
            }

            return Ok(Some((result_columns, result_rows)));
        }

        // Fallback: general single-column path with Value storage
        // Use hash -> Vec to handle collisions (different values with same hash)
        let mut groups: FxHashMap<u64, Vec<SingleColGroupState>> =
            FxHashMap::with_capacity_and_hasher(estimated_groups, Default::default());
        let mut current_group_count: usize = 0;

        for (_, row) in rows {
            // OPTIMIZATION: Hash directly from row reference (no clone for hashing)
            let row_value = row.get(col_idx);
            let mut hasher = AHasher::default();
            if let Some(value) = row_value {
                hash_value_into(value, &mut hasher);
            } else {
                hash_value_into(&Value::null_unknown(), &mut hasher);
            }
            let hash = hasher.finish();

            // Get or create bucket for this hash
            let bucket = groups.entry(hash).or_default();

            // OPTIMIZATION: Compare row value directly against stored keys (no clone for lookup)
            // Inline is_null() check as pattern match to avoid function call overhead
            let existing_idx = bucket.iter().position(|s| match row_value {
                Some(rv) => &s.key_value == rv,
                None => matches!(s.key_value, Value::Null(_)),
            });

            let state = if let Some(idx) = existing_idx {
                // Existing group - no clone needed!
                &mut bucket[idx]
            } else {
                // New group - check limit before creating
                if has_limit && current_group_count >= group_limit {
                    continue;
                }
                // Only clone when creating a new group
                let key_value = row_value.cloned().unwrap_or_else(Value::null_unknown);
                bucket.push(SingleColGroupState {
                    key_value,
                    sums: smallvec::smallvec![SumAccumulator::default(); num_aggs],
                    counts: smallvec::smallvec![0; num_aggs],
                    min_values: smallvec::smallvec![None; num_aggs],
                    max_values: smallvec::smallvec![None; num_aggs],
                });
                current_group_count += 1;
                bucket.last_mut().unwrap()
            };

            // Accumulate aggregates
            for (i, agg) in simple_aggs.iter().enumerate() {
                match agg {
                    SimpleAgg::Count => {
                        state.counts[i] += 1;
                    }
                    SimpleAgg::CountColumn(col_idx) => {
                        if let Some(value) = row.get(*col_idx) {
                            if !value.is_null() {
                                state.counts[i] += 1;
                            }
                        }
                    }
                    SimpleAgg::Sum(sum_col_idx) | SimpleAgg::Avg(sum_col_idx) => {
                        if let Some(value) = row.get(*sum_col_idx) {
                            state.sums[i].add(value);
                        }
                    }
                    SimpleAgg::Min(min_col_idx) => {
                        if let Some(value) = row.get(*min_col_idx) {
                            if !value.is_null() {
                                match &state.min_values[i] {
                                    None => state.min_values[i] = Some(value.clone()),
                                    Some(current) if value < current => {
                                        state.min_values[i] = Some(value.clone())
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    SimpleAgg::Max(max_col_idx) => {
                        if let Some(value) = row.get(*max_col_idx) {
                            if !value.is_null() {
                                match &state.max_values[i] {
                                    None => state.max_values[i] = Some(value.clone()),
                                    Some(current) if value > current => {
                                        state.max_values[i] = Some(value.clone())
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build result columns
        let mut result_columns = Vec::with_capacity(1 + aggregations.len());

        // Single GROUP BY column name
        let group_col_name = match &group_by_items[0] {
            GroupByItem::Column(col_name) => col_name.clone(),
            _ => "col0".to_string(),
        };
        result_columns.push(group_col_name);

        // Add aggregate column names
        for agg in aggregations {
            let col_name = if let Some(ref alias) = agg.alias {
                alias.clone()
            } else {
                agg.get_expression_name()
            };
            result_columns.push(col_name);
        }

        // Build result rows with inline HAVING filter (supports AND combinations)
        let mut result_rows = RowVec::new();
        let mut row_id = 0i64;
        for mut state in groups.into_values().flatten() {
            // Apply inline HAVING filter
            if let Some(filter) = having_filter {
                let mut passes = true;
                for cond in &filter.conditions {
                    let agg_value = match &simple_aggs[cond.agg_index] {
                        SimpleAgg::Count | SimpleAgg::CountColumn(_) => {
                            Some(Cow::Owned(Value::Integer(state.counts[cond.agg_index])))
                        }
                        SimpleAgg::Sum(_) => Some(state.sums[cond.agg_index].sum()?)
                            .filter(|v| !v.is_null())
                            .map(Cow::Owned),
                        SimpleAgg::Avg(_) => Some(state.sums[cond.agg_index].avg())
                            .filter(|v| !v.is_null())
                            .map(Cow::Owned),
                        SimpleAgg::Min(_) => {
                            state.min_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                        }
                        SimpleAgg::Max(_) => {
                            state.max_values[cond.agg_index].as_ref().map(Cow::Borrowed)
                        }
                    };
                    match agg_value {
                        Some(val) => {
                            if !cond.matches(&val) {
                                passes = false;
                                break;
                            }
                        }
                        None => {
                            passes = false;
                            break;
                        }
                    }
                }
                if !passes {
                    continue;
                }
            }

            // Use CompactVec directly to avoid Vec→CompactVec conversion
            let mut values: CompactVec<Value> = CompactVec::with_capacity(1 + simple_aggs.len());
            values.push(state.key_value);

            for (i, agg) in simple_aggs.iter().enumerate() {
                let value = match agg {
                    SimpleAgg::Count | SimpleAgg::CountColumn(_) => Value::Integer(state.counts[i]),
                    SimpleAgg::Sum(_) => state.sums[i].sum()?,
                    SimpleAgg::Avg(_) => state.sums[i].avg(),
                    SimpleAgg::Min(_) => state.min_values[i]
                        .take()
                        .unwrap_or_else(Value::null_unknown),
                    SimpleAgg::Max(_) => state.max_values[i]
                        .take()
                        .unwrap_or_else(Value::null_unknown),
                };
                values.push(value);
            }

            result_rows.push((row_id, Row::from_compact_vec(values)));
            row_id += 1;
        }

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
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        limit: Option<usize>,
    ) -> Result<(Vec<String>, RowVec, bool)> {
        // Try to parse simple HAVING for inline filtering optimization
        let inline_having = stmt
            .having
            .as_ref()
            .and_then(|h| try_parse_simple_having(h, aggregations));

        // FAST PATH: For simple aggregates (COUNT/SUM/AVG/MIN/MAX without DISTINCT/FILTER/ORDER BY/expression),
        // use single-pass streaming aggregation that accumulates values directly
        // When inline HAVING is available, it's applied during row generation
        if let Some(result) = self.try_fast_aggregation(
            aggregations,
            group_by_items,
            rows,
            columns,
            col_index_map,
            limit,
            inline_having.as_ref(),
        )? {
            // Return true for having_applied if we had an inline HAVING filter
            return Ok((result.0, result.1, inline_having.is_some()));
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

            // One raw-entry loop for any column count: hash and compare
            // against borrowed row values, clone key Values only when a
            // new group is actually inserted (same pattern as
            // try_fast_aggregation above)
            // The map hasher and the manual hash must agree exactly
            // (hash_one(&Vec<Value>) length-prefixes the slice): hashbrown
            // rehashes with the map's own hasher on resize
            type FxBuild = BuildHasherDefault<FxHasher>;
            let mut keyed_groups: hashbrown::HashMap<Vec<Value>, Vec<usize>, FxBuild> =
                hashbrown::HashMap::with_capacity_and_hasher(64, FxBuild::default());
            let num_cols = column_indices.len();
            let mut group_count: usize = 0;

            for (row_idx, (_, row)) in rows.iter().enumerate() {
                let mut hasher = FxHasher::default();
                hasher.write_usize(num_cols);
                for &idx in &column_indices {
                    match row.get(idx) {
                        Some(value) => value.hash(&mut hasher),
                        None => Value::null_unknown().hash(&mut hasher),
                    }
                }
                let hash = hasher.finish();

                let entry = keyed_groups.raw_entry_mut().from_hash(hash, |stored| {
                    stored.len() == num_cols
                        && stored.iter().zip(column_indices.iter()).all(|(s, &idx)| {
                            match row.get(idx) {
                                Some(rv) => s == rv,
                                None => matches!(s, Value::Null(_)),
                            }
                        })
                });
                match entry {
                    RawEntryMut::Occupied(mut occupied) => occupied.get_mut().push(row_idx),
                    RawEntryMut::Vacant(vacant) => {
                        // Early termination: skip rows that would create
                        // new groups beyond the limit
                        if has_limit && group_count >= group_limit {
                            continue;
                        }
                        group_count += 1;
                        let key: Vec<Value> = column_indices
                            .iter()
                            .map(|&idx| row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                            .collect();
                        vacant.insert_hashed_nocheck(hash, key, vec![row_idx]);
                    }
                }
            }

            // Convert to GroupEntry format for downstream processing
            for (key_values, row_indices) in keyed_groups {
                groups.entry(0).or_default().push(GroupEntry {
                    key_values,
                    row_indices,
                });
            }
        } else {
            // Slow path: need to evaluate expressions, use buffer. A key that
            // fails is returned after the loop; an early return in it is slower
            let mut key_error = None;
            for (row_idx, (_, row)) in rows.iter().enumerate() {
                key_buffer.clear();

                // Create execution context for this row
                // CRITICAL: Include params for parameterized queries
                let exec_ctx = ExecuteContext::new(row)
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
                                match vm.execute_cow(program, &exec_ctx) {
                                    Ok(value) => value,
                                    Err(e) => keep_first_error(&mut key_error, e),
                                }
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

                // OPTIMIZATION: Single scan to find existing group OR check limit
                // Previously we scanned twice: once for key_exists check, once for find()
                match groups.entry(hash) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        let bucket = e.get_mut();
                        // Single scan: find position of matching group
                        let existing_idx = bucket
                            .iter()
                            .position(|entry| entry.key_values == key_buffer);

                        if let Some(idx) = existing_idx {
                            // Existing group - just add this row to it
                            bucket[idx].row_indices.push(row_idx);
                        } else {
                            // Hash collision: different key with same hash
                            // Check limit before creating new group
                            if has_limit && current_group_count >= group_limit {
                                continue;
                            }
                            bucket.push(GroupEntry {
                                key_values: key_buffer.clone(),
                                row_indices: vec![row_idx],
                            });
                            current_group_count += 1;
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        // First entry for this hash - check limit before creating
                        if has_limit && current_group_count >= group_limit {
                            continue;
                        }
                        e.insert(vec![GroupEntry {
                            key_values: key_buffer.clone(),
                            row_indices: vec![row_idx],
                        }]);
                        current_group_count += 1;
                    }
                }
            }
            if let Some(e) = key_error {
                return Err(e);
            }
        }

        // Convert groups to Vec for parallel processing
        // Flatten buckets: each bucket may contain multiple groups (hash collisions)
        let groups_vec: Vec<GroupEntry> = groups.into_values().flatten().collect();

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
        #[cfg(feature = "parallel")]
        let total_rows: usize = groups_vec.iter().map(|g| g.row_indices.len()).sum();
        #[cfg(feature = "parallel")]
        let avg_rows_per_group = total_rows / groups_vec.len().max(1);
        #[cfg(feature = "parallel")]
        let use_parallel = groups_vec.len() >= 4
            && total_rows >= 10_000
            && avg_rows_per_group >= 50
            && !has_agg_expression;
        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;

        // Process groups (parallel or sequential based on data size)
        let result_rows: RowVec = if use_parallel {
            // PARALLEL: Process each group independently using Rayon
            let function_registry = &self.function_registry;

            #[cfg(feature = "parallel")]
            let rows_vec: Vec<Row> = groups_vec
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
                        let (_, row) = &rows[row_idx];
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
                    // Use CompactVec directly to avoid Vec→CompactVec conversion
                    let mut row_values: CompactVec<Value> =
                        CompactVec::with_capacity(group_by_items.len() + aggregations.len());
                    row_values.extend(group.key_values);

                    for (i, agg) in aggregations.iter().enumerate() {
                        let value = if let Some(ref func) = agg_funcs[i] {
                            func.result()?
                        } else if agg.name == "COUNT" && agg.column == "*" {
                            Value::Integer(group.row_indices.len() as i64)
                        } else {
                            Value::null_unknown()
                        };
                        row_values.push(value);
                    }

                    Ok(Row::from_compact_vec(row_values))
                })
                .collect::<Result<_>>()?;
            #[cfg(not(feature = "parallel"))]
            let rows_vec: Vec<Row> = groups_vec
                .into_iter()
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
                        let (_, row) = &rows[row_idx];
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
                    // Use CompactVec directly to avoid Vec→CompactVec conversion
                    let mut row_values: CompactVec<Value> =
                        CompactVec::with_capacity(group_by_items.len() + aggregations.len());
                    row_values.extend(group.key_values);

                    for (i, agg) in aggregations.iter().enumerate() {
                        let value = if let Some(ref func) = agg_funcs[i] {
                            func.result()?
                        } else if agg.name == "COUNT" && agg.column == "*" {
                            Value::Integer(group.row_indices.len() as i64)
                        } else {
                            Value::null_unknown()
                        };
                        row_values.push(value);
                    }

                    Ok(Row::from_compact_vec(row_values))
                })
                .collect::<Result<_>>()?;
            // Convert to RowVec with sequential IDs
            rows_vec
                .into_iter()
                .enumerate()
                .map(|(idx, row)| (idx as i64, row))
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
                        func.set_order_by(Self::aggregate_order_keys(&agg.order_by));
                    }
                }
            }

            // Buffer for evaluated expression values (to avoid repeated allocation)
            let mut expr_values: Vec<Value> = vec![Value::null_unknown(); aggregations.len()];

            let mut result_rows_seq = RowVec::with_capacity(groups_vec.len());
            let mut row_id = 0i64;
            for group in groups_vec {
                // Reset aggregate functions for this group
                for f in agg_funcs.iter_mut().flatten() {
                    f.reset();
                }

                // Accumulate values for this group
                // Pre-create static Value for COUNT(*)
                let count_star_value = Value::Integer(1);
                for &row_idx in &group.row_indices {
                    let (_, row) = &rows[row_idx];

                    // Create execution context for this row
                    // CRITICAL: Include params for parameterized queries
                    let exec_ctx = ExecuteContext::new(row)
                        .with_params(ctx.params())
                        .with_named_params(ctx.named_params());

                    for (i, agg) in aggregations.iter().enumerate() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            // Check FILTER clause first - skip row if filter is false
                            if let Some(ref filter_program) = compiled_agg_filters[i] {
                                if let Some(ref mut vm) = expr_vm {
                                    match vm.execute_cow(filter_program, &exec_ctx) {
                                        Ok(Value::Boolean(true)) => {} // Continue with accumulation
                                        Ok(Value::Boolean(false)) | Ok(Value::Null(_)) => continue, // Skip this row
                                        Ok(_) => continue, // Non-boolean treated as false
                                        Err(e) => return Err(e),
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
                                    match vm.execute_cow(expr_program, &exec_ctx) {
                                        Ok(val) => {
                                            expr_values[i] = val;
                                            Some(&expr_values[i])
                                        }
                                        Err(e) => {
                                            return Err(crate::core::Error::expression_evaluation(
                                                format!("{}({}): {}", agg.name, agg.column, e),
                                            ));
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
                                        for order_program in &compiled_agg_order_by[i] {
                                            sort_keys
                                                .push(vm.execute_cow(order_program, &exec_ctx)?);
                                        }
                                        func.accumulate_with_sort_key(v, sort_keys, agg.distinct);
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
                // Use CompactVec directly to avoid Vec→CompactVec conversion
                let mut row_values: CompactVec<Value> =
                    CompactVec::with_capacity(group_by_items.len() + aggregations.len());
                row_values.extend(group.key_values);

                for (i, agg) in aggregations.iter().enumerate() {
                    let value = if let Some(ref func) = agg_funcs[i] {
                        func.result()?
                    } else if agg.name == "COUNT" && agg.column == "*" {
                        Value::Integer(group.row_indices.len() as i64)
                    } else {
                        Value::null_unknown()
                    };
                    row_values.push(value);
                }

                result_rows_seq.push((row_id, Row::from_compact_vec(row_values)));
                row_id += 1;
            }
            result_rows_seq
        };

        // Build result columns
        let mut result_columns: Vec<String> =
            self.resolve_group_by_column_names_new(group_by_items, columns, col_index_map);
        result_columns.extend(aggregations.iter().map(|a| a.get_column_name()));

        // Slow path doesn't apply HAVING inline, so return false
        Ok((result_columns, result_rows, false))
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
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, RowVec)> {
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

        // Pre-compile FILTER expressions for aggregate functions
        let has_filters = aggregations.iter().any(|a| a.filter.is_some());
        let compiled_filters: Vec<Option<SharedProgram>> = if has_filters {
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

        let mut expr_vm = if has_expr_group_by || has_agg_expression || has_filters {
            Some(ExprVM::new())
        } else {
            None
        };

        // Collect all results from all grouping sets
        let mut all_result_rows: RowVec = RowVec::new();
        let mut row_id = 0i64;

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

                for (_, row) in rows {
                    // Create execution context for this row
                    // CRITICAL: Include params for parameterized queries
                    let exec_ctx = ExecuteContext::new(row)
                        .with_params(ctx.params())
                        .with_named_params(ctx.named_params());

                    for (i, agg) in aggregations.iter().enumerate() {
                        if let Some(ref mut func) = agg_funcs[i] {
                            // Check FILTER clause first - skip row if filter is false
                            if let Some(ref filter_program) = compiled_filters[i] {
                                if let Some(ref mut vm) = expr_vm {
                                    match vm.execute_cow(filter_program, &exec_ctx)? {
                                        Value::Boolean(true) => {}
                                        _ => continue,
                                    }
                                } else {
                                    continue;
                                }
                            }

                            if let Some(ref expr_program) = compiled_agg_expressions[i] {
                                if let Some(ref mut vm) = expr_vm {
                                    expr_values[i] = vm.execute_cow(expr_program, &exec_ctx)?;
                                    func.accumulate(&expr_values[i], agg.distinct);
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
                // Use CompactVec directly to avoid Vec→CompactVec conversion
                let mut row_values: CompactVec<Value> = CompactVec::with_capacity(
                    group_by_items.len() + aggregations.len() + group_by_items.len(),
                );
                for _ in group_by_items {
                    row_values.push(Value::null_unknown());
                }
                for (i, agg) in aggregations.iter().enumerate() {
                    let value = if let Some(ref func) = agg_funcs[i] {
                        func.result()?
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
                all_result_rows.push((row_id, Row::from_compact_vec(row_values)));
                row_id += 1;
            } else {
                // Partial grouping: group by active columns only
                // Use hash-based grouping with collision handling: u64 hash -> Vec<GroupEntry>
                // Each hash bucket can contain multiple groups (handles hash collisions correctly)
                // FxHashMap is optimized for trusted keys in embedded database context
                let mut groups: FxHashMap<u64, Vec<GroupEntry>> = FxHashMap::default();
                let mut key_buffer: Vec<Value> = Vec::with_capacity(active_count);

                for (row_idx, (_, row)) in rows.iter().enumerate() {
                    key_buffer.clear();

                    // Create execution context for this row
                    // CRITICAL: Include params for parameterized queries
                    let exec_ctx = ExecuteContext::new(row)
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
                                        vm.execute_cow(program, &exec_ctx)?
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
                            let (_, row) = &rows[row_idx];

                            // Create execution context for this row
                            // CRITICAL: Include params for parameterized queries
                            let exec_ctx = ExecuteContext::new(row)
                                .with_params(ctx.params())
                                .with_named_params(ctx.named_params());

                            for (i, agg) in aggregations.iter().enumerate() {
                                if let Some(ref mut func) = agg_funcs[i] {
                                    // Check FILTER clause first - skip row if filter is false
                                    if let Some(ref filter_program) = compiled_filters[i] {
                                        if let Some(ref mut vm) = expr_vm {
                                            match vm.execute_cow(filter_program, &exec_ctx)? {
                                                Value::Boolean(true) => {}
                                                _ => continue,
                                            }
                                        } else {
                                            continue;
                                        }
                                    }

                                    if let Some(ref expr_program) = compiled_agg_expressions[i] {
                                        if let Some(ref mut vm) = expr_vm {
                                            expr_values[i] =
                                                vm.execute_cow(expr_program, &exec_ctx)?;
                                            func.accumulate(&expr_values[i], agg.distinct);
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
                        // Use CompactVec directly to avoid Vec→CompactVec conversion
                        let mut row_values: CompactVec<Value> = CompactVec::with_capacity(
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
                                func.result()?
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

                        all_result_rows.push((row_id, Row::from_compact_vec(row_values)));
                        row_id += 1;
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

        let item_to_index: FxHashMap<ExprKey, usize> = group_by_items
            .iter()
            .enumerate()
            .map(|(i, item)| (group_by_item_key(item), i))
            .collect();

        for set in sets {
            let mut active_columns = vec![false; num_columns];

            for expr in set {
                if let Some(&idx) = item_to_index.get(&ExprKey::of(expr)) {
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
        col_index_map: &StringMap<usize>,
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
    fn lookup_column_index(column_lower: &str, col_index_map: &StringMap<usize>) -> Option<usize> {
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

    /// Apply HAVING clause to aggregated results
    #[allow(clippy::too_many_arguments)]
    fn apply_having(
        &self,
        result: Box<dyn QueryResult>,
        having: &Expression,
        columns: &[String],
        slots: &FxHashMap<ExprKey, usize>,
        outer_table: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Materialize the result
        let mut rows = RowVec::new();
        let mut result = result;
        let mut row_id = 0i64;
        while result.next() {
            rows.push((row_id, result.take_row()));
            row_id += 1;
        }

        // A subquery that reads the group it is asked about is run once per
        // group, with the group's row standing as the outer row, the way a
        // correlated subquery in the WHERE clause is run once per row. The
        // group's columns are offered under their bare names and under the
        // table's, since the subquery may name either
        if Self::has_correlated_subqueries(having) {
            let columns_arc = CompactArc::new(columns.to_vec());
            let keys: Vec<(CompactArc<str>, Option<CompactArc<str>>)> = columns
                .iter()
                .map(|c| {
                    let bare = c.to_lowercase();
                    let qualified = outer_table.filter(|_| !bare.contains('.')).map(|t| {
                        CompactArc::from(format!("{}.{}", t.to_lowercase(), bare).as_str())
                    });
                    (CompactArc::from(bare.as_str()), qualified)
                })
                .collect();
            let mut filtered_rows = RowVec::new();
            let mut new_id = 0i64;
            let mut vm = ExprVM::new();
            for (_, row) in rows {
                let mut outer_row_map: FxHashMap<CompactArc<str>, Value> = FxHashMap::default();
                for (i, (bare, qualified)) in keys.iter().enumerate() {
                    let value = row.get(i).cloned().unwrap_or_else(Value::null_unknown);
                    if let Some(q) = qualified {
                        outer_row_map.insert(q.clone(), value.clone());
                    }
                    outer_row_map.insert(bare.clone(), value);
                }
                let correlated_ctx =
                    ctx.with_outer_row(outer_row_map, CompactArc::clone(&columns_arc));
                let processed = self.process_correlated_where(having, &correlated_ctx)?;
                let filter = compile_over_slots(
                    &processed,
                    columns,
                    crate::functions::registry::global_registry(),
                    Some(slots),
                    None,
                )?;
                if vm.execute_bool_checked(&filter, &slot_context(&row, &correlated_ctx))? {
                    filtered_rows.push((new_id, row));
                    new_id += 1;
                }
            }
            return Ok(Box::new(ExecutorResult::new(
                columns.to_vec(),
                filtered_rows,
            )));
        }

        let having_filter = compile_over_slots(
            having,
            columns,
            crate::functions::registry::global_registry(),
            Some(slots),
            None,
        )?;
        let mut filtered_rows = RowVec::new();
        let mut new_id = 0i64;
        let mut vm = ExprVM::new();
        for (_, row) in rows {
            if vm.execute_bool_checked(&having_filter, &slot_context(&row, ctx))? {
                filtered_rows.push((new_id, row));
                new_id += 1;
            }
        }

        Ok(Box::new(ExecutorResult::new(
            columns.to_vec(),
            filtered_rows,
        )))
    }

    /// The SUM of one column over materialized rows, for SUM and AVG to
    /// read; large inputs sum their chunks in parallel and merge them
    fn sum_column_rows(rows: &[(i64, Row)], col_idx: usize) -> SumAccumulator {
        let sum_chunk = |chunk: &[(i64, Row)]| {
            let mut sum = SumAccumulator::default();
            for (_, row) in chunk {
                if let Some(value) = row.get(col_idx) {
                    sum.add(value);
                }
            }
            sum
        };
        #[cfg(feature = "parallel")]
        if rows.len() >= 10_000 {
            let chunk_size = (rows.len() / rayon::current_num_threads()).max(1000);
            return rows.par_chunks(chunk_size).map(sum_chunk).reduce(
                SumAccumulator::default,
                |mut total, part| {
                    total.merge(&part);
                    total
                },
            );
        }
        sum_chunk(rows)
    }

    /// Check if an expression is a PURE aggregate function call.
    ///
    /// Returns true only for:
    /// - `COUNT(*)`, `SUM(col)`, `MIN(col)`, `MAX(col)`, `AVG(col)` directly
    /// - Same with alias: `COUNT(*) AS cnt`
    ///
    /// Returns false for:
    /// - `SUM(col) + 10` (wrapped in Infix)
    /// - `-SUM(col)` (wrapped in Prefix)
    /// - `col * 2` (not an aggregate)
    fn is_pure_aggregate_expression(expr: &Expression) -> bool {
        match expr {
            Expression::FunctionCall(func) => {
                // Check if it's an aggregate function
                matches!(
                    func.function.to_uppercase().as_str(),
                    "COUNT" | "SUM" | "MIN" | "MAX" | "AVG"
                )
            }
            Expression::Aliased(aliased) => {
                // Check the inner expression
                Self::is_pure_aggregate_expression(&aliased.expression)
            }
            _ => false,
        }
    }

    /// Try to compute aggregates directly on the table without materializing rows.
    ///
    /// This is the "deferred aggregation" optimization for simple queries like:
    /// - `SELECT COUNT(*) FROM table`
    /// - `SELECT SUM(col), MIN(col), MAX(col) FROM table`
    ///
    /// Returns None if the optimization cannot be applied.
    /// Returns Some(result) if the aggregates were computed directly.
    ///
    /// # Eligibility
    /// - No WHERE clause (or simplified to nothing)
    /// - No GROUP BY clause
    /// - No HAVING clause
    /// - No window functions
    /// - Simple column aggregates only (no expressions like SUM(a+b))
    /// - No DISTINCT
    /// - No ORDER BY on aggregates (like STRING_AGG with ORDER BY)
    /// - No FILTER clause on aggregates
    pub(crate) fn try_aggregation_pushdown(
        &self,
        table: &dyn crate::storage::traits::Table,
        stmt: &SelectStatement,
        ctx: &super::context::ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> Result<Option<Box<dyn crate::storage::traits::QueryResult>>> {
        // classification is passed from caller to avoid redundant cache lookups
        if Self::aggregates_hold_subquery(stmt) {
            return Ok(None);
        }

        // Quick eligibility checks using cached classification
        if classification.has_where {
            return Ok(None);
        }
        if classification.has_group_by {
            return Ok(None);
        }
        if classification.has_having {
            return Ok(None);
        }
        if classification.has_window_functions {
            return Ok(None);
        }

        // CRITICAL: Check that each column expression is a PURE aggregate function
        // We cannot pushdown expressions like SUM(val) + 10, -SUM(val), etc.
        // Only handle direct function calls: SUM(val), COUNT(*), MAX(val), etc.
        for col in &stmt.columns {
            if !Self::is_pure_aggregate_expression(col) {
                return Ok(None);
            }
        }

        // Parse aggregations
        let (aggregations, non_agg_columns) = self.parse_aggregations(stmt)?;

        // Must have only aggregations, no regular columns
        if !non_agg_columns.is_empty() {
            return Ok(None);
        }
        if aggregations.is_empty() {
            return Ok(None);
        }

        // Check all aggregations are simple (no expression, no ORDER BY, no FILTER)
        // COUNT(DISTINCT col) is allowed if column has an index
        for agg in &aggregations {
            if agg.expression.is_some() || !agg.order_by.is_empty() || agg.filter.is_some() {
                return Ok(None);
            }
            // COUNT(DISTINCT col) is allowed, other DISTINCT aggregates are not
            if agg.distinct && agg.name != "COUNT" {
                return Ok(None);
            }
            // Only support COUNT, SUM, MIN, MAX, AVG
            match agg.name.as_str() {
                "COUNT" | "SUM" | "MIN" | "MAX" | "AVG" => {}
                _ => return Ok(None),
            }
        }

        // Build column index map using schema's cached lowercase column names
        let schema_lower = table.schema().column_names_lower_arc();
        let col_index_map: StringMap<usize> = schema_lower
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        // Compute each aggregate
        // Use CompactVec directly to avoid Vec→CompactVec conversion
        let mut result_values: CompactVec<Value> = CompactVec::with_capacity(aggregations.len());
        let mut result_columns: Vec<String> = Vec::with_capacity(aggregations.len());

        for agg in &aggregations {
            result_columns.push(agg.get_column_name());

            match agg.name.as_str() {
                "COUNT" => {
                    if agg.distinct {
                        // COUNT(DISTINCT col) - try to get count from index without cloning values
                        if let Some(count) = table.get_partition_count(&agg.column_lower)? {
                            // get_partition_count already excludes NULL values per SQL standard
                            result_values.push(Value::Integer(count as i64));
                        } else {
                            // No index on this column, can't pushdown
                            return Ok(None);
                        }
                    } else if agg.column == "*" {
                        // COUNT(*) - use row_count
                        let count = table.row_count()?;
                        result_values.push(Value::Integer(count as i64));
                    } else {
                        // COUNT(col) - need to count non-null values, can't pushdown easily
                        return Ok(None);
                    }
                }
                "SUM" => {
                    let col_idx = col_index_map.get(&agg.column_lower).copied();
                    if let Some(idx) = col_idx {
                        if let Some(sum) = table.sum_column(idx)? {
                            result_values.push(sum.sum()?);
                        } else {
                            return Ok(None); // Pushdown not available
                        }
                    } else {
                        return Ok(None); // Column not found
                    }
                }
                "AVG" => {
                    let col_idx = col_index_map.get(&agg.column_lower).copied();
                    if let Some(idx) = col_idx {
                        if let Some(sum) = table.avg_column(idx)? {
                            result_values.push(sum.avg());
                        } else {
                            return Ok(None); // Pushdown not available
                        }
                    } else {
                        return Ok(None); // Column not found
                    }
                }
                "MIN" => {
                    let col_idx = col_index_map.get(&agg.column_lower).copied();
                    if let Some(idx) = col_idx {
                        if let Some(min_val) = table.min_column(idx)? {
                            result_values
                                .push(min_val.unwrap_or_else(|| {
                                    Value::null(crate::core::DataType::Integer)
                                }));
                        } else {
                            return Ok(None); // Pushdown not available
                        }
                    } else {
                        return Ok(None); // Column not found
                    }
                }
                "MAX" => {
                    let col_idx = col_index_map.get(&agg.column_lower).copied();
                    if let Some(idx) = col_idx {
                        if let Some(max_val) = table.max_column(idx)? {
                            result_values
                                .push(max_val.unwrap_or_else(|| {
                                    Value::null(crate::core::DataType::Integer)
                                }));
                        } else {
                            return Ok(None); // Pushdown not available
                        }
                    } else {
                        return Ok(None); // Column not found
                    }
                }
                _ => return Ok(None),
            }
        }

        let row = Row::from_compact_vec(result_values);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));
        let slots = AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations: &aggregations,
            group_by_count: 0,
        };
        self.finish_aggregation(AggregatePath::Pushdown, slots, ctx, result_columns, rows)
            .map(Some)
    }

    /// Try to push filtered aggregation (WHERE + aggregates) directly to the storage layer.
    ///
    /// This handles queries like:
    /// - `SELECT COUNT(*) FROM orders WHERE status = 'shipped'`
    /// - `SELECT SUM(amount), AVG(amount) FROM sales WHERE region = 'US'`
    ///
    /// The WHERE clause is converted to a storage expression and passed alongside the
    /// aggregate operations to `Table::compute_filtered_aggregates`, which can scan and
    /// aggregate in a single pass without materializing Row objects in the executor.
    ///
    /// # Eligibility
    /// - Must have WHERE and aggregation, no HAVING/window functions/joins/GROUP BY
    /// - No DISTINCT aggregates, no subqueries or parameters in WHERE
    /// - All SELECT columns must be pure aggregate function calls
    ///
    /// # Returns
    /// - `Ok(Some(result))` if the pushdown was applied
    /// - `Ok(None)` if the query is not eligible (falls through to next path)
    pub(crate) fn try_filtered_aggregation_pushdown(
        &self,
        table: &dyn crate::storage::traits::Table,
        stmt: &SelectStatement,
        ctx: &super::context::ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
        columns: &[String],
    ) -> Result<Option<Box<dyn crate::storage::traits::QueryResult>>> {
        use crate::storage::mvcc::version_store::AggregateOp;

        if Self::aggregates_hold_subquery(stmt) {
            return Ok(None);
        }

        // --- Eligibility checks ---

        if !classification.has_where {
            return Ok(None);
        }
        if !classification.has_aggregation {
            return Ok(None);
        }
        if classification.has_having {
            return Ok(None);
        }
        if classification.has_window_functions {
            return Ok(None);
        }
        if classification.has_joins {
            return Ok(None);
        }
        if classification.has_group_by {
            return Ok(None);
        }
        // Parameters are not yet resolved in storage expressions
        if classification.where_has_parameters {
            return Ok(None);
        }
        // Subqueries in WHERE cannot be pushed to storage
        if classification.where_has_subqueries {
            return Ok(None);
        }

        // All SELECT columns must be pure aggregate function calls
        for col in &stmt.columns {
            if !Self::is_pure_aggregate_expression(col) {
                return Ok(None);
            }
        }

        // Parse aggregations to validate and extract details
        let (aggregations, non_agg_columns) = self.parse_aggregations(stmt)?;

        // Must have only aggregations, no regular columns
        if !non_agg_columns.is_empty() || aggregations.is_empty() {
            return Ok(None);
        }

        // Check all aggregations are simple (no expression, no ORDER BY, no FILTER, no DISTINCT)
        for agg in &aggregations {
            if agg.expression.is_some() || !agg.order_by.is_empty() || agg.filter.is_some() {
                return Ok(None);
            }
            if agg.distinct {
                return Ok(None);
            }
            match agg.name.as_str() {
                "COUNT" | "SUM" | "MIN" | "MAX" | "AVG" => {}
                _ => return Ok(None),
            }
        }

        // --- Convert WHERE clause to storage expression ---

        let where_expr = match stmt.where_clause.as_ref() {
            Some(expr) => expr,
            None => return Ok(None),
        };

        let schema = table.schema();
        let (storage_expr, needs_memory_filter) =
            super::pushdown::try_pushdown(where_expr, schema, Some(ctx));

        // Bail when the WHERE clause is only partially pushed to storage.
        // The residual predicate would be ignored, producing wrong aggregates.
        if needs_memory_filter {
            return Ok(None);
        }

        let storage_expr = match storage_expr {
            Some(expr) => expr,
            None => return Ok(None), // Cannot convert WHERE to storage expression
        };

        // --- Build (AggregateOp, column_index) pairs ---

        let col_map: FxHashMap<&str, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        let mut agg_ops: Vec<(AggregateOp, usize)> = Vec::with_capacity(aggregations.len());
        let mut result_columns: Vec<String> = Vec::with_capacity(aggregations.len());

        for agg in &aggregations {
            result_columns.push(agg.get_column_name());

            let (op, col_idx) = match agg.name.as_str() {
                "COUNT" => {
                    if agg.column == "*" {
                        (AggregateOp::CountStar, 0)
                    } else if let Some(&idx) = col_map.get(agg.column_lower.as_str()) {
                        (AggregateOp::Count, idx)
                    } else {
                        return Ok(None);
                    }
                }
                "SUM" => {
                    if let Some(&idx) = col_map.get(agg.column_lower.as_str()) {
                        (AggregateOp::Sum, idx)
                    } else {
                        return Ok(None);
                    }
                }
                "AVG" => {
                    if let Some(&idx) = col_map.get(agg.column_lower.as_str()) {
                        (AggregateOp::Avg, idx)
                    } else {
                        return Ok(None);
                    }
                }
                "MIN" => {
                    if let Some(&idx) = col_map.get(agg.column_lower.as_str()) {
                        (AggregateOp::Min, idx)
                    } else {
                        return Ok(None);
                    }
                }
                "MAX" => {
                    if let Some(&idx) = col_map.get(agg.column_lower.as_str()) {
                        (AggregateOp::Max, idx)
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            };
            agg_ops.push((op, col_idx));
        }

        // --- Call storage-level filtered aggregation ---

        let values = match table.compute_filtered_aggregates(&agg_ops, storage_expr.as_ref())? {
            Some(v) => v,
            None => return Ok(None),
        };

        // --- Build result ---

        let mut result_values: CompactVec<Value> = CompactVec::with_capacity(values.len());
        for v in values {
            result_values.push(v);
        }
        let row = Row::from_compact_vec(result_values);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));
        let slots = AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations: &aggregations,
            group_by_count: 0,
        };
        self.finish_aggregation(
            AggregatePath::FilteredPushdown,
            slots,
            ctx,
            result_columns,
            rows,
        )
        .map(Some)
    }

    /// Try to compute global aggregates using streaming (no row materialization).
    ///
    /// This is a fallback for queries that can't use direct aggregation pushdown,
    /// but can still avoid collecting all rows by streaming through a scanner.
    ///
    /// Examples of eligible queries:
    /// - `SELECT AVG(col) * 100 FROM table`  (expression wrapping aggregate)
    /// - `SELECT SUM(col), COUNT(*) FROM table`  (multiple simple aggregates)
    ///
    /// # Returns
    /// - `Some(result)` if streaming aggregation was used
    /// - `None` if the query is not eligible for streaming
    pub(crate) fn try_streaming_global_aggregation(
        &self,
        table: &dyn crate::storage::traits::Table,
        stmt: &SelectStatement,
        ctx: &super::context::ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> Result<Option<Box<dyn crate::storage::traits::QueryResult>>> {
        // Quick eligibility checks using cached classification
        if classification.has_where || Self::aggregates_hold_subquery(stmt) {
            return Ok(None);
        }
        if classification.has_group_by {
            return Ok(None);
        }
        if classification.has_having {
            return Ok(None);
        }
        if classification.has_window_functions {
            return Ok(None);
        }
        if classification.has_order_by {
            return Ok(None);
        }
        if classification.has_limit {
            return Ok(None);
        }

        // Parse aggregations - allow non-pure expressions (like AVG(col) * 100)
        let (aggregations, non_agg_columns) = self.parse_aggregations(stmt)?;

        // Must have only aggregations, no regular columns
        if !non_agg_columns.is_empty() {
            return Ok(None);
        }
        if aggregations.is_empty() {
            return Ok(None);
        }

        // Check all aggregations are simple enough for streaming
        // (no ORDER BY, no FILTER, no DISTINCT except COUNT, no expression arguments)
        for agg in &aggregations {
            if !agg.order_by.is_empty() || agg.filter.is_some() {
                return Ok(None);
            }
            if agg.distinct && agg.name != "COUNT" {
                return Ok(None);
            }
            // Can't handle expression arguments like SUM(a + b) - need full evaluation
            if agg.expression.is_some() {
                return Ok(None);
            }
            // Only support COUNT, SUM, MIN, MAX, AVG for streaming
            match agg.name.as_str() {
                "COUNT" | "SUM" | "MIN" | "MAX" | "AVG" => {}
                _ => return Ok(None),
            }
        }

        // Build column index map using schema's cached lowercase column names
        let schema_lower = table.schema().column_names_lower_arc();
        let col_index_map: StringMap<usize> = schema_lower
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        // Pre-compute column indices for each aggregation
        let agg_col_indices: Vec<Option<usize>> = aggregations
            .iter()
            .map(|agg| {
                if agg.column == "*" || agg.expression.is_some() {
                    None
                } else {
                    Self::lookup_column_index(&agg.column_lower, &col_index_map)
                }
            })
            .collect();

        // FAST PATH: Try deferred aggregation (no row materialization)
        // This bypasses the lazy scanner entirely for simple aggregates
        let can_use_deferred = aggregations.iter().all(|agg| !agg.distinct);
        if can_use_deferred {
            let mut deferred_values: Vec<Option<Value>> = Vec::with_capacity(aggregations.len());
            let mut all_succeeded = true;

            for (i, agg) in aggregations.iter().enumerate() {
                let col_idx = agg_col_indices[i];
                let value = match agg.name.as_str() {
                    "COUNT" => {
                        if agg.column == "*" {
                            // COUNT(*) - use row_count()
                            Some(Value::Integer(table.row_count()? as i64))
                        } else {
                            // COUNT(col) - need scanner for NULL checking
                            // Could optimize with a dedicated count_non_null method
                            None
                        }
                    }
                    "SUM" => match col_idx {
                        Some(idx) => table.sum_column(idx)?.map(|sum| sum.sum()).transpose()?,
                        None => None,
                    },
                    "AVG" => match col_idx {
                        Some(idx) => table.avg_column(idx)?.map(|sum| sum.avg()),
                        None => None,
                    },
                    "MIN" => match col_idx {
                        Some(idx) => table.min_column(idx)?.map(|min_opt| {
                            min_opt.unwrap_or_else(|| Value::null(crate::core::DataType::Integer))
                        }),
                        None => None,
                    },
                    "MAX" => match col_idx {
                        Some(idx) => table.max_column(idx)?.map(|max_opt| {
                            max_opt.unwrap_or_else(|| Value::null(crate::core::DataType::Integer))
                        }),
                        None => None,
                    },
                    _ => None,
                };

                if let Some(v) = value {
                    deferred_values.push(Some(v));
                } else {
                    all_succeeded = false;
                    break;
                }
            }

            if all_succeeded && deferred_values.len() == aggregations.len() {
                // Build result from deferred values
                // Use CompactVec directly to avoid Vec→CompactVec conversion
                let mut agg_result_values: CompactVec<Value> =
                    CompactVec::with_capacity(aggregations.len());
                let mut agg_result_columns: Vec<String> = Vec::with_capacity(aggregations.len());

                for (i, agg) in aggregations.iter().enumerate() {
                    agg_result_columns.push(agg.get_column_name());
                    agg_result_values.push(deferred_values[i].take().unwrap());
                }

                let agg_row = Row::from_compact_vec(agg_result_values);
                let mut agg_rows = RowVec::with_capacity(1);
                agg_rows.push((0, agg_row));
                let slots = AggregateSlots {
                    stmt,
                    written_stmt: stmt,
                    aggregations: &aggregations,
                    group_by_count: 0,
                };
                return self
                    .finish_aggregation(
                        AggregatePath::StreamingGlobal,
                        slots,
                        ctx,
                        agg_result_columns,
                        agg_rows,
                    )
                    .map(Some);
            }
        }

        // SLOW PATH: Fall back to scanner-based streaming
        // Initialize aggregate states
        struct AggState {
            sum: SumAccumulator,
            count: i64,
            min: Option<Value>,
            max: Option<Value>,
            distinct_set: Option<ahash::AHashSet<u64>>,
        }

        let mut states: Vec<AggState> = aggregations
            .iter()
            .map(|agg| AggState {
                sum: SumAccumulator::default(),
                count: 0,
                min: None,
                max: None,
                distinct_set: if agg.distinct {
                    Some(ahash::AHashSet::new())
                } else {
                    None
                },
            })
            .collect();

        // Get a scanner and stream through rows
        let mut scanner = table.scan(&[], None)?;

        while scanner.next() {
            let row = scanner.row();

            for (i, agg) in aggregations.iter().enumerate() {
                let state = &mut states[i];

                if agg.column == "*" {
                    // COUNT(*)
                    state.count += 1;
                    continue;
                }

                let col_idx = match agg_col_indices[i] {
                    Some(idx) => idx,
                    None => continue,
                };

                let value = match row.get(col_idx) {
                    Some(v) if !v.is_null() => v,
                    _ => continue, // Skip NULL values
                };

                // Handle DISTINCT
                if let Some(ref mut distinct_set) = state.distinct_set {
                    let hash = {
                        use std::hash::{Hash, Hasher};
                        let mut hasher = ahash::AHasher::default();
                        match value {
                            Value::Integer(i) => i.hash(&mut hasher),
                            Value::Float(f) => grouping_float(*f).to_bits().hash(&mut hasher),
                            Value::Text(s) => s.hash(&mut hasher),
                            Value::Boolean(b) => b.hash(&mut hasher),
                            _ => continue,
                        }
                        hasher.finish()
                    };
                    if !distinct_set.insert(hash) {
                        continue; // Already seen this value
                    }
                }

                match agg.name.as_str() {
                    "COUNT" => {
                        state.count += 1;
                    }
                    "SUM" | "AVG" => state.sum.add(value),
                    "MIN" => {
                        let is_smaller = match (&state.min, value) {
                            (None, _) => true,
                            (Some(current), new) => {
                                new.compare(current).unwrap_or(std::cmp::Ordering::Equal)
                                    == std::cmp::Ordering::Less
                            }
                        };
                        if is_smaller {
                            state.min = Some(value.clone());
                        }
                    }
                    "MAX" => {
                        let is_larger = match (&state.max, value) {
                            (None, _) => true,
                            (Some(current), new) => {
                                new.compare(current).unwrap_or(std::cmp::Ordering::Equal)
                                    == std::cmp::Ordering::Greater
                            }
                        };
                        if is_larger {
                            state.max = Some(value.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(e) = scanner.err() {
            return Err(crate::core::Error::internal(format!("scan error: {}", e)));
        }
        scanner.close()?;

        // Build intermediate result columns (raw aggregate values)
        // Use CompactVec directly to avoid Vec→CompactVec conversion
        let mut agg_result_values: CompactVec<Value> =
            CompactVec::with_capacity(aggregations.len());
        let mut agg_result_columns: Vec<String> = Vec::with_capacity(aggregations.len());

        for (i, agg) in aggregations.iter().enumerate() {
            let state = &states[i];
            agg_result_columns.push(agg.get_column_name());

            let value = match agg.name.as_str() {
                "COUNT" => Value::Integer(state.count),
                "SUM" => state.sum.sum()?,
                "AVG" => state.sum.avg(),
                "MIN" => state
                    .min
                    .clone()
                    .unwrap_or_else(|| Value::null(crate::core::DataType::Integer)),
                "MAX" => state
                    .max
                    .clone()
                    .unwrap_or_else(|| Value::null(crate::core::DataType::Integer)),
                _ => Value::null(crate::core::DataType::Integer),
            };
            agg_result_values.push(value);
        }

        let agg_row = Row::from_compact_vec(agg_result_values);
        let mut agg_rows = RowVec::with_capacity(1);
        agg_rows.push((0, agg_row));
        let slots = AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations: &aggregations,
            group_by_count: 0,
        };
        self.finish_aggregation(
            AggregatePath::StreamingGlobalScan,
            slots,
            ctx,
            agg_result_columns,
            agg_rows,
        )
        .map(Some)
    }

    /// Try streaming aggregation for derived tables (FROM subqueries).
    ///
    /// OPTIMIZATION: For simple GROUP BY + COUNT(*) on derived tables without WHERE clause,
    /// stream directly to aggregation HashMap without materializing all rows first.
    /// This reduces memory allocations from O(N) to O(groups).
    ///
    /// Returns None if the optimization cannot be applied.
    ///
    /// Supported patterns:
    /// - Single-column GROUP BY (column reference)
    /// - Simple aggregates: COUNT(*), COUNT(col), SUM, AVG, MIN, MAX
    /// - No HAVING, no DISTINCT on non-COUNT aggregates
    /// - No ORDER BY clause, no LIMIT on groups
    pub(crate) fn try_streaming_derived_table_aggregation(
        &self,
        mut result: Box<dyn QueryResult>,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        classification: &std::sync::Arc<QueryClassification>,
    ) -> Result<Option<Box<dyn QueryResult>>> {
        use crate::common::SmartString;
        use smallvec::SmallVec;

        if Self::aggregates_hold_subquery(stmt) {
            return Ok(None);
        }

        // Quick eligibility checks
        if !classification.has_group_by {
            return Ok(None);
        }
        if classification.has_having {
            return Ok(None);
        }
        if classification.has_window_functions {
            return Ok(None);
        }
        // Skip ROLLUP/CUBE/GROUPING SETS
        if stmt.group_by.modifier != crate::parser::ast::GroupByModifier::None {
            return Ok(None);
        }

        // Only single-column GROUP BY for this optimization
        if stmt.group_by.columns.len() != 1 {
            return Ok(None);
        }

        // The derived table's columns carry no qualifier
        let Some((group_col_name, _)) = self.single_group_column(stmt)? else {
            return Ok(None);
        };

        // Build column index map from result columns
        let source_columns = result.columns().to_vec();
        let col_index_map: StringMap<usize> = source_columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_lowercase(), i))
            .collect();

        // Find GROUP BY column index
        let group_col_idx = match col_index_map.get(&group_col_name) {
            Some(&idx) => idx,
            None => return Ok(None),
        };

        // Parse aggregations
        let (aggregations, non_agg_columns) = self.parse_aggregations(stmt)?;

        // The select list reads the group column, bare or qualified, and
        // aggregate calls; the shared end projects the slots onto it
        let readable = |expr: &crate::parser::ast::Expression| {
            use crate::parser::ast::Expression;
            let inner = match expr {
                Expression::Aliased(aliased) => aliased.expression.as_ref(),
                other => other,
            };
            match inner {
                Expression::Identifier(id) => id.value_lower.eq_ignore_ascii_case(&group_col_name),
                Expression::QualifiedIdentifier(qid) => {
                    qid.name.value_lower.eq_ignore_ascii_case(&group_col_name)
                }
                Expression::FunctionCall(func) => {
                    crate::executor::utils::is_aggregate_function(&func.function)
                }
                _ => false,
            }
        };
        if !stmt.columns.iter().all(readable)
            || !non_agg_columns
                .iter()
                .all(|column| column.eq_ignore_ascii_case(&group_col_name))
        {
            return Ok(None);
        }

        // Check all aggregations are simple enough for streaming
        let simple_aggs: Vec<Option<SimpleAgg>> = aggregations
            .iter()
            .map(|agg| {
                // Must not have DISTINCT, FILTER, ORDER BY, or expression.
                // This path has no dedup state, so even COUNT(DISTINCT)
                // cannot stream here.
                if agg.filter.is_some() || !agg.order_by.is_empty() || agg.expression.is_some() {
                    return None;
                }
                if agg.distinct {
                    return None;
                }

                match agg.name.to_uppercase().as_str() {
                    "COUNT" => {
                        if agg.column == "*" {
                            Some(SimpleAgg::Count)
                        } else {
                            Self::lookup_column_index(&agg.column_lower, &col_index_map)
                                .map(SimpleAgg::CountColumn)
                        }
                    }
                    "SUM" => {
                        if agg.column == "*" {
                            None
                        } else {
                            Self::lookup_column_index(&agg.column_lower, &col_index_map)
                                .map(SimpleAgg::Sum)
                        }
                    }
                    "AVG" => {
                        if agg.column == "*" {
                            None
                        } else {
                            Self::lookup_column_index(&agg.column_lower, &col_index_map)
                                .map(SimpleAgg::Avg)
                        }
                    }
                    "MIN" => {
                        if agg.column == "*" {
                            None
                        } else {
                            Self::lookup_column_index(&agg.column_lower, &col_index_map)
                                .map(SimpleAgg::Min)
                        }
                    }
                    "MAX" => {
                        if agg.column == "*" {
                            None
                        } else {
                            Self::lookup_column_index(&agg.column_lower, &col_index_map)
                                .map(SimpleAgg::Max)
                        }
                    }
                    _ => None,
                }
            })
            .collect();

        // All aggregates must be resolved for streaming path
        if simple_aggs.iter().any(|a| a.is_none()) {
            return Ok(None);
        }

        let simple_aggs: Vec<SimpleAgg> = simple_aggs.into_iter().map(|a| a.unwrap()).collect();
        let num_aggs = simple_aggs.len();

        // State for streaming aggregation
        type AggVec<T> = SmallVec<[T; 4]>;

        #[derive(Clone)]
        struct StreamGroupState {
            sums: AggVec<SumAccumulator>,
            counts: AggVec<i64>,
            min_values: AggVec<Option<Value>>,
            max_values: AggVec<Option<Value>>,
        }

        // Template for new group state
        let state_template = StreamGroupState {
            sums: smallvec::smallvec![SumAccumulator::default(); num_aggs],
            counts: smallvec::smallvec![0; num_aggs],
            min_values: smallvec::smallvec![None; num_aggs],
            max_values: smallvec::smallvec![None; num_aggs],
        };

        let mut slot_columns = Vec::with_capacity(1 + aggregations.len());
        slot_columns.push(source_columns[group_col_idx].clone());
        slot_columns.extend(aggregations.iter().map(|agg| agg.get_column_name()));

        // Sample first row to detect key type
        if !result.next() {
            if let Some(err) = result.last_error() {
                return Err(err);
            }
            // No group: no row, under the columns the shared end names
            let slots = AggregateSlots {
                stmt,
                written_stmt: stmt,
                aggregations: &aggregations,
                group_by_count: 1,
            };
            return self
                .finish_aggregation(
                    AggregatePath::StreamingDerivedTable,
                    slots,
                    ctx,
                    slot_columns,
                    RowVec::new(),
                )
                .map(Some);
        }

        let first_row = result.row();
        let first_value = first_row.get(group_col_idx);

        // Use string fast path for Text values (common for derived tables with CASE expressions)
        let use_string_path = first_value
            .map(|v| matches!(v, Value::Text(_)))
            .unwrap_or(false);

        if use_string_path {
            // String GROUP BY streaming path
            // OPTIMIZATION: Use hashbrown::HashMap with raw_entry_mut to avoid SmartString allocation on lookup
            type FxBuildHasher = std::hash::BuildHasherDefault<FxHasher>;
            let mut groups: hashbrown::HashMap<SmartString, StreamGroupState, FxBuildHasher> =
                hashbrown::HashMap::with_capacity_and_hasher(64, FxBuildHasher::default());
            let mut null_group: Option<StreamGroupState> = None;

            // Process first row
            let process_row =
                |row: &Row,
                 groups: &mut hashbrown::HashMap<SmartString, StreamGroupState, FxBuildHasher>,
                 null_group: &mut Option<StreamGroupState>,
                 template: &StreamGroupState| {
                    let key_opt = match row.get(group_col_idx) {
                        Some(Value::Text(s)) => Some(s),
                        Some(Value::Null(_)) | None => None,
                        _ => return, // Skip non-text, non-NULL
                    };

                    let state = if let Some(key_str) = key_opt {
                        // OPTIMIZATION: Use raw_entry_mut to avoid SmartString allocation on lookup
                        // Only create SmartString when inserting a new group
                        let mut hasher = FxHasher::default();
                        std::hash::Hash::hash(key_str, &mut hasher);
                        let hash = hasher.finish();

                        let entry = groups
                            .raw_entry_mut()
                            .from_hash(hash, |k| k.as_str() == key_str);
                        match entry {
                            RawEntryMut::Occupied(o) => o.into_mut(),
                            RawEntryMut::Vacant(v) => {
                                v.insert_hashed_nocheck(
                                    hash,
                                    SmartString::new(key_str),
                                    template.clone(),
                                )
                                .1
                            }
                        }
                    } else {
                        if null_group.is_none() {
                            *null_group = Some(template.clone());
                        }
                        null_group.as_mut().unwrap()
                    };

                    // Accumulate aggregates
                    for (i, agg) in simple_aggs.iter().enumerate() {
                        match agg {
                            SimpleAgg::Count => {
                                state.counts[i] += 1;
                            }
                            SimpleAgg::CountColumn(col_idx) => {
                                if let Some(value) = row.get(*col_idx) {
                                    if !value.is_null() {
                                        state.counts[i] += 1;
                                    }
                                }
                            }
                            SimpleAgg::Sum(col_idx) | SimpleAgg::Avg(col_idx) => {
                                if let Some(value) = row.get(*col_idx) {
                                    state.sums[i].add(value);
                                }
                            }
                            SimpleAgg::Min(col_idx) => {
                                if let Some(value) = row.get(*col_idx) {
                                    if !value.is_null() {
                                        match &state.min_values[i] {
                                            None => state.min_values[i] = Some(value.clone()),
                                            Some(current) if value < current => {
                                                state.min_values[i] = Some(value.clone())
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            SimpleAgg::Max(col_idx) => {
                                if let Some(value) = row.get(*col_idx) {
                                    if !value.is_null() {
                                        match &state.max_values[i] {
                                            None => state.max_values[i] = Some(value.clone()),
                                            Some(current) if value > current => {
                                                state.max_values[i] = Some(value.clone())
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

            // Process first row (already fetched)
            process_row(first_row, &mut groups, &mut null_group, &state_template);

            // Stream through remaining rows
            while result.next() {
                let row = result.row();
                process_row(row, &mut groups, &mut null_group, &state_template);
            }
            if let Some(err) = result.last_error() {
                return Err(err);
            }

            // Build result rows
            let build_row = |key_value: Value, state: StreamGroupState| -> Result<Row> {
                let mut values: crate::common::CompactVec<Value> =
                    crate::common::CompactVec::with_capacity(1 + simple_aggs.len());
                values.push(key_value);
                for (i, agg) in simple_aggs.iter().enumerate() {
                    let value = match agg {
                        SimpleAgg::Count | SimpleAgg::CountColumn(_) => {
                            Value::Integer(state.counts[i])
                        }
                        SimpleAgg::Sum(_) => state.sums[i].sum()?,
                        SimpleAgg::Avg(_) => state.sums[i].avg(),
                        SimpleAgg::Min(_) => state.min_values[i]
                            .clone()
                            .unwrap_or_else(|| Value::null(crate::core::DataType::Integer)),
                        SimpleAgg::Max(_) => state.max_values[i]
                            .clone()
                            .unwrap_or_else(|| Value::null(crate::core::DataType::Integer)),
                    };
                    values.push(value);
                }
                Ok(Row::from_compact_vec(values))
            };

            let mut result_rows = RowVec::with_capacity(groups.len() + 1);
            let mut row_id = 0i64;
            for (key, state) in groups.into_iter() {
                result_rows.push((row_id, build_row(Value::Text(key), state)?));
                row_id += 1;
            }
            if let Some(ng) = null_group {
                result_rows.push((row_id, build_row(Value::null_unknown(), ng)?));
            }

            let slots = AggregateSlots {
                stmt,
                written_stmt: stmt,
                aggregations: &aggregations,
                group_by_count: 1,
            };
            return self
                .finish_aggregation(
                    AggregatePath::StreamingDerivedTable,
                    slots,
                    ctx,
                    slot_columns,
                    result_rows,
                )
                .map(Some);
        }

        // Fallback: not eligible for streaming optimization
        Ok(None)
    }

    /// Try to use storage-level aggregation for GROUP BY queries.
    ///
    /// This optimization bypasses row materialization by computing aggregates
    /// directly from arena storage using Arc::clone for group keys.
    ///
    /// Returns None if the optimization cannot be applied.
    ///
    /// Currently only applies to simple queries with:
    /// - GROUP BY columns that match SELECT identifiers exactly (same order)
    /// - Simple aggregates (COUNT, SUM, AVG, MIN, MAX) on column references
    /// - No WHERE, HAVING, ROLLUP, CUBE, or GROUPING SETS
    pub fn try_storage_aggregation(
        &self,
        table: &dyn crate::storage::traits::Table,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        all_columns: &[String],
        classification: &QueryClassification,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        use crate::parser::ast::GroupByModifier;
        use crate::storage::mvcc::version_store::AggregateOp;

        if Self::aggregates_hold_subquery(stmt) {
            return None;
        }

        // Only for GROUP BY without WHERE or HAVING
        if classification.has_where || !classification.has_group_by || classification.has_having {
            return None;
        }

        // Only for simple GROUP BY (no ROLLUP, CUBE, or GROUPING SETS)
        if !matches!(stmt.group_by.modifier, GroupByModifier::None) {
            return None;
        }

        // Only for simple GROUP BY expressions (column references)
        let group_by_cols = &stmt.group_by.columns;
        if group_by_cols.is_empty() {
            return None;
        }

        // Only for single-column GROUP BY (multi-column causes Vec allocation per row)
        if group_by_cols.len() != 1 {
            return None;
        }

        // Build column name -> index map
        let col_map: FxHashMap<&str, usize> = all_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        // Extract group-by column names and indices (in GROUP BY order)
        let col_name = match self.single_group_column(stmt) {
            Ok(Some((name, false))) => name,
            Ok(_) => return None,
            Err(e) => return Some(Err(e)),
        };
        let &idx = col_map.get(col_name.as_str())?;
        let group_by_indices: Vec<usize> = vec![idx];
        let group_by_col_names: Vec<String> = vec![col_name];

        // The SELECT list reads the group column and plain aggregates; the
        // shared end projects the slots onto it, in any order and under any alias
        for col_expr in &stmt.columns {
            let expr = match col_expr {
                Expression::Aliased(aliased) => aliased.expression.as_ref(),
                other => other,
            };
            match expr {
                Expression::Identifier(ident) => {
                    if !group_by_col_names.contains(&ident.value.to_lowercase().to_string()) {
                        return None;
                    }
                }
                Expression::FunctionCall(fc)
                    if matches!(
                        fc.function.to_uppercase().as_str(),
                        "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"
                    ) => {}
                _ => return None,
            }
        }

        // One storage aggregate per slot, ORDER BY's own included
        let (aggregations, _) = match self.parse_aggregations(stmt) {
            Ok(parsed) => parsed,
            Err(error) => return Some(Err(error)),
        };
        if aggregations.is_empty() {
            return None;
        }
        let mut aggregates: Vec<(AggregateOp, usize)> = Vec::with_capacity(aggregations.len());
        for agg in &aggregations {
            if agg.distinct
                || agg.filter.is_some()
                || agg.expression.is_some()
                || !agg.order_by.is_empty()
                || !agg.extra_args.is_empty()
            {
                return None;
            }
            let column = || col_map.get(agg.column_lower.as_str()).copied();
            aggregates.push(match (agg.name.as_str(), agg.column.as_str()) {
                ("COUNT", "*") => (AggregateOp::CountStar, 0),
                ("COUNT", _) => (AggregateOp::Count, column()?),
                ("SUM", _) => (AggregateOp::Sum, column()?),
                ("AVG", _) => (AggregateOp::Avg, column()?),
                ("MIN", _) => (AggregateOp::Min, column()?),
                ("MAX", _) => (AggregateOp::Max, column()?),
                _ => return None,
            });
        }

        let results = match table.compute_grouped_aggregates(&group_by_indices, &aggregates) {
            Ok(results) => results?,
            Err(error) => return Some(Err(error)),
        };

        let mut slot_columns: Vec<String> = group_by_indices
            .iter()
            .map(|&i| all_columns[i].clone())
            .collect();
        slot_columns.extend(aggregations.iter().map(|agg| agg.get_column_name()));
        let mut rows = RowVec::new();
        for (row_id, r) in results.into_iter().enumerate() {
            let mut values = r.group_values;
            values.extend(r.aggregate_values);
            rows.push((
                row_id as i64,
                Row::from_compact_vec(CompactVec::from_vec(values)),
            ));
        }
        let slots = AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations: &aggregations,
            group_by_count: group_by_indices.len(),
        };
        Some(self.finish_aggregation(
            AggregatePath::StorageGrouped,
            slots,
            ctx,
            slot_columns,
            rows,
        ))
    }

    /// Try fast COUNT(DISTINCT col) using compiled cache
    ///
    /// This is a compiled fast path for simple `SELECT COUNT(DISTINCT col) FROM table` queries.
    /// On first execution, it analyzes and compiles the query pattern.
    /// On subsequent executions, it skips parsing and directly fetches the distinct count.
    ///
    /// # Arguments
    /// * `stmt` - The SELECT statement
    /// * `compiled` - The compiled execution state (shared via RwLock)
    ///
    /// # Returns
    /// * `Some(Ok(result))` - Query succeeded via fast path
    /// * `Some(Err(e))` - Query failed
    /// * `None` - Query doesn't qualify for this fast path (use normal path)
    pub(crate) fn try_fast_count_distinct_compiled(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Caller (try_compiled_fast_paths) guarantees no explicit transaction
        // is active; this path reads committed state only.

        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable(epoch)
                    if self.engine.schema_epoch() == *epoch =>
                {
                    return None
                }
                CompiledExecution::CountDistinct(cd) => {
                    // Fast path: validate epoch and execute directly
                    if self.engine.schema_epoch() == cd.cached_epoch {
                        return self.execute_compiled_count_distinct(cd).transpose();
                    }
                    // Schema changed - fall through to recompile
                }
                CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - fall through to recompile
                // Other variants - not a COUNT DISTINCT query
                _ => return None,
            }
        }

        // First execution or schema changed - compile and cache (write lock)
        self.compile_and_execute_count_distinct(stmt, compiled)
    }

    /// Execute using pre-compiled COUNT(DISTINCT col) info; None when the
    /// index cannot answer for this statement and the regular path does
    fn execute_compiled_count_distinct(
        &self,
        cd: &CompiledCountDistinct,
    ) -> Result<Option<Box<dyn QueryResult>>> {
        // Under read committed the table's own distinct count is the latest
        // committed one: it answers only when no commit published across its
        // reads. Else the count is read through a view
        let quiet_count = || -> Result<Option<usize>> {
            let tx = self.engine.begin_transaction()?;
            let table = tx.get_statement_table_at(&cd.table_name, None)?;
            table.get_partition_count(&cd.column_name)
        };
        let read_committed =
            self.engine.get_isolation_level() == crate::core::IsolationLevel::ReadCommitted;
        let count = match read_committed.then(quiet_count).transpose()?.flatten() {
            Some(count) => count,
            None => {
                let view = self.autocommit_view();
                let tx = self.engine.begin_transaction()?;
                let table = tx.get_statement_table_at(&cd.table_name, view.as_ref())?;
                let Some(count) = table.get_partition_count(&cd.column_name)? else {
                    return Ok(None);
                };
                view.as_ref()
                    .map_or(Ok(()), crate::storage::mvcc::registry::StatementView::held)?;
                count
            }
        };

        // Build result
        let mut result_values = CompactVec::with_capacity(1);
        result_values.push(Value::Integer(count as i64));
        let row = Row::from_compact_vec(result_values);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));

        Ok(Some(Box::new(ExecutorResult::new(
            vec![cd.result_column_name.clone()],
            rows,
        ))))
    }

    /// Compile and execute COUNT(DISTINCT col), caching the compiled state
    fn compile_and_execute_count_distinct(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        use crate::common::SmartString;

        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check (another thread may have compiled while we waited)
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(epoch) if self.engine.schema_epoch() == *epoch => {
                return None
            }
            CompiledExecution::CountDistinct(cd) => {
                if self.engine.schema_epoch() == cd.cached_epoch {
                    return self.execute_compiled_count_distinct(cd).transpose();
                }
                // Schema changed, continue to recompile
            }
            CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - recompile
            _ => return None,
        }

        // Pattern detection: SELECT COUNT(DISTINCT col) FROM table
        // Must have:
        // - Exactly one column expression
        // - That column is COUNT(DISTINCT col) function call
        // - No WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, CTEs, set operations
        // - Single table source (no joins)

        if stmt.columns.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Check for disqualifying clauses
        if stmt.where_clause.is_some()
            || !stmt.group_by.columns.is_empty()
            || stmt.having.is_some()
            || !stmt.order_by.is_empty()
            || stmt.limit.is_some()
            || stmt.offset.is_some()
            || stmt.with.is_some()
            || !stmt.set_operations.is_empty()
        {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Extract COUNT(DISTINCT col) pattern
        let (column_name, result_column_name) = match &stmt.columns[0] {
            Expression::FunctionCall(func) => {
                // Must be COUNT function
                if func.function.to_uppercase() != "COUNT" {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
                // Must be DISTINCT - if not, don't mark as NotOptimizable
                // because COUNT(*) fast path might handle it
                if !func.is_distinct {
                    return None;
                }
                if func.arguments.len() != 1 {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
                // Get column name from argument
                let col = match &func.arguments[0] {
                    Expression::Identifier(ident) => ident.value.to_lowercase(),
                    _ => {
                        *compiled_guard =
                            CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                        return None;
                    }
                };
                let result_name = format!("COUNT(DISTINCT {})", col);
                (col, result_name)
            }
            Expression::Aliased(aliased) => {
                // Handle COUNT(DISTINCT col) AS alias
                match aliased.expression.as_ref() {
                    Expression::FunctionCall(func) => {
                        // Must be COUNT function
                        if func.function.to_uppercase() != "COUNT" {
                            *compiled_guard =
                                CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                            return None;
                        }
                        // Must be DISTINCT - if not, don't mark as NotOptimizable
                        // because COUNT(*) fast path might handle it
                        if !func.is_distinct {
                            return None;
                        }
                        if func.arguments.len() != 1 {
                            *compiled_guard =
                                CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                            return None;
                        }
                        let col = match &func.arguments[0] {
                            Expression::Identifier(ident) => ident.value.to_lowercase(),
                            _ => {
                                *compiled_guard =
                                    CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                                return None;
                            }
                        };
                        (col, aliased.alias.value.to_string())
                    }
                    _ => {
                        *compiled_guard =
                            CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                        return None;
                    }
                }
            }
            _ => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        // Extract table name (bail out for AS OF temporal queries)
        let table_name = match stmt.table_expr.as_deref() {
            Some(Expression::TableSource(ts)) => {
                if ts.as_of.is_some() {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
                ts.name.value_lower.clone()
            }
            _ => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        // Try to get table and verify index exists
        let tx = match self.engine.begin_transaction() {
            Ok(tx) => tx,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        let table = match tx.get_statement_table_at(&table_name, self.autocommit_view().as_ref()) {
            Ok(t) => t,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        // Cache only an available count; storage failures must reach the caller.
        let count = match table.get_partition_count(&column_name) {
            Ok(Some(c)) => c,
            // An index that cannot answer for this statement may for the next
            Ok(None) if table.get_index_on_column(&column_name).is_some() => return None,
            Ok(None) => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
            Err(error) => return Some(Err(error)),
        };

        // Cache the compiled state
        let compiled_cd = CompiledCountDistinct {
            table_name: SmartString::new(&table_name),
            column_name: SmartString::new(&column_name),
            result_column_name: result_column_name.clone(),
            cached_epoch: self.engine.schema_epoch(),
        };
        *compiled_guard = CompiledExecution::CountDistinct(compiled_cd);
        drop(compiled_guard);

        // Build result
        let mut result_values = CompactVec::with_capacity(1);
        result_values.push(Value::Integer(count as i64));
        let row = Row::from_compact_vec(result_values);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));

        Some(Ok(Box::new(ExecutorResult::new(
            vec![result_column_name],
            rows,
        ))))
    }

    /// COUNT(*) fast path for simple queries
    ///
    /// This is a compiled fast path for simple `SELECT COUNT(*) FROM table` queries.
    /// On first execution, it analyzes and compiles the query pattern.
    /// On subsequent executions, it skips parsing and directly fetches the row count.
    ///
    /// # Arguments
    /// * `stmt` - The SELECT statement
    /// * `compiled` - The compiled execution state (shared via RwLock)
    ///
    /// # Returns
    /// * `Some(Ok(result))` - Query succeeded via fast path
    /// * `Some(Err(e))` - Query failed
    /// * `None` - Query doesn't qualify for this fast path (use normal path)
    pub(crate) fn try_fast_count_star_compiled(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Caller (try_compiled_fast_paths) guarantees no explicit transaction
        // is active; this path reads committed state only.

        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable(epoch)
                    if self.engine.schema_epoch() == *epoch =>
                {
                    return None
                }
                CompiledExecution::CountStar(cs) => {
                    // Fast path: validate epoch and execute directly
                    if self.engine.schema_epoch() == cs.cached_epoch {
                        return Some(self.execute_compiled_count_star(cs));
                    }
                    // Schema changed - fall through to recompile
                }
                CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - fall through to recompile
                // Other variants - not a COUNT(*) query
                _ => return None,
            }
        }

        // First execution or schema changed - compile and cache (write lock)
        self.compile_and_execute_count_star(stmt, compiled)
    }

    /// Execute using pre-compiled COUNT(*) info
    fn execute_compiled_count_star(
        &self,
        cs: &crate::executor::query_cache::CompiledCountStar,
    ) -> Result<Box<dyn QueryResult>> {
        // Under read committed the table's own count is the latest committed
        // one: it answers only when no commit published across its reads.
        // Else the count is read through a view
        let quiet_count = || -> Result<Option<usize>> {
            let tx = self.engine.begin_transaction()?;
            let table = tx.get_statement_table_at(&cs.table_name, None)?;
            table.fast_row_count()
        };
        let read_committed =
            self.engine.get_isolation_level() == crate::core::IsolationLevel::ReadCommitted;
        let count = match read_committed.then(quiet_count).transpose()?.flatten() {
            Some(count) => count,
            None => {
                let view = self.autocommit_view();
                let tx = self.engine.begin_transaction()?;
                let table = tx.get_statement_table_at(&cs.table_name, view.as_ref())?;
                let count = table.row_count()?;
                view.as_ref()
                    .map_or(Ok(()), crate::storage::mvcc::registry::StatementView::held)?;
                count
            }
        };

        // Build result
        let mut result_values = CompactVec::with_capacity(1);
        result_values.push(Value::Integer(count as i64));
        let row = Row::from_compact_vec(result_values);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));

        Ok(Box::new(ExecutorResult::new(
            vec![cs.result_column_name.clone()],
            rows,
        )))
    }

    /// Compile and execute COUNT(*), caching the compiled state
    fn compile_and_execute_count_star(
        &self,
        stmt: &SelectStatement,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        use crate::common::SmartString;
        use crate::executor::query_cache::CompiledCountStar;

        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check (another thread may have compiled while we waited)
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(epoch) if self.engine.schema_epoch() == *epoch => {
                return None
            }
            CompiledExecution::CountStar(cs) => {
                if self.engine.schema_epoch() == cs.cached_epoch {
                    return Some(self.execute_compiled_count_star(cs));
                }
                // Schema changed, continue to recompile
            }
            CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - recompile
            _ => return None,
        }

        // Pattern detection: SELECT COUNT(*) FROM table
        // Must have:
        // - Exactly one column expression
        // - That column is COUNT(*) or COUNT(1) function call (not DISTINCT)
        // - No WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, CTEs, set operations
        // - Single table source (no joins)

        if stmt.columns.len() != 1 {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Check for disqualifying clauses
        if stmt.where_clause.is_some()
            || !stmt.group_by.columns.is_empty()
            || stmt.having.is_some()
            || !stmt.order_by.is_empty()
            || stmt.limit.is_some()
            || stmt.offset.is_some()
            || stmt.with.is_some()
            || !stmt.set_operations.is_empty()
        {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Extract COUNT(*) pattern
        let result_column_name = match &stmt.columns[0] {
            Expression::FunctionCall(func) => {
                // Must be COUNT function
                if func.function.to_uppercase() != "COUNT" {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
                // Must NOT be DISTINCT - if DISTINCT, don't mark as NotOptimizable
                // because COUNT DISTINCT fast path might handle it
                if func.is_distinct {
                    return None;
                }
                // Must not have FILTER clause
                if func.filter.is_some() {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
                // Must be COUNT(*) or COUNT(1)
                match func.arguments.len() {
                    0 => {
                        // COUNT(*) without explicit star is rare but handle it
                        "COUNT(*)".to_string()
                    }
                    1 => {
                        match &func.arguments[0] {
                            Expression::Star(_) => "COUNT(*)".to_string(),
                            Expression::IntegerLiteral(lit) => {
                                // COUNT(1) is equivalent to COUNT(*)
                                if lit.value == 1 {
                                    "COUNT(1)".to_string()
                                } else {
                                    *compiled_guard = CompiledExecution::NotOptimizable(
                                        self.engine.schema_epoch(),
                                    );
                                    return None;
                                }
                            }
                            _ => {
                                // COUNT(col) without DISTINCT - not our fast path
                                *compiled_guard =
                                    CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                                return None;
                            }
                        }
                    }
                    _ => {
                        *compiled_guard =
                            CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                        return None;
                    }
                }
            }
            Expression::Aliased(aliased) => {
                // Handle COUNT(*) AS alias
                match aliased.expression.as_ref() {
                    Expression::FunctionCall(func) => {
                        // Must be COUNT function
                        if func.function.to_uppercase() != "COUNT" {
                            *compiled_guard =
                                CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                            return None;
                        }
                        // Must NOT be DISTINCT - if DISTINCT, don't mark as NotOptimizable
                        // because COUNT DISTINCT fast path might handle it
                        if func.is_distinct {
                            return None;
                        }
                        // Must not have FILTER clause
                        if func.filter.is_some() {
                            *compiled_guard =
                                CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                            return None;
                        }
                        match func.arguments.len() {
                            0 => aliased.alias.value.to_string(),
                            1 => match &func.arguments[0] {
                                Expression::Star(_) => aliased.alias.value.to_string(),
                                Expression::IntegerLiteral(lit) if lit.value == 1 => {
                                    aliased.alias.value.to_string()
                                }
                                Expression::IntegerLiteral(_) => {
                                    *compiled_guard = CompiledExecution::NotOptimizable(
                                        self.engine.schema_epoch(),
                                    );
                                    return None;
                                }
                                _ => {
                                    *compiled_guard = CompiledExecution::NotOptimizable(
                                        self.engine.schema_epoch(),
                                    );
                                    return None;
                                }
                            },
                            _ => {
                                *compiled_guard =
                                    CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                                return None;
                            }
                        }
                    }
                    _ => {
                        *compiled_guard =
                            CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                        return None;
                    }
                }
            }
            _ => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        // Extract table name (bail out for AS OF temporal queries)
        let table_name = match stmt.table_expr.as_deref() {
            Some(Expression::TableSource(ts)) => {
                if ts.as_of.is_some() {
                    *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                    return None;
                }
                ts.name.value_lower.clone()
            }
            _ => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        // Try to get table and get count
        let tx = match self.engine.begin_transaction() {
            Ok(tx) => tx,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        let view = self.autocommit_view();
        let table = match tx.get_statement_table_at(&table_name, view.as_ref()) {
            Ok(t) => t,
            Err(_) => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        // Get the count; a truncate or drop that overtook the view leaves it
        // to the run again
        let count = match table.row_count().and_then(|count| {
            view.as_ref()
                .map_or(Ok(()), crate::storage::mvcc::registry::StatementView::held)
                .map(|()| count)
        }) {
            Ok(count) => count,
            Err(error) => return Some(Err(error)),
        };

        // Cache the compiled state
        let compiled_cs = CompiledCountStar {
            table_name: SmartString::new(&table_name),
            result_column_name: result_column_name.clone(),
            cached_epoch: self.engine.schema_epoch(),
        };
        *compiled_guard = CompiledExecution::CountStar(compiled_cs);
        drop(compiled_guard);

        // Build result
        let mut result_values = CompactVec::with_capacity(1);
        result_values.push(Value::Integer(count as i64));
        let row = Row::from_compact_vec(result_values);
        let mut rows = RowVec::with_capacity(1);
        rows.push((0, row));

        Some(Ok(Box::new(ExecutorResult::new(
            vec![result_column_name],
            rows,
        ))))
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

        let mut found = ahash::AHashMap::new();
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
