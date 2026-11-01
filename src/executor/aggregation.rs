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

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
// SmallVec removed - Vec is faster due to spilled() check overhead in hot loops

use crate::common::{CompactArc, CompactVec, StringMap};
use crate::core::{Result, Row, RowVec, Value};
use crate::functions::aggregate::{
    AggregateState, CallTemplate, ColumnsKey, CompiledAggregate, GroupKey, GroupedAggregates,
};
use crate::parser::ast::*;
use crate::storage::traits::QueryResult;

use super::context::ExecutionContext;
use super::expr_key::ExprKey;
use super::expression::{ExprVM, SubqueryExecutor};
use super::query::order_by_inputs;
use super::query_classification::QueryClassification;
use super::result::ExecutorResult;
use super::utils::{
    bare_name, build_column_index_map, eq_folded, item_name, item_spelled_name, qualified,
    qualified_index, qualified_key_index, shown_name, split_qualified,
};
use super::window::WindowOutput;
use super::Executor;

// Re-export for backward compatibility
pub use super::utils::{expression_contains_aggregate, is_aggregate_function};

/// Every path that builds an aggregate result. Each ends in
/// `finish_aggregation`, which records the path for the equivalence tests
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AggregatePath {
    General,
    /// One row from what the table keeps: counts, distinct counts, sums, bounds
    Stats,
    /// One row the table computes over the rows the WHERE passes
    StorageFiltered,
    /// One row fed by every row of the table
    GlobalScan,
    StorageGrouped,
    StreamingGroupBy,
    StreamingDerivedTable,
    /// MIN or MAX read from an index's end
    IndexBound,
    /// A prepared subquery block that aggregates, per call or by key
    PreparedBlock,
    /// A prepared block's COUNT(*) of one indexed key, counted from the index
    IndexCount,
}

impl AggregatePath {
    /// Notes, under test-failpoints, that this path answered; the shortcuts
    /// that do not end in `finish_aggregation` call it themselves
    #[inline]
    pub(crate) fn taken(self) {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::aggregate_path_taken(self as usize);
    }
}

#[cfg(any(test, feature = "test-failpoints"))]
impl AggregatePath {
    pub(crate) const ALL: [AggregatePath; 10] = [
        AggregatePath::General,
        AggregatePath::Stats,
        AggregatePath::StorageFiltered,
        AggregatePath::GlobalScan,
        AggregatePath::StorageGrouped,
        AggregatePath::StreamingGroupBy,
        AggregatePath::StreamingDerivedTable,
        AggregatePath::IndexBound,
        AggregatePath::PreparedBlock,
        AggregatePath::IndexCount,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            AggregatePath::General => "general",
            AggregatePath::Stats => "stats",
            AggregatePath::StorageFiltered => "storage_filtered",
            AggregatePath::GlobalScan => "global_scan",
            AggregatePath::StorageGrouped => "storage_grouped",
            AggregatePath::StreamingGroupBy => "streaming_group_by",
            AggregatePath::StreamingDerivedTable => "streaming_derived_table",
            AggregatePath::IndexBound => "index_bound",
            AggregatePath::PreparedBlock => "prepared_block",
            AggregatePath::IndexCount => "index_count",
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
    /// The columns of the rows aggregated, which the statement may read
    /// only through its groups
    pub source_columns: &'a [String],
    /// The statement's blocks, which answer its SELECT list's subqueries
    pub blocks: Option<&'a dyn SubqueryExecutor>,
    /// The output names a plan found for a SELECT list of nothing but the
    /// one row's calls, see [`Executor::calls_alone`]
    pub alone: Option<&'a CompactArc<Vec<String>>>,
}

/// A program over aggregate slot rows: `aliases` read a group item or
/// aggregate from its slot by identity, `outer_columns` the outer row by name;
/// its subqueries run as the statement's blocks when `blocks`, and `params`
/// fold when it serves only the execution they belong to
pub(crate) fn compile_over_slots(
    expr: &Expression,
    columns: &[String],
    registry: &crate::functions::FunctionRegistry,
    aliases: Option<&FxHashMap<ExprKey, usize>>,
    outer_columns: Option<&[String]>,
    blocks: bool,
    params: Option<&[Value]>,
) -> Result<super::expression::SharedProgram> {
    let mut compile = super::expression::CompileContext::new(columns, registry);
    if blocks {
        compile = compile.with_prepared_subqueries();
    }
    if let Some(params) = params {
        compile = compile.with_params(params);
    }
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
pub(crate) fn slot_context<'a>(
    row: &'a Row,
    ctx: &'a ExecutionContext,
) -> super::expression::ExecuteContext<'a> {
    let exec = super::expression::ExecuteContext::with_common_params(
        row,
        ctx.params(),
        ctx.transaction_id(),
    );
    match ctx.outer_row() {
        Some(outer) => exec.with_outer_row(outer),
        None => exec,
    }
}

/// The parameters a slot row's programs read: the `blocks`' when the
/// statement has them, as `with_blocks` gives them, else the statement's
pub(crate) fn slot_params<'a>(
    ctx: &'a ExecutionContext,
    blocks: Option<&'a dyn SubqueryExecutor>,
) -> &'a [Value] {
    blocks
        .and_then(|blocks| blocks.params())
        .unwrap_or(ctx.params())
}

/// `exec` with its subqueries answered by the statement's `blocks`, which
/// hold the parameters past the statement's own
pub(crate) fn with_blocks<'a>(
    exec: super::expression::ExecuteContext<'a>,
    blocks: Option<&'a dyn SubqueryExecutor>,
) -> super::expression::ExecuteContext<'a> {
    let Some(blocks) = blocks else {
        return exec;
    };
    let exec = exec.with_subquery_executor(blocks);
    match blocks.params() {
        Some(params) => exec.with_params(params),
        None => exec,
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

/// The expressions `stmt` groups by, in the order its groups hold them:
/// GROUPING SETS's each once, as it first appears
pub(crate) fn grouping_items(stmt: &SelectStatement) -> Vec<&Expression> {
    match &stmt.group_by.modifier {
        GroupByModifier::GroupingSets(sets) => {
            let mut seen = FxHashSet::default();
            sets.iter()
                .flatten()
                .filter(|expr| seen.insert(ExprKey::of(expr)))
                .collect()
        }
        _ => stmt.group_by.columns.iter().collect(),
    }
}

/// The SELECT item of `stmt` the GROUP BY key `key` names: the last item
/// holding no aggregate whose alias it is written as, or its position
pub(crate) fn grouped_item(stmt: &SelectStatement, key: &Expression) -> Option<usize> {
    match key {
        Expression::Identifier(id) => stmt.columns.iter().rposition(|col| {
            matches!(col, Expression::Aliased(aliased)
                if aliased.alias.value_lower == id.value_lower
                    && !expression_contains_aggregate(&aliased.expression))
        }),
        Expression::IntegerLiteral(lit) => usize::try_from(lit.value)
            .ok()
            .filter(|at| (1..=stmt.columns.len()).contains(at))
            .map(|at| at - 1),
        _ => None,
    }
}

/// The identity of a GROUP BY item
fn group_by_item_key(item: &GroupByItem) -> ExprKey {
    match item {
        GroupByItem::Column(name) => ExprKey::column(name),
        GroupByItem::Expression { expr, .. } => ExprKey::of(expr),
    }
}

/// The error of a column a query that aggregates reads outside its groups
pub(crate) fn ungrouped_column(name: &str) -> crate::core::Error {
    crate::core::Error::internal(format!(
        "Compile error: {}",
        super::expression::CompileError::ColumnNotFound(name.to_string())
    ))
}

/// The error of a GROUP BY key that is or holds an aggregate call, however
/// the key is written: as the call, by position or by its item's alias
pub(crate) fn aggregate_group_key() -> crate::core::Error {
    crate::core::Error::invalid_argument("an aggregate function cannot be a GROUP BY key")
}

/// Whether the GROUP BY key `written` is a bare name that, `binds` false,
/// binds to no column, enclosing column or other SELECT alias, and is the
/// alias of an item of `select` holding an aggregate: the one rule every
/// path refuses such a key by
pub(crate) fn aggregate_alias_key(
    written: Option<&Expression>,
    select: &[Expression],
    binds: bool,
) -> bool {
    let Some(Expression::Identifier(name)) = written else {
        return false;
    };
    !binds
        && select.iter().any(|item| {
            matches!(item, Expression::Aliased(aliased)
                if aliased.alias.value.eq_ignore_ascii_case(&name.value)
                    && expression_contains_aggregate(&aliased.expression))
        })
}

/// The quoted name `stmt` writes as its GROUP BY key at `at`, itself or as
/// the SELECT item a position names
fn written_quoted(stmt: &SelectStatement, at: usize) -> Option<&Expression> {
    let key = match grouping_items(stmt).get(at) {
        Some(Expression::IntegerLiteral(position)) => usize::try_from(position.value)
            .ok()
            .and_then(|position| stmt.columns.get(position.checked_sub(1)?))
            .map(|item| match item {
                Expression::Aliased(aliased) => &*aliased.expression,
                item => item,
            }),
        key => key.copied(),
    };
    key.filter(|key| matches!(key, Expression::Identifier(id) if id.quoted))
}

/// Whether `name`, a GROUP BY column the input does not hold, is a column of
/// the outer row `ctx` runs for
fn names_outer_column(name: &str, ctx: &ExecutionContext) -> bool {
    let Some(outer) = ctx.outer_row() else {
        return false;
    };
    let lower = name.to_lowercase();
    outer.contains_key(lower.as_str())
        || split_qualified(&lower).is_some_and(|(_, bare)| outer.contains_key(bare))
}

/// The error the grouped `stmt` raises when it runs over `source` and reads
/// a column outside its groups, raised without running it
pub(crate) fn check_grouped_reads(
    executor: &Executor,
    stmt: &SelectStatement,
    source: &[String],
    ctx: &ExecutionContext,
) -> Result<()> {
    let items = executor.parse_group_by(stmt, source)?;
    match ungrouped_read(stmt, &items, source, ctx) {
        Some(name) => Err(ungrouped_column(&name)),
        None => Ok(()),
    }
}

/// A column of `source` that `stmt` reads outside every GROUP BY item and
/// aggregate call, in its SELECT list, HAVING or ORDER BY. Each name binds
/// as the compiler binds it, the outer row of `ctx` included; a HAVING name
/// or a bare ORDER BY name a SELECT alias carries is the alias's
fn ungrouped_read(
    stmt: &SelectStatement,
    items: &[GroupByItem],
    source: &[String],
    ctx: &ExecutionContext,
) -> Option<String> {
    use std::cell::OnceCell;
    let grouped_exprs: smallvec::SmallVec<[ExprKey; 4]> = items
        .iter()
        .filter_map(|item| match item {
            GroupByItem::Expression { expr, .. } => Some(ExprKey::of(expr)),
            GroupByItem::Column(_) => None,
        })
        .collect();
    let skip = |expr: &Expression| match expr {
        Expression::FunctionCall(call) if is_aggregate_function(&call.function) => true,
        _ => !grouped_exprs.is_empty() && grouped_exprs.contains(&ExprKey::of(expr)),
    };
    // A reference spelled as a group column binds as that column does
    let spelled_grouped = |reference: &Expression| {
        items.iter().any(|item| match (item, reference) {
            (GroupByItem::Column(c), Expression::Identifier(id)) => {
                c.eq_ignore_ascii_case(&id.value)
            }
            (GroupByItem::Column(c), Expression::QualifiedIdentifier(q)) => split_qualified(c)
                .is_some_and(|(table, name)| {
                    table.eq_ignore_ascii_case(&q.qualifier.value)
                        && name.eq_ignore_ascii_case(&q.name.value)
                }),
            _ => false,
        })
    };
    let outer: OnceCell<Option<Vec<String>>> = OnceCell::new();
    let compile: OnceCell<super::expression::CompileContext<'_>> = OnceCell::new();
    let column_of = |reference: &Expression| {
        let compile = compile.get_or_init(|| {
            let outer = outer.get_or_init(|| {
                ctx.outer_row()
                    .map(|row| row.keys().map(|k| k.to_string()).collect())
            });
            let compile = super::expression::CompileContext::new(
                source,
                crate::functions::registry::global_registry(),
            );
            match outer {
                Some(outer) => compile.with_outer_columns(outer),
                None => compile,
            }
        });
        compile.reference(reference).and_then(|r| r.row_column())
    };
    let grouped_columns: OnceCell<smallvec::SmallVec<[Option<u16>; 4]>> = OnceCell::new();
    let is_alias = |reference: &Expression| match reference {
        Expression::Identifier(id) => stmt
            .columns
            .iter()
            .any(|c| matches!(c, Expression::Aliased(a) if a.alias.value_lower == id.value_lower)),
        _ => false,
    };
    let mut found = None;
    let mut check = |reference: &Expression, aliases: bool| {
        if found.is_some() || spelled_grouped(reference) || (aliases && is_alias(reference)) {
            return;
        }
        let Some(column) = column_of(reference) else {
            return;
        };
        let grouped = grouped_columns.get_or_init(|| {
            items
                .iter()
                .filter_map(|item| match item {
                    GroupByItem::Column(name) => Some(column_of(&Expression::Identifier(
                        crate::parser::ast::Identifier::new(
                            crate::parser::token::Token::new(
                                crate::parser::token::TokenType::Identifier,
                                name,
                                crate::parser::token::Position::default(),
                            ),
                            name.as_str(),
                        ),
                    ))),
                    GroupByItem::Expression { .. } => None,
                })
                .collect()
        });
        if !grouped.contains(&Some(column)) {
            found = Some(match reference {
                Expression::Identifier(id) => id.value.to_string(),
                other => other.to_string(),
            });
        }
    };
    for column in &stmt.columns {
        super::utils::visit_column_refs_outside(column, &skip, &mut |r| check(r, false));
    }
    if let Some(having) = stmt.having.as_deref() {
        super::utils::visit_column_refs_outside(having, &skip, &mut |r| check(r, true));
    }
    for ob in &stmt.order_by {
        if super::utils::order_by_names_an_alias(&ob.expression, stmt) {
            continue;
        }
        super::utils::visit_column_refs_outside(&ob.expression, &skip, &mut |r| check(r, false));
    }
    found
}

/// The slot of each group item and aggregate in an aggregated row, which
/// holds the group items first and the aggregates after them; a SELECT
/// alias names its expression's slot, unless a group column, qualified or
/// not, has that name
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
    // The alias comes lowercased; the column keeps its written spelling
    let names_a_group_column = |alias_lower: &str| {
        group_items.iter().any(|item| match item {
            GroupByItem::Column(name) => eq_folded(bare_name(name), alias_lower),
            GroupByItem::Expression { .. } => false,
        })
    };
    for col in select {
        if let Expression::Aliased(aliased) = col {
            if names_a_group_column(&aliased.alias.value_lower) {
                continue;
            }
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
    /// The error an extra argument gave when evaluated, the result of
    /// every group that reads this call
    pub(crate) extra_error: Option<crate::core::Error>,
    /// The extra arguments as written when one is not a literal, which a
    /// subquery block evaluates for each call; None otherwise
    pub(crate) extra_exprs: Option<Box<[Expression]>>,
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
    /// The modifiers this call carries
    pub(crate) fn modifiers(&self) -> Modifiers {
        let mut modifiers = Modifiers::NONE;
        if self.distinct {
            modifiers = modifiers.and(Modifiers::DISTINCT);
        }
        if self.filter.is_some() {
            modifiers = modifiers.and(Modifiers::FILTER);
        }
        if !self.order_by.is_empty() {
            modifiers = modifiers.and(Modifiers::ORDER_BY);
        }
        if self.expression.is_some() {
            modifiers = modifiers.and(Modifiers::EXPRESSION);
        }
        if !self.extra_args.is_empty() {
            modifiers = modifiers.and(Modifiers::EXTRA_ARGS);
        }
        modifiers
    }

    /// Whether a path implementing `supported` may answer this call
    pub(crate) fn answered_with(&self, supported: Modifiers) -> bool {
        self.modifiers().within(supported)
    }

    /// Whether this is a COUNT of every row: COUNT(*), or a COUNT of a
    /// constant that is never NULL
    pub(crate) fn counts_rows(&self) -> bool {
        self.name == "COUNT"
            && self.answered_with(Modifiers::EXPRESSION)
            && match &self.expression {
                None => self.column == "*",
                Some(Expression::IntegerLiteral(_) | Expression::FloatLiteral(_)) => true,
                Some(Expression::StringLiteral(_) | Expression::BooleanLiteral(_)) => true,
                Some(_) => false,
            }
    }

    /// The running state of this call, see [`Executor::aggregate_state`]
    pub(crate) fn state(
        &self,
        registry: &crate::functions::FunctionRegistry,
    ) -> Option<AggregateState> {
        Executor::aggregate_state(
            registry,
            &self.name,
            self.distinct,
            &self.extra_args,
            &self.order_by,
        )
    }

    /// How every group runs this call
    pub(crate) fn template(
        &self,
        registry: &crate::functions::FunctionRegistry,
    ) -> Result<CallTemplate> {
        self.template_reading(registry, &self.extra_args, self.extra_error.clone())
    }

    /// How every group runs this call with the extra arguments `extra_args`
    /// a block read for one call, and `failed` the error one of them gave
    pub(crate) fn template_reading(
        &self,
        registry: &crate::functions::FunctionRegistry,
        extra_args: &[Value],
        failed: Option<crate::core::Error>,
    ) -> Result<CallTemplate> {
        let function = registry.get_aggregate(&self.name).ok_or_else(|| {
            crate::core::Error::internal(format!("unknown aggregate function {}", self.name))
        })?;
        Ok(CallTemplate::new(
            function,
            self.name == "COUNT" && self.column == "*",
            self.distinct,
            extra_args,
            Executor::aggregate_order_keys(&self.order_by),
            failed,
        ))
    }

    /// Get the result column name
    pub fn get_column_name(&self) -> String {
        match self.alias {
            Some(ref alias) => alias.clone(),
            None => aggregate_call_name(&self.name, self.distinct, &self.column, &self.extra_args),
        }
    }

    /// Get the expression name (without alias) for HAVING clause matching
    /// This returns `SUM(price)` even if there's an alias like `AS total`
    pub fn get_expression_name(&self) -> String {
        aggregate_call_name(&self.name, self.distinct, &self.column, &[])
    }
}

/// The name an unaliased aggregate call's result column takes, as written:
/// `COUNT(*)`, `SUM(a)`, `COUNT(DISTINCT g)`, `STRING_AGG(b, ' | ')`
pub(crate) fn aggregate_call_name(
    name: &str,
    distinct: bool,
    column: &str,
    extra_args: &[Value],
) -> String {
    if column == "*" {
        return format!("{}(*)", name);
    }
    let mut args = String::new();
    if distinct {
        args.push_str("DISTINCT ");
    }
    args.push_str(column);
    for arg in extra_args {
        args.push_str(", ");
        match arg {
            Value::Text(s) => {
                args.push('\'');
                args.push_str(s);
                args.push('\'');
            }
            other => args.push_str(&other.to_string()),
        }
    }
    format!("{}({})", name, args)
}

/// The modifiers an aggregate call carries. A path that answers aggregates
/// states the ones it implements, and takes a call only when the call's
/// modifiers are among them: the one admission rule for every such path
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Modifiers(u8);

impl Modifiers {
    pub(crate) const NONE: Modifiers = Modifiers(0);
    pub(crate) const DISTINCT: Modifiers = Modifiers(1);
    pub(crate) const FILTER: Modifiers = Modifiers(1 << 1);
    pub(crate) const ORDER_BY: Modifiers = Modifiers(1 << 2);
    /// The argument is an expression, not a column or `*`
    pub(crate) const EXPRESSION: Modifiers = Modifiers(1 << 3);
    /// Arguments after the first, such as a separator
    pub(crate) const EXTRA_ARGS: Modifiers = Modifiers(1 << 4);

    pub(crate) const fn and(self, other: Modifiers) -> Modifiers {
        Modifiers(self.0 | other.0)
    }

    /// Whether every modifier here is among `supported`
    pub(crate) const fn within(self, supported: Modifiers) -> bool {
        self.0 & !supported.0 == 0
    }

    /// The modifiers of a call as the AST writes it, for a path that reads
    /// the statement before its aggregates are parsed
    pub(crate) fn of_call(call: &crate::parser::ast::FunctionCall) -> Modifiers {
        let mut modifiers = Modifiers::NONE;
        if call.is_distinct {
            modifiers = modifiers.and(Modifiers::DISTINCT);
        }
        if call.filter.is_some() {
            modifiers = modifiers.and(Modifiers::FILTER);
        }
        if !call.order_by.is_empty() {
            modifiers = modifiers.and(Modifiers::ORDER_BY);
        }
        if call.arguments.first().is_some_and(|arg| {
            !matches!(
                arg,
                Expression::Identifier(_)
                    | Expression::QualifiedIdentifier(_)
                    | Expression::Star(_)
            )
        }) {
            modifiers = modifiers.and(Modifiers::EXPRESSION);
        }
        if call.arguments.len() > 1 {
            modifiers = modifiers.and(Modifiers::EXTRA_ARGS);
        }
        modifiers
    }
}

/// An aggregate call's extra argument values and the error one of them gave
pub(crate) type ExtraArgs = (Vec<Value>, Option<crate::core::Error>);

/// What a parse of aggregate calls does with their extra arguments
#[derive(Clone, Copy)]
enum ExtraArgsRead<'a> {
    /// Reads them, a parameter bound by the context
    With(Option<&'a ExecutionContext>),
    /// Reads the literal ones and keeps them as written, for a subquery
    /// block that reads the others for each call
    Block,
    /// Leaves them unread, the calls wanted for their identities alone
    Unread,
}

impl ExtraArgsRead<'_> {
    /// The extra arguments of a call to `args` as a block keeps them
    fn written(self, args: &[Expression]) -> Option<Box<[Expression]>> {
        match self {
            ExtraArgsRead::Block => written_extra_args(args),
            ExtraArgsRead::With(_) | ExtraArgsRead::Unread => None,
        }
    }
}

/// The error an extra argument of one of `aggregations` gave, which a path
/// that reads groups itself reports at its first group
pub(crate) fn extra_args_failure(
    aggregations: &[SqlAggregateFunction],
) -> Option<&crate::core::Error> {
    aggregations.iter().find_map(|agg| agg.extra_error.as_ref())
}

/// The extra arguments of an aggregate call (a STRING_AGG separator), each
/// one value for the statement, read once with the parameters of `ctx`. An
/// argument that fails to evaluate is NULL here and its error comes second,
/// for the first group or frame that reads the call to report
pub(crate) fn aggregate_extra_args(
    args: &[Expression],
    ctx: Option<&ExecutionContext>,
) -> Result<ExtraArgs> {
    let mut values = Vec::with_capacity(args.len().saturating_sub(1));
    let mut failed = None;
    for arg in args.iter().skip(1) {
        let value = match literal_extra_arg(arg) {
            Some(value) => value,
            None => {
                extra_arg_constant(arg)?;
                let program = super::expression::compile_once(arg)?;
                super::expression::execute_once(&program, ctx).unwrap_or_else(|error| {
                    failed.get_or_insert(error);
                    Value::null_unknown()
                })
            }
        };
        values.push(value);
    }
    Ok((values, failed))
}

/// Whether an aggregate's extra argument is written as its value: a
/// literal, or a quoted name, which reads as text
fn is_literal_extra_arg(arg: &Expression) -> bool {
    match arg {
        Expression::Identifier(id) => id.quoted,
        Expression::IntegerLiteral(_)
        | Expression::FloatLiteral(_)
        | Expression::StringLiteral(_)
        | Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_) => true,
        _ => false,
    }
}

/// The value of an extra argument written as its value
fn literal_extra_arg(arg: &Expression) -> Option<Value> {
    match arg {
        _ if !is_literal_extra_arg(arg) => None,
        Expression::Identifier(id) => Some(Value::text(id.value.as_str())),
        arg => super::utils::extract_literal_value(arg),
    }
}

/// Whether every argument of a call to `args` after the first is a literal,
/// which `aggregate_extra_args` reads without a run's parameters
pub(crate) fn extra_args_literal(args: &[Expression]) -> bool {
    args.get(1..)
        .is_none_or(|extra| extra.iter().all(is_literal_extra_arg))
}

/// The extra arguments of a call to `args` as written, when one of them is
/// not a literal
fn written_extra_args(args: &[Expression]) -> Option<Box<[Expression]>> {
    let extra = args.get(1..)?;
    (!extra.iter().all(is_literal_extra_arg)).then(|| extra.into())
}

/// An error unless `arg`, an aggregate argument after the first, is one
/// value for the call: it reads no row the aggregate reads
fn extra_arg_constant(arg: &Expression) -> Result<()> {
    if super::volatility::statement_constant(arg, crate::functions::registry::global_registry()) {
        return Ok(());
    }
    Err(crate::core::Error::invalid_argument(format!(
        "an aggregate argument after the first must not read the row: {}",
        arg
    )))
}

/// What one aggregate call reads from each input row, compiled once: the
/// FILTER that decides whether the row counts, the argument expression, and
/// the keys of the call's own ORDER BY
pub(crate) struct CallInputs {
    filter: Option<super::expression::SharedProgram>,
    argument: Option<super::expression::SharedProgram>,
    order: Vec<super::expression::SharedProgram>,
}

impl CallInputs {
    /// The inputs of every call, each program made by `compile`
    pub(crate) fn compile_with(
        aggregations: &[SqlAggregateFunction],
        compile: impl Fn(&Expression) -> Result<super::expression::SharedProgram>,
    ) -> Result<Vec<CallInputs>> {
        aggregations
            .iter()
            .map(|agg| {
                Ok(CallInputs {
                    filter: agg.filter.as_ref().map(&compile).transpose()?,
                    argument: agg.expression.as_ref().map(&compile).transpose()?,
                    order: agg
                        .order_by
                        .iter()
                        .map(|o| compile(&o.expression))
                        .collect::<Result<_>>()?,
                })
            })
            .collect()
    }

    /// Feeds one input row to the call's run. `column` is the call's
    /// column value, read when it has no argument expression; a VM is made
    /// on the first program that runs
    #[inline(always)]
    pub(crate) fn feed(
        &self,
        state: &mut CompiledAggregate,
        column: Option<&Value>,
        vm: &mut Option<ExprVM>,
        row: &super::expression::ExecuteContext<'_>,
        agg: &SqlAggregateFunction,
    ) -> Result<()> {
        if self.filter.is_none() && self.argument.is_none() && self.order.is_empty() {
            if let Some(value) = column {
                state.accumulate(value);
            }
            return Ok(());
        }
        self.feed_programs(state, column, vm, row, agg)
    }

    /// `feed` for a call with a FILTER, an argument expression or ORDER BY
    fn feed_programs(
        &self,
        state: &mut CompiledAggregate,
        column: Option<&Value>,
        vm: &mut Option<ExprVM>,
        row: &super::expression::ExecuteContext<'_>,
        agg: &SqlAggregateFunction,
    ) -> Result<()> {
        if let Some(filter) = &self.filter {
            let vm = vm.get_or_insert_with(ExprVM::new);
            if !matches!(vm.execute(filter, row)?, Value::Boolean(true)) {
                return Ok(());
            }
        }
        let argument;
        let value = match &self.argument {
            Some(program) => {
                let vm = vm.get_or_insert_with(ExprVM::new);
                argument = vm.execute(program, row).map_err(|e| {
                    crate::core::Error::expression_evaluation(format!(
                        "{}({}): {}",
                        agg.name, agg.column, e
                    ))
                })?;
                &argument
            }
            None => match column {
                Some(value) => value,
                None => return Ok(()),
            },
        };
        if !self.order.is_empty() && state.supports_order_by() {
            let vm = vm.get_or_insert_with(ExprVM::new);
            let mut keys = Vec::with_capacity(self.order.len());
            for program in &self.order {
                keys.push(vm.execute(program, row)?);
            }
            state.accumulate_with_sort_key(value, keys);
        } else {
            state.accumulate(value);
        }
        Ok(())
    }
}

/// The value COUNT(*) is fed for each row
pub(crate) static COUNT_STAR: Value = Value::Integer(1);

/// Rows at which an aggregation whose calls all merge splits its input
const PARALLEL_ROWS: usize = 100_000;

/// One GROUP BY item as each row is read for it
enum KeyItem {
    Column(usize),
    Program(super::expression::SharedProgram),
    Null,
}

/// A group key's items within the key of every item, for a grouping set
struct Picked<'a> {
    key: &'a [Value],
    picks: &'a [usize],
}

impl GroupKey for Picked<'_> {
    #[inline(always)]
    fn width(&self) -> usize {
        self.picks.len()
    }
    #[inline(always)]
    fn at(&self, i: usize) -> &Value {
        &self.key[self.picks[i]]
    }
}

/// What hands an aggregation its rows one at a time, each borrowed until
/// the next is asked for
pub(super) trait RowSource {
    /// The next row; None once every row was handed over
    fn next_row(&mut self) -> Result<Option<&Row>>;

    /// Every row left, taken at once; their ids are not read
    fn held(self) -> Result<RowVec>;
}

/// The rows an aggregation reads: held at once, which a large input feeds
/// on the thread pool, or handed over one at a time, with about how many
/// there are; the feed takes those at once when it splits them
pub(super) enum FeedInput<'r, S> {
    Rows(&'r [(i64, Row)]),
    Each(S, usize),
}

/// A table's scanner as a source of rows, its error raised at the end
pub(super) struct ScannedRows<'s>(pub(super) &'s mut dyn crate::storage::traits::Scanner);

impl RowSource for ScannedRows<'_> {
    fn next_row(&mut self) -> Result<Option<&Row>> {
        if self.0.next() {
            return Ok(Some(self.0.row()));
        }
        match self.0.err() {
            Some(err) => Err(err.clone()),
            None => Ok(None),
        }
    }

    fn held(self) -> Result<RowVec> {
        let mut rows = RowVec::with_capacity(self.0.estimated_count().unwrap_or(64));
        while self.0.next() {
            rows.push((0, self.0.take_row()));
        }
        match self.0.err() {
            Some(err) => Err(err.clone()),
            None => Ok(rows),
        }
    }
}

/// The VM and key buffer one pass over rows reuses
#[derive(Default)]
struct FeedState {
    vm: Option<ExprVM>,
    key: Vec<Value>,
}

/// What each row a pass feeds reads beside itself, taken once for the pass:
/// the parameters, and the blocks answering its subqueries
#[derive(Clone, Copy)]
struct RowReads<'b> {
    params: &'b [Value],
    blocks: Option<&'b dyn SubqueryExecutor>,
}

impl<'b> RowReads<'b> {
    fn of(ctx: &'b ExecutionContext, blocks: Option<&'b dyn SubqueryExecutor>) -> Self {
        Self {
            params: slot_params(ctx, blocks),
            blocks,
        }
    }

    #[inline(always)]
    fn exec<'a>(self, row: &'a Row) -> super::expression::ExecuteContext<'a>
    where
        'b: 'a,
    {
        let exec = super::expression::ExecuteContext::new(row).with_params(self.params);
        match self.blocks {
            Some(blocks) => exec.with_subquery_executor(blocks),
            None => exec,
        }
    }
}

/// What an aggregation reads from each input row, compiled once over the
/// input's columns: the group key, and each call's column or inputs
struct RowFeed<'a> {
    aggregations: &'a [SqlAggregateFunction],
    inputs: Vec<CallInputs>,
    /// Each call's column; None for COUNT(*) and an argument expression
    columns: Vec<Option<usize>>,
    items: Vec<KeyItem>,
    /// The key's columns, when every item is a column
    key_columns: Option<Vec<usize>>,
    /// Whether a row runs a program, which the parallel split does not
    programs: bool,
}

impl<'a> RowFeed<'a> {
    /// The feed of `aggregations` grouped by `group_items`, the keys `stmt`
    /// writes, over `columns`, for this execution of `ctx`'s statement: its
    /// subqueries run as the statement's `blocks`, each set they read tested
    /// in place, and what reads a parameter folds it
    fn new(
        aggregations: &'a [SqlAggregateFunction],
        group_items: &[GroupByItem],
        stmt: Option<&SelectStatement>,
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Self> {
        let compile = |expr: &Expression| {
            if blocks.is_none() && !super::utils::expression_has_parameters(expr) {
                return super::expression::compile_expression(expr, columns);
            }
            let program = compile_over_slots(
                expr,
                columns,
                crate::functions::registry::global_registry(),
                None,
                None,
                blocks.is_some(),
                Some(slot_params(ctx, blocks)),
            )?;
            Ok(blocks
                .and_then(|blocks| blocks.bind_members(&program))
                .unwrap_or(program))
        };
        let items = group_items
            .iter()
            .enumerate()
            .map(|(at, item)| {
                Ok(match item {
                    GroupByItem::Column(name) => {
                        match Executor::lookup_column_index(&name.to_lowercase(), col_index_map) {
                            Some(idx) => KeyItem::Column(idx),
                            // The outer row's column is one value for every row
                            None if names_outer_column(name, ctx) => KeyItem::Null,
                            // No column or outer column binds it, nor another alias
                            None if stmt.is_some_and(|s| {
                                aggregate_alias_key(
                                    grouping_items(s).get(at).copied(),
                                    &s.columns,
                                    false,
                                )
                            }) =>
                            {
                                return Err(aggregate_group_key())
                            }
                            None => match stmt.and_then(|s| written_quoted(s, at)) {
                                // A quoted name no column carries is its text,
                                // as the compiler reads it
                                Some(quoted) => KeyItem::Program(compile(quoted)?),
                                None => {
                                    return Err(ungrouped_column(&match split_qualified(name) {
                                        Some((table, column)) => format!("{table}.{column}"),
                                        None => name.clone(),
                                    }))
                                }
                            },
                        }
                    }
                    GroupByItem::Expression { expr, .. } => KeyItem::Program(compile(expr)?),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let key_columns = items
            .iter()
            .map(|item| match item {
                KeyItem::Column(idx) => Some(*idx),
                _ => None,
            })
            .collect::<Option<Vec<_>>>();
        let programs = key_columns.is_none()
            || !aggregations
                .iter()
                .all(|a| a.answered_with(Modifiers::DISTINCT.and(Modifiers::EXTRA_ARGS)));
        Ok(Self {
            aggregations,
            inputs: CallInputs::compile_with(aggregations, compile)?,
            columns: aggregations
                .iter()
                .map(|agg| {
                    if agg.column == "*" || agg.expression.is_some() {
                        return Ok(None);
                    }
                    match Executor::lookup_column_index(&agg.column_lower, col_index_map) {
                        Some(idx) => Ok(Some(idx)),
                        // A qualified name its source lacks, unless the outer row's
                        None if split_qualified(&agg.column).is_some()
                            && !names_outer_column(&agg.column, ctx) =>
                        {
                            Err(ungrouped_column(&shown_name(&agg.column)))
                        }
                        None => Ok(None),
                    }
                })
                .collect::<Result<_>>()?,
            items,
            key_columns,
            programs,
        })
    }

    /// Reads every item of the row's key into `key`
    fn read_key(
        &self,
        row: &Row,
        exec: &super::expression::ExecuteContext<'_>,
        state: &mut FeedState,
    ) -> Result<()> {
        state.key.clear();
        for item in &self.items {
            let value = match item {
                KeyItem::Column(idx) => row.get(*idx).cloned().unwrap_or_else(Value::null_unknown),
                KeyItem::Program(program) => state
                    .vm
                    .get_or_insert_with(ExprVM::new)
                    .execute(program, exec)?,
                KeyItem::Null => Value::null_unknown(),
            };
            state.key.push(value);
        }
        Ok(())
    }

    /// Feeds the row to the runs of every call in its group
    #[inline]
    fn feed(
        &self,
        runs: &mut [CompiledAggregate],
        row: &Row,
        exec: &super::expression::ExecuteContext<'_>,
        vm: &mut Option<ExprVM>,
    ) -> Result<()> {
        for (i, run) in runs.iter_mut().enumerate() {
            let column = match self.columns[i] {
                Some(idx) => row.get(idx),
                None => Some(&COUNT_STAR),
            };
            self.inputs[i].feed(run, column, vm, exec, &self.aggregations[i])?;
        }
        Ok(())
    }

    /// `feed` when no call runs a program: each call reads its column
    #[inline(always)]
    fn feed_columns(&self, runs: &mut [CompiledAggregate], row: &Row) {
        for (run, column) in runs.iter_mut().zip(&self.columns) {
            match column {
                Some(idx) => {
                    if let Some(value) = row.get(*idx) {
                        run.accumulate(value);
                    }
                }
                None => run.accumulate(&COUNT_STAR),
            }
        }
    }

    /// `run_one` when no program runs: the key is `columns` of the row
    #[inline(always)]
    fn run_columns(&self, groups: &mut GroupedAggregates, row: &Row, columns: &[usize]) {
        let key = ColumnsKey {
            values: row.as_slice(),
            columns,
        };
        if let Some(group) = groups.group(&key) {
            self.feed_columns(groups.aggregates(group), row);
        }
    }

    /// Runs one row into its group
    #[inline]
    fn run_one(
        &self,
        groups: &mut GroupedAggregates,
        row: &Row,
        reads: &RowReads<'_>,
        state: &mut FeedState,
    ) -> Result<()> {
        if let (false, Some(columns)) = (self.programs, &self.key_columns) {
            self.run_columns(groups, row, columns);
            return Ok(());
        }
        let exec = reads.exec(row);
        let group = match &self.key_columns {
            Some(columns) => groups.group(&ColumnsKey {
                values: row.as_slice(),
                columns,
            }),
            None => {
                self.read_key(row, &exec, state)?;
                groups.group(state.key.as_slice())
            }
        };
        match group {
            Some(group) => self.feed(groups.aggregates(group), row, &exec, &mut state.vm),
            None => Ok(()),
        }
    }

    /// Whether `run_rows` splits an input of `rows` rows over the thread
    /// pool: a large one whose calls all merge and run no program
    fn splits(&self, groups: &GroupedAggregates, rows: usize) -> bool {
        cfg!(feature = "parallel") && rows >= PARALLEL_ROWS && !self.programs && groups.can_merge()
    }

    /// Runs every row into its group, its subqueries answered by `blocks`;
    /// a large input whose calls all merge is split, and its parts merge in
    /// input order
    fn run_rows(
        &self,
        groups: &mut GroupedAggregates,
        rows: &[(i64, Row)],
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<()> {
        #[cfg(feature = "parallel")]
        if self.splits(groups, rows.len()) {
            let chunk_size = (rows.len() / rayon::current_num_threads()).max(1000);
            let params = ctx.params();
            let parts: Vec<GroupedAggregates> = rows
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut part = groups.empty_like();
                    let mut state = FeedState::default();
                    let reads = RowReads {
                        params,
                        blocks: None,
                    };
                    for (_, row) in chunk {
                        self.run_one(&mut part, row, &reads, &mut state)?;
                    }
                    Ok(part)
                })
                .collect::<Result<_>>()?;
            for part in parts {
                groups.merge(part);
            }
            return Ok(());
        }
        if let (false, Some(columns)) = (self.programs, &self.key_columns) {
            if columns.is_empty() {
                if let Some(group) = groups.group(&[] as &[Value]) {
                    let runs = groups.aggregates(group);
                    for (_, row) in rows {
                        self.feed_columns(runs, row);
                    }
                }
                return Ok(());
            }
            for (_, row) in rows {
                self.run_columns(groups, row, columns);
            }
            return Ok(());
        }
        let mut state = FeedState::default();
        let reads = RowReads::of(ctx, blocks);
        for (_, row) in rows {
            self.run_one(groups, row, &reads, &mut state)?;
        }
        Ok(())
    }

    /// Runs into its group every row `source` hands over, as `run_rows`
    /// does a slice's, one at a time
    fn run_each(
        &self,
        groups: &mut GroupedAggregates,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
        mut source: impl RowSource,
    ) -> Result<()> {
        if let (false, Some(columns)) = (self.programs, &self.key_columns) {
            if columns.is_empty() {
                let Some(group) = groups.group(&[] as &[Value]) else {
                    return Ok(());
                };
                let runs = groups.aggregates(group);
                while let Some(row) = source.next_row()? {
                    self.feed_columns(runs, row);
                }
                return Ok(());
            }
            while let Some(row) = source.next_row()? {
                self.run_columns(groups, row, columns);
            }
            return Ok(());
        }
        let mut state = FeedState::default();
        let reads = RowReads::of(ctx, blocks);
        while let Some(row) = source.next_row()? {
            self.run_one(groups, row, &reads, &mut state)?;
        }
        Ok(())
    }

    /// Runs every row into its group in each grouping set, `picks` naming
    /// the key items each set groups by, its subqueries answered by `blocks`
    fn run_sets(
        &self,
        sets: &mut [GroupedAggregates],
        picks: &[Vec<usize>],
        rows: &[(i64, Row)],
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<()> {
        let mut state = FeedState::default();
        let reads = RowReads::of(ctx, blocks);
        for (_, row) in rows {
            let exec = reads.exec(row);
            self.read_key(row, &exec, &mut state)?;
            for (groups, picks) in sets.iter_mut().zip(picks) {
                let key = Picked {
                    key: &state.key,
                    picks,
                };
                if let Some(group) = groups.group(&key) {
                    self.feed(groups.aggregates(group), row, &exec, &mut state.vm)?;
                }
            }
        }
        Ok(())
    }
}

impl Executor {
    /// How every group runs each of the calls
    fn call_templates(&self, aggregations: &[SqlAggregateFunction]) -> Result<Vec<CallTemplate>> {
        aggregations
            .iter()
            .map(|agg| agg.template(&self.function_registry))
            .collect()
    }

    /// Execute SELECT with aggregation (GROUP BY support); the statement's
    /// `blocks`, when it has them, answer its SELECT list and HAVING subqueries
    pub(crate) fn execute_select_with_aggregation(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: RowVec,
        base_columns: &[String],
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Box<dyn QueryResult>> {
        let written_stmt = stmt;
        // A subquery inside an aggregate is resolved per input row and read
        // back as a column, since an aggregate cannot resolve one itself
        let lifted_stmt;
        let lifted_columns;
        let lifts = blocks.is_none() && Self::aggregates_hold_subquery(stmt);
        let (stmt, base_rows, base_columns) = match lifts
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
        let (aggregations, _non_agg_columns) = self.parse_aggregations(stmt, Some(ctx))?;
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
            super::row_count::rows_needed(stmt, ctx)
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
                blocks,
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
                blocks,
            )?
        } else {
            // Regular grouped aggregation - pass limit for early termination
            self.execute_grouped_aggregation(
                &aggregations,
                &group_by_columns,
                stmt,
                &base_rows,
                base_columns,
                &col_index_map,
                ctx,
                aggregation_limit,
                blocks,
            )?
        };

        // Apply HAVING clause BEFORE projection (HAVING may reference aggregates not in SELECT)
        let (having_columns, having_rows) = if let Some(ref having) = stmt.having {
            // Pre-process scalar subqueries in HAVING clause
            // This executes subqueries like (SELECT AVG(a) FROM t) and replaces them with values.
            // One that reads the group is left as written; apply_having
            // runs it once per group with that group as the outer row
            let has_correlated_having = Self::has_correlated_subqueries(having);
            let processed_having = if has_correlated_having || blocks.is_some() {
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
                blocks,
            )?;

            // Collect rows after HAVING filter
            let mut filtered_rows = RowVec::new();
            let mut row_id = 0i64;
            while having_result.next() {
                filtered_rows.push((row_id, having_result.take_row()));
                row_id += 1;
            }
            (result_columns, filtered_rows)
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
                source_columns: base_columns,
                blocks,
                alone: None,
            },
            ctx,
            having_columns,
            having_rows,
        )
    }

    /// The output names of `stmt` when its SELECT list is nothing but its
    /// one row's calls, in their order, with no HAVING or ORDER BY: that row
    /// is its output as it stands
    pub(crate) fn calls_alone(
        stmt: &SelectStatement,
        aggregations: &[SqlAggregateFunction],
    ) -> Option<Vec<String>> {
        // Each item is its own slot's call: a slot another clause added, or
        // one the parse shared between items, moves the others
        let alone = stmt.having.is_none()
            && stmt.order_by.is_empty()
            && stmt.columns.len() == aggregations.len()
            && aggregations.iter().all(|agg| !agg.hidden)
            && stmt.columns.iter().zip(aggregations).all(|(column, agg)| {
                let call = match column {
                    Expression::Aliased(aliased) => aliased.expression.as_ref(),
                    other => other,
                };
                matches!(call, Expression::FunctionCall(f)
                    if is_aggregate_function(&f.function) && ExprKey::of_call(f) == agg.key)
            });
        alone.then(|| stmt.columns.iter().map(item_name).collect())
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
        path.taken();
        let AggregateSlots {
            stmt,
            written_stmt,
            aggregations,
            group_by_count,
            source_columns,
            blocks,
            alone,
        } = slots;

        // A SELECT list of nothing but its calls is the row of their values
        if let Some(names) = alone {
            return Ok(Box::new(ExecutorResult::with_arc_columns(
                CompactArc::clone(names),
                having_rows,
            )));
        }
        if group_by_count == 0 && std::ptr::eq(stmt, written_stmt) {
            if let Some(names) = Self::calls_alone(stmt, aggregations) {
                return Ok(Box::new(ExecutorResult::new(names, having_rows)));
            }
        }

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
                source_columns,
                aggregations,
                having_columns,
                having_rows,
                blocks,
            )?
        } else {
            // Hidden aggregates exist: extract only the needed columns
            // up front, then move the aggregated rows instead of cloning
            // the whole result set
            // Lifting rewrote the aggregates; the sort looks for them as written
            let written_keys: Option<Vec<ExprKey>> = if std::ptr::eq(stmt, written_stmt) {
                None
            } else {
                let (written, _) = self.parse_aggregations(written_stmt, Some(ctx))?;
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
                source_columns,
                aggregations,
                having_columns,
                having_rows,
                blocks,
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

    /// The grouped columns ORDER BY reads that no SELECT item copies, paired
    /// with their per-group values
    pub(crate) fn collect_hidden_order_by_columns(
        stmt: &SelectStatement,
        source_columns: &[String],
        source_rows: &RowVec,
    ) -> Vec<(String, Vec<Value>)> {
        let inputs = order_by_inputs(stmt, source_columns);
        if inputs.is_empty() {
            return Vec::new();
        }
        let copied = Self::select_sources(&stmt.columns, source_columns, None);
        inputs
            .into_iter()
            .filter(|&idx| !copied.iter().any(|source| source.input == Some(idx)))
            .map(|idx| {
                let values = source_rows
                    .iter()
                    .map(|(_, row)| row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                    .collect();
                (source_columns[idx].clone(), values)
            })
            .collect()
    }

    fn append_hidden_order_by_columns(
        mut columns: Vec<String>,
        mut rows: RowVec,
        hidden: Vec<(String, Vec<Value>)>,
    ) -> (Vec<String>, RowVec) {
        for (name, values) in hidden {
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_post_aggregation_expressions(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        source_columns: &[String],
        aggregations: &[SqlAggregateFunction],
        agg_columns: Vec<String>,
        agg_rows: RowVec,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<(Vec<String>, RowVec)> {
        // Parse GROUP BY items for GROUPING() function support
        let group_by_columns = self.parse_group_by(stmt, &agg_columns)?;
        if let Some(name) = ungrouped_read(stmt, &group_by_columns, source_columns, ctx) {
            return Err(ungrouped_column(&name));
        }

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
        let processed_columns = if has_any_correlated || blocks.is_some() {
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
                            == qualified(&qid.qualifier.value_lower, &qid.name.value_lower)
                    }
                    Expression::Aliased(a) => {
                        slot_of(&a.expression) == Some(i)
                            && name.to_lowercase() == a.alias.value_lower.as_str()
                    }
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
                // The rows are the SELECT list's, named as its items are: a
                // slot of a column or an alias may carry another spelling
                let mut agg_columns = agg_columns;
                for (name, item) in agg_columns.iter_mut().zip(&stmt.columns) {
                    if let Some(spelled) = item_spelled_name(item).filter(|s| *s != name) {
                        *name = spelled.to_string();
                    }
                }
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

            final_columns.push(item_name(original_expr));
            match col_expr {
                Expression::Identifier(id) => {
                    // A name no result column carries may be another alias of a shared slot
                    match slot_of(col_expr)
                        .filter(|_| !agg_col_index_map.contains_key(id.value_lower.as_str()))
                    {
                        Some(slot) => column_sources.push(ColumnSource::AggSlot(slot)),
                        None if agg_col_index_map.contains_key(id.value_lower.as_str()) => {
                            column_sources.push(ColumnSource::AggColumn(id.value_lower.to_string()))
                        }
                        // Neither grouped nor aggregated: the slot compiler refuses it
                        None => column_sources.push(make_source(col_expr, is_correlated)),
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    let (qualifier, name) = (&qid.qualifier.value_lower, &qid.name.value_lower);
                    match qualified_index(&agg_col_index_map, qualifier, name) {
                        Some(index) => column_sources
                            .push(ColumnSource::AggColumn(agg_columns[index].to_lowercase())),
                        None => column_sources.push(make_source(col_expr, is_correlated)),
                    }
                }
                Expression::FunctionCall(func) => {
                    if is_aggregate_function(&func.function) {
                        column_sources.push(match slot_of(col_expr) {
                            Some(slot) => ColumnSource::AggSlot(slot),
                            None => make_source(col_expr, is_correlated),
                        });
                    } else if func.function.eq_ignore_ascii_case("GROUPING") {
                        // GROUPING() function - map to the appropriate grouping flag column
                        if let Some(idx) = self.find_grouping_column_index(func, &group_by_columns)
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
                        match slot_of(col_expr).filter(|_| !is_correlated) {
                            Some(slot) => column_sources.push(ColumnSource::AggSlot(slot)),
                            None => column_sources.push(make_source(col_expr, is_correlated)),
                        }
                    }
                }
                Expression::Aliased(aliased) => {
                    // The item by its own identity, never by its alias: an
                    // alias may repeat the name another slot carries
                    let slot = slot_of(&aliased.expression).filter(|_| !is_correlated);
                    if let Some(slot) = slot {
                        column_sources.push(ColumnSource::AggSlot(slot));
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
                                if let Some(idx) =
                                    self.find_grouping_column_index(func, &group_by_columns)
                                {
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
                                CompactArc::from(qualified(alias, &col_lower).as_str())
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
                    blocks.is_some(),
                    Some(slot_params(ctx, blocks)),
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
                        Some(p) => vm.execute(p, &with_blocks(slot_context(&row, ctx), blocks))?,
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
                                    false,
                                    None,
                                )?;
                                vm.execute(&program, &slot_context(&row, &correlated_ctx))
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
        super::utils::function_call_to_string(func)
    }

    /// Find the GROUP BY column index for a GROUPING() function call
    /// Returns the index (0-based) of the GROUP BY column that matches the GROUPING() argument
    fn find_grouping_column_index(
        &self,
        func: &crate::parser::ast::FunctionCall,
        group_by_columns: &[GroupByItem],
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

    /// The running state of one aggregate call: the registry's function with
    /// the call's extra arguments, ORDER BY and DISTINCT applied, the one way
    /// every path builds it; None for a name the registry does not know
    pub(crate) fn aggregate_state(
        registry: &crate::functions::FunctionRegistry,
        name: &str,
        distinct: bool,
        extra_args: &[Value],
        order_by: &[crate::parser::ast::OrderByExpression],
    ) -> Option<AggregateState> {
        registry.get_aggregate(name).map(|function| {
            AggregateState::new(
                function,
                distinct,
                extra_args,
                Self::aggregate_order_keys(order_by),
            )
        })
    }

    /// Aggregate, then run the window functions over the aggregate's output.
    /// A subquery inside an aggregate is resolved per input row first, unless
    /// the statement's `blocks` answer its subqueries, and both stages read
    /// the same rewritten statement
    pub(crate) fn execute_aggregation_then_windows(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<WindowOutput> {
        let lifted_stmt;
        let lifted_rows;
        let lifted_columns;
        let (stmt, base_rows, base_columns) = match (blocks.is_none()
            && Self::aggregates_hold_subquery(stmt))
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
            self.execute_aggregation_for_window(stmt, ctx, base_rows, base_columns, blocks)?;
        self.execute_select_with_window_functions_over_groups(
            stmt,
            ctx,
            &agg_rows,
            &agg_columns,
            &slots,
            blocks,
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
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<(Vec<String>, RowVec, FxHashMap<ExprKey, usize>)> {
        // Parse aggregations and group by columns
        let (aggregations, _non_agg_columns) = self.parse_aggregations(stmt, Some(ctx))?;
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
                blocks,
            )?
        } else {
            // Grouped aggregation - no limit since window functions need all groups
            self.execute_grouped_aggregation(
                &aggregations,
                &group_by_columns,
                stmt,
                base_rows,
                base_columns,
                &col_index_map,
                ctx,
                None, // Window functions need all groups
                blocks,
            )?
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
                blocks.is_some(),
                Some(slot_params(ctx, blocks)),
            )?;
            let mut vm = ExprVM::new();
            for (id, row) in result_rows {
                if vm.execute_bool_checked(
                    &having_filter,
                    &with_blocks(slot_context(&row, ctx), blocks),
                )? {
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

    /// Parse aggregate functions from SELECT list; `ctx` binds the parameters
    /// an extra argument reads, None where only the shape is asked
    /// Returns: (aggregations, non_agg_columns, post_agg_expressions)
    pub(crate) fn parse_aggregations(
        &self,
        stmt: &SelectStatement,
        ctx: Option<&ExecutionContext>,
    ) -> Result<(Vec<SqlAggregateFunction>, Vec<String>)> {
        self.parse_calls(stmt, ExtraArgsRead::With(ctx))
    }

    /// The aggregate calls of `stmt` for their identities alone, their extra
    /// arguments unread: a block's name an enclosing row before it binds
    pub(crate) fn parse_aggregation_calls(
        &self,
        stmt: &SelectStatement,
    ) -> Result<Vec<SqlAggregateFunction>> {
        Ok(self.parse_calls(stmt, ExtraArgsRead::Unread)?.0)
    }

    /// The aggregate calls of a subquery block's bound `stmt`, each keeping
    /// its extra arguments as written when one is not a literal
    pub(crate) fn parse_block_aggregations(
        &self,
        stmt: &SelectStatement,
    ) -> Result<Vec<SqlAggregateFunction>> {
        Ok(self.parse_calls(stmt, ExtraArgsRead::Block)?.0)
    }

    fn parse_calls(
        &self,
        stmt: &SelectStatement,
        ctx: ExtraArgsRead<'_>,
    ) -> Result<(Vec<SqlAggregateFunction>, Vec<String>)> {
        let mut aggregations = Vec::new();
        let mut non_agg_columns = Vec::new();

        for col_expr in &stmt.columns {
            self.extract_aggregates_from_expr(
                col_expr,
                &mut aggregations,
                &mut non_agg_columns,
                ctx,
            )?;
        }
        let select_count = aggregations.len();

        // Also extract aggregates from HAVING clause
        // These need to be computed even if they're not in SELECT
        if let Some(ref having) = stmt.having {
            self.extract_aggregates_from_expr(
                having,
                &mut aggregations,
                &mut non_agg_columns,
                ctx,
            )?;
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
                ctx,
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
                self.extract_aggregates_from_expr(
                    expr,
                    &mut aggregations,
                    &mut non_agg_columns,
                    ctx,
                )?;
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
        ctx: ExtraArgsRead<'_>,
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

                    let (column, distinct, (extra_args, extra_error), expression) =
                        self.extract_agg_column(&func.arguments, ctx)?;
                    let column_lower = column.to_lowercase();
                    aggregations.push(SqlAggregateFunction {
                        name: func.function.to_string(),
                        column,
                        column_lower,
                        alias: None,
                        distinct: distinct || func.is_distinct,
                        extra_args,
                        extra_error,
                        extra_exprs: ctx.written(&func.arguments),
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
                        self.extract_aggregates_from_expr(arg, aggregations, non_agg_columns, ctx)?;
                    }
                }
            }
            Expression::Aliased(aliased) => {
                // For aliased expressions, extract aggregates from the inner expression
                self.extract_aggregates_from_aliased(aliased, aggregations, non_agg_columns, ctx)?;
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
                        ctx,
                    )?;
                    self.extract_aggregates_from_expr(
                        &when_clause.then_result,
                        aggregations,
                        non_agg_columns,
                        ctx,
                    )?;
                }
                if let Some(ref else_val) = case.else_value {
                    self.extract_aggregates_from_expr(
                        else_val,
                        aggregations,
                        non_agg_columns,
                        ctx,
                    )?;
                }
            }
            Expression::Infix(infix) => {
                self.extract_aggregates_from_expr(&infix.left, aggregations, non_agg_columns, ctx)?;
                self.extract_aggregates_from_expr(
                    &infix.right,
                    aggregations,
                    non_agg_columns,
                    ctx,
                )?;
            }
            Expression::Prefix(prefix) => {
                self.extract_aggregates_from_expr(
                    &prefix.right,
                    aggregations,
                    non_agg_columns,
                    ctx,
                )?;
            }
            Expression::Cast(cast) => {
                self.extract_aggregates_from_expr(&cast.expr, aggregations, non_agg_columns, ctx)?;
            }
            Expression::QuantifiedSet(set) => {
                self.extract_aggregates_from_expr(&set.left, aggregations, non_agg_columns, ctx)?;
            }
            Expression::In(in_expr) if expression_contains_aggregate(&in_expr.left) => {
                self.extract_aggregates_from_expr(
                    &in_expr.left,
                    aggregations,
                    non_agg_columns,
                    ctx,
                )?;
            }
            Expression::AllAny(all_any) if expression_contains_aggregate(&all_any.left) => {
                self.extract_aggregates_from_expr(
                    &all_any.left,
                    aggregations,
                    non_agg_columns,
                    ctx,
                )?;
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
                    self.extract_aggregates_from_expr(expr, aggregations, non_agg_columns, ctx)?;
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
        ctx: ExtraArgsRead<'_>,
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

                    let (column, distinct, (extra_args, extra_error), expression) =
                        self.extract_agg_column(&func.arguments, ctx)?;
                    let column_lower = column.to_lowercase();
                    aggregations.push(SqlAggregateFunction {
                        name: func.function.to_string(),
                        column,
                        column_lower,
                        alias: Some(aliased.alias.value.to_string()),
                        distinct: distinct || func.is_distinct,
                        extra_args,
                        extra_error,
                        extra_exprs: ctx.written(&func.arguments),
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
                        self.extract_aggregates_from_expr(arg, aggregations, non_agg_columns, ctx)?;
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
                        ctx,
                    )?;
                    self.extract_aggregates_from_expr(
                        &when_clause.then_result,
                        aggregations,
                        non_agg_columns,
                        ctx,
                    )?;
                }
                if let Some(ref else_val) = case.else_value {
                    self.extract_aggregates_from_expr(
                        else_val,
                        aggregations,
                        non_agg_columns,
                        ctx,
                    )?;
                }
            }
            Expression::Cast(cast) => {
                // Extract aggregates from CAST expression (e.g., CAST(SUM(val) AS TEXT) AS sum_text)
                self.extract_aggregates_from_expr(&cast.expr, aggregations, non_agg_columns, ctx)?;
            }
            _ => {
                self.extract_aggregates_from_expr(
                    &aliased.expression,
                    aggregations,
                    non_agg_columns,
                    ctx,
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
    /// - extra_args: Additional arguments (e.g., separator for STRING_AGG) and
    ///   the error one gave, see [`aggregate_extra_args`]
    /// - expression: The expression to evaluate (Some for complex expressions like val * 2)
    fn extract_agg_column(
        &self,
        args: &[Expression],
        ctx: ExtraArgsRead<'_>,
    ) -> Result<(String, bool, ExtraArgs, Option<Expression>)> {
        if args.is_empty() {
            return Ok(("*".to_string(), false, (Vec::new(), None), None));
        }

        let (column, distinct, expression) = match &args[0] {
            Expression::Star(_) => ("*".to_string(), false, None),
            Expression::Identifier(id) => (id.value.to_string(), false, None),
            Expression::QualifiedIdentifier(qid) => {
                // Use full qualified name (e.g., "p.price" instead of just "price")
                // This is needed for JOIN queries where columns are qualified with table aliases
                let qualified_name = qualified(&qid.qualifier.value, &qid.name.value);
                (qualified_name, false, None)
            }
            // For expressions like val * 2, a + b, etc. - store the expression
            expr => {
                let expr_str = self.expression_to_string(expr);
                (expr_str, false, Some(expr.clone()))
            }
        };

        let extra = match ctx {
            ExtraArgsRead::With(ctx) => aggregate_extra_args(args, ctx)?,
            ExtraArgsRead::Block => aggregate_extra_args(args, None)?,
            ExtraArgsRead::Unread => (Vec::new(), None),
        };
        Ok((column, distinct, extra, expression))
    }

    /// Parse GROUP BY clause
    pub(crate) fn parse_group_by(
        &self,
        stmt: &SelectStatement,
        _base_columns: &[String],
    ) -> Result<Vec<GroupByItem>> {
        let mut group_items = Vec::new();

        for expr in grouping_items(stmt) {
            if expression_contains_aggregate(expr) {
                return Err(aggregate_group_key());
            }
            match expr {
                Expression::Identifier(id) => {
                    let named = grouped_item(stmt, expr).map(|item| &stmt.columns[item]);
                    if let Some(Expression::Aliased(aliased)) = named {
                        // Use the aliased expression, with the alias as the display name
                        group_items.push(GroupByItem::Expression {
                            expr: (*aliased.expression).clone(),
                            display_name: id.value.to_string(),
                        });
                    } else {
                        // Regular column reference
                        group_items.push(GroupByItem::Column(id.value.to_string()));
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    // Use full qualified name (e.g., "c.name" instead of just "name")
                    let qualified_name = qualified(&qid.qualifier.value, &qid.name.value);
                    group_items.push(GroupByItem::Column(qualified_name));
                }
                Expression::IntegerLiteral(lit) => {
                    // GROUP BY 1 refers to first SELECT column (1-indexed)
                    if let Some(at) = grouped_item(stmt, expr) {
                        // Convert position to the actual SELECT column expression
                        let select_col = &stmt.columns[at];
                        if expression_contains_aggregate(select_col) {
                            return Err(aggregate_group_key());
                        }
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
                        return Err(crate::core::Error::invalid_argument(format!(
                            "GROUP BY position {} is not in the select list",
                            lit.value
                        )));
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

    /// Execute global aggregation (no GROUP BY): the one group of the
    /// grouped path, there with no rows
    fn execute_global_aggregation(
        &self,
        aggregations: &[SqlAggregateFunction],
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<(Vec<String>, RowVec)> {
        let mut groups = GroupedAggregates::global(self.call_templates(aggregations)?);
        RowFeed::new(aggregations, &[], None, columns, col_index_map, ctx, blocks)?.run_rows(
            &mut groups,
            rows,
            ctx,
            blocks,
        )?;
        let result_columns = aggregations.iter().map(|a| a.get_column_name()).collect();
        Ok((result_columns, groups.into_rows()?))
    }

    /// Execute grouped aggregation (with GROUP BY). With `limit`, no group
    /// past the first `limit` is made, and their rows are not aggregated
    #[allow(clippy::too_many_arguments)]
    fn execute_grouped_aggregation(
        &self,
        aggregations: &[SqlAggregateFunction],
        group_by_items: &[GroupByItem],
        stmt: &SelectStatement,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        ctx: &ExecutionContext,
        limit: Option<usize>,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<(Vec<String>, RowVec)> {
        let mut groups =
            GroupedAggregates::new(self.call_templates(aggregations)?, group_by_items.len());
        if let Some(limit) = limit {
            groups = groups.with_group_cap(limit);
        }
        RowFeed::new(
            aggregations,
            group_by_items,
            Some(stmt),
            columns,
            col_index_map,
            ctx,
            blocks,
        )?
        .run_rows(&mut groups, rows, ctx, blocks)?;

        let mut result_columns: Vec<String> =
            self.resolve_group_by_column_names_new(group_by_items, columns, col_index_map);
        result_columns.extend(aggregations.iter().map(|a| a.get_column_name()));
        Ok((result_columns, groups.into_rows()?))
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
        blocks: Option<&dyn SubqueryExecutor>,
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

        let feed = RowFeed::new(
            aggregations,
            group_by_items,
            Some(stmt),
            columns,
            col_index_map,
            ctx,
            blocks,
        )?;
        let picks: Vec<Vec<usize>> = grouping_sets
            .iter()
            .map(|set| {
                set.active_columns
                    .iter()
                    .enumerate()
                    .filter(|(_, &active)| active)
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        let mut sets = picks
            .iter()
            .map(|picks| {
                let calls = self.call_templates(aggregations)?;
                Ok(match picks.len() {
                    0 => GroupedAggregates::global(calls),
                    width => GroupedAggregates::new(calls, width),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        feed.run_sets(&mut sets, &picks, rows, ctx, blocks)?;

        // Each group's items where its set groups by them and NULL where the
        // set rolls them up, then its aggregates, then the GROUPING flags
        let mut all_result_rows: RowVec = RowVec::new();
        let mut row_id = 0i64;
        for (grouping_set, groups) in grouping_sets.iter().zip(sets) {
            for (_, row) in groups.into_rows()? {
                let mut slots = row.into_values().into_iter();
                let mut row_values: CompactVec<Value> = CompactVec::with_capacity(
                    group_by_items.len() + aggregations.len() + group_by_items.len(),
                );
                for &is_active in &grouping_set.active_columns {
                    row_values.push(match is_active {
                        true => slots.next().unwrap_or_else(Value::null_unknown),
                        false => Value::null_unknown(),
                    });
                }
                row_values.extend(slots);
                for &is_active in &grouping_set.active_columns {
                    row_values.push(Value::Integer(if is_active { 0 } else { 1 }));
                }
                all_result_rows.push((row_id, Row::from_compact_vec(row_values)));
                row_id += 1;
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
    pub(crate) fn resolve_group_by_column_names_new(
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
                GroupByItem::Expression { display_name, .. } => {
                    // Use the display name (alias or generated name)
                    names.push(display_name.clone());
                }
            }
        }
        names
    }

    /// Look up column index, handling both qualified (e.g., "o.amount") and unqualified names,
    /// a qualified one by [`qualified_index`]
    pub(crate) fn lookup_column_index(
        column_lower: &str,
        col_index_map: &StringMap<usize>,
    ) -> Option<usize> {
        match split_qualified(column_lower) {
            Some(_) => qualified_key_index(col_index_map, column_lower),
            None => col_index_map.get(column_lower).copied(),
        }
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
        blocks: Option<&dyn SubqueryExecutor>,
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
        if blocks.is_none() && Self::has_correlated_subqueries(having) {
            let columns_arc = CompactArc::new(columns.to_vec());
            let keys: Vec<(CompactArc<str>, Option<CompactArc<str>>)> = columns
                .iter()
                .map(|c| {
                    let bare = c.to_lowercase();
                    let qualified = outer_table
                        .filter(|_| split_qualified(&bare).is_none())
                        .map(|t| CompactArc::from(qualified(&t.to_lowercase(), &bare).as_str()));
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
                    false,
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
            blocks.is_some(),
            Some(slot_params(ctx, blocks)),
        )?;
        let mut filtered_rows = RowVec::new();
        let mut new_id = 0i64;
        let mut vm = ExprVM::new();
        for (_, row) in rows {
            if vm.execute_bool_checked(
                &having_filter,
                &with_blocks(slot_context(&row, ctx), blocks),
            )? {
                filtered_rows.push((new_id, row));
                new_id += 1;
            }
        }

        Ok(Box::new(ExecutorResult::new(
            columns.to_vec(),
            filtered_rows,
        )))
    }

    /// The one group of `aggregations`, every row of `table` fed to it; the
    /// statement's `blocks` answer what the calls ask of each row
    pub(super) fn global_scan_rows(
        &self,
        table: &dyn crate::storage::traits::Table,
        aggregations: &[SqlAggregateFunction],
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<RowVec> {
        let columns = table.schema().column_names_lower_arc();
        let mut scanner = table.scan(&[], None)?;
        let input = FeedInput::Each(ScannedRows(&mut *scanner), table.row_count_hint());
        let rows = self.global_row(aggregations, &columns, ctx, blocks, input)?;
        scanner.close()?;
        Ok(rows)
    }

    /// The one group of `aggregations` over the rows of `input`, whose
    /// columns are `columns`; the statement's `blocks` answer what the calls
    /// ask of each row
    pub(super) fn global_row(
        &self,
        aggregations: &[SqlAggregateFunction],
        columns: &[String],
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
        input: FeedInput<'_, impl RowSource>,
    ) -> Result<RowVec> {
        let col_index_map = build_column_index_map(columns);
        let mut groups = GroupedAggregates::global(self.call_templates(aggregations)?);
        let feed = RowFeed::new(
            aggregations,
            &[],
            None,
            columns,
            &col_index_map,
            ctx,
            blocks,
        )?;
        match input {
            FeedInput::Rows(rows) => feed.run_rows(&mut groups, rows, ctx, blocks)?,
            FeedInput::Each(source, rows) if feed.splits(&groups, rows) => {
                feed.run_rows(&mut groups, &source.held()?, ctx, blocks)?
            }
            FeedInput::Each(source, _) => feed.run_each(&mut groups, ctx, blocks, source)?,
        }
        groups.into_rows()
    }

    /// Try streaming aggregation for derived tables (FROM subqueries).
    ///
    /// OPTIMIZATION: For simple GROUP BY + COUNT(*) on derived tables without WHERE clause,
    /// stream directly to aggregation HashMap without materializing all rows first.
    /// This reduces memory allocations from O(N) to O(groups).
    ///
    /// Hands `result` back unread when the optimization cannot be applied,
    /// so the derived table's statement runs once either way.
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
    ) -> Result<std::result::Result<Box<dyn QueryResult>, Box<dyn QueryResult>>> {
        if Self::aggregates_hold_subquery(stmt) {
            return Ok(Err(result));
        }

        // Quick eligibility checks
        if !classification.has_group_by {
            return Ok(Err(result));
        }
        if classification.has_having {
            return Ok(Err(result));
        }
        if classification.has_window_functions {
            return Ok(Err(result));
        }
        // Skip ROLLUP/CUBE/GROUPING SETS
        if stmt.group_by.modifier != crate::parser::ast::GroupByModifier::None {
            return Ok(Err(result));
        }

        // Only single-column GROUP BY for this optimization
        if stmt.group_by.columns.len() != 1 {
            return Ok(Err(result));
        }

        // The derived table's columns carry no qualifier
        let Some((group_col_name, _)) = self.single_group_column(stmt)? else {
            return Ok(Err(result));
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
            None => return Ok(Err(result)),
        };

        // Parse aggregations
        let (aggregations, non_agg_columns) = self.parse_aggregations(stmt, Some(ctx))?;

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
            return Ok(Err(result));
        }

        // Plain COUNT, SUM, AVG, MIN and MAX over the derived table's columns
        let plain = aggregations.iter().all(|agg| {
            let column = || Self::lookup_column_index(&agg.column_lower, &col_index_map).is_some();
            agg.answered_with(Modifiers::NONE)
                && match agg.name.as_str() {
                    "COUNT" => agg.column == "*" || column(),
                    "SUM" | "AVG" | "MIN" | "MAX" => agg.column != "*" && column(),
                    _ => false,
                }
        });
        if !plain {
            return Ok(Err(result));
        }

        let mut slot_columns = Vec::with_capacity(1 + aggregations.len());
        slot_columns.push(source_columns[group_col_idx].clone());
        slot_columns.extend(aggregations.iter().map(|agg| agg.get_column_name()));

        let group_items = [GroupByItem::Column(group_col_name)];
        let mut groups = GroupedAggregates::new(self.call_templates(&aggregations)?, 1);
        let feed = RowFeed::new(
            &aggregations,
            &group_items,
            Some(stmt),
            &source_columns,
            &col_index_map,
            ctx,
            None,
        )?;
        let mut state = FeedState::default();
        let reads = RowReads::of(ctx, None);
        while result.next() {
            feed.run_one(&mut groups, result.row(), &reads, &mut state)?;
        }
        if let Some(err) = result.last_error() {
            return Err(err);
        }

        let slots = AggregateSlots {
            stmt,
            written_stmt: stmt,
            aggregations: &aggregations,
            group_by_count: 1,
            source_columns: &source_columns,
            blocks: None,
            alone: None,
        };
        self.finish_aggregation(
            AggregatePath::StreamingDerivedTable,
            slots,
            ctx,
            slot_columns,
            groups.into_rows()?,
        )
        .map(Ok)
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
