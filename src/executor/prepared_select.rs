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

//! The prepared family: a single-table SELECT whose WHERE holds scalar
//! subqueries over single tables. It is classified from its syntax, prepared
//! once per cached plan into programs, and run with each subquery as a
//! parameterised scan whose outer reads fill parameter slots.

use std::cell::{Cell, OnceCell, RefCell};
use std::sync::{Arc, RwLock};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::api::params::ParamVec;
use crate::common::{CompactArc, SmartString};
use crate::core::predicate::{kind, KeySide, PredicateSet};
use crate::core::{DataType, Error, Operator, Result, Row, RowVec, Schema, SchemaColumn, Value};
use crate::functions::aggregate::{CallTemplate, ColumnsKey, CompiledAggregate, GroupedAggregates};
use crate::functions::{FunctionRegistry, Volatility};
use crate::parser::ast::*;
use crate::parser::scope::unknown_qualifier;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::expression::{
    AndExpr, ComparisonExpr, ConstBoolExpr, Expression as StorageExpr,
};
use crate::storage::mvcc::engine::ViewDefinition;
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::traits::transaction::TableUse;
use crate::storage::traits::{
    CappedEqual, Engine, ProbeScratch, QueryResult, Scanner, Table, Transaction,
};

use super::aggregation::{
    aggregate_alias_key, aggregate_group_key, aggregate_slot_keys, check_grouped_reads,
    grouped_item, grouping_items, ungrouped_column, AggregatePath, CallInputs, GroupByItem,
    Modifiers, SqlAggregateFunction, COUNT_STAR,
};
use super::context::{
    cache_block_answer, get_cached_block_answer, subquery_cache_tables, BlockAnswer, BlockOrigin,
    ExecutionContext, ExistsKeys, ExistsOutcome, KeptMembers, MemberKeys, MemberSet,
};
use super::expr_key::ExprKey;
use super::expression::{
    more_than_one_row, CompareOp, CompileContext, CompileError, ExecuteContext, ExprCompiler,
    ExprVM, Op, QuantifiedMembers, RowComparison, RowFilter, RowMembers, RowTest, SharedProgram,
    SubqueryExecutor,
};
use super::operators::index_nested_loop::{crosses_inner_key, EQUALITY_CANDIDATE_CAP};
use super::prepared_join::{JoinSide, JoinSides};
use super::projection::{Item, Projection};
use super::pushdown::{self, coerce_constant, read_as_column, scan_compares};
use super::query::{ColumnSource, OutputLayout};
use super::query_cache::{CompiledExecution, NestedPlan};
use super::query_classification::{get_classification, QueryClassification};
use super::result::{
    DistinctRows, ExecutorResult, FilteredResult, ProjectionResult, ScannerResult,
};
use super::utils::{
    bare_name, build_column_index_map, combine_predicates_with_and, expression_contains_aggregate,
    extract_literal_value, flip_operator, for_each_qualified_star_column,
    get_table_alias_from_expr, infix_to_operator, is_aggregate_function, item_name,
    order_by_names_an_alias, split_qualified, strip_table_qualifier, QUALIFIER_SEP,
};
use super::volatility::{expression_volatility, statement_draws_per_call};
use super::Executor;

/// Scopes one chain may hold: a block's own table and every enclosing one
const MAX_SCOPES: usize = 32;

/// The qualifier of a column the resolver bound: NUL, which the lexer
/// takes in no name, so no table or column can spell it
const RESOLVED: &str = "\u{0}";

/// Why a statement runs on the general path instead of the prepared one
pub(crate) type OutOfFamily = &'static str;

/// The rows, their columns, whether LIMIT and OFFSET were applied, how many
/// of the columns the statement shows, and their layout for an ORDER BY
type PreparedOutput = (
    Box<dyn crate::storage::traits::QueryResult>,
    CompactArc<Vec<String>>,
    bool,
    usize,
    Option<CompactArc<OutputLayout>>,
    // Whether its rows are already distinct
    bool,
);

/// The classifier's walk over one statement
struct Family<'a> {
    executor: &'a Executor,
    /// Every block's FROM that names a table, a view or a CTE
    blocks: RefCell<SmallVec<[&'a SimpleTableSource; 4]>>,
    /// The statement's own CTEs, which its FROM may name and no block may
    with: Option<&'a WithClause>,
}

impl<'a> Family<'a> {
    /// One name, read without AS OF
    fn one_table(
        table_expr: Option<&'a Expression>,
    ) -> std::result::Result<&'a SimpleTableSource, OutOfFamily> {
        let Some(Expression::TableSource(ts)) = table_expr else {
            return Err("FROM is not one table");
        };
        if ts.as_of.is_some() {
            return Err("FROM reads AS OF");
        }
        Ok(ts)
    }

    /// The statement's FROM: one table, view or CTE, a derived table, a
    /// VALUES list or a table function; the statement of a view, a CTE or a
    /// derived table runs as any statement does; or none, which reads one
    /// row without columns
    fn statement_from(
        &self,
        table_expr: Option<&'a Expression>,
    ) -> std::result::Result<(), OutOfFamily> {
        match table_expr {
            None | Some(Expression::SubquerySource(_)) => Ok(()),
            Some(Expression::ValuesSource(values)) => {
                match values.rows.iter().flatten().any(Executor::has_subqueries) {
                    true => Err("a VALUES row holding a subquery"),
                    false => Ok(()),
                }
            }
            Some(Expression::FunctionTableSource(function)) => {
                match function.arguments.iter().any(Executor::has_subqueries) {
                    true => Err("a table function argument holding a subquery"),
                    false => Ok(()),
                }
            }
            Some(Expression::TableSource(ts)) if ts.as_of.is_none() => Ok(()),
            other => Self::one_table(other).map(drop),
        }
    }

    /// A block's FROM: one table, view or CTE, a VALUES list or a table
    /// function whose values name no column, read once per run; or none,
    /// which reads one row without columns
    fn block_from(
        &self,
        table_expr: Option<&'a Expression>,
    ) -> std::result::Result<(), OutOfFamily> {
        match table_expr {
            None => Ok(()),
            Some(Expression::ValuesSource(values)) => {
                match values.rows.iter().flatten().any(mentions_column) {
                    true => Err("a subquery's VALUES row naming a column or holding a subquery"),
                    false => Ok(()),
                }
            }
            Some(Expression::FunctionTableSource(function)) => {
                match function.arguments.iter().any(mentions_column) {
                    true => Err("a subquery's table function argument naming a column"),
                    false => Ok(()),
                }
            }
            Some(Expression::SubquerySource(derived)) => self.derived(&derived.subquery),
            Some(Expression::JoinSource(join)) => self.join_from(join),
            some => {
                self.blocks.borrow_mut().push(Self::one_table(some)?);
                Ok(())
            }
        }
    }

    /// A join in the FROM of a block or of a derived table in one, which
    /// the block runs as a statement of its own: each side a source a
    /// derived table reads, each subquery in it a derived table of its own
    fn join_from(&self, join: &'a JoinTableSource) -> std::result::Result<(), OutOfFamily> {
        let mut nested = Vec::new();
        for side in [&*join.left, &*join.right] {
            match side {
                Expression::JoinSource(join) => self.join_from(join)?,
                Expression::SubquerySource(derived) => self.derived(&derived.subquery)?,
                Expression::ValuesSource(values) => values
                    .rows
                    .iter()
                    .flatten()
                    .for_each(|e| nested_statements(e, &mut nested)),
                Expression::FunctionTableSource(function) => function
                    .arguments
                    .iter()
                    .for_each(|e| nested_statements(e, &mut nested)),
                side => match Self::one_table(Some(side)) {
                    Ok(ts) => self.blocks.borrow_mut().push(ts),
                    Err("FROM reads AS OF") => return Err("a join side read AS OF"),
                    Err(_) => return Err("a join side the family does not read"),
                },
            }
        }
        if let Some(on) = join.condition.as_deref() {
            nested_statements(on, &mut nested);
        }
        nested.into_iter().try_for_each(|s| self.derived(s))
    }

    /// A derived table in a block's FROM, which runs as a statement of its
    /// own: every FROM in it, its subqueries' too, one source whose names
    /// the walk binds over, no WITH, and no LIMIT or OFFSET naming a column
    fn derived(&self, stmt: &'a SelectStatement) -> std::result::Result<(), OutOfFamily> {
        if stmt.with.is_some() {
            return Err("a derived table with its own WITH");
        }
        if stmt
            .limit
            .iter()
            .chain(&stmt.offset)
            .any(|c| mentions_column(c))
        {
            return Err("a derived table's LIMIT or OFFSET naming a column");
        }
        let mut nested = Vec::new();
        match stmt.table_expr.as_deref() {
            None => {}
            Some(Expression::ValuesSource(values)) => {
                values
                    .rows
                    .iter()
                    .flatten()
                    .for_each(|e| nested_statements(e, &mut nested));
            }
            Some(Expression::FunctionTableSource(function)) => {
                function
                    .arguments
                    .iter()
                    .for_each(|e| nested_statements(e, &mut nested));
            }
            Some(Expression::SubquerySource(derived)) => self.derived(&derived.subquery)?,
            Some(Expression::JoinSource(join)) => {
                // A star over a join is named only by its run
                if stmt
                    .columns
                    .iter()
                    .any(|c| matches!(c, Expression::Star(_)))
                {
                    return Err("a derived table selecting * over a join");
                }
                self.join_from(join)?
            }
            some => match Self::one_table(some) {
                Ok(ts) => self.blocks.borrow_mut().push(ts),
                Err("FROM reads AS OF") => return Err("a derived table read AS OF"),
                Err(_) => return Err("a derived table over a source the family does not read"),
            },
        }
        for branch in &stmt.set_operations {
            self.derived(&branch.right)?;
        }
        let grouping_sets = match &stmt.group_by.modifier {
            GroupByModifier::GroupingSets(sets) => sets.as_slice(),
            _ => &[],
        };
        let windows = stmt.window_defs.iter().flat_map(|def| {
            def.partition_by
                .iter()
                .chain(def.order_by.iter().map(|o| &o.expression))
        });
        stmt.columns
            .iter()
            .chain(&stmt.distinct_on)
            .chain(stmt.where_clause.as_deref())
            .chain(&stmt.group_by.columns)
            .chain(grouping_sets.iter().flatten())
            .chain(stmt.having.as_deref())
            .chain(stmt.order_by.iter().map(|o| &o.expression))
            .chain(windows)
            .for_each(|e| nested_statements(e, &mut nested));
        nested.into_iter().try_for_each(|s| self.derived(s))
    }

    /// A block's FROM names a table, a view or a CTE the context holds, and
    /// no CTE of the statement's own WITH, which may inline into a derived
    /// table
    fn catalog(&self) -> std::result::Result<(), OutOfFamily> {
        let own_cte = |name: &str| {
            self.with
                .is_some_and(|with| with.ctes.iter().any(|cte| cte.name.value_lower == name))
        };
        let view = |name: &str| {
            self.executor
                .engine
                .get_view_lowercase(name)
                .map(|view| view.is_some())
                .map_err(|_| "the catalog could not be read")
        };
        for ts in self.blocks.borrow().iter() {
            let name = ts.name.value_lower.as_str();
            if own_cte(name) {
                return Err("a subquery's FROM names a CTE of the statement's WITH");
            }
            view(name)?;
        }
        Ok(())
    }

    /// `expr` holds only the family's expression forms; `site` says what
    /// may appear where it stands, `found` records a subquery that did
    fn expression(
        &self,
        expr: &'a Expression,
        scopes: usize,
        site: Site,
        found: &mut bool,
    ) -> std::result::Result<(), OutOfFamily> {
        let subqueries = matches!(
            site,
            Site::Subqueries | Site::GroupedOutput | Site::Rows | Site::GroupedList
        );
        let mut walk = |e: &'a Expression| self.expression(e, scopes, site, found);
        match expr {
            Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_) => Ok(()),
            Expression::Infix(infix) => {
                if infix.op_type == InfixOperator::Other {
                    return Err("an operator outside the family");
                }
                if let Some((row, sq, _)) = row_subquery(infix) {
                    if !subqueries {
                        return Err("a subquery outside WHERE and the subquery blocks");
                    }
                    self.expression(row, scopes, site.operand(), found)?;
                    *found = true;
                    return self.block(&sq.subquery, scopes + 1, width(row));
                }
                walk(&infix.left)?;
                walk(&infix.right)
            }
            Expression::Prefix(prefix) => {
                if prefix.op_type == PrefixOperator::Other {
                    return Err("a prefix operator outside the family");
                }
                walk(&prefix.right)
            }
            Expression::Between(between) => {
                walk(&between.expr)?;
                walk(&between.lower)?;
                walk(&between.upper)
            }
            Expression::In(in_expr) => {
                let list = match &*in_expr.right {
                    Expression::List(list) => &list.elements,
                    Expression::ExpressionList(list) => &list.expressions,
                    Expression::ScalarSubquery(sq) => {
                        if !subqueries {
                            return Err("a subquery outside WHERE and the subquery blocks");
                        }
                        self.expression(&in_expr.left, scopes, site.operand(), found)?;
                        *found = true;
                        return self.block(&sq.subquery, scopes + 1, width(&in_expr.left));
                    }
                    _ => return Err("IN over something other than a list"),
                };
                walk(&in_expr.left)?;
                list.iter().try_for_each(walk)
            }
            Expression::Like(like) => {
                walk(&like.left)?;
                walk(&like.pattern)?;
                like.escape.as_deref().map_or(Ok(()), walk)
            }
            Expression::Case(case) => {
                if let Some(value) = case.value.as_deref() {
                    walk(value)?;
                }
                for when in &case.when_clauses {
                    walk(&when.condition)?;
                    walk(&when.then_result)?;
                }
                case.else_value.as_deref().map_or(Ok(()), walk)
            }
            Expression::Cast(cast) => walk(&cast.expr),
            Expression::List(list) => list.elements.iter().try_for_each(walk),
            Expression::ExpressionList(list) => list.expressions.iter().try_for_each(walk),
            Expression::FunctionCall(call) => {
                if is_aggregate_function(&call.function) {
                    if !matches!(
                        site,
                        Site::Grouped | Site::GroupedOutput | Site::GroupedList
                    ) {
                        return Err("an aggregate outside a subquery block");
                    }
                    return self.aggregate_call(call, scopes, site, found);
                }
                if call.is_distinct || !call.order_by.is_empty() || call.filter.is_some() {
                    return Err("DISTINCT, ORDER BY or FILTER in a call");
                }
                call.arguments.iter().try_for_each(walk)
            }
            Expression::ScalarSubquery(sq) => {
                if !subqueries {
                    return Err("a subquery outside WHERE and the subquery blocks");
                }
                *found = true;
                self.block(&sq.subquery, scopes + 1, 1)
            }
            Expression::Exists(exists) => {
                if !subqueries {
                    return Err("a subquery outside WHERE and the subquery blocks");
                }
                *found = true;
                self.existence_block(&exists.subquery, scopes + 1)
            }
            Expression::AllAny(all_any) => {
                if !subqueries {
                    return Err("a subquery outside WHERE and the subquery blocks");
                }
                if QuantifiedMembers::operator(&all_any.operator).is_none() {
                    return Err("an operator outside the family");
                }
                self.expression(&all_any.left, scopes, site.operand(), found)?;
                *found = true;
                self.block(&all_any.subquery, scopes + 1, width(&all_any.left))
            }
            Expression::Window(window) => {
                let inner = match site {
                    Site::Rows => Site::Subqueries,
                    Site::GroupedList => Site::GroupedOutput,
                    _ => return Err("a window function outside the SELECT list"),
                };
                if window.window_ref.is_some() {
                    return Err("a named window");
                }
                // NTILE's argument, the ones after a call's first and the frame
                // offsets are read once for the call: a subquery there runs once
                let call = &window.function;
                let inputs = usize::from(!call.function.eq_ignore_ascii_case("NTILE"));
                for (i, arg) in call.arguments.iter().enumerate() {
                    let at = if i < inputs { inner } else { Site::Subqueries };
                    if !matches!(arg, Expression::Star(_)) {
                        self.expression(arg, scopes, at, found)?;
                    }
                }
                let read = call
                    .filter
                    .as_deref()
                    .into_iter()
                    .chain(call.order_by.iter().map(|o| &o.expression))
                    .chain(&window.partition_by)
                    .chain(window.order_by.iter().map(|o| &o.expression));
                for expr in read {
                    self.expression(expr, scopes, inner, found)?;
                }
                let bounds = window
                    .frame
                    .iter()
                    .flat_map(|frame| std::iter::once(&frame.start).chain(&frame.end));
                for bound in bounds {
                    if let WindowFrameBound::Preceding(offset)
                    | WindowFrameBound::Following(offset) = bound
                    {
                        self.expression(offset, scopes, Site::Subqueries, found)?;
                    }
                }
                Ok(())
            }
            _ => Err("an expression form outside the family"),
        }
    }

    /// One subquery block whose chain holds `scopes` scopes, selecting the
    /// `width` values of the row it is compared with, one for a value
    fn block(
        &self,
        sq: &'a SelectStatement,
        scopes: usize,
        width: usize,
    ) -> std::result::Result<(), OutOfFamily> {
        if scopes > MAX_SCOPES {
            return Err("more than 32 scopes in a chain");
        }
        if !block_clauses(sq) {
            return Err("a subquery block with a clause outside the family");
        }
        if sq.columns.len() != width {
            return Err(match width {
                1 => "a subquery block without exactly one SELECT expression",
                _ => "a subquery block selecting another number of values than its row holds",
            });
        }
        let columns = || sq.columns.iter().map(unaliased);
        if columns().any(|c| matches!(c, Expression::Star(_) | Expression::QualifiedStar(_))) {
            return Err("a subquery block selecting *");
        }
        if windowed_block(sq) {
            return self.derived(sq);
        }
        self.order(sq, scopes, false)?;
        self.block_from(sq.table_expr.as_deref())?;
        let mut found = false;
        for column in columns() {
            self.expression(column, scopes, list_site(sq), &mut found)?;
        }
        self.grouping(sq, scopes)?;
        match sq.where_clause.as_deref() {
            Some(where_clause) => {
                self.expression(where_clause, scopes, Site::Subqueries, &mut found)
            }
            None => Ok(()),
        }
    }

    /// A block's DISTINCT ON, ORDER BY, LIMIT and OFFSET: its counts literals
    /// or parameters, its keys values over its row and the enclosing ones; a
    /// block that groups holds none of them
    fn order(
        &self,
        sq: &'a SelectStatement,
        scopes: usize,
        exists: bool,
    ) -> std::result::Result<(), OutOfFamily> {
        if sq.order_by.is_empty()
            && sq.distinct_on.is_empty()
            && sq.limit.is_none()
            && sq.offset.is_none()
        {
            return Ok(());
        }
        if grouped_block(sq) {
            return Err("a grouped subquery block with DISTINCT ON, ORDER BY, LIMIT or OFFSET");
        }
        let counts = [sq.limit.as_deref(), sq.offset.as_deref()];
        if counts.into_iter().flatten().any(|count| {
            !matches!(
                count,
                Expression::IntegerLiteral(_) | Expression::Parameter(_)
            )
        }) {
            return Err("a subquery block's LIMIT or OFFSET that is not a literal or a parameter");
        }
        // Keys that choose no row are bound but never computed
        let chooses =
            !exists && (sq.limit.is_some() || sq.offset.is_some() || !sq.distinct_on.is_empty());
        let volatile_aliases: SmallVec<[&str; 1]> = sq
            .columns
            .iter()
            .filter_map(|column| match column {
                Expression::Aliased(item)
                    if volatile(&item.expression, &self.executor.function_registry) =>
                {
                    Some(item.alias.value_lower.as_str())
                }
                _ => None,
            })
            .collect();
        let reads_volatile_alias =
            |key: &Expression| names_any(key, &|name| volatile_aliases.contains(&name));
        let mut found = false;
        for ob in &sq.order_by {
            let key = &ob.expression;
            // An EXISTS block checks its positions where it is bound
            if matches!(key, Expression::IntegerLiteral(_)) && !exists {
                return Err("an ORDER BY position in a subquery block");
            }
            if !chooses && mentions_subquery(key) {
                return Err("a subquery block's ORDER BY key holding a subquery without LIMIT");
            }
            if !order_by_names_an_alias(key, sq) && reads_volatile_alias(key) {
                return Err("an ORDER BY expression reading a volatile SELECT alias");
            }
            self.expression(key, scopes, Site::Values, &mut found)?;
        }
        for key in &sq.distinct_on {
            if matches!(key, Expression::IntegerLiteral(_)) {
                return Err("a DISTINCT ON position in a subquery block");
            }
            if mentions_subquery(key) {
                return Err("a subquery block's DISTINCT ON key holding a subquery");
            }
            if reads_volatile_alias(key) {
                return Err("a DISTINCT ON key reading a volatile SELECT alias");
            }
            self.expression(key, scopes, Site::Values, &mut found)?;
        }
        Ok(())
    }

    /// A grouped block's GROUP BY items read values; its HAVING aggregates
    fn grouping(
        &self,
        sq: &'a SelectStatement,
        scopes: usize,
    ) -> std::result::Result<(), OutOfFamily> {
        let mut found = false;
        for item in &sq.group_by.columns {
            self.expression(item, scopes, Site::Values, &mut found)?;
        }
        match sq.having.as_deref() {
            Some(having) => self.expression(having, scopes, Site::Grouped, &mut found),
            None => Ok(()),
        }
    }

    /// An aggregate call at `site`, and an argument after the first is a
    /// literal: what a block's call reads per row holds values, and what
    /// the statement's own call reads may hold subqueries too
    fn aggregate_call(
        &self,
        call: &'a FunctionCall,
        scopes: usize,
        site: Site,
        found: &mut bool,
    ) -> std::result::Result<(), OutOfFamily> {
        let inputs = match site {
            Site::GroupedOutput | Site::GroupedList => Site::Subqueries,
            _ => Site::Values,
        };
        let mut values = |e: &'a Expression| self.expression(e, scopes, inputs, found);
        let mut arguments = call.arguments.iter();
        match arguments.next() {
            None | Some(Expression::Star(_)) => {}
            Some(first) => values(first)?,
        }
        // The statement's own call reads one value for an argument after the
        // first, which a subquery there gives once; a block's call a literal
        let literal = |arg: &Expression| {
            extract_literal_value(arg).is_some()
                || matches!(arg, Expression::Identifier(id) if id.quoted)
        };
        if inputs == Site::Subqueries {
            arguments.try_for_each(&mut values)?;
        } else if !arguments.all(literal) {
            return Err("an aggregate argument after the first that is not a literal");
        }
        if let Some(filter) = call.filter.as_deref() {
            values(filter)?;
        }
        call.order_by.iter().try_for_each(|o| values(&o.expression))
    }

    /// The block of an EXISTS: its SELECT list is never evaluated, but its
    /// WHERE may read an alias of it, and an aggregate in it decides the
    /// row count, so the list holds the family's forms too
    fn existence_block(
        &self,
        sq: &'a SelectStatement,
        scopes: usize,
    ) -> std::result::Result<(), OutOfFamily> {
        if scopes > MAX_SCOPES {
            return Err("more than 32 scopes in a chain");
        }
        if !block_clauses(sq) {
            return Err("a subquery block with a clause outside the family");
        }
        if windowed_block(sq) {
            return self.derived(sq);
        }
        self.order(sq, scopes, true)?;
        self.block_from(sq.table_expr.as_deref())?;
        let star = |c: &Expression| matches!(c, Expression::Star(_) | Expression::QualifiedStar(_));
        // Only its run knows how wide a star over a join merging columns is
        if sq.table_expr.as_deref().is_some_and(merges_columns)
            && sq.columns.iter().any(star)
            && crate::parser::scope::order_position_outside(sq, Some(0)).is_some()
        {
            return Err("an EXISTS ordered by position in a star over a NATURAL or USING join");
        }
        let mut found = false;
        for column in &sq.columns {
            if !star(column) {
                self.expression(unaliased(column), scopes, list_site(sq), &mut found)?;
            }
        }
        self.grouping(sq, scopes)?;
        match sq.where_clause.as_deref() {
            Some(where_clause) => {
                self.expression(where_clause, scopes, Site::Subqueries, &mut found)
            }
            None => Ok(()),
        }
    }
}

/// Whether `from` holds a NATURAL or USING join, whose run merges columns
fn merges_columns(from: &Expression) -> bool {
    match from {
        Expression::JoinSource(join) => {
            join.join_type.to_uppercase().contains("NATURAL")
                || !join.using_columns.is_empty()
                || merges_columns(&join.left)
                || merges_columns(&join.right)
        }
        _ => false,
    }
}

/// What an expression of the family may hold where it stands
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Site {
    /// Subqueries: a WHERE, or the SELECT list of a block that does not group
    Subqueries,
    /// Neither subqueries nor aggregates
    Values,
    /// Aggregate calls: a grouped block's SELECT list and HAVING
    Grouped,
    /// A grouped statement's own SELECT list and HAVING: aggregate calls,
    /// and subqueries outside them
    GroupedOutput,
    /// The SELECT list of a statement that does not group: subqueries and
    /// window functions
    Rows,
    /// A grouped statement's own SELECT list: what `GroupedOutput` holds,
    /// and window functions
    GroupedList,
}

/// Whether a block aggregates its rows: it groups, has HAVING, or its
/// SELECT list calls an aggregate
impl Site {
    /// What the left of a set test over a block may hold: no window function,
    /// which the window path computes only outside one
    fn operand(self) -> Site {
        match self {
            Site::Rows => Site::Subqueries,
            Site::GroupedList => Site::GroupedOutput,
            other => other,
        }
    }
}

fn grouped_block(sq: &SelectStatement) -> bool {
    !sq.group_by.columns.is_empty()
        || matches!(sq.group_by.modifier, GroupByModifier::GroupingSets(_))
        || sq.having.is_some()
        || sq.columns.iter().any(expression_contains_aggregate)
}

/// A block whose SELECT list or ORDER BY computes a window function, which
/// runs as one statement for each call
fn windowed_block(sq: &SelectStatement) -> bool {
    sq.columns
        .iter()
        .chain(sq.order_by.iter().map(|ob| &ob.expression))
        .any(QueryClassification::expression_has_window_function)
}

/// What a block's SELECT list may hold
fn list_site(sq: &SelectStatement) -> Site {
    if grouped_block(sq) {
        Site::Grouped
    } else {
        Site::Subqueries
    }
}

/// `(a, b) op (SELECT x, y ..)`, either way round: the row, the block, and
/// the comparison with the row on its left
fn row_subquery(infix: &InfixExpression) -> Option<(&Expression, &ScalarSubquery, RowComparison)> {
    let comparison = match infix.op_type {
        InfixOperator::Equal => RowComparison::Eq,
        InfixOperator::NotEqual => RowComparison::Ne,
        InfixOperator::LessThan => RowComparison::Lt,
        InfixOperator::LessEqual => RowComparison::Le,
        InfixOperator::GreaterThan => RowComparison::Gt,
        InfixOperator::GreaterEqual => RowComparison::Ge,
        InfixOperator::IsDistinctFrom => RowComparison::Distinct,
        InfixOperator::IsNotDistinctFrom => RowComparison::NotDistinct,
        _ => return None,
    };
    match (&*infix.left, &*infix.right) {
        (row, Expression::ScalarSubquery(sq)) if width(row) > 1 => Some((row, sq, comparison)),
        (Expression::ScalarSubquery(sq), row) if width(row) > 1 => {
            Some((row, sq, comparison.flipped()))
        }
        _ => None,
    }
}

/// `infix` with its row and its block replaced, each on its side
fn row_against(
    infix: &InfixExpression,
    row: Expression,
    block: &ScalarSubquery,
    mark: Box<SelectStatement>,
) -> Expression {
    let block = Expression::ScalarSubquery(ScalarSubquery {
        token: block.token.clone(),
        subquery: mark,
    });
    let (left, right) = match &*infix.right {
        Expression::ScalarSubquery(_) => (row, block),
        _ => (block, row),
    };
    Expression::Infix(InfixExpression {
        token: infix.token.clone(),
        left: Box::new(left),
        operator: infix.operator.clone(),
        op_type: infix.op_type,
        right: Box::new(right),
    })
}

/// The values `expr` holds: a row's, `(a, b)`, else one
fn width(expr: &Expression) -> usize {
    match expr {
        Expression::List(list) => list.elements.len().max(1),
        Expression::ExpressionList(list) => list.expressions.len().max(1),
        _ => 1,
    }
}

/// A statement of the family's clauses: its WITH runs before it, as its own
/// statements, and its DISTINCT, DISTINCT ON and set operation over the rows
/// it gives, as the general path runs them; no grouping, HAVING or WINDOW
fn plain_statement(stmt: &SelectStatement) -> bool {
    matches!(stmt.group_by.modifier, GroupByModifier::None)
        && stmt.window_defs.is_empty()
        && stmt.group_by.columns.is_empty()
        && stmt.having.is_none()
}

/// No WITH, set operation, DISTINCT, grouping set or WINDOW
fn block_clauses(stmt: &SelectStatement) -> bool {
    stmt.with.is_none() && statement_clauses(stmt)
}

/// No set operation, DISTINCT, grouping set or WINDOW; a DISTINCT ON is not
/// a DISTINCT
fn statement_clauses(stmt: &SelectStatement) -> bool {
    stmt.set_operations.is_empty()
        && (!stmt.distinct || !stmt.distinct_on.is_empty())
        && matches!(stmt.group_by.modifier, GroupByModifier::None)
        && stmt.window_defs.is_empty()
}

fn unaliased(expr: &Expression) -> &Expression {
    match expr {
        Expression::Aliased(aliased) => &aliased.expression,
        other => other,
    }
}

impl Executor {
    /// Whether `stmt` belongs to the prepared family, or why not. Decided
    /// from its syntax and the catalog only, before anything is bound
    pub(crate) fn prepared_family(
        &self,
        stmt: &SelectStatement,
        ctx: Option<&ExecutionContext>,
    ) -> std::result::Result<(), OutOfFamily> {
        if ctx.is_some_and(|ctx| ctx.outer_row().is_some()) {
            return Err("it runs for a row of an enclosing statement");
        }
        if !plain_statement(stmt) {
            return Err("a clause outside the family");
        }
        let family = Family {
            executor: self,
            blocks: RefCell::new(SmallVec::new()),
            with: stmt.with.as_ref(),
        };
        let mut found = false;
        if let Some(where_clause) = stmt.where_clause.as_deref() {
            family.expression(where_clause, 1, Site::Subqueries, &mut found)?;
        }
        for column in &stmt.columns {
            if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
                family.expression(unaliased(column), 1, Site::Subqueries, &mut found)?;
            }
        }
        // A set operation's ORDER BY is the whole one's, over its rows
        if !stmt.set_operations.is_empty() {
            if stmt
                .order_by
                .iter()
                .any(|ob| mentions_subquery(&ob.expression))
            {
                return Err("an ORDER BY subquery over a set operation");
            }
            // Its DISTINCT ON reads the first branch's items over every row
            if !stmt.distinct_on.is_empty() {
                return Err("a DISTINCT ON over a set operation");
            }
        } else {
            // A DISTINCT ON key is computed for each row as an ORDER BY key is
            for key in stmt
                .order_by
                .iter()
                .map(|ob| &ob.expression)
                .chain(&stmt.distinct_on)
            {
                family.expression(key, 1, Site::Subqueries, &mut found)?;
            }
        }
        if !found {
            return Err("no WHERE, SELECT list, ORDER BY or DISTINCT ON subquery");
        }
        let aliases: SmallVec<[&str; 4]> = stmt
            .columns
            .iter()
            .filter_map(|c| match c {
                Expression::Aliased(a) => Some(a.alias.value_lower.as_str()),
                _ => None,
            })
            .collect();
        let names_an_alias = |key: &Expression| {
            mentions_subquery(key) && names_any_within(key, &|name| aliases.contains(&name))
        };
        if stmt
            .order_by
            .iter()
            .any(|ob| names_an_alias(&ob.expression))
        {
            return Err("an ORDER BY subquery that names a SELECT alias");
        }
        if stmt.distinct_on.iter().any(names_an_alias) {
            return Err("a DISTINCT ON subquery that names a SELECT alias");
        }
        counts_fit(stmt, ctx)?;
        family.statement_from(stmt.table_expr.as_deref())?;
        family.catalog()
    }

    /// Whether `stmt`, as `classification` says, is a statement the grouped
    /// family may take, which `grouped_family` then decides
    pub(crate) fn grouped_family_reach(
        stmt: &SelectStatement,
        classification: &QueryClassification,
    ) -> bool {
        let rows_where = classification.where_has_subqueries
            && !matches!(stmt.table_expr.as_deref(), Some(Expression::JoinSource(_)));
        let grouped_keys = classification.has_aggregation
            && stmt
                .order_by
                .iter()
                .any(|ob| Self::has_subqueries(&ob.expression));
        let distinct_on = stmt.distinct_on.iter().any(Self::has_subqueries);
        (classification.has_aggregation || classification.has_window_functions)
            && (classification.grouped_has_subqueries || rows_where || grouped_keys || distinct_on)
    }

    /// Whether `stmt` groups its rows or has window functions, and holds
    /// subqueries in its SELECT list, HAVING, GROUP BY or DISTINCT ON, in the
    /// ORDER BY of one that groups without windows, or in the WHERE of one
    /// that does not join, all of the family's forms, or why not
    pub(crate) fn grouped_family(
        &self,
        stmt: &SelectStatement,
        ctx: Option<&ExecutionContext>,
    ) -> std::result::Result<(), OutOfFamily> {
        if ctx.is_some_and(|ctx| ctx.outer_row().is_some()) {
            return Err("it runs for a row of an enclosing statement");
        }
        if !stmt.window_defs.is_empty() {
            return Err("a named window");
        }
        // A set operation's branch carries no DISTINCT ON key as a column
        if !stmt.set_operations.is_empty() && stmt.distinct_on.iter().any(mentions_subquery) {
            return Err("a DISTINCT ON subquery in a set operation's branch");
        }
        let grouped = grouped_block(stmt);
        // A grouped statement's DISTINCT ON key runs as an item, its windows
        // too, and an ORDER BY window as a hidden item the window path adds
        let windowed = stmt
            .columns
            .iter()
            .chain(stmt.distinct_on.iter().filter(|_| grouped))
            .chain(stmt.order_by.iter().map(|ob| &ob.expression))
            .any(QueryClassification::expression_has_window_function);
        if !grouped && !windowed {
            return Err("a statement that neither groups nor has window functions");
        }
        // The windows run over the groups when the statement aggregates first
        if windowed && grouped != get_classification(stmt).has_aggregation {
            return Err("windows over a grouping the aggregation does not run");
        }
        // A set operation's ORDER BY, LIMIT and OFFSET are the whole one's,
        // read over its rows once the blocks gave the first branch's
        let whole = stmt.set_operations.is_empty();
        // A grouped key holding a subquery is computed for each group, where
        // a SELECT alias names no column
        let keys: SmallVec<[&Expression; 2]> = stmt
            .order_by
            .iter()
            .map(|ob| &ob.expression)
            .filter(|key| whole && mentions_subquery(key))
            .collect();
        if !keys.is_empty() && windowed {
            return Err("an ORDER BY holding a subquery");
        }
        let aliases: SmallVec<[&str; 4]> = stmt
            .columns
            .iter()
            .filter_map(|c| match c {
                Expression::Aliased(a) => Some(a.alias.value_lower.as_str()),
                _ => None,
            })
            .collect();
        if keys
            .iter()
            .any(|key| names_anywhere(key, &|name| aliases.contains(&name)))
        {
            return Err("an ORDER BY subquery key naming a SELECT alias");
        }
        // A GROUP BY item naming a SELECT alias or position groups that item
        self.parse_group_by(stmt, &[])
            .map_err(|_| "a GROUP BY position outside the SELECT list")?;
        let family = Family {
            executor: self,
            blocks: RefCell::new(SmallVec::new()),
            with: stmt.with.as_ref(),
        };
        let mut found = false;
        let list = if grouped {
            Site::GroupedList
        } else {
            Site::Rows
        };
        for column in &stmt.columns {
            if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
                family.expression(unaliased(column), 1, list, &mut found)?;
            }
        }
        if let Some(having) = stmt.having.as_deref() {
            family.expression(having, 1, Site::GroupedOutput, &mut found)?;
        }
        for key in keys {
            family.expression(key, 1, Site::GroupedOutput, &mut found)?;
        }
        // A DISTINCT ON key is computed as a SELECT item of the statement is
        for key in &stmt.distinct_on {
            family.expression(key, 1, list, &mut found)?;
        }
        for item in grouping_items(stmt) {
            family.expression(item, 1, Site::Subqueries, &mut found)?;
        }
        // The WHERE runs as the statement's blocks too, unless a join's path
        // asks it
        let joined = matches!(stmt.table_expr.as_deref(), Some(Expression::JoinSource(_)));
        if !joined {
            found |= stmt.where_clause.as_deref().is_some_and(mentions_subquery);
        }
        if !found {
            return Err("no clause the blocks run holds a subquery");
        }
        if whole {
            counts_fit(stmt, ctx)?;
        }
        // A join's rows pass its ON and WHERE as the join path asks them
        match stmt.table_expr.as_deref() {
            Some(Expression::JoinSource(join)) => {
                let side_merges = |side: &Expression| {
                    matches!(side, Expression::JoinSource(_)) && merges_columns(side)
                };
                if join.join_type.to_uppercase().contains("NATURAL")
                    || !join.using_columns.is_empty()
                    || side_merges(&join.left)
                    || side_merges(&join.right)
                {
                    return Err("a NATURAL or USING join whose output holds a subquery");
                }
            }
            table_expr => {
                if let Some(where_clause) = stmt.where_clause.as_deref() {
                    family.expression(where_clause, 1, Site::Subqueries, &mut found)?;
                }
                family.statement_from(table_expr)?;
            }
        }
        family.catalog()
    }
}

/// Whether the LIMIT and OFFSET are literals or parameters, or counts read
/// before the statement's rows
fn counts_fit(
    stmt: &SelectStatement,
    ctx: Option<&ExecutionContext>,
) -> std::result::Result<(), OutOfFamily> {
    for (count, clause) in [(&stmt.limit, "LIMIT"), (&stmt.offset, "OFFSET")] {
        let Some(count) = count else {
            continue;
        };
        if !matches!(
            **count,
            Expression::IntegerLiteral(_) | Expression::Parameter(_)
        ) && !ctx.is_some_and(|ctx| ctx.read_count(count, clause).is_some())
        {
            return Err("LIMIT or OFFSET that is not a literal or a parameter");
        }
    }
    Ok(())
}

/// One scope of a chain: the tables it reads, each by the name that
/// qualifies it and the position of its first column, and their columns
/// in that order; a join's statement scope holds its two sides
#[derive(Debug)]
struct ScopeTable {
    tables: SmallVec<[(SmartString, usize); 2]>,
    /// The names its programs compile against
    columns: CompactArc<Vec<String>>,
    /// Each column's own name, lowercase, as a bare reference names it
    lower: CompactArc<Vec<String>>,
    types: Vec<DataType>,
    /// The first columns, which a bare name may bind; past them a column is
    /// read only through its table's name
    bare: usize,
    /// The columns a NATURAL or USING join merges, each a left side's column
    /// and the right side's it is one column with under a bare name
    merged: SmallVec<[(usize, usize); 1]>,
}

impl ScopeTable {
    fn of(name: SmartString, schema: &Schema) -> Self {
        Self {
            tables: smallvec::smallvec![(name, 0)],
            columns: schema.column_names_arc(),
            lower: schema.column_names_lower_arc(),
            types: schema.columns.iter().map(|c| c.data_type).collect(),
            bare: schema.columns.len(),
            merged: SmallVec::new(),
        }
    }

    /// The scope of a join of `left` and `right`: their tables side by side,
    /// the right's columns after the left's
    fn beside(left: Self, right: Self) -> Self {
        let offset = left.columns.len();
        let mut tables = left.tables;
        tables.extend(
            right
                .tables
                .into_iter()
                .map(|(name, at)| (name, at + offset)),
        );
        let joined = |left: &[String], right: &[String]| {
            CompactArc::new(left.iter().chain(right).cloned().collect::<Vec<_>>())
        };
        let mut merged = left.merged;
        merged.extend(
            right
                .merged
                .iter()
                .map(|(left, right)| (left + offset, right + offset)),
        );
        Self {
            tables,
            columns: joined(&left.columns, &right.columns),
            lower: joined(&left.lower, &right.lower),
            types: left.types.into_iter().chain(right.types).collect(),
            bare: offset + right.lower.len(),
            merged,
        }
    }

    /// The scope of `join` of `left` and `right`, each name its USING list or
    /// a NATURAL join merges one column: the left side's first column of
    /// that name with the right side's first
    fn joined(join: &JoinTableSource, left: Self, right: Self) -> Self {
        let (left_names, right_names) = (&left.lower[..left.bare], &right.lower[..right.bare]);
        let natural = join.join_type.to_uppercase().contains("NATURAL");
        let names = join
            .using_columns
            .iter()
            .map(|c| c.value_lower.as_str())
            .chain(left_names.iter().map(String::as_str).filter(|_| natural));
        let mut merged: SmallVec<[(usize, usize); 1]> = SmallVec::new();
        for name in names {
            let at = |names: &[String]| names.iter().position(|c| c == name);
            if let (Some(l), Some(r)) = (at(left_names), at(right_names)) {
                let pair = (l, left.lower.len() + r);
                if !merged.contains(&pair) {
                    merged.push(pair);
                }
            }
        }
        let mut scope = Self::beside(left, right);
        scope.merged.extend(merged);
        scope
    }

    /// The scope of the FROM entry `side`, its tables `leaves` in turn and
    /// each join among them merging columns as [`Self::joined`] does
    fn of_side(side: &Expression, leaves: &mut impl Iterator<Item = Self>) -> Option<Self> {
        match side {
            Expression::JoinSource(join) => {
                let left = Self::of_side(&join.left, leaves)?;
                let right = Self::of_side(&join.right, leaves)?;
                Some(Self::joined(join, left, right))
            }
            _ => leaves.next(),
        }
    }

    /// This scope read by position, each column under a name no column can
    /// have: a name two of its tables share binds the one its qualifier
    /// names, never the first, and no name a table chose binds it elsewhere
    fn by_position(&self) -> Self {
        let width = self.lower.len();
        Self {
            tables: self.tables.clone(),
            columns: CompactArc::new((0..width).map(|i| format!("{RESOLVED}#{i}")).collect()),
            lower: CompactArc::clone(&self.lower),
            types: self.types.clone(),
            bare: self.bare,
            merged: self.merged.clone(),
        }
    }

    /// Whether more than one of its tables has a column named `name`
    fn shared(&self, name: &str) -> bool {
        let mut tables = self.tables.iter().enumerate().filter(|(at, _)| {
            let (_, range) = self.table(self.tables[*at].1);
            self.lower[range].iter().any(|c| c == name)
        });
        tables.next().is_some() && tables.next().is_some()
    }

    /// Column `at` as a reference spells it: qualified by its table when
    /// the scope has more than one
    fn spelled(&self, at: usize) -> String {
        let table = self.tables.iter().rev().find(|(_, start)| *start <= at);
        match table {
            Some((table, _)) if self.tables.len() > 1 => format!("{table}.{}", self.lower[at]),
            _ => self.lower[at].clone(),
        }
    }

    /// The scope of a block without FROM: no columns, and no name a
    /// qualifier could find, so every name reads an enclosing scope
    fn unnamed() -> Self {
        Self {
            tables: SmallVec::new(),
            columns: CompactArc::new(Vec::new()),
            lower: CompactArc::new(Vec::new()),
            types: Vec::new(),
            bare: 0,
            merged: SmallVec::new(),
        }
    }

    /// The joined row a join path lays out as `columns` from its `leaves`,
    /// the first `sides.0` of them read as its left side's `sides.1` columns:
    /// the one split, in leaf order, where each column spells its leaf's
    /// name, a dot and its own name, and a leaf has its known names and width.
    /// Read by position as `joined`'s are; None when no split or several fit
    fn of_join_columns(
        columns: &[String],
        leaves: &[JoinLeaf],
        sides: (usize, usize),
    ) -> Option<Self> {
        let (n, m) = (columns.len(), leaves.len());
        if m < 2 || sides.0 == 0 || sides.0 >= m || sides.1 > n {
            return None;
        }
        let spelled: Vec<String> = columns.iter().map(|c| c.to_lowercase()).collect();
        // The own name column `at` spells as the `offset`-th of `leaf`
        let own = |leaf: &JoinLeaf, offset: usize, at: usize| -> Option<&str> {
            let name = spelled[at]
                .strip_prefix(leaf.name.as_str())?
                .strip_prefix(QUALIFIER_SEP)?;
            match &leaf.columns {
                Some(names) => names.get(offset).filter(|n| n.to_lowercase() == name)?,
                None => &spelled[at],
            };
            Some(name)
        };
        // How many splits give the first `e` columns to the first `i` leaves,
        // 2 for several, and where the last of them starts
        let mut ways = vec![vec![0u8; n + 1]; m + 1];
        let mut starts = vec![vec![0usize; n + 1]; m + 1];
        ways[0][0] = 1;
        for (i, leaf) in leaves.iter().enumerate() {
            // The right side starts where the left side's columns, as read, end
            if i == sides.0 {
                for (end, reached) in ways[i].iter_mut().enumerate() {
                    if end != sides.1 {
                        *reached = 0;
                    }
                }
            }
            for start in 0..n {
                let reached = ways[i][start];
                if reached == 0 {
                    continue;
                }
                let width = leaf.width;
                let mut end = start;
                while end < n && width.is_none_or(|w| end - start < w) {
                    if own(leaf, end - start, end).is_none() {
                        break;
                    }
                    end += 1;
                    if width.is_none_or(|w| end - start == w) {
                        ways[i + 1][end] = (ways[i + 1][end] + reached).min(2);
                        starts[i + 1][end] = start;
                    }
                }
            }
        }
        if ways[m][n] != 1 {
            return None;
        }
        let mut bounds = vec![n; m + 1];
        for i in (0..m).rev() {
            bounds[i] = starts[i + 1][bounds[i + 1]];
        }
        let mut tables: SmallVec<[(SmartString, usize); 2]> = SmallVec::new();
        let mut lower = Vec::with_capacity(n);
        for (i, leaf) in leaves.iter().enumerate() {
            tables.push((SmartString::from(leaf.name.as_str()), bounds[i]));
            for at in bounds[i]..bounds[i + 1] {
                lower.push(own(leaf, at - bounds[i], at)?.to_string());
            }
        }
        Some(Self {
            tables,
            columns: CompactArc::new(
                (0..columns.len())
                    .map(|i| format!("{RESOLVED}#{i}"))
                    .collect(),
            ),
            lower: CompactArc::new(lower),
            types: vec![DataType::Null; columns.len()],
            bare: columns.len(),
            merged: SmallVec::new(),
        })
    }

    /// The name qualifying the table whose columns include `index`, and the
    /// columns of that table
    fn table(&self, index: usize) -> (&SmartString, std::ops::Range<usize>) {
        let at = self
            .tables
            .iter()
            .rposition(|(_, start)| *start <= index)
            .unwrap_or(0);
        let end = self
            .tables
            .get(at + 1)
            .map_or(self.lower.len(), |(_, start)| *start);
        (&self.tables[at].0, self.tables[at].1..end)
    }

    /// An error when a column in `rest` is also named `name`: a reference to
    /// a name two columns have is ambiguous, in one table or in two
    fn unique(
        &self,
        rest: impl IntoIterator<Item = usize>,
        name: &str,
        spelled: &dyn std::fmt::Display,
    ) -> Result<()> {
        match rest.into_iter().any(|i| self.lower[i] == name) {
            true => Err(Error::invalid_argument(format!(
                "column reference '{}' is ambiguous",
                spelled
            ))),
            false => Ok(()),
        }
    }
}

/// What an expression reads, one bit per scope of the chain: bit 0 the
/// block's own table, bit 1 the enclosing scope, and so on
type Reads = u32;

/// What the walk over a statement a block runs whole meets: a name bound
/// beyond the block, with the depth of its scope in the block's chain and
/// its column there, or a parameter
enum Reach<'x> {
    Outer(&'x Expression, usize, usize),
    Param(&'x Expression),
}

/// One statement level of a [`StatementWalk`]: its scope, then those it
/// sees, the first `inside` of them inside the block, and its SELECT
/// aliases; `check` raises a name no scope has, for a clause nothing runs
#[derive(Clone)]
struct Level<'s> {
    scopes: Vec<&'s ScopeTable>,
    inside: usize,
    aliases: Vec<SmartString>,
    check: bool,
}

/// Which clauses of a walked statement raise a name no scope has, since
/// the statement runs without them: none, its ORDER BY, or all of them
#[derive(Clone, Copy, PartialEq)]
enum Check {
    Off,
    Order,
    All,
}

/// A statement a block runs whole, its names bound as a block's are: the
/// nearest scope with that column, the statement's own SELECT aliases
/// among its own. A name bound beyond the block goes to `on`, which gives
/// what stands in its place
struct StatementWalk<'w> {
    executor: &'w Executor,
    ctx: &'w ExecutionContext,
    on: &'w mut dyn FnMut(Reach<'_>) -> Result<Option<Expression>>,
}

impl StatementWalk<'_> {
    /// Each subquery of `expr`, walked over the scopes `enclosing`
    fn subqueries_of(&mut self, expr: &Expression, enclosing: &[&ScopeTable]) -> Result<()> {
        let block = |walk: &mut Self, sq: &SelectStatement| {
            walk.statement(sq, enclosing, 0, Check::All).map(drop)
        };
        match expr {
            Expression::ScalarSubquery(sq) => block(self, &sq.subquery),
            Expression::Exists(exists) => block(self, &exists.subquery),
            Expression::AllAny(all_any) => {
                self.subqueries_of(&all_any.left, enclosing)?;
                block(self, &all_any.subquery)
            }
            other => for_each_child(other, &mut |child| self.subqueries_of(child, enclosing)),
        }
    }

    /// `stmt` over the scopes `enclosing`, nearest first, the first `inside`
    /// of them inside the block, with the names its rows have, as
    /// `output_names` gives them, and the scope its FROM gives it
    #[allow(clippy::type_complexity)]
    fn statement(
        &mut self,
        stmt: &SelectStatement,
        enclosing: &[&ScopeTable],
        inside: usize,
        check: Check,
    ) -> Result<(SelectStatement, Option<Vec<String>>, ScopeTable)> {
        let (table_expr, own) = self.from(stmt.table_expr.as_deref(), enclosing, inside, check)?;
        let mut scopes = Vec::with_capacity(enclosing.len() + 1);
        scopes.push(&own);
        scopes.extend_from_slice(enclosing);
        let aliases = stmt
            .columns
            .iter()
            .filter_map(|item| match item {
                Expression::Aliased(aliased) => Some(aliased.alias.value_lower.clone()),
                _ => None,
            })
            .collect();
        // A SELECT item reads no alias of its own list; the clauses after it
        // do. Its names are bound even where the statement runs without it
        let items = Level {
            scopes: scopes.clone(),
            inside: inside + 1,
            aliases: Vec::new(),
            check: check != Check::Off,
        };
        let level = Level {
            scopes,
            inside: inside + 1,
            aliases,
            check: check == Check::All,
        };
        let sorted = Level {
            check: check != Check::Off,
            ..level.clone()
        };
        let names = output_names(stmt, &own)?;
        // Its ORDER BY positions count the columns it shows, whether or not
        // it ever runs
        if let Some(position) =
            crate::parser::scope::order_position_outside(stmt, names.as_ref().map(Vec::len))
        {
            return Err(Error::invalid_argument(
                crate::parser::scope::order_position_message(position.value),
            ));
        }
        let all = |walk: &mut Self, exprs: &[Expression]| {
            exprs
                .iter()
                .map(|e| walk.expr(e, &level))
                .collect::<Result<Vec<_>>>()
        };
        let one = |walk: &mut Self, expr: &Option<Box<Expression>>| {
            expr.as_deref()
                .map(|e| walk.expr(e, &level).map(Box::new))
                .transpose()
        };
        let order = |walk: &mut Self, order: &[OrderByExpression], at: &Level<'_>| {
            order
                .iter()
                .map(|o| {
                    Ok(OrderByExpression {
                        expression: walk.expr(&o.expression, at)?,
                        ..o.clone()
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        let group_by = match &stmt.group_by.modifier {
            GroupByModifier::GroupingSets(sets) => GroupByClause::grouping_sets(
                sets.iter()
                    .map(|set| all(self, set))
                    .collect::<Result<_>>()?,
            ),
            modifier => GroupByClause {
                columns: all(self, &stmt.group_by.columns)?,
                modifier: modifier.clone(),
            },
        };
        let walked = SelectStatement {
            distinct_on: all(self, &stmt.distinct_on)?,
            columns: stmt
                .columns
                .iter()
                .map(|e| self.item(e, &items))
                .collect::<Result<_>>()?,
            table_expr: table_expr.map(Box::new),
            where_clause: one(self, &stmt.where_clause)?,
            group_by,
            having: one(self, &stmt.having)?,
            window_defs: stmt
                .window_defs
                .iter()
                .map(|def| {
                    Ok(WindowDefinition {
                        partition_by: all(self, &def.partition_by)?,
                        order_by: order(self, &def.order_by, &level)?,
                        frame: def
                            .frame
                            .as_ref()
                            .map(|frame| self.frame(frame, &level))
                            .transpose()?,
                        name: def.name.clone(),
                    })
                })
                .collect::<Result<_>>()?,
            order_by: order(self, &stmt.order_by, &sorted)?,
            limit: one(self, &stmt.limit)?,
            offset: one(self, &stmt.offset)?,
            set_operations: stmt
                .set_operations
                .iter()
                .map(|op| {
                    let (right, ..) = self.statement(&op.right, enclosing, inside, check)?;
                    Ok(SetOperation {
                        operation: op.operation.clone(),
                        right: Box::new(right),
                    })
                })
                .collect::<Result<_>>()?,
            ..stmt.clone()
        };
        Ok((walked, names, own))
    }

    /// A FROM walked over the scopes it sees, `enclosing` and not its
    /// siblings, and the scope it gives its statement
    fn from(
        &mut self,
        table_expr: Option<&Expression>,
        enclosing: &[&ScopeTable],
        inside: usize,
        check: Check,
    ) -> Result<(Option<Expression>, ScopeTable)> {
        // A FROM runs with its statement unless no clause of it does
        let check = match check {
            Check::All => Check::All,
            _ => Check::Off,
        };
        let beside = Level {
            scopes: enclosing.to_vec(),
            inside,
            aliases: Vec::new(),
            check: check == Check::All,
        };
        let untyped = |name: &SmartString, columns: &[String]| {
            ScopeTable::of(name.clone(), &untyped_schema(name, columns))
        };
        let Some(table_expr) = table_expr else {
            return Ok((None, ScopeTable::unnamed()));
        };
        let scope = match table_expr {
            Expression::TableSource(ts) => {
                let name = ts.alias.as_ref().unwrap_or(&ts.name).value_lower.clone();
                let lower = &ts.name.value_lower;
                if let Some((columns, _)) = self.ctx.get_cte_by_lower(lower) {
                    untyped(&name, columns)
                } else if let Some(view) = self.executor.engine.get_view_lowercase(lower)? {
                    untyped(&name, &view_names(self.executor, &view, self.ctx)?)
                } else {
                    let schema =
                        self.executor
                            .engine
                            .get_table_schema(lower)
                            .map_err(|e| match e {
                                Error::TableNotFound(_) => {
                                    Error::TableOrViewNotFound(ts.name.value.to_string())
                                }
                                other => other,
                            })?;
                    ScopeTable::of(name, &schema)
                }
            }
            Expression::ValuesSource(values) => {
                let rows = values
                    .rows
                    .iter()
                    .map(|row| row.iter().map(|e| self.expr(e, &beside)).collect())
                    .collect::<Result<_>>()?;
                let name = values.alias.as_ref().map(|a| a.value_lower.clone());
                let scope = untyped(&name.unwrap_or_default(), &Executor::values_columns(values));
                let walked = ValuesTableSource {
                    rows,
                    ..(**values).clone()
                };
                return Ok((Some(Expression::ValuesSource(Box::new(walked))), scope));
            }
            Expression::FunctionTableSource(function) => {
                let arguments = function
                    .arguments
                    .iter()
                    .map(|e| self.expr(e, &beside))
                    .collect::<Result<_>>()?;
                let name = &function
                    .alias
                    .as_ref()
                    .unwrap_or(&function.function)
                    .value_lower;
                let scope = untyped(name, &Executor::tvf_columns(function)?);
                let walked = FunctionTableSource {
                    arguments,
                    ..(**function).clone()
                };
                return Ok((
                    Some(Expression::FunctionTableSource(Box::new(walked))),
                    scope,
                ));
            }
            Expression::SubquerySource(derived) => {
                let (subquery, names, _) =
                    self.statement(&derived.subquery, enclosing, inside, check)?;
                let name = derived.alias.as_ref().map(|a| a.value_lower.clone());
                let scope = untyped(&name.unwrap_or_default(), &named(names)?);
                let walked = SubqueryTableSource {
                    subquery: Box::new(subquery),
                    ..(**derived).clone()
                };
                return Ok((Some(Expression::SubquerySource(Box::new(walked))), scope));
            }
            // Each side sees what the join does, not its sibling; the ON
            // sees both, as the statement's other clauses do
            Expression::JoinSource(join) => {
                let (left, left_scope) = self.from(Some(&join.left), enclosing, inside, check)?;
                let (right, right_scope) =
                    self.from(Some(&join.right), enclosing, inside, check)?;
                let (Some(left), Some(right)) = (left, right) else {
                    return Err(Error::internal("a join side without a source"));
                };
                let scope = ScopeTable::joined(join, left_scope, right_scope);
                let condition = match join.condition.as_deref() {
                    Some(on) => {
                        let mut scopes = Vec::with_capacity(enclosing.len() + 1);
                        scopes.push(&scope);
                        scopes.extend_from_slice(enclosing);
                        let level = Level {
                            scopes,
                            inside: inside + 1,
                            aliases: Vec::new(),
                            check: check == Check::All,
                        };
                        Some(Box::new(self.expr(on, &level)?))
                    }
                    None => None,
                };
                let walked = JoinTableSource {
                    left: Box::new(left),
                    right: Box::new(right),
                    condition,
                    ..(**join).clone()
                };
                return Ok((Some(Expression::JoinSource(Box::new(walked))), scope));
            }
            _ => {
                return Err(Error::internal(
                    "a derived table's FROM the family does not read",
                ))
            }
        };
        Ok((Some(table_expr.clone()), scope))
    }

    /// `expr` at `level`, each name bound beyond the block replaced
    fn expr(&mut self, expr: &Expression, level: &Level<'_>) -> Result<Expression> {
        let chain = Chain {
            scopes: &level.scopes,
        };
        let beyond = |depth: usize| depth.checked_sub(level.inside).map(|d| d + 1);
        let nested = |exists: bool| match (level.check, exists) {
            (false, _) => Check::Off,
            (true, true) => Check::Order,
            (true, false) => Check::All,
        };
        match expr {
            Expression::Identifier(id) => {
                if level.aliases.contains(&id.value_lower) {
                    return Ok(expr.clone());
                }
                match chain.bare(id) {
                    Ok(Some((depth, index))) => match beyond(depth) {
                        Some(depth) => self.outer(expr, depth, index),
                        None => Ok(expr.clone()),
                    },
                    Err(e) if level.check => Err(e),
                    _ => Ok(expr.clone()),
                }
            }
            Expression::QualifiedIdentifier(q) => {
                let qualifier = &q.qualifier.value_lower;
                let named = level
                    .scopes
                    .iter()
                    .position(|scope| scope.tables.iter().any(|(name, _)| name == qualifier));
                match named.and_then(beyond) {
                    Some(_) => {
                        let (depth, index) = chain.qualified(qualifier, &q.name.value_lower, q)?;
                        match beyond(depth) {
                            Some(depth) => self.outer(expr, depth, index),
                            None => Ok(expr.clone()),
                        }
                    }
                    None => {
                        if level.check {
                            chain.qualified(qualifier, &q.name.value_lower, q)?;
                        }
                        Ok(expr.clone())
                    }
                }
            }
            Expression::Parameter(_) => {
                (self.on)(Reach::Param(expr))?;
                Ok(expr.clone())
            }
            Expression::ScalarSubquery(sq) => Ok(Expression::ScalarSubquery(ScalarSubquery {
                token: sq.token.clone(),
                subquery: Box::new(
                    self.statement(&sq.subquery, &level.scopes, level.inside, nested(false))?
                        .0,
                ),
            })),
            Expression::Exists(exists) => Ok(Expression::Exists(ExistsExpression {
                token: exists.token.clone(),
                subquery: Box::new(
                    self.statement(&exists.subquery, &level.scopes, level.inside, nested(true))?
                        .0,
                ),
            })),
            Expression::AllAny(all_any) => Ok(Expression::AllAny(AllAnyExpression {
                left: Box::new(self.expr(&all_any.left, level)?),
                subquery: Box::new(
                    self.statement(
                        &all_any.subquery,
                        &level.scopes,
                        level.inside,
                        nested(false),
                    )?
                    .0,
                ),
                ..all_any.clone()
            })),
            Expression::Aliased(aliased) => Ok(Expression::Aliased(AliasedExpression {
                expression: Box::new(self.expr(&aliased.expression, level)?),
                ..aliased.clone()
            })),
            Expression::Distinct(distinct) => Ok(Expression::Distinct(DistinctExpression {
                token: distinct.token.clone(),
                expr: Box::new(self.expr(&distinct.expr, level)?),
            })),
            other => rebuild(other, &mut |child| self.expr(child, level)),
        }
    }

    /// What stands for the enclosing name `expr`: a slot spelled as `expr` is,
    /// so every name the statement derives from it is the one written
    fn outer(&mut self, expr: &Expression, depth: usize, index: usize) -> Result<Expression> {
        let stands = (self.on)(Reach::Outer(expr, depth, index))?
            .ok_or_else(|| Error::internal("an enclosing name without its slot"))?;
        Ok(spelled_as(stands, expr.to_string()))
    }

    /// The SELECT item `item`, walked; an enclosing column that is the whole
    /// item stands as a slot under the column's own name, as the item's name
    fn item(&mut self, item: &Expression, level: &Level<'_>) -> Result<Expression> {
        let walked = self.expr(item, level)?;
        Ok(match item {
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
                spelled_as(walked, item_name(item))
            }
            _ => walked,
        })
    }

    /// The frame of a named window, its offsets walked
    fn frame(&mut self, frame: &WindowFrame, level: &Level<'_>) -> Result<WindowFrame> {
        let mut bound = |bound: &WindowFrameBound| -> Result<WindowFrameBound> {
            Ok(match bound {
                WindowFrameBound::Preceding(offset) => {
                    WindowFrameBound::Preceding(Box::new(self.expr(offset, level)?))
                }
                WindowFrameBound::Following(offset) => {
                    WindowFrameBound::Following(Box::new(self.expr(offset, level)?))
                }
                other => other.clone(),
            })
        };
        Ok(WindowFrame {
            unit: frame.unit,
            start: bound(&frame.start)?,
            end: frame.end.as_ref().map(&mut bound).transpose()?,
        })
    }
}

/// `stands`, when it is a parameter, spelled `name`; a parameter is read by
/// its index, so its spelling only names it
fn spelled_as(stands: Expression, name: String) -> Expression {
    match stands {
        Expression::Parameter(parameter) => Expression::Parameter(Parameter {
            name: SmartString::from_string(name),
            ..parameter
        }),
        other => other,
    }
}

/// The names `stmt`'s rows have over its FROM's scope `own`: a star's
/// columns, every other item by `item_name`; None for a star over a join,
/// whose names only its run gives
fn output_names(stmt: &SelectStatement, own: &ScopeTable) -> Result<Option<Vec<String>>> {
    let mut names = Vec::with_capacity(stmt.columns.len());
    for item in &stmt.columns {
        match item {
            Expression::Star(_) if own.tables.len() > 1 => return Ok(None),
            Expression::Star(_) => names.extend(own.columns.iter().cloned()),
            Expression::QualifiedStar(star) => {
                let qualifier = star.qualifier.to_lowercase();
                let (_, start) = own
                    .tables
                    .iter()
                    .find(|(name, _)| **name == *qualifier)
                    .ok_or_else(|| Error::invalid_argument(unknown_qualifier(&star.qualifier)))?;
                let (_, range) = own.table(*start);
                names.extend(own.columns[range].iter().cloned());
            }
            _ => names.push(item_name(item)),
        }
    }
    Ok(Some(names))
}

/// The names a walked statement's rows have, or the error of a star over
/// a join, which a source named before its run cannot carry
fn named(names: Option<Vec<String>>) -> Result<Vec<String>> {
    names.ok_or_else(|| Error::internal("a star over a join, named only by its run"))
}

/// The names of `view`'s columns, recorded when it was created; a view
/// recorded without them runs once here, which records them
fn view_names(
    executor: &Executor,
    view: &ViewDefinition,
    ctx: &ExecutionContext,
) -> Result<Vec<String>> {
    if let Some(columns) = view.columns() {
        return Ok(columns.to_vec());
    }
    let mut result = executor.execute_view_body(view, ctx)?;
    let columns = result.columns().to_vec();
    result.close()?;
    Ok(columns)
}

/// A reference resolved to a scope of the chain and a column of its table
type Resolved = Option<(usize, usize)>;

/// A block's scope chain, nearest first
struct Chain<'a> {
    scopes: &'a [&'a ScopeTable],
}

/// An identifier spelled `value`, matched as `lower`
fn ident(value: &str, lower: &str) -> Identifier {
    let mut id = Identifier::new(
        Token::new(TokenType::Identifier, value, Position::default()),
        value,
    );
    id.value_lower = SmartString::from(lower);
    id
}

impl Chain<'_> {
    /// `id` by the name rules: the nearest scope with that column, or None
    /// for a name the compiler reads as a literal
    fn bare(&self, id: &Identifier) -> Result<Resolved> {
        if !id.quoted && matches!(id.value_lower.as_str(), "true" | "false") {
            return Ok(None);
        }
        // A name is a column's whole name, dots and all, before a qualifier's
        for (depth, scope) in self.scopes.iter().enumerate() {
            let bare = &scope.lower[..scope.bare];
            if let Some(index) = bare.iter().position(|c| *c == id.value_lower) {
                let rest =
                    (index + 1..scope.bare).filter(|&at| !scope.merged.contains(&(index, at)));
                scope.unique(rest, &id.value_lower, &id.value)?;
                return Ok(Some((depth, index)));
            }
        }
        if let Some((qualifier, name)) = split_qualified(&id.value_lower) {
            let resolved = self.qualified(qualifier, name, &id.value);
            return match resolved {
                Err(_) if id.quoted => Ok(None),
                other => other.map(Some),
            };
        }
        if id.quoted {
            return Ok(None);
        }
        Err(Error::ColumnNotFound(id.value.to_string()))
    }

    /// `qualifier.column` in the table of the nearest scope the qualifier names
    fn qualified(
        &self,
        qualifier: &str,
        column: &str,
        spelled: &dyn std::fmt::Display,
    ) -> Result<(usize, usize)> {
        let (depth, start) = self
            .scopes
            .iter()
            .enumerate()
            .find_map(|(depth, scope)| {
                scope
                    .tables
                    .iter()
                    .find(|(name, _)| name == qualifier)
                    .map(|(_, start)| (depth, *start))
            })
            .ok_or_else(|| Error::invalid_argument(unknown_qualifier(qualifier)))?;
        let scope = self.scopes[depth];
        let (_, mut range) = scope.table(start);
        let index = range
            .find(|i| scope.lower[*i] == column)
            .ok_or_else(|| Error::ColumnNotFound(spelled.to_string()))?;
        scope.unique(range, column, spelled)?;
        Ok((depth, index))
    }

    fn resolve(&self, expr: &Expression) -> Result<Resolved> {
        match expr {
            Expression::Identifier(id) => self.bare(id),
            Expression::QualifiedIdentifier(q) => self
                .qualified(&q.qualifier.value_lower, &q.name.value_lower, q)
                .map(Some),
            _ => Ok(None),
        }
    }

    /// The column `index` of the scope at `depth`, bound for the programs
    /// that read that scope's row
    fn column(&self, depth: usize, index: usize) -> Expression {
        let name = self.scopes[depth].lower[index].as_str();
        Expression::Column(ColumnRef {
            index: index as u32,
            name: ident(name, name),
        })
    }

    /// The type of the column `expr` is, in whichever scope it binds to
    fn column_type(&self, expr: &Expression) -> Result<Option<DataType>> {
        Ok(self
            .resolve(expr)?
            .map(|(depth, index)| self.scopes[depth].types[index])
            .filter(|t| *t != DataType::Null))
    }

    /// What `expr` reads; a scalar subquery counts as reading this row
    fn reads(&self, expr: &Expression) -> Result<Reads> {
        match expr {
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
                Ok(self.resolve(expr)?.map_or(0, |(depth, _)| 1 << depth))
            }
            Expression::ScalarSubquery(_) | Expression::Exists(_) => Ok(1),
            Expression::AllAny(all_any) => Ok(1 | self.reads(&all_any.left)?),
            other => {
                let mut reads = 0;
                for_each_child(other, &mut |child| {
                    reads |= self.reads(child)?;
                    Ok(())
                })?;
                Ok(reads)
            }
        }
    }

    /// `expr`, read wholly from the scope at `depth`, with each reference
    /// spelled as that scope's own column
    fn in_scope(&self, expr: &Expression) -> Result<Expression> {
        match self.resolve(expr)? {
            Some((depth, index)) => Ok(self.column(depth, index)),
            None => rebuild(expr, &mut |child| self.in_scope(child)),
        }
    }
}

/// Whether an argument a call in `expr` reads once rather than for each row
/// passes `matches`: an aggregate's after its first, a window function's
/// after its first (each of NTILE's), and a frame offset
fn constant_args_any(expr: &Expression, matches: &impl Fn(&Expression) -> bool) -> bool {
    let within = |input: &Expression| constant_args_any(input, matches);
    match expr {
        Expression::ScalarSubquery(_) | Expression::Exists(_) => false,
        Expression::Window(window) => {
            let call = &window.function;
            let first = usize::from(!call.function.eq_ignore_ascii_case("NTILE"));
            call.arguments
                .iter()
                .skip(first)
                .chain(window.frame_offsets())
                .any(matches)
                || call
                    .arguments
                    .iter()
                    .take(first)
                    .chain(call.filter.as_deref())
                    .chain(call.order_by.iter().map(|o| &o.expression))
                    .chain(&window.partition_by)
                    .chain(window.order_by.iter().map(|o| &o.expression))
                    .any(within)
        }
        Expression::FunctionCall(call) if is_aggregate_function(&call.function) => {
            call.arguments.iter().skip(1).any(matches)
                || call
                    .arguments
                    .iter()
                    .take(1)
                    .chain(call.filter.as_deref())
                    .chain(call.order_by.iter().map(|o| &o.expression))
                    .any(within)
        }
        other => {
            let mut found = false;
            let _ = for_each_child(other, &mut |child| {
                found |= within(child);
                Ok(())
            });
            found
        }
    }
}

/// Each direct operand of an expression of the family's forms
fn for_each_child<'e>(
    expr: &'e Expression,
    f: &mut impl FnMut(&'e Expression) -> Result<()>,
) -> Result<()> {
    match expr {
        Expression::Infix(infix) => {
            f(&infix.left)?;
            f(&infix.right)
        }
        Expression::Prefix(prefix) => f(&prefix.right),
        Expression::Between(between) => {
            f(&between.expr)?;
            f(&between.lower)?;
            f(&between.upper)
        }
        Expression::In(in_expr) => {
            f(&in_expr.left)?;
            match &*in_expr.right {
                Expression::List(list) => list.elements.iter().try_for_each(f),
                Expression::ExpressionList(list) => list.expressions.iter().try_for_each(f),
                other => f(other),
            }
        }
        Expression::Like(like) => {
            f(&like.left)?;
            f(&like.pattern)?;
            like.escape.as_deref().map_or(Ok(()), f)
        }
        Expression::Case(case) => {
            if let Some(value) = case.value.as_deref() {
                f(value)?;
            }
            for when in &case.when_clauses {
                f(&when.condition)?;
                f(&when.then_result)?;
            }
            case.else_value.as_deref().map_or(Ok(()), f)
        }
        Expression::Cast(cast) => f(&cast.expr),
        Expression::FunctionCall(call) => {
            call.arguments.iter().try_for_each(&mut *f)?;
            if let Some(filter) = call.filter.as_deref() {
                f(filter)?;
            }
            call.order_by.iter().try_for_each(|o| f(&o.expression))
        }
        Expression::AllAny(all_any) => f(&all_any.left),
        Expression::List(list) => list.elements.iter().try_for_each(f),
        Expression::ExpressionList(list) => list.expressions.iter().try_for_each(f),
        Expression::Window(window) => {
            let call = &window.function;
            call.arguments.iter().try_for_each(&mut *f)?;
            if let Some(filter) = call.filter.as_deref() {
                f(filter)?;
            }
            call.order_by.iter().try_for_each(|o| f(&o.expression))?;
            window.partition_by.iter().try_for_each(&mut *f)?;
            window.order_by.iter().try_for_each(|o| f(&o.expression))?;
            window.frame_offsets().try_for_each(f)
        }
        _ => Ok(()),
    }
}

/// `expr` with each direct operand replaced by what `f` makes of it
fn rebuild(
    expr: &Expression,
    f: &mut impl FnMut(&Expression) -> Result<Expression>,
) -> Result<Expression> {
    fn boxed(
        f: &mut impl FnMut(&Expression) -> Result<Expression>,
        e: &Expression,
    ) -> Result<Box<Expression>> {
        f(e).map(Box::new)
    }
    Ok(match expr {
        Expression::Infix(infix) => Expression::Infix(InfixExpression {
            token: infix.token.clone(),
            left: boxed(f, &infix.left)?,
            operator: infix.operator.clone(),
            op_type: infix.op_type,
            right: boxed(f, &infix.right)?,
        }),
        Expression::Prefix(prefix) => Expression::Prefix(PrefixExpression {
            token: prefix.token.clone(),
            operator: prefix.operator.clone(),
            op_type: prefix.op_type,
            right: boxed(f, &prefix.right)?,
        }),
        Expression::Between(between) => Expression::Between(BetweenExpression {
            token: between.token.clone(),
            expr: boxed(f, &between.expr)?,
            lower: boxed(f, &between.lower)?,
            upper: boxed(f, &between.upper)?,
            not: between.not,
        }),
        Expression::In(in_expr) => {
            let left = boxed(f, &in_expr.left)?;
            let right = match &*in_expr.right {
                Expression::List(list) => Expression::List(Box::new(ListExpression {
                    token: list.token.clone(),
                    elements: list.elements.iter().map(&mut *f).collect::<Result<_>>()?,
                })),
                Expression::ExpressionList(list) => {
                    Expression::ExpressionList(Box::new(ExpressionList {
                        token: list.token.clone(),
                        expressions: list
                            .expressions
                            .iter()
                            .map(&mut *f)
                            .collect::<Result<_>>()?,
                    }))
                }
                other => f(other)?,
            };
            Expression::In(InExpression {
                token: in_expr.token.clone(),
                left,
                right: Box::new(right),
                not: in_expr.not,
            })
        }
        Expression::Like(like) => Expression::Like(LikeExpression {
            token: like.token.clone(),
            left: boxed(f, &like.left)?,
            pattern: boxed(f, &like.pattern)?,
            operator: like.operator.clone(),
            escape: like.escape.as_deref().map(|e| boxed(f, e)).transpose()?,
        }),
        Expression::Case(case) => Expression::Case(Box::new(CaseExpression {
            token: case.token.clone(),
            value: case.value.as_deref().map(|e| boxed(f, e)).transpose()?,
            when_clauses: case
                .when_clauses
                .iter()
                .map(|when| {
                    Ok(WhenClause {
                        token: when.token.clone(),
                        condition: f(&when.condition)?,
                        then_result: f(&when.then_result)?,
                    })
                })
                .collect::<Result<_>>()?,
            else_value: case
                .else_value
                .as_deref()
                .map(|e| boxed(f, e))
                .transpose()?,
        })),
        Expression::Cast(cast) => Expression::Cast(CastExpression {
            token: cast.token.clone(),
            expr: boxed(f, &cast.expr)?,
            type_name: cast.type_name.clone(),
        }),
        Expression::FunctionCall(call) => Expression::FunctionCall(Box::new(FunctionCall {
            token: call.token.clone(),
            function: call.function.clone(),
            arguments: call.arguments.iter().map(&mut *f).collect::<Result<_>>()?,
            is_distinct: call.is_distinct,
            order_by: call
                .order_by
                .iter()
                .map(|o| {
                    Ok(OrderByExpression {
                        expression: f(&o.expression)?,
                        ascending: o.ascending,
                        nulls_first: o.nulls_first,
                    })
                })
                .collect::<Result<_>>()?,
            filter: call.filter.as_deref().map(|e| boxed(f, e)).transpose()?,
        })),
        Expression::AllAny(all_any) => Expression::AllAny(AllAnyExpression {
            token: all_any.token.clone(),
            left: boxed(f, &all_any.left)?,
            operator: all_any.operator.clone(),
            all_any_type: all_any.all_any_type,
            subquery: all_any.subquery.clone(),
        }),
        Expression::List(list) => Expression::List(Box::new(ListExpression {
            token: list.token.clone(),
            elements: list.elements.iter().map(&mut *f).collect::<Result<_>>()?,
        })),
        Expression::ExpressionList(list) => Expression::ExpressionList(Box::new(ExpressionList {
            token: list.token.clone(),
            expressions: list
                .expressions
                .iter()
                .map(&mut *f)
                .collect::<Result<_>>()?,
        })),
        Expression::Window(window) => {
            let mut order_by = |order: &[OrderByExpression]| {
                order
                    .iter()
                    .map(|o| {
                        Ok(OrderByExpression {
                            expression: f(&o.expression)?,
                            ascending: o.ascending,
                            nulls_first: o.nulls_first,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            };
            let call = &window.function;
            let call_order_by = order_by(&call.order_by)?;
            let window_order_by = order_by(&window.order_by)?;
            Expression::Window(Box::new(WindowExpression {
                token: window.token.clone(),
                function: Box::new(FunctionCall {
                    token: call.token.clone(),
                    function: call.function.clone(),
                    arguments: call.arguments.iter().map(&mut *f).collect::<Result<_>>()?,
                    is_distinct: call.is_distinct,
                    order_by: call_order_by,
                    filter: call.filter.as_deref().map(|e| boxed(f, e)).transpose()?,
                }),
                window_ref: window.window_ref.clone(),
                partition_by: window
                    .partition_by
                    .iter()
                    .map(&mut *f)
                    .collect::<Result<_>>()?,
                order_by: window_order_by,
                frame: window
                    .frame
                    .as_ref()
                    .map(|frame| {
                        let mut bound = |bound: &WindowFrameBound| -> Result<WindowFrameBound> {
                            Ok(match bound {
                                WindowFrameBound::Preceding(offset) => {
                                    WindowFrameBound::Preceding(boxed(f, offset)?)
                                }
                                WindowFrameBound::Following(offset) => {
                                    WindowFrameBound::Following(boxed(f, offset)?)
                                }
                                other => other.clone(),
                            })
                        };
                        Ok::<_, Error>(WindowFrame {
                            unit: frame.unit,
                            start: bound(&frame.start)?,
                            end: frame.end.as_ref().map(&mut bound).transpose()?,
                        })
                    })
                    .transpose()?,
            }))
        }
        other => other.clone(),
    })
}

/// `expr`, bound over a side alone, with each of its columns read from
/// the joined row, where the side's first is at `offset`
fn in_joined(expr: &Expression, offset: usize) -> Result<Expression> {
    match expr {
        Expression::Column(c) => Ok(Expression::Column(ColumnRef {
            index: (offset + c.index as usize) as u32,
            name: c.name.clone(),
        })),
        _ => rebuild(expr, &mut |child| in_joined(child, offset)),
    }
}

/// `expr` with each name of the chain's own scope bound to its column; a
/// name `keep` takes, or one the compiler reads as a literal, stays
fn bind_own(
    chain: &Chain<'_>,
    expr: &Expression,
    keep: &dyn Fn(&Identifier) -> bool,
) -> Result<Expression> {
    match expr {
        Expression::Identifier(id) if keep(id) => Ok(expr.clone()),
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
            match chain.resolve(expr)? {
                Some((0, index)) => Ok(chain.column(0, index)),
                Some(_) => Err(Error::internal("an enclosing column read by a statement")),
                None => Ok(expr.clone()),
            }
        }
        _ => rebuild(expr, &mut |child| bind_own(chain, child, keep)),
    }
}

/// `expr` with each column bound over `scope` named again, quoted as its
/// column is, for a statement the general path runs
fn spelled_columns(expr: &Expression, scope: &ScopeTable) -> Result<Expression> {
    match expr {
        Expression::Column(c) => {
            let name = scope.columns[c.index as usize].as_str();
            let mut id = ident(name, &scope.lower[c.index as usize]);
            id.quoted = true;
            Ok(Expression::Identifier(id))
        }
        _ => rebuild(expr, &mut |child| spelled_columns(child, scope)),
    }
}

/// The column of its scope's row a name the builder bound reads
fn bound_column(expr: &Expression) -> Option<usize> {
    match expr {
        Expression::Column(c) => Some(c.index as usize),
        _ => None,
    }
}

/// Each bare name in `key` of an alias in `positions`, by node address, to
/// that alias's position on the projected row
fn bind_aliases(
    key: &Expression,
    positions: &FxHashMap<&str, u16>,
    out: &mut FxHashMap<usize, u16>,
) {
    match key {
        Expression::Identifier(id) => {
            if let Some(&i) = positions.get(id.value_lower.as_str()) {
                out.insert(id as *const Identifier as usize, i);
            }
        }
        other => {
            let _ = for_each_child(other, &mut |child| {
                bind_aliases(child, positions, out);
                Ok(())
            });
        }
    }
}

/// Whether every column `expr` reads is one the scan compares
fn scans_whole(expr: &Expression, schema: &Schema) -> bool {
    match expr {
        Expression::Column(c) => schema
            .columns
            .get(c.index as usize)
            .is_none_or(|column| scan_compares(column.data_type)),
        other => {
            let mut whole = true;
            let _ = for_each_child(other, &mut |child| {
                whole &= scans_whole(child, schema);
                Ok(())
            });
            whole
        }
    }
}

/// Whether `expr` reads a column of the row its program runs over; a
/// subquery's value is a value, whatever its own block reads
fn reads_row_column(expr: &Expression) -> bool {
    match expr {
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) | Expression::Column(_) => {
            true
        }
        other => {
            let mut reads = false;
            let _ = for_each_child(other, &mut |child| {
                reads |= reads_row_column(child);
                Ok(())
            });
            reads
        }
    }
}

/// Every value a WHERE program compares with a column of `schema`, by node
/// address, to the column's type: the program reads it as the scan reads a
/// pushed comparison's value, so both give one answer
fn compare_as_scan(expr: &Expression, schema: &Schema, out: &mut FxHashMap<usize, DataType>) {
    let column_type = |e: &Expression| {
        bound_column(e)
            .and_then(|i| schema.columns.get(i))
            .map(|column| column.data_type)
            .filter(|t| *t != DataType::Null)
    };
    let mut read_as = |value: &Expression, t: DataType| {
        // A literal of the column's own type, or NULL, already reads as the scan's
        let as_is = extract_literal_value(value).is_some_and(|v| v.is_null() || v.data_type() == t);
        if !as_is && !reads_row_column(value) {
            out.insert(value as *const Expression as usize, t);
        }
    };
    match expr {
        Expression::Infix(infix)
            if is_comparison(infix.op_type)
                || matches!(
                    infix.op_type,
                    InfixOperator::IsDistinctFrom | InfixOperator::IsNotDistinctFrom
                ) =>
        {
            match (column_type(&infix.left), column_type(&infix.right)) {
                (Some(t), None) => read_as(&infix.right, t),
                (None, Some(t)) => read_as(&infix.left, t),
                _ => {}
            }
        }
        Expression::Between(between) => {
            if let Some(t) = column_type(&between.expr) {
                read_as(&between.lower, t);
                read_as(&between.upper, t);
            }
        }
        Expression::In(in_expr) => {
            if let Some(t) = column_type(&in_expr.left) {
                let elements = match &*in_expr.right {
                    Expression::List(list) => &list.elements[..],
                    Expression::ExpressionList(list) => &list.expressions[..],
                    _ => &[],
                };
                for element in elements {
                    read_as(element, t);
                }
            }
        }
        _ => {}
    }
    let _ = for_each_child(expr, &mut |child| {
        compare_as_scan(child, schema, out);
        Ok(())
    });
}

/// Whether a call anywhere in `expr` draws a new value each time it runs
fn volatile(expr: &Expression, registry: &FunctionRegistry) -> bool {
    expression_volatility(expr, registry) == Volatility::PerCall
}

fn is_leaf(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::Column(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::Parameter(_)
    )
}

fn is_comparison(op: InfixOperator) -> bool {
    matches!(
        op,
        InfixOperator::Equal
            | InfixOperator::NotEqual
            | InfixOperator::LessThan
            | InfixOperator::LessEqual
            | InfixOperator::GreaterThan
            | InfixOperator::GreaterEqual
    )
}

/// The top-level AND chain of a WHERE, in order
fn conjuncts<'e>(expr: &'e Expression, out: &mut Vec<&'e Expression>) {
    match expr {
        Expression::Infix(infix) if infix.op_type == InfixOperator::And => {
            conjuncts(&infix.left, out);
            conjuncts(&infix.right, out);
        }
        other => out.push(other),
    }
}

/// `parts` joined by AND, or None when there are none
fn conjunction(parts: Vec<Expression>) -> Option<Expression> {
    parts.into_iter().reduce(|left, right| {
        Expression::Infix(InfixExpression::new(
            Token::new(TokenType::Keyword, "AND", Position::default()),
            Box::new(left),
            "AND",
            Box::new(right),
        ))
    })
}

/// The condition every row passes
fn true_literal() -> Expression {
    Expression::BooleanLiteral(BooleanLiteral {
        token: Token::new(TokenType::Keyword, "TRUE", Position::default()),
        value: true,
    })
}

fn slot_parameter(slot: usize) -> Expression {
    let name = format!("${}", slot + 1);
    Expression::Parameter(Parameter {
        token: Token::new(TokenType::Parameter, &name, Position::default()),
        name: SmartString::from_string(name),
        index: slot + 1,
    })
}

/// The EXISTS block `stmt`, run as a statement, asked only whether it has
/// a row: one value per row it returns in place of its SELECT items, each
/// alias or position that read an item reading its expression, no ORDER
/// BY. What it drops is still bound, as the general path binds it
fn existence(
    executor: &Executor,
    stmt: SelectStatement,
    from_names: &[String],
    ctx: &ExecutionContext,
) -> Result<SelectStatement> {
    let grouped = !stmt.group_by.columns.is_empty()
        || stmt.having.is_some()
        || stmt.columns.iter().any(expression_contains_aggregate);
    if grouped {
        check_grouped_reads(executor, &stmt, from_names, ctx)?;
    }
    let where_clause = stmt
        .where_clause
        .as_deref()
        .map(|w| Box::new(with_aliases(w, &stmt.columns, from_names).into_owned()));
    let group_by = GroupByClause {
        columns: group_by_expressions(executor, &stmt, from_names)?,
        modifier: stmt.group_by.modifier.clone(),
    };
    let having = stmt
        .having
        .as_deref()
        .map(|h| grouped_having(executor, &stmt, h).map(Box::new))
        .transpose()?;
    let token = |kind, text: &str| Token::new(kind, text, Position::default());
    let row = match grouped {
        true => Expression::FunctionCall(Box::new(FunctionCall {
            token: token(TokenType::Identifier, "COUNT"),
            function: SmartString::from("COUNT"),
            arguments: vec![Expression::Star(StarExpression {
                token: token(TokenType::Operator, "*"),
            })],
            is_distinct: false,
            order_by: Vec::new(),
            filter: None,
        })),
        false => Expression::IntegerLiteral(IntegerLiteral {
            token: token(TokenType::Integer, "1"),
            value: 1,
        }),
    };
    Ok(SelectStatement {
        columns: vec![row],
        where_clause,
        group_by,
        having,
        order_by: Vec::new(),
        ..stmt
    })
}

/// `sq`'s GROUP BY items as the general path binds a SELECT alias or a
/// position, over the columns `columns` of its FROM
fn group_by_expressions(
    executor: &Executor,
    sq: &SelectStatement,
    columns: &[String],
) -> Result<Vec<Expression>> {
    let bound = executor.parse_group_by(sq, columns)?;
    let mut items = Vec::with_capacity(bound.len());
    for (item, written) in bound.iter().zip(&sq.group_by.columns) {
        let expr = match (item, written) {
            (GroupByItem::Expression { expr, .. }, _) => expr,
            (GroupByItem::Column(_), Expression::IntegerLiteral(position)) => {
                let at = usize::try_from(position.value - 1).ok();
                at.and_then(|at| sq.columns.get(at))
                    .map(unaliased)
                    .ok_or_else(|| Error::internal("a GROUP BY position not bound"))?
            }
            (GroupByItem::Column(_), written) => written,
        };
        items.push(expr.clone());
    }
    Ok(items)
}

/// What a block answers as written, the signature of its form, and
/// whether a keyed set may answer it
fn block_identity(sq: &SelectStatement, form: Form<'_>) -> (Expression, String, bool) {
    let as_value = |block: &SelectStatement| {
        Expression::ScalarSubquery(ScalarSubquery {
            token: block.token.clone(),
            subquery: Box::new(block.clone()),
        })
    };
    match form {
        Form::Scalar => (as_value(sq), "value".to_string(), false),
        Form::Exists(exists) => (Expression::Exists(exists.clone()), String::new(), true),
        Form::Members {
            block,
            read_as,
            test,
        } => (
            as_value(block),
            format!("{read_as:?}{test:?}"),
            test == MemberTest::In,
        ),
    }
}

/// What the block answers and how its members are read, then the names,
/// columns and types of the scopes `outer`: part of what is kept
fn scope_signature(signature: String, outer: &[&ScopeTable]) -> Arc<str> {
    use std::fmt::Write as _;
    let mut scopes = signature;
    for scope in outer {
        for (name, _) in &scope.tables {
            scopes.push_str(name);
            scopes.push('\u{2}');
        }
        for (column, data_type) in scope.lower.iter().zip(&scope.types) {
            // Writing to a String cannot fail
            let _ = write!(scopes, "\u{0}{column}:{data_type:?}");
        }
        scopes.push('\u{1}');
    }
    Arc::from(scopes)
}

fn compile_error(error: CompileError) -> Error {
    Error::internal(format!("Compile error: {}", error))
}

/// That each ORDER BY position of the block `sq`, whose FROM gives `columns`
/// columns, names a column its list shows: a star shows them all, as a
/// block's FROM is one source
fn check_block_positions(sq: &SelectStatement, columns: usize) -> Result<()> {
    let width = sq
        .columns
        .iter()
        .map(|item| match item {
            Expression::Star(_) | Expression::QualifiedStar(_) => columns,
            _ => 1,
        })
        .sum();
    match crate::parser::scope::order_position_outside(sq, Some(width)) {
        Some(position) => Err(Error::invalid_argument(
            crate::parser::scope::order_position_message(position.value),
        )),
        None => Ok(()),
    }
}

/// A block's WHERE with its SELECT aliases read as their expressions, as
/// the general path reads them: not inside subqueries, and not where a
/// column of the block's table has the alias's name
fn with_aliases<'e>(
    where_clause: &'e Expression,
    select: &[Expression],
    columns: &[String],
) -> std::borrow::Cow<'e, Expression> {
    let alias_map = Executor::build_alias_map_excluding(select, Some(columns));
    if alias_map.is_empty() || !names_any(where_clause, &|name| alias_map.contains_key(name)) {
        return std::borrow::Cow::Borrowed(where_clause);
    }
    std::borrow::Cow::Owned(Executor::substitute_aliases(where_clause, &alias_map))
}

/// `having` of the grouped `stmt`, each bare name outside its subqueries
/// that the grouped path reads as a SELECT alias spelled as that item
fn grouped_having(
    executor: &Executor,
    stmt: &SelectStatement,
    having: &Expression,
) -> Result<Expression> {
    let (aggregations, _) = executor.parse_aggregations(stmt, None)?;
    let items = executor.parse_group_by(stmt, &[])?;
    let slots = aggregate_slot_keys(&items, &aggregations, &stmt.columns);
    let aliased = |name: &str| {
        let slot = slots.get(&ExprKey::column(name))?;
        stmt.columns.iter().find_map(|column| match column {
            Expression::Aliased(aliased)
                if aliased.alias.value_lower == name
                    && slots.get(&ExprKey::of(&aliased.expression)) == Some(slot) =>
            {
                Some(&*aliased.expression)
            }
            _ => None,
        })
    };
    // A group item or aggregate is read from its slot whole, and an
    // aggregate's arguments read the input row, as the grouped compile reads them
    let bound = |expr: &Expression| match expr {
        Expression::FunctionCall(call) if is_aggregate_function(&call.function) => true,
        _ => slots.contains_key(&ExprKey::of(expr)),
    };
    fn open<'s>(
        expr: &Expression,
        aliased: &impl Fn(&str) -> Option<&'s Expression>,
        bound: &impl Fn(&Expression) -> bool,
    ) -> Result<Expression> {
        match expr {
            Expression::Identifier(id) => Ok(aliased(&id.value_lower)
                .cloned()
                .unwrap_or_else(|| expr.clone())),
            _ if bound(expr) => Ok(expr.clone()),
            other => rebuild(other, &mut |child| open(child, aliased, bound)),
        }
    }
    open(having, &aliased, &bound)
}

/// `expr` of a grouped statement with each GROUP BY item `items` holds, by
/// its identity as written, read as its rewritten form, and each aggregate
/// call holding a subquery rewritten by `call`, outside its subqueries
fn read_inputs(
    expr: &Expression,
    items: &FxHashMap<ExprKey, Expression>,
    call: &mut impl FnMut(&Expression) -> Result<Expression>,
) -> Result<Expression> {
    if let Some(item) = items.get(&ExprKey::of(expr)) {
        return Ok(item.clone());
    }
    match expr {
        Expression::FunctionCall(f) if is_aggregate_function(&f.function) => {
            match mentions_subquery(expr) {
                true => call(expr),
                false => Ok(expr.clone()),
            }
        }
        other => rebuild(other, &mut |child| read_inputs(child, items, call)),
    }
}

/// Whether a bare name in `expr` or in a clause the family admits of one
/// of its subqueries passes `matches`
pub(super) fn names_any_within(expr: &Expression, matches: &impl Fn(&str) -> bool) -> bool {
    let block = |sq: &SelectStatement| {
        sq.columns
            .iter()
            .chain(sq.where_clause.as_deref())
            .chain(sq.having.as_deref())
            .chain(&sq.group_by.columns)
            .chain(sq.order_by.iter().map(|ob| &ob.expression))
            .any(|e| names_any_within(e, matches))
    };
    match expr {
        Expression::Identifier(id) => matches(&id.value_lower),
        Expression::Aliased(aliased) => names_any_within(&aliased.expression, matches),
        Expression::ScalarSubquery(sq) => block(&sq.subquery),
        Expression::Exists(exists) => block(&exists.subquery),
        Expression::AllAny(all_any) => {
            names_any_within(&all_any.left, matches) || block(&all_any.subquery)
        }
        other => {
            let mut found = false;
            let _ = for_each_child(other, &mut |child| {
                found |= names_any_within(child, matches);
                Ok(())
            });
            found
        }
    }
}

/// Whether a bare name outside any subquery of `expr` passes `matches`
pub(super) fn names_any(expr: &Expression, matches: &impl Fn(&str) -> bool) -> bool {
    match expr {
        Expression::Identifier(id) => matches(&id.value_lower),
        other => {
            let mut found = false;
            let _ = for_each_child(other, &mut |child| {
                found |= names_any(child, matches);
                Ok(())
            });
            found
        }
    }
}

fn compile_with(ctx: &CompileContext<'_>, expr: &Expression) -> Result<SharedProgram> {
    ExprCompiler::new(ctx)
        .compile(expr)
        .map(CompactArc::new)
        .map_err(compile_error)
}

/// The subquery blocks of one statement and the slots they read, which a
/// `PreparedRunner` runs for the statement's rows
#[derive(Debug)]
struct PreparedBlocks {
    /// Positional parameters the statement takes; slots are numbered after them
    param_count: usize,
    /// The statement's parameters and every slot
    slot_count: usize,
    /// Values computed once per run before the statement reads a row
    run_slots: Vec<(usize, SharedProgram)>,
    /// Indexed by subquery id, nested ones included
    subqueries: Vec<PreparedSubquery>,
}

/// A statement's WHERE over its table's row: its program, what of it a scan
/// takes, and the rows a set test's keys may narrow it to
#[derive(Debug)]
struct PreparedWhere {
    whole: WhereProgram,
    /// What a row the scan read is still asked when the scan took
    /// `pushdown` whole: the conjuncts outside it
    scanned: Option<WhereProgram>,
    /// What a row the narrowing named is still asked when the narrowing's
    /// keys raised no error: the WHERE without the narrowing's conjunct
    narrowed: Option<WhereProgram>,
    /// `narrowed` without the conjuncts `pushdown` holds, for a named row
    /// the scan's filter passed when it took `pushdown` whole
    narrowed_scanned: Option<WhereProgram>,
    /// The conjuncts without a subquery, which the scan is narrowed by
    pushdown: Option<Expression>,
    /// `pushdown` prepared, for a run to give its values
    push: Option<PreparedPush>,
    /// A top-level EXISTS or NOT EXISTS conjunct with one key, whose keyed
    /// set's keys name the rows the WHERE may pass
    narrowing: Option<Narrowing>,
    /// The top-level conjuncts testing a column against an uncorrelated
    /// block's members, which the scan takes once the run read them
    sets: Vec<SetSlot>,
    /// The narrowing's own set conjunct, the scan's when it declines
    narrowing_set: Option<SetSlot>,
    /// How many conjuncts `pushdown` holds, of the WHERE's `conjuncts`
    pushed: usize,
    conjuncts: usize,
}

/// A program over the statement's row, with whether a run binds it: it
/// loads a parameter or asks an uncorrelated block, and binding can
/// shrink it
#[derive(Debug)]
struct WhereProgram {
    program: SharedProgram,
    binds: bool,
    /// Which of the WHERE's programs it is, the run's bound copy's place
    rest: WhereRest,
}

impl WhereProgram {
    fn of(program: SharedProgram, rest: WhereRest, blocks: &PreparedBlocks) -> Self {
        let asks_blocks = program.ops().iter().any(|op| match *op {
            Op::ExecScalarSubquery(id)
            | Op::ExecExists(id)
            | Op::ExecInSubquery(id)
            | Op::ExecQuantified(id, ..) => blocks
                .subqueries
                .get(id as usize)
                .is_some_and(|sq| sq.uncorrelated),
            _ => false,
        });
        let loads = asks_blocks
            || program
                .ops()
                .iter()
                .any(|op| matches!(op, Op::LoadParam(_)));
        let binds = loads && (asks_blocks || !program.bind_never_shrinks());
        Self {
            program,
            binds,
            rest,
        }
    }
}

/// Which of a prepared WHERE's programs a row is asked
#[derive(Debug, Clone, Copy)]
enum WhereRest {
    Whole,
    Scanned,
    Narrowed,
    NarrowedScanned,
}

impl PreparedWhere {
    /// The program `rest` names, the whole WHERE when there is none
    fn rest(&self, rest: WhereRest) -> &WhereProgram {
        let program = match rest {
            WhereRest::Whole => None,
            WhereRest::Scanned => self.scanned.as_ref(),
            WhereRest::Narrowed => self.narrowed.as_ref(),
            WhereRest::NarrowedScanned => self.narrowed_scanned.as_ref(),
        };
        program.unwrap_or(&self.whole)
    }
}

/// `column [NOT] IN (block)` or `column op ANY|ALL (block)` at the top of
/// a WHERE, over an uncorrelated block
#[derive(Debug, Clone, Copy)]
struct SetSlot {
    id: u16,
    column: usize,
    negated: bool,
}

/// The filter a scan of a prepared WHERE takes, whether it is the whole
/// WHERE, whether it takes the extra conditions whole, and which program
/// a row it reads is still asked
type WhereScan = (Option<Box<dyn StorageExpr>>, bool, bool, WhereRest);

/// What a set slot's members make of its conjunct for one run
enum SetScan {
    /// The scan takes it as this filter
    Filter(Box<dyn StorageExpr>),
    /// Every row passes it
    Every,
    /// No row passes it
    None,
    /// The program asks it of each row
    Program,
}

/// How a prepared WHERE's rows were read
enum Passing {
    /// The scan proves that no row passes
    Pruned,
    /// The scan took the whole WHERE: its answers up to the target
    Answers(RowVec),
    /// The scan took the whole WHERE and the extra conditions: its rows,
    /// for a caller that takes them as they come
    Scan(Box<dyn Scanner>),
    /// The scan took the extra conditions, and its rows need the WHERE,
    /// bound to the sets the run read: the rows and that program, for a
    /// caller that asks it of each row as it comes
    Filtered(Box<dyn Scanner>, SharedProgram),
    /// Every row read that passes went to the caller
    Read,
}

/// A statement's WHERE rewritten for its blocks, before it compiles
struct WhereParts {
    clause: Expression,
    pushed: Vec<Expression>,
    /// The clause without the pushed conjuncts, when some are pushed and
    /// some are not
    scanned: Option<Expression>,
    /// For each set test, the clause without it, and without the pushed
    /// conjuncts too when some are pushed
    narrowed: Vec<(Expression, Option<Expression>)>,
    /// Each top-level EXISTS or IN test: its block, whether negated, the
    /// statement column an IN tests, and its conjunct's position
    set_tests: Vec<(u16, bool, Option<usize>, usize)>,
    conjuncts: usize,
}

/// A WHERE's programs: the whole one, the one without the pushed
/// conjuncts when some are pushed and some are not, and for each set test
/// the one without the test, and without the pushed conjuncts too when
/// some are pushed
struct WherePrograms {
    whole: SharedProgram,
    scanned: Option<SharedProgram>,
    narrowed: Vec<(SharedProgram, Option<SharedProgram>)>,
}

impl WhereParts {
    /// The WHERE's programs, each compiled by `compile`
    fn programs(
        &self,
        compile: impl Fn(&Expression) -> Result<SharedProgram>,
    ) -> Result<WherePrograms> {
        let optional = |clause: &Option<Expression>| clause.as_ref().map(&compile).transpose();
        let narrowed = self
            .narrowed
            .iter()
            .map(|(clause, scanned)| Ok((compile(clause)?, optional(scanned)?)))
            .collect::<Result<_>>()?;
        Ok(WherePrograms {
            whole: compile(&self.clause)?,
            scanned: optional(&self.scanned)?,
            narrowed,
        })
    }

    /// Each value a clause of the WHERE compares with a column of `schema`,
    /// read as the scan reads it, by node address
    fn read_as(&self, schema: &Schema, out: &mut FxHashMap<usize, DataType>) {
        let narrowed = self
            .narrowed
            .iter()
            .flat_map(|(clause, scanned)| std::iter::once(clause).chain(scanned));
        for clause in std::iter::once(&self.clause)
            .chain(&self.scanned)
            .chain(narrowed)
        {
            compare_as_scan(clause, schema, out);
        }
    }

    /// The WHERE as `programs` over `schema`, narrowed by the first set
    /// test whose block in `blocks` can name its rows
    fn prepare(
        self,
        programs: WherePrograms,
        blocks: &PreparedBlocks,
        schema: &Schema,
    ) -> PreparedWhere {
        let whole = self.conjuncts == 1;
        let narrowing_test =
            self.set_tests
                .iter()
                .enumerate()
                .find_map(|(at, &(id, negated, column, _))| {
                    let sq = &blocks.subqueries[id as usize];
                    let (column, members) = match (&sq.answer, column) {
                        (
                            Answer::Exists(Kept {
                                keyed: Some(plan), ..
                            }),
                            None,
                        ) if plan.keys.len() == 1 => (plan.keys[0].2, false),
                        (_, Some(column)) if sq.member_set_test() == Some(MemberTest::In) => {
                            (column, true)
                        }
                        _ => return None,
                    };
                    Some((
                        at,
                        Narrowing {
                            id,
                            column,
                            whole,
                            negated,
                            members,
                        },
                    ))
                });
        let narrowing = narrowing_test.map(|(_, narrowing)| narrowing);
        let slot = |&(id, negated, column, _): &(u16, bool, Option<usize>, usize)| {
            let sq = &blocks.subqueries[id as usize];
            let column = column.filter(|_| sq.uncorrelated)?;
            sq.member_set_test().is_some().then_some(SetSlot {
                id,
                column,
                negated,
            })
        };
        // The block a narrowing reads the rows by is the scan's only when
        // the narrowing declines
        let narrowed = |id: u16| narrowing.is_some_and(|n| n.id == id);
        let sets = self
            .set_tests
            .iter()
            .filter(|test| !narrowed(test.0))
            .filter_map(slot)
            .collect();
        let narrowing_set = self
            .set_tests
            .iter()
            .filter(|test| narrowed(test.0))
            .find_map(slot);
        let (narrowed_program, narrowed_scanned) = narrowing_test
            .and_then(|(at, _)| programs.narrowed.get(at).cloned())
            .map_or((None, None), |(program, scanned)| {
                (
                    Some(WhereProgram::of(program, WhereRest::Narrowed, blocks)),
                    scanned.map(|p| WhereProgram::of(p, WhereRest::NarrowedScanned, blocks)),
                )
            });
        let pushed = self.pushed.len();
        let pushdown = conjunction(self.pushed);
        PreparedWhere {
            push: pushdown.as_ref().and_then(|p| PreparedPush::of(p, schema)),
            whole: WhereProgram::of(programs.whole, WhereRest::Whole, blocks),
            scanned: programs
                .scanned
                .map(|program| WhereProgram::of(program, WhereRest::Scanned, blocks)),
            narrowed: narrowed_program,
            narrowed_scanned,
            pushed,
            conjuncts: self.conjuncts,
            pushdown,
            narrowing,
            sets,
            narrowing_set,
        }
    }
}

/// One statement shape compiled against its tables' columns
#[derive(Debug)]
pub struct PreparedSelect {
    cached_epoch: u64,
    /// The columns of the statement's table or derived table at preparation
    columns: CompactArc<Vec<String>>,
    blocks: PreparedBlocks,
    holes: HoleSet,
    where_: PreparedWhere,
    projection: PreparedProjection,
    /// The uncorrelated member blocks the output reads, read before any row
    ahead: Vec<u16>,
    output_columns: CompactArc<Vec<String>>,
    /// The output columns the statement shows; the sort's keys follow them
    visible: usize,
    /// Where each output column comes from, for an ORDER BY to bind by
    layout: Option<CompactArc<OutputLayout>>,
}

/// A subquery over one table, prepared as its own scope: every value it
/// reads from an enclosing scope becomes a parameter slot, filled per call
/// by a hole program over the row it runs for
#[derive(Debug)]
struct PreparedSubquery {
    /// The inner table's name as the catalog knows it; None without FROM
    table: Option<SmartString>,
    /// A FROM that is no table: its rows, read once per run
    source: Option<BlockSource>,
    column_indices: Vec<usize>,
    /// The table's schema the programs read positions of
    schema: CompactArc<Schema>,
    /// The WHERE's conjuncts the scan can take
    pushdown: Vec<PreparedComparison>,
    /// Every conjunct is one the scan takes: a call that pushes them all
    /// has the scan's answer, as the general path does, and skips the filter
    pushed_whole: bool,
    /// The pushed equality on the table's primary key, by its position in
    /// `pushdown`: a call reads the row its key names instead of scanning
    point: Option<usize>,
    /// An EXISTS block's correlation equalities, which pick its candidate rows
    candidates: Option<SharedProgram>,
    /// The WHERE; for an EXISTS block, what of it the candidates leave
    filter: Option<SharedProgram>,
    /// An EXISTS block's `filter` by conjunct, with the slots each reads:
    /// asked in its place when a slot's value raised an error
    rest: Vec<(SharedProgram, SmallVec<[usize; 2]>)>,
    /// The slots its own programs read; a scalar block raises a failed one
    reads: SmallVec<[usize; 2]>,
    answer: Answer,
    /// A block that aggregates: its answer reads the slot rows of its groups
    grouping: Option<Box<Grouping>>,
    /// A LIMIT or OFFSET, which choose the rows the block answers from
    order: Option<Box<BlockOrder>>,
    /// The subquery whose row this one runs for; None for the statement's row
    parent: Option<u16>,
    /// What fills each slot from the parent's row, including values a
    /// deeper subquery reads from it; empty for the statement's own blocks,
    /// whose holes its `HoleSet` keeps
    holes: Vec<(usize, Hole)>,
    /// A join statement's own block's holes as written, compiled again for
    /// each layout of the row its sites run over
    hole_exprs: Vec<(usize, Expression)>,
    /// A grouped statement's own block's holes as the block wrote them,
    /// which choose between group items holding one column
    hole_written: Vec<(usize, Expression)>,
    /// Reads no enclosing scope: its value is computed once per run
    uncorrelated: bool,
}

/// A block's LIMIT and OFFSET over the rows its WHERE passes, in the order
/// of its ORDER BY; an EXISTS keeps no keys, its answer the count alone
#[derive(Debug)]
struct BlockOrder {
    /// Each key over the block's row, its direction and its NULLS clause;
    /// no program for a key that is the SELECT item, whose value it keeps
    keys: Vec<(Option<SharedProgram>, bool, Option<bool>)>,
    /// The first key that is the SELECT item, which the answer reads
    item: Option<usize>,
    /// For each of a row's values, the key that is it, which the member
    /// reads; empty for a single item
    values: Vec<Option<usize>>,
    /// A row's key that names a value again, and the key that computed it
    repeats: Vec<(usize, usize)>,
    /// The DISTINCT ON keys over the block's row: the first row in order of
    /// each of their values is kept, before the OFFSET
    distinct: Vec<SharedProgram>,
    limit: Option<SharedProgram>,
    offset: Option<SharedProgram>,
}

impl BlockOrder {
    /// The order of two rows' key values
    fn compare(&self, a: &[Value], b: &[Value]) -> std::cmp::Ordering {
        self.keys
            .iter()
            .zip(a.iter().zip(b))
            .map(|((_, ascending, nulls), (a, b))| {
                crate::core::value::compare_order_key(Some(a), Some(b), *ascending, *nulls)
            })
            .find(|ordering| ordering.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// A row a block's LIMIT may keep: by its keys, then by when it was read
struct HeldRow<'o> {
    keys: SmallVec<[Value; 2]>,
    read: usize,
    row: Row,
    order: &'o BlockOrder,
}

impl Ord for HeldRow<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order
            .compare(&self.keys, &other.keys)
            .then(self.read.cmp(&other.read))
    }
}

impl PartialOrd for HeldRow<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeldRow<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for HeldRow<'_> {}

impl PreparedSubquery {
    /// The test an uncorrelated block's members answer, read once per run
    /// as a set: of a block over one source, or of one run as a statement
    fn member_set_test(&self) -> Option<MemberTest> {
        if !self.uncorrelated {
            return None;
        }
        match &self.answer {
            Answer::Members(_, members) => Some(members.test),
            Answer::Statement(_, block) => match block.form {
                StatementForm::Members(_, test) => Some(test),
                _ => None,
            },
            _ => None,
        }
    }

    /// How many values a member of the block holds: one, or a row's
    fn member_width(&self) -> usize {
        match &self.answer {
            Answer::Members(_, members) => members.width(),
            Answer::Statement(_, block) => block.run.statement.columns.len(),
            _ => 1,
        }
    }

    /// Whether the block may read `table`: its own table, or one its view's
    /// query reads, any when they are not known. A CTE's rows are read before
    /// the statement writes; a VALUES list or table function reads none
    fn may_read(&self, table: &str) -> bool {
        if let Answer::Statement(_, block) = &self.answer {
            return block
                .tables
                .as_ref()
                .is_none_or(|tables| tables.iter().any(|read| read == table));
        }
        match &self.source {
            None => self.table.as_deref() == Some(table),
            Some(BlockSource::View(_, tables, _) | BlockSource::Derived(_, tables)) => tables
                .as_ref()
                .is_none_or(|tables| tables.iter().any(|read| read == table)),
            Some(_) => false,
        }
    }
}

/// What a block reads in place of a table, materialized once per run
/// before the statement's rows, as the statement's own source is
#[derive(Debug)]
enum BlockSource {
    /// A view, with the tables its query reads, None when they are not
    /// known, and the query its columns were named for
    View(SmartString, Option<Vec<SmartString>>, String),
    /// A derived table that reads no enclosing scope, with the tables its
    /// statement reads
    Derived(Box<StatementRun>, Option<Vec<SmartString>>),
    Cte(SmartString),
    Values(Box<ValuesTableSource>),
    /// A table function, with the WHERE's conjuncts that narrow its range,
    /// and the LIMIT and OFFSET of a block whose first rows are its answer
    Function(
        Box<FunctionTableSource>,
        Option<Box<Expression>>,
        Option<BlockCounts>,
    ),
}

impl BlockSource {
    /// Whether its rows are read when a call first needs them, not when the
    /// run opens: a CTE's are there already
    fn lazy(&self) -> bool {
        !matches!(self, Self::Cte(_))
    }
}

/// A block's LIMIT and OFFSET
type BlockCounts = (Box<Expression>, Option<Box<Expression>>);

/// A source a block's FROM names and its columns
type NamedSource = (BlockSource, Vec<String>);

#[derive(Debug, Clone, Copy)]
struct Narrowing {
    /// The block's id
    id: u16,
    /// The statement column its key is
    column: usize,
    /// It is the whole WHERE
    whole: bool,
    /// NOT EXISTS or NOT IN: the rows its keys do not name
    negated: bool,
    /// An uncorrelated `column IN (b)`, whose members are the keys
    members: bool,
}

/// One key's members as read, and the first error one of its candidates raised
type MemberGroup = (Vec<Value>, Option<Error>);

/// An error a candidate raised, kept until no candidate settles the answer;
/// a cancel ends the statement and is raised at once, so no answer kept
/// across statements holds it
fn deferred(e: Error) -> Result<Error> {
    match e {
        Error::QueryCancelled => Err(e),
        other => Ok(other),
    }
}

/// Slots whose value raised an error, each with its error
type FailedSlots = SmallVec<[(usize, Error); 1]>;

/// One slot's value over the row a subquery runs for
#[derive(Debug)]
enum Hole {
    /// A column of that row, copied without a program
    Column(usize),
    /// A column of the second row of a pair, copied without a program
    Column2(usize),
    Program(SharedProgram),
}

impl Hole {
    fn value(&self, vm: &mut ExprVM, row: &Row, ctx: &ExecuteContext) -> Result<Value> {
        match self {
            Hole::Column(index) => Ok(row.get(*index).cloned().unwrap_or_else(Value::null_unknown)),
            Hole::Column2(index) => Ok(ctx
                .row2
                .and_then(|row| row.get(*index))
                .cloned()
                .unwrap_or_else(Value::null_unknown)),
            Hole::Program(program) => vm.execute(program, ctx),
        }
    }
}

/// What fills the slots of the statement's own blocks from the row its
/// programs run over, by block id; a nested block's holes are its own
#[derive(Debug, Default)]
pub(crate) struct HoleSet {
    holes: Vec<Vec<(usize, Hole)>>,
}

/// What a subquery block answers
#[derive(Debug)]
enum Answer {
    /// Its one row's SELECT expression, NULL for no row
    Value(SharedProgram, Kept),
    /// Whether a candidate row passes its WHERE
    Exists(Kept),
    /// A value tested against the SELECT expression of its candidates
    Members(Kept, Members),
    /// The block run as a statement of its own for each call
    Statement(Kept, Box<StatementBlock>),
}

/// A block whose derived table reads an enclosing scope: the block with
/// each name bound beyond it reading a slot, what it answers, and the
/// tables it reads, None when they are not known
#[derive(Debug)]
struct StatementBlock {
    run: StatementRun,
    form: StatementForm,
    tables: Option<Vec<SmartString>>,
    /// Whether it computes a window function
    windowed: bool,
}

#[derive(Debug, Clone, Copy)]
enum StatementForm {
    Value,
    Exists,
    /// The tested value's column type and the test
    Members(Option<DataType>, MemberTest),
}

/// How a block that aggregates feeds each candidate row to its group. A
/// slot row holds the GROUP BY items, then one value per distinct call
struct Grouping {
    aggregations: Vec<SqlAggregateFunction>,
    calls: Vec<CallTemplate>,
    inputs: Vec<CallInputs>,
    /// Each call's own column; None for COUNT(*) and an argument program
    columns: Vec<Option<usize>>,
    /// The GROUP BY items over the candidate row
    items: Vec<Hole>,
    /// Over a slot row
    having: Option<SharedProgram>,
    /// The groups of every key read in one pass, for a block that reads
    /// its parent only through key equalities
    keyed: Option<KeyedPlan>,
    /// An uncorrelated block without subqueries, as written: run once as a
    /// statement, so every aggregate path the general path admits answers it
    statement: Option<Box<StatementRun>>,
    /// One group whose every call counts rows, which an index can answer
    counts_rows: bool,
}

/// A block run as a statement of its own, with the plan its runs keep, as
/// a cached statement keeps its own
struct StatementRun {
    statement: SelectStatement,
    classification: std::sync::OnceLock<Arc<super::query_classification::QueryClassification>>,
    compiled: NestedPlan,
}

impl std::fmt::Debug for StatementRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatementRun")
            .field("statement", &self.statement.to_string())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Grouping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grouping")
            .field("aggregations", &self.aggregations)
            .field("columns", &self.columns)
            .field("items", &self.items)
            .field("having", &self.having)
            .field("keyed", &self.keyed)
            .field("statement", &self.statement.is_some())
            .finish()
    }
}

impl Grouping {
    /// No groups yet, for keys of `prefix` key columns before the items;
    /// without either, the one group there with no rows
    fn groups(&self, prefix: usize) -> GroupedAggregates {
        let calls = self.calls.iter().map(CallTemplate::duplicate).collect();
        match prefix + self.items.len() {
            0 => GroupedAggregates::global(calls),
            width => GroupedAggregates::new(calls, width),
        }
    }

    /// Feeds `row` to the runs of every call in `group`
    fn feed(
        &self,
        groups: &mut GroupedAggregates,
        group: usize,
        row: &Row,
        exec: &ExecuteContext,
        vm: &mut Option<ExprVM>,
    ) -> Result<()> {
        for (i, run) in groups.aggregates(group).iter_mut().enumerate() {
            let column = match self.columns[i] {
                Some(column) => row.get(column),
                None => Some(&COUNT_STAR),
            };
            self.inputs[i].feed(run, column, vm, exec, &self.aggregations[i])?;
        }
        Ok(())
    }
}

/// How one run reads a candidate's group key: key columns, then the
/// items; borrowed from the row when every part is a column
struct KeyReader {
    columns: Option<Vec<usize>>,
    prefix: Vec<usize>,
    values: Vec<Value>,
}

impl KeyReader {
    fn new(grouping: &Grouping, prefix: Vec<usize>) -> Self {
        let columns = grouping
            .items
            .iter()
            .map(|item| match item {
                Hole::Column(column) => Some(*column),
                Hole::Column2(_) | Hole::Program(_) => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(|items| prefix.iter().copied().chain(items).collect());
        Self {
            columns,
            prefix,
            values: Vec::new(),
        }
    }

    /// The group of `row`, made at its first row
    fn group(
        &mut self,
        grouping: &Grouping,
        groups: &mut GroupedAggregates,
        row: &Row,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
    ) -> Result<Option<usize>> {
        if let Some(columns) = &self.columns {
            return Ok(groups.group(&ColumnsKey {
                values: row.as_slice(),
                columns,
            }));
        }
        self.values.clear();
        for column in &self.prefix {
            self.values.push(
                row.get(*column)
                    .cloned()
                    .unwrap_or_else(Value::null_unknown),
            );
        }
        for item in &grouping.items {
            self.values.push(item.value(vm, row, exec)?);
        }
        Ok(groups.group(self.values.as_slice()))
    }
}

/// A keyed grouped block's slot rows by key, read in one pass. A key no
/// candidate has, or a NULL one, reads `missing`: without GROUP BY, the
/// one row of no input
struct GroupedKeys {
    groups: FxHashMap<SmallVec<[Value; 1]>, Result<Vec<Row>>>,
    missing: Vec<Row>,
}

/// The test a value block answers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemberTest {
    In,
    /// The operator, and whether it is ALL
    Quantified(CompareOp, bool),
    /// A row compared with the block's one row, `(a, b) = (SELECT x, y ..)`
    One(RowComparison),
}

impl MemberTest {
    /// The answer one member settles: TRUE for IN and ANY, FALSE for ALL
    fn decides(self) -> bool {
        !matches!(self, MemberTest::Quantified(_, true))
    }
}

/// A value block's members: its SELECT expression, read as the tested
/// value's column type when that value is a column
#[derive(Debug)]
struct Members {
    projection: SharedProgram,
    /// A row's values after the first, which `projection` reads
    rest: Vec<SharedProgram>,
    read_as: Option<DataType>,
    test: MemberTest,
}

impl Members {
    /// The values a member holds: one, or a row's
    fn width(&self) -> usize {
        1 + self.rest.len()
    }

    /// A value of the SELECT expression as the tested value's column reads it
    fn read(&self, value: Value) -> Value {
        match self.read_as {
            Some(column_type) => read_as_column(column_type, value),
            None => value,
        }
    }
}

/// What a block is prepared to answer
#[derive(Clone, Copy)]
enum Form<'a> {
    Scalar,
    Exists(&'a ExistsExpression),
    Members {
        block: &'a SelectStatement,
        read_as: Option<DataType>,
        test: MemberTest,
    },
}

/// What an EXISTS or value block needs beyond its probe
#[derive(Debug)]
struct Kept {
    /// The block as written: what a result kept across statements is of
    identity: Arc<Expression>,
    identity_key: ExprKey,
    /// The enclosing scopes' names, columns and types, which decide what
    /// each name of the block binds to
    scopes: Arc<str>,
    /// Every table the block and the blocks in it read
    tables: Vec<SmartString>,
    /// It or a block in it reads a source no stamp checks: never kept
    reads_source: bool,
    /// The statement parameters the block and the blocks in it read, by
    /// position, which its answer depends on; None when it reads another slot
    params: Option<Arc<[usize]>>,
    /// Its candidates grouped by key once per run, when that answers as the probe
    keyed: Option<KeyedPlan>,
}

/// An EXISTS block that reads its parent only through `own column =
/// parent column` equalities of one type a hash tells apart
#[derive(Debug)]
struct KeyedPlan {
    /// Each key's own column, the slot its parent value fills, and the
    /// parent's column that value is
    keys: Vec<(usize, usize, usize)>,
    /// The comparisons of what the keys leave of the WHERE that the scan takes
    pushdown: Vec<PreparedComparison>,
    pushed_whole: bool,
    /// The WHERE is the keys alone, which is when a probe may be cheaper
    pure: bool,
    /// A parent column has no type: a call whose key value is not of its
    /// own column's type is asked without the keys
    checked: bool,
}

/// `column operator value` with the column resolved at preparation; only
/// the value is bound per call
#[derive(Debug)]
struct PreparedComparison {
    column: SmartString,
    index: usize,
    data_type: DataType,
    operator: Operator,
    value: PreparedValue,
}

#[derive(Debug)]
enum PreparedValue {
    Const(Value),
    /// A parameter or a slot, zero-based
    Slot(usize),
}

impl PreparedComparison {
    /// The conjunct as `own column operator value`, when it is one
    fn from_conjunct(conjunct: &Expression, schema: &Schema) -> Option<Self> {
        let Expression::Infix(infix) = conjunct else {
            return None;
        };
        let operator = infix_to_operator(infix.op_type)?;
        for (column, value, operator) in [
            (&infix.left, &infix.right, operator),
            (&infix.right, &infix.left, flip_operator(operator)),
        ] {
            let Some(column) = bound_column(column) else {
                continue;
            };
            let value = match value.as_ref() {
                Expression::Parameter(p) if p.index > 0 => PreparedValue::Slot(p.index - 1),
                other => PreparedValue::Const(extract_literal_value(other)?),
            };
            let read = schema.columns.get(column)?;
            if !scan_compares(read.data_type) {
                return None;
            }
            return Some(Self {
                column: SmartString::new(&read.name),
                index: column,
                data_type: read.data_type,
                operator,
                value,
            });
        }
        None
    }

    /// The value this call compares against, as the scan reads it, or
    /// None when the scan cannot read it by the comparison's own rule
    fn value(&self, params: &[Value]) -> Option<Value> {
        let value = match &self.value {
            PreparedValue::Const(value) => value.clone(),
            PreparedValue::Slot(slot) => params.get(*slot)?.clone(),
        };
        coerce_constant(Some(self.data_type), value)
    }

    /// Whether the value compared with is NULL, which equals no row
    fn compares_null(&self, params: &[Value]) -> bool {
        match &self.value {
            PreparedValue::Const(value) => value.is_null(),
            PreparedValue::Slot(slot) => params.get(*slot).is_some_and(Value::is_null),
        }
    }
}

/// A statement's pushed conjuncts as comparisons, in the AND tree that
/// `try_pushdown` builds from them, when every one is a comparison
#[derive(Debug)]
enum PreparedPush {
    Comparison(PreparedComparison),
    And(Box<PreparedPush>, Box<PreparedPush>),
}

impl PreparedPush {
    fn of(expr: &Expression, schema: &Schema) -> Option<Self> {
        match expr {
            Expression::Infix(infix) if infix.op_type == InfixOperator::And => Some(Self::And(
                Box::new(Self::of(&infix.left, schema)?),
                Box::new(Self::of(&infix.right, schema)?),
            )),
            _ => PreparedComparison::from_conjunct(expr, schema).map(Self::Comparison),
        }
    }

    /// The scan's filter for a run with `params`, as `try_pushdown` builds
    /// it; None when a value is not read as its column's type
    fn storage(&self, params: &[Value]) -> Option<Box<dyn StorageExpr>> {
        Some(match self {
            Self::Comparison(c) => Box::new(
                ComparisonExpr::new(c.column.clone(), c.operator, c.value(params)?)
                    .with_column_index(c.index),
            ),
            Self::And(left, right) => Box::new(AndExpr::new(vec![
                left.storage(params)?,
                right.storage(params)?,
            ])),
        })
    }
}

/// A subquery's pushed comparisons bound for one call: built on its first
/// call, then given each later call's values in place
enum BoundPushdown {
    One(ComparisonExpr),
    All(AndExpr),
}

impl BoundPushdown {
    /// `bound` holding `values` for `pushdown`
    fn bind(pushdown: &[PreparedComparison], bound: &mut Option<Self>, values: Vec<Value>) {
        match (pushdown, bound.as_mut()) {
            (_, Some(BoundPushdown::One(expr))) => {
                for value in values {
                    expr.set_comparison_value(value);
                }
            }
            (_, Some(BoundPushdown::All(and))) => {
                for (index, value) in values.into_iter().enumerate() {
                    and.set_comparison_value_at(index, value);
                }
            }
            ([one], None) => {
                *bound = values.into_iter().next().map(|value| {
                    BoundPushdown::One(
                        ComparisonExpr::new(one.column.clone(), one.operator, value)
                            .with_column_index(one.index),
                    )
                })
            }
            (many, None) => {
                *bound = Some(BoundPushdown::All(AndExpr::new(
                    many.iter()
                        .zip(values)
                        .map(|(c, value)| {
                            Box::new(
                                ComparisonExpr::new(c.column.clone(), c.operator, value)
                                    .with_column_index(c.index),
                            ) as Box<dyn StorageExpr>
                        })
                        .collect(),
                )))
            }
        }
    }

    fn expr(&self) -> &dyn StorageExpr {
        match self {
            BoundPushdown::One(expr) => expr,
            BoundPushdown::All(and) => and,
        }
    }
}

/// One output column: read from the row, or computed by a program
#[derive(Debug)]
enum Output {
    Column(usize),
    Program(Computed),
    /// A SELECT item holding subqueries, which the blocks answer
    Block(Computed),
    /// An ORDER BY or DISTINCT ON key, NULL where it fails, as on the general path
    Key(Computed),
    /// A key that reads SELECT aliases from the projected row, as its second
    /// row, and the positions of the items it reads
    AliasKey(Computed, SmallVec<[usize; 2]>),
    /// A key that is the SELECT item at this position, read from it
    Copy(usize),
}

/// A computed output's program, the item it reads as before a run binds
/// it, and its place among the outputs a run binds, when a run binds it
#[derive(Debug)]
struct Computed {
    program: SharedProgram,
    item: Item,
    slot: Option<usize>,
}

impl Computed {
    fn new(program: SharedProgram) -> Self {
        let item = Item::of(CompactArc::clone(&program));
        Self {
            program,
            item,
            slot: None,
        }
    }

    /// Whether a run binds `program`: it loads a parameter or asks a block
    fn binds(program: &SharedProgram) -> bool {
        program.ops().iter().any(|op| {
            matches!(
                op,
                Op::LoadParam(_)
                    | Op::ExecScalarSubquery(_)
                    | Op::ExecExists(_)
                    | Op::ExecInSubquery(_)
                    | Op::ExecQuantified(..)
            )
        })
    }
}

/// The SELECT list, then the columns and expressions ORDER BY appends for the sort
#[derive(Debug)]
pub(crate) struct PreparedProjection {
    outputs: Vec<Output>,
    /// The outputs are the row's columns in order: each row is its own projection
    whole_row: bool,
    /// The row's columns the outputs are, when every output is one
    columns: Option<Vec<usize>>,
    /// How many outputs' programs load a parameter or ask a block, which a
    /// run may bind them to
    slots: usize,
    /// Whether a key reads SELECT aliases from the projected row
    alias_keys: bool,
}

/// A SELECT list's items, each `*` spelled out as its columns
struct SelectItems<'a> {
    exprs: Vec<Expression>,
    /// The column each item of a `*` is, by position: names may repeat
    stars: Vec<Option<usize>>,
    /// Each item as the blocks rewrote it, when it holds subqueries
    rewritten: Vec<Option<&'a Expression>>,
}

/// What preparing one statement builds up
struct Builder<'e> {
    executor: &'e Executor,
    /// The statement's context, which a block's CTE and view are read in
    ctx: &'e ExecutionContext,
    /// The next free parameter slot
    next_slot: usize,
    /// Indexed by subquery id; None while that subquery is being prepared
    subqueries: Vec<Option<PreparedSubquery>>,
    /// The highest parameter the statement reads, one-based
    max_param: usize,
    /// The statement's own blocks keep their holes as written, for a row
    /// laid out apart from its table's
    keeps_holes: bool,
}

/// A value a block reads from a scope beyond its parent: its slot, the
/// expression a scope further out computes into it, that scope's depth,
/// and the expression as the block wrote it
type Delegated = (usize, Expression, usize, Expression);

/// What rewriting one block's expressions collects
#[derive(Default)]
struct Rewritten {
    holes: Vec<(usize, Expression)>,
    /// Each hole as the block wrote it, which a grouped statement binds to
    /// its group items as its output binds a name
    written: Vec<(usize, Expression)>,
    delegated: Vec<Delegated>,
    /// Whether a hole reads the parent's row
    reads_parent: bool,
    /// Whether anything read is a column of the block's own row
    reads_row: bool,
    /// The slot of each enclosing column read, by scope depth and column,
    /// and by its spelling when the holes are kept as written
    outer: FxHashMap<(usize, usize, Option<ExprKey>), usize>,
}

impl Builder<'_> {
    /// The blocks prepared for a statement taking `param_count` positional
    /// parameters, with its run slots compiled, and the holes of its own
    /// blocks over the statement's row
    fn finish(
        self,
        param_count: usize,
        run_slots: Vec<(usize, SharedProgram)>,
    ) -> Result<(PreparedBlocks, HoleSet)> {
        // Slots are numbered after the values given, so a value not given would read one
        if self.max_param > param_count {
            return Err(Error::internal(format!(
                "Query requires {} parameters but only {} provided",
                self.max_param, param_count
            )));
        }
        let mut subqueries = self
            .subqueries
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::internal("a prepared subquery left unfinished"))?;
        // A block's tables include those of every block nested in it
        for nested in 0..subqueries.len() {
            let mut ancestor = subqueries[nested].parent;
            let reads_source = subqueries[nested].source.is_some()
                || matches!(subqueries[nested].answer, Answer::Statement(..));
            while let Some(at) = ancestor {
                let table = subqueries[nested].table.clone();
                if table.is_none() && !reads_source {
                    break;
                }
                let (Answer::Value(_, kept)
                | Answer::Exists(kept)
                | Answer::Members(kept, _)
                | Answer::Statement(kept, _)) = &mut subqueries[at as usize].answer;
                kept.reads_source |= reads_source;
                if let Some(table) = table.filter(|table| !kept.tables.contains(table)) {
                    kept.tables.push(table);
                }
                ancestor = subqueries[at as usize].parent;
            }
        }
        // A block's answer depends on the parameters it and the blocks in it
        // read; any other slot it reads leaves it every parameter
        let mut params: Vec<Option<SmallVec<[usize; 2]>>> = subqueries
            .iter()
            .map(|sq| {
                sq.reads
                    .iter()
                    .all(|slot| *slot < param_count)
                    .then(|| sq.reads.clone())
            })
            .collect();
        for nested in 0..subqueries.len() {
            let mut ancestor = subqueries[nested].parent;
            while let Some(at) = ancestor {
                let at = at as usize;
                params[at] = match (params[at].take(), &params[nested]) {
                    (Some(mut read), Some(more)) => {
                        for slot in more {
                            if !read.contains(slot) {
                                read.push(*slot);
                            }
                        }
                        Some(read)
                    }
                    _ => None,
                };
                ancestor = subqueries[at].parent;
            }
        }
        for (sq, read) in subqueries.iter_mut().zip(params) {
            let (Answer::Value(_, kept)
            | Answer::Exists(kept)
            | Answer::Members(kept, _)
            | Answer::Statement(kept, _)) = &mut sq.answer;
            kept.params = read.map(|mut read| {
                read.sort_unstable();
                Arc::from(read.as_slice())
            });
        }
        let holes = HoleSet {
            holes: subqueries
                .iter_mut()
                .map(|sq| match sq.parent {
                    None => std::mem::take(&mut sq.holes),
                    Some(_) => Vec::new(),
                })
                .collect(),
        };
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_slots(self.next_slot);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_blocks(subqueries.len());
        let blocks = PreparedBlocks {
            param_count,
            slot_count: self.next_slot,
            run_slots,
            subqueries,
        };
        Ok((blocks, holes))
    }

    fn slot(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        slot
    }

    fn note_param(&mut self, expr: &Expression) {
        if let Expression::Parameter(p) = expr {
            self.max_param = self.max_param.max(p.index);
        }
    }

    /// The slot that carries column `index` of the enclosing scope at
    /// `depth` of `chain` into the block, `expr` as the block wrote it
    fn outer_slot(
        &mut self,
        chain: &Chain<'_>,
        depth: usize,
        index: usize,
        expr: &Expression,
        work: &mut Rewritten,
    ) -> Expression {
        let spelled = self.keeps_holes.then(|| ExprKey::of(expr));
        if let Some(&slot) = work.outer.get(&(depth, index, spelled)) {
            return slot_parameter(slot);
        }
        let slot = self.slot();
        work.outer.insert((depth, index, spelled), slot);
        let value = chain.column(depth, index);
        if depth == 1 {
            work.reads_parent = true;
            work.holes.push((slot, value));
            work.written.push((slot, expr.clone()));
        } else {
            work.delegated.push((slot, value, depth, expr.clone()));
        }
        slot_parameter(slot)
    }

    /// What fills a block's slots from the row of `outer[0]` it runs for,
    /// the holes as a join statement's sites lay them out, and as written
    #[allow(clippy::type_complexity)]
    fn compiled_holes(
        &self,
        outer: &[&ScopeTable],
        parent: Option<u16>,
        work: &Rewritten,
    ) -> Result<(
        Vec<(usize, Hole)>,
        Vec<(usize, Expression)>,
        Vec<(usize, Expression)>,
    )> {
        let holes = if work.holes.is_empty() {
            Vec::new()
        } else {
            let parent = CompileContext::bound(&self.executor.function_registry);
            work.holes
                .iter()
                .map(|(slot, hole)| match hole {
                    Expression::Column(c) => Ok((*slot, Hole::Column(c.index as usize))),
                    _ => compile_with(&parent, hole).map(|p| (*slot, Hole::Program(p))),
                })
                .collect::<Result<Vec<_>>>()?
        };
        let hole_exprs = if parent.is_none() && outer[0].tables.len() > 1 {
            work.holes.clone()
        } else {
            Vec::new()
        };
        let hole_written = match parent {
            None if self.keeps_holes => work.written.clone(),
            _ => Vec::new(),
        };
        Ok((holes, hole_exprs, hole_written))
    }

    /// Whether the derived table `stmt` of a block, which sees the scopes
    /// `outer`, binds a name in one of them, and the names its rows have
    fn reads_enclosing(
        &mut self,
        stmt: &SelectStatement,
        outer: &[&ScopeTable],
    ) -> Result<(bool, Vec<String>)> {
        let (mut reads, mut max_param) = (false, 0);
        let (_, names, _) = StatementWalk {
            executor: self.executor,
            ctx: self.ctx,
            on: &mut |reach| match reach {
                Reach::Outer(expr, ..) => {
                    reads = true;
                    Ok(Some(expr.clone()))
                }
                Reach::Param(Expression::Parameter(p)) => {
                    max_param = max_param.max(p.index);
                    Ok(None)
                }
                Reach::Param(_) => Ok(None),
            },
        }
        .statement(stmt, outer, 0, Check::Off)?;
        self.max_param = self.max_param.max(max_param);
        Ok((reads, named(names)?))
    }

    /// Bind the ORDER BY the EXISTS block `sq` runs without, as a block binds
    /// a key that chooses no row, over the scope `own` its FROM gives it
    fn bind_dropped_order(
        &mut self,
        sq: &SelectStatement,
        own: &ScopeTable,
        outer: &[&ScopeTable],
        id: u16,
        work: &Rewritten,
    ) -> Result<()> {
        check_block_positions(sq, own.columns.len())?;
        let mut scopes = Vec::with_capacity(outer.len() + 1);
        scopes.push(own);
        scopes.extend_from_slice(outer);
        let chain = Chain { scopes: &scopes };
        let registry = &self.executor.function_registry;
        let compile = CompileContext::bound(registry).with_prepared_subqueries();
        for ob in &sq.order_by {
            let key = with_aliases(&ob.expression, &sq.columns, &own.columns);
            let key = self.trial_rewrite(&chain, id, &key, &work.outer)?;
            compile_with(&compile, &key)?;
        }
        Ok(())
    }

    /// Bind the block `sq` over a join as a block over one table is bound
    /// when prepared, computing nothing: its clauses over its FROM's scope
    /// `own` read by position, its grouping checked as `grouping` checks it.
    /// A window function, or a bare name two of its tables have, binds when
    /// the block runs, which binds that name as the prepared side cannot know
    fn bind_join_block(
        &mut self,
        sq: &SelectStatement,
        own: &ScopeTable,
        outer: &[&ScopeTable],
        id: u16,
        work: &Rewritten,
        exists: bool,
    ) -> Result<()> {
        let clauses = sq
            .columns
            .iter()
            .chain(sq.where_clause.as_deref())
            .chain(&sq.group_by.columns)
            .chain(sq.having.as_deref())
            .chain(sq.order_by.iter().map(|ob| &ob.expression))
            .chain(&sq.distinct_on);
        if clauses
            .into_iter()
            .any(|e| names_any(e, &|name| own.shared(name)))
        {
            return Ok(());
        }
        let scope = own.by_position();
        let mut scopes = Vec::with_capacity(outer.len() + 1);
        scopes.push(&scope);
        scopes.extend_from_slice(outer);
        let chain = Chain { scopes: &scopes };
        let registry = &self.executor.function_registry;
        let compile = CompileContext::bound(registry).with_prepared_subqueries();
        let windowed = windowed_block(sq);
        let grouped = grouped_block(sq) && !windowed;
        let items: Vec<&Expression> = sq
            .columns
            .iter()
            .filter(|c| !matches!(c, Expression::Star(_) | Expression::QualifiedStar(_)))
            .map(unaliased)
            .collect();
        let named = |expr: &Expression| with_aliases(expr, &sq.columns, &scope.lower).into_owned();
        let mut bound: Vec<Expression> =
            sq.where_clause.as_deref().map(named).into_iter().collect();
        if !windowed {
            bound.extend(sq.order_by.iter().map(|ob| named(&ob.expression)));
            bound.extend(sq.distinct_on.iter().map(named));
        }
        if !grouped && !windowed {
            bound.extend(items.iter().map(|item| (*item).clone()));
        }
        for expr in &bound {
            let expr = self.trial_rewrite(&chain, id, expr, &work.outer)?;
            compile_with(&compile, &expr)?;
        }
        if grouped {
            let values = match exists {
                true => Vec::new(),
                false => items
                    .iter()
                    .map(|item| self.trial_rewrite(&chain, id, item, &work.outer))
                    .collect::<Result<Vec<_>>>()?,
            };
            let (next_slot, blocks) = (self.next_slot, self.subqueries.len());
            let mut trial = Rewritten {
                outer: work.outer.clone(),
                ..Rewritten::default()
            };
            let grouping = self.grouping(&chain, id, sq, &values, exists, &mut trial);
            self.next_slot = next_slot;
            self.subqueries.truncate(blocks);
            grouping?;
        }
        Ok(())
    }

    /// The block `sq`, whose FROM is a join or a derived table reading an
    /// enclosing scope, or which computes a window function, run as one
    /// statement: each name it binds beyond itself reads a slot its parent
    /// fills, as any block's does, and one that binds none answers once per run
    fn statement_block(
        &mut self,
        sq: &SelectStatement,
        outer: &[&ScopeTable],
        parent: Option<u16>,
        form: Form<'_>,
    ) -> Result<(u16, Vec<Delegated>, bool)> {
        let id = u16::try_from(self.subqueries.len())
            .map_err(|_| Error::internal("too many prepared subqueries"))?;
        self.subqueries.push(None);
        let unnamed = ScopeTable::unnamed();
        let mut scopes = Vec::with_capacity(outer.len() + 1);
        scopes.push(&unnamed);
        scopes.extend_from_slice(outer);
        let chain = Chain { scopes: &scopes };
        let mut work = Rewritten::default();
        let (executor, ctx) = (self.executor, self.ctx);
        // An EXISTS runs without its ORDER BY, which is still bound; one whose
        // rows its items choose, by a window or a DISTINCT ON, runs as written
        let dropped =
            matches!(form, Form::Exists(_)) && !windowed_block(sq) && sq.distinct_on.is_empty();
        let mut max_param = 0;
        // The parameters it reads key its kept answer; one not positional
        // leaves it every parameter
        let mut params = SmallVec::<[usize; 2]>::new();
        let (statement, _, own) = StatementWalk {
            executor,
            ctx,
            on: &mut |reach| match reach {
                Reach::Outer(expr, depth, index) => {
                    Ok(Some(self.outer_slot(&chain, depth, index, expr, &mut work)))
                }
                Reach::Param(Expression::Parameter(p)) if p.index > 0 => {
                    max_param = max_param.max(p.index);
                    if !params.contains(&(p.index - 1)) {
                        params.push(p.index - 1);
                    }
                    Ok(None)
                }
                Reach::Param(_) => {
                    params.push(usize::MAX);
                    Ok(None)
                }
            },
        }
        .statement(sq, outer, 0, Check::All)?;
        self.max_param = self.max_param.max(max_param);
        let joined = own.tables.len() > 1;
        if joined {
            if dropped {
                check_block_positions(sq, own.columns.len())?;
            }
            self.bind_join_block(sq, &own, outer, id, &work, matches!(form, Form::Exists(_)))?;
        }
        let statement = match dropped {
            // A join's run names its columns, so it binds what it reads
            true if joined => SelectStatement {
                order_by: Vec::new(),
                ..statement
            },
            true => {
                self.bind_dropped_order(sq, &own, outer, id, &work)?;
                existence(executor, statement, &own.lower, ctx)?
            }
            false => statement,
        };
        let (identity, signature, _) = block_identity(sq, form);
        // One that binds nothing beyond itself keeps its answer across
        // statements while the tables it reads, all known, stay as they were
        let uncorrelated = !work.reads_parent && work.delegated.is_empty();
        let tables = executor.read_tables(&statement);
        let kept_tables = tables.clone().filter(|_| uncorrelated);
        let kept = Kept {
            identity_key: ExprKey::exact(&identity),
            identity: Arc::new(identity),
            scopes: scope_signature(signature, outer),
            reads_source: kept_tables.is_none(),
            tables: kept_tables.unwrap_or_default(),
            params: None,
            keyed: None,
        };
        let form = match form {
            Form::Scalar => StatementForm::Value,
            Form::Exists(_) => StatementForm::Exists,
            Form::Members { read_as, test, .. } => StatementForm::Members(read_as, test),
        };
        let block = StatementBlock {
            tables,
            run: StatementRun {
                statement,
                classification: std::sync::OnceLock::new(),
                compiled: NestedPlan::default(),
            },
            form,
            windowed: windowed_block(sq),
        };
        let (holes, hole_exprs, hole_written) = self.compiled_holes(outer, parent, &work)?;
        self.subqueries[id as usize] = Some(PreparedSubquery {
            table: None,
            source: None,
            column_indices: Vec::new(),
            schema: CompactArc::new(Schema::new("", Vec::new())),
            pushdown: Vec::new(),
            pushed_whole: true,
            point: None,
            candidates: None,
            filter: None,
            rest: Vec::new(),
            reads: work.outer.values().copied().chain(params).collect(),
            answer: Answer::Statement(kept, Box::new(block)),
            grouping: None,
            order: None,
            parent,
            holes,
            hole_exprs,
            hole_written,
            uncorrelated,
        });
        Ok((id, work.delegated, work.reads_parent))
    }

    /// `expr` of the statement as written, with each subquery in it prepared
    /// as a block of the statement's row; what stands outside them, which
    /// the grouped path reads by name, stays as written
    fn rewrite_blocks(&mut self, chain: &Chain<'_>, expr: &Expression) -> Result<Expression> {
        let mut work = Rewritten::default();
        match expr {
            // A part prepared already, as an input of a grouped statement
            Expression::ScalarSubquery(ScalarSubquery { subquery, .. })
            | Expression::Exists(ExistsExpression { subquery, .. })
            | Expression::AllAny(AllAnyExpression { subquery, .. })
                if subquery.prepared.is_some() =>
            {
                Ok(expr.clone())
            }
            Expression::In(in_expr)
                if matches!(&*in_expr.right, Expression::ScalarSubquery(nested)
                    if nested.subquery.prepared.is_some()) =>
            {
                Ok(expr.clone())
            }
            Expression::Infix(infix) => match row_subquery(infix) {
                Some((row, nested, comparison)) if nested.subquery.prepared.is_none() => {
                    let form = Form::Members {
                        block: &nested.subquery,
                        read_as: None,
                        test: MemberTest::One(comparison),
                    };
                    let row = self.rewrite_blocks(chain, row)?;
                    let mark = self.nested(chain, None, &nested.subquery, form, &mut work)?;
                    Ok(row_against(infix, row, nested, mark))
                }
                Some(_) => Ok(expr.clone()),
                None => rebuild(expr, &mut |child| self.rewrite_blocks(chain, child)),
            },
            Expression::ScalarSubquery(_) | Expression::Exists(_) => {
                self.rewrite(chain, None, expr, &mut work)
            }
            Expression::In(in_expr) => {
                let Expression::ScalarSubquery(nested) = &*in_expr.right else {
                    return rebuild(expr, &mut |child| self.rewrite_blocks(chain, child));
                };
                let form = Form::Members {
                    block: &nested.subquery,
                    read_as: chain.column_type(&in_expr.left)?,
                    test: MemberTest::In,
                };
                let left = self.rewrite_blocks(chain, &in_expr.left)?;
                let mark = self.nested(chain, None, &nested.subquery, form, &mut work)?;
                Ok(Expression::In(InExpression {
                    token: in_expr.token.clone(),
                    left: Box::new(left),
                    right: Box::new(Expression::ScalarSubquery(ScalarSubquery {
                        token: nested.token.clone(),
                        subquery: mark,
                    })),
                    not: in_expr.not,
                }))
            }
            Expression::AllAny(all_any) => {
                let op = QuantifiedMembers::operator(&all_any.operator).ok_or_else(|| {
                    Error::internal("ANY or ALL with an operator outside the family")
                })?;
                let all = matches!(all_any.all_any_type, AllAnyType::All);
                let form = Form::Members {
                    block: &all_any.subquery,
                    read_as: chain.column_type(&all_any.left)?,
                    test: MemberTest::Quantified(op, all),
                };
                let left = self.rewrite_blocks(chain, &all_any.left)?;
                let mark = self.nested(chain, None, &all_any.subquery, form, &mut work)?;
                Ok(Expression::AllAny(AllAnyExpression {
                    token: all_any.token.clone(),
                    left: Box::new(left),
                    operator: all_any.operator.clone(),
                    all_any_type: all_any.all_any_type,
                    subquery: mark,
                }))
            }
            other => rebuild(other, &mut |child| self.rewrite_blocks(chain, child)),
        }
    }

    /// `expr`, rewritten for the statement's row, with each uncorrelated
    /// block it reads for a value or an existence test a run slot
    fn slotted(
        &mut self,
        expr: &Expression,
        run_slots: &mut Vec<(usize, Expression)>,
    ) -> Result<Expression> {
        let block = match expr {
            Expression::ScalarSubquery(sq) => sq.subquery.prepared,
            Expression::Exists(exists) => exists.subquery.prepared,
            _ => None,
        };
        // A window's block runs when a row asks it, as over one source
        let once = block
            .and_then(|id| self.subqueries.get(usize::from(id))?.as_ref())
            .is_some_and(|sq| {
                sq.uncorrelated
                    && sq.parent.is_none()
                    && match &sq.answer {
                        Answer::Value(..) | Answer::Exists(_) => true,
                        Answer::Statement(_, block) => {
                            !matches!(block.form, StatementForm::Members(..)) && !block.windowed
                        }
                        Answer::Members(..) => false,
                    }
            });
        match (block, once) {
            (Some(_), true) => {
                let slot = self.slot();
                run_slots.push((slot, expr.clone()));
                Ok(slot_parameter(slot))
            }
            (Some(_), false) => Ok(expr.clone()),
            (None, _) => rebuild(expr, &mut |child| self.slotted(child, run_slots)),
        }
    }

    /// [`Self::rewrite`] of `expr` for a look at its bound form only: its
    /// enclosing reads take the slots `outer` holds, and a new one is given
    /// back, so the statement's buffer does not grow for it. A block it
    /// prepares is checked and dropped again, so no run sets it up
    fn trial_rewrite(
        &mut self,
        chain: &Chain<'_>,
        id: u16,
        expr: &Expression,
        outer: &FxHashMap<(usize, usize, Option<ExprKey>), usize>,
    ) -> Result<Expression> {
        let (next_slot, blocks) = (self.next_slot, self.subqueries.len());
        let mut trial = Rewritten {
            outer: outer.clone(),
            ..Rewritten::default()
        };
        let rewritten = self.rewrite(chain, Some(id), expr, &mut trial);
        self.next_slot = next_slot;
        self.subqueries.truncate(blocks);
        rewritten
    }

    /// `expr` of the block whose chain is `chain` (id `id`, None for the
    /// statement), with own columns spelled for its programs, enclosing
    /// reads turned into slots and nested blocks prepared
    fn rewrite(
        &mut self,
        chain: &Chain<'_>,
        id: Option<u16>,
        expr: &Expression,
        work: &mut Rewritten,
    ) -> Result<Expression> {
        self.note_param(expr);
        match expr {
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
                match chain.resolve(expr)? {
                    None => Ok(expr.clone()),
                    Some((0, index)) => {
                        work.reads_row = true;
                        Ok(chain.column(0, index))
                    }
                    Some((depth, index)) => Ok(self.outer_slot(chain, depth, index, expr, work)),
                }
            }
            Expression::ScalarSubquery(nested) => {
                let mark = self.nested(chain, id, &nested.subquery, Form::Scalar, work)?;
                Ok(Expression::ScalarSubquery(ScalarSubquery {
                    token: nested.token.clone(),
                    subquery: mark,
                }))
            }
            Expression::Exists(nested) => {
                let mark = self.nested(chain, id, &nested.subquery, Form::Exists(nested), work)?;
                Ok(Expression::Exists(ExistsExpression {
                    token: nested.token.clone(),
                    subquery: mark,
                }))
            }
            Expression::In(in_expr) => {
                let Expression::ScalarSubquery(nested) = &*in_expr.right else {
                    return rebuild(expr, &mut |child| self.rewrite(chain, id, child, work));
                };
                let form = Form::Members {
                    block: &nested.subquery,
                    read_as: chain.column_type(&in_expr.left)?,
                    test: MemberTest::In,
                };
                let left = self.rewrite(chain, id, &in_expr.left, work)?;
                let mark = self.nested(chain, id, &nested.subquery, form, work)?;
                Ok(Expression::In(InExpression {
                    token: in_expr.token.clone(),
                    left: Box::new(left),
                    right: Box::new(Expression::ScalarSubquery(ScalarSubquery {
                        token: nested.token.clone(),
                        subquery: mark,
                    })),
                    not: in_expr.not,
                }))
            }
            Expression::AllAny(all_any) => {
                let op = QuantifiedMembers::operator(&all_any.operator).ok_or_else(|| {
                    Error::internal("ANY or ALL with an operator outside the family")
                })?;
                let all = matches!(all_any.all_any_type, AllAnyType::All);
                let form = Form::Members {
                    block: &all_any.subquery,
                    read_as: chain.column_type(&all_any.left)?,
                    test: MemberTest::Quantified(op, all),
                };
                let left = self.rewrite(chain, id, &all_any.left, work)?;
                let mark = self.nested(chain, id, &all_any.subquery, form, work)?;
                Ok(Expression::AllAny(AllAnyExpression {
                    token: all_any.token.clone(),
                    left: Box::new(left),
                    operator: all_any.operator.clone(),
                    all_any_type: all_any.all_any_type,
                    subquery: mark,
                }))
            }
            Expression::Infix(infix) => match row_subquery(infix) {
                Some((row, nested, comparison)) => {
                    let form = Form::Members {
                        block: &nested.subquery,
                        read_as: None,
                        test: MemberTest::One(comparison),
                    };
                    let row = self.rewrite(chain, id, row, work)?;
                    let mark = self.nested(chain, id, &nested.subquery, form, work)?;
                    Ok(row_against(infix, row, nested, mark))
                }
                None => rebuild(expr, &mut |child| self.rewrite(chain, id, child, work)),
            },
            other => rebuild(other, &mut |child| self.rewrite(chain, id, child, work)),
        }
    }

    /// Prepare the block `sq` nested in the block whose chain is `chain`,
    /// passing on what it reads beyond that block. Returns the mark the
    /// program compiles by its id: a mark, not a copy of the block
    fn nested(
        &mut self,
        chain: &Chain<'_>,
        id: Option<u16>,
        sq: &SelectStatement,
        form: Form<'_>,
        work: &mut Rewritten,
    ) -> Result<Box<SelectStatement>> {
        let (nested_id, delegated, reads_this_row) = self.subquery(sq, chain.scopes, id, form)?;
        work.reads_row |= reads_this_row;
        for (slot, value, depth, written) in delegated {
            if depth == 2 {
                work.reads_parent = true;
                work.holes.push((slot, value));
                work.written.push((slot, written));
            } else {
                work.delegated.push((slot, value, depth - 1, written));
            }
        }
        let mark = Box::new(SelectStatement {
            param_count: 0,
            token: sq.token.clone(),
            distinct: false,
            distinct_on: Vec::new(),
            columns: Vec::new(),
            with: None,
            table_expr: None,
            where_clause: None,
            group_by: GroupByClause::default(),
            having: None,
            window_defs: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            set_operations: Vec::new(),
            prepared: Some(nested_id),
        });
        Ok(mark)
    }

    /// A top-level conjunct of a subquery block's WHERE: the value an own
    /// column is compared with becomes one slot when it reads at most one
    /// enclosing scope, so the scan can take the comparison
    fn rewrite_conjunct(
        &mut self,
        chain: &Chain<'_>,
        id: u16,
        conjunct: &Expression,
        work: &mut Rewritten,
    ) -> Result<Expression> {
        if let Expression::Infix(infix) = conjunct {
            if is_comparison(infix.op_type) {
                let left = chain.reads(&infix.left)?;
                let right = chain.reads(&infix.right)?;
                let hoist = |value: &Expression, reads: Reads| {
                    reads & 1 == 0
                        && reads.count_ones() <= 1
                        && !is_leaf(value)
                        && !volatile(value, &self.executor.function_registry)
                };
                let sides = [(&infix.left, left), (&infix.right, right)];
                if let Some(index) = sides.iter().position(|(value, reads)| hoist(value, *reads)) {
                    let (value, reads) = sides[index];
                    let other = sides[1 - index].0;
                    if matches!(chain.resolve(other)?, Some((0, _))) {
                        self.note_params(value);
                        let slot = self.slot();
                        let depth = reads.trailing_zeros() as usize;
                        let value_in_scope = chain.in_scope(value)?;
                        if reads == 0 || depth == 1 {
                            work.reads_parent |= reads != 0;
                            work.holes.push((slot, value_in_scope));
                            work.written.push((slot, (**value).clone()));
                        } else {
                            work.delegated
                                .push((slot, value_in_scope, depth, (**value).clone()));
                        }
                        let mut out = infix.clone();
                        let other = self.rewrite(chain, Some(id), other, work)?;
                        let (left, right) = if index == 0 {
                            (slot_parameter(slot), other)
                        } else {
                            (other, slot_parameter(slot))
                        };
                        *out.left = left;
                        *out.right = right;
                        return Ok(Expression::Infix(out));
                    }
                }
            }
        }
        self.rewrite(chain, Some(id), conjunct, work)
    }

    /// Record every parameter `expr` reads
    fn note_params(&mut self, expr: &Expression) {
        self.note_param(expr);
        let _ = for_each_child(expr, &mut |child| {
            self.note_params(child);
            Ok(())
        });
    }

    /// The CTE or view a block's FROM names, a CTE before a view as the
    /// general path reads them, with its columns; None for a table. A view
    /// is read when a call first needs it
    fn named_source(&self, name: &str) -> Result<Option<NamedSource>> {
        if let Some((columns, _)) = self.ctx.get_cte_by_lower(name) {
            return Ok(Some((BlockSource::Cte(name.into()), columns.to_vec())));
        }
        let Some(view) = self.executor.engine.get_view_lowercase(name)? else {
            return Ok(None);
        };
        let parsed = self.executor.parse_view_statement(&view.query)?;
        let tables = match parsed.as_ref() {
            Statement::Select(select) => self.executor.read_tables(select),
            _ => None,
        };
        let named = view_names(self.executor, &view, self.ctx)?;
        let source = BlockSource::View(name.into(), tables, view.query.clone());
        Ok(Some((source, named)))
    }

    /// Prepare `sq`, run once per row of the scope `outer[0]`, as `form`
    /// says. Returns its id, the values it reads from scopes beyond its
    /// parent, and whether it reads its parent's row
    fn subquery(
        &mut self,
        sq: &SelectStatement,
        outer: &[&ScopeTable],
        parent: Option<u16>,
        form: Form<'_>,
    ) -> Result<(u16, Vec<Delegated>, bool)> {
        if windowed_block(sq) {
            return self.statement_block(sq, outer, parent, form);
        }
        // EXISTS, value and grouped blocks pick candidates by their correlation equalities
        let picks = grouped_block(sq) || !matches!(form, Form::Scalar);
        let executor = self.executor;
        let registry = &executor.function_registry;
        // A source's columns have no type, as a statement's own source's
        let over_source = |name: SmartString, source: BlockSource, columns: Vec<String>| {
            let schema = CompactArc::new(untyped_schema(&name, &columns));
            let scope = ScopeTable::of(name, &schema);
            (None, Some(source), schema, scope)
        };
        let (table, source, schema, scope) = match sq.table_expr.as_deref() {
            Some(Expression::TableSource(ts)) => {
                let name = ts.alias.as_ref().unwrap_or(&ts.name).value_lower.clone();
                match self.named_source(&ts.name.value_lower)? {
                    Some((source, columns)) => over_source(name, source, columns),
                    None => {
                        let schema = executor
                            .engine
                            .get_table_schema(&ts.name.value_lower)
                            .map_err(|e| match e {
                                Error::TableNotFound(_) => {
                                    Error::TableOrViewNotFound(ts.name.value.to_string())
                                }
                                other => other,
                            })?;
                        let scope = ScopeTable::of(name, &schema);
                        (Some(ts.name.value_lower.clone()), None, schema, scope)
                    }
                }
            }
            Some(Expression::ValuesSource(values)) => over_source(
                values
                    .alias
                    .as_ref()
                    .map(|a| a.value_lower.clone())
                    .unwrap_or_default(),
                BlockSource::Values(values.clone()),
                Executor::values_columns(values),
            ),
            Some(Expression::FunctionTableSource(function)) => {
                let name = &function
                    .alias
                    .as_ref()
                    .unwrap_or(&function.function)
                    .value_lower;
                let range = function_range(sq.where_clause.as_deref(), name);
                // Without a WHERE, an ORDER BY or a DISTINCT ON the first rows
                // are the answer
                let counts = sq
                    .limit
                    .clone()
                    .filter(|_| {
                        sq.where_clause.is_none()
                            && sq.order_by.is_empty()
                            && sq.distinct_on.is_empty()
                    })
                    .map(|limit| (limit, sq.offset.clone()));
                over_source(
                    name.clone(),
                    BlockSource::Function(function.clone(), range, counts),
                    Executor::tvf_columns(function)?,
                )
            }
            // A join runs with the block as one statement: for each call when
            // it reads an enclosing scope, else once per run
            Some(Expression::JoinSource(_)) => {
                return self.statement_block(sq, outer, parent, form);
            }
            // A derived table that reads an enclosing scope runs with the block
            // as one statement for each call; one that reads none, once per
            // run, when a call first needs it
            Some(Expression::SubquerySource(derived)) => {
                let (reads, columns) = self.reads_enclosing(&derived.subquery, outer)?;
                if reads {
                    return self.statement_block(sq, outer, parent, form);
                }
                let run = StatementRun {
                    statement: (*derived.subquery).clone(),
                    classification: std::sync::OnceLock::new(),
                    compiled: NestedPlan::default(),
                };
                let tables = executor.read_tables(&derived.subquery);
                let name = derived.alias.as_ref().map(|a| a.value_lower.clone());
                over_source(
                    name.unwrap_or_default(),
                    BlockSource::Derived(Box::new(run), tables),
                    columns,
                )
            }
            // A block without FROM reads one row without columns, under no name
            None => (
                None,
                None,
                CompactArc::new(Schema::new("", Vec::new())),
                ScopeTable::unnamed(),
            ),
            Some(_) => return Err(Error::internal("a prepared subquery without one table")),
        };
        let id = u16::try_from(self.subqueries.len())
            .map_err(|_| Error::internal("too many prepared subqueries"))?;
        self.subqueries.push(None);

        let mut scopes: Vec<&ScopeTable> = Vec::with_capacity(outer.len() + 1);
        scopes.push(&scope);
        scopes.extend_from_slice(outer);
        let chain = Chain { scopes: &scopes };
        let mut work = Rewritten::default();
        // The SELECT values a member is read from: one, or a row's
        let values = match form {
            Form::Exists(_) => Vec::new(),
            Form::Scalar | Form::Members { .. } => sq
                .columns
                .iter()
                .map(|column| self.rewrite(&chain, Some(id), unaliased(column), &mut work))
                .collect::<Result<Vec<_>>>()?,
        };
        let projection = values.first().filter(|_| values.len() == 1);
        let where_clause = sq
            .where_clause
            .as_deref()
            .map(|w| with_aliases(w, &sq.columns, &scope.columns));
        let mut pushdown = Vec::new();
        let mut pushed_whole = true;
        // The correlation equalities apart from the rest
        let mut keys = Vec::new();
        let mut key_parts = Vec::new();
        let mut rest = Vec::new();
        if let Some(where_clause) = &where_clause {
            let mut parts = Vec::new();
            conjuncts(where_clause, &mut parts);
            for part in parts {
                let key = if picks {
                    correlation_key(&chain, part)?
                } else {
                    None
                };
                let part = self.rewrite_conjunct(&chain, id, part, &mut work)?;
                // A source's rows are asked the WHERE program: no scan takes it
                let comparison = match source {
                    Some(_) => None,
                    None => PreparedComparison::from_conjunct(&part, &schema),
                };
                match comparison {
                    Some(comparison) => pushdown.push(comparison),
                    None => pushed_whole = false,
                }
                match key.and_then(|key| Some((key, slot_compared(&part)?))) {
                    Some((key, slot)) => {
                        keys.push((key, slot));
                        key_parts.push(part);
                    }
                    None => rest.push(part),
                }
            }
        }
        let candidates = conjunction(key_parts);
        let filter = conjunction(rest);
        // A LIMIT or OFFSET chooses the rows the block answers from, by its
        // ORDER BY, whose keys read the block's row and enclosing values as
        // its WHERE does; an EXISTS needs the count alone
        let exists = matches!(form, Form::Exists(_));
        let cut =
            sq.limit.is_some() || sq.offset.is_some() || (!sq.distinct_on.is_empty() && !exists);
        let chooses = cut && !exists;
        let item = projection.and(sq.columns.first());
        let (mut order_keys, mut bound_only) = (Vec::new(), Vec::new());
        // For each of a row's values, the key that sorts by it, whose value
        // the member reads
        let mut row_keys = vec![None; if values.len() > 1 { values.len() } else { 0 }];
        let mut repeats = Vec::new();
        if matches!(form, Form::Exists(_)) {
            check_block_positions(sq, scope.columns.len())?;
        }
        for ob in &sq.order_by {
            // A key naming one of a row's values, by its alias or as written,
            // sorts by that value as a single item's does
            if !row_keys.is_empty() && chooses {
                let named = match &ob.expression {
                    Expression::Identifier(name) if order_by_names_an_alias(&ob.expression, sq) => {
                        sq.columns.iter().position(|column| {
                            matches!(column, Expression::Aliased(a)
                                if a.alias.value_lower == name.value_lower)
                        })
                    }
                    key if !mentions_subquery(key) => self
                        .trial_rewrite(&chain, id, key, &work.outer)
                        .ok()
                        .and_then(|key| {
                            values
                                .iter()
                                .position(|value| ExprKey::exact(value) == ExprKey::exact(&key))
                        }),
                    _ => None,
                };
                if let Some(at) = named {
                    // A value named again sorts by the value its first key computed
                    let key = match row_keys[at] {
                        Some(first) => {
                            repeats.push((order_keys.len(), first));
                            None
                        }
                        None => {
                            row_keys[at] = Some(order_keys.len());
                            Some(values[at].clone())
                        }
                    };
                    order_keys.push((key, ob.ascending, ob.nulls_first));
                    continue;
                }
            }
            // A bare alias sorts by the item's value
            let alias = item.is_some() && order_by_names_an_alias(&ob.expression, sq);
            let key = match alias {
                true => None,
                false => Some(with_aliases(&ob.expression, &sq.columns, &scope.columns)),
            };
            match (key, chooses) {
                (None, true) => order_keys.push((None, ob.ascending, ob.nulls_first)),
                (Some(key), true) => {
                    // So does the item as written over the columns it binds,
                    // matched before an alias in it is read as its item
                    let same = !mentions_subquery(&ob.expression)
                        && projection.as_ref().is_some_and(|item| {
                            self.trial_rewrite(&chain, id, &ob.expression, &work.outer)
                                .is_ok_and(|key| ExprKey::exact(item) == ExprKey::exact(&key))
                        });
                    let key = match same {
                        true => None,
                        false => Some(self.rewrite(&chain, Some(id), &key, &mut work)?),
                    };
                    order_keys.push((key, ob.ascending, ob.nulls_first));
                }
                // A key that chooses no row is still bound, as any ORDER BY is
                (Some(key), false) => {
                    bound_only.push(self.trial_rewrite(&chain, id, &key, &work.outer)?);
                }
                (None, false) => {}
            }
        }
        let mut distinct_keys = Vec::new();
        for key in &sq.distinct_on {
            // A bare alias reads its item, as an ORDER BY key does
            let item = match key {
                Expression::Identifier(id) if order_by_names_an_alias(key, sq) => {
                    sq.columns.iter().find_map(|item| match item {
                        Expression::Aliased(a) if a.alias.value_lower == id.value_lower => {
                            Some(std::borrow::Cow::Borrowed(&*a.expression))
                        }
                        _ => None,
                    })
                }
                _ => None,
            };
            let key = item.unwrap_or_else(|| with_aliases(key, &sq.columns, &scope.columns));
            match cut {
                true => distinct_keys.push(self.rewrite(&chain, Some(id), &key, &mut work)?),
                false => bound_only.push(self.trial_rewrite(&chain, id, &key, &work.outer)?),
            }
        }
        for count in sq.limit.iter().chain(&sq.offset) {
            self.note_params(count);
        }
        let grouped = if grouped_block(sq) {
            let exists = matches!(form, Form::Exists(_));
            Some(self.grouping(&chain, id, sq, &values, exists, &mut work)?)
        } else {
            None
        };

        let mut read_as = FxHashMap::default();
        for expr in candidates.iter().chain(&filter) {
            compare_as_scan(expr, &schema, &mut read_as);
        }
        let own = CompileContext::bound(registry)
            .with_prepared_subqueries()
            .with_read_as(&read_as);
        let compile = |expr: &Option<Expression>| match expr {
            Some(expr) => compile_with(&own, expr).map(Some),
            None => Ok(None),
        };
        // The first SELECT value's program, and a row's others
        let projection_programs = || -> Result<(SharedProgram, Vec<SharedProgram>)> {
            let mut programs = match grouped.as_ref().filter(|g| !g.projection.is_empty()) {
                Some(grouped) => grouped.projection.clone(),
                None => values
                    .iter()
                    .map(|value| compile_with(&own, value))
                    .collect::<Result<Vec<_>>>()?,
            };
            if programs.is_empty() {
                return Err(Error::internal(
                    "a prepared subquery without a SELECT expression",
                ));
            }
            let first = programs.remove(0);
            Ok((first, programs))
        };
        let (identity, signature, keyed_test) = block_identity(sq, form);
        let mut others = SmallVec::new();
        for expr in values.iter().chain(&filter) {
            slots_read(expr, &mut others);
        }
        if let Some(grouped) = &grouped {
            others.extend(grouped.reads.iter().copied());
        }
        // A keyed set holds every row of a key, which a LIMIT or OFFSET cuts
        let keyed = (keyed_test
            && grouped.is_none()
            && !cut
            && self.keyed(&chain, &keys, &work, sq, &others))
        .then(|| keyed_plan(&chain, &keys, filter.as_ref(), &schema));
        let kept = Kept {
            identity_key: ExprKey::exact(&identity),
            identity: Arc::new(identity),
            scopes: scope_signature(signature, outer),
            tables: table.iter().cloned().collect(),
            reads_source: source.is_some(),
            params: None,
            keyed,
        };
        let answer = match form {
            Form::Scalar => Answer::Value(projection_programs()?.0, kept),
            Form::Members { read_as, test, .. } => {
                let (projection, rest) = projection_programs()?;
                Answer::Members(
                    kept,
                    Members {
                        projection,
                        rest,
                        read_as,
                        test,
                    },
                )
            }
            Form::Exists(_) => Answer::Exists(kept),
        };
        let mut reads = SmallVec::new();
        let counts = sq.limit.iter().chain(&sq.offset).map(|count| &**count);
        for expr in values
            .iter()
            .chain(&candidates)
            .chain(&filter)
            .chain(order_keys.iter().filter_map(|(key, ..)| key.as_ref()))
            .chain(&distinct_keys)
            .chain(counts)
        {
            slots_read(expr, &mut reads);
        }
        let grouping = match grouped {
            Some(Grouped {
                mut grouping,
                reads: grouping_reads,
                ..
            }) => {
                for slot in grouping_reads {
                    if !reads.contains(&slot) {
                        reads.push(slot);
                    }
                }
                if self.keyed(&chain, &keys, &work, sq, &others) {
                    grouping.keyed = Some(keyed_plan(&chain, &keys, filter.as_ref(), &schema));
                }
                Some(Box::new(grouping))
            }
            None => None,
        };
        let candidates = compile(&candidates)?;
        for key in &bound_only {
            compile_with(&own, key)?;
        }
        let order = match cut {
            true => {
                let count = |count: &Option<Box<Expression>>| {
                    count.as_deref().map(|c| compile_with(&own, c)).transpose()
                };
                Some(Box::new(BlockOrder {
                    item: match row_keys.is_empty() {
                        true => order_keys.iter().position(|(key, ..)| key.is_none()),
                        false => None,
                    },
                    values: row_keys,
                    repeats,
                    keys: order_keys
                        .iter()
                        .map(|(key, ascending, nulls)| {
                            let key = key.as_ref().map(|key| compile_with(&own, key));
                            key.transpose().map(|p| (p, *ascending, *nulls))
                        })
                        .collect::<Result<Vec<_>>>()?,
                    distinct: distinct_keys
                        .iter()
                        .map(|key| compile_with(&own, key))
                        .collect::<Result<Vec<_>>>()?,
                    limit: count(&sq.limit)?,
                    offset: count(&sq.offset)?,
                }))
            }
            false => None,
        };
        let mut rest = Vec::new();
        if let (true, Some(filter)) = (picks, &filter) {
            let mut parts = Vec::new();
            conjuncts(filter, &mut parts);
            for part in parts {
                let mut slots = SmallVec::new();
                slots_read(part, &mut slots);
                rest.push((compile_with(&own, part)?, slots));
            }
        }
        let filter = compile(&filter)?;
        let (holes, hole_exprs, hole_written) = self.compiled_holes(outer, parent, &work)?;
        let correlated = work.reads_parent || !work.delegated.is_empty();
        let mut grouping = grouping;
        if let Some(grouping) = grouping.as_mut() {
            // Without FROM it groups its one row here, as its correlated form does
            if table.is_some()
                && !correlated
                && !sq.where_clause.as_deref().is_some_and(mentions_subquery)
            {
                grouping.statement = Some(Box::new(StatementRun {
                    statement: sq.clone(),
                    classification: std::sync::OnceLock::new(),
                    compiled: NestedPlan::default(),
                }));
            }
        }
        self.subqueries[id as usize] = Some(PreparedSubquery {
            table,
            source,
            column_indices: (0..scope.columns.len()).collect(),
            schema: CompactArc::clone(&schema),
            point: schema.pk_column_index().and_then(|pk| {
                pushdown
                    .iter()
                    .position(|c| c.operator == Operator::Eq && c.index == pk)
            }),
            pushdown,
            pushed_whole,
            candidates,
            filter,
            rest,
            reads,
            answer,
            grouping,
            order,
            parent,
            holes,
            hole_exprs,
            hole_written,
            uncorrelated: !correlated,
        });
        Ok((id, work.delegated, work.reads_parent))
    }

    /// The grouping of the block `sq` whose chain is `chain`: its GROUP BY
    /// items, HAVING and calls rewritten as its other expressions are, and
    /// its SELECT expression `projection` compiled over the slot rows. A
    /// column of the block read outside a call and no item is an error,
    /// as on the general path; `exists` checks the list an EXISTS never reads
    fn grouping(
        &mut self,
        chain: &Chain<'_>,
        id: u16,
        sq: &SelectStatement,
        projection: &[Expression],
        exists: bool,
        work: &mut Rewritten,
    ) -> Result<Grouped> {
        let executor = self.executor;
        let registry = &executor.function_registry;
        let scope = chain.scopes[0];
        let bound = group_by_expressions(executor, sq, &scope.columns)?;
        let mut items = Vec::with_capacity(bound.len());
        for (expr, written) in bound.into_iter().zip(grouping_items(sq)) {
            // It binds where the resolver took another alias or a scope holds it
            let binds = &expr != written || matches!(chain.resolve(written), Ok(Some(_)));
            if aggregate_alias_key(Some(written), &sq.columns, binds) {
                return Err(aggregate_group_key());
            }
            items.push(self.rewrite(chain, Some(id), &expr, work)?);
        }
        let having = match sq.having.as_deref() {
            Some(having) => {
                let having = grouped_having(executor, sq, having)?;
                Some(self.rewrite(chain, Some(id), &having, work)?)
            }
            None => None,
        };

        let item_keys: FxHashSet<ExprKey> = items.iter().map(ExprKey::of).collect();
        let mut unread = Rewritten {
            outer: work.outer.clone(),
            ..Rewritten::default()
        };
        let mut listed = Vec::new();
        if exists {
            for column in &sq.columns {
                if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
                    listed.push(self.rewrite(chain, Some(id), unaliased(column), &mut unread)?);
                }
            }
        }
        for expr in projection.iter().chain(&having).chain(&listed) {
            if let Some(at) = ungrouped(expr, &item_keys) {
                let name = match scope.tables.len() > 1 || scope.columns[at].starts_with(RESOLVED) {
                    true => scope.spelled(at),
                    false => scope.columns[at].clone(),
                };
                return Err(ungrouped_column(&name));
            }
        }

        let calls_of = SelectStatement {
            param_count: 0,
            token: sq.token.clone(),
            distinct: false,
            distinct_on: Vec::new(),
            columns: projection.to_vec(),
            with: None,
            table_expr: None,
            where_clause: None,
            group_by: GroupByClause::default(),
            having: having.clone().map(Box::new),
            window_defs: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            set_operations: Vec::new(),
            prepared: None,
        };
        let (mut aggregations, _) = executor.parse_aggregations(&calls_of, None)?;
        // A call whose argument is one bound column reads that column
        let columns = aggregations
            .iter_mut()
            .map(|agg| match agg.expression.take() {
                Some(Expression::Column(c)) => Ok(Some(c.index as usize)),
                expression if expression.is_some() || agg.column == "*" => {
                    agg.expression = expression;
                    Ok(None)
                }
                _ => Err(Error::internal("an aggregate's column outside its block")),
            })
            .collect::<Result<Vec<_>>>()?;
        let calls = aggregations
            .iter()
            .map(|agg| agg.template(registry))
            .collect::<Result<Vec<_>>>()?;
        let own = CompileContext::bound(registry);
        let inputs = CallInputs::compile_with(&aggregations, |expr| compile_with(&own, expr))?;

        let slot = |i: usize| {
            u16::try_from(i).map_err(|_| Error::internal("too many grouped values in a block"))
        };
        let mut slots = FxHashMap::default();
        for (i, item) in items.iter().enumerate() {
            slots.entry(ExprKey::of(item)).or_insert(slot(i)?);
        }
        for (i, agg) in aggregations.iter().enumerate() {
            slots.entry(agg.key).or_insert(slot(items.len() + i)?);
        }
        // A group's values are read by identity alone: no column is in reach
        let over_slots = CompileContext::new(&[], registry).with_expression_aliases(slots);
        let projection = projection
            .iter()
            .map(|expr| compile_with(&over_slots, expr))
            .collect::<Result<Vec<_>>>()?;
        let having_program = having
            .as_ref()
            .map(|expr| compile_with(&over_slots, expr))
            .transpose()?;

        let mut reads = SmallVec::new();
        for expr in items.iter().chain(&having) {
            slots_read(expr, &mut reads);
        }
        let items = items
            .iter()
            .map(|item| {
                Ok(match bound_column(item) {
                    Some(column) => Hole::Column(column),
                    None => Hole::Program(compile_with(&own, item)?),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // Every call a bare COUNT(*): a FILTER or DISTINCT reads its rows
        let counts_rows = items.is_empty()
            && !calls.is_empty()
            && aggregations
                .iter()
                .all(|agg| agg.answered_with(Modifiers::NONE))
            && calls
                .iter()
                .all(|call| matches!(call.instance(), CompiledAggregate::CountStar { .. }));
        Ok(Grouped {
            grouping: Grouping {
                aggregations,
                calls,
                inputs,
                columns,
                items,
                having: having_program,
                keyed: None,
                statement: None,
                counts_rows,
            },
            projection,
            reads,
        })
    }

    /// Whether an EXISTS block whose correlation equalities are `keys`
    /// answers by a keyed set as its probe does: it reads its parent through
    /// them alone, holds nothing volatile, and each key's two columns are of
    /// one type whose equality is its hash's. No expression but the keys
    /// reads a key's slot: `others` holds the slots the rest reads
    fn keyed(
        &self,
        chain: &Chain<'_>,
        keys: &[((usize, usize, usize), usize)],
        work: &Rewritten,
        block: &SelectStatement,
        others: &[usize],
    ) -> bool {
        let key_slot = |slot: usize| keys.iter().any(|(_, key)| *key == slot);
        !keys.is_empty()
            && keys.iter().all(|(_, slot)| !others.contains(slot))
            && work.delegated.is_empty()
            && keys.iter().all(|((own, depth, index), _)| {
                let own = chain.scopes[0].types[*own];
                let parent = chain.scopes[1].types[*index];
                *depth == 1
                    && (own == parent || parent == DataType::Null)
                    && matches!(
                        own,
                        DataType::Integer
                            | DataType::Text
                            | DataType::Boolean
                            | DataType::Timestamp
                    )
            })
            && work
                .holes
                .iter()
                .all(|(slot, hole)| key_slot(*slot) || !reads_row_column(hole))
            && !statement_draws_per_call(block, &self.executor.function_registry)
    }
}

/// What preparing a block's grouping gives: the grouping, its SELECT
/// expression over the slot rows, and the slots its items and HAVING read
struct Grouped {
    grouping: Grouping,
    /// Each SELECT value over a slot row
    projection: Vec<SharedProgram>,
    reads: SmallVec<[usize; 2]>,
}

/// The block's own column that `expr` reads outside every aggregate call
/// and every GROUP BY item, keyed by `items`
fn ungrouped(expr: &Expression, items: &FxHashSet<ExprKey>) -> Option<usize> {
    if items.contains(&ExprKey::of(expr)) {
        return None;
    }
    match expr {
        Expression::FunctionCall(call) if is_aggregate_function(&call.function) => None,
        Expression::Column(c) => Some(c.index as usize),
        other => {
            let mut found = None;
            let _ = for_each_child(other, &mut |child| {
                if found.is_none() {
                    found = ungrouped(child, items);
                }
                Ok(())
            });
            found
        }
    }
}

/// The keyed plan of a block whose correlation equalities are `keys`, the
/// rest of its WHERE being `filter`, its parent scope the chain's second
fn keyed_plan(
    chain: &Chain<'_>,
    keys: &[((usize, usize, usize), usize)],
    filter: Option<&Expression>,
    schema: &Schema,
) -> KeyedPlan {
    let mut pushdown = Vec::new();
    let mut pushed_whole = true;
    let mut parts = Vec::new();
    if let Some(filter) = filter {
        conjuncts(filter, &mut parts);
    }
    for part in &parts {
        match PreparedComparison::from_conjunct(part, schema) {
            Some(comparison) => pushdown.push(comparison),
            None => pushed_whole = false,
        }
    }
    KeyedPlan {
        keys: keys
            .iter()
            .map(|((own, _, parent), slot)| (*own, *slot, *parent))
            .collect(),
        pushdown,
        pushed_whole,
        pure: filter.is_none(),
        checked: keys
            .iter()
            .any(|((_, _, parent), _)| chain.scopes[1].types[*parent] == DataType::Null),
    }
}

/// `own column = enclosing column` as the own column, the enclosing
/// scope's depth and that scope's column, when `conjunct` is one
fn correlation_key(
    chain: &Chain<'_>,
    conjunct: &Expression,
) -> Result<Option<(usize, usize, usize)>> {
    let Expression::Infix(infix) = conjunct else {
        return Ok(None);
    };
    if infix.op_type != InfixOperator::Equal {
        return Ok(None);
    }
    Ok(
        match (chain.resolve(&infix.left)?, chain.resolve(&infix.right)?) {
            (Some((0, own)), Some((depth, index))) | (Some((depth, index)), Some((0, own)))
                if depth > 0 =>
            {
                Some((own, depth, index))
            }
            _ => None,
        },
    )
}

/// Every positional parameter or slot `expr` reads, zero-based
fn slots_read(expr: &Expression, out: &mut SmallVec<[usize; 2]>) {
    match expr {
        Expression::Parameter(p) if p.index > 0 => {
            if !out.contains(&(p.index - 1)) {
                out.push(p.index - 1);
            }
        }
        other => {
            let _ = for_each_child(other, &mut |child| {
                slots_read(child, out);
                Ok(())
            });
        }
    }
}

/// The slot a rewritten correlation equality compares its own column with
fn slot_compared(conjunct: &Expression) -> Option<usize> {
    let Expression::Infix(infix) = conjunct else {
        return None;
    };
    [&infix.left, &infix.right]
        .into_iter()
        .find_map(|side| match side.as_ref() {
            Expression::Parameter(p) if p.index > 0 => Some(p.index - 1),
            _ => None,
        })
}

/// Whether the pushdown would have to evaluate part of `expr` itself:
/// an operand that reads no column and is more than a literal or parameter
fn needs_folding(expr: &Expression) -> bool {
    let mut folds = false;
    let _ = for_each_child(expr, &mut |child| {
        folds |= (!is_leaf(child) && !mentions_column(child)) || needs_folding(child);
        Ok(())
    });
    folds
}

/// Each statement `expr` holds as a subquery, not those inside them
fn nested_statements<'e>(expr: &'e Expression, out: &mut Vec<&'e SelectStatement>) {
    match expr {
        Expression::ScalarSubquery(sq) => out.push(&sq.subquery),
        Expression::Exists(exists) => out.push(&exists.subquery),
        Expression::AllAny(all_any) => {
            nested_statements(&all_any.left, out);
            out.push(&all_any.subquery);
        }
        Expression::Aliased(aliased) => nested_statements(&aliased.expression, out),
        Expression::Distinct(distinct) => nested_statements(&distinct.expr, out),
        other => {
            let _ = for_each_child(other, &mut |child| {
                nested_statements(child, out);
                Ok(())
            });
        }
    }
}

fn mentions_column(expr: &Expression) -> bool {
    match expr {
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) | Expression::Column(_) => {
            true
        }
        Expression::ScalarSubquery(_) | Expression::Exists(_) | Expression::AllAny(_) => true,
        other => {
            let mut mentions = false;
            let _ = for_each_child(other, &mut |child| {
                mentions |= mentions_column(child);
                Ok(())
            });
            mentions
        }
    }
}

/// Every reference `expr` makes resolves by the name rules; `aliases`
/// are names ORDER BY may use without a column behind them
fn check_names(chain: &Chain<'_>, expr: &Expression, aliases: &[&str]) -> Result<()> {
    match expr {
        Expression::Identifier(id) if aliases.contains(&id.value_lower.as_str()) => Ok(()),
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
            chain.resolve(expr).map(|_| ())
        }
        other => for_each_child(other, &mut |child| check_names(chain, child, aliases)),
    }
}

/// Each ORDER BY and DISTINCT ON key's names as `chain` binds them, over the
/// rows of `columns`: a bare key names a SELECT alias first; inside an
/// expression a column of the same name comes first
fn check_key_names(stmt: &SelectStatement, chain: &Chain<'_>, columns: &[String]) -> Result<()> {
    let aliases: Vec<&str> = stmt
        .columns
        .iter()
        .filter_map(|c| match c {
            Expression::Aliased(a) => Some(a.alias.value_lower.as_str()),
            _ => None,
        })
        .collect();
    let alias_map = Executor::build_alias_map_excluding(&stmt.columns, Some(columns));
    let inner_aliases: Vec<&str> = alias_map.keys().map(String::as_str).collect();
    let keys = stmt
        .order_by
        .iter()
        .map(|ob| &ob.expression)
        .chain(&stmt.distinct_on);
    for key in keys {
        match key {
            Expression::Identifier(id) if aliases.contains(&id.value_lower.as_str()) => {}
            key => check_names(chain, key, &inner_aliases)?,
        }
    }
    Ok(())
}

impl PreparedProjection {
    /// The SELECT list over `chain`'s one table and what ORDER BY and
    /// DISTINCT ON append, with the names and the sources of the appended
    /// columns. `rewritten` holds each SELECT item, ORDER BY key and DISTINCT
    /// ON key holding subqueries as the blocks rewrote it, which compiles in
    /// its place
    fn prepare(
        executor: &Executor,
        stmt: &SelectStatement,
        chain: &Chain<'_>,
        rewritten: [&[Option<Expression>]; 3],
    ) -> Result<(Self, Vec<String>, Vec<ColumnSource>)> {
        let scope = chain.scopes[0];
        let columns: &[String] = &scope.columns;
        let mut select_exprs = Vec::with_capacity(stmt.columns.len());
        // The column each item of a `*` is, by position: names may repeat
        let mut star_columns: Vec<Option<usize>> = Vec::with_capacity(stmt.columns.len());
        // Each item's rewritten form, by its position in `select_exprs`
        let mut item_rewritten: Vec<Option<&Expression>> = Vec::with_capacity(stmt.columns.len());
        // A `*` item is read by position, under its column's name, by which
        // an ORDER BY key finds it selected
        let star_item = |i: usize| Expression::Identifier(ident(&columns[i], &scope.lower[i]));
        for (column, rewritten) in stmt.columns.iter().zip(rewritten[0]) {
            match column {
                Expression::Star(_) => {
                    select_exprs.extend((0..columns.len()).map(star_item));
                    star_columns.extend((0..columns.len()).map(Some));
                    item_rewritten.extend((0..columns.len()).map(|_| None));
                }
                Expression::QualifiedStar(star) => {
                    if star.qualifier.to_lowercase() != scope.tables[0].0.as_str() {
                        return Err(Error::invalid_argument(unknown_qualifier(&star.qualifier)));
                    }
                    select_exprs.extend((0..columns.len()).map(star_item));
                    star_columns.extend((0..columns.len()).map(Some));
                    item_rewritten.extend((0..columns.len()).map(|_| None));
                }
                other => {
                    check_names(chain, unaliased(other), &[])?;
                    select_exprs.push(other.clone());
                    star_columns.push(None);
                    item_rewritten.push(rewritten.as_ref());
                }
            }
        }
        check_key_names(stmt, chain, columns)?;
        let items = SelectItems {
            exprs: select_exprs,
            stars: star_columns,
            rewritten: item_rewritten,
        };
        let keys = (&stmt.order_by[..], &stmt.distinct_on[..]);
        let rewritten_keys = [rewritten[1], rewritten[2]];
        Self::build(executor, stmt, chain, items, keys, columns, rewritten_keys)
    }

    /// The SELECT list over a join's rows of `all_columns` and, unless the
    /// join sorted them already, what ORDER BY and DISTINCT ON append; an
    /// item or key holding subqueries compiles as `blocks` rewrote it
    pub(crate) fn prepare_joined(
        executor: &Executor,
        stmt: &SelectStatement,
        all_columns: &[String],
        sorted: bool,
        blocks: &JoinBlocks,
    ) -> Result<(Self, Vec<String>, Vec<ColumnSource>)> {
        let scopes = [&blocks.scope];
        let chain = Chain { scopes: &scopes };
        check_key_names(stmt, &chain, all_columns)?;
        let mut items = SelectItems {
            exprs: Vec::with_capacity(stmt.columns.len()),
            stars: Vec::with_capacity(stmt.columns.len()),
            rewritten: Vec::with_capacity(stmt.columns.len()),
        };
        let star = |items: &mut SelectItems<'_>, i: usize| {
            let spelled = all_columns[i].as_str();
            items
                .exprs
                .push(Expression::Identifier(ident(spelled, spelled)));
            items.stars.push(Some(i));
            items.rewritten.push(None);
        };
        for (item, rewritten) in stmt.columns.iter().zip(&blocks.select) {
            match item {
                Expression::Star(_) => (0..all_columns.len()).for_each(|i| star(&mut items, i)),
                Expression::QualifiedStar(q) => {
                    for_each_qualified_star_column(all_columns, &q.qualifier, None, |i, _| {
                        star(&mut items, i)
                    })
                }
                other => {
                    check_names(&chain, unaliased(other), &[])?;
                    items.exprs.push(other.clone());
                    items.stars.push(None);
                    items.rewritten.push(rewritten.as_ref());
                }
            }
        }
        let (keys, rewritten): (_, [&[Option<Expression>]; 2]) = match sorted {
            true => ((&[][..], &[][..]), [&[], &[]]),
            false => (
                (&stmt.order_by[..], &stmt.distinct_on[..]),
                [&blocks.order_by, &blocks.distinct_on],
            ),
        };
        Self::build(executor, stmt, &chain, items, keys, all_columns, rewritten)
    }

    /// `items` and the columns `order_by` and `distinct_on` append over the
    /// rows of `columns`, each bound in `chain`, or compiled as the blocks
    /// rewrote it
    fn build(
        executor: &Executor,
        stmt: &SelectStatement,
        chain: &Chain<'_>,
        items: SelectItems<'_>,
        (order_by, distinct_on): (&[OrderByExpression], &[Expression]),
        columns: &[String],
        rewritten: [&[Option<Expression>]; 2],
    ) -> Result<(Self, Vec<String>, Vec<ColumnSource>)> {
        let SelectItems {
            exprs: select_exprs,
            stars: star_columns,
            rewritten: item_rewritten,
        } = items;
        let alias_map = Executor::build_alias_map_excluding(&stmt.columns, Some(columns));
        let sort_keys = || order_by.iter().map(|ob| &ob.expression).chain(distinct_on);
        // A key holding a subquery is computed into a column named after it
        let plan = executor.order_by_projection(
            &select_exprs,
            order_by,
            distinct_on,
            stmt.table_expr.as_deref(),
            columns,
            true,
        );
        let registry = &executor.function_registry;
        let bound = CompileContext::bound(registry).with_prepared_subqueries();
        // An expression bound here, or as the blocks rewrote it
        let compile = |expr: &Expression, rewritten: Option<&Expression>| match rewritten {
            Some(rewritten) => compile_with(&bound, rewritten),
            None => compile_with(&bound, &bind_own(chain, unaliased(expr), &|_| false)?),
        };
        let mut outputs = Vec::with_capacity(select_exprs.len() + plan.extra_names.len());
        for (((expr, index), star), rewritten) in select_exprs
            .iter()
            .zip(&plan.select_column_indices)
            .zip(&star_columns)
            .zip(&item_rewritten)
        {
            outputs.push(match (star.or(*index), rewritten) {
                (Some(index), _) => Output::Column(index),
                (None, Some(_)) => Output::Block(Computed::new(compile(expr, *rewritten)?)),
                (None, None) => Output::Program(Computed::new(compile(expr, None)?)),
            });
        }
        outputs.extend(plan.extra_order_indices.iter().map(|&i| Output::Column(i)));
        // Each key as the blocks rewrote it, found by the node it is
        let rewritten_key = |expr: &Expression| {
            let order_at = order_by
                .iter()
                .position(|ob| std::ptr::eq(&ob.expression, expr));
            let distinct_at = distinct_on.iter().position(|k| std::ptr::eq(k, expr));
            match (order_at, distinct_at) {
                (Some(at), _) => rewritten[0][at].as_ref(),
                (None, Some(at)) => rewritten[1][at].as_ref(),
                (None, None) => None,
            }
        };
        for expr in plan
            .computed_order_by
            .iter()
            .chain(&plan.computed_distinct_on)
        {
            let rewritten = rewritten_key(expr);
            // A key holding subqueries that is a SELECT item reads its value,
            // so the subqueries run once per row
            let key = ExprKey::of(expr);
            let item = mentions_subquery(expr)
                .then(|| {
                    select_exprs
                        .iter()
                        .zip(&item_rewritten)
                        .position(|(e, r)| r.is_some() && ExprKey::of(unaliased(e)) == key)
                })
                .flatten();
            outputs.push(match item {
                Some(position) => Output::Copy(position),
                None => Output::Key(Computed::new(compile(expr, rewritten)?)),
            });
        }
        // A key reading a SELECT alias reads the value the projection gave
        // it, in a column named after the key's identity, as the sort finds it
        let mut extra_names = plan.extra_names;
        let mut extra_sources = plan.extra_sources;
        // Each alias reference is bound to its position by node, not by name,
        // so no column name can meet it; a repeated alias is its last use,
        // as the general path's evaluator over the output columns reads it
        let mut positions: FxHashMap<&str, u16> = FxHashMap::default();
        for (i, e) in select_exprs.iter().enumerate() {
            if let Expression::Aliased(a) = e {
                if alias_map.contains_key(a.alias.value_lower.as_str()) {
                    let i = u16::try_from(i)
                        .map_err(|_| Error::internal("too many SELECT expressions"))?;
                    positions.insert(a.alias.value_lower.as_str(), i);
                }
            }
        }
        let keys: Vec<&Expression> = sort_keys()
            .filter(|key| !is_leaf(key) && names_any(key, &|name| positions.contains_key(name)))
            .collect();
        // Each key's columns bound, its aliases left for their positions
        let is_alias = |id: &Identifier| positions.contains_key(id.value_lower.as_str());
        let bound_keys = keys
            .iter()
            .map(|key| bind_own(chain, key, &is_alias))
            .collect::<Result<Vec<_>>>()?;
        let mut bindings = FxHashMap::default();
        for key in &bound_keys {
            bind_aliases(key, &positions, &mut bindings);
        }
        let alias_ctx = CompileContext::bound(registry).with_row2_bindings(&bindings);
        for (key, bound_key) in keys.into_iter().zip(&bound_keys) {
            let name = ExprKey::of(key).hidden_column_name();
            if extra_names.contains(&name) {
                continue;
            }
            let items = positions
                .iter()
                .filter(|(alias, _)| names_any(key, &|name| name == **alias))
                .map(|(_, position)| usize::from(*position))
                .collect();
            outputs.push(Output::AliasKey(
                Computed::new(compile_with(&alias_ctx, bound_key)?),
                items,
            ));
            extra_names.push(name);
            extra_sources.push(ColumnSource {
                item: None,
                input: None,
            });
        }
        let whole_row = outputs.len() == columns.len()
            && outputs
                .iter()
                .enumerate()
                .all(|(i, output)| matches!(output, Output::Column(c) if *c == i));
        let column_outputs = outputs
            .iter()
            .map(|output| match output {
                Output::Column(c) => Some(*c),
                _ => None,
            })
            .collect();
        let mut slots = 0;
        for output in &mut outputs {
            if let Output::Program(computed)
            | Output::Key(computed)
            | Output::Block(computed)
            | Output::AliasKey(computed, _) = output
            {
                if Computed::binds(&computed.program) {
                    computed.slot = Some(slots);
                    slots += 1;
                }
            }
        }
        let alias_keys = outputs.iter().any(|o| matches!(o, Output::AliasKey(..)));
        Ok((
            Self {
                outputs,
                whole_row,
                columns: column_outputs,
                slots,
                alias_keys,
            },
            extra_names,
            extra_sources,
        ))
    }

    /// Each row's output, its subqueries answered by the run's blocks
    pub(crate) fn project(&self, mut rows: RowVec, run: &PreparedRunner<'_>) -> Result<RowVec> {
        if self.whole_row {
            for (row_id, (id, _)) in rows.iter_mut().enumerate() {
                *id = row_id as i64;
            }
            return Ok(rows);
        }
        // Columns only: each row keeps them in its own storage, as the
        // general path's projection takes them
        if let Some(columns) = &self.columns {
            for (row_id, (id, row)) in rows.iter_mut().enumerate() {
                *row = std::mem::take(row).take_columns(columns);
                *id = row_id as i64;
            }
            return Ok(rows);
        }
        if rows.is_empty() {
            return Ok(rows);
        }
        let mut vm = ExprVM::new();
        let mut bound = self.unbound();
        // Each row's output takes its place
        for (row_id, (id, row)) in rows.iter_mut().enumerate() {
            *row = self.project_row::<false>(&mut vm, row, run, &[], &mut bound)?;
            *id = row_id as i64;
        }
        Ok(rows)
    }

    /// Whether a SELECT item is computed, which the sort can leave for later
    fn defers(&self) -> bool {
        self.outputs
            .iter()
            .any(|o| matches!(o, Output::Program(_) | Output::Block(_)))
    }

    /// The source row's column the output at `position` holds as it is
    fn source_column(&self, position: usize) -> Option<usize> {
        match self.outputs.get(position)? {
            Output::Column(column) => Some(*column),
            Output::Copy(item) => match self.outputs.get(*item)? {
                Output::Column(column) => Some(*column),
                _ => None,
            },
            _ => None,
        }
    }

    /// The output of the rows `order` keeps past `offset`, at most `limit`:
    /// the keys and the SELECT items they read are computed for every row,
    /// any other computed item only for a row kept, so it and its blocks
    /// are asked of the rows returned
    fn project_first(
        &self,
        mut rows: RowVec,
        run: &PreparedRunner<'_>,
        order: Vec<(Option<usize>, bool, Option<bool>)>,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<RowVec> {
        // The items the sort reads, directly, through a copying key or
        // through a key reading their aliases
        let mut read: SmallVec<[usize; 4]> = order.iter().filter_map(|(c, ..)| *c).collect();
        for output in &self.outputs {
            match output {
                Output::Copy(position) => read.push(*position),
                Output::AliasKey(_, items) => read.extend_from_slice(items),
                _ => {}
            }
        }
        let width = self.outputs.len();
        let mut vm = ExprVM::new();
        let mut sources = Vec::with_capacity(rows.len());
        let mut top = super::result::TopN::new(
            move |a: &Row, b: &Row| Executor::compare_rows_with_indices(a, b, &order),
            limit.unwrap_or(usize::MAX),
            offset,
        );
        // The storage of a row the sort does not keep holds the next row's keys
        let mut spare: Vec<Value> = Vec::new();
        let mut bound = self.unbound();
        for (i, row) in rows.drain_rows().enumerate() {
            let values = match spare.capacity() {
                0 => Vec::with_capacity(width + 1),
                _ => {
                    spare.clear();
                    std::mem::take(&mut spare)
                }
            };
            let mut out =
                self.project_into::<true>(&mut vm, &row, run, &read, &mut bound, values)?;
            // The row's place among the sources, past the outputs
            out.set(width, Value::Integer(i as i64))?;
            sources.push(row);
            if let Some(dropped) = top.push(out) {
                spare = dropped.into_values();
            }
        }
        let mut kept = RowVec::new();
        for mut out in top.into_rows() {
            let Some(Value::Integer(i)) = out.get(width).cloned() else {
                return Err(Error::internal("a sorted row lost its source"));
            };
            let row = &sources[i as usize];
            let exec = ExecuteContext::with_common_params(row, &run.base, run.transaction_id)
                .with_subquery_executor(run);
            for (position, output) in self.outputs.iter().enumerate() {
                if let Output::Program(computed) | Output::Block(computed) = output {
                    if !read.contains(&position) {
                        let item = Self::output_item(run, &mut bound, computed);
                        out.set(position, item.value(&mut vm, &exec)?)?;
                    }
                }
            }
            out.truncate(width);
            kept.push((kept.len() as i64, out));
        }
        Ok(kept)
    }

    /// One row's output. In the `FIRST` phase of a sort, a computed SELECT
    /// item that `read` does not hold is left NULL, and one more column
    /// follows for the row's place
    #[inline]
    fn project_row<const FIRST: bool>(
        &self,
        vm: &mut ExprVM,
        row: &Row,
        run: &PreparedRunner<'_>,
        read: &[usize],
        bound: &mut OutputsBound,
    ) -> Result<Row> {
        let values = Vec::with_capacity(self.outputs.len() + usize::from(FIRST));
        self.project_into::<FIRST>(vm, row, run, read, bound, values)
    }

    /// [`Self::project_row`] built in `values`, an empty buffer whose room
    /// a caller may keep across rows
    #[inline]
    fn project_into<const FIRST: bool>(
        &self,
        vm: &mut ExprVM,
        row: &Row,
        run: &PreparedRunner<'_>,
        read: &[usize],
        bound: &mut OutputsBound,
        mut values: Vec<Value>,
    ) -> Result<Row> {
        let exec = ExecuteContext::with_common_params(row, &run.base, run.transaction_id)
            .with_subquery_executor(run);
        for (position, output) in self.outputs.iter().enumerate() {
            let value = match output {
                Output::Column(i) => row.get(*i).cloned().unwrap_or_else(Value::null_unknown),
                Output::Program(_) | Output::Block(_) if FIRST && !read.contains(&position) => {
                    Value::null_unknown()
                }
                Output::Program(computed) | Output::Key(computed) | Output::Block(computed) => {
                    Self::output_item(run, bound, computed).value(vm, &exec)?
                }
                Output::AliasKey(..) => Value::null_unknown(),
                Output::Copy(item) => values
                    .get(*item)
                    .cloned()
                    .unwrap_or_else(Value::null_unknown),
            };
            values.push(value);
        }
        if FIRST {
            values.push(Value::null_unknown());
        }
        let mut out = Row::from_values(values);
        if !self.alias_keys {
            return Ok(out);
        }
        for (position, output) in self.outputs.iter().enumerate() {
            if let Output::AliasKey(computed, _) = output {
                let mut exec =
                    ExecuteContext::with_common_params(row, &run.base, run.transaction_id);
                exec.row2 = Some(&out);
                let key = vm.execute(Self::bound_program(run, bound, computed), &exec)?;
                out.set(position, key)?;
            }
        }
        Ok(out)
    }

    /// No output bound yet, for one run: a slot for each output a run binds
    fn unbound(&self) -> OutputsBound {
        (0..self.slots).map(|_| None).collect()
    }

    /// The item of `computed`, bound to `run` when a run binds it
    #[inline]
    fn output_item<'b>(
        run: &PreparedRunner<'_>,
        bound: &'b mut OutputsBound,
        computed: &'b Computed,
    ) -> &'b Item {
        match computed.slot.and_then(|slot| bound.get_mut(slot)) {
            Some(slot) => &BoundOutput::current(slot, run, &computed.program).item,
            None => &computed.item,
        }
    }

    /// The program of `computed`, bound to `run` when a run binds it
    #[inline]
    fn bound_program<'b>(
        run: &PreparedRunner<'_>,
        bound: &'b mut OutputsBound,
        computed: &'b Computed,
    ) -> &'b SharedProgram {
        match computed.slot.and_then(|slot| bound.get_mut(slot)) {
            Some(slot) => {
                &BoundOutput::current(slot, run, &computed.program)
                    .bound
                    .program
            }
            None => &computed.program,
        }
    }
}

/// An output's program bound to one run, and the item the bound program reads
struct BoundOutput {
    bound: Bound,
    item: Item,
}

impl BoundOutput {
    /// `slot`'s output bound to `run`, bound first from `program` when no row
    /// computed it yet and again when a block it asks was read since
    #[inline]
    fn current<'s>(
        slot: &'s mut Option<Self>,
        run: &PreparedRunner<'_>,
        program: &SharedProgram,
    ) -> &'s Self {
        // A binding that waits for no block stays as it is for the run
        if !slot.as_ref().is_some_and(|o| o.bound.pending.is_empty()) {
            Self::refresh(slot, run, program);
        }
        slot.get_or_insert_with(|| Self::bind(run, program))
    }

    /// [`Self::current`] binding `slot`'s output for the first time or again
    #[inline(never)]
    fn refresh(slot: &mut Option<Self>, run: &PreparedRunner<'_>, program: &SharedProgram) {
        let output = slot.get_or_insert_with(|| Self::bind(run, program));
        if run.rebound(&mut output.bound) {
            output.item = Item::of(CompactArc::clone(&output.bound.program));
        }
    }

    /// `program` bound to all `run` knows now, and the item it reads
    fn bind(run: &PreparedRunner<'_>, program: &SharedProgram) -> Self {
        let bound = run.bind(program);
        let item = Item::of(CompactArc::clone(&bound.program));
        Self { bound, item }
    }
}

/// Each output's program bound to one run, once a row computes it
type OutputsBound = SmallVec<[Option<BoundOutput>; 4]>;

impl PreparedSelect {
    /// Prepare `stmt` over its table, named `scope_name` and holding
    /// `columns`, for executions with `param_count` positional parameters
    fn prepare(
        executor: &Executor,
        ctx: &ExecutionContext,
        stmt: &SelectStatement,
        schema: &Schema,
        scope_name: &SmartString,
        param_count: usize,
        epoch: u64,
    ) -> Result<Self> {
        let registry = &executor.function_registry;
        let scope = ScopeTable::of(scope_name.clone(), schema);
        let scopes = [&scope];
        let chain = Chain { scopes: &scopes };
        let mut builder = Builder {
            executor,
            ctx,
            next_slot: param_count,
            subqueries: Vec::new(),
            max_param: 0,
            keeps_holes: false,
        };
        let mut work = Rewritten::default();
        let mut run_slots = Vec::new();
        let parts = match stmt.where_clause.as_deref() {
            Some(where_clause) => {
                let where_clause = with_aliases(where_clause, &stmt.columns, &scope.columns);
                builder.statement_where(&chain, schema, &where_clause, &mut work, &mut run_slots)?
            }
            // No conjunct: the scan reads every row with no filter, and takes it whole
            None => WhereParts {
                clause: true_literal(),
                pushed: Vec::new(),
                scanned: None,
                narrowed: Vec::new(),
                set_tests: Vec::new(),
                conjuncts: 0,
            },
        };
        // The SELECT items and ORDER BY keys holding subqueries, rewritten for
        // the blocks; an uncorrelated value they read is a run slot, read once
        // before the statement reads a row
        let first_output_block = builder.subqueries.len();
        let mut output = |expr: &Expression| -> Result<Option<Expression>> {
            if !mentions_subquery(expr) {
                return Ok(None);
            }
            let rewritten = builder.rewrite(&chain, None, expr, &mut Rewritten::default())?;
            builder.slotted(&rewritten, &mut run_slots).map(Some)
        };
        let select = stmt
            .columns
            .iter()
            .map(|column| output(unaliased(column)))
            .collect::<Result<Vec<_>>>()?;
        // A key that is a SELECT item holding subqueries reads that item,
        // so its blocks and run slots are the item's
        let repeats = |key: &Expression| {
            mentions_subquery(key)
                && stmt
                    .columns
                    .iter()
                    .any(|c| ExprKey::of(unaliased(c)) == ExprKey::of(key))
        };
        let order_by = stmt
            .order_by
            .iter()
            .map(|ob| match repeats(&ob.expression) {
                true => Ok(None),
                false => output(&ob.expression),
            })
            .collect::<Result<Vec<_>>>()?;
        let distinct_on = stmt
            .distinct_on
            .iter()
            .map(|key| match repeats(key) {
                true => Ok(None),
                false => output(key),
            })
            .collect::<Result<Vec<_>>>()?;
        let output_blocks = first_output_block..builder.subqueries.len();
        let mut read_as = FxHashMap::default();
        parts.read_as(schema, &mut read_as);
        let own = CompileContext::bound(registry)
            .with_prepared_subqueries()
            .with_read_as(&read_as);
        let compile = |expr: &Expression| compile_with(&own, expr);
        let where_programs = parts.programs(compile)?;
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile(value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let (projection, extra_names, extra_sources) = PreparedProjection::prepare(
            executor,
            stmt,
            &chain,
            [&select, &order_by, &distinct_on],
        )?;
        for column in stmt
            .columns
            .iter()
            .chain(stmt.order_by.iter().map(|ob| &ob.expression))
            .chain(&stmt.distinct_on)
        {
            builder.note_params(unaliased(column));
        }
        let (blocks, holes) = builder.finish(param_count, run_slots)?;
        let ahead = output_blocks
            .filter(|&id| {
                let sq = &blocks.subqueries[id];
                sq.parent.is_none() && sq.member_set_test().is_some()
            })
            .map(|id| u16::try_from(id).map_err(|_| Error::internal("too many subqueries")))
            .collect::<Result<Vec<_>>>()?;
        // Over one table `t.*` is `*`, the projection's expansion; the
        // names' prefix rule for a joined `t.*` would drop columns
        let select: std::borrow::Cow<[Expression]> = if stmt
            .columns
            .iter()
            .any(|c| matches!(c, Expression::QualifiedStar(_)))
        {
            stmt.columns
                .iter()
                .map(|c| match c {
                    Expression::QualifiedStar(star) => Expression::Star(StarExpression {
                        token: star.token.clone(),
                    }),
                    other => other.clone(),
                })
                .collect()
        } else {
            std::borrow::Cow::Borrowed(&stmt.columns)
        };
        let mut output_columns =
            executor.get_output_column_names(&select, &scope.columns, Some(scope_name));
        let visible = output_columns.len();
        output_columns.extend(extra_names);
        // The layout an ORDER BY or DISTINCT ON binds its names by, built
        // with the plan
        let layout = (!stmt.order_by.is_empty() || !stmt.distinct_on.is_empty()).then(|| {
            let mut layout = Executor::select_layout(&select, &scope.columns, Some(scope_name));
            layout.sources.extend(extra_sources);
            CompactArc::new(layout)
        });
        let where_ = parts.prepare(where_programs, &blocks, schema);
        Ok(Self {
            cached_epoch: epoch,
            columns: scope.columns.clone(),
            blocks,
            holes,
            where_,
            projection,
            ahead,
            output_columns: CompactArc::new(output_columns),
            visible,
            layout,
        })
    }
}

/// A grouped or windowed statement whose clauses hold subqueries, prepared
/// once as blocks of its rows: the statement the grouped and window paths
/// run, with those clauses rewritten and each value read once a run slot
#[derive(Debug)]
pub struct PreparedGrouped {
    cached_epoch: u64,
    /// The table's schema the clauses were prepared over, or the columns of
    /// the rows of another source
    schema: CompactArc<Schema>,
    source: GroupedSource,
    blocks: PreparedBlocks,
    /// Each block's holes: an input block's read from the row it is asked
    /// of, any other's from the GROUP BY items of its group's row
    holes: HoleSet,
    /// The uncorrelated member blocks past its WHERE, read before any row;
    /// the WHERE's are read by the first row that reaches them
    ahead: Vec<u16>,
    /// A WHERE holding subqueries, which the table's rows pass before they group
    where_: Option<PreparedWhere>,
    pub(crate) stmt: SelectStatement,
    pub(crate) classification: Arc<super::query_classification::QueryClassification>,
    /// An argument read once for its call holds a run slot, which the
    /// statement's parameters then carry beside the blocks'
    pub(crate) reads_slots: bool,
    /// Every subquery is a run slot: the statement runs as any grouped
    /// statement does, reading them as parameters, under this plan of its
    /// own over a table, and groups another source's rows so
    pub(crate) runs_alone: Option<NestedPlan>,
}

/// What a grouped statement's rows come from, which says what its blocks run
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedSource {
    /// A table, whose scan answers a WHERE without subqueries
    Table,
    /// Another source's rows, whose WHERE the blocks run whole
    Rows,
    /// A join's rows, which its WHERE has already passed
    Joined,
}

/// One execution of a grouped statement of the family: its plan, and the
/// rows of a source that is not a base table with the name its scope bears
pub(crate) struct GroupedRun<'s> {
    /// None when the blocks cannot run the statement, whose source's rows,
    /// read already, the general path takes
    pub(crate) held: Option<Arc<PreparedGrouped>>,
    source: Option<(StatementSource<'s>, SmartString)>,
}

impl GroupedRun<'_> {
    /// The name the rows' scope bears, None over a base table
    pub(crate) fn scope_name(&self) -> Option<&str> {
        self.source.as_ref().map(|(_, name)| name.as_str())
    }

    /// The columns of the source's rows and each row the WHERE passes, read
    /// through the `run`'s blocks; every row without them. None over a base
    /// table
    pub(crate) fn source_rows(
        self,
        run: Option<&PreparedRunner<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<Option<(CompactArc<Vec<String>>, RowVec)>> {
        let Some((source, _)) = self.source else {
            return Ok(None);
        };
        let Some(columns) = source.columns() else {
            return Err(Error::internal(
                "a grouped statement read its table as rows",
            ));
        };
        let columns = CompactArc::new(columns.to_vec());
        let where_ = self.held.as_deref().and_then(|held| held.where_.as_ref());
        let Some((run, where_)) = run.zip(where_) else {
            let rows = match source {
                StatementSource::Rows(_, rows) => rows,
                StatementSource::Values(_, values) => (0i64..)
                    .zip(&values.rows)
                    .map(|(id, exprs)| Executor::values_row(exprs, ctx).map(|row| (id, row)))
                    .collect::<Result<_>>()?,
                StatementSource::Cte((_, rows)) => rows.iter().cloned().collect(),
                StatementSource::Table(_) => {
                    return Err(Error::internal(
                        "a grouped statement read its table as rows",
                    ))
                }
            };
            return Ok(Some((columns, rows)));
        };
        let mut rows = RowVec::new();
        let keep = |id: i64, _: u64, row: Row| {
            rows.push((id, row));
            true
        };
        let read = match source {
            StatementSource::Rows(_, source) => {
                run.rows_passing(where_, source.into_iter().map(Ok), ctx, None, keep)
            }
            StatementSource::Values(_, values) => run.rows_passing(
                where_,
                (0i64..)
                    .zip(&values.rows)
                    .map(|(id, exprs)| Executor::values_row(exprs, ctx).map(|row| (id, row))),
                ctx,
                None,
                keep,
            ),
            StatementSource::Cte((_, source)) => run.rows_passing(
                where_,
                source.iter().map(|(id, row)| Ok((*id, row))),
                ctx,
                None,
                keep,
            ),
            StatementSource::Table(_) => {
                return Err(Error::internal(
                    "a grouped statement read its table as rows",
                ))
            }
        };
        read?;
        Ok(Some((columns, rows)))
    }
}

impl PreparedGrouped {
    /// Prepare `stmt` over the `scope` of its `source`'s rows, whose columns
    /// `schema` holds; None when a block reads a column of the row that is
    /// not a GROUP BY item
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        executor: &Executor,
        ctx: &ExecutionContext,
        stmt: &SelectStatement,
        scope: ScopeTable,
        schema: CompactArc<Schema>,
        source: GroupedSource,
        param_count: usize,
        epoch: u64,
    ) -> Result<Option<Self>> {
        // A DISTINCT ON key rides along as the hidden item it runs with
        let keyed = Executor::with_distinct_on_items(stmt, &get_classification(stmt));
        let stmt = keyed.as_ref().unwrap_or(stmt);
        let written = stmt;
        let scopes = [&scope];
        let chain = Chain { scopes: &scopes };
        let mut builder = Builder {
            executor,
            ctx,
            next_slot: param_count,
            subqueries: Vec::new(),
            max_param: 0,
            keeps_holes: true,
        };
        let mut run_slots = Vec::new();
        fn rewrite(
            builder: &mut Builder<'_>,
            chain: &Chain<'_>,
            run_slots: &mut Vec<(usize, Expression)>,
            expr: &Expression,
        ) -> Result<Expression> {
            let rewritten = builder.rewrite_blocks(chain, expr)?;
            builder.slotted(&rewritten, run_slots)
        }
        // What each input row reads comes first: its blocks' holes read the row
        let mut work = Rewritten::default();
        let where_parts = match stmt.where_clause.as_deref() {
            Some(where_clause)
                if match source {
                    GroupedSource::Table => mentions_subquery(where_clause),
                    GroupedSource::Rows => true,
                    GroupedSource::Joined => false,
                } =>
            {
                let where_clause = with_aliases(where_clause, &stmt.columns, &scope.columns);
                Some(builder.statement_where(
                    &chain,
                    &schema,
                    &where_clause,
                    &mut work,
                    &mut run_slots,
                )?)
            }
            _ => None,
        };
        // The WHERE's blocks come first; a row that reaches one reads it
        let where_blocks = builder.subqueries.len();
        let opened_having = match stmt.having.as_deref() {
            Some(having) if mentions_subquery(having) => {
                Some(grouped_having(executor, stmt, having)?)
            }
            _ => None,
        };
        // Whether each block is asked of an input row, the WHERE's so far
        let mut input = vec![true; builder.subqueries.len()];
        fn input_rewrite(
            builder: &mut Builder<'_>,
            chain: &Chain<'_>,
            run_slots: &mut Vec<(usize, Expression)>,
            input: &mut Vec<bool>,
            expr: &Expression,
        ) -> Result<Expression> {
            input.resize(builder.subqueries.len(), false);
            let rewritten = rewrite(builder, chain, run_slots, expr)?;
            input.resize(builder.subqueries.len(), true);
            Ok(rewritten)
        }
        // The keys are refused as every grouping path refuses them; a SELECT
        // item holding a subquery that one names by alias or position groups
        let group_items = executor.parse_group_by(stmt, &[])?;
        let mut keys = vec![false; stmt.columns.len()];
        for key in grouping_items(stmt) {
            if let Some(at) = grouped_item(stmt, key) {
                let item = unaliased(&stmt.columns[at]);
                keys[at] |= mentions_subquery(item) && !expression_contains_aggregate(item);
            }
        }
        // A grouped SELECT item or GROUP BY expression holding a subquery is
        // rewritten where it is written; a SELECT item or HAVING written as
        // one reads the first of that identity
        let mut items = FxHashMap::default();
        let mut grouped = vec![None; stmt.columns.len()];
        for ((column, &key), slot) in stmt.columns.iter().zip(&keys).zip(&mut grouped) {
            if key {
                let item = unaliased(column);
                let rewritten =
                    input_rewrite(&mut builder, &chain, &mut run_slots, &mut input, item)?;
                items
                    .entry(ExprKey::of(item))
                    .or_insert_with(|| rewritten.clone());
                *slot = Some(rewritten);
            }
        }
        let mut group_item = |item: &Expression| -> Result<Expression> {
            if !mentions_subquery(item) {
                return Ok(item.clone());
            }
            let rewritten = input_rewrite(&mut builder, &chain, &mut run_slots, &mut input, item)?;
            Ok(items
                .entry(ExprKey::of(item))
                .or_insert_with(|| rewritten.clone())
                .clone())
        };
        let group_by = match &stmt.group_by.modifier {
            GroupByModifier::GroupingSets(sets) => GroupByClause::grouping_sets(
                sets.iter()
                    .map(|set| set.iter().map(&mut group_item).collect())
                    .collect::<Result<_>>()?,
            ),
            modifier => GroupByClause {
                columns: stmt
                    .group_by
                    .columns
                    .iter()
                    .map(&mut group_item)
                    .collect::<Result<_>>()?,
                modifier: modifier.clone(),
            },
        };
        // An item the GROUP BY names is read for each input row, as rewritten
        // above, and so is every item of rows that do not group; any other
        // for each group
        let groups = grouped_block(stmt);
        let mut output = |expr: &Expression, key: Option<&Expression>| -> Result<Expression> {
            if let Some(rewritten) = key {
                return Ok(rewritten.clone());
            }
            if !groups {
                return input_rewrite(&mut builder, &chain, &mut run_slots, &mut input, expr);
            }
            let read = read_inputs(expr, &items, &mut |call| {
                input_rewrite(&mut builder, &chain, &mut run_slots, &mut input, call)
            })?;
            rewrite(&mut builder, &chain, &mut run_slots, &read)
        };
        // An item holding a subquery keeps the name it is written under
        let mut columns = Vec::with_capacity(stmt.columns.len());
        for (column, key) in stmt.columns.iter().zip(&grouped) {
            let key = key.as_ref();
            columns.push(match column {
                column if !mentions_subquery(unaliased(column)) => column.clone(),
                Expression::Aliased(aliased) => Expression::Aliased(AliasedExpression {
                    token: aliased.token.clone(),
                    expression: Box::new(output(&aliased.expression, key)?),
                    alias: aliased.alias.clone(),
                }),
                column => {
                    let name = item_name(column);
                    Expression::Aliased(AliasedExpression {
                        token: Token::new(TokenType::Identifier, &name, Position::default()),
                        expression: Box::new(output(column, key)?),
                        alias: Identifier::new(
                            Token::new(TokenType::Identifier, &name, Position::default()),
                            &name,
                        ),
                    })
                }
            });
        }
        let having = match (&opened_having, stmt.having.as_deref()) {
            (Some(opened), _) => Some(Box::new(output(opened, None)?)),
            (None, other) => other.map(|having| Box::new(having.clone())),
        };
        // An ORDER BY key holding a subquery is computed for each group into a
        // column named by its identity, which the sort reads as the key
        let mut order_by = stmt.order_by.clone();
        for ob in order_by
            .iter_mut()
            .filter(|_| stmt.set_operations.is_empty())
        {
            if !mentions_subquery(&ob.expression) {
                continue;
            }
            let name = ExprKey::of(&ob.expression).hidden_column_name();
            let token = Token::new(TokenType::Identifier, &name, Position::default());
            let column = Identifier::new(token.clone(), &name);
            if !columns
                .iter()
                .any(|c| matches!(c, Expression::Aliased(a) if a.alias.value == column.value))
            {
                columns.push(Expression::Aliased(AliasedExpression {
                    token,
                    expression: Box::new(output(&ob.expression, None)?),
                    alias: column.clone(),
                }));
            }
            ob.expression = Expression::Identifier(column);
        }
        let blocks_read = builder.subqueries.len();
        let values = CompileContext::bound(&executor.function_registry).with_prepared_subqueries();
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile_with(&values, value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let mut read_as = FxHashMap::default();
        if let Some(parts) = &where_parts {
            parts.read_as(&schema, &mut read_as);
        }
        let own = CompileContext::bound(&executor.function_registry)
            .with_prepared_subqueries()
            .with_read_as(&read_as);
        let where_program = where_parts
            .as_ref()
            .map(|parts| parts.programs(|expr| compile_with(&own, expr)))
            .transpose()?;
        let (blocks, mut row_holes) = builder.finish(param_count, run_slots)?;
        // The statement the grouped run reads names its columns again
        let where_clause = where_parts
            .as_ref()
            .map(|parts| spelled_columns(&parts.clause, &scope))
            .transpose()?;
        let where_ = where_parts
            .zip(where_program)
            .map(|(parts, program)| parts.prepare(program, &blocks, &schema));
        // A group's row holds its GROUP BY items first, each holding the
        // column its key binds to when it is one. A hole reads an item holding
        // the column it binds to: between several, the one the grouped output
        // reads for what the block wrote, by that output's own keys and item
        // names, which it binds by text until it reads bound columns
        let held: Vec<Option<usize>> = grouping_items(stmt)
            .into_iter()
            .map(|key| {
                let read = grouped_item(stmt, key).map_or(key, |at| unaliased(&stmt.columns[at]));
                match chain.resolve(read) {
                    Ok(Some((0, column))) => Some(column),
                    _ => None,
                }
            })
            .collect();
        let names = schema.column_names_arc();
        let item_names = executor.resolve_group_by_column_names_new(
            &group_items,
            &names,
            &build_column_index_map(&names),
        );
        let item_keys = aggregate_slot_keys(&group_items, &[], &stmt.columns);
        let item_ctx = CompileContext::new(&item_names, &executor.function_registry);
        let item_of = |column: usize, reference: &Expression| -> Option<usize> {
            let holding: SmallVec<[usize; 2]> = (0..held.len())
                .filter(|&item| held[item] == Some(column))
                .collect();
            let output = item_keys.get(&ExprKey::of(reference)).copied().or_else(|| {
                item_ctx
                    .reference(reference)
                    .and_then(|reference| reference.row_column())
                    .map(usize::from)
            });
            output
                .filter(|item| holding.contains(item))
                .or(holding.first().copied())
        };
        let slots_ctx = CompileContext::bound(&executor.function_registry);
        let mut holes = Vec::with_capacity(blocks.subqueries.len());
        for (id, sq) in blocks.subqueries.iter().enumerate() {
            if input.get(id) == Some(&true) {
                holes.push(std::mem::take(&mut row_holes.holes[id]));
                continue;
            }
            let mut grouped = Vec::with_capacity(sq.hole_written.len());
            for (slot, written) in &sq.hole_written {
                let read = match chain.resolve(written) {
                    Ok(Some((_, column))) => match item_of(column, written) {
                        Some(item) => Hole::Column(item),
                        None => return Ok(None),
                    },
                    _ => match over_items(written, &chain, &item_of)
                        .and_then(|expr| compile_with(&slots_ctx, &expr))
                    {
                        Ok(program) => Hole::Program(program),
                        Err(_) => return Ok(None),
                    },
                };
                grouped.push((*slot, read));
            }
            holes.push(grouped);
        }
        let ahead = (where_blocks..blocks_read)
            .filter(|&id| {
                let sq = &blocks.subqueries[id];
                sq.parent.is_none() && sq.member_set_test().is_some()
            })
            .map(|id| u16::try_from(id).map_err(|_| Error::internal("too many subqueries")))
            .collect::<Result<Vec<_>>>()?;
        let stmt = SelectStatement {
            columns,
            having,
            group_by,
            where_clause: where_clause
                .map(Box::new)
                .or_else(|| stmt.where_clause.clone()),
            order_by,
            ..stmt.clone()
        };
        // A set operation's first branch: the whole one reads the rest
        let stmt = match stmt.set_operations.is_empty() {
            true => stmt,
            false => SelectStatement {
                set_operations: Vec::new(),
                order_by: Vec::new(),
                limit: None,
                offset: None,
                ..stmt
            },
        };
        let runs_alone = (!stmt.columns.iter().any(|c| mentions_subquery(unaliased(c)))
            && !stmt.having.as_deref().is_some_and(mentions_subquery)
            && !grouping_items(&stmt).into_iter().any(mentions_subquery)
            && !stmt.where_clause.as_deref().is_some_and(mentions_subquery))
        .then(NestedPlan::default);
        // An argument read once for its call holds only run slots: one reading
        // the row is the general path's to refuse
        // The ORDER BY keys a hidden item computes count with the list
        let constants = |stmt: &SelectStatement, matches: &dyn Fn(&Expression) -> bool| {
            stmt.columns
                .iter()
                .chain(stmt.having.as_deref())
                .chain(stmt.order_by.iter().map(|ob| &ob.expression))
                .any(|e| constant_args_any(e, &matches))
        };
        if constants(&stmt, &mentions_subquery) {
            return Ok(None);
        }
        Ok(Some(Self {
            cached_epoch: epoch,
            schema,
            source,
            blocks,
            holes: HoleSet { holes },
            ahead,
            where_,
            reads_slots: constants(written, &mentions_subquery),
            classification: super::query_classification::get_classification(&stmt),
            stmt,
            runs_alone,
        }))
    }

    /// Whether the table's rows pass a WHERE holding subqueries before they group
    pub(crate) fn reads_through_where(&self) -> bool {
        self.where_.is_some()
    }

    /// The columns of table `name` and each of its rows the WHERE passes,
    /// read through the `run`'s blocks as a prepared statement reads them
    pub(crate) fn passing_rows(
        &self,
        executor: &Executor,
        run: &PreparedRunner<'_>,
        name: &str,
        ctx: &ExecutionContext,
    ) -> Result<(CompactArc<Vec<String>>, RowVec)> {
        let Some(where_) = &self.where_ else {
            return Err(Error::internal("a grouped statement without its WHERE"));
        };
        let table = executor.prepared_table(name, ctx, None)?;
        self.table_fits(&table.schema_arc())?;
        let mut rows = RowVec::new();
        let keep = |id: i64, _: u64, row: Row| {
            rows.push((id, row));
            true
        };
        match run.passing_rows(where_, None, table.as_ref(), ctx, None, false, keep)? {
            Passing::Pruned | Passing::Read => Ok((table.schema().column_names_arc(), rows)),
            Passing::Answers(_) | Passing::Scan(_) | Passing::Filtered(..) => Err(Error::internal(
                "a grouped statement read the scan's answers",
            )),
        }
    }

    /// SchemaChanged when `schema`, which the table's rows were read with,
    /// does not hold the columns the clauses were prepared over
    pub(crate) fn table_fits(&self, schema: &CompactArc<Schema>) -> Result<()> {
        if CompactArc::ptr_eq(&self.schema, schema) || self.schema.columns == schema.columns {
            return Ok(());
        }
        Err(Error::SchemaChanged {
            table: schema.table_name.clone(),
        })
    }
}

impl Executor {
    /// The plan's prepared grouped statement, with the rows of its source
    /// read now when that is not a base table; prepared now when the plan
    /// holds none for this schema, parameter count and source columns. None
    /// when the statement's blocks read a column outside its groups
    pub(crate) fn prepared_grouped<'s>(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        stmt: &'s SelectStatement,
        ctx: &'s ExecutionContext,
    ) -> Result<Option<GroupedRun<'s>>> {
        let epoch = self.engine.schema_epoch();
        let held = compiled
            .and_then(|slot| slot.read().ok())
            .and_then(|guard| match &*guard {
                CompiledExecution::PreparedGrouped(p)
                    if p.cached_epoch == epoch && p.blocks.param_count == ctx.params().len() =>
                {
                    Some(Arc::clone(p))
                }
                _ => None,
            });
        // A table held a plan for this epoch, so no view bears its name
        let base_table = match stmt.table_expr.as_deref() {
            Some(Expression::TableSource(ts)) if !ctx.has_cte_by_lower(&ts.name.value_lower) => {
                Some(ts)
            }
            _ => None,
        };
        if let Some(held) = held
            .as_ref()
            .filter(|p| p.source == GroupedSource::Table && base_table.is_some())
        {
            return Ok(Some(GroupedRun {
                held: Some(Arc::clone(held)),
                source: None,
            }));
        }
        let inner_ctx = ctx.with_incremented_query_depth();
        // A grouped statement reads every row
        let (source, scope_name) = self.read_source(stmt, ctx, &inner_ctx, None)?;
        let scope_name = scope_name.cloned().unwrap_or_default();
        let columns = source.as_ref().and_then(StatementSource::columns);
        if let Some(held) = held.filter(|p| {
            p.source == GroupedSource::Rows
                && columns.is_some_and(|columns| *p.schema.column_names_arc() == columns)
        }) {
            return Ok(Some(GroupedRun {
                held: Some(held),
                source: source.map(|source| (source, scope_name)),
            }));
        }
        let (schema, kind) = match (columns, base_table) {
            (Some(columns), _) => (
                CompactArc::new(untyped_schema(&scope_name, columns)),
                GroupedSource::Rows,
            ),
            (None, Some(ts)) => (
                self.prepared_table(&ts.name.value_lower, ctx, None)?
                    .schema_arc(),
                GroupedSource::Table,
            ),
            (None, None) => return Err(Error::internal("a grouped statement without one table")),
        };
        let scope = ScopeTable::of(scope_name.clone(), &schema);
        let Some(prepared) = PreparedGrouped::prepare(
            self,
            ctx,
            stmt,
            scope,
            schema,
            kind,
            ctx.params().len(),
            epoch,
        )?
        else {
            // Rows read already go to the general path, which reads no source again
            return Ok(source.map(|source| GroupedRun {
                held: None,
                source: Some((source, scope_name)),
            }));
        };
        let prepared = Arc::new(prepared);
        if let Some(mut guard) = compiled.and_then(|slot| slot.write().ok()) {
            if matches!(
                &*guard,
                CompiledExecution::Unknown
                    | CompiledExecution::NotOptimizable(_)
                    | CompiledExecution::PreparedGrouped(_)
            ) {
                *guard = CompiledExecution::PreparedGrouped(Arc::clone(&prepared));
            }
        }
        Ok(Some(GroupedRun {
            held: Some(prepared),
            source: source.map(|source| (source, scope_name)),
        }))
    }

    /// A grouped join statement prepared over the joined `columns` its join
    /// path laid out from its `leaves` and `sides`, as `of_join_columns`
    /// takes them; None when they do not line up or a block reads a column
    /// outside its groups
    pub(crate) fn prepare_grouped_join(
        &self,
        stmt: &SelectStatement,
        columns: &[String],
        (leaves, sides): (&[JoinLeaf], (usize, usize)),
        ctx: &ExecutionContext,
    ) -> Result<Option<PreparedGrouped>> {
        let Some(scope) = ScopeTable::of_join_columns(columns, leaves, sides) else {
            return Ok(None);
        };
        PreparedGrouped::prepare(
            self,
            ctx,
            stmt,
            scope,
            CompactArc::new(untyped_schema("", columns)),
            GroupedSource::Joined,
            ctx.params().len(),
            self.engine.schema_epoch(),
        )
    }

    /// Whether the plan holds a grouped statement prepared for the current
    /// schema, which puts the statement in the family without walking it
    pub(crate) fn holds_prepared_grouped(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        ctx: &ExecutionContext,
    ) -> bool {
        compiled
            .and_then(|slot| slot.read().ok())
            .is_some_and(|guard| match &*guard {
                CompiledExecution::PreparedGrouped(p) => {
                    p.cached_epoch == self.engine.schema_epoch()
                        && p.blocks.param_count == ctx.params().len()
                        && ctx.outer_row().is_none()
                }
                CompiledExecution::PreparedJoin(p) => {
                    p.cached_epoch == self.engine.schema_epoch()
                        && p.holds_grouped()
                        && ctx.outer_row().is_none()
                }
                _ => false,
            })
    }

    /// One execution's run of a grouped statement's blocks, each value read
    /// once and each set the output tests read before any row; None when a
    /// table they read no longer has the columns they were prepared for
    pub(crate) fn grouped_runner<'a>(
        &'a self,
        held: &'a PreparedGrouped,
        ctx: &'a ExecutionContext,
    ) -> Result<Option<PreparedRunner<'a>>> {
        let inner_ctx = ctx.with_incremented_query_depth();
        let Some(tables) = PreparedRunner::open_tables(self, &held.blocks, &inner_ctx, None)?
        else {
            return Ok(None);
        };
        let mut runner = PreparedRunner::new(
            self,
            &held.blocks,
            &held.holes,
            ctx,
            inner_ctx,
            tables,
            None,
            None,
        );
        for (slot, value) in runner.run_slot_values()? {
            runner.base[slot] = value;
        }
        runner.read_ahead(&held.ahead, &held.holes)?;
        Ok(Some(runner))
    }
}

/// The WHERE and SET of an UPDATE or DELETE, or the VALUES of an INSERT,
/// whose subqueries run prepared, compiled against its target's columns over
/// one set of blocks
#[derive(Debug)]
pub struct PreparedDml {
    cached_epoch: u64,
    /// The target's columns at preparation
    columns: CompactArc<Vec<String>>,
    blocks: PreparedBlocks,
    holes: HoleSet,
    where_: Option<PreparedWhere>,
    /// Each SET value with the column it writes
    sets: Vec<(usize, SharedProgram)>,
    /// Each VALUES row's programs, None for a value without a subquery
    values: Vec<Vec<Option<SharedProgram>>>,
    /// An INSERT's ON CONFLICT DO UPDATE, None without a subquery there
    conflict: Option<PreparedConflict>,
    /// The RETURNING, None without a subquery there
    returning: Option<PreparedReturning>,
}

/// A statement's RETURNING values over the row it returns, one program for
/// each column of its result
#[derive(Debug)]
struct PreparedReturning {
    /// The target's columns the row returned is laid out for
    columns: CompactArc<Vec<String>>,
    /// The names of the result's columns, shared by every execution
    names: CompactArc<Vec<String>>,
    programs: Vec<SharedProgram>,
    /// A value read for each row returned holds a block that reads the target
    reads_target: bool,
}

/// An INSERT's ON CONFLICT SET values and DO UPDATE WHERE, over the row
/// met joined with the excluded row
struct PreparedConflict {
    /// The target's columns the joined row is laid out for
    columns: CompactArc<Vec<String>>,
    upsert: Arc<super::dml::CompiledUpsert>,
    /// A value read for each conflict holds a block that reads the target
    reads_target: bool,
}

impl std::fmt::Debug for PreparedConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedConflict")
            .field("columns", &self.columns)
            .field("reads_target", &self.reads_target)
            .finish_non_exhaustive()
    }
}

/// The ON CONFLICT DO UPDATE of `stmt`, an INSERT into `table` of `schema`
pub(crate) struct ConflictSites<'e> {
    pub(crate) table: &'e SmartString,
    pub(crate) schema: &'e Schema,
    pub(crate) stmt: &'e InsertStatement,
}

/// The RETURNING `values` of a statement writing `table` of `schema`
pub(crate) struct ReturningSites<'e> {
    pub(crate) table: &'e str,
    pub(crate) schema: &'e Schema,
    pub(crate) values: &'e [Expression],
}

/// The values of a RETURNING the family walks: each without its alias, a
/// `*` left out
pub(crate) fn returning_values(returning: &[Expression]) -> impl Iterator<Item = &Expression> {
    returning
        .iter()
        .map(unaliased)
        .filter(|value| !matches!(value, Expression::Star(_)))
}

impl PreparedDml {
    /// Prepare the WHERE, `sets`, `values`, `conflict` and `returning` of a
    /// statement over its target `schema`, named `scope_name`, for executions with
    /// `param_count` parameters
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        executor: &Executor,
        ctx: &ExecutionContext,
        schema: &Schema,
        scope_name: &SmartString,
        where_clause: Option<&Expression>,
        sets: &[(usize, &Expression)],
        values: &[impl AsRef<[Expression]>],
        conflict: Option<&ConflictSites<'_>>,
        returning: Option<&ReturningSites<'_>>,
        param_count: usize,
        epoch: u64,
    ) -> Result<Self> {
        let scope = ScopeTable::of(scope_name.clone(), schema);
        let scopes = [&scope];
        let chain = Chain { scopes: &scopes };
        let mut builder = Builder {
            executor,
            ctx,
            next_slot: param_count,
            subqueries: Vec::new(),
            max_param: 0,
            keeps_holes: false,
        };
        let mut work = Rewritten::default();
        // A value that reads no row is computed once per run, as a run slot
        let mut run_slots = Vec::new();
        let where_parts = where_clause
            .map(|w| builder.statement_where(&chain, schema, w, &mut work, &mut run_slots))
            .transpose()?;
        let mut rewritten = Vec::with_capacity(sets.len());
        for (column, value) in sets {
            let mut value_work = Rewritten::default();
            let rewritten_value = builder.rewrite(&chain, None, value, &mut value_work)?;
            let value = if !value_work.reads_row
                && !is_leaf(value)
                && !volatile(value, &executor.function_registry)
            {
                let slot = builder.slot();
                run_slots.push((slot, rewritten_value));
                slot_parameter(slot)
            } else {
                rewritten_value
            };
            rewritten.push((*column, value));
        }
        let sets = rewritten;
        let mut rewritten_values = Vec::with_capacity(values.len());
        for row in values {
            let row = row.as_ref();
            let mut programs = Vec::with_capacity(row.len());
            for value in row {
                if matches!(value, Expression::Default(_)) {
                    programs.push(None);
                    continue;
                }
                let mut value_work = Rewritten::default();
                // Rewritten for its parameters too, so a slot never takes one not given
                let rewritten_value = builder.rewrite(&chain, None, value, &mut value_work)?;
                if !mentions_subquery(value) {
                    programs.push(None);
                    continue;
                }
                programs.push(Some(
                    if !value_work.reads_row
                        && !is_leaf(value)
                        && !volatile(value, &executor.function_registry)
                    {
                        let slot = builder.slot();
                        run_slots.push((slot, rewritten_value));
                        slot_parameter(slot)
                    } else {
                        rewritten_value
                    },
                ));
            }
            rewritten_values.push(programs);
        }
        // ON CONFLICT reads the row met and the excluded row, joined
        let joined = conflict.map(|sites| {
            let excluded = SmartString::from("excluded");
            // `excluded` names the excluded row, also when the target bears
            // that name; the target then has a name no qualifier can spell
            let target = if *sites.table == excluded {
                SmartString::from(format!("{RESOLVED}{excluded}"))
            } else {
                sites.table.clone()
            };
            // A bare name reads the row met; the excluded row only by its name
            let mut joined = ScopeTable::beside(
                ScopeTable::of(target, sites.schema),
                ScopeTable::of(excluded, sites.schema),
            )
            .by_position();
            joined.bare = sites.schema.columns.len();
            joined
        });
        let mut conflict_parts = None;
        if let (Some(sites), Some(joined)) = (conflict, &joined) {
            let scopes = [joined];
            let chain = Chain { scopes: &scopes };
            let mut reads_target = false;
            let mut per_value = |builder: &mut Builder<'_>, value: &Expression| {
                let first = builder.subqueries.len();
                let mut value_work = Rewritten::default();
                let rewritten_value = builder.rewrite(&chain, None, value, &mut value_work)?;
                if !value_work.reads_row
                    && !is_leaf(value)
                    && !volatile(value, &executor.function_registry)
                {
                    let slot = builder.slot();
                    run_slots.push((slot, rewritten_value));
                    return Ok(slot_parameter(slot));
                }
                // Read for each conflict, after the statement's earlier writes
                reads_target |= builder.subqueries[first..]
                    .iter()
                    .flatten()
                    .any(|sq| sq.may_read(sites.table));
                Ok::<_, Error>(rewritten_value)
            };
            let site_sets = Executor::conflict_sets(sites.schema, sites.stmt);
            let mut sets = Vec::with_capacity(site_sets.len());
            for (column, value) in site_sets {
                sets.push((column, per_value(&mut builder, value)?));
            }
            let where_ = sites
                .stmt
                .update_where
                .as_deref()
                .map(|clause| per_value(&mut builder, clause))
                .transpose()?;
            conflict_parts = Some((sets, where_, reads_target));
        }
        // RETURNING reads the row returned, `*` each of its columns
        let returning_scope =
            returning.map(|sites| ScopeTable::of(scope_name.clone(), sites.schema));
        let mut returning_parts = None;
        if let (Some(sites), Some(returning_scope)) = (returning, &returning_scope) {
            let scopes = [returning_scope];
            let chain = Chain { scopes: &scopes };
            let mut reads_target = false;
            let mut values = Vec::with_capacity(sites.values.len());
            for value in sites.values.iter().map(unaliased) {
                if matches!(value, Expression::Star(_)) {
                    values.extend((0..sites.schema.columns.len()).map(|i| chain.column(0, i)));
                    continue;
                }
                let first = builder.subqueries.len();
                let mut value_work = Rewritten::default();
                let rewritten_value = builder.rewrite(&chain, None, value, &mut value_work)?;
                if !value_work.reads_row
                    && !is_leaf(value)
                    && !volatile(value, &executor.function_registry)
                {
                    let slot = builder.slot();
                    run_slots.push((slot, rewritten_value));
                    values.push(slot_parameter(slot));
                    continue;
                }
                // Read for each row returned, after the statement's writes
                reads_target |= builder.subqueries[first..]
                    .iter()
                    .flatten()
                    .any(|sq| sq.may_read(sites.table));
                values.push(rewritten_value);
            }
            returning_parts = Some((values, reads_target));
        }
        let mut read_as = FxHashMap::default();
        if let Some(parts) = &where_parts {
            parts.read_as(schema, &mut read_as);
        }
        let own = CompileContext::bound(&executor.function_registry)
            .with_prepared_subqueries()
            .with_read_as(&read_as);
        let where_program = where_parts
            .as_ref()
            .map(|parts| parts.programs(|expr| compile_with(&own, expr)))
            .transpose()?;
        let sets = sets
            .iter()
            .map(|(column, value)| compile_with(&own, value).map(|p| (*column, p)))
            .collect::<Result<Vec<_>>>()?;
        let values = rewritten_values
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| value.as_ref().map(|v| compile_with(&own, v)).transpose())
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let prepared_conflict = match (conflict, &joined, conflict_parts) {
            (Some(sites), Some(_), Some((sets, where_, reads_target))) => {
                let over_joined =
                    CompileContext::bound(&executor.function_registry).with_prepared_subqueries();
                let updates = sets
                    .iter()
                    .map(|(index, value)| {
                        let column = &sites.schema.columns[*index];
                        compile_with(&over_joined, value)
                            .map(|p| (*index, column.data_type, column.vector_dimensions, p))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let where_ = where_
                    .as_ref()
                    .map(|clause| compile_with(&over_joined, clause))
                    .transpose()?;
                Some(PreparedConflict {
                    columns: sites.schema.column_names_arc(),
                    upsert: Arc::new(Executor::upsert_of(sites.schema, updates, where_)?),
                    reads_target,
                })
            }
            _ => None,
        };
        let prepared_returning = match (returning, &returning_scope, returning_parts) {
            (Some(sites), Some(_), Some((values, reads_target))) => {
                let over_returned =
                    CompileContext::bound(&executor.function_registry).with_prepared_subqueries();
                let columns = sites.schema.column_names_arc();
                Some(PreparedReturning {
                    names: CompactArc::new(Executor::returning_names(sites.values, &columns)),
                    columns,
                    programs: values
                        .iter()
                        .map(|value| compile_with(&over_returned, value))
                        .collect::<Result<Vec<_>>>()?,
                    reads_target,
                })
            }
            _ => None,
        };
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile_with(&own, value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let (blocks, holes) = builder.finish(param_count, run_slots)?;
        let where_ = where_parts
            .zip(where_program)
            .map(|(parts, program)| parts.prepare(program, &blocks, schema));
        Ok(Self {
            cached_epoch: epoch,
            columns: scope.columns.clone(),
            blocks,
            holes,
            where_,
            sets,
            values,
            conflict: prepared_conflict,
            returning: prepared_returning,
        })
    }
}

/// The subqueries of a join statement's ON and of the WHERE it asks of its
/// joined rows, prepared once over the joined scope as one set of blocks.
/// The clauses keep their shape, each column spelled by its position
#[derive(Debug)]
pub(crate) struct JoinBlocks {
    blocks: PreparedBlocks,
    /// The holes of sites over the joined row, and over a (left, right) pair
    joined: HoleSet,
    pair: HoleSet,
    pub(crate) on: Option<Expression>,
    pub(crate) cross: Option<Expression>,
    /// The WHERE's conditions on the left side alone, and on the right,
    /// over the joined row
    pub(crate) sides: [Option<Expression>; 2],
    /// The same conditions as written, which the sides' reads are given,
    /// when a subquery that reads the other side moved one to the joined row
    pub(crate) written: Option<[Option<Expression>; 2]>,
    /// The same conditions as their side's scan reads them
    side_wheres: [Option<PreparedWhere>; 2],
    /// The columns of each side the clauses were prepared over
    side_columns: [Vec<SchemaColumn>; 2],
    /// The schema each side was found to hold those columns with; held, any
    /// change to the table's schema replaces it, so the same one skips the check
    fitted: [std::sync::OnceLock<CompactArc<Schema>>; 2],
    /// Whether each side is read whole as rows, not scanned as a table
    pub(crate) rows_sides: [bool; 2],
    /// The plan of a side that is itself a join, which runs its ON's
    /// subqueries as its own blocks; None for a side that is one source
    pub(crate) nested: [Option<NestedPlan>; 2],
    /// The conditions on a side asked of the joined row with the WHERE's
    /// others there: one on a side read as rows that asks a block of each
    /// row, or one whose subquery reads the other side
    pub(crate) moved: Option<Expression>,
    /// Each SELECT item, ORDER BY key and DISTINCT ON key holding a
    /// subquery, over the joined row
    pub(crate) select: Vec<Option<Expression>>,
    pub(crate) order_by: Vec<Option<Expression>>,
    pub(crate) distinct_on: Vec<Option<Expression>>,
    /// The uncorrelated member blocks those read, read before the join
    /// reads a row; a value block they read is a run slot
    ahead: Vec<u16>,
    /// The scope's names for the left side's columns, then the right side's
    pub(crate) left: Vec<String>,
    pub(crate) right: Vec<String>,
    pub(crate) all: CompactArc<Vec<String>>,
    /// The joined scope, which the output's names bind in
    scope: ScopeTable,
}

impl JoinBlocks {
    /// SchemaChanged when `schema`, which side `side` was read with, does not
    /// hold the columns the clauses were prepared over
    pub(crate) fn side_fits(&self, side: usize, schema: &CompactArc<Schema>) -> Result<()> {
        if self.fitted[side]
            .get()
            .is_some_and(|fitted| CompactArc::ptr_eq(fitted, schema))
        {
            return Ok(());
        }
        if schema.columns == self.side_columns[side] {
            let _ = self.fitted[side].set(CompactArc::clone(schema));
            return Ok(());
        }
        Err(Error::SchemaChanged {
            table: schema.table_name.clone(),
        })
    }

    /// Whether the statement holds a condition on side `side` alone
    pub(crate) fn side_asks(&self, side: usize) -> bool {
        self.side_wheres[side].is_some()
    }

    /// The pairs a joined side merges, each its left and right column in the
    /// joined row, which its run keeps apart; the join's own pairs straddle
    /// the sides
    pub(crate) fn side_merges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let left_width = self.left.len();
        self.scope
            .merged
            .iter()
            .copied()
            .filter(move |&(left, right)| (left < left_width) == (right < left_width))
    }

    /// SchemaChanged when `columns`, the names side `side`, `name`, was read
    /// as rows with, are not the ones the clauses were prepared over
    fn rows_fit(&self, side: usize, name: &str, columns: &[String]) -> Result<()> {
        let prepared = &self.side_columns[side];
        // A joined side's rows hold its tables' columns in turn, each named
        // by its table and its own name
        let fits = |c: &SchemaColumn, n: &String| match self.nested[side] {
            Some(_) => bare_name(n).eq_ignore_ascii_case(&c.name),
            None => c.name == *n,
        };
        if prepared.len() == columns.len() && prepared.iter().zip(columns).all(|(c, n)| fits(c, n))
        {
            return Ok(());
        }
        Err(Error::SchemaChanged {
            table: name.to_string(),
        })
    }

    /// `join`'s `on`, `inner_on` when it is an INNER join's, `cross`, the
    /// conditions on one side, `side_filters`, and the SELECT items, ORDER BY
    /// keys and DISTINCT ON keys holding a subquery, `output`, over the
    /// joined row of `sides`, each a name, its columns' schema and whether it
    /// is read as rows; None when the join's `width` columns do not line up
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        executor: &Executor,
        ctx: &ExecutionContext,
        (join, on, inner_on): (&JoinTableSource, Option<&Expression>, bool),
        cross: Option<&Expression>,
        side_filters: [Option<&Expression>; 2],
        output: [&[Option<&Expression>]; 3],
        sides: &JoinSides,
        width: usize,
        param_count: usize,
    ) -> Result<Option<Self>> {
        let rows_sides = [sides[0].rows, sides[1].rows];
        let table = |(name, schema): &(SmartString, CompactArc<Schema>)| {
            ScopeTable::of(SmartString::from(name.to_lowercase().as_str()), schema)
        };
        // Each side's tables side by side, read by position in the joined row
        let side_scope = |expr: &Expression, side: &JoinSide| {
            ScopeTable::of_side(expr, &mut side.leaves.iter().map(table))
        };
        let (Some(left_scope), Some(right_scope)) = (
            side_scope(&join.left, &sides[0]),
            side_scope(&join.right, &sides[1]),
        ) else {
            return Ok(None);
        };
        let left_width = left_scope.columns.len();
        let scope = ScopeTable::joined(join, left_scope, right_scope).by_position();
        if scope.columns.len() != width {
            return Ok(None);
        }
        let scopes = [&scope];
        let chain = Chain { scopes: &scopes };
        let mut builder = Builder {
            executor,
            ctx,
            next_slot: param_count,
            subqueries: Vec::new(),
            max_param: 0,
            keeps_holes: false,
        };
        // A side read as rows keeps the conditions its read asks: those
        // without a block, which the read takes, and those whose blocks bind
        // before its first row; one that asks a block of each row is asked
        // of the joined row
        let mut kept: [Option<Expression>; 2] = [None, None];
        let mut written: [Option<Expression>; 2] = [None, None];
        let mut crossed = false;
        let mut moved = Vec::new();
        for (i, filter) in side_filters.into_iter().enumerate() {
            let Some(filter) = filter else {
                continue;
            };
            // A condition whose subquery reads the other side is the joined row's
            let mut parts = Vec::new();
            conjuncts(filter, &mut parts);
            let mut own = Vec::new();
            for part in parts {
                let other = 2 >> i;
                match mentions_subquery(part)
                    && sides_read_within(executor, ctx, part, &scope, left_width)? & other != 0
                {
                    true => {
                        crossed = true;
                        moved.push(part.clone())
                    }
                    false => own.push(part.clone()),
                }
            }
            let Some(filter) = conjunction(own) else {
                continue;
            };
            let filter = &*written[i].insert(filter);
            if !rows_sides[i] {
                kept[i] = Some(filter.clone());
                continue;
            }
            // Only a side of one source is given conditions of its own
            let Some(schema) = sides[i].schema() else {
                moved.push(filter.clone());
                continue;
            };
            let side = table(&sides[i].leaves[0]);
            let side_scopes = [&side];
            let side_chain = Chain {
                scopes: &side_scopes,
            };
            let (mut parts, mut binds) = (Vec::new(), Vec::new());
            conjuncts(filter, &mut parts);
            for part in parts.into_iter().filter(|part| mentions_subquery(part)) {
                match builder.binds_before_rows(&side_chain, schema, part)? {
                    true => binds.push(part.clone()),
                    false => moved.push(part.clone()),
                }
            }
            kept[i] = conjunction(binds);
        }
        let side_filters = [kept[0].as_ref(), kept[1].as_ref()];
        // An INNER join's ON asks them of each pair, as its WHERE would, so a
        // join that stops at its LIMIT still does; another join's WHERE does
        let moved = conjunction(moved);
        let with_moved = |clause: Option<&Expression>, moved: &Expression| {
            conjunction(clause.cloned().into_iter().chain([moved.clone()]).collect())
        };
        let (on, cross) = match moved.as_ref() {
            Some(moved) if inner_on => (with_moved(on, moved), cross.cloned()),
            Some(moved) => (on.cloned(), with_moved(cross, moved)),
            None => (on.cloned(), cross.cloned()),
        };
        let mut rewrite = |clause: Option<&Expression>| {
            clause
                .map(|clause| builder.rewrite(&chain, None, clause, &mut Rewritten::default()))
                .transpose()
        };
        let on = rewrite(on.as_ref())?;
        let cross = rewrite(cross.as_ref())?;
        // An uncorrelated value the output reads is a run slot, read once
        // before the join reads a row
        let mut run_slots = Vec::new();
        let first_output_block = builder.subqueries.len();
        let [select, order_by, distinct_on] = output.map(|exprs| {
            exprs
                .iter()
                .map(|expr| {
                    expr.map(|expr| {
                        let rewritten =
                            builder.rewrite(&chain, None, expr, &mut Rewritten::default())?;
                        builder.slotted(&rewritten, &mut run_slots)
                    })
                    .transpose()
                })
                .collect::<Result<Vec<_>>>()
        });
        let (select, order_by, distinct_on) = (select?, order_by?, distinct_on?);
        let output_blocks = first_output_block..builder.subqueries.len();
        // Each side's conditions as its own scan reads them, a value a column
        // is compared with a run slot; the joined row reads the same blocks
        let registry = &executor.function_registry;
        let mut side_parts = [None, None];
        for (i, filter) in side_filters.into_iter().enumerate() {
            let (Some(filter), Some(schema)) = (filter, sides[i].schema()) else {
                continue;
            };
            let side = table(&sides[i].leaves[0]);
            let side_scopes = [&side];
            let side_chain = Chain {
                scopes: &side_scopes,
            };
            let parts = builder.statement_where(
                &side_chain,
                schema,
                filter,
                &mut Rewritten::default(),
                &mut run_slots,
            )?;
            let mut read_as = FxHashMap::default();
            parts.read_as(schema, &mut read_as);
            let own = CompileContext::bound(registry)
                .with_prepared_subqueries()
                .with_read_as(&read_as);
            let program = parts.programs(|expr| compile_with(&own, expr))?;
            let offset = if i == 0 { 0 } else { left_width };
            let joined_clause = in_joined(&parts.clause, offset)?;
            side_parts[i] = Some((parts, program, joined_clause));
        }
        let values = CompileContext::bound(registry).with_prepared_subqueries();
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile_with(&values, value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let (blocks, joined) = builder.finish(param_count, run_slots)?;
        let ahead = output_blocks
            .filter(|&id| {
                let sq = &blocks.subqueries[id];
                sq.parent.is_none() && sq.member_set_test().is_some()
            })
            .map(|id| u16::try_from(id).map_err(|_| Error::internal("too many subqueries")))
            .collect::<Result<Vec<_>>>()?;
        let mut side_filters = [None, None];
        let mut side_wheres = [None, None];
        let side_columns = sides.each_ref().map(|side| {
            let columns = side.leaves.iter().flat_map(|(_, schema)| &schema.columns);
            columns.cloned().collect()
        });
        for (i, part) in side_parts.into_iter().enumerate() {
            if let (Some((parts, program, joined_clause)), Some(schema)) = (part, sides[i].schema())
            {
                side_wheres[i] = Some(parts.prepare(program, &blocks, schema));
                side_filters[i] = Some(joined_clause);
            }
        }
        let left = scope.columns[..left_width].to_vec();
        let right = scope.columns[left_width..].to_vec();
        let pair_ctx = CompileContext::bound_pair(registry, left_width as u32);
        let position = |hole: &Expression| match hole {
            Expression::Column(c) => Some(match (c.index as usize).checked_sub(left_width) {
                Some(i) => Hole::Column2(i),
                None => Hole::Column(c.index as usize),
            }),
            _ => None,
        };
        let pair = HoleSet {
            holes: blocks
                .subqueries
                .iter()
                .map(|sq| {
                    sq.hole_exprs
                        .iter()
                        .map(|(slot, hole)| match position(hole) {
                            Some(hole) => Ok((*slot, hole)),
                            None => {
                                compile_with(&pair_ctx, hole).map(|p| (*slot, Hole::Program(p)))
                            }
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?,
        };
        Ok(Some(Self {
            blocks,
            joined,
            pair,
            on,
            cross,
            sides: side_filters,
            written: crossed.then_some(written),
            side_wheres,
            side_columns,
            fitted: Default::default(),
            rows_sides,
            nested: sides
                .each_ref()
                .map(|side| (side.leaves.len() > 1).then(NestedPlan::default)),
            moved,
            select,
            order_by,
            distinct_on,
            ahead,
            left,
            right,
            all: CompactArc::clone(&scope.columns),
            scope,
        }))
    }

    /// `clause`, one of the rewritten clauses or a part of one, compiled
    /// over the joined row, or over the (left, right) pair when `pair`
    pub(crate) fn compile(
        &self,
        clause: &Expression,
        pair: bool,
        registry: &FunctionRegistry,
    ) -> Result<SharedProgram> {
        let ctx = match pair {
            true => CompileContext::bound_pair(registry, self.left.len() as u32),
            false => CompileContext::bound(registry),
        };
        compile_with(&ctx.with_prepared_subqueries(), clause)
    }

    /// The holes of sites over the joined row, or over the pair when `pair`
    pub(crate) fn holes(&self, pair: bool) -> &HoleSet {
        match pair {
            true => &self.pair,
            false => &self.joined,
        }
    }
}

impl Executor {
    /// Whether a join's WHERE asked of its joined rows holds subqueries
    /// only of the family's forms, or why not
    pub(crate) fn join_where_family(
        &self,
        clause: &Expression,
    ) -> std::result::Result<(), OutOfFamily> {
        let family = Family {
            executor: self,
            blocks: RefCell::new(SmallVec::new()),
            with: None,
        };
        let mut found = false;
        family.expression(clause, 1, Site::Subqueries, &mut found)?;
        family.catalog()
    }

    /// The columns a join side that is not a base table gives, a CTE's, a
    /// view's, a derived table's, a VALUES list's or a table function's,
    /// named as a FROM names them without reading the side, and untyped
    pub(crate) fn join_side_schema(
        &self,
        side: &Expression,
        ctx: &ExecutionContext,
    ) -> Result<CompactArc<Schema>> {
        let mut on = |reach: Reach<'_>| match reach {
            Reach::Outer(..) => Err(Error::internal("a join side naming an enclosing column")),
            Reach::Param(_) => Ok(None),
        };
        let mut walk = StatementWalk {
            executor: self,
            ctx,
            on: &mut on,
        };
        let (_, scope) = walk.from(Some(side), &[], 0, Check::Off)?;
        let name = scope.tables.first().map(|(name, _)| name.as_str());
        Ok(CompactArc::new(untyped_schema(
            name.unwrap_or_default(),
            &scope.columns,
        )))
    }

    /// A context holding each CTE of `with` under the columns a FROM names
    /// it with, and no rows: a recursive one named by its first branch
    pub(crate) fn named_ctes(&self, with: &WithClause) -> Result<ExecutionContext> {
        let mut registry = super::cte::CteRegistry::new();
        let mut on = |_: Reach<'_>| Ok(None);
        for cte in &with.ctes {
            let ctx = ExecutionContext::new().with_cte_data(registry.data());
            let columns = match cte.column_names.is_empty() {
                false => cte
                    .column_names
                    .iter()
                    .map(|c| c.value.to_string())
                    .collect(),
                true => {
                    let anchor;
                    let query = match cte.is_recursive {
                        true => {
                            anchor = SelectStatement {
                                set_operations: Vec::new(),
                                ..(*cte.query).clone()
                            };
                            &anchor
                        }
                        false => &*cte.query,
                    };
                    let mut walk = StatementWalk {
                        executor: self,
                        ctx: &ctx,
                        on: &mut on,
                    };
                    named(walk.statement(query, &[], 0, Check::Off)?.1)?
                }
            };
            registry.store(&cte.name.value, columns, RowVec::new());
        }
        Ok(ExecutionContext::new().with_cte_data(registry.data()))
    }

    /// A join statement's `on`, `cross` and `output`, over the joined rows of
    /// `sides` laid out as `width` columns, with their subqueries prepared
    /// as blocks
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_join_blocks(
        &self,
        on: (&JoinTableSource, Option<&Expression>, bool),
        cross: Option<&Expression>,
        side_filters: [Option<&Expression>; 2],
        output: [&[Option<&Expression>]; 3],
        sides: &JoinSides,
        width: usize,
        ctx: &ExecutionContext,
    ) -> Result<Option<JoinBlocks>> {
        JoinBlocks::prepare(
            self,
            ctx,
            on,
            cross,
            side_filters,
            output,
            sides,
            width,
            ctx.params().len(),
        )
    }

    /// One execution's run of a join statement's blocks, its rows unknown
    /// until a site has them; None when a table they read no longer has
    /// the columns they were prepared for
    pub(crate) fn join_runner<'a>(
        &'a self,
        held: &'a JoinBlocks,
        ctx: &'a ExecutionContext,
    ) -> Result<Option<PreparedRunner<'a>>> {
        let (blocks, holes) = (&held.blocks, &held.joined);
        let inner_ctx = ctx.with_incremented_query_depth();
        let Some(tables) = PreparedRunner::open_tables(self, blocks, &inner_ctx, None)? else {
            return Ok(None);
        };
        let mut runner =
            PreparedRunner::new(self, blocks, holes, ctx, inner_ctx, tables, None, None);
        for (slot, value) in runner.run_slot_values()? {
            runner.base[slot] = value;
        }
        runner.read_ahead(&held.ahead, holes)?;
        Ok(Some(runner))
    }

    /// The filter that answers, for the rows of `table`, the conditions
    /// `held` puts on side `side` alone, when the scan takes them whole and
    /// no narrowing reads the rows by keys: Some(None) when no row needs
    /// one; None when a row needs the program
    pub(crate) fn join_side_filter(
        &self,
        held: &JoinBlocks,
        run: &PreparedRunner<'_>,
        side: usize,
        table: &dyn Table,
        ctx: &ExecutionContext,
    ) -> Result<Option<Option<Box<dyn StorageExpr>>>> {
        // A narrowing reads rows by keys the fetch has already chosen; its
        // set, when it has one, is a filter like the others
        let Some(where_) = held.side_wheres[side]
            .as_ref()
            .filter(|w| w.narrowing.is_none() || w.narrowing_set.is_some())
        else {
            return Ok(None);
        };
        Ok(
            match run.where_scan(where_, None, where_.narrowing_set, table, ctx)? {
                None => Some(Some(Box::new(ConstBoolExpr::new(false)))),
                Some((filter, true, ..)) => Some(filter),
                Some(_) => None,
            },
        )
    }

    /// The rows of table `name`, side `side` of the join `held` prepared,
    /// that its conditions on that side pass and the `extra` ones too, read
    /// as `run` reads a prepared statement's rows: from the `offset`-th up
    /// to `limit` of them, with the table's layout. None when the statement
    /// holds no condition on that side alone
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn join_side_rows(
        &self,
        held: &JoinBlocks,
        run: &PreparedRunner<'_>,
        side: usize,
        (name, alias): (&str, &str),
        ctx: &ExecutionContext,
        extra: Option<&Expression>,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<Option<(Box<dyn crate::storage::traits::QueryResult>, LayoutToken)>> {
        let Some(where_) = &held.side_wheres[side] else {
            return Ok(None);
        };
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::join_side_opening();
        let table = self.prepared_table(name, ctx, None)?;
        held.side_fits(side, &table.schema_arc())?;
        let token = table.layout_token();
        let columns = table.schema().column_names_arc();
        let extra = extra.map(|extra| strip_table_qualifier(extra, alias));
        let filter = extra
            .as_ref()
            .map(|extra| RowFilter::new(extra, &columns).map(|f| f.with_context(ctx)))
            .transpose()?;
        let mut kept = SideKeep::new(filter.as_ref(), offset, limit);
        if limit != Some(0) {
            let mut keep = |id: i64, _: u64, row: Row| kept.keep(id, row);
            let stream = limit.is_none() && offset == 0;
            // The scan's own answers stop at the rows wanted, which the extra
            // conditions would thin after the count
            let target = limit
                .filter(|_| extra.is_none())
                .map(|limit| limit.saturating_add(offset));
            match run.passing_rows(
                where_,
                extra.as_ref(),
                &*table,
                ctx,
                target,
                stream,
                &mut keep,
            )? {
                Passing::Pruned | Passing::Read => {}
                Passing::Scan(scan) => {
                    return Ok(Some((Box::new(ScannerResult::new(scan, columns)), token)));
                }
                Passing::Filtered(scan, bound) => {
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::bound_side_read();
                    let rows = Box::new(ScannerResult::new(scan, columns));
                    let mut filter = RowFilter::from_program(bound).with_context(ctx);
                    // The values the run computed once sit past the statement's own
                    if run.base.len() > ctx.params().len() {
                        filter = filter.with_params(CompactArc::new(run.base.clone()));
                    }
                    let filtered = FilteredResult::from_filter(rows, filter);
                    return Ok(Some((Box::new(filtered), token)));
                }
                Passing::Answers(answers) => {
                    kept.rows.extend(answers.into_iter().skip(offset));
                }
            }
        }
        Ok(Some((
            Box::new(ExecutorResult::with_arc_columns(columns, kept.rows()?)),
            token,
        )))
    }

    /// `result`, side `expr` as the general path read it with its conditions
    /// without a block, side `side` of the join `held` prepared, cut to the
    /// rows its conditions with a block pass
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn join_source_side_rows(
        &self,
        held: &JoinBlocks,
        run: &PreparedRunner<'_>,
        side: usize,
        (expr, result): (&Expression, Box<dyn crate::storage::traits::QueryResult>),
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn crate::storage::traits::QueryResult>> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::join_side_opening();
        let alias = get_table_alias_from_expr(expr).unwrap_or_default();
        let columns = CompactArc::new(result.columns().to_vec());
        held.rows_fit(side, &alias, &columns)?;
        // A side with no condition of its own passes on as the join reads it
        let Some(where_) = &held.side_wheres[side] else {
            return Ok(result);
        };
        // Its conditions bind before its first row, so they pass its rows
        // as they come, as a table side's scan does
        if let Some(bound) = run.bound_reading(&where_.whole.program) {
            let mut own = RowFilter::from_program(bound).with_context(ctx);
            // The values the run computed once sit past the statement's own
            if run.base.len() > ctx.params().len() {
                own = own.with_params(CompactArc::new(run.base.clone()));
            }
            return Ok(Box::new(FilteredResult::from_filter(result, own)));
        }
        // A set whose read failed leaves its error with the row that needs it
        let mut result = result;
        let mut id = 0i64;
        let source = std::iter::from_fn(|| {
            result.next().then(|| {
                id += 1;
                (id - 1, result.take_row())
            })
        });
        let mut rows = RowVec::new();
        let keep = |id: i64, _: u64, row: Row| {
            rows.push((id, row));
            true
        };
        run.rows_passing(where_, source.into_iter().map(Ok), ctx, None, keep)?;
        if let Some(err) = result.last_error() {
            return Err(err);
        }
        Ok(Box::new(ExecutorResult::with_arc_columns(columns, rows)))
    }
}

/// The rows a join side's read keeps: those its `filter` passes, from the
/// `offset`-th up to `limit` of them; an error the filter raises stops it
struct SideKeep<'f> {
    filter: Option<&'f RowFilter>,
    offset: usize,
    limit: Option<usize>,
    skipped: usize,
    failed: Option<Error>,
    rows: RowVec,
}

impl<'f> SideKeep<'f> {
    fn new(filter: Option<&'f RowFilter>, offset: usize, limit: Option<usize>) -> Self {
        Self {
            filter,
            offset,
            limit,
            skipped: 0,
            failed: None,
            rows: RowVec::new(),
        }
    }

    /// Whether the read goes on past `row`
    fn keep(&mut self, id: i64, row: Row) -> bool {
        match self.filter.map(|filter| filter.matches_checked(&row)) {
            Some(Ok(false)) => return true,
            Some(Err(err)) => {
                self.failed = Some(err);
                return false;
            }
            Some(Ok(true)) | None => {}
        }
        if self.skipped < self.offset {
            self.skipped += 1;
            return true;
        }
        self.rows.push((id, row));
        self.limit.is_none_or(|limit| self.rows.len() < limit)
    }

    /// The rows kept, or the error that stopped the read
    fn rows(self) -> Result<RowVec> {
        match self.failed {
            Some(err) => Err(err),
            None => Ok(self.rows),
        }
    }
}

/// The tables a prepared UPDATE's or DELETE's blocks read, opened before
/// the statement writes, and the context they read under
pub(crate) struct DmlTables {
    tables: BlockTables,
    inner_ctx: ExecutionContext,
}

/// The rows an UPDATE may write, each asked its WHERE as it is written
pub(crate) enum Candidates {
    /// The scan proves that no row passes
    Pruned,
    /// The rows a set test's keys name, by row id
    Keyed(Vec<i64>),
    /// A scan narrowed by the conjuncts without a subquery
    Scan(Option<Box<dyn StorageExpr>>),
}

/// One execution of a prepared UPDATE or DELETE over its target's rows
pub(crate) struct DmlRunner<'a> {
    runner: PreparedRunner<'a>,
    prepared: &'a PreparedDml,
}

impl<'a> DmlRunner<'a> {
    /// A run of `prepared` for a target of about `rows` rows, its run slots
    /// computed before the statement reads a row
    pub(crate) fn new(
        executor: &'a Executor,
        prepared: &'a PreparedDml,
        ctx: &'a ExecutionContext,
        tables: DmlTables,
        rows: usize,
    ) -> Result<Self> {
        let mut runner = PreparedRunner::new(
            executor,
            &prepared.blocks,
            &prepared.holes,
            ctx,
            tables.inner_ctx,
            tables.tables,
            Some(rows),
            None,
        );
        for (slot, value) in runner.run_slot_values()? {
            runner.base[slot] = value;
        }
        Ok(Self { runner, prepared })
    }

    /// `program` bound to all the run knows now, see [`PreparedRunner::bind`]
    pub(crate) fn bind(&self, program: &SharedProgram) -> Bound {
        self.runner.bind(program)
    }

    /// `bound`'s program, bound again when a block was read since
    pub(crate) fn current<'b>(&self, bound: &'b mut Bound) -> &'b SharedProgram {
        self.runner.current(bound)
    }

    /// What the statement's programs read for `row`: its parameters, its
    /// run slots and its blocks
    pub(crate) fn context<'r>(&'r self, row: &'r Row) -> ExecuteContext<'r> {
        ExecuteContext::with_common_params(row, &self.runner.base, self.runner.transaction_id)
            .with_subquery_executor(&self.runner)
    }

    /// Whether the row of `exec` passes the WHERE; every row passes a
    /// statement without one
    pub(crate) fn matches(&self, vm: &mut ExprVM, exec: &ExecuteContext) -> Result<bool> {
        match &self.prepared.where_ {
            Some(where_) => self.runner.passes(vm, &where_.whole, exec),
            None => Ok(true),
        }
    }

    /// The rows of `table` the WHERE may pass, read as a prepared SELECT
    /// reads them but not yet asked the WHERE
    pub(crate) fn candidates(
        &self,
        table: &dyn Table,
        ctx: &ExecutionContext,
    ) -> Result<Candidates> {
        let Some(where_) = &self.prepared.where_ else {
            return Ok(Candidates::Scan(None));
        };
        let Some(scan) = self.runner.where_scan(where_, None, None, table, ctx)? else {
            return Ok(Candidates::Pruned);
        };
        if let Some((rows, _)) = self.runner.narrowed_rows(where_.narrowing, table, None)? {
            return Ok(Candidates::Keyed(
                rows.into_iter().map(|(id, _)| id).collect(),
            ));
        }
        // The scan takes the narrowing's own set once it declined
        let scan = match where_.narrowing_set {
            Some(slot) => self
                .runner
                .where_scan(where_, None, Some(slot), table, ctx)?,
            None => Some(scan),
        };
        Ok(match scan {
            Some((storage_expr, ..)) => Candidates::Scan(storage_expr),
            None => Candidates::Pruned,
        })
    }

    /// Hand `keep` the row id and row of each row of `table` the WHERE
    /// passes, read as a prepared SELECT reads its rows: through its
    /// narrowing, or a scan the conjuncts without a subquery narrow
    pub(crate) fn passing_rows(
        &self,
        table: &dyn Table,
        ctx: &ExecutionContext,
        mut keep: impl FnMut(i64, Row),
    ) -> Result<()> {
        let Some(where_) = &self.prepared.where_ else {
            return Err(Error::internal("a prepared statement without WHERE"));
        };
        let mut each = |id: i64, _: u64, row: Row| {
            keep(id, row);
            true
        };
        match self
            .runner
            .passing_rows(where_, None, table, ctx, None, false, &mut each)?
        {
            Passing::Pruned | Passing::Read => Ok(()),
            Passing::Answers(_) | Passing::Scan(_) | Passing::Filtered(..) => Err(Error::internal(
                "a statement without a LIMIT read the scan's answers",
            )),
        }
    }

    /// Each SET value's program with the column it writes
    pub(crate) fn sets(&self) -> &'a [(usize, SharedProgram)] {
        &self.prepared.sets
    }

    /// The programs of each VALUES row, None for a value without a subquery
    pub(crate) fn values(&self) -> &'a [Vec<Option<SharedProgram>>] {
        &self.prepared.values
    }

    /// The ON CONFLICT DO UPDATE, its programs over the row met joined with
    /// the excluded row; None without a prepared one
    pub(crate) fn conflict_upsert(&self) -> Option<&'a Arc<super::dml::CompiledUpsert>> {
        self.prepared
            .conflict
            .as_ref()
            .map(|conflict| &conflict.upsert)
    }

    /// Whether a value read for each conflict holds a block that reads the
    /// target, which inside a transaction would see the statement's writes
    pub(crate) fn conflict_reads_target(&self) -> bool {
        self.prepared
            .conflict
            .as_ref()
            .is_some_and(|conflict| conflict.reads_target)
    }

    /// The RETURNING's programs over the row returned, one for each column
    /// of its result; None without a prepared one
    pub(crate) fn returning(&self) -> Option<&'a [SharedProgram]> {
        self.prepared
            .returning
            .as_ref()
            .map(|returning| returning.programs.as_slice())
    }

    /// The names of the RETURNING's result columns; None without a
    /// prepared RETURNING
    pub(crate) fn returning_names(&self) -> Option<&'a CompactArc<Vec<String>>> {
        self.prepared
            .returning
            .as_ref()
            .map(|returning| &returning.names)
    }

    /// Whether a RETURNING value read for each row returned holds a block
    /// that reads the target, which inside a transaction would see the
    /// statement's writes
    pub(crate) fn returning_reads_target(&self) -> bool {
        self.prepared
            .returning
            .as_ref()
            .is_some_and(|returning| returning.reads_target)
    }
}

/// The conjuncts of a block's `where_clause` that read no column, or compare
/// a column of its table function `name` with a value that reads none: they
/// narrow the range it generates, or skip it, as a statement's WHERE does
fn function_range(where_clause: Option<&Expression>, name: &str) -> Option<Box<Expression>> {
    let own = |expr: &Expression| match expr {
        Expression::Identifier(_) => true,
        Expression::QualifiedIdentifier(qi) => qi.qualifier.value_lower == name,
        _ => false,
    };
    let mut parts = Vec::new();
    conjuncts(where_clause?, &mut parts);
    let range = parts
        .into_iter()
        .filter(|part| {
            !mentions_column(part)
                || match part {
                    Expression::Infix(infix) => {
                        (own(&infix.left) && !mentions_column(&infix.right))
                            || (own(&infix.right) && !mentions_column(&infix.left))
                    }
                    Expression::Between(between) => {
                        !between.not
                            && own(&between.expr)
                            && !mentions_column(&between.lower)
                            && !mentions_column(&between.upper)
                    }
                    _ => false,
                }
        })
        .cloned()
        .collect();
    combine_predicates_with_and(range).map(Box::new)
}

/// Whether a name anywhere in `expr`, its subqueries' clauses included, bare,
/// as the part after a dotted name's last dot or as the column of a qualified
/// name, passes `matches`
fn names_anywhere(expr: &Expression, matches: &impl Fn(&str) -> bool) -> bool {
    let statement = |sq: &SelectStatement| {
        sq.columns
            .iter()
            .chain(sq.where_clause.as_deref())
            .chain(sq.having.as_deref())
            .chain(&sq.group_by.columns)
            .chain(sq.order_by.iter().map(|o| &o.expression))
            .any(|e| names_anywhere(e, matches))
    };
    match expr {
        Expression::Identifier(id) => {
            matches(&id.value_lower)
                || split_qualified(&id.value_lower).is_some_and(|(_, column)| matches(column))
        }
        Expression::QualifiedIdentifier(qid) => matches(&qid.name.value_lower),
        Expression::Aliased(aliased) => names_anywhere(&aliased.expression, matches),
        Expression::ScalarSubquery(sq) => statement(&sq.subquery),
        Expression::Exists(exists) => statement(&exists.subquery),
        Expression::AllAny(all_any) => {
            names_anywhere(&all_any.left, matches) || statement(&all_any.subquery)
        }
        other => {
            let mut found = false;
            let _ = for_each_child(other, &mut |child| {
                found |= names_anywhere(child, matches);
                Ok(())
            });
            found
        }
    }
}

/// Whether `expr`, rewritten with its blocks in `blocks`, asks one of them
/// of each row: any but an uncorrelated block's value or members, which a
/// run reads once
fn asks_per_row(expr: &Expression, blocks: &[Option<PreparedSubquery>]) -> bool {
    let block = |mark: &SelectStatement| {
        mark.prepared
            .and_then(|id| blocks.get(usize::from(id))?.as_ref())
    };
    let members =
        |mark: &SelectStatement| block(mark).is_some_and(|sq| sq.member_set_test().is_some());
    let once = |mark: &SelectStatement| block(mark).is_some_and(|sq| sq.uncorrelated);
    match expr {
        Expression::In(in_expr) => match in_expr.right.as_ref() {
            Expression::ScalarSubquery(mark) => {
                !members(&mark.subquery) || asks_per_row(&in_expr.left, blocks)
            }
            right => asks_per_row(&in_expr.left, blocks) || asks_per_row(right, blocks),
        },
        Expression::AllAny(all_any) => {
            !members(&all_any.subquery) || asks_per_row(&all_any.left, blocks)
        }
        Expression::ScalarSubquery(mark) => !once(&mark.subquery),
        Expression::Exists(mark) => !once(&mark.subquery),
        other => {
            let mut found = false;
            let _ = for_each_child(other, &mut |child| {
                found |= asks_per_row(child, blocks);
                Ok(())
            });
            found
        }
    }
}

fn mentions_subquery(expr: &Expression) -> bool {
    match expr {
        Expression::ScalarSubquery(_) | Expression::Exists(_) | Expression::AllAny(_) => true,
        other => {
            let mut found = false;
            let _ = for_each_child(other, &mut |child| {
                found |= mentions_subquery(child);
                Ok(())
            });
            found
        }
    }
}

/// The sides of the joined row `scope`, the left one its first `left_width`
/// columns, that a subquery of `expr` reads: bit 0 the left, bit 1 the right
fn sides_read_within(
    executor: &Executor,
    ctx: &ExecutionContext,
    expr: &Expression,
    scope: &ScopeTable,
    left_width: usize,
) -> Result<u8> {
    let mut read = 0;
    let mut on = |reach: Reach<'_>| match reach {
        Reach::Outer(expr, depth, index) => {
            if depth == 1 {
                read |= if index < left_width { 1 } else { 2 };
            }
            Ok(Some(expr.clone()))
        }
        Reach::Param(_) => Ok(None),
    };
    let mut walk = StatementWalk {
        executor,
        ctx,
        on: &mut on,
    };
    walk.subqueries_of(expr, &[scope])?;
    Ok(read)
}

impl Builder<'_> {
    /// The WHERE of the statement whose row is `chain`'s first scope, over
    /// `schema`: each conjunct rewritten, the ones a scan takes, and the
    /// EXISTS and IN tests whose keys may name its rows
    fn statement_where(
        &mut self,
        chain: &Chain<'_>,
        schema: &Schema,
        where_clause: &Expression,
        work: &mut Rewritten,
        run_slots: &mut Vec<(usize, Expression)>,
    ) -> Result<WhereParts> {
        let mut parts = Vec::new();
        conjuncts(where_clause, &mut parts);
        let mut rewritten = Vec::with_capacity(parts.len());
        let mut pushed = Vec::new();
        let mut set_tests = Vec::new();
        for (at, part) in parts.into_iter().enumerate() {
            let part = self.statement_conjunct(chain, part, work, run_slots)?;
            let scanned =
                !mentions_subquery(&part) && !needs_folding(&part) && scans_whole(&part, schema);
            if scanned {
                pushed.push(part.clone());
            }
            let set_test = match &part {
                Expression::Exists(mark) => Some((&mark.subquery, false, None)),
                Expression::Prefix(prefix) if prefix.op_type == PrefixOperator::Not => {
                    match prefix.right.as_ref() {
                        Expression::Exists(mark) => Some((&mark.subquery, true, None)),
                        _ => None,
                    }
                }
                Expression::In(in_expr) => match (&*in_expr.right, bound_column(&in_expr.left)) {
                    (Expression::ScalarSubquery(mark), Some(column)) => {
                        Some((&mark.subquery, in_expr.not, Some(column)))
                    }
                    _ => None,
                },
                Expression::AllAny(all_any) => bound_column(&all_any.left)
                    .map(|column| (&all_any.subquery, false, Some(column))),
                _ => None,
            };
            if let Some((mark, negated, column)) = set_test {
                set_tests.extend(mark.prepared.map(|id| (id, negated, column, at)));
            }
            rewritten.push((part, scanned));
        }
        let conjuncts = rewritten.len();
        let clause = conjunction(rewritten.iter().map(|(part, _)| part.clone()).collect())
            .ok_or_else(|| Error::internal("a prepared statement without WHERE"))?;
        // The conjuncts the `kept` test keeps; TRUE for none
        let rest = |kept: &dyn Fn(usize, bool) -> bool| {
            let parts = rewritten
                .iter()
                .enumerate()
                .filter(|&(i, &(_, pushed))| kept(i, pushed))
                .map(|(_, (part, _))| part.clone())
                .collect();
            conjunction(parts).unwrap_or_else(true_literal)
        };
        let scanned = match pushed.len() {
            0 => None,
            n if n == conjuncts => None,
            _ => Some(rest(&|_, pushed| !pushed)),
        };
        let narrowed = set_tests
            .iter()
            .map(|&(.., at)| {
                let scanned = (!pushed.is_empty()).then(|| rest(&|i, pushed| i != at && !pushed));
                (rest(&|i, _| i != at), scanned)
            })
            .collect();
        Ok(WhereParts {
            clause,
            pushed,
            scanned,
            narrowed,
            set_tests,
            conjuncts,
        })
    }

    /// Whether `conjunct`, a condition on the one table of `chain` over
    /// `schema`, binds to what a run reads before the table's first row: its
    /// blocks uncorrelated, read once for their values or sets. Tried as the
    /// WHERE prepares it, and given back
    fn binds_before_rows(
        &mut self,
        chain: &Chain<'_>,
        schema: &Schema,
        conjunct: &Expression,
    ) -> Result<bool> {
        let (next_slot, blocks) = (self.next_slot, self.subqueries.len());
        let parts = self.statement_where(
            chain,
            schema,
            conjunct,
            &mut Rewritten::default(),
            &mut Vec::new(),
        );
        let binds = parts.map(|parts| !asks_per_row(&parts.clause, &self.subqueries));
        self.next_slot = next_slot;
        self.subqueries.truncate(blocks);
        binds
    }

    /// A top-level conjunct of the statement's WHERE. The value a column
    /// is compared with becomes a run slot when it reads no row, so the
    /// scan can take the comparison; it may read uncorrelated subqueries
    fn statement_conjunct(
        &mut self,
        chain: &Chain<'_>,
        conjunct: &Expression,
        work: &mut Rewritten,
        run_slots: &mut Vec<(usize, Expression)>,
    ) -> Result<Expression> {
        let Expression::Infix(infix) = conjunct else {
            return self.rewrite(chain, None, conjunct, work);
        };
        if !is_comparison(infix.op_type) {
            return self.rewrite(chain, None, conjunct, work);
        }
        let column_left = matches!(chain.resolve(&infix.left)?, Some((0, _)));
        let column_right = matches!(chain.resolve(&infix.right)?, Some((0, _)));
        let value = match (column_left, column_right) {
            (true, false) => &infix.right,
            (false, true) => &infix.left,
            _ => return self.rewrite(chain, None, conjunct, work),
        };
        let mut value_work = Rewritten::default();
        let rewritten_value = self.rewrite(chain, None, value, &mut value_work)?;
        let value = if !value_work.reads_row
            && !is_leaf(value)
            && !volatile(value, &self.executor.function_registry)
        {
            let slot = self.slot();
            run_slots.push((slot, rewritten_value));
            slot_parameter(slot)
        } else {
            work.reads_row |= value_work.reads_row;
            rewritten_value
        };
        let column = if column_left {
            &infix.left
        } else {
            &infix.right
        };
        let column = self.rewrite(chain, None, column, work)?;
        let (left, right) = if column_left {
            (column, value)
        } else {
            (value, column)
        };
        Ok(Expression::Infix(InfixExpression {
            token: infix.token.clone(),
            left: Box::new(left),
            operator: infix.operator.clone(),
            op_type: infix.op_type,
            right: Box::new(right),
        }))
    }
}

impl Executor {
    /// The tables `select` reads, unknown when one of them is a view or it
    /// reads anything else
    fn read_tables(&self, select: &SelectStatement) -> Option<Vec<SmartString>> {
        subquery_cache_tables(select)
            .filter(|tables| {
                tables
                    .iter()
                    .all(|table| matches!(self.engine.get_view_lowercase(table), Ok(None)))
            })
            .map(|tables| {
                tables
                    .iter()
                    .map(|table| SmartString::from(&**table))
                    .collect()
            })
    }

    /// The columns and rows of `run`'s statement as `ctx` reads them
    fn statement_rows(
        &self,
        run: &StatementRun,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, RowVec)> {
        let result = self.execute_select_with_plan(
            &run.statement,
            ctx,
            Some(&run.classification),
            Some(run.compiled.slot()),
            None,
        )?;
        let columns = result.columns().to_vec();
        Ok((columns, Self::materialize_result(result)?))
    }

    /// Whether a source read when a call first needs it is the one the plan
    /// named its columns for; a view's query may have changed since
    fn source_holds(&self, source: &BlockSource) -> Result<bool> {
        Ok(match source {
            BlockSource::View(name, _, query) => self
                .engine
                .get_view_lowercase(name)?
                .is_some_and(|view| view.query == *query),
            _ => true,
        })
    }

    /// A block source's rows for this run, as `ctx` reads them; None when
    /// they no longer have the columns `schema` was prepared with
    fn source_rows(
        &self,
        source: &BlockSource,
        schema: &Schema,
        ctx: &ExecutionContext,
    ) -> Result<Option<SourceRows>> {
        let fits = |columns: &[String]| {
            columns.len() == schema.columns.len()
                && columns
                    .iter()
                    .zip(&schema.columns)
                    .all(|(c, s)| c.eq_ignore_ascii_case(&s.name))
        };
        let (columns, rows) = match source {
            BlockSource::Cte(name) => {
                let (columns, rows) = ctx
                    .get_cte_by_lower(name)
                    .ok_or_else(|| Error::TableOrViewNotFound(name.to_string()))?;
                return Ok(fits(columns).then(|| CompactArc::clone(rows)));
            }
            BlockSource::View(name, ..) => {
                let Some(view) = self.engine.get_view_lowercase(name)? else {
                    return Ok(None);
                };
                let result = self.execute_view_body(&view, ctx)?;
                let columns = result.columns().to_vec();
                (columns, Self::materialize_result(result)?)
            }
            BlockSource::Derived(run, ..) => self.statement_rows(run, ctx)?,
            BlockSource::Values(values) => {
                let mut rows = RowVec::with_capacity(values.rows.len());
                for (id, exprs) in (0i64..).zip(&values.rows) {
                    rows.push((id, Self::values_row(exprs, ctx)?));
                }
                (Self::values_columns(values), rows)
            }
            BlockSource::Function(function, range, counts) => {
                // A count the block cannot read is its error, raised when it runs
                let first = counts.as_ref().and_then(|(limit, offset)| {
                    use super::row_count::row_count;
                    let limit = row_count(limit, ctx, "LIMIT").ok()?;
                    let offset = offset.as_deref().map(|o| row_count(o, ctx, "OFFSET"));
                    Some(limit.saturating_add(offset.transpose().ok()?.unwrap_or(0)))
                });
                let (rows, columns) = Self::tvf_rows(function, range.as_deref(), first, ctx)?;
                (columns, rows)
            }
        };
        Ok(fits(&columns).then(|| CompactArc::new(rows.into_vec())))
    }

    /// `name` as this statement reads it: through `through`, the statement's
    /// own transaction, else the explicit transaction, else the statement's
    /// snapshot, each at the statement's view
    fn prepared_table(
        &self,
        name: &str,
        ctx: &ExecutionContext,
        through: Option<&dyn Transaction>,
    ) -> Result<Box<dyn Table>> {
        use crate::storage::traits::transaction::StatementTable;
        let table = match through {
            Some(tx) => tx.get_read_table_at(name, ctx.scan_view(name), ctx.table_use()),
            None => {
                let active = self
                    .active_transaction
                    .lock()
                    .map_err(|_| Error::internal("active transaction lock poisoned"))?;
                match active.as_ref() {
                    Some(tx) => {
                        tx.transaction
                            .get_read_table_at(name, ctx.scan_view(name), ctx.table_use())
                    }
                    None => {
                        drop(active);
                        ctx.statement_snapshot()
                            .ok_or_else(|| {
                                Error::internal("a prepared statement reads without a snapshot")
                            })?
                            .statement_table_at(name, ctx.scan_view(name))
                    }
                }
            }
        };
        let table = table.map_err(|e| match e {
            Error::TableNotFound(_) => Error::TableOrViewNotFound(name.to_string()),
            other => other,
        })?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::select_table_opened();
        Ok(table)
    }

    /// The plan's prepared statement, prepared now over the source `schema`
    /// gives when the plan holds none for this schema, these `columns` and
    /// this parameter count
    #[allow(clippy::too_many_arguments)]
    fn prepared_for<'s>(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        stmt: &SelectStatement,
        columns: &[String],
        schema: impl FnOnce() -> std::borrow::Cow<'s, Schema>,
        scope_name: &SmartString,
        ctx: &ExecutionContext,
        fresh: bool,
    ) -> Result<Arc<PreparedSelect>> {
        let epoch = self.engine.schema_epoch();
        let fits = |p: &PreparedSelect| {
            p.cached_epoch == epoch
                && p.blocks.param_count == ctx.params().len()
                && (std::ptr::eq(p.columns.as_slice(), columns) || p.columns.as_slice() == columns)
        };
        if let Some(slot) = compiled.filter(|_| !fresh) {
            if let Ok(guard) = slot.read() {
                if let CompiledExecution::PreparedSelect(p) = &*guard {
                    if fits(p) {
                        return Ok(Arc::clone(p));
                    }
                }
            }
        }
        // A set operation's first branch: the whole one reads the rest
        let branch;
        let stmt = match stmt.set_operations.is_empty() {
            true => stmt,
            false => {
                branch = SelectStatement {
                    set_operations: Vec::new(),
                    order_by: Vec::new(),
                    limit: None,
                    offset: None,
                    ..stmt.clone()
                };
                &branch
            }
        };
        let prepared = Arc::new(PreparedSelect::prepare(
            self,
            ctx,
            stmt,
            &schema(),
            scope_name,
            ctx.params().len(),
            epoch,
        )?);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_select_built();
        if let Some(slot) = compiled {
            if let Ok(mut guard) = slot.write() {
                if matches!(
                    &*guard,
                    CompiledExecution::Unknown
                        | CompiledExecution::NotOptimizable(_)
                        | CompiledExecution::PreparedSelect(_)
                ) {
                    *guard = CompiledExecution::PreparedSelect(Arc::clone(&prepared));
                }
            }
        }
        Ok(prepared)
    }

    /// Whether the plan holds a statement prepared for the current schema,
    /// which puts the statement in the family without classifying it again
    pub(crate) fn holds_prepared_select(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        ctx: &ExecutionContext,
    ) -> bool {
        compiled
            .and_then(|slot| slot.read().ok())
            .is_some_and(|guard| match &*guard {
                CompiledExecution::PreparedSelect(p) => {
                    p.cached_epoch == self.engine.schema_epoch()
                        && p.blocks.param_count == ctx.params().len()
                        && ctx.outer_row().is_none()
                }
                _ => false,
            })
    }

    /// Whether the plan holds a DML statement prepared for the current
    /// schema, which puts it in the family without walking it again
    pub(crate) fn holds_prepared_dml(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        ctx: &ExecutionContext,
    ) -> bool {
        compiled
            .and_then(|slot| slot.read().ok())
            .is_some_and(|guard| match &*guard {
                CompiledExecution::PreparedDml(p) => {
                    p.cached_epoch == self.engine.schema_epoch()
                        && p.blocks.param_count == ctx.params().len()
                        && ctx.outer_row().is_none()
                }
                _ => false,
            })
    }

    /// Whether an UPDATE's or DELETE's WHERE and SET values run prepared:
    /// they hold a subquery, and every form in them is the family's
    pub(crate) fn dml_family<'e>(
        &self,
        where_clause: Option<&'e Expression>,
        sets: impl IntoIterator<Item = &'e Expression>,
        ctx: &'e ExecutionContext,
    ) -> std::result::Result<(), OutOfFamily> {
        if ctx.outer_row().is_some() {
            return Err("it runs for a row of an enclosing statement");
        }
        let family = Family {
            executor: self,
            blocks: RefCell::new(SmallVec::new()),
            with: None,
        };
        let mut found = false;
        for expr in where_clause.into_iter().chain(sets) {
            family.expression(expr, 1, Site::Subqueries, &mut found)?;
        }
        if !found {
            return Err("the statement holds no subquery");
        }
        family.catalog()
    }

    /// The plan's prepared UPDATE, DELETE or INSERT over a target of
    /// `schema`, named `scope_name`, with its blocks' tables open, through
    /// the statement's own transaction `through` when given; prepared now
    /// when the plan holds none for this schema, these columns and this
    /// parameter count; a compiled INSERT keeps it beside its own parts
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open_prepared_dml(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        schema: &Schema,
        scope_name: &SmartString,
        where_clause: Option<&Expression>,
        sets: &[(usize, &Expression)],
        values: &[impl AsRef<[Expression]>],
        conflict: Option<&ConflictSites<'_>>,
        returning: Option<&ReturningSites<'_>>,
        through: Option<&dyn Transaction>,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<PreparedDml>, DmlTables)> {
        let epoch = self.engine.schema_epoch();
        let columns = schema.column_names_arc();
        let same = |held: &CompactArc<Vec<String>>, now: &CompactArc<Vec<String>>| {
            CompactArc::ptr_eq(held, now) || held == now
        };
        let conflict_columns = conflict.map(|sites| sites.schema.column_names_arc());
        let returning_columns = returning.map(|sites| sites.schema.column_names_arc());
        let fits = |p: &PreparedDml| {
            p.cached_epoch == epoch
                && p.blocks.param_count == ctx.params().len()
                && same(&p.columns, &columns)
                && match (&p.conflict, &conflict_columns) {
                    (Some(held), Some(now)) => same(&held.columns, now),
                    (None, None) => true,
                    _ => false,
                }
                && match (&p.returning, &returning_columns) {
                    (Some(held), Some(now)) => same(&held.columns, now),
                    (None, None) => true,
                    _ => false,
                }
        };
        // The statement writes its target, open before its blocks, and what
        // its foreign keys cascade to
        let inner_ctx = ctx
            .with_incremented_query_depth()
            .with_table_use(TableUse::ReadBesideWrites);
        // A column change between taking the plan and opening a block's
        // table prepares again, against the table as it now is
        let mut fresh = false;
        loop {
            let held = compiled
                .filter(|_| !fresh)
                .and_then(|slot| slot.read().ok())
                .and_then(|guard| match &*guard {
                    CompiledExecution::PreparedDml(p) if fits(p) => Some(Arc::clone(p)),
                    CompiledExecution::Insert(insert) => {
                        insert.prepared.as_ref().filter(|p| fits(p)).cloned()
                    }
                    _ => None,
                });
            let prepared = match held {
                Some(prepared) => prepared,
                None => {
                    let prepared = Arc::new(PreparedDml::prepare(
                        self,
                        ctx,
                        schema,
                        scope_name,
                        where_clause,
                        sets,
                        values,
                        conflict,
                        returning,
                        ctx.params().len(),
                        epoch,
                    )?);
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::prepared_dml_was_built();
                    if let Some(mut guard) = compiled.and_then(|slot| slot.write().ok()) {
                        match &mut *guard {
                            CompiledExecution::Insert(insert) => {
                                insert.prepared = Some(Arc::clone(&prepared));
                            }
                            CompiledExecution::Unknown
                            | CompiledExecution::NotOptimizable(_)
                            | CompiledExecution::PreparedDml(_) => {
                                *guard = CompiledExecution::PreparedDml(Arc::clone(&prepared));
                            }
                            _ => {}
                        }
                    }
                    prepared
                }
            };
            // A block read for each conflict or row returned that reads the
            // target would see the statement's writes: it reads a snapshot taken before them
            let reads_target = prepared.conflict.as_ref().is_some_and(|c| c.reads_target)
                || prepared.returning.as_ref().is_some_and(|r| r.reads_target);
            let snapshot_ctx = match through {
                Some(_) if reads_target => {
                    let snapshot = self.new_statement_snapshot(
                        self.isolation_level(),
                        inner_ctx.statement_view(),
                    )?;
                    Some(inner_ctx.with_statement_snapshot(snapshot))
                }
                _ => None,
            };
            let block_ctx = snapshot_ctx.as_ref().unwrap_or(&inner_ctx);
            let through = through.filter(|_| snapshot_ctx.is_none());
            if let Some(tables) =
                PreparedRunner::open_tables(self, &prepared.blocks, block_ctx, through)?
            {
                let inner_ctx = snapshot_ctx.unwrap_or(inner_ctx);
                return Ok((prepared, DmlTables { tables, inner_ctx }));
            }
            if fresh {
                return Err(Error::internal(
                    "a table's columns changed while the statement prepared",
                ));
            }
            fresh = true;
        }
    }

    /// What `stmt` reads its rows from, and the name its scope bears: None
    /// for a base table, left closed; else a CTE's rows, a VALUES list, or a
    /// view's, a derived table's or a table function's rows, read once under
    /// the statement's snapshot, before a plan is taken, or without FROM the
    /// one row without columns. A table function
    /// generates at most `first` rows when the caller needs no more
    fn read_source<'s>(
        &self,
        stmt: &'s SelectStatement,
        ctx: &'s ExecutionContext,
        inner_ctx: &ExecutionContext,
        first: Option<usize>,
    ) -> Result<(Option<StatementSource<'s>>, Option<&'s SmartString>)> {
        let named = |alias: &'s Option<Identifier>| alias.as_ref().map(|alias| &alias.value_lower);
        Ok(match stmt.table_expr.as_deref() {
            Some(Expression::TableSource(ts)) => {
                let name = &ts.name.value_lower;
                // A CTE comes before a view, and a view before a table, of
                // one name, as the general path reads them
                let source = match ctx.get_cte_by_lower(name) {
                    Some(cte) => Some(StatementSource::Cte(cte)),
                    None => match self.engine.get_view_lowercase(name)? {
                        Some(view) => {
                            let result = self.execute_view_body(&view, inner_ctx)?;
                            let columns = result.columns().to_vec();
                            Some(StatementSource::Rows(
                                columns,
                                Self::materialize_result(result)?,
                            ))
                        }
                        None => None,
                    },
                };
                let scope = ts.alias.as_ref().map_or(name, |alias| &alias.value_lower);
                (source, Some(scope))
            }
            Some(Expression::SubquerySource(ss)) => {
                let result = self.execute_select(&ss.subquery, inner_ctx)?;
                let columns = result.columns().to_vec();
                let rows = Self::materialize_result(result)?;
                (Some(StatementSource::Rows(columns, rows)), named(&ss.alias))
            }
            Some(Expression::ValuesSource(values)) => (
                Some(StatementSource::Values(
                    Self::values_columns(values),
                    values,
                )),
                named(&values.alias),
            ),
            Some(Expression::FunctionTableSource(function)) => {
                let (rows, columns) =
                    Self::tvf_rows(function, stmt.where_clause.as_deref(), first, ctx)?;
                let scope = function
                    .alias
                    .as_ref()
                    .map_or(&function.function.value_lower, |alias| &alias.value_lower);
                (Some(StatementSource::Rows(columns, rows)), Some(scope))
            }
            None => {
                let mut row = RowVec::with_capacity(1);
                row.push((0, Row::new()));
                (Some(StatementSource::Rows(Vec::new(), row)), None)
            }
            _ => return Err(Error::internal("a prepared statement without one table")),
        })
    }

    /// The output rows a DISTINCT keeps of `scan`, each projected as it
    /// passes `bound` and held only when its values are new, `wanted` of
    /// them at most: the scan's rows are read in place, as the general
    /// path's stream reads them
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn distinct_scan(
        &self,
        mut scan: Box<dyn Scanner>,
        bound: Option<SharedProgram>,
        runner: &PreparedRunner<'_>,
        projection: &PreparedProjection,
        mut distinct: DistinctRows,
        wanted: Option<usize>,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        let (mut vm, mut filter_vm) = (ExprVM::new(), ExprVM::new());
        let (mut values, mut kept) = (Vec::new(), RowVec::new());
        let mut outputs = projection.unbound();
        let mut scanned = 0u64;
        while scan.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                ctx.check_cancelled()?;
            }
            let row = scan.row();
            if let Some(bound) = &bound {
                if !runner.matches(&mut filter_vm, bound, row)? {
                    continue;
                }
            }
            // A row that is its own output is copied only when new
            let out = if projection.whole_row {
                if !distinct.is_new(row) {
                    continue;
                }
                row.clone()
            } else {
                let out = projection.project_into::<false>(
                    &mut vm,
                    row,
                    runner,
                    &[],
                    &mut outputs,
                    values,
                )?;
                if !distinct.is_new(&out) {
                    values = out.into_values();
                    values.clear();
                    continue;
                }
                values = Vec::new();
                out
            };
            kept.push((kept.len() as i64, out));
            if wanted.is_some_and(|wanted| kept.len() >= wanted) {
                break;
            }
        }
        if let Some(err) = scan.err() {
            return Err(err.clone());
        }
        ctx.check_cancelled()?;
        Ok(kept)
    }

    /// Run an in-family statement: scan its table narrowed by the
    /// conjuncts without a subquery, keep the rows the WHERE program
    /// passes, and project them. With no ORDER BY the scan stops once
    /// `limit` rows past `offset` passed, and both are applied here
    pub(crate) fn execute_prepared_select(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        compiled: Option<&RwLock<CompiledExecution>>,
        limit: Option<usize>,
        offset: usize,
    ) -> Result<PreparedOutput> {
        let inner_ctx = ctx.with_incremented_query_depth();
        // Without a WHERE or an ORDER BY the first rows are the answer
        let first = limit
            .filter(|_| stmt.where_clause.is_none() && stmt.order_by.is_empty())
            .map(|limit| limit.saturating_add(offset));
        let (source, scope_name) = self.read_source(stmt, ctx, &inner_ctx, first)?;
        let source = match (source, stmt.table_expr.as_deref()) {
            (Some(source), _) => source,
            (None, Some(Expression::TableSource(ts))) => {
                StatementSource::Table(self.prepared_table(&ts.name.value_lower, ctx, None)?)
            }
            (None, _) => return Err(Error::internal("a prepared statement without one table")),
        };
        let unnamed = SmartString::new("");
        let scope_name = scope_name.unwrap_or(&unnamed);
        let table_columns;
        let columns: &[String] = match &source {
            StatementSource::Table(table) => {
                table_columns = table.schema().column_names_arc();
                &table_columns
            }
            StatementSource::Rows(columns, _) | StatementSource::Values(columns, _) => columns,
            StatementSource::Cte((columns, _)) => columns,
        };
        let schema = || match &source {
            StatementSource::Table(table) => std::borrow::Cow::Borrowed(table.schema()),
            StatementSource::Rows(..) | StatementSource::Values(..) | StatementSource::Cte(_) => {
                std::borrow::Cow::Owned(untyped_schema(scope_name, columns))
            }
        };
        // A column change between taking the plan and opening a subquery's
        // table prepares again, against the table as it now is
        let mut fresh = false;
        let (prepared, tables) = loop {
            let prepared =
                self.prepared_for(compiled, stmt, columns, schema, scope_name, ctx, fresh)?;
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::prepared_plan_taken();
            if let Some(tables) =
                PreparedRunner::open_tables(self, &prepared.blocks, &inner_ctx, None)?
            {
                break (prepared, tables);
            }
            if fresh {
                return Err(Error::internal(
                    "a table's columns changed while the statement prepared",
                ));
            }
            fresh = true;
        };

        let statement_limit = limit
            .filter(|_| stmt.order_by.is_empty())
            .map(|limit| i64::try_from(limit).unwrap_or(i64::MAX));
        let statement_rows = match &source {
            StatementSource::Table(table) => table.row_count_hint(),
            StatementSource::Rows(_, rows) => rows.len(),
            StatementSource::Values(_, values) => values.rows.len(),
            StatementSource::Cte((_, rows)) => rows.len(),
        };
        let mut runner = PreparedRunner::new(
            self,
            &prepared.blocks,
            &prepared.holes,
            ctx,
            inner_ctx,
            tables,
            Some(statement_rows),
            statement_limit,
        );
        if !prepared.blocks.run_slots.is_empty() {
            let values = runner.run_slot_values()?;
            for (slot, value) in values {
                runner.base[slot] = value;
            }
        }
        runner.read_ahead(&prepared.ahead, &prepared.holes)?;
        let output_columns = CompactArc::clone(&prepared.output_columns);
        // Every value of a VALUES list is evaluated, as the general path does
        let target = limit
            .filter(|_| stmt.order_by.is_empty())
            .filter(|_| !matches!(source, StatementSource::Values(..)))
            .map(|limit| limit.saturating_add(offset));
        // Made on the first row: the scan's own answers need none
        let mut kept: Option<RowVec> = None;
        let mut keep = |_: i64, scanned: u64, row: Row| {
            let rows = kept.get_or_insert_with(|| RowVec::with_capacity(64));
            rows.push((scanned as i64, row));
            target.is_none_or(|target| rows.len() < target)
        };
        // An ORDER BY of one column under a LIMIT reads the rows it keeps in
        // order, through the column's index or the table's top-k scan
        let ordered = match (&source, limit) {
            (StatementSource::Table(table), Some(limit)) => match Self::ordered_read_column(stmt) {
                Some(key) => {
                    runner.ordered_rows(&prepared.where_, &**table, ctx, key, limit, offset)?
                }
                None => None,
            },
            _ => None,
        };
        let read_in_order = ordered.is_some();
        if read_in_order {
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::prepared_ordered_read();
            super::result::OrderLimitPath::PreparedOrderedRead.taken();
        }
        // A DISTINCT whose output asks no block keeps the first row of each
        // value as the scan passes it; one that does is told its rows first.
        // A set operation's DISTINCT is its first branch's, and a first
        // branch whose set operation removes duplicates is one too
        let dedups_branch = stmt.set_operations.first().is_some_and(|op| {
            matches!(
                op.operation,
                SetOperationType::Union | SetOperationType::Intersect | SetOperationType::Except
            )
        });
        // A DISTINCT ON keeps its first row per key once the rows are sorted
        let distinct = ((stmt.distinct || dedups_branch)
            && stmt.distinct_on.is_empty()
            && !stmt
                .columns
                .iter()
                .chain(stmt.order_by.iter().map(|ob| &ob.expression))
                .any(mentions_subquery))
        .then(|| DistinctRows::new(prepared.visible));
        // Rows nothing sorts, counts or merges, whose output is the row or
        // columns of it, reach the caller as the scan reads them, which checks
        // the cancel; a timed read is collected, so its deadline holds while it reads
        let projection = &prepared.projection;
        let flows = !ctx.has_timeout()
            && distinct.is_none()
            && stmt.order_by.is_empty()
            && limit.is_none()
            && offset == 0
            && (projection.whole_row || projection.columns.is_some());
        let read = match (ordered, source) {
            (Some(rows), _) => Ok(Passing::Answers(rows)),
            (None, StatementSource::Table(table)) => runner.passing_rows(
                &prepared.where_,
                None,
                &*table,
                ctx,
                target,
                distinct.is_some() || flows,
                &mut keep,
            ),
            (None, StatementSource::Rows(_, rows)) => runner.rows_passing(
                &prepared.where_,
                rows.into_iter().map(Ok),
                ctx,
                target,
                &mut keep,
            ),
            (None, StatementSource::Values(_, values)) => runner.rows_passing(
                &prepared.where_,
                (0i64..)
                    .zip(&values.rows)
                    .map(|(id, exprs)| Self::values_row(exprs, ctx).map(|row| (id, row))),
                ctx,
                target,
                &mut keep,
            ),
            (None, StatementSource::Cte((_, rows))) => runner.rows_passing(
                &prepared.where_,
                rows.iter().map(|(id, row)| Ok((*id, row))),
                ctx,
                target,
                &mut keep,
            ),
        };
        let streamed = |scan: Box<dyn Scanner>,
                        bound: Option<SharedProgram>|
         -> Result<PreparedOutput> {
            // A flowing read: the general path's scan, filter and projection
            let Some(distinct) = distinct else {
                let names = match projection.whole_row {
                    true => &output_columns,
                    false => &prepared.columns,
                };
                let mut rows: Box<dyn QueryResult> = Box::new(
                    ScannerResult::new(scan, CompactArc::clone(names))
                        .with_cancel(ctx.cancellation_handle()),
                );
                if let Some(bound) = bound {
                    let mut filter = RowFilter::from_program(bound).with_context(ctx);
                    // The values the run computed once sit past the statement's own
                    if runner.base.len() > ctx.params().len() {
                        filter = filter.with_params(CompactArc::new(runner.base.clone()));
                    }
                    rows = Box::new(FilteredResult::from_filter(rows, filter));
                }
                if let Some(columns) = projection
                    .columns
                    .as_ref()
                    .filter(|_| !projection.whole_row)
                {
                    rows = Box::new(ProjectionResult::new(
                        rows,
                        Projection::columns(columns.iter().copied()),
                        output_columns.to_vec(),
                    ));
                }
                return Ok((
                    rows,
                    CompactArc::clone(&output_columns),
                    false,
                    prepared.visible,
                    prepared.layout.clone(),
                    false,
                ));
            };
            // Without an ORDER BY the first rows a LIMIT keeps end the read;
            // a set operation's LIMIT counts its own rows, not the branch's
            let branch = !stmt.set_operations.is_empty();
            let wanted = super::row_count::rows_needed(stmt, ctx)
                .filter(|_| stmt.order_by.is_empty() && !branch);
            let projection = &prepared.projection;
            let rows =
                self.distinct_scan(scan, bound, &runner, projection, distinct, wanted, ctx)?;
            let result = ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), rows);
            Ok((
                Box::new(result),
                CompactArc::clone(&output_columns),
                false,
                prepared.visible,
                prepared.layout.clone(),
                true,
            ))
        };
        let mut rows = match read? {
            Passing::Pruned => {
                let result = ExecutorResult::with_arc_columns(
                    CompactArc::clone(&output_columns),
                    RowVec::new(),
                );
                return Ok((
                    Box::new(result),
                    output_columns,
                    true,
                    prepared.visible,
                    None,
                    false,
                ));
            }
            Passing::Answers(answers) => answers,
            Passing::Read => kept.unwrap_or_else(|| RowVec::with_capacity(64)),
            Passing::Scan(scan) => return streamed(scan, None),
            Passing::Filtered(scan, bound) => return streamed(scan, Some(bound)),
        };
        // Without an ORDER BY the rows LIMIT and OFFSET keep are the ones the
        // output is computed for, whether or not the read stopped early
        let mut limit_applied = stmt.order_by.is_empty() && (limit.is_some() || offset > 0);
        if limit_applied {
            let keep = limit.unwrap_or(usize::MAX);
            rows = rows.into_iter().skip(offset).take(keep).collect();
        }
        limit_applied |= read_in_order;
        // With an ORDER BY a SELECT item is computed for the rows LIMIT and
        // OFFSET keep, once the sort chose them by the columns the general
        // sort reads
        let projection = &prepared.projection;
        let order = (!read_in_order
            && !stmt.order_by.is_empty()
            && (limit.is_some() || projection.defers() && offset > 0))
            .then(|| {
                let layout = prepared.layout.as_deref();
                let (mapped, _) = self.order_by_columns(stmt, &output_columns, layout, false);
                mapped
                    .into_iter()
                    .zip(&stmt.order_by)
                    .map(|(column, ob)| column.map(|c| (Some(c), ob.ascending, ob.nulls_first)))
                    .collect::<Option<Vec<_>>>()
            })
            .flatten();
        // Keys that are source columns choose the rows before any is projected
        let source = |column: Option<usize>| column.and_then(|c| projection.source_column(c));
        let source_order = limit
            .and(order.as_ref())
            .filter(|order| order.iter().all(|&(column, ..)| source(column).is_some()))
            .map(|order| {
                order
                    .iter()
                    .map(|&(column, ascending, nulls_first)| {
                        (source(column), ascending, nulls_first)
                    })
                    .collect::<Vec<_>>()
            });
        if let (Some(source_order), Some(limit)) = (source_order, limit) {
            super::result::OrderLimitPath::PreparedTopRows.taken();
            rows = top_rows(rows, source_order, limit, offset);
            limit_applied = true;
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_projection_ran();
        // The output's blocks are asked of these rows, not of the table's
        runner.ask_rows(Some(rows.len()));
        let rows = match order.filter(|_| !limit_applied) {
            Some(order) => {
                super::result::OrderLimitPath::PreparedProjectFirst.taken();
                limit_applied = true;
                projection.project_first(rows, &runner, order, limit, offset)?
            }
            None => projection.project(rows, &runner)?,
        };
        let result = ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), rows);
        Ok((
            Box::new(result),
            output_columns,
            limit_applied,
            prepared.visible,
            prepared.layout.clone(),
            false,
        ))
    }
}

/// The rows past `offset`, at most `limit`, in `order` over their own
/// columns: the general sort's top-N, so ties keep the rows it keeps
fn top_rows(
    rows: RowVec,
    order: Vec<(Option<usize>, bool, Option<bool>)>,
    limit: usize,
    offset: usize,
) -> RowVec {
    let mut top = super::result::TopN::new(
        move |a: &Row, b: &Row| Executor::compare_rows_with_indices(a, b, &order),
        limit,
        offset,
    );
    for (_, row) in rows {
        top.push(row);
    }
    (0i64..).zip(top.into_rows()).collect()
}

/// A row a source hands the runner: its own, or one it lends, copied only
/// when the WHERE keeps it
trait SourceRow {
    fn row(&self) -> &Row;
    fn into_row(self) -> Row;
}

impl SourceRow for Row {
    fn row(&self) -> &Row {
        self
    }
    fn into_row(self) -> Row {
        self
    }
}

impl SourceRow for &Row {
    fn row(&self) -> &Row {
        self
    }
    fn into_row(self) -> Row {
        self.clone()
    }
}

/// What a prepared statement reads its rows from
enum StatementSource<'s> {
    Table(Box<dyn Table>),
    /// A derived table's columns and rows, read once
    Rows(Vec<String>, RowVec),
    /// A VALUES list's columns and rows, each evaluated as the WHERE reads it
    Values(Vec<String>, &'s ValuesTableSource),
    /// A CTE's columns and rows, materialized by the statement's WITH
    Cte(&'s super::cte::CteData),
}

impl StatementSource<'_> {
    /// The columns of the rows it reads; None for a table, whose schema has them
    fn columns(&self) -> Option<&[String]> {
        match self {
            StatementSource::Table(_) => None,
            StatementSource::Rows(columns, _) | StatementSource::Values(columns, _) => {
                Some(columns)
            }
            StatementSource::Cte((columns, _)) => Some(columns),
        }
    }
}

/// `expr`, read over a group's row: each column it names, resolved in the
/// `chain`, bound to the group item `item_of` reads it from
fn over_items(
    expr: &Expression,
    chain: &Chain<'_>,
    item_of: &dyn Fn(usize, &Expression) -> Option<usize>,
) -> Result<Expression> {
    match expr {
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
            match chain.resolve(expr)? {
                None => Ok(expr.clone()),
                Some((depth, column)) => {
                    let item = item_of(column, expr)
                        .ok_or_else(|| Error::internal("a column outside the groups"))?;
                    let name = chain.scopes[depth].lower[column].as_str();
                    Ok(Expression::Column(ColumnRef {
                        index: item as u32,
                        name: ident(name, name),
                    }))
                }
            }
        }
        other => rebuild(other, &mut |child| over_items(child, chain, item_of)),
    }
}

/// One source a join reads: the name qualifying its columns, lowercase,
/// their names when known before the join reads it, and how many there are
/// when that is known
pub(crate) struct JoinLeaf {
    pub(crate) name: String,
    pub(crate) columns: Option<Vec<String>>,
    pub(crate) width: Option<usize>,
}

/// A derived table's columns as its scope's schema: their names and no
/// type, so a value compared with one is not read as a column's type
fn untyped_schema(name: &str, columns: &[String]) -> Schema {
    let columns = columns
        .iter()
        .enumerate()
        .map(|(i, name)| SchemaColumn::new(i, name.as_str(), DataType::Null, true, false))
        .collect();
    Schema::new(name, columns)
}

/// Rows read in place of a scan: the ones a primary key equality read by
/// key, or a block source's, lent for one call
struct PointRows<R> {
    rows: R,
    at: Option<usize>,
}

impl<R: std::ops::Deref<Target = Vec<(i64, Row)>> + Send> Scanner for PointRows<R> {
    fn next(&mut self) -> bool {
        let at = self.at.map_or(0, |at| at + 1);
        self.at = Some(at);
        at < self.rows.len()
    }

    fn row(&self) -> &Row {
        &self.rows[self.at.unwrap_or(0)].1
    }

    fn err(&self) -> Option<&Error> {
        None
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The schema of a scope read before any row: a table without columns
pub(crate) static NO_COLUMNS: std::sync::LazyLock<Schema> =
    std::sync::LazyLock::new(|| Schema::new("", Vec::new()));

/// Each subquery's table or source rows by its id; None for a block without
/// FROM, or one run as a statement of its own
type BlockTables = SmallVec<[Option<BlockInput>; 2]>;

/// A block source's rows, shared by every outer row of a run
type SourceRows = CompactArc<Vec<(i64, Row)>>;

/// What a block reads its candidates from
enum BlockInput {
    Table(Box<dyn Table>),
    /// A view's, CTE's, VALUES list's, table function's or derived table's
    /// rows, read once per run
    Rows(SourceRows),
    /// The same, read when a call first needs them
    Lazy(std::cell::OnceCell<SourceRows>),
}

impl BlockInput {
    fn table(&self) -> Option<&dyn Table> {
        match self {
            Self::Table(table) => Some(&**table),
            Self::Rows(_) | Self::Lazy(_) => None,
        }
    }

    fn row_count(&self) -> usize {
        match self {
            Self::Table(table) => table.row_count_hint(),
            Self::Rows(rows) => rows.len(),
            Self::Lazy(rows) => rows.get().map_or(1, |rows| rows.len()),
        }
    }
}

/// The one row without columns a block without FROM reads
struct OneRow {
    row: Row,
    read: bool,
}

impl Scanner for OneRow {
    fn next(&mut self) -> bool {
        !std::mem::replace(&mut self.read, true)
    }

    fn row(&self) -> &Row {
        &self.row
    }

    fn err(&self) -> Option<&Error> {
        None
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// One execution of a prepared statement. Each subquery has its own
/// parameter buffer: the statement's parameters, then every slot, its
/// parent's values copied in before its own holes are filled
pub(crate) struct PreparedRunner<'a> {
    executor: &'a Executor,
    prepared: &'a PreparedBlocks,
    /// The holes of the statement's own blocks
    holes: &'a HoleSet,
    /// The statement's own context, whose view and parameters a kept answer is of
    statement: &'a ExecutionContext,
    /// The rows the statement asks its blocks for, by its hint; a join
    /// knows them only once a site has them
    statement_rows: Cell<Option<usize>>,
    /// The rows the statement stops at, when no ORDER BY reads them all
    statement_limit: Option<i64>,
    /// The statement's parameters, then its run slots; the other slots NULL
    base: ParamVec,
    transaction_id: Option<u64>,
    /// The context the subqueries read their tables under
    ctx: ExecutionContext,
    /// Each subquery's table, opened before the run reads a row
    tables: BlockTables,
    /// Each subquery's state for this run, by its id
    blocks: Vec<BlockState>,
    /// The WHERE's programs bound to the run, each once a row asks it, by
    /// [`WhereRest`]; boxed, as a larger runner costs a point query its move
    where_bound: OnceCell<Box<[OnceCell<RefCell<Bound>>; 4]>>,
}

/// A statement program bound to its run's parameters and run slots and to
/// the uncorrelated blocks read when it was bound
pub(crate) struct Bound {
    source: SharedProgram,
    program: SharedProgram,
    /// The uncorrelated block ops not read when it was bound
    pending: SmallVec<[Op; 2]>,
}

/// What one run keeps for one subquery
struct BlockState {
    buffer: RefCell<ParamVec>,
    vm: RefCell<ExprVM>,
    /// Its pushed comparisons, given each call's values in place
    bound: RefCell<Option<BoundPushdown>>,
    /// An uncorrelated block's value, once computed
    value: RefCell<Option<Value>>,
    /// How many times its buffer was filled in this run
    fills: std::cell::Cell<u64>,
    /// A block reading only its ancestors' values: its answer, by the parent's fill
    answered: RefCell<Option<(u64, BlockAnswer)>>,
    /// A keyed EXISTS or IN block's choice, once made
    keyed_choice: std::cell::Cell<Option<bool>>,
    keyed_set: RefCell<Option<CompactArc<ExistsKeys>>>,
    /// A value block's members: an uncorrelated block's, or a keyed IN
    /// block's grouped by key, set once per run
    members: std::cell::OnceCell<BlockAnswer>,
    /// The error a read of an uncorrelated block's value, of the members or
    /// of the keyed set raised in this run, which every row that needs them
    /// raises without reading again
    read_error: RefCell<Option<Error>>,
    /// The slots whose value raised an error for this call, its parent's
    /// included: an EXISTS or value block raises one only on a candidate
    failed: RefCell<FailedSlots>,
    /// A keyed grouped block's slot rows by key, once read
    grouped: RefCell<Option<GroupedKeys>>,
    /// The VM a grouped block's calls read their inputs with
    feed_vm: RefCell<Option<ExprVM>>,
    /// The row a primary key equality read by key, kept for the next call
    point_rows: RefCell<RowVec>,
    /// The row ids an index count reads, kept for the next call
    count_ids: RefCell<Option<Box<CountIds>>>,
}

/// The row ids of one index probe and the scratch it reads them with
type CountIds = (Vec<i64>, ProbeScratch);

/// The most row ids an index count keeps room for between calls
const COUNT_IDS_KEPT: usize = 4096;

impl<'a> PreparedRunner<'a> {
    /// Every subquery's table as `inner_ctx` reads it, through `through`
    /// when given, or None when one no longer holds the columns its
    /// programs were prepared for
    fn open_tables(
        executor: &Executor,
        prepared: &PreparedBlocks,
        inner_ctx: &ExecutionContext,
        through: Option<&dyn Transaction>,
    ) -> Result<Option<BlockTables>> {
        let mut tables = BlockTables::with_capacity(prepared.subqueries.len());
        for sq in &prepared.subqueries {
            if let Some(source) = &sq.source {
                let input = match source.lazy() {
                    true => match executor.source_holds(source)? {
                        true => BlockInput::Lazy(std::cell::OnceCell::new()),
                        false => return Ok(None),
                    },
                    false => match executor.source_rows(source, &sq.schema, inner_ctx)? {
                        Some(rows) => BlockInput::Rows(rows),
                        None => return Ok(None),
                    },
                };
                tables.push(Some(input));
                continue;
            }
            // A block run as a statement of its own opens its table itself
            let runs_alone = sq.grouping.as_ref().is_some_and(|g| g.statement.is_some());
            let (Some(name), false) = (&sq.table, runs_alone) else {
                tables.push(None);
                continue;
            };
            let table = executor.prepared_table(name, inner_ctx, through)?;
            if table.schema().columns != sq.schema.columns {
                return Ok(None);
            }
            tables.push(Some(BlockInput::Table(table)));
        }
        Ok(Some(tables))
    }

    /// Sets the rows a join's site asks its blocks for, before its first ask
    pub(crate) fn ask_rows(&self, rows: Option<usize>) {
        self.statement_rows.set(rows);
    }

    /// The rows block `index` reads, through `expr` when its scan takes one:
    /// its table's, its source's, which no pushdown reaches, or the one row
    /// without columns of a block without FROM
    fn scan(&self, index: usize, expr: Option<&dyn StorageExpr>) -> Result<Box<dyn Scanner>> {
        match &self.tables[index] {
            Some(BlockInput::Table(table)) => {
                table.scan(&self.prepared.subqueries[index].column_indices, expr)
            }
            Some(BlockInput::Rows(rows)) => Ok(Box::new(PointRows {
                rows: CompactArc::clone(rows),
                at: None,
            })),
            Some(BlockInput::Lazy(read)) => Ok(Box::new(PointRows {
                rows: CompactArc::clone(self.lazy_rows(index, read)?),
                at: None,
            })),
            None => Ok(Box::new(OneRow {
                row: Row::new(),
                read: false,
            })),
        }
    }

    /// The rows of block `index`'s source, read on the first call that needs them
    fn lazy_rows<'c>(
        &self,
        index: usize,
        read: &'c std::cell::OnceCell<SourceRows>,
    ) -> Result<&'c SourceRows> {
        if read.get().is_none() {
            let sq = &self.prepared.subqueries[index];
            let source = sq
                .source
                .as_ref()
                .ok_or_else(|| Error::internal("a block read lazily without a source"))?;
            let rows = self
                .executor
                .source_rows(source, &sq.schema, &self.ctx)?
                .ok_or_else(|| Error::internal("a block source's columns changed in its run"))?;
            let _ = read.set(rows);
        }
        read.get()
            .ok_or_else(|| Error::internal("a block source left unread"))
    }

    /// [`Self::scan`] through the block's own bound pushdown `expr`: a
    /// primary key equality bound to an integer reads that key's row, the
    /// whole pushdown its filter, so it passes the rows the scan would
    fn point_or_scan(
        &self,
        index: usize,
        expr: Option<&dyn StorageExpr>,
    ) -> Result<Box<dyn Scanner>> {
        let sq = &self.prepared.subqueries[index];
        let table = self.tables[index].as_ref().and_then(BlockInput::table);
        if let (Some(point), Some(expr), Some(table)) = (sq.point, expr, table) {
            let params = self.blocks[index].buffer.borrow();
            if let Some(Value::Integer(key)) = sq.pushdown[point].value(&params) {
                let rows = table.fetch_rows_by_ids(&[key], expr)?;
                return Ok(Box::new(PointRows { rows, at: None }));
            }
        }
        self.scan(index, expr)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        executor: &'a Executor,
        prepared: &'a PreparedBlocks,
        holes: &'a HoleSet,
        ctx: &'a ExecutionContext,
        inner_ctx: ExecutionContext,
        tables: BlockTables,
        statement_rows: Option<usize>,
        statement_limit: Option<i64>,
    ) -> Self {
        let mut base: ParamVec = ctx.params().iter().cloned().collect();
        base.resize(prepared.slot_count, Value::null_unknown());
        let count = prepared.subqueries.len();
        Self {
            executor,
            prepared,
            holes,
            statement: ctx,
            statement_rows: Cell::new(statement_rows),
            statement_limit,
            blocks: (0..count)
                .map(|_| BlockState {
                    buffer: RefCell::new(base.clone()),
                    vm: RefCell::new(ExprVM::new()),
                    bound: RefCell::new(None),
                    value: RefCell::new(None),
                    fills: std::cell::Cell::new(0),
                    answered: RefCell::new(None),
                    keyed_choice: std::cell::Cell::new(None),
                    keyed_set: RefCell::new(None),
                    members: std::cell::OnceCell::new(),
                    read_error: RefCell::new(None),
                    failed: RefCell::new(SmallVec::new()),
                    grouped: RefCell::new(None),
                    feed_vm: RefCell::new(None),
                    point_rows: RefCell::new(RowVec::new()),
                    count_ids: RefCell::new(None),
                })
                .collect(),
            base,
            transaction_id: ctx.transaction_id(),
            ctx: inner_ctx,
            tables,
            where_bound: OnceCell::new(),
        }
    }

    /// Read the members of each uncorrelated block of `ids` now, which keeps
    /// them for every row that asks later
    fn read_ahead(&self, ids: &[u16], holes: &HoleSet) -> Result<()> {
        let empty = Row::new();
        let caller = ExecuteContext::with_common_params(&empty, &self.base, self.transaction_id);
        for &id in ids {
            let width = self.prepared.subqueries[usize::from(id)].member_width();
            let nulls: SmallVec<[Value; 2]> = smallvec::smallvec![Value::null_unknown(); width];
            self.members_answer(usize::from(id), &nulls, &caller, holes)?;
        }
        Ok(())
    }

    /// The statement's run slots, each computed once over no row
    fn run_slot_values(&self) -> Result<Vec<(usize, Value)>> {
        let empty = Row::new();
        let mut vm = ExprVM::new();
        let exec = ExecuteContext::with_common_params(&empty, &self.base, self.transaction_id)
            .with_subquery_executor(self);
        self.prepared
            .run_slots
            .iter()
            .map(|(slot, program)| Ok((*slot, vm.execute(program, &exec)?)))
            .collect()
    }

    /// Whether `row` passes `condition`, a program over the statement's row
    fn matches(&self, vm: &mut ExprVM, condition: &SharedProgram, row: &Row) -> Result<bool> {
        let exec = ExecuteContext::with_common_params(row, &self.base, self.transaction_id)
            .with_subquery_executor(self);
        vm.execute_bool_checked(condition, &exec)
    }

    /// Whether `row` passes `program`, see [`Self::passes`]. Out of line: a
    /// scan loop that builds the context costs a point query its codegen
    #[inline(never)]
    fn row_passes(&self, vm: &mut ExprVM, program: &WhereProgram, row: &Row) -> Result<bool> {
        let exec = ExecuteContext::with_common_params(row, &self.base, self.transaction_id)
            .with_subquery_executor(self);
        self.passes(vm, program, &exec)
    }

    /// Whether the row of `exec` passes `program`, one of a WHERE's, bound
    /// to what the run knows once a row asks it, unless binding cannot
    /// shrink it
    fn passes(
        &self,
        vm: &mut ExprVM,
        program: &WhereProgram,
        exec: &ExecuteContext,
    ) -> Result<bool> {
        if !program.binds {
            return vm.execute_bool_checked(&program.program, exec);
        }
        let cell = &self.where_bound.get_or_init(Box::default)[program.rest as usize];
        let bound = cell.get_or_init(|| RefCell::new(self.bind(&program.program)));
        match bound.try_borrow_mut() {
            Ok(mut bound) if CompactArc::ptr_eq(&bound.source, &program.program) => {
                vm.execute_bool_checked(self.current(&mut bound), exec)
            }
            _ => vm.execute_bool_checked(&program.program, exec),
        }
    }

    /// The subquery's pushed comparisons for this call, None when there
    /// are none or one value is not one the scan can compare by
    fn bind_pushdown(&self, index: usize, params: &[Value]) -> bool {
        let sq = &self.prepared.subqueries[index];
        if sq.pushdown.is_empty() {
            return false;
        }
        let values: Option<Vec<Value>> = sq.pushdown.iter().map(|c| c.value(params)).collect();
        match values {
            Some(values) => {
                BoundPushdown::bind(
                    &sq.pushdown,
                    &mut self.blocks[index].bound.borrow_mut(),
                    values,
                );
                true
            }
            None => false,
        }
    }

    /// The scan `where_`'s conjuncts without a subquery and the `extra`
    /// ones narrow, and its set conjuncts once their members are read, the
    /// `narrowed` one too when given, with whether it takes the whole
    /// WHERE; None when it proves no row passes
    fn where_scan(
        &self,
        where_: &PreparedWhere,
        extra: Option<&Expression>,
        narrowed: Option<SetSlot>,
        table: &dyn Table,
        ctx: &ExecutionContext,
    ) -> Result<Option<WhereScan>> {
        let both;
        let pushdown = match (&where_.pushdown, extra) {
            (Some(own), Some(extra)) => {
                both = conjunction(vec![own.clone(), extra.clone()]);
                both.as_ref()
            }
            (own, extra) => own.as_ref().or(extra),
        };
        let prepared = match (&where_.push, extra) {
            (Some(push), None) => push.storage(&self.base),
            _ => None,
        };
        #[cfg(feature = "test-failpoints")]
        if prepared.is_none() && pushdown.is_some() {
            crate::test_failpoints::where_pushdown_converted();
        }
        let (mut storage_expr, residual) = match (prepared, pushdown) {
            (Some(expr), _) => (Some(expr), false),
            (None, Some(pushdown)) if self.prepared.run_slots.is_empty() => {
                pushdown::try_pushdown(pushdown, table.schema(), Some(ctx))
            }
            (None, Some(pushdown)) => {
                let mut run_ctx = ctx.clone();
                run_ctx.set_params(self.base.clone());
                pushdown::try_pushdown(pushdown, table.schema(), Some(&run_ctx))
            }
            (None, None) => (None, true),
        };
        // The conjuncts the scan takes whole, of the WHERE's
        let static_whole = where_.pushdown.is_none() || (storage_expr.is_some() && !residual);
        let extra_whole = extra.is_none() || (storage_expr.is_some() && !residual);
        let mut taken = if where_.pushdown.is_some() && static_whole {
            where_.pushed
        } else {
            0
        };
        if !where_.sets.is_empty() || narrowed.is_some() {
            let slots = where_.sets.iter().chain(narrowed.as_ref());
            match self.with_sets(slots, storage_expr.take(), &mut taken, table, ctx)? {
                Some(expr) => storage_expr = expr,
                None => return Ok(None),
            }
        }
        if let Some(expr) = &storage_expr {
            if self
                .executor
                .get_query_planner()
                .can_prune_entire_scan(table, expr.as_ref())
            {
                return Ok(None);
            }
        }
        // The scan took the whole WHERE: its rows need no second look
        let scanned_whole = static_whole && taken == where_.conjuncts;
        // The scan took the pushed conjuncts: its rows need the others only
        let rest = match where_.pushdown.is_some() && static_whole {
            true => WhereRest::Scanned,
            false => WhereRest::Whole,
        };
        Ok(Some((storage_expr, scanned_whole, extra_whole, rest)))
    }

    /// `storage_expr` with the filters the set conjuncts of `slots` make
    /// for this run, each one the scan takes counted in `taken`; None when
    /// one keeps no row
    fn with_sets<'s>(
        &self,
        slots: impl Iterator<Item = &'s SetSlot>,
        storage_expr: Option<Box<dyn StorageExpr>>,
        taken: &mut usize,
        table: &dyn Table,
        ctx: &ExecutionContext,
    ) -> Result<Option<Option<Box<dyn StorageExpr>>>> {
        let mut filters: SmallVec<[Box<dyn StorageExpr>; 2]> = storage_expr.into_iter().collect();
        for slot in slots {
            let scan = self.set_scan(slot, table, ctx)?;
            #[cfg(feature = "test-failpoints")]
            if !matches!(scan, SetScan::Program) {
                crate::test_failpoints::prepared_strategy_ran(
                    crate::test_failpoints::PreparedStrategy::ScanTakesMembers,
                );
            }
            match scan {
                SetScan::Filter(filter) => {
                    filters.push(filter);
                    *taken += 1;
                }
                SetScan::Every => *taken += 1,
                SetScan::None => return Ok(None),
                SetScan::Program => {}
            }
        }
        Ok(Some(match filters.len() {
            0 | 1 => filters.pop(),
            _ => Some(Box::new(AndExpr::new(filters.into_vec()))),
        }))
    }

    /// What `slot`'s members make of its conjunct for this run, read now,
    /// before the scan: by the three-valued rule, an UNKNOWN keeps no row
    fn set_scan(
        &self,
        slot: &SetSlot,
        table: &dyn Table,
        ctx: &ExecutionContext,
    ) -> Result<SetScan> {
        let index = slot.id as usize;
        // An error reading the members, or one a member raised, belongs to
        // the row that needs the answer: the program asks the block then
        let empty = Row::new();
        let read = self.read_member_set(index, &ExecuteContext::new(&empty), self.holes);
        if read.is_err() {
            return Ok(SetScan::Program);
        }
        let Some(BlockAnswer::Members(set)) = self.blocks[index].members.get() else {
            return Ok(SetScan::Program);
        };
        if set.failed.is_some() {
            return Ok(SetScan::Program);
        }
        let schema = table.schema();
        let column = || schema.columns[slot.column].name_lower.clone();
        let scan = pushdown::PushdownContext::new(schema, Some(ctx));
        let filter = |filter: Option<Box<dyn StorageExpr>>| match filter {
            Some(filter) => SetScan::Filter(filter),
            None => SetScan::Program,
        };
        let float_column = schema.columns[slot.column].data_type == DataType::Float;
        let list = |values: &mut dyn Iterator<Item = Value>, negated: bool| {
            match pushdown::in_list_filter(column(), values, negated, &scan) {
                Some(list) if list.tests_cheaply(float_column) => SetScan::Filter(Box::new(list)),
                _ => SetScan::Program,
            }
        };
        Ok(match &set.set {
            // A key's set is a correlated block's, which no scan takes, and
            // a row's is tested against no column
            MemberSet::In(_) | MemberSet::Rows(_) => SetScan::Program,
            MemberSet::SharedIn(set_of) => {
                let members = set_of.members();
                let mut values = members.iter().filter(|v| !v.is_null()).cloned();
                match (slot.negated, set_of.has_null()) {
                    // Over no member NOT IN holds for every value, NULL too
                    (true, _) if members.is_empty() => SetScan::Every,
                    (true, true) => SetScan::None,
                    (false, _) if members.iter().all(Value::is_null) => SetScan::None,
                    (negated, _) => list(&mut values, negated),
                }
            }
            MemberSet::Quantified(quantified) => {
                let (op, all, values, has_null) = quantified.parts();
                match (op, all) {
                    // Over no member ALL holds for every value and ANY for none
                    _ if values.is_empty() && !has_null => match all {
                        true => SetScan::Every,
                        false => SetScan::None,
                    },
                    _ if values.is_empty() || (all && has_null) => SetScan::None,
                    (CompareOp::Eq, false) => list(&mut values.iter().cloned(), false),
                    (CompareOp::Ne, true) => list(&mut values.iter().cloned(), true),
                    _ => filter(pushdown::quantified_filter(
                        SmartString::new(&schema.columns[slot.column].name_lower),
                        op,
                        all,
                        values,
                        &scan,
                    )),
                }
            }
        })
    }

    /// The rows of `table` an ORDER BY of `column` keeps past `offset`, at
    /// most `limit`, read in order as the general path reads them: through
    /// the column's index without a filter, else the table's top-k scan.
    /// None when the scan does not take the whole WHERE, the WHERE narrows
    /// the rows by a set, or the table cannot read them in order
    fn ordered_rows(
        &self,
        where_: &PreparedWhere,
        table: &dyn Table,
        ctx: &ExecutionContext,
        (column, ascending, nulls_first): (&str, bool, Option<bool>),
        limit: usize,
        offset: usize,
    ) -> Result<Option<RowVec>> {
        if where_.narrowing.is_some() || where_.narrowing_set.is_some() {
            return Ok(None);
        }
        let Some((storage_expr, scanned_whole, ..)) =
            self.where_scan(where_, None, None, table, ctx)?
        else {
            return Ok(Some(RowVec::new()));
        };
        if !scanned_whole {
            return Ok(None);
        }
        if storage_expr.is_none() {
            let nulls_first = crate::core::value::nulls_first(ascending, nulls_first);
            let rows = table.collect_rows_ordered_by_index(
                column,
                ascending,
                nulls_first,
                limit,
                offset,
            )?;
            if rows.is_some() {
                return Ok(rows);
            }
        }
        let rows = table.scan_top_k(storage_expr.as_deref(), column, ascending, limit, offset)?;
        // A read that checks for no cancel is the statement's all the same
        ctx.check_cancelled()?;
        Ok(rows)
    }

    /// Hand `keep` each row of `table` that passes `where_`, with its row
    /// id and the count of rows read so far, until `keep` says stop: the
    /// rows its narrowing names, or a scan the conjuncts without a subquery
    /// narrow. With a `target` and a scan that takes the whole WHERE, the
    /// scan's own answers up to it instead. The scan takes the `extra`
    /// conditions too, which `keep` still judges; with `stream`, a scan
    /// that takes them and the WHERE whole is handed back instead
    #[allow(clippy::too_many_arguments)]
    fn passing_rows(
        &self,
        where_: &PreparedWhere,
        extra: Option<&Expression>,
        table: &dyn Table,
        ctx: &ExecutionContext,
        target: Option<usize>,
        stream: bool,
        mut keep: impl FnMut(i64, u64, Row) -> bool,
    ) -> Result<Passing> {
        let Some((mut storage_expr, mut scanned_whole, mut extra_whole, mut rest)) =
            self.where_scan(where_, extra, None, table, ctx)?
        else {
            return Ok(Passing::Pruned);
        };
        if target == Some(0) {
            return Ok(Passing::Read);
        }
        let mut vm = ExprVM::new();
        let mut scanned = 0u64;
        // The rows the statement reads: the ones its narrowing names, which
        // the conjuncts the scan would take still judge; the scan's answers
        // up to the LIMIT when it takes the whole WHERE, which are the rows;
        // or the scan, which takes the narrowing's own set when it declined
        let (mut listed, mut scanner, judged, passed) =
            match self.narrowed_rows(where_.narrowing, table, target)? {
                Some((candidates, settled)) => {
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::prepared_rows_narrowed_by_keys(
                        where_.narrowing.is_some_and(|n| n.negated),
                    );
                    // A named row needs no second look at the narrowing,
                    // nor at the pushed conjuncts the scan's filter took
                    rest = match (settled, rest) {
                        (true, WhereRest::Scanned) => WhereRest::NarrowedScanned,
                        (true, _) => WhereRest::Narrowed,
                        (false, _) => WhereRest::Whole,
                    };
                    let whole = where_.narrowing.is_some_and(|n| n.whole);
                    (Some(candidates.into_iter()), None, true, settled && whole)
                }
                None => {
                    if let Some(slot) = where_.narrowing_set {
                        let Some(scan) = self.where_scan(where_, extra, Some(slot), table, ctx)?
                        else {
                            return Ok(Passing::Pruned);
                        };
                        (storage_expr, scanned_whole, extra_whole, rest) = scan;
                    }
                    match target.filter(|_| scanned_whole) {
                        Some(target) => {
                            let answers = table.collect_rows_with_limit_unordered(
                                storage_expr.as_deref(),
                                target,
                                0,
                            )?;
                            // A read that checks for no cancel is the statement's all the same
                            ctx.check_cancelled()?;
                            return Ok(Passing::Answers(answers));
                        }
                        None => {
                            let all: Vec<usize> = (0..table.schema().columns.len()).collect();
                            let scan = table.scan(&all, storage_expr.as_deref())?;
                            if stream && scanned_whole && extra_whole {
                                return Ok(Passing::Scan(scan));
                            }
                            if stream && extra_whole {
                                if let Some(bound) = self.bound_program(&where_.rest(rest).program)
                                {
                                    return Ok(Passing::Filtered(scan, bound));
                                }
                            }
                            (None, Some(scan), false, scanned_whole)
                        }
                    }
                }
            };
        let chosen = where_.rest(rest);
        loop {
            let (id, row) = match (&mut listed, &mut scanner) {
                (Some(listed), _) => match listed.next() {
                    Some(entry) => entry,
                    None => break,
                },
                (None, Some(scanner)) => match scanner.next() {
                    true => (scanner.current_row_id(), scanner.take_row()),
                    false => break,
                },
                (None, None) => break,
            };
            scanned += 1;
            if scanned.is_multiple_of(100) {
                ctx.check_cancelled()?;
            }
            // One that fails to evaluate leaves the answer to the WHERE
            let taken = match storage_expr
                .as_ref()
                .filter(|_| judged)
                .map(|e| e.evaluate(&row))
            {
                Some(Ok(false)) => continue,
                Some(Err(_)) => false,
                Some(Ok(true)) | None => true,
            };
            let asked = if taken { chosen } else { &where_.whole };
            if ((passed && taken) || self.row_passes(&mut vm, asked, &row)?)
                && !keep(id, scanned, row)
            {
                break;
            }
        }
        if let Some(err) = scanner.as_ref().and_then(|scanner| scanner.err()) {
            return Err(err.clone());
        }
        // A read that checks for no cancel is the statement's all the same
        ctx.check_cancelled()?;
        Ok(Passing::Read)
    }

    /// Hand `keep` each of `rows` that passes `where_`, with the count of
    /// rows read so far, until `keep` says stop; a borrowed row is copied
    /// only when it passes. The set conjuncts test the members read before
    /// the first row, as a table's scan does. Inlined into a caller as large
    /// as `execute_prepared_select`, its loop costs more per row
    #[inline(never)]
    fn rows_passing<R: SourceRow>(
        &self,
        where_: &PreparedWhere,
        rows: impl IntoIterator<Item = Result<(i64, R)>>,
        ctx: &ExecutionContext,
        target: Option<usize>,
        mut keep: impl FnMut(i64, u64, Row) -> bool,
    ) -> Result<Passing> {
        if target == Some(0) {
            return Ok(Passing::Read);
        }
        let mut vm = ExprVM::new();
        let mut program = None;
        for (scanned, read) in (1u64..).zip(rows) {
            if scanned.is_multiple_of(100) {
                ctx.check_cancelled()?;
            }
            let (id, row) = read?;
            // Rows read in no table are narrowed by no key: the narrowing's
            // set is a set conjunct like the others
            let bound = program.get_or_insert_with(|| {
                let slots: SmallVec<[SetSlot; 2]> = where_
                    .sets
                    .iter()
                    .copied()
                    .chain(where_.narrowing_set)
                    .collect();
                self.with_bound_sets(&where_.whole.program, &slots)
            });
            if self.matches(&mut vm, self.current(bound), row.row())?
                && !keep(id, scanned, row.into_row())
            {
                break;
            }
        }
        ctx.check_cancelled()?;
        Ok(Passing::Read)
    }

    /// The statement's rows its narrowing EXISTS may pass, read by the keys
    /// of the block's keyed set when the run takes it, and whether each
    /// passes the narrowing's conjunct without asking it: no key holds an
    /// error; None to scan
    fn narrowed_rows(
        &self,
        narrowing: Option<Narrowing>,
        table: &dyn Table,
        target: Option<usize>,
    ) -> Result<Option<(RowVec, bool)>> {
        let Some(Narrowing {
            id,
            column,
            whole,
            negated,
            members,
        }) = narrowing
        else {
            return Ok(None);
        };
        let index = id as usize;
        let sq = &self.prepared.subqueries[index];
        if members {
            return self.rows_by_members(table, index, column, whole, negated, target);
        }
        let Answer::Exists(existence) = &sq.answer else {
            return Ok(None);
        };
        let Some(plan) = existence.keyed.as_ref().filter(|plan| !plan.checked) else {
            return Ok(None);
        };
        if !self.takes_keyed_set(index, plan) {
            return Ok(None);
        }
        // Over no row: the holes besides the keys read none
        {
            self.filled(index);
            let mut vm = self.blocks[index].vm.borrow_mut();
            let mut params = self.blocks[index].buffer.borrow_mut();
            let mut failed = self.blocks[index].failed.borrow_mut();
            failed.clear();
            let empty = Row::new();
            let exec = ExecuteContext::with_common_params(&empty, &self.base, self.transaction_id);
            for (slot, hole) in self.holes_of(index, self.holes) {
                if !plan.keys.iter().any(|(_, key, _)| key == slot) {
                    match hole.value(&mut vm, &empty, &exec) {
                        Ok(value) => params[*slot] = value,
                        Err(e) => {
                            params[*slot] = Value::null_unknown();
                            failed.push((*slot, deferred(e)?));
                        }
                    }
                }
            }
        }
        // An error building the set belongs to the row that needs it
        let Ok(set) = self.keyed_set(index, existence, plan) else {
            return Ok(None);
        };
        let passed = whole && !set.failed;
        // Only rows that pass without the WHERE may stop at the LIMIT
        let enough = target.filter(|_| passed);
        let column = &table.schema().columns[column].name_lower;
        let rows = if negated {
            // A key holding an error is not left out: its rows raise it
            let passing = set
                .outcomes
                .iter()
                .filter(|(_, outcome)| matches!(outcome, ExistsOutcome::Passed))
                .filter_map(|(key, _)| key.first());
            Executor::rows_without_members(table, column, passing, enough)?
        } else {
            let members = set.outcomes.keys().filter_map(|key| key.first());
            Executor::rows_of_members(table, column, members, target.is_some(), enough)?
        };
        Ok(rows.map(|rows| (rows, !set.failed)))
    }

    /// What fills the block's slots: the calling site's holes for one of
    /// the statement's own blocks, the block's for a nested one
    fn holes_of<'s>(&'s self, index: usize, site: &'s HoleSet) -> &'s [(usize, Hole)] {
        let sq = &self.prepared.subqueries[index];
        match sq.parent {
            None => site.holes.get(index).map_or(&[], Vec::as_slice),
            Some(_) => &sq.holes,
        }
    }

    /// Count a new fill of block `index`'s buffer
    fn filled(&self, index: usize) {
        let fills = &self.blocks[index].fills;
        fills.set(fills.get() + 1);
    }

    /// The parent's fill a correlated block reading no row of its parent
    /// answers for: one answer holds until the parent is filled again
    fn parent_call(&self, sq: &PreparedSubquery) -> Option<u64> {
        let parent = sq
            .parent
            .filter(|_| !sq.uncorrelated && sq.holes.is_empty())?;
        Some(self.blocks[parent as usize].fills.get())
    }

    /// Block `index`'s answer kept for the parent's fill `call`
    fn answered(&self, index: usize, call: Option<u64>) -> Option<BlockAnswer> {
        let call = call?;
        match self.blocks[index].answered.borrow().as_ref() {
            Some((at, answer)) if *at == call => Some(answer.clone()),
            _ => None,
        }
    }

    /// Keep `answer` as block `index`'s for the parent's fill `call`
    fn answer_call(&self, index: usize, call: Option<u64>, answer: impl FnOnce() -> BlockAnswer) {
        if let Some(call) = call {
            *self.blocks[index].answered.borrow_mut() = Some((call, answer()));
        }
    }

    /// The subquery's parameter buffer for a call from `caller` at `site`:
    /// its parent's values, then its own holes over the caller's row or pair
    fn fill(
        &self,
        index: usize,
        vm: &mut ExprVM,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<()> {
        let sq = &self.prepared.subqueries[index];
        self.filled(index);
        let mut params = self.blocks[index].buffer.borrow_mut();
        let mut failed = self.blocks[index].failed.borrow_mut();
        failed.clear();
        if let Some(parent) = sq.parent {
            params.clone_from(&self.blocks[parent as usize].buffer.borrow());
            failed.extend(self.blocks[parent as usize].failed.borrow().iter().cloned());
        }
        let mut hole_ctx =
            ExecuteContext::with_common_params(caller.row, &self.base, self.transaction_id);
        hole_ctx.row2 = caller.row2;
        for (slot, hole) in self.holes_of(index, site) {
            match hole.value(vm, caller.row, &hole_ctx) {
                Ok(value) => params[*slot] = value,
                Err(e) => {
                    params[*slot] = Value::null_unknown();
                    failed.push((*slot, deferred(e)?));
                }
            }
        }
        // A failed slot its nested blocks read is theirs to raise; a block
        // that aggregates, cuts its rows by a LIMIT or OFFSET or runs as a
        // statement raises its own
        if sq.grouping.is_none()
            && sq.order.is_none()
            && !matches!(sq.answer, Answer::Value(..) | Answer::Statement(..))
        {
            return Ok(());
        }
        match failed.iter().find(|(slot, _)| sq.reads.contains(slot)) {
            Some((_, e)) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Whether a candidate passes what the correlation equalities leave of
    /// the WHERE. With a slot failed, each conjunct is asked in turn as the
    /// conjunction asks it, and one that reads the slot raises its error
    fn passes_rest(
        &self,
        sq: &PreparedSubquery,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
        failed: &[(usize, Error)],
    ) -> Result<bool> {
        if failed.is_empty() {
            return match &sq.filter {
                Some(filter) => vm.execute_bool_checked(filter, exec),
                None => Ok(true),
            };
        }
        // The conjunction's own steps: only a FALSE skips what follows it,
        // and each pair of operands meets as the VM's AND meets them
        let mut acc: Option<Value> = None;
        for (program, slots) in &sq.rest {
            if matches!(acc, Some(Value::Boolean(false))) {
                break;
            }
            if let Some((_, e)) = failed.iter().find(|(slot, _)| slots.contains(slot)) {
                return Err(e.clone());
            }
            let value = vm.execute(program, exec)?;
            acc = Some(match acc {
                None => value,
                Some(left) => ExprVM::and(&left, &value),
            });
        }
        Ok(acc.as_ref().is_none_or(ExprVM::passes))
    }

    /// Whether an EXISTS block's candidate passes for the buffer filled:
    /// TRUE at the first that does; else the first error a candidate
    /// raised; else FALSE
    fn probe(&self, index: usize) -> Result<bool> {
        self.ctx.check_cancelled()?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_exists_probed();
        let sq = &self.prepared.subqueries[index];
        if let Some(order) = &sq.order {
            return Ok(!self.cut_rows(index, order, None, true)?.is_empty());
        }
        let mut vm = self.blocks[index].vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let slots_failed = self.blocks[index].failed.borrow();
        // A failed slot holds NULL, which the scan must not compare with
        let pushed = slots_failed.is_empty() && self.bind_pushdown(index, &params);
        let mut scanner = if pushed {
            let bound = self.blocks[index].bound.borrow();
            self.point_or_scan(index, bound.as_ref().map(BoundPushdown::expr))?
        } else {
            self.scan(index, None)?
        };
        let decided = pushed && sq.pushed_whole;
        let mut failed = None;
        let mut scanned = 0u64;
        while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            if !decided {
                let exec =
                    ExecuteContext::with_common_params(scanner.row(), &params, self.transaction_id)
                        .with_subquery_executor(self);
                if let Some(candidates) = &sq.candidates {
                    if !vm.execute_bool_checked(candidates, &exec)? {
                        continue;
                    }
                }
                match self.passes_rest(sq, &mut vm, &exec, &slots_failed) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(e) => {
                        failed.get_or_insert(deferred(e)?);
                        continue;
                    }
                }
            }
            return Ok(true);
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        failed.map_or(Ok(false), Err)
    }

    /// Whether this run answers the block by its keyed set: decided on its
    /// first call, by the rule the general path's index nested loop uses
    fn takes_keyed_set(&self, index: usize, plan: &KeyedPlan) -> bool {
        if let Some(keyed) = self.blocks[index].keyed_choice.get() {
            return keyed;
        }
        #[cfg(feature = "test-failpoints")]
        let forced = crate::test_failpoints::forced_exists_strategy();
        #[cfg(not(feature = "test-failpoints"))]
        let forced: Option<bool> = None;
        let keyed = forced.unwrap_or_else(|| !(plan.pure && self.probe_cheaper(index, plan)));
        self.blocks[index].keyed_choice.set(Some(keyed));
        keyed
    }

    /// Whether the keyed set answers the call the buffer holds: each key
    /// value of a checked plan NULL or of its own column's type
    fn keys_fit(&self, index: usize, plan: &KeyedPlan) -> bool {
        if !plan.checked {
            return true;
        }
        let columns = &self.prepared.subqueries[index].schema.columns;
        let params = self.blocks[index].buffer.borrow();
        plan.keys.iter().all(|(own, slot, _)| {
            let value = &params[*slot];
            value.is_null() || value.data_type() == columns[*own].data_type
        })
    }

    /// The general path's rule for its index nested loop, with the parent
    /// block's rows as the outer rows; only the statement's LIMIT counts
    fn probe_cheaper(&self, index: usize, plan: &KeyedPlan) -> bool {
        let sq = &self.prepared.subqueries[index];
        let parent_rows = match sq.parent {
            // A parent without FROM reads one row
            Some(parent) => Some(
                self.tables[parent as usize]
                    .as_ref()
                    .map_or(1, BlockInput::row_count),
            ),
            None => self.statement_rows.get(),
        };
        // A keyed plan reads a typed column, so its block has a table
        let Some(table) = self.tables[index].as_ref().and_then(BlockInput::table) else {
            return true;
        };
        Executor::probe_cheaper(
            table,
            &sq.schema.columns[plan.keys[0].0].name,
            self.statement_limit.filter(|_| sq.parent.is_none()),
            parent_rows,
        )
    }

    /// The keyed set's answer for the buffer filled: FALSE for a NULL key
    fn keyed_lookup(&self, index: usize, existence: &Kept, plan: &KeyedPlan) -> Result<bool> {
        if self.blocks[index].keyed_set.borrow().is_none() {
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::prepared_strategy_ran(
                crate::test_failpoints::PreparedStrategy::KeyedSet,
            );
            self.keyed_set(index, existence, plan)?;
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_exists_looked_up();
        let sets = self.blocks[index].keyed_set.borrow();
        let set = sets
            .as_ref()
            .ok_or_else(|| Error::internal("a keyed set not built"))?;
        let params = self.blocks[index].buffer.borrow();
        let outcome = match plan.keys.as_slice() {
            [(_, slot, _)] => {
                let key = &params[*slot];
                if key.is_null() {
                    return Ok(false);
                }
                set.outcomes.get(std::slice::from_ref(key))
            }
            keys => {
                let mut key: SmallVec<[Value; 4]> = SmallVec::with_capacity(keys.len());
                for (_, slot, _) in keys {
                    if params[*slot].is_null() {
                        return Ok(false);
                    }
                    key.push(params[*slot].clone());
                }
                set.outcomes.get(key.as_slice())
            }
        };
        match outcome {
            Some(ExistsOutcome::Passed) => Ok(true),
            Some(ExistsOutcome::Failed(e)) => Err((**e).clone()),
            None => Ok(false),
        }
    }

    /// The block's keyed set, built on first need or taken from a run
    /// before whose stamp still holds
    fn keyed_set(
        &self,
        index: usize,
        existence: &Kept,
        plan: &KeyedPlan,
    ) -> Result<CompactArc<ExistsKeys>> {
        if let Some(set) = self.blocks[index].keyed_set.borrow().as_ref() {
            return Ok(CompactArc::clone(set));
        }
        let answer = self.read_once(index, || {
            self.kept(existence, || {
                self.build_keyed_set(index, plan)
                    .map(|set| BlockAnswer::Keyed(CompactArc::new(set)))
            })
        })?;
        let BlockAnswer::Keyed(set) = answer else {
            return Err(Error::internal("a value kept for a keyed set"));
        };
        *self.blocks[index].keyed_set.borrow_mut() = Some(CompactArc::clone(&set));
        Ok(set)
    }

    /// Every candidate of the block grouped by its key: whether one passed
    /// what the keys leave of the WHERE, or else the first error one raised.
    /// A row with a NULL key is no key's candidate
    fn build_keyed_set(&self, index: usize, plan: &KeyedPlan) -> Result<ExistsKeys> {
        #[cfg(feature = "test-failpoints")]
        {
            crate::test_failpoints::prepared_exists_set_built();
            crate::test_failpoints::inner_table_read_whole();
        }
        let sq = &self.prepared.subqueries[index];
        let mut vm = self.blocks[index].vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let slots_failed = self.blocks[index].failed.borrow();
        let values: Option<Vec<Value>> = plan.pushdown.iter().map(|c| c.value(&params)).collect();
        let mut bound = None;
        if let Some(values) =
            values.filter(|_| !plan.pushdown.is_empty() && slots_failed.is_empty())
        {
            BoundPushdown::bind(&plan.pushdown, &mut bound, values);
        }
        let mut scanner = self.scan(index, bound.as_ref().map(BoundPushdown::expr))?;
        let decided = sq.filter.is_none() || (bound.is_some() && plan.pushed_whole);
        let mut set = ExistsKeys::default();
        let mut scanned = 0u64;
        'rows: while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            let row = scanner.row();
            let mut key: SmallVec<[Value; 1]> = SmallVec::with_capacity(plan.keys.len());
            for (column, _, _) in &plan.keys {
                match row.get(*column) {
                    Some(value) if !value.is_null() => key.push(value.clone()),
                    _ => continue 'rows,
                }
            }
            if matches!(
                set.outcomes.get(key.as_slice()),
                Some(ExistsOutcome::Passed)
            ) {
                continue;
            }
            let outcome = if decided {
                ExistsOutcome::Passed
            } else {
                let exec = ExecuteContext::with_common_params(row, &params, self.transaction_id)
                    .with_subquery_executor(self);
                match self.passes_rest(sq, &mut vm, &exec, &slots_failed) {
                    Ok(true) => ExistsOutcome::Passed,
                    Ok(false) => continue,
                    Err(e) => ExistsOutcome::Failed(Box::new(deferred(e)?)),
                }
            };
            match set.outcomes.entry(key) {
                std::collections::hash_map::Entry::Occupied(mut kept) => {
                    if matches!(outcome, ExistsOutcome::Passed) {
                        kept.insert(outcome);
                    }
                }
                std::collections::hash_map::Entry::Vacant(free) => {
                    free.insert(outcome);
                }
            }
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        set.failed = set
            .outcomes
            .values()
            .any(|outcome| matches!(outcome, ExistsOutcome::Failed(_)));
        Ok(set)
    }

    /// The block's answer for this run: kept from a run before when its
    /// stamp still holds, else computed and kept under the stamp taken
    /// before the read, where the run may keep one
    fn kept(
        &self,
        existence: &Kept,
        compute: impl FnOnce() -> Result<BlockAnswer>,
    ) -> Result<BlockAnswer> {
        let statement = self.statement;
        let keeps = !existence.reads_source
            && Executor::reads_cacheable_rows(statement)
            && !Executor::expression_varies_by_statement(&existence.identity, statement);
        let stamp = if keeps {
            self.executor
                .subquery_cache_stamp(&existence.tables, statement.statement_view())
        } else {
            None
        };
        let origin = BlockOrigin {
            block_key: existence.identity_key,
            scopes: &existence.scopes,
            params: statement.params(),
            read: existence.params.as_deref(),
        };
        let key = origin.key();
        if let Some(stamp) = &stamp {
            if let Some(answer) = get_cached_block_answer(&origin, key, stamp) {
                return Ok(answer);
            }
        }
        let answer = compute()?;
        if let Some(stamp) = stamp {
            cache_block_answer(&origin, key, stamp, answer.clone());
        }
        Ok(answer)
    }
}

impl SubqueryExecutor for PreparedRunner<'_> {
    fn execute_scalar(&self, subquery_index: u16, caller: &ExecuteContext) -> Result<Value> {
        self.scalar_at(subquery_index, caller, self.holes)
    }

    fn execute_exists(&self, subquery_index: u16, caller: &ExecuteContext) -> Result<bool> {
        self.exists_answer(subquery_index as usize, caller, self.holes)
    }

    fn execute_in(
        &self,
        subquery_index: u16,
        value: &Value,
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        let tested = std::slice::from_ref(value);
        self.members_answer(subquery_index as usize, tested, caller, self.holes)
    }

    fn execute_quantified(
        &self,
        subquery_index: u16,
        value: &Value,
        _op: CompareOp,
        _all: bool,
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        let tested = std::slice::from_ref(value);
        self.members_answer(subquery_index as usize, tested, caller, self.holes)
    }

    fn execute_row_members(
        &self,
        subquery_index: u16,
        row: &[Value],
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        self.row_members_answer(subquery_index as usize, row, caller, self.holes)
    }

    fn kept_members(&self, subquery_index: u16) -> Option<&KeptMembers> {
        match self.blocks.get(subquery_index as usize)?.members.get()? {
            BlockAnswer::Members(set) => Some(set),
            _ => None,
        }
    }

    fn params(&self) -> Option<&[Value]> {
        Some(&self.base)
    }

    fn bind_members(&self, program: &SharedProgram) -> Option<SharedProgram> {
        self.bound_program(program)
    }
}

/// A run's blocks as one site of a join statement calls them: the run's
/// state, and the holes compiled for the layout of the row the site reads
pub(crate) struct SiteRunner<'r> {
    runner: &'r PreparedRunner<'r>,
    holes: &'r HoleSet,
}

impl<'r> SiteRunner<'r> {
    pub(crate) fn new(runner: &'r PreparedRunner<'r>, holes: &'r HoleSet) -> Self {
        Self { runner, holes }
    }
}

impl SubqueryExecutor for SiteRunner<'_> {
    fn execute_scalar(&self, subquery_index: u16, caller: &ExecuteContext) -> Result<Value> {
        self.runner.scalar_at(subquery_index, caller, self.holes)
    }

    fn execute_exists(&self, subquery_index: u16, caller: &ExecuteContext) -> Result<bool> {
        self.runner
            .exists_answer(subquery_index as usize, caller, self.holes)
    }

    fn execute_in(
        &self,
        subquery_index: u16,
        value: &Value,
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        let tested = std::slice::from_ref(value);
        self.runner
            .members_answer(subquery_index as usize, tested, caller, self.holes)
    }

    fn execute_quantified(
        &self,
        subquery_index: u16,
        value: &Value,
        _op: CompareOp,
        _all: bool,
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        let tested = std::slice::from_ref(value);
        self.runner
            .members_answer(subquery_index as usize, tested, caller, self.holes)
    }

    fn execute_row_members(
        &self,
        subquery_index: u16,
        row: &[Value],
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        self.runner
            .row_members_answer(subquery_index as usize, row, caller, self.holes)
    }

    fn kept_members(&self, subquery_index: u16) -> Option<&KeptMembers> {
        self.runner.kept_members(subquery_index)
    }

    fn params(&self) -> Option<&[Value]> {
        Some(&self.runner.base)
    }
}

impl PreparedRunner<'_> {
    /// A scalar block's value for a call from `caller` at `site`
    fn scalar_at(
        &self,
        subquery_index: u16,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Value> {
        let index = subquery_index as usize;
        let sq = self
            .prepared
            .subqueries
            .get(index)
            .ok_or_else(|| Error::internal("unknown prepared subquery"))?;
        if sq.uncorrelated {
            if let Some(value) = self.blocks[index].value.borrow().as_ref() {
                return Ok(value.clone());
            }
        }
        let call = self.parent_call(sq);
        if let Some(BlockAnswer::Value(value)) = self.answered(index, call) {
            return Ok(value);
        }
        self.ctx.check_cancelled()?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_scalar_next();
        if let Answer::Statement(kept, block) = &sq.answer {
            let read = || {
                let mut rows = self.block_statement(index, block, caller, site)?;
                let mut value = None;
                while rows.next() {
                    if value.is_some() {
                        return Err(more_than_one_row());
                    }
                    value = Some(first_value(rows.take_row()));
                }
                if let Some(e) = rows.last_error() {
                    return Err(e);
                }
                Ok(value.unwrap_or_else(Value::null_unknown))
            };
            if !sq.uncorrelated {
                return read();
            }
            let answer =
                self.read_once(index, || self.kept(kept, || read().map(BlockAnswer::Value)))?;
            let BlockAnswer::Value(value) = answer else {
                return Err(Error::internal("a set kept for a scalar value"));
            };
            *self.blocks[index].value.borrow_mut() = Some(value.clone());
            return Ok(value);
        }
        let Answer::Value(projection, kept) = &sq.answer else {
            return Err(Error::internal("an EXISTS block run for a value"));
        };
        if !sq.uncorrelated {
            let value = self.scalar_value(index, sq, projection, caller, site)?;
            self.answer_call(index, call, || BlockAnswer::Value(value.clone()));
            return Ok(value);
        }
        let answer = self.read_once(index, || {
            self.kept(kept, || {
                self.scalar_value(index, sq, projection, caller, site)
                    .map(BlockAnswer::Value)
            })
        })?;
        let BlockAnswer::Value(value) = answer else {
            return Err(Error::internal("a set kept for a scalar value"));
        };
        *self.blocks[index].value.borrow_mut() = Some(value.clone());
        Ok(value)
    }
}

impl PreparedRunner<'_> {
    /// A scalar block's value for the row of `caller`: its one row's
    /// SELECT expression, NULL for no row
    fn scalar_value(
        &self,
        index: usize,
        sq: &PreparedSubquery,
        projection: &SharedProgram,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Value> {
        if let Some(grouping) = &sq.grouping {
            let Some(statement) = &grouping.statement else {
                return self.grouped_value(index, grouping, projection, caller, site);
            };
            let mut values = self.statement_values(statement, 1)?;
            if values.len() > 1 {
                return Err(more_than_one_row());
            }
            return Ok(values.pop().unwrap_or_else(Value::null_unknown));
        }
        if let Some(order) = &sq.order {
            return self.cut_value(index, order, projection, caller, site);
        }
        let mut vm = self.blocks[index].vm.borrow_mut();
        self.fill(index, &mut vm, caller, site)?;
        // Read-only from here on: a nested subquery copies these values
        let params = self.blocks[index].buffer.borrow();
        let mut value = None;
        // A WHERE that is one primary key equality reads that key's row
        if let Some((table, key)) = self.point_key(index, sq, &params) {
            let mut rows = self.blocks[index].point_rows.borrow_mut();
            rows.clear();
            table.fetch_rows_by_ids_into(&[key], &ConstBoolExpr::true_expr(), &mut rows)?;
            for (_, row) in rows.iter() {
                self.take_one(&mut vm, projection, row, &params, &mut value)?;
            }
            return Ok(value.unwrap_or_else(Value::null_unknown));
        }
        // The scanner keeps what it needs of the expression, so the next
        // call may rebind it once this scan is set up
        let pushed = self.bind_pushdown(index, &params);
        let mut scanner = if pushed {
            let bound = self.blocks[index].bound.borrow();
            self.point_or_scan(index, bound.as_ref().map(BoundPushdown::expr))?
        } else {
            self.scan(index, None)?
        };
        let filter = sq.filter.as_ref().filter(|_| !(pushed && sq.pushed_whole));
        let mut scanned = 0u64;
        while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            if let Some(filter) = filter {
                let exec =
                    ExecuteContext::with_common_params(scanner.row(), &params, self.transaction_id)
                        .with_subquery_executor(self);
                if !vm.execute_bool_checked(filter, &exec)? {
                    continue;
                }
            }
            self.take_one(&mut vm, projection, scanner.row(), &params, &mut value)?;
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        Ok(value.unwrap_or_else(Value::null_unknown))
    }

    /// The table and key of block `index` when its WHERE is one equality
    /// on the table's primary key, given this call's integer
    fn point_key(
        &self,
        index: usize,
        sq: &PreparedSubquery,
        params: &[Value],
    ) -> Option<(&dyn Table, i64)> {
        let point = sq
            .point
            .filter(|_| sq.pushed_whole && sq.pushdown.len() == 1)?;
        let table = self.tables[index].as_ref()?.table()?;
        match sq.pushdown[point].value(params)? {
            Value::Integer(key) => Some((table, key)),
            _ => None,
        }
    }

    /// `value` taking the SELECT expression over a candidate `row` that
    /// passed; a second candidate is the scalar's error
    fn take_one(
        &self,
        vm: &mut ExprVM,
        projection: &SharedProgram,
        row: &Row,
        params: &[Value],
        value: &mut Option<Value>,
    ) -> Result<()> {
        if value.is_some() {
            return Err(more_than_one_row());
        }
        let exec = ExecuteContext::with_common_params(row, params, self.transaction_id)
            .with_subquery_executor(self);
        *value = Some(vm.execute(projection, &exec)?);
        Ok(())
    }

    /// [`Self::scalar_value`] of a block with a LIMIT or OFFSET
    #[inline(never)]
    fn cut_value(
        &self,
        index: usize,
        order: &BlockOrder,
        projection: &SharedProgram,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Value> {
        self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
        let rows = self.cut_rows(index, order, Some(projection), false)?;
        let mut vm = self.blocks[index].vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let mut value = None;
        for (mut keys, row) in rows {
            match order.item {
                Some(_) if value.is_some() => return Err(more_than_one_row()),
                Some(item) => value = Some(keys.swap_remove(item)),
                None => self.take_one(&mut vm, projection, &row, &params, &mut value)?,
            }
        }
        Ok(value.unwrap_or_else(Value::null_unknown))
    }

    /// The rows block `index` answers from for the buffer filled, with
    /// their key values: those its WHERE passes in its ORDER BY, a tie in
    /// the order read, the first of each DISTINCT ON key, past its OFFSET
    /// and up to its LIMIT, or the first of them for an `exists`. A
    /// candidate's error is the block's
    fn cut_rows(
        &self,
        index: usize,
        order: &BlockOrder,
        item: Option<&SharedProgram>,
        exists: bool,
    ) -> Result<Vec<(SmallVec<[Value; 2]>, Row)>> {
        self.ctx.check_cancelled()?;
        let sq = &self.prepared.subqueries[index];
        let mut vm = self.blocks[index].vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let slots_failed = self.blocks[index].failed.borrow();
        let empty = Row::new();
        let none = ExecuteContext::with_common_params(&empty, &params, self.transaction_id);
        let mut count = |program: &Option<SharedProgram>, clause| {
            program
                .as_ref()
                .map(|program| super::row_count::count_of(vm.execute(program, &none)?, clause))
                .transpose()
        };
        let limit = count(&order.limit, "LIMIT")?;
        let offset = count(&order.offset, "OFFSET")?.unwrap_or(0);
        if !order.keys.is_empty() && (limit.is_some() || offset > 0) {
            super::result::OrderLimitPath::PreparedBlockCut.taken();
        }
        let mut keep = limit.map_or(usize::MAX, |limit| limit.saturating_add(offset));
        if keep == 0 {
            return Ok(Vec::new());
        }
        if exists {
            keep = keep.min(offset.saturating_add(1));
        }
        let pushed = slots_failed.is_empty() && self.bind_pushdown(index, &params);
        let mut scanner = if pushed {
            let bound = self.blocks[index].bound.borrow();
            self.point_or_scan(index, bound.as_ref().map(BoundPushdown::expr))?
        } else {
            self.scan(index, None)?
        };
        let decided = pushed && sq.pushed_whole;
        let (mut kept, mut all) = (Vec::new(), Vec::new());
        // The worst row held on top, which a better one replaces
        let mut held = std::collections::BinaryHeap::new();
        let (mut passed, mut scanned) = (0usize, 0u64);
        while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            let row = scanner.row();
            let exec = ExecuteContext::with_common_params(row, &params, self.transaction_id)
                .with_subquery_executor(self);
            if !decided {
                if let Some(candidates) = &sq.candidates {
                    if !vm.execute_bool_checked(candidates, &exec)? {
                        continue;
                    }
                }
                if !self.passes_rest(sq, &mut vm, &exec, &slots_failed)? {
                    continue;
                }
            }
            passed += 1;
            // Without keys the rows come in the order read: OFFSET's are passed
            if order.keys.is_empty() && order.distinct.is_empty() {
                if passed > offset {
                    kept.push((SmallVec::new(), row.clone()));
                }
                if passed == keep {
                    break;
                }
                continue;
            }
            let mut keys: SmallVec<[Value; 2]> = SmallVec::new();
            for (i, (key, ..)) in order.keys.iter().enumerate() {
                let repeated = order.repeats.iter().find(|(at, _)| *at == i);
                let value = match (key, order.item, repeated) {
                    (Some(key), ..) => vm.execute(key, &exec)?,
                    (None, _, Some(&(_, first))) => keys[first].clone(),
                    (None, Some(first), None) if first < i => keys[first].clone(),
                    (None, ..) => {
                        let item =
                            item.ok_or_else(|| Error::internal("a sort key without item"))?;
                        vm.execute(item, &exec)?
                    }
                };
                keys.push(value);
            }
            // A DISTINCT ON keeps the first row of a key in the whole order
            if !order.distinct.is_empty() {
                let mut seen = Row::with_capacity(order.distinct.len());
                for key in &order.distinct {
                    seen.push(vm.execute(key, &exec)?);
                }
                all.push((keys, seen, row.clone()));
                continue;
            }
            if held.len() < keep {
                let row = row.clone();
                held.push(HeldRow {
                    keys,
                    read: passed,
                    row,
                    order,
                });
            } else if let Some(mut worst) = held.peek_mut() {
                if order.compare(&keys, &worst.keys).is_lt() {
                    let row = row.clone();
                    *worst = HeldRow {
                        keys,
                        read: passed,
                        row,
                        order,
                    };
                }
            }
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        if !order.distinct.is_empty() {
            // A stable sort: a tie stays in the order read
            all.sort_by(|a, b| order.compare(&a.0, &b.0));
            let mut first = super::result::DistinctRows::new(order.distinct.len());
            let rows = all.into_iter().filter(|(_, seen, _)| first.is_new(seen));
            let rows = rows.skip(offset).take(keep - offset);
            return Ok(rows.map(|(keys, _, row)| (keys, row)).collect());
        }
        if order.keys.is_empty() {
            return Ok(kept);
        }
        let held = held.into_sorted_vec().into_iter().skip(offset);
        Ok(held.map(|held| (held.keys, held.row)).collect())
    }

    fn exists_answer(&self, index: usize, caller: &ExecuteContext, site: &HoleSet) -> Result<bool> {
        let sq = self
            .prepared
            .subqueries
            .get(index)
            .ok_or_else(|| Error::internal("unknown prepared subquery"))?;
        if let Answer::Statement(kept, block) = &sq.answer {
            let read = || {
                let mut rows = self.block_statement(index, block, caller, site)?;
                let exists = rows.next();
                match rows.last_error() {
                    Some(e) => Err(e),
                    None => Ok(exists),
                }
            };
            if !sq.uncorrelated {
                return read();
            }
            if let Some(Value::Boolean(exists)) = self.blocks[index].value.borrow().as_ref() {
                return Ok(*exists);
            }
            let answer =
                self.read_once(index, || self.kept(kept, || read().map(BlockAnswer::Once)))?;
            let BlockAnswer::Once(exists) = answer else {
                return Err(Error::internal("a set kept for an EXISTS value"));
            };
            *self.blocks[index].value.borrow_mut() = Some(Value::Boolean(exists));
            return Ok(exists);
        }
        let Answer::Exists(existence) = &sq.answer else {
            return Err(Error::internal("a scalar block run for EXISTS"));
        };
        if sq.uncorrelated {
            if let Some(Value::Boolean(exists)) = self.blocks[index].value.borrow().as_ref() {
                return Ok(*exists);
            }
        }
        let call = self.parent_call(sq);
        if let Some(BlockAnswer::Once(exists)) = self.answered(index, call) {
            return Ok(exists);
        }
        let exists = match (&sq.grouping, sq.uncorrelated) {
            (Some(grouping), false) => {
                self.with_groups(index, grouping, caller, site, |rows, params, vm| {
                    self.any_group_passes(grouping, rows, params, vm)
                })?
            }
            (grouping, true) => {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::prepared_strategy_ran(
                    crate::test_failpoints::PreparedStrategy::Once,
                );
                let once = || match grouping.as_deref() {
                    Some(Grouping {
                        statement: Some(statement),
                        ..
                    }) => self
                        .statement_values(statement, 1)
                        .map(|rows| !rows.is_empty()),
                    Some(grouping) => {
                        self.with_groups(index, grouping, caller, site, |rows, params, vm| {
                            self.any_group_passes(grouping, rows, params, vm)
                        })
                    }
                    None => {
                        self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
                        self.probe(index)
                    }
                };
                match self.read_once(index, || {
                    self.kept(existence, || once().map(BlockAnswer::Once))
                })? {
                    BlockAnswer::Once(exists) => exists,
                    _ => return Err(Error::internal("a set kept for an EXISTS value")),
                }
            }
            (None, false) => {
                self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
                match &existence.keyed {
                    Some(plan)
                        if self.takes_keyed_set(index, plan) && self.keys_fit(index, plan) =>
                    {
                        self.keyed_lookup(index, existence, plan)?
                    }
                    _ => self.probe(index)?,
                }
            }
        };
        if sq.uncorrelated {
            *self.blocks[index].value.borrow_mut() = Some(Value::Boolean(exists));
        }
        self.answer_call(index, call, || BlockAnswer::Once(exists));
        Ok(exists)
    }
}

/// A row's first value, NULL for a row without columns
fn first_value(row: Row) -> Value {
    let mut values = row.into_values();
    match values.is_empty() {
        true => Value::null_unknown(),
        false => values.swap_remove(0),
    }
}

/// `values` as `test` asks them, NULL members included; an uncorrelated
/// block's IN set `shared`, a key's kept in place. A row block's members
/// are `width` values each
fn member_set(test: MemberTest, width: usize, values: Vec<Value>, shared: bool) -> MemberSet {
    let (op, all) = match test {
        MemberTest::One(comparison) => {
            return MemberSet::Rows(RowMembers::new(RowTest::One(comparison), width, values));
        }
        MemberTest::In => (CompareOp::Eq, false),
        MemberTest::Quantified(op, all) => (op, all),
    };
    if width > 1 {
        let test = RowTest::Quantified(op, all);
        return MemberSet::Rows(RowMembers::new(test, width, values));
    }
    match test {
        MemberTest::In => {
            let set = PredicateSet::of(CompactArc::new(values.into_iter().collect()));
            match shared {
                true => MemberSet::SharedIn(CompactArc::new(set)),
                false => MemberSet::In(set),
            }
        }
        _ => {
            let has_null = values.iter().any(Value::is_null);
            let members: Vec<Value> = values.into_iter().filter(|v| !v.is_null()).collect();
            MemberSet::Quantified(QuantifiedMembers::new(
                op,
                all,
                CompactArc::new(members),
                has_null,
            ))
        }
    }
}

impl PreparedRunner<'_> {
    /// A row block's answer to `row` for the row it runs for. A row compared
    /// with the block's one row reads every row the block gives first; an
    /// uncorrelated block keeps them for the run
    fn row_members_answer(
        &self,
        index: usize,
        row: &[Value],
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Option<bool>> {
        let sq = &self.prepared.subqueries[index];
        let one = match &sq.answer {
            Answer::Members(_, members) => matches!(members.test, MemberTest::One(_)),
            Answer::Statement(_, block) => {
                matches!(block.form, StatementForm::Members(_, MemberTest::One(_)))
            }
            _ => false,
        };
        if !one || sq.uncorrelated {
            return self.members_answer(index, row, caller, site);
        }
        let set = match &sq.answer {
            Answer::Statement(_, block) => {
                let StatementForm::Members(read_as, test) = block.form else {
                    return Err(Error::internal("a block run for its members that has none"));
                };
                let mut rows = self.block_statement(index, block, caller, site)?;
                let mut values = Vec::new();
                while rows.next() {
                    Member::of_row(rows.take_row(), row.len(), read_as).push_into(&mut values);
                }
                if let Some(e) = rows.last_error() {
                    return Err(e);
                }
                KeptMembers {
                    set: member_set(test, row.len(), values, false),
                    failed: None,
                }
            }
            Answer::Members(_, members) => {
                self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
                self.collect_members(index, members, None)?
            }
            _ => return Err(Error::internal("a block run for its members that has none")),
        };
        set.answer_row(row)
    }

    /// A value block's answer to `tested`, a value or a row of the block's
    /// width, for the row it runs for
    fn members_answer(
        &self,
        index: usize,
        tested: &[Value],
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Option<bool>> {
        let sq = self
            .prepared
            .subqueries
            .get(index)
            .ok_or_else(|| Error::internal("unknown prepared subquery"))?;
        if let Answer::Statement(_, block) = &sq.answer {
            let StatementForm::Members(read_as, test) = block.form else {
                return Err(Error::internal("a block run for its members that has none"));
            };
            // One that binds nothing beyond itself reads its members once
            if sq.uncorrelated {
                self.read_member_set(index, caller, site)?;
                let Some(BlockAnswer::Members(set)) = self.blocks[index].members.get() else {
                    return Err(Error::internal("a value block's members not kept"));
                };
                return set.answer_values(tested);
            }
            let mut rows = self.block_statement(index, block, caller, site)?;
            let mut fold = MemberFold::new(test);
            while rows.next() {
                let member = Member::of_row(rows.take_row(), tested.len(), read_as);
                if let Some(answer) = fold.take(tested, Ok(member))? {
                    return Ok(Some(answer));
                }
            }
            if let Some(e) = rows.last_error() {
                return Err(e);
            }
            return fold.finish();
        }
        let Answer::Members(kept, members) = &sq.answer else {
            return Err(Error::internal("a block run for its members that has none"));
        };
        if sq.uncorrelated {
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::kept_member_called();
            // Read on the first row; every row after takes the kept members
            if self.blocks[index].members.get().is_none() {
                self.read_members(index, kept, members, caller, site)?;
            }
            let Some(BlockAnswer::Members(set)) = self.blocks[index].members.get() else {
                return Err(Error::internal("a value block's members not kept"));
            };
            return set.answer_values(tested);
        }
        let call = self.parent_call(sq);
        if let Some(BlockAnswer::Members(set)) = self.answered(index, call) {
            return set.answer_values(tested);
        }
        if call.is_some() {
            // A keyed set read for the run answers each call by lookup already
            self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
            let keyed = match &sq.grouping {
                Some(grouping) => grouping.keyed.as_ref(),
                None => kept.keyed.as_ref(),
            };
            if !keyed
                .is_some_and(|plan| self.takes_keyed_set(index, plan) && self.keys_fit(index, plan))
            {
                let set = CompactArc::new(self.collect_members(index, members, None)?);
                self.answer_call(index, call, || {
                    BlockAnswer::Members(CompactArc::clone(&set))
                });
                return set.answer_values(tested);
            }
        }
        if let Some(grouping) = &sq.grouping {
            return match tested {
                [value] => {
                    self.fold_groups(index, grouping, members, caller, site, |fold, vm, exec| {
                        fold.member(value, self.member(members, vm, exec))
                    })
                }
                row => {
                    self.fold_groups(index, grouping, members, caller, site, |fold, vm, exec| {
                        fold.row(row, self.member_row(members, vm, exec))
                    })
                }
            };
        }
        self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
        match &kept.keyed {
            Some(plan) if self.takes_keyed_set(index, plan) && self.keys_fit(index, plan) => {
                self.keyed_members_answer(index, kept, members, plan, tested)
            }
            _ => self.probe_members(index, members, tested),
        }
    }

    /// A grouped block's answer for the call: `take` folds each passing
    /// group's member
    fn fold_groups(
        &self,
        index: usize,
        grouping: &Grouping,
        members: &Members,
        caller: &ExecuteContext,
        site: &HoleSet,
        mut take: impl FnMut(&mut MemberFold, &mut ExprVM, &ExecuteContext) -> Result<Option<bool>>,
    ) -> Result<Option<bool>> {
        self.with_groups(index, grouping, caller, site, |rows, params, vm| {
            let mut fold = MemberFold::new(members.test);
            for slot_row in rows {
                let exec = self.slot_context(slot_row, params);
                match self.group_passes(grouping, vm, &exec) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(e) => {
                        fold.fail(e)?;
                        continue;
                    }
                }
                if let Some(answer) = take(&mut fold, vm, &exec)? {
                    return Ok(Some(answer));
                }
            }
            fold.finish()
        })
    }

    /// The members uncorrelated block `index` holds, read once per run on
    /// first need whether it is a value block or a statement
    fn read_member_set(&self, index: usize, caller: &ExecuteContext, site: &HoleSet) -> Result<()> {
        let sq = &self.prepared.subqueries[index];
        let (kept, block, read_as, test) = match &sq.answer {
            Answer::Members(kept, members) => {
                return self.read_members(index, kept, members, caller, site)
            }
            Answer::Statement(kept, block) if sq.uncorrelated => match block.form {
                StatementForm::Members(read_as, test) => (kept, block, read_as, test),
                _ => return Err(Error::internal("a block run for its members that has none")),
            },
            _ => return Err(Error::internal("a block run for its members that has none")),
        };
        if self.blocks[index].members.get().is_some() {
            return Ok(());
        }
        let width = sq.member_width();
        let answer = self.read_once(index, || {
            self.kept(kept, || {
                let mut rows = self.block_statement(index, block, caller, site)?;
                let mut values = Vec::new();
                while rows.next() {
                    Member::of_row(rows.take_row(), width, read_as).push_into(&mut values);
                }
                if let Some(e) = rows.last_error() {
                    return Err(e);
                }
                let set = member_set(test, width, values, true);
                let members = KeptMembers { set, failed: None };
                Ok(BlockAnswer::Members(CompactArc::new(members)))
            })
        })?;
        let _ = self.blocks[index].members.set(answer);
        Ok(())
    }

    /// An uncorrelated value block's members, read once per run on first
    /// need, or taken from a run before whose stamp still holds
    fn read_members(
        &self,
        index: usize,
        kept: &Kept,
        members: &Members,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<()> {
        if self.blocks[index].members.get().is_some() {
            return Ok(());
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_strategy_ran(
            crate::test_failpoints::PreparedStrategy::MemberSet,
        );
        let answer = self.read_once(index, || {
            self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
            self.kept(kept, || {
                self.collect_members(index, members, None)
                    .map(|set| BlockAnswer::Members(CompactArc::new(set)))
            })
        })?;
        let _ = self.blocks[index].members.set(answer);
        Ok(())
    }

    /// `program` with its member tests bound to the sets this run read, so
    /// it answers without the run as a test over a literal set does; None
    /// when it asks another block, a read fails or a candidate failed,
    /// whose error stays with the row that needs it
    pub(crate) fn bound_program(&self, program: &SharedProgram) -> Option<SharedProgram> {
        let unknown_value = program.ops().iter().any(|op| {
            matches!(op, Op::ExecScalarSubquery(_) | Op::ExecExists(_))
                && self.known_op(op).is_none()
        });
        if unknown_value {
            return None;
        }
        for op in program.ops() {
            if let Op::ExecInSubquery(index) | Op::ExecQuantified(index, ..) = op {
                self.read_set(*index as usize)?;
            }
        }
        let bound = self.bound_now(program, program.ops().iter().any(|op| self.asks_block(op)));
        (!bound.has_subqueries()).then_some(bound)
    }

    /// `program` bound as [`Self::bound_program`] binds it, once the values
    /// of the uncorrelated blocks it asks are read: a side read once asks
    /// them alike of each of its rows. None when a read fails, whose error
    /// stays with the row that needs it
    pub(crate) fn bound_reading(&self, program: &SharedProgram) -> Option<SharedProgram> {
        let empty = Row::new();
        let caller = ExecuteContext::with_common_params(&empty, &self.base, self.transaction_id);
        for op in program.ops().iter().filter(|op| self.asks_block(op)) {
            let read = match *op {
                Op::ExecScalarSubquery(index) => self.execute_scalar(index, &caller).map(drop),
                Op::ExecExists(index) => self.execute_exists(index, &caller).map(drop),
                _ => Ok(()),
            };
            read.ok()?;
        }
        self.bound_program(program)
    }

    /// `program` with the member tests of `slots` bound to the sets this run
    /// read, read now, and all else this run knows bound; a test whose set
    /// is not read stays one the run answers
    fn with_bound_sets(&self, program: &SharedProgram, slots: &[SetSlot]) -> Bound {
        for slot in slots {
            if program.ops().iter().any(|op| {
                matches!(op, Op::ExecInSubquery(index) | Op::ExecQuantified(index, ..) if *index == slot.id)
            }) && self.read_set(slot.id as usize).is_some()
            {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::prepared_strategy_ran(
                    crate::test_failpoints::PreparedStrategy::RowsTakeMembers,
                );
            }
        }
        self.bind(program)
    }

    /// The members uncorrelated block `index` holds for this run, read now,
    /// when a test over them binds; None when the read fails or a candidate
    /// failed, whose error stays with the row that needs it
    fn read_set(&self, index: usize) -> Option<()> {
        self.prepared.subqueries.get(index)?.member_set_test()?;
        let empty = Row::new();
        self.read_member_set(index, &ExecuteContext::new(&empty), self.holes)
            .ok()?;
        self.members_op(index).map(drop)
    }

    /// The test over the members uncorrelated block `index` has read in
    /// this run; None before the read, or when it failed for a candidate
    fn members_op(&self, index: usize) -> Option<Op> {
        if !self.prepared.subqueries.get(index)?.uncorrelated {
            return None;
        }
        let Some(BlockAnswer::Members(set)) = self.blocks[index].members.get() else {
            return None;
        };
        if set.failed.is_some() {
            return None;
        }
        Some(match &set.set {
            MemberSet::SharedIn(set) => Op::InSet(CompactArc::clone(set)),
            MemberSet::Quantified(set) => Op::QuantifiedSet(CompactArc::new(set.clone())),
            MemberSet::In(_) | MemberSet::Rows(_) => return None,
        })
    }

    /// What `op` reads in this run, when the run knows it already: a
    /// parameter or run slot, an uncorrelated block's value or members read
    fn known_op(&self, op: &Op) -> Option<Op> {
        match *op {
            Op::LoadParam(slot) => self.base.get(slot as usize).cloned().map(Op::LoadConst),
            Op::ExecScalarSubquery(index) | Op::ExecExists(index) => {
                if !self.prepared.subqueries.get(index as usize)?.uncorrelated {
                    return None;
                }
                let value = self.blocks[index as usize].value.borrow();
                value.clone().map(Op::LoadConst)
            }
            Op::ExecInSubquery(index) | Op::ExecQuantified(index, ..) => {
                self.members_op(index as usize)
            }
            _ => None,
        }
    }

    /// `program` bound to all this run knows now; one that asks no
    /// uncorrelated block binds only when that shrinks it
    fn bound_now(&self, program: &SharedProgram, asks_blocks: bool) -> SharedProgram {
        let bound = match asks_blocks {
            true => program.bind_with(|op| self.known_op(op)),
            false => program.bind(&self.base),
        };
        match bound {
            Some(bound) => CompactArc::new(bound),
            None => CompactArc::clone(program),
        }
    }

    /// Whether `op` asks an uncorrelated block, whose answer a run reads once
    fn asks_block(&self, op: &Op) -> bool {
        match *op {
            Op::ExecScalarSubquery(index)
            | Op::ExecExists(index)
            | Op::ExecInSubquery(index)
            | Op::ExecQuantified(index, ..) => self
                .prepared
                .subqueries
                .get(index as usize)
                .is_some_and(|sq| sq.uncorrelated),
            _ => false,
        }
    }

    /// Whether the run has read the answer block op `op` asks, as
    /// [`Self::known_op`] binds it
    fn knows(&self, op: &Op) -> bool {
        match *op {
            Op::ExecScalarSubquery(index) | Op::ExecExists(index) => {
                self.blocks[index as usize].value.borrow().is_some()
            }
            Op::ExecInSubquery(index) | Op::ExecQuantified(index, ..) => matches!(
                self.blocks[index as usize].members.get(),
                Some(BlockAnswer::Members(set))
                    if set.failed.is_none() && !matches!(set.set, MemberSet::In(_))
            ),
            _ => false,
        }
    }

    /// `program` bound to all this run knows now, to bind again once it
    /// knows more
    pub(crate) fn bind(&self, program: &SharedProgram) -> Bound {
        let pending: SmallVec<[Op; 2]> = program
            .ops()
            .iter()
            .filter(|op| self.asks_block(op))
            .cloned()
            .collect();
        let program_bound = self.bound_now(program, !pending.is_empty());
        Bound {
            source: CompactArc::clone(program),
            program: program_bound,
            pending: pending.into_iter().filter(|op| !self.knows(op)).collect(),
        }
    }

    /// `bound`'s program, bound again when a block it asks was read since
    #[inline]
    pub(crate) fn current<'b>(&self, bound: &'b mut Bound) -> &'b SharedProgram {
        self.rebound(bound);
        &bound.program
    }

    /// Whether `bound` was bound again, a block it asks having been read since
    #[inline]
    pub(crate) fn rebound(&self, bound: &mut Bound) -> bool {
        let read = bound.pending.iter().any(|op| self.knows(op));
        if read {
            *bound = self.bind(&bound.source);
        }
        read
    }

    /// `read` of block `index`'s set or uncorrelated value, or the error a
    /// read of it raised before in this run, which every row that needs it
    /// raises
    fn read_once<T>(&self, index: usize, read: impl FnOnce() -> Result<T>) -> Result<T> {
        if let Some(e) = self.blocks[index].read_error.borrow().as_ref() {
            return Err(e.clone());
        }
        read().inspect_err(|e| *self.blocks[index].read_error.borrow_mut() = Some(e.clone()))
    }

    /// The member a candidate's SELECT expression gives, read as the tested
    /// value's column reads it
    fn member(&self, members: &Members, vm: &mut ExprVM, exec: &ExecuteContext) -> Result<Value> {
        Ok(members.read(vm.execute(&members.projection, exec)?))
    }

    /// The values a candidate gives a row block's members
    fn member_row(
        &self,
        members: &Members,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
    ) -> Result<SmallVec<[Value; 2]>> {
        std::iter::once(&members.projection)
            .chain(&members.rest)
            .map(|program| vm.execute(program, exec))
            .collect()
    }

    /// The candidate's member, a value or a row's values
    fn candidate(
        &self,
        members: &Members,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
    ) -> Result<Member> {
        match members.rest.is_empty() {
            true => self.member(members, vm, exec).map(Member::Value),
            false => self.member_row(members, vm, exec).map(Member::Row),
        }
    }

    /// `fold` given the candidate's member, tested against `tested`
    #[inline(always)]
    fn fold_member(
        &self,
        fold: &mut MemberFold,
        tested: &[Value],
        members: &Members,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
    ) -> Result<Option<bool>> {
        match tested {
            [value] => fold.member(value, self.member(members, vm, exec)),
            row => fold.row(row, self.member_row(members, vm, exec)),
        }
    }

    /// Appends the candidate's member, a value or a row's values, to `values`
    #[inline]
    fn push_member(
        &self,
        members: &Members,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
        values: &mut Vec<Value>,
    ) -> Result<()> {
        match members.rest.is_empty() {
            true => values.push(self.member(members, vm, exec)?),
            false => values.extend(self.member_row(members, vm, exec)?),
        }
        Ok(())
    }

    /// The member of a row a block's LIMIT kept: the SELECT expression's
    /// value its sort computed, or the expression over the row
    fn cut_member(
        &self,
        members: &Members,
        order: &BlockOrder,
        (mut keys, row): (SmallVec<[Value; 2]>, Row),
        vm: &mut ExprVM,
        params: &[Value],
    ) -> Result<Member> {
        if let Some(item) = order.item {
            return Ok(Member::Value(members.read(keys.swap_remove(item))));
        }
        let exec = ExecuteContext::with_common_params(&row, params, self.transaction_id)
            .with_subquery_executor(self);
        if order.values.is_empty() {
            return self.candidate(members, vm, &exec);
        }
        let programs = std::iter::once(&members.projection).chain(&members.rest);
        let mut values = SmallVec::with_capacity(order.values.len());
        for (key, program) in order.values.iter().zip(programs) {
            values.push(match key {
                Some(key) => std::mem::replace(&mut keys[*key], Value::null_unknown()),
                None => vm.execute(program, &exec)?,
            });
        }
        Ok(Member::Row(values))
    }

    /// `value` tested by the probe against the buffer filled: a member that
    /// settles the test ends it; else the first error a candidate raised;
    /// else what the members say, the test of no members included
    fn probe_members(
        &self,
        index: usize,
        members: &Members,
        tested: &[Value],
    ) -> Result<Option<bool>> {
        self.ctx.check_cancelled()?;
        #[cfg(feature = "test-failpoints")]
        {
            crate::test_failpoints::prepared_exists_probed();
            crate::test_failpoints::in_subquery_ran();
        }
        let sq = &self.prepared.subqueries[index];
        if let Some(order) = &sq.order {
            let rows = self.cut_rows(index, order, Some(&members.projection), false)?;
            let mut vm = self.blocks[index].vm.borrow_mut();
            let params = self.blocks[index].buffer.borrow();
            let mut fold = MemberFold::new(members.test);
            for row in rows {
                let member = self.cut_member(members, order, row, &mut vm, &params);
                if let Some(answer) = fold.take(tested, member)? {
                    return Ok(Some(answer));
                }
            }
            return fold.finish();
        }
        let mut vm = self.blocks[index].vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let slots_failed = self.blocks[index].failed.borrow();
        let pushed = slots_failed.is_empty() && self.bind_pushdown(index, &params);
        let mut scanner = if pushed {
            let bound = self.blocks[index].bound.borrow();
            self.point_or_scan(index, bound.as_ref().map(BoundPushdown::expr))?
        } else {
            self.scan(index, None)?
        };
        let decided = pushed && sq.pushed_whole;
        let mut fold = MemberFold::new(members.test);
        let mut scanned = 0u64;
        while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            let exec =
                ExecuteContext::with_common_params(scanner.row(), &params, self.transaction_id)
                    .with_subquery_executor(self);
            if !decided {
                if let Some(candidates) = &sq.candidates {
                    if !vm.execute_bool_checked(candidates, &exec)? {
                        continue;
                    }
                }
                match self.passes_rest(sq, &mut vm, &exec, &slots_failed) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(e) => {
                        fold.fail(e)?;
                        continue;
                    }
                }
            }
            if let Some(answer) = self.fold_member(&mut fold, tested, members, &mut vm, &exec)? {
                return Ok(Some(answer));
            }
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        fold.finish()
    }

    /// Every member the block's candidates give, with the first error a
    /// candidate raised; grouped by key when `plan` is given, a row with a
    /// NULL key no key's candidate
    fn collect_members(
        &self,
        index: usize,
        members: &Members,
        plan: Option<&KeyedPlan>,
    ) -> Result<KeptMembers> {
        let width = members.width();
        if let Some(grouping) = &self.prepared.subqueries[index].grouping {
            if let Some(statement) = &grouping.statement {
                let values = self.statement_values(statement, width)?;
                let values = match members.read_as {
                    Some(column_type) => values
                        .into_iter()
                        .map(|value| read_as_column(column_type, value))
                        .collect(),
                    None => values,
                };
                return Ok(KeptMembers {
                    set: member_set(members.test, width, values, true),
                    failed: None,
                });
            }
            let rows = self.group_rows(index, grouping)?;
            let mut vm = self.blocks[index].vm.borrow_mut();
            let params = self.blocks[index].buffer.borrow();
            let (mut values, mut failed) = (Vec::with_capacity(rows.len() * width), None);
            for slot_row in &rows {
                let exec = self.slot_context(slot_row, &params);
                let pushed = match self.group_passes(grouping, &mut vm, &exec) {
                    Ok(true) => self.push_member(members, &mut vm, &exec, &mut values),
                    Ok(false) => continue,
                    Err(e) => Err(e),
                };
                if let Err(e) = pushed {
                    failed.get_or_insert(deferred(e)?);
                }
            }
            return Ok(KeptMembers {
                set: member_set(members.test, width, values, true),
                failed,
            });
        }
        let mut grouped = self.group_members(index, members, plan)?;
        Ok(grouped.remove(&[][..]).unwrap_or_else(|| KeptMembers {
            set: member_set(members.test, width, Vec::new(), true),
            failed: None,
        }))
    }

    /// The members per key the block's candidates give, the key empty
    /// without `plan`
    fn group_members(
        &self,
        index: usize,
        members: &Members,
        plan: Option<&KeyedPlan>,
    ) -> Result<MemberKeys> {
        #[cfg(feature = "test-failpoints")]
        {
            crate::test_failpoints::prepared_exists_set_built();
            crate::test_failpoints::inner_table_read_whole();
            crate::test_failpoints::in_subquery_ran();
        }
        let sq = &self.prepared.subqueries[index];
        if let Some(order) = &sq.order {
            let rows = self.cut_rows(index, order, Some(&members.projection), false)?;
            let mut vm = self.blocks[index].vm.borrow_mut();
            let params = self.blocks[index].buffer.borrow();
            let (mut values, mut failed) = (Vec::with_capacity(rows.len() * members.width()), None);
            for row in rows {
                match self.cut_member(members, order, row, &mut vm, &params) {
                    Ok(member) => member.push_into(&mut values),
                    Err(e) => {
                        failed.get_or_insert(deferred(e)?);
                    }
                }
            }
            let set = member_set(members.test, members.width(), values, true);
            return Ok(std::iter::once((SmallVec::new(), KeptMembers { set, failed })).collect());
        }
        let mut vm = self.blocks[index].vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let slots_failed = self.blocks[index].failed.borrow();
        let (pushdown, pushed_whole) = match plan {
            Some(plan) => (&plan.pushdown, plan.pushed_whole),
            None => (&sq.pushdown, sq.pushed_whole),
        };
        let values: Option<Vec<Value>> = pushdown.iter().map(|c| c.value(&params)).collect();
        let mut bound = None;
        if let Some(values) = values.filter(|_| !pushdown.is_empty() && slots_failed.is_empty()) {
            BoundPushdown::bind(pushdown, &mut bound, values);
        }
        let mut scanner = self.scan(index, bound.as_ref().map(BoundPushdown::expr))?;
        // A plan's keys take the place of the correlation equalities
        let candidates = sq.candidates.as_ref().filter(|_| plan.is_none());
        let decided =
            (sq.filter.is_none() && candidates.is_none()) || (bound.is_some() && pushed_whole);
        let mut groups: FxHashMap<SmallVec<[Value; 1]>, MemberGroup> = FxHashMap::default();
        let mut scanned = 0u64;
        'rows: while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            let row = scanner.row();
            let mut key: SmallVec<[Value; 1]> = SmallVec::new();
            for (column, _, _) in plan.map_or(&[][..], |plan| &plan.keys) {
                match row.get(*column) {
                    Some(value) if !value.is_null() => key.push(value.clone()),
                    _ => continue 'rows,
                }
            }
            let exec = ExecuteContext::with_common_params(row, &params, self.transaction_id)
                .with_subquery_executor(self);
            if let Some(candidates) = candidates.filter(|_| !decided) {
                if !vm.execute_bool_checked(candidates, &exec)? {
                    continue;
                }
            }
            let passed = if decided {
                Ok(true)
            } else {
                self.passes_rest(sq, &mut vm, &exec, &slots_failed)
            };
            if let Ok(false) = passed {
                continue;
            }
            let group = groups.entry(key).or_default();
            let pushed =
                passed.and_then(|_| self.push_member(members, &mut vm, &exec, &mut group.0));
            if let Err(e) = pushed {
                group.1.get_or_insert(deferred(e)?);
            }
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        Ok(groups
            .into_iter()
            .map(|(key, (values, failed))| {
                (
                    key,
                    KeptMembers {
                        set: member_set(members.test, members.width(), values, plan.is_none()),
                        failed,
                    },
                )
            })
            .collect())
    }

    /// The statement's rows an uncorrelated `column IN (b)` or `column NOT
    /// IN (b)` in its top-level AND may pass, read by the block's members,
    /// and whether each passes that conjunct. The members are read
    /// before the scan, as a block compared with a column there is; a
    /// member's error leaves the rows to the scan, which raises it
    fn rows_by_members(
        &self,
        table: &dyn Table,
        index: usize,
        column: usize,
        whole: bool,
        negated: bool,
        target: Option<usize>,
    ) -> Result<Option<(RowVec, bool)>> {
        if self.prepared.subqueries[index].member_set_test().is_none() {
            return Ok(None);
        }
        // An error reading the members belongs to the row that needs them
        let empty = Row::new();
        let read = self.read_member_set(index, &ExecuteContext::new(&empty), self.holes);
        if read.is_err() {
            return Ok(None);
        }
        let Some(BlockAnswer::Members(set)) = self.blocks[index].members.get() else {
            return Ok(None);
        };
        let MemberSet::SharedIn(set_of) = &set.set else {
            return Ok(None);
        };
        if set.failed.is_some() {
            return Ok(None);
        }
        let schema_column = &table.schema().columns[column];
        let Some((values, unknown)) =
            Executor::members_for_column(Some(schema_column.data_type), set_of)
        else {
            return Ok(None);
        };
        let name = &schema_column.name_lower;
        let values = values.iter().filter(|v| !v.is_null());
        if negated {
            // A member NULL or UNKNOWN to every value leaves NOT IN TRUE for no row
            if set_of.has_null() || unknown {
                return Ok(whole.then(|| (RowVec::new(), true)));
            }
            let rows =
                Executor::rows_without_members(table, name, values, target.filter(|_| whole))?;
            return Ok(rows.map(|rows| (rows, true)));
        }
        let rows = Executor::rows_of_members(
            table,
            name,
            values,
            target.is_some(),
            target.filter(|_| whole),
        )?;
        Ok(rows.map(|rows| (rows, true)))
    }

    /// A keyed IN block's answer to `value` for the buffer filled: no
    /// candidates for a NULL key or a key without rows
    fn keyed_members_answer(
        &self,
        index: usize,
        kept: &Kept,
        members: &Members,
        plan: &KeyedPlan,
        tested: &[Value],
    ) -> Result<Option<bool>> {
        if self.blocks[index].members.get().is_none() {
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::prepared_strategy_ran(
                crate::test_failpoints::PreparedStrategy::KeyedMembers,
            );
            let answer = self.kept(kept, || {
                self.group_members(index, members, Some(plan))
                    .map(|groups| BlockAnswer::KeyedMembers(CompactArc::new(groups)))
            })?;
            let _ = self.blocks[index].members.set(answer);
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_exists_looked_up();
        let Some(BlockAnswer::KeyedMembers(groups)) = self.blocks[index].members.get() else {
            return Err(Error::internal("a keyed IN block's members not kept"));
        };
        let params = self.blocks[index].buffer.borrow();
        let no_candidates = Ok(Some(!members.test.decides()));
        if plan.keys.iter().any(|(_, slot, _)| params[*slot].is_null()) {
            return no_candidates;
        }
        let group = match plan.keys.as_slice() {
            [(_, slot, _)] => groups.get(std::slice::from_ref(&params[*slot])),
            keys => {
                let key: SmallVec<[Value; 4]> = keys
                    .iter()
                    .map(|(_, slot, _)| params[*slot].clone())
                    .collect();
                groups.get(key.as_slice())
            }
        };
        match group {
            Some(group) => group.answer_values(tested),
            None => no_candidates,
        }
    }
}

/// What a candidate gives a block's members: its SELECT value, or a row's
/// values
enum Member {
    Value(Value),
    Row(SmallVec<[Value; 2]>),
}

impl Member {
    /// A statement block's row as a member of `width` values
    fn of_row(row: Row, width: usize, read_as: Option<DataType>) -> Self {
        match width {
            1 => {
                let member = first_value(row);
                Member::Value(match read_as {
                    Some(column_type) => read_as_column(column_type, member),
                    None => member,
                })
            }
            _ => {
                let mut values = row.into_values();
                values.resize(width, Value::null_unknown());
                Member::Row(values.into_iter().collect())
            }
        }
    }

    /// Appends the member's values to `values`
    fn push_into(self, values: &mut Vec<Value>) {
        match self {
            Member::Value(value) => values.push(value),
            Member::Row(row) => values.extend(row),
        }
    }
}

/// A value tested against a block's members one at a time: a member that
/// settles the test ends it; else the first error one raised; else what
/// the members say, the test of no members included
struct MemberFold {
    op: CompareOp,
    decides: bool,
    any: bool,
    unknown: bool,
    failed: Option<Error>,
}

impl MemberFold {
    fn new(test: MemberTest) -> Self {
        debug_assert!(
            !matches!(test, MemberTest::One(_)),
            "a row compared with a block's one row reads all its rows first"
        );
        let (op, decides) = match test {
            MemberTest::In | MemberTest::One(_) => (CompareOp::Eq, true),
            MemberTest::Quantified(op, all) => (op, !all),
        };
        Self {
            op,
            decides,
            any: false,
            unknown: false,
            failed: None,
        }
    }

    /// The answer, once `member` settles the test
    fn member(&mut self, value: &Value, member: Result<Value>) -> Result<Option<bool>> {
        match member {
            Ok(member) => {
                self.any = true;
                match QuantifiedMembers::compares(self.op, value, &member) {
                    Some(answer) if answer == self.decides => return Ok(Some(answer)),
                    Some(_) => {}
                    None => self.unknown = true,
                }
            }
            Err(e) => self.fail(e)?,
        }
        Ok(None)
    }

    /// The answer, once a row block's `member` settles the test of `row`
    fn row(&mut self, row: &[Value], member: Result<SmallVec<[Value; 2]>>) -> Result<Option<bool>> {
        match member {
            Ok(member) => Ok(self.settle(self.op.compare_rows(row, &member))),
            Err(e) => self.fail(e).map(|_| None),
        }
    }

    /// [`Self::member`] or [`Self::row`], as `tested` and `member` are
    fn take(&mut self, tested: &[Value], member: Result<Member>) -> Result<Option<bool>> {
        match (tested, member) {
            ([value], Ok(Member::Value(member))) => self.member(value, Ok(member)),
            (row, Ok(Member::Row(member))) => self.row(row, Ok(member)),
            (_, Err(e)) => self.fail(e).map(|_| None),
            _ => Err(Error::internal("a block's members of another width")),
        }
    }

    /// The answer one member's comparison settles
    #[inline]
    fn settle(&mut self, answer: Option<bool>) -> Option<bool> {
        self.any = true;
        match answer {
            Some(answer) if answer == self.decides => Some(answer),
            Some(_) => None,
            None => {
                self.unknown = true;
                None
            }
        }
    }

    /// Keeps the first error a candidate raised; a cancel ends the test
    fn fail(&mut self, e: Error) -> Result<()> {
        let e = deferred(e)?;
        self.failed.get_or_insert(e);
        Ok(())
    }

    fn finish(self) -> Result<Option<bool>> {
        match self.failed {
            Some(e) => Err(e),
            None if !self.any => Ok(Some(!self.decides)),
            None => Ok((!self.unknown).then_some(!self.decides)),
        }
    }
}

impl PreparedRunner<'_> {
    /// The first value of every row an uncorrelated block that aggregates
    /// gives, run as a statement of its own under the run's view
    /// The rows of statement block `index` for the row it runs for, its
    /// slots filled and read as the statement's parameters
    fn block_statement(
        &self,
        index: usize,
        block: &StatementBlock,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Box<dyn crate::storage::traits::QueryResult>> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_strategy_ran(
            crate::test_failpoints::PreparedStrategy::BlockStatement,
        );
        self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
        let ctx = self
            .ctx
            .with_slot_params(self.blocks[index].buffer.borrow().clone());
        self.executor.execute_select_with_plan(
            &block.run.statement,
            &ctx,
            Some(&block.run.classification),
            Some(block.run.compiled.slot()),
            None,
        )
    }

    fn statement_values(&self, run: &StatementRun, width: usize) -> Result<Vec<Value>> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_strategy_ran(
            crate::test_failpoints::PreparedStrategy::Statement,
        );
        let mut result = self.executor.execute_select_with_plan(
            &run.statement,
            &self.ctx,
            Some(&run.classification),
            Some(run.compiled.slot()),
            None,
        )?;
        let mut values = Vec::new();
        while result.next() {
            Member::of_row(result.take_row(), width, None).push_into(&mut values);
        }
        match result.last_error() {
            Some(e) => Err(e),
            None => Ok(values),
        }
    }

    /// The context a grouped block's HAVING and SELECT expression read a
    /// slot row in, with the buffer filled for the call
    fn slot_context<'r>(&'r self, slot_row: &'r Row, params: &'r [Value]) -> ExecuteContext<'r> {
        ExecuteContext::with_common_params(slot_row, params, self.transaction_id)
    }

    /// Whether a slot row passes the block's HAVING
    fn group_passes(
        &self,
        grouping: &Grouping,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
    ) -> Result<bool> {
        match &grouping.having {
            Some(having) => vm.execute_bool_checked(having, exec),
            None => Ok(true),
        }
    }

    /// A grouped scalar block's value: the SELECT expression over the one
    /// slot row HAVING passes, NULL for none
    fn grouped_value(
        &self,
        index: usize,
        grouping: &Grouping,
        projection: &SharedProgram,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Value> {
        self.with_groups(index, grouping, caller, site, |rows, params, vm| {
            let mut value = None;
            for slot_row in rows {
                let exec = self.slot_context(slot_row, params);
                if !self.group_passes(grouping, vm, &exec)? {
                    continue;
                }
                if value.is_some() {
                    return Err(more_than_one_row());
                }
                value = Some(vm.execute(projection, &exec)?);
            }
            Ok(value.unwrap_or_else(Value::null_unknown))
        })
    }

    /// Whether a slot row passes HAVING: TRUE at the first that does; else
    /// the first error HAVING raised; else FALSE
    fn any_group_passes(
        &self,
        grouping: &Grouping,
        rows: &[Row],
        params: &[Value],
        vm: &mut ExprVM,
    ) -> Result<bool> {
        let mut failed = None;
        for slot_row in rows {
            match self.group_passes(grouping, vm, &self.slot_context(slot_row, params)) {
                Ok(true) => return Ok(true),
                Ok(false) => {}
                Err(e) => {
                    failed.get_or_insert(deferred(e)?);
                }
            }
        }
        failed.map_or(Ok(false), Err)
    }

    /// `answer` over the block's slot rows for `row`, with the buffer they
    /// read and the block's VM: looked up in the run's keyed groups when it
    /// takes them, else aggregated for this call
    fn with_groups<T>(
        &self,
        index: usize,
        grouping: &Grouping,
        caller: &ExecuteContext,
        site: &HoleSet,
        answer: impl FnOnce(&[Row], &[Value], &mut ExprVM) -> Result<T>,
    ) -> Result<T> {
        self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
        if let Some(plan) = grouping.keyed.as_ref() {
            if self.takes_keyed_set(index, plan) && self.keys_fit(index, plan) {
                if self.blocks[index].grouped.borrow().is_none() {
                    let keys = self.grouped_keys(index, grouping, plan)?;
                    *self.blocks[index].grouped.borrow_mut() = Some(keys);
                }
                let keys = self.blocks[index].grouped.borrow();
                let keys = keys
                    .as_ref()
                    .ok_or_else(|| Error::internal("a keyed block's groups not read"))?;
                let params = self.blocks[index].buffer.borrow();
                let group = if plan.keys.iter().any(|(_, slot, _)| params[*slot].is_null()) {
                    None
                } else {
                    match plan.keys.as_slice() {
                        [(_, slot, _)] => keys.groups.get(std::slice::from_ref(&params[*slot])),
                        many => {
                            let key: SmallVec<[Value; 4]> = many
                                .iter()
                                .map(|(_, slot, _)| params[*slot].clone())
                                .collect();
                            keys.groups.get(key.as_slice())
                        }
                    }
                };
                let rows = match group {
                    Some(Ok(rows)) => rows.as_slice(),
                    Some(Err(e)) => return Err(e.clone()),
                    None => keys.missing.as_slice(),
                };
                return answer(rows, &params, &mut self.blocks[index].vm.borrow_mut());
            }
        }
        let rows = self.group_rows(index, grouping)?;
        let params = self.blocks[index].buffer.borrow();
        answer(&rows, &params, &mut self.blocks[index].vm.borrow_mut())
    }

    /// The block's slot rows for the buffer filled: every candidate its
    /// WHERE passes, fed to its group
    fn group_rows(&self, index: usize, grouping: &Grouping) -> Result<Vec<Row>> {
        self.ctx.check_cancelled()?;
        if let Some(rows) = self.index_count(index, grouping)? {
            return Ok(rows);
        }
        AggregatePath::PreparedBlock.taken();
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_strategy_ran(
            crate::test_failpoints::PreparedStrategy::Grouped,
        );
        let sq = &self.prepared.subqueries[index];
        let mut vm = self.blocks[index].vm.borrow_mut();
        let mut feed_vm = self.blocks[index].feed_vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let pushed = self.bind_pushdown(index, &params);
        let mut scanner = if pushed {
            let bound = self.blocks[index].bound.borrow();
            self.point_or_scan(index, bound.as_ref().map(BoundPushdown::expr))?
        } else {
            self.scan(index, None)?
        };
        let decided = pushed && sq.pushed_whole;
        let mut groups = grouping.groups(0);
        let mut key = KeyReader::new(grouping, Vec::new());
        let mut scanned = 0u64;
        'rows: while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            let row = scanner.row();
            let exec = ExecuteContext::with_common_params(row, &params, self.transaction_id)
                .with_subquery_executor(self);
            if !decided {
                for program in sq.candidates.iter().chain(&sq.filter) {
                    if !vm.execute_bool_checked(program, &exec)? {
                        continue 'rows;
                    }
                }
            }
            if let Some(group) = key.group(grouping, &mut groups, row, &mut vm, &exec)? {
                grouping.feed(&mut groups, group, row, &exec, &mut feed_vm)?;
            }
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        Ok(groups
            .into_rows()?
            .into_iter()
            .map(|(_, row)| row)
            .collect())
    }

    /// The slot row of a block whose calls count rows and whose WHERE is one
    /// equality an index holds, counted from the index's ids as the
    /// statement sees them; None when the index cannot answer for this call
    fn index_count(&self, index: usize, grouping: &Grouping) -> Result<Option<Vec<Row>>> {
        if !grouping.counts_rows {
            return Ok(None);
        }
        let sq = &self.prepared.subqueries[index];
        let ([equality], true) = (sq.pushdown.as_slice(), sq.pushed_whole) else {
            return Ok(None);
        };
        let table = self.tables[index].as_ref().and_then(BlockInput::table);
        let Some(table) = table.filter(|_| equality.operator == Operator::Eq) else {
            return Ok(None);
        };
        let params = self.blocks[index].buffer.borrow();
        // An equality with NULL holds for no row, a value read as NULL by
        // the column's type included
        let count = match equality.value(&params) {
            _ if equality.compares_null(&params) => 0,
            Some(key) if key.is_null() => 0,
            Some(key) => match self.key_count(index, table, equality, &key)? {
                Some(count) => count,
                None => return Ok(None),
            },
            None => return Ok(None),
        };
        drop(params);
        AggregatePath::IndexCount.taken();
        let count = CompiledAggregate::CountStar {
            count: i64::try_from(count).map_err(|_| Error::internal("a count past i64"))?,
        };
        let mut groups = grouping.groups(0);
        for run in groups.aggregates(0) {
            run.merge(&count);
        }
        Ok(Some(
            groups
                .into_rows()?
                .into_iter()
                .map(|(_, row)| row)
                .collect(),
        ))
    }

    /// How many rows the statement sees holding `key` in the column of
    /// `equality`, read from the ids its index holds for the key; None when
    /// the index cannot answer, the key crosses the column's type or the
    /// index moved while the ids were taken
    fn key_count(
        &self,
        index: usize,
        table: &dyn Table,
        equality: &PreparedComparison,
        key: &Value,
    ) -> Result<Option<usize>> {
        let Some(epoch) = table.index_view_epoch() else {
            return Ok(None);
        };
        let column = &table.schema().columns[equality.index];
        if crosses_inner_key(KeySide::of_kinds(kind::of_type(column.data_type)), key) {
            return Ok(None);
        }
        let mut held = self.blocks[index].count_ids.borrow_mut();
        let (ids, scratch) = &mut **held.get_or_insert_with(Box::default);
        ids.clear();
        // The cap bounds sealed reads; hot ids cost only their visibility
        let cap = match table.has_sealed_rows() {
            true => EQUALITY_CANDIDATE_CAP,
            false => usize::MAX,
        };
        let found = table.equality_candidates(&equality.column, key, cap, ids, scratch)?;
        if !matches!(found, Some(CappedEqual::Copied)) {
            return Ok(None);
        }
        let count =
            table.count_equal_candidates(ids, equality.index, key, usize::MAX, Some(epoch))?;
        // A key that held many rows does not keep their room for every call
        if ids.capacity() > COUNT_IDS_KEPT {
            *ids = Vec::new();
        }
        Ok((table.index_view_epoch() == Some(epoch)).then_some(count))
    }

    /// Every candidate of a keyed grouped block fed to its group under its
    /// key, in one pass. A row with a NULL key is no key's candidate; a key
    /// whose candidate raised an error answers with it
    fn grouped_keys(
        &self,
        index: usize,
        grouping: &Grouping,
        plan: &KeyedPlan,
    ) -> Result<GroupedKeys> {
        AggregatePath::PreparedBlock.taken();
        #[cfg(feature = "test-failpoints")]
        {
            crate::test_failpoints::prepared_strategy_ran(
                crate::test_failpoints::PreparedStrategy::KeyedGroups,
            );
            crate::test_failpoints::inner_table_read_whole();
        }
        let sq = &self.prepared.subqueries[index];
        let mut vm = self.blocks[index].vm.borrow_mut();
        let mut feed_vm = self.blocks[index].feed_vm.borrow_mut();
        let params = self.blocks[index].buffer.borrow();
        let values: Option<Vec<Value>> = plan.pushdown.iter().map(|c| c.value(&params)).collect();
        let mut bound = None;
        if let Some(values) = values.filter(|_| !plan.pushdown.is_empty()) {
            BoundPushdown::bind(&plan.pushdown, &mut bound, values);
        }
        let mut scanner = self.scan(index, bound.as_ref().map(BoundPushdown::expr))?;
        let decided = sq.filter.is_none() || (bound.is_some() && plan.pushed_whole);
        let prefix: Vec<usize> = plan.keys.iter().map(|(own, _, _)| *own).collect();
        let mut groups = grouping.groups(prefix.len());
        let mut key = KeyReader::new(grouping, prefix.clone());
        let mut failed: FxHashMap<SmallVec<[Value; 1]>, Error> = FxHashMap::default();
        let mut scanned = 0u64;
        'rows: while scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                self.ctx.check_cancelled()?;
            }
            let row = scanner.row();
            for column in &prefix {
                if row.get(*column).is_none_or(Value::is_null) {
                    continue 'rows;
                }
            }
            let exec = ExecuteContext::with_common_params(row, &params, self.transaction_id)
                .with_subquery_executor(self);
            let fed = (|| {
                if let (false, Some(filter)) = (decided, &sq.filter) {
                    if !vm.execute_bool_checked(filter, &exec)? {
                        return Ok(());
                    }
                }
                match key.group(grouping, &mut groups, row, &mut vm, &exec)? {
                    Some(group) => grouping.feed(&mut groups, group, row, &exec, &mut feed_vm),
                    None => Ok(()),
                }
            })();
            if let Err(e) = fed {
                let at: SmallVec<[Value; 1]> = prefix
                    .iter()
                    .map(|column| {
                        row.get(*column)
                            .cloned()
                            .unwrap_or_else(Value::null_unknown)
                    })
                    .collect();
                failed.entry(at).or_insert(deferred(e)?);
            }
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        let mut by_key: FxHashMap<SmallVec<[Value; 1]>, Result<Vec<Row>>> = FxHashMap::default();
        // A group's error belongs to its key alone
        for group in groups.into_group_results()? {
            match group {
                Ok(row) => {
                    let mut values = row.into_values();
                    let rest = values.split_off(prefix.len());
                    let rows = by_key
                        .entry(values.into_iter().collect())
                        .or_insert_with(|| Ok(Vec::new()));
                    if let Ok(rows) = rows {
                        rows.push(Row::from_values(rest));
                    }
                }
                Err((values, e)) => {
                    let at = values.into_iter().take(prefix.len()).collect();
                    failed.entry(at).or_insert(deferred(e)?);
                }
            }
        }
        for (at, e) in failed {
            by_key.insert(at, Err(e));
        }
        let missing = if grouping.items.is_empty() {
            let none = grouping.groups(0).into_rows()?;
            none.into_iter().map(|(_, row)| row).collect()
        } else {
            Vec::new()
        };
        Ok(GroupedKeys {
            groups: by_key,
            missing,
        })
    }
}
