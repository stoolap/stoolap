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

use std::cell::{Cell, RefCell};
use std::sync::{Arc, RwLock};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::api::params::ParamVec;
use crate::common::{CompactArc, SmartString};
use crate::core::predicate::PredicateSet;
use crate::core::{DataType, Error, Operator, Result, Row, RowVec, Schema, SchemaColumn, Value};
use crate::functions::aggregate::{CallTemplate, ColumnsKey, GroupedAggregates};
use crate::functions::{FunctionRegistry, Volatility};
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::expression::{
    AndExpr, ComparisonExpr, ConstBoolExpr, Expression as StorageExpr,
};
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::traits::transaction::TableUse;
use crate::storage::traits::{Engine, Scanner, Table, Transaction};

use super::aggregation::{
    aggregate_slot_keys, ungrouped_column, AggregatePath, CallInputs, GroupByItem,
    SqlAggregateFunction, COUNT_STAR,
};
use super::context::{
    cache_block_answer, get_cached_block_answer, BlockAnswer, BlockOrigin, ExecutionContext,
    ExistsKeys, ExistsOutcome, KeptMembers, MemberKeys, MemberSet,
};
use super::expr_key::ExprKey;
use super::expression::{
    CompareOp, CompileContext, CompileError, ExecuteContext, ExprCompiler, ExprVM, Op,
    QuantifiedMembers, RowFilter, SharedProgram, SubqueryExecutor,
};
use super::pushdown::{self, coerce_constant, read_as_column, scan_compares};
use super::query::{ColumnSource, OutputLayout};
use super::query_cache::CompiledExecution;
use super::result::{ExecutorResult, FilteredResult, ScannerResult};
use super::utils::{
    expression_contains_aggregate, extract_literal_value, flip_operator, infix_to_operator,
    is_aggregate_function, strip_table_qualifier,
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
);

/// The classifier's walk over one statement
struct Family<'a> {
    executor: &'a Executor,
    ctx: Option<&'a ExecutionContext>,
    /// Every FROM met, checked against the catalog once the syntax passed
    tables: RefCell<SmallVec<[&'a SimpleTableSource; 4]>>,
    /// The statement's own FROM when it names a table or a view
    statement_table: std::cell::Cell<Option<&'a SimpleTableSource>>,
}

impl<'a> Family<'a> {
    /// A FROM the family reads by its syntax: one table, no AS OF
    fn from(&self, table_expr: Option<&'a Expression>) -> std::result::Result<(), OutOfFamily> {
        let Some(Expression::TableSource(ts)) = table_expr else {
            return Err("FROM is not one table");
        };
        if ts.as_of.is_some() {
            return Err("FROM reads AS OF");
        }
        self.tables.borrow_mut().push(ts);
        Ok(())
    }

    /// The statement's FROM: one table or view, or a derived table; a view's
    /// or a derived table's own statement runs as any statement does
    fn statement_from(
        &self,
        table_expr: Option<&'a Expression>,
    ) -> std::result::Result<(), OutOfFamily> {
        match table_expr {
            Some(Expression::SubquerySource(_)) => Ok(()),
            Some(Expression::TableSource(ts)) if ts.as_of.is_none() => {
                self.statement_table.set(Some(ts));
                Ok(())
            }
            other => self.from(other),
        }
    }

    /// A block's FROM: one table, or none, which reads one row without columns
    fn block_from(
        &self,
        table_expr: Option<&'a Expression>,
    ) -> std::result::Result<(), OutOfFamily> {
        match table_expr {
            None => Ok(()),
            some => self.from(some),
        }
    }

    /// Every FROM met names a base table, not a CTE or a view; the
    /// statement's own may name a view
    fn catalog(&self) -> std::result::Result<(), OutOfFamily> {
        if let Some(ts) = self.statement_table.get() {
            if self
                .ctx
                .is_some_and(|ctx| ctx.has_cte_by_lower(&ts.name.value_lower))
            {
                return Err("FROM names a CTE");
            }
        }
        for ts in self.tables.borrow().iter() {
            let name = ts.name.value_lower.as_str();
            if self.ctx.is_some_and(|ctx| ctx.has_cte_by_lower(name)) {
                return Err("FROM names a CTE");
            }
            match self.executor.engine.get_view_lowercase(name) {
                Ok(None) => {}
                Ok(Some(_)) => return Err("FROM names a view"),
                Err(_) => return Err("the catalog could not be read"),
            }
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
        let subqueries = matches!(site, Site::Subqueries | Site::GroupedOutput);
        let mut walk = |e: &'a Expression| self.expression(e, scopes, site, found);
        match expr {
            Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::Parameter(_) => Ok(()),
            Expression::Infix(infix) => {
                use InfixOperator as I;
                match infix.op_type {
                    I::Equal
                    | I::NotEqual
                    | I::LessThan
                    | I::LessEqual
                    | I::GreaterThan
                    | I::GreaterEqual
                    | I::And
                    | I::Or
                    | I::Add
                    | I::Subtract
                    | I::Multiply
                    | I::Divide
                    | I::Modulo
                    | I::Like
                    | I::NotLike
                    | I::IsDistinctFrom
                    | I::IsNotDistinctFrom => {}
                    I::Is | I::IsNot if matches!(*infix.right, Expression::NullLiteral(_)) => {}
                    _ => return Err("an operator outside the family"),
                }
                walk(&infix.left)?;
                walk(&infix.right)
            }
            Expression::Prefix(prefix) => match prefix.op_type {
                PrefixOperator::Negate | PrefixOperator::Not => walk(&prefix.right),
                _ => Err("a prefix operator outside the family"),
            },
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
                        if is_tuple(&in_expr.left) {
                            return Err("a tuple on the left of IN (subquery)");
                        }
                        walk(&in_expr.left)?;
                        *found = true;
                        return self.block(&sq.subquery, scopes + 1);
                    }
                    _ => return Err("IN over something other than a list"),
                };
                walk(&in_expr.left)?;
                list.iter().try_for_each(walk)
            }
            Expression::Like(like) => {
                if !like.operator.eq_ignore_ascii_case("LIKE")
                    && !like.operator.eq_ignore_ascii_case("NOT LIKE")
                {
                    return Err("a pattern operator outside the family");
                }
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
            Expression::FunctionCall(call) => {
                if is_aggregate_function(&call.function) {
                    if !matches!(site, Site::Grouped | Site::GroupedOutput) {
                        return Err("an aggregate outside a subquery block");
                    }
                    return self.aggregate_call(call, scopes);
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
                self.block(&sq.subquery, scopes + 1)
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
                if is_tuple(&all_any.left) {
                    return Err("a tuple on the left of ANY or ALL");
                }
                walk(&all_any.left)?;
                *found = true;
                self.block(&all_any.subquery, scopes + 1)
            }
            Expression::Window(_) => Err("a window function"),
            _ => Err("an expression form outside the family"),
        }
    }

    /// One subquery block whose chain holds `scopes` scopes
    fn block(
        &self,
        sq: &'a SelectStatement,
        scopes: usize,
    ) -> std::result::Result<(), OutOfFamily> {
        if scopes > MAX_SCOPES {
            return Err("more than 32 scopes in a chain");
        }
        if !block_clauses(sq)
            || !sq.order_by.is_empty()
            || sq.limit.is_some()
            || sq.offset.is_some()
        {
            return Err("a subquery block with a clause outside the family");
        }
        self.block_from(sq.table_expr.as_deref())?;
        let [column] = sq.columns.as_slice() else {
            return Err("a subquery block without exactly one SELECT expression");
        };
        let column = unaliased(column);
        if matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
            return Err("a subquery block selecting *");
        }
        let mut found = false;
        self.expression(column, scopes, list_site(sq), &mut found)?;
        self.grouping(sq, scopes)?;
        match sq.where_clause.as_deref() {
            Some(where_clause) => {
                self.expression(where_clause, scopes, Site::Subqueries, &mut found)
            }
            None => Ok(()),
        }
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

    /// An aggregate call of a grouped block: what it reads per row holds
    /// values, and an argument after the first is a literal
    fn aggregate_call(
        &self,
        call: &'a FunctionCall,
        scopes: usize,
    ) -> std::result::Result<(), OutOfFamily> {
        let mut found = false;
        let mut values = |e: &'a Expression| self.expression(e, scopes, Site::Values, &mut found);
        let mut arguments = call.arguments.iter();
        match arguments.next() {
            None | Some(Expression::Star(_)) => {}
            Some(first) => values(first)?,
        }
        let literal = |arg: &Expression| {
            extract_literal_value(arg).is_some()
                || matches!(arg, Expression::Identifier(id) if id.quoted)
        };
        if !arguments.all(literal) {
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
        if !block_clauses(sq)
            || !sq.order_by.is_empty()
            || sq.limit.is_some()
            || sq.offset.is_some()
        {
            return Err("a subquery block with a clause outside the family");
        }
        self.block_from(sq.table_expr.as_deref())?;
        let mut found = false;
        for column in &sq.columns {
            if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
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
}

/// Whether a block aggregates its rows: it groups, has HAVING, or its
/// SELECT list calls an aggregate
fn grouped_block(sq: &SelectStatement) -> bool {
    !sq.group_by.columns.is_empty()
        || sq.having.is_some()
        || sq.columns.iter().any(expression_contains_aggregate)
}

/// What a block's SELECT list may hold
fn list_site(sq: &SelectStatement) -> Site {
    if grouped_block(sq) {
        Site::Grouped
    } else {
        Site::Subqueries
    }
}

/// `(a, b)`: more than one value on the left of a set test
fn is_tuple(expr: &Expression) -> bool {
    match expr {
        Expression::List(list) => list.elements.len() > 1,
        Expression::ExpressionList(list) => list.expressions.len() > 1,
        _ => false,
    }
}

/// No WITH, set operation, DISTINCT, GROUP BY, HAVING or WINDOW
fn plain_block(stmt: &SelectStatement) -> bool {
    block_clauses(stmt) && stmt.group_by.columns.is_empty() && stmt.having.is_none()
}

/// No WITH, set operation, DISTINCT, grouping set or WINDOW
fn block_clauses(stmt: &SelectStatement) -> bool {
    stmt.with.is_none()
        && stmt.set_operations.is_empty()
        && !stmt.distinct
        && stmt.distinct_on.is_empty()
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
        if !plain_block(stmt) {
            return Err("a clause outside the family");
        }
        let family = Family {
            executor: self,
            ctx,
            tables: RefCell::new(SmallVec::new()),
            statement_table: std::cell::Cell::new(None),
        };
        let mut found = false;
        if let Some(where_clause) = stmt.where_clause.as_deref() {
            family.expression(where_clause, 1, Site::Subqueries, &mut found)?;
        }
        if !found {
            return Err("WHERE holds no subquery");
        }
        for column in &stmt.columns {
            if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
                family.expression(unaliased(column), 1, Site::Values, &mut found)?;
            }
        }
        for ob in &stmt.order_by {
            family.expression(&ob.expression, 1, Site::Values, &mut found)?;
        }
        counts_fit(stmt, ctx)?;
        family.statement_from(stmt.table_expr.as_deref())?;
        family.catalog()
    }

    /// Whether `stmt` groups one table and holds subqueries in its SELECT
    /// list or HAVING, all of the family's forms and none inside an
    /// aggregate, while its WHERE, GROUP BY and ORDER BY hold none, or why not
    pub(crate) fn grouped_family(
        &self,
        stmt: &SelectStatement,
        ctx: Option<&ExecutionContext>,
    ) -> std::result::Result<(), OutOfFamily> {
        if ctx.is_some_and(|ctx| ctx.outer_row().is_some()) {
            return Err("it runs for a row of an enclosing statement");
        }
        if !block_clauses(stmt) {
            return Err("a clause outside the family");
        }
        if !grouped_block(stmt) {
            return Err("a statement that does not group");
        }
        let held = |expr: Option<&Expression>| expr.is_some_and(mentions_subquery);
        // A GROUP BY item naming a SELECT alias or position groups that item
        let items = self
            .parse_group_by(stmt, &[])
            .map_err(|_| "a GROUP BY position outside the SELECT list")?;
        if held(stmt.where_clause.as_deref())
            || stmt.group_by.columns.iter().any(|item| held(Some(item)))
            || items.iter().any(|item| match item {
                GroupByItem::Expression { expr, .. } => mentions_subquery(expr),
                GroupByItem::Column(_) => false,
            })
            || stmt.order_by.iter().any(|ob| held(Some(&ob.expression)))
        {
            return Err("a WHERE, GROUP BY or ORDER BY holding a subquery");
        }
        let family = Family {
            executor: self,
            ctx,
            tables: RefCell::new(SmallVec::new()),
        };
        let mut found = false;
        for column in &stmt.columns {
            if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
                family.expression(unaliased(column), 1, Site::GroupedOutput, &mut found)?;
            }
        }
        if let Some(having) = stmt.having.as_deref() {
            family.expression(having, 1, Site::GroupedOutput, &mut found)?;
        }
        if !found {
            return Err("neither the SELECT list nor HAVING holds a subquery");
        }
        counts_fit(stmt, ctx)?;
        family.from(stmt.table_expr.as_deref())?;
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
struct ScopeTable {
    tables: SmallVec<[(SmartString, usize); 2]>,
    /// The names its programs compile against
    columns: CompactArc<Vec<String>>,
    /// Each column's own name, lowercase, as a bare reference names it
    lower: CompactArc<Vec<String>>,
    types: Vec<DataType>,
}

impl ScopeTable {
    fn of(name: SmartString, schema: &Schema) -> Self {
        Self {
            tables: smallvec::smallvec![(name, 0)],
            columns: schema.column_names_arc(),
            lower: schema.column_names_lower_arc(),
            types: schema.columns.iter().map(|c| c.data_type).collect(),
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
        }
    }

    /// The joined row of `sides`, each a name and its table's schema, as a
    /// join of `width` columns lays it out; None when they do not line up.
    /// Its programs read each column by position, under a name no column
    /// can have, so no name a table chose can bind it elsewhere
    fn joined(sides: [(SmartString, &Schema); 2], width: usize) -> Option<Self> {
        if sides
            .iter()
            .map(|(_, schema)| schema.columns.len())
            .sum::<usize>()
            != width
        {
            return None;
        }
        let mut tables = SmallVec::new();
        let mut lower = Vec::with_capacity(width);
        let mut types = Vec::with_capacity(width);
        for (name, schema) in sides {
            tables.push((name, lower.len()));
            lower.extend(schema.columns.iter().map(|c| c.name.to_lowercase()));
            types.extend(schema.columns.iter().map(|c| c.data_type));
        }
        Some(Self {
            tables,
            columns: CompactArc::new((0..width).map(|i| format!("{RESOLVED}#{i}")).collect()),
            lower: CompactArc::new(lower),
            types,
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

    /// An error when a column in `rest` of one table is also named `name`:
    /// a derived table may repeat a name, and a reference to it is ambiguous
    fn unique(&self, mut rest: std::ops::Range<usize>, name: &str, spelled: &str) -> Result<()> {
        match rest.any(|i| self.lower[i] == name) {
            true => Err(Error::invalid_argument(format!(
                "column reference '{}' is ambiguous",
                spelled
            ))),
            false => Ok(()),
        }
    }

    /// The column `expr` names as this scope spells its own columns
    fn position(&self, expr: &Expression) -> Option<usize> {
        match expr {
            Expression::Identifier(id) if self.tables.len() > 1 => {
                self.columns.iter().position(|c| *c == id.value_lower)
            }
            Expression::QualifiedIdentifier(q) if q.qualifier.value_lower == RESOLVED => {
                self.lower.iter().position(|c| *c == q.name.value_lower)
            }
            _ => None,
        }
    }
}

/// What an expression reads, one bit per scope of the chain: bit 0 the
/// block's own table, bit 1 the enclosing scope, and so on
type Reads = u32;

/// A reference resolved to a scope of the chain and a column of its table
type Resolved = Option<(usize, usize)>;

/// A block's scope chain, nearest first
struct Chain<'a> {
    scopes: &'a [&'a ScopeTable],
}

impl Chain<'_> {
    /// `id` by the name rules: the nearest scope with that column, or None
    /// for a name the compiler reads as a literal
    fn bare(&self, id: &Identifier) -> Result<Resolved> {
        if !id.quoted && matches!(id.value_lower.as_str(), "true" | "false") {
            return Ok(None);
        }
        // A name is a column's whole name before its dot is a qualifier's
        for (depth, scope) in self.scopes.iter().enumerate() {
            if let Some(index) = scope.lower.iter().position(|c| *c == id.value_lower) {
                let (_, range) = scope.table(index);
                scope.unique(index + 1..range.end, &id.value_lower, &id.value)?;
                return Ok(Some((depth, index)));
            }
        }
        if let Some(dot) = id.value_lower.rfind('.') {
            let resolved = self.qualified(
                &id.value_lower[..dot],
                &id.value_lower[dot + 1..],
                &id.value,
            );
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
    fn qualified(&self, qualifier: &str, column: &str, spelled: &str) -> Result<(usize, usize)> {
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
            .ok_or_else(|| {
                Error::invalid_argument(format!("unknown table or alias '{}'", qualifier))
            })?;
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
                .qualified(
                    &q.qualifier.value_lower,
                    &q.name.value_lower,
                    &q.to_string(),
                )
                .map(Some),
            _ => Ok(None),
        }
    }

    /// The column `index` of the scope at `depth`, as a name its own
    /// programs compile against. Over one table, qualified by a name no
    /// table can have, so the compiler reads it as a column wherever it
    /// stands, never a literal; over a join, the scope's name for its position
    fn column(&self, depth: usize, index: usize) -> Expression {
        let ident = |value: &str, lower: &str| {
            let mut id = Identifier::new(
                Token::new(TokenType::Identifier, value, Position::default()),
                value,
            );
            id.value_lower = SmartString::from(lower);
            id
        };
        let scope = self.scopes[depth];
        let spelled = scope.columns[index].as_str();
        if scope.tables.len() > 1 {
            return Expression::Identifier(ident(spelled, spelled));
        }
        Expression::QualifiedIdentifier(QualifiedIdentifier {
            qualifier: ident(RESOLVED, RESOLVED),
            name: ident(spelled, &scope.lower[index]),
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

/// Each direct operand of an expression of the family's forms
fn for_each_child(expr: &Expression, f: &mut impl FnMut(&Expression) -> Result<()>) -> Result<()> {
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
        other => other.clone(),
    })
}

/// `expr`, rewritten over the scope `side` alone, with each of its columns
/// spelled as the scope `joined` names it, the side's first at `offset`
fn in_joined(
    expr: &Expression,
    side: &ScopeTable,
    joined: &ScopeTable,
    offset: usize,
) -> Result<Expression> {
    let index = bound_column(expr).and_then(|name| side.lower.iter().position(|c| c == name));
    match index {
        Some(index) => {
            let spelled = joined.columns[offset + index].as_str();
            Ok(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, spelled, Position::default()),
                spelled,
            )))
        }
        None => rebuild(expr, &mut |child| in_joined(child, side, joined, offset)),
    }
}

/// The lowercase name of a column the resolver bound, as `column` spells it
fn bound_column(expr: &Expression) -> Option<&str> {
    match expr {
        Expression::QualifiedIdentifier(q) if q.qualifier.value_lower == RESOLVED => {
            Some(q.name.value_lower.as_str())
        }
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
        Expression::QualifiedIdentifier(_) => bound_column(expr)
            .and_then(|name| schema.column_index_map().get(name))
            .is_none_or(|&i| scan_compares(schema.columns[i].data_type)),
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
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => true,
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
            .and_then(|name| schema.column_index_map().get(name))
            .map(|&i| schema.columns[i].data_type)
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

fn slot_parameter(slot: usize) -> Expression {
    let name = format!("${}", slot + 1);
    Expression::Parameter(Parameter {
        token: Token::new(TokenType::Parameter, &name, Position::default()),
        name: SmartString::from_string(name),
        index: slot + 1,
    })
}

fn compile_error(error: CompileError) -> Error {
    Error::internal(format!("Compile error: {}", error))
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
    fn open<'s>(
        expr: &Expression,
        aliased: &impl Fn(&str) -> Option<&'s Expression>,
    ) -> Result<Expression> {
        match expr {
            Expression::Identifier(id) => Ok(aliased(&id.value_lower)
                .cloned()
                .unwrap_or_else(|| expr.clone())),
            other => rebuild(other, &mut |child| open(child, aliased)),
        }
    }
    open(having, &aliased)
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
    program: SharedProgram,
    /// The conjuncts without a subquery, which the scan is narrowed by
    pushdown: Option<Expression>,
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

/// `column [NOT] IN (block)` or `column op ANY|ALL (block)` at the top of
/// a WHERE, over an uncorrelated block
#[derive(Debug, Clone, Copy)]
struct SetSlot {
    id: u16,
    column: usize,
    negated: bool,
}

/// The filter a scan of a prepared WHERE takes, whether it is the whole
/// WHERE, and whether it takes the extra conditions whole
type WhereScan = (Option<Box<dyn StorageExpr>>, bool, bool);

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
    /// Each top-level EXISTS or IN test: its block, whether negated, and
    /// the statement column an IN tests
    set_tests: Vec<(u16, bool, Option<usize>)>,
    conjuncts: usize,
}

impl WhereParts {
    /// The WHERE as `program`, narrowed by the first set test whose block
    /// in `blocks` can name its rows
    fn prepare(self, program: SharedProgram, blocks: &PreparedBlocks) -> PreparedWhere {
        let whole = self.conjuncts == 1;
        let narrowing = self.set_tests.iter().find_map(|&(id, negated, column)| {
            let sq = &blocks.subqueries[id as usize];
            let (column, members) = match (&sq.answer, column) {
                (
                    Answer::Exists(Kept {
                        keyed: Some(plan), ..
                    }),
                    None,
                ) if plan.keys.len() == 1 => (plan.keys[0].2, false),
                (Answer::Members(_, members), Some(column))
                    if sq.uncorrelated && members.test == MemberTest::In =>
                {
                    (column, true)
                }
                _ => return None,
            };
            Some(Narrowing {
                id,
                column,
                whole,
                negated,
                members,
            })
        });
        let slot = |&(id, negated, column): &(u16, bool, Option<usize>)| {
            let sq = &blocks.subqueries[id as usize];
            let column = column.filter(|_| sq.uncorrelated)?;
            matches!(sq.answer, Answer::Members(..)).then_some(SetSlot {
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
        PreparedWhere {
            program,
            pushed: self.pushed.len(),
            conjuncts: self.conjuncts,
            pushdown: conjunction(self.pushed),
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
    /// The subquery whose row this one runs for; None for the statement's row
    parent: Option<u16>,
    /// What fills each slot from the parent's row, including values a
    /// deeper subquery reads from it; empty for the statement's own blocks,
    /// whose holes its `HoleSet` keeps
    holes: Vec<(usize, Hole)>,
    /// A join statement's own block's holes as written, compiled again for
    /// each layout of the row its sites run over
    hole_exprs: Vec<(usize, Expression)>,
    /// Reads no enclosing scope: its value is computed once per run
    uncorrelated: bool,
}

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
}

/// A block run as a statement of its own, with the plan its runs keep, as
/// a cached statement keeps its own
struct StatementRun {
    statement: SelectStatement,
    classification: std::sync::OnceLock<Arc<super::query_classification::QueryClassification>>,
    compiled: RwLock<CompiledExecution>,
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
    read_as: Option<DataType>,
    test: MemberTest,
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
}

/// `column operator value` with the column resolved at preparation; only
/// the value is bound per call
#[derive(Debug)]
struct PreparedComparison {
    column: String,
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
            let index = *schema.column_index_map().get(column)?;
            if !scan_compares(schema.columns[index].data_type) {
                return None;
            }
            return Some(Self {
                column: schema.columns[index].name.clone(),
                index,
                data_type: schema.columns[index].data_type,
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
    Program(SharedProgram),
    /// An ORDER BY or DISTINCT ON key, NULL where it fails, as on the general path
    Key(SharedProgram),
    /// A key that reads SELECT aliases from the projected row, as its second row
    AliasKey(SharedProgram),
}

/// The SELECT list, then the columns and expressions ORDER BY appends for the sort
#[derive(Debug)]
struct PreparedProjection {
    outputs: Vec<Output>,
    /// The outputs are the row's columns in order: each row is its own projection
    whole_row: bool,
}

/// What preparing one statement builds up
struct Builder<'e> {
    executor: &'e Executor,
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
/// expression a scope further out computes into it, and that scope's depth
type Delegated = (usize, Expression, usize);

/// What rewriting one block's expressions collects
#[derive(Default)]
struct Rewritten {
    holes: Vec<(usize, Expression)>,
    delegated: Vec<Delegated>,
    /// Whether a hole reads the parent's row
    reads_parent: bool,
    /// Whether anything read is a column of the block's own row
    reads_row: bool,
    /// The slot of each enclosing column read, by scope depth and column:
    /// one per column, so two reads of it are one expression
    outer: FxHashMap<(usize, usize), usize>,
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
            while let Some(at) = ancestor {
                let Some(table) = subqueries[nested].table.clone() else {
                    break;
                };
                let (Answer::Value(_, kept) | Answer::Exists(kept) | Answer::Members(kept, _)) =
                    &mut subqueries[at as usize].answer;
                if !kept.tables.contains(&table) {
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
            let (Answer::Value(_, kept) | Answer::Exists(kept) | Answer::Members(kept, _)) =
                &mut sq.answer;
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

    /// `expr` of the statement as written, with each subquery in it prepared
    /// as a block of the statement's row; what stands outside them, which
    /// the grouped path reads by name, stays as written
    fn rewrite_blocks(&mut self, chain: &Chain<'_>, expr: &Expression) -> Result<Expression> {
        let mut work = Rewritten::default();
        match expr {
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
        let once = block
            .and_then(|id| self.subqueries.get(usize::from(id))?.as_ref())
            .is_some_and(|sq| {
                sq.uncorrelated
                    && sq.parent.is_none()
                    && matches!(sq.answer, Answer::Value(..) | Answer::Exists(_))
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
                    Some((depth, index)) => {
                        if let Some(&slot) = work.outer.get(&(depth, index)) {
                            return Ok(slot_parameter(slot));
                        }
                        let slot = self.slot();
                        work.outer.insert((depth, index), slot);
                        let value = chain.column(depth, index);
                        if depth == 1 {
                            work.reads_parent = true;
                            work.holes.push((slot, value));
                        } else {
                            work.delegated.push((slot, value, depth));
                        }
                        Ok(slot_parameter(slot))
                    }
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
        for (slot, value, depth) in delegated {
            if depth == 2 {
                work.reads_parent = true;
                work.holes.push((slot, value));
            } else {
                work.delegated.push((slot, value, depth - 1));
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
                        } else {
                            work.delegated.push((slot, value_in_scope, depth));
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
        // EXISTS, value and grouped blocks pick candidates by their correlation equalities
        let picks = grouped_block(sq) || !matches!(form, Form::Scalar);
        let executor = self.executor;
        let registry = &executor.function_registry;
        let (table, schema, scope) = match sq.table_expr.as_deref() {
            Some(Expression::TableSource(ts)) => {
                let schema = executor
                    .engine
                    .get_table_schema(&ts.name.value_lower)
                    .map_err(|e| match e {
                        Error::TableNotFound(_) => {
                            Error::TableOrViewNotFound(ts.name.value.to_string())
                        }
                        other => other,
                    })?;
                let name = ts.alias.as_ref().unwrap_or(&ts.name).value_lower.clone();
                let scope = ScopeTable::of(name, &schema);
                (Some(ts.name.value_lower.clone()), schema, scope)
            }
            // A block without FROM reads one row without columns, under no name
            None => (
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
        let projection = match form {
            Form::Exists(_) => None,
            Form::Scalar | Form::Members { .. } => {
                Some(self.rewrite(&chain, Some(id), unaliased(&sq.columns[0]), &mut work)?)
            }
        };
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
                match PreparedComparison::from_conjunct(&part, &schema) {
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
        let grouped = if grouped_block(sq) {
            let exists = matches!(form, Form::Exists(_));
            Some(self.grouping(&chain, id, sq, projection.as_ref(), exists, &mut work)?)
        } else {
            None
        };

        let mut read_as = FxHashMap::default();
        for expr in candidates.iter().chain(&filter) {
            compare_as_scan(expr, &schema, &mut read_as);
        }
        let own = CompileContext::new(&scope.columns, registry)
            .with_prepared_subqueries()
            .with_read_as(&read_as);
        let compile = |expr: &Option<Expression>| match expr {
            Some(expr) => compile_with(&own, expr).map(Some),
            None => Ok(None),
        };
        let projection_program = || match grouped.as_ref().and_then(|g| g.projection.as_ref()) {
            Some(over_slots) => Ok(CompactArc::clone(over_slots)),
            None => compile(&projection)?
                .ok_or_else(|| Error::internal("a prepared subquery without a SELECT expression")),
        };
        let as_value = |block: &SelectStatement| {
            Expression::ScalarSubquery(ScalarSubquery {
                token: block.token.clone(),
                subquery: Box::new(block.clone()),
            })
        };
        let (identity, signature, keyed_test) = match form {
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
        };
        let mut others = SmallVec::new();
        for expr in projection.iter().chain(&filter) {
            slots_read(expr, &mut others);
        }
        if let Some(grouped) = &grouped {
            others.extend(grouped.reads.iter().copied());
        }
        let keyed =
            (keyed_test && grouped.is_none() && self.keyed(&chain, &keys, &work, sq, &others))
                .then(|| keyed_plan(&keys, filter.as_ref(), &schema));
        // What the block answers and how its members are read is part of what is kept
        let mut scopes = signature;
        for scope in outer {
            for (name, _) in &scope.tables {
                scopes.push_str(name);
                scopes.push('\u{2}');
            }
            for (column, data_type) in scope.lower.iter().zip(&scope.types) {
                scopes.push_str(&format!("\u{0}{column}:{data_type:?}"));
            }
            scopes.push('\u{1}');
        }
        let kept = Kept {
            identity_key: ExprKey::of(&identity),
            identity: Arc::new(identity),
            scopes: Arc::from(scopes),
            tables: table.iter().cloned().collect(),
            params: None,
            keyed,
        };
        let answer = match form {
            Form::Scalar => Answer::Value(projection_program()?, kept),
            Form::Members { read_as, test, .. } => Answer::Members(
                kept,
                Members {
                    projection: projection_program()?,
                    read_as,
                    test,
                },
            ),
            Form::Exists(_) => Answer::Exists(kept),
        };
        let mut reads = SmallVec::new();
        for expr in projection.iter().chain(&candidates).chain(&filter) {
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
                    grouping.keyed = Some(keyed_plan(&keys, filter.as_ref(), &schema));
                }
                Some(Box::new(grouping))
            }
            None => None,
        };
        let candidates = compile(&candidates)?;
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
        let holes = if work.holes.is_empty() {
            Vec::new()
        } else {
            let parent = CompileContext::new(&outer[0].columns, registry);
            work.holes
                .iter()
                .map(|(slot, hole)| match outer[0].position(hole) {
                    Some(index) => Ok((*slot, Hole::Column(index))),
                    None => compile_with(&parent, hole).map(|p| (*slot, Hole::Program(p))),
                })
                .collect::<Result<Vec<_>>>()?
        };
        let hole_exprs = if parent.is_none() && (outer[0].tables.len() > 1 || self.keeps_holes) {
            work.holes.clone()
        } else {
            Vec::new()
        };
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
                    compiled: RwLock::new(CompiledExecution::Unknown),
                }));
            }
        }
        self.subqueries[id as usize] = Some(PreparedSubquery {
            table,
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
            parent,
            holes,
            hole_exprs,
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
        projection: Option<&Expression>,
        exists: bool,
        work: &mut Rewritten,
    ) -> Result<Grouped> {
        let executor = self.executor;
        let registry = &executor.function_registry;
        let scope = chain.scopes[0];
        // The items as the general path binds a SELECT alias or a position
        let bound = executor.parse_group_by(sq, &scope.columns)?;
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
            items.push(self.rewrite(chain, Some(id), expr, work)?);
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
        for expr in projection.into_iter().chain(&having).chain(&listed) {
            if let Some(name) = ungrouped(expr, &item_keys) {
                return Err(ungrouped_column(&name));
            }
        }

        let calls_of = SelectStatement {
            param_count: 0,
            token: sq.token.clone(),
            distinct: false,
            distinct_on: Vec::new(),
            columns: projection.into_iter().cloned().collect(),
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
        let (aggregations, _) = executor.parse_aggregations(&calls_of, None)?;
        let calls = aggregations
            .iter()
            .map(|agg| agg.template(registry))
            .collect::<Result<Vec<_>>>()?;
        let own = CompileContext::new(&scope.columns, registry);
        let inputs = CallInputs::compile_with(&aggregations, |expr| compile_with(&own, expr))?;
        let columns = aggregations
            .iter()
            .map(|agg| {
                if agg.column == "*" || agg.expression.is_some() {
                    return Ok(None);
                }
                agg.column_lower
                    .strip_prefix(RESOLVED)
                    .and_then(|name| name.strip_prefix('.'))
                    .and_then(|name| scope.lower.iter().position(|c| c == name))
                    .map(Some)
                    .ok_or_else(|| Error::internal("an aggregate's column outside its block"))
            })
            .collect::<Result<Vec<_>>>()?;

        let width = items.len() + aggregations.len();
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
        let names: Vec<String> = (0..width).map(|i| format!("{RESOLVED}#{i}")).collect();
        let over_slots = CompileContext::new(&names, registry).with_expression_aliases(slots);
        let projection = projection
            .map(|expr| compile_with(&over_slots, expr))
            .transpose()?;
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
                let column =
                    bound_column(item).and_then(|name| scope.lower.iter().position(|c| c == name));
                Ok(match column {
                    Some(column) => Hole::Column(column),
                    None => Hole::Program(compile_with(&own, item)?),
                })
            })
            .collect::<Result<Vec<_>>>()?;
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
                *depth == 1
                    && own == chain.scopes[1].types[*index]
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
    projection: Option<SharedProgram>,
    reads: SmallVec<[usize; 2]>,
}

/// The name of a block's own column that `expr` reads outside every
/// aggregate call and every GROUP BY item, keyed by `items`
fn ungrouped(expr: &Expression, items: &FxHashSet<ExprKey>) -> Option<String> {
    if items.contains(&ExprKey::of(expr)) {
        return None;
    }
    match expr {
        Expression::FunctionCall(call) if is_aggregate_function(&call.function) => None,
        Expression::QualifiedIdentifier(q) if q.qualifier.value_lower == RESOLVED => {
            Some(q.name.value.to_string())
        }
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
/// rest of its WHERE being `filter`
fn keyed_plan(
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

fn mentions_column(expr: &Expression) -> bool {
    match expr {
        Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => true,
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

impl PreparedProjection {
    /// The SELECT list over `chain`'s one table and what ORDER BY appends,
    /// with the names and the sources of the appended columns
    fn prepare(
        executor: &Executor,
        stmt: &SelectStatement,
        chain: &Chain<'_>,
    ) -> Result<(Self, Vec<String>, Vec<ColumnSource>)> {
        let scope = chain.scopes[0];
        let columns: &[String] = &scope.columns;
        let aliases: Vec<&str> = stmt
            .columns
            .iter()
            .filter_map(|c| match c {
                Expression::Aliased(a) => Some(a.alias.value_lower.as_str()),
                _ => None,
            })
            .collect();
        let mut select_exprs = Vec::with_capacity(stmt.columns.len());
        // The column each item of a `*` is, by position: names may repeat
        let mut star_columns: Vec<Option<usize>> = Vec::with_capacity(stmt.columns.len());
        for column in &stmt.columns {
            match column {
                Expression::Star(_) => {
                    select_exprs.extend((0..columns.len()).map(|i| chain.column(0, i)));
                    star_columns.extend((0..columns.len()).map(Some));
                }
                Expression::QualifiedStar(star) => {
                    if star.qualifier.to_lowercase() != scope.tables[0].0.as_str() {
                        return Err(Error::invalid_argument(format!(
                            "unknown table or alias '{}'",
                            star.qualifier
                        )));
                    }
                    select_exprs.extend((0..columns.len()).map(|i| chain.column(0, i)));
                    star_columns.extend((0..columns.len()).map(Some));
                }
                other => {
                    check_names(chain, unaliased(other), &[])?;
                    select_exprs.push(other.clone());
                    star_columns.push(None);
                }
            }
        }
        // A bare key names an alias first; inside an expression a column of
        // the same name comes first, so only the aliases this map keeps
        let alias_map = Executor::build_alias_map_excluding(&stmt.columns, Some(columns));
        let inner_aliases: Vec<&str> = alias_map.keys().map(String::as_str).collect();
        for ob in &stmt.order_by {
            match &ob.expression {
                Expression::Identifier(id) if aliases.contains(&id.value_lower.as_str()) => {}
                key => check_names(chain, key, &inner_aliases)?,
            }
        }
        let plan = executor.order_by_projection(&select_exprs, &stmt.order_by, &[], columns, false);
        let registry = &executor.function_registry;
        let ctx = std::cell::OnceCell::new();
        let compile = |expr: &Expression| {
            compile_with(
                ctx.get_or_init(|| CompileContext::new(columns, registry)),
                unaliased(expr),
            )
        };
        let mut outputs = Vec::with_capacity(select_exprs.len() + plan.extra_names.len());
        for ((expr, index), star) in select_exprs
            .iter()
            .zip(&plan.select_column_indices)
            .zip(&star_columns)
        {
            outputs.push(match star.or(*index) {
                Some(index) => Output::Column(index),
                None => Output::Program(compile(expr)?),
            });
        }
        outputs.extend(plan.extra_order_indices.iter().map(|&i| Output::Column(i)));
        for expr in &plan.computed_order_by {
            outputs.push(Output::Key(compile(expr)?));
        }
        // A key reading a SELECT alias reads the value the projection gave
        // it, under the key's own text, which is how the sort finds it
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
        let keys: Vec<&Expression> = stmt
            .order_by
            .iter()
            .map(|ob| &ob.expression)
            .filter(|key| !is_leaf(key) && names_any(key, &|name| positions.contains_key(name)))
            .collect();
        let mut bindings = FxHashMap::default();
        for key in &keys {
            bind_aliases(key, &positions, &mut bindings);
        }
        let alias_ctx = CompileContext::new(columns, registry).with_row2_bindings(&bindings);
        for key in keys {
            let name = key.to_string();
            if extra_names.contains(&name) {
                continue;
            }
            outputs.push(Output::AliasKey(compile_with(&alias_ctx, key)?));
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
        Ok((Self { outputs, whole_row }, extra_names, extra_sources))
    }

    fn project(&self, mut rows: RowVec, ctx: &ExecutionContext) -> Result<RowVec> {
        let mut projected = RowVec::with_capacity(rows.len());
        if self.whole_row {
            for (row_id, row) in rows.drain_rows().enumerate() {
                projected.push((row_id as i64, row));
            }
            return Ok(projected);
        }
        let mut vm = ExprVM::new();
        for (row_id, row) in rows.drain_rows().enumerate() {
            let mut values = Vec::with_capacity(self.outputs.len());
            let exec = ExecuteContext::with_common_params(&row, ctx.params(), ctx.transaction_id());
            for output in &self.outputs {
                values.push(match output {
                    Output::Column(i) => row.get(*i).cloned().unwrap_or_else(Value::null_unknown),
                    Output::Program(program) => vm.execute(program, &exec)?,
                    Output::Key(program) => vm.execute(program, &exec)?,
                    Output::AliasKey(_) => Value::null_unknown(),
                });
            }
            let mut out = Row::from_values(values);
            for (position, output) in self.outputs.iter().enumerate() {
                if let Output::AliasKey(program) = output {
                    let mut exec = ExecuteContext::with_common_params(
                        &row,
                        ctx.params(),
                        ctx.transaction_id(),
                    );
                    exec.row2 = Some(&out);
                    let key = vm.execute(program, &exec)?;
                    out.set(position, key)?;
                }
            }
            projected.push((row_id as i64, out));
        }
        Ok(projected)
    }
}

impl PreparedSelect {
    /// Prepare `stmt` over its table, named `scope_name` and holding
    /// `columns`, for executions with `param_count` positional parameters
    fn prepare(
        executor: &Executor,
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
            next_slot: param_count,
            subqueries: Vec::new(),
            max_param: 0,
            keeps_holes: false,
        };
        let where_clause = stmt
            .where_clause
            .as_deref()
            .ok_or_else(|| Error::internal("a prepared statement without WHERE"))?;
        let where_clause = with_aliases(where_clause, &stmt.columns, &scope.columns);
        let mut work = Rewritten::default();
        let mut run_slots = Vec::new();
        let parts =
            builder.statement_where(&chain, schema, &where_clause, &mut work, &mut run_slots)?;
        let mut read_as = FxHashMap::default();
        compare_as_scan(&parts.clause, schema, &mut read_as);
        let own = CompileContext::new(&scope.columns, registry)
            .with_prepared_subqueries()
            .with_read_as(&read_as);
        let compile = |expr: &Expression| compile_with(&own, expr);
        let where_program = compile(&parts.clause)?;
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile(value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let (projection, extra_names, extra_sources) =
            PreparedProjection::prepare(executor, stmt, &chain)?;
        for column in stmt
            .columns
            .iter()
            .chain(stmt.order_by.iter().map(|ob| &ob.expression))
        {
            builder.note_params(unaliased(column));
        }
        let (blocks, holes) = builder.finish(param_count, run_slots)?;
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
        // The layout an ORDER BY binds its names by, built with the plan
        let layout = (!stmt.order_by.is_empty()).then(|| {
            let mut layout = Executor::select_layout(&select, &scope.columns, Some(scope_name));
            layout.sources.extend(extra_sources);
            CompactArc::new(layout)
        });
        let where_ = parts.prepare(where_program, &blocks);
        Ok(Self {
            cached_epoch: epoch,
            columns: scope.columns.clone(),
            blocks,
            holes,
            where_,
            projection,
            output_columns: CompactArc::new(output_columns),
            visible,
            layout,
        })
    }
}

/// A grouped statement over one table whose SELECT list or HAVING holds
/// subqueries, prepared once as blocks of its rows: the statement the
/// grouped path runs, with those clauses rewritten and each value read once
/// a run slot
#[derive(Debug)]
pub struct PreparedGrouped {
    cached_epoch: u64,
    /// The table's schema the clauses were prepared over
    schema: CompactArc<Schema>,
    blocks: PreparedBlocks,
    /// Each block's holes, read from the GROUP BY items of its group's row
    holes: HoleSet,
    /// The uncorrelated member blocks it reads, read before any row
    ahead: Vec<u16>,
    pub(crate) stmt: SelectStatement,
    pub(crate) classification: Arc<super::query_classification::QueryClassification>,
    /// Every subquery is a run slot: the statement runs as any grouped
    /// statement does, reading them as parameters, under this plan of its own
    pub(crate) runs_alone: Option<RwLock<CompiledExecution>>,
}

impl PreparedGrouped {
    /// Prepare `stmt` over its table, named `scope_name`; None when a block
    /// reads a column of the row that is not a GROUP BY item
    fn prepare(
        executor: &Executor,
        stmt: &SelectStatement,
        schema: CompactArc<Schema>,
        scope_name: &SmartString,
        param_count: usize,
        epoch: u64,
    ) -> Result<Option<Self>> {
        let scope = ScopeTable::of(scope_name.clone(), &schema);
        let scopes = [&scope];
        let chain = Chain { scopes: &scopes };
        let mut builder = Builder {
            executor,
            next_slot: param_count,
            subqueries: Vec::new(),
            max_param: 0,
            keeps_holes: true,
        };
        let mut run_slots = Vec::new();
        let mut rewrite = |expr: &Expression| -> Result<Expression> {
            let rewritten = builder.rewrite_blocks(&chain, expr)?;
            builder.slotted(&rewritten, &mut run_slots)
        };
        // An item holding a subquery keeps the name it is written under
        let mut columns = Vec::with_capacity(stmt.columns.len());
        for column in &stmt.columns {
            columns.push(match column {
                column if !mentions_subquery(unaliased(column)) => column.clone(),
                Expression::Aliased(aliased) => Expression::Aliased(AliasedExpression {
                    token: aliased.token.clone(),
                    expression: Box::new(rewrite(&aliased.expression)?),
                    alias: aliased.alias.clone(),
                }),
                column => {
                    let name = executor.grouped_item_name(column);
                    Expression::Aliased(AliasedExpression {
                        token: Token::new(TokenType::Identifier, &name, Position::default()),
                        expression: Box::new(rewrite(column)?),
                        alias: Identifier::new(
                            Token::new(TokenType::Identifier, &name, Position::default()),
                            &name,
                        ),
                    })
                }
            });
        }
        let having = match stmt.having.as_deref() {
            Some(having) if mentions_subquery(having) => {
                let opened = grouped_having(executor, stmt, having)?;
                Some(Box::new(rewrite(&opened)?))
            }
            other => other.map(|having| Box::new(having.clone())),
        };
        let blocks_read = builder.subqueries.len();
        let values = CompileContext::new(&scope.columns, &executor.function_registry)
            .with_prepared_subqueries();
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile_with(&values, value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let (blocks, _) = builder.finish(param_count, run_slots)?;
        // A group's row holds its GROUP BY items first: a column a hole reads
        // is the item naming it, and a column no item names cannot be read
        let group_slots: Vec<String> = stmt
            .group_by
            .columns
            .iter()
            .enumerate()
            .map(|(item, expr)| {
                let named = |name: &str| schema.columns.iter().position(|c| c.name_lower == name);
                let column = match expr {
                    Expression::Identifier(id) => named(&id.value_lower),
                    Expression::QualifiedIdentifier(q)
                        if q.qualifier.value_lower == *scope_name =>
                    {
                        named(&q.name.value_lower)
                    }
                    _ => None,
                };
                match column {
                    Some(column) => scope.columns[column].clone(),
                    None => format!("\0group item {item}"),
                }
            })
            .collect();
        let slots_ctx = CompileContext::new(&group_slots, &executor.function_registry);
        let mut holes = Vec::with_capacity(blocks.subqueries.len());
        for sq in &blocks.subqueries {
            let mut grouped = Vec::with_capacity(sq.hole_exprs.len());
            for (slot, hole) in &sq.hole_exprs {
                let read = match hole {
                    Expression::Identifier(id) => group_slots
                        .iter()
                        .position(|name| *name == id.value_lower)
                        .map(Hole::Column),
                    _ => None,
                };
                let read = match read {
                    Some(read) => read,
                    None => match compile_with(&slots_ctx, hole) {
                        Ok(program) => Hole::Program(program),
                        Err(_) => return Ok(None),
                    },
                };
                grouped.push((*slot, read));
            }
            holes.push(grouped);
        }
        let ahead = (0..blocks_read)
            .filter(|&id| {
                let sq = &blocks.subqueries[id];
                sq.uncorrelated && sq.parent.is_none() && matches!(sq.answer, Answer::Members(..))
            })
            .map(|id| u16::try_from(id).map_err(|_| Error::internal("too many subqueries")))
            .collect::<Result<Vec<_>>>()?;
        let stmt = SelectStatement {
            columns,
            having,
            ..stmt.clone()
        };
        let runs_alone = (!stmt.columns.iter().any(|c| mentions_subquery(unaliased(c)))
            && !stmt.having.as_deref().is_some_and(mentions_subquery))
        .then(|| RwLock::new(CompiledExecution::Unknown));
        Ok(Some(Self {
            cached_epoch: epoch,
            schema,
            blocks,
            holes: HoleSet { holes },
            ahead,
            classification: super::query_classification::get_classification(&stmt),
            stmt,
            runs_alone,
        }))
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
    /// The plan's prepared grouped statement over `table`, prepared now when
    /// it holds none for this schema and parameter count; None when the
    /// statement's blocks read a column outside its groups
    pub(crate) fn prepared_grouped(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Result<Option<Arc<PreparedGrouped>>> {
        let Some(Expression::TableSource(ts)) = stmt.table_expr.as_deref() else {
            return Err(Error::internal("a grouped statement without one table"));
        };
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
        if held.is_some() {
            return Ok(held);
        }
        let table = self.prepared_table(&ts.name.value_lower, ctx, None)?;
        let scope_name = ts
            .alias
            .as_ref()
            .map_or(&ts.name.value_lower, |alias| &alias.value_lower);
        let Some(prepared) = PreparedGrouped::prepare(
            self,
            stmt,
            table.schema_arc(),
            scope_name,
            ctx.params().len(),
            epoch,
        )?
        else {
            return Ok(None);
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
        Ok(Some(prepared))
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
        let joined = conflict
            .map(|sites| {
                let width = sites.schema.columns.len();
                let excluded = SmartString::from("excluded");
                // `excluded` names the excluded row, also when the target bears
                // that name; the target then has a name no qualifier can spell
                let target = if *sites.table == excluded {
                    SmartString::from(format!("{RESOLVED}{excluded}"))
                } else {
                    sites.table.clone()
                };
                ScopeTable::joined(
                    [(target, sites.schema), (excluded, sites.schema)],
                    2 * width,
                )
                .ok_or_else(|| Error::internal("an ON CONFLICT row that does not line up"))
            })
            .transpose()?;
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
                    .any(|sq| sq.table.as_ref() == Some(sites.table));
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
                    values.extend(sites.schema.columns.iter().map(|column| {
                        let token =
                            Token::new(TokenType::Identifier, &column.name, Position::default());
                        Expression::Identifier(Identifier::new(token, column.name.as_str()))
                    }));
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
                    .any(|sq| sq.table.as_deref() == Some(sites.table));
                values.push(rewritten_value);
            }
            returning_parts = Some((values, reads_target));
        }
        let mut read_as = FxHashMap::default();
        if let Some(parts) = &where_parts {
            compare_as_scan(&parts.clause, schema, &mut read_as);
        }
        let own = CompileContext::new(&scope.columns, &executor.function_registry)
            .with_prepared_subqueries()
            .with_read_as(&read_as);
        let where_program = where_parts
            .as_ref()
            .map(|parts| compile_with(&own, &parts.clause))
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
            (Some(sites), Some(joined), Some((sets, where_, reads_target))) => {
                let over_joined = CompileContext::new(&joined.columns, &executor.function_registry)
                    .with_prepared_subqueries();
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
            (Some(sites), Some(scope), Some((values, reads_target))) => {
                let over_returned =
                    CompileContext::new(&scope.columns, &executor.function_registry)
                        .with_prepared_subqueries();
                Some(PreparedReturning {
                    columns: sites.schema.column_names_arc(),
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
            .map(|(parts, program)| parts.prepare(program, &blocks));
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
    /// The same conditions as their side's scan reads them
    side_wheres: [Option<PreparedWhere>; 2],
    /// The columns of each side the clauses were prepared over
    side_columns: [Vec<SchemaColumn>; 2],
    /// The schema each side was found to hold those columns with; held, any
    /// change to the table's schema replaces it, so the same one skips the check
    fitted: [std::sync::OnceLock<CompactArc<Schema>>; 2],
    /// Each SELECT item and ORDER BY key holding a subquery, over the joined row
    pub(crate) select: Vec<Option<Expression>>,
    pub(crate) order_by: Vec<Option<Expression>>,
    /// The uncorrelated member blocks those read, read before the join
    /// reads a row; a value block they read is a run slot
    ahead: Vec<u16>,
    /// The scope's names for the left side's columns, then the right side's
    pub(crate) left: Vec<String>,
    pub(crate) right: Vec<String>,
    pub(crate) all: CompactArc<Vec<String>>,
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

    /// `on`, `cross`, the conditions on one side, `side_filters`, and the
    /// SELECT items and ORDER BY keys holding a subquery, `output`, over the
    /// joined row of `sides`, each a name and its table's schema; None when
    /// the join's `width` columns do not line up
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        executor: &Executor,
        on: Option<&Expression>,
        cross: Option<&Expression>,
        side_filters: [Option<&Expression>; 2],
        output: [&[Option<&Expression>]; 2],
        sides: [(SmartString, &Schema); 2],
        width: usize,
        param_count: usize,
    ) -> Result<Option<Self>> {
        let left_width = sides[0].1.columns.len();
        let Some(scope) = ScopeTable::joined(sides.clone(), width) else {
            return Ok(None);
        };
        let scopes = [&scope];
        let chain = Chain { scopes: &scopes };
        let mut builder = Builder {
            executor,
            next_slot: param_count,
            subqueries: Vec::new(),
            max_param: 0,
            keeps_holes: false,
        };
        let mut rewrite = |clause: Option<&Expression>| {
            clause
                .map(|clause| builder.rewrite(&chain, None, clause, &mut Rewritten::default()))
                .transpose()
        };
        let on = rewrite(on)?;
        let cross = rewrite(cross)?;
        // An uncorrelated value the output reads is a run slot, read once
        // before the join reads a row
        let mut run_slots = Vec::new();
        let first_output_block = builder.subqueries.len();
        let [select, order_by] = output.map(|exprs| {
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
        let (select, order_by) = (select?, order_by?);
        let output_blocks = first_output_block..builder.subqueries.len();
        // Each side's conditions as its own scan reads them, a value a column
        // is compared with a run slot; the joined row reads the same blocks
        let registry = &executor.function_registry;
        let mut side_parts = [None, None];
        for (i, filter) in side_filters.into_iter().enumerate() {
            let Some(filter) = filter else {
                continue;
            };
            let (name, schema) = (&sides[i].0, sides[i].1);
            let side = ScopeTable::of(name.clone(), schema);
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
            compare_as_scan(&parts.clause, schema, &mut read_as);
            let own = CompileContext::new(&side.columns, registry)
                .with_prepared_subqueries()
                .with_read_as(&read_as);
            let program = compile_with(&own, &parts.clause)?;
            let offset = if i == 0 { 0 } else { left_width };
            let joined_clause = in_joined(&parts.clause, &side, &scope, offset)?;
            side_parts[i] = Some((parts, program, joined_clause));
        }
        let values = CompileContext::new(&scope.columns, registry).with_prepared_subqueries();
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile_with(&values, value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let (blocks, joined) = builder.finish(param_count, run_slots)?;
        let ahead = output_blocks
            .filter(|&id| {
                let sq = &blocks.subqueries[id];
                sq.uncorrelated && sq.parent.is_none() && matches!(sq.answer, Answer::Members(..))
            })
            .map(|id| u16::try_from(id).map_err(|_| Error::internal("too many subqueries")))
            .collect::<Result<Vec<_>>>()?;
        let mut side_filters = [None, None];
        let mut side_wheres = [None, None];
        let side_columns = [sides[0].1.columns.clone(), sides[1].1.columns.clone()];
        for (i, part) in side_parts.into_iter().enumerate() {
            if let Some((parts, program, joined_clause)) = part {
                side_wheres[i] = Some(parts.prepare(program, &blocks));
                side_filters[i] = Some(joined_clause);
            }
        }
        let left = scope.columns[..left_width].to_vec();
        let right = scope.columns[left_width..].to_vec();
        let pair_ctx = CompileContext::new(&left, registry).with_second_row(&right);
        let position = |hole: &Expression| match hole {
            Expression::Identifier(id) => scope
                .columns
                .iter()
                .position(|c| *c == id.value_lower)
                .map(|i| match i.checked_sub(left_width) {
                    Some(i) => Hole::Column2(i),
                    None => Hole::Column(i),
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
            side_wheres,
            side_columns,
            fitted: Default::default(),
            select,
            order_by,
            ahead,
            left,
            right,
            all: CompactArc::clone(&scope.columns),
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
            true => CompileContext::new(&self.left, registry).with_second_row(&self.right),
            false => CompileContext::new(&self.all, registry),
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
        ctx: Option<&ExecutionContext>,
    ) -> std::result::Result<(), OutOfFamily> {
        let family = Family {
            executor: self,
            ctx,
            tables: RefCell::new(SmallVec::new()),
            statement_table: std::cell::Cell::new(None),
        };
        let mut found = false;
        family.expression(clause, 1, Site::Subqueries, &mut found)?;
        family.catalog()
    }

    /// A join statement's `on`, `cross` and `output`, over the joined rows of
    /// `sides` laid out as `width` columns, with their subqueries prepared
    /// as blocks
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_join_blocks(
        &self,
        on: Option<&Expression>,
        cross: Option<&Expression>,
        side_filters: [Option<&Expression>; 2],
        output: [&[Option<&Expression>]; 2],
        sides: [(SmartString, &Schema); 2],
        width: usize,
        ctx: &ExecutionContext,
    ) -> Result<Option<JoinBlocks>> {
        JoinBlocks::prepare(
            self,
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
                Some((filter, true, _)) => Some(filter),
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
        let mut rows = RowVec::new();
        if limit != Some(0) {
            let mut skipped = 0;
            let mut failed = None;
            let mut keep = |id: i64, _: u64, row: Row| {
                match filter.as_ref().map(|filter| filter.matches_checked(&row)) {
                    Some(Ok(false)) => return true,
                    Some(Err(err)) => {
                        failed = Some(err);
                        return false;
                    }
                    Some(Ok(true)) | None => {}
                }
                if skipped < offset {
                    skipped += 1;
                    return true;
                }
                rows.push((id, row));
                limit.is_none_or(|limit| rows.len() < limit)
            };
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
                    rows.extend(answers.into_iter().skip(offset));
                }
            }
            if let Some(err) = failed {
                return Err(err);
            }
        }
        Ok(Some((
            Box::new(ExecutorResult::with_arc_columns(columns, rows)),
            token,
        )))
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
            Some(where_) => vm.execute_bool_checked(&where_.program, exec),
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
        for part in parts {
            let part = self.statement_conjunct(chain, part, work, run_slots)?;
            if !mentions_subquery(&part) && !needs_folding(&part) && scans_whole(&part, schema) {
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
                    (Expression::ScalarSubquery(mark), Some(name)) => schema
                        .column_index_map()
                        .get(name)
                        .map(|&column| (&mark.subquery, in_expr.not, Some(column))),
                    _ => None,
                },
                Expression::AllAny(all_any) => bound_column(&all_any.left)
                    .and_then(|name| schema.column_index_map().get(name).copied())
                    .map(|column| (&all_any.subquery, false, Some(column))),
                _ => None,
            };
            if let Some((mark, negated, column)) = set_test {
                set_tests.extend(mark.prepared.map(|id| (id, negated, column)));
            }
            rewritten.push(part);
        }
        let conjuncts = rewritten.len();
        let clause = conjunction(rewritten)
            .ok_or_else(|| Error::internal("a prepared statement without WHERE"))?;
        Ok(WhereParts {
            clause,
            pushed,
            set_tests,
            conjuncts,
        })
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
        let prepared = Arc::new(PreparedSelect::prepare(
            self,
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
            ctx: Some(ctx),
            tables: RefCell::new(SmallVec::new()),
            statement_table: std::cell::Cell::new(None),
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
        let unnamed = SmartString::new("");
        // A view's or a derived table's rows are read once, under the
        // statement's snapshot, before the plan is taken
        let (source, scope_name) = match stmt.table_expr.as_deref() {
            Some(Expression::TableSource(ts)) => {
                let name = &ts.name.value_lower;
                // A view comes before a table of its name, as the general path reads it
                let source = match self.engine.get_view_lowercase(name)? {
                    Some(view) => {
                        let result = self.execute_view_body(&view, &inner_ctx)?;
                        let columns = result.columns().to_vec();
                        StatementSource::Rows(columns, Self::materialize_result(result)?)
                    }
                    None => StatementSource::Table(self.prepared_table(name, ctx, None)?),
                };
                (
                    source,
                    ts.alias.as_ref().map_or(name, |alias| &alias.value_lower),
                )
            }
            Some(Expression::SubquerySource(ss)) => {
                let result = self.execute_select(&ss.subquery, &inner_ctx)?;
                let columns = result.columns().to_vec();
                let rows = Self::materialize_result(result)?;
                (
                    StatementSource::Rows(columns, rows),
                    ss.alias
                        .as_ref()
                        .map_or(&unnamed, |alias| &alias.value_lower),
                )
            }
            _ => return Err(Error::internal("a prepared statement without one table")),
        };
        let table_columns;
        let columns: &[String] = match &source {
            StatementSource::Table(table) => {
                table_columns = table.schema().column_names_arc();
                &table_columns
            }
            StatementSource::Rows(columns, _) => columns,
        };
        let schema = || match &source {
            StatementSource::Table(table) => std::borrow::Cow::Borrowed(table.schema()),
            StatementSource::Rows(columns, _) => {
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
        let output_columns = CompactArc::clone(&prepared.output_columns);
        let target = limit
            .filter(|_| stmt.order_by.is_empty())
            .map(|limit| limit.saturating_add(offset));
        // Made on the first row: the scan's own answers need none
        let mut kept: Option<RowVec> = None;
        let mut keep = |_: i64, scanned: u64, row: Row| {
            let rows = kept.get_or_insert_with(|| RowVec::with_capacity(64));
            rows.push((scanned as i64, row));
            target.is_none_or(|target| rows.len() < target)
        };
        let read = match source {
            StatementSource::Table(table) => runner.passing_rows(
                &prepared.where_,
                None,
                &*table,
                ctx,
                target,
                false,
                &mut keep,
            ),
            StatementSource::Rows(_, rows) => {
                runner.rows_passing(&prepared.where_, rows, ctx, target, &mut keep)
            }
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
                ));
            }
            Passing::Answers(answers) => answers,
            Passing::Read => kept.unwrap_or_else(|| RowVec::with_capacity(64)),
            Passing::Scan(_) | Passing::Filtered(..) => {
                return Err(Error::internal("a prepared statement streamed its scan"))
            }
        };
        let limit_applied = target.is_some();
        if limit_applied {
            let keep = limit.unwrap_or(usize::MAX);
            rows = rows.into_iter().skip(offset).take(keep).collect();
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_projection_ran();
        let rows = prepared.projection.project(rows, ctx)?;
        let result = ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), rows);
        Ok((
            Box::new(result),
            output_columns,
            limit_applied,
            prepared.visible,
            prepared.layout.clone(),
        ))
    }
}

/// What a prepared statement reads its rows from
enum StatementSource {
    Table(Box<dyn Table>),
    /// A derived table's columns and rows, read once
    Rows(Vec<String>, RowVec),
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

/// The rows a primary key equality read by key, in place of a scan
struct PointRows {
    rows: RowVec,
    at: Option<usize>,
}

impl Scanner for PointRows {
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

/// Each subquery's table by its id; None for a block without FROM, or one
/// run as a statement of its own
type BlockTables = SmallVec<[Option<Box<dyn Table>>; 2]>;

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
}

/// What one run keeps for one subquery
struct BlockState {
    buffer: RefCell<ParamVec>,
    vm: RefCell<ExprVM>,
    /// Its pushed comparisons, given each call's values in place
    bound: RefCell<Option<BoundPushdown>>,
    /// An uncorrelated block's value, once computed
    value: RefCell<Option<Value>>,
    /// A keyed EXISTS or IN block's choice, once made
    keyed_choice: std::cell::Cell<Option<bool>>,
    keyed_set: RefCell<Option<CompactArc<ExistsKeys>>>,
    /// A value block's members: an uncorrelated block's, or a keyed IN
    /// block's grouped by key, set once per run
    members: std::cell::OnceCell<BlockAnswer>,
    /// The error a read of the members or of the keyed set raised in this
    /// run, which every row that needs them raises without reading again
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
}

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
            tables.push(Some(table));
        }
        Ok(Some(tables))
    }

    /// Sets the rows a join's site asks its blocks for, before its first ask
    pub(crate) fn ask_rows(&self, rows: Option<usize>) {
        self.statement_rows.set(rows);
    }

    /// The rows block `index` reads, through `expr` when its scan takes one:
    /// its table's, or the one row without columns of a block without FROM
    fn scan(&self, index: usize, expr: Option<&dyn StorageExpr>) -> Result<Box<dyn Scanner>> {
        match &self.tables[index] {
            Some(table) => table.scan(&self.prepared.subqueries[index].column_indices, expr),
            None => Ok(Box::new(OneRow {
                row: Row::new(),
                read: false,
            })),
        }
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
        if let (Some(point), Some(expr), Some(table)) = (sq.point, expr, &self.tables[index]) {
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
                    keyed_choice: std::cell::Cell::new(None),
                    keyed_set: RefCell::new(None),
                    members: std::cell::OnceCell::new(),
                    read_error: RefCell::new(None),
                    failed: RefCell::new(SmallVec::new()),
                    grouped: RefCell::new(None),
                    feed_vm: RefCell::new(None),
                    point_rows: RefCell::new(RowVec::new()),
                })
                .collect(),
            base,
            transaction_id: ctx.transaction_id(),
            ctx: inner_ctx,
            tables,
        }
    }

    /// Read the members of each uncorrelated block of `ids` now, which keeps
    /// them for every row that asks later
    fn read_ahead(&self, ids: &[u16], holes: &HoleSet) -> Result<()> {
        let empty = Row::new();
        let caller = ExecuteContext::with_common_params(&empty, &self.base, self.transaction_id);
        for &id in ids {
            self.members_answer(usize::from(id), &Value::null_unknown(), &caller, holes)?;
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
        let (mut storage_expr, residual) = match pushdown {
            Some(pushdown) if self.prepared.run_slots.is_empty() => {
                pushdown::try_pushdown(pushdown, table.schema(), Some(ctx))
            }
            Some(pushdown) => {
                let mut run_ctx = ctx.clone();
                run_ctx.set_params(self.base.clone());
                pushdown::try_pushdown(pushdown, table.schema(), Some(&run_ctx))
            }
            None => (None, true),
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
        Ok(Some((storage_expr, scanned_whole, extra_whole)))
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
        let Answer::Members(kept, members) = &self.prepared.subqueries[index].answer else {
            return Ok(SetScan::Program);
        };
        // An error reading the members, or one a member raised, belongs to
        // the row that needs the answer: the program asks the block then
        let empty = Row::new();
        let read = self.read_members(
            index,
            kept,
            members,
            &ExecuteContext::new(&empty),
            self.holes,
        );
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
            MemberSet::In(set_of) => {
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
                        column(),
                        op,
                        all,
                        values,
                        &scan,
                    )),
                }
            }
        })
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
        let Some((mut storage_expr, mut scanned_whole, mut extra_whole)) =
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
                Some((candidates, passed)) => {
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::prepared_rows_narrowed_by_keys(
                        where_.narrowing.is_some_and(|n| n.negated),
                    );
                    (Some(candidates.into_iter()), None, true, passed)
                }
                None => {
                    if let Some(slot) = where_.narrowing_set {
                        let Some(scan) = self.where_scan(where_, extra, Some(slot), table, ctx)?
                        else {
                            return Ok(Passing::Pruned);
                        };
                        (storage_expr, scanned_whole, extra_whole) = scan;
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
                                if let Some(bound) = self.bound_program(&where_.program) {
                                    return Ok(Passing::Filtered(scan, bound));
                                }
                            }
                            (None, Some(scan), false, scanned_whole)
                        }
                    }
                }
            };
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
            if ((passed && taken) || self.matches(&mut vm, &where_.program, &row)?)
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
    /// rows read so far, until `keep` says stop
    fn rows_passing(
        &self,
        where_: &PreparedWhere,
        rows: RowVec,
        ctx: &ExecutionContext,
        target: Option<usize>,
        mut keep: impl FnMut(i64, u64, Row) -> bool,
    ) -> Result<Passing> {
        if target == Some(0) {
            return Ok(Passing::Read);
        }
        let mut vm = ExprVM::new();
        for (scanned, (id, row)) in (1u64..).zip(rows) {
            if scanned.is_multiple_of(100) {
                ctx.check_cancelled()?;
            }
            if self.matches(&mut vm, &where_.program, &row)? && !keep(id, scanned, row) {
                break;
            }
        }
        ctx.check_cancelled()?;
        Ok(Passing::Read)
    }

    /// The statement's rows its narrowing EXISTS may pass, read by the keys
    /// of the block's keyed set when the run takes it, and whether each
    /// passes the WHERE without asking it: the EXISTS is the whole WHERE and
    /// no key holds an error; None to scan
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
        let Some(plan) = &existence.keyed else {
            return Ok(None);
        };
        if !self.takes_keyed_set(index, plan) {
            return Ok(None);
        }
        // Over no row: the holes besides the keys read none
        {
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
        Ok(rows.map(|rows| (rows, passed)))
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
        // that aggregates reads every candidate, so it raises its own
        if sq.grouping.is_none() && !matches!(sq.answer, Answer::Value(..)) {
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

    /// The general path's rule for its index nested loop, with the parent
    /// block's rows as the outer rows; only the statement's LIMIT counts
    fn probe_cheaper(&self, index: usize, plan: &KeyedPlan) -> bool {
        let sq = &self.prepared.subqueries[index];
        let parent_rows = match sq.parent {
            // A parent without FROM reads one row
            Some(parent) => Some(
                self.tables[parent as usize]
                    .as_ref()
                    .map_or(1, |table| table.row_count_hint()),
            ),
            None => self.statement_rows.get(),
        };
        // A keyed plan reads a column, so its block has a table
        let Some(table) = &self.tables[index] else {
            return true;
        };
        Executor::probe_cheaper(
            &**table,
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
        let keeps = Executor::reads_cacheable_rows(statement)
            && !Executor::expression_varies_by_statement(&existence.identity, statement);
        let stamp = if keeps {
            self.executor
                .subquery_cache_stamp(&existence.tables, statement.statement_view())
        } else {
            None
        };
        let origin = BlockOrigin {
            block: &existence.identity,
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
        self.members_answer(subquery_index as usize, value, caller, self.holes)
    }

    fn execute_quantified(
        &self,
        subquery_index: u16,
        value: &Value,
        _op: CompareOp,
        _all: bool,
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        self.members_answer(subquery_index as usize, value, caller, self.holes)
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
        self.runner
            .members_answer(subquery_index as usize, value, caller, self.holes)
    }

    fn execute_quantified(
        &self,
        subquery_index: u16,
        value: &Value,
        _op: CompareOp,
        _all: bool,
        caller: &ExecuteContext,
    ) -> Result<Option<bool>> {
        self.runner
            .members_answer(subquery_index as usize, value, caller, self.holes)
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
        self.ctx.check_cancelled()?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_scalar_next();
        let Answer::Value(projection, kept) = &sq.answer else {
            return Err(Error::internal("an EXISTS block run for a value"));
        };
        if !sq.uncorrelated {
            return self.scalar_value(index, sq, projection, caller, site);
        }
        let answer = self.kept(kept, || {
            self.scalar_value(index, sq, projection, caller, site)
                .map(BlockAnswer::Value)
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
            let mut values = self.statement_values(statement)?;
            if values.len() > 1 {
                return Err(more_than_one_row());
            }
            return Ok(values.pop().unwrap_or_else(Value::null_unknown));
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
        let table = self.tables[index].as_deref()?;
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

    fn exists_answer(&self, index: usize, caller: &ExecuteContext, site: &HoleSet) -> Result<bool> {
        let sq = self
            .prepared
            .subqueries
            .get(index)
            .ok_or_else(|| Error::internal("unknown prepared subquery"))?;
        let Answer::Exists(existence) = &sq.answer else {
            return Err(Error::internal("a scalar block run for EXISTS"));
        };
        if sq.uncorrelated {
            if let Some(Value::Boolean(exists)) = self.blocks[index].value.borrow().as_ref() {
                return Ok(*exists);
            }
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
                        .statement_values(statement)
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
                match self.kept(existence, || once().map(BlockAnswer::Once))? {
                    BlockAnswer::Once(exists) => exists,
                    _ => return Err(Error::internal("a set kept for an EXISTS value")),
                }
            }
            (None, false) => {
                self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
                match &existence.keyed {
                    Some(plan) if self.takes_keyed_set(index, plan) => {
                        self.keyed_lookup(index, existence, plan)?
                    }
                    _ => self.probe(index)?,
                }
            }
        };
        if sq.uncorrelated {
            *self.blocks[index].value.borrow_mut() = Some(Value::Boolean(exists));
        }
        Ok(exists)
    }
}

fn more_than_one_row() -> Error {
    Error::Internal {
        message: "scalar subquery returned more than one row".to_string(),
    }
}

/// `values` as `test` asks them, NULL members included
fn member_set(test: MemberTest, values: Vec<Value>) -> MemberSet {
    match test {
        MemberTest::In => MemberSet::In(PredicateSet::of(CompactArc::new(
            values.into_iter().collect(),
        ))),
        MemberTest::Quantified(op, all) => {
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
    /// A value block's answer to `value` for the row it runs for
    fn members_answer(
        &self,
        index: usize,
        value: &Value,
        caller: &ExecuteContext,
        site: &HoleSet,
    ) -> Result<Option<bool>> {
        let sq = self
            .prepared
            .subqueries
            .get(index)
            .ok_or_else(|| Error::internal("unknown prepared subquery"))?;
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
            return set.answer(value);
        }
        if let Some(grouping) = &sq.grouping {
            return self.with_groups(index, grouping, caller, site, |rows, params, vm| {
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
                    if let Some(answer) = fold.member(value, self.member(members, vm, &exec))? {
                        return Ok(Some(answer));
                    }
                }
                fold.finish()
            });
        }
        self.fill(index, &mut self.blocks[index].vm.borrow_mut(), caller, site)?;
        match &kept.keyed {
            Some(plan) if self.takes_keyed_set(index, plan) => {
                self.keyed_members_answer(index, kept, members, plan, value)
            }
            _ => self.probe_members(index, members, value),
        }
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
        if !program.has_subqueries() {
            return Some(CompactArc::clone(program));
        }
        let empty = Row::new();
        let caller = ExecuteContext::new(&empty);
        let mut ops = Vec::with_capacity(program.len());
        for op in program.ops() {
            let index = match op {
                Op::ExecInSubquery(index) | Op::ExecQuantified(index, ..) => *index as usize,
                Op::ExecScalarSubquery(_) | Op::ExecExists(_) => return None,
                op => {
                    ops.push(op.clone());
                    continue;
                }
            };
            let sq = self.prepared.subqueries.get(index)?;
            let Answer::Members(kept, members) = &sq.answer else {
                return None;
            };
            if !sq.uncorrelated {
                return None;
            }
            self.read_members(index, kept, members, &caller, self.holes)
                .ok()?;
            let Some(BlockAnswer::Members(set)) = self.blocks[index].members.get() else {
                return None;
            };
            if set.failed.is_some() {
                return None;
            }
            ops.push(match &set.set {
                MemberSet::In(set) => Op::InSet(CompactArc::new(set.clone())),
                MemberSet::Quantified(set) => Op::QuantifiedSet(CompactArc::new(set.clone())),
            });
        }
        Some(CompactArc::new(super::expression::Program::new(ops)))
    }

    /// `read` of block `index`'s set, or the error a read of it raised
    /// before in this run, which every row that needs the set raises
    fn read_once<T>(&self, index: usize, read: impl FnOnce() -> Result<T>) -> Result<T> {
        if let Some(e) = self.blocks[index].read_error.borrow().as_ref() {
            return Err(e.clone());
        }
        read().inspect_err(|e| *self.blocks[index].read_error.borrow_mut() = Some(e.clone()))
    }

    /// The member a candidate's SELECT expression gives, read as the tested
    /// value's column reads it
    fn member(&self, members: &Members, vm: &mut ExprVM, exec: &ExecuteContext) -> Result<Value> {
        let value = vm.execute(&members.projection, exec)?;
        Ok(match members.read_as {
            Some(column_type) => read_as_column(column_type, value),
            None => value,
        })
    }

    /// `value` tested by the probe against the buffer filled: a member that
    /// settles the test ends it; else the first error a candidate raised;
    /// else what the members say, the test of no members included
    fn probe_members(
        &self,
        index: usize,
        members: &Members,
        value: &Value,
    ) -> Result<Option<bool>> {
        self.ctx.check_cancelled()?;
        #[cfg(feature = "test-failpoints")]
        {
            crate::test_failpoints::prepared_exists_probed();
            crate::test_failpoints::in_subquery_ran();
        }
        let sq = &self.prepared.subqueries[index];
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
            if let Some(answer) = fold.member(value, self.member(members, &mut vm, &exec))? {
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
        if let Some(grouping) = &self.prepared.subqueries[index].grouping {
            if let Some(statement) = &grouping.statement {
                let values = self.statement_values(statement)?;
                let values = match members.read_as {
                    Some(column_type) => values
                        .into_iter()
                        .map(|value| read_as_column(column_type, value))
                        .collect(),
                    None => values,
                };
                return Ok(KeptMembers {
                    set: member_set(members.test, values),
                    failed: None,
                });
            }
            let rows = self.group_rows(index, grouping)?;
            let mut vm = self.blocks[index].vm.borrow_mut();
            let params = self.blocks[index].buffer.borrow();
            let (mut values, mut failed) = (Vec::with_capacity(rows.len()), None);
            for slot_row in &rows {
                let exec = self.slot_context(slot_row, &params);
                let member = match self.group_passes(grouping, &mut vm, &exec) {
                    Ok(true) => self.member(members, &mut vm, &exec),
                    Ok(false) => continue,
                    Err(e) => Err(e),
                };
                match member {
                    Ok(member) => values.push(member),
                    Err(e) => {
                        failed.get_or_insert(deferred(e)?);
                    }
                }
            }
            return Ok(KeptMembers {
                set: member_set(members.test, values),
                failed,
            });
        }
        let mut grouped = self.group_members(index, members, plan)?;
        Ok(grouped.remove(&[][..]).unwrap_or_else(|| KeptMembers {
            set: member_set(members.test, Vec::new()),
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
        let decided = sq.filter.is_none() || (bound.is_some() && pushed_whole);
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
            let passed = if decided {
                Ok(true)
            } else {
                self.passes_rest(sq, &mut vm, &exec, &slots_failed)
            };
            let member = match passed {
                Ok(true) => self.member(members, &mut vm, &exec),
                Ok(false) => continue,
                Err(e) => Err(e),
            };
            let group = groups.entry(key).or_default();
            match member {
                Ok(member) => group.0.push(member),
                Err(e) => {
                    group.1.get_or_insert(deferred(e)?);
                }
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
                        set: member_set(members.test, values),
                        failed,
                    },
                )
            })
            .collect())
    }

    /// The statement's rows an uncorrelated `column IN (b)` or `column NOT
    /// IN (b)` in its top-level AND may pass, read by the block's members,
    /// and whether each passes without the WHERE. The members are read
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
        let sq = &self.prepared.subqueries[index];
        let Answer::Members(kept, members) = &sq.answer else {
            return Ok(None);
        };
        // An error reading the members belongs to the row that needs them
        let empty = Row::new();
        let read = self.read_members(
            index,
            kept,
            members,
            &ExecuteContext::new(&empty),
            self.holes,
        );
        if read.is_err() {
            return Ok(None);
        }
        let Some(BlockAnswer::Members(set)) = self.blocks[index].members.get() else {
            return Ok(None);
        };
        let MemberSet::In(set_of) = &set.set else {
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
            return Ok(rows.map(|rows| (rows, whole)));
        }
        let rows = Executor::rows_of_members(
            table,
            name,
            values,
            target.is_some(),
            target.filter(|_| whole),
        )?;
        Ok(rows.map(|rows| (rows, whole)))
    }

    /// A keyed IN block's answer to `value` for the buffer filled: no
    /// candidates for a NULL key or a key without rows
    fn keyed_members_answer(
        &self,
        index: usize,
        kept: &Kept,
        members: &Members,
        plan: &KeyedPlan,
        value: &Value,
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
            Some(group) => group.answer(value),
            None => no_candidates,
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
        let (op, decides) = match test {
            MemberTest::In => (CompareOp::Eq, true),
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
    fn statement_values(&self, run: &StatementRun) -> Result<Vec<Value>> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_strategy_ran(
            crate::test_failpoints::PreparedStrategy::Statement,
        );
        let mut result = self.executor.execute_select_with_plan(
            &run.statement,
            &self.ctx,
            Some(&run.classification),
            Some(&run.compiled),
            None,
        )?;
        let mut values = Vec::new();
        while result.next() {
            let mut row = result.take_row().into_values();
            values.push(if row.is_empty() {
                Value::null_unknown()
            } else {
                row.swap_remove(0)
            });
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
            if self.takes_keyed_set(index, plan) {
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
