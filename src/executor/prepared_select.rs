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

use std::cell::RefCell;
use std::sync::{Arc, RwLock};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::api::params::ParamVec;
use crate::common::{CompactArc, SmartString};
use crate::core::{DataType, Error, Operator, Result, Row, RowVec, Schema, Value};
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::expression::{AndExpr, ComparisonExpr, Expression as StorageExpr};
use crate::storage::traits::{Engine, Table};

use super::context::ExecutionContext;
use super::expression::{
    CompileContext, CompileError, ExecuteContext, ExprCompiler, ExprVM, SharedProgram,
    SubqueryExecutor,
};
use super::pushdown::{self, coerce_constant, contains_non_deterministic_volatile};
use super::query_cache::CompiledExecution;
use super::result::ExecutorResult;
use super::utils::{
    extract_literal_value, flip_operator, infix_to_operator, is_aggregate_function,
};
use super::Executor;

/// Scopes one chain may hold: a block's own table and every enclosing one
const MAX_SCOPES: usize = 32;

/// Why a statement runs on the general path instead of the prepared one
pub(crate) type OutOfFamily = &'static str;

/// The rows, their columns, and whether LIMIT and OFFSET were applied
type PreparedOutput = (
    Box<dyn crate::storage::traits::QueryResult>,
    CompactArc<Vec<String>>,
    bool,
);

/// The classifier's walk over one statement
struct Family<'a> {
    executor: &'a Executor,
    ctx: Option<&'a ExecutionContext>,
    /// Every FROM met, checked against the catalog once the syntax passed
    tables: RefCell<SmallVec<[&'a SimpleTableSource; 4]>>,
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

    /// Every FROM met names a base table, not a CTE or a view
    fn catalog(&self) -> std::result::Result<(), OutOfFamily> {
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

    /// `expr` holds only the family's expression forms; `subqueries` says
    /// whether scalar subqueries may appear, `found` records one that did
    fn expression(
        &self,
        expr: &'a Expression,
        scopes: usize,
        subqueries: bool,
        found: &mut bool,
    ) -> std::result::Result<(), OutOfFamily> {
        let mut walk = |e: &'a Expression| self.expression(e, scopes, subqueries, found);
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
                    Expression::ScalarSubquery(_) => return Err("IN (subquery)"),
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
                    return Err("an aggregate");
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
            Expression::Exists(_) => Err("EXISTS"),
            Expression::AllAny(_) => Err("ANY or ALL"),
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
        if !plain_block(sq) || !sq.order_by.is_empty() || sq.limit.is_some() || sq.offset.is_some()
        {
            return Err("a subquery block with a clause outside the family");
        }
        self.from(sq.table_expr.as_deref())?;
        let [column] = sq.columns.as_slice() else {
            return Err("a subquery block without exactly one SELECT expression");
        };
        let column = unaliased(column);
        if matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
            return Err("a subquery block selecting *");
        }
        let mut found = false;
        self.expression(column, scopes, true, &mut found)?;
        match sq.where_clause.as_deref() {
            Some(where_clause) => self.expression(where_clause, scopes, true, &mut found),
            None => Ok(()),
        }
    }
}

/// No WITH, set operation, DISTINCT, GROUP BY, HAVING or WINDOW
fn plain_block(stmt: &SelectStatement) -> bool {
    stmt.with.is_none()
        && stmt.set_operations.is_empty()
        && !stmt.distinct
        && stmt.distinct_on.is_empty()
        && stmt.group_by.columns.is_empty()
        && matches!(stmt.group_by.modifier, GroupByModifier::None)
        && stmt.having.is_none()
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
        };
        let mut found = false;
        if let Some(where_clause) = stmt.where_clause.as_deref() {
            family.expression(where_clause, 1, true, &mut found)?;
        }
        if !found {
            return Err("WHERE holds no scalar subquery");
        }
        for column in &stmt.columns {
            if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
                family.expression(unaliased(column), 1, false, &mut found)?;
            }
        }
        for ob in &stmt.order_by {
            family.expression(&ob.expression, 1, false, &mut found)?;
        }
        for clause in [&stmt.limit, &stmt.offset].into_iter().flatten() {
            if !matches!(
                **clause,
                Expression::IntegerLiteral(_) | Expression::Parameter(_)
            ) {
                return Err("LIMIT or OFFSET that is not a literal or a parameter");
            }
        }
        family.from(stmt.table_expr.as_deref())?;
        family.catalog()
    }
}

/// One scope of a chain: the name that qualifies it and its columns
struct ScopeTable {
    name: SmartString,
    columns: CompactArc<Vec<String>>,
    lower: CompactArc<Vec<String>>,
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
        if !id.quoted && matches!(id.value_lower.as_str(), "true" | "false") {
            return Ok(None);
        }
        for (depth, scope) in self.scopes.iter().enumerate() {
            if let Some(index) = scope.lower.iter().position(|c| *c == id.value_lower) {
                return Ok(Some((depth, index)));
            }
        }
        if id.quoted {
            return Ok(None);
        }
        Err(Error::ColumnNotFound(id.value.to_string()))
    }

    /// `qualifier.column` in the nearest scope the qualifier names
    fn qualified(&self, qualifier: &str, column: &str, spelled: &str) -> Result<(usize, usize)> {
        let depth = self
            .scopes
            .iter()
            .position(|scope| scope.name == qualifier)
            .ok_or_else(|| {
                Error::invalid_argument(format!("unknown table or alias '{}'", qualifier))
            })?;
        let index = self.scopes[depth]
            .lower
            .iter()
            .position(|c| c == column)
            .ok_or_else(|| Error::ColumnNotFound(spelled.to_string()))?;
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
    /// programs compile against
    fn column(&self, depth: usize, index: usize) -> Expression {
        let name = &self.scopes[depth].columns[index];
        let mut id = Identifier::new(
            Token::new(TokenType::Identifier, name, Position::default()),
            name.as_str(),
        );
        id.value_lower = SmartString::from(self.scopes[depth].lower[index].as_str());
        id.quoted = true;
        Expression::Identifier(id)
    }

    /// What `expr` reads; a scalar subquery counts as reading this row
    fn reads(&self, expr: &Expression) -> Result<Reads> {
        match expr {
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
                Ok(self.resolve(expr)?.map_or(0, |(depth, _)| 1 << depth))
            }
            Expression::ScalarSubquery(_) => Ok(1),
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
        Expression::FunctionCall(call) => call.arguments.iter().try_for_each(f),
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
            order_by: call.order_by.clone(),
            filter: call.filter.clone(),
        })),
        other => other.clone(),
    })
}

/// Whether a call anywhere in `expr` draws a new value each time it runs,
/// through every operand the family's forms have
fn volatile(expr: &Expression) -> bool {
    if contains_non_deterministic_volatile(expr) {
        return true;
    }
    let mut found = false;
    let _ = for_each_child(expr, &mut |child| {
        found |= volatile(child);
        Ok(())
    });
    found
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

/// Whether a bare name outside any subquery of `expr` passes `matches`
fn names_any(expr: &Expression, matches: &impl Fn(&str) -> bool) -> bool {
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

/// One statement shape compiled against its tables' columns
#[derive(Debug)]
pub struct PreparedSelect {
    cached_epoch: u64,
    /// The statement table's columns at preparation
    columns: CompactArc<Vec<String>>,
    /// Positional parameters the statement takes; slots are numbered after them
    param_count: usize,
    /// The statement's parameters and every slot
    slot_count: usize,
    /// Named parameters read anywhere; each must have a value
    named: Vec<SmartString>,
    where_program: SharedProgram,
    /// Values the statement's conjuncts compare its columns with, each
    /// computed once per run before the scan
    run_slots: Vec<(usize, SharedProgram)>,
    /// The conjuncts without a subquery, which the scan is narrowed by
    pushdown: Option<Expression>,
    /// Indexed by subquery id, nested ones included
    subqueries: Vec<PreparedScalarSubquery>,
    projection: PreparedProjection,
    output_columns: CompactArc<Vec<String>>,
}

/// A scalar subquery over one table, prepared as its own scope: every
/// value it reads from an enclosing scope becomes a parameter slot, filled
/// per call by a hole program over the row it runs for
#[derive(Debug)]
struct PreparedScalarSubquery {
    /// The inner table's name as the catalog knows it
    table: SmartString,
    column_indices: Vec<usize>,
    /// The table's schema the programs read positions of
    schema: CompactArc<Schema>,
    /// The WHERE's conjuncts the scan can take
    pushdown: Vec<PreparedComparison>,
    /// Every conjunct is one the scan takes: a call that pushes them all
    /// has the scan's answer, as the general path does, and skips the filter
    pushed_whole: bool,
    filter: Option<SharedProgram>,
    projection: SharedProgram,
    /// The subquery whose row this one runs for; None for the statement's row
    parent: Option<u16>,
    /// Programs over that row, each filling one slot, including values a
    /// deeper subquery reads from it
    holes: Vec<(usize, SharedProgram)>,
    /// Reads no enclosing scope: its value is computed once per run
    uncorrelated: bool,
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
    /// A positional parameter or a slot, zero-based
    Slot(usize),
    Named(SmartString),
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
            let Expression::Identifier(column) = &**column else {
                continue;
            };
            let value = match value.as_ref() {
                Expression::Parameter(p) if p.name.starts_with(':') => {
                    PreparedValue::Named(SmartString::from(&p.name[1..]))
                }
                Expression::Parameter(p) if p.index > 0 => PreparedValue::Slot(p.index - 1),
                other => PreparedValue::Const(extract_literal_value(other)?),
            };
            let index = *schema.column_index_map().get(column.value_lower.as_str())?;
            return Some(Self {
                column: column.value.to_string(),
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
    fn value(&self, params: &[Value], named: &FxHashMap<String, Value>) -> Option<Value> {
        let value = match &self.value {
            PreparedValue::Const(value) => value.clone(),
            PreparedValue::Slot(slot) => params.get(*slot)?.clone(),
            PreparedValue::Named(name) => named.get(name.as_str())?.clone(),
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
}

/// The SELECT list, then the columns and expressions ORDER BY appends for the sort
#[derive(Debug)]
struct PreparedProjection {
    outputs: Vec<Output>,
}

/// What preparing one statement builds up
struct Builder<'e> {
    executor: &'e Executor,
    /// The next free parameter slot
    next_slot: usize,
    /// Indexed by subquery id; None while that subquery is being prepared
    subqueries: Vec<Option<PreparedScalarSubquery>>,
    named: Vec<SmartString>,
    /// The highest positional parameter the statement reads, one-based
    max_param: usize,
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
    /// Ids of the subquery nodes made while rewriting, by node address
    ids: FxHashMap<usize, u16>,
}

impl Builder<'_> {
    fn slot(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        slot
    }

    fn note_named(&mut self, expr: &Expression) {
        if let Expression::Parameter(p) = expr {
            match p.name.strip_prefix(':') {
                Some(name) => {
                    if !self.named.iter().any(|n| n == name) {
                        self.named.push(SmartString::from(name));
                    }
                }
                None => self.max_param = self.max_param.max(p.index),
            }
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
        self.note_named(expr);
        match expr {
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
                match chain.resolve(expr)? {
                    None => Ok(expr.clone()),
                    Some((0, index)) => {
                        work.reads_row = true;
                        Ok(chain.column(0, index))
                    }
                    Some((depth, index)) => {
                        let slot = self.slot();
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
                let (nested_id, delegated, reads_this_row) =
                    self.subquery(&nested.subquery, chain.scopes, id)?;
                work.reads_row |= reads_this_row;
                for (slot, value, depth) in delegated {
                    if depth == 2 {
                        work.reads_parent = true;
                        work.holes.push((slot, value));
                    } else {
                        work.delegated.push((slot, value, depth - 1));
                    }
                }
                // Compiled by its id only: the node is a mark, not a copy of the block
                let node = ScalarSubquery {
                    token: nested.token.clone(),
                    subquery: Box::new(SelectStatement {
                        token: nested.subquery.token.clone(),
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
                    }),
                };
                work.ids.insert(
                    &*node.subquery as *const SelectStatement as usize,
                    nested_id,
                );
                Ok(Expression::ScalarSubquery(node))
            }
            other => rebuild(other, &mut |child| self.rewrite(chain, id, child, work)),
        }
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
                    reads & 1 == 0 && reads.count_ones() <= 1 && !is_leaf(value) && !volatile(value)
                };
                let sides = [(&infix.left, left), (&infix.right, right)];
                if let Some(index) = sides.iter().position(|(value, reads)| hoist(value, *reads)) {
                    let (value, reads) = sides[index];
                    let other = sides[1 - index].0;
                    if matches!(chain.resolve(other)?, Some((0, _))) {
                        self.note_names(value);
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

    /// Record every named parameter `expr` reads
    fn note_names(&mut self, expr: &Expression) {
        self.note_named(expr);
        let _ = for_each_child(expr, &mut |child| {
            self.note_names(child);
            Ok(())
        });
    }

    /// Prepare `sq`, run once per row of the scope `outer[0]`. Returns its
    /// id, the values it reads from scopes beyond its parent, and whether
    /// it reads its parent's row at all
    fn subquery(
        &mut self,
        sq: &SelectStatement,
        outer: &[&ScopeTable],
        parent: Option<u16>,
    ) -> Result<(u16, Vec<Delegated>, bool)> {
        let executor = self.executor;
        let registry = &executor.function_registry;
        let Some(Expression::TableSource(ts)) = sq.table_expr.as_deref() else {
            return Err(Error::internal("a prepared subquery without one table"));
        };
        let schema = executor
            .engine
            .get_table_schema(&ts.name.value_lower)
            .map_err(|e| match e {
                Error::TableNotFound(_) => Error::TableOrViewNotFound(ts.name.value.to_string()),
                other => other,
            })?;
        let scope = ScopeTable {
            name: ts
                .alias
                .as_ref()
                .map_or(&ts.name.value_lower, |alias| &alias.value_lower)
                .clone(),
            columns: schema.column_names_arc(),
            lower: schema.column_names_lower_arc(),
        };
        let id = u16::try_from(self.subqueries.len())
            .map_err(|_| Error::internal("too many prepared subqueries"))?;
        self.subqueries.push(None);

        let mut scopes: Vec<&ScopeTable> = Vec::with_capacity(outer.len() + 1);
        scopes.push(&scope);
        scopes.extend_from_slice(outer);
        let chain = Chain { scopes: &scopes };
        let mut work = Rewritten::default();
        let select = unaliased(&sq.columns[0]);
        let projection = self.rewrite(&chain, Some(id), select, &mut work)?;
        let where_clause = sq
            .where_clause
            .as_deref()
            .map(|w| with_aliases(w, &sq.columns, &scope.columns));
        let mut pushdown = Vec::new();
        let mut pushed_whole = true;
        let filter = match &where_clause {
            Some(where_clause) => {
                let mut parts = Vec::new();
                conjuncts(where_clause, &mut parts);
                let mut rewritten = Vec::with_capacity(parts.len());
                for part in parts {
                    let part = self.rewrite_conjunct(&chain, id, part, &mut work)?;
                    match PreparedComparison::from_conjunct(&part, &schema) {
                        Some(comparison) => pushdown.push(comparison),
                        None => pushed_whole = false,
                    }
                    rewritten.push(part);
                }
                conjunction(rewritten)
            }
            None => None,
        };

        let own = CompileContext::new(&scope.columns, registry).with_subquery_ids(&work.ids);
        let projection = compile_with(&own, &projection)?;
        let filter = match &filter {
            Some(filter) => Some(compile_with(&own, filter)?),
            None => None,
        };
        let holes = if work.holes.is_empty() {
            Vec::new()
        } else {
            let parent = CompileContext::new(&outer[0].columns, registry);
            work.holes
                .iter()
                .map(|(slot, hole)| compile_with(&parent, hole).map(|p| (*slot, p)))
                .collect::<Result<Vec<_>>>()?
        };
        let correlated = work.reads_parent || !work.delegated.is_empty();
        self.subqueries[id as usize] = Some(PreparedScalarSubquery {
            table: ts.name.value_lower.clone(),
            column_indices: (0..scope.columns.len()).collect(),
            schema: CompactArc::clone(&schema),
            pushdown,
            pushed_whole,
            filter,
            projection,
            parent,
            holes,
            uncorrelated: !correlated,
        });
        Ok((id, work.delegated, work.reads_parent))
    }
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
        Expression::ScalarSubquery(_) => true,
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
    /// with the names of the appended columns
    fn prepare(
        executor: &Executor,
        stmt: &SelectStatement,
        chain: &Chain<'_>,
    ) -> Result<(Self, Vec<String>)> {
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
        for column in &stmt.columns {
            match column {
                Expression::Star(_) => {
                    select_exprs.extend((0..columns.len()).map(|i| chain.column(0, i)))
                }
                Expression::QualifiedStar(star) => {
                    if star.qualifier.to_lowercase() != scope.name.as_str() {
                        return Err(Error::invalid_argument(format!(
                            "unknown table or alias '{}'",
                            star.qualifier
                        )));
                    }
                    select_exprs.extend((0..columns.len()).map(|i| chain.column(0, i)))
                }
                other => {
                    check_names(chain, unaliased(other), &[])?;
                    select_exprs.push(other.clone());
                }
            }
        }
        for ob in &stmt.order_by {
            check_names(chain, &ob.expression, &aliases)?;
        }
        let plan = executor.order_by_projection(&select_exprs, &stmt.order_by, &[], columns);
        let registry = &executor.function_registry;
        let ctx = std::cell::OnceCell::new();
        let compile = |expr: &Expression| {
            compile_with(
                ctx.get_or_init(|| CompileContext::new(columns, registry)),
                unaliased(expr),
            )
        };
        let mut outputs = Vec::with_capacity(select_exprs.len() + plan.extra_names.len());
        for (expr, index) in select_exprs.iter().zip(&plan.select_column_indices) {
            outputs.push(match index {
                Some(index) => Output::Column(*index),
                None => Output::Program(compile(expr)?),
            });
        }
        outputs.extend(plan.extra_order_indices.iter().map(|&i| Output::Column(i)));
        for expr in &plan.computed_order_by {
            outputs.push(Output::Key(compile(expr)?));
        }
        // A key reading a SELECT alias is computed from what the alias
        // names, under the key's own text, which is how the sort finds it
        let mut extra_names = plan.extra_names;
        let alias_map = Executor::build_alias_map_excluding(&stmt.columns, Some(columns));
        for ob in &stmt.order_by {
            let key = &ob.expression;
            if is_leaf(key) || !names_any(key, &|name| alias_map.contains_key(name)) {
                continue;
            }
            let name = key.to_string();
            if extra_names.contains(&name) {
                continue;
            }
            outputs.push(Output::Key(compile(&Executor::substitute_aliases(
                key, &alias_map,
            ))?));
            extra_names.push(name);
        }
        Ok((Self { outputs }, extra_names))
    }

    fn project(&self, mut rows: RowVec, ctx: &ExecutionContext) -> Result<RowVec> {
        let mut vm = ExprVM::new();
        let mut projected = RowVec::with_capacity(rows.len());
        for (row_id, row) in rows.drain_rows().enumerate() {
            let mut values = Vec::with_capacity(self.outputs.len());
            let exec = ExecuteContext::with_common_params(
                &row,
                ctx.params(),
                Some(ctx.named_params()),
                ctx.transaction_id(),
            );
            for output in &self.outputs {
                values.push(match output {
                    Output::Column(i) => row.get(*i).cloned().unwrap_or_else(Value::null_unknown),
                    Output::Program(program) => vm.execute(program, &exec)?,
                    Output::Key(program) => vm
                        .execute(program, &exec)
                        .unwrap_or_else(|_| Value::null_unknown()),
                });
            }
            projected.push((row_id as i64, Row::from_values(values)));
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
        let scope = ScopeTable {
            name: scope_name.clone(),
            columns: schema.column_names_arc(),
            lower: schema.column_names_lower_arc(),
        };
        let scopes = [&scope];
        let chain = Chain { scopes: &scopes };
        let mut builder = Builder {
            executor,
            next_slot: param_count,
            subqueries: Vec::new(),
            named: Vec::new(),
            max_param: 0,
        };
        let where_clause = stmt
            .where_clause
            .as_deref()
            .ok_or_else(|| Error::internal("a prepared statement without WHERE"))?;
        let where_clause = with_aliases(where_clause, &stmt.columns, &scope.columns);
        let mut parts = Vec::new();
        conjuncts(&where_clause, &mut parts);

        let mut work = Rewritten::default();
        let mut rewritten = Vec::with_capacity(parts.len());
        let mut run_slots = Vec::new();
        let mut pushed = Vec::new();
        for part in parts {
            let part = builder.statement_conjunct(&chain, part, &mut work, &mut run_slots)?;
            if !mentions_subquery(&part) && !needs_folding(&part) {
                pushed.push(part.clone());
            }
            rewritten.push(part);
        }
        let where_clause = conjunction(rewritten)
            .ok_or_else(|| Error::internal("a prepared statement without WHERE"))?;
        let own = CompileContext::new(&scope.columns, registry).with_subquery_ids(&work.ids);
        let compile = |expr: &Expression| compile_with(&own, expr);
        let where_program = compile(&where_clause)?;
        let run_slots = run_slots
            .iter()
            .map(|(slot, value)| compile(value).map(|p| (*slot, p)))
            .collect::<Result<Vec<_>>>()?;
        let (projection, extra_names) = PreparedProjection::prepare(executor, stmt, &chain)?;
        for column in stmt
            .columns
            .iter()
            .chain(stmt.order_by.iter().map(|ob| &ob.expression))
        {
            builder.note_names(unaliased(column));
        }
        // Slots are numbered after the values given, so a value not given would read one
        if builder.max_param > param_count {
            return Err(Error::internal(format!(
                "Query requires {} parameters but only {} provided",
                builder.max_param, param_count
            )));
        }
        let mut output_columns =
            executor.get_output_column_names(&stmt.columns, &scope.columns, Some(scope_name));
        output_columns.extend(extra_names);
        let subqueries = builder
            .subqueries
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::internal("a prepared subquery left unfinished"))?;
        Ok(Self {
            cached_epoch: epoch,
            columns: scope.columns.clone(),
            param_count,
            slot_count: builder.next_slot,
            named: builder.named,
            where_program,
            run_slots,
            pushdown: conjunction(pushed),
            subqueries,
            projection,
            output_columns: CompactArc::new(output_columns),
        })
    }
}

fn mentions_subquery(expr: &Expression) -> bool {
    match expr {
        Expression::ScalarSubquery(_) => true,
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
        work.ids.extend(value_work.ids);
        let value = if !value_work.reads_row && !is_leaf(value) && !volatile(value) {
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
    /// `name` as this statement reads it: through the explicit transaction,
    /// or through the statement's snapshot, each at the statement's view
    fn prepared_table(&self, name: &str, ctx: &ExecutionContext) -> Result<Box<dyn Table>> {
        use crate::storage::traits::transaction::StatementTable;
        let active = self
            .active_transaction
            .lock()
            .map_err(|_| Error::internal("active transaction lock poisoned"))?;
        let table = match active.as_ref() {
            Some(tx) => tx
                .transaction
                .get_statement_table_at(name, ctx.scan_view(name)),
            None => {
                drop(active);
                ctx.statement_snapshot()
                    .ok_or_else(|| {
                        Error::internal("a prepared statement reads without a snapshot")
                    })?
                    .statement_table_at(name, ctx.scan_view(name))
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

    /// The plan's prepared statement, prepared now when the plan holds
    /// none for this schema, these columns and this parameter count
    fn prepared_for(
        &self,
        compiled: Option<&RwLock<CompiledExecution>>,
        stmt: &SelectStatement,
        table: &dyn Table,
        scope_name: &SmartString,
        ctx: &ExecutionContext,
        fresh: bool,
    ) -> Result<Arc<PreparedSelect>> {
        let epoch = self.engine.schema_epoch();
        let columns = table.schema().column_names_arc();
        let fits = |p: &PreparedSelect| {
            p.cached_epoch == epoch
                && p.param_count == ctx.params().len()
                && (CompactArc::ptr_eq(&p.columns, &columns) || p.columns == columns)
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
            table.schema(),
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
                        && p.param_count == ctx.params().len()
                        && ctx.outer_row().is_none()
                }
                _ => false,
            })
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
        let Some(Expression::TableSource(ts)) = stmt.table_expr.as_deref() else {
            return Err(Error::internal("a prepared statement without one table"));
        };
        let table = self.prepared_table(&ts.name.value_lower, ctx)?;
        let scope_name = ts
            .alias
            .as_ref()
            .map_or(&ts.name.value_lower, |alias| &alias.value_lower);
        let inner_ctx = ctx.with_incremented_query_depth();
        // A column change between taking the plan and opening a subquery's
        // table prepares again, against the table as it now is
        let mut fresh = false;
        let (prepared, tables) = loop {
            let prepared = self.prepared_for(compiled, stmt, &*table, scope_name, ctx, fresh)?;
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::prepared_plan_taken();
            if let Some(tables) = PreparedRunner::open_tables(self, &prepared, &inner_ctx)? {
                break (prepared, tables);
            }
            if fresh {
                return Err(Error::internal(
                    "a table's columns changed while the statement prepared",
                ));
            }
            fresh = true;
        };
        for name in &prepared.named {
            if !ctx.named_params().contains_key(name.as_str()) {
                return Err(Error::invalid_argument(format!(
                    "named parameter :{} has no value",
                    name
                )));
            }
        }

        let mut runner = PreparedRunner::new(&prepared, ctx, inner_ctx, tables);
        if !prepared.run_slots.is_empty() {
            let values = runner.run_slot_values()?;
            for (slot, value) in values {
                runner.base[slot] = value;
            }
        }
        let storage_expr = match &prepared.pushdown {
            Some(pushdown) if prepared.run_slots.is_empty() => {
                pushdown::try_pushdown(pushdown, table.schema(), Some(ctx)).0
            }
            Some(pushdown) => {
                let mut run_ctx = ctx.clone();
                run_ctx.set_params(runner.base.clone());
                pushdown::try_pushdown(pushdown, table.schema(), Some(&run_ctx)).0
            }
            None => None,
        };
        let output_columns = CompactArc::clone(&prepared.output_columns);
        if let Some(expr) = &storage_expr {
            if self
                .get_query_planner()
                .can_prune_entire_scan(&*table, expr.as_ref())
            {
                let result = ExecutorResult::with_arc_columns(
                    CompactArc::clone(&output_columns),
                    RowVec::new(),
                );
                return Ok((Box::new(result), output_columns, true));
            }
        }

        let target = limit
            .filter(|_| stmt.order_by.is_empty())
            .map(|limit| limit.saturating_add(offset));
        let all: Vec<usize> = (0..prepared.columns.len()).collect();
        let mut scanner = table.scan(&all, storage_expr.as_deref())?;
        let mut vm = ExprVM::new();
        let mut rows = RowVec::with_capacity(64);
        let mut scanned = 0u64;
        while target.is_none_or(|target| rows.len() < target) && scanner.next() {
            scanned += 1;
            if scanned.is_multiple_of(100) {
                ctx.check_cancelled()?;
            }
            let row = scanner.take_row();
            if runner.matches(&mut vm, &row)? {
                rows.push((scanned as i64, row));
            }
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        drop(scanner);
        let limit_applied = target.is_some();
        if limit_applied {
            let keep = limit.unwrap_or(usize::MAX);
            rows = rows.into_iter().skip(offset).take(keep).collect();
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_projection_ran();
        let rows = prepared.projection.project(rows, ctx)?;
        let result = ExecutorResult::with_arc_columns(CompactArc::clone(&output_columns), rows);
        Ok((Box::new(result), output_columns, limit_applied))
    }
}

/// One execution of a prepared statement. Each subquery has its own
/// parameter buffer: the statement's parameters, then every slot, its
/// parent's values copied in before its own holes are filled
pub(crate) struct PreparedRunner<'a> {
    prepared: &'a PreparedSelect,
    /// The statement's parameters, then its run slots; the other slots NULL
    base: ParamVec,
    named: &'a FxHashMap<String, Value>,
    transaction_id: Option<u64>,
    /// The context the subqueries read their tables under
    ctx: ExecutionContext,
    buffers: Vec<RefCell<ParamVec>>,
    vms: Vec<RefCell<ExprVM>>,
    /// Each subquery's table, opened before the run reads a row
    tables: Vec<Box<dyn Table>>,
    /// Each subquery's pushed comparisons, owned by this run
    bound: Vec<RefCell<Option<BoundPushdown>>>,
    /// Each uncorrelated subquery's value, once computed
    values: Vec<RefCell<Option<Value>>>,
}

impl<'a> PreparedRunner<'a> {
    /// Every subquery's table as `inner_ctx` reads it, or None when one no
    /// longer holds the columns its programs were prepared for
    fn open_tables(
        executor: &Executor,
        prepared: &PreparedSelect,
        inner_ctx: &ExecutionContext,
    ) -> Result<Option<Vec<Box<dyn Table>>>> {
        let mut tables = Vec::with_capacity(prepared.subqueries.len());
        for sq in &prepared.subqueries {
            let table = executor.prepared_table(&sq.table, inner_ctx)?;
            if table.schema().columns != sq.schema.columns {
                return Ok(None);
            }
            tables.push(table);
        }
        Ok(Some(tables))
    }

    fn new(
        prepared: &'a PreparedSelect,
        ctx: &'a ExecutionContext,
        inner_ctx: ExecutionContext,
        tables: Vec<Box<dyn Table>>,
    ) -> Self {
        let mut base: ParamVec = ctx.params().iter().cloned().collect();
        base.resize(prepared.slot_count, Value::null_unknown());
        let count = prepared.subqueries.len();
        Self {
            prepared,
            buffers: (0..count).map(|_| RefCell::new(base.clone())).collect(),
            base,
            named: ctx.named_params(),
            transaction_id: ctx.transaction_id(),
            ctx: inner_ctx,
            vms: (0..count).map(|_| RefCell::new(ExprVM::new())).collect(),
            tables,
            bound: (0..count).map(|_| RefCell::new(None)).collect(),
            values: (0..count).map(|_| RefCell::new(None)).collect(),
        }
    }

    /// The statement's run slots, each computed once over no row
    fn run_slot_values(&self) -> Result<Vec<(usize, Value)>> {
        let empty = Row::new();
        let mut vm = ExprVM::new();
        let exec = ExecuteContext::with_common_params(
            &empty,
            &self.base,
            Some(self.named),
            self.transaction_id,
        )
        .with_subquery_executor(self);
        self.prepared
            .run_slots
            .iter()
            .map(|(slot, program)| Ok((*slot, vm.execute(program, &exec)?)))
            .collect()
    }

    /// Whether `row` passes the WHERE
    fn matches(&self, vm: &mut ExprVM, row: &Row) -> Result<bool> {
        let exec = ExecuteContext::with_common_params(
            row,
            &self.base,
            Some(self.named),
            self.transaction_id,
        )
        .with_subquery_executor(self);
        vm.execute_bool_checked(&self.prepared.where_program, &exec)
    }

    /// The subquery's pushed comparisons for this call, None when there
    /// are none or one value is not one the scan can compare by
    fn bind_pushdown(&self, index: usize, params: &[Value]) -> bool {
        let sq = &self.prepared.subqueries[index];
        if sq.pushdown.is_empty() {
            return false;
        }
        let values: Option<Vec<Value>> = sq
            .pushdown
            .iter()
            .map(|c| c.value(params, self.named))
            .collect();
        match values {
            Some(values) => {
                BoundPushdown::bind(&sq.pushdown, &mut self.bound[index].borrow_mut(), values);
                true
            }
            None => false,
        }
    }
}

impl SubqueryExecutor for PreparedRunner<'_> {
    fn execute_scalar(&self, subquery_index: u16, row: &Row) -> Result<Value> {
        let index = subquery_index as usize;
        let sq = self
            .prepared
            .subqueries
            .get(index)
            .ok_or_else(|| Error::internal("unknown prepared subquery"))?;
        if sq.uncorrelated {
            if let Some(value) = self.values[index].borrow().as_ref() {
                return Ok(value.clone());
            }
        }
        self.ctx.check_cancelled()?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_scalar_next();
        let mut vm = self.vms[index].borrow_mut();
        {
            let mut params = self.buffers[index].borrow_mut();
            if let Some(parent) = sq.parent {
                params.clone_from(&self.buffers[parent as usize].borrow());
            }
            let hole_ctx = ExecuteContext::with_common_params(
                row,
                &self.base,
                Some(self.named),
                self.transaction_id,
            );
            for (slot, hole) in &sq.holes {
                params[*slot] = vm.execute(hole, &hole_ctx)?;
            }
        }
        // Read-only from here on: a nested subquery copies these values
        let params = self.buffers[index].borrow();
        let table = &*self.tables[index];
        // The scanner keeps what it needs of the expression, so the next
        // call may rebind it once this scan is set up
        let pushed = self.bind_pushdown(index, &params);
        let mut scanner = if pushed {
            let bound = self.bound[index].borrow();
            table.scan(&sq.column_indices, bound.as_ref().map(BoundPushdown::expr))?
        } else {
            table.scan(&sq.column_indices, None)?
        };
        let filter = sq.filter.as_ref().filter(|_| !(pushed && sq.pushed_whole));
        let mut value = None;
        while scanner.next() {
            let exec = ExecuteContext::with_common_params(
                scanner.row(),
                &params,
                Some(self.named),
                self.transaction_id,
            )
            .with_subquery_executor(self);
            if let Some(filter) = filter {
                if !vm.execute_bool_checked(filter, &exec)? {
                    continue;
                }
            }
            if value.is_some() {
                return Err(Error::Internal {
                    message: "scalar subquery returned more than one row".to_string(),
                });
            }
            value = Some(vm.execute(&sq.projection, &exec)?);
        }
        if let Some(err) = scanner.err() {
            return Err(err.clone());
        }
        let value = value.unwrap_or_else(Value::null_unknown);
        if sq.uncorrelated {
            *self.values[index].borrow_mut() = Some(value.clone());
        }
        Ok(value)
    }

    fn execute_exists(
        &self,
        _subquery_index: u16,
        _outer_row: Option<&FxHashMap<CompactArc<str>, Value>>,
    ) -> Result<bool> {
        Err(Error::internal("EXISTS is not prepared"))
    }

    fn execute_in(
        &self,
        _subquery_index: u16,
        _value: &Value,
        _outer_row: Option<&FxHashMap<CompactArc<str>, Value>>,
    ) -> Result<bool> {
        Err(Error::internal("IN is not prepared"))
    }

    fn execute_all(
        &self,
        _subquery_index: u16,
        _value: &Value,
        _op: super::expression::CompareOp,
        _outer_row: Option<&FxHashMap<CompactArc<str>, Value>>,
    ) -> Result<bool> {
        Err(Error::internal("ALL is not prepared"))
    }

    fn execute_any(
        &self,
        _subquery_index: u16,
        _value: &Value,
        _op: super::expression::CompareOp,
        _outer_row: Option<&FxHashMap<CompactArc<str>, Value>>,
    ) -> Result<bool> {
        Err(Error::internal("ANY is not prepared"))
    }
}
