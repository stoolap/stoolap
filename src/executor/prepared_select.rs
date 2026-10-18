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
use crate::functions::{FunctionRegistry, Volatility};
use crate::parser::ast::*;
use crate::parser::token::{Position, Token, TokenType};
use crate::storage::expression::{AndExpr, ComparisonExpr, Expression as StorageExpr};
use crate::storage::traits::{Engine, Table};

use super::context::{
    cache_exists_answer, get_cached_exists_answer, ExecutionContext, ExistsAnswer, ExistsKeys,
    ExistsOrigin, ExistsOutcome,
};
use super::expr_key::ExprKey;
use super::expression::{
    CompileContext, CompileError, ExecuteContext, ExprCompiler, ExprVM, SharedProgram,
    SubqueryExecutor,
};
use super::pushdown::{self, coerce_constant};
use super::query_cache::CompiledExecution;
use super::result::ExecutorResult;
use super::utils::{
    extract_literal_value, flip_operator, infix_to_operator, is_aggregate_function,
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

/// The rows, their columns, whether LIMIT and OFFSET were applied, and how
/// many of the columns the statement shows
type PreparedOutput = (
    Box<dyn crate::storage::traits::QueryResult>,
    CompactArc<Vec<String>>,
    bool,
    usize,
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
            Expression::Exists(exists) => {
                if !subqueries {
                    return Err("a subquery outside WHERE and the subquery blocks");
                }
                *found = true;
                self.existence_block(&exists.subquery, scopes + 1)
            }
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
        if !plain_block(sq) || !sq.order_by.is_empty() || sq.limit.is_some() || sq.offset.is_some()
        {
            return Err("a subquery block with a clause outside the family");
        }
        self.from(sq.table_expr.as_deref())?;
        let mut found = false;
        for column in &sq.columns {
            if !matches!(column, Expression::Star(_) | Expression::QualifiedStar(_)) {
                self.expression(unaliased(column), scopes, true, &mut found)?;
            }
        }
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
            return Err("WHERE holds no subquery");
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
    types: Vec<DataType>,
}

impl ScopeTable {
    fn of(name: SmartString, schema: &Schema) -> Self {
        Self {
            name,
            columns: schema.column_names_arc(),
            lower: schema.column_names_lower_arc(),
            types: schema.columns.iter().map(|c| c.data_type).collect(),
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
    /// programs compile against. Qualified by a name no table can have, so
    /// the compiler reads it as a column wherever it stands, never a literal
    fn column(&self, depth: usize, index: usize) -> Expression {
        let ident = |value: &str, lower: &str| {
            let mut id = Identifier::new(
                Token::new(TokenType::Identifier, value, Position::default()),
                value,
            );
            id.value_lower = SmartString::from(lower);
            id
        };
        Expression::QualifiedIdentifier(QualifiedIdentifier {
            qualifier: ident(RESOLVED, RESOLVED),
            name: ident(
                &self.scopes[depth].columns[index],
                &self.scopes[depth].lower[index],
            ),
        })
    }

    /// What `expr` reads; a scalar subquery counts as reading this row
    fn reads(&self, expr: &Expression) -> Result<Reads> {
        match expr {
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_) => {
                Ok(self.resolve(expr)?.map_or(0, |(depth, _)| 1 << depth))
            }
            Expression::ScalarSubquery(_) | Expression::Exists(_) => Ok(1),
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

/// Whether the scan compares values of `data_type` by the program's rule;
/// a JSON or VECTOR comparison is left to the program
fn scan_compares(data_type: DataType) -> bool {
    !matches!(data_type, DataType::Json | DataType::Vector)
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
    subqueries: Vec<PreparedSubquery>,
    /// A top-level EXISTS or NOT EXISTS conjunct with one key, whose keyed
    /// set's keys name the rows the WHERE may pass
    narrowing: Option<Narrowing>,
    projection: PreparedProjection,
    output_columns: CompactArc<Vec<String>>,
    /// The output columns the statement shows; the sort's keys follow them
    visible: usize,
}

/// A subquery over one table, prepared as its own scope: every value it
/// reads from an enclosing scope becomes a parameter slot, filled per call
/// by a hole program over the row it runs for
#[derive(Debug)]
struct PreparedSubquery {
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
    /// The subquery whose row this one runs for; None for the statement's row
    parent: Option<u16>,
    /// What fills each slot from that row, including values a deeper
    /// subquery reads from it
    holes: Vec<(usize, Hole)>,
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
    /// NOT EXISTS: the rows its keys do not name
    negated: bool,
}

/// Slots whose value raised an error, each with its error
type FailedSlots = SmallVec<[(usize, Error); 1]>;

/// One slot's value over the row a subquery runs for
#[derive(Debug)]
enum Hole {
    /// A column of that row, copied without a program
    Column(usize),
    Program(SharedProgram),
}

impl Hole {
    fn value(&self, vm: &mut ExprVM, row: &Row, ctx: &ExecuteContext) -> Result<Value> {
        match self {
            Hole::Column(index) => Ok(row.get(*index).cloned().unwrap_or_else(Value::null_unknown)),
            Hole::Program(program) => vm.execute(program, ctx),
        }
    }
}

/// What a subquery block answers
#[derive(Debug)]
enum Answer {
    /// Its one row's SELECT expression, NULL for no row
    Value(SharedProgram),
    /// Whether a candidate row passes its WHERE
    Exists(Existence),
}

/// What an EXISTS block needs beyond its probe
#[derive(Debug)]
struct Existence {
    /// The block as written: what a result kept across statements is of
    identity: Arc<Expression>,
    identity_key: ExprKey,
    /// The enclosing scopes' names, columns and types, which decide what
    /// each name of the block binds to
    scopes: Arc<str>,
    /// Every table the block and the blocks in it read
    tables: Vec<SmartString>,
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
            let Some(column) = bound_column(column) else {
                continue;
            };
            let value = match value.as_ref() {
                Expression::Parameter(p) if p.name.starts_with(':') => {
                    PreparedValue::Named(SmartString::from(&p.name[1..]))
                }
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
                let mark = self.nested(chain, id, &nested.subquery, None, work)?;
                Ok(Expression::ScalarSubquery(ScalarSubquery {
                    token: nested.token.clone(),
                    subquery: mark,
                }))
            }
            Expression::Exists(nested) => {
                let mark = self.nested(chain, id, &nested.subquery, Some(nested), work)?;
                Ok(Expression::Exists(ExistsExpression {
                    token: nested.token.clone(),
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
        exists: Option<&ExistsExpression>,
        work: &mut Rewritten,
    ) -> Result<Box<SelectStatement>> {
        let (nested_id, delegated, reads_this_row) = self.subquery(sq, chain.scopes, id, exists)?;
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
        });
        work.ids
            .insert(&*mark as *const SelectStatement as usize, nested_id);
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

    /// Prepare `sq`, run once per row of the scope `outer[0]`, as the block
    /// of `exists` or as a scalar block. Returns its id, the values it reads
    /// from scopes beyond its parent, and whether it reads its parent's row
    fn subquery(
        &mut self,
        sq: &SelectStatement,
        outer: &[&ScopeTable],
        parent: Option<u16>,
        exists: Option<&ExistsExpression>,
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
        let scope = ScopeTable::of(
            ts.alias
                .as_ref()
                .map_or(&ts.name.value_lower, |alias| &alias.value_lower)
                .clone(),
            &schema,
        );
        let id = u16::try_from(self.subqueries.len())
            .map_err(|_| Error::internal("too many prepared subqueries"))?;
        self.subqueries.push(None);

        let mut scopes: Vec<&ScopeTable> = Vec::with_capacity(outer.len() + 1);
        scopes.push(&scope);
        scopes.extend_from_slice(outer);
        let chain = Chain { scopes: &scopes };
        let mut work = Rewritten::default();
        let projection = match exists {
            Some(_) => None,
            None => Some(self.rewrite(&chain, Some(id), unaliased(&sq.columns[0]), &mut work)?),
        };
        let where_clause = sq
            .where_clause
            .as_deref()
            .map(|w| with_aliases(w, &sq.columns, &scope.columns));
        let mut pushdown = Vec::new();
        let mut pushed_whole = true;
        // An EXISTS block's correlation equalities apart from the rest
        let mut keys = Vec::new();
        let mut key_parts = Vec::new();
        let mut rest = Vec::new();
        if let Some(where_clause) = &where_clause {
            let mut parts = Vec::new();
            conjuncts(where_clause, &mut parts);
            for part in parts {
                let key = match exists {
                    Some(_) => correlation_key(&chain, part)?,
                    None => None,
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

        let mut read_as = FxHashMap::default();
        for expr in candidates.iter().chain(&filter) {
            compare_as_scan(expr, &schema, &mut read_as);
        }
        let own = CompileContext::new(&scope.columns, registry)
            .with_subquery_ids(&work.ids)
            .with_read_as(&read_as);
        let compile = |expr: &Option<Expression>| match expr {
            Some(expr) => compile_with(&own, expr).map(Some),
            None => Ok(None),
        };
        let answer = match exists {
            None => Answer::Value(compile(&projection)?.ok_or_else(|| {
                Error::internal("a prepared scalar subquery without a SELECT expression")
            })?),
            Some(exists) => {
                let identity = Expression::Exists(exists.clone());
                let keyed = self.keyed(&chain, &keys, &work, sq).then(|| {
                    let mut pushdown = Vec::new();
                    let mut pushed_whole = true;
                    let mut parts = Vec::new();
                    if let Some(filter) = &filter {
                        conjuncts(filter, &mut parts);
                    }
                    for part in &parts {
                        match PreparedComparison::from_conjunct(part, &schema) {
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
                });
                let mut scopes = String::new();
                for scope in outer {
                    scopes.push_str(&scope.name);
                    for (column, data_type) in scope.lower.iter().zip(&scope.types) {
                        scopes.push_str(&format!("\u{0}{column}:{data_type:?}"));
                    }
                    scopes.push('\u{1}');
                }
                Answer::Exists(Existence {
                    identity_key: ExprKey::of(&identity),
                    identity: Arc::new(identity),
                    scopes: Arc::from(scopes),
                    tables: vec![ts.name.value_lower.clone()],
                    keyed,
                })
            }
        };
        let mut reads = SmallVec::new();
        for expr in projection.iter().chain(&candidates).chain(&filter) {
            slots_read(expr, &mut reads);
        }
        let candidates = compile(&candidates)?;
        let mut rest = Vec::new();
        if let (Some(_), Some(filter)) = (exists, &filter) {
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
            let column = |hole: &Expression| {
                bound_column(hole).and_then(|name| outer[0].lower.iter().position(|c| c == name))
            };
            work.holes
                .iter()
                .map(|(slot, hole)| match column(hole) {
                    Some(index) => Ok((*slot, Hole::Column(index))),
                    None => compile_with(&parent, hole).map(|p| (*slot, Hole::Program(p))),
                })
                .collect::<Result<Vec<_>>>()?
        };
        let correlated = work.reads_parent || !work.delegated.is_empty();
        self.subqueries[id as usize] = Some(PreparedSubquery {
            table: ts.name.value_lower.clone(),
            column_indices: (0..scope.columns.len()).collect(),
            schema: CompactArc::clone(&schema),
            pushdown,
            pushed_whole,
            candidates,
            filter,
            rest,
            reads,
            answer,
            parent,
            holes,
            uncorrelated: !correlated,
        });
        Ok((id, work.delegated, work.reads_parent))
    }

    /// Whether an EXISTS block whose correlation equalities are `keys`
    /// answers by a keyed set as its probe does: it reads its parent through
    /// them alone, holds nothing volatile, and each key's two columns are of
    /// one type whose equality is its hash's
    fn keyed(
        &self,
        chain: &Chain<'_>,
        keys: &[((usize, usize, usize), usize)],
        work: &Rewritten,
        block: &SelectStatement,
    ) -> bool {
        let key_slot = |slot: usize| keys.iter().any(|(_, key)| *key == slot);
        !keys.is_empty()
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
        Expression::Parameter(p) if p.index > 0 && !p.name.starts_with(':') => {
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
            Expression::Parameter(p) if p.index > 0 && !p.name.starts_with(':') => {
                Some(p.index - 1)
            }
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
        Expression::ScalarSubquery(_) | Expression::Exists(_) => true,
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
        // A key reading a SELECT alias reads the value the projection gave
        // it, under the key's own text, which is how the sort finds it
        let mut extra_names = plan.extra_names;
        let alias_map = Executor::build_alias_map_excluding(&stmt.columns, Some(columns));
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
        }
        let whole_row = outputs.len() == columns.len()
            && outputs
                .iter()
                .enumerate()
                .all(|(i, output)| matches!(output, Output::Column(c) if *c == i));
        Ok((Self { outputs, whole_row }, extra_names))
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
                        Some(ctx.named_params()),
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
        let mut exists_conjuncts = Vec::new();
        for part in parts {
            let part = builder.statement_conjunct(&chain, part, &mut work, &mut run_slots)?;
            if !mentions_subquery(&part) && !needs_folding(&part) && scans_whole(&part, schema) {
                pushed.push(part.clone());
            }
            let exists = match &part {
                Expression::Exists(mark) => Some((mark, false)),
                Expression::Prefix(prefix) if prefix.op_type == PrefixOperator::Not => {
                    match prefix.right.as_ref() {
                        Expression::Exists(mark) => Some((mark, true)),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some((mark, negated)) = exists {
                exists_conjuncts.extend(
                    work.ids
                        .get(&(&*mark.subquery as *const SelectStatement as usize))
                        .map(|&id| (id, negated)),
                );
            }
            rewritten.push(part);
        }
        let rewritten_count = rewritten.len();
        let where_clause = conjunction(rewritten)
            .ok_or_else(|| Error::internal("a prepared statement without WHERE"))?;
        let mut read_as = FxHashMap::default();
        compare_as_scan(&where_clause, schema, &mut read_as);
        let own = CompileContext::new(&scope.columns, registry)
            .with_subquery_ids(&work.ids)
            .with_read_as(&read_as);
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
        // Over one table `t.*` is `*`, the projection's expansion; the
        // names' prefix rule for a joined `t.*` would drop columns
        let mut output_columns = if stmt
            .columns
            .iter()
            .any(|c| matches!(c, Expression::QualifiedStar(_)))
        {
            let unqualified: Vec<Expression> = stmt
                .columns
                .iter()
                .map(|c| match c {
                    Expression::QualifiedStar(star) => Expression::Star(StarExpression {
                        token: star.token.clone(),
                    }),
                    other => other.clone(),
                })
                .collect();
            executor.get_output_column_names(&unqualified, &scope.columns, Some(scope_name))
        } else {
            executor.get_output_column_names(&stmt.columns, &scope.columns, Some(scope_name))
        };
        let visible = output_columns.len();
        output_columns.extend(extra_names);
        let mut subqueries = builder
            .subqueries
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::internal("a prepared subquery left unfinished"))?;
        let narrowing = exists_conjuncts.iter().find_map(|&(id, negated)| {
            match &subqueries[id as usize].answer {
                Answer::Exists(Existence {
                    keyed: Some(plan), ..
                }) if plan.keys.len() == 1 => Some(Narrowing {
                    id,
                    column: plan.keys[0].2,
                    whole: rewritten_count == 1,
                    negated,
                }),
                _ => None,
            }
        });
        // A block's tables include those of every block nested in it
        for nested in 0..subqueries.len() {
            let mut ancestor = subqueries[nested].parent;
            while let Some(at) = ancestor {
                let table = subqueries[nested].table.clone();
                if let Answer::Exists(existence) = &mut subqueries[at as usize].answer {
                    if !existence.tables.contains(&table) {
                        existence.tables.push(table);
                    }
                }
                ancestor = subqueries[at as usize].parent;
            }
        }
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
            narrowing,
            projection,
            output_columns: CompactArc::new(output_columns),
            visible,
        })
    }
}

fn mentions_subquery(expr: &Expression) -> bool {
    match expr {
        Expression::ScalarSubquery(_) | Expression::Exists(_) => true,
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

        let statement_limit = limit
            .filter(|_| stmt.order_by.is_empty())
            .map(|limit| i64::try_from(limit).unwrap_or(i64::MAX));
        let mut runner = PreparedRunner::new(
            self,
            &prepared,
            ctx,
            inner_ctx,
            tables,
            table.row_count_hint(),
            statement_limit,
        );
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
                return Ok((Box::new(result), output_columns, true, prepared.visible));
            }
        }

        let target = limit
            .filter(|_| stmt.order_by.is_empty())
            .map(|limit| limit.saturating_add(offset));
        let mut vm = ExprVM::new();
        let mut rows = RowVec::with_capacity(64);
        let mut scanned = 0u64;
        if let Some((candidates, passed)) = runner.narrowed_rows(&*table, target)? {
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::prepared_rows_narrowed_by_keys();
            for (_, row) in candidates {
                if target.is_some_and(|target| rows.len() >= target) {
                    break;
                }
                scanned += 1;
                if scanned.is_multiple_of(100) {
                    ctx.check_cancelled()?;
                }
                // The conjuncts the scan would have taken, before the WHERE
                if storage_expr
                    .as_ref()
                    .is_some_and(|expr| !expr.evaluate_fast(&row))
                {
                    continue;
                }
                if passed || runner.matches(&mut vm, &row)? {
                    rows.push((scanned as i64, row));
                }
            }
        } else {
            let all: Vec<usize> = (0..prepared.columns.len()).collect();
            let mut scanner = table.scan(&all, storage_expr.as_deref())?;
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
        }
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
        ))
    }
}

/// One execution of a prepared statement. Each subquery has its own
/// parameter buffer: the statement's parameters, then every slot, its
/// parent's values copied in before its own holes are filled
pub(crate) struct PreparedRunner<'a> {
    executor: &'a Executor,
    prepared: &'a PreparedSelect,
    /// The statement's own context, whose view and parameters a kept answer is of
    statement: &'a ExecutionContext,
    /// The statement table's rows, by its hint
    statement_rows: usize,
    /// The rows the statement stops at, when no ORDER BY reads them all
    statement_limit: Option<i64>,
    /// Each keyed EXISTS block's choice for this run, once made
    keyed_choice: Vec<std::cell::Cell<Option<bool>>>,
    keyed_sets: Vec<RefCell<Option<CompactArc<ExistsKeys>>>>,
    /// Each subquery's slots whose value raised an error for this call,
    /// its parent's included: an EXISTS block raises one only on a candidate
    failed: Vec<RefCell<FailedSlots>>,
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
        executor: &'a Executor,
        prepared: &'a PreparedSelect,
        ctx: &'a ExecutionContext,
        inner_ctx: ExecutionContext,
        tables: Vec<Box<dyn Table>>,
        statement_rows: usize,
        statement_limit: Option<i64>,
    ) -> Self {
        let mut base: ParamVec = ctx.params().iter().cloned().collect();
        base.resize(prepared.slot_count, Value::null_unknown());
        let count = prepared.subqueries.len();
        Self {
            executor,
            prepared,
            statement: ctx,
            statement_rows,
            statement_limit,
            keyed_choice: (0..count).map(|_| std::cell::Cell::new(None)).collect(),
            keyed_sets: (0..count).map(|_| RefCell::new(None)).collect(),
            failed: (0..count).map(|_| RefCell::new(SmallVec::new())).collect(),
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

    /// The statement's rows its narrowing EXISTS may pass, read by the keys
    /// of the block's keyed set when the run takes it, and whether each
    /// passes the WHERE without asking it: the EXISTS is the whole WHERE and
    /// no key holds an error; None to scan
    fn narrowed_rows(
        &self,
        table: &dyn Table,
        target: Option<usize>,
    ) -> Result<Option<(RowVec, bool)>> {
        let Some(Narrowing {
            id,
            column,
            whole,
            negated,
        }) = self.prepared.narrowing
        else {
            return Ok(None);
        };
        let index = id as usize;
        let sq = &self.prepared.subqueries[index];
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
            let mut vm = self.vms[index].borrow_mut();
            let mut params = self.buffers[index].borrow_mut();
            let mut failed = self.failed[index].borrow_mut();
            failed.clear();
            let empty = Row::new();
            let exec = ExecuteContext::with_common_params(
                &empty,
                &self.base,
                Some(self.named),
                self.transaction_id,
            );
            for (slot, hole) in &sq.holes {
                if !plan.keys.iter().any(|(_, key, _)| key == slot) {
                    match hole.value(&mut vm, &empty, &exec) {
                        Ok(value) => params[*slot] = value,
                        Err(e) => {
                            params[*slot] = Value::null_unknown();
                            failed.push((*slot, e));
                        }
                    }
                }
            }
        }
        let set = self.keyed_set(index, existence, plan)?;
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

    /// The subquery's parameter buffer for a call for `row`: its parent's
    /// values, then its own holes
    fn fill(&self, index: usize, vm: &mut ExprVM, row: &Row) -> Result<()> {
        let sq = &self.prepared.subqueries[index];
        let mut params = self.buffers[index].borrow_mut();
        let mut failed = self.failed[index].borrow_mut();
        failed.clear();
        if let Some(parent) = sq.parent {
            params.clone_from(&self.buffers[parent as usize].borrow());
            failed.extend(self.failed[parent as usize].borrow().iter().cloned());
        }
        let hole_ctx = ExecuteContext::with_common_params(
            row,
            &self.base,
            Some(self.named),
            self.transaction_id,
        );
        for (slot, hole) in &sq.holes {
            match hole.value(vm, row, &hole_ctx) {
                Ok(value) => params[*slot] = value,
                Err(e) => {
                    params[*slot] = Value::null_unknown();
                    failed.push((*slot, e));
                }
            }
        }
        // A failed slot its nested blocks read is theirs to raise
        match &sq.answer {
            Answer::Value(_) => match failed.iter().find(|(slot, _)| sq.reads.contains(slot)) {
                Some((_, e)) => Err(e.clone()),
                None => Ok(()),
            },
            Answer::Exists(_) => Ok(()),
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
        let mut vm = self.vms[index].borrow_mut();
        let params = self.buffers[index].borrow();
        let slots_failed = self.failed[index].borrow();
        let table = &*self.tables[index];
        // A failed slot holds NULL, which the scan must not compare with
        let pushed = slots_failed.is_empty() && self.bind_pushdown(index, &params);
        let mut scanner = if pushed {
            let bound = self.bound[index].borrow();
            table.scan(&sq.column_indices, bound.as_ref().map(BoundPushdown::expr))?
        } else {
            table.scan(&sq.column_indices, None)?
        };
        let decided = pushed && sq.pushed_whole;
        let mut failed = None;
        while scanner.next() {
            if !decided {
                let exec = ExecuteContext::with_common_params(
                    scanner.row(),
                    &params,
                    Some(self.named),
                    self.transaction_id,
                )
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
                        failed.get_or_insert(e);
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
        if let Some(keyed) = self.keyed_choice[index].get() {
            return keyed;
        }
        #[cfg(feature = "test-failpoints")]
        let forced = crate::test_failpoints::forced_exists_strategy();
        #[cfg(not(feature = "test-failpoints"))]
        let forced: Option<bool> = None;
        let keyed = forced.unwrap_or_else(|| !(plan.pure && self.probe_cheaper(index, plan)));
        self.keyed_choice[index].set(Some(keyed));
        keyed
    }

    /// The general path's rule for its index nested loop, with the parent
    /// block's rows as the outer rows; only the statement's LIMIT counts
    fn probe_cheaper(&self, index: usize, plan: &KeyedPlan) -> bool {
        let sq = &self.prepared.subqueries[index];
        let parent_rows = match sq.parent {
            Some(parent) => self.tables[parent as usize].row_count_hint(),
            None => self.statement_rows,
        };
        Executor::probe_cheaper(
            &*self.tables[index],
            &sq.schema.columns[plan.keys[0].0].name,
            self.statement_limit.filter(|_| sq.parent.is_none()),
            Some(parent_rows),
        )
    }

    /// The keyed set's answer for the buffer filled: FALSE for a NULL key
    fn keyed_lookup(&self, index: usize, existence: &Existence, plan: &KeyedPlan) -> Result<bool> {
        if self.keyed_sets[index].borrow().is_none() {
            self.keyed_set(index, existence, plan)?;
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::prepared_exists_looked_up();
        let sets = self.keyed_sets[index].borrow();
        let set = sets
            .as_ref()
            .ok_or_else(|| Error::internal("a keyed set not built"))?;
        let params = self.buffers[index].borrow();
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
            Some(ExistsOutcome::Failed(e)) => Err(e.clone()),
            None => Ok(false),
        }
    }

    /// The block's keyed set, built on first need or taken from a run
    /// before whose stamp still holds
    fn keyed_set(
        &self,
        index: usize,
        existence: &Existence,
        plan: &KeyedPlan,
    ) -> Result<CompactArc<ExistsKeys>> {
        if let Some(set) = self.keyed_sets[index].borrow().as_ref() {
            return Ok(CompactArc::clone(set));
        }
        let answer = self.kept(existence, || {
            self.build_keyed_set(index, plan)
                .map(|set| ExistsAnswer::Keyed(CompactArc::new(set)))
        })?;
        let ExistsAnswer::Keyed(set) = answer else {
            return Err(Error::internal("a value kept for a keyed set"));
        };
        *self.keyed_sets[index].borrow_mut() = Some(CompactArc::clone(&set));
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
        let mut vm = self.vms[index].borrow_mut();
        let params = self.buffers[index].borrow();
        let slots_failed = self.failed[index].borrow();
        let values: Option<Vec<Value>> = plan
            .pushdown
            .iter()
            .map(|c| c.value(&params, self.named))
            .collect();
        let mut bound = None;
        if let Some(values) =
            values.filter(|_| !plan.pushdown.is_empty() && slots_failed.is_empty())
        {
            BoundPushdown::bind(&plan.pushdown, &mut bound, values);
        }
        let table = &*self.tables[index];
        let mut scanner =
            table.scan(&sq.column_indices, bound.as_ref().map(BoundPushdown::expr))?;
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
                let exec = ExecuteContext::with_common_params(
                    row,
                    &params,
                    Some(self.named),
                    self.transaction_id,
                )
                .with_subquery_executor(self);
                match self.passes_rest(sq, &mut vm, &exec, &slots_failed) {
                    Ok(true) => ExistsOutcome::Passed,
                    Ok(false) => continue,
                    Err(e) => ExistsOutcome::Failed(e),
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
        existence: &Existence,
        compute: impl FnOnce() -> Result<ExistsAnswer>,
    ) -> Result<ExistsAnswer> {
        let statement = self.statement;
        let keeps = Executor::reads_cacheable_rows(statement)
            && !Executor::expression_varies_by_statement(&existence.identity, statement);
        let stamp = if keeps {
            self.executor
                .subquery_cache_stamp(&existence.tables, statement.statement_view())
        } else {
            None
        };
        let origin = ExistsOrigin {
            block: &existence.identity,
            block_key: existence.identity_key,
            scopes: &existence.scopes,
            params: statement.params(),
            named_params: statement.named_params(),
        };
        let key = origin.key();
        if let Some(stamp) = &stamp {
            if let Some(answer) = get_cached_exists_answer(&origin, key, stamp) {
                return Ok(answer);
            }
        }
        let answer = compute()?;
        if let Some(stamp) = stamp {
            cache_exists_answer(&origin, key, stamp, answer.clone());
        }
        Ok(answer)
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
        let Answer::Value(projection) = &sq.answer else {
            return Err(Error::internal("an EXISTS block run for a value"));
        };
        let mut vm = self.vms[index].borrow_mut();
        self.fill(index, &mut vm, row)?;
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
            value = Some(vm.execute(projection, &exec)?);
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

    fn execute_exists(&self, subquery_index: u16, row: &Row) -> Result<bool> {
        let index = subquery_index as usize;
        let sq = self
            .prepared
            .subqueries
            .get(index)
            .ok_or_else(|| Error::internal("unknown prepared subquery"))?;
        let Answer::Exists(existence) = &sq.answer else {
            return Err(Error::internal("a scalar block run for EXISTS"));
        };
        if sq.uncorrelated {
            if let Some(Value::Boolean(exists)) = self.values[index].borrow().as_ref() {
                return Ok(*exists);
            }
        }
        self.fill(index, &mut self.vms[index].borrow_mut(), row)?;
        let exists = if sq.uncorrelated {
            match self.kept(existence, || self.probe(index).map(ExistsAnswer::Once))? {
                ExistsAnswer::Once(exists) => exists,
                ExistsAnswer::Keyed(_) => {
                    return Err(Error::internal("a keyed set kept for a value"))
                }
            }
        } else {
            match &existence.keyed {
                Some(plan) if self.takes_keyed_set(index, plan) => {
                    self.keyed_lookup(index, existence, plan)?
                }
                _ => self.probe(index)?,
            }
        };
        if sq.uncorrelated {
            *self.values[index].borrow_mut() = Some(Value::Boolean(exists));
        }
        Ok(exists)
    }

    fn execute_in(&self, _subquery_index: u16, _value: &Value, _row: &Row) -> Result<Option<bool>> {
        Err(Error::internal("IN is not prepared"))
    }

    fn execute_quantified(
        &self,
        _subquery_index: u16,
        _value: &Value,
        _op: super::expression::CompareOp,
        _all: bool,
        _row: &Row,
    ) -> Result<Option<bool>> {
        Err(Error::internal("ANY or ALL is not prepared"))
    }
}
