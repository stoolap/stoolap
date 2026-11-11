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

//! DML Statement Execution
//!
//! This module implements execution of Data Manipulation Language (DML) statements:
//! - INSERT
//! - UPDATE
//! - DELETE

use crate::common::CompactArc;
use crate::common::I64Set;
use crate::common::SmartString;
use crate::core::integer::integer_neg;
use crate::core::{DataType, Error, Result, Row, RowVec, Schema, Value, ROWID_SLOT};
use crate::parser::ast::*;
use crate::storage::expression::{ComparisonExpr, Expression as StorageExpr};
use crate::storage::mvcc::version_store::LayoutToken;
use crate::storage::traits::transaction::StatementTable;
use crate::storage::traits::{Engine, QueryResult, ScanAccess, Table, Transaction};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

use super::context::ExecutionContext;
use super::pushdown;
use super::query_cache::{
    CompiledConflictInsert, CompiledExecution, CompiledInsert, NestedPlan, PlanValidity,
};
use super::result::{ExecResult, ExecutorResult};
use super::utils::qualified;
use super::Executor;
use std::sync::RwLock;

pub(super) struct StatementRollback<'a> {
    engine: &'a crate::storage::mvcc::MVCCEngine,
    checkpoint: Option<(i64, i64)>,
}

impl<'a> StatementRollback<'a> {
    pub(super) fn new(
        engine: &'a crate::storage::mvcc::MVCCEngine,
        active: Option<&super::ActiveTransaction>,
    ) -> Self {
        Self {
            engine,
            checkpoint: active.map(|tx| {
                (
                    tx.transaction.id(),
                    crate::storage::mvcc::last_write_order(),
                )
            }),
        }
    }

    pub(super) fn finish<T>(mut self, result: Result<T>) -> Result<T> {
        if result.is_ok() {
            self.checkpoint = None;
        }
        result
    }
}

impl Drop for StatementRollback<'_> {
    fn drop(&mut self) {
        if let Some((txn_id, timestamp)) = self.checkpoint {
            self.engine.rollback_dml_after(txn_id, timestamp);
        }
    }
}

/// The parameter reading a value the executor adds from `slot`, one past the
/// slots of the statement's own values, so no parameter of the statement
/// reads it: a VALUES answer, a CTE's pushed LIMIT
pub(crate) fn answer_parameter(slot: usize) -> Expression {
    let spelled = SmartString::from(format!("${slot}"));
    Expression::Parameter(Parameter {
        token: crate::parser::token::Token::new(
            crate::parser::token::TokenType::Parameter,
            spelled.clone(),
            crate::parser::token::Position::default(),
        ),
        name: spelled,
        index: slot,
    })
}

/// Check whether a constraint violation matches the ON CONFLICT target columns.
/// Returns true if the conflict should be handled (DO UPDATE / DO NOTHING).
/// Returns false if the violation is on a different constraint, meaning the
/// error should be re-raised.
///
/// When `conflict_target` is empty (MySQL ON DUPLICATE KEY UPDATE style),
/// all conflicts match. When specified (PostgreSQL ON CONFLICT (cols) style),
/// only violations on matching columns are handled.
fn conflict_matches_target(conflict_target: &[Identifier], schema: &Schema, error: &Error) -> bool {
    // Empty target = match all conflicts (MySQL semantics)
    if conflict_target.is_empty() {
        return true;
    }

    // Build normalized set of target column names
    let target_set: rustc_hash::FxHashSet<&str> = conflict_target
        .iter()
        .map(|id| id.value_lower.as_str())
        .collect();

    match error {
        Error::PrimaryKeyConstraint { .. } => {
            // Collect PK column names from schema
            let pk_cols: rustc_hash::FxHashSet<&str> = schema
                .columns
                .iter()
                .filter(|c| c.primary_key)
                .map(|c| c.name_lower.as_str())
                .collect();
            target_set == pk_cols
        }
        Error::UniqueConstraint { column, .. } => {
            // `column` is comma-separated (e.g. "a" or "a, b")
            let violated_cols: rustc_hash::FxHashSet<&str> =
                column.split(", ").map(|s| s.trim()).collect();
            // Compare lowercase: conflict_target is already lowered by Identifier::new;
            // UniqueConstraint.column comes from index column_names which are lowercase
            target_set == violated_cols
        }
        _ => false,
    }
}

/// Validate type coercion didn't silently fail.
/// Returns an error if a non-null value became null during coercion.
pub(crate) fn validate_coercion(
    original: &Value,
    coerced: &Value,
    column_name: &str,
    target_type: DataType,
    vector_dimensions: u16,
) -> Result<()> {
    // If original was non-null but coerced is null, the conversion failed
    if !original.is_null() && coerced.is_null() {
        return Err(Error::Type(format!(
            "cannot convert value '{}' to {:?} for column '{}'",
            original, target_type, column_name
        )));
    }
    // Validate vector dimension matches column definition
    if target_type == DataType::Vector {
        if let Value::Extension(data) = coerced {
            if data.first() == Some(&(DataType::Vector as u8)) {
                // Payload is packed LE f32 bytes after the tag byte
                let got_dim = u16::try_from((data.len() - 1) / 4).unwrap_or(u16::MAX);
                // vector_dimensions == 0 means unspecified, skip check
                if vector_dimensions > 0 && got_dim != vector_dimensions {
                    return Err(Error::VectorDimensionMismatch {
                        expected: vector_dimensions,
                        got: got_dim,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Try to extract a literal value directly from an expression without VM compilation.
/// Returns Some(value) for simple literals, None for complex expressions that need VM.
#[inline]
fn try_extract_literal(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
        Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
        Expression::StringLiteral(lit) => Some(super::utils::string_literal_value(lit)),
        Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
        Expression::NullLiteral(_) => Some(Value::null_unknown()),
        // Negative numbers: -5, -3.14
        // An overflow is left to the VM, which reports it
        Expression::Prefix(prefix) if prefix.operator == "-" => match prefix.right.as_ref() {
            Expression::IntegerLiteral(lit) => integer_neg(lit.value).ok().map(Value::Integer),
            Expression::FloatLiteral(lit) => Some(Value::Float(-lit.value)),
            _ => None,
        },
        _ => None, // Complex expression - needs VM
    }
}

/// Literal extraction plus direct parameter resolution: a bound $n is
/// already a Value in the context, and routing it through the program
/// cache costs an AST hash plus a global mutex per parameter per row.
#[inline]
fn try_extract_simple_value(expr: &Expression, ctx: &ExecutionContext) -> Option<Value> {
    if let Some(v) = try_extract_literal(expr) {
        return Some(v);
    }
    if let Expression::Parameter(param) = expr {
        if param.index > 0 {
            return ctx.params().get(param.index - 1).cloned();
        }
    }
    None
}

/// Evaluate one default slot for a row: run the per-row program when the
/// default is non-literal, else clone the constant template value.
pub(crate) fn eval_default_slot(
    vm: &mut super::expression::ExprVM,
    exec_ctx: &super::expression::ExecuteContext,
    program: &DefaultSlot,
    template_val: &Value,
) -> Result<Value> {
    Ok(match program {
        Some((Ok(program), target)) => vm
            .execute(program, exec_ctx)?
            .try_into_coerce_to_type(*target)?,
        Some((Err(e), _)) => return Err(e.clone()),
        None => template_val.clone(),
    })
}

/// A non-literal default's program, or the error it failed to compile
/// with, raised only when a row takes the default
pub(crate) type DefaultSlot = Option<(Result<super::expression::SharedProgram>, DataType)>;

use crate::parser::parse_stored_expression;

/// Per-statement default machinery: constant defaults land in the value
/// template; non-literal defaults compile ONCE and execute PER ROW, so
/// DEFAULT (RANDOM()) differs per row. A default that fails to parse or
/// compile is the error of a row that takes it, as in evaluate_default_expr.
pub(crate) fn build_default_slots(schema: &crate::core::Schema) -> (Vec<Value>, Vec<DefaultSlot>) {
    let registry = crate::functions::global_registry();
    let n = schema.columns.len();
    let mut values = Vec::with_capacity(n);
    let mut programs: Vec<DefaultSlot> = vec![None; n];
    for (i, c) in schema.columns.iter().enumerate() {
        let Some(expr_str) = c.default_expr.as_ref() else {
            values.push(Value::null_unknown());
            continue;
        };
        let target = c.data_type;
        values.push(Value::null_unknown());
        let expr = match parse_stored_expression(expr_str, "default", &c.name) {
            Ok(expr) => expr,
            Err(e) => {
                programs[i] = Some((Err(e), target));
                continue;
            }
        };
        // A literal outside the column's range runs as a program, whose error
        // a row that takes it raises
        if let Some(Ok(value)) =
            try_extract_literal(&expr).map(|v| v.try_into_coerce_to_type(target))
        {
            values[i] = value;
            continue;
        }
        let program =
            super::aggregation::compile_over_slots(&expr, &[], registry, None, false, None);
        programs[i] = Some((program, target));
    }
    (values, programs)
}

/// The defaults and VALUES programs of one INSERT, compiled once per plan
#[derive(Debug)]
pub(crate) struct InsertPrograms {
    pub(crate) default_values: Vec<Value>,
    pub(crate) default_programs: Vec<DefaultSlot>,
    /// The VALUES positions (row, value) a program computes, in order, each
    /// with its program's index; equal expressions share one program
    values: Vec<((usize, usize), usize)>,
    programs: Vec<Result<super::expression::SharedProgram>>,
    /// The RETURNING's programs, when it holds no subquery
    returning: Option<Vec<Result<super::expression::SharedProgram>>>,
}

impl InsertPrograms {
    /// A literal, a parameter and DEFAULT resolve without a program, and a
    /// subquery's answer is read before the rows
    pub(crate) fn of(schema: &Schema, stmt: &InsertStatement) -> Self {
        let (default_values, default_programs) = build_default_slots(schema);
        let mut values = Vec::new();
        let mut programs = Vec::new();
        let mut seen = FxHashMap::default();
        for (row, value_row) in stmt.values.iter().enumerate() {
            for (at, expr) in value_row.iter().enumerate() {
                if matches!(expr, Expression::Default(_) | Expression::Parameter(_))
                    || try_extract_literal(expr).is_some()
                    || Executor::has_subqueries(expr)
                {
                    continue;
                }
                let index = *seen
                    .entry(super::expr_key::ExprKey::of(expr))
                    .or_insert_with(|| {
                        programs.push(compile_value(expr));
                        programs.len() - 1
                    });
                values.push(((row, at), index));
            }
        }
        let returning = (!stmt.returning.is_empty()
            && !stmt.returning.iter().any(Executor::has_subqueries))
        .then(|| {
            let widened = returns_slot(&stmt.returning, schema);
            let columns = super::prepared_select::read_names(schema, widened);
            let rowid = schema.rowid_column();
            let at = rowid.or(widened.then_some(schema.columns.len()));
            returning_programs(
                &stmt.returning,
                &columns,
                at.map(|at| (at, schema.table_name.as_str())),
            )
        });
        Self {
            default_values,
            default_programs,
            values,
            programs,
            returning,
        }
    }

    pub(crate) fn returning(&self) -> Option<&[Result<super::expression::SharedProgram>]> {
        self.returning.as_deref()
    }

    /// The value at `at` of a VALUES row that neither a literal, a parameter
    /// nor an answer gives
    fn value(
        &self,
        vm: &mut super::expression::ExprVM,
        exec: &super::expression::ExecuteContext,
        at: (usize, usize),
        expr: &Expression,
    ) -> Result<Value> {
        match self.values.binary_search_by_key(&at, |(at, _)| *at) {
            Ok(found) => match &self.programs[self.values[found].1] {
                Ok(program) => vm.execute(program, exec),
                Err(e) => Err(e.clone()),
            },
            Err(_) => vm.execute(&*compile_value(expr)?, exec),
        }
    }
}

/// A VALUES expression's program: it reads no column
fn compile_value(expr: &Expression) -> Result<super::expression::SharedProgram> {
    super::aggregation::compile_over_slots(
        expr,
        &[],
        crate::functions::global_registry(),
        None,
        false,
        None,
    )
}

/// A RETURNING's programs over the target's `columns`, a star expanded, a
/// rowid read from the column `rowid` gives of the target it names
fn returning_programs(
    returning: &[Expression],
    columns: &[String],
    rowid: Option<(usize, &str)>,
) -> Vec<Result<super::expression::SharedProgram>> {
    use crate::parser::{Identifier, Position, Token, TokenType};
    let registry = crate::functions::global_registry();
    let compile = |expr: &Expression| {
        let expr = super::utils::rowids_respelled(expr, &mut |_| {
            Ok(rowid.map(|(at, table)| super::utils::qualified_reference(table, &columns[at])))
        })?;
        super::aggregation::compile_over_slots(&expr, columns, registry, None, false, None)
    };
    let mut programs = Vec::with_capacity(returning.len());
    for expr in returning {
        match expr {
            Expression::Star(_) => {
                programs.extend(columns.iter().filter(|c| *c != ROWID_SLOT).map(|name| {
                    let token =
                        Token::new(TokenType::Identifier, name.clone(), Position::new(0, 0, 0));
                    compile(&Expression::Identifier(Identifier::new(
                        token,
                        name.clone(),
                    )))
                }))
            }
            _ => programs.push(compile(expr)),
        }
    }
    programs
}

/// Compiled ON DUPLICATE KEY UPDATE clause, reused for every conflicting row
/// and kept in the statement's plan cache
pub(crate) struct CompiledUpsert {
    /// (column_index, column_type, vector_dimensions, compiled_program)
    compiled_updates: Vec<(usize, DataType, u16, super::expression::SharedProgram)>,
    /// DO UPDATE ... WHERE, read against the row met and the EXCLUDED row
    compiled_where: Option<super::expression::SharedProgram>,
    /// CHECK constraints of the updated columns
    compiled_checks: Vec<CompiledCheck>,
    /// Whether the programs read the row met with its row id past its columns
    widens: bool,
}

/// An ON CONFLICT DO UPDATE whose programs are prepared: `upsert` holds
/// them, over the row met joined with the excluded row, and `runner`
/// answers their subqueries
struct ConflictRun<'r, 'a> {
    runner: &'r super::prepared_select::DmlRunner<'a>,
    upsert: Arc<CompiledUpsert>,
    /// The row met and the excluded row as one row, reused across conflicts
    joined: std::cell::RefCell<Row>,
}

/// The answers of an INSERT's VALUES subqueries, each at its row and position
type ValueAnswers = SmallVec<[((usize, usize), Value); 4]>;

/// The CHECK programs of a table without CHECK constraints, shared
static NO_CHECKS: std::sync::LazyLock<Arc<Vec<CompiledCheck>>> =
    std::sync::LazyLock::new(|| Arc::new(Vec::new()));

/// One INSERT's CHECK programs and ON DUPLICATE KEY UPDATE clause: taken from
/// the plan cache when compiled against this table layout, else compiled
/// before the first write and cached
struct InsertPlan<'a> {
    checks: Arc<Vec<CompiledCheck>>,
    programs: Arc<InsertPrograms>,
    upsert: Option<Arc<CompiledUpsert>>,
    cache: Option<&'a RwLock<CompiledExecution>>,
    table_name: &'a str,
    /// When what this compiled holds; None for what the cache held already
    validity: Option<PlanValidity>,
    token: LayoutToken,
}

impl<'a> InsertPlan<'a> {
    /// `epoch` read before the table was opened
    fn new(
        cache: Option<&'a RwLock<CompiledExecution>>,
        table_name: &'a str,
        (executor, epoch): (&Executor, u64),
        token: LayoutToken,
        schema: &Schema,
        stmt: &InsertStatement,
        prepared: Option<Arc<CompiledUpsert>>,
    ) -> Result<Self> {
        let cached = cache.and_then(|cache| match &*cache.read().ok()? {
            CompiledExecution::ConflictInsert(plan)
                if plan.validity.holds(&*executor.engine, epoch)
                    && plan.token == token
                    && *plan.table_name == *table_name =>
            {
                Some((
                    Arc::clone(&plan.checks),
                    Arc::clone(&plan.programs),
                    plan.upsert.clone(),
                ))
            }
            _ => None,
        });
        let hit = cached.is_some();
        let (checks, programs, upsert) = match cached {
            Some(parts) => parts,
            None => {
                let checks = compile_check_programs(schema)?;
                let checks = if checks.is_empty() {
                    Arc::clone(&NO_CHECKS)
                } else {
                    Arc::new(checks)
                };
                // Compiled before the first write, so an invalid clause
                // fails the statement even when no row conflicts
                let upsert = match prepared {
                    Some(prepared) => Some(prepared),
                    None if stmt.on_duplicate => {
                        Some(Arc::new(Executor::compile_upsert(schema, stmt)?))
                    }
                    None => None,
                };
                (checks, Arc::new(InsertPrograms::of(schema, stmt)), upsert)
            }
        };
        let plan = Self {
            checks,
            programs,
            upsert,
            cache,
            table_name,
            validity: (!hit && cache.is_some()).then(|| executor.insert_validity(stmt, epoch)),
            token,
        };
        if !hit {
            plan.store();
        }
        Ok(plan)
    }

    fn upsert(&mut self, schema: &Schema, stmt: &InsertStatement) -> Result<Arc<CompiledUpsert>> {
        if let Some(upsert) = &self.upsert {
            return Ok(Arc::clone(upsert));
        }
        let upsert = Arc::new(Executor::compile_upsert(schema, stmt)?);
        self.upsert = Some(Arc::clone(&upsert));
        self.store();
        Ok(upsert)
    }

    fn store(&self) {
        let Some(validity) = &self.validity else {
            return;
        };
        if let Some(Ok(mut slot)) = self.cache.map(|cache| cache.write()) {
            *slot = CompiledExecution::ConflictInsert(CompiledConflictInsert {
                checks: Arc::clone(&self.checks),
                programs: Arc::clone(&self.programs),
                upsert: self.upsert.clone(),
                table_name: SmartString::new(self.table_name),
                validity: validity.clone(),
                token: self.token,
            });
        }
    }
}

/// A CHECK constraint compiled once over the table's whole row: a column's,
/// or the table's own when `column` is None
#[derive(Clone, Debug)]
pub(crate) struct CompiledCheck {
    pub(crate) column: Option<SmartString>,
    pub(crate) table: SmartString,
    pub(crate) text: SmartString,
    /// The columns it reads, which a write into the row may change
    pub(crate) reads: SmallVec<[usize; 4]>,
    /// Whether its answer may change with no column changing (NOW, RANDOM)
    pub(crate) volatile: bool,
    pub(crate) program: super::expression::SharedProgram,
}

/// Whether an INSERT's RETURNING reads the row ids of a table without a
/// primary key, which its rows returned then carry past their columns
fn returns_slot(returning: &[Expression], schema: &Schema) -> bool {
    schema.rowid_column().is_none() && returning.iter().any(super::utils::expr_names_rowid)
}

/// `row` inserted into `table` as a RETURNING reads it: with its row id
/// past its columns when `widened`
fn inserted(table: &mut dyn Table, row: Row, widened: bool) -> Result<Row> {
    match widened {
        true => table
            .insert_with_id(row)
            .map(|(id, row)| super::prepared_select::widened(row, id)),
        false => table.insert(row),
    }
}

/// An ON CONFLICT DO UPDATE's SET values, each with the column it writes,
/// and its WHERE
type ConflictClauses = (Vec<(usize, Expression)>, Option<Expression>);

/// The column a statement writes under the lowercase `name`, None for no
/// column; an error for rowid, the row's identity, which nothing writes
pub(crate) fn written_column(schema: &Schema, name: &str) -> Result<Option<usize>> {
    match schema.column_index_map().get(name) {
        Some(&at) => Ok(Some(at)),
        None if crate::core::is_rowid_name(name) => Err(Error::invalid_argument(format!(
            "cannot write {name}: it is the row's identity"
        ))),
        None => Ok(None),
    }
}

/// Moves each row of `moves`, (old row id, row), to the id its primary key
/// now holds: every old id is deleted first, then every row inserted, so a
/// key collides only with a row the statement leaves or with another new
/// key, whatever order the rows were set in; a NULL key is refused before
/// an insert could generate one
pub(crate) fn move_rows(table: &mut dyn Table, moves: Vec<(i64, Row)>) -> Result<()> {
    if moves.is_empty() {
        return Ok(());
    }
    let schema = table.schema();
    if let Some(pk) = schema.pk_column_index() {
        if moves
            .iter()
            .any(|(_, row)| row.get(pk).is_none_or(Value::is_null))
        {
            return Err(Error::not_null_constraint(&schema.columns[pk].name));
        }
    }
    let ids: Vec<i64> = moves.iter().map(|(id, _)| *id).collect();
    table.delete_by_row_ids(&ids)?;
    for (_, row) in moves {
        table.insert_discard(row)?;
    }
    Ok(())
}

/// Compile every CHECK constraint of `schema`, the columns' and the table's.
///
/// Parsing and compiling a constraint costs dozens of allocations. The COPY
/// path used to do it again for every row, which made a CHECK the dominant
/// cost of a bulk load. A constraint that fails to compile is an error rather
/// than a silently skipped check.
pub(crate) fn compile_check_programs(schema: &Schema) -> Result<Vec<CompiledCheck>> {
    if schema.table_checks.is_empty() && schema.columns.iter().all(|c| c.check_expr.is_none()) {
        return Ok(Vec::new());
    }
    let columns: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
    let column_checks = schema.columns.iter().filter_map(|c| {
        c.check_expr
            .as_ref()
            .map(|text| (Some(c.name.as_str()), text))
    });
    let table_checks = schema.table_checks.iter().map(|text| (None, text));
    column_checks
        .chain(table_checks)
        .map(|(column, text)| compile_check(schema, &columns, column, text))
        .collect()
}

/// The CHECK constraints of `checks` a write of `columns` can change the
/// answer of: those reading one of them, and every volatile one
pub(crate) fn checks_reading(
    mut checks: Vec<CompiledCheck>,
    columns: impl Fn(usize) -> bool,
) -> Vec<CompiledCheck> {
    checks.retain(|check| check.volatile || check.reads.iter().any(|&c| columns(c)));
    checks
}

fn compile_check(
    schema: &Schema,
    columns: &[String],
    column: Option<&str>,
    text: &str,
) -> Result<CompiledCheck> {
    let owner = column.unwrap_or(&schema.table_name);
    let expr = parse_stored_expression(text, "CHECK", owner)?;
    let binding = super::expression::CompileContext::with_global_registry(columns);
    let program = super::expression::ExprCompiler::new(&binding)
        .compile(&expr)
        .map(CompactArc::new)
        .map_err(|e| {
            Error::internal(format!(
                "the CHECK on {owner} failed to compile: Compile error: {e}"
            ))
        })?;
    let mut reads = SmallVec::new();
    super::utils::visit_column_refs(&expr, &mut |reference| {
        let read = binding.reference(reference).and_then(|r| r.row_column());
        if let Some(i) = read.map(usize::from) {
            if !reads.contains(&i) {
                reads.push(i);
            }
        }
    });
    let registry = crate::functions::registry::global_registry();
    Ok(CompiledCheck {
        column: column.map(SmartString::from),
        table: SmartString::from(schema.table_name.as_str()),
        text: SmartString::from(text),
        reads,
        volatile: super::volatility::expression_volatility(&expr, registry)
            != crate::functions::Volatility::Fixed,
        program,
    })
}

/// Run `check` over the row it guards. A NULL result passes, per the standard
pub(crate) fn run_check(
    vm: &mut super::expression::ExprVM,
    check: &CompiledCheck,
    row: &Row,
) -> Result<()> {
    let result = vm.execute(&check.program, &super::expression::ExecuteContext::new(row))?;
    let passes = match &result {
        Value::Boolean(b) => *b,
        Value::Null(_) => true,
        Value::Integer(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::Text(t) => !t.is_empty(),
        _ => true, // Timestamp, Extension, etc. are truthy
    };
    if passes {
        return Ok(());
    }
    Err(match &check.column {
        Some(column) => Error::CheckConstraintViolation {
            column: column.to_string(),
            expression: check.text.to_string(),
        },
        None => Error::TableCheckViolation {
            table: check.table.to_string(),
            expression: check.text.to_string(),
        },
    })
}

/// `run_check` over a row still held as `values`, copying the columns the
/// check reads into `scratch`, a row of the table's width built on the
/// statement's first CHECK and reused row to row
pub(crate) fn run_check_on_values(
    vm: &mut super::expression::ExprVM,
    scratch: &mut Option<Row>,
    check: &CompiledCheck,
    values: &[Value],
) -> Result<()> {
    let scratch =
        scratch.get_or_insert_with(|| Row::from_values(vec![Value::null_unknown(); values.len()]));
    for &column in &check.reads {
        scratch.set(column, values[column].clone())?;
    }
    run_check(vm, check, scratch)
}

impl Executor {
    /// Whether the INSERT's VALUES, ON CONFLICT DO UPDATE or RETURNING hold
    /// a subquery of the prepared family
    #[inline(never)]
    fn insert_reads_prepared(
        &self,
        stmt: &InsertStatement,
        ctx: &ExecutionContext,
        plan_cache: Option<&RwLock<CompiledExecution>>,
    ) -> bool {
        let values = || {
            stmt.values
                .iter()
                .flatten()
                .filter(|e| !matches!(e, Expression::Default(_)))
        };
        let conflict = || {
            stmt.update_expressions
                .iter()
                .chain(stmt.update_where.as_deref())
        };
        let returning = || {
            stmt.returning
                .iter()
                .any(Self::has_subqueries)
                .then(|| super::prepared_select::returning_values(&stmt.returning))
                .into_iter()
                .flatten()
        };
        (values().any(Self::has_subqueries)
            || conflict().any(Self::has_subqueries)
            || returning().next().is_some())
            && (self.holds_prepared_dml(plan_cache, ctx)
                || self
                    .dml_family(None, values().chain(conflict()).chain(returning()))
                    .is_ok())
    }

    /// The refusal of a RETURNING value read for each row returned whose
    /// block reads the target inside a transaction
    fn returning_reads_target() -> Error {
        Error::invalid_argument(
            "a RETURNING subquery that reads the row returned cannot read the target table \
             inside a transaction",
        )
    }

    /// Refuse an INSERT whose block read for each conflict or row returned
    /// reads the target inside a transaction, where it would see the
    /// statement's writes
    fn refuse_target_reads_in_transaction(
        run: &super::prepared_select::DmlRunner,
        in_transaction: bool,
    ) -> Result<()> {
        if in_transaction && run.conflict_reads_target() {
            return Err(Error::invalid_argument(
                "an ON CONFLICT subquery that reads the row met or the excluded row \
                 cannot read the target table inside a transaction",
            ));
        }
        if in_transaction && run.returning_reads_target() {
            return Err(Self::returning_reads_target());
        }
        Ok(())
    }

    /// The answer of each VALUES subquery with its row and position, all
    /// read before the statement writes a row
    fn insert_value_answers(run: &super::prepared_select::DmlRunner) -> Result<ValueAnswers> {
        let mut answers = ValueAnswers::new();
        if !run.values().iter().flatten().any(Option::is_some) {
            return Ok(answers);
        }
        let mut vm = super::expression::ExprVM::new();
        let no_row = Row::new();
        let exec = run.context(&no_row);
        for (row, programs) in run.values().iter().enumerate() {
            for (at, program) in programs.iter().enumerate() {
                if let Some(program) = program {
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::prepared_dml_site_ran(
                        crate::test_failpoints::PreparedDmlSite::InsertValues,
                    );
                    answers.push(((row, at), vm.execute(program, &exec)?));
                }
            }
        }
        Ok(answers)
    }

    /// Execute an INSERT statement
    pub(crate) fn execute_insert(
        &self,
        stmt: &InsertStatement,
        ctx: &ExecutionContext,
        plan: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        let plan_cache = plan.map(NestedPlan::slot);
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;
        let reads_through_view = super::insert_reads_through_a_view(stmt);
        let reads_prepared =
            reads_through_view && self.insert_reads_prepared(stmt, ctx, plan_cache);

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();

        // INSERT is dispatched before the dispatcher puts the transaction
        // id in the context, so the expressions are given it here
        let active_txn_id = (*active_tx).as_ref().map(|t| t.transaction.id());
        let ctx_with_txn;
        let ctx = match active_txn_id {
            Some(txn_id) => {
                ctx_with_txn = ctx.with_transaction_id(txn_id as u64);
                &ctx_with_txn
            }
            None => ctx,
        };
        // Its query and subqueries read through one view; the target keeps the newest
        let ctx_with_view;
        let ctx = match self.statement_view(reads_through_view, ctx, active_txn_id) {
            Some(view) => {
                ctx_with_view = ctx.with_statement_view(view);
                &ctx_with_view
            }
            None => ctx,
        };
        // Without a view the prepared subqueries read a snapshot taken before the write
        let snapshot_ctx;
        let ctx = if reads_prepared
            && active_txn_id.is_none()
            && ctx.statement_view().is_none()
            && ctx.statement_snapshot().is_none()
        {
            let snapshot = self.new_statement_snapshot(self.isolation_level(), None)?;
            snapshot_ctx = ctx.with_statement_snapshot(snapshot);
            &snapshot_ctx
        } else {
            ctx
        };

        let statement_rollback = StatementRollback::new(&self.engine, active_tx.as_ref());

        // Read before the table: a change since leaves a plan made now behind
        let plan_epoch = self.engine.schema_epoch();
        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                // NOTE: table_name is already lowercase (value_lower from AST)
                let table = tx_state.transaction.get_statement_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name.as_str()) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_statement_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.begin_transaction()?;
                let table = tx.get_statement_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::dml_table_opened();

        // Prepared subqueries are read before the first write, through this
        // statement's transaction; a commit-time retry runs the statement again
        let retry_of = (stmt, ctx, plan);
        let prepared;
        let mut runner = None;
        let conflict_run;
        let written_ctx;
        let written;
        let (stmt, ctx, plan_cache, conflict, returned) = if reads_prepared {
            let target = table.schema_arc();
            let conflict_reads = stmt.on_duplicate
                && stmt
                    .update_expressions
                    .iter()
                    .chain(stmt.update_where.as_deref())
                    .any(Self::has_subqueries);
            let sites = conflict_reads.then(|| super::prepared_select::ConflictSites {
                table: table_name,
                schema: &target,
                stmt,
            });
            let returning = stmt.returning.iter().any(Self::has_subqueries).then(|| {
                super::prepared_select::ReturningSites {
                    table: table_name,
                    schema: &target,
                    values: &stmt.returning,
                }
            });
            // A VALUES row has no target row: its scope is a table without columns
            let (plan, tables) = self.open_prepared_dml(
                plan_cache,
                &super::prepared_select::NO_COLUMNS,
                table_name,
                None,
                &[],
                &stmt.values,
                sites.as_ref(),
                returning.as_ref(),
                standalone_tx
                    .as_deref()
                    .filter(|_| ctx.statement_snapshot().is_none()),
                ctx,
            )?;
            prepared = plan;
            let run = &*runner.insert(super::prepared_select::DmlRunner::new(
                self, &prepared, ctx, tables, 0,
            )?);
            Self::refuse_target_reads_in_transaction(run, active_txn_id.is_some())?;
            // The statement is copied only for the VALUES answers it takes
            let mut answered_ctx = std::borrow::Cow::Borrowed(ctx);
            let mut answered = std::borrow::Cow::Borrowed(stmt);
            let answers = Self::insert_value_answers(run)?;
            if !answers.is_empty() {
                let (answered_ctx, answered) = (answered_ctx.to_mut(), answered.to_mut());
                for ((row, at), answer) in answers {
                    answered_ctx.add_param(answer);
                    answered.values[row][at] = answer_parameter(answered_ctx.params().len());
                }
            }
            written_ctx = answered_ctx;
            written = answered;
            let returned = run.returning().map(|_| run);
            match run.conflict_upsert() {
                Some(upsert) => {
                    conflict_run = ConflictRun {
                        runner: run,
                        upsert: Arc::clone(upsert),
                        joined: std::cell::RefCell::new(Row::with_capacity(
                            2 * target.columns.len(),
                        )),
                    };
                    (
                        &*written,
                        &*written_ctx,
                        None,
                        Some(&conflict_run),
                        returned,
                    )
                }
                None if returned.is_none() => {
                    // Every value is read: the blocks' tables close before a row is written
                    drop(runner.take());
                    (&*written, &*written_ctx, None, None, None)
                }
                None => (&*written, &*written_ctx, None, None, returned),
            }
        } else {
            (stmt, ctx, plan_cache, None, None)
        };

        // Acquire per-table upsert mutex for ON CONFLICT DO UPDATE only.
        // Serializes check+insert+commit to prevent TOCTOU races where two
        // concurrent upserts both detect the same conflict and try to UPDATE
        // the same row (second would hit an unhandled WriteConflict).
        // DO NOTHING needs no lock: it never writes to the conflicting row,
        // and commit-time unique violations are handled by returning 0 rows.
        // Plain INSERTs proceed lock-free (UniqueConstraint error is correct).
        let _upsert_guard = if stmt.on_duplicate {
            table.acquire_upsert_lock()
        } else {
            None
        };

        // Pre-compute schema information to avoid repeated borrows during insert
        let schema_column_count: usize;
        let column_indices: Vec<usize>;
        // Pre-compute column types for type coercion
        let column_types: Vec<crate::core::DataType>;
        // Pre-compute vector dimensions for vector columns (0 for non-vector)
        let column_vector_dims: Vec<u16>;
        // Pre-compute column names for error messages
        let column_names: Vec<String>;
        // Pre-compute ALL column types for default values and check constraints
        let all_column_types: Vec<crate::core::DataType>;
        let programs: Arc<InsertPrograms>;
        let mut insert_plan: InsertPlan<'_>;
        let check_programs: Arc<Vec<CompiledCheck>>;
        let mut check_vm = super::expression::ExprVM::new();
        {
            let schema = table.schema();
            schema_column_count = schema.columns.len();

            // Only a conflict-handling INSERT keeps its plan here
            insert_plan = InsertPlan::new(
                plan_cache.filter(|_| stmt.on_duplicate || stmt.do_nothing),
                table_name,
                (self, plan_epoch),
                table.layout_token(),
                schema,
                stmt,
                conflict.map(|conflict| Arc::clone(&conflict.upsert)),
            )?;
            check_programs = Arc::clone(&insert_plan.checks);
            programs = Arc::clone(&insert_plan.programs);
            all_column_types = schema.columns.iter().map(|c| c.data_type).collect();

            // OPTIMIZATION: When no columns specified, insert into all columns in order
            // Skip all column name lookups - just use sequential indices
            if stmt.columns.is_empty() {
                column_indices = (0..schema_column_count).collect();
                column_types = all_column_types.clone();
                column_vector_dims = schema.columns.iter().map(|c| c.vector_dimensions).collect();
                // Use schema's cached column names - avoids re-collecting on every INSERT
                column_names = schema.column_names_owned().to_vec();
            } else {
                // Validate columns exist and pre-compute their indices
                // OPTIMIZATION: Use cached column_index_map for O(1) lookups instead of O(n) linear scan
                column_indices = stmt
                    .columns
                    .iter()
                    .map(|id| {
                        // Use pre-computed lowercase value from AST
                        written_column(schema, id.value_lower.as_str())?
                            .ok_or_else(|| Error::ColumnNotFound(id.value.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                // Get column types for the specified columns
                column_types = column_indices
                    .iter()
                    .map(|&idx| schema.columns[idx].data_type)
                    .collect();
                column_vector_dims = column_indices
                    .iter()
                    .map(|&idx| schema.columns[idx].vector_dimensions)
                    .collect();
                // Get column names for error messages
                column_names = column_indices
                    .iter()
                    .map(|&idx| schema.columns[idx].name.clone())
                    .collect();
            }
        }
        let (default_values, default_programs) =
            (&programs.default_values, &programs.default_programs);
        // Columns the INSERT names; only the rest run their defaults. Inline,
        // so a statement allocates nothing for it
        let mut provided: smallvec::SmallVec<[bool; 32]> =
            smallvec::smallvec![false; schema_column_count];
        for &idx in &column_indices {
            provided[idx] = true;
        }

        // Pre-compute FK info for parent validation (CompactArc ref-count bump, not deep clone)
        let fk_schema = if !table.schema().foreign_keys.is_empty() {
            Some(table.schema_arc())
        } else {
            None
        };
        // Keys into this table whose parent is a later row of the statement
        let mut pending_parents = super::foreign_key::PendingParents::new(table_name);

        // Create VM for constant expression evaluation (reused for all INSERT values)
        use super::expression::{ExecuteContext, ExprVM};
        let mut vm = ExprVM::new();
        let mut check_row = None;
        let params = ctx.params();
        let empty_row = Row::new();

        // OPTIMIZATION: Pre-build ExecuteContext once (reused for all expressions)
        let base_exec_ctx =
            ExecuteContext::new(&empty_row).with_run(params, ctx.transaction_id(), None);

        let mut rows_affected = 0i64;

        // RETURNING clause support - collect inserted rows if RETURNING is specified
        let has_returning = !stmt.returning.is_empty();
        let widened = has_returning && returns_slot(&stmt.returning, table.schema());
        let mut returning_rows: Vec<Row> = Vec::new();
        // OPTIMIZATION: Only get column names Arc when RETURNING is used (avoids 7ms clone)
        let schema_column_names_arc = if has_returning {
            Some(table.schema().column_names_arc())
        } else {
            None
        };

        // Check if this is INSERT ... SELECT
        if let Some(ref select_stmt) = stmt.select {
            // Get schema for conflict handling (needed for duplicate row lookup and target matching)
            let select_schema = if stmt.on_duplicate || stmt.do_nothing {
                Some(table.schema_arc())
            } else {
                None
            };

            // For explicit transactions, materialize the SELECT BEFORE any inserts
            // to ensure statement atomicity: a late runtime error (e.g., invalid REGEXP)
            // won't leave partial inserts pending for commit. Auto-commit transactions
            // stream rows directly since the standalone transaction rolls back on drop.
            // Run as a parsed tree, the SELECT holds the plan its structure
            // keys, as a SELECT run alone does
            let tree_plan = match plan {
                Some(_) => None,
                None => self.query_cache.statement_plan(select_stmt),
            };
            let source = plan.and_then(NestedPlan::source).or(tree_plan.as_deref());
            let mut select_result =
                self.execute_select_with_plan(select_stmt, ctx, None, source)?;
            if !should_auto_commit {
                // Explicit tx: fully materialize before writes for atomicity.
                // A late runtime error (e.g., invalid REGEXP) will fail here
                // before any inserts, preventing partial writes in the transaction.
                let columns = select_result.columns().to_vec();
                let rows = Self::materialize_result(select_result)?;
                select_result = Box::new(super::result::ExecutorResult::new(columns, rows));
            }

            // Process each row from the SELECT result (streaming or materialized)
            while select_result.next() {
                let select_row = select_result.row();
                if select_row.len() != column_indices.len() {
                    return Err(Error::InvalidArgument(format!(
                        "INSERT has {} columns but SELECT returns {} columns",
                        column_indices.len(),
                        select_row.len()
                    )));
                }

                // Build row values - initialize with DEFAULT values for missing columns
                // This matches the behavior of regular INSERT
                let mut row_values = Vec::with_capacity(schema_column_count);
                for i in 0..schema_column_count {
                    // A named column is filled below; a default it would
                    // replace must not run, so it cannot fail the row
                    row_values.push(if provided[i] {
                        Value::null_unknown()
                    } else {
                        eval_default_slot(
                            &mut vm,
                            &base_exec_ctx,
                            &default_programs[i],
                            &default_values[i],
                        )?
                    });
                }

                // Fill in values from SELECT using pre-computed indices with type coercion
                for (i, value) in select_row.iter().enumerate() {
                    // Coerce value to target column type
                    let coerced = value.coerce_to_type(column_types[i]);
                    // Validate coercion didn't silently fail
                    validate_coercion(
                        value,
                        &coerced,
                        &column_names[i],
                        column_types[i],
                        column_vector_dims[i],
                    )?;
                    row_values[column_indices[i]] = coerced;
                }

                // Validate CHECK constraints (must happen before insert)
                for check in check_programs.iter() {
                    run_check_on_values(&mut check_vm, &mut check_row, check, &row_values)?;
                }

                // Save row_values for ON DUPLICATE KEY UPDATE before moving into Row
                let saved_row_values = if stmt.on_duplicate {
                    Some(row_values.clone())
                } else {
                    None
                };
                let row = Row::from_values(row_values);

                // FK parent validation (zero-cost if no FKs)
                if let Some(ref fks) = fk_schema {
                    super::foreign_key::check_inserted_parents(
                        &self.engine,
                        table.txn_id(),
                        fks,
                        &row,
                        &mut pending_parents,
                    )?;
                }

                if stmt.do_nothing {
                    let schema_ref = select_schema.as_ref().unwrap();
                    // ON CONFLICT DO NOTHING — silently skip duplicates
                    let insert_result: Result<Option<Row>> = if has_returning {
                        inserted(&mut *table, row, widened).map(Some)
                    } else {
                        table.insert_discard(row).map(|()| None)
                    };
                    match insert_result {
                        Ok(opt_row) => {
                            if let Some(inserted) = opt_row {
                                returning_rows.push(inserted);
                            }
                            rows_affected += 1;
                        }
                        Err(e @ Error::PrimaryKeyConstraint { .. })
                        | Err(e @ Error::UniqueConstraint { .. }) => {
                            if !conflict_matches_target(&stmt.conflict_target, schema_ref, &e) {
                                return Err(e);
                            }
                            // DO NOTHING: conflict skipped, no RETURNING row
                        }
                        Err(e) => return Err(e),
                    }
                } else if stmt.on_duplicate {
                    let row_values = saved_row_values.as_ref().unwrap();
                    let schema_ref = select_schema.as_ref().unwrap();
                    // ON CONFLICT DO UPDATE / ON DUPLICATE KEY UPDATE
                    let insert_result: Result<Option<Row>> = if has_returning {
                        inserted(&mut *table, row, widened).map(Some)
                    } else {
                        table.insert_discard(row).map(|()| None)
                    };
                    match insert_result {
                        Ok(opt_row) => {
                            if let Some(inserted) = opt_row {
                                returning_rows.push(inserted);
                            }
                            rows_affected += 1;
                        }
                        Err(ref e @ Error::PrimaryKeyConstraint { row_id }) => {
                            if !conflict_matches_target(&stmt.conflict_target, schema_ref, e) {
                                return Err(Error::PrimaryKeyConstraint { row_id });
                            }
                            match self.apply_on_duplicate_update(
                                &mut table,
                                schema_ref,
                                row_id,
                                None,
                                row_values,
                                stmt,
                                &mut insert_plan,
                                ctx,
                                (has_returning, widened),
                                conflict,
                                &mut pending_parents,
                            ) {
                                Ok((updated, returned)) => {
                                    if let Some(updated_row) = returned {
                                        returning_rows.push(updated_row);
                                    }
                                    if updated {
                                        rows_affected += 1;
                                    }
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        Err(mut e @ Error::UniqueConstraint { .. }) => {
                            if !conflict_matches_target(&stmt.conflict_target, schema_ref, &e) {
                                return Err(e);
                            }
                            let Error::UniqueConstraint {
                                index,
                                column,
                                row_id: conflict_rid,
                                ..
                            } = &e
                            else {
                                return Err(e);
                            };
                            let conflict_rid = *conflict_rid;
                            // Use row_id from the error if available (cold segment check
                            // already found it). Only fall back to re-search if row_id < 0
                            // (hot index path sets row_id = -1 when unknown).
                            let found_row_id = if conflict_rid >= 0 {
                                Ok(Some(conflict_rid))
                            } else {
                                self.find_row_by_unique_index(
                                    &*table, schema_ref, index, column, row_values,
                                )
                            };
                            match found_row_id {
                                Ok(Some(row_id)) => {
                                    match self.apply_on_duplicate_update(
                                        &mut table,
                                        schema_ref,
                                        row_id,
                                        Some(column),
                                        row_values,
                                        stmt,
                                        &mut insert_plan,
                                        ctx,
                                        (has_returning, widened),
                                        conflict,
                                        &mut pending_parents,
                                    ) {
                                        Ok((updated, returned)) => {
                                            if let Some(updated_row) = returned {
                                                returning_rows.push(updated_row);
                                            }
                                            if updated {
                                                rows_affected += 1;
                                            }
                                        }
                                        Err(e) => return Err(e),
                                    }
                                }
                                Ok(None) => {
                                    if let Error::UniqueConstraint { row_id, .. } = &mut e {
                                        *row_id = -1;
                                    }
                                    return Err(e);
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        Err(e) => return Err(e),
                    }
                } else if has_returning {
                    // Use insert() which returns row (for RETURNING clause)
                    let inserted_row = inserted(&mut *table, row, widened)?;
                    returning_rows.push(inserted_row);
                    rows_affected += 1;
                } else {
                    // Use insert_discard() to avoid clone overhead
                    table.insert_discard(row)?;
                    rows_affected += 1;
                }
            }
            // For streaming (auto-commit) path, check for runtime filter errors
            if let Some(err) = select_result.last_error() {
                return Err(err);
            }
            pending_parents.check(&self.engine, table.txn_id())?;

            // Invalidate semantic cache for this table BEFORE commit
            // CRITICAL: Must invalidate before commit to prevent stale data window
            // where concurrent queries could see new data in storage but get old cached results
            if rows_affected > 0 {
                self.semantic_cache.invalidate_table(table_name);
            }

            let mut returning_result = if let Some(column_names) = &schema_column_names_arc {
                Some(self.build_returning_result(
                    &stmt.returning,
                    returning_rows,
                    column_names,
                    ctx,
                    returned,
                    programs.returning(),
                )?)
            } else {
                None
            };

            // A statement whose reads a truncate or drop overtook commits nothing
            ctx.view_held()?;
            // Commit if this is a standalone (auto-commit) transaction
            if should_auto_commit {
                if let Some(mut tx) = standalone_tx {
                    match tx.commit() {
                        Ok(()) => {}
                        Err(e)
                            if (stmt.on_duplicate || stmt.do_nothing)
                                && e.is_pk_or_unique_violation() =>
                        {
                            if stmt.on_duplicate && ctx.query_depth == 0 {
                                // Commit-time PK/unique violation during upsert:
                                // a concurrent plain INSERT committed first. Retry once.
                                drop(returning_result);
                                drop(_upsert_guard);
                                let (stmt, ctx, plan) = retry_of;
                                let retry_ctx = ctx.with_incremented_query_depth();
                                return self.execute_insert(stmt, &retry_ctx, plan);
                            }
                            if stmt.do_nothing {
                                // DO NOTHING: returning 0 rows is the correct semantic
                                rows_affected = 0;
                                if let Some(result) = &mut returning_result {
                                    *result = Box::new(ExecutorResult::with_columns(
                                        result.columns().to_vec(),
                                    ));
                                }
                            } else {
                                return Err(e);
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
            }

            if let Some(result) = returning_result {
                return statement_rollback.finish(Ok(result));
            }

            return statement_rollback
                .finish(Ok(Box::new(ExecResult::with_rows_affected(rows_affected))));
        }

        // Process each row of values - use fast path for normal INSERT, slow path for conflict handling
        if stmt.do_nothing || stmt.on_duplicate {
            // ON DUPLICATE KEY UPDATE requires schema (CompactArc ref-count bump, not deep clone)
            let schema = table.schema_arc();

            for (row_at, value_row) in stmt.values.iter().enumerate() {
                if value_row.len() != column_indices.len() {
                    return Err(Error::InvalidArgument(format!(
                        "INSERT has {} columns but {} values",
                        column_indices.len(),
                        value_row.len()
                    )));
                }

                // Build row values - need Vec for error handling paths
                let mut row_values = Vec::with_capacity(schema_column_count);
                for i in 0..schema_column_count {
                    // A named column is filled below; a default it would
                    // replace must not run, so it cannot fail the row
                    row_values.push(if provided[i] {
                        Value::null_unknown()
                    } else {
                        eval_default_slot(
                            &mut vm,
                            &base_exec_ctx,
                            &default_programs[i],
                            &default_values[i],
                        )?
                    });
                }
                // Fill in provided values using pre-computed indices with type coercion
                for (i, expr) in value_row.iter().enumerate() {
                    // DEFAULT keyword: the column's default, run now
                    if matches!(expr, Expression::Default(_)) {
                        let idx = column_indices[i];
                        row_values[idx] = eval_default_slot(
                            &mut vm,
                            &base_exec_ctx,
                            &default_programs[idx],
                            &default_values[idx],
                        )?;
                        continue;
                    }
                    // OPTIMIZATION: Try to extract literal value directly without VM compilation
                    // This avoids ~1-2μs per expression for simple literals (INTEGER, TEXT, etc.)
                    let value = match try_extract_simple_value(expr, ctx) {
                        Some(value) => value,
                        None => programs.value(&mut vm, &base_exec_ctx, (row_at, i), expr)?,
                    };
                    // Coerce to target type
                    let coerced = value.coerce_to_type(column_types[i]);
                    // Validate coercion didn't silently fail
                    validate_coercion(
                        &value,
                        &coerced,
                        &column_names[i],
                        column_types[i],
                        column_vector_dims[i],
                    )?;
                    row_values[column_indices[i]] = coerced;
                }

                // Validate CHECK constraints (must happen before insert)
                for check in check_programs.iter() {
                    run_check_on_values(&mut check_vm, &mut check_row, check, &row_values)?;
                }

                // Create row from values (ON DUPLICATE KEY needs values for error handling)
                let row = Row::from_values(row_values.clone());

                // FK parent validation (zero-cost if no FKs)
                if let Some(ref fks) = fk_schema {
                    super::foreign_key::check_inserted_parents(
                        &self.engine,
                        table.txn_id(),
                        fks,
                        &row,
                        &mut pending_parents,
                    )?;
                }

                if stmt.do_nothing {
                    // ON CONFLICT DO NOTHING — silently skip duplicates
                    let insert_result: Result<Option<Row>> = if has_returning {
                        inserted(&mut *table, row, widened).map(Some)
                    } else {
                        table.insert_discard(row).map(|()| None)
                    };
                    match insert_result {
                        Ok(opt_row) => {
                            if let Some(inserted) = opt_row {
                                returning_rows.push(inserted);
                            }
                            rows_affected += 1;
                        }
                        Err(e @ Error::PrimaryKeyConstraint { .. })
                        | Err(e @ Error::UniqueConstraint { .. }) => {
                            if !conflict_matches_target(&stmt.conflict_target, &schema, &e) {
                                return Err(e);
                            }
                            // DO NOTHING: conflict skipped, no RETURNING row
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    // ON CONFLICT DO UPDATE / ON DUPLICATE KEY UPDATE
                    let insert_result: Result<Option<Row>> = if has_returning {
                        inserted(&mut *table, row, widened).map(Some)
                    } else {
                        table.insert_discard(row).map(|()| None)
                    };
                    match insert_result {
                        Ok(opt_row) => {
                            if let Some(inserted) = opt_row {
                                returning_rows.push(inserted);
                            }
                            rows_affected += 1;
                        }
                        Err(ref e @ Error::PrimaryKeyConstraint { row_id }) => {
                            if !conflict_matches_target(&stmt.conflict_target, &schema, e) {
                                return Err(Error::PrimaryKeyConstraint { row_id });
                            }
                            match self.apply_on_duplicate_update(
                                &mut table,
                                &schema,
                                row_id,
                                None,
                                &row_values,
                                stmt,
                                &mut insert_plan,
                                ctx,
                                (has_returning, widened),
                                conflict,
                                &mut pending_parents,
                            ) {
                                Ok((updated, returned)) => {
                                    if let Some(updated_row) = returned {
                                        returning_rows.push(updated_row);
                                    }
                                    if updated {
                                        rows_affected += 1;
                                    }
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        Err(mut e @ Error::UniqueConstraint { .. }) => {
                            if !conflict_matches_target(&stmt.conflict_target, &schema, &e) {
                                return Err(e);
                            }
                            let Error::UniqueConstraint {
                                index,
                                column,
                                row_id: conflict_rid,
                                ..
                            } = &e
                            else {
                                return Err(e);
                            };
                            let conflict_rid = *conflict_rid;
                            let found_row_id = if conflict_rid >= 0 {
                                Ok(Some(conflict_rid))
                            } else {
                                self.find_row_by_unique_index(
                                    &*table,
                                    &schema,
                                    index,
                                    column,
                                    &row_values,
                                )
                            };
                            match found_row_id {
                                Ok(Some(row_id)) => {
                                    match self.apply_on_duplicate_update(
                                        &mut table,
                                        &schema,
                                        row_id,
                                        Some(column),
                                        &row_values,
                                        stmt,
                                        &mut insert_plan,
                                        ctx,
                                        (has_returning, widened),
                                        conflict,
                                        &mut pending_parents,
                                    ) {
                                        Ok((updated, returned)) => {
                                            if let Some(updated_row) = returned {
                                                returning_rows.push(updated_row);
                                            }
                                            if updated {
                                                rows_affected += 1;
                                            }
                                        }
                                        Err(e) => return Err(e),
                                    }
                                }
                                Ok(None) => {
                                    if let Error::UniqueConstraint { row_id, .. } = &mut e {
                                        *row_id = -1;
                                    }
                                    return Err(e);
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        } else {
            // Fast path: normal INSERT without clones
            for (row_at, value_row) in stmt.values.iter().enumerate() {
                if value_row.len() != column_indices.len() {
                    return Err(Error::InvalidArgument(format!(
                        "INSERT has {} columns but {} values",
                        column_indices.len(),
                        value_row.len()
                    )));
                }

                // Build row values - initialize with DEFAULT values for missing columns
                let mut row_values = Vec::with_capacity(schema_column_count);
                for i in 0..schema_column_count {
                    // A named column is filled below; a default it would
                    // replace must not run, so it cannot fail the row
                    row_values.push(if provided[i] {
                        Value::null_unknown()
                    } else {
                        eval_default_slot(
                            &mut vm,
                            &base_exec_ctx,
                            &default_programs[i],
                            &default_values[i],
                        )?
                    });
                }

                // Fill in provided values using pre-computed indices with type coercion
                for (i, expr) in value_row.iter().enumerate() {
                    // DEFAULT keyword: the column's default, run now
                    if matches!(expr, Expression::Default(_)) {
                        let idx = column_indices[i];
                        row_values[idx] = eval_default_slot(
                            &mut vm,
                            &base_exec_ctx,
                            &default_programs[idx],
                            &default_values[idx],
                        )?;
                        continue;
                    }
                    // OPTIMIZATION: Try to extract literal value directly without VM compilation
                    // This avoids ~1-2μs per expression for simple literals (INTEGER, TEXT, etc.)
                    let value = match try_extract_simple_value(expr, ctx) {
                        Some(value) => value,
                        None => programs.value(&mut vm, &base_exec_ctx, (row_at, i), expr)?,
                    };
                    // Coerce to target type
                    let coerced = value.coerce_to_type(column_types[i]);
                    // Validate coercion didn't silently fail
                    validate_coercion(
                        &value,
                        &coerced,
                        &column_names[i],
                        column_types[i],
                        column_vector_dims[i],
                    )?;
                    row_values[column_indices[i]] = coerced;
                }

                // Validate CHECK constraints (only iterates columns that have CHECK constraints)
                for check in check_programs.iter() {
                    run_check_on_values(&mut check_vm, &mut check_row, check, &row_values)?;
                }

                // Insert row
                let row = Row::from_values(row_values);

                // FK parent validation (zero-cost if no FKs)
                if let Some(ref fks) = fk_schema {
                    super::foreign_key::check_inserted_parents(
                        &self.engine,
                        table.txn_id(),
                        fks,
                        &row,
                        &mut pending_parents,
                    )?;
                }

                if has_returning {
                    // Use insert() which returns row (for RETURNING clause)
                    let inserted_row = inserted(&mut *table, row, widened)?;
                    returning_rows.push(inserted_row);
                } else {
                    // Use insert_discard() to avoid clone overhead
                    table.insert_discard(row)?;
                }
                rows_affected += 1;
            }
        }
        pending_parents.check(&self.engine, table.txn_id())?;

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        let mut returning_result = if let Some(column_names) = &schema_column_names_arc {
            Some(self.build_returning_result(
                &stmt.returning,
                returning_rows,
                column_names,
                ctx,
                returned,
                programs.returning(),
            )?)
        } else {
            None
        };

        // A statement whose reads a truncate or drop overtook commits nothing
        ctx.view_held()?;
        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            if let Some(mut tx) = standalone_tx {
                match tx.commit() {
                    Ok(()) => {}
                    Err(e)
                        if (stmt.on_duplicate || stmt.do_nothing)
                            && e.is_pk_or_unique_violation() =>
                    {
                        if stmt.on_duplicate && ctx.query_depth == 0 {
                            drop(returning_result);
                            drop(_upsert_guard);
                            let (stmt, ctx, plan) = retry_of;
                            let retry_ctx = ctx.with_incremented_query_depth();
                            return self.execute_insert(stmt, &retry_ctx, plan);
                        }
                        if stmt.do_nothing {
                            // DO NOTHING: returning 0 rows is the correct semantic
                            rows_affected = 0;
                            if let Some(result) = &mut returning_result {
                                *result = Box::new(ExecutorResult::with_columns(
                                    result.columns().to_vec(),
                                ));
                            }
                        } else {
                            return Err(e);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        if let Some(result) = returning_result {
            return statement_rollback.finish(Ok(result));
        }

        statement_rollback.finish(Ok(Box::new(ExecResult::with_rows_affected(rows_affected))))
    }

    /// Execute an INSERT statement with compiled cache support
    /// This variant uses the query cache to avoid recomputing schema-derived metadata
    /// on every INSERT execution, significantly reducing allocations for prepared statements.
    pub(crate) fn execute_insert_with_compiled_cache(
        &self,
        stmt: &InsertStatement,
        ctx: &ExecutionContext,
        plan: &NestedPlan,
    ) -> Result<Box<dyn QueryResult>> {
        // Conflict handling requires special handling - fall back to non-cached path
        if stmt.on_duplicate || stmt.do_nothing {
            return self.execute_insert(stmt, ctx, Some(plan));
        }
        let compiled_cache = plan.slot();

        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;

        // The plan is read before the transaction begins, which a snapshot
        // for prepared subqueries must precede
        let current_epoch = self.engine.schema_epoch();
        let cached_insert = match &*compiled_cache.read().unwrap() {
            CompiledExecution::Insert(cached)
                if cached.validity.holds(&*self.engine, current_epoch)
                    && *cached.table_name == *table_name =>
            {
                Some(cached.copied())
            }
            _ => None,
        };
        let (reads_through_view, reads_prepared) = match &cached_insert {
            Some(cached) => (cached.reads_through_view, cached.reads_prepared),
            None => {
                let reads_through_view = super::insert_reads_through_a_view(stmt);
                let reads_prepared =
                    reads_through_view && self.insert_reads_prepared(stmt, ctx, None);
                (reads_through_view, reads_prepared)
            }
        };

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();
        let statement_rollback = StatementRollback::new(&self.engine, active_tx.as_ref());
        let active_txn_id = (*active_tx).as_ref().map(|t| t.transaction.id());

        // INSERT is dispatched before the dispatcher puts the transaction
        // id in the context, so the expressions are given it here
        let ctx_with_txn;
        let ctx = match active_txn_id {
            Some(txn_id) => {
                ctx_with_txn = ctx.with_transaction_id(txn_id as u64);
                &ctx_with_txn
            }
            None => ctx,
        };
        // Its query and subqueries read through one view, taken before them;
        // the target keeps the newest
        let ctx_with_view;
        let ctx = match self.statement_view(reads_through_view, ctx, active_txn_id) {
            Some(view) => {
                ctx_with_view = ctx.with_statement_view(view);
                &ctx_with_view
            }
            None => ctx,
        };
        // Without a view the prepared subqueries read a snapshot taken before the write
        let snapshot_ctx;
        let ctx = if reads_prepared
            && active_txn_id.is_none()
            && ctx.statement_view().is_none()
            && ctx.statement_snapshot().is_none()
        {
            let snapshot = self.new_statement_snapshot(self.isolation_level(), None)?;
            snapshot_ctx = ctx.with_statement_snapshot(snapshot);
            &snapshot_ctx
        } else {
            ctx
        };

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                let table = tx_state.transaction.get_statement_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name.as_str()) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_statement_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.begin_transaction()?;
                let table = tx.get_statement_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::dml_table_opened();

        // Use cached metadata or compile fresh; a rename since the plan was
        // read keeps the token but not the epoch
        let token = table.layout_token();
        let opened_epoch = self.engine.schema_epoch();
        let (
            column_indices,
            column_types,
            column_vector_dims,
            column_names,
            programs,
            compiled_checks,
        ) = if let Some(cached) = cached_insert
            .filter(|cached| cached.token == token && cached.validity.epoch() == opened_epoch)
        {
            // Use cached values (Arc clone is cheap)
            (
                cached.column_indices,
                cached.column_types,
                cached.column_vector_dims,
                cached.column_names,
                cached.programs,
                cached.compiled_checks,
            )
        } else {
            // Compile and cache
            let schema = table.schema();
            let schema_column_count = schema.columns.len();

            // Pre-compile CHECK constraints once per plan. A constraint that
            // does not parse is skipped; one that fails to compile fails the
            // INSERT instead of silently disabling the constraint (e.g. a
            // stale check_expr after RENAME COLUMN).
            let compiled_checks = compile_check_programs(schema)?;

            let all_column_types: Vec<DataType> =
                schema.columns.iter().map(|c| c.data_type).collect();

            // Constant defaults land in the template; non-literal defaults
            // and VALUES expressions compile once and execute per row
            let programs = Arc::new(InsertPrograms::of(schema, stmt));

            let (column_indices, column_types, column_vector_dims, column_names) =
                if stmt.columns.is_empty() {
                    // No columns specified - insert into all columns in order
                    let indices: Vec<usize> = (0..schema_column_count).collect();
                    let types = all_column_types.clone();
                    let dims: Vec<u16> =
                        schema.columns.iter().map(|c| c.vector_dimensions).collect();
                    let names: Vec<SmartString> = schema
                        .columns
                        .iter()
                        .map(|c| SmartString::new(&c.name))
                        .collect();
                    (indices, types, dims, names)
                } else {
                    // Validate columns exist and pre-compute their indices
                    let indices: Vec<usize> = stmt
                        .columns
                        .iter()
                        .map(|id| {
                            written_column(schema, id.value_lower.as_str())?
                                .ok_or_else(|| Error::ColumnNotFound(id.value.to_string()))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let types: Vec<DataType> = indices
                        .iter()
                        .map(|&idx| schema.columns[idx].data_type)
                        .collect();
                    let dims: Vec<u16> = indices
                        .iter()
                        .map(|&idx| schema.columns[idx].vector_dimensions)
                        .collect();
                    let names: Vec<SmartString> = indices
                        .iter()
                        .map(|&idx| SmartString::new(&schema.columns[idx].name))
                        .collect();
                    (indices, types, dims, names)
                };

            // Store in cache for next execution
            let compiled = CompiledInsert {
                table_name: SmartString::new(table_name),
                column_indices: Arc::new(column_indices.clone()),
                column_types: Arc::new(column_types.clone()),
                column_vector_dims: Arc::new(column_vector_dims.clone()),
                column_names: Arc::new(column_names.clone()),
                programs: Arc::clone(&programs),
                compiled_checks: Arc::new(compiled_checks.clone()),
                validity: self.insert_validity(stmt, current_epoch),
                token,
                reads_through_view,
                reads_prepared,
                prepared: None,
            };

            // Update the cache
            if let Ok(mut cache_write) = compiled_cache.write() {
                *cache_write = CompiledExecution::Insert(compiled);
            }

            (
                Arc::new(column_indices),
                Arc::new(column_types),
                Arc::new(column_vector_dims),
                Arc::new(column_names),
                programs,
                Arc::new(compiled_checks),
            )
        };
        let (default_row_template, default_programs) =
            (&programs.default_values, &programs.default_programs);

        // Prepared subqueries are read before the first write, through this
        // statement's transaction
        let prepared;
        let mut runner = None;
        let mut answers = ValueAnswers::new();
        if reads_prepared {
            let target = table.schema_arc();
            let returning = stmt.returning.iter().any(Self::has_subqueries).then(|| {
                super::prepared_select::ReturningSites {
                    table: table_name,
                    schema: &target,
                    values: &stmt.returning,
                }
            });
            // A VALUES row has no target row: its scope is a table without columns
            let (plan, tables) = self.open_prepared_dml(
                Some(compiled_cache),
                &super::prepared_select::NO_COLUMNS,
                table_name,
                None,
                &[],
                &stmt.values,
                None,
                returning.as_ref(),
                standalone_tx
                    .as_deref()
                    .filter(|_| ctx.statement_snapshot().is_none()),
                ctx,
            )?;
            prepared = plan;
            let run = &*runner.insert(super::prepared_select::DmlRunner::new(
                self, &prepared, ctx, tables, 0,
            )?);
            Self::refuse_target_reads_in_transaction(run, active_txn_id.is_some())?;
            answers = Self::insert_value_answers(run)?;
            if run.returning().is_none() {
                // Every value is read: the blocks' tables close before a row is written
                drop(runner.take());
            }
        }

        // Pre-compute FK info for parent validation (CompactArc ref-count bump, not deep clone)
        let fk_schema = if !table.schema().foreign_keys.is_empty() {
            Some(table.schema_arc())
        } else {
            None
        };
        // Keys into this table whose parent is a later row of the statement
        let mut pending_parents = super::foreign_key::PendingParents::new(table_name);

        // Create VM for constant expression evaluation (reused for all INSERT values)
        use super::expression::{ExecuteContext, ExprVM};
        let mut vm = ExprVM::new();
        let mut check_row = None;
        let params = ctx.params();
        let empty_row = Row::new();

        // OPTIMIZATION: Pre-build ExecuteContext once (reused for all expressions)
        let base_exec_ctx =
            ExecuteContext::new(&empty_row).with_run(params, ctx.transaction_id(), None);

        let mut rows_affected = 0i64;

        // RETURNING clause support - collect inserted rows if RETURNING is specified
        let has_returning = !stmt.returning.is_empty();
        let widened = has_returning && returns_slot(&stmt.returning, table.schema());
        let mut returning_rows: Vec<Row> = Vec::new();
        let schema_column_names_arc = if has_returning {
            Some(table.schema().column_names_arc())
        } else {
            None
        };

        // Build lookup from schema column index to value position
        // This allows building row_values directly without cloning entire template
        let col_to_value_pos: Vec<Option<usize>> = {
            let mut lookup = vec![None; default_row_template.len()];
            for (value_pos, &col_idx) in column_indices.iter().enumerate() {
                lookup[col_idx] = Some(value_pos);
            }
            lookup
        };
        let num_columns = default_row_template.len();

        // Check if this is INSERT ... SELECT
        if let Some(ref select_stmt) = stmt.select {
            let mut select_result =
                self.execute_select_with_plan(select_stmt, ctx, None, plan.source())?;
            if !should_auto_commit {
                // Explicit tx: fully materialize before writes for atomicity
                let columns = select_result.columns().to_vec();
                let rows = Self::materialize_result(select_result)?;
                select_result = Box::new(super::result::ExecutorResult::new(columns, rows));
            }

            // Process each row from the SELECT result
            while select_result.next() {
                let select_row = select_result.row();
                if select_row.len() != column_indices.len() {
                    return Err(Error::InvalidArgument(format!(
                        "INSERT has {} columns but SELECT returns {} columns",
                        column_indices.len(),
                        select_row.len()
                    )));
                }

                // OPTIMIZATION: Build row_values directly without cloning entire template
                // Only clone defaults for columns NOT in the insert list
                let mut row_values = Vec::with_capacity(num_columns);
                for (col_idx, default_val) in default_row_template.iter().enumerate() {
                    if let Some(value_pos) = col_to_value_pos[col_idx] {
                        // Column in insert list - use value from SELECT row
                        let value = &select_row[value_pos];
                        let coerced = value.coerce_to_type(column_types[value_pos]);
                        validate_coercion(
                            value,
                            &coerced,
                            &column_names[value_pos],
                            column_types[value_pos],
                            column_vector_dims[value_pos],
                        )?;
                        row_values.push(coerced);
                    } else {
                        // Column not in insert list - use default
                        row_values.push(eval_default_slot(
                            &mut vm,
                            &base_exec_ctx,
                            &default_programs[col_idx],
                            default_val,
                        )?);
                    }
                }

                // Validate CHECK constraints via the pre-compiled programs
                // (NULL passes per the SQL standard)
                for check in compiled_checks.iter() {
                    run_check_on_values(&mut vm, &mut check_row, check, &row_values)?;
                }

                // Insert row
                let row = Row::from_values(row_values);

                // FK parent validation (zero-cost if no FKs)
                if let Some(ref fks) = fk_schema {
                    super::foreign_key::check_inserted_parents(
                        &self.engine,
                        table.txn_id(),
                        fks,
                        &row,
                        &mut pending_parents,
                    )?;
                }

                if has_returning {
                    let inserted_row = inserted(&mut *table, row, widened)?;
                    returning_rows.push(inserted_row);
                } else {
                    table.insert_discard(row)?;
                }
                rows_affected += 1;
            }
            if let Some(err) = select_result.last_error() {
                return Err(err);
            }
        } else {
            // Regular INSERT with VALUES
            for (row, value_list) in stmt.values.iter().enumerate() {
                if value_list.len() != column_indices.len() {
                    return Err(Error::InvalidArgument(format!(
                        "INSERT has {} columns but {} values provided",
                        column_indices.len(),
                        value_list.len()
                    )));
                }

                // OPTIMIZATION: Build row_values directly without cloning entire template
                // Only clone defaults for columns NOT in the insert list
                let mut row_values = Vec::with_capacity(num_columns);
                for (col_idx, default_val) in default_row_template.iter().enumerate() {
                    if let Some(value_pos) = col_to_value_pos[col_idx] {
                        // Column in insert list - evaluate expression
                        let expr = &value_list[value_pos];
                        if matches!(expr, Expression::Default(_)) {
                            // DEFAULT keyword - same per-row slot machinery
                            row_values.push(eval_default_slot(
                                &mut vm,
                                &base_exec_ctx,
                                &default_programs[col_idx],
                                default_val,
                            )?);
                        } else {
                            // OPTIMIZATION: literals and bound parameters
                            // resolve directly (avoids VM compilation)
                            let value = if let Some(simple) = try_extract_simple_value(expr, ctx) {
                                simple
                            } else if let Ok(at) =
                                answers.binary_search_by_key(&(row, value_pos), |(at, _)| *at)
                            {
                                std::mem::replace(&mut answers[at].1, Value::null_unknown())
                            } else {
                                programs.value(&mut vm, &base_exec_ctx, (row, value_pos), expr)?
                            };

                            let target_type = column_types[value_pos];
                            let coerced = value.coerce_to_type(target_type);
                            validate_coercion(
                                &value,
                                &coerced,
                                &column_names[value_pos],
                                target_type,
                                column_vector_dims[value_pos],
                            )?;
                            row_values.push(coerced);
                        }
                    } else {
                        // Column not in insert list - use default
                        row_values.push(eval_default_slot(
                            &mut vm,
                            &base_exec_ctx,
                            &default_programs[col_idx],
                            default_val,
                        )?);
                    }
                }

                // Validate CHECK constraints via the pre-compiled programs
                for check in compiled_checks.iter() {
                    run_check_on_values(&mut vm, &mut check_row, check, &row_values)?;
                }

                // Insert row
                let row = Row::from_values(row_values);

                // FK parent validation (zero-cost if no FKs)
                if let Some(ref fks) = fk_schema {
                    super::foreign_key::check_inserted_parents(
                        &self.engine,
                        table.txn_id(),
                        fks,
                        &row,
                        &mut pending_parents,
                    )?;
                }

                if has_returning {
                    let inserted_row = inserted(&mut *table, row, widened)?;
                    returning_rows.push(inserted_row);
                } else {
                    table.insert_discard(row)?;
                }
                rows_affected += 1;
            }
        }
        pending_parents.check(&self.engine, table.txn_id())?;

        // CRITICAL: Must invalidate before commit to prevent stale data window
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        if let Some(column_names) = &schema_column_names_arc {
            return statement_rollback.finish(self.finish_returning(
                &stmt.returning,
                returning_rows,
                column_names,
                ctx,
                standalone_tx,
                runner.as_ref(),
                programs.returning(),
            ));
        }

        // A statement whose reads a truncate or drop overtook commits nothing
        ctx.view_held()?;
        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        statement_rollback.finish(Ok(Box::new(ExecResult::with_rows_affected(rows_affected))))
    }

    /// Execute an UPDATE statement
    /// The expression a column's DEFAULT stands for in an UPDATE: its
    /// default value, else its default expression, else NULL
    fn column_default_expression(schema: &crate::core::Schema, name: &str) -> Result<Expression> {
        let column = schema
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        // The expression comes first: a value kept beside it was computed
        // once, when the column was added, and a RANDOM() default must be
        // drawn again for every row
        match (&column.default_expr, &column.default_value) {
            (None, Some(value)) => Ok(super::utils::value_to_expression(value)),
            (Some(text), _) => parse_stored_expression(text, "default", name),
            (None, None) => Ok(super::utils::value_to_expression(&Value::null_unknown())),
        }
    }

    pub(crate) fn execute_update(
        &self,
        stmt: &UpdateStatement,
        ctx: &ExecutionContext,
        plan: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        let compiled = plan.map(NestedPlan::slot);
        // A WITH clause is run first, so the subqueries can read it
        let ctx_with_ctes;
        let ctx = match &stmt.with {
            Some(with_clause) => {
                let plans = plan.map_or(&[][..], |plan| plan.dml_ctes(with_clause));
                let registry = self.materialize_ctes(with_clause, plans, ctx, None)?;
                ctx_with_ctes = ctx.with_cte_data(registry.data());
                &ctx_with_ctes
            }
            None => ctx,
        };
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();
        let statement_rollback = StatementRollback::new(&self.engine, active_tx.as_ref());

        // Without a view the prepared subqueries read a snapshot taken before the write
        let snapshot_ctx;
        let ctx = if active_tx.is_none()
            && ctx.statement_view().is_none()
            && ctx.statement_snapshot().is_none()
            && (stmt
                .where_clause
                .as_deref()
                .is_some_and(Self::has_subqueries)
                || stmt.updates.values().any(Self::has_subqueries)
                || stmt.returning.iter().any(Self::has_subqueries))
            && self
                .dml_family(
                    stmt.where_clause.as_deref(),
                    stmt.updates
                        .values()
                        .filter(|e| !matches!(e, Expression::Default(_)))
                        .chain(
                            stmt.returning
                                .iter()
                                .any(Self::has_subqueries)
                                .then(|| super::prepared_select::returning_values(&stmt.returning))
                                .into_iter()
                                .flatten(),
                        ),
                )
                .is_ok()
        {
            let snapshot = self.new_statement_snapshot(self.isolation_level(), None)?;
            snapshot_ctx = ctx.with_statement_snapshot(snapshot);
            &snapshot_ctx
        } else {
            ctx
        };

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                // NOTE: table_name is already lowercase (value_lower from AST)
                let table = tx_state.transaction.get_statement_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name.as_str()) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_statement_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.begin_transaction()?;
                let table = tx.get_statement_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::dml_table_opened();

        // Check for RETURNING clause
        let has_returning = !stmt.returning.is_empty();

        // Pre-compute column names and indices to avoid schema borrow conflicts
        let schema = table.schema();

        // SET col = DEFAULT reads the column's default
        let defaulted_stmt;
        let stmt = if stmt
            .updates
            .values()
            .any(|e| matches!(e, Expression::Default(_)))
        {
            let mut updates = stmt.updates.clone();
            for (name, expr) in updates.iter_mut() {
                if matches!(expr, Expression::Default(_)) {
                    *expr = Self::column_default_expression(schema, name.as_str())?;
                }
            }
            defaulted_stmt = UpdateStatement {
                updates,
                ..stmt.clone()
            };
            &defaulted_stmt
        } else {
            stmt
        };

        // OPTIMIZATION: Use CompactArc<Vec<String>> to share column names without cloning
        let column_names = schema.column_names_arc();

        // Pre-compute FK info for UPDATE validation
        // Determine which FK columns are being updated (for parent validation)
        let fk_cols_in_update: Vec<(usize, crate::core::ForeignKeyConstraint)> = {
            let col_map = schema.column_index_map();
            let updated_col_indices: Vec<usize> = stmt
                .updates
                .keys()
                .filter_map(|col_name| {
                    let col_lower = col_name.to_lowercase();
                    col_map.get(col_lower.as_str()).copied()
                })
                .collect();
            schema
                .foreign_keys
                .iter()
                .filter(|fk| updated_col_indices.contains(&fk.column_index))
                .map(|fk| (fk.column_index, fk.clone()))
                .collect()
        };
        let has_fk_updates = !fk_cols_in_update.is_empty();

        // A row's id is its primary key's value: a row whose key the statement
        // changes is moved to its new id once every row is set (move_rows)
        let mut key_set = None;
        for col_name in stmt.updates.keys() {
            let column = written_column(schema, &col_name.to_lowercase())?;
            key_set = key_set.or(column.filter(|&at| schema.pk_column_index() == Some(at)));
        }

        // Check if this table is referenced by child tables via columns being updated.
        // This handles CASCADE/RESTRICT/SET NULL for UNIQUE columns referenced by child FKs.
        let all_referencing_fks =
            super::foreign_key::find_referencing_fks(&self.engine, table_name);
        let referencing_fks_for_update: Arc<Vec<(String, crate::core::ForeignKeyConstraint)>> =
            if all_referencing_fks.is_empty() {
                Arc::new(Vec::new())
            } else {
                let col_map = schema.column_index_map();
                let updated_cols: Vec<usize> = stmt
                    .updates
                    .keys()
                    .filter_map(|c| col_map.get(c.to_lowercase().as_str()).copied())
                    .collect();
                let relevant: Vec<_> = all_referencing_fks
                    .iter()
                    .filter(|(_, fk)| {
                        col_map
                            .get(fk.referenced_column.to_lowercase().as_str())
                            .is_some_and(|&idx| updated_cols.contains(&idx))
                    })
                    .cloned()
                    .collect();
                Arc::new(relevant)
            };

        // FK schema of the handle the rows are read through (CompactArc ref-count bump)
        let fk_update_schema = if has_fk_updates {
            Some(table.schema_arc())
        } else {
            None
        };

        // A statement of the prepared family runs through one runner, whose
        // subqueries read the tables as the statement found them
        let returning_values = super::prepared_select::returning_values(&stmt.returning);
        let in_family = self
            .dml_family(
                stmt.where_clause.as_deref(),
                stmt.updates.values().chain(returning_values),
            )
            .is_ok();
        let where_prepared = in_family && stmt.where_clause.is_some();
        let held_dml;
        let dml_run = if in_family {
            let col_map = schema.column_index_map();
            let sets: Vec<(usize, &Expression)> = stmt
                .updates
                .iter()
                .filter_map(|(name, expr)| {
                    col_map
                        .get(name.to_lowercase().as_str())
                        .map(|&i| (i, expr))
                })
                .collect();
            let (prepared, tables) = self.open_prepared_dml(
                compiled,
                schema,
                table_name,
                stmt.where_clause.as_deref(),
                &sets,
                &[] as &[Vec<Expression>],
                None,
                (!stmt.returning.is_empty())
                    .then_some(super::prepared_select::ReturningSites {
                        table: table_name,
                        schema,
                        values: &stmt.returning,
                    })
                    .as_ref(),
                standalone_tx
                    .as_deref()
                    .filter(|_| ctx.statement_snapshot().is_none()),
                ctx,
            )?;
            held_dml = prepared;
            let run = super::prepared_select::DmlRunner::new(
                self,
                &held_dml,
                ctx,
                tables,
                table.row_count_hint(),
            )?;
            if !should_auto_commit && run.returning_reads_target() {
                return Err(Self::returning_reads_target());
            }
            Some(run)
        } else {
            None
        };

        // Build WHERE expression for storage layer
        // Try to convert to storage expression, fall back to in-memory filtering if not possible
        let (where_expr, memory_where_clause): (Option<Box<dyn StorageExpr>>, Option<Expression>) =
            if where_prepared {
                // The runner reads the rows a prepared WHERE passes
                (None, None)
            } else if let Some(ref where_clause) = stmt.where_clause {
                // Try to push down predicate to storage layer
                let (storage_expr, needs_mem) =
                    pushdown::try_pushdown(where_clause, schema, Some(ctx));
                if needs_mem {
                    // Complex expression (like a + b > 100) - use in-memory filtering
                    (storage_expr, Some((**where_clause).clone()))
                } else {
                    (storage_expr, None)
                }
            } else {
                (None, None)
            };

        // The WHERE the prepared family does not ask, compiled once over the
        // target's row; a failure to compile is the error of a row reaching it
        let memory_where_program = memory_where_clause.as_ref().map(|expr| {
            super::aggregation::compile_over_slots(
                expr,
                &column_names,
                crate::functions::global_registry(),
                None,
                false,
                Some(ctx.params()),
            )
        });

        // Use RefCell to collect updated rows for RETURNING clause and FK validation
        use std::cell::RefCell;
        let returning_rows: RefCell<Vec<Row>> = RefCell::new(Vec::new());
        // (old row id, row) of each row whose primary key the statement changes
        let moves: RefCell<Vec<(i64, Row)>> = RefCell::new(Vec::new());
        // Collect new FK values and referenced-column old/new pairs from setter
        let fk_new_values: RefCell<Vec<Row>> = RefCell::new(Vec::new());
        // (referenced_col_idx, old_value, new_value) for FK cascade enforcement
        let ref_col_changes: RefCell<Vec<(usize, Value, Value)>> = RefCell::new(Vec::new());

        // Pre-compute referenced column indices for FK cascade enforcement.
        // Shared by both correlated and non-correlated update paths.
        let ref_col_indices_for_fk: Vec<usize> = if !referencing_fks_for_update.is_empty() {
            let col_map = schema.column_index_map();
            referencing_fks_for_update
                .iter()
                .filter_map(|(_, fk)| {
                    col_map
                        .get(fk.referenced_column.to_lowercase().as_str())
                        .copied()
                })
                .collect::<rustc_hash::FxHashSet<usize>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };

        // Pre-partition FKs by referenced column index for post-update dispatch.
        // This avoids re-borrowing schema after table.update().
        let fks_by_ref_col: FxHashMap<usize, Vec<(String, crate::core::ForeignKeyConstraint)>> =
            if !referencing_fks_for_update.is_empty() {
                let col_map = schema.column_index_map();
                let mut map: FxHashMap<usize, Vec<(String, crate::core::ForeignKeyConstraint)>> =
                    FxHashMap::default();
                for (tbl, fk) in referencing_fks_for_update.iter() {
                    if let Some(&idx) = col_map.get(fk.referenced_column.to_lowercase().as_str()) {
                        map.entry(idx).or_default().push((tbl.clone(), fk.clone()));
                    }
                }
                map
            } else {
                FxHashMap::default()
            };

        // Create a setter function that applies updates using pre-computed indices
        // If we need memory filtering, include the WHERE check in the setter
        let rows_affected = {
            // CRITICAL: Pre-compile update expressions ONCE before the loop
            // Compile directly from source expressions (no intermediate cloning!)
            use super::expression::{ExecuteContext, ExprVM, SharedProgram};
            use super::prepared_select::Candidates;

            let col_map = schema.column_index_map();
            let compiled_updates: Vec<(usize, crate::core::DataType, u16, SharedProgram)> =
                if let Some(run) = &dml_run {
                    run.sets()
                        .iter()
                        .map(|(idx, program)| {
                            (
                                *idx,
                                schema.columns[*idx].data_type,
                                schema.columns[*idx].vector_dimensions,
                                program.clone(),
                            )
                        })
                        .collect()
                } else {
                    // Compile directly from statement expressions
                    let registry = crate::functions::global_registry();
                    stmt.updates
                        .iter()
                        .filter_map(|(col_name, expr)| {
                            let col_lower: String = col_name.to_lowercase().into();
                            col_map.get(&col_lower).map(|&idx| {
                                let program = super::aggregation::compile_over_slots(
                                    expr,
                                    &column_names,
                                    registry,
                                    None,
                                    false,
                                    None,
                                );
                                program.map(|program| {
                                    (
                                        idx,
                                        schema.columns[idx].data_type,
                                        schema.columns[idx].vector_dimensions,
                                        program,
                                    )
                                })
                            })
                        })
                        .collect::<Result<_>>()?
                };

            // Column names for coercion errors, captured before the setter
            // borrows the table mutably
            let update_col_names: Vec<String> = compiled_updates
                .iter()
                .map(|(idx, _, _, _)| schema.columns[*idx].name.to_string())
                .collect();

            // The CHECK constraints reading an updated column, held by the plan
            let compiled_check_exprs = match &dml_run {
                Some(run) => run.update_checks(schema)?,
                None => Arc::new(checks_reading(compile_check_programs(schema)?, |c| {
                    compiled_updates.iter().any(|(idx, _, _, _)| *idx == c)
                })),
            };

            // Create VM once and reuse for all rows
            let mut vm = ExprVM::new();
            // Extract params before the closure so they can be captured
            let params = ctx.params();
            let transaction_id = ctx.transaction_id();
            let candidates = match dml_run.as_ref().filter(|_| where_prepared) {
                Some(run) => Some(run.candidates(&*table, ctx)?),
                None => None,
            };
            // Who asks each row its WHERE, chosen once for the statement
            enum RowWhere<'w, 'r> {
                Passed,
                Runner(&'w super::prepared_select::DmlRunner<'r>),
                Program(&'w SharedProgram),
                Failed(&'w Error),
            }
            let row_where = match (&dml_run, &candidates) {
                (Some(run), Some(Candidates::Keyed(_) | Candidates::Scan(_))) => {
                    RowWhere::Runner(run)
                }
                (_, Some(_)) => RowWhere::Passed,
                (_, None) => match &memory_where_program {
                    Some(Ok(program)) => RowWhere::Program(program),
                    Some(Err(error)) => RowWhere::Failed(error),
                    None => RowWhere::Passed,
                },
            };
            // The context each row's programs read, built once over no row
            let no_row = Row::new();
            let row_ctx = match &dml_run {
                Some(run) => run.context(&no_row),
                None => ExecuteContext::new(&no_row).with_run(params, transaction_id, None),
            };
            let widen = dml_run.as_ref().is_some_and(|run| run.widens());
            let mut set_row = |row_id: i64, row: Row| -> Result<(Row, bool)> {
                let mut row = match widen {
                    true => super::prepared_select::widened(row, row_id),
                    false => row,
                };
                // Execute pre-compiled programs (no recompilation per row)
                let updates_to_apply: Vec<(usize, Value)> = {
                    let exec_ctx = ExecuteContext {
                        row: &row,
                        ..row_ctx
                    };

                    // In-memory WHERE filtering shares the row context
                    match row_where {
                        RowWhere::Passed => {}
                        RowWhere::Runner(run) => {
                            #[cfg(feature = "test-failpoints")]
                            crate::test_failpoints::prepared_dml_site_ran(
                                crate::test_failpoints::PreparedDmlSite::UpdateWhere,
                            );
                            if !run.matches(&mut vm, &exec_ctx)? {
                                return Ok((row, false));
                            }
                        }
                        RowWhere::Program(program) => {
                            if !vm.execute_bool_checked(program, &exec_ctx)? {
                                return Ok((row, false));
                            }
                        }
                        RowWhere::Failed(error) => return Err(error.clone()),
                    }

                    let mut updates = Vec::with_capacity(compiled_updates.len());
                    for (k, (idx, col_type, vec_dims, program)) in
                        compiled_updates.iter().enumerate()
                    {
                        // Propagate VM errors (overflow etc.): silently
                        // skipping a SET column would half-apply the row.
                        // Coercion validates like INSERT and the fast path
                        // (vector dims, no silent non-null-to-NULL).
                        #[cfg(feature = "test-failpoints")]
                        if dml_run.is_some() {
                            crate::test_failpoints::prepared_dml_site_ran(
                                crate::test_failpoints::PreparedDmlSite::UpdateSet,
                            );
                        }
                        let v = vm.execute(program, &exec_ctx)?;
                        let coerced = v.coerce_to_type(*col_type);
                        validate_coercion(
                            &v,
                            &coerced,
                            &update_col_names[k],
                            *col_type,
                            *vec_dims,
                        )?;
                        updates.push((*idx, coerced));
                    }
                    updates
                };

                // Now apply all the computed values to the row
                let changed = !updates_to_apply.is_empty();

                // Capture old values of referenced columns before applying changes
                let ref_old_values: Vec<(usize, Value)> =
                    if changed && !ref_col_indices_for_fk.is_empty() {
                        ref_col_indices_for_fk
                            .iter()
                            .filter_map(|&ci| row.get(ci).map(|v| (ci, v.clone())))
                            .collect()
                    } else {
                        Vec::new()
                    };

                let original = key_set.map(|_| row.clone());
                for (idx, new_value) in updates_to_apply {
                    let _ = row.set(idx, new_value);
                }

                // Each CHECK an updated column reads, over the row as it now is
                if changed {
                    for check in compiled_check_exprs.iter() {
                        run_check(&mut vm, check, &row)?;
                    }
                }

                // Collect FK values for post-update validation
                if changed && has_fk_updates {
                    fk_new_values.borrow_mut().push(row.clone());
                }

                // Track referenced column changes for FK cascade enforcement
                if !ref_old_values.is_empty() {
                    for (ci, old_val) in &ref_old_values {
                        if let Some(new_val) = row.get(*ci) {
                            if old_val != new_val {
                                ref_col_changes.borrow_mut().push((
                                    *ci,
                                    old_val.clone(),
                                    new_val.clone(),
                                ));
                            }
                        }
                    }
                }

                // Collect row for RETURNING clause
                if changed && has_returning {
                    returning_rows.borrow_mut().push(row.clone());
                }

                // A row whose key changes is held as it was, claimed for the
                // statement, and moved once every row is set
                if let (Some(key), Some(original)) = (key_set, original) {
                    if row.get(key) != original.get(key) {
                        let mut row = row;
                        if widen {
                            row.pop();
                        }
                        moves.borrow_mut().push((row_id, row));
                        return Ok((original, true));
                    }
                }

                Ok((row, changed))
            };
            // A widened row is written without its row id
            let mut setter = |row_id: i64, row: Row| -> Result<(Row, bool)> {
                let (mut row, changed) = set_row(row_id, row)?;
                if widen {
                    row.pop();
                }
                Ok((row, changed))
            };

            // OPTIMIZATION: Use SELECT executor to find matching row_ids, then batch update.
            // This reuses ALL SELECT optimizations: indexes, semi-joins, parallel execution, etc.
            let rows = if let Some(candidates) = candidates {
                match candidates {
                    Candidates::Pruned => 0,
                    Candidates::Keyed(mut row_ids) => {
                        row_ids.sort_unstable();
                        table.update_by_row_ids(&row_ids, &mut setter)?
                    }
                    Candidates::Scan(expr) | Candidates::Whole(expr) => {
                        // The access the plan decided, else the rule's answer now
                        let plan;
                        let access = match dml_run.as_ref().and_then(|run| run.access(&*table)) {
                            Some(access) => access,
                            None => {
                                plan = expr.as_deref().and_then(|e| table.index_plan(e));
                                plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index)
                            }
                        };
                        table.update(access, expr.as_deref(), &mut setter)?
                    }
                }
            } else {
                // The storage layer reads the rows the WHERE names; the setter
                // asks each the part it does not take
                let plan = where_expr.as_deref().and_then(|e| table.index_plan(e));
                let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
                table.update(access, where_expr.as_deref(), &mut setter)?
            };
            rows
        };
        move_rows(&mut *table, moves.into_inner())?;

        // Post-update FK validation: check new FK values reference existing parent rows
        if has_fk_updates {
            let fk_rows = fk_new_values.into_inner();
            if let Some(ref fk_schema) = fk_update_schema {
                for row in &fk_rows {
                    super::foreign_key::check_parent_exists(
                        &self.engine,
                        table.txn_id(),
                        fk_schema,
                        row,
                    )?;
                }
            }
        }

        // Post-update referenced-column change enforcement: each value the
        // statement changed is checked against RESTRICT, then CASCADE/SET NULL apply
        if !fks_by_ref_col.is_empty() {
            // Each column's changes cascade together, so a child row is
            // matched by the value the statement found, once
            let mut by_column: FxHashMap<usize, Vec<(Value, Value)>> = FxHashMap::default();
            let restricted: Vec<usize> = fks_by_ref_col
                .iter()
                .filter(|(_, fks)| super::foreign_key::fk_tree_needs_precheck(&self.engine, fks))
                .map(|(&col_idx, _)| col_idx)
                .collect();
            for (col_idx, old_val, new_val) in ref_col_changes.into_inner() {
                if let Some(fks) = fks_by_ref_col.get(&col_idx) {
                    if restricted.contains(&col_idx) {
                        super::foreign_key::pre_check_restrict_for_update(
                            &self.engine,
                            table.txn_id(),
                            table_name,
                            &old_val,
                            fks,
                        )?;
                    }
                    by_column
                        .entry(col_idx)
                        .or_default()
                        .push((old_val, new_val));
                }
            }
            if !by_column.is_empty() {
                super::foreign_key::enforce_update_actions(
                    &self.engine,
                    table.txn_id(),
                    table_name,
                    by_column.into_iter().collect(),
                    None,
                )?;
            }
        }

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        if has_returning {
            return statement_rollback.finish(self.finish_returning(
                &stmt.returning,
                returning_rows.into_inner(),
                &column_names,
                ctx,
                standalone_tx,
                dml_run.as_ref(),
                None,
            ));
        }

        // A statement whose reads a truncate or drop overtook commits nothing
        ctx.view_held()?;
        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        statement_rollback.finish(Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        ))))
    }

    /// Execute a DELETE statement
    pub(crate) fn execute_delete(
        &self,
        stmt: &DeleteStatement,
        ctx: &ExecutionContext,
        plan: Option<&NestedPlan>,
    ) -> Result<Box<dyn QueryResult>> {
        let compiled = plan.map(NestedPlan::slot);
        // A WITH clause is run first, so the subqueries can read it
        let ctx_with_ctes;
        let ctx = match &stmt.with {
            Some(with_clause) => {
                let plans = plan.map_or(&[][..], |plan| plan.dml_ctes(with_clause));
                let registry = self.materialize_ctes(with_clause, plans, ctx, None)?;
                ctx_with_ctes = ctx.with_cte_data(registry.data());
                &ctx_with_ctes
            }
            None => ctx,
        };
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;
        // Use alias if provided, otherwise use table name
        let effective_name = stmt
            .alias
            .as_ref()
            .map(|a| a.value_lower.as_str())
            .unwrap_or(table_name.as_str());

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();
        let statement_rollback = StatementRollback::new(&self.engine, active_tx.as_ref());

        // A statement of the prepared family runs through one runner, whose
        // subqueries read the tables as the statement found them
        let where_has_subqueries = stmt
            .where_clause
            .as_deref()
            .is_some_and(Self::has_subqueries);
        let returning_has_subqueries = stmt.returning.iter().any(Self::has_subqueries);
        let returning_values = super::prepared_select::returning_values(&stmt.returning);
        let in_family = self
            .dml_family(stmt.where_clause.as_deref(), returning_values)
            .is_ok();
        let where_prepared = in_family && stmt.where_clause.is_some();
        // Without a view the prepared subqueries read a snapshot taken before the write
        let snapshot_ctx;
        let ctx = if in_family
            && (where_has_subqueries || returning_has_subqueries)
            && active_tx.is_none()
            && ctx.statement_view().is_none()
            && ctx.statement_snapshot().is_none()
        {
            let snapshot = self.new_statement_snapshot(self.isolation_level(), None)?;
            snapshot_ctx = ctx.with_statement_snapshot(snapshot);
            &snapshot_ctx
        } else {
            ctx
        };

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Use the active transaction
                // NOTE: table_name is already lowercase (value_lower from AST)
                let table = tx_state.transaction.get_statement_table(table_name)?;

                // Store a reference to this table for commit/rollback
                if !tx_state.tables.contains_key(table_name.as_str()) {
                    tx_state.tables.insert(
                        table_name.to_string(),
                        tx_state.transaction.get_statement_table(table_name)?,
                    );
                }

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.begin_transaction()?;
                let table = tx.get_statement_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::dml_table_opened();

        // Check for RETURNING clause
        let has_returning = !stmt.returning.is_empty();
        let mut returning_rows: Vec<Row> = Vec::new();

        // Build WHERE expression - try to convert to storage expression
        // If that fails (complex expression like a + b > 100), fall back to in-memory filtering
        let schema = table.schema();

        let held_dml;
        let dml_run = if in_family {
            let (prepared, tables) = self.open_prepared_dml(
                compiled,
                schema,
                &SmartString::from(effective_name),
                stmt.where_clause.as_deref().filter(|_| where_prepared),
                &[],
                &[] as &[Vec<Expression>],
                None,
                (!stmt.returning.is_empty())
                    .then_some(super::prepared_select::ReturningSites {
                        table: table_name,
                        schema,
                        values: &stmt.returning,
                    })
                    .as_ref(),
                standalone_tx
                    .as_deref()
                    .filter(|_| ctx.statement_snapshot().is_none()),
                ctx,
            )?;
            held_dml = prepared;
            let run = super::prepared_select::DmlRunner::new(
                self,
                &held_dml,
                ctx,
                tables,
                table.row_count_hint(),
            )?;
            if !should_auto_commit && run.returning_reads_target() {
                return Err(Self::returning_reads_target());
            }
            Some(run)
        } else {
            None
        };

        let (where_expr, needs_memory_filter, memory_where_clause): (
            Option<Box<dyn StorageExpr>>,
            bool,
            Option<Expression>,
        ) = if let Some(ref where_clause) = stmt.where_clause {
            if where_prepared {
                // A scan that takes the whole WHERE deletes as it reads; else
                // the runner reads the rows a prepared WHERE passes
                match dml_run.as_ref().filter(|_| !has_returning) {
                    Some(run) => match run.whole_scan(&*table, ctx)? {
                        Some(expr) => (expr, false, None),
                        None => (None, true, None),
                    },
                    None => (None, true, None),
                }
            } else {
                // Try to push down predicate to storage layer
                let (storage_expr, needs_mem) =
                    pushdown::try_pushdown(where_clause, schema, Some(ctx));
                if needs_mem {
                    // Complex expression (like a + b > 100) - use in-memory filtering
                    (storage_expr, true, Some((**where_clause).clone()))
                } else {
                    (storage_expr, false, None)
                }
            }
        } else {
            (None, false, None)
        };

        // Check if this table is referenced by child tables (for FK enforcement)
        let referencing_fks = super::foreign_key::find_referencing_fks(&self.engine, table_name);

        // Get schema info for RETURNING clause processing
        let column_names_owned = schema.column_names_owned().to_vec();
        let column_count = schema.columns.len();
        // Use schema's cached pk_column_index for O(1) lookup
        let pk_col_idx = schema.pk_column_index();

        let has_referencing_fks = !referencing_fks.is_empty();

        // Delete rows
        let rows_affected = if needs_memory_filter || has_returning || has_referencing_fks {
            // Complex WHERE expression, RETURNING, or FK enforcement - need to scan rows first
            // Scan all rows, filter with evaluator, collect for RETURNING, delete matching ones by primary key
            // Schema of the handle the rows are read through (CompactArc ref-count bump)
            let schema_arc = table.schema_arc();

            // Build column names with effective prefix (alias or table name)
            // This allows WHERE clauses to reference columns using the alias
            // OPTIMIZATION: Only build when needed (memory filter or RETURNING)
            let column_names_with_prefix: Vec<String> = column_names_owned
                .iter()
                .map(|c| qualified(effective_name, c))
                .collect();

            // The WHERE the prepared family does not ask, compiled once over
            // the prefixed columns, so it reads alias.column; a failure to
            // compile is the error of a row reaching it
            use super::expression::{ExecuteContext, ExprVM};
            let mut delete_vm = ExprVM::new();
            let params = ctx.params();
            let memory_where_program = memory_where_clause.as_ref().map(|expr| {
                super::aggregation::compile_over_slots(
                    expr,
                    &column_names_with_prefix,
                    crate::functions::global_registry(),
                    None,
                    false,
                    Some(params),
                )
            });
            let transaction_id = ctx.transaction_id();

            let mut rows_to_delete: Vec<(Value, Option<Row>)> = Vec::new();
            let mut rows_to_delete_by_row_id: Vec<(i64, Option<Row>)> = Vec::new();
            // Each key's values, read as the rows are found from the column
            // it names: a key need not name the primary key
            let fk_columns: Vec<Option<usize>> = referencing_fks
                .iter()
                .map(|(_, fk)| {
                    let referenced = fk.referenced_column.to_lowercase();
                    schema_arc
                        .columns
                        .iter()
                        .position(|c| c.name_lower == referenced)
                })
                .collect();
            let mut fk_values: Vec<Vec<Value>> = vec![Vec::new(); fk_columns.len()];
            // A row to delete, by its primary key or, without one, its row id
            let mut collect = |row_id: i64, row: &Row| {
                let row_data = has_returning.then(|| row.clone());
                for (values, column) in fk_values.iter_mut().zip(&fk_columns) {
                    if let Some(value) = column.and_then(|c| row.get(c)) {
                        values.push(value.clone());
                    }
                }
                match pk_col_idx {
                    Some(pk_idx) => {
                        if let Some(pk_value) = row.get(pk_idx) {
                            rows_to_delete.push((pk_value.clone(), row_data));
                        }
                    }
                    None => rows_to_delete_by_row_id.push((row_id, row_data)),
                }
            };
            // A prepared WHERE's rows are read as a prepared SELECT reads them
            if let Some(run) = dml_run.as_ref().filter(|_| where_prepared) {
                run.passing_rows(&*table, ctx, &mut |row_id, row| {
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::prepared_dml_site_ran(
                        crate::test_failpoints::PreparedDmlSite::DeleteWhere,
                    );
                    collect(row_id, &row);
                })?;
            }

            // Scan all rows and collect IDs of rows to delete; a RETURNING
            // reading the row id has it past the row's columns
            let returns_widened = has_returning && dml_run.as_ref().is_some_and(|r| r.widens());
            let column_indices: Vec<usize> = (0..column_count).collect();
            let mut scanner = if where_prepared {
                None
            } else {
                let plan = where_expr.as_deref().and_then(|e| table.index_plan(e));
                let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
                Some(table.scan_access(access, &column_indices, None, where_expr.as_deref())?)
            };

            while let Some(scanner) = scanner.as_mut() {
                if !scanner.next() {
                    break;
                }
                let row = scanner.row();

                // Check memory filter if needed
                let matches = match &memory_where_program {
                    Some(program) => {
                        let program = program.as_ref().map_err(Clone::clone)?;
                        let exec_ctx =
                            ExecuteContext::new(row).with_run(params, transaction_id, None);
                        delete_vm.execute_bool_checked(program, &exec_ctx)?
                    }
                    None => true, // Storage layer already filtered
                };

                if matches {
                    let row_id = scanner.current_row_id();
                    match returns_widened {
                        true => collect(
                            row_id,
                            &super::prepared_select::widened(row.clone(), row_id),
                        ),
                        false => collect(row_id, row),
                    }
                }
            }
            if let Some(error) = scanner.as_ref().and_then(|scanner| scanner.err()) {
                return Err(error.clone());
            }
            // Drop scanner to release borrow
            drop(scanner);

            // The keys' actions are worked out from the rows as found, and
            // a refusal comes before any row goes
            let delete_plan = if has_referencing_fks {
                let ids: Vec<i64> = rows_to_delete
                    .iter()
                    .filter_map(|(pk, _)| match pk {
                        Value::Integer(id) => Some(*id),
                        _ => None,
                    })
                    .chain(rows_to_delete_by_row_id.iter().map(|(id, _)| *id))
                    .collect();
                Some(super::foreign_key::plan_delete_actions(
                    &self.engine,
                    table.txn_id(),
                    table_name,
                    &ids,
                    referencing_fks.clone(),
                    fk_values,
                )?)
            } else {
                None
            };

            // Delete matching rows by primary key
            let mut delete_count = 0;
            if !rows_to_delete.is_empty() {
                // An integer key is the row id, so the rows go in one call,
                // read once from the volumes they sit in, as the rows named
                // by id below do
                let mut row_ids: Vec<i64> = rows_to_delete
                    .iter()
                    .map(|(pk, _)| match pk {
                        Value::Integer(id) => *id,
                        _ => 0,
                    })
                    .collect();
                #[cfg(any(test, feature = "test-failpoints"))]
                crate::test_failpoints::delete_rows_scanned();
                let scanned = row_ids.len();
                delete_count = table.delete_scanned_rows(&mut row_ids)?;
                if has_returning {
                    // A row gone since the scan was not deleted and is not
                    // returned
                    if row_ids.len() == scanned {
                        returning_rows
                            .extend(rows_to_delete.into_iter().filter_map(|(_, row)| row));
                    } else {
                        let deleted: I64Set = row_ids.iter().copied().collect();
                        returning_rows.extend(rows_to_delete.into_iter().filter_map(
                            |(pk, row)| match pk {
                                Value::Integer(id) if deleted.contains(id) => row,
                                _ => None,
                            },
                        ));
                    }
                }
            } else if !rows_to_delete_by_row_id.is_empty() {
                // A table without a primary key: the rows are named by the
                // ids the scan read them under
                let row_ids: Vec<i64> =
                    rows_to_delete_by_row_id.iter().map(|(id, _)| *id).collect();
                delete_count = table.delete_by_row_ids(&row_ids)?;
                if has_returning {
                    returning_rows.extend(
                        rows_to_delete_by_row_id
                            .into_iter()
                            .filter_map(|(_, row)| row),
                    );
                }
            }
            if let Some(plan) = delete_plan {
                plan.write(&self.engine, table.txn_id())?;
            }
            delete_count
        } else {
            // A WHERE the storage layer takes whole: it reads the rows the
            // WHERE names. The access the plan decided, else the rule's answer now
            let decided = dml_run
                .as_ref()
                .filter(|_| where_prepared)
                .and_then(|run| run.access(&*table));
            let plan;
            let access = match decided {
                Some(access) => access,
                None => {
                    plan = where_expr.as_deref().and_then(|e| table.index_plan(e));
                    plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index)
                }
            };
            table.delete(access, where_expr.as_deref())?
        };

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
        }

        if has_returning {
            return statement_rollback.finish(self.finish_returning(
                &stmt.returning,
                returning_rows,
                &column_names_owned,
                ctx,
                standalone_tx,
                dml_run.as_ref(),
                None,
            ));
        }

        // A statement whose reads a truncate or drop overtook commits nothing
        ctx.view_held()?;
        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        statement_rollback.finish(Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        ))))
    }

    /// Execute a TRUNCATE statement
    /// TRUNCATE is equivalent to DELETE without WHERE clause, but more efficient.
    ///
    /// **Non-rollbackable**: Like MySQL and Oracle, TRUNCATE physically destroys
    /// versions, arena, and indexes immediately. ROLLBACK cannot undo it.
    /// This is a deliberate trade-off: O(1) truncation vs rollback safety.
    /// Use `DELETE FROM table` if transactional rollback is needed.
    ///
    /// Fails with `TableHasActiveTransactions` if:
    /// - The current explicit transaction has already modified this table (INSERT/UPDATE/DELETE)
    /// - Another transaction holds uncommitted UPDATE/DELETE claims on the table
    pub(crate) fn execute_truncate(
        &self,
        stmt: &TruncateStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // OPTIMIZATION: Use pre-computed lowercase name to avoid allocation per query
        let table_name = &stmt.table_name.value_lower;

        // Check if there's an active explicit transaction
        let mut active_tx = self.active_transaction.lock().unwrap();

        let (mut table, should_auto_commit, standalone_tx) =
            if let Some(ref mut tx_state) = *active_tx {
                // Block TRUNCATE if this transaction has already modified the table.
                // Uncommitted INSERTs live in TransactionVersionStore.local_versions
                // and would be silently destroyed by truncate_all().
                if tx_state.tables.contains_key(table_name.as_str()) {
                    return Err(Error::TableHasActiveTransactions);
                }

                // Use the active transaction
                let table = tx_state.transaction.get_statement_table(table_name)?;

                // Register this table for commit/rollback
                tx_state.tables.insert(
                    table_name.to_string(),
                    tx_state.transaction.get_statement_table(table_name)?,
                );

                (table, false, None)
            } else {
                // No active transaction - create a standalone transaction with auto-commit
                let tx = self.begin_transaction()?;
                let table = tx.get_statement_table(table_name)?;
                (table, true, Some(tx))
            };

        // Drop the lock before doing work
        drop(active_tx);

        // FK enforcement: block truncate if child tables reference this table
        // Uses the table's transaction for visibility (sees uncommitted child deletes)
        super::foreign_key::check_no_referencing_rows(
            &self.engine,
            table_name,
            Some(table.txn_id()),
        )?;

        // Truncate all rows (fast path: drops storage directly)
        // WAL is recorded AFTER success to prevent phantom records on failure.
        // If a crash occurs between truncate and WAL write, recovery restores
        // the pre-truncate state from the snapshot — the user simply retries.
        // A view taken before the rows go, or while they go, is overtaken
        let _destruction = self.engine.registry_ref().begin_destruction();
        let rows_affected = table.truncate()?;

        // Record TRUNCATE to WAL for persistence (only after successful truncate)
        self.engine.record_truncate_table(table_name)?;

        // Invalidate semantic cache for this table BEFORE commit
        // CRITICAL: Must invalidate before commit to prevent stale data window
        // (TRUNCATE always invalidates, regardless of rows_affected, for safety)
        self.semantic_cache.invalidate_table(table_name);

        // Commit if this is a standalone (auto-commit) transaction
        if should_auto_commit {
            // Commit the transaction - it will commit all tables via commit_all_tables()
            if let Some(mut tx) = standalone_tx {
                tx.commit()?;
            }
        }

        Ok(Box::new(ExecResult::with_rows_affected(
            rows_affected as i64,
        )))
    }

    /// Build a `CompiledUpsert` from `schema` and `stmt` once per INSERT statement.
    /// This avoids re-compiling update expressions and re-parsing CHECK constraint SQL
    /// for every conflicting row in an upsert batch.
    fn compile_upsert(
        schema: &crate::core::Schema,
        stmt: &InsertStatement,
    ) -> Result<CompiledUpsert> {
        use super::expression::{CompileContext, ExprCompiler, SharedProgram};
        use crate::functions::registry::global_registry;

        let (update_specs, update_where) = Self::conflict_sets(schema, stmt)?;
        let widens = Self::conflict_widens(schema, stmt);

        let column_names: Vec<String> = super::prepared_select::read_names(schema, widens).to_vec();

        // Build EXCLUDED column name list for the second row source.
        let excluded_columns: Vec<String> = (schema.column_names_owned().iter())
            .map(|c| qualified("excluded", c))
            .collect();

        // Compile each update expression with EXCLUDED as the second row source.
        let mut compiled_updates: Vec<(usize, DataType, u16, SharedProgram)> =
            Vec::with_capacity(update_specs.len());
        for (idx, expr) in &update_specs {
            let compile_ctx = CompileContext::new(&column_names, global_registry())
                .with_second_row(&excluded_columns);
            let compiler = ExprCompiler::new(&compile_ctx);
            match compiler.compile(expr) {
                Ok(program) => {
                    let column = &schema.columns[*idx];
                    compiled_updates.push((
                        *idx,
                        column.data_type,
                        column.vector_dimensions,
                        CompactArc::new(program),
                    ));
                }
                Err(e) => {
                    return Err(Error::internal(format!(
                        "failed to compile ON CONFLICT update expression: {}",
                        e
                    )));
                }
            }
        }

        // DO UPDATE ... WHERE is compiled the same way, with EXCLUDED as the
        // second row source
        let compiled_where = match &update_where {
            Some(expr) => {
                let compile_ctx = CompileContext::new(&column_names, global_registry())
                    .with_second_row(&excluded_columns);
                let program = ExprCompiler::new(&compile_ctx).compile(expr).map_err(|e| {
                    Error::internal(format!(
                        "failed to compile ON CONFLICT update condition: {}",
                        e
                    ))
                })?;
                Some(CompactArc::new(program))
            }
            None => None,
        };

        Self::upsert_of(schema, compiled_updates, compiled_where, widens)
    }

    /// Whether `stmt`'s ON CONFLICT SET values or WHERE read the row id of
    /// a row met in `schema`'s table, which has no primary key
    pub(super) fn conflict_widens(schema: &crate::core::Schema, stmt: &InsertStatement) -> bool {
        let clauses = stmt.update_expressions.iter();
        schema.rowid_column().is_none()
            && (clauses.chain(stmt.update_where.as_deref())).any(super::utils::expr_names_rowid)
    }

    /// Each ON CONFLICT SET value of `stmt` with the column it writes, and
    /// its WHERE, a rowid of the row met spelled as the column it reads: the
    /// primary key, quoted, or the row id slot
    pub(super) fn conflict_sets(
        schema: &crate::core::Schema,
        stmt: &InsertStatement,
    ) -> Result<ConflictClauses> {
        use crate::parser::{Position, Token, TokenType};
        let key = match schema.rowid_column() {
            Some(at) => schema.columns[at].name.as_str(),
            None => ROWID_SLOT,
        };
        let target = &stmt.table_name.value_lower;
        let spell = |expr: &Expression| {
            super::utils::rowids_respelled(expr, &mut |rowid| {
                let met = match rowid {
                    Expression::QualifiedIdentifier(q) => {
                        q.qualifier.value_lower == *target && q.qualifier.value_lower != "excluded"
                    }
                    _ => true,
                };
                Ok(met.then(|| {
                    let token = Token::new(TokenType::Identifier, key, Position::new(0, 0, 0));
                    Expression::Identifier(Identifier {
                        quoted: true,
                        ..Identifier::new(token, key.to_string())
                    })
                }))
            })
        };
        let mut sets = Vec::with_capacity(stmt.update_columns.len());
        for (column, value) in stmt.update_columns.iter().zip(&stmt.update_expressions) {
            if let Some(at) = written_column(schema, &column.value_lower)? {
                sets.push((at, spell(value)?));
            }
        }
        let where_ = stmt.update_where.as_deref().map(spell).transpose()?;
        Ok((sets, where_))
    }

    /// The upsert writing each program of `compiled_updates` into its column,
    /// of the type and vector width it gives, where `compiled_where` holds,
    /// with the CHECK constraints of those columns
    pub(super) fn upsert_of(
        schema: &crate::core::Schema,
        compiled_updates: Vec<(usize, DataType, u16, super::expression::SharedProgram)>,
        compiled_where: Option<super::expression::SharedProgram>,
        widens: bool,
    ) -> Result<CompiledUpsert> {
        // The CHECK constraints reading an updated column, shared immutably:
        // the VM that runs them lives in the setter closure
        let compiled_checks = checks_reading(compile_check_programs(schema)?, |c| {
            compiled_updates.iter().any(|(idx, _, _, _)| *idx == c)
        });
        Ok(CompiledUpsert {
            compiled_updates,
            compiled_where,
            compiled_checks,
            widens,
        })
    }

    /// Apply ON DUPLICATE KEY UPDATE to an existing row.
    /// `insert_values` contains the attempted insert row — accessible via `EXCLUDED.column`
    /// in update expressions (PostgreSQL-style).
    /// `conflict_column` is the unique constraint column(s) that caused the conflict (comma-separated).
    /// Used to build WHERE clause when the table has no primary key.
    /// When `capture_row` is true, returns the post-update row for RETURNING clauses,
    /// its row id past its columns when `returns_id`.
    #[allow(clippy::too_many_arguments)]
    fn apply_on_duplicate_update(
        &self,
        table: &mut Box<dyn Table>,
        schema: &crate::core::Schema,
        row_id: i64,
        conflict_column: Option<&str>,
        insert_values: &[Value],
        stmt: &InsertStatement,
        insert_plan: &mut InsertPlan<'_>,
        ctx: &ExecutionContext,
        (capture_row, returns_id): (bool, bool),
        conflict: Option<&ConflictRun<'_, '_>>,
        pending_parents: &mut super::foreign_key::PendingParents<'_>,
    ) -> Result<(bool, Option<Row>)> {
        // Build a WHERE clause to find the specific row
        // Use schema's cached pk_column_index for O(1) lookup
        let pk_col = schema
            .pk_column_index()
            .map(|idx| schema.columns[idx].name.as_str());

        let where_expr: Option<Box<dyn StorageExpr>> = if let Some(pk_name) = pk_col {
            // Table has PK — target by primary key value
            let mut expr =
                ComparisonExpr::new(pk_name, crate::core::Operator::Eq, Value::Integer(row_id));
            expr.prepare_for_schema(schema);
            Some(Box::new(expr))
        } else if let Some(conflict_cols) = conflict_column {
            // No PK — target by the unique constraint columns that caused the conflict
            use crate::storage::expression::AndExpr;
            let col_names: Vec<&str> = conflict_cols.split(", ").collect();
            let col_map = schema.column_index_map();
            let mut comparisons: Vec<Box<dyn StorageExpr>> = Vec::with_capacity(col_names.len());
            for col_name in &col_names {
                let col_lower = col_name.to_lowercase();
                if let Some(&idx) = col_map.get(col_lower.as_str()) {
                    let value = insert_values
                        .get(idx)
                        .cloned()
                        .unwrap_or(Value::null_unknown());
                    let mut expr = ComparisonExpr::new(*col_name, crate::core::Operator::Eq, value);
                    expr.prepare_for_schema(schema);
                    comparisons.push(Box::new(expr));
                }
            }
            if comparisons.len() == 1 {
                Some(comparisons.pop().unwrap())
            } else if comparisons.len() > 1 {
                let mut and_expr = AndExpr::new(comparisons);
                and_expr.prepare_for_schema(schema);
                Some(Box::new(and_expr))
            } else {
                None
            }
        } else {
            None
        };

        use super::expression::{ExecuteContext, ExprVM};

        let effective = insert_plan.upsert(schema, stmt)?;

        // Build the EXCLUDED row from insert_values; prepared programs read it joined
        let excluded_row = match conflict {
            Some(_) => Row::new(),
            None => Row::from_values(insert_values.to_vec()),
        };

        // Create VM once and reuse for all rows
        let mut vm = ExprVM::new();

        // Extract params from execution context for use in the setter closure
        let params = ctx.params();
        let transaction_id = ctx.transaction_id();

        // Capture the post-update row for RETURNING clause
        let mut captured_row: Option<Row> = None;

        // The keys the setter changes are asked after the write: a row whose
        // foreign key moved, and each value a child references that it gave up
        let referencing =
            super::foreign_key::find_referencing_fks(&self.engine, &schema.table_name);
        let mut fks_by_ref_col: FxHashMap<usize, Vec<(String, crate::core::ForeignKeyConstraint)>> =
            FxHashMap::default();
        for (child, fk) in referencing.iter() {
            if let Some(idx) = schema.get_column_index(&fk.referenced_column) {
                fks_by_ref_col
                    .entry(idx)
                    .or_default()
                    .push((child.clone(), fk.clone()));
            }
        }
        let mut fk_rows: Vec<Row> = Vec::new();
        let mut ref_changes: Vec<(usize, Value, Value)> = Vec::new();
        // A row whose primary key the update changes moves to its new id (move_rows)
        let key_set = schema
            .pk_column_index()
            .filter(|pk| effective.compiled_updates.iter().any(|(at, ..)| at == pk));
        let mut moves: Vec<(i64, Row)> = Vec::new();

        // Prepared programs read the row met and the excluded row as one row
        let mut joined = conflict.map(|conflict| conflict.joined.borrow_mut());
        let conflict = conflict.map(|conflict| conflict.runner);

        // Create a setter function that applies the ON DUPLICATE KEY UPDATE
        let mut setter = |id: i64, row: Row| -> Result<(Row, bool)> {
            let mut row = match effective.widens {
                true => super::prepared_select::widened(row, id),
                false => row,
            };
            if let Some(joined) = joined.as_mut() {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::prepared_dml_site_ran(
                    crate::test_failpoints::PreparedDmlSite::ConflictUpdate,
                );
                joined.clear();
                joined.extend_from_slice(row.as_slice());
                joined.extend_from_slice(insert_values);
            }
            // DO UPDATE ... WHERE leaves the row as it is where it does not hold
            if let Some(program) = &effective.compiled_where {
                let exec_ctx = match (conflict, joined.as_deref()) {
                    (Some(run), Some(joined)) => run.context(joined),
                    _ => ExecuteContext::for_join(&row, &excluded_row).with_run(
                        params,
                        transaction_id,
                        None,
                    ),
                };
                if !vm.execute_bool_checked(program, &exec_ctx)? {
                    if effective.widens {
                        row.pop();
                    }
                    return Ok((row, false));
                }
            }

            // Collect all updates first to avoid borrow conflicts
            let updates_to_apply: Vec<(usize, Value)> = {
                // Use for_join to make EXCLUDED columns available as row2
                let exec_ctx = match (conflict, joined.as_deref()) {
                    (Some(run), Some(joined)) => run.context(joined),
                    _ => ExecuteContext::for_join(&row, &excluded_row).with_run(
                        params,
                        transaction_id,
                        None,
                    ),
                };

                let mut updates = Vec::with_capacity(effective.compiled_updates.len());
                for (idx, col_type, vec_dims, program) in &effective.compiled_updates {
                    let coerced = vm
                        .execute(program, &exec_ctx)?
                        .try_into_coerce_to_type(*col_type)?;
                    // Validate vector dimensions
                    if *col_type == DataType::Vector && *vec_dims > 0 {
                        if let Value::Extension(data) = &coerced {
                            if data.first() == Some(&(DataType::Vector as u8)) {
                                let got_dim =
                                    u16::try_from((data.len() - 1) / 4).unwrap_or(u16::MAX);
                                if got_dim != *vec_dims {
                                    return Err(Error::VectorDimensionMismatch {
                                        expected: *vec_dims,
                                        got: got_dim,
                                    });
                                }
                            }
                        }
                    }
                    updates.push((*idx, coerced));
                }
                updates
            };

            // Apply the updates, then each CHECK an updated column reads over the row
            let changed = !updates_to_apply.is_empty();
            let original = key_set.map(|_| row.clone());
            let mut fk_moved = false;
            for (idx, new_value) in updates_to_apply {
                if let Some(old) = row.get(idx).filter(|old| **old != new_value) {
                    if !old.is_null() && fks_by_ref_col.contains_key(&idx) {
                        ref_changes.push((idx, old.clone(), new_value.clone()));
                    }
                    fk_moved |= schema.foreign_keys.iter().any(|fk| fk.column_index == idx);
                }
                let _ = row.set(idx, new_value);
            }
            if changed {
                for check in &effective.compiled_checks {
                    run_check(&mut vm, check, &row)?;
                }
            }

            // The row written has the table's columns; one RETURNING reads has its id too
            if effective.widens {
                row.pop();
            }
            if fk_moved {
                fk_rows.push(row.clone());
            }
            if capture_row {
                captured_row = Some(match returns_id {
                    true => super::prepared_select::widened(row.clone(), id),
                    false => row.clone(),
                });
            }
            if let (Some(key), Some(mut original)) = (key_set, original) {
                if row.get(key) != original.get(key) {
                    if effective.widens {
                        original.pop();
                    }
                    moves.push((id, row));
                    return Ok((original, true));
                }
            }

            Ok((row, changed))
        };

        // Prefer direct row_id lookup when we have a concrete conflicting row_id.
        // This avoids a second scan on non-PK upserts after conflict resolution.
        // row_id < 0 means "unknown" (sentinel from UniqueConstraint error).
        // A row DO UPDATE ... WHERE left alone is not an updated row
        let updated = if row_id >= 0 {
            table.update_by_row_ids(&[row_id], &mut setter)?
        } else {
            let plan = where_expr.as_deref().and_then(|e| table.index_plan(e));
            let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
            table.update(access, where_expr.as_deref(), &mut setter)?
        };
        move_rows(&mut **table, moves)?;

        let txn_id = table.txn_id();
        for row in &fk_rows {
            super::foreign_key::check_inserted_parents(
                &self.engine,
                txn_id,
                schema,
                row,
                pending_parents,
            )?;
        }
        let mut by_column: Vec<(usize, Vec<(Value, Value)>)> = Vec::new();
        for (idx, old, new) in ref_changes {
            let Some(fks) = fks_by_ref_col.get(&idx) else {
                continue;
            };
            if super::foreign_key::fk_tree_needs_precheck(&self.engine, fks) {
                super::foreign_key::pre_check_restrict_for_update(
                    &self.engine,
                    txn_id,
                    &schema.table_name,
                    &old,
                    fks,
                )?;
            }
            by_column.push((idx, vec![(old, new)]));
        }
        if !by_column.is_empty() {
            super::foreign_key::enforce_update_actions(
                &self.engine,
                txn_id,
                &schema.table_name,
                by_column,
                Some(pending_parents),
            )?;
        }

        Ok((updated > 0, captured_row))
    }

    /// Find a row by unique index value (supports single and composite unique constraints).
    /// Uses direct index lookup (O(1) hash) instead of full table scan.
    fn find_row_by_unique_index(
        &self,
        table: &dyn Table,
        schema: &crate::core::Schema,
        index_name: &str,
        column_name: &str,
        row_values: &[Value],
    ) -> Result<Option<i64>> {
        // Try direct index lookup first (O(1) for hash/multi-column indexes)
        if let Some(index) = table.get_index(index_name) {
            // Build the value array in the index's column order
            let col_ids = index.column_ids();
            let mut lookup_values: Vec<Value> = Vec::with_capacity(col_ids.len());
            for &col_id in col_ids {
                let value = row_values
                    .get(col_id as usize)
                    .cloned()
                    .unwrap_or(Value::null_unknown());
                lookup_values.push(value);
            }

            let row_ids = index.get_row_ids_equal(&lookup_values);
            if let Some(&row_id) = row_ids.first() {
                return Ok(Some(row_id));
            }
            // Hot index didn't find it — fall through to scan-based lookup
            // which searches both hot buffer AND cold segments.
        }

        if let Some(row_id) =
            table.find_unique_conflict_row_id(index_name, column_name, row_values)?
        {
            return Ok(Some(row_id));
        }

        // Fallback: scan with filter expression (searches hot + cold via SegmentedTable)
        let col_names: Vec<&str> = column_name.split(", ").collect();

        let mut comparisons: Vec<Box<dyn StorageExpr>> = Vec::with_capacity(col_names.len());
        for col_name in &col_names {
            let col_lower = col_name.to_lowercase();
            let col_idx = match schema.column_index_map().get(&col_lower) {
                Some(&idx) => idx,
                None => return Ok(None),
            };
            let value = row_values
                .get(col_idx)
                .cloned()
                .unwrap_or(Value::null_unknown());

            let mut expr = ComparisonExpr::new(*col_name, crate::core::Operator::Eq, value);
            expr.prepare_for_schema(schema);
            comparisons.push(Box::new(expr));
        }

        let scan_expr: Box<dyn StorageExpr> = if comparisons.len() == 1 {
            comparisons.pop().unwrap()
        } else {
            use crate::storage::expression::AndExpr;
            let mut and_expr = AndExpr::new(comparisons);
            and_expr.prepare_for_schema(schema);
            Box::new(and_expr)
        };

        // Only project the PK column (if any) — we only need the row_id,
        // not the full row. This avoids materializing all columns.
        let pk_idx = schema.pk_column_index();
        let column_indices: Vec<usize> = if let Some(pk) = pk_idx {
            vec![pk]
        } else {
            vec![0]
        };
        let plan = table.index_plan(&*scan_expr);
        let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
        let mut scanner = table.scan_access(access, &column_indices, None, Some(&*scan_expr))?;

        let result = if scanner.next() {
            let row_id = scanner.current_row_id();
            if row_id >= 0 {
                Some(row_id)
            } else if pk_idx.is_some() {
                // PK column is at index 0 in our minimal projection
                let row = scanner.row();
                if let Some(Value::Integer(id)) = row.get(0) {
                    Some(*id)
                } else {
                    Some(row_id)
                }
            } else {
                Some(row_id)
            }
        } else {
            None
        };

        let scan_error = scanner.err().cloned();
        scanner.close()?;
        if let Some(error) = scan_error {
            return Err(error);
        }
        Ok(result)
    }

    /// The value of column `column`'s default text, evaluated once
    pub(crate) fn evaluate_default_expr(
        &self,
        default_expr: &str,
        column: &str,
        target_type: crate::core::DataType,
    ) -> Result<Value> {
        let expr = parse_stored_expression(default_expr, "default", column)?;
        // Constant expression - no row context needed
        let value = super::expression::evaluate_once(&expr, None)?;
        value.try_into_coerce_to_type(target_type)
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn finish_returning(
        &self,
        returning: &[Expression],
        source_rows: Vec<Row>,
        column_names: &[String],
        ctx: &ExecutionContext,
        standalone_tx: Option<Box<dyn Transaction>>,
        prepared: Option<&super::prepared_select::DmlRunner<'_>>,
        held: Option<&[Result<super::expression::SharedProgram>]>,
    ) -> Result<Box<dyn QueryResult>> {
        let result =
            self.build_returning_result(returning, source_rows, column_names, ctx, prepared, held)?;
        // A statement whose reads a truncate or drop overtook commits nothing
        ctx.view_held()?;
        if let Some(mut tx) = standalone_tx {
            tx.commit()?;
        }
        Ok(result)
    }

    /// Build a result from RETURNING clause expressions
    ///
    /// Evaluates the RETURNING expressions for each affected row and returns
    /// the results as a QueryResult; a `prepared` RETURNING runs its own
    /// programs, which read its blocks, else the plan's `held` ones run.
    fn build_returning_result(
        &self,
        returning: &[Expression],
        source_rows: Vec<Row>,
        column_names: &[String],
        ctx: &ExecutionContext,
        prepared: Option<&super::prepared_select::DmlRunner<'_>>,
        held: Option<&[Result<super::expression::SharedProgram>]>,
    ) -> Result<Box<dyn QueryResult>> {
        let result_columns = match prepared.and_then(|run| run.returning_names()) {
            Some(names) => CompactArc::clone(names),
            None => CompactArc::new(Self::returning_names(returning, column_names)),
        };

        // If no source rows, return empty result
        if source_rows.is_empty() {
            return Ok(Box::new(ExecutorResult::with_arc_columns(
                result_columns,
                RowVec::new(),
            )));
        }

        use super::expression::ExprVM;

        if let Some((run, programs)) = prepared.and_then(|run| Some((run, run.returning()?))) {
            let mut vm = ExprVM::new();
            let mut result_rows = RowVec::with_capacity(source_rows.len());
            let mut bound: smallvec::SmallVec<[_; 4]> =
                programs.iter().map(|program| run.bind(program)).collect();
            for (row_id, row) in source_rows.into_iter().enumerate() {
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::prepared_dml_site_ran(
                    crate::test_failpoints::PreparedDmlSite::Returning,
                );
                let exec_ctx = run.context(&row);
                let mut row_values = Vec::with_capacity(programs.len());
                for program in &mut bound {
                    row_values.push(vm.execute(run.current(program), &exec_ctx)?);
                }
                result_rows.push((row_id as i64, Row::from_values(row_values)));
            }
            return Ok(Box::new(ExecutorResult::with_arc_columns(
                result_columns,
                result_rows,
            )));
        }

        let compiled;
        let programs = match held {
            Some(held) => held,
            None => {
                compiled = returning_programs(returning, column_names, None);
                &compiled[..]
            }
        };
        if let Some(Err(e)) = programs.iter().find(|program| program.is_err()) {
            return Err(e.clone());
        }
        // Bound to this run's parameters once, before the rows
        let programs: SmallVec<[_; 8]> = programs
            .iter()
            .flatten()
            .map(|program| {
                super::expression::Program::bound(CompactArc::clone(program), ctx.params())
            })
            .collect();

        // Create VM for execution (reused for all rows)
        let mut vm = ExprVM::new();

        // Evaluate RETURNING expressions for each row
        let mut result_rows = RowVec::with_capacity(source_rows.len());
        for (row_id, row) in source_rows.into_iter().enumerate() {
            // CRITICAL: Include params from context for parameterized queries
            let exec_ctx = super::aggregation::slot_context(&row, ctx);

            let mut row_values = Vec::with_capacity(programs.len());
            for program in &programs {
                // CRITICAL: Propagate errors instead of silently returning NULL
                let value = vm.execute(program, &exec_ctx)?;
                row_values.push(value);
            }
            result_rows.push((row_id as i64, Row::from_values(row_values)));
        }

        Ok(Box::new(ExecutorResult::with_arc_columns(
            result_columns,
            result_rows,
        )))
    }

    /// The names of the columns `returning` gives over the target's
    /// `columns`, a star expanded
    pub(crate) fn returning_names(returning: &[Expression], columns: &[String]) -> Vec<String> {
        let mut names = Vec::with_capacity(returning.len());
        for expr in returning {
            match expr {
                Expression::Star(_) => names.extend(columns.iter().cloned()),
                _ => names.push(super::utils::item_name(expr)),
            }
        }
        names
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

    #[test]
    fn a_stored_text_that_does_not_parse_is_an_error() {
        let schema = crate::core::SchemaBuilder::new("t")
            .add_primary_key("id", DataType::Integer)
            .add_with_constraints(
                "v",
                DataType::Integer,
                true,
                false,
                false,
                Some("(((".to_string()),
                Some("v <".to_string()),
            )
            .build();
        assert!(compile_check_programs(&schema).is_err(), "the CHECK");
        let (_, programs) = build_default_slots(&schema);
        assert!(
            matches!(programs[1], Some((Err(_), _))),
            "the DEFAULT, raised by a row that takes it"
        );
        let executor = create_test_executor();
        assert!(executor
            .evaluate_default_expr("(((", "v", DataType::Integer)
            .is_err());
    }

    fn setup_test_table(executor: &Executor) {
        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            .unwrap();
    }

    #[test]
    fn test_insert_single_row() {
        let executor = create_test_executor();
        setup_test_table(&executor);

        let result = executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        assert_eq!(result.rows_affected(), 1);
    }

    #[test]
    fn test_insert_multiple_rows() {
        let executor = create_test_executor();
        setup_test_table(&executor);

        let result = executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)")
            .unwrap();
        assert_eq!(result.rows_affected(), 2);
    }

    #[test]
    fn test_insert_and_select() {
        let executor = create_test_executor();
        setup_test_table(&executor);

        executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();

        let mut result = executor.execute("SELECT * FROM users").unwrap();
        assert!(result.next());
        let row = result.row();
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::text("Alice")));
        assert_eq!(row.get(2), Some(&Value::Integer(30)));
    }

    #[test]
    fn test_type_coercion_insert_int_to_float() {
        let executor = create_test_executor();
        // Create table with FLOAT column
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();

        // Insert integer into float column - should coerce 5 -> 5.0
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5)")
            .unwrap();

        let mut result = executor.execute("SELECT price FROM products").unwrap();
        assert!(result.next());
        let row = result.row();
        // Value should be Float(5.0), not Integer(5)
        assert_eq!(row.get(0), Some(&Value::Float(5.0)));
    }

    #[test]
    fn test_type_coercion_insert_float_to_int() {
        let executor = create_test_executor();
        // Create table with INTEGER column
        executor
            .execute("CREATE TABLE counts (id INTEGER PRIMARY KEY, amount INTEGER)")
            .unwrap();

        // Insert float into integer column - should coerce 5.9 -> 5
        executor
            .execute("INSERT INTO counts (id, amount) VALUES (1, 5.9)")
            .unwrap();

        let mut result = executor.execute("SELECT amount FROM counts").unwrap();
        assert!(result.next());
        let row = result.row();
        // Value should be Integer(5), not Float(5.9)
        assert_eq!(row.get(0), Some(&Value::Integer(5)));
    }

    #[test]
    fn test_type_coercion_where_int_literal_on_float_column() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5.0)")
            .unwrap();

        // Query with integer literal against float column
        let mut result = executor
            .execute("SELECT * FROM products WHERE price = 5")
            .unwrap();
        assert!(result.next(), "Should find row with WHERE price = 5");
    }

    #[test]
    fn test_type_coercion_where_float_literal_on_int_column() {
        let executor = create_test_executor();
        setup_test_table(&executor);
        executor
            .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();

        // Query with float literal against integer column
        let mut result = executor
            .execute("SELECT * FROM users WHERE age = 30.0")
            .unwrap();
        assert!(result.next(), "Should find row with WHERE age = 30.0");
    }

    #[test]
    fn test_type_coercion_between() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5.0)")
            .unwrap();

        // BETWEEN with integer literals against float column
        let mut result = executor
            .execute("SELECT * FROM products WHERE price BETWEEN 4 AND 6")
            .unwrap();
        assert!(result.next(), "Should find row with BETWEEN 4 AND 6");
    }

    #[test]
    fn test_type_coercion_in_list() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)")
            .unwrap();
        executor
            .execute("INSERT INTO products (id, price) VALUES (1, 5.0)")
            .unwrap();

        // IN with integer literals against float column
        let mut result = executor
            .execute("SELECT * FROM products WHERE price IN (4, 5, 6)")
            .unwrap();
        assert!(result.next(), "Should find row with IN (4, 5, 6)");
    }

    #[test]
    fn test_type_coercion_insert_text_to_timestamp() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE events (id INTEGER PRIMARY KEY, created_at TIMESTAMP)")
            .unwrap();

        // Insert text string into timestamp column - should parse to timestamp
        executor
            .execute("INSERT INTO events (id, created_at) VALUES (1, '2024-01-15 10:30:00')")
            .unwrap();

        let mut result = executor.execute("SELECT created_at FROM events").unwrap();
        assert!(result.next());
        let row = result.row();
        // Value should be Timestamp, not Text
        match row.get(0) {
            Some(Value::Timestamp(_)) => {} // Success
            other => panic!("Expected Timestamp, got {:?}", other),
        }
    }

    #[test]
    fn test_type_coercion_where_text_on_timestamp_column() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE events (id INTEGER PRIMARY KEY, created_at TIMESTAMP)")
            .unwrap();
        executor
            .execute("INSERT INTO events (id, created_at) VALUES (1, '2024-01-15 10:30:00')")
            .unwrap();

        // Query with text literal against timestamp column
        let mut result = executor
            .execute("SELECT * FROM events WHERE created_at > '2024-01-01'")
            .unwrap();
        assert!(
            result.next(),
            "Should find row with WHERE created_at > '2024-01-01'"
        );

        // Query with exact match
        let mut result = executor
            .execute("SELECT * FROM events WHERE created_at = '2024-01-15 10:30:00'")
            .unwrap();
        assert!(
            result.next(),
            "Should find row with WHERE created_at = '2024-01-15 10:30:00'"
        );
    }

    #[test]
    fn test_type_coercion_timestamp_between() {
        let executor = create_test_executor();
        executor
            .execute("CREATE TABLE events (id INTEGER PRIMARY KEY, created_at TIMESTAMP)")
            .unwrap();
        executor
            .execute("INSERT INTO events (id, created_at) VALUES (1, '2024-01-15 10:30:00')")
            .unwrap();

        // BETWEEN with text literals against timestamp column
        let mut result = executor
            .execute("SELECT * FROM events WHERE created_at BETWEEN '2024-01-01' AND '2024-02-01'")
            .unwrap();
        assert!(
            result.next(),
            "Should find row with BETWEEN '2024-01-01' AND '2024-02-01'"
        );
    }
}
