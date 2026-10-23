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

// Expression Compiler
//
// Transforms AST Expressions into compiled Programs.
// This is where the magic happens - we convert recursive AST into linear bytecode.
//
// Design principles:
// 1. Resolve everything at compile time (column indices, function pointers, patterns)
// 2. Flatten recursion into linear instruction sequences
// 3. Handle short-circuit evaluation with jumps
// 4. Pre-compute constant expressions where possible

use std::cell::Cell;
use std::sync::Arc;

use crate::common::CompactArc;
use crate::common::SmartString;
use crate::common::StringMap;
use rustc_hash::{FxHashMap, FxHashSet};

use super::ops::{IntervalStep, Op, QuantifiedMembers};
use super::program::{Program, ProgramBuilder};
use super::vm::{ExecuteContext, ExprVM};
use crate::core::interval::IntervalUnit;
use crate::core::pattern::Pattern;
use crate::core::predicate::PredicateSet;
use crate::core::{DataType, Row, Value, ValueSet};
use crate::executor::expr_key::ExprKey;
use crate::executor::utils::{
    derived_aggregate_column_name, extract_literal_value, string_literal_value, string_to_datatype,
};
use crate::executor::volatility::expression_volatility;
use crate::functions::{global_registry, FunctionRegistry, Volatility};
use crate::parser::ast::*;
use smallvec::SmallVec;

/// Compilation error
#[derive(Debug, Clone)]
pub enum CompileError {
    /// Column not found
    ColumnNotFound(String),
    /// Function not found
    FunctionNotFound(String),
    /// Invalid expression
    InvalidExpression(String),
    /// Unsupported expression type
    UnsupportedExpression(String),
    /// Type error
    TypeError(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::ColumnNotFound(name) => {
                write!(f, "Column '{}' not found", name)
            }
            CompileError::FunctionNotFound(name) => write!(f, "Function not found: {}", name),
            CompileError::InvalidExpression(msg) => write!(f, "Invalid expression: {}", msg),
            CompileError::UnsupportedExpression(msg) => {
                write!(f, "Unsupported expression: {}", msg)
            }
            CompileError::TypeError(msg) => write!(f, "Type error: {}", msg),
        }
    }
}

impl std::error::Error for CompileError {}

/// An INTERVAL literal, negated or not, alone or times the expressions in `by`
struct IntervalOperand<'a> {
    literal: &'a IntervalLiteral,
    negated: bool,
    by: SmallVec<[&'a Expression; 2]>,
}

impl IntervalOperand<'_> {
    fn step(&self) -> Result<IntervalStep, CompileError> {
        let unit = self
            .literal
            .value
            .split_whitespace()
            .nth(1)
            .and_then(IntervalUnit::parse)
            .ok_or_else(|| {
                CompileError::TypeError(format!("unknown INTERVAL unit in {}", self.literal))
            })?;
        let quantity = if self.negated {
            self.literal.quantity.checked_neg().ok_or_else(|| {
                CompileError::TypeError(format!("-{} is out of range", self.literal))
            })?
        } else {
            self.literal.quantity
        };
        Ok(IntervalStep { quantity, unit })
    }
}

/// Result of column resolution - indicates which row the column is from
#[derive(Debug, Clone, Copy)]
pub enum ColumnSource {
    /// Column from first row with given index
    Row1(u16),
    /// Column from second row (for joins) with given index
    Row2(u16),
}

/// What a name reads
#[derive(Debug)]
pub(crate) enum Reference {
    Column(ColumnSource),
    /// A column of the outer row, by its name there
    Outer(CompactArc<str>),
    /// A double-quoted name no column has: its own text
    Text,
    /// No column has it: the name as the error spells it
    Missing(String),
}

impl Reference {
    /// The column of the first row it reads, when it reads one
    pub(crate) fn row_column(&self) -> Option<u16> {
        match self {
            Reference::Column(ColumnSource::Row1(idx)) => Some(*idx),
            _ => None,
        }
    }
}

/// Compilation context
///
/// Contains all the information needed to compile expressions:
/// - Column name to index mapping
/// - Function registry
/// - Outer query columns (for correlated subqueries)
pub struct CompileContext<'a> {
    /// Column name -> index (case-insensitive)
    columns: StringMap<u16>,

    /// Qualified column name -> (index, is_row2)
    /// For row1 columns: index is the direct row1 index
    /// For row2 columns: index is the row2 index (not offset)
    qualified_columns: StringMap<StringMap<ColumnSource>>,

    /// Second row columns (for joins)
    columns2: Option<StringMap<u16>>,

    /// Tables that belong to row2 (for tracking which tables are from second row)
    row2_tables: FxHashSet<String>,

    /// Outer query columns (for correlated subqueries)
    outer_columns: Option<FxHashMap<CompactArc<str>, u16>>,

    /// Function registry
    functions: &'a FunctionRegistry,

    /// The aggregated row's slot of each group item and aggregate, by identity
    expression_aliases: FxHashMap<ExprKey, u16>,

    /// Column aliases
    column_aliases: StringMap<String>,

    /// A prepared plan compiles: a subquery mark compiles to its block's id
    prepared_subqueries: bool,

    /// Identifiers read from the second row at a fixed position, by node address
    row2_bindings: Option<&'a FxHashMap<usize, u16>>,

    /// Values read as the type of the column they are compared with, by node address
    read_as: Option<&'a FxHashMap<usize, DataType>>,

    /// The parameters of the one execution the program serves, which folding reads
    params: Option<&'a [Value]>,
}

impl<'a> CompileContext<'a> {
    /// Create a new compilation context
    pub fn new(columns: &[String], functions: &'a FunctionRegistry) -> Self {
        let mut col_map = StringMap::new();
        let mut qualified_map: StringMap<StringMap<ColumnSource>> = StringMap::new();

        for (i, col) in columns.iter().enumerate() {
            let lower = col.to_lowercase();
            col_map.insert(lower.clone(), i as u16);

            // Handle qualified names (table.column)
            if let Some(dot_idx) = col.rfind('.') {
                let table = col[..dot_idx].to_lowercase();
                let column = col[dot_idx + 1..].to_lowercase();
                qualified_map
                    .entry(table)
                    .or_default()
                    .insert(column.clone(), ColumnSource::Row1(i as u16));

                // Also map unqualified column name for lookup without table prefix
                // Don't overwrite if already exists (first occurrence wins)
                col_map.entry(column).or_insert(i as u16);
            }
        }

        Self {
            columns: col_map,
            qualified_columns: qualified_map,
            columns2: None,
            row2_tables: FxHashSet::default(),
            outer_columns: None,
            functions,
            expression_aliases: FxHashMap::default(),
            column_aliases: StringMap::new(),
            prepared_subqueries: false,
            row2_bindings: None,
            read_as: None,
            params: None,
        }
    }

    /// Create context using global function registry
    pub fn with_global_registry(columns: &[String]) -> Self {
        Self::new(columns, global_registry())
    }

    /// Add second row columns (for join compilation)
    pub fn with_second_row(mut self, columns2: &[String]) -> Self {
        let mut col_map = StringMap::new();
        for (i, col) in columns2.iter().enumerate() {
            let lower = col.to_lowercase();
            col_map.insert(lower.clone(), i as u16);

            // Handle qualified names (table.column)
            if let Some(dot_idx) = col.rfind('.') {
                let table = col[..dot_idx].to_lowercase();
                let column = col[dot_idx + 1..].to_lowercase();

                // Track this table as belonging to row2
                self.row2_tables.insert(table.clone());

                // Add qualified name with Row2 source (index is local to row2)
                self.qualified_columns
                    .entry(table)
                    .or_default()
                    .insert(column.clone(), ColumnSource::Row2(i as u16));

                // Also map unqualified column name (don't overwrite if exists)
                col_map.entry(column).or_insert(i as u16);
            }
        }
        self.columns2 = Some(col_map);
        self
    }

    /// Add outer columns for correlated subqueries
    pub fn with_outer_columns(mut self, outer_cols: &[String]) -> Self {
        let mut map = FxHashMap::default();
        for (i, col) in outer_cols.iter().enumerate() {
            map.insert(CompactArc::from(col.to_lowercase().as_str()), i as u16);
        }
        self.outer_columns = Some(map);
        self
    }

    /// Add expression aliases (for HAVING clause)
    pub(crate) fn with_expression_aliases(mut self, aliases: FxHashMap<ExprKey, u16>) -> Self {
        self.expression_aliases = aliases;
        self
    }

    /// Add column aliases
    pub fn with_column_aliases(mut self, aliases: StringMap<String>) -> Self {
        self.column_aliases = aliases;
        self
    }

    /// Compile each subquery mark to the prepared id it carries; a written
    /// subquery is then a compile error
    pub fn with_prepared_subqueries(mut self) -> Self {
        self.prepared_subqueries = true;
        self
    }

    /// Compile each listed identifier node to the second row's column at
    /// its position, whatever its name; other names resolve as usual
    pub fn with_row2_bindings(mut self, bindings: &'a FxHashMap<usize, u16>) -> Self {
        self.row2_bindings = Some(bindings);
        self
    }

    /// Read each listed expression node's value as the type the scan reads
    /// a value compared with a column of that type as
    pub fn with_read_as(mut self, read_as: &'a FxHashMap<usize, DataType>) -> Self {
        self.read_as = Some(read_as);
        self
    }

    /// Fold what reads only `params` and constants, for a program compiled
    /// for one execution and dropped with it; a fold that errors is not taken
    pub fn with_params(mut self, params: &'a [Value]) -> Self {
        self.params = Some(params);
        self
    }

    /// Resolve a column name to its source (Row1 or Row2)
    fn resolve_column(&self, name: &str) -> Option<ColumnSource> {
        let lower = name.to_lowercase();

        // Check column aliases first
        if let Some(original) = self.column_aliases.get(&lower) {
            if let Some(&idx) = self.columns.get(original) {
                return Some(ColumnSource::Row1(idx));
            }
        }

        // Direct lookup in primary columns (Row1)
        if let Some(&idx) = self.columns.get(&lower) {
            return Some(ColumnSource::Row1(idx));
        }

        // Try second row if available (Row2)
        if let Some(ref cols2) = self.columns2 {
            if let Some(&idx) = cols2.get(&lower) {
                return Some(ColumnSource::Row2(idx));
            }
        }

        None
    }

    /// Resolve a qualified column name (table.column)
    fn resolve_qualified(&self, table: &str, column: &str) -> Option<ColumnSource> {
        let table_lower = table.to_lowercase();
        let column_lower = column.to_lowercase();

        if let Some(table_cols) = self.qualified_columns.get(&table_lower) {
            if let Some(&source) = table_cols.get(&column_lower) {
                return Some(source);
            }
        }

        // Check if the FULLY QUALIFIED name (table.column) exists in outer_columns.
        // This distinguishes between `t.id` (outer reference) and `t2.id` (current row).
        // Only if the qualified name is in outer context should we skip the fallback.
        if let Some(ref outer_cols) = self.outer_columns {
            let qualified_name = format!("{}.{}", table_lower, column_lower);
            if outer_cols.contains_key(qualified_name.as_str()) {
                // Qualified name exists in outer context - don't fall back
                return None;
            }
        }

        // Qualified name not in outer context - safe to fall back to unqualified lookup
        self.resolve_column(&column_lower)
    }

    /// What a bare or qualified name reads, by the one rule every compiled
    /// name is read by; None for an expression that is not a name
    pub(crate) fn reference(&self, expr: &Expression) -> Option<Reference> {
        let found = |source: Option<ColumnSource>| source.map(Reference::Column);
        Some(match expr {
            Expression::Identifier(id) => match id.value_lower.rfind('.') {
                Some(dot) => {
                    let column = &id.value_lower[dot + 1..];
                    found(self.resolve_qualified(&id.value_lower[..dot], column))
                        .or_else(|| self.resolve_outer_column(column).map(Reference::Outer))
                }
                None => found(self.resolve_column(&id.value_lower)).or_else(|| {
                    self.resolve_outer_column(&id.value_lower)
                        .map(Reference::Outer)
                }),
            }
            .unwrap_or_else(|| match id.quoted {
                true => Reference::Text,
                false => Reference::Missing(id.value.to_string()),
            }),
            Expression::QualifiedIdentifier(qid) => {
                let (table, column) = (&qid.qualifier.value_lower, &qid.name.value_lower);
                // The qualified name in the outer row first: its bare name may
                // hold an inner row's column of the same name
                found(self.resolve_qualified(table, column))
                    .or_else(|| {
                        self.resolve_outer_column(&format!("{table}.{column}"))
                            .or_else(|| self.resolve_outer_column(column))
                            .map(Reference::Outer)
                    })
                    .unwrap_or_else(|| Reference::Missing(format!("{table}.{column}")))
            }
            _ => return None,
        })
    }

    /// Resolve outer column (for correlated subqueries)
    fn resolve_outer_column(&self, name: &str) -> Option<CompactArc<str>> {
        let lower = name.to_lowercase();
        self.outer_columns.as_ref().and_then(|cols| {
            if cols.contains_key(lower.as_str()) {
                Some(CompactArc::from(lower.as_str()))
            } else {
                None
            }
        })
    }

    /// Check if an expression matches an expression alias
    fn check_expression_alias(&self, expr: &Expression) -> Option<u16> {
        if self.expression_aliases.is_empty() {
            return None;
        }
        self.expression_aliases.get(&ExprKey::of(expr)).copied()
    }
}

/// Expression compiler
pub struct ExprCompiler<'a> {
    ctx: &'a CompileContext<'a>,
    /// Guard flag to prevent recursive constant folding
    folding: Cell<bool>,
}

impl<'a> ExprCompiler<'a> {
    pub fn new(ctx: &'a CompileContext<'a>) -> Self {
        Self {
            ctx,
            folding: Cell::new(false),
        }
    }

    /// The prepared id of a subquery block. Only a prepared plan compiles one
    /// into the program; any other caller substitutes its result first
    fn subquery_id(&self, block: &SelectStatement, form: &str) -> Result<u16, CompileError> {
        if !self.ctx.prepared_subqueries {
            return Err(CompileError::UnsupportedExpression(format!(
                "{form} outside a prepared plan"
            )));
        }
        block
            .prepared
            .ok_or_else(|| CompileError::InvalidExpression(format!("unprepared {form}")))
    }

    /// Compile an expression into a Program
    pub fn compile(&self, expr: &Expression) -> Result<Program, CompileError> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::program_compiled();
        let mut builder = ProgramBuilder::new();
        self.compile_expr(expr, &mut builder)?;
        builder.emit(Op::Return);
        Ok(builder.build())
    }

    /// Flatten chained concatenation operators into a list of operands.
    /// For `a || b || c || d`, returns [a, b, c, d] in order.
    fn flatten_concat_chain_infix<'b>(
        infix: &'b InfixExpression,
        operands: &mut Vec<&'b Expression>,
    ) {
        // Flatten left side
        if let Expression::Infix(left_infix) = &*infix.left {
            if left_infix.op_type == InfixOperator::Concat {
                Self::flatten_concat_chain_infix(left_infix, operands);
            } else {
                operands.push(&infix.left);
            }
        } else {
            operands.push(&infix.left);
        }
        // Flatten right side
        if let Expression::Infix(right_infix) = &*infix.right {
            if right_infix.op_type == InfixOperator::Concat {
                Self::flatten_concat_chain_infix(right_infix, operands);
            } else {
                operands.push(&infix.right);
            }
        } else {
            operands.push(&infix.right);
        }
    }

    /// Try to fold a column-free expression into a constant at compile time.
    /// Compiles the expression into a temporary program, executes it with an empty
    /// row context, and returns the result if successful.
    fn try_fold_constant(&self, expr: &Expression) -> Option<Value> {
        use std::cell::RefCell;

        thread_local! {
            static FOLD_VM: RefCell<ExprVM> = RefCell::new(ExprVM::new());
            static FOLD_ROW: Row = Row::new();
            static FOLD_OPS: RefCell<Vec<Op>> = const { RefCell::new(Vec::new()) };
        }

        self.folding.set(true);

        let empty_cols: &[String] = &[];
        let ctx = CompileContext::new(empty_cols, self.ctx.functions);
        let compiler = ExprCompiler::new(&ctx);
        compiler.folding.set(true);

        let ops = FOLD_OPS.with(|ops| std::mem::take(&mut *ops.borrow_mut()));
        let mut builder = ProgramBuilder::reusing(ops);
        let ok = compiler.compile_expr(expr, &mut builder);
        self.folding.set(false);

        ok.ok()?;
        builder.emit(Op::Return);
        let program = builder.build_unoptimized();

        let value = FOLD_ROW.with(|empty_row| {
            let exec_ctx =
                ExecuteContext::new(empty_row).with_params(self.ctx.params.unwrap_or(&[]));
            FOLD_VM.with(|vm_cell| {
                let mut vm = vm_cell.borrow_mut();
                vm.execute(&program, &exec_ctx).ok()
            })
        });
        let mut spent = program.into_ops();
        spent.clear();
        FOLD_OPS.with(|ops| *ops.borrow_mut() = spent);
        value
    }

    /// Compile an expression, emitting ops to the builder
    fn compile_expr(
        &self,
        expr: &Expression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        self.compile_node(expr, builder)?;
        if let Some(&column_type) = self
            .ctx
            .read_as
            .and_then(|r| r.get(&(expr as *const Expression as usize)))
        {
            builder.emit(Op::ReadAsColumn(column_type));
        }
        Ok(())
    }

    fn compile_node(
        &self,
        expr: &Expression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        // Check if this expression matches an expression alias (for HAVING)
        if let Some(idx) = self.ctx.check_expression_alias(expr) {
            builder.emit(Op::LoadAggregateResult(idx));
            return Ok(());
        }
        // A derived table names an aggregate's column after it
        if let Some(source) =
            derived_aggregate_column_name(expr).and_then(|name| self.ctx.resolve_column(&name))
        {
            match source {
                ColumnSource::Row1(idx) => builder.emit(Op::LoadColumn(idx)),
                ColumnSource::Row2(idx) => builder.emit(Op::LoadColumn2(idx)),
            }
            return Ok(());
        }

        // Constant folding: if not already folding and expression is column-free
        // and non-trivial, evaluate once at compile time and emit as LoadConst.
        // This optimizes deterministic expressions like ABS(-5) + 1 or UPPER('text').
        // Non-deterministic functions (NOW, RANDOM, etc.) are excluded by is_foldable_expr
        // and handled separately by pushdown's try_eval_constant_expr at query time.
        if !self.folding.get() && is_foldable_expr(expr, self.ctx) {
            if let Some(value) = self.try_fold_constant(expr) {
                builder.emit(Op::LoadConst(value));
                return Ok(());
            }
        }

        match expr {
            // === LITERALS ===
            Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_) => {
                if let Some(value) = literal_value(expr) {
                    builder.emit(Op::LoadConst(value));
                }
            }

            Expression::NullLiteral(_) => {
                builder.emit(Op::LoadNull(DataType::Null));
            }

            // === IDENTIFIERS ===
            Expression::Identifier(id) => {
                if let Some(&position) = self
                    .ctx
                    .row2_bindings
                    .and_then(|b| b.get(&(id as *const Identifier as usize)))
                {
                    builder.emit(Op::LoadColumn2(position));
                    return Ok(());
                }
                // CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP are now parsed as
                // FunctionCall by the parser, so they no longer reach this path.
                // A double-quoted "true" or "false" names a column, not a literal.
                match id.value_lower.as_str() {
                    "true" if !id.quoted => {
                        builder.emit(Op::LoadConst(Value::Boolean(true)));
                        return Ok(());
                    }
                    "false" if !id.quoted => {
                        builder.emit(Op::LoadConst(Value::Boolean(false)));
                        return Ok(());
                    }
                    _ => {}
                }

                self.emit_reference(expr, builder)?;
            }

            Expression::QualifiedIdentifier(_) => {
                self.emit_reference(expr, builder)?;
            }

            // === PARAMETERS ===
            Expression::Parameter(param) => {
                let slot = parameter_slot(param)?;
                match self.ctx.params.and_then(|params| params.get(slot as usize)) {
                    Some(value) => builder.emit(Op::LoadConst(value.clone())),
                    None => builder.emit(Op::LoadParam(slot)),
                }
            }

            // === INFIX EXPRESSIONS ===
            Expression::Infix(infix) => {
                self.compile_infix(infix, builder)?;
            }

            // === PREFIX EXPRESSIONS ===
            Expression::Prefix(prefix) => {
                self.compile_prefix(prefix, builder)?;
            }

            // === IN EXPRESSION ===
            Expression::In(in_expr) => {
                self.compile_in(in_expr, builder)?;
            }

            Expression::InHashSet(in_hash) => {
                self.compile_expr(&in_hash.column, builder)?;
                let set = CompactArc::clone(&in_hash.values);
                if in_hash.not {
                    builder.emit(Op::NotInSet(set));
                } else {
                    builder.emit(Op::InSet(set));
                }
            }

            Expression::QuantifiedSet(quantified) => {
                let op = QuantifiedMembers::operator(&quantified.operator).ok_or_else(|| {
                    CompileError::UnsupportedExpression(format!(
                        "{} ANY or ALL",
                        quantified.operator
                    ))
                })?;
                self.compile_expr(&quantified.left, builder)?;
                builder.emit(Op::QuantifiedSet(CompactArc::new(QuantifiedMembers::new(
                    op,
                    quantified.all,
                    CompactArc::clone(&quantified.members),
                    quantified.has_null,
                ))));
            }

            // === BETWEEN EXPRESSION ===
            Expression::Between(between) => {
                self.compile_expr(&between.expr, builder)?;
                self.compile_expr(&between.lower, builder)?;
                self.compile_expr(&between.upper, builder)?;
                if between.not {
                    builder.emit(Op::NotBetween);
                } else {
                    builder.emit(Op::Between);
                }
            }

            // === LIKE EXPRESSION ===
            Expression::Like(like) => {
                self.compile_like(like, builder)?;
            }

            // === CASE EXPRESSION ===
            Expression::Case(case) => {
                self.compile_case(case, builder)?;
            }

            // === CAST EXPRESSION ===
            Expression::Cast(cast) => {
                self.compile_expr(&cast.expr, builder)?;
                // DATE type requires special handling - truncate time to midnight
                if cast.type_name.eq_ignore_ascii_case("DATE") {
                    builder.emit(Op::TruncateToDate);
                } else {
                    let dt = string_to_datatype(&cast.type_name);
                    builder.emit(Op::Cast(dt));
                }
            }

            // === FUNCTION CALL ===
            Expression::FunctionCall(func) => {
                self.compile_function(func, builder)?;
            }

            // === ALIASED EXPRESSION ===
            Expression::Aliased(aliased) => {
                self.compile_expr(&aliased.expression, builder)?;
            }

            // === DISTINCT ===
            Expression::Distinct(distinct) => {
                self.compile_expr(&distinct.expr, builder)?;
            }

            // === LIST ===
            Expression::List(list) => {
                if list.elements.is_empty() {
                    builder.emit(Op::LoadNull(DataType::Null));
                } else {
                    // Compile first item (for single-value IN)
                    self.compile_expr(&list.elements[0], builder)?;
                }
            }

            Expression::ExpressionList(list) => {
                if list.expressions.is_empty() {
                    builder.emit(Op::LoadNull(DataType::Null));
                } else {
                    self.compile_expr(&list.expressions[0], builder)?;
                }
            }

            // === INTERVAL ===
            Expression::IntervalLiteral(literal) => {
                let operand = IntervalOperand {
                    literal,
                    negated: false,
                    by: SmallVec::new(),
                };
                self.compile_interval(&operand, builder)?;
            }

            // === SUBQUERIES ===
            Expression::ScalarSubquery(sq) => {
                let id = self.subquery_id(&sq.subquery, "scalar subquery")?;
                builder.emit(Op::ExecScalarSubquery(id));
            }

            Expression::Exists(exists) => {
                let id = self.subquery_id(&exists.subquery, "EXISTS")?;
                builder.emit(Op::ExecExists(id));
            }

            Expression::AllAny(all_any) => {
                let id = self.subquery_id(&all_any.subquery, "ANY or ALL")?;
                let compare_op =
                    QuantifiedMembers::operator(&all_any.operator).ok_or_else(|| {
                        CompileError::UnsupportedExpression(format!(
                            "ANY or ALL with {}",
                            all_any.operator
                        ))
                    })?;
                self.compile_expr(&all_any.left, builder)?;
                let all = matches!(all_any.all_any_type, AllAnyType::All);
                builder.emit(Op::ExecQuantified(id, compare_op, all));
            }

            // === WINDOW (not supported in VM, requires special handling) ===
            Expression::Window(_) => {
                return Err(CompileError::UnsupportedExpression(
                    "Window functions require special execution context".to_string(),
                ));
            }

            // === TABLE SOURCES (not expressions) ===
            Expression::TableSource(_)
            | Expression::JoinSource(_)
            | Expression::SubquerySource(_)
            | Expression::ValuesSource(_)
            | Expression::CteReference(_)
            | Expression::FunctionTableSource(_)
            | Expression::Star(_)
            | Expression::QualifiedStar(_)
            | Expression::Default(_) => {
                return Err(CompileError::InvalidExpression(
                    "unexpected table reference or '*' in expression context".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Emit the load of the name `expr`, as `CompileContext::reference` reads it
    fn emit_reference(
        &self,
        expr: &Expression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        match self.ctx.reference(expr) {
            Some(Reference::Column(ColumnSource::Row1(idx))) => builder.emit(Op::LoadColumn(idx)),
            Some(Reference::Column(ColumnSource::Row2(idx))) => builder.emit(Op::LoadColumn2(idx)),
            Some(Reference::Outer(name)) => builder.emit(Op::LoadOuterColumn(name)),
            Some(Reference::Text) => {
                let Expression::Identifier(id) = expr else {
                    return Err(CompileError::InvalidExpression(expr.to_string()));
                };
                builder.emit(Op::LoadConst(Value::text(id.value.as_str())));
            }
            Some(Reference::Missing(name)) => return Err(CompileError::ColumnNotFound(name)),
            None => return Err(CompileError::InvalidExpression(expr.to_string())),
        }
        Ok(())
    }

    /// Compile an infix expression
    fn compile_infix(
        &self,
        infix: &InfixExpression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        match infix.op_type {
            // Short-circuit AND
            InfixOperator::And => {
                // Compile left side
                self.compile_expr(&infix.left, builder)?;

                // Emit AND with placeholder jump target
                let and_pos = builder.position();
                builder.emit(Op::And(0)); // Placeholder

                // Compile right side
                self.compile_expr(&infix.right, builder)?;

                // Emit finalize
                builder.emit(Op::AndFinalize);

                // Patch jump to skip right side if left is false
                let end_pos = builder.position();
                builder.patch_jump(and_pos as usize, end_pos);
            }

            // Short-circuit OR
            InfixOperator::Or => {
                // Compile left side
                self.compile_expr(&infix.left, builder)?;

                // Emit OR with placeholder jump target
                let or_pos = builder.position();
                builder.emit(Op::Or(0)); // Placeholder

                // Compile right side
                self.compile_expr(&infix.right, builder)?;

                // Emit finalize
                builder.emit(Op::OrFinalize);

                // Patch jump to skip right side if left is true
                let end_pos = builder.position();
                builder.patch_jump(or_pos as usize, end_pos);
            }

            // Comparison operators
            InfixOperator::Equal => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Eq);
            }

            InfixOperator::NotEqual => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Ne);
            }

            InfixOperator::LessThan => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Lt);
            }

            InfixOperator::LessEqual => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Le);
            }

            InfixOperator::GreaterThan => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Gt);
            }

            InfixOperator::GreaterEqual => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Ge);
            }

            // Arithmetic operators
            InfixOperator::Add => {
                if !self.compile_interval_shift(infix, false, builder)? {
                    self.compile_expr(&infix.left, builder)?;
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::Add);
                }
            }

            InfixOperator::Subtract => {
                if !self.compile_interval_shift(infix, true, builder)? {
                    self.compile_expr(&infix.left, builder)?;
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::Sub);
                }
            }

            InfixOperator::Multiply => {
                if let Some(operand) = Self::interval_product(infix)? {
                    self.compile_interval(&operand, builder)?;
                } else {
                    self.compile_expr(&infix.left, builder)?;
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::Mul);
                }
            }

            InfixOperator::Divide | InfixOperator::Modulo => {
                if Self::interval_operand(&infix.left)?.is_some()
                    || Self::interval_operand(&infix.right)?.is_some()
                {
                    return Err(CompileError::TypeError(format!(
                        "an INTERVAL takes no {}",
                        infix.operator
                    )));
                }
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(if matches!(infix.op_type, InfixOperator::Divide) {
                    Op::Div
                } else {
                    Op::Mod
                });
            }

            // String concatenation - optimize chained || into single ConcatN
            InfixOperator::Concat => {
                // Flatten chained concatenations: a || b || c -> ConcatN(3)
                let mut operands = Vec::new();
                Self::flatten_concat_chain_infix(infix, &mut operands);

                if operands.len() > 2 && operands.len() <= 255 {
                    // Compile all operands in order
                    for operand in &operands {
                        self.compile_expr(operand, builder)?;
                    }
                    builder.emit(Op::ConcatN(operands.len() as u8));
                } else {
                    // Fallback to binary concat
                    self.compile_expr(&infix.left, builder)?;
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::Concat);
                }
            }

            // Bitwise operators
            InfixOperator::BitwiseAnd => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::BitAnd);
            }

            InfixOperator::BitwiseOr => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::BitOr);
            }

            InfixOperator::BitwiseXor => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::BitXor);
            }

            InfixOperator::LeftShift => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Shl);
            }

            InfixOperator::RightShift => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Shr);
            }

            // XOR
            InfixOperator::Xor => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Xor);
            }

            // IS / IS NOT
            InfixOperator::Is => {
                self.compile_expr(&infix.left, builder)?;
                // Check if right side is NULL, TRUE, or FALSE
                match &*infix.right {
                    Expression::NullLiteral(_) => {
                        builder.emit(Op::IsNull);
                    }
                    Expression::BooleanLiteral(lit) if lit.value => {
                        builder.emit(Op::IsTrue);
                    }
                    Expression::BooleanLiteral(lit) if !lit.value => {
                        builder.emit(Op::IsFalse);
                    }
                    Expression::Identifier(id) if id.value_lower == "true" => {
                        builder.emit(Op::IsTrue);
                    }
                    Expression::Identifier(id) if id.value_lower == "false" => {
                        builder.emit(Op::IsFalse);
                    }
                    _ => {
                        self.compile_expr(&infix.right, builder)?;
                        builder.emit(Op::IsNotDistinctFrom);
                    }
                }
            }

            InfixOperator::IsNot => {
                self.compile_expr(&infix.left, builder)?;
                match &*infix.right {
                    Expression::NullLiteral(_) => {
                        builder.emit(Op::IsNotNull);
                    }
                    Expression::BooleanLiteral(lit) if lit.value => {
                        builder.emit(Op::IsNotTrue);
                    }
                    Expression::BooleanLiteral(lit) if !lit.value => {
                        builder.emit(Op::IsNotFalse);
                    }
                    Expression::Identifier(id) if id.value_lower == "true" => {
                        builder.emit(Op::IsNotTrue);
                    }
                    Expression::Identifier(id) if id.value_lower == "false" => {
                        builder.emit(Op::IsNotFalse);
                    }
                    _ => {
                        self.compile_expr(&infix.right, builder)?;
                        builder.emit(Op::IsDistinctFrom);
                    }
                }
            }

            InfixOperator::IsDistinctFrom => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::IsDistinctFrom);
            }

            InfixOperator::IsNotDistinctFrom => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::IsNotDistinctFrom);
            }

            // Pattern matching via infix
            InfixOperator::Like => {
                self.compile_expr(&infix.left, builder)?;
                let pattern_str = Self::extract_pattern_string(&infix.right);
                if let Some(s) = pattern_str {
                    builder.emit(Op::Like(Arc::new(Pattern::like(&s, false))));
                } else {
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::LikeDynamic(false));
                }
            }

            InfixOperator::ILike => {
                self.compile_expr(&infix.left, builder)?;
                let pattern_str = Self::extract_pattern_string(&infix.right);
                if let Some(s) = pattern_str {
                    builder.emit(Op::Like(Arc::new(Pattern::like(&s, true))));
                } else {
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::LikeDynamic(true));
                }
            }

            InfixOperator::NotLike => {
                self.compile_expr(&infix.left, builder)?;
                let pattern_str = Self::extract_pattern_string(&infix.right);
                if let Some(s) = pattern_str {
                    builder.emit(Op::Like(Arc::new(Pattern::like(&s, false))));
                    builder.emit(Op::Not);
                } else {
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::LikeDynamic(false));
                    builder.emit(Op::Not);
                }
            }

            InfixOperator::NotILike => {
                self.compile_expr(&infix.left, builder)?;
                let pattern_str = Self::extract_pattern_string(&infix.right);
                if let Some(s) = pattern_str {
                    builder.emit(Op::Like(Arc::new(Pattern::like(&s, true))));
                    builder.emit(Op::Not);
                } else {
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::LikeDynamic(true));
                    builder.emit(Op::Not);
                }
            }

            InfixOperator::Glob | InfixOperator::NotGlob => {
                self.compile_expr(&infix.left, builder)?;
                let pattern_str = Self::extract_pattern_string(&infix.right);
                if let Some(s) = pattern_str {
                    builder.emit(Op::Glob(Arc::new(Pattern::glob(&s))));
                } else {
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::GlobDynamic);
                }
                if matches!(infix.op_type, InfixOperator::NotGlob) {
                    builder.emit(Op::Not);
                }
            }

            InfixOperator::Regexp | InfixOperator::NotRegexp => {
                self.compile_expr(&infix.left, builder)?;
                let pattern_str = Self::extract_pattern_string(&infix.right);
                if let Some(s) = pattern_str {
                    let regex = regex::Regex::new(&s).map_err(|e| {
                        CompileError::InvalidExpression(format!("Invalid regex: {}", e))
                    })?;
                    builder.emit(Op::Regexp(Arc::new(regex)));
                } else {
                    self.compile_expr(&infix.right, builder)?;
                    builder.emit(Op::RegexpDynamic);
                }
                if matches!(infix.op_type, InfixOperator::NotRegexp) {
                    builder.emit(Op::Not);
                }
            }

            // JSON operators
            InfixOperator::JsonAccess => {
                // json -> key (returns JSON)
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::JsonAccess);
            }

            InfixOperator::JsonAccessText => {
                // json ->> key (returns TEXT)
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::JsonAccessText);
            }

            InfixOperator::Index => {
                // Array/JSON index access - treat as JsonAccess
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::JsonAccess);
            }

            // Vector distance operator (<=>)
            InfixOperator::VectorDistance => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::VectorDistanceL2);
            }

            // Other/unknown operators
            InfixOperator::Other => {
                return Err(CompileError::UnsupportedExpression(format!(
                    "Unknown infix operator: {}",
                    infix.operator
                )));
            }
        }

        Ok(())
    }

    /// Compile a prefix expression
    fn compile_prefix(
        &self,
        prefix: &PrefixExpression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        if let Some(operand) = Self::interval_sign(prefix)? {
            return self.compile_interval(&operand, builder);
        }
        self.compile_expr(&prefix.right, builder)?;

        match prefix.operator.to_uppercase().as_str() {
            "NOT" => builder.emit(Op::Not),
            "-" => builder.emit(Op::Neg),
            "+" => {} // Unary plus is a no-op
            "~" => builder.emit(Op::BitNot),
            _ => {
                return Err(CompileError::InvalidExpression(format!(
                    "Unknown prefix operator: {}",
                    prefix.operator
                )));
            }
        }

        Ok(())
    }

    /// `expr` as an interval operand; None when it holds no INTERVAL literal
    /// at its top, Err when it multiplies two of them
    fn interval_operand(expr: &Expression) -> Result<Option<IntervalOperand<'_>>, CompileError> {
        match expr {
            Expression::IntervalLiteral(literal) => Ok(Some(IntervalOperand {
                literal,
                negated: false,
                by: SmallVec::new(),
            })),
            Expression::Prefix(prefix) => Self::interval_sign(prefix),
            Expression::Infix(infix) if matches!(infix.op_type, InfixOperator::Multiply) => {
                Self::interval_product(infix)
            }
            _ => Ok(None),
        }
    }

    fn interval_sign(
        prefix: &PrefixExpression,
    ) -> Result<Option<IntervalOperand<'_>>, CompileError> {
        let negate = match prefix.op_type {
            PrefixOperator::Negate => true,
            PrefixOperator::Plus => false,
            _ => return Ok(None),
        };
        Ok(Self::interval_operand(&prefix.right)?.map(|mut operand| {
            operand.negated ^= negate;
            operand
        }))
    }

    fn interval_product(
        infix: &InfixExpression,
    ) -> Result<Option<IntervalOperand<'_>>, CompileError> {
        match (
            Self::interval_operand(&infix.left)?,
            Self::interval_operand(&infix.right)?,
        ) {
            (Some(_), Some(_)) => Err(CompileError::TypeError(
                "an INTERVAL cannot be multiplied by an INTERVAL".to_string(),
            )),
            (Some(mut operand), None) => {
                operand.by.push(&infix.right);
                Ok(Some(operand))
            }
            (None, Some(mut operand)) => {
                operand.by.push(&infix.left);
                Ok(Some(operand))
            }
            (None, None) => Ok(None),
        }
    }

    /// Emits an interval operand's value: the literal's text, or the
    /// product of its multipliers scaled into interval text
    fn compile_interval(
        &self,
        operand: &IntervalOperand<'_>,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        if !operand.by.is_empty() {
            let step = operand.step()?;
            let count = self.compile_multipliers(operand, builder)?;
            builder.emit(Op::ScaleInterval(step, count));
            return Ok(());
        }
        let text = if operand.negated {
            let step = operand.step()?;
            format!("{} {}", step.quantity, step.unit.name())
        } else {
            format!("{} {}", operand.literal.quantity, operand.literal.unit)
        };
        builder.emit(Op::LoadConst(Value::Text(SmartString::from_string(text))));
        Ok(())
    }

    /// Emits each multiplier and returns their count; the interval op
    /// checks and multiplies them, so a non-number is never a NULL
    fn compile_multipliers(
        &self,
        operand: &IntervalOperand<'_>,
        builder: &mut ProgramBuilder,
    ) -> Result<u8, CompileError> {
        let count = u8::try_from(operand.by.len()).map_err(|_| {
            CompileError::UnsupportedExpression(format!(
                "{} multipliers of one INTERVAL",
                operand.by.len()
            ))
        })?;
        for multiplier in &operand.by {
            self.compile_expr(multiplier, builder)?;
        }
        Ok(count)
    }

    /// Compiles `infix`, a `+` or, with `subtract`, a `-`, when one side is
    /// an interval operand; false when neither is
    fn compile_interval_shift(
        &self,
        infix: &InfixExpression,
        subtract: bool,
        builder: &mut ProgramBuilder,
    ) -> Result<bool, CompileError> {
        let (base, operand) = match (
            Self::interval_operand(&infix.left)?,
            Self::interval_operand(&infix.right)?,
        ) {
            (None, None) => return Ok(false),
            (Some(_), Some(_)) => {
                return Err(CompileError::TypeError(format!(
                    "an INTERVAL takes no {} with another INTERVAL",
                    infix.operator
                )))
            }
            (None, Some(operand)) => (&infix.left, operand),
            (Some(operand), None) if !subtract => (&infix.right, operand),
            (Some(_), None) => {
                return Err(CompileError::TypeError(
                    "nothing can be subtracted from an INTERVAL".to_string(),
                ))
            }
        };
        self.compile_expr(base, builder)?;
        if operand.by.is_empty() {
            self.compile_interval(&operand, builder)?;
            builder.emit(if subtract { Op::Sub } else { Op::Add });
        } else {
            let step = operand.step()?;
            let count = self.compile_multipliers(&operand, builder)?;
            builder.emit(Op::TimestampAddScaledInterval(step, subtract, count));
        }
        Ok(true)
    }

    /// Compile an IN expression
    fn compile_in(
        &self,
        in_expr: &InExpression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        // Check if this is a multi-column IN: (a, b) IN ((1, 2), (3, 4))
        let left_columns: Vec<&Expression> = match &*in_expr.left {
            Expression::List(list) if list.elements.len() > 1 => list.elements.iter().collect(),
            Expression::ExpressionList(list) if list.expressions.len() > 1 => {
                list.expressions.iter().collect()
            }
            _ => Vec::new(),
        };

        if !left_columns.is_empty() {
            // Multi-column IN expression
            return self.compile_multi_column_in(in_expr, &left_columns, builder);
        }

        // The block's members answer, NOT IN keeping UNKNOWN
        if let Expression::ScalarSubquery(sq) = &*in_expr.right {
            let id = self.subquery_id(&sq.subquery, "IN (subquery)")?;
            self.compile_expr(&in_expr.left, builder)?;
            builder.emit(Op::ExecInSubquery(id));
            if in_expr.not {
                builder.emit(Op::Not);
            }
            return Ok(());
        }

        // Single-value IN expression
        // Build the set of values at compile time if possible
        let mut values = ValueSet::default();
        let mut has_null = false;
        let mut all_constant = true;

        // Check if right side is a list of constants
        let constant = |item: &Expression| {
            let value = try_eval_constant(item)?;
            Some(
                match self
                    .ctx
                    .read_as
                    .and_then(|r| r.get(&(item as *const Expression as usize)))
                {
                    Some(&column_type) => {
                        crate::executor::pushdown::read_as_column(column_type, value)
                    }
                    None => value,
                },
            )
        };
        match &*in_expr.right {
            Expression::List(list) => {
                for item in &list.elements {
                    if let Some(value) = constant(item) {
                        if value.is_null() {
                            has_null = true;
                        } else {
                            values.insert(value);
                        }
                    } else {
                        all_constant = false;
                        break;
                    }
                }
            }
            Expression::ExpressionList(list) => {
                for item in &list.expressions {
                    if let Some(value) = constant(item) {
                        if value.is_null() {
                            has_null = true;
                        } else {
                            values.insert(value);
                        }
                    } else {
                        all_constant = false;
                        break;
                    }
                }
            }
            _ => {
                all_constant = false;
            }
        }

        if all_constant {
            if values.is_empty() && !has_null {
                // Empty IN list with no NULLs:
                // x IN () -> FALSE (nothing matches)
                // x NOT IN () -> TRUE (x is not in empty set)
                if in_expr.not {
                    builder.emit(Op::LoadConst(Value::Boolean(true)));
                } else {
                    builder.emit(Op::LoadConst(Value::Boolean(false)));
                }
            } else {
                // Optimized: use pre-built HashSet
                self.compile_expr(&in_expr.left, builder)?;
                let set = CompactArc::new(PredicateSet::new(CompactArc::new(values), has_null));
                if in_expr.not {
                    builder.emit(Op::NotInSet(set));
                } else {
                    builder.emit(Op::InSet(set));
                }
            }
        } else {
            // An item that is not a constant is compared in turn:
            // left = item OR left = item ..., with IN's NULL rule falling
            // out of OR's
            let items: Vec<&Expression> = match &*in_expr.right {
                Expression::List(list) => list.elements.iter().collect(),
                Expression::ExpressionList(list) => list.expressions.iter().collect(),
                _ => {
                    return Err(CompileError::UnsupportedExpression(
                        "IN over something that is not a list".to_string(),
                    ))
                }
            };
            // The value is evaluated once and judged against every item,
            // so a volatile left side rolls a single time
            if items.is_empty() {
                builder.emit(Op::LoadConst(Value::Boolean(in_expr.not)));
            } else {
                self.compile_expr(&in_expr.left, builder)?;
                for item in &items {
                    self.compile_expr(item, builder)?;
                }
                let count = u32::try_from(items.len()).map_err(|_| {
                    CompileError::UnsupportedExpression(format!(
                        "IN list holds {} items, more than an expression can hold",
                        items.len()
                    ))
                })?;
                builder.emit(Op::InList(count));
                if in_expr.not {
                    builder.emit(Op::Not);
                }
            }
        }

        Ok(())
    }

    /// Compile multi-column IN expression: (a, b) IN ((1, 2), (3, 4))
    fn compile_multi_column_in(
        &self,
        in_expr: &InExpression,
        left_columns: &[&Expression],
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        let tuple_size = left_columns.len();

        // Extract tuples from right side
        let mut tuple_values: Vec<Vec<Value>> = Vec::new();
        let mut all_constant = true;

        match &*in_expr.right {
            Expression::List(list) => {
                for item in &list.elements {
                    if let Some(tuple) = self.extract_tuple_values(item, tuple_size) {
                        tuple_values.push(tuple);
                    } else {
                        all_constant = false;
                        break;
                    }
                }
            }
            Expression::ExpressionList(list) => {
                for item in &list.expressions {
                    if let Some(tuple) = self.extract_tuple_values(item, tuple_size) {
                        tuple_values.push(tuple);
                    } else {
                        all_constant = false;
                        break;
                    }
                }
            }
            _ => {
                all_constant = false;
            }
        }

        if !all_constant || tuple_values.is_empty() {
            return Err(CompileError::UnsupportedExpression(
                "Dynamic multi-column IN not yet supported in VM".to_string(),
            ));
        }

        // Compile each column expression to push onto stack
        for col in left_columns {
            self.compile_expr(col, builder)?;
        }

        // Emit InTupleSet operation
        builder.emit(Op::InTupleSet {
            tuple_size: tuple_size as u8,
            values: Arc::new(tuple_values),
            negated: in_expr.not,
        });

        Ok(())
    }

    /// Extract tuple values from an expression (e.g., (1, 2) -> [1, 2])
    fn extract_tuple_values(&self, expr: &Expression, expected_size: usize) -> Option<Vec<Value>> {
        let elements: Vec<&Expression> = match expr {
            Expression::List(list) => list.elements.iter().collect(),
            Expression::ExpressionList(list) => list.expressions.iter().collect(),
            _ => return None,
        };

        if elements.len() != expected_size {
            return None;
        }

        let mut values = Vec::with_capacity(expected_size);
        for element in elements {
            values.push(try_eval_constant(element)?);
        }

        Some(values)
    }

    /// Compile a LIKE expression
    fn compile_like(
        &self,
        like: &LikeExpression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        self.compile_expr(&like.left, builder)?;

        // Determine case sensitivity and negation from operator
        let op_upper = like.operator.to_uppercase();
        let case_insensitive = op_upper.contains("ILIKE");
        let negated = op_upper.contains("NOT");
        let is_glob = op_upper.contains("GLOB");
        let is_regexp = op_upper.contains("REGEXP") || op_upper.contains("RLIKE");

        // ESCAPE is baked in when it is a literal or folds to one. Anything else
        // is evaluated per row, never dropped.
        let mut runtime_escape: Option<&Expression> = None;
        let escape_char: Option<char> = match like.escape.as_deref() {
            None => None,
            Some(Expression::StringLiteral(lit)) => lit.value.chars().next(),
            Some(escape_expr) => {
                match may_fold(escape_expr, self.ctx)
                    .then(|| self.try_fold_constant(escape_expr))
                    .flatten()
                {
                    Some(Value::Text(text)) => text.chars().next(),
                    _ => {
                        runtime_escape = Some(escape_expr);
                        None
                    }
                }
            }
        };

        if let Some(escape_expr) = runtime_escape.filter(|_| !is_glob && !is_regexp) {
            self.compile_expr(&like.pattern, builder)?;
            self.compile_expr(escape_expr, builder)?;
            builder.emit(Op::LikeDynamicEscapeExpr(case_insensitive));
            if negated {
                builder.emit(Op::Not);
            }
            return Ok(());
        }

        // Try to compile pattern at compile time
        let pattern_str = Self::extract_pattern_string(&like.pattern);
        if let Some(s) = pattern_str {
            if is_regexp {
                let regex = regex::Regex::new(&s).map_err(|e| {
                    CompileError::InvalidExpression(format!("Invalid regex: {}", e))
                })?;
                builder.emit(Op::Regexp(Arc::new(regex)));
            } else if is_glob {
                builder.emit(Op::Glob(Arc::new(Pattern::glob(&s))));
            } else if let Some(esc) = escape_char {
                // LIKE with ESCAPE - pre-process pattern to handle escape character
                let processed_pattern = self.process_like_escape(&s, esc);
                let pattern = Pattern::like(&processed_pattern, case_insensitive);
                builder.emit(Op::LikeEscape(Arc::new(pattern), esc));
            } else {
                builder.emit(Op::Like(Arc::new(Pattern::like(&s, case_insensitive))));
            }

            if negated {
                builder.emit(Op::Not);
            }
        } else {
            // Dynamic pattern (e.g. parameter $1) — compile the pattern expression
            // onto the stack and use the dynamic op
            self.compile_expr(&like.pattern, builder)?;
            if is_regexp {
                builder.emit(Op::RegexpDynamic);
            } else if is_glob {
                builder.emit(Op::GlobDynamic);
            } else if let Some(esc) = escape_char {
                builder.emit(Op::LikeDynamicEscape(case_insensitive, esc));
            } else {
                builder.emit(Op::LikeDynamic(case_insensitive));
            }
            if negated {
                builder.emit(Op::Not);
            }
        }

        Ok(())
    }

    /// A pattern known when compiled: a literal read as its text form, a
    /// typed literal by its value, or a double-quoted identifier's text. None
    /// for a NULL and for an expression read per row
    fn extract_pattern_string(expr: &Expression) -> Option<SmartString> {
        match expr {
            Expression::Identifier(id) if id.quoted => Some(id.value.clone()),
            expr => extract_literal_value(expr)?
                .as_string()
                .map(SmartString::from_string),
        }
    }

    /// Process LIKE pattern with escape character
    /// Converts escaped wildcards to special markers and then to literal characters
    fn process_like_escape(&self, pattern: &str, escape: char) -> String {
        let mut result = String::with_capacity(pattern.len());
        let mut chars = pattern.chars().peekable();

        while let Some(c) = chars.next() {
            if c == escape {
                // Whatever follows stands for itself. A wildcard and the
                // backslash the pattern reads as its own marker keep that
                // marker in front of them; anything else is already itself
                match chars.next() {
                    Some(next) => {
                        if next == '%' || next == '_' || next == '\\' {
                            result.push('\\');
                        }
                        result.push(next);
                    }
                    // Nothing follows, so the escape stands for itself
                    None => result.push(c),
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Compile a CASE expression
    fn compile_case(
        &self,
        case: &CaseExpression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        builder.emit(Op::CaseStart);

        let is_simple = case.value.is_some();
        let mut end_jumps = Vec::new();

        // For simple CASE, compile the operand once
        if let Some(ref operand) = case.value {
            self.compile_expr(operand, builder)?;
        }

        for when_clause in &case.when_clauses {
            if is_simple {
                // Simple CASE: compare the operand, which CaseCompare reads
                // without taking, so it is there for the branch after this
                self.compile_expr(&when_clause.condition, builder)?;
                builder.emit(Op::CaseCompare);
            } else {
                // Searched CASE: evaluate condition
                self.compile_expr(&when_clause.condition, builder)?;
            }

            // Jump to next branch if condition is false
            let when_pos = builder.position();
            builder.emit(Op::CaseWhen(0)); // Placeholder

            // Compile THEN result
            if is_simple {
                builder.emit(Op::Pop); // The operand has served its purpose
            }
            self.compile_expr(&when_clause.then_result, builder)?;

            // Jump to end after THEN
            let then_pos = builder.position();
            builder.emit(Op::CaseThen(0)); // Placeholder
            end_jumps.push(then_pos);

            // Patch WHEN jump to here
            let next_pos = builder.position();
            builder.patch_jump(when_pos as usize, next_pos);
        }

        // Compile ELSE
        if is_simple {
            builder.emit(Op::Pop); // Remove operand
        }
        if let Some(ref else_value) = case.else_value {
            builder.emit(Op::CaseElse);
            self.compile_expr(else_value, builder)?;
        } else {
            builder.emit(Op::LoadNull(DataType::Null));
        }

        // Patch all THEN jumps to end
        let end_pos = builder.position();
        builder.emit(Op::CaseEnd);

        for pos in end_jumps {
            builder.patch_jump(pos as usize, end_pos);
        }

        Ok(())
    }

    /// Compile a function call
    fn compile_function(
        &self,
        func: &FunctionCall,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        let func_name = func.function.to_uppercase();

        // Special handling for certain functions
        match func_name.as_str() {
            "CURRENT_TRANSACTION_ID" => {
                // Context-dependent function - loads from ExecuteContext
                builder.emit(Op::LoadTransactionId);
                return Ok(());
            }

            "COALESCE" => {
                // Short-circuit COALESCE: stop evaluation as soon as we find non-null
                // Bytecode pattern:
                //   Eval(Arg1)
                //   JumpIfNotNull(End)  // If not null, jump to end (keep value)
                //   Pop                  // Pop the null value
                //   Eval(Arg2)
                //   JumpIfNotNull(End)
                //   Pop
                //   ...
                //   Eval(ArgN)          // Last arg: keep on stack (null or not)
                //   Label(End)
                if func.arguments.is_empty() {
                    builder.emit(Op::LoadNull(DataType::Null));
                    return Ok(());
                }

                let mut jump_positions = Vec::new();
                let last_idx = func.arguments.len() - 1;

                for (i, arg) in func.arguments.iter().enumerate() {
                    self.compile_expr(arg, builder)?;

                    if i < last_idx {
                        // For all but last: jump to end if not null, else pop and continue
                        let jump_pos = builder.position();
                        builder.emit(Op::JumpIfNotNull(0)); // Placeholder, will patch
                        jump_positions.push(jump_pos);
                        builder.emit(Op::Pop); // Pop the null value
                    }
                    // Last argument: just leave on stack
                }

                // Patch all jumps to point to end
                let end_pos = builder.position();
                for pos in jump_positions {
                    builder.patch_jump(pos as usize, end_pos);
                }

                return Ok(());
            }

            "NULLIF" if func.arguments.len() == 2 => {
                self.compile_expr(&func.arguments[0], builder)?;
                self.compile_expr(&func.arguments[1], builder)?;
                builder.emit(Op::NullIf);
                return Ok(());
            }

            "GREATEST" => {
                for arg in &func.arguments {
                    self.compile_expr(arg, builder)?;
                }
                builder.emit(Op::Greatest(func.arguments.len() as u8));
                return Ok(());
            }

            "LEAST" => {
                for arg in &func.arguments {
                    self.compile_expr(arg, builder)?;
                }
                builder.emit(Op::Least(func.arguments.len() as u8));
                return Ok(());
            }

            _ => {}
        }

        // Get function from registry
        if let Some(scalar_func) = self.ctx.functions.get_scalar(&func_name) {
            // Compile arguments
            for arg in &func.arguments {
                self.compile_expr(arg, builder)?;
            }

            // Try native function pointer for single-arg functions (no dynamic dispatch)
            if func.arguments.len() == 1 {
                if let Some(native_fn) = scalar_func.native_fn1() {
                    builder.emit(Op::NativeFn1(native_fn));
                    return Ok(());
                }
            }

            // Fallback to dynamic dispatch
            builder.emit(Op::CallScalar {
                func: scalar_func.into(),
                arg_count: func.arguments.len() as u8,
            });
            Ok(())
        } else {
            // Check if it's an aggregate being referenced post-aggregation
            // This would be handled via LoadAggregateResult in a real implementation
            Err(CompileError::FunctionNotFound(func_name.to_string()))
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Whether `expr` is worth folding into a constant and may be: an operator,
/// call or cast (a literal is a constant already)
fn is_foldable_expr(expr: &Expression, ctx: &CompileContext) -> bool {
    matches!(
        expr,
        Expression::Infix(_)
            | Expression::Prefix(_)
            | Expression::FunctionCall(_)
            | Expression::Cast(_)
    ) && may_fold(expr, ctx)
}

/// Whether `expr` may be folded into a constant under `ctx`: it reads no
/// column, no slot and no parameter `ctx` does not hold, and answers the
/// same every time
fn may_fold(expr: &Expression, ctx: &CompileContext) -> bool {
    is_column_free(expr, ctx) && expression_volatility(expr, ctx.functions) == Volatility::Fixed
}

/// Check if an expression references no columns (is entirely self-contained);
/// a part read from an aggregated row's slot counts as one, and so does a
/// parameter unless `ctx` holds the values
fn is_column_free(expr: &Expression, ctx: &CompileContext) -> bool {
    if ctx.check_expression_alias(expr).is_some() {
        return false;
    }
    let params = ctx.params.is_some();
    let is_column_free = |expr: &Expression| is_column_free(expr, ctx);
    match expr {
        // Literals are always column-free
        Expression::IntegerLiteral(_)
        | Expression::FloatLiteral(_)
        | Expression::StringLiteral(_)
        | Expression::BooleanLiteral(_)
        | Expression::NullLiteral(_)
        | Expression::IntervalLiteral(_) => true,

        // Identifiers reference columns (not column-free)
        Expression::Identifier(_) | Expression::QualifiedIdentifier { .. } => false,

        // Parameters: known only for a program that serves one execution
        Expression::Parameter(_) => params,

        // Binary operations
        Expression::Infix(infix) => is_column_free(&infix.left) && is_column_free(&infix.right),

        // Unary operations
        Expression::Prefix(prefix) => is_column_free(&prefix.right),

        // Function calls (NOW(), UPPER('text'), etc.)
        Expression::FunctionCall(func) => func.arguments.iter().all(is_column_free),

        // CAST
        Expression::Cast(cast) => is_column_free(&cast.expr),

        // CASE WHEN
        Expression::Case(case) => {
            case.value.as_ref().is_none_or(|e| is_column_free(e))
                && case
                    .when_clauses
                    .iter()
                    .all(|wc| is_column_free(&wc.condition) && is_column_free(&wc.then_result))
                && case.else_value.as_ref().is_none_or(|e| is_column_free(e))
        }

        // Subqueries, EXISTS — not column-free
        Expression::ScalarSubquery(_) | Expression::Exists(_) => false,

        // Between
        Expression::Between(between) => {
            is_column_free(&between.expr)
                && is_column_free(&between.lower)
                && is_column_free(&between.upper)
        }

        // Anything else: conservatively assume it references columns
        _ => false,
    }
}

/// The zero-based slot `param` reads; the parser numbers `$n`, `?` and
/// `:name` alike from one. A slot a program cannot address is an error,
/// never another slot
pub(crate) fn parameter_slot(param: &Parameter) -> Result<u32, CompileError> {
    param
        .index
        .checked_sub(1)
        .and_then(|slot| u32::try_from(slot).ok())
        .ok_or_else(|| CompileError::InvalidExpression(format!("Invalid parameter {}", param.name)))
}

/// The value a program loads for a literal; None for any other expression
pub(crate) fn literal_value(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
        Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
        Expression::StringLiteral(lit) => Some(string_literal_value(lit)),
        Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
        Expression::NullLiteral(_) => Some(Value::Null(DataType::Null)),
        _ => None,
    }
}

/// Try to evaluate a constant expression at compile time
fn try_eval_constant(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
        Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
        Expression::StringLiteral(lit) => Some(string_literal_value(lit)),
        Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
        Expression::NullLiteral(_) => Some(Value::null_unknown()),
        Expression::Identifier(id) if id.quoted => Some(Value::Text(id.value.clone())),
        _ => None,
    }
}

// Note: string_to_datatype and expression_to_string are now imported from utils

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::IntegerLiteral;
    use crate::parser::token::{Position, Token, TokenType};

    fn make_token() -> Token {
        Token {
            token_type: TokenType::Integer,
            literal: "1".into(),
            position: Position {
                offset: 0,
                line: 1,
                column: 1,
            },
            quoted: false,
            keyword: None,
        }
    }

    #[test]
    fn test_compile_simple_comparison() {
        let columns = vec!["a".to_string(), "b".to_string()];
        let ctx = CompileContext::with_global_registry(&columns);
        let compiler = ExprCompiler::new(&ctx);

        // a > 5
        let expr = Expression::Infix(InfixExpression {
            token: make_token(),
            left: Box::new(Expression::Identifier(Identifier::new(
                make_token(),
                "a".to_string(),
            ))),
            operator: ">".into(),
            op_type: InfixOperator::GreaterThan,
            right: Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: make_token(),
                value: 5,
            })),
        });

        let program = compiler.compile(&expr).unwrap();
        assert!(!program.is_empty());
        println!("{}", program.disassemble());
    }

    #[test]
    fn test_compile_and_expression() {
        let columns = vec!["a".to_string(), "b".to_string()];
        let ctx = CompileContext::with_global_registry(&columns);
        let compiler = ExprCompiler::new(&ctx);

        // a > 5 AND b < 10
        let expr = Expression::Infix(InfixExpression {
            token: make_token(),
            left: Box::new(Expression::Infix(InfixExpression {
                token: make_token(),
                left: Box::new(Expression::Identifier(Identifier::new(
                    make_token(),
                    "a".to_string(),
                ))),
                operator: ">".into(),
                op_type: InfixOperator::GreaterThan,
                right: Box::new(Expression::IntegerLiteral(IntegerLiteral {
                    token: make_token(),
                    value: 5,
                })),
            })),
            operator: "AND".into(),
            op_type: InfixOperator::And,
            right: Box::new(Expression::Infix(InfixExpression {
                token: make_token(),
                left: Box::new(Expression::Identifier(Identifier::new(
                    make_token(),
                    "b".to_string(),
                ))),
                operator: "<".into(),
                op_type: InfixOperator::LessThan,
                right: Box::new(Expression::IntegerLiteral(IntegerLiteral {
                    token: make_token(),
                    value: 10,
                })),
            })),
        });

        let program = compiler.compile(&expr).unwrap();
        assert!(!program.is_empty());
        println!("{}", program.disassemble());
    }

    #[test]
    fn a_timestamp_moved_by_an_interval_product_compiles_to_one_op() {
        let columns = vec!["ts".to_string(), "n".to_string()];
        let ctx = CompileContext::with_global_registry(&columns);
        let compiler = ExprCompiler::new(&ctx);
        for (sql, subtract) in [
            ("ts + n * INTERVAL '1 hour'", false),
            ("n * INTERVAL '1 hour' + ts", false),
            ("ts - INTERVAL '1 hour' * n", true),
        ] {
            let expr = crate::parser::Parser::new(sql)
                .parse_expression(crate::parser::precedence::Precedence::Lowest)
                .unwrap();
            let program = compiler.compile(&expr).unwrap();
            let shifts: Vec<_> = program
                .ops()
                .iter()
                .filter_map(|op| match op {
                    Op::TimestampAddScaledInterval(step, s, 1) => Some((*step, *s)),
                    _ => None,
                })
                .collect();
            let step = IntervalStep {
                quantity: 1,
                unit: IntervalUnit::Hour,
            };
            assert_eq!(
                shifts,
                [(step, subtract)],
                "{sql}: {}",
                program.disassemble()
            );
            assert!(
                !program
                    .ops()
                    .iter()
                    .any(|op| matches!(op, Op::ScaleInterval(_, _) | Op::Mul)),
                "{sql}: no interval text per row"
            );
        }
    }
}
