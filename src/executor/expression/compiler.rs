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

use std::collections::HashSet;
use std::sync::Arc;

use ahash::AHashSet;
use rustc_hash::FxHashMap;

use super::ops::{CompareOp, CompiledPattern, Op};
use super::program::{Program, ProgramBuilder};
use crate::core::{DataType, Value};
use crate::executor::utils::{expression_to_string, string_to_datatype};
use crate::functions::{global_registry, FunctionRegistry};
use crate::parser::ast::*;

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
            CompileError::ColumnNotFound(name) => write!(f, "Column not found: {}", name),
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

/// Result of column resolution - indicates which row the column is from
#[derive(Debug, Clone, Copy)]
pub enum ColumnSource {
    /// Column from first row with given index
    Row1(u16),
    /// Column from second row (for joins) with given index
    Row2(u16),
}

/// Compilation context
///
/// Contains all the information needed to compile expressions:
/// - Column name to index mapping
/// - Function registry
/// - Outer query columns (for correlated subqueries)
pub struct CompileContext<'a> {
    /// Column name -> index (case-insensitive)
    columns: FxHashMap<String, u16>,

    /// Qualified column name -> (index, is_row2)
    /// For row1 columns: index is the direct row1 index
    /// For row2 columns: index is the row2 index (not offset)
    qualified_columns: FxHashMap<String, FxHashMap<String, ColumnSource>>,

    /// Second row columns (for joins)
    columns2: Option<FxHashMap<String, u16>>,

    /// Tables that belong to row2 (for tracking which tables are from second row)
    row2_tables: HashSet<String>,

    /// Column offset for second row in combined index space (for fallback)
    column2_offset: u16,

    /// Outer query columns (for correlated subqueries)
    outer_columns: Option<FxHashMap<Arc<str>, u16>>,

    /// Function registry
    functions: &'a FunctionRegistry,

    /// Expression alias mapping (for HAVING with GROUP BY expressions)
    expression_aliases: FxHashMap<String, u16>,

    /// Column aliases
    column_aliases: FxHashMap<String, String>,
}

impl<'a> CompileContext<'a> {
    /// Create a new compilation context
    pub fn new(columns: &[String], functions: &'a FunctionRegistry) -> Self {
        let mut col_map = FxHashMap::default();
        let mut qualified_map: FxHashMap<String, FxHashMap<String, ColumnSource>> =
            FxHashMap::default();

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
            row2_tables: HashSet::new(),
            column2_offset: columns.len() as u16,
            outer_columns: None,
            functions,
            expression_aliases: FxHashMap::default(),
            column_aliases: FxHashMap::default(),
        }
    }

    /// Create context using global function registry
    pub fn with_global_registry(columns: &[String]) -> Self {
        Self::new(columns, global_registry())
    }

    /// Add second row columns (for join compilation)
    pub fn with_second_row(mut self, columns2: &[String]) -> Self {
        let mut col_map = FxHashMap::default();
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
            map.insert(Arc::from(col.to_lowercase().as_str()), i as u16);
        }
        self.outer_columns = Some(map);
        self
    }

    /// Add expression aliases (for HAVING clause)
    pub fn with_expression_aliases(mut self, aliases: FxHashMap<String, u16>) -> Self {
        self.expression_aliases = aliases;
        self
    }

    /// Add column aliases
    pub fn with_column_aliases(mut self, aliases: FxHashMap<String, String>) -> Self {
        self.column_aliases = aliases;
        self
    }

    /// Resolve a column name to its index
    fn resolve_column(&self, name: &str) -> Option<u16> {
        let lower = name.to_lowercase();

        // Check column aliases first
        if let Some(original) = self.column_aliases.get(&lower) {
            if let Some(&idx) = self.columns.get(original) {
                return Some(idx);
            }
        }

        // Direct lookup
        if let Some(&idx) = self.columns.get(&lower) {
            return Some(idx);
        }

        // Try second row if available
        if let Some(ref cols2) = self.columns2 {
            if let Some(&idx) = cols2.get(&lower) {
                return Some(self.column2_offset + idx);
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
        self.resolve_column(&column_lower).map(ColumnSource::Row1)
    }

    /// Resolve outer column (for correlated subqueries)
    fn resolve_outer_column(&self, name: &str) -> Option<Arc<str>> {
        let lower = name.to_lowercase();
        self.outer_columns.as_ref().and_then(|cols| {
            if cols.contains_key(lower.as_str()) {
                Some(Arc::from(lower.as_str()))
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
        let expr_str = expression_to_string(expr).to_lowercase();
        self.expression_aliases.get(&expr_str).copied()
    }
}

/// Expression compiler
pub struct ExprCompiler<'a> {
    ctx: &'a CompileContext<'a>,
}

impl<'a> ExprCompiler<'a> {
    pub fn new(ctx: &'a CompileContext<'a>) -> Self {
        Self { ctx }
    }

    /// Compile an expression into a Program
    pub fn compile(&self, expr: &Expression) -> Result<Program, CompileError> {
        let mut builder = ProgramBuilder::new();
        self.compile_expr(expr, &mut builder)?;
        builder.emit(Op::Return);
        Ok(builder.build())
    }

    /// Compile an expression for use as a boolean filter
    pub fn compile_filter(&self, expr: &Expression) -> Result<Program, CompileError> {
        // For simple filter expressions, we can optimize
        let mut builder = ProgramBuilder::new();
        self.compile_expr(expr, &mut builder)?;
        builder.emit(Op::Return);
        Ok(builder.build())
    }

    /// Compile an expression, emitting ops to the builder
    fn compile_expr(
        &self,
        expr: &Expression,
        builder: &mut ProgramBuilder,
    ) -> Result<(), CompileError> {
        // Check if this expression matches an expression alias (for HAVING)
        if let Some(idx) = self.ctx.check_expression_alias(expr) {
            builder.emit(Op::LoadAggregateResult(idx));
            return Ok(());
        }

        match expr {
            // === LITERALS ===
            Expression::IntegerLiteral(lit) => {
                builder.emit(Op::LoadConst(Value::Integer(lit.value)));
            }

            Expression::FloatLiteral(lit) => {
                builder.emit(Op::LoadConst(Value::Float(lit.value)));
            }

            Expression::StringLiteral(lit) => {
                // Handle type hints (DATE, TIMESTAMP, etc.)
                let value = if let Some(ref hint) = lit.type_hint {
                    match hint.to_uppercase().as_str() {
                        "TIMESTAMP" | "DATETIME" => crate::core::value::parse_timestamp(&lit.value)
                            .map(Value::Timestamp)
                            .unwrap_or_else(|_| Value::Text(Arc::from(lit.value.as_str()))),
                        "DATE" => crate::core::value::parse_timestamp(&lit.value)
                            .map(Value::Timestamp)
                            .unwrap_or_else(|_| Value::Text(Arc::from(lit.value.as_str()))),
                        _ => Value::Text(Arc::from(lit.value.as_str())),
                    }
                } else {
                    Value::Text(Arc::from(lit.value.as_str()))
                };
                builder.emit(Op::LoadConst(value));
            }

            Expression::BooleanLiteral(lit) => {
                builder.emit(Op::LoadConst(Value::Boolean(lit.value)));
            }

            Expression::NullLiteral(_) => {
                builder.emit(Op::LoadNull(DataType::Null));
            }

            // === IDENTIFIERS ===
            Expression::Identifier(id) => {
                // Check if it's a special keyword like CURRENT_DATE
                match id.value_lower.as_str() {
                    "current_date" => {
                        let now = chrono::Utc::now();
                        let date = chrono::TimeZone::with_ymd_and_hms(
                            &chrono::Utc,
                            now.year(),
                            now.month(),
                            now.day(),
                            0,
                            0,
                            0,
                        )
                        .single()
                        .unwrap_or(now);
                        builder.emit(Op::LoadConst(Value::Timestamp(date)));
                        return Ok(());
                    }
                    "current_timestamp" => {
                        builder.emit(Op::LoadConst(Value::Timestamp(chrono::Utc::now())));
                        return Ok(());
                    }
                    "current_time" => {
                        // Return current time as text "HH:MM:SS"
                        let now = chrono::Utc::now();
                        let time_str = now.format("%H:%M:%S").to_string();
                        builder.emit(Op::LoadConst(Value::Text(Arc::from(time_str.as_str()))));
                        return Ok(());
                    }
                    "true" => {
                        builder.emit(Op::LoadConst(Value::Boolean(true)));
                        return Ok(());
                    }
                    "false" => {
                        builder.emit(Op::LoadConst(Value::Boolean(false)));
                        return Ok(());
                    }
                    _ => {}
                }

                // Try to resolve as column
                // First, check if the identifier contains a dot (qualified name like "table.column")
                if let Some(dot_idx) = id.value_lower.rfind('.') {
                    // Treat as qualified identifier
                    let table = &id.value_lower[..dot_idx];
                    let column = &id.value_lower[dot_idx + 1..];
                    if let Some(source) = self.ctx.resolve_qualified(table, column) {
                        match source {
                            ColumnSource::Row1(idx) => builder.emit(Op::LoadColumn(idx)),
                            ColumnSource::Row2(idx) => builder.emit(Op::LoadColumn2(idx)),
                        }
                    } else if let Some(name) = self.ctx.resolve_outer_column(column) {
                        builder.emit(Op::LoadOuterColumn(name));
                    } else {
                        return Err(CompileError::ColumnNotFound(id.value.clone()));
                    }
                } else if let Some(idx) = self.ctx.resolve_column(&id.value_lower) {
                    builder.emit(Op::LoadColumn(idx));
                } else if let Some(name) = self.ctx.resolve_outer_column(&id.value_lower) {
                    builder.emit(Op::LoadOuterColumn(name));
                } else {
                    return Err(CompileError::ColumnNotFound(id.value.clone()));
                }
            }

            Expression::QualifiedIdentifier(qid) => {
                let table = &qid.qualifier.value_lower;
                let column = &qid.name.value_lower;

                if let Some(source) = self.ctx.resolve_qualified(table, column) {
                    match source {
                        ColumnSource::Row1(idx) => builder.emit(Op::LoadColumn(idx)),
                        ColumnSource::Row2(idx) => builder.emit(Op::LoadColumn2(idx)),
                    }
                } else if let Some(name) = self.ctx.resolve_outer_column(column) {
                    builder.emit(Op::LoadOuterColumn(name));
                } else {
                    return Err(CompileError::ColumnNotFound(format!(
                        "{}.{}",
                        table, column
                    )));
                }
            }

            // === PARAMETERS ===
            Expression::Parameter(param) => {
                if param.name.starts_with(':') {
                    let name = &param.name[1..];
                    builder.emit(Op::LoadNamedParam(Arc::from(name)));
                } else if param.index > 0 {
                    builder.emit(Op::LoadParam((param.index - 1) as u16));
                } else {
                    return Err(CompileError::InvalidExpression(
                        "Invalid parameter".to_string(),
                    ));
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
                let has_null = in_hash.values.iter().any(|v| v.is_null());
                if in_hash.not {
                    builder.emit(Op::NotInSet(in_hash.values.clone(), has_null));
                } else {
                    builder.emit(Op::InSet(in_hash.values.clone(), has_null));
                }
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
            Expression::IntervalLiteral(interval) => {
                let s = format!("{} {}", interval.quantity, interval.unit);
                builder.emit(Op::LoadConst(Value::Text(Arc::from(s.as_str()))));
            }

            // === SUBQUERIES ===
            Expression::ScalarSubquery(_) => {
                // Subqueries are handled via the subquery executor
                // For now, emit a placeholder that will be resolved at runtime
                builder.emit(Op::ExecScalarSubquery(0));
            }

            Expression::Exists(exists) => {
                let _ = exists; // Subquery index would be resolved during planning
                builder.emit(Op::ExecExists(0));
            }

            Expression::AllAny(all_any) => {
                self.compile_expr(&all_any.left, builder)?;
                let compare_op = match all_any.operator.as_str() {
                    "=" => CompareOp::Eq,
                    "!=" | "<>" => CompareOp::Ne,
                    "<" => CompareOp::Lt,
                    "<=" => CompareOp::Le,
                    ">" => CompareOp::Gt,
                    ">=" => CompareOp::Ge,
                    _ => CompareOp::Eq,
                };
                if matches!(all_any.all_any_type, AllAnyType::All) {
                    builder.emit(Op::ExecAll(0, compare_op));
                } else {
                    builder.emit(Op::ExecAny(0, compare_op));
                }
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
            | Expression::Star(_)
            | Expression::QualifiedStar(_)
            | Expression::Default(_) => {
                return Err(CompileError::InvalidExpression(
                    "Table sources and special expressions cannot be compiled".to_string(),
                ));
            }
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
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Add);
            }

            InfixOperator::Subtract => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Sub);
            }

            InfixOperator::Multiply => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Mul);
            }

            InfixOperator::Divide => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Div);
            }

            InfixOperator::Modulo => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Mod);
            }

            // String concatenation
            InfixOperator::Concat => {
                self.compile_expr(&infix.left, builder)?;
                self.compile_expr(&infix.right, builder)?;
                builder.emit(Op::Concat);
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
                if let Expression::StringLiteral(lit) = &*infix.right {
                    let pattern = CompiledPattern::compile(&lit.value, false);
                    builder.emit(Op::Like(Arc::new(pattern), false));
                } else {
                    // Dynamic pattern - less optimal
                    self.compile_expr(&infix.right, builder)?;
                    // Would need runtime pattern compilation
                    return Err(CompileError::UnsupportedExpression(
                        "Dynamic LIKE patterns not yet supported in VM".to_string(),
                    ));
                }
            }

            InfixOperator::ILike => {
                self.compile_expr(&infix.left, builder)?;
                if let Expression::StringLiteral(lit) = &*infix.right {
                    let pattern = CompiledPattern::compile(&lit.value, true);
                    builder.emit(Op::Like(Arc::new(pattern), true));
                } else {
                    return Err(CompileError::UnsupportedExpression(
                        "Dynamic ILIKE patterns not yet supported in VM".to_string(),
                    ));
                }
            }

            InfixOperator::NotLike => {
                self.compile_expr(&infix.left, builder)?;
                if let Expression::StringLiteral(lit) = &*infix.right {
                    let pattern = CompiledPattern::compile(&lit.value, false);
                    builder.emit(Op::Like(Arc::new(pattern), false));
                    builder.emit(Op::Not);
                } else {
                    return Err(CompileError::UnsupportedExpression(
                        "Dynamic NOT LIKE patterns not yet supported in VM".to_string(),
                    ));
                }
            }

            InfixOperator::NotILike => {
                self.compile_expr(&infix.left, builder)?;
                if let Expression::StringLiteral(lit) = &*infix.right {
                    let pattern = CompiledPattern::compile(&lit.value, true);
                    builder.emit(Op::Like(Arc::new(pattern), true));
                    builder.emit(Op::Not);
                } else {
                    return Err(CompileError::UnsupportedExpression(
                        "Dynamic NOT ILIKE patterns not yet supported in VM".to_string(),
                    ));
                }
            }

            InfixOperator::Glob | InfixOperator::NotGlob => {
                self.compile_expr(&infix.left, builder)?;
                if let Expression::StringLiteral(lit) = &*infix.right {
                    // Use compile_glob for GLOB patterns (uses * and ? wildcards)
                    let pattern = CompiledPattern::compile_glob(&lit.value);
                    builder.emit(Op::Glob(Arc::new(pattern)));
                    if matches!(infix.op_type, InfixOperator::NotGlob) {
                        builder.emit(Op::Not);
                    }
                } else {
                    return Err(CompileError::UnsupportedExpression(
                        "Dynamic GLOB patterns not yet supported in VM".to_string(),
                    ));
                }
            }

            InfixOperator::Regexp | InfixOperator::NotRegexp => {
                self.compile_expr(&infix.left, builder)?;
                if let Expression::StringLiteral(lit) = &*infix.right {
                    let regex = regex::Regex::new(&lit.value).map_err(|e| {
                        CompileError::InvalidExpression(format!("Invalid regex: {}", e))
                    })?;
                    builder.emit(Op::Regexp(Arc::new(regex)));
                    if matches!(infix.op_type, InfixOperator::NotRegexp) {
                        builder.emit(Op::Not);
                    }
                } else {
                    return Err(CompileError::UnsupportedExpression(
                        "Dynamic REGEXP patterns not yet supported in VM".to_string(),
                    ));
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

        // Single-value IN expression
        // Build the set of values at compile time if possible
        let mut values = AHashSet::default();
        let mut has_null = false;
        let mut all_constant = true;

        // Check if right side is a list of constants
        match &*in_expr.right {
            Expression::List(list) => {
                for item in &list.elements {
                    if let Some(value) = try_eval_constant(item) {
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
                    if let Some(value) = try_eval_constant(item) {
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
                if in_expr.not {
                    builder.emit(Op::NotInSet(Arc::new(values), has_null));
                } else {
                    builder.emit(Op::InSet(Arc::new(values), has_null));
                }
            }
        } else {
            // Fallback: evaluate each item (less efficient)
            // For now, return error - would need runtime set building
            return Err(CompileError::UnsupportedExpression(
                "Dynamic IN list not yet supported in VM".to_string(),
            ));
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
            if let Some(value) = try_eval_constant(element) {
                values.push(value);
            } else {
                return None;
            }
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

        // Extract escape character if present
        let escape_char: Option<char> = if let Some(ref escape_expr) = like.escape {
            if let Expression::StringLiteral(lit) = &**escape_expr {
                lit.value.chars().next()
            } else {
                None
            }
        } else {
            None
        };

        // Try to compile pattern at compile time
        if let Expression::StringLiteral(lit) = &*like.pattern {
            if is_regexp {
                let regex = regex::Regex::new(&lit.value).map_err(|e| {
                    CompileError::InvalidExpression(format!("Invalid regex: {}", e))
                })?;
                builder.emit(Op::Regexp(Arc::new(regex)));
            } else if is_glob {
                // Use compile_glob for GLOB patterns (uses * and ? wildcards)
                let pattern = CompiledPattern::compile_glob(&lit.value);
                builder.emit(Op::Glob(Arc::new(pattern)));
            } else if let Some(esc) = escape_char {
                // LIKE with ESCAPE - pre-process pattern to handle escape character
                let processed_pattern = self.process_like_escape(&lit.value, esc);
                let pattern = CompiledPattern::compile(&processed_pattern, case_insensitive);
                builder.emit(Op::LikeEscape(Arc::new(pattern), case_insensitive, esc));
            } else {
                let pattern = CompiledPattern::compile(&lit.value, case_insensitive);
                builder.emit(Op::Like(Arc::new(pattern), case_insensitive));
            }

            if negated {
                builder.emit(Op::Not);
            }
        } else {
            return Err(CompileError::UnsupportedExpression(
                "Dynamic LIKE patterns not yet supported in VM".to_string(),
            ));
        }

        Ok(())
    }

    /// Process LIKE pattern with escape character
    /// Converts escaped wildcards to special markers and then to literal characters
    fn process_like_escape(&self, pattern: &str, escape: char) -> String {
        let mut result = String::with_capacity(pattern.len());
        let mut chars = pattern.chars().peekable();

        while let Some(c) = chars.next() {
            if c == escape {
                // Next character should be treated literally
                if let Some(&next) = chars.peek() {
                    if next == '%' || next == '_' || next == escape {
                        // Escape the wildcard - use regex escape sequence
                        result.push('\\');
                        result.push(chars.next().unwrap());
                    } else {
                        // Not escaping a special character, keep the escape char
                        result.push(c);
                    }
                } else {
                    // Escape at end of pattern
                    result.push(c);
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
                // Simple CASE: compare operand with WHEN value
                builder.emit(Op::Dup); // Keep operand on stack
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
                builder.emit(Op::Pop); // Remove operand copy
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
                for arg in &func.arguments {
                    self.compile_expr(arg, builder)?;
                }
                builder.emit(Op::Coalesce(func.arguments.len() as u8));
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

            builder.emit(Op::CallScalar {
                func: scalar_func.into(),
                arg_count: func.arguments.len() as u8,
            });
            Ok(())
        } else {
            // Check if it's an aggregate being referenced post-aggregation
            // This would be handled via LoadAggregateResult in a real implementation
            Err(CompileError::FunctionNotFound(func_name))
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Try to evaluate a constant expression at compile time
fn try_eval_constant(expr: &Expression) -> Option<Value> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(Value::Integer(lit.value)),
        Expression::FloatLiteral(lit) => Some(Value::Float(lit.value)),
        Expression::StringLiteral(lit) => Some(Value::Text(Arc::from(lit.value.as_str()))),
        Expression::BooleanLiteral(lit) => Some(Value::Boolean(lit.value)),
        Expression::NullLiteral(_) => Some(Value::null_unknown()),
        _ => None,
    }
}

// Note: string_to_datatype and expression_to_string are now imported from utils

use chrono::Datelike;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::IntegerLiteral;
    use crate::parser::token::{Position, Token, TokenType};

    fn make_token() -> Token {
        Token {
            token_type: TokenType::Integer,
            literal: "1".to_string(),
            position: Position {
                offset: 0,
                line: 1,
                column: 1,
            },
            error: None,
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
            operator: ">".to_string(),
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
                operator: ">".to_string(),
                op_type: InfixOperator::GreaterThan,
                right: Box::new(Expression::IntegerLiteral(IntegerLiteral {
                    token: make_token(),
                    value: 5,
                })),
            })),
            operator: "AND".to_string(),
            op_type: InfixOperator::And,
            right: Box::new(Expression::Infix(InfixExpression {
                token: make_token(),
                left: Box::new(Expression::Identifier(Identifier::new(
                    make_token(),
                    "b".to_string(),
                ))),
                operator: "<".to_string(),
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
}
