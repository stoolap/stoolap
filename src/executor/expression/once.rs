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

//! Expressions a statement evaluates once, over no row: compiled for that
//! one evaluation and kept nowhere, so they never enter a program cache

use std::cell::RefCell;

use super::compiler::{
    literal_value, parameter_slot, CompileContext, CompileError, ExprCompiler, ParameterSlot,
};
use super::program::Program;
use super::vm::{ExecuteContext, ExprVM};
use crate::core::{Error, Result, Row, Value};
use crate::executor::context::ExecutionContext;
use crate::parser::ast::Expression;

thread_local! {
    static ONCE_VM: RefCell<ExprVM> = RefCell::new(ExprVM::new());
}

fn compile_error(e: CompileError) -> Error {
    Error::internal(format!("Compile error: {}", e))
}

/// Compile `expr` over no columns, as a one-shot evaluation does
pub(crate) fn compile_once(expr: &Expression) -> Result<Program> {
    let empty: &[String] = &[];
    ExprCompiler::new(&CompileContext::with_global_registry(empty))
        .compile(expr)
        .map_err(compile_error)
}

/// The value of `expr` over no row, with `ctx`'s parameters, outer row and
/// transaction
pub(crate) fn evaluate_once(expr: &Expression, ctx: Option<&ExecutionContext>) -> Result<Value> {
    // A literal or a parameter is the value its program would load
    if let Some(value) = literal_value(expr) {
        return Ok(value);
    }
    let row = Row::new();
    let exec = context(&row, ctx);
    if let Expression::Parameter(param) = expr {
        let value = match parameter_slot(param).map_err(compile_error)? {
            ParameterSlot::Named(name) => exec.named_param_ref(name),
            ParameterSlot::Positional(index) => exec.param_ref(index as usize),
        };
        return Ok(value.clone());
    }
    run(&compile_once(expr)?, &exec)
}

/// What [`evaluate_once`] does after compiling
pub(crate) fn execute_once(program: &Program, ctx: Option<&ExecutionContext>) -> Result<Value> {
    let row = Row::new();
    run(program, &context(&row, ctx))
}

fn context<'a>(row: &'a Row, ctx: Option<&'a ExecutionContext>) -> ExecuteContext<'a> {
    let mut exec = ExecuteContext::new(row);
    if let Some(ctx) = ctx {
        exec = exec
            .with_params(ctx.params())
            .with_named_params(ctx.named_params())
            .with_transaction_id(ctx.transaction_id());
        if let Some(outer) = ctx.outer_row() {
            exec = exec.with_outer_row(outer);
        }
    }
    exec
}

fn run(program: &Program, exec: &ExecuteContext) -> Result<Value> {
    // A function that runs a statement of its own evaluates here again
    ONCE_VM.with(|vm| match vm.try_borrow_mut() {
        Ok(mut vm) => vm.execute(program, exec),
        Err(_) => ExprVM::new().execute(program, exec),
    })
}

/// A column-free expression's value for pushdown, or None when it reads
/// columns, needs the transaction, or fails to evaluate. NOW() and RANDOM()
/// are evaluated: pushdown resolves them once per statement
pub(crate) fn try_eval_constant_expr_with_context(
    expr: &Expression,
    context: Option<&ExecutionContext>,
) -> Option<Value> {
    if contains_context_dependent_function(expr) {
        return None;
    }
    evaluate_once(expr, context).ok()
}

/// Whether `expr` calls CURRENT_TRANSACTION_ID, which reads the transaction
fn contains_context_dependent_function(expr: &Expression) -> bool {
    match expr {
        Expression::FunctionCall(func) => {
            func.function.eq_ignore_ascii_case("CURRENT_TRANSACTION_ID")
                || func
                    .arguments
                    .iter()
                    .any(contains_context_dependent_function)
        }
        Expression::Infix(infix) => {
            contains_context_dependent_function(&infix.left)
                || contains_context_dependent_function(&infix.right)
        }
        Expression::Prefix(prefix) => contains_context_dependent_function(&prefix.right),
        Expression::Cast(cast) => contains_context_dependent_function(&cast.expr),
        Expression::Case(case) => {
            case.value
                .as_ref()
                .is_some_and(|v| contains_context_dependent_function(v))
                || case.when_clauses.iter().any(|w| {
                    contains_context_dependent_function(&w.condition)
                        || contains_context_dependent_function(&w.then_result)
                })
                || case
                    .else_value
                    .as_ref()
                    .is_some_and(|v| contains_context_dependent_function(v))
        }
        Expression::Between(between) => {
            contains_context_dependent_function(&between.expr)
                || contains_context_dependent_function(&between.lower)
                || contains_context_dependent_function(&between.upper)
        }
        Expression::In(in_expr) => {
            contains_context_dependent_function(&in_expr.left)
                || contains_context_dependent_function(&in_expr.right)
        }
        Expression::Like(like) => {
            contains_context_dependent_function(&like.left)
                || contains_context_dependent_function(&like.pattern)
                || like
                    .escape
                    .as_ref()
                    .is_some_and(|e| contains_context_dependent_function(e))
        }
        Expression::List(list) => list
            .elements
            .iter()
            .any(contains_context_dependent_function),
        Expression::ExpressionList(list) => list
            .expressions
            .iter()
            .any(contains_context_dependent_function),
        Expression::Aliased(aliased) => contains_context_dependent_function(&aliased.expression),
        Expression::Distinct(distinct) => contains_context_dependent_function(&distinct.expr),
        Expression::AllAny(all_any) => contains_context_dependent_function(&all_any.left),
        Expression::InHashSet(in_hash) => contains_context_dependent_function(&in_hash.column),
        Expression::QuantifiedSet(quantified) => {
            contains_context_dependent_function(&quantified.left)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::executor::expression::evaluator_bridge::program_cached;
    use crate::executor::Executor;
    use crate::parser::ast::Statement;
    use crate::storage::mvcc::engine::MVCCEngine;

    /// The SELECT expression of `SELECT {text}`
    fn expression(text: &str) -> crate::parser::ast::Expression {
        match &crate::parser::parse_sql(&format!("SELECT {text}")).unwrap()[0] {
            Statement::Select(select) => select.columns[0].clone(),
            other => panic!("not a SELECT: {other:?}"),
        }
    }

    #[test]
    fn a_value_a_statement_reads_once_is_not_cached() {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        let executor = Executor::new(Arc::new(engine));
        executor
            .execute("CREATE TABLE once_t (id INTEGER PRIMARY KEY)")
            .unwrap();
        executor
            .execute("INSERT INTO once_t VALUES (1), (2), (3)")
            .unwrap();
        for (sql, value) in [
            ("SELECT id FROM once_t LIMIT 1 + 700001", "1 + 700001"),
            (
                "SELECT id FROM once_t LIMIT 1 OFFSET 0 * 700002",
                "0 * 700002",
            ),
            ("SELECT * FROM generate_series(1, 2 + 700003)", "2 + 700003"),
            ("SELECT 3 + 700004", "3 + 700004"),
            ("SELECT 1 WHERE 4 + 700005 > 0", "4 + 700005 > 0"),
            (
                "SELECT id FROM once_t WHERE id > 0 * 700006 ORDER BY id LIMIT 2",
                "0 * 700006",
            ),
            (
                "SELECT id FROM once_t WHERE id IN (0 * 700007, 1)",
                "0 * 700007",
            ),
            (
                "SELECT id FROM once_t WHERE 8 + 700008 < 0",
                "8 + 700008 < 0",
            ),
            (
                "SELECT STRING_AGG(CAST(id AS TEXT), '-' || '700009') FROM once_t",
                "'-' || '700009'",
            ),
            (
                "ALTER TABLE once_t ADD COLUMN c INTEGER DEFAULT 10 + 700010",
                "10 + 700010",
            ),
        ] {
            let _ = executor.execute(sql).unwrap();
            assert!(
                !program_cached(&expression(value), &[]),
                "{sql}: `{value}` is read once, yet it entered the program cache"
            );
        }
    }
}
