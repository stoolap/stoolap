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

//! A join condition over a compiled program, run per pair of rows with the
//! statement's parameters and transaction

use crate::api::params::ParamVec;
use crate::common::CompactArc;
use crate::core::Row;
use crate::executor::context::ExecutionContext;

use super::program::{Program, SharedProgram};
use super::row_filter::EMPTY_PARAMS;
use super::vm::{ExecuteContext, ExprVM, SubqueryExecutor};

/// A filter for join condition evaluation between two rows.
#[derive(Clone)]
pub struct JoinFilter {
    /// Pre-compiled program
    program: SharedProgram,
    /// Query parameters (shared Arc to avoid cloning)
    params: CompactArc<ParamVec>,
    /// Transaction ID for CURRENT_TRANSACTION_ID()
    transaction_id: Option<u64>,
}

impl JoinFilter {
    /// A filter over a program compiled earlier for the same column lists
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            program,
            params: EMPTY_PARAMS.clone(),
            transaction_id: None,
        }
    }

    /// Set parameters and the transaction from the execution context.
    /// Required when the join condition holds parameter placeholders
    /// ($1, $2, ...) or reads CURRENT_TRANSACTION_ID().
    #[inline]
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        self.params = CompactArc::clone(ctx.params_arc());
        self.program = Program::bound(self.program, &self.params);
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Check if a pair of rows satisfies the join condition. An error in it
    /// reads as false, sets `failed` and is kept on this thread until
    /// `take_error`
    #[inline]
    pub fn matches(&self, left_row: &Row, right_row: &Row, failed: &mut bool) -> bool {
        self.matches_with(left_row, right_row, failed, None)
    }

    /// `matches` for a condition whose subqueries run as the statement's
    /// prepared blocks, which `blocks` answers for the pair
    #[inline]
    pub fn matches_with(
        &self,
        left_row: &Row,
        right_row: &Row,
        failed: &mut bool,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> bool {
        JOIN_VM.with(|vm| {
            let mut ctx = ExecuteContext::for_join(left_row, right_row);

            // Apply params if present (required for parameter placeholders like $1, $2)
            let params = blocks.and_then(|b| b.params()).unwrap_or(&self.params);
            if !params.is_empty() {
                ctx = ctx.with_params(params);
            }
            ctx = ctx.with_transaction_id(self.transaction_id);
            if let Some(blocks) = blocks {
                ctx = ctx.with_subquery_executor(blocks);
            }

            // Use try_borrow_mut to avoid panic on recursive calls (e.g., nested subqueries).
            // If the VM is already borrowed, create a temporary one for this call.
            if let Ok(mut borrowed_vm) = vm.try_borrow_mut() {
                let matched = borrowed_vm.execute_bool(&self.program, &ctx);
                if borrowed_vm.has_bool_error() {
                    *failed = true;
                }
                matched
            } else {
                // Fallback: create a fresh VM for recursive calls
                let mut temp_vm = ExprVM::new();
                let matched = temp_vm.execute_bool(&self.program, &ctx);
                if let Some(e) = temp_vm.take_bool_error() {
                    JOIN_FALLBACK_ERROR.with(|slot| {
                        slot.borrow_mut().get_or_insert(e);
                    });
                    *failed = true;
                }
                matched
            }
        })
    }

    /// The first error a condition on this thread read as false since the
    /// last call. A join clears it when it opens
    pub fn take_error(&self) -> Option<crate::core::Error> {
        JOIN_FALLBACK_ERROR
            .with(|slot| slot.borrow_mut().take())
            .or_else(|| JOIN_VM.with(|vm| vm.try_borrow_mut().ok()?.take_bool_error()))
    }

    /// The error behind a `failed` flag that `matches` set
    #[cold]
    pub fn failure(&self) -> crate::core::Error {
        self.take_error().unwrap_or_else(|| {
            crate::core::Error::internal("a join condition failed without an error")
        })
    }

    /// Get the underlying program.
    pub fn program(&self) -> &SharedProgram {
        &self.program
    }
}

thread_local! {
    /// The VM join conditions run in on this thread
    static JOIN_VM: std::cell::RefCell<ExprVM> = std::cell::RefCell::new(ExprVM::new());
    /// An error from a join condition run on a temporary VM, for `take_error`
    static JOIN_FALLBACK_ERROR: std::cell::RefCell<Option<crate::core::Error>> =
        const { std::cell::RefCell::new(None) };
}

// Static assertions to verify JoinFilter implements Send + Sync.
// This is safer than unsafe impl because it will fail at compile time
// if any field doesn't implement Send/Sync, rather than causing UB at runtime.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<JoinFilter>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;
    use crate::executor::expression::{CompileContext, ExprCompiler};
    use crate::functions::global_registry;

    #[test]
    fn a_join_filter_given_another_context_reads_its_parameters() {
        let sql = "SELECT 1 FROM t WHERE a = $1";
        let condition = match crate::parser::parse_sql(sql).unwrap().remove(0) {
            crate::parser::ast::Statement::Select(select) => *select.where_clause.unwrap(),
            _ => unreachable!(),
        };
        let (left, right) = (["a".to_string()], ["b".to_string()]);
        let compile = CompileContext::new(&left, global_registry()).with_second_row(&right);
        let program = ExprCompiler::new(&compile).compile(&condition).unwrap();
        let context = |value: i64| {
            ExecutionContext::with_params(ParamVec::from_iter([Value::Integer(value)]))
        };
        let join = JoinFilter::from_program(CompactArc::new(program))
            .with_context(&context(1))
            .with_context(&context(2));
        let mut failed = false;
        let (row, other) = (
            Row::from(vec![Value::Integer(2)]),
            Row::from(vec![Value::Integer(0)]),
        );
        assert!(
            join.matches(&row, &other, &mut failed),
            "JoinFilter kept $1 = 1"
        );
    }
}
