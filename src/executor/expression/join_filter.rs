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

use std::cell::RefCell;

use crate::core::Row;
use crate::executor::context::ExecutionContext;

use super::bound_program::BoundProgram;
use super::program::SharedProgram;
use super::vm::{ExecuteContext, ExprVM, SubqueryExecutor};

/// A filter for join condition evaluation between two rows.
#[derive(Clone)]
pub struct JoinFilter {
    bound: BoundProgram,
}

impl JoinFilter {
    /// A filter over a program compiled earlier for the same column lists
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            bound: BoundProgram::new(program),
        }
    }

    /// This filter bound to the parameters and transaction of `ctx`, which
    /// a condition reading a parameter or CURRENT_TRANSACTION_ID() needs
    #[inline]
    pub fn with_context(self, ctx: &ExecutionContext) -> Self {
        Self {
            bound: self.bound.with_context(ctx),
        }
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
            let pair = ExecuteContext::for_join(left_row, right_row);
            let ctx = self.bound.context(pair, blocks);
            // A condition calling a function that runs a join again on this
            // thread gets a fresh VM, whose error goes to the fallback slot
            if let Ok(mut vm) = vm.try_borrow_mut() {
                let matched = vm.execute_bool(&self.bound.program, &ctx);
                if vm.has_bool_error() {
                    *failed = true;
                }
                matched
            } else {
                let mut vm = ExprVM::new();
                let matched = vm.execute_bool(&self.bound.program, &ctx);
                if let Some(e) = vm.take_bool_error() {
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
        &self.bound.program
    }
}

thread_local! {
    /// The VM join conditions run in on this thread
    static JOIN_VM: RefCell<ExprVM> = RefCell::new(ExprVM::new());
    /// An error from a join condition run on a temporary VM, for `take_error`
    static JOIN_FALLBACK_ERROR: RefCell<Option<crate::core::Error>> =
        const { RefCell::new(None) };
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
    use crate::api::params::ParamVec;
    use crate::common::CompactArc;
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
