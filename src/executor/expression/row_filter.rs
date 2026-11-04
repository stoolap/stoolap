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

//! A filter over a compiled program, run per row with the statement's
//! parameters and transaction

use crate::api::params::ParamVec;
use crate::common::CompactArc;
use crate::core::{Result, Row};
use crate::executor::context::ExecutionContext;

use super::program::{Program, SharedProgram};
use super::vm::{ExecuteContext, ExprVM, SubqueryExecutor};

/// The parameters of a filter no context gave any, shared by every filter
pub(super) static EMPTY_PARAMS: std::sync::LazyLock<CompactArc<ParamVec>> =
    std::sync::LazyLock::new(|| CompactArc::new(ParamVec::new()));

/// A lightweight, thread-safe row filter for closure-based filtering.
///
/// `RowFilter` holds a compiled program and can be cloned cheaply (it uses
/// `CompactArc<Program>` internally). Each thread runs it in a VM of its own.
#[derive(Clone)]
pub struct RowFilter {
    /// Pre-compiled program (shared across clones)
    program: SharedProgram,
    /// Query parameters (shared) - uses CompactArc<Vec<Value>> to match ExecutionContext
    params: CompactArc<ParamVec>,
    /// Transaction ID for CURRENT_TRANSACTION_ID()
    transaction_id: Option<u64>,
}

impl RowFilter {
    /// A filter over a program compiled earlier for the same columns
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            program,
            params: EMPTY_PARAMS.clone(),
            transaction_id: None,
        }
    }

    /// Create a filter from execution context.
    ///
    /// PERF: `params` shares the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed (both use CompactArc<Vec<Value>>)
        self.params = CompactArc::clone(ctx.params_arc());
        self.program = Program::bound(self.program, &self.params);
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// The parameters the program reads, in place of the context's
    pub(crate) fn with_params(mut self, params: CompactArc<ParamVec>) -> Self {
        self.params = params;
        self.program = Program::bound(self.program, &self.params);
        self
    }

    /// Whether every thread may run this filter as it is, without a
    /// [`Self::worker`] of its own
    #[cfg(feature = "parallel")]
    pub(crate) fn shares_across_threads(&self) -> bool {
        !self.program.holds_thread_state()
    }

    /// This filter for one more thread, over the program's worker copy
    #[cfg(feature = "parallel")]
    pub(crate) fn worker(&self) -> Self {
        Self {
            program: CompactArc::new(self.program.worker_copy()),
            params: CompactArc::clone(&self.params),
            transaction_id: self.transaction_id,
        }
    }

    /// Check a row against the filter, returning an evaluation error instead of swallowing it.
    ///
    /// Returns `Err` when the VM encounters a runtime error (e.g. invalid
    /// REGEXP pattern supplied via a parameter). Used by FilteredResult
    /// to surface errors through the Rows iterator.
    #[inline]
    pub fn matches_checked(&self, row: &Row) -> Result<bool> {
        self.matches_checked_with(row, None)
    }

    /// `matches_checked` for a program whose subqueries run as the
    /// statement's prepared blocks, which `blocks` answers
    #[inline]
    pub fn matches_checked_with(
        &self,
        row: &Row,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<bool> {
        thread_local! {
            static VM: std::cell::RefCell<ExprVM> = std::cell::RefCell::new(ExprVM::new());
        }

        VM.with(|vm| {
            let mut ctx = ExecuteContext::new(row);

            let params = blocks.and_then(|b| b.params()).unwrap_or(&self.params);
            if !params.is_empty() {
                ctx = ctx.with_params(params);
            }
            ctx = ctx.with_transaction_id(self.transaction_id);
            if let Some(blocks) = blocks {
                ctx = ctx.with_subquery_executor(blocks);
            }

            if let Ok(mut borrowed_vm) = vm.try_borrow_mut() {
                borrowed_vm.execute_bool_checked(&self.program, &ctx)
            } else {
                let mut temp_vm = ExprVM::new();
                temp_vm.execute_bool_checked(&self.program, &ctx)
            }
        })
    }

    /// Filter a RowVec in-place, removing rows that don't match.
    /// Returns Err if the filter expression produces a runtime error
    /// (e.g. invalid REGEXP pattern supplied via a parameter).
    pub fn retain_checked(&self, rows: &mut crate::core::RowVec) -> Result<()> {
        self.retain_checked_with(rows, None)
    }

    /// `retain_checked` for a program whose subqueries run as the
    /// statement's prepared blocks, which `blocks` answers
    pub fn retain_checked_with(
        &self,
        rows: &mut crate::core::RowVec,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<()> {
        let mut error: Option<crate::core::Error> = None;
        rows.retain(|(_, row)| {
            if error.is_some() {
                return false;
            }
            match self.matches_checked_with(row, blocks) {
                Ok(b) => b,
                Err(e) => {
                    error = Some(e);
                    false
                }
            }
        });
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

// Static assertions to verify RowFilter implements Send + Sync.
// This is safer than unsafe impl because it will fail at compile time
// if any field doesn't implement Send/Sync, rather than causing UB at runtime.
// All fields are Send + Sync:
// - CompactArc<Program> is Send + Sync (Program is immutable)
// - CompactArc<Value> is Send + Sync
// - Arc<FxHashMap<String, Value>> is Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<RowFilter>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;
    use crate::executor::expression::{CompileContext, ExprCompiler};
    use crate::parser::ast::Expression;

    fn where_of(sql: &str) -> Expression {
        match crate::parser::parse_sql(sql).unwrap().remove(0) {
            crate::parser::ast::Statement::Select(select) => *select.where_clause.unwrap(),
            _ => unreachable!(),
        }
    }

    fn filter(sql: &str, columns: &[String]) -> RowFilter {
        let compile = CompileContext::with_global_registry(columns);
        let program = ExprCompiler::new(&compile).compile(&where_of(sql)).unwrap();
        RowFilter::from_program(CompactArc::new(program))
    }

    fn context(value: i64) -> ExecutionContext {
        ExecutionContext::with_params(ParamVec::from_iter([Value::Integer(value)]))
    }

    #[test]
    fn a_filter_keeps_the_rows_its_program_passes() {
        let filter = filter("SELECT 1 FROM t WHERE col > 5", &["col".to_string()]);
        let cloned = filter.clone();
        for keep in [&filter, &cloned] {
            assert!(keep
                .matches_checked(&Row::from(vec![Value::Integer(10)]))
                .unwrap());
            assert!(!keep
                .matches_checked(&Row::from(vec![Value::Integer(3)]))
                .unwrap());
        }
    }

    #[test]
    fn a_filter_given_another_context_reads_its_parameters() {
        let filter = filter("SELECT 1 FROM t WHERE a = $1", &["a".to_string()])
            .with_context(&context(1))
            .with_context(&context(2));
        assert!(
            filter
                .matches_checked(&Row::from(vec![Value::Integer(2)]))
                .unwrap(),
            "RowFilter kept $1 = 1"
        );
    }
}
