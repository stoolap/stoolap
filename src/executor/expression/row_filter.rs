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

use std::cell::RefCell;

use crate::api::params::ParamVec;
use crate::common::CompactArc;
use crate::core::{Result, Row};
use crate::executor::context::ExecutionContext;

use super::bound_program::BoundProgram;
use super::program::SharedProgram;
use super::vm::{ExecuteContext, ExprVM, SubqueryExecutor};

thread_local! {
    /// The VM row filters run in on this thread
    static VM: RefCell<ExprVM> = RefCell::new(ExprVM::new());
}

/// A lightweight, thread-safe row filter for closure-based filtering.
///
/// `RowFilter` holds a compiled program and can be cloned cheaply (it uses
/// `CompactArc<Program>` internally). Each thread runs it in a VM of its own.
#[derive(Clone)]
pub struct RowFilter {
    bound: BoundProgram,
}

impl RowFilter {
    /// A filter over a program compiled earlier for the same columns
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            bound: BoundProgram::new(program),
        }
    }

    /// This filter bound to the parameters and transaction of `ctx`
    pub fn with_context(self, ctx: &ExecutionContext) -> Self {
        Self {
            bound: self.bound.with_context(ctx),
        }
    }

    /// The parameters the program reads, in place of the context's
    pub(crate) fn with_params(self, params: CompactArc<ParamVec>) -> Self {
        Self {
            bound: self.bound.with_params(params),
        }
    }

    /// Whether every thread may run this filter as it is, without a
    /// [`Self::worker`] of its own
    #[cfg(feature = "parallel")]
    pub(crate) fn shares_across_threads(&self) -> bool {
        !self.bound.program.holds_thread_state()
    }

    /// This filter for one more thread, over the program's worker copy
    #[cfg(feature = "parallel")]
    pub(crate) fn worker(&self) -> Self {
        Self {
            bound: self.bound.worker(),
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
        VM.with(|vm| {
            let ctx = self.bound.context(ExecuteContext::new(row), blocks);
            // A filter calling a function that runs a filter again on this
            // thread gets a fresh VM
            match vm.try_borrow_mut() {
                Ok(mut vm) => vm.execute_bool_checked(&self.bound.program, &ctx),
                Err(_) => ExprVM::new().execute_bool_checked(&self.bound.program, &ctx),
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
