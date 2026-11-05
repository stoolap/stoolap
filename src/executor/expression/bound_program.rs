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

//! A compiled program with the parameters and transaction a statement's run
//! gives it

use crate::api::params::ParamVec;
use crate::common::CompactArc;
use crate::executor::context::ExecutionContext;

use super::program::{Program, SharedProgram};
use super::vm::{ExecuteContext, SubqueryExecutor};

/// The parameters of a program no context gave any, shared by every one
static EMPTY_PARAMS: std::sync::LazyLock<CompactArc<ParamVec>> =
    std::sync::LazyLock::new(|| CompactArc::new(ParamVec::new()));

/// A program, the parameters it was bound to and the transaction it reads
#[derive(Clone)]
pub(crate) struct BoundProgram {
    pub(crate) program: SharedProgram,
    params: CompactArc<ParamVec>,
    transaction_id: Option<u64>,
}

impl BoundProgram {
    /// `program` with no parameters and no transaction
    pub(crate) fn new(program: SharedProgram) -> Self {
        Self {
            program,
            params: EMPTY_PARAMS.clone(),
            transaction_id: None,
        }
    }

    /// This program bound to the parameters and transaction of `ctx`
    pub(crate) fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        self = self.with_params(CompactArc::clone(ctx.params_arc()));
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// This program bound to `params`
    pub(crate) fn with_params(mut self, params: CompactArc<ParamVec>) -> Self {
        self.params = params;
        self.program = Program::bound(self.program, &self.params);
        self
    }

    /// The program over a worker copy, for one more thread
    #[cfg(feature = "parallel")]
    pub(crate) fn worker(&self) -> Self {
        Self {
            program: CompactArc::new(self.program.worker_copy()),
            params: CompactArc::clone(&self.params),
            transaction_id: self.transaction_id,
        }
    }

    /// `base` reading the run this program was bound to, its subqueries
    /// answered by the statement's `blocks`
    #[inline(always)]
    pub(crate) fn context<'a>(
        &'a self,
        base: ExecuteContext<'a>,
        blocks: Option<&'a dyn SubqueryExecutor>,
    ) -> ExecuteContext<'a> {
        base.with_run(&self.params, self.transaction_id, blocks)
    }
}
