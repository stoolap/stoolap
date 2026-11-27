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

//! An aggregate call's argument as the table computes it: the program the
//! statement's own path compiles, run by the same VM over chunks of the
//! rows a reader holds, a step at a time where every op allows it

use std::sync::Arc;

use super::{ExecuteContext, ExprVM, SharedProgram};
use crate::core::{Error, Result, Row, Value};
use crate::functions::aggregate::GroupedAggregates;

/// Rows a reader holds before it computes their arguments
pub const ARGUMENT_CHUNK: usize = 2048;

/// An argument compiled over the table columns it reads, `columns` in the
/// order its program reads them, with the statement's parameters, the call
/// as written, which an error names, and whether it runs a step at a time
#[derive(Debug)]
pub struct ArgumentProgram {
    program: SharedProgram,
    pub columns: Vec<usize>,
    params: Vec<Value>,
    call: (String, String),
    batched: bool,
}

impl ArgumentProgram {
    pub fn new(
        program: SharedProgram,
        columns: Vec<usize>,
        params: Vec<Value>,
        call: (String, String),
    ) -> Self {
        let batched = ExprVM::runs_batched(&program);
        Self {
            program,
            columns,
            params,
            call,
            batched,
        }
    }
}

/// The VM, the row and the vectors one reader runs arguments with
#[derive(Default)]
pub struct ArgumentRun {
    vm: ExprVM,
    row: Row,
    out: Vec<Value>,
    pool: Vec<Vec<Value>>,
}

impl ArgumentRun {
    /// The values of `argument` over the `n` rows whose cell of its column
    /// `k` is `leaves[k]`: a step at a time when it runs so, otherwise, or
    /// when a step fails, a row at a time, which gives the first row's error
    pub fn evaluate_chunk(
        &mut self,
        argument: &ArgumentProgram,
        leaves: &[Vec<Value>],
        n: usize,
    ) -> Result<&[Value]> {
        if argument.batched {
            let run = self.vm.execute_batch(
                &argument.program,
                (leaves, n),
                &argument.params,
                &mut self.out,
                &mut self.pool,
            );
            if run.is_ok() {
                return Ok(&self.out);
            }
        }
        self.out.clear();
        self.out.reserve(n);
        for row in 0..n {
            let value = self.evaluate_row(argument, leaves, row)?;
            self.out.push(value);
        }
        Ok(&self.out)
    }

    /// The value of `argument` for row `row` of `leaves`; an error is the
    /// call's, as `Error::aggregate_argument` names it
    pub fn evaluate_row(
        &mut self,
        argument: &ArgumentProgram,
        leaves: &[Vec<Value>],
        row: usize,
    ) -> Result<Value> {
        let width = leaves.len();
        if self.row.len() < width {
            self.row = Row::from_values(vec![Value::null_unknown(); width]);
        }
        for (at, leaf) in leaves.iter().enumerate() {
            if let Some(slot) = self.row.get_mut(at) {
                *slot = leaf[row].clone();
            }
        }
        let mut run = ExecuteContext::new(&self.row);
        run.params = &argument.params;
        let (name, written) = &argument.call;
        self.vm
            .execute(&argument.program, &run)
            .map_err(|error| Error::aggregate_argument(name, written, error))
    }
}

/// One argument's call, program and the cells of its columns a batch holds
struct HeldArgument {
    call: usize,
    program: Arc<ArgumentProgram>,
    leaves: Vec<Vec<Value>>,
}

/// The rows one reader holds for the calls of a statement that read an
/// expression: each row's group, and each argument's cells or the row's
/// position to read them from, computed and fed to the calls a chunk at a
/// time
pub struct ExpressionBatch {
    arguments: Vec<HeldArgument>,
    groups: Vec<usize>,
    positions: Vec<usize>,
    run: ArgumentRun,
}

impl ExpressionBatch {
    /// A batch for `arguments`, each call's index and program
    pub fn new(arguments: impl Iterator<Item = (usize, Arc<ArgumentProgram>)>) -> Self {
        let arguments = arguments
            .map(|(call, program)| HeldArgument {
                call,
                leaves: (0..program.columns.len())
                    .map(|_| Vec::with_capacity(ARGUMENT_CHUNK))
                    .collect(),
                program,
            })
            .collect();
        Self {
            arguments,
            groups: Vec::with_capacity(ARGUMENT_CHUNK),
            positions: Vec::new(),
            run: ArgumentRun::default(),
        }
    }

    /// Holds the row at `position` of group `group`, its cells read when
    /// the chunk is fed; true once a chunk is held
    #[inline(always)]
    pub fn hold_position(&mut self, group: usize, position: usize) -> bool {
        if self.positions.capacity() == 0 {
            self.positions.reserve(ARGUMENT_CHUNK);
        }
        self.positions.push(position);
        self.groups.push(group);
        self.groups.len() >= ARGUMENT_CHUNK
    }

    /// Holds a row of group `group` whose cell of argument `a`'s column `k`,
    /// table column `column`, `cell(a, k, column)` gives; true once a chunk
    /// is held
    #[inline]
    pub fn hold(&mut self, group: usize, cell: impl Fn(usize, usize, usize) -> Value) -> bool {
        for (a, held) in self.arguments.iter_mut().enumerate() {
            for (k, leaf) in held.leaves.iter_mut().enumerate() {
                leaf.push(cell(a, k, held.program.columns[k]));
            }
        }
        self.groups.push(group);
        self.groups.len() >= ARGUMENT_CHUNK
    }

    /// Computes the held rows' arguments and feeds each to its call in
    /// `groups`, the rows let go: the cells of rows held by position are
    /// read by `gather(a, k, positions, leaf)` into argument `a`'s column
    /// `k`. The error is the first row's, and of its calls the first's, as
    /// the statement's own path finds it
    pub fn flush(
        &mut self,
        groups: &mut GroupedAggregates,
        gather: impl Fn(usize, usize, &[usize], &mut Vec<Value>),
    ) -> Result<()> {
        let n = self.groups.len();
        if n == 0 {
            return Ok(());
        }
        if !self.positions.is_empty() {
            for (a, held) in self.arguments.iter_mut().enumerate() {
                for (k, leaf) in held.leaves.iter_mut().enumerate() {
                    gather(a, k, &self.positions, leaf);
                }
            }
        }
        let fed = self.feed(groups, n);
        for held in &mut self.arguments {
            held.leaves.iter_mut().for_each(Vec::clear);
        }
        self.groups.clear();
        self.positions.clear();
        fed
    }

    fn feed(&mut self, groups: &mut GroupedAggregates, n: usize) -> Result<()> {
        let Self {
            arguments,
            groups: held_groups,
            run,
            ..
        } = self;
        for held in arguments.iter() {
            match run.evaluate_chunk(&held.program, &held.leaves, n) {
                Ok(values) => {
                    for (value, &group) in values.iter().zip(held_groups.iter()) {
                        groups.aggregates(group)[held.call].accumulate(value);
                    }
                }
                Err(error) => return Err(first_error(arguments, run, n).unwrap_or(error)),
            }
        }
        Ok(())
    }
}

/// The error of the first row, and of its calls the first's, that fails
#[cold]
fn first_error(arguments: &[HeldArgument], run: &mut ArgumentRun, n: usize) -> Option<Error> {
    (0..n).find_map(|row| {
        arguments
            .iter()
            .find_map(|held| run.evaluate_row(&held.program, &held.leaves, row).err())
    })
}
