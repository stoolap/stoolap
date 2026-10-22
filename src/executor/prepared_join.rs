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

//! What a join statement builds once, held with its plan: programs by the
//! operator's choice they were compiled for, the join's analysis and its
//! output, each by the columns it reads

use std::sync::{Arc, RwLock};

use smallvec::SmallVec;

use crate::core::Result;

use super::expression::SharedProgram;
use super::join_executor::JoinAnalysis;
use super::prepared_select::JoinBlocks;
use super::query::JoinOutput;

/// The programs one compile builds, shared only once they are held
pub type BuiltPrograms = SmallVec<[SharedProgram; 2]>;

/// What a held program was compiled for: the part of the ON clause an
/// operator asks of each pair, by the keys it answers itself, or the WHERE
/// asked of each joined row
#[derive(Debug)]
pub enum JoinProgramKey {
    /// The ON clause without the equality an index probe answers, and the
    /// WHERE's filter on the inner table, over a (left, right) pair
    IndexResidual {
        swapped: bool,
        outer_col: String,
        inner_col: String,
        outer_cols: Vec<String>,
        inner_cols: Vec<String>,
        /// The inner fetch takes the WHERE's filter on the inner table
        inner_taken: bool,
    },
    /// The ON clause without the keys a hash join matches, over the joined row
    HashResidual {
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        columns: Vec<String>,
    },
    /// The whole ON clause over a (left, right) pair
    Pair {
        left_cols: Vec<String>,
        right_cols: Vec<String>,
    },
    /// The ORDER BY keys an index join sorts its joined rows by
    OrderKeys { columns: Vec<String> },
}

/// Where a join asks its WHERE of the joined rows
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhereSite {
    /// An index nested loop, per joined row toward its LIMIT
    IndexJoin,
    /// After the hash join or nested loop produced its rows
    AfterJoin,
}

/// The joined rows a WHERE was compiled over, by where it is asked
#[derive(Debug)]
struct WhereKey {
    site: WhereSite,
    columns: Vec<String>,
}

/// The key equalities a join's analysis was asked for: the sides' columns,
/// and the key pairs it was told to leave in the residual
#[derive(Debug)]
struct AnalysisKey {
    left_cols: Vec<String>,
    right_cols: Vec<String>,
    kept_out: Vec<(usize, usize)>,
}

/// The joined rows a statement's output was built for: their columns, and
/// whether the join ordered them already
#[derive(Debug)]
struct OutputKey {
    columns: Vec<String>,
    sorted: bool,
}

/// A value held with the plan, or built for this execution alone
pub enum Held<T> {
    Shared(Arc<T>),
    Own(T),
}

impl<T> std::ops::Deref for Held<T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Held::Shared(value) => value,
            Held::Own(value) => value,
        }
    }
}

/// More of one kind than one statement's joins build are not kept
const MAX_HELD: usize = 16;

/// Values of one kind held by key; a lookup copies no key
#[derive(Debug)]
struct Shelf<K, T: ?Sized> {
    held: RwLock<Vec<(K, Arc<T>)>>,
}

impl<K, T: ?Sized> Shelf<K, T> {
    fn new() -> Self {
        Self {
            held: RwLock::new(Vec::new()),
        }
    }

    fn find(&self, wanted: impl Fn(&K) -> bool) -> Option<Arc<T>> {
        let held = self.held.read().ok()?;
        held.iter()
            .find(|(key, _)| wanted(key))
            .map(|(_, value)| Arc::clone(value))
    }

    /// Holds `value` under `key` while there is room and no key `wanted` accepts
    fn hold(
        &self,
        wanted: impl Fn(&K) -> bool,
        key: impl FnOnce() -> K,
        value: impl FnOnce() -> Arc<T>,
    ) {
        if let Ok(mut held) = self.held.write() {
            if held.len() < MAX_HELD && !held.iter().any(|(k, _)| wanted(k)) {
                held.push((key(), value()));
            }
        }
    }
}

/// The programs one join statement compiled, valid for one schema epoch
#[derive(Debug)]
pub struct PreparedJoin {
    pub cached_epoch: u64,
    /// An empty list stands for a clause that leaves nothing to ask
    programs: Shelf<JoinProgramKey, [SharedProgram]>,
    analyses: Shelf<AnalysisKey, JoinAnalysis>,
    /// None stands for a SELECT list each execution projects anew
    outputs: Shelf<OutputKey, Option<JoinOutput>>,
    wheres: Shelf<WhereKey, SharedProgram>,
    /// Whether the ON's and the WHERE's subqueries run as the statement's
    /// blocks; decided on first need
    blocks_where: std::sync::OnceLock<bool>,
    join_blocks: RwLock<Option<Arc<JoinBlocks>>>,
}

impl PreparedJoin {
    pub fn new(cached_epoch: u64) -> Self {
        Self {
            cached_epoch,
            programs: Shelf::new(),
            analyses: Shelf::new(),
            outputs: Shelf::new(),
            wheres: Shelf::new(),
            blocks_where: std::sync::OnceLock::new(),
            join_blocks: RwLock::new(None),
        }
    }

    /// Whether the WHERE's subqueries run as the statement's blocks, as
    /// `decide` finds once
    pub fn blocks_where(&self, decide: impl FnOnce() -> bool) -> bool {
        *self.blocks_where.get_or_init(decide)
    }

    /// The output held for joined rows of `columns`, or the one `build`
    /// makes; none when the statement's output is not held
    pub(crate) fn output(
        &self,
        columns: &[String],
        sorted: bool,
        build: impl FnOnce() -> Result<Option<JoinOutput>>,
    ) -> Result<Arc<Option<JoinOutput>>> {
        let wanted = |held: &OutputKey| held.sorted == sorted && held.columns == columns;
        if let Some(held) = self.outputs.find(wanted) {
            return Ok(held);
        }
        let output = Arc::new(build()?);
        self.outputs.hold(
            wanted,
            || OutputKey {
                columns: columns.to_vec(),
                sorted,
            },
            || Arc::clone(&output),
        );
        Ok(output)
    }

    /// The analysis of the join of `left_cols` and `right_cols` that leaves
    /// the key pairs of `kept_out` in the residual, or the one `analyze` makes
    pub fn analysis(
        &self,
        left_cols: &[String],
        right_cols: &[String],
        kept_out: &[(usize, usize)],
        analyze: impl FnOnce() -> JoinAnalysis,
    ) -> Arc<JoinAnalysis> {
        let wanted = |held: &AnalysisKey| {
            held.left_cols == left_cols
                && held.right_cols == right_cols
                && held.kept_out == kept_out
        };
        if let Some(held) = self.analyses.find(wanted) {
            return held;
        }
        let analysis = Arc::new(analyze());
        self.analyses.hold(
            wanted,
            || AnalysisKey {
                left_cols: left_cols.to_vec(),
                right_cols: right_cols.to_vec(),
                kept_out: kept_out.to_vec(),
            },
            || Arc::clone(&analysis),
        );
        analysis
    }

    /// `read` over the programs held under a key `wanted` accepts, or over
    /// the ones `compile` builds, which are then held under `key`; a lookup
    /// copies no key
    pub fn programs<R>(
        &self,
        wanted: impl Fn(&JoinProgramKey) -> bool,
        key: impl FnOnce() -> JoinProgramKey,
        compile: impl FnOnce() -> Result<BuiltPrograms>,
        read: impl FnOnce(&[SharedProgram]) -> R,
    ) -> Result<R> {
        if let Some(held) = self.programs.find(&wanted) {
            return Ok(read(&held));
        }
        let built = compile()?;
        self.programs
            .hold(&wanted, key, || built.iter().cloned().collect());
        Ok(read(&built))
    }

    /// The WHERE program held for `site` over `columns`, or the one
    /// `compile` builds
    pub(crate) fn where_program(
        &self,
        site: WhereSite,
        columns: &[String],
        compile: impl FnOnce() -> Result<SharedProgram>,
    ) -> Result<SharedProgram> {
        let wanted = |held: &WhereKey| held.site == site && held.columns == columns;
        if let Some(held) = self.wheres.find(wanted) {
            return Ok(SharedProgram::clone(&held));
        }
        let built = compile()?;
        self.wheres.hold(
            wanted,
            || WhereKey {
                site,
                columns: columns.to_vec(),
            },
            || Arc::new(SharedProgram::clone(&built)),
        );
        Ok(built)
    }

    /// The statement's blocks, or the ones `build` prepares, then held;
    /// none when `build` prepares none
    pub(crate) fn join_blocks(
        &self,
        build: impl FnOnce() -> Result<Option<JoinBlocks>>,
    ) -> Result<Option<Arc<JoinBlocks>>> {
        if let Some(held) = self.join_blocks.read().ok().and_then(|held| held.clone()) {
            return Ok(Some(held));
        }
        let Some(built) = build()? else {
            return Ok(None);
        };
        let built = Arc::new(built);
        if let Ok(mut held) = self.join_blocks.write() {
            held.get_or_insert_with(|| Arc::clone(&built));
        }
        Ok(Some(built))
    }
}
