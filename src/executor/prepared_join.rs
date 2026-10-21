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

//! The programs a join statement compiles, held with its plan: each one by
//! the operator's choice it was compiled for and the columns it reads

use std::sync::RwLock;

use smallvec::SmallVec;

use crate::core::{Error, Result};

use super::expression::SharedProgram;

/// The programs held under one key, read without an allocation for two
pub type HeldPrograms = SmallVec<[SharedProgram; 2]>;

/// What a held program was compiled for: the part of the ON clause an
/// operator asks of each pair, by the keys it answers itself, or the WHERE
/// asked of each joined row
#[derive(Debug)]
pub enum JoinProgramKey {
    /// The ON clause without the equality an index probe answers, and the
    /// WHERE's filter on the inner table, over an (outer, inner) pair
    IndexResidual {
        swapped: bool,
        outer_col: String,
        inner_col: String,
        outer_cols: Vec<String>,
        inner_cols: Vec<String>,
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
    /// The WHERE asked of each joined row, by where it is asked
    Where {
        site: WhereSite,
        columns: Vec<String>,
    },
}

/// Where a join asks its WHERE of the joined rows
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhereSite {
    /// An index nested loop, per joined row toward its LIMIT
    IndexJoin,
    /// After the hash join or nested loop produced its rows
    AfterJoin,
}

/// More programs than one statement's joins compile are not kept
const MAX_PROGRAMS: usize = 16;

/// The programs one join statement compiled, valid for one schema epoch
#[derive(Debug)]
pub struct PreparedJoin {
    pub cached_epoch: u64,
    /// An empty list stands for a clause that leaves nothing to ask
    programs: RwLock<Vec<(JoinProgramKey, HeldPrograms)>>,
}

impl PreparedJoin {
    pub fn new(cached_epoch: u64) -> Self {
        Self {
            cached_epoch,
            programs: RwLock::new(Vec::new()),
        }
    }

    /// The programs held under a key `wanted` accepts, or the ones `compile`
    /// builds, which are then held under `key`; a lookup copies no key
    pub fn programs(
        &self,
        wanted: impl Fn(&JoinProgramKey) -> bool,
        key: impl FnOnce() -> JoinProgramKey,
        compile: impl FnOnce() -> Result<HeldPrograms>,
    ) -> Result<HeldPrograms> {
        if let Ok(programs) = self.programs.read() {
            if let Some((_, program)) = programs.iter().find(|(held, _)| wanted(held)) {
                return Ok(program.clone());
            }
        }
        let program = compile()?;
        if let Ok(mut programs) = self.programs.write() {
            if programs.len() < MAX_PROGRAMS && !programs.iter().any(|(held, _)| wanted(held)) {
                programs.push((key(), program.clone()));
            }
        }
        Ok(program)
    }

    /// The WHERE program held for `site` over `columns`, or the one
    /// `compile` builds
    pub fn where_program(
        &self,
        site: WhereSite,
        columns: &[String],
        compile: impl FnOnce() -> Result<SharedProgram>,
    ) -> Result<SharedProgram> {
        let programs = self.programs(
            |held| {
                matches!(held, JoinProgramKey::Where { site: s, columns: c }
                    if *s == site && c == columns)
            },
            || JoinProgramKey::Where {
                site,
                columns: columns.to_vec(),
            },
            || Ok(std::iter::once(compile()?).collect()),
        )?;
        programs
            .into_iter()
            .next()
            .ok_or_else(|| Error::internal("a join WHERE held without its program"))
    }
}
