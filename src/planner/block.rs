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

//! How a subquery block answers, decided once when its statement is
//! prepared; what only a run knows picks between the arms a node names

use crate::storage::traits::Table;

/// How often a block computes its answer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockRuns {
    /// It reads no enclosing scope: once per run
    Once,
    /// It reads only values of its parent block's row: once per parent row
    PerParentRow,
    /// For each call
    PerCall,
}

/// How a block keyed on its parent's values answers a call
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyedRead {
    /// From the keyed set it builds once per run
    Set,
    /// By a probe of the index on its key where [`probe_cheaper`] holds for
    /// the run, else from the keyed set: its WHERE is the keys alone
    ProbeOrSet,
}

/// How a grouped block reads its rows for a call it answers itself
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CallRead {
    /// Its pushed conditions or a scan
    Rows,
    /// Every call counts rows and its WHERE is one equality: the count the
    /// index on its column holds for the key, where it answers, else `Rows`
    CountByIndex,
}

/// Whether a LIMIT is small enough that probing per outer row stops early
pub(crate) fn small_probe_limit(limit: Option<i64>) -> bool {
    limit.is_some_and(|limit| (1..=500).contains(&limit))
}

/// Whether an EXISTS on `column` of `table` probes it per outer row more
/// cheaply than it builds a set: for a small LIMIT or few outer rows,
/// since a probe per outer row costs about what a set pays for sixteen
/// inner rows, and only when an index answers the probe
pub(crate) fn probe_cheaper(
    table: &dyn Table,
    column: &str,
    outer_limit: Option<i64>,
    outer_rows: Option<usize>,
) -> bool {
    let few_outer =
        outer_rows.is_some_and(|rows| rows.saturating_mul(16) <= table.row_count_hint());
    (small_probe_limit(outer_limit) || few_outer) && exists_probes(table, column)
}

fn exists_probes(table: &dyn Table, column: &str) -> bool {
    // The table vouches for its indexes; one with local changes or an
    // older snapshot refuses every probe
    if table.index_view_epoch().is_none() {
        return false;
    }
    // An index the table keeps whole
    if table.lookup_index_on_column(column).is_some() {
        return true;
    }
    // Sealed rows answer through side files
    let Some(at) = table
        .schema()
        .columns
        .iter()
        .position(|c| c.name_lower.eq_ignore_ascii_case(column))
    else {
        return false;
    };
    // A key probe asks an ordered i64 entry, or a single-column TEXT one
    table.side_index_defs().iter().any(|def| {
        use crate::storage::volume::side_index::{Domain, KeyKind};
        match def.components.as_slice() {
            [only] if only.column as usize == at => {
                def.kind == KeyKind::OrderedI64 || only.domain == Domain::Text
            }
            _ => false,
        }
    })
}
