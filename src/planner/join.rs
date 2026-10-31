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

//! How a join runs, decided once from its statement, its schemas and its
//! indexes; what only a run knows picks among the arms the plan names

use std::sync::Arc;

use crate::storage::traits::Index;

/// What the planner decided for one join. A run reads it and chooses only
/// among its arms, by the LIMIT it binds and the sizes it finds
#[derive(Debug)]
pub(crate) struct JoinPlan {
    /// A GROUP BY + LIMIT over the left side's primary key: the left and
    /// the right key columns the left chunk narrows the right side by
    pub reduction: Option<(String, String)>,
    /// Whether the grouped index join runs a reduced statement
    pub grouped_index: bool,
    /// The probe each side offers as the inner side of an index join, the
    /// right side's first
    pub probes: [Option<IndexProbe>; 2],
    /// A small LIMIT streams the join through a hash table: whether it
    /// builds on the left side
    pub limit_hash: Option<bool>,
}

/// The inner side of an index join: its table, how a key finds its rows,
/// and the key columns
#[derive(Debug, Clone)]
pub(crate) struct IndexProbe {
    pub table: String,
    pub index: ProbeIndex,
    /// Unqualified, of the inner table
    pub inner_col: String,
    /// As the ON names it, of the outer side
    pub outer_col: String,
}

/// How a key finds the inner table's rows
#[derive(Clone)]
pub(crate) enum ProbeIndex {
    PrimaryKey,
    Secondary(Arc<dyn Index>),
    /// The table's candidates for one key, which a run takes only when the
    /// table's size and the join's limit say a probe pays
    Equality {
        outer_is_sure: bool,
    },
}

impl std::fmt::Debug for ProbeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbeIndex::PrimaryKey => f.write_str("PrimaryKey"),
            ProbeIndex::Secondary(index) => write!(f, "Secondary({})", index.name()),
            ProbeIndex::Equality { outer_is_sure } => {
                write!(f, "Equality {{ outer_is_sure: {outer_is_sure} }}")
            }
        }
    }
}

/// How two read sides are joined
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PairMethod {
    NestedLoop,
    Hash { build_left: bool },
}

/// The method for sides of `left_rows` and `right_rows` rows: a nested loop
/// without an equality key, with a side empty, or where comparing every
/// pair costs less than a hash table; else a hash table on the side the
/// join type keeps whole, or on the smaller side of an inner join
pub(crate) fn pair_method(
    left_rows: usize,
    right_rows: usize,
    has_keys: bool,
    join_type: &str,
) -> PairMethod {
    const NESTED_LOOP_MAX: usize = 200;
    const HASH_JOIN_MIN_BENEFIT: usize = 50;
    if !has_keys {
        return PairMethod::NestedLoop;
    }
    let small = left_rows <= NESTED_LOOP_MAX && right_rows <= NESTED_LOOP_MAX;
    if !small {
        let product = left_rows.saturating_mul(right_rows);
        if left_rows == 0
            || right_rows == 0
            || product < left_rows + right_rows + HASH_JOIN_MIN_BENEFIT
        {
            return PairMethod::NestedLoop;
        }
    }
    let build_left = if join_type.contains("LEFT") || join_type.contains("FULL") {
        false
    } else if join_type.contains("RIGHT") {
        true
    } else {
        left_rows <= right_rows
    };
    PairMethod::Hash { build_left }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_method_follows_keys_sizes_and_join_type() {
        assert_eq!(pair_method(10, 10, false, "INNER"), PairMethod::NestedLoop);
        assert_eq!(
            pair_method(5, 100, true, "INNER"),
            PairMethod::Hash { build_left: true }
        );
        assert_eq!(
            pair_method(100, 5, true, "INNER"),
            PairMethod::Hash { build_left: false }
        );
        assert_eq!(
            pair_method(5, 100, true, "LEFT"),
            PairMethod::Hash { build_left: false }
        );
        assert_eq!(
            pair_method(100, 5, true, "RIGHT"),
            PairMethod::Hash { build_left: true }
        );
        assert_eq!(pair_method(0, 5000, true, "INNER"), PairMethod::NestedLoop);
        assert_eq!(pair_method(1, 5000, true, "INNER"), PairMethod::NestedLoop);
        assert_eq!(
            pair_method(2, 5000, true, "INNER"),
            PairMethod::Hash { build_left: true }
        );
    }
}
