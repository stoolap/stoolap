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

//! Index implementations for Stoolap
//!
//! This module provides all index structures used by the storage engine:
//!
//! - [`BTreeIndex`] - B-tree index for range queries and sorted access
//! - [`HashIndex`] - Hash index for O(1) equality lookups
//! - [`BitmapIndex`] - Bitmap index for low-cardinality columns
//! - [`HnswIndex`] - HNSW index for approximate nearest neighbor search
//! - [`MultiColumnIndex`] - Composite index for multi-column queries
//! - [`PkIndex`] - Primary key index (virtual, auto-created)

pub mod bitmap;
pub mod btree;
pub mod hash;
pub mod hnsw;
pub mod multi_column;
pub mod pk;

/// Removes the sorted `ids` from the sorted `rows`, touching only the rows
/// from the first removed id onwards: taking the newest rows off a large
/// group stays cheap, taking the whole group is one pass.
pub(crate) fn subtract_sorted(rows: &mut crate::common::CompactVec<i64>, ids: &[i64]) {
    let Some(&first) = ids.first() else {
        return;
    };
    let start = rows.binary_search(&first).unwrap_or_else(|pos| pos);
    let slice = rows.as_mut_slice();
    let mut keep = start;
    for read in start..slice.len() {
        if ids.binary_search(&slice[read]).is_err() {
            slice[keep] = slice[read];
            keep += 1;
        }
    }
    rows.truncate(keep);
}

// Re-export main types
pub use bitmap::BitmapIndex;
pub use btree::{
    intersect_multiple_sorted_ids, intersect_sorted_ids, union_multiple_sorted_ids,
    union_sorted_ids, BTreeIndex,
};
pub use hash::HashIndex;
pub use hnsw::{
    default_ef_construction, default_ef_search, default_m_for_dims, HnswDistanceMetric, HnswIndex,
};
pub use multi_column::{CompositeKey, MultiColumnIndex};
pub use pk::PkIndex;
