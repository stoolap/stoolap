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
pub mod id_list;
pub mod multi_column;
pub mod pk;

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

/// Calls `copy` for each found entry in the members' order, skipping one an
/// earlier member found: members an index reads as one key find the same
/// entry, named by the pair's first field (its address under the read lock)
pub(crate) fn each_entry_once<K: Ord + Copy, T: Copy>(found: &[(K, T)], mut copy: impl FnMut(T)) {
    let mut order: smallvec::SmallVec<[u32; 16]> = (0..found.len() as u32).collect();
    order.sort_unstable_by_key(|&i| (found[i as usize].0, i));
    let mut repeated: smallvec::SmallVec<[bool; 16]> = smallvec::smallvec![false; found.len()];
    for pair in order.windows(2) {
        if found[pair[0] as usize].0 == found[pair[1] as usize].0 {
            repeated[pair[1] as usize] = true;
        }
    }
    for (&(_, entry), repeated) in found.iter().zip(repeated) {
        if !repeated {
            copy(entry);
        }
    }
}

#[cfg(test)]
mod in_list_tests {
    use super::seal_removal_tests::indexes;
    use crate::core::Value;

    #[test]
    fn an_in_list_appends_each_row_once_after_the_buffer() {
        let members: Vec<Value> = (0..24)
            .map(|i| Value::Integer([2, 0, 5, 9][i % 4] + if i == 23 { 50 } else { 0 }))
            .collect();
        for (name, index) in indexes() {
            let key = |id: i64| if name == "pk" { id } else { id % 10 };
            for id in 1..=60 {
                let values = match name {
                    "multi_column" => vec![Value::Integer(key(id)), Value::Integer(id)],
                    _ => vec![Value::Integer(key(id))],
                };
                index.add(&values, id, id).unwrap();
            }
            let mut buffer = vec![-1, -1];
            index.get_row_ids_in_into(&members, &mut buffer);
            let mut rows = buffer.split_off(2);
            assert_eq!(buffer, vec![-1, -1], "{name}: the buffer's rows are kept");
            rows.sort_unstable();
            let mut want: Vec<i64> = (1..=60)
                .filter(|&id| members.iter().any(|m| *m == Value::Integer(key(id))))
                .collect();
            want.sort_unstable();
            assert_eq!(rows, want, "{name}: each matching row once");
        }
    }
}

#[cfg(test)]
mod seal_removal_tests {
    use super::*;
    use crate::common::i64_map::slot_visits;
    use crate::core::{DataType, Value};
    use crate::storage::traits::Index;

    /// Past the PK index's bitset, so its rows sit in the overflow set
    const FIRST: i64 = 10_000_000;
    const ROWS: i64 = 200_000;

    pub(super) fn indexes() -> Vec<(&'static str, Box<dyn Index>)> {
        let cols = || (vec!["k".to_string()], vec![1], vec![DataType::Integer]);
        let (hn, hi, ht) = cols();
        let (bn, bi, bt) = cols();
        vec![
            (
                "btree",
                Box::new(BTreeIndex::new(
                    "i".into(),
                    "t".into(),
                    1,
                    "k".into(),
                    DataType::Integer,
                    false,
                    0,
                )),
            ),
            (
                "hash",
                Box::new(HashIndex::new("i".into(), "t".into(), hn, hi, ht, false, 0)),
            ),
            (
                "bitmap",
                Box::new(BitmapIndex::new(
                    "i".into(),
                    "t".into(),
                    bn,
                    bi,
                    bt,
                    false,
                    0,
                )),
            ),
            (
                "multi_column",
                Box::new(MultiColumnIndex::new(
                    "i".into(),
                    "t".into(),
                    vec!["k".into(), "id".into()],
                    vec![1, 0],
                    vec![DataType::Integer, DataType::Integer],
                    false,
                    0,
                )),
            ),
            (
                "pk",
                Box::new(PkIndex::new("i".into(), "t".into(), 0, "id".into())),
            ),
        ]
    }

    fn key(name: &str, id: i64) -> Vec<Value> {
        match name {
            "pk" => vec![Value::Integer(id)],
            "multi_column" => vec![Value::Integer(id % 100), Value::Integer(id)],
            _ => vec![Value::Integer(id % 100)],
        }
    }

    #[test]
    fn a_seal_removal_keeps_capacity_and_releases_an_emptied_map() {
        for (name, index) in indexes() {
            for id in FIRST..FIRST + ROWS {
                index.add(&key(name, id), id, id).unwrap();
            }
            let ids: Vec<i64> = (FIRST..FIRST + ROWS).collect();
            let mut buffers = index.removal_buffers(2_000).unwrap();
            let before = slot_visits();
            let mut next = 0usize;
            for left in [4_096usize, 1, 0] {
                let upto = ids.len() - left;
                for chunk in ids[next..upto].chunks(2_000) {
                    index
                        .remove_batch_ids(chunk, &mut buffers)
                        .unwrap()
                        .unwrap();
                }
                next = upto;
                assert_eq!(slot_visits() - before, 0, "{name}: {left} rows left");
            }
            if let Some(set) = buffers.downcast_ref::<crate::common::I64Set>() {
                assert!(
                    set.capacity() >= ROWS as usize,
                    "pk: the buffers hold the old set"
                );
            } else {
                drop(buffers);
                assert!(
                    slot_visits() - before > 0,
                    "{name}: the buffers hold the old map, walked when dropped"
                );
            }
            let id = FIRST + ROWS;
            index.add(&key(name, id), id, id).unwrap();
            assert_eq!(
                index
                    .get_row_ids_equal(&key(name, id))
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![id],
                "{name} works after"
            );
        }
    }

    #[test]
    fn a_delete_removal_keeps_the_old_shrink_policy() {
        let index = BTreeIndex::new(
            "i".into(),
            "t".into(),
            1,
            "k".into(),
            DataType::Integer,
            false,
            0,
        );
        let rows: Vec<(i64, Vec<Value>)> = (0..ROWS)
            .map(|id| (id, vec![Value::Integer(id % 100)]))
            .collect();
        for (id, values) in &rows {
            index.add(values, *id, *id).unwrap();
        }
        let before = slot_visits();
        let entries: Vec<(i64, &[Value])> = rows[1..]
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        index.remove_batch_slice(&entries).unwrap();
        assert!(
            slot_visits() - before > 0,
            "a DELETE still shrinks the row map"
        );
    }
}
