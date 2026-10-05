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

//! An index rebound to another column name shares its data with the object
//! it came from: a write through the old object is seen through the new one,
//! cached minimums and maximums included, and the binding is the new one.

use std::sync::Arc;

use stoolap::core::{DataType, Value};
use stoolap::storage::index::{BTreeIndex, BitmapIndex, HashIndex, MultiColumnIndex, PkIndex};
use stoolap::storage::traits::Index;

fn int(v: i64) -> Value {
    Value::Integer(v)
}

/// Writes through `old` after rebinding and reads through the rebound one
fn assert_shares_data(old: Arc<dyn Index>, width: usize) {
    let key = |v: i64| vec![int(v); width];
    old.add(&key(10), 1, 0).unwrap();
    let names: Vec<String> = (0..width).map(|i| format!("renamed{i}")).collect();
    let rebound = old.rebound(&names, old.column_ids());
    assert_eq!(rebound.column_names(), names.as_slice());
    assert_eq!(rebound.column_ids(), old.column_ids());
    assert_eq!(rebound.name(), old.name());
    old.add(&key(20), 2, 0).unwrap();
    old.remove(&key(10), 1, 0).unwrap();
    assert_eq!(
        &*rebound.get_row_ids_equal(&key(20)),
        &[2],
        "{}",
        old.name()
    );
    assert!(
        rebound.get_row_ids_equal(&key(10)).is_empty(),
        "{}",
        old.name()
    );
    rebound.add(&key(30), 3, 0).unwrap();
    assert_eq!(&*old.get_row_ids_equal(&key(30)), &[3], "{}", old.name());
}

#[test]
fn every_index_kind_shares_its_data_with_its_rebound_object() {
    let names = vec!["a".to_string()];
    assert_shares_data(
        Arc::new(BTreeIndex::new(
            "ib".into(),
            "t".into(),
            1,
            "a".into(),
            DataType::Integer,
            false,
            0,
        )),
        1,
    );
    assert_shares_data(
        Arc::new(HashIndex::new(
            "ih".into(),
            "t".into(),
            names.clone(),
            vec![1],
            vec![DataType::Integer],
            false,
            0,
        )),
        1,
    );
    assert_shares_data(
        Arc::new(BitmapIndex::new(
            "im".into(),
            "t".into(),
            names,
            vec![1],
            vec![DataType::Integer],
            false,
            0,
        )),
        1,
    );
    assert_shares_data(
        Arc::new(MultiColumnIndex::new(
            "ic".into(),
            "t".into(),
            vec!["a".into(), "b".into()],
            vec![1, 2],
            vec![DataType::Integer, DataType::Integer],
            false,
            0,
        )),
        2,
    );
}

#[test]
fn a_rebound_btree_sees_the_extremes_the_old_object_wrote() {
    let old: Arc<dyn Index> = Arc::new(BTreeIndex::new(
        "ib".into(),
        "t".into(),
        1,
        "a".into(),
        DataType::Integer,
        false,
        0,
    ));
    old.add(&[int(10)], 1, 0).unwrap();
    let rebound = old.rebound(&["b".to_string()], &[1]);
    assert_eq!(rebound.get_max_value(), Some(int(10)));
    old.add(&[int(50)], 2, 0).unwrap();
    old.add(&[int(-5)], 3, 0).unwrap();
    assert_eq!(rebound.get_max_value(), Some(int(50)));
    assert_eq!(rebound.get_min_value(), Some(int(-5)));
    old.remove(&[int(50)], 2, 0).unwrap();
    assert_eq!(rebound.get_max_value(), Some(int(10)));
}

#[test]
fn a_rebound_primary_key_shares_its_rows() {
    let old: Arc<dyn Index> = Arc::new(PkIndex::new("pk".into(), "t".into(), 0, "id".into()));
    old.add(&[int(7)], 7, 0).unwrap();
    let rebound = old.rebound(&["key".to_string()], &[0]);
    old.add(&[int(8)], 8, 0).unwrap();
    assert_eq!(rebound.column_names(), ["key".to_string()]);
    assert_eq!(&*rebound.get_row_ids_equal(&[int(8)]), &[8]);
    assert_eq!(&*rebound.get_row_ids_equal(&[int(7)]), &[7]);
}

#[test]
fn a_rebound_vector_index_follows_the_unique_rule_of_its_graph() {
    use stoolap::storage::index::hnsw::{HnswDistanceMetric, HnswIndex};
    let mut old = HnswIndex::new(
        "iv".into(),
        "t".into(),
        "v".into(),
        1,
        2,
        8,
        32,
        32,
        HnswDistanceMetric::L2,
    );
    let rebound = old.rebound(&["w".into()], &[1]);
    old.set_unique(true);
    assert!(rebound.is_unique());
    let value = Value::vector(vec![1.0, 2.0]);
    old.add(std::slice::from_ref(&value), 1, 1).unwrap();
    assert!(
        rebound.add(std::slice::from_ref(&value), 2, 2).is_err(),
        "the rebound object keeps the shared graph unique"
    );
}
