// Copyright 2026 Stoolap Contributors
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

use std::sync::Arc;

use stoolap::core::{DataType, Schema, SchemaColumn};
use stoolap::storage::index::IndexMemory;
use stoolap::storage::mvcc::MVCCEngine;

#[test]
fn metadata_charges_survive_strong_owners_until_weak_backing_is_released() {
    let observer = MVCCEngine::in_memory();
    let bytes = || observer.memory_stats().last().unwrap().hot_metadata_bytes;
    let baseline = bytes();
    let target = MVCCEngine::in_memory();
    target.open_engine().unwrap();
    drop(
        target
            .create_table(Schema::new(
                "metadata_owners",
                vec![SchemaColumn::primary_key(0, "id", DataType::Integer)],
            ))
            .unwrap(),
    );
    let store = target.get_version_store("metadata_owners").unwrap();
    let index = store.get_all_indexes().pop().unwrap();
    let account = Arc::clone(index.memory_account().unwrap());
    drop(store.remove_index(index.name()));
    drop(index);
    let strong = bytes();
    drop(account);
    let weak = bytes();
    let object_bytes = strong - weak;
    assert!(object_bytes >= std::mem::size_of::<IndexMemory>());
    drop(target.memory_stats());
    assert_eq!(
        weak - bytes(),
        object_bytes,
        "the engine Weak keeps the same allocation envelope charged"
    );

    target.drop_table_internal("metadata_owners").unwrap();
    drop(store);
    let weak_table = bytes();
    drop(target.memory_stats());
    assert!(
        bytes() < weak_table,
        "the dead table account releases its Weak backing on registry pruning"
    );
    drop(target);
    assert_eq!(bytes(), baseline);
}
