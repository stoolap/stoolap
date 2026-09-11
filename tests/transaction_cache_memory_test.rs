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

use stoolap::storage::mvcc::TransactionVersionStore;
use stoolap::storage::Engine;
use stoolap::Database;

#[test]
fn table_handles_keep_transaction_store_objects_charged_after_rollback() {
    let db = Database::open(
        "memory://table_handles_keep_transaction_store_objects_charged_after_rollback",
    )
    .unwrap();
    for table in 0..8 {
        db.execute(
            &format!(
                "CREATE TABLE table_with_a_heap_allocated_name_{table} (id INTEGER PRIMARY KEY)"
            ),
            (),
        )
        .unwrap();
    }
    let bytes = || {
        db.engine()
            .memory_stats()
            .last()
            .unwrap()
            .hot_metadata_bytes
    };
    let baseline = bytes();
    let mut transaction = db.engine().begin_transaction().unwrap();
    let tables: Vec<_> = (0..8)
        .map(|table| {
            transaction
                .get_table(&format!("table_with_a_heap_allocated_name_{table}"))
                .unwrap()
        })
        .collect();
    let active = bytes();
    transaction.rollback().unwrap();
    drop(transaction);
    let retained = bytes();
    assert!(
        active > retained,
        "cache names and spilled entries were released"
    );
    assert!(
        retained
            >= baseline + 8 * std::mem::size_of::<std::sync::RwLock<TransactionVersionStore>>(),
        "retained table handles still own their transaction stores",
    );
    drop(tables);
    assert_eq!(bytes(), baseline, "last table owners release store objects");
}
