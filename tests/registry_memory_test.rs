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

use stoolap::core::IsolationLevel;
use stoolap::storage::mvcc::TransactionRegistry;
use stoolap::Database;

#[test]
fn registry_capacity_survives_shared_owners_and_releases_on_destruction() {
    let db = Database::open(
        "memory://registry_capacity_survives_shared_owners_and_releases_on_destruction",
    )
    .unwrap();
    let bytes = || {
        db.engine()
            .memory_stats()
            .last()
            .unwrap()
            .transaction_registry_bytes
    };
    let baseline = bytes();
    let registry = Arc::new(TransactionRegistry::with_capacity(8));
    let empty = bytes();
    assert!(empty > baseline, "empty maps own their minimum capacity");
    registry.set_global_isolation_level(IsolationLevel::SnapshotIsolation);
    let ids: Vec<_> = (0..2048).map(|_| registry.begin_transaction().0).collect();
    assert!(
        bytes() > empty,
        "active transaction capacity must be charged"
    );
    for id in ids {
        registry.set_transaction_isolation_level(id, IsolationLevel::ReadCommitted);
        registry.commit_transaction(id);
    }
    let committed = bytes();
    assert!(
        committed > empty,
        "snapshot and isolation maps retain capacity"
    );
    registry.run_gc();
    let collected = bytes();
    assert!(collected < committed);
    assert!(collected >= empty);
    let retained = Arc::clone(&registry);
    drop(registry);
    assert_eq!(bytes(), collected, "the final owner still retains the maps");
    drop(retained);
    assert_eq!(bytes(), baseline, "destroying maps releases their charge");
}
