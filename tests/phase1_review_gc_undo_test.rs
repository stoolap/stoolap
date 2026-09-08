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

use std::sync::Arc;
use stoolap::core::{DataType, Row, SchemaBuilder};
use stoolap::storage::mvcc::{
    registry::TransactionRegistry,
    version_store::{TransactionVersionStore, VersionStore},
};

#[test]
fn failed_delete_publication_survives_concurrent_vacuum() {
    let registry = Arc::new(TransactionRegistry::new());
    let schema = SchemaBuilder::new("gc_undo_review")
        .column("id", DataType::Integer, false, true)
        .build();
    let parent = Arc::new(VersionStore::with_visibility_checker(
        "gc_undo_review",
        schema,
        registry.clone(),
    ));
    let (seed_id, _) = registry.begin_transaction();
    let mut seed = TransactionVersionStore::new(parent.clone(), seed_id);
    seed.put(1, Row::from(vec![1.into()]), false).unwrap();
    seed.commit().unwrap();
    registry.commit_transaction(seed_id);
    let (delete_id, _) = registry.begin_transaction();
    let mut deletion = TransactionVersionStore::new(parent.clone(), delete_id);
    deletion.put(1, Row::from(vec![1.into()]), true).unwrap();
    registry.start_commit(delete_id);
    deletion.prepare_publication().unwrap();
    deletion.apply_prepared_publication().unwrap();
    registry.abort_transaction(delete_id);
    std::thread::sleep(std::time::Duration::from_millis(1));
    // VACUUM/zero-retention background cleanup races between abort and undo.
    assert_eq!(parent.cleanup_deleted_rows(std::time::Duration::ZERO), 0);
    assert_eq!(
        parent.cleanup_old_previous_versions_with_retention(std::time::Duration::ZERO),
        0
    );
    deletion.finish_publication(false).unwrap();
    let (viewer_id, _) = registry.begin_transaction();
    assert!(
        parent.get_visible_version(1, viewer_id).is_some(),
        "vacuum removed the claimed published delete, so failed-commit undo lost the original row"
    );
    assert_eq!(parent.committed_row_count(), 1);
}
