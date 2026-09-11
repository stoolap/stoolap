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

use stoolap::core::{Error, Row, Value};
use stoolap::storage::mvcc::persistence::PersistenceManager;
use stoolap::storage::mvcc::version_store::RowVersion;
use stoolap::storage::mvcc::wal_manager::WALOperationType;
use stoolap::{PersistenceConfig, SyncMode};

#[test]
fn dml_receipts_match_interleaved_replay_entries() {
    let dir = tempfile::tempdir().unwrap();
    let config = PersistenceConfig {
        enabled: true,
        sync_mode: SyncMode::Full,
        ..Default::default()
    };
    let pm = PersistenceManager::new(Some(dir.path()), &config).unwrap();
    pm.start().unwrap();
    pm.record_ddl_operation("t", WALOperationType::CreateTable, b"schema")
        .unwrap();
    let mut receipts = Vec::new();
    for (txn_id, row_id, op) in [
        (10, 1, WALOperationType::Insert),
        (20, 2, WALOperationType::Insert),
        (10, 3, WALOperationType::Delete),
    ] {
        let version = RowVersion::new(txn_id, Row::from_values(vec![Value::Integer(row_id)]));
        let receipt = pm
            .record_dml_operation(txn_id, "t", row_id, op, &version)
            .unwrap();
        receipts.push((txn_id, row_id, format!("{receipt:?}")));
    }
    pm.record_commit(20).unwrap();
    pm.record_commit(10).unwrap();
    let end_lsn = pm.current_lsn();
    pm.stop().unwrap();

    let pm = PersistenceManager::new(Some(dir.path()), &config).unwrap();
    let mut replayed = Vec::new();
    let mut commit_lsn = 0;
    pm.replay_two_phase(0, |entry| {
        if entry.is_commit_marker() && entry.txn_id == 10 {
            commit_lsn = entry.lsn;
        } else if matches!(
            entry.operation,
            WALOperationType::Insert | WALOperationType::Delete
        ) {
            replayed.push((entry.txn_id, entry.row_id, entry.lsn));
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(replayed.len(), receipts.len());
    for ((txn_id, row_id, receipt), &(replayed_txn, replayed_row, lsn)) in
        receipts.iter().zip(&replayed)
    {
        assert_eq!((*txn_id, *row_id), (replayed_txn, replayed_row));
        assert_eq!(
            receipt,
            &format!("Some({lsn})"),
            "receipt must identify the DML frame"
        );
    }
    let first_dml = replayed[0].2;
    assert!(first_dml < commit_lsn);
    assert!(first_dml < end_lsn);
    pm.stop().unwrap();
}

#[test]
fn disabled_persistence_has_no_dml_receipt() {
    let pm = PersistenceManager::new(None, &PersistenceConfig::default()).unwrap();
    let version = RowVersion::new(1, Row::from_values(vec![Value::Integer(1)]));
    let receipt = pm
        .record_dml_operation(1, "t", 1, WALOperationType::Insert, &version)
        .unwrap();
    assert_eq!(format!("{receipt:?}"), "None");
}

#[test]
fn closed_wal_returns_error_instead_of_dml_receipt() {
    let dir = tempfile::tempdir().unwrap();
    let pm = PersistenceManager::new(Some(dir.path()), &PersistenceConfig::default()).unwrap();
    pm.start().unwrap();
    pm.stop().unwrap();
    let version = RowVersion::new(1, Row::from_values(vec![Value::Integer(1)]));
    assert!(matches!(
        pm.record_dml_operation(1, "t", 1, WALOperationType::Insert, &version),
        Err(Error::WalNotRunning)
    ));
}
