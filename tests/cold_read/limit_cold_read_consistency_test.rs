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

#![cfg(feature = "test-failpoints")]

use std::collections::BTreeSet;
use stoolap::core::RowVec;
use stoolap::storage::expression::{ComparisonExpr, Expression};
use stoolap::storage::traits::{Engine, Table};
use stoolap::Database;

fn fixture() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=3",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    for batch in 0..3 {
        let values = (batch * 1000 + 1..=batch * 1000 + 1000)
            .map(|id| format!("({id},{},{})", id % 7, id * 10))
            .collect::<Vec<_>>()
            .join(",");
        db.execute(&format!("INSERT INTO t VALUES {values}"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(db.engine().volume_stats().len(), 3);
    db.execute("DELETE FROM t WHERE id <= 10", ()).unwrap();
    (dir, db)
}

fn volume_files(dir: &std::path::Path) -> BTreeSet<std::path::PathBuf> {
    std::fs::read_dir(dir.join("volumes/t"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "vol"))
        .collect()
}

fn collect(
    table: &dyn Table,
    unordered: bool,
    filter: Option<&dyn Expression>,
    limit: usize,
    offset: usize,
) -> RowVec {
    if unordered {
        table.collect_rows_with_limit_unordered(filter, limit, offset)
    } else {
        table.collect_rows_with_limit(filter, limit, offset)
    }
    .unwrap()
}

fn assert_compaction_view(unordered: bool, cold: bool, drop_key: bool) {
    let (dir, db) = fixture();
    if drop_key {
        db.execute("ALTER TABLE t DROP COLUMN k", ()).unwrap();
    }
    if cold {
        assert_eq!(db.engine().cold_volumes_for_test("t"), (3, 3));
    }
    let files = volume_files(dir.path());
    assert_eq!(files.len(), 3);
    let held_files = files.clone();
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        other.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(other.engine().volume_stats().len(), 1);
        for path in &held_files {
            assert!(path.exists(), "captured file must survive: {path:?}");
        }
    });
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = collect(&*table, unordered, None, 4000, 0);
    assert_eq!(rows.len(), 2990);
    let width = if drop_key { 2 } else { 3 };
    let mut actual = Vec::new();
    for (id, row) in rows.iter() {
        assert_eq!(row.len(), width);
        actual.push((*id, row[width - 1].as_int64().unwrap()));
    }
    actual.sort_unstable();
    assert_eq!(
        actual,
        (11..=3000).map(|id| (id, id * 10)).collect::<Vec<_>>()
    );
    tx.rollback().unwrap();
    for path in files {
        assert!(!path.exists(), "retired file must be released: {path:?}");
    }
}

#[test]
fn resident_limit_keeps_deleted_rows_hidden_during_compaction() {
    assert_compaction_view(false, false, false);
}

#[test]
fn ordered_limit_reads_retired_metadata_only_volumes() {
    assert_compaction_view(false, true, false);
}

#[test]
fn unordered_limit_reads_retired_metadata_only_volumes() {
    assert_compaction_view(true, true, false);
}

#[test]
fn ordered_limit_keeps_the_retired_volumes_column_mapping() {
    assert_compaction_view(false, false, true);
}

#[test]
fn unordered_limit_keeps_the_retired_volumes_column_mapping() {
    assert_compaction_view(true, false, true);
}

#[test]
fn limit_rejects_a_mapping_from_a_newer_schema() {
    for unordered in [false, true] {
        let (_dir, db) = fixture();
        let mut tx = db.engine().begin_transaction().unwrap();
        let table = tx.get_table("t").unwrap();
        db.clone()
            .execute("ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 4", ())
            .unwrap();
        let result = if unordered {
            table.collect_rows_with_limit_unordered(None, 1, 0)
        } else {
            table.collect_rows_with_limit(None, 1, 0)
        };
        assert!(matches!(result, Err(stoolap::Error::SchemaChanged { .. })));
        tx.rollback().unwrap();
    }
}

#[test]
fn limit_and_offset_count_filtered_cold_and_hot_rows() {
    for unordered in [false, true] {
        let (_dir, db) = fixture();
        db.execute("INSERT INTO t VALUES (10001,4,100010)", ())
            .unwrap();
        let mut filter = ComparisonExpr::gte("k", stoolap::Value::Integer(3));
        let store = db.engine().get_version_store("t").unwrap();
        filter.prepare_for_schema(&store.schema());
        let mut expected = Vec::new();
        if unordered {
            expected.push(10001);
            for batch in (0..3).rev() {
                expected.extend(
                    (batch * 1000 + 1..=batch * 1000 + 1000).filter(|id| *id > 10 && id % 7 >= 3),
                );
            }
        } else {
            expected.extend((11..=3000).filter(|id| id % 7 >= 3));
            expected.push(10001);
        }
        let mut tx = db.engine().begin_transaction().unwrap();
        let table = tx.get_table("t").unwrap();
        for offset in [0, 4, expected.len() - 5, expected.len() + 2] {
            let rows = collect(&*table, unordered, Some(&filter), 7, offset);
            let actual: Vec<_> = rows.iter().map(|(id, _)| *id).collect();
            assert_eq!(
                actual,
                expected
                    .iter()
                    .copied()
                    .skip(offset)
                    .take(7)
                    .collect::<Vec<_>>(),
                "unordered={unordered}, offset={offset}"
            );
        }
        tx.rollback().unwrap();
    }
}

#[test]
fn a_small_limit_reloads_only_the_volume_it_needs() {
    for unordered in [false, true] {
        let (_dir, db) = fixture();
        assert_eq!(db.engine().cold_volumes_for_test("t"), (3, 3));
        let mut tx = db.engine().begin_transaction().unwrap();
        let table = tx.get_table("t").unwrap();
        let rows = collect(&*table, unordered, None, 1, 0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, if unordered { 2001 } else { 11 });
        assert_eq!(
            db.engine()
                .volume_stats()
                .iter()
                .filter(|v| v.2 == "cold")
                .count(),
            2
        );
        tx.rollback().unwrap();
    }
}
