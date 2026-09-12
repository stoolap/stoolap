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

use stoolap::storage::Engine;
use stoolap::{test_failpoints, Database};

#[test]
fn memory_stats_include_retained_version_payloads() {
    let db = Database::open("memory://memory_stats_include_retained_version_payloads").unwrap();
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO items VALUES (1, 10)", ()).unwrap();
    db.execute("UPDATE items SET value = 20 WHERE id = 1", ())
        .unwrap();
    let stats: Vec<_> = db
        .query("PRAGMA MEMORY_STATS", ())
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let table = stats
        .iter()
        .find(|row| row.get::<String>(0).unwrap() == "items")
        .unwrap();
    assert_eq!(table.get::<i64>(2).unwrap(), 48);
    assert_eq!(table.get::<i64>(8).unwrap(), 96);
    assert_eq!(table.get::<i64>(9).unwrap(), 0);
    assert_eq!(table.get::<i64>(10).unwrap(), 0);
    assert!(table.get::<i64>(11).unwrap() > 0);
    assert_eq!(table.get::<i64>(12).unwrap(), 0);
    assert_eq!(table.get::<i64>(13).unwrap(), 0);
    assert!(table.get::<i64>(16).unwrap() > 0);
    db.execute("TRUNCATE TABLE items", ()).unwrap();
    let stats = db.engine().memory_stats();
    let table = stats.iter().find(|row| row.table_name == "items").unwrap();
    assert_eq!(table.version_payload_bytes, 0);
    assert_eq!(table.pinned_version_payload_bytes, 0);
    assert_eq!(table.retired_arena_payload_bytes, 0);
    assert_eq!(table.version_tree_bytes, 0);
    assert_eq!(table.pinned_version_tree_bytes, 0);
}

#[test]
fn memory_stats_keep_transaction_history_until_release() {
    let db =
        Database::open("memory://memory_stats_keep_transaction_history_until_release").unwrap();
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO items VALUES (1, 10)", ()).unwrap();
    let bytes = || {
        db.engine()
            .memory_stats()
            .last()
            .unwrap()
            .transaction_version_bytes
    };
    assert_eq!(bytes(), 48);
    db.execute("SAVEPOINT first_write", ()).unwrap();
    db.execute("UPDATE items SET value = 20 WHERE id = 1", ())
        .unwrap();
    let with_history = bytes();
    assert!(with_history > 96);
    db.execute("ROLLBACK TO SAVEPOINT first_write", ()).unwrap();
    assert_eq!(bytes(), with_history - 48);
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(bytes(), 0);
}

#[test]
fn memory_totals_retain_dropped_table_until_payload_destruction() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://memory_totals_retain_dropped_table").unwrap();
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO items VALUES (1, 10)", ()).unwrap();
    let old = db.engine().get_version_store("items").unwrap();
    let old_capacity = old.arena_footprint().1;
    db.execute("DROP TABLE items", ()).unwrap();
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (2)", ()).unwrap();
    let current_capacity = db
        .engine()
        .get_version_store("items")
        .unwrap()
        .arena_footprint()
        .1;
    let stats = db.engine().memory_stats();
    let current = stats.iter().find(|row| row.table_name == "items").unwrap();
    assert_eq!(current.version_payload_bytes, 32);
    let tree_bytes = current.version_tree_bytes;
    assert!(tree_bytes > 0);
    let total = stats.last().unwrap();
    assert_eq!(total.version_payload_bytes, 80);
    assert_eq!(total.version_tree_bytes, tree_bytes * 2);
    assert_eq!(total.arena_capacity_bytes, old_capacity + current_capacity);
    let weak = std::sync::Arc::downgrade(&old);
    let observed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed_drop = std::sync::Arc::clone(&observed);
    let reader = db.clone();
    test_failpoints::before_hot_owner_drop(move || {
        assert!(
            weak.upgrade().is_none(),
            "the last store owner has been released"
        );
        let stats = reader.engine().memory_stats();
        let total = stats.last().unwrap();
        assert_eq!(
            total.version_payload_bytes, 80,
            "payloads remain charged during destruction"
        );
        assert_eq!(total.version_tree_bytes, tree_bytes * 2);
        assert_eq!(total.hot_bytes, 80);
        assert_eq!(total.arena_capacity_bytes, old_capacity + current_capacity);
        observed_drop.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    drop(old);
    assert!(observed.load(std::sync::atomic::Ordering::Relaxed));
    let stats = db.engine().memory_stats();
    let total = stats.last().unwrap();
    assert_eq!(total.version_payload_bytes, 32);
    assert_eq!(total.version_tree_bytes, tree_bytes);
    assert_eq!(total.hot_bytes, 32);
    assert_eq!(total.arena_capacity_bytes, current_capacity);
}

#[test]
fn dropped_table_keeps_retained_primary_index_charged() {
    let db = Database::open("memory://dropped_table_keeps_retained_primary_index_charged").unwrap();
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (4096), (-1)", ())
        .unwrap();
    let index = db
        .engine()
        .get_version_store("items")
        .unwrap()
        .get_index_by_column("id")
        .unwrap();
    let account = std::sync::Arc::clone(index.memory_account().unwrap());
    let retained = db
        .engine()
        .memory_stats()
        .last()
        .unwrap()
        .index_requested_bytes;
    assert!(retained > 0);
    db.execute("DROP TABLE items", ()).unwrap();
    assert_eq!(
        db.engine()
            .memory_stats()
            .last()
            .unwrap()
            .index_requested_bytes,
        retained
    );
    assert_eq!(
        index
            .get_row_ids_equal(&[stoolap::Value::Integer(4096)])
            .len(),
        1
    );
    drop(index);
    assert_eq!(
        db.engine()
            .memory_stats()
            .last()
            .unwrap()
            .index_requested_bytes,
        0,
        "an account handle keeps no index allocation charged after destruction"
    );
    drop(account);
}

#[test]
fn memory_stats_retain_indexes_after_drop_index() {
    for kind in ["HASH", "BTREE", "BITMAP", "COMPOSITE"] {
        let db = Database::open(&format!(
            "memory://memory_stats_retain_indexes_after_drop_index_{kind}"
        ))
        .unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO items VALUES (1, 'a long retained index key'), (2, 'another retained index key')", ())
        .unwrap();
        let before = db
            .engine()
            .memory_stats()
            .last()
            .unwrap()
            .index_requested_bytes;
        let create = if kind == "COMPOSITE" {
            "CREATE INDEX idx_key ON items(label, id)".to_string()
        } else {
            format!("CREATE INDEX idx_key ON items(label) USING {kind}")
        };
        db.execute(&create, ()).unwrap();
        let index = db
            .engine()
            .get_version_store("items")
            .unwrap()
            .get_index("idx_key")
            .unwrap();
        let retained = db.engine().memory_stats().pop().unwrap();
        assert!(retained.index_requested_bytes > before);
        assert!(retained.index_estimated_bytes > 0);
        db.execute("DROP INDEX idx_key ON items", ()).unwrap();
        let dropped = db.engine().memory_stats().pop().unwrap();
        assert_eq!(
            dropped.index_requested_bytes,
            retained.index_requested_bytes
        );
        assert_eq!(
            dropped.index_estimated_bytes,
            retained.index_estimated_bytes
        );
        drop(index);
        let released = db.engine().memory_stats().pop().unwrap();
        assert_eq!(released.index_requested_bytes, before);
        assert_eq!(released.index_estimated_bytes, 0);
    }
}

#[test]
fn changed_table_mapping_aborts_before_any_table_commit() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://changed_table_mapping_aborts_prepared_commit").unwrap();
    for table in ["left_rows", "right_rows"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY)"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO {table} VALUES (1)"), ())
            .unwrap();
    }
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO left_rows VALUES (2)", ()).unwrap();
    db.execute("INSERT INTO right_rows VALUES (2)", ()).unwrap();
    let writer = db.clone();
    test_failpoints::after_commit_preparation(move || {
        writer
            .execute("ALTER TABLE right_rows RENAME TO moved_rows", ())
            .unwrap();
    });
    let result = db.execute("COMMIT", ());
    for table in ["left_rows", "moved_rows"] {
        assert_eq!(
            db.query_one::<i64, _>(&format!("SELECT COUNT(*) FROM {table}"), ())
                .unwrap(),
            1,
            "a changed mapping must be rejected before either table commits"
        );
    }
    assert!(matches!(result, Err(stoolap::Error::TransactionAborted)));
}

#[test]
fn commit_releases_an_already_dropped_touched_table() {
    let db = Database::open("memory://commit_releases_dropped_touched_table").unwrap();
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO items VALUES (1)", ()).unwrap();
    db.execute("DROP TABLE items", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    assert!(db.query("SELECT * FROM items", ()).is_err());
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (1)", ()).unwrap();
}

#[test]
fn indexed_callback_keeps_captured_payloads_after_update() {
    let db = Database::open("memory://indexed_callback_keeps_captured_payloads").unwrap();
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO items VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let store = db.engine().get_version_store("items").unwrap();
    let mut reader = db.engine().begin_transaction().unwrap();
    let writer = db.clone();
    let mut seen = Vec::new();
    store.for_each_visible(&[1, 2], reader.id(), |id, row| {
        if id == 1 {
            writer
                .execute("UPDATE items SET value = 900 WHERE id = 2", ())
                .unwrap();
        }
        seen.push((id, row));
        true
    });
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[1].1.get(1),
        Some(&stoolap::Value::Integer(20)),
        "later callbacks must use the captured payload"
    );
    reader.rollback().unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT value FROM items WHERE id = 2", ())
            .unwrap(),
        900
    );
}

#[test]
fn row_policy_changes_preserve_chunk_identity_and_seal_releases_capacity() {
    let directory = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&hot_max_rows=0",
        directory.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE rows (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    let count = stoolap::storage::mvcc::arena::ARENA_CHUNK_ROWS as i64 + 6;
    db.execute(
        "INSERT INTO rows SELECT n, n FROM generate_series(1, $1) AS g(n)",
        (count,),
    )
    .unwrap();
    let store = db.engine().get_version_store("rows").unwrap();
    let footprint = store.arena_footprint();
    assert_eq!(footprint.0, count as usize);
    for limit in [0, 1000, 1048576, 0] {
        db.execute(&format!("PRAGMA HOT_MAX_ROWS = {limit}"), ())
            .unwrap();
        assert_eq!(
            store.arena_footprint(),
            footprint,
            "policy changes must not repartition chunks"
        );
        assert_eq!(
            db.query_one::<i64, _>("SELECT value FROM rows WHERE id = $1", (count,))
                .unwrap(),
            count
        );
    }
    db.execute("UPDATE rows SET value = -1 WHERE id = 1", ())
        .unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT value FROM rows WHERE id = 1", ())
            .unwrap(),
        -1
    );
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        store.arena_footprint(),
        (0, 0),
        "seal must release every empty chunk and directory"
    );
    drop(store);
    drop(db);
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM rows", ())
            .unwrap(),
        count
    );
    assert_eq!(
        db.query_one::<i64, _>("SELECT value FROM rows WHERE id = 1", ())
            .unwrap(),
        -1
    );
    assert_eq!(
        db.query_one::<i64, _>("SELECT value FROM rows WHERE id = $1", (count,))
            .unwrap(),
        count
    );
}

#[test]
fn growth_failure_rolls_back_tables_created_by_the_transaction() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://growth_failure_rolls_back_ddl").unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("CREATE TABLE new_rows (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO new_rows VALUES (1)", ()).unwrap();
    test_failpoints::fail_arena_growth_after(0);
    let error = db.execute("COMMIT", ()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected hot arena growth failure"),
        "unexpected commit error: {error}"
    );
    assert!(db.query("SELECT * FROM new_rows", ()).is_err());
    db.execute("CREATE TABLE new_rows (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO new_rows VALUES (1)", ()).unwrap();
}

#[test]
fn second_table_growth_failure_leaves_both_tables_unchanged() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://second_table_growth_failure").unwrap();
    for table in ["left_rows", "right_rows"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, value INTEGER UNIQUE)"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO {table} VALUES (1, 10)"), ())
            .unwrap();
    }
    db.execute("BEGIN", ()).unwrap();
    for table in ["left_rows", "right_rows"] {
        db.execute(
            &format!(
                "INSERT INTO {table} VALUES (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70)"
            ),
            (),
        )
        .unwrap();
    }
    test_failpoints::fail_arena_growth_after(1);
    let error = db.execute("COMMIT", ()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected hot arena growth failure"),
        "unexpected commit error: {error}"
    );
    for table in ["left_rows", "right_rows"] {
        let count: i64 = db
            .query_one(&format!("SELECT COUNT(*) FROM {table}"), ())
            .unwrap();
        assert_eq!(
            count, 1,
            "a reservation failure must not commit either table"
        );
        let value: i64 = db
            .query_one(&format!("SELECT value FROM {table} WHERE id = 1"), ())
            .unwrap();
        assert_eq!(value, 10);
        db.execute(&format!("INSERT INTO {table} VALUES (2, 20)"), ())
            .unwrap();
        db.execute(&format!("TRUNCATE TABLE {table}"), ()).unwrap();
        let count: i64 = db
            .query_one(&format!("SELECT COUNT(*) FROM {table}"), ())
            .unwrap();
        assert_eq!(
            count, 0,
            "failed preparation must release claims and reserved slots"
        );
    }
}
