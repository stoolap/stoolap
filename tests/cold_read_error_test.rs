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

//! Cold read errors must survive optimization fallback and cached access.

#![cfg(feature = "test-failpoints")]

use stoolap::core::Row;
use stoolap::{test_failpoints, Database, Result};

fn fixture() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&sync_mode=full",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER, code TEXT UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX items_category ON items(category)", ())
        .unwrap();
    db.execute(
        "INSERT INTO items VALUES (1, 'a', 10, 'one'), (2, 'a', 20, 'two'), (3, 'b', 30, 'three'), (6, 'b', 60, 'six')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    (dir, dsn)
}

fn rows(db: &Database, sql: &str) -> Result<Vec<Row>> {
    db.query(sql, ())?
        .map(|row| row.map(|row| row.into_inner()))
        .collect()
}

fn assert_injected<T>(result: Result<T>, context: &str) {
    let error = match result {
        Ok(_) => panic!("{context}: injected read was lost or answered as a partial result"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("injected cold read failure"),
        "{context}: expected the read error, received {error}"
    );
}

#[test]
fn cold_query_errors_survive_first_touch_and_cached_fast_paths() {
    let _guard = test_failpoints::FailpointGuard::new();
    let (_dir, dsn) = fixture();
    let queries = [
        "SELECT amount, code FROM items WHERE id = 2",
        "SELECT amount + 1, code FROM items ORDER BY id",
        "SELECT COUNT(*) FROM items WHERE amount >= 20",
        "SELECT SUM(amount) FROM items WHERE category = 'a'",
        "SELECT MIN(amount), MAX(amount) FROM items WHERE category = 'a'",
        "SELECT category, SUM(amount) FROM items GROUP BY category ORDER BY category",
        "SELECT DISTINCT category FROM items ORDER BY category",
        "SELECT COUNT(DISTINCT category) FROM items",
        "SELECT * FROM items ORDER BY id LIMIT 2",
        "SELECT category, ROW_NUMBER() OVER (PARTITION BY category ORDER BY amount) FROM items ORDER BY category, amount",
    ];

    for sql in queries {
        // A new engine resets its volume views and compiled statement cache.
        let db = Database::open(&dsn).unwrap();
        test_failpoints::fail_cold_read_on(1);
        assert_injected(rows(&db, sql), &format!("first access: {sql}"));
        test_failpoints::fail_cold_read_on(0);
        let expected = rows(&db, sql).unwrap();
        assert!(
            !expected.is_empty(),
            "fixture query must produce rows: {sql}"
        );

        // The same SQL now has a compiled plan and warmed column/index state.
        test_failpoints::fail_cold_read_on(1);
        assert_injected(rows(&db, sql), &format!("cached access: {sql}"));
        test_failpoints::fail_cold_read_on(0);
        assert_eq!(
            rows(&db, sql).unwrap(),
            expected,
            "retry after error: {sql}"
        );
        db.close().unwrap();
    }
}

#[test]
fn cold_projection_error_after_identity_lookup_is_not_a_missing_row() {
    let _guard = test_failpoints::FailpointGuard::new();
    let (_dir, dsn) = fixture();
    let db = Database::open(&dsn).unwrap();
    let sql = "SELECT amount, code FROM items WHERE id = 2";
    test_failpoints::fail_cold_read_on(2);
    assert_injected(rows(&db, sql), "projection after identity lookup");
    test_failpoints::fail_cold_read_on(0);
    let actual = rows(&db, sql).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].get(0), Some(&stoolap::Value::Integer(20)));
}

#[test]
fn cold_constraint_read_error_never_admits_a_write() {
    let _guard = test_failpoints::FailpointGuard::new();
    let (_dir, dsn) = fixture();
    let db = Database::open(&dsn).unwrap();
    // The missing ID is inside the cold volume's range, so the PK proof must read it.
    let insert = "INSERT INTO items VALUES (4, 'a', 40, 'four')";
    test_failpoints::fail_cold_read_on(1);
    assert_injected(db.execute(insert, ()), "cold primary-key absence proof");
    test_failpoints::fail_cold_read_on(0);
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM items", ())
            .unwrap(),
        4
    );

    // A PK match followed by a distinct UNIQUE probe must preserve the read failure.
    test_failpoints::fail_cold_read_on(2);
    assert_injected(db.execute(insert, ()), "cold UNIQUE proof");
    test_failpoints::fail_cold_read_on(0);
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM items", ())
            .unwrap(),
        4
    );
    db.execute(insert, ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM items", ())
            .unwrap(),
        5
    );
    assert!(db
        .execute("INSERT INTO items VALUES (5, 'a', 50, 'two')", ())
        .is_err());
}

#[test]
fn cold_error_after_scan_progress_discards_rows_and_aggregate() {
    let _guard = test_failpoints::FailpointGuard::new();
    let (_dir, dsn) = fixture();
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "INSERT INTO items VALUES (7, 'c', 70, 'seven'), (8, 'c', 80, 'eight')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 2);
    for (sql, nth) in [
        // Each volume binds four projected columns; the fifth read belongs
        // to the second volume after the first has contributed visible rows.
        ("SELECT id, amount, code FROM items ORDER BY id", 5),
        // Predicate and SUM share the amount column. The second binding is
        // in the next volume, after the first partial aggregate was accumulated.
        ("SELECT SUM(amount) FROM items WHERE amount >= 10", 2),
    ] {
        let expected = rows(&db, sql).unwrap();
        test_failpoints::fail_cold_read_on(nth);
        assert_injected(rows(&db, sql), &format!("later read: {sql}"));
        test_failpoints::fail_cold_read_on(0);
        assert_eq!(rows(&db, sql).unwrap(), expected);
    }
}

#[test]
fn lazy_window_partition_propagates_first_cold_read_error() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, category TEXT, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_category ON t(category)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 30)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    test_failpoints::fail_cold_read_on(1);
    let result = db
        .query(
            "SELECT category, ROW_NUMBER() OVER (PARTITION BY category ORDER BY v) FROM t LIMIT 2",
            (),
        )
        .and_then(|rows| rows.collect::<std::result::Result<Vec<_>, _>>());
    assert!(
        result.is_err(),
        "lazy-window optimization swallowed cold read error and returned a fallback result"
    );
}

#[test]
fn dictionary_filter_prepass_preserves_the_original_read_error() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_interval=3600", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE strings (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO strings VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 30)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    let db = Database::open(&dsn).unwrap();
    let sql = "SELECT amount + 1 FROM strings WHERE category = 'a'";
    test_failpoints::fail_cold_read_on(1);
    assert_injected(rows(&db, sql), "dictionary prepass");
    test_failpoints::fail_cold_read_on(0);
    assert_eq!(rows(&db, sql).unwrap().len(), 2);
}

#[test]
fn failed_cold_index_build_never_installs_a_partial_index() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_interval=3600", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE ddl_reads (id INTEGER PRIMARY KEY, v INTEGER, embedding VECTOR(3))",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO ddl_reads VALUES (1, 10, '[1,0,0]'), (2, 20, '[0,1,0]'), (3, 30, '[0,0,1]')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let store = db.engine().get_version_store("ddl_reads").unwrap();
    for (name, sql) in [
        (
            "ddl_unique",
            "CREATE UNIQUE INDEX ddl_unique ON ddl_reads(v)",
        ),
        (
            "ddl_vector",
            "CREATE INDEX ddl_vector ON ddl_reads(embedding) USING HNSW",
        ),
    ] {
        test_failpoints::fail_cold_read_on(1);
        assert_injected(db.execute(sql, ()), "detached cold index build");
        test_failpoints::fail_cold_read_on(0);
        assert!(!store.index_exists(name), "failed build installed {name}");
        db.execute(sql, ()).unwrap();
        assert!(store.index_exists(name));
    }
}

#[test]
fn unique_index_build_checks_duplicates_across_hot_and_cold() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE ddl_dupes (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO ddl_dupes VALUES (1, 10)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO ddl_dupes VALUES (2, 10)", ())
        .unwrap();
    assert!(db
        .execute("CREATE UNIQUE INDEX duplicate_key ON ddl_dupes(v)", ())
        .is_err());
    assert!(!db
        .engine()
        .get_version_store("ddl_dupes")
        .unwrap()
        .index_exists("duplicate_key"));
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM ddl_dupes", ())
            .unwrap(),
        2
    );
}
