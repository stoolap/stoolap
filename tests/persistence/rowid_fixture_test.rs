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

//! A database an older build left with a column named after the row
//! identity opens, and that table serves DDL alone until the column is
//! renamed or dropped. The fixtures and how they were written are in
//! tests/fixtures/rowid

use std::path::Path;

use stoolap::Database;

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A copy of the fixture `name` in a directory of its own, and its DSN
fn copied(name: &str) -> (tempfile::TempDir, String) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rowid")
        .join(name);
    let dir = tempfile::tempdir().unwrap();
    copy_tree(&fixture, dir.path());
    let dsn = format!("file://{}", dir.path().display());
    (dir, dsn)
}

fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect()
}

fn refused(db: &Database, sql: &str) {
    let text = db.execute(sql, ()).map(drop);
    let prepared = db.prepare(sql).and_then(|stmt| stmt.execute(()).map(drop));
    for (how, result) in [("text", text), ("prepared", prepared)] {
        let Err(message) = result.map_err(|e| e.to_string()) else {
            panic!("{how}: {sql} was served");
        };
        assert!(
            message.contains("table t") && message.contains("RENAME COLUMN _rowid_"),
            "{how}: {sql}: {message}"
        );
    }
}

#[test]
fn a_table_with_a_column_named_after_the_row_identity_serves_ddl_alone() {
    let (_dir, dsn) = copied("column_named_rowid_beside");
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(pairs(&db, "SELECT id, v FROM u"), [(1, 100)]);
        db.execute("INSERT INTO u VALUES (2, 200)", ()).unwrap();
        assert!(db.table_exists("t").unwrap(), "t does not exist");
        for sql in ["SHOW CREATE TABLE t", "SHOW INDEXES FROM t", "DESCRIBE t"] {
            let rows = db.query(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
            assert!(rows.count() > 0, "{sql} showed nothing");
        }
        for sql in [
            "SELECT a FROM t",
            "SELECT COUNT(*) FROM t",
            "INSERT INTO t VALUES (3, 30)",
            "UPDATE t SET a = 5",
            "DELETE FROM t",
            "SELECT * FROM t",
            "SELECT _rowid_ FROM t",
            "SELECT a, COUNT(*) FROM t GROUP BY a",
            "SELECT u.v FROM u JOIN t ON t.a = u.id",
            "SELECT v FROM u WHERE id IN (SELECT a FROM t)",
            "WITH s AS (SELECT a FROM t) SELECT a FROM s",
            "INSERT INTO u SELECT a + 10, a FROM t",
            "EXPLAIN ANALYZE SELECT a FROM t",
            "WITH t AS (SELECT 1 AS a) UPDATE t SET a = 5",
            "WITH t AS (SELECT 1 AS a) DELETE FROM t",
            "WITH RECURSIVE t(a) AS (SELECT a FROM t UNION ALL SELECT a FROM t WHERE FALSE) \
             SELECT SUM(a) FROM t",
            "CREATE TABLE t2 AS SELECT a FROM t",
            "WITH t AS (SELECT a FROM t) SELECT a FROM t",
            "SELECT a FROM t WHERE EXISTS (WITH t AS (SELECT 1 AS a) SELECT a FROM t)",
        ] {
            refused(&db, sql);
        }
        for (sql, answer) in [
            ("WITH t AS (SELECT 7 AS a) SELECT a FROM t", 7),
            (
                "WITH RECURSIVE t(a) AS (SELECT 1 UNION ALL SELECT a + 1 FROM t WHERE a < 3) \
                 SELECT COUNT(*) FROM t",
                3,
            ),
        ] {
            assert_eq!(db.query_one::<i64, _>(sql, ()).unwrap(), answer, "{sql}");
        }
        refused(&db, "SELECT x FROM v");
        let created = db
            .execute("CREATE VIEW w AS SELECT a FROM t", ())
            .unwrap_err()
            .to_string();
        assert!(created.contains("RENAME COLUMN _rowid_"), "{created}");
        for sql in [
            "ALTER TABLE t RENAME COLUMN _rowid_ TO b",
            "ALTER TABLE t DROP COLUMN _rowid_",
        ] {
            let message = db.execute(sql, ()).unwrap_err().to_string();
            assert!(message.contains("view v reads _rowid_"), "{sql}: {message}");
        }
        db.execute("DROP VIEW v", ()).unwrap();
        db.execute("ALTER TABLE t RENAME COLUMN _rowid_ TO b", ())
            .unwrap();
        assert_eq!(
            pairs(&db, "SELECT a, b FROM t ORDER BY a"),
            [(1, 10), (2, 20)]
        );
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        pairs(&db, "SELECT a, b FROM t ORDER BY a"),
        [(1, 10), (2, 20)]
    );
    assert_eq!(
        pairs(&db, "SELECT id, v FROM u ORDER BY id"),
        [(1, 100), (2, 200)]
    );
}

#[test]
fn a_table_with_a_column_named_after_the_row_identity_opens_when_the_column_is_dropped() {
    let (_dir, dsn) = copied("column_named_rowid");
    {
        let db = Database::open(&dsn).unwrap();
        refused(&db, "SELECT a FROM t");
        db.execute("ALTER TABLE t DROP COLUMN _rowid_", ()).unwrap();
        assert_eq!(
            pairs(&db, "SELECT a, a * 10 FROM t ORDER BY a"),
            [(1, 10), (2, 20)]
        );
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        pairs(&db, "SELECT a, a * 10 FROM t ORDER BY a"),
        [(1, 10), (2, 20)]
    );
}

#[test]
fn a_table_with_a_column_named_after_the_row_identity_can_be_dropped() {
    for in_transaction in [false, true] {
        let (_dir, dsn) = copied("column_named_rowid_beside");
        let db = Database::open(&dsn).unwrap();
        let statements: &[&str] = match in_transaction {
            true => &["BEGIN", "DROP TABLE t", "COMMIT"],
            false => &["DROP TABLE t"],
        };
        for sql in statements {
            db.execute(sql, ()).unwrap();
        }
        assert!(
            db.query("SELECT a FROM t", ()).is_err(),
            "transaction {in_transaction}: t is still there"
        );
        assert_eq!(pairs(&db, "SELECT id, v FROM u"), [(1, 100)]);
    }
}

#[test]
fn a_snapshot_with_a_column_named_after_the_row_identity_restores_to_ddl_alone() {
    let (_dir, dsn) = copied("snapshot_named_rowid");
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            pairs(&db, "SELECT a, b FROM t ORDER BY a"),
            [(1, 10), (2, 20)]
        );
        db.execute("PRAGMA RESTORE", ()).unwrap();
        refused(&db, "SELECT a FROM t");
        db.execute("ALTER TABLE t RENAME COLUMN _rowid_ TO c", ())
            .unwrap();
        assert_eq!(
            pairs(&db, "SELECT a, c FROM t ORDER BY a"),
            [(1, 10), (2, 20)]
        );
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        pairs(&db, "SELECT a, c FROM t ORDER BY a"),
        [(1, 10), (2, 20)]
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_failed_rename_of_the_column_leaves_the_table_serving_ddl_alone() {
    use std::sync::atomic::Ordering;
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (_dir, dsn) = copied("column_named_rowid");
    let db = Database::open(&dsn).unwrap();
    stoolap::test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
    let renamed = db.execute("ALTER TABLE t RENAME COLUMN _rowid_ TO b", ());
    stoolap::test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
    assert!(renamed.is_err(), "the rename went through");
    refused(&db, "SELECT a FROM t");
    refused(&db, "INSERT INTO t VALUES (3, 30)");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_restore_stopped_after_a_table_of_the_reserved_name_leaves_it_serving_ddl_alone() {
    use std::sync::atomic::Ordering;
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (_dir, dsn) = copied("snapshot_named_rowid_twice");
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        pairs(&db, "SELECT a, b FROM t ORDER BY a"),
        [(1, 10), (2, 20)]
    );
    stoolap::test_failpoints::RESTORE_LOADS_BEFORE_FAIL.store(2, Ordering::Release);
    let restored = db.execute("PRAGMA RESTORE", ());
    stoolap::test_failpoints::RESTORE_LOADS_BEFORE_FAIL.store(0, Ordering::Release);
    assert!(restored.is_err(), "the restore went through");
    for table in ["t", "t2"] {
        refused(&db, &format!("SELECT a FROM {table}"));
        refused(&db, &format!("INSERT INTO {table} VALUES (3, 30)"));
    }
}

#[test]
fn an_index_of_a_table_with_a_column_named_after_the_row_identity_drops() {
    let (_dir, dsn) = copied("column_named_rowid_beside");
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("DROP INDEX uk ON t", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "DROP VIEW v",
        "ALTER TABLE t RENAME COLUMN _rowid_ TO b",
        "INSERT INTO t VALUES (1, 30)",
    ] {
        db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    assert_eq!(count_of(&db, "SELECT COUNT(*) FROM t WHERE a = 1"), 2);
}

fn count_of(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}
