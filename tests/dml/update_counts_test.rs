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

//! An UPDATE reports the rows it matched and, of those, the rows its SET
//! changed, whether the rows are hot or sealed

use stoolap::{Database, ExecCounts};

/// Where the rows (1, 1, 'a'), (2, 2, 'b') and (3, 3, NULL) of t live
#[derive(Clone, Copy, Debug)]
enum Rows {
    Hot,
    Sealed,
    /// Rows 1 and 2 sealed, row 3 hot
    Mixed,
}

const LAYOUTS: [Rows; 3] = [Rows::Hot, Rows::Sealed, Rows::Mixed];

fn database(rows: Rows, dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 'a'), (2, 2, 'b')", ())
        .unwrap();
    if matches!(rows, Rows::Mixed) {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("INSERT INTO t VALUES (3, 3, NULL)", ()).unwrap();
    if matches!(rows, Rows::Sealed) {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn counts(matched: i64, changed: i64) -> ExecCounts {
    ExecCounts { matched, changed }
}

#[test]
fn update_counts_matched_and_changed_rows() {
    for rows in LAYOUTS {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        for (sql, expected) in [
            ("UPDATE t SET v = 2 WHERE id > 0", counts(3, 2)),
            ("UPDATE t SET v = v", counts(3, 0)),
            ("UPDATE t SET v = 2, s = 'b' WHERE v = 2", counts(3, 2)),
            ("UPDATE t SET s = NULL WHERE id = 1", counts(1, 1)),
            ("UPDATE t SET s = NULL WHERE id = 1", counts(1, 0)),
            ("UPDATE t SET v = 9 WHERE id > 100", counts(0, 0)),
            ("UPDATE t SET id = id + 10", counts(3, 3)),
        ] {
            assert_eq!(
                db.execute_counts(sql, ()).unwrap(),
                expected,
                "{rows:?}: {sql}"
            );
        }
        assert_eq!(
            db.execute("UPDATE t SET v = v", ()).unwrap(),
            3,
            "{rows:?}: execute still returns the matched rows"
        );
    }
}

#[test]
fn update_counts_by_primary_key_and_prepared() {
    for rows in LAYOUTS {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        let sql = "UPDATE t SET v = $1 WHERE id = $2";
        for (params, expected) in [((2, 2), counts(1, 0)), ((5, 2), counts(1, 1))] {
            assert_eq!(
                db.execute_counts(sql, params).unwrap(),
                expected,
                "{rows:?}: execute_counts {params:?}"
            );
        }
        let by_key = db.prepare(sql).unwrap();
        for (params, expected) in [
            ((1, 1), counts(1, 0)),
            ((7, 1), counts(1, 1)),
            ((7, 99), counts(0, 0)),
        ] {
            assert_eq!(
                by_key.execute_counts(params).unwrap(),
                expected,
                "{rows:?}: prepared by key {params:?}"
            );
        }
        let by_value = db.prepare("UPDATE t SET s = $1 WHERE v >= $2").unwrap();
        assert_eq!(
            by_value.execute_counts(("b", 0)).unwrap(),
            counts(3, 2),
            "{rows:?}: prepared by value"
        );
    }
}

#[test]
fn update_counts_in_a_transaction() {
    let db = Database::open("memory://update_counts_in_a_transaction").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    assert_eq!(
        tx.execute_counts("INSERT INTO t VALUES (1, 1, 'a'), (2, 2, 'b')", ())
            .unwrap(),
        counts(2, 2)
    );
    assert_eq!(
        tx.execute_counts("UPDATE t SET v = 1", ()).unwrap(),
        counts(2, 1)
    );
    assert_eq!(
        tx.execute_counts("UPDATE t SET v = $1 WHERE id = $2", (1, 2))
            .unwrap(),
        counts(1, 0)
    );
    assert_eq!(
        tx.execute_counts("DELETE FROM t WHERE id = 1", ()).unwrap(),
        counts(1, 1)
    );
    tx.commit().unwrap();
}

#[test]
fn update_counts_other_statements_report_rows_affected_twice() {
    let db = Database::open("memory://update_counts_other_statements").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    for (sql, expected) in [
        (
            "INSERT INTO t VALUES (1, 1, 'a'), (2, 2, 'b')",
            counts(2, 2),
        ),
        ("DELETE FROM t WHERE id = 2", counts(1, 1)),
        ("DELETE FROM t WHERE v > 0", counts(1, 1)),
    ] {
        assert_eq!(db.execute_counts(sql, ()).unwrap(), expected, "{sql}");
    }
}

#[test]
fn update_counts_judge_a_column_by_the_value_it_ends_with() {
    for rows in LAYOUTS {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        for (sql, matched) in [
            ("UPDATE t SET v = 1, V = v + 1 WHERE id = 1", 1),
            ("UPDATE t SET v = 1, V = v + 1 WHERE v = 1", 1),
        ] {
            let got = db.execute_counts(sql, ()).unwrap();
            let v: i64 = db.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
            assert_eq!(
                got,
                counts(matched, i64::from(v != 1)),
                "{rows:?}: {sql} left v = {v}"
            );
            db.execute("UPDATE t SET v = 1 WHERE id = 1", ()).unwrap();
        }
    }
}

#[test]
fn update_counts_returning_reports_rows_not_counts() {
    let db = Database::open("memory://update_counts_returning").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
    assert_eq!(
        db.execute_counts("UPDATE t SET v = 2 RETURNING id", ())
            .unwrap(),
        counts(0, 0)
    );
}
