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

//! A table read AS OF answers in every place a table stands as a copy of
//! its rows at that point would, and every AS OF of a statement resolves
//! against bounds taken when the statement starts.

#![cfg(feature = "test-failpoints")]

use std::time::Duration;

use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

type Rows = Result<Vec<Vec<String>>, String>;

fn rows(db: &Database, sql: &str) -> Rows {
    let mut out = Vec::new();
    for row in db.query(sql, ()).map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect(),
        );
    }
    Ok(out)
}

/// The wall clock as AS OF TIMESTAMP reads it, apart from the commits around it
fn now(db: &Database) -> String {
    std::thread::sleep(Duration::from_millis(3));
    let time: String = db
        .query("SELECT CAST(NOW() AS TEXT)", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    std::thread::sleep(Duration::from_millis(3));
    time
}

const ROWS: &str = "(1, 1, 5), (2, 2, 7), (3, 1, NULL), (4, 3, 9)";

/// Tables t and u, and u0 holding u's rows as they stood at the time
/// returned, after which u changes; sealed first when `sealed`
fn setup(db: &Database, sealed: bool) -> String {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "CREATE TABLE u0 (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 4)",
        &format!("INSERT INTO u VALUES {ROWS}"),
        &format!("INSERT INTO u0 VALUES {ROWS}"),
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let before = now(db);
    for sql in [
        "UPDATE u SET v = v + 100",
        "DELETE FROM u WHERE id = 4",
        "INSERT INTO u VALUES (5, 2, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    before
}

/// Each shape, `{u}` standing for the table read
const SHAPES: &[&str] = &[
    "SELECT id, k, v FROM {u} ORDER BY id",
    "SELECT x.id FROM {u} x WHERE x.v > 5 ORDER BY x.id",
    "SELECT x.k, COUNT(*), MAX(x.v) FROM {u} x GROUP BY x.k ORDER BY x.k",
    "SELECT DISTINCT x.k FROM {u} x ORDER BY x.k LIMIT 2",
    "SELECT t.id, (SELECT MAX(x.v) FROM {u} x WHERE x.k = t.k) FROM t ORDER BY t.id",
    "SELECT t.id FROM t WHERE EXISTS (SELECT 1 FROM {u} x WHERE x.k = t.k) ORDER BY t.id",
    "SELECT t.id FROM t WHERE NOT EXISTS (SELECT 1 FROM {u} x WHERE x.k = t.k) ORDER BY t.id",
    "SELECT t.id FROM t WHERE t.k IN (SELECT x.k FROM {u} x WHERE x.v > 5) ORDER BY t.id",
    "SELECT t.id FROM t WHERE t.k = ANY (SELECT x.k FROM {u} x) ORDER BY t.id",
    "SELECT d.id FROM (SELECT x.id, x.v FROM {u} x) d WHERE d.v > 5 ORDER BY d.id",
    "SELECT t.id, (SELECT MAX(d.v) FROM (SELECT * FROM {u} x) d WHERE d.k = t.k) FROM t ORDER BY t.id",
    "WITH c AS (SELECT * FROM {u} x) SELECT id, v FROM c ORDER BY id",
    "SELECT t.id, x.v FROM t JOIN {u} x ON x.k = t.k ORDER BY t.id, x.id",
    "SELECT t.id, x.v FROM t LEFT JOIN {u} x ON x.k = t.k ORDER BY t.id, x.id",
    "SELECT id FROM t UNION SELECT x.id FROM {u} x ORDER BY 1",
    "SELECT x.id, ROW_NUMBER() OVER (ORDER BY x.id) FROM {u} x ORDER BY x.id",
];

/// Every shape over `u` AS OF `before` answers as over `u0`
fn matrix(db: &Database, before: &str, mode: &str) {
    let as_of = format!("u AS OF TIMESTAMP '{before}'");
    for shape in SHAPES {
        let read = shape.replace("{u}", &as_of);
        let copy = shape.replace("{u}", "u0");
        assert_eq!(rows(db, &read), rows(db, &copy), "{mode}: {read}");
    }
}

#[test]
fn every_shape_reads_the_table_as_it_stood() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://as_of_sources_shapes").unwrap();
    let before = setup(&db, false);
    matrix(&db, &before, "autocommit");
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE u SET v = 0", ()).unwrap();
    matrix(&db, &before, "explicit transaction with a local write");
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn every_shape_reads_a_sealed_table_as_it_stood() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_interval=0",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    let before = setup(&db, true);
    matrix(&db, &before, "sealed");
}

#[test]
fn a_transaction_reads_the_committed_past_through_the_api() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://as_of_sources_api").unwrap();
    let before = setup(&db, false);
    let mut tx = db.begin().unwrap();
    tx.execute("UPDATE u SET v = 0", ()).unwrap();
    let sql = format!("SELECT id, v FROM u AS OF TIMESTAMP '{before}' ORDER BY id");
    let got: Vec<(i64, Option<i64>)> = tx
        .query(&sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect();
    assert_eq!(got, [(1, Some(5)), (2, Some(7)), (3, None), (4, Some(9))]);
    tx.rollback().unwrap();
}

#[test]
fn a_parameter_names_the_point() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://as_of_sources_parameter").unwrap();
    let before = setup(&db, false);
    let sql = "SELECT t.id, (SELECT MAX(x.v) FROM u AS OF TIMESTAMP $1 x WHERE x.k = t.k) \
               FROM t ORDER BY t.id";
    let read = |db: &Database| -> Vec<Option<i64>> {
        db.query(sql, (before.as_str(),))
            .unwrap()
            .map(|row| row.unwrap().get(1).unwrap())
            .collect()
    };
    assert_eq!(read(&db), [Some(5), Some(7), Some(9), None]);
    assert_eq!(
        read(&db),
        [Some(5), Some(7), Some(9), None],
        "the prepared plan again"
    );
}

#[test]
fn a_dml_query_reads_the_table_as_it_stood() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://as_of_sources_dml").unwrap();
    let before = setup(&db, false);
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        &format!("INSERT INTO w SELECT * FROM u AS OF TIMESTAMP '{before}'"),
        (),
    )
    .unwrap();
    assert_eq!(
        rows(&db, "SELECT * FROM w ORDER BY id"),
        rows(&db, "SELECT * FROM u0 ORDER BY id")
    );
    db.execute(
        &format!(
            "DELETE FROM t WHERE k IN (SELECT x.k FROM u AS OF TIMESTAMP '{before}' x WHERE x.v = 9)"
        ),
        (),
    )
    .unwrap();
    assert_eq!(
        rows(&db, "SELECT id FROM t ORDER BY id"),
        Ok(vec![vec!["1".into()], vec!["2".into()], vec!["4".into()]])
    );
    db.execute(
        &format!(
            "UPDATE t SET k = 10 WHERE k IN (SELECT x.k FROM u AS OF TIMESTAMP '{before}' x WHERE x.v = 5)"
        ),
        (),
    )
    .unwrap();
    assert_eq!(
        rows(&db, "SELECT id FROM t WHERE k = 10"),
        Ok(vec![vec!["1".into()]])
    );
}

#[test]
fn as_of_names_a_table_and_no_view_reads_one() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://as_of_sources_views").unwrap();
    let before = setup(&db, false);
    db.execute("CREATE VIEW v AS SELECT * FROM u", ()).unwrap();
    let refused = |sql: String| {
        let err = rows(&db, &sql).expect_err(&sql);
        assert!(err.contains("AS OF"), "{sql}: {err}");
    };
    refused(format!("SELECT * FROM v AS OF TIMESTAMP '{before}'"));
    refused(format!(
        "SELECT t.id FROM t WHERE EXISTS (SELECT 1 FROM v AS OF TIMESTAMP '{before}' x WHERE x.k = t.k)"
    ));
    refused(format!(
        "WITH c AS (SELECT * FROM u) SELECT * FROM c AS OF TIMESTAMP '{before}'"
    ));
    refused(format!(
        "SELECT t.id FROM t JOIN v AS OF TIMESTAMP '{before}' x ON x.k = t.k"
    ));
    for sql in [
        format!(
            "WITH c AS (SELECT * FROM u), d AS (SELECT * FROM c AS OF TIMESTAMP '{before}') \
             DELETE FROM t WHERE k IN (SELECT k FROM d)"
        ),
        format!(
            "WITH c AS (SELECT * FROM u), d AS (SELECT * FROM c AS OF TIMESTAMP '{before}') \
             UPDATE t SET k = 0 WHERE k IN (SELECT k FROM d)"
        ),
        format!(
            "WITH c AS (SELECT * FROM u) DELETE FROM t WHERE k IN (SELECT k FROM c AS OF TIMESTAMP '{before}')"
        ),
        format!(
            "WITH c AS (SELECT * FROM u) UPDATE t SET k = 0 WHERE k IN (SELECT k FROM c AS OF TIMESTAMP '{before}')"
        ),
    ] {
        let err = db.execute(&sql, ()).expect_err(&sql).to_string();
        assert!(err.contains("AS OF"), "{sql}: {err}");
    }
    for body in [
        format!("SELECT * FROM u AS OF TIMESTAMP '{before}'"),
        format!("SELECT * FROM t WHERE k IN (SELECT k FROM u AS OF TIMESTAMP '{before}')"),
        format!("SELECT * FROM (SELECT * FROM u AS OF TIMESTAMP '{before}') d"),
    ] {
        let sql = format!("CREATE VIEW w AS {body}");
        let err = db.execute(&sql, ()).expect_err(&sql).to_string();
        assert!(err.contains("AS OF"), "{sql}: {err}");
    }
}

#[test]
fn a_cte_body_reads_the_table_its_name_shadows_only_after_it() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://as_of_sources_cte_scope").unwrap();
    let before = setup(&db, false);
    let want = rows(&db, "SELECT id, v FROM u0 ORDER BY id");
    for sql in [
        format!("WITH u AS (SELECT * FROM u AS OF TIMESTAMP '{before}') SELECT id, v FROM u ORDER BY id"),
        format!(
            "WITH a AS (SELECT * FROM u AS OF TIMESTAMP '{before}'), u AS (SELECT 1 AS id) \
             SELECT id, v FROM a ORDER BY id"
        ),
    ] {
        assert_eq!(rows(&db, &sql), want, "{sql}");
    }
}

/// `sql` while another handle commits `u.v = 20` just before the
/// statement's first AS OF resolves
fn with_commit_before_resolution(sql: &str, name: &str) -> Rows {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 10)",
        "INSERT INTO u VALUES (1, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    fp::before_as_of_resolved(move || {
        other
            .execute("UPDATE u SET v = 20 WHERE id = 1", ())
            .unwrap();
    });
    if sql.starts_with("INSERT") {
        db.execute(sql, ()).map_err(|e| e.to_string())?;
        return rows(&db, "SELECT * FROM w");
    }
    rows(&db, sql)
}

#[test]
fn an_insert_reads_its_query_at_its_start() {
    let _guard = FailpointGuard::new();
    let got = with_commit_before_resolution(
        "INSERT INTO w SELECT * FROM u AS OF TIMESTAMP '2099-01-01'",
        "as_of_sources_insert_cut",
    );
    assert_eq!(got, Ok(vec![vec!["1".into(), "10".into()]]));
}

#[test]
fn the_point_of_a_time_not_yet_passed_is_the_statements_start() {
    let _guard = FailpointGuard::new();
    let got = with_commit_before_resolution(
        "SELECT t.id FROM t WHERE t.v = \
         (SELECT x.v FROM u AS OF TIMESTAMP '2099-01-01' x WHERE x.id = t.id)",
        "as_of_sources_cut_at_entry",
    );
    assert_eq!(got, Ok(vec![vec!["1".into()]]));
}

#[test]
fn two_sources_at_one_time_read_one_point() {
    let _guard = FailpointGuard::new();
    let got = with_commit_before_resolution(
        "SELECT a.v, b.v FROM u AS OF TIMESTAMP '2099-01-01' a \
         JOIN u AS OF TIMESTAMP '2099-01-01' b ON a.id = b.id",
        "as_of_sources_one_point",
    );
    assert_eq!(got, Ok(vec![vec!["10".into(), "10".into()]]));
}
