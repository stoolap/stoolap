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

//! A grouped aggregate under a WHERE the table runs whole is computed by the
//! table over its volumes and its hot rows, each row seen as the statement's
//! snapshot sees it

use stoolap::Database;

/// Rows sealed into a volume, then one updated, one deleted and one
/// inserted, so the hot store hides, removes and adds to the sealed rows
fn mixed(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", dir.display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, a INTEGER, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_a ON t(a)", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 'x', 1, '2026-01-01'), (2, 1, 'x', 2, '2026-01-02'), \
         (3, 2, 'y', 10, '2026-01-03'), (4, 2, 'y', NULL, '2026-01-04'), \
         (5, NULL, NULL, 7, '2026-01-05')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 100 WHERE id = 1", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
    db.execute("INSERT INTO t VALUES (6, 2, 'y', 5, '2026-01-06')", ())
        .unwrap();
    db
}

/// The rows of `sql` as text, in its order
fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// The aggregate path EXPLAIN ANALYZE names for `sql`
fn path(db: &Database, sql: &str) -> String {
    rows(db, &format!("EXPLAIN ANALYZE {sql}"))
        .into_iter()
        .find_map(|line| line.trim().strip_prefix("Paths: ").map(str::to_string))
        .unwrap_or_default()
}

#[test]
fn a_grouped_where_over_volumes_and_hot_rows_is_the_tables() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT g, COUNT(*), SUM(a), MIN(a), MAX(a) FROM t WHERE a > 1 GROUP BY g ORDER BY g",
            &["1|2|102|2|100", "2|1|5|5|5", "NULL|1|7|7|7"],
        ),
        (
            "SELECT s, COUNT(*) FROM t WHERE ts >= '2026-01-02' GROUP BY s ORDER BY s",
            &["x|1", "y|2", "NULL|1"],
        ),
        (
            "SELECT g FROM t WHERE a > 4 GROUP BY g ORDER BY g",
            &["1", "2", "NULL"],
        ),
        ("SELECT g, COUNT(*) FROM t WHERE a > 1000 GROUP BY g", &[]),
    ];
    for &(sql, want) in cases {
        assert_eq!(rows(&db, sql), want, "{sql}");
        assert!(
            path(&db, sql).ends_with("aggregate.storage_grouped"),
            "{sql}: {}",
            path(&db, sql)
        );
    }
}

#[test]
fn a_grouped_where_sees_the_rows_of_its_own_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (7, 1, 'x', 50, '2026-01-07')", ())
        .unwrap();
    db.execute("UPDATE t SET a = 3 WHERE id = 2", ()).unwrap();
    assert_eq!(
        rows(
            &db,
            "SELECT g, COUNT(*), SUM(a) FROM t WHERE a > 1 GROUP BY g ORDER BY g"
        ),
        ["1|3|153", "2|1|5", "NULL|1|7"]
    );
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(
        rows(
            &db,
            "SELECT g, COUNT(*), SUM(a) FROM t WHERE a > 1 GROUP BY g ORDER BY g"
        ),
        ["1|2|102", "2|1|5", "NULL|1|7"]
    );
}

#[test]
fn a_float_sum_passing_the_range_across_volumes_is_an_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v FLOAT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 1.5e308)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 1, 1.5e308), (3, 1, -1.5e308)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for sql in [
        "SELECT g, SUM(v) FROM t GROUP BY g",
        "SELECT g, SUM(v) FROM t WHERE v > -1e309 GROUP BY g",
        "SELECT SUM(v) FROM t WHERE v > -1e309",
    ] {
        let answer = db
            .query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
        let error = answer.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(error.contains("Float overflow in SUM"), "{sql}: {error}");
    }
}

#[test]
fn a_grouped_where_a_side_file_answers_is_read_by_the_scan() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    let sql = "SELECT g, COUNT(*) FROM t WHERE a = 2 GROUP BY g";
    assert_eq!(rows(&db, sql), ["1|1"]);
}
