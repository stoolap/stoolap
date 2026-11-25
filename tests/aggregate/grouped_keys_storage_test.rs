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

//! A GROUP BY of several columns is computed by the table over its volumes
//! and its hot rows, each row seen as the statement's snapshot sees it

use stoolap::Database;

/// Rows sealed into a volume, then one updated, one deleted and one
/// inserted, so the hot store hides, removes and adds to the sealed rows
fn mixed(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", dir.display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 'x', 1), (2, 1, 'x', 2), (3, 2, 'y', 10), \
         (4, 2, 'y', NULL), (5, NULL, NULL, 7)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 100 WHERE id = 1", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
    db.execute("INSERT INTO t VALUES (6, 2, 'y', 5)", ())
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
fn several_group_columns_over_volumes_and_hot_rows_are_the_tables() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT g, s, COUNT(*), SUM(a) FROM t GROUP BY g, s ORDER BY g, s",
            &["1|x|2|102", "2|y|2|5", "NULL|NULL|1|7"],
        ),
        (
            "SELECT s, g, COUNT(*), MIN(a), MAX(a) FROM t WHERE a > 1 GROUP BY s, g ORDER BY s",
            &["x|1|2|2|100", "y|2|1|5|5", "NULL|NULL|1|7|7"],
        ),
        (
            "SELECT s, g FROM t GROUP BY s, g ORDER BY s",
            &["x|1", "y|2", "NULL|NULL"],
        ),
        (
            "SELECT g, s, COUNT(*) FROM t GROUP BY g, s HAVING COUNT(*) > 1 ORDER BY g",
            &["1|x|2", "2|y|2"],
        ),
        (
            "SELECT g AS k, s, COUNT(*) FROM t GROUP BY g, s HAVING k > 1",
            &["2|y|2"],
        ),
        (
            "SELECT g, s, a, COUNT(*) FROM t GROUP BY g, s, a ORDER BY g, a",
            &[
                "1|x|2|1",
                "1|x|100|1",
                "2|y|5|1",
                "2|y|NULL|1",
                "NULL|NULL|7|1",
            ],
        ),
        (
            "SELECT g, s, COUNT(*) FROM t WHERE a > 1000 GROUP BY g, s",
            &[],
        ),
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
fn integer_group_columns_over_a_large_volume_keep_their_groups_apart() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, h INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value % 3, value % 5 FROM generate_series(1, 70000)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let sql = "SELECT g, h, COUNT(*) FROM t GROUP BY g, h ORDER BY g, h";
    let got = rows(&db, sql);
    assert_eq!(got.len(), 15, "{got:?}");
    assert_eq!(got[0], "0|0|4666");
    assert_eq!(got[6], "1|1|4667");
    assert_eq!(got[14], "2|4|4666");
    assert!(path(&db, sql).ends_with("aggregate.storage_grouped"));
}

#[test]
fn a_limit_that_bounds_the_groups_of_several_columns_keeps_the_hash_grouping() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    for sql in [
        "SELECT g, s, COUNT(*) FROM t GROUP BY g, s LIMIT 1",
        "SELECT g, s, COUNT(*) FROM t GROUP BY g, s LIMIT 2 OFFSET 1",
    ] {
        assert!(path(&db, sql).ends_with("aggregate.hash_groups"), "{sql}");
    }
    assert_eq!(
        rows(
            &db,
            "SELECT g, s, COUNT(*) FROM t GROUP BY g, s LIMIT 2 OFFSET 1"
        )
        .len(),
        2
    );
    let sql = "SELECT g, s, COUNT(*) FROM t GROUP BY g, s ORDER BY g LIMIT 1";
    assert_eq!(rows(&db, sql), ["1|x|2"]);
    assert!(
        path(&db, sql).contains("aggregate.storage_grouped"),
        "{sql}: {}",
        path(&db, sql)
    );
}

#[test]
fn a_group_column_a_volume_holds_as_its_default_is_one_value() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    db.execute("ALTER TABLE t ADD COLUMN d INTEGER DEFAULT 3", ())
        .unwrap();
    let sql = "SELECT g, d, COUNT(*) FROM t GROUP BY g, d ORDER BY g";
    assert_eq!(rows(&db, sql), ["1|3|2", "2|3|2", "NULL|3|1"]);
    assert!(path(&db, sql).ends_with("aggregate.storage_grouped"));
}

#[test]
fn several_group_columns_see_the_rows_of_their_own_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    let sql = "SELECT g, s, COUNT(*), SUM(a) FROM t GROUP BY g, s ORDER BY g";
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (7, 1, 'x', 50)", ())
        .unwrap();
    db.execute("UPDATE t SET a = 3 WHERE id = 2", ()).unwrap();
    assert_eq!(rows(&db, sql), ["1|x|3|153", "2|y|2|5", "NULL|NULL|1|7"]);
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(rows(&db, sql), ["1|x|2|102", "2|y|2|5", "NULL|NULL|1|7"]);
}

#[test]
fn a_float_sum_of_several_group_columns_passing_the_range_is_an_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, h TEXT, v FLOAT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 'x', 1.5e308)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (2, 1, 'x', 1.5e308), (3, 1, 'x', -1.5e308)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let sql = "SELECT g, h, SUM(v) FROM t GROUP BY g, h";
    let answer = db
        .query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    let error = answer.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(error.contains("Float overflow in SUM"), "{sql}: {error}");
}

#[test]
fn explain_names_every_group_column() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed(dir.path());
    let plan = rows(&db, "EXPLAIN SELECT g, s, COUNT(*) FROM t GROUP BY g, s");
    assert!(
        plan.iter()
            .any(|line| line.trim() == "StorageGrouped t (g, s)"),
        "{plan:#?}"
    );
}
