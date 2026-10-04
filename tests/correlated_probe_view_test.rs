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

//! A correlated EXISTS, NOT EXISTS or COUNT(*) that probes the inner
//! table's index reads the rows the statement sees: a transaction's own
//! changes, and a snapshot's state rather than later commits.

use stoolap::Database;

const EXISTS: &str =
    "SELECT o.id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.k = o.k) ORDER BY o.id LIMIT 10";
const EXISTS_WHERE: &str = "SELECT o.id FROM o WHERE EXISTS \
     (SELECT 1 FROM t WHERE t.k = o.k AND t.v > 0) ORDER BY o.id LIMIT 10";
const NOT_EXISTS: &str =
    "SELECT o.id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = o.k) ORDER BY o.id LIMIT 5";
const COUNT: &str =
    "SELECT o.id, (SELECT COUNT(*) FROM t WHERE t.k = o.k) FROM o ORDER BY o.id LIMIT 10";

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 1), (2, 20, 1)", ())
        .unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 10), (2, 30)", ())
        .unwrap();
    db
}

fn rows(result: stoolap::Rows) -> Vec<String> {
    result
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| r.get::<i64>(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

#[test]
fn a_transaction_probes_its_own_changes() {
    let db = setup("probe_own_changes");
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (3, 30, 1)", ()).unwrap();
    tx.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(rows(tx.query(EXISTS, ()).unwrap()), ["2"], "EXISTS");
    assert_eq!(
        rows(tx.query(EXISTS_WHERE, ()).unwrap()),
        ["2"],
        "EXISTS with a predicate"
    );
    assert_eq!(rows(tx.query(NOT_EXISTS, ()).unwrap()), ["1"], "NOT EXISTS");
    assert_eq!(rows(tx.query(COUNT, ()).unwrap()), ["1,0", "2,1"], "COUNT");
    tx.rollback().unwrap();
    assert_eq!(rows(db.query(EXISTS, ()).unwrap()), ["1"]);
    assert_eq!(rows(db.query(COUNT, ()).unwrap()), ["1,1", "2,0"]);
}

#[test]
fn a_snapshot_probes_the_rows_it_began_with() {
    let db = setup("probe_snapshot");
    let snapshot = db.clone();
    snapshot
        .execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    assert_eq!(rows(snapshot.query(EXISTS, ()).unwrap()), ["1"]);
    db.execute("INSERT INTO t VALUES (3, 30, 1)", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(rows(snapshot.query(EXISTS, ()).unwrap()), ["1"], "EXISTS");
    assert_eq!(
        rows(snapshot.query(EXISTS_WHERE, ()).unwrap()),
        ["1"],
        "EXISTS with a predicate"
    );
    assert_eq!(
        rows(snapshot.query(NOT_EXISTS, ()).unwrap()),
        ["2"],
        "NOT EXISTS"
    );
    assert_eq!(
        rows(snapshot.query(COUNT, ()).unwrap()),
        ["1,1", "2,0"],
        "COUNT"
    );
    snapshot.execute("COMMIT", ()).unwrap();
    assert_eq!(rows(db.query(EXISTS, ()).unwrap()), ["2"]);
    assert_eq!(rows(db.query(COUNT, ()).unwrap()), ["1,0", "2,1"]);
}

#[test]
fn a_sealed_row_whose_key_moved_is_not_counted_under_its_old_key() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value * 10, 1 FROM generate_series(1, 1000)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 10), (2, 5)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET k = 5 WHERE id = 1", ()).unwrap();
    assert_eq!(rows(db.query(COUNT, ()).unwrap()), ["1,0", "2,1"], "COUNT");
    assert_eq!(rows(db.query(EXISTS, ()).unwrap()), ["2"], "EXISTS");
    assert_eq!(
        rows(db.query(EXISTS_WHERE, ()).unwrap()),
        ["2"],
        "EXISTS with a predicate"
    );
    let projected = "SELECT o.id, CASE WHEN EXISTS \
         (SELECT 1 FROM t WHERE t.k = o.k AND t.v > 0) THEN 1 ELSE 0 END FROM o ORDER BY o.id";
    assert_eq!(
        rows(db.query(projected, ()).unwrap()),
        ["1,0", "2,1"],
        "EXISTS per row"
    );
}

fn counts(db: &Database, sql: &str) -> Vec<Option<i64>> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<Option<i64>>(1).unwrap())
        .collect()
}

fn keyed(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value FROM generate_series(1, 1000)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 1), (2, 2000)", ())
        .unwrap();
    db
}

#[test]
fn a_count_with_having_limit_or_offset_runs_the_subquery() {
    let db = keyed("probe_count_tail");
    for tail in ["HAVING COUNT(*) > 1", "LIMIT 0", "LIMIT 1 OFFSET 1"] {
        assert_eq!(
            counts(
                &db,
                &format!(
                    "SELECT o.id, (SELECT COUNT(*) FROM t WHERE t.k = o.k {tail}) \
                     FROM o ORDER BY o.id"
                )
            ),
            [None, None],
            "{tail}"
        );
    }
}

#[test]
fn a_cte_named_like_the_table_hides_it() {
    let db = keyed("probe_cte_shadow");
    assert_eq!(
        counts(
            &db,
            "WITH t AS (SELECT 2000 AS k) \
             SELECT o.id, (SELECT COUNT(*) FROM t WHERE t.k = o.k) FROM o ORDER BY o.id"
        ),
        [Some(0), Some(1)]
    );
}

#[test]
fn every_conjunct_beside_the_correlation_is_checked() {
    let db = keyed("probe_every_conjunct");
    assert_eq!(
        counts(
            &db,
            "SELECT o.id, CASE WHEN EXISTS \
             (SELECT 1 FROM t WHERE t.k = o.k AND t.id > 10 AND t.id > 0) THEN 1 ELSE 0 END \
             FROM o ORDER BY o.id"
        ),
        [Some(0), Some(0)]
    );
    assert_eq!(
        counts(
            &db,
            "SELECT o.id, (SELECT COUNT(*) FROM t WHERE t.k = o.k AND t.id > 10 AND t.id > 0) \
             FROM o ORDER BY o.id"
        ),
        [Some(0), Some(0)]
    );
}

#[test]
fn an_as_of_subquery_reads_its_point() {
    let db = Database::open("memory://probe_as_of").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 10)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let tx: i64 = db.query_one("SELECT CURRENT_TRANSACTION_ID()", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    db.execute("UPDATE t SET k = 20 WHERE id = 1", ()).unwrap();
    assert_eq!(
        counts(
            &db,
            &format!("SELECT o.id, (SELECT COUNT(*) FROM t AS OF TRANSACTION {tx} WHERE t.k = o.k) FROM o")
        ),
        [Some(1)]
    );
}
