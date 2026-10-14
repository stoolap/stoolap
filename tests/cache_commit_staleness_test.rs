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

//! A cached query or subquery result serves a statement only when that
//! statement sees the data the result was computed from, whichever handle,
//! thread or transaction wrote it

use stoolap::{Database, Value};

/// A scalar subquery, an IN subquery and a semi-join over b, each giving
/// one number that follows b.bv of row 1
const QUERIES: [&str; 3] = [
    "SELECT (SELECT bv FROM b WHERE bid = 1)",
    "SELECT COUNT(*) FROM a WHERE x IN (SELECT bv FROM b WHERE bid = 1)",
    "SELECT COUNT(*) FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bv = a.x AND b.bid = 1)",
];

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE b (bid INTEGER PRIMARY KEY, bv INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (1, 10)", ()).unwrap();
    db.execute(
        "INSERT INTO a SELECT g.value, CASE WHEN g.value <= 2 THEN 10 ELSE 20 END \
         FROM generate_series(1, 5) g",
        (),
    )
    .unwrap();
    db
}

fn one(db: &Database, sql: &str) -> String {
    let mut rows = db.query(sql, ()).unwrap();
    match rows.next() {
        Some(row) => match row.unwrap().get_value(0) {
            Some(Value::Integer(n)) => n.to_string(),
            other => format!("{other:?}"),
        },
        None => "no rows".to_string(),
    }
}

/// The queries whose result on `db` is not the one row 1 holding `bv` gives
fn wrong(db: &Database, bv: i64, when: &str) -> Vec<String> {
    let expected = match bv {
        10 => ["10", "2", "2"],
        20 => ["20", "3", "3"],
        _ => ["99", "0", "0"],
    };
    QUERIES
        .iter()
        .zip(expected)
        .filter_map(|(sql, want)| {
            let got = one(db, sql);
            (got != want).then(|| format!("{when}: {sql} gave {got}, expected {want}"))
        })
        .collect()
}

#[test]
fn test_select_cache_sees_an_autocommit_write_through_another_handle() {
    let db = open("cache_staleness_select_handle");
    db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO s VALUES (1, 5)", ()).unwrap();
    let other = db.clone();
    let sql = "SELECT * FROM s WHERE v > 0";
    assert_eq!(one(&db, sql), "1");
    other
        .execute("UPDATE s SET v = -1 WHERE id = 1", ())
        .unwrap();
    assert_eq!(one(&db, sql), "no rows", "after a write through a clone");
}

#[test]
fn test_subquery_caches_see_a_write_from_another_thread() {
    let db = open("cache_staleness_thread");
    let mut found = wrong(&db, 10, "first read");
    let other = db.clone();
    std::thread::spawn(move || {
        other
            .execute("UPDATE b SET bv = 20 WHERE bid = 1", ())
            .unwrap();
    })
    .join()
    .unwrap();
    found.extend(wrong(&db, 20, "after another thread's write"));
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn test_subquery_caches_see_a_commit_on_another_handle() {
    let db = open("cache_staleness_commit");
    let other = db.clone();
    let mut found = wrong(&db, 10, "first read");
    other.execute("BEGIN", ()).unwrap();
    other
        .execute("UPDATE b SET bv = 20 WHERE bid = 1", ())
        .unwrap();
    found.extend(wrong(&db, 10, "before the commit"));
    other.execute("COMMIT", ()).unwrap();
    found.extend(wrong(&db, 20, "after the commit"));
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn test_subquery_caches_keep_an_open_transaction_write_to_itself() {
    let db = open("cache_staleness_own_write");
    let other = db.clone();
    other.execute("BEGIN", ()).unwrap();
    other
        .execute("UPDATE b SET bv = 99 WHERE bid = 1", ())
        .unwrap();
    let mut found = wrong(&other, 99, "in the writer");
    found.extend(wrong(&db, 10, "beside the open writer"));
    other.execute("ROLLBACK", ()).unwrap();
    found.extend(wrong(&db, 10, "after the rollback"));
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn test_subquery_caches_keep_a_snapshot_read_to_its_snapshot() {
    let db = open("cache_staleness_snapshot");
    let other = db.clone();
    other
        .execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    assert_eq!(one(&other, "SELECT COUNT(*) FROM a"), "5");
    db.execute("UPDATE b SET bv = 20 WHERE bid = 1", ())
        .unwrap();
    let mut found = wrong(&other, 10, "in the snapshot");
    found.extend(wrong(&db, 20, "after the snapshot read"));
    other.execute("COMMIT", ()).unwrap();
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn test_subquery_caches_tell_two_databases_with_equal_epochs_apart() {
    let first = open("cache_staleness_first_database");
    let second = open("cache_staleness_second_database");
    second
        .execute("UPDATE b SET bv = 20 WHERE bid = 1", ())
        .unwrap();
    first
        .execute("UPDATE b SET bv = 10 WHERE bid = 1", ())
        .unwrap();
    let mut found = wrong(&first, 10, "first database");
    found.extend(wrong(&second, 20, "second database after the first"));
    found.extend(wrong(&first, 10, "first database after the second"));
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn test_subquery_caches_see_writes_to_every_table_they_read() {
    let db = open("cache_staleness_tables");
    db.execute("CREATE TABLE c (cid INTEGER PRIMARY KEY, cv INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap();
    db.execute("CREATE VIEW bv_view AS SELECT bid, bv FROM b", ())
        .unwrap();
    let reads = [
        "SELECT (SELECT COUNT(*) FROM b JOIN c ON b.bid = c.cid)",
        "SELECT (SELECT bv FROM bv_view WHERE bid = 1)",
        "SELECT (SELECT bv FROM b WHERE bid = (SELECT MAX(cid) FROM c))",
        "SELECT (SELECT COUNT(*) FROM b WHERE EXISTS (SELECT 1 FROM c WHERE c.cid = b.bid))",
        "SELECT COUNT(*) FROM a WHERE x IN (SELECT bv FROM b WHERE bid = 99 UNION SELECT cv * 10 FROM c)",
        "SELECT COUNT(*) FROM a WHERE EXISTS \
         (SELECT 1 FROM b WHERE b.bv = a.x AND b.bid = (SELECT MAX(cid) FROM c))",
    ];
    let before: Vec<String> = reads.iter().map(|sql| one(&db, sql)).collect();
    assert_eq!(before, ["1", "10", "10", "1", "2", "2"]);
    let other = db.clone();
    std::thread::spawn(move || {
        other.execute("INSERT INTO b VALUES (2, 30)", ()).unwrap();
        other.execute("INSERT INTO c VALUES (2, 2)", ()).unwrap();
        other
            .execute("UPDATE b SET bv = 20 WHERE bid = 1", ())
            .unwrap();
    })
    .join()
    .unwrap();
    let after: Vec<String> = reads.iter().map(|sql| one(&db, sql)).collect();
    assert_eq!(
        after,
        ["2", "20", "30", "2", "5", "0"],
        "after writes from another thread"
    );
}

#[test]
fn test_subquery_caches_see_a_table_dropped_created_or_truncated_elsewhere() {
    let db = open("cache_staleness_ddl");
    let mut found = wrong(&db, 10, "first read");
    let other = db.clone();
    std::thread::spawn(move || {
        other.execute("DROP TABLE b", ()).unwrap();
        other
            .execute("CREATE TABLE b (bid INTEGER PRIMARY KEY, bv INTEGER)", ())
            .unwrap();
        other.execute("INSERT INTO b VALUES (1, 20)", ()).unwrap();
    })
    .join()
    .unwrap();
    found.extend(wrong(&db, 20, "after DROP and CREATE"));
    let other = db.clone();
    std::thread::spawn(move || {
        other.execute("TRUNCATE TABLE b", ()).unwrap();
        other.execute("INSERT INTO b VALUES (1, 99)", ()).unwrap();
    })
    .join()
    .unwrap();
    found.extend(wrong(&db, 99, "after TRUNCATE"));
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn test_subquery_caches_see_writes_to_a_table_only_an_inner_subquery_reads() {
    let db = open("cache_staleness_inner_tables");
    db.execute("CREATE TABLE c (cid INTEGER PRIMARY KEY, cv INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap();
    let reads = [
        "SELECT (SELECT bv FROM b WHERE bid = (SELECT MAX(cid) FROM c))",
        "SELECT (SELECT COUNT(*) FROM b WHERE EXISTS (SELECT 1 FROM c WHERE c.cid = b.bid))",
        "SELECT COUNT(*) FROM a WHERE x IN (SELECT bv FROM b WHERE bid = 99 UNION SELECT cv * 10 FROM c)",
        "SELECT COUNT(*) FROM a WHERE EXISTS \
         (SELECT 1 FROM b WHERE b.bv = a.x AND b.bid = (SELECT MAX(cid) FROM c))",
    ];
    let before: Vec<String> = reads.iter().map(|sql| one(&db, sql)).collect();
    assert_eq!(before, ["10", "1", "2", "2"]);
    let other = db.clone();
    std::thread::spawn(move || {
        other.execute("DELETE FROM c WHERE cid = 1", ()).unwrap();
        other.execute("INSERT INTO c VALUES (2, 2)", ()).unwrap();
    })
    .join()
    .unwrap();
    let after: Vec<String> = reads.iter().map(|sql| one(&db, sql)).collect();
    assert_eq!(
        after,
        ["Some(Null(Null))", "0", "3", "0"],
        "after writes to c alone from another thread"
    );
}
