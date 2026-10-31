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

//! A prepared UPDATE or DELETE reads its rows by the access decided when it
//! was prepared, and decides again once an index of its target changes

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap()
}

fn fill(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_a ON t(a)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value % 10, value % 7, 0 FROM generate_series(1, 200)",
        (),
    )
    .unwrap();
}

#[test]
fn a_prepared_update_writes_the_rows_its_where_names_across_index_changes() {
    let db = Database::open("memory://prepared_dml_access_update").unwrap();
    fill(&db);
    let by_a = db.prepare("UPDATE t SET v = v + 1 WHERE a = $1").unwrap();
    let by_range = db
        .prepare("UPDATE t SET v = v + 10 WHERE id >= $1 AND id < $2")
        .unwrap();
    let by_both = db
        .prepare("UPDATE t SET v = v + 100 WHERE a = $1 AND b = $2")
        .unwrap();
    let mut changes = [
        "DROP INDEX t_a ON t",
        "CREATE INDEX t_b ON t(b)",
        "CREATE INDEX t_a ON t(a)",
        "DROP INDEX t_b ON t",
    ]
    .into_iter();
    for round in 0..5 {
        assert_eq!(by_a.execute((3,)).unwrap(), 20, "round {round}: a = 3");
        assert_eq!(
            by_range.execute((50, 60)).unwrap(),
            10,
            "round {round}: range"
        );
        assert_eq!(
            by_both.execute((3, 3)).unwrap(),
            3,
            "round {round}: a and b"
        );
        if let Some(change) = changes.next() {
            db.execute(change, ()).unwrap();
        }
    }
    assert_eq!(
        count(&db, "SELECT SUM(v) FROM t"),
        5 * (20 + 10 * 10 + 3 * 100),
        "each round's writes, once each"
    );
}

#[test]
fn a_prepared_delete_removes_the_rows_its_where_names_across_index_changes() {
    let db = Database::open("memory://prepared_dml_access_delete").unwrap();
    fill(&db);
    let by_a = db.prepare("DELETE FROM t WHERE a = $1").unwrap();
    assert_eq!(by_a.execute((1,)).unwrap(), 20);
    db.execute("DROP INDEX t_a ON t", ()).unwrap();
    assert_eq!(by_a.execute((2,)).unwrap(), 20);
    db.execute("CREATE INDEX t_a ON t(a)", ()).unwrap();
    // A row written after the index is made is found by it, not by the dropped one
    db.execute("INSERT INTO t VALUES (500, 3, 0, 0)", ())
        .unwrap();
    assert_eq!(by_a.execute((3,)).unwrap(), 21);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 200 - 20 - 20 - 20);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_write_reads_by_the_indexes_of_the_table_it_opened() {
    use std::cell::Cell;
    use std::rc::Rc;
    let db = Database::open("memory://prepared_dml_access_renamed").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 7, 0)", ()).unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_dml_table_opened(move || {
        for sql in [
            "ALTER TABLE t RENAME TO old_t",
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
            "CREATE INDEX t_k2 ON t(k)",
            "INSERT INTO t VALUES (2, 7, 0)",
        ] {
            other.execute(sql, ()).unwrap();
        }
        hook_fired.set(true);
    });
    let updated = db.execute("UPDATE t SET v = v + 1 WHERE k = 7", ());
    assert!(fired.get(), "the rename ran between the open and the plan");
    // The write is the opened table's, or refused; never a quiet miss
    if let Ok(updated) = updated {
        assert_eq!(updated, 1, "the opened table's row was not written");
        assert_eq!(count(&db, "SELECT v FROM old_t WHERE id = 1"), 1);
    }
    assert_eq!(count(&db, "SELECT v FROM t WHERE id = 2"), 0);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_write_reads_the_comparison_a_set_test_makes_by_its_index() {
    let db = Database::open("memory://prepared_dml_access_set").unwrap();
    fill(&db);
    db.execute("CREATE TABLE rhs (id INTEGER PRIMARY KEY, b INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO rhs VALUES (1, 5), (2, 7)", ())
        .unwrap();
    for sql in [
        "UPDATE t SET v = v + 1 WHERE a > ALL (SELECT b FROM rhs)",
        "DELETE FROM t WHERE a > ALL (SELECT b FROM rhs)",
    ] {
        let statement = db.prepare(sql).unwrap();
        let before = stoolap::test_failpoints::hot_index_ids_read();
        assert_eq!(statement.execute(()).unwrap(), 40, "{sql}");
        assert!(
            stoolap::test_failpoints::hot_index_ids_read() > before,
            "{sql}: the comparison the set test made read no index"
        );
    }
}
