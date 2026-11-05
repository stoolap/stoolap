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

//! An UPDATE or DELETE whose WHERE ANDs its primary key equality with
//! other conditions writes the row the key names only when the whole WHERE
//! holds for it

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://pk_write_where_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 'a'), (2, NULL, 'b'), (3, 30, 'c')",
        (),
    )
    .unwrap();
    db
}

fn rows(db: &Database) -> Vec<(i64, Option<i64>)> {
    db.query("SELECT id, v FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect()
}

#[test]
fn a_key_write_writes_the_row_only_when_the_whole_where_holds() {
    let all = [(1, Some(10)), (2, None), (3, Some(30))];
    for (at, (sql, affected, want)) in [
        (
            "UPDATE t SET v = 11 WHERE id = 1 AND v <> 11",
            1,
            vec![(1, Some(11)), (2, None), (3, Some(30))],
        ),
        (
            "UPDATE t SET v = 11 WHERE id = 1 AND v = 99",
            0,
            all.to_vec(),
        ),
        (
            "UPDATE t SET v = 11 WHERE id = 2 AND v > 0",
            0,
            all.to_vec(),
        ),
        (
            "UPDATE t SET v = 11 WHERE id = 2 AND v IS NULL",
            1,
            vec![(1, Some(10)), (2, Some(11)), (3, Some(30))],
        ),
        (
            "UPDATE t SET v = v + 1 WHERE id = 3 AND s = 'c'",
            1,
            vec![(1, Some(10)), (2, None), (3, Some(31))],
        ),
        (
            "UPDATE t SET v = 11 WHERE v = 10 AND id = 1",
            1,
            vec![(1, Some(11)), (2, None), (3, Some(30))],
        ),
        (
            "UPDATE t SET v = 11 WHERE id = 1 AND id = 3",
            0,
            all.to_vec(),
        ),
        (
            "DELETE FROM t WHERE id = 1 AND v = 10",
            1,
            vec![(2, None), (3, Some(30))],
        ),
        ("DELETE FROM t WHERE id = 1 AND v <> 10", 0, all.to_vec()),
        ("DELETE FROM t WHERE id = 2 AND v < 100", 0, all.to_vec()),
        (
            "DELETE FROM t WHERE s = 'b' AND id = 2 AND v IS NULL",
            1,
            vec![(1, Some(10)), (3, Some(30))],
        ),
        ("DELETE FROM t WHERE id = 9 AND v = 10", 0, all.to_vec()),
    ]
    .into_iter()
    .enumerate()
    {
        let db = open(&format!("case_{at}"));
        assert_eq!(db.execute(sql, ()).unwrap(), affected, "{sql}");
        assert_eq!(rows(&db), want, "{sql}");
    }
}

#[test]
fn a_prepared_key_write_reads_each_runs_values() {
    let db = open("prepared");
    let update = db
        .prepare("UPDATE t SET v = $3 WHERE id = $1 AND v = $2")
        .unwrap();
    assert_eq!(update.execute((1, 10, 11)).unwrap(), 1);
    assert_eq!(update.execute((1, 10, 12)).unwrap(), 0);
    assert_eq!(update.execute((3, 30, 31)).unwrap(), 1);
    let delete = db
        .prepare("DELETE FROM t WHERE id = $1 AND v = $2")
        .unwrap();
    assert_eq!(delete.execute((1, 10)).unwrap(), 0);
    assert_eq!(delete.execute((1, 11)).unwrap(), 1);
    assert_eq!(rows(&db), vec![(2, None), (3, Some(31))]);
}

#[test]
fn a_key_write_in_a_transaction_reads_its_own_rows() {
    let db = open("transaction");
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (4, 40, 'd')", ()).unwrap();
    tx.execute("UPDATE t SET v = 41 WHERE id = 1 AND v = 10", ())
        .unwrap();
    assert_eq!(
        tx.execute("UPDATE t SET v = 42 WHERE id = 4 AND v = 40", ())
            .unwrap(),
        1
    );
    assert_eq!(
        tx.execute("UPDATE t SET v = 43 WHERE id = 1 AND v = 10", ())
            .unwrap(),
        0
    );
    assert_eq!(
        tx.execute("DELETE FROM t WHERE id = 4 AND v = 42", ())
            .unwrap(),
        1
    );
    tx.commit().unwrap();
    assert_eq!(rows(&db), vec![(1, Some(41)), (2, None), (3, Some(30))]);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_key_delete_asks_the_where_of_the_version_it_deletes() {
    let db = open("committed_between");
    let other = db.clone();
    stoolap::test_failpoints::before_key_delete_read(move || {
        other
            .execute("UPDATE t SET v = 20 WHERE id = 1", ())
            .unwrap();
    });
    let sql = "DELETE FROM t WHERE id = 1 AND v = 10";
    assert_eq!(db.execute(sql, ()).unwrap(), 0, "{sql}");
    assert_eq!(rows(&db), vec![(1, Some(20)), (2, None), (3, Some(30))]);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_key_update_with_other_conditions_runs_on_the_key_plan() {
    use std::cell::Cell;
    use std::rc::Rc;
    let db = open("route");
    for (sql, planned) in [
        ("UPDATE t SET v = 11 WHERE id = 1 AND v <> 11", true),
        ("UPDATE t SET v = v + 1 WHERE s = 'c' AND id = 3", true),
        ("DELETE FROM t WHERE id = 2 AND s = 'b'", false),
    ] {
        let fired = Rc::new(Cell::new(false));
        let hook = Rc::clone(&fired);
        stoolap::test_failpoints::after_compiled_epoch_checked(move || hook.set(true));
        db.execute(sql, ()).unwrap();
        assert_eq!(fired.get(), planned, "{sql} ran on the key plan");
    }
}
