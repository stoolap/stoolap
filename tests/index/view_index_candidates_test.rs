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

//! An index lookup under a statement view a later commit overtook: a
//! commit that only adds entries leaves the lookup to the index, one that
//! takes an entry out or builds an index sends it to the scan

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::test_failpoints as fp;
use stoolap::Database;

fn open(name: &str, indexed: bool) -> Database {
    fill(
        Database::open(&format!("memory://view_index_candidates_{name}")).unwrap(),
        indexed,
    )
}

fn open_file(dir: &std::path::Path) -> Database {
    fill(
        Database::open(&format!(
            "file://{}?checkpoint_interval=3600",
            dir.display()
        ))
        .unwrap(),
        true,
    )
}

fn fill(db: Database, indexed: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    if indexed {
        db.execute("CREATE INDEX t_k ON t (k)", ()).unwrap();
    }
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 2, 30), (4, 3, 40), (5, 5, 50)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 2), (2, 3)", ())
        .unwrap();
    db
}

/// Once this thread's next statement has its view, another handle runs
/// `sqls`; the flag says they ran
fn after_the_view(db: &Database, sqls: &'static [&'static str]) -> Rc<Cell<bool>> {
    let done = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&done));
    fp::after_statement_view_taken(move || {
        for sql in sqls {
            other.execute(sql, ()).unwrap();
        }
        flag.set(true);
    });
    done
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// The ids `sql` gives while another handle runs `sqls` after its view,
/// the index lookups it answered and the joins that read their keys by a
/// scan
fn overtaken(db: &Database, sql: &str, sqls: &'static [&'static str]) -> (Vec<i64>, usize, usize) {
    let ran = after_the_view(db, sqls);
    let (lookups, scans) = (fp::view_index_lookups(), fp::join_view_key_scans());
    let seen = ids(db, sql);
    assert!(ran.get(), "{sql}: the commit did not land after the view");
    (
        seen,
        fp::view_index_lookups() - lookups,
        fp::join_view_key_scans() - scans,
    )
}

const LOOKUPS: &[(&str, &[i64])] = &[
    ("SELECT id FROM t WHERE k = 2", &[2, 3]),
    ("SELECT id FROM t WHERE k >= 3", &[4, 5]),
];
const JOIN: &str = "SELECT t.id FROM u JOIN t ON t.k = u.x WHERE u.id = 1";

#[test]
fn an_insert_after_the_view_leaves_the_lookup_to_the_index() {
    let mut wrong = Vec::new();
    for (n, (sql, expected)) in LOOKUPS.iter().enumerate() {
        let db = open(&format!("insert_{n}"), true);
        let (seen, lookups, _) =
            overtaken(&db, sql, &["INSERT INTO t VALUES (6, 2, 60), (7, 4, 70)"]);
        if seen != *expected || lookups == 0 {
            wrong.push(format!("{sql}: {seen:?} with {lookups} index lookups"));
        }
    }
    let db = open("insert_join", true);
    let (seen, _, scans) = overtaken(&db, JOIN, &["INSERT INTO t VALUES (6, 2, 60)"]);
    if seen != [2, 3] || scans != 0 {
        wrong.push(format!("{JOIN}: {seen:?} with {scans} key scans"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_insert_after_the_view_leaves_a_file_tables_equality_candidates_to_the_index() {
    let mut wrong = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    let (seen, _, scans) = overtaken(&db, JOIN, &["INSERT INTO t VALUES (6, 2, 60)"]);
    if seen != [2, 3] || scans != 0 {
        wrong.push(format!("{JOIN}: {seen:?} with {scans} key scans"));
    }
    let by_candidates = fp::in_member_reads().0;
    let (seen, _, _) = overtaken(
        &db,
        "SELECT id FROM t WHERE k IN (2, 3)",
        &["INSERT INTO t VALUES (7, 3, 70)"],
    );
    let read = fp::in_member_reads().0 - by_candidates;
    // Row 6 came in before this statement's view, row 7 after it
    if seen != [2, 3, 4, 6] || read == 0 {
        wrong.push(format!("IN: {seen:?} with {read} candidate reads"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_key_taken_out_after_the_view_is_found_by_the_removal_history() {
    let mut wrong = Vec::new();
    for (change, n) in [
        ("UPDATE t SET k = 9 WHERE id = 3", "update"),
        ("DELETE FROM t WHERE id = 3", "delete"),
        ("UPDATE t SET k = 9 WHERE id >= 3", "update_rows"),
        ("DELETE FROM t WHERE id >= 3", "delete_rows"),
    ] {
        let sqls: &'static [&'static str] = Box::leak(Box::new([change]));
        let db = open(&format!("{n}_lookup"), true);
        let (seen, lookups, _) = overtaken(&db, "SELECT id FROM t WHERE k = 2", sqls);
        if seen != [2, 3] || lookups == 0 {
            wrong.push(format!("{change}: {seen:?} with {lookups} index lookups"));
        }
        let db = open(&format!("{n}_join"), true);
        let (seen, _, scans) = overtaken(&db, JOIN, sqls);
        if seen != [2, 3] || scans != 0 {
            wrong.push(format!("{change}, join: {seen:?} with {scans} key scans"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_index_built_after_the_view_sends_the_lookup_to_the_scan() {
    let db = open("built", false);
    let (seen, lookups, _) = overtaken(
        &db,
        "SELECT id FROM t WHERE k = 2",
        &[
            "UPDATE t SET k = 9 WHERE id = 3",
            "CREATE INDEX t_k ON t (k)",
        ],
    );
    assert_eq!(
        (seen, lookups),
        (vec![2, 3], 0),
        "the rows the view sees under the key, by the scan"
    );
}
