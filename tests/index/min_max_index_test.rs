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

//! MIN and MAX of an indexed column alone answer as over the same column
//! without an index: index forms, types, NULLs, empty tables, rows deleted
//! and updated hot and after a seal, a transaction's own rows and an older
//! snapshot

use stoolap::Database;

fn value(db: &Database, sql: &str) -> String {
    let rows: Vec<String> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| {
            let r = r.unwrap();
            if r.is_null(0) {
                "NULL".to_string()
            } else {
                format!("{:?}", r.get_value(0))
            }
        })
        .collect();
    rows.join(",")
}

/// Each MIN and MAX on `t` and on `u`, the same rows without an index,
/// against the aggregate over a derived table, which takes no shortcut
fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    for f in ["MIN", "MAX"] {
        let want = value(db, &format!("SELECT {f}(x) FROM (SELECT c AS x FROM u) s"));
        for shape in ["SELECT {f}(c) FROM {t}", "SELECT {f}(c) AS m FROM {t}"] {
            for table in ["t", "u"] {
                let sql = shape.replace("{f}", f).replace("{t}", table);
                let got = value(db, &sql);
                if got != want {
                    wrong.push(format!("{phase}: {sql} gave {got}, want {want}"));
                }
            }
        }
    }
}

fn twin(db: &Database, ty: &str, using: &str) {
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, c {ty})"),
            (),
        )
        .unwrap();
    }
    db.execute(&format!("CREATE INDEX t_c ON t(c) USING {using}"), ())
        .unwrap();
}

fn both(db: &Database, sql: &str) {
    for table in ["t", "u"] {
        db.execute(&sql.replace("{t}", table), ()).unwrap();
    }
}

fn key(ty: &str, k: i64) -> String {
    match ty {
        "TEXT" => format!("'k{k:05}'"),
        "FLOAT" => format!("{k}.5"),
        "TIMESTAMP" => format!("TIMESTAMP '2024-01-01 00:00:00' + INTERVAL '{k} seconds'"),
        _ => k.to_string(),
    }
}

/// Empty, all NULL, then rows with NULLs; the extremes deleted and moved;
/// for a file database, sealed and changed after the seal
fn exercise(db: &Database, ty: &str, file: bool, wrong: &mut Vec<String>) {
    check(db, "empty", wrong);
    both(db, "INSERT INTO {t} VALUES (1, NULL), (2, NULL)");
    check(db, "all NULL", wrong);
    for id in 3..300i64 {
        let c = if id % 13 == 0 {
            "NULL".to_string()
        } else {
            key(ty, (id * 37) % 500)
        };
        both(db, &format!("INSERT INTO {{t}} VALUES ({id}, {c})"));
    }
    check(db, "rows", wrong);
    both(db, &format!("DELETE FROM {{t}} WHERE c = {}", key(ty, 0)));
    both(
        db,
        &format!("UPDATE {{t}} SET c = {} WHERE id = 7", key(ty, 9999)),
    );
    check(db, "extremes changed", wrong);
    if file {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        check(db, "sealed", wrong);
        both(
            db,
            &format!("INSERT INTO {{t}} VALUES (900, {})", key(ty, -7)),
        );
        both(
            db,
            &format!("INSERT INTO {{t}} VALUES (901, {})", key(ty, 8888)),
        );
        check(db, "sealed with hot extremes", wrong);
        both(
            db,
            &format!("DELETE FROM {{t}} WHERE c = {}", key(ty, 9999)),
        );
        both(db, &format!("DELETE FROM {{t}} WHERE c = {}", key(ty, 1)));
        check(db, "sealed extremes deleted", wrong);
        both(
            db,
            &format!("INSERT INTO {{t}} VALUES (1000, {})", key(ty, -5)),
        );
        check(db, "sealed and a hot extreme", wrong);
    }
    db.execute("BEGIN", ()).unwrap();
    both(
        db,
        &format!("INSERT INTO {{t}} VALUES (2000, {})", key(ty, -50)),
    );
    both(db, &format!("DELETE FROM {{t}} WHERE c = {}", key(ty, 498)));
    check(db, "own rows", wrong);
    db.execute("ROLLBACK", ()).unwrap();
    check(db, "rolled back", wrong);
}

#[test]
fn min_and_max_through_an_index_answer_as_without_it() {
    let mut wrong = Vec::new();
    for (ty, using) in [
        ("INTEGER", "BTREE"),
        ("TEXT", "BTREE"),
        ("FLOAT", "BTREE"),
        ("TIMESTAMP", "BTREE"),
        ("INTEGER", "HASH"),
        ("TEXT", "HASH"),
    ] {
        let db = Database::open(&format!("memory://min_max_index_{ty}_{using}")).unwrap();
        twin(&db, ty, using);
        exercise(&db, ty, false, &mut wrong);
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        twin(&db, ty, using);
        exercise(&db, ty, true, &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn min_and_max_after_sealed_extremes_are_updated_and_sealed_again() {
    let mut wrong = Vec::new();
    for ty in ["INTEGER", "TEXT", "FLOAT", "TIMESTAMP"] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        twin(&db, ty, "BTREE");
        both(
            &db,
            &format!(
                "INSERT INTO {{t}} VALUES (1, {}), (2, {}), (3, {})",
                key(ty, 0),
                key(ty, 5),
                key(ty, 10)
            ),
        );
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        both(
            &db,
            &format!("UPDATE {{t}} SET c = {} WHERE id = 1", key(ty, 3)),
        );
        both(
            &db,
            &format!("UPDATE {{t}} SET c = {} WHERE id = 3", key(ty, 7)),
        );
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        check(&db, &format!("{ty} resealed"), &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn min_and_max_with_nan_answer_as_without_an_index() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://min_max_index_nan").unwrap();
    twin(&db, "FLOAT", "BTREE");
    both(
        &db,
        "INSERT INTO {t} VALUES (1, 1.5), (2, CAST('NaN' AS FLOAT)), (3, -2.5)",
    );
    check(&db, "NaN", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A snapshot older than a commit that brings a new extreme answers from
/// the rows it sees
#[test]
fn min_and_max_under_an_older_snapshot_answer_as_without_an_index() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://min_max_index_snapshot").unwrap();
    twin(&db, "INTEGER", "BTREE");
    both(
        &db,
        "INSERT INTO {t} SELECT value, value FROM generate_series(10, 100)",
    );
    let reader = db.clone();
    reader
        .execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    check(&reader, "snapshot before", &mut wrong);
    both(&db, "INSERT INTO {t} VALUES (1, 1), (1000, 1000)");
    check(&reader, "snapshot after a commit", &mut wrong);
    reader.execute("COMMIT", ()).unwrap();
    check(&db, "after", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A seal built before the hot extreme moved publishes the old value's copy
/// between the bound's hot read and its volume read
#[cfg(feature = "test-failpoints")]
#[test]
fn min_and_max_with_a_seal_publishing_inside_the_bound_answer_as_without_an_index() {
    use std::sync::mpsc;
    use stoolap::test_failpoints::{self as fp, FailpointGuard};
    let _guard = FailpointGuard::new();
    let mut wrong = Vec::new();
    // (aggregate, hot value sealed, value it moves to); the sealed row holds 5
    for (f, sealed, moved) in [("MIN", 0, 9), ("MAX", 10, 1)] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        // One table, so the paused seal is this table's
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, c INTEGER)",
            "CREATE INDEX t_c ON t(c) USING BTREE",
            "INSERT INTO t VALUES (1, 5)",
            "PRAGMA CHECKPOINT",
        ] {
            db.execute(sql, ()).unwrap();
        }
        db.execute(&format!("INSERT INTO t VALUES (2, {sealed})"), ())
            .unwrap();
        let (built_tx, built) = mpsc::channel();
        let (go, go_rx) = mpsc::channel::<()>();
        let sealer = db.clone();
        let seal = std::thread::spawn(move || {
            fp::after_seal_volumes_built(move || {
                built_tx.send(()).unwrap();
                let _ = go_rx.recv();
            });
            sealer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        built.recv().unwrap();
        db.execute(&format!("UPDATE t SET c = {moved} WHERE id = 2"), ())
            .unwrap();
        let seal = std::rc::Rc::new(std::cell::RefCell::new(Some(seal)));
        let joined = std::rc::Rc::clone(&seal);
        fp::after_index_bound_hot(move || {
            go.send(()).unwrap();
            if let Some(seal) = joined.borrow_mut().take() {
                seal.join().unwrap();
            }
        });
        let got = value(&db, &format!("SELECT {f}(c) FROM t"));
        let want = value(&db, &format!("SELECT {f}(x) FROM (SELECT c AS x FROM t) s"));
        if let Some(seal) = seal.borrow_mut().take() {
            fp::after_index_bound_hot(|| {});
            seal.join().unwrap();
            wrong.push(format!("{f}: the bound never read the hot rows"));
        }
        if got != want {
            wrong.push(format!("{f}: gave {got}, want {want}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn min_and_max_of_an_indexed_column_read_the_index() {
    use stoolap::test_failpoints::min_max_index_answers;
    let db = Database::open("memory://min_max_index_answers").unwrap();
    twin(&db, "INTEGER", "BTREE");
    both(
        &db,
        "INSERT INTO {t} SELECT value, value % 1000 FROM generate_series(1, 5000)",
    );
    for sql in ["SELECT MIN(c) FROM t", "SELECT MAX(c) AS m FROM t"] {
        let before = min_max_index_answers();
        value(&db, sql);
        assert_eq!(min_max_index_answers() - before, 1, "{sql}");
    }
    for sql in ["SELECT MIN(c) FROM u", "SELECT MIN(c) FROM t WHERE id > 5"] {
        let before = min_max_index_answers();
        value(&db, sql);
        assert_eq!(min_max_index_answers() - before, 0, "{sql}");
    }
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    twin(&db, "INTEGER", "BTREE");
    both(
        &db,
        "INSERT INTO {t} SELECT value, value % 1000 FROM generate_series(1, 5000)",
    );
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    both(&db, "INSERT INTO {t} VALUES (6000, 7)");
    let before = min_max_index_answers();
    value(&db, "SELECT MAX(c) FROM t");
    assert_eq!(min_max_index_answers() - before, 1, "sealed MAX(c)");
}
