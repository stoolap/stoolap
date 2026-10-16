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

//! A small outer side narrows a large inner table by TEXT, FLOAT and
//! TIMESTAMP keys as it does by INTEGER keys, hot and sealed, and answers
//! as the same join over an unindexed copy

use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

// The index page ledger is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const ROWS: i64 = 70000;

/// A key of type `ty` for the integer expression `expr`
fn key(ty: &str, expr: &str) -> String {
    match ty {
        "TEXT" => format!("'k' || CAST({expr} AS TEXT)"),
        "FLOAT" => format!("CAST({expr} AS FLOAT) + 0.5"),
        "TIMESTAMP" => format!(
            "DATE_ADD(TIMESTAMP '2024-01-01 00:00:00', {expr}, 'second') + INTERVAL '250 milliseconds'"
        ),
        _ => expr.to_string(),
    }
}

/// `t` and its unindexed copy `u`, keyed by `inner` type, and the small
/// outer side `s`, keyed by `outer` type; sealed when asked
fn tables(dir: &tempfile::TempDir, inner: &str, outer: &str, seal: bool) -> Database {
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k {inner}, v INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, {}, value FROM generate_series(1, {ROWS})",
                key(inner, "value % 1000")
            ),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute(
        &format!("CREATE TABLE s (id INTEGER PRIMARY KEY, k {outer}, w INTEGER)"),
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO s VALUES (1, {}, 10), (2, {}, 20), (3, {}, 30), (4, NULL, 40)",
            key(outer, "1"),
            key(outer, "2"),
            key(outer, "2")
        ),
        (),
    )
    .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<i64>> {
    let mut out: Vec<Vec<i64>> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| r.get::<i64>(i).unwrap_or(-1))
                .collect()
        })
        .collect();
    out.sort_unstable();
    out
}

const JOINS: &[&str] = &[
    "SELECT t.id, s.w FROM s JOIN t ON t.k = s.k",
    "SELECT t.id FROM t JOIN (SELECT k FROM s WHERE w > 10) c ON t.k = c.k",
    "SELECT t.id, s.w FROM s JOIN t ON t.k = s.k AND t.v > 500 WHERE t.v % 3 = 0",
    "SELECT s.id, COUNT(*) FROM s JOIN t ON t.k = s.k GROUP BY s.id ORDER BY s.id LIMIT 2",
];

/// Each join's rows on `t` equal its rows on `u`; their count
fn same_as_control(db: &Database, wrong: &mut Vec<String>, label: &str) -> usize {
    let mut found = 0;
    for sql in JOINS {
        let got = rows(db, sql);
        let control = rows(db, &sql.replace(" t ", " u ").replace("t.", "u."));
        if got != control {
            wrong.push(format!(
                "{label}: {sql}: {} rows, control {}",
                got.len(),
                control.len()
            ));
        }
        found += got.len();
    }
    found
}

#[test]
fn keys_of_every_type_answer_as_their_control() {
    let _serial = serial();
    let mut wrong = Vec::new();
    for seal in [false, true] {
        for (inner, outer) in [
            ("TEXT", "TEXT"),
            ("FLOAT", "FLOAT"),
            ("TIMESTAMP", "TIMESTAMP"),
            ("INTEGER", "INTEGER"),
            ("TEXT", "INTEGER"),
            ("INTEGER", "TEXT"),
            ("FLOAT", "INTEGER"),
            ("INTEGER", "FLOAT"),
            ("TIMESTAMP", "TEXT"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let db = tables(&dir, inner, outer, seal);
            let label = format!("{inner} column, {outer} keys, sealed {seal}");
            same_as_control(&db, &mut wrong, &label);
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn float_keys_at_the_edges_answer_as_their_control() {
    let _serial = serial();
    let mut wrong = Vec::new();
    for seal in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = tables(&dir, "FLOAT", "FLOAT", seal);
        for table in ["t", "u"] {
            db.execute(
                &format!(
                    "INSERT INTO {table} VALUES (900001, CAST('NaN' AS FLOAT), 1), \
                     (900002, 0.0, 2), (900003, -0.0, 3)"
                ),
                (),
            )
            .unwrap();
        }
        db.execute(
            "INSERT INTO s VALUES (5, CAST('NaN' AS FLOAT), 50), (6, -0.0, 60)",
            (),
        )
        .unwrap();
        same_as_control(
            &db,
            &mut wrong,
            &format!("NaN and -0.0 keys, sealed {seal}"),
        );
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A grouped join under a LIMIT over a derived inner side finds the row a
/// TIMESTAMP key matches
#[test]
fn a_grouped_join_over_a_derived_side_finds_its_timestamp_row() {
    let _serial = serial();
    let mut wrong = Vec::new();
    for seal in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        for sql in [
            "CREATE TABLE s (id INTEGER PRIMARY KEY, k TIMESTAMP)",
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k TIMESTAMP)",
            "INSERT INTO s VALUES (1, TIMESTAMP '2024-01-01 00:00:01.25')",
            "INSERT INTO t VALUES (1, TIMESTAMP '2024-01-01 00:00:01.25')",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if seal {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (sql, want) in [
            (
                "SELECT s.id, COUNT(r.k) FROM s \
                 LEFT JOIN (SELECT k FROM t WHERE id > 0) r ON s.k = r.k \
                 GROUP BY s.id LIMIT 1",
                vec![vec![1, 1]],
            ),
            (
                "SELECT s.id, COUNT(*) FROM s \
                 JOIN (SELECT k FROM t WHERE id > 0) r ON s.k = r.k \
                 GROUP BY s.id LIMIT 1",
                vec![vec![1, 1]],
            ),
        ] {
            let got = rows(&db, sql);
            if got != want {
                wrong.push(format!("sealed {seal}: {sql}: gave {got:?}, want {want:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A TIMESTAMP literal in an IN over a TEXT column answers the same on the
/// plain and the prepared single-table path
#[test]
fn a_typed_literal_on_a_text_column_answers_alike_on_the_prepared_path() {
    let _serial = serial();
    let db = Database::open("memory://small_outer_typed_literal_text").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT)",
        "INSERT INTO t VALUES (1, '2024-01-01')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let plain = rows(&db, "SELECT id FROM t WHERE k IN (TIMESTAMP '2024-01-01')");
    let prepared = rows(
        &db,
        "SELECT t.id FROM t WHERE t.k IN (TIMESTAMP '2024-01-01') \
         AND t.id = (SELECT x.id FROM t x WHERE x.id = t.id)",
    );
    assert_eq!(plain, vec![vec![1]]);
    assert_eq!(prepared, plain);
}

#[cfg(feature = "test-failpoints")]
mod counted {
    use super::*;
    use stoolap::test_failpoints::small_outer_joins;

    fn narrowed(db: &Database, sql: &str) -> usize {
        let before = small_outer_joins();
        rows(db, sql);
        small_outer_joins() - before
    }

    #[test]
    fn keys_of_the_column_s_type_narrow_and_others_do_not() {
        let _serial = serial();
        let mut wrong = Vec::new();
        for seal in [false, true] {
            for (inner, outer, want) in [
                ("TEXT", "TEXT", 1),
                ("FLOAT", "FLOAT", 1),
                ("TIMESTAMP", "TIMESTAMP", 1),
                ("INTEGER", "INTEGER", 1),
                ("TEXT", "INTEGER", 1),
                ("INTEGER", "TEXT", 0),
                ("INTEGER", "FLOAT", 0),
                ("TIMESTAMP", "TEXT", 0),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let db = tables(&dir, inner, outer, seal);
                let got = narrowed(&db, JOINS[0]);
                if got != want {
                    wrong.push(format!(
                        "{inner} column, {outer} keys, sealed {seal}: narrowed {got}, want {want}"
                    ));
                }
            }
            let dir = tempfile::tempdir().unwrap();
            let db = tables(&dir, "FLOAT", "FLOAT", seal);
            db.execute("INSERT INTO s VALUES (5, CAST('NaN' AS FLOAT), 50)", ())
                .unwrap();
            let got = narrowed(&db, JOINS[0]);
            if got != 0 {
                wrong.push(format!("a NaN key, sealed {seal}: narrowed {got}, want 0"));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} wrong:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }
}
