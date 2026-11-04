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

//! A join whose WHERE keeps no row names its columns and raises the errors
//! its names and its sides' reads raise, reading no table side's rows and
//! compiling nothing through the evaluator bridge once its blocks are
//! prepared

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{after_select_table_opened, bridge_compiles};
use stoolap::Database;

// The families switch and the bridge counter are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The column names and each row, or the error
type Answer = Result<(Vec<String>, Vec<String>), String>;

fn answer(db: &Database, sql: &str) -> Answer {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
    let mut rows = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        rows.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok((columns, rows))
}

fn open(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO u VALUES (1, 10), (3, 30)",
        "CREATE TABLE w (k INTEGER PRIMARY KEY, y INTEGER)",
        "INSERT INTO w VALUES (1, 5)",
        "CREATE VIEW uv AS SELECT id, b FROM u",
        "CREATE VIEW uo AS SELECT id + 9223372036854775807 AS x FROM u",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Each side `u` may be read as, and whether it has the column b
const SIDES: &[(&str, &str, bool)] = &[
    ("", "u", true),
    ("", "uv AS u", true),
    ("WITH c AS (SELECT id, b FROM u) ", "c AS u", true),
    ("", "(SELECT id, b FROM u) AS u", true),
    ("", "(VALUES (1, 10), (3, 30)) AS u(id, b)", true),
    ("", "generate_series(1, 3) AS u(id)", false),
];

/// The columns and rows a statement answers, or a text its error holds
type Written = Result<(&'static [&'static str], &'static [&'static str]), &'static str>;

/// Each statement and what it answers, `u.b` a column only where the side
/// has b
const SHAPES: &[(&str, Written)] = &[
    (
        "SELECT * FROM t JOIN {u} ON u.id = t.id WHERE 1 = 0",
        Ok((&["t.id", "t.a", "u.id", "u.b"], &[])),
    ),
    (
        "SELECT t.id, u.id FROM t LEFT JOIN {u} ON u.id = t.id WHERE 1 = 0",
        Ok((&["id", "id"], &[])),
    ),
    (
        "SELECT t.a FROM t RIGHT JOIN {u} ON u.id = t.id WHERE t.a > 0 AND 1 = 0",
        Ok((&["a"], &[])),
    ),
    (
        "SELECT * FROM t FULL JOIN {u} ON u.id = t.id WHERE FALSE",
        Ok((&["t.id", "t.a", "u.id", "u.b"], &[])),
    ),
    (
        "SELECT COUNT(*) FROM t JOIN {u} ON u.id = t.id WHERE 1 = 0",
        Ok((&["COUNT(*)"], &["Integer(0)"])),
    ),
    (
        "SELECT * FROM t JOIN {u} USING (id) WHERE 1 = 0",
        Ok((&["id", "t.a", "u.b"], &[])),
    ),
    (
        "SELECT * FROM t NATURAL JOIN {u} WHERE 1 = 0",
        Ok((&["id", "t.a", "u.b"], &[])),
    ),
    (
        "SELECT * FROM t JOIN {u} USING (id) JOIN w ON w.k = t.id WHERE 1 = 0",
        Ok((&["id", "t.a", "u.b", "w.k", "w.y"], &[])),
    ),
    (
        "SELECT * FROM w JOIN t ON w.k = t.id JOIN {u} USING (id) WHERE 1 = 0",
        Ok((&["w.k", "w.y", "id", "t.a", "u.b"], &[])),
    ),
    (
        "SELECT * FROM w CROSS JOIN t LEFT JOIN {u} ON u.id = t.id WHERE 1 = 0",
        Ok((&["w.k", "w.y", "t.id", "t.a", "u.id", "u.b"], &[])),
    ),
    (
        "SELECT t.missing FROM t JOIN {u} ON u.id = t.id WHERE 1 = 0",
        Err("column 't.missing' not found"),
    ),
    (
        "SELECT t.id FROM t JOIN {u} ON u.id = t.id WHERE u.missing = 1 AND 1 = 0",
        Err("column 'u.missing' not found"),
    ),
    (
        "SELECT t.id FROM t JOIN {u} ON u.id = t.id WHERE u.id + 9223372036854775807 > 0 AND 1 = 0",
        Ok((&["id"], &[])),
    ),
];

#[test]
fn a_join_that_keeps_no_row_answers_with_its_columns_and_no_row() {
    let _guard = serial();
    let db = open("memory://join_where_false_answers");
    let mut wrong = Vec::new();
    for (with, side, has_b) in SIDES {
        for (shape, want) in SHAPES {
            let sql = format!("{with}{}", shape.replace("{u}", side));
            let got = answer(&db, &sql);
            let fits = match (want, &got) {
                (Ok((columns, rows)), Ok((got_columns, got_rows))) => {
                    let columns: Vec<&str> = (columns.iter().copied())
                        .filter(|c| *has_b || *c != "u.b")
                        .collect();
                    got_columns == &columns && got_rows == rows
                }
                (Err(want), Err(got)) => got.contains(want),
                _ => false,
            };
            if !fits {
                wrong.push(format!("{sql}\n  got  {got:?}\n  want {want:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_join_that_keeps_no_row_reads_its_other_sides_for_their_errors() {
    let _guard = serial();
    let db = open("memory://join_where_false_side_errors");
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT t.id FROM t JOIN (VALUES (9223372036854775807 + 1)) AS v(x) ON TRUE WHERE FALSE",
            "overflow",
        ),
        (
            "SELECT t.id FROM t JOIN u ON t.missing = u.id JOIN w ON w.k = t.id WHERE FALSE",
            "missing",
        ),
        (
            "SELECT t.id FROM t JOIN (SELECT missing FROM u) AS d ON TRUE WHERE FALSE",
            "missing",
        ),
        (
            "SELECT t.id FROM t JOIN (SELECT id + 9223372036854775807 AS x FROM u) AS d ON TRUE WHERE FALSE",
            "overflow",
        ),
    ] {
        match answer(&db, sql) {
            Err(e) if e.to_lowercase().contains(want) => {}
            got => wrong.push(format!("{sql}\n  got  {got:?}\n  want an error naming {want}")),
        }
    }
    // A view's rows are never pulled, so its body raises nothing
    let sql = "SELECT t.id FROM t JOIN uo ON TRUE WHERE FALSE";
    let got = answer(&db, sql);
    if got != Ok((vec!["id".to_string()], Vec::new())) {
        wrong.push(format!("{sql}\n  got  {got:?}\n  want no row"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_join_that_keeps_no_row_raises_schema_changed_for_a_side_read_over_new_columns() {
    let _guard = serial();
    let sql = "SELECT d.* FROM (SELECT * FROM w) AS x JOIN (SELECT * FROM uv) AS d ON TRUE \
               WHERE FALSE";
    let mut wrong = Vec::new();
    for prepared in [false, true] {
        let db = open(&format!("memory://join_where_false_side_schema_{prepared}"));
        let statement = db.prepare(sql).unwrap();
        let run = || -> Result<usize, String> {
            let rows = match prepared {
                true => statement.query(()),
                false => db.query(sql, ()),
            };
            rows.map_err(|e| e.to_string())?
                .map(|row| row.map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, _>>()
                .map(|rows| rows.len())
        };
        run().unwrap();
        let writer = db.clone();
        let done = Rc::new(Cell::new(false));
        let flag = Rc::clone(&done);
        // The view's columns change once the left side opened w
        after_select_table_opened(move || {
            writer.execute("DROP VIEW uv", ()).unwrap();
            writer
                .execute("CREATE VIEW uv AS SELECT id, b, b AS c FROM u", ())
                .unwrap();
            flag.set(true);
        });
        let got = run();
        if !done.get() || !got.as_ref().is_err_and(|e| e.contains("schema changed")) {
            wrong.push(format!(
                "prepared {prepared}: changed {}, {got:?}",
                done.get()
            ));
        }
    }
    assert!(wrong.is_empty(), "{sql}\n{}", wrong.join("\n"));
}

#[test]
fn a_join_that_keeps_no_row_compiles_nothing_through_the_bridge_again() {
    let _guard = serial();
    let db = open("memory://join_where_false_bridge");
    let mut wrong = Vec::new();
    for (with, side, _) in SIDES {
        let sql = format!(
            "{with}SELECT t.id, u.id FROM t JOIN {side} ON u.id = t.id WHERE t.a > 0 AND 1 = 0"
        );
        let statement = db.prepare(&sql).unwrap();
        statement.query(()).unwrap().count();
        let before = bridge_compiles();
        statement.query(()).unwrap().count();
        let compiled = bridge_compiles() - before;
        if compiled != 0 {
            wrong.push(format!("{sql}: run again, {compiled} bridge compiles"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
