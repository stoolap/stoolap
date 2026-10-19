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

//! A STRING_AGG or GROUP_CONCAT separator is one value for the statement,
//! read as CONCAT_WS reads its separator: a NULL one gives NULL, and one
//! that reads the row is an error.

use stoolap::{Database, Value};

const SEPARATORS: [&str; 9] = [
    "' | '",
    "''",
    "1",
    "1.5",
    "TRUE",
    "NULL",
    "TIMESTAMP '2024-01-01 12:00:00'",
    "'a' || 'b'",
    "$1",
];

/// Every statement shape a separator is read in, with `sep` in it
const SHAPES: [&str; 8] = [
    "SELECT STRING_AGG(b, {sep}) FROM t",
    "SELECT STRING_AGG(b, {sep} ORDER BY id) FROM t",
    "SELECT GROUP_CONCAT(b, {sep} ORDER BY id) FROM t",
    "SELECT g, STRING_AGG(b, {sep} ORDER BY id) FROM t GROUP BY g",
    "SELECT g, STRING_AGG(b, {sep} ORDER BY id) FROM t GROUP BY g HAVING COUNT(*) > 1",
    "SELECT a FROM (SELECT STRING_AGG(b, {sep} ORDER BY id) AS a FROM t) d",
    "SELECT STRING_AGG(b, {sep} ORDER BY id) OVER (PARTITION BY g) FROM t ORDER BY id LIMIT 1",
    "SELECT STRING_AGG(b, {sep}) FROM t WHERE b <> 'z'",
];

fn last_of_first_row(rows: stoolap::Result<stoolap::Rows>) -> Result<Value, String> {
    let mut rows = rows.map_err(|e| e.to_string())?;
    let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
    Ok(row.get_value(row.len() - 1).cloned().unwrap())
}

fn create(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, b TEXT, sep TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 'x', '-'), (2, 1, 'y', '-')",
        (),
    )
    .unwrap();
}

fn check(db: &Database, storage: &str, wrong: &mut Vec<String>) {
    for sep in SEPARATORS {
        let want =
            last_of_first_row(db.query(&format!("SELECT CONCAT_WS({sep}, 'x', 'y')"), ("+",)));
        for shape in SHAPES {
            let sql = shape.replace("{sep}", sep);
            let got = last_of_first_row(db.query(&sql, ("+",)));
            if got != want {
                wrong.push(format!("{storage}: {sql} gave {got:?}, want {want:?}"));
            }
        }
    }
    for shape in SHAPES {
        let sql = shape.replace("{sep}", "sep");
        if let Ok(got) = last_of_first_row(db.query(&sql, ())) {
            wrong.push(format!("{storage}: {sql} gave {got:?}, want an error"));
        }
    }
    for shape in SHAPES {
        let sql = shape.replace("{sep}", "$1");
        let prepared = db.prepare(&sql).unwrap();
        for sep in ["+", "-"] {
            let got = last_of_first_row(prepared.query((sep,)));
            let want = Ok(Value::text(format!("x{sep}y")));
            if got != want {
                wrong.push(format!(
                    "{storage}: prepared {sql} with {sep} gave {got:?}, want {want:?}"
                ));
            }
        }
    }
}

#[test]
fn test_a_separator_is_read_as_concat_ws_reads_it() {
    let mut wrong = Vec::new();
    let memory = Database::open("memory://string_agg_separator").unwrap();
    create(&memory);
    check(&memory, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    create(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed", &mut wrong);

    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_separator_names_its_column_as_written() {
    let db = Database::open("memory://string_agg_separator_names").unwrap();
    create(&db);
    let mut wrong = Vec::new();
    for sep in [
        "' | '",
        "'it''s'",
        "TIMESTAMP '2024-01-01 12:00:00'",
        "'a' || 'b'",
        "$1",
    ] {
        let sql = format!("SELECT STRING_AGG(b, {sep}) FROM t");
        let rows = db.query(&sql, ("+",)).unwrap();
        let want = format!("STRING_AGG(b, {sep})");
        if rows.columns() != [want.as_str()] {
            wrong.push(format!("{sql} named {:?}, want {want}", rows.columns()));
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
fn test_a_grouped_index_join_reads_the_separator_parameter() {
    let db = Database::open("memory://string_agg_separator_join").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER, b TEXT)",
        "CREATE INDEX c_pid ON c (pid)",
        "INSERT INTO p VALUES (1), (2)",
        "INSERT INTO c VALUES (1, 1, 'x'), (2, 1, 'x'), (3, 2, 'x'), (4, 2, 'x')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql =
        "SELECT p.id, STRING_AGG(c.b, $1) FROM p JOIN c ON p.id = c.pid GROUP BY p.id LIMIT 1";
    let mut wrong = Vec::new();
    let got = last_of_first_row(db.query(sql, ("|",)));
    if got != Ok(Value::text("x|x")) {
        wrong.push(format!("{sql} gave {got:?}"));
    }
    let prepared = db.prepare(sql).unwrap();
    for sep in ["+", "-"] {
        let got = last_of_first_row(prepared.query((sep,)));
        if got != Ok(Value::text(format!("x{sep}x"))) {
            wrong.push(format!("prepared {sql} with {sep} gave {got:?}"));
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
fn test_a_separator_reads_the_transaction() {
    let db = Database::open("memory://string_agg_separator_txn").unwrap();
    create(&db);
    db.execute("BEGIN", ()).unwrap();
    let want =
        last_of_first_row(db.query("SELECT CONCAT_WS(CURRENT_TRANSACTION_ID(), 'x', 'y')", ()));
    let got = last_of_first_row(db.query(
        "SELECT STRING_AGG(b, CURRENT_TRANSACTION_ID() ORDER BY id) FROM t",
        (),
    ));
    db.execute("COMMIT", ()).unwrap();
    assert!(want.is_ok() && got == want, "gave {got:?}, want {want:?}");
}

#[test]
fn test_a_window_separator_is_one_value_for_every_partition() {
    let db = Database::open("memory://string_agg_separator_partitions").unwrap();
    create(&db);
    db.execute(
        "INSERT INTO t VALUES (3, 2, 'x', '-'), (4, 2, 'y', '-')",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t (g)", ()).unwrap();
    let sql = "SELECT g, STRING_AGG(b, NOW() ORDER BY id) OVER (PARTITION BY g) FROM t LIMIT 4";
    let separators: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let joined: String = row.unwrap().get(1).unwrap();
            joined[1..joined.len() - 1].to_string()
        })
        .collect();
    assert_eq!(separators.len(), 4, "{sql}");
    assert!(
        separators.iter().all(|sep| sep == &separators[0]),
        "{sql} used {separators:?}"
    );
}

#[test]
fn test_a_window_separator_error_is_reported_by_a_row_that_reads_it() {
    let db = Database::open("memory://string_agg_separator_empty").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t (g)", ()).unwrap();
    let windows = [
        "SELECT STRING_AGG(b, $1 + 1) OVER () FROM t",
        "SELECT STRING_AGG(b, $1 + 1 ORDER BY id) OVER (PARTITION BY g) FROM t",
        "SELECT g, STRING_AGG(b, $1 + 1) OVER (PARTITION BY g) FROM t LIMIT 4",
    ];
    let mut wrong = Vec::new();
    let mut run = |sql: &str, want_error: bool| {
        let got = db.query(sql, (i64::MAX,)).and_then(|rows| {
            rows.collect::<stoolap::Result<Vec<_>>>()
                .map(|rows| rows.len())
        });
        if got.is_err() != want_error {
            wrong.push(format!("{sql} gave {got:?}, want an error: {want_error}"));
        }
    };
    for sql in windows {
        run(sql, false);
    }
    db.execute("INSERT INTO t VALUES (1, 1, 'x'), (2, 2, 'y')", ())
        .unwrap();
    for sql in windows {
        run(sql, true);
    }
    drop(run);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_separator_error_is_reported_by_a_group_that_reads_it() {
    let db = Database::open("memory://string_agg_separator_groups").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, b TEXT)",
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER, b TEXT)",
        "CREATE INDEX c_pid ON c (pid)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let grouped = [
        "SELECT g, STRING_AGG(b, $1 + 1) FROM t GROUP BY g",
        "SELECT g, STRING_AGG(b, $1 + 1 ORDER BY id) FROM t GROUP BY g",
        "SELECT g, STRING_AGG(b, $1 + 1) FROM t GROUP BY g LIMIT 1",
        "SELECT p.id, STRING_AGG(c.b, $1 + 1) FROM p JOIN c ON p.id = c.pid GROUP BY p.id LIMIT 1",
    ];
    let mut wrong = Vec::new();
    let mut run = |sql: &str, want_error: bool| {
        let got = db.query(sql, (i64::MAX,)).and_then(|rows| {
            rows.collect::<stoolap::Result<Vec<_>>>()
                .map(|rows| rows.len())
        });
        if got.is_err() != want_error {
            wrong.push(format!("{sql} gave {got:?}, want an error: {want_error}"));
        }
    };
    for sql in grouped {
        run(sql, false);
    }
    run("SELECT STRING_AGG(b, $1 + 1) FROM t", true);
    for sql in [
        "INSERT INTO t VALUES (1, 1, 'x')",
        "INSERT INTO p VALUES (1)",
        "INSERT INTO c VALUES (1, 1, 'x')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for sql in grouped {
        run(sql, true);
    }
    drop(run);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
