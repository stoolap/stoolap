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

//! An error a WHERE raises on a row it reaches is the statement's error,
//! wherever the filter is evaluated: OR and NOT follow the expression
//! evaluator, left to right, stopping at the first TRUE

use stoolap::Database;

const ERR: &str = "error";

fn answer(db: &Database, sql: &str) -> String {
    if sql.starts_with("SELECT") {
        match db.query(sql, ()).and_then(|rows| {
            rows.map(|row| row.map(|row| format!("{:?}", row.as_row().as_slice())))
                .collect::<stoolap::Result<Vec<_>>>()
        }) {
            Ok(rows) => rows.join(" "),
            Err(_) => ERR.to_string(),
        }
    } else {
        match db.execute(sql, ()) {
            Ok(n) => format!("affected {n}"),
            Err(_) => ERR.to_string(),
        }
    }
}

const CASES: [(&str, &str); 15] = [
    ("SELECT id FROM t WHERE ABS(x) >= 0", ERR),
    ("SELECT id FROM t WHERE ABS(x) >= 0 LIMIT 2", ERR),
    (
        "SELECT id FROM t WHERE ABS(x) >= 0 ORDER BY id LIMIT 1",
        ERR,
    ),
    ("SELECT COUNT(*) FROM t WHERE ABS(x) >= 0", ERR),
    ("SELECT id FROM t WHERE id IN (1, 2) AND ABS(x) >= 0", ERR),
    ("SELECT id FROM t WHERE k = 1 AND ABS(x) >= 0", ERR),
    (
        "SELECT id FROM t WHERE k = 1 AND ABS(x) >= 0 ORDER BY o LIMIT 1",
        ERR,
    ),
    ("SELECT id FROM t WHERE ABS(x) >= 0 OR id = 1", ERR),
    ("SELECT id FROM t WHERE NOT (ABS(x) < 0)", ERR),
    (
        "SELECT id FROM t WHERE id = 1 OR ABS(x) >= 0 ORDER BY id",
        "[Integer(1)] [Integer(2)]",
    ),
    (
        "SELECT id FROM t WHERE id = 2 AND ABS(x) >= 0",
        "[Integer(2)]",
    ),
    (
        "SELECT id FROM t WHERE CAST(s AS INTEGER) > 0",
        "[Integer(2)]",
    ),
    ("UPDATE t SET k = 0 WHERE ABS(x) >= 0", ERR),
    ("DELETE FROM t WHERE ABS(x) >= 0", ERR),
    ("SELECT COUNT(*) FROM t WHERE k > 0", "[Integer(2)]"),
];

fn filled(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, s TEXT, k INTEGER, o INTEGER NOT NULL)",
        "CREATE INDEX tk ON t(k, o)",
        "INSERT INTO t VALUES (1, -9223372036854775808, 'a', 1, 1), (2, 5, '2', 2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

#[test]
fn a_filter_error_is_the_statements_on_hot_and_sealed_rows() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            Database::open(&format!("file://{}", dir.path().display())).unwrap()
        } else {
            Database::open("memory://storage_filter_error_rows").unwrap()
        };
        filled(&db);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (sql, expected) in CASES {
            let got = answer(&db, sql);
            if got != expected {
                wrong.push(format!("sealed {sealed}: {sql} gave {got}"));
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

#[test]
fn a_filter_error_is_the_statements_on_a_transactions_own_rows() {
    let db = Database::open("memory://storage_filter_error_own_rows").unwrap();
    filled(&db);
    db.execute("DELETE FROM t", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, -9223372036854775808, 'a', 1, 1), (2, 5, '2', 2, 2)",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for (sql, expected) in CASES {
        if expected == ERR && sql.starts_with("SELECT") {
            let got = answer(&db, sql);
            if got != expected {
                wrong.push(format!("{sql} gave {got}"));
            }
        }
    }
    db.execute("ROLLBACK", ()).unwrap();
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_filter_error_is_the_statements_as_of_a_transaction() {
    let db = Database::open("memory://storage_filter_error_as_of").unwrap();
    filled(&db);
    for as_of in ["TIMESTAMP '2200-01-01 00:00:00'", "TRANSACTION 100"] {
        let all = answer(&db, &format!("SELECT id FROM t AS OF {as_of}"));
        let some = answer(
            &db,
            &format!("SELECT id FROM t AS OF {as_of} WHERE ABS(x) >= 0"),
        );
        assert_eq!(
            (all.as_str(), some.as_str()),
            ("[Integer(1)] [Integer(2)]", ERR),
            "{as_of}"
        );
    }
}
