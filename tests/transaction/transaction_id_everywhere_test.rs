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

//! Every program a statement runs reads the statement's parameters and its
//! transaction: CURRENT_TRANSACTION_ID() inside a transaction is its id in
//! every clause, an aggregate's argument and FILTER included, over every
//! source, hot and sealed, with a parameter beside it or not

use stoolap::{Database, Value};

/// Each statement over `{s}`, a source of t's rows named t, and `{x}`, the
/// call; each answers one row whose one value is `{x}`'s, or the count
/// of rows a condition on `{x}` keeps
const SHAPES: &[&str] = &[
    "SELECT MAX({x}) FROM {s}",
    "SELECT MIN({x} + t.a - t.a) FROM {s} WHERE t.a > 0",
    "SELECT MAX(m) FROM (SELECT t.g, MAX({x}) AS m FROM {s} GROUP BY t.g) q",
    "SELECT COUNT(*) FILTER (WHERE {x} > 0) - 3 + {x} FROM {s}",
    "SELECT MAX({x}) FROM {s} GROUP BY t.g HAVING MAX({x}) = {x} ORDER BY 1 LIMIT 1",
    "SELECT COUNT(DISTINCT {x}) - 1 + MIN({x}) FROM {s}",
    "SELECT CAST(MAX(CAST({x} AS TEXT)) AS INTEGER) FROM {s}",
    "SELECT SUM({x}) / COUNT(*) FROM {s}",
    "SELECT MAX({x}) FROM {s} JOIN u ON u.tid = t.id",
    "SELECT {x} FROM {s} WHERE t.id = 1",
    "SELECT MAX({x} + $1 - $1) FROM {s}",
    "SELECT (SELECT MAX({x}) FROM {s})",
    "WITH c AS (SELECT MAX({x}) AS m FROM {s}) SELECT m FROM c",
    "SELECT MAX(m) FROM (SELECT MAX({x}) AS m FROM {s} UNION ALL SELECT MAX({x}) FROM u) q",
    "SELECT MAX({x}) OVER () FROM {s} ORDER BY 1 LIMIT 1",
];

/// Each source of t's rows under the name t
const SOURCES: &[&str] = &[
    "t",
    "tv t",
    "(SELECT id, a, g FROM t) t",
    "(VALUES (1, 1, 1), (2, 2, 1), (3, 3, 2)) AS t(id, a, g)",
];

fn open(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, g INTEGER)",
        "INSERT INTO t VALUES (1, 1, 1), (2, 2, 1), (3, 3, 2)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 3)",
        "CREATE VIEW tv AS SELECT id, a, g FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn answer(rows: stoolap::Result<stoolap::Rows>) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for row in rows.map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(row.get_value(0).unwrap().clone());
    }
    Ok(out)
}

#[test]
fn a_statement_reads_its_transaction_in_every_program() {
    let dir = tempfile::tempdir().unwrap();
    let sealed = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let mut wrong = Vec::new();
    let mut count = 0;
    for (context, db) in [
        ("hot", open("memory://transaction_id_everywhere", false)),
        ("sealed", open(&sealed, true)),
    ] {
        let mut tx = db.begin().unwrap();
        let want = Ok(vec![Value::Integer(tx.id())]);
        for source in SOURCES {
            for shape in SHAPES {
                let sql = shape
                    .replace("{s}", source)
                    .replace("{x}", "CURRENT_TRANSACTION_ID()");
                count += 1;
                let got = answer(tx.query(&sql, vec![Value::Integer(7)]));
                if got != want {
                    wrong.push(format!("{context}: {sql}: {got:?}, want {want:?}"));
                }
            }
        }
        tx.rollback().unwrap();
    }
    assert!(count > 100, "{count} statements");
    assert!(
        wrong.is_empty(),
        "{} of {count} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_statement_outside_a_transaction_reads_null_in_every_program() {
    let db = open("memory://transaction_id_everywhere_autocommit", false);
    for shape in SHAPES {
        let sql = shape
            .replace("{s}", "t")
            .replace("{x}", "CURRENT_TRANSACTION_ID()");
        let got = answer(db.query(&sql, vec![Value::Integer(7)]));
        let null = got
            .as_ref()
            .is_ok_and(|values| values.iter().all(Value::is_null));
        assert!(null, "{sql}: {got:?}");
    }
}
