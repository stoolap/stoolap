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

//! A prepared statement, the same SQL text and the same statement inside a
//! transaction answer alike, whichever compiled path each one takes

use stoolap::{Database, Value};

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 'a', 1.5), (2, 'b', NULL), (3, NULL, 3.5)",
        (),
    )
    .unwrap();
    db
}

fn collect(rows: stoolap::Result<stoolap::Rows>) -> stoolap::Result<Vec<Vec<Value>>> {
    let mut out = Vec::new();
    let mut rows = rows?;
    while rows.advance() {
        out.push(rows.current_row().as_slice().to_vec());
    }
    match rows.error() {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

fn key(result: &stoolap::Result<Vec<Vec<Value>>>) -> String {
    match result {
        Ok(rows) => format!("{rows:?}"),
        Err(e) => format!("error: {e}"),
    }
}

#[test]
fn point_lookups_answer_alike_on_every_path() {
    let db = open("api_paths_point");
    let sql = "SELECT id, name, score FROM t WHERE id = $1";
    let stmt = db.prepare(sql).unwrap();
    let params = [
        Value::Integer(1),
        Value::Integer(3),
        Value::Integer(9),
        Value::Float(2.0),
        Value::Float(2.5),
        Value::text("2"),
        Value::null_unknown(),
    ];
    for param in params {
        let adhoc = collect(db.query(sql, vec![param.clone()]));
        let prepared = collect(stmt.query(vec![param.clone()]));
        let mut tx = db.begin().unwrap();
        let in_tx = collect(stmt.query_in(&mut tx, vec![param.clone()]));
        let tx_sql = collect(tx.query(sql, vec![param.clone()]));
        tx.rollback().unwrap();
        assert_eq!(key(&prepared), key(&adhoc), "prepared, param {param:?}");
        assert_eq!(
            key(&in_tx),
            key(&adhoc),
            "prepared in a transaction, param {param:?}"
        );
        assert_eq!(
            key(&tx_sql),
            key(&adhoc),
            "SQL in a transaction, param {param:?}"
        );
    }
}

#[test]
fn point_writes_affect_alike_on_every_path() {
    let sqls = [
        "UPDATE t SET score = score + 1 WHERE id = $1",
        "DELETE FROM t WHERE id = $1",
    ];
    for sql in sqls {
        for param in [Value::Integer(2), Value::Integer(9), Value::Float(1.0)] {
            let mut seen = Vec::new();
            for path in 0..3 {
                let db = open(&format!("api_paths_write_{}_{path}", sql.len()));
                let affected = match path {
                    0 => db.execute(sql, vec![param.clone()]),
                    1 => db.prepare(sql).unwrap().execute(vec![param.clone()]),
                    _ => {
                        let stmt = db.prepare(sql).unwrap();
                        let mut tx = db.begin().unwrap();
                        let affected = stmt.execute_in(&mut tx, vec![param.clone()]);
                        tx.commit().unwrap();
                        affected
                    }
                };
                let table = collect(db.query("SELECT id, name, score FROM t ORDER BY id", ()));
                seen.push(format!("{affected:?} {}", key(&table)));
            }
            assert_eq!(seen[1], seen[0], "prepared: {sql} with {param:?}");
            assert_eq!(seen[2], seen[0], "in a transaction: {sql} with {param:?}");
        }
    }
}

#[test]
fn a_statement_prepared_on_another_database_reads_the_transaction_s_own() {
    let here = open("api_paths_here");
    let there = Database::open("memory://api_paths_there").unwrap();
    there
        .execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, score FLOAT, name TEXT)",
            (),
        )
        .unwrap();
    there
        .execute("INSERT INTO t VALUES (1, 0.0, 'there')", ())
        .unwrap();
    let stmt = here.prepare("SELECT name FROM t WHERE id = $1").unwrap();
    let mut tx = here.begin().unwrap();
    let rows = collect(stmt.query_in(&mut tx, (1,))).unwrap();
    assert_eq!(rows, vec![vec![Value::text("a")]]);
    tx.commit().unwrap();
    let mut tx = there.begin().unwrap();
    let rows = collect(stmt.query_in(&mut tx, (1,))).unwrap();
    assert_eq!(rows, vec![vec![Value::text("there")]]);
    tx.commit().unwrap();
}

#[test]
fn get_by_name_matches_names_without_regard_to_case() {
    let db = Database::open("memory://api_paths_names").unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, \"Ärger\" TEXT, Name TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, 'x', 'y')", ())
        .unwrap();
    for row in db.query("SELECT id, \"Ärger\", Name FROM u", ()).unwrap() {
        let row = row.unwrap();
        assert_eq!(row.get_by_name::<String>("NAME").unwrap(), "y");
        assert_eq!(row.get_by_name::<String>("name").unwrap(), "y");
        assert_eq!(row.get_by_name::<String>("äRGER").unwrap(), "x");
        assert_eq!(row.get_by_name::<i64>("ID").unwrap(), 1);
        assert!(row.get_by_name::<String>("missing").is_err());
    }
}
