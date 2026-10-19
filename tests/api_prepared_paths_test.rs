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

#[test]
fn point_writes_inside_a_transaction_see_and_undo_its_own_rows() {
    let db = open("api_paths_tx_writes");
    let update = db.prepare("UPDATE t SET score = $1 WHERE id = $2").unwrap();
    let delete = db.prepare("DELETE FROM t WHERE id = $1").unwrap();
    let table = |db: &Database| {
        key(&collect(
            db.query("SELECT id, name, score FROM t ORDER BY id", ()),
        ))
    };
    let committed = table(&db);

    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (4, 'd', 4.0)", ())
        .unwrap();
    assert_eq!(
        update.execute_in(&mut tx, (40.0, 4)).unwrap(),
        1,
        "own insert"
    );
    assert_eq!(update.execute_in(&mut tx, (10.0, 1)).unwrap(), 1);
    assert_eq!(delete.execute_in(&mut tx, (2,)).unwrap(), 1);
    assert_eq!(
        delete.execute_in(&mut tx, (2,)).unwrap(),
        0,
        "deleted twice"
    );
    let inside = collect(tx.query("SELECT id, score FROM t ORDER BY id", ())).unwrap();
    assert_eq!(
        inside,
        vec![
            vec![Value::Integer(1), Value::Float(10.0)],
            vec![Value::Integer(3), Value::Float(3.5)],
            vec![Value::Integer(4), Value::Float(40.0)],
        ]
    );
    assert_eq!(table(&db), committed, "uncommitted writes leaked");
    tx.rollback().unwrap();
    assert_eq!(table(&db), committed, "rollback kept a write");

    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET score = 9.0 WHERE id = 1", ())
        .unwrap();
    db.execute("SAVEPOINT s", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    db.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    let score: f64 = db
        .query_one("SELECT score FROM t WHERE id = 1", ())
        .unwrap();
    assert_eq!(score, 9.0);
}

#[test]
fn two_snapshot_transactions_writing_one_row_conflict() {
    let db = open("api_paths_tx_conflict");
    let other = db.clone();
    let update = db.prepare("UPDATE t SET score = $1 WHERE id = $2").unwrap();
    let mut first = db
        .begin_with_isolation(stoolap::IsolationLevel::SnapshotIsolation)
        .unwrap();
    let mut second = other
        .begin_with_isolation(stoolap::IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(update.execute_in(&mut first, (1.0, 1)).unwrap(), 1);
    let clash = update
        .execute_in(&mut second, (2.0, 1))
        .and_then(|_| second.commit());
    first.commit().unwrap();
    assert!(clash.is_err(), "both writes of one row committed");
    let score: f64 = db
        .query_one("SELECT score FROM t WHERE id = 1", ())
        .unwrap();
    assert_eq!(score, 1.0);
}

#[test]
fn named_parameters_on_a_prepared_statement_answer_as_the_sql_text_does() {
    let db = open("api_paths_named");
    let sql = "SELECT id, name FROM t WHERE id = :id";
    let stmt = db.prepare(sql).unwrap();
    for id in [1i64, 3, 9] {
        let adhoc = collect(db.query_named(sql, stoolap::named_params! { id: id }));
        let prepared = collect(stmt.query_named(stoolap::named_params! { id: id }));
        assert_eq!(key(&prepared), key(&adhoc), "id {id}");
    }
    let update = db
        .prepare("UPDATE t SET score = :s WHERE id = :id")
        .unwrap();
    let affected = update
        .execute_named(stoolap::named_params! { s: 5.0, id: 2 })
        .unwrap();
    assert_eq!(affected, 1);
    let score: f64 = db
        .query_one("SELECT score FROM t WHERE id = 2", ())
        .unwrap();
    assert_eq!(score, 5.0);
}

/// Two databases with the same DDL count, so their schema epochs match,
/// and `t.v` of a different type in each
fn twins(name: &str, here_v: &str, there_v: &str) -> (Database, Database) {
    let open_one = |side: &str, v_type: &str| {
        let db = Database::open(&format!("memory://{name}_{side}")).unwrap();
        db.execute(
            &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, v {v_type})"),
            (),
        )
        .unwrap();
        db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO u VALUES (1)", ()).unwrap();
        db
    };
    (open_one("here", here_v), open_one("there", there_v))
}

#[test]
fn a_statement_from_another_database_is_planned_for_the_transaction_s_tables() {
    let (here, there) = twins("api_paths_rebind_update", "INTEGER", "TEXT");
    here.execute("INSERT INTO t VALUES (1, 5)", ()).unwrap();
    there.execute("INSERT INTO t VALUES (1, 'x')", ()).unwrap();
    let update = here.prepare("UPDATE t SET v = $1 WHERE id = $2").unwrap();
    update.execute((6, 1)).unwrap();
    let mut tx = there.begin().unwrap();
    assert_eq!(update.execute_in(&mut tx, ("foo", 1)).unwrap(), 1);
    tx.commit().unwrap();
    let v: String = there.query_one("SELECT v FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(v, "foo");

    let (here, there) = twins("api_paths_rebind_select", "BOOLEAN", "TEXT");
    here.execute("INSERT INTO t VALUES (1, true)", ()).unwrap();
    there
        .execute("INSERT INTO t VALUES (1, 'foo')", ())
        .unwrap();
    let select = here
        .prepare("SELECT v FROM t WHERE v = $1 AND EXISTS (SELECT 1 FROM u WHERE id = 1)")
        .unwrap();
    assert_eq!(collect(select.query(("true",))).unwrap().len(), 1);
    let mut tx = there.begin().unwrap();
    let rows = collect(select.query_in(&mut tx, ("foo",))).unwrap();
    tx.commit().unwrap();
    assert_eq!(rows, vec![vec![Value::text("foo")]]);
}

#[test]
fn current_transaction_id_reads_the_open_transaction_on_every_path() {
    let db = Database::open("memory://api_paths_txn_id").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut tx = db.begin().unwrap();
    let id = tx.id();
    tx.execute("UPDATE t SET v = CURRENT_TRANSACTION_ID() WHERE id = 1", ())
        .unwrap();
    tx.execute("INSERT INTO t SELECT 2, CURRENT_TRANSACTION_ID()", ())
        .unwrap();
    tx.execute("INSERT INTO t VALUES (3, CURRENT_TRANSACTION_ID())", ())
        .unwrap();
    let rows = collect(tx.query("SELECT id, v FROM t ORDER BY id", ())).unwrap();
    tx.commit().unwrap();
    let ids: Vec<Value> = rows.iter().map(|row| row[1].clone()).collect();
    assert_eq!(ids, vec![Value::Integer(id); 3], "{rows:?}");
}

#[test]
fn a_missing_named_update_parameter_writes_what_the_general_path_writes() {
    // RETURNING keeps the statement off the compiled PK path
    let general = "UPDATE t SET v = :v WHERE id = :id RETURNING id";
    let compiled = "UPDATE t SET v = :v WHERE id = :id";
    let outcome = |name: &str, sql: &str, in_tx: bool| {
        let db = Database::open(&format!("memory://api_paths_missing_{name}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        let mut seen = Vec::new();
        // The second run takes the plan compiled by the first
        for _ in 0..2 {
            db.execute("DELETE FROM t", ()).unwrap();
            db.execute("INSERT INTO t VALUES (1, 9)", ()).unwrap();
            let affected = if in_tx {
                let mut tx = db.begin().unwrap();
                let affected = tx.execute_named(sql, stoolap::named_params! { id: 1 });
                tx.commit().unwrap();
                affected
            } else {
                db.execute_named(sql, stoolap::named_params! { id: 1 })
            };
            let table = key(&collect(db.query("SELECT id, v FROM t", ())));
            assert!(affected.is_ok(), "{name}: {affected:?}");
            seen.push(table);
        }
        seen
    };
    let expected = outcome("general", general, false);
    assert!(
        expected[1].contains("Null"),
        "the general path wrote {expected:?}"
    );
    assert_eq!(
        outcome("autocommit", compiled, false),
        expected,
        "autocommit"
    );
    assert_eq!(
        outcome("transaction", compiled, true),
        expected,
        "transaction"
    );
}
