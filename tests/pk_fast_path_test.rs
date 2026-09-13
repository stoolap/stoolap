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

//! The primary-key fast path must give the same answer as the general
//! planner for every statement shape it accepts, including a plain column
//! list, which it now takes instead of only SELECT *.

use stoolap::Database;

fn seed(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 'ann', 31, 1.5), (2, 'bob', 42, 2.5), (3, 'cy', 53, 3.5)",
        (),
    )
    .unwrap();
}

fn one_row(db: &Database, sql: &str) -> (Vec<String>, Vec<String>) {
    let mut rows = db.query(sql, ()).unwrap();
    let columns: Vec<String> = rows.columns().iter().map(|c| c.to_string()).collect();
    let row = rows.next().unwrap().unwrap();
    let values = (0..columns.len())
        .map(|i| format!("{:?}", row.get::<stoolap::Value>(i).unwrap()))
        .collect();
    assert!(rows.next().is_none(), "one row expected for {sql}");
    (columns, values)
}

#[test]
fn projected_pk_lookup_returns_only_the_named_columns_in_order() {
    let db = Database::open("memory://pk_proj_order").unwrap();
    seed(&db);
    let (columns, values) = one_row(&db, "SELECT age, name FROM t WHERE id = 2");
    assert_eq!(columns, vec!["age", "name"]);
    assert_eq!(values, vec!["Integer(42)", "Text(\"bob\")"]);
}

#[test]
fn projected_pk_lookup_matches_star_lookup_column_by_column() {
    let db = Database::open("memory://pk_proj_star").unwrap();
    seed(&db);
    let (star_cols, star_vals) = one_row(&db, "SELECT * FROM t WHERE id = 3");
    let (cols, vals) = one_row(&db, "SELECT score, id, name, age FROM t WHERE id = 3");
    for (c, v) in cols.iter().zip(&vals) {
        let i = star_cols.iter().position(|s| s == c).unwrap();
        assert_eq!(&star_vals[i], v, "column {c}");
    }
}

#[test]
fn projected_pk_lookup_repeats_a_column() {
    let db = Database::open("memory://pk_proj_dup").unwrap();
    seed(&db);
    let (columns, values) = one_row(&db, "SELECT name, name FROM t WHERE id = 1");
    assert_eq!(columns, vec!["name", "name"]);
    assert_eq!(values, vec!["Text(\"ann\")", "Text(\"ann\")"]);
}

#[test]
fn projected_pk_lookup_with_unknown_column_is_an_error() {
    let db = Database::open("memory://pk_proj_unknown").unwrap();
    seed(&db);
    assert!(db.query("SELECT nope FROM t WHERE id = 1", ()).is_err());
}

#[test]
fn projected_pk_lookup_reads_a_sealed_row() {
    let dir = std::env::temp_dir().join(format!("pk_proj_cold_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    seed(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 'dee', 64, 4.5)", ())
        .unwrap();
    let (columns, values) = one_row(&db, "SELECT name, score FROM t WHERE id = 2");
    assert_eq!(columns, vec!["name", "score"]);
    assert_eq!(values, vec!["Text(\"bob\")", "Float(2.5)"]);
    let (_, hot) = one_row(&db, "SELECT age FROM t WHERE id = 4");
    assert_eq!(hot, vec!["Integer(64)"]);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn projected_pk_lookup_sees_an_uncommitted_update_in_its_transaction() {
    let db = Database::open("memory://pk_proj_txn").unwrap();
    seed(&db);
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET age = 99 WHERE id = 1", ())
        .unwrap();
    let (_, values) = one_row(&db, "SELECT age FROM t WHERE id = 1");
    assert_eq!(values, vec!["Integer(99)"]);
    db.execute("ROLLBACK", ()).unwrap();
    let (_, values) = one_row(&db, "SELECT age FROM t WHERE id = 1");
    assert_eq!(values, vec!["Integer(31)"]);
}

fn seed_with_history(db: &Database) {
    db.execute(
        "CREATE TABLE h (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO h VALUES (1, 'ann', 31)", ())
        .unwrap();
    db.execute("UPDATE h SET age = 99 WHERE id = 1", ())
        .unwrap();
}

#[test]
fn star_pk_lookup_with_as_of_reads_the_historical_row() {
    let db = Database::open("memory://pk_as_of_star").unwrap();
    seed_with_history(&db);
    let (_, now) = one_row(&db, "SELECT * FROM h WHERE id = 1");
    assert_eq!(now[2], "Integer(99)");
    let (_, then) = one_row(&db, "SELECT * FROM h AS OF TRANSACTION 1 WHERE id = 1");
    assert_eq!(
        then[2], "Integer(31)",
        "AS OF must not take the PK fast path"
    );
}

#[test]
fn projected_pk_lookup_with_as_of_reads_the_historical_row() {
    let db = Database::open("memory://pk_as_of_proj").unwrap();
    seed_with_history(&db);
    let (_, now) = one_row(&db, "SELECT age FROM h WHERE id = 1");
    assert_eq!(now, vec!["Integer(99)"]);
    let (_, then) = one_row(&db, "SELECT age FROM h AS OF TRANSACTION 1 WHERE id = 1");
    assert_eq!(
        then,
        vec!["Integer(31)"],
        "AS OF must not take the PK fast path"
    );
}

#[test]
fn projected_pk_lookup_keeps_the_query_spelling_of_column_names() {
    let db = Database::open("memory://pk_proj_spelling").unwrap();
    seed(&db);
    let (fast, _) = one_row(&db, "SELECT \"NAME\", Age FROM t WHERE id = 1");
    let (general, _) = one_row(&db, "SELECT \"NAME\", Age FROM t WHERE name = 'ann'");
    assert_eq!(fast[0], "NAME", "a quoted identifier keeps its case");
    assert_eq!(
        fast, general,
        "both paths must name the columns the same way"
    );
}
