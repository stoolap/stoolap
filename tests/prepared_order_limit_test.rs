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

//! A prepared statement with ORDER BY and LIMIT answers as the general path
//! does, whether it chooses its rows before or after projecting them

use stoolap::Database;

const PREPARED: &str = "(SELECT v FROM u WHERE u.id = 1)";
// An ORDER BY position in the block keeps the statement out of the family
const GENERAL: &str = "(SELECT v FROM u WHERE u.id = 1 ORDER BY 1)";

fn setup(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, c FLOAT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 0.5)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, CASE WHEN value % 7 = 0 THEN NULL ELSE value % 5 END, \
         'b' || (value % 3), (value * 37 % 100) / 10.0 FROM generate_series(1, 400)",
        (),
    )
    .unwrap();
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), (1,))
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

/// Each row's values, and the values of the ORDER BY keys in row order
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, (3,)).map_err(|e| e.to_string())?;
    rows.map(|row| {
        let row = row.map_err(|e| e.to_string())?;
        Ok(format!(
            "{:?}",
            (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
        ))
    })
    .collect()
}

fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    for shape in [
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a, id LIMIT 10",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a DESC, id DESC LIMIT 10",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a NULLS FIRST, id LIMIT 10",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a DESC NULLS LAST, id LIMIT 10",
        "SELECT id, b, a FROM t WHERE c > {sq} ORDER BY b, a DESC, id LIMIT 7 OFFSET 4",
        "SELECT id, c FROM t WHERE c > {sq} ORDER BY c DESC, id LIMIT 5",
        "SELECT id, a AS x FROM t WHERE c > {sq} ORDER BY x, id LIMIT 6",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY 2, 1 LIMIT 6",
        "SELECT id FROM t WHERE c > {sq} ORDER BY a, id LIMIT 6",
        "SELECT id, a + 1 AS a1 FROM t WHERE c > {sq} ORDER BY a, id LIMIT 6",
        "SELECT * FROM t WHERE c > {sq} ORDER BY b DESC, id LIMIT 6",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a, id LIMIT 0",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a, id LIMIT 5 OFFSET 1000",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a, id LIMIT $1",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY a, id LIMIT $1 OFFSET 2",
        "SELECT id, a FROM t WHERE c > {sq} ORDER BY id DESC LIMIT 4",
    ] {
        let prepared = shape.replace("{sq}", PREPARED);
        let general = shape.replace("{sq}", GENERAL);
        if prepared_line(db, &prepared) != "Prepared: yes" {
            wrong.push(format!("{phase}: not prepared: {prepared}"));
        }
        if !prepared_line(db, &general).starts_with("Prepared: no") {
            wrong.push(format!("{phase}: prepared: {general}"));
        }
        let (got, want) = (answer(db, &prepared), answer(db, &general));
        if got != want {
            wrong.push(format!("{phase}: {shape}\n  got  {got:?}\n  want {want:?}"));
        }
    }
}

#[test]
fn a_prepared_order_by_with_a_limit_answers_as_the_general_path() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://prepared_order_limit_hot").unwrap();
    setup(&db);
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 9 WHERE id % 50 = 0", ())
        .unwrap();
    check(&db, "sealed with hot updates", &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_computed_item_of_a_row_the_limit_drops_raises_no_error() {
    let db = Database::open("memory://prepared_order_limit_errors").unwrap();
    setup(&db);
    // The product overflows for a > 1 only
    let item = |order: &str| {
        answer(
            &db,
            &format!(
                "SELECT id, a * 9223372036854775807 FROM t WHERE c > {PREPARED} \
                 ORDER BY {order} LIMIT 3"
            ),
        )
    };
    assert!(item("a, id").is_ok(), "the kept rows have a = 0");
    assert!(
        item("a DESC NULLS LAST, id").is_err(),
        "the kept rows have a = 4"
    );
}
