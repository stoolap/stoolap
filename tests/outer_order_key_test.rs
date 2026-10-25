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

//! An enclosing query's column is one value inside a subquery: as an ORDER BY
//! key it never binds the subquery's own column of its name

use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE parent (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO parent VALUES (1, 5), (2, 7)",
        "CREATE TABLE child (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "INSERT INTO child VALUES (1, 1, 100), (2, 2, 0), (3, 3, 50)",
        "CREATE TABLE side (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO side VALUES (1, 1), (2, 2), (3, 3)",
        "CREATE VIEW cv AS SELECT id, a, b FROM child",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(format!("{:?}", row.get::<stoolap::Value>(0)));
    }
    Ok(out)
}

const SOURCES: [(&str, &str); 6] = [
    ("", "child AS c"),
    ("", "(SELECT * FROM child) AS c"),
    ("", "cv AS c"),
    ("WITH w AS (SELECT * FROM child) ", "w AS c"),
    ("", "child AS c JOIN side AS s ON s.k = c.a"),
    (
        "",
        "(SELECT id, a, b FROM child) AS c JOIN side AS s ON s.k = c.a",
    ),
];

const SHAPES: [&str; 6] = [
    "SELECT c.a FROM {from} ORDER BY {key}, c.a LIMIT 1",
    "SELECT c.a FROM {from} ORDER BY {key}, c.a DESC LIMIT 1",
    "SELECT DISTINCT c.a FROM {from} ORDER BY {key}, c.a LIMIT 1",
    "SELECT c.a + c.b * 0 FROM {from} ORDER BY {key}, c.a LIMIT 1",
    "SELECT ROW_NUMBER() OVER () * 0 + c.a FROM {from} ORDER BY {key}, c.a LIMIT 1",
    "SELECT c.a FROM {from} GROUP BY c.a ORDER BY {key}, c.a LIMIT 1",
];

#[test]
fn an_enclosing_column_sorts_as_one_value() {
    let db = Database::open("memory://outer_order_key").unwrap();
    setup(&db);
    let mut wrong = Vec::new();
    for (with, from) in SOURCES {
        for shape in SHAPES {
            let inner = |key: &str| shape.replace("{from}", from).replace("{key}", key);
            let outer =
                |inner: String| format!("{with}SELECT ({inner}) FROM parent AS p ORDER BY p.id");
            let want = rows(
                &db,
                &outer(shape.replace("{key}, ", "").replace("{from}", from)),
            );
            for key in ["p.b", "p.b DESC", "p.b + 0", "-p.b", "p.id"] {
                let sql = outer(inner(key));
                let got = rows(&db, &sql);
                if got != want {
                    wrong.push(format!("{sql}\n  got {got:?}, want {want:?}"));
                }
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
fn the_subquerys_own_qualifier_binds_its_own_column() {
    let db = Database::open("memory://outer_order_key_own").unwrap();
    setup(&db);
    let mut wrong = Vec::new();
    let want = Ok(vec![String::from("Ok(Integer(2))"); 2]);
    for (with, from) in SOURCES {
        // A grouped query cannot sort by a column it does not group
        for shape in SHAPES.iter().filter(|shape| !shape.contains("GROUP BY")) {
            let inner = shape.replace("{from}", from).replace("{key}", "c.b");
            // The enclosing query names its table c as well, or not
            for outer in ["parent AS p", "parent AS c"] {
                let sql = format!("{with}SELECT ({inner}) FROM {outer}");
                let got = rows(&db, &sql);
                if got != want {
                    wrong.push(format!("{sql}\n  got {got:?}, want {want:?}"));
                }
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
