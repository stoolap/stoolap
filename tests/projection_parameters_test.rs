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

//! A parameter in a projection reads its value on every path that projects
//! rows: a streamed scan, a primary key lookup, an aggregate's result

use stoolap::{named_params, Database};

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://projection_parameters_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 1), (2, 20, 1), (3, 30, 2)",
        (),
    )
    .unwrap();
    db.execute("CREATE VIEW w AS SELECT id, v FROM t", ())
        .unwrap();
    db
}

fn column(rows: stoolap::Rows, at: usize) -> Vec<String> {
    rows.map(|row| format!("{:?}", row.unwrap().get_value(at)))
        .collect()
}

#[test]
fn a_positional_parameter_reads_in_every_projection() {
    let db = open("positional");
    let mut wrong = Vec::new();
    for (sql, at, expected) in [
        (
            "SELECT id + $1 FROM t",
            0,
            vec![
                "Some(Integer(101))",
                "Some(Integer(102))",
                "Some(Integer(103))",
            ],
        ),
        (
            "SELECT id + $1 FROM t WHERE v > 15",
            0,
            vec!["Some(Integer(102))", "Some(Integer(103))"],
        ),
        (
            "SELECT id + $1 FROM t WHERE id = 1",
            0,
            vec!["Some(Integer(101))"],
        ),
        (
            "SELECT id, $1 FROM t WHERE id = 2",
            1,
            vec!["Some(Integer(100))"],
        ),
        (
            "SELECT id + $1 FROM t LIMIT 1",
            0,
            vec!["Some(Integer(101))"],
        ),
        ("SELECT COUNT(*) + $1 FROM t", 0, vec!["Some(Integer(103))"]),
        ("SELECT SUM(v) + $1 FROM t", 0, vec!["Some(Integer(160))"]),
        (
            "SELECT g, COUNT(*) + $1 FROM t GROUP BY g ORDER BY g",
            1,
            vec!["Some(Integer(102))", "Some(Integer(101))"],
        ),
        (
            "SELECT id + $1 FROM w",
            0,
            vec![
                "Some(Integer(101))",
                "Some(Integer(102))",
                "Some(Integer(103))",
            ],
        ),
    ] {
        let mut seen = column(db.query(sql, (100,)).unwrap(), at);
        if !sql.contains("ORDER BY") {
            seen.sort();
        }
        if seen != expected {
            wrong.push(format!("{sql}: {seen:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_named_parameter_reads_in_a_streamed_projection() {
    let db = open("named");
    let seen = column(
        db.query_named(
            "SELECT id + :shift FROM t WHERE id = 3",
            named_params! { shift: 100 },
        )
        .unwrap(),
        0,
    );
    assert_eq!(seen, ["Some(Integer(103))"]);
}

#[test]
fn an_insert_select_writes_the_parameter_it_projects() {
    let db = open("insert_select");
    db.execute("CREATE TABLE kept (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO kept SELECT id + $1, v FROM t", (100,))
        .unwrap();
    let ids = column(db.query("SELECT id FROM kept ORDER BY id", ()).unwrap(), 0);
    assert_eq!(
        ids,
        [
            "Some(Integer(101))",
            "Some(Integer(102))",
            "Some(Integer(103))"
        ]
    );
}
