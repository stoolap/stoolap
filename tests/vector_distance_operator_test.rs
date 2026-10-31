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

//! `a <=> b` answers what `VEC_DISTANCE_L2(a, b)` answers: the same value,
//! the same NULL and the same error, in every clause, prepared or not, over
//! hot and sealed rows

use stoolap::{Database, Value};

/// Each statement with `{d}` standing for the distance
const CLAUSES: &[&str] = &[
    "SELECT id, {d} FROM t ORDER BY id",
    "SELECT id FROM t WHERE {d} < 2 ORDER BY id",
    "SELECT COUNT(*) FROM t HAVING SUM({d}) >= 0",
    "SELECT id, CASE WHEN {d} < 1 THEN 'near' ELSE 'far' END FROM t ORDER BY id",
    "SELECT id FROM t ORDER BY {d} LIMIT 2",
];

/// The query vector of each case: as written, and the parameter's value
const QUERIES: &[(&str, Option<&str>)] = &[
    ("'[1, 0, 0]'", None),
    ("'[1, 0]'", None),
    ("NULL", None),
    ("'abc'", None),
    ("5", None),
    ("$1", Some("[0, 1, 0]")),
    ("$1", Some("[0, 1]")),
];

fn setup(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, e VECTOR(3))", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, '[1, 0, 0]'), (2, '[0, 1, 0]'), (3, '[0, 0, 2]'), (4, NULL)",
        (),
    )
    .unwrap();
}

/// The rows of `sql` with their values, or its error
fn answer(db: &Database, sql: &str, param: Option<&str>, prepared: bool) -> String {
    let params: Vec<Value> = param.map(Value::text).into_iter().collect();
    let rows = match prepared {
        true => db.prepare(sql).and_then(|stmt| stmt.query(params)),
        false => db.query(sql, params),
    };
    let collected = rows.and_then(|rows| {
        rows.map(|row| {
            row.map(|row| {
                (0..row.len())
                    .map(|i| format!("{:?}", row.get_value(i)))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
        })
        .collect::<stoolap::Result<Vec<_>>>()
    });
    match collected {
        Ok(rows) => rows.join("; "),
        Err(e) => format!("error: {e}"),
    }
}

fn mismatches(db: &Database, storage: &str) -> Vec<String> {
    let mut wrong = Vec::new();
    for clause in CLAUSES {
        for (query, param) in QUERIES {
            let (written, call) = (
                format!("e <=> {query}"),
                format!("VEC_DISTANCE_L2(e, {query})"),
            );
            let operator = clause.replace("{d}", &written);
            let function = clause.replace("{d}", &call);
            for prepared in [false, true] {
                // An error names the expression as written
                let got = answer(db, &operator, *param, prepared).replace(&written, &call);
                let want = answer(db, &function, *param, prepared);
                if got != want {
                    wrong.push(format!(
                        "{storage}, prepared {prepared}, {param:?}: {operator}\n  got  {got}\n  want {want}"
                    ));
                }
            }
        }
    }
    wrong
}

#[test]
fn the_distance_operator_answers_as_vec_distance_l2() {
    let hot = Database::open("memory://vector_distance_operator_hot").unwrap();
    setup(&hot);
    let dir = tempfile::tempdir().unwrap();
    let sealed = Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap();
    setup(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let mut wrong = mismatches(&hot, "hot");
    wrong.extend(mismatches(&sealed, "sealed"));
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
