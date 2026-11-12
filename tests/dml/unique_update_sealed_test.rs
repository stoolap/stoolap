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

//! An UPDATE's new unique keys are judged against the rows as the
//! statement leaves them, whether the rows are hot, sealed or both

use stoolap::Database;

/// Where the rows (1, 10) and (2, 20) of t live
#[derive(Clone, Copy, Debug)]
enum Rows {
    Hot,
    Sealed,
    /// Row 1 sealed, row 2 hot
    Mixed,
}

/// t(id, u UNIQUE, w) and s(id, a, b, UNIQUE (a, b)), laid out as `rows` says
fn database(rows: Rows, dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, u INTEGER NOT NULL UNIQUE, w INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, UNIQUE (a, b))",
        "INSERT INTO t VALUES (1, 10, 0)",
        "INSERT INTO s VALUES (1, 1, 'x')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if matches!(rows, Rows::Mixed) {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    for sql in [
        "INSERT INTO t VALUES (2, 20, 0)",
        "INSERT INTO s VALUES (2, 1, 'y')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if matches!(rows, Rows::Sealed) {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each statement's outcome, `ok` or the start of its error, then the rows
fn run(db: &Database, steps: &[&str]) -> String {
    let outcomes: Vec<String> = steps
        .iter()
        .map(|sql| match db.execute(sql, ()) {
            Ok(_) => "ok".to_string(),
            Err(e) => {
                let text = e.to_string();
                match text.find(" with value") {
                    Some(end) => text[..end].to_string(),
                    None => text,
                }
            }
        })
        .collect();
    let rows: Vec<String> = db
        .query("SELECT id, u FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            format!(
                "{}:{}",
                row.get::<i64>(0).unwrap(),
                row.get::<i64>(1).unwrap()
            )
        })
        .collect();
    format!("{} => {}", outcomes.join(" | "), rows.join(","))
}

const TAKEN: &str = "unique constraint failed for index unique_t_u on column u";

/// Each sequence with the outcome every layout gives
fn cases() -> Vec<(Vec<&'static str>, String)> {
    let swap = "UPDATE t SET u = CASE id WHEN 1 THEN 20 WHEN 2 THEN 10 END";
    vec![
        (vec![swap], "ok => 1:20,2:10".into()),
        (
            vec!["BEGIN", swap, "COMMIT"],
            "ok | ok | ok => 1:20,2:10".into(),
        ),
        (vec!["UPDATE t SET u = u + 10"], "ok => 1:20,2:30".into()),
        (
            vec!["UPDATE t SET u = u - 10 + 20 * (id - 1) WHERE id > 0"],
            "ok => 1:0,2:30".into(),
        ),
        (vec!["UPDATE t SET w = w + 1"], "ok => 1:10,2:20".into()),
        (
            vec!["UPDATE t SET u = 20 WHERE id = 1"],
            format!("{TAKEN} => 1:10,2:20"),
        ),
        (vec!["UPDATE t SET u = 30"], format!("{TAKEN} => 1:10,2:20")),
        (
            vec![
                "BEGIN",
                "DELETE FROM t WHERE id = 2",
                "UPDATE t SET u = 20 WHERE id = 1",
                "COMMIT",
            ],
            "ok | ok | ok | ok => 1:20".into(),
        ),
    ]
}

#[test]
fn a_statement_s_new_unique_keys_are_judged_as_it_leaves_the_rows() {
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        for (steps, want) in cases() {
            let dir = tempfile::tempdir().unwrap();
            let db = database(rows, &dir);
            let got = run(&db, &steps);
            if got != want {
                wrong.push(format!("{rows:?} {steps:?}\n  got  {got}\n  want {want}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_composite_unique_key_swaps_in_one_statement() {
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        let swap = "UPDATE s SET b = CASE b WHEN 'x' THEN 'y' ELSE 'x' END";
        let taken = "UPDATE s SET b = 'x' WHERE id = 1";
        let got = [swap, taken].map(|sql| db.execute(sql, ()).map_err(|e| e.to_string()));
        let left: Vec<String> = db
            .query("SELECT id, b FROM s ORDER BY id", ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                format!(
                    "{}:{}",
                    row.get::<i64>(0).unwrap(),
                    row.get::<String>(1).unwrap()
                )
            })
            .collect();
        let as_asked = matches!(&got[0], Ok(2))
            && matches!(&got[1], Err(e) if e.contains("unique constraint failed for index unique_s_0 on column a, b"))
            && left == ["1:y", "2:x"];
        if !as_asked {
            wrong.push(format!("{rows:?}: {got:?} left {left:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
