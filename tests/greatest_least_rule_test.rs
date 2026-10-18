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

//! GREATEST and LEAST skip NULL arguments and are NULL only when every
//! argument is, in every place a call can stand, hot and sealed

use stoolap::core::Value;
use stoolap::Database;

/// (id, a, b, c), and GREATEST and LEAST of a, b, c
type Case = (
    i64,
    &'static str,
    &'static str,
    &'static str,
    Option<i64>,
    Option<i64>,
);

const ROWS: &[Case] = &[
    (1, "1", "NULL", "3", Some(3), Some(1)),
    (2, "5", "3", "NULL", Some(5), Some(3)),
    (3, "NULL", "NULL", "NULL", None, None),
    (4, "2", "2", "2", Some(2), Some(2)),
];

fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => n.to_string(),
        other => format!("{other}"),
    }
}

fn collect(rows: stoolap::Result<stoolap::Rows>) -> Vec<String> {
    match rows {
        Ok(rows) => rows
            .map(|row| {
                let row = row.unwrap();
                (0..row.len())
                    .map(|i| render(row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect(),
        Err(e) => vec![format!("error: {e}")],
    }
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let mut expect = |sql: String, got: Vec<String>, want: Vec<String>| {
        if got != want {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
        }
    };
    for (func, pick) in [("GREATEST", 4usize), ("LEAST", 5)] {
        let value = |row: &Case| match pick {
            4 => row.4,
            _ => row.5,
        };
        let call = format!("{func}(a, b, c)");
        let ids = |keep: &dyn Fn(Option<i64>) -> bool| -> Vec<String> {
            ROWS.iter()
                .filter(|row| keep(value(row)))
                .map(|row| row.0.to_string())
                .collect()
        };

        let sql = format!("SELECT id, {call} FROM t ORDER BY id");
        let want = ROWS
            .iter()
            .map(|row| match value(row) {
                Some(v) => format!("{}|{v}", row.0),
                None => format!("{}|NULL", row.0),
            })
            .collect();
        expect(sql.clone(), collect(db.query(&sql, ())), want);

        let sql = format!("SELECT id FROM t WHERE {call} IS NULL ORDER BY id");
        expect(
            sql.clone(),
            collect(db.query(&sql, ())),
            ids(&|v| v.is_none()),
        );
        let sql = format!("SELECT id FROM t WHERE {call} IS NOT NULL ORDER BY id");
        expect(
            sql.clone(),
            collect(db.query(&sql, ())),
            ids(&|v| v.is_some()),
        );
        let sql = format!("SELECT id FROM t WHERE {call} > 2 ORDER BY id");
        expect(
            sql.clone(),
            collect(db.query(&sql, ())),
            ids(&|v| v.is_some_and(|v| v > 2)),
        );

        for v in ROWS.iter().filter_map(value) {
            let want = ids(&|x| x == Some(v));
            for form in [
                format!("SELECT id FROM t WHERE {call} = {v} ORDER BY id"),
                format!("SELECT id FROM t WHERE {call} + 0 = {v} ORDER BY id"),
                format!(
                    "SELECT id FROM t WHERE CASE WHEN {call} = {v} THEN TRUE ELSE FALSE END \
                     ORDER BY id"
                ),
                format!(
                    "SELECT id FROM t GROUP BY id \
                     HAVING {func}(MAX(a), MAX(b), MAX(c)) = {v} ORDER BY id"
                ),
            ] {
                expect(form.clone(), collect(db.query(&form, ())), want.clone());
            }
            let sql = format!("SELECT id FROM t WHERE {call} = $1 ORDER BY id");
            expect(sql.clone(), collect(db.query(&sql, (v,))), want.clone());
            let prepared = db.prepare(&sql).unwrap();
            expect(
                format!("prepared {sql}"),
                collect(prepared.query((v,))),
                want.clone(),
            );
        }

        let first = if func == "GREATEST" { "3" } else { "1" };
        let sql = format!("SELECT {func}(1, NULL, 3), {func}(NULL, NULL)");
        expect(
            sql.clone(),
            collect(db.query(&sql, ())),
            vec![format!("{first}|NULL")],
        );
        let sql = format!("SELECT {func}($1, NULL, $2)");
        expect(
            sql.clone(),
            collect(db.query(&sql, (1i64, 3i64))),
            vec![first.to_string()],
        );
    }
}

fn create(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)",
        (),
    )
    .unwrap();
    for (id, a, b, c, _, _) in ROWS {
        db.execute(&format!("INSERT INTO t VALUES ({id}, {a}, {b}, {c})"), ())
            .unwrap();
    }
}

#[test]
fn greatest_and_least_skip_null_arguments_everywhere() {
    let mut wrong = Vec::new();
    let memory = Database::open("memory://greatest_least_rule").unwrap();
    create(&memory);
    check(&memory, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    create(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed", &mut wrong);

    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
