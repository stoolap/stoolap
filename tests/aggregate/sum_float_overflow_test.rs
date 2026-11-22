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

//! SUM and AVG of FLOAT values answer as `+` does: a sum finite values take
//! past the FLOAT range is a `Float overflow` error on every path, hot and
//! sealed, prepared or not, while an infinite value gives an infinite sum

use stoolap::Database;

fn setup(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE f (id INTEGER PRIMARY KEY, x FLOAT, g INTEGER)",
        "INSERT INTO f VALUES (1, 1e308, 1), (2, 1e308, 1), (3, 1.0, 2)",
        "CREATE TABLE fi (id INTEGER PRIMARY KEY, x FLOAT)",
        "INSERT INTO fi VALUES (1, CAST('inf' AS FLOAT)), (2, 1.0)",
        "CREATE TABLE fb (id INTEGER PRIMARY KEY, x FLOAT)",
        "INSERT INTO fb VALUES (1, 1e308), (2, -1e308), (3, 1.0)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    // The volumes sealed before it carry no such column: its default stands in
    db.execute("ALTER TABLE f ADD COLUMN y FLOAT DEFAULT 1e308", ())
        .unwrap();
    db
}

fn answer(rows: stoolap::Result<stoolap::Rows>) -> Result<Vec<String>, String> {
    rows.map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

const OVERFLOWS: &[(&str, &str)] = &[
    ("SELECT SUM(x) FROM f", "SUM"),
    ("SELECT AVG(x) FROM f", "AVG"),
    ("SELECT SUM(x) FROM f WHERE id < 3", "SUM"),
    ("SELECT g, SUM(x) FROM f GROUP BY g", "SUM"),
    ("SELECT g, AVG(x) FROM f GROUP BY g", "AVG"),
    ("SELECT id, SUM(x) OVER (ORDER BY id) FROM f", "SUM"),
    (
        "SELECT id, AVG(x) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM f",
        "AVG",
    ),
    ("SELECT SUM(y) FROM f", "SUM"),
];

const ANSWERED: &[(&str, &[&str])] = &[
    ("SELECT SUM(x) FROM f WHERE id = 3", &["1"]),
    ("SELECT SUM(DISTINCT x) FROM f", &["1e308"]),
    ("SELECT SUM(x), AVG(x) FROM fi", &["Infinity|Infinity"]),
    ("SELECT SUM(x), AVG(x) FROM fb", &["1|0.3333333333333333"]),
];

#[test]
fn a_float_sum_past_the_range_fails_and_an_infinite_value_does_not() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        for (sql, function) in OVERFLOWS {
            let statement = db.prepare(sql).unwrap();
            for (how, got) in [
                ("ad hoc", answer(db.query(sql, ()))),
                ("prepared", answer(statement.query(()))),
            ] {
                let prefix = format!("type error: Float overflow in {function}:");
                if !got.as_ref().is_err_and(|e| e.starts_with(&prefix)) {
                    wrong.push(format!("sealed {sealed}, {how}: {sql}: {got:?}"));
                }
            }
        }
        for (sql, want) in ANSWERED {
            let want: Result<Vec<String>, String> =
                Ok(want.iter().map(|row| row.to_string()).collect());
            let statement = db.prepare(sql).unwrap();
            for (how, got) in [
                ("ad hoc", answer(db.query(sql, ()))),
                ("prepared", answer(statement.query(()))),
            ] {
                if got != want {
                    wrong.push(format!("sealed {sealed}, {how}: {sql}: {got:?}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Rows past one row group, so a hot scan and a sealed one each add them in
/// several runs; (table, ids holding 1e308, id holding infinity, answer)
const RUNS: &[(&str, &[i64], Option<i64>, &str)] = &[
    ("over_then_inf", &[10, 11], Some(69_000), "Infinity"),
    ("inf_then_over", &[69_000, 69_001], Some(10), "Infinity"),
    ("over_apart", &[10, 69_000], None, "overflow"),
    ("over_only", &[10, 11], None, "overflow"),
];

#[test]
fn an_overflow_and_an_infinite_value_in_different_runs_answer_as_in_one() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
        let db = Database::open(&dsn).unwrap();
        for (table, big, inf, _) in RUNS {
            let big = big
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let inf = inf.unwrap_or(-1);
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, x FLOAT)"),
                (),
            )
            .unwrap();
            db.execute(
                &format!(
                    "INSERT INTO {table} SELECT value, CASE WHEN value IN ({big}) THEN 1e308 \
                     WHEN value = {inf} THEN CAST('inf' AS FLOAT) ELSE 1.0 END \
                     FROM generate_series(1, 70000)"
                ),
                (),
            )
            .unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (table, _, _, want) in RUNS {
            // A deleted row makes the sealed sum scan its volumes
            db.execute(&format!("DELETE FROM {table} WHERE id = 5"), ())
                .unwrap();
            for sql in [
                format!("SELECT SUM(x) FROM {table}"),
                format!("SELECT AVG(x) FROM {table}"),
            ] {
                let got = answer(db.query(&sql, ()));
                let right = match *want {
                    "overflow" => got.as_ref().is_err_and(|e| e.contains("Float overflow")),
                    want => got.as_deref() == Ok(&[want.to_string()][..]),
                };
                if !right {
                    wrong.push(format!("sealed {sealed}: {sql}: {got:?}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
