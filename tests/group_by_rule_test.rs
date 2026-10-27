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

//! GROUP BY puts two rows in one group exactly when their keys are equal as
//! DISTINCT and UNION take them, every row lands in a group, and every
//! registered aggregate answers per group as it answers over that group alone

use stoolap::core::Value;
use stoolap::Database;

fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        other => format!("{other}"),
    }
}

/// The rows sorted, or the error
fn rows(db: &Database, sql: &str) -> Vec<String> {
    let result = db.query(sql, ()).and_then(|rows| {
        let mut out = Vec::new();
        for row in rows {
            let row = row?;
            out.push(
                (0..row.len())
                    .map(|i| render(row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
        Ok(out)
    });
    match result {
        Ok(mut out) => {
            out.sort();
            out
        }
        Err(e) => vec![format!("error: {e}")],
    }
}

fn create(db: &Database) {
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, f FLOAT, s TEXT)",
        (),
    )
    .unwrap();
    for i in 1..=40 {
        let g = if i % 7 == 0 {
            "NULL".to_string()
        } else {
            (i % 3).to_string()
        };
        let x = if i % 5 == 0 {
            "NULL".to_string()
        } else {
            (i * 7 % 11).to_string()
        };
        db.execute(
            &format!(
                "INSERT INTO d VALUES ({i}, {g}, {x}, {}, 's{}')",
                (i * 3 % 13) as f64 * 0.5,
                i % 4
            ),
            (),
        )
        .unwrap();
    }
}

/// (key expression over d, the groups as `key|count`, or the counts alone
/// where the group's key may be either of two equal values)
const KEYS: &[(&str, &[&str])] = &[
    (
        "CASE WHEN id = 1 THEN 'a' ELSE 5 END",
        &["a|i:1", "i:5|i:39"],
    ),
    (
        "CASE WHEN id = 30 THEN 'a' ELSE id % 3 END",
        &["a|i:1", "i:0|i:12", "i:1|i:14", "i:2|i:13"],
    ),
    ("CASE WHEN id < 20 THEN 1 ELSE 1.0 END", &["i:40"]),
    ("CASE WHEN id = 1 THEN 1.0 ELSE 1 END", &["i:40"]),
    ("CASE WHEN id = 1 THEN -0.0 ELSE 0.0 END", &["i:40"]),
    (
        "CASE WHEN id < 3 THEN CAST('NaN' AS FLOAT) ELSE 2.0 END",
        &["f:2.0|i:38", "f:NaN|i:2"],
    ),
    (
        "CASE WHEN id < 5 THEN NULL ELSE 7 END",
        &["NULL|i:4", "i:7|i:36"],
    ),
    (
        "CASE WHEN id = 1 THEN TRUE ELSE 1 END",
        &["i:1|i:39", "true|i:1"],
    ),
];

fn check_keys(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    for &(key, want) in KEYS {
        let counts_only = !want[0].contains('|');
        let select = if counts_only { "" } else { "k, " };
        let derived = format!("(SELECT {key} AS k, g, x FROM d) t");
        let forms = [
            format!("SELECT {select}COUNT(*) FROM {derived} GROUP BY k"),
            format!(
                "SELECT {select}COUNT(*) FROM {derived} \
                 WHERE k = k OR k IS NULL OR k <> k GROUP BY k"
            ),
            format!("SELECT {select}COUNT(*) FILTER (WHERE TRUE) FROM {derived} GROUP BY k"),
            format!("SELECT {select}COUNT(x) + COUNT(*) - COUNT(x) FROM {derived} GROUP BY k"),
            format!("SELECT {select}SUM(1) FROM {derived} GROUP BY k"),
            format!("SELECT {select}COUNT(*) FROM {derived} GROUP BY k HAVING COUNT(*) > 0"),
            format!("SELECT {select}COUNT(*) FROM {derived} GROUP BY k ORDER BY COUNT(*)"),
            format!(
                "SELECT {}COUNT(*) FROM d GROUP BY {key}",
                if counts_only {
                    String::new()
                } else {
                    format!("{key}, ")
                }
            ),
        ];
        for sql in forms {
            let got = rows(db, &sql);
            if got != want {
                wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
            }
        }
        let two_keys = format!(
            "SELECT {select}SUM(n) FROM (SELECT k, g, COUNT(*) AS n FROM \
             (SELECT {key} AS k, g FROM d) t GROUP BY k, g) u GROUP BY k"
        );
        let got = rows(db, &two_keys);
        if got != want {
            wrong.push(format!("{ctx}: {two_keys}: got {got:?}, want {want:?}"));
        }
        let limited = format!("SELECT {select}COUNT(*) FROM {derived} GROUP BY k LIMIT 100");
        let got = rows(db, &limited);
        if got != want {
            wrong.push(format!("{ctx}: {limited}: got {got:?}, want {want:?}"));
        }
        let rollup = format!("SELECT {select}COUNT(*) FROM {derived} GROUP BY ROLLUP(k)");
        let mut want_rollup = want.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        want_rollup.push(if counts_only { "i:40" } else { "NULL|i:40" }.to_string());
        want_rollup.sort();
        let got = rows(db, &rollup);
        if got != want_rollup {
            wrong.push(format!(
                "{ctx}: {rollup}: got {got:?}, want {want_rollup:?}"
            ));
        }
    }
}

/// The calls each registered aggregate is checked with; a function that
/// needs a second argument gets a separator
fn calls(db: &Database, name: &str) -> Vec<String> {
    let needs_separator = rows(db, &format!("SELECT {name}(s) FROM d"))[0].starts_with("error");
    let call = |arg: &str, modifiers: &str| {
        if needs_separator {
            format!("{name}({arg}, '-'{modifiers})")
        } else {
            format!("{name}({arg}{modifiers})")
        }
    };
    let mut out = Vec::new();
    for arg in ["x", "f", "s"] {
        out.push(call(arg, ""));
        out.push(call(arg, " ORDER BY id DESC"));
        out.push(format!("{} FILTER (WHERE id % 3 = 0)", call(arg, "")));
        out.push(call(&format!("DISTINCT {arg}"), ""));
    }
    out.push(call("x * 2 + id", ""));
    out
}

fn check_registry(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let mut names = stoolap::functions::global_registry().list_aggregates();
    names.sort();
    assert!(names.len() >= 17, "registered aggregates: {names:?}");
    for name in &names {
        for call in calls(db, name) {
            let grouped = rows(db, &format!("SELECT g, {call} FROM d GROUP BY g"));
            let mut alone = Vec::new();
            for (key, filter) in [
                ("i:0", "g = 0"),
                ("i:1", "g = 1"),
                ("i:2", "g = 2"),
                ("NULL", "g IS NULL"),
            ] {
                for value in rows(db, &format!("SELECT {call} FROM d WHERE {filter}")) {
                    alone.push(format!("{key}|{value}"));
                }
            }
            alone.sort();
            if grouped.iter().any(|row| row.starts_with("error")) {
                wrong.push(format!("{ctx}: {call} GROUP BY g: {grouped:?}"));
            } else if grouped != alone {
                wrong.push(format!(
                    "{ctx}: {call} GROUP BY g: got {grouped:?}, each group alone {alone:?}"
                ));
            }
        }
    }
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    check_keys(db, ctx, wrong);
    check_registry(db, ctx, wrong);
}

#[test]
fn group_by_groups_equal_keys_and_answers_every_aggregate() {
    let mut wrong = Vec::new();
    let memory = Database::open("memory://group_by_rule").unwrap();
    create(&memory);
    check(&memory, "memory", &mut wrong);

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

/// A LIMIT keeps some groups whole: over an input large enough to split,
/// the kept group counts every one of its rows, wherever they fall
#[test]
fn a_limited_group_counts_all_its_rows() {
    const ROWS: i64 = 200_000;
    const SEEDS: i64 = 8;
    let db = Database::open("memory://group_by_rule_limit").unwrap();
    let columns: Vec<String> = (0..SEEDS).map(|s| format!("g{s} INTEGER")).collect();
    db.execute(
        &format!(
            "CREATE TABLE p (id INTEGER PRIMARY KEY, v INTEGER, {})",
            columns.join(", ")
        ),
        (),
    )
    .unwrap();
    let keys: Vec<String> = (0..SEEDS)
        .map(|s| format!("((value * 48271 + {}) % 65537) % 2", s * 7919))
        .collect();
    db.execute(
        &format!(
            "INSERT INTO p SELECT value, value % 10, {} FROM generate_series(1, {ROWS})",
            keys.join(", ")
        ),
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for s in 0..SEEDS {
        let sql =
            format!("SELECT g{s}, COUNT(*), SUM(v) FROM p WHERE id > 0 GROUP BY g{s} LIMIT 1");
        let got = rows(&db, &sql);
        let key = got[0]
            .split('|')
            .next()
            .unwrap()
            .trim_start_matches("i:")
            .to_string();
        let whole = rows(
            &db,
            &format!("SELECT {key}, COUNT(*), SUM(v) FROM p WHERE g{s} = {key}"),
        );
        if got != whole {
            wrong.push(format!("{sql}: got {got:?}, the group holds {whole:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_select_item_naming_a_grouped_column_reads_it_beside_an_aggregate_of_that_alias() {
    let db = Database::open("memory://group_by_rule_alias_shadow").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE one (id INTEGER PRIMARY KEY)",
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 10), (3, 3, 20)",
        "INSERT INTO one VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // An alias is not visible to the other items of its SELECT list
    for (sql, want) in [
        (
            "SELECT SUM(a) AS b, b FROM t GROUP BY b",
            vec!["i:3|i:10", "i:3|i:20"],
        ),
        (
            "SELECT b, SUM(a) AS b FROM t GROUP BY b",
            vec!["i:10|i:3", "i:20|i:3"],
        ),
        (
            "SELECT SUM(a) AS b, b, (SELECT MAX(id) FROM one) FROM t GROUP BY b",
            vec!["i:3|i:10|i:1", "i:3|i:20|i:1"],
        ),
        (
            "SELECT COUNT(*) AS b, b + 1 FROM t GROUP BY b",
            vec!["i:1|i:21", "i:2|i:11"],
        ),
    ] {
        assert_eq!(rows(&db, sql), want, "{sql}");
    }
}
