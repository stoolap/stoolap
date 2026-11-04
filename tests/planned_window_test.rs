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

//! A statement with window functions over one table runs on the planner's
//! plan, its rows read through the index the window rule chooses, and
//! answers as recorded

use stoolap::core::Value;
use stoolap::Database;

mod oracle;

/// Each statement's answer, recorded when the general path gave the same
const RECORDED: oracle::Answers = answers!("planned_window");

/// The statements over another source than a table, recorded likewise
const RECORDED_SOURCES: oracle::Answers = answers!("planned_window_sources");

/// t: a indexed, b indexed, NOT NULL and distinct, so an ORDER BY b orders
/// the rows one way, c with NULLs
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER)",
    "CREATE INDEX t_a ON t(a)",
    "CREATE INDEX t_b ON t(b)",
    "INSERT INTO t SELECT g.value, g.value % 7, (g.value * 37) % 1009, \
     CASE WHEN g.value % 5 = 0 THEN NULL ELSE g.value % 13 END FROM generate_series(1, 120) g",
    "CREATE VIEW tv AS SELECT id, a, b, c FROM t WHERE id > 3",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://planned_window_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Sealed, then some rows changed and added hot
fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for sql in [
        "UPDATE t SET c = c + 1 WHERE id % 9 = 0",
        "DELETE FROM t WHERE id % 17 = 0",
        "INSERT INTO t VALUES (500, 3, 2000, 4), (501, 6, 2001, NULL)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

const WINDOWS: &[&str] = &[
    "ROW_NUMBER() OVER (PARTITION BY a ORDER BY id)",
    "RANK() OVER (PARTITION BY a ORDER BY c)",
    "SUM(c) OVER (PARTITION BY a)",
    "ROW_NUMBER() OVER (ORDER BY b)",
    "LAG(c, 2) OVER (ORDER BY b DESC)",
    "COUNT(*) OVER ()",
    "SUM(b) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)",
    "NTILE(3) OVER (ORDER BY b)",
    "DENSE_RANK() OVER (ORDER BY c NULLS FIRST)",
];

/// Each statement over `{w}`, with the parameters it reads
const SHAPES: &[(&str, &[i64])] = &[
    ("SELECT id, {w} AS x FROM t ORDER BY id", &[]),
    ("SELECT id, {w} AS x FROM t WHERE a < 5 ORDER BY id", &[]),
    (
        "SELECT id, {w} AS x FROM t WHERE c + 0 > 6 ORDER BY id",
        &[],
    ),
    ("SELECT id, {w} AS x FROM t ORDER BY x, id LIMIT 7", &[]),
    (
        "SELECT id, {w} AS x FROM t ORDER BY id LIMIT 5 OFFSET 3",
        &[],
    ),
    ("SELECT DISTINCT a, {w} AS x FROM t ORDER BY a, x", &[]),
    (
        "SELECT DISTINCT ON (a) a, id, {w} AS x FROM t ORDER BY a, id",
        &[],
    ),
    ("SELECT id FROM t ORDER BY {w}, id LIMIT 10", &[]),
    ("SELECT id, {w} + 1 AS x FROM t ORDER BY id", &[]),
    (
        "SELECT id, {w} AS x FROM t WHERE a < $1 ORDER BY id LIMIT $2",
        &[4, 9],
    ),
    ("SELECT id, {w} AS x FROM t LIMIT 6", &[]),
    (
        "SELECT DISTINCT ON ({w}) id, COUNT(*) OVER () AS n FROM t ORDER BY id",
        &[],
    ),
];

/// The rows of `sql` as text, in the order given
fn answer(db: &Database, sql: &str, params: &[i64]) -> Result<Vec<String>, String> {
    let params: Vec<Value> = params.iter().map(|&p| Value::Integer(p)).collect();
    let rows = db.query(sql, params).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get::<Option<String>>(i) {
                    Ok(value) => value.unwrap_or_else(|| "NULL".into()),
                    Err(e) => format!("<{e}>"),
                })
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok(out)
}

#[test]
fn a_windowed_statement_answers_as_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    let mut count = 0;
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir))] {
        for window in WINDOWS {
            for &(shape, params) in SHAPES {
                let sql = shape.replace("{w}", window);
                count += 1;
                let planned = answer(&db, &sql, params);
                // Rows a LIMIT keeps without an ORDER BY are any of the
                // statement's, each carrying its own window value: the
                // whole statement's rows are recorded in sorted order
                let unordered = !shape.contains("ORDER BY") && shape.contains("LIMIT");
                let failure = match (&planned, unordered) {
                    (Ok(rows), true) => {
                        let (whole, limit) = sql.split_once(" LIMIT ").unwrap();
                        let mut all = answer(&db, whole, params);
                        if let Ok(all) = &mut all {
                            all.sort();
                        }
                        let fits = all.as_ref().is_ok_and(|all| {
                            rows.len() == all.len().min(limit.parse().unwrap())
                                && rows.iter().all(|row| all.contains(row))
                        });
                        let key = format!("{context}: {whole}");
                        oracle::check_answer(&RECORDED, &key, &all)
                            .or_else(|| (!fits).then(|| format!("{sql}: gave {rows:?}")))
                    }
                    _ => oracle::check_answer(&RECORDED, &format!("{context}: {sql}"), &planned),
                };
                wrong.extend(failure);
            }
        }
    }
    assert!(count > 150, "{count} statements");
    assert!(
        wrong.is_empty(),
        "{} of {count} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Each statement over `{w}` reading another source than a table
const SOURCE_SHAPES: &[&str] = &[
    "SELECT id, {w} AS x FROM tv ORDER BY id",
    "SELECT id, {w} AS x FROM (SELECT id, a, b, c FROM t WHERE a < 5) AS d ORDER BY id",
    "WITH s AS (SELECT id, a, b, c FROM t WHERE b > 100) \
     SELECT id, {w} AS x FROM s ORDER BY x, id LIMIT 7",
    "SELECT id, {w} + 1 AS x FROM tv WHERE c IS NOT NULL ORDER BY id LIMIT 5 OFFSET 2",
    "SELECT DISTINCT ON (a) a, id, {w} AS x FROM tv ORDER BY a, id",
    "SELECT id FROM tv ORDER BY {w}, id LIMIT 10",
    "SELECT id, {w} AS x FROM (VALUES (1, 1, 5, 2), (2, 1, 3, NULL), (3, 2, 9, 4)) \
     AS v(id, a, b, c) ORDER BY id",
];

#[test]
fn a_windowed_statement_over_a_source_answers_as_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    let mut count = 0;
    for (context, db) in [("hot", hot("sources")), ("sealed", sealed(&dir))] {
        for window in WINDOWS {
            for shape in SOURCE_SHAPES {
                let sql = shape.replace("{w}", window);
                count += 1;
                let planned = answer(&db, &sql, &[]);
                if let Err(e) = &planned {
                    wrong.push(format!("{context}: {sql}: {e}"));
                }
                let key = format!("{context}: {sql}");
                wrong.extend(oracle::check_answer(&RECORDED_SOURCES, &key, &planned));
            }
        }
    }
    assert!(count > 100, "{count} statements");
    assert!(
        wrong.is_empty(),
        "{} of {count} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_window_over_a_source_under_a_limit_without_order_reads_every_row() {
    let db = hot("source_limit");
    let tv_rows = answer(&db, "SELECT COUNT(*) FROM tv", &[]).unwrap();
    for (sql, want) in [
        (
            "SELECT SUM(value) OVER () FROM generate_series(1, 3) LIMIT 1",
            vec!["6".to_string()],
        ),
        (
            "SELECT COUNT(*) OVER () FROM generate_series(1, 3) LIMIT 1",
            vec!["3".to_string()],
        ),
        (
            "SELECT value, SUM(value) OVER () FROM generate_series(1, 3) LIMIT 1 OFFSET 1",
            vec!["2|6".to_string()],
        ),
        ("SELECT COUNT(*) OVER () FROM tv LIMIT 1", tv_rows),
        (
            "SELECT COUNT(*) OVER () FROM (VALUES (1), (2), (3)) AS v(x) LIMIT 2",
            vec!["3".to_string(), "3".to_string()],
        ),
    ] {
        assert_eq!(answer(&db, sql, &[]), Ok(want), "{sql}");
    }
}

#[test]
fn explain_shows_the_window_step_over_a_source() {
    let db = hot("explain_sources");
    for (sql, window, source) in [
        (
            "SELECT id, ROW_NUMBER() OVER (ORDER BY b) FROM tv ORDER BY id LIMIT 5",
            "Window ROW_NUMBER() OVER (ORDER BY b ASC)",
            "View tv",
        ),
        (
            "SELECT id, SUM(c) OVER (PARTITION BY a) FROM (SELECT id, a, c FROM t) AS d",
            "Window SUM(c) OVER (PARTITION BY a)",
            "Derived d",
        ),
        (
            "SELECT x, LAG(x) OVER (ORDER BY x) FROM (VALUES (3), (1), (2)) AS v(x) ORDER BY x",
            "Window LAG(x) OVER (ORDER BY x ASC)",
            "Values v",
        ),
    ] {
        let got: Vec<String> = db
            .query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap().trim().to_string())
            .collect();
        let at = |line: &str| got.iter().position(|l| l == line);
        assert!(
            matches!((at(window), at(source)), (Some(w), Some(s)) if w < s),
            "{sql}: {got:#?}"
        );
    }
}

#[test]
fn explain_shows_the_window_step_and_the_index_read_it_takes() {
    let db = hot("explain");
    let lines = |sql: &str| -> Vec<String> {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap().trim().to_string())
            .collect()
    };
    for (sql, window, read) in [
        (
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY a ORDER BY id) FROM t LIMIT 5",
            "Window ROW_NUMBER() OVER (PARTITION BY a ORDER BY id ASC)",
            Some("PartitionsByIndex t (a) one at a time to the LIMIT"),
        ),
        (
            "SELECT id, SUM(c) OVER (PARTITION BY a) FROM t ORDER BY id",
            "Window SUM(c) OVER (PARTITION BY a)",
            Some("PartitionsByIndex t (a)"),
        ),
        (
            "SELECT id, ROW_NUMBER() OVER (ORDER BY b) FROM t LIMIT 5",
            "Window ROW_NUMBER() OVER (ORDER BY b ASC)",
            Some("IndexOrder t (b ASC) to the LIMIT"),
        ),
        (
            "SELECT id, ROW_NUMBER() OVER (ORDER BY c) FROM t LIMIT 5",
            "Window ROW_NUMBER() OVER (ORDER BY c ASC)",
            None,
        ),
        (
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY a ORDER BY id) FROM t WHERE b > 3",
            "Window ROW_NUMBER() OVER (PARTITION BY a ORDER BY id ASC)",
            None,
        ),
    ] {
        let got = lines(sql);
        assert!(got.iter().any(|l| l == window), "{sql}: {got:#?}");
        match read {
            Some(read) => {
                assert!(got.iter().any(|l| l == read), "{sql}: {got:#?}");
                assert!(
                    got.iter().any(|l| l.starts_with("else ")),
                    "{sql}: the scan the read falls back to: {got:#?}"
                );
            }
            None => assert!(
                !got.iter()
                    .any(|l| l.contains("PartitionsByIndex") || l.contains("IndexOrder")),
                "{sql}: {got:#?}"
            ),
        }
    }
}

#[test]
fn a_prepared_window_reads_the_parameters_of_each_run() {
    let db = hot("parameter_runs");
    let stmt = db
        .prepare("SELECT id, STRING_AGG(CAST(a AS TEXT), $1) OVER () FROM t WHERE id <= 3")
        .unwrap();
    for separator in [",", "-", "+"] {
        let rows: Vec<String> = stmt
            .query((separator,))
            .unwrap()
            .map(|row| row.unwrap().get::<String>(1).unwrap())
            .collect();
        assert_eq!(rows.len(), 3);
        for joined in rows {
            assert_eq!(
                joined.matches(separator).count(),
                2,
                "{separator}: {joined}"
            );
        }
    }
}
