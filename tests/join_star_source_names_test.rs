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

//! A derived table, CTE or view whose body selects * over a join names each
//! column by its own name, so a reference reads it bare or through the
//! source's name, and a name two of its columns carry is ambiguous

use stoolap::Database;

const SETUP: &[&str] = &[
    "CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
    "INSERT INTO a VALUES (1, 1, 10), (2, 2, 20), (3, 3, 30)",
    "CREATE TABLE b (bid INTEGER PRIMARY KEY, k2 INTEGER, y INTEGER)",
    "INSERT INTO b VALUES (1, 1, 100), (2, 2, 200), (3, 9, 300)",
    "CREATE TABLE c (id INTEGER PRIMARY KEY, z INTEGER)",
    "INSERT INTO c VALUES (1, 7), (2, 8)",
    "CREATE VIEW vj AS SELECT * FROM a JOIN b ON b.k2 = a.k",
];

fn open(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The rows of `sql` as sorted text, or the error
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
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
    out.sort();
    Ok(out)
}

/// The column names of `sql` and its rows in order, or the error
fn shown(db: &Database, sql: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = rows.columns().to_vec();
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
    Ok((columns, out))
}

fn check(db: &Database, mode: &str) {
    let source = |s: &str| {
        [
            (
                "derived",
                format!("(SELECT * FROM a JOIN b ON b.k2 = a.k) AS {s}"),
                "",
            ),
            (
                "cte",
                s.to_string(),
                "WITH {s} AS (SELECT * FROM a JOIN b ON b.k2 = a.k) ",
            ),
            ("view", format!("vj AS {s}"), ""),
        ]
        .map(|(kind, from, with)| (kind, from, with.replace("{s}", s)))
    };
    let mut wrong = Vec::new();
    for (kind, from, with) in source("d") {
        for (shape, want) in [
            ("SELECT k, d.y FROM {d}", vec!["1|100", "2|200"]),
            (
                "SELECT * FROM {d}",
                vec!["1|1|10|1|1|100", "2|2|20|2|2|200"],
            ),
            ("SELECT id FROM {d} WHERE d.y > 150", vec!["2"]),
            (
                "SELECT id FROM {d} WHERE y > (SELECT MIN(z) * 20 FROM c)",
                vec!["2"],
            ),
            (
                "SELECT k, (SELECT MAX(z) FROM c WHERE c.id = d.bid) FROM {d}",
                vec!["1|7", "2|8"],
            ),
            (
                "SELECT d.k, COUNT(*) FROM {d} GROUP BY d.k",
                vec!["1|1", "2|1"],
            ),
            ("SELECT x FROM {d} ORDER BY d.y DESC LIMIT 1", vec!["20"]),
            (
                "SELECT c.z, d.x FROM c JOIN {d} ON d.id = c.id",
                vec!["7|10", "8|20"],
            ),
            (
                "SELECT c.z, d.x FROM c JOIN {d} ON d.id = c.id WHERE d.y < (SELECT 150)",
                vec!["7|10"],
            ),
            (
                "SELECT c.id, (SELECT COUNT(*) FROM {d} WHERE d.k = c.id) FROM c",
                vec!["1|1", "2|1"],
            ),
        ] {
            let sql = format!("{with}{}", shape.replace("{d}", &from));
            let want: Vec<String> = want.into_iter().map(String::from).collect();
            if answer(db, &sql) != Ok(want.clone()) {
                wrong.push(format!(
                    "{mode} {kind}: {sql}: {:?}, want {want:?}",
                    answer(db, &sql)
                ));
            }
        }
    }
    // A name both tables of the body carry is ambiguous wherever it is read
    let shared = "(SELECT * FROM a JOIN c ON c.id = a.id) AS d";
    for sql in [
        format!("SELECT id FROM {shared}"),
        format!("SELECT d.id FROM {shared}"),
        format!("SELECT d.z FROM {shared} WHERE d.id > 0"),
        format!("SELECT d.z FROM {shared} WHERE id > (SELECT 0)"),
        format!("SELECT b.y, d.id FROM b JOIN {shared} ON TRUE"),
        format!("SELECT b.y FROM b JOIN {shared} ON d.id = b.bid"),
        format!("SELECT d.id, COUNT(*) FROM b JOIN {shared} ON TRUE GROUP BY d.id"),
        format!("SELECT b.y FROM b JOIN {shared} ON TRUE ORDER BY d.id"),
        "WITH d AS (SELECT * FROM a JOIN c ON c.id = a.id) SELECT id FROM d".to_string(),
        "SELECT b.y, d.id FROM b JOIN (SELECT 1 AS id, 2 AS id) AS d ON TRUE".to_string(),
        format!(
            "SELECT ROW_NUMBER() OVER w FROM b JOIN {shared} ON TRUE WINDOW w AS (ORDER BY d.id)"
        ),
        format!(
            "SELECT ROW_NUMBER() OVER (w ORDER BY b.y) FROM b JOIN {shared} ON TRUE \
             WINDOW w AS (PARTITION BY d.id)"
        ),
    ] {
        match answer(db, &sql) {
            Err(e) if e.contains("is ambiguous") => {}
            got => wrong.push(format!("{mode}: {sql}: {got:?}, want an ambiguous column")),
        }
    }
    // The columns a body's USING merges are one column
    let sql = "SELECT * FROM (SELECT * FROM a JOIN c USING (id)) AS d";
    let want = Ok(vec!["1|1|10|7".to_string(), "2|2|20|8".to_string()]);
    if answer(db, sql) != want {
        wrong.push(format!(
            "{mode}: {sql}: {:?}, want {want:?}",
            answer(db, sql)
        ));
    }
    // A body with its own WITH names its columns the same way
    for (sql, want) in [
        (
            "SELECT (SELECT COUNT(*) FROM (WITH q AS (SELECT 1 AS id) \
             SELECT * FROM q JOIN (SELECT 2 AS bid) b ON TRUE) d)",
            vec!["1"],
        ),
        (
            "SELECT d.id, d.bid FROM (WITH q AS (SELECT 1 AS id) \
             SELECT * FROM q JOIN (SELECT 2 AS bid) b ON TRUE) d WHERE d.id > (SELECT 0)",
            vec!["1|2"],
        ),
    ] {
        let want: Vec<String> = want.into_iter().map(String::from).collect();
        if answer(db, sql) != Ok(want.clone()) {
            wrong.push(format!(
                "{mode}: {sql}: {:?}, want {want:?}",
                answer(db, sql)
            ));
        }
    }
    // A named window part no window reads names nothing
    for (sql, want) in [
        (
            format!(
                "SELECT 7 FROM b JOIN {shared} ON TRUE WHERE b.bid = 1 WINDOW w AS (ORDER BY d.id)"
            ),
            vec!["7", "7"],
        ),
        (
            format!(
                "SELECT ROW_NUMBER() OVER (w ORDER BY b.y) FROM b JOIN {shared} ON TRUE \
                 WHERE b.bid = 1 WINDOW w AS (ORDER BY d.id)"
            ),
            vec!["1", "2"],
        ),
    ] {
        let want: Vec<String> = want.into_iter().map(String::from).collect();
        if answer(db, &sql) != Ok(want.clone()) {
            wrong.push(format!(
                "{mode}: {sql}: {:?}, want {want:?}",
                answer(db, &sql)
            ));
        }
    }
    // A set operation keeps its names and rows when its ORDER BY holds a subquery
    let set = "SELECT * FROM a JOIN b ON b.k2 = a.k UNION ALL SELECT 0, 0, 0, 0, 0, 0";
    for keys in ["y DESC", "6 DESC", "k2, x"] {
        let plain = format!("{set} ORDER BY {keys}");
        let keyed = format!("{set} ORDER BY {keys}, (SELECT 0)");
        if shown(db, &keyed) != shown(db, &plain) {
            wrong.push(format!(
                "{mode}: {keyed}: {:?}, want {:?}",
                shown(db, &keyed),
                shown(db, &plain)
            ));
        }
    }
    // A view whose body names a column twice is refused when created
    match db.execute(
        "CREATE VIEW vs AS SELECT * FROM a JOIN c ON c.id = a.id",
        (),
    ) {
        Err(e) if e.to_string().contains("named more than once") => {}
        got => wrong.push(format!("{mode}: CREATE VIEW vs: {got:?}, want a refusal")),
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_star_over_a_join_names_each_column_by_its_own_name_in_a_source() {
    check(&open("memory://join_star_source_names"), "hot");
    let dir = tempfile::tempdir().unwrap();
    let db = open(&format!("file://{}/db", dir.path().display()));
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed");
}
