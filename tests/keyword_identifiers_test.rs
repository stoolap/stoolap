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

//! Every keyword is usable as a name in every position that takes a name:
//! bare when it is not reserved, quoted when it is. A keyword that CREATE
//! TABLE accepts as a column must be accepted everywhere else too.

use std::collections::BTreeSet;
use stoolap::parser::token::KEYWORDS;
use stoolap::Database;

fn err(db: &Database, sql: &str) -> Option<String> {
    db.execute(sql, ()).err().map(|e| format!("{sql}: {e}"))
}

/// Statements naming a column `col` in the table `t`, in order
fn column_positions(t: &str, col: &str) -> Vec<String> {
    vec![
        format!("INSERT INTO {t} (id, {col}) VALUES (1, 1)"),
        format!("SELECT {col} FROM {t}"),
        format!("SELECT id FROM {t} WHERE {col} = 1"),
        format!("SELECT id FROM {t} ORDER BY {col}"),
        format!("SELECT {col}, COUNT(*) FROM {t} GROUP BY {col}"),
        format!("UPDATE {t} SET {col} = 2 WHERE id = 1"),
        format!("SELECT id AS {col} FROM {t}"),
        format!("CREATE INDEX idx_{t} ON {t} ({col})"),
        format!("DROP INDEX idx_{t} ON {t}"),
        format!("ALTER TABLE {t} RENAME COLUMN {col} TO renamed"),
        format!("ALTER TABLE {t} RENAME COLUMN renamed TO {col}"),
        format!("ALTER TABLE {t} DROP COLUMN {col}"),
        format!("ALTER TABLE {t} ADD COLUMN {col} INTEGER"),
        format!("DELETE FROM {t} WHERE {col} IS NULL"),
    ]
}

#[test]
fn test_every_keyword_works_as_a_name_bare_or_quoted() {
    let db = Database::open("memory://keyword_identifiers_all").unwrap();
    let keywords: BTreeSet<&str> = KEYWORDS.iter().copied().collect();
    let mut failures = Vec::new();
    let mut reserved = Vec::new();
    for (i, kw) in keywords.iter().enumerate() {
        let bare = kw.to_lowercase();
        let t = format!("t{i}");
        let col = if err(
            &db,
            &format!("CREATE TABLE {t} (id INTEGER PRIMARY KEY, {bare} INTEGER)"),
        )
        .is_some()
        {
            reserved.push(*kw);
            let quoted = format!("\"{bare}\"");
            if let Some(e) = err(
                &db,
                &format!("CREATE TABLE {t} (id INTEGER PRIMARY KEY, {quoted} INTEGER)"),
            ) {
                failures.push(e);
                continue;
            }
            quoted
        } else {
            bare
        };
        failures.extend(
            column_positions(&t, &col)
                .iter()
                .filter_map(|sql| err(&db, sql)),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // The reserved words are the SQL skeleton; a name-like word must not be among them
    for word in [
        "LEVEL", "FORMAT", "DATE", "TEXT", "ROW", "FIRST", "LAST", "RELEASE",
    ] {
        assert!(!reserved.contains(&word), "{word} is reserved");
    }
    assert!(reserved.contains(&"SELECT") && reserved.contains(&"DEFAULT"));
}

#[test]
fn test_non_reserved_keywords_name_tables_views_and_savepoints() {
    let db = Database::open("memory://keyword_identifiers_objects").unwrap();
    let mut failures = Vec::new();
    for kw in [
        "level", "format", "window", "row", "date", "release", "copy",
    ] {
        for sql in [
            format!("CREATE TABLE {kw} (id INTEGER PRIMARY KEY, v INTEGER)"),
            format!("INSERT INTO {kw} VALUES (1, 1)"),
            format!("UPDATE {kw} SET v = 2 WHERE id = 1"),
            format!("CREATE VIEW {kw}_view AS SELECT id FROM {kw}"),
            format!("DROP VIEW {kw}_view"),
            format!("DELETE FROM {kw} WHERE id = 1"),
            format!("TRUNCATE TABLE {kw}"),
            format!("VACUUM {kw}"),
            format!("CREATE INDEX {kw} ON {kw} (v)"),
            format!("DROP INDEX {kw} ON {kw}"),
            "BEGIN".to_string(),
            format!("SAVEPOINT {kw}"),
            format!("ROLLBACK TO SAVEPOINT {kw}"),
            format!("RELEASE SAVEPOINT {kw}"),
            "COMMIT".to_string(),
            format!("DROP TABLE {kw}"),
        ] {
            failures.extend(err(&db, &sql));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn test_cast_extract_and_interval_are_columns_without_their_shape() {
    let db = Database::open("memory://keyword_identifiers_shape").unwrap();
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, cast INTEGER, extract INTEGER, interval INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO d VALUES (1, 2, 3, 4)", ()).unwrap();
    let row: Vec<i64> = db
        .query(
            "SELECT cast, extract, interval, CAST(cast AS TEXT) || '', \
             EXTRACT(YEAR FROM TIMESTAMP '2024-05-06 00:00:00') FROM d \
             WHERE interval = 4 AND cast + extract = 5",
            (),
        )
        .unwrap()
        .flat_map(|r| {
            let r = r.unwrap();
            [
                r.get::<i64>(0).unwrap(),
                r.get::<i64>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
                r.get::<String>(3).unwrap().parse::<i64>().unwrap(),
                r.get::<i64>(4).unwrap(),
            ]
        })
        .collect();
    assert_eq!(row, [2, 3, 4, 2, 2024]);
    let intervals: Vec<String> = db
        .query(
            "SELECT TIMESTAMP '2024-01-31 00:00:00' + INTERVAL '1 day', \
             TIMESTAMP '2024-01-31 00:00:00' + INTERVAL 1 DAY",
            (),
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            format!(
                "{} {}",
                r.get::<String>(0).unwrap(),
                r.get::<String>(1).unwrap()
            )
        })
        .collect();
    assert_eq!(intervals.len(), 1);
    assert!(intervals[0].starts_with("2024-02-01"), "{}", intervals[0]);
}

#[test]
fn test_default_is_reserved_and_keeps_its_meaning() {
    let db = Database::open("memory://keyword_identifiers_default").unwrap();
    let e = db
        .execute(
            "CREATE TABLE d (id INTEGER PRIMARY KEY, default INTEGER)",
            (),
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("reserved"), "{e}");
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, \"default\" INTEGER DEFAULT 7, v INTEGER DEFAULT 9)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO d (id) VALUES (1)", ()).unwrap();
    db.execute("UPDATE d SET v = 1, \"default\" = 3 WHERE id = 1", ())
        .unwrap();
    db.execute("UPDATE d SET v = DEFAULT WHERE id = 1", ())
        .unwrap();
    let row: Vec<i64> = db
        .query("SELECT \"default\", v FROM d", ())
        .unwrap()
        .flat_map(|r| {
            let r = r.unwrap();
            [r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap()]
        })
        .collect();
    assert_eq!(row, [3, 9]);
}
