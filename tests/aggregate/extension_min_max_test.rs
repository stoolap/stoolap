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

//! MIN and MAX of a JSON or VECTOR column answer as the general path does,
//! whichever path the query takes

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Vec<String> {
    match db.query(sql, ()) {
        Ok(result) => result
            .map(|row| match row {
                Ok(row) => (0..row.len())
                    .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|"),
                Err(e) => format!("error: {e}"),
            })
            .collect(),
        Err(e) => vec![format!("error: {e}")],
    }
}

const JSON_ROWS: &str = "(1, 1, '{\"k\":1}'), (2, 1, '{\"k\":2}'), (3, 1, '{\"a\":0}'), \
                         (4, 2, '{\"m\":5}'), (5, 2, '{\"b\":1}')";
const VECTOR_ROWS: &str = "(1, 1, '[1.0, 2.0]'), (2, 1, '[0.5, 9.0]'), (3, 1, '[3.0, 0.0]'), \
                           (4, 2, '[2.0, 2.0]'), (5, 2, '[0.25, 1.0]')";

/// (table, a value one row holds)
const TABLES: &[(&str, &str)] = &[
    ("j", "CAST('{\"a\":0}' AS JSON)"),
    ("jb", "CAST('{\"a\":0}' AS JSON)"),
    ("e", "CAST('[3.0, 0.0]' AS VECTOR)"),
];

fn create(db: &Database) {
    for sql in [
        "CREATE TABLE j (id INTEGER PRIMARY KEY, g INTEGER, v JSON)".to_string(),
        format!("INSERT INTO j VALUES {JSON_ROWS}"),
        "CREATE TABLE jb (id INTEGER PRIMARY KEY, g INTEGER, v JSON)".to_string(),
        "CREATE INDEX jb_v ON jb(v) USING BTREE".to_string(),
        format!("INSERT INTO jb VALUES {JSON_ROWS}"),
        "CREATE TABLE e (id INTEGER PRIMARY KEY, g INTEGER, v VECTOR(2))".to_string(),
        format!("INSERT INTO e VALUES {VECTOR_ROWS}"),
    ] {
        db.execute(&sql, ()).unwrap();
    }
}

fn sorted(mut rows: Vec<String>) -> Vec<String> {
    rows.sort();
    rows
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    const V: &str = "CASE WHEN id > 0 THEN v END";
    for (table, held) in TABLES {
        for (sql, general) in [
            (
                format!("SELECT MIN(v), MAX(v) FROM {table}"),
                format!("SELECT MIN({V}), MAX({V}) FROM {table}"),
            ),
            (
                format!("SELECT MIN(v) FROM {table}"),
                format!("SELECT MIN({V}) FROM {table}"),
            ),
            (
                format!("SELECT MAX(v) FROM {table}"),
                format!("SELECT MAX({V}) FROM {table}"),
            ),
            (
                format!("SELECT MIN(v), MAX(v) FROM {table} WHERE id > 0"),
                format!("SELECT MIN({V}), MAX({V}) FROM {table} WHERE id > 0"),
            ),
            (
                format!("SELECT g, MIN(v), MAX(v) FROM {table} GROUP BY g"),
                format!("SELECT g, MIN({V}), MAX({V}) FROM {table} GROUP BY g"),
            ),
            (
                format!("SELECT v FROM {table} ORDER BY v LIMIT 2"),
                format!("SELECT v FROM {table} ORDER BY {V} LIMIT 2"),
            ),
            (
                format!("SELECT v FROM {table} ORDER BY v DESC LIMIT 2"),
                format!("SELECT v FROM {table} ORDER BY {V} DESC LIMIT 2"),
            ),
        ] {
            let (got, want) = (rows(db, &sql), rows(db, &general));
            let (got, want) = if sql.contains("GROUP BY") {
                (sorted(got), sorted(want))
            } else {
                (got, want)
            };
            if got != want {
                wrong.push(format!("{ctx}: {sql}: got {got:?}, general {want:?}"));
            }
        }
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE v = {held}");
        let got = rows(db, &sql);
        if got != ["Integer(1)"] {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want [Integer(1)]"));
        }
    }
}

#[test]
fn json_and_vector_min_max_answer_as_the_general_path() {
    let mut wrong = Vec::new();
    let memory = Database::open("memory://extension_min_max").unwrap();
    create(&memory);
    check(&memory, "memory", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        create(&db);
        check(&db, "file hot", &mut wrong);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        check(&db, "file sealed", &mut wrong);
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    check(&db, "file reopened", &mut wrong);
    for table in ["j", "jb"] {
        db.execute(
            &format!("INSERT INTO {table} VALUES (6, 2, '{{\"c\":7}}')"),
            (),
        )
        .unwrap();
    }
    db.execute("INSERT INTO e VALUES (6, 2, '[9.0, 9.0]')", ())
        .unwrap();
    check(&db, "file reopened, sealed and hot", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
