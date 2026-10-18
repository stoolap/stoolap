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

fn create(db: &Database) {
    for sql in [
        "CREATE TABLE j (id INTEGER PRIMARY KEY, v JSON)",
        "INSERT INTO j VALUES (1, '{\"k\":1}'), (2, '{\"k\":2}'), (3, '{\"a\":0}')",
        "CREATE TABLE e (id INTEGER PRIMARY KEY, v VECTOR(2))",
        "INSERT INTO e VALUES (1, '[1.0, 2.0]'), (2, '[0.5, 9.0]'), (3, '[3.0, 0.0]')",
        "CREATE TABLE ji (id INTEGER PRIMARY KEY, v JSON)",
        "CREATE INDEX ji_v ON ji(v)",
        "INSERT INTO ji VALUES (1, '{\"k\":1}'), (2, '{\"k\":2}'), (3, '{\"a\":0}')",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    for table in ["j", "e", "ji"] {
        for (sql, general) in [
            (
                format!("SELECT MIN(v), MAX(v) FROM {table}"),
                format!(
                    "SELECT MIN(CASE WHEN id > 0 THEN v END), \
                     MAX(CASE WHEN id > 0 THEN v END) FROM {table}"
                ),
            ),
            (
                format!("SELECT MIN(v), MAX(v) FROM {table} WHERE id > 0"),
                format!(
                    "SELECT MIN(CASE WHEN id > 0 THEN v END), \
                     MAX(CASE WHEN id > 0 THEN v END) FROM {table} WHERE id > 0"
                ),
            ),
        ] {
            let (got, want) = (rows(db, &sql), rows(db, &general));
            if got != want {
                wrong.push(format!("{ctx}: {sql}: got {got:?}, general {want:?}"));
            }
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
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
