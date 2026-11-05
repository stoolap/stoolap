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

//! A name or a stored text the files cannot hold is refused by the
//! statement, which leaves the database as it was after a reopen

use stoolap::Database;

/// The most bytes a stored name or text may take
const MAX: usize = u16::MAX as usize;

/// The most bytes a table name may take: it names a directory
const TABLE_NAME: usize = 255;

/// The most bytes an HNSW index name may take: its graph file is named
/// `hnsw_{name}-{YYYYMMDD-HHMMSS.mmm}.bin`
const HNSW_NAME: usize = 226;

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!("file://{}?checkpoint_interval=0", dir.display())).unwrap()
}

fn tables(db: &Database) -> Vec<String> {
    let mut names: Vec<String> = db
        .query("SHOW TABLES", ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    names.sort();
    names
}

/// Each row of t as (id, x), or the column count when t has other columns
fn rows_of_t(db: &Database) -> Result<Vec<(i64, i64)>, String> {
    db.query("SELECT * FROM t ORDER BY id", ())
        .and_then(|rows| {
            rows.map(|row| {
                let row = row?;
                if row.len() != 2 {
                    return Ok((-1, row.len() as i64));
                }
                Ok((row.get::<i64>(0)?, row.get::<i64>(1)?))
            })
            .collect()
        })
        .map_err(|e| e.to_string())
}

/// Statements over t (id, x) and w (id, v VECTOR(2)), with the error each
/// is refused with
fn refused_statements() -> Vec<(&'static str, String, &'static str)> {
    let long = "a".repeat(MAX + 1);
    let check = vec!["x > 0"; MAX / 8].join(" AND ");
    let stored = "a schema can store";
    let file = "a file name leaves it";
    let path = "since it names a file";
    vec![
        (
            "table name",
            format!(
                "CREATE TABLE {} (id INTEGER PRIMARY KEY)",
                "a".repeat(TABLE_NAME + 1)
            ),
            file,
        ),
        (
            "table name with a /",
            "CREATE TABLE \"a/b\" (id INTEGER PRIMARY KEY)".to_string(),
            path,
        ),
        (
            "table name ..",
            "CREATE TABLE \"..\" (id INTEGER PRIMARY KEY)".to_string(),
            path,
        ),
        (
            "column name",
            format!("CREATE TABLE u (id INTEGER PRIMARY KEY, {long} INTEGER)"),
            stored,
        ),
        (
            "column CHECK",
            format!("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER CHECK ({check}))"),
            stored,
        ),
        (
            "table CHECK",
            format!("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, CHECK ({check}))"),
            stored,
        ),
        (
            "DEFAULT text",
            format!("CREATE TABLE u (id INTEGER PRIMARY KEY, s TEXT DEFAULT '{long}')"),
            stored,
        ),
        (
            "name of the index a UNIQUE column gets",
            format!(
                "CREATE TABLE u (id INTEGER PRIMARY KEY, {} INTEGER UNIQUE)",
                "a".repeat(MAX - 4)
            ),
            stored,
        ),
        (
            "added column name",
            format!("ALTER TABLE t ADD COLUMN {long} INTEGER"),
            stored,
        ),
        (
            "added column DEFAULT text",
            format!("ALTER TABLE t ADD COLUMN s TEXT DEFAULT '{long}'"),
            stored,
        ),
        (
            "added column DEFAULT value",
            format!(
                "ALTER TABLE t ADD COLUMN s TEXT DEFAULT (REPEAT('a', {}))",
                MAX + 1
            ),
            stored,
        ),
        (
            "renamed column",
            format!("ALTER TABLE t RENAME COLUMN x TO {long}"),
            stored,
        ),
        (
            "renamed table",
            format!("ALTER TABLE t RENAME TO {}", "a".repeat(TABLE_NAME + 1)),
            file,
        ),
        (
            "renamed table with ..",
            "ALTER TABLE t RENAME TO \"../t\"".to_string(),
            path,
        ),
        (
            "view name",
            format!("CREATE VIEW {long} AS SELECT 1"),
            stored,
        ),
        (
            "index name",
            format!("CREATE INDEX {long} ON t (x)"),
            stored,
        ),
        (
            "HNSW index name",
            format!(
                "CREATE INDEX {} ON w (v) USING HNSW",
                "a".repeat(HNSW_NAME + 1)
            ),
            file,
        ),
        (
            "HNSW index name with a /",
            "CREATE INDEX \"a/b\" ON w (v) USING HNSW".to_string(),
            path,
        ),
        (
            "name of the HNSW index a UNIQUE VECTOR column gets",
            "CREATE TABLE u (id INTEGER PRIMARY KEY, \"a/b\" VECTOR(2) UNIQUE)".to_string(),
            path,
        ),
        (
            "DEFAULT literal a reopen reads as a value past the limit",
            format!(
                "CREATE TABLE u (id INTEGER PRIMARY KEY, s TEXT DEFAULT '{}')",
                "a".repeat(MAX - 4)
            ),
            stored,
        ),
        (
            "name of the HNSW index a table UNIQUE on a VECTOR gets",
            format!(
                "CREATE TABLE {} (id INTEGER PRIMARY KEY, v VECTOR(2), UNIQUE (v))",
                "a".repeat(220)
            ),
            file,
        ),
        (
            "HNSW index with options, name with a /",
            "CREATE INDEX \"a/b\" ON w (v) USING HNSW WITH (m = 16)".to_string(),
            path,
        ),
        (
            "HNSW index with options, name too long",
            format!("CREATE INDEX {long} ON w (v) USING HNSW WITH (m = 16)"),
            stored,
        ),
        (
            "view name whose lowercase is past the limit",
            format!("CREATE VIEW \"{}\" AS SELECT 1", "İ".repeat(22_000)),
            stored,
        ),
    ]
}

/// t and w, the tables every refused statement runs beside
fn setup(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 5)", ()).unwrap();
    db.execute("INSERT INTO w VALUES (1, '[1, 2]')", ())
        .unwrap();
}

/// What a reopen of `dir` finds wrong: tables other than t and w, or t
/// changed
fn reopened(dir: &std::path::Path) -> Option<String> {
    let db = open(dir);
    let (tables, rows) = (tables(&db), rows_of_t(&db));
    db.close().unwrap();
    (tables != ["t", "w"] || rows != Ok(vec![(1, 5)])).then(|| {
        let tables: Vec<String> = tables
            .iter()
            .map(|t| t.chars().take(20).collect())
            .collect();
        format!("{tables:?} {rows:?}")
    })
}

/// What each refused statement leaves wrong, run alone or, for a CREATE
/// TABLE, inside a transaction, which makes the table on a path of its own
fn refusals(in_transaction: bool) -> Vec<String> {
    let mut wrong = Vec::new();
    for (shape, sql, error) in refused_statements() {
        if in_transaction && !sql.starts_with("CREATE TABLE") {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(dir.path());
            setup(&db);
            if in_transaction {
                db.execute("BEGIN", ()).unwrap();
            }
            match db.execute(&sql, ()) {
                Err(e) if e.to_string().contains(error) => {}
                other => wrong.push(format!(
                    "{shape}: {}",
                    format!("{:?}", other.map_err(|e| e.to_string()))
                        .chars()
                        .take(200)
                        .collect::<String>()
                )),
            }
            if in_transaction {
                db.execute("COMMIT", ()).unwrap();
            }
            db.close().unwrap();
        }
        // Reopened from the WAL, then checkpointed and reopened from the
        // snapshot: a reopen may compute what the statement left to compute
        if let Some(found) = reopened(dir.path()) {
            wrong.push(format!("{shape}, reopened: {found}"));
        }
        {
            let db = open(dir.path());
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db.close().unwrap();
        }
        if let Some(found) = reopened(dir.path()) {
            wrong.push(format!("{shape}, checkpointed and reopened again: {found}"));
        }
    }
    wrong
}

#[test]
fn a_name_or_text_the_files_cannot_hold_is_refused_and_leaves_nothing() {
    let wrong = refusals(false);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn a_create_table_in_a_transaction_is_refused_the_same() {
    let wrong = refusals(true);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn a_name_or_text_at_the_limits_survives_a_reopen() {
    let table = "a".repeat(TABLE_NAME);
    let column = "c".repeat(MAX);
    let index = "i".repeat(HNSW_NAME);
    let text = "b".repeat(MAX - 10);
    for checkpoint in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(dir.path());
            db.execute(
                &format!(
                    "CREATE TABLE {table} (id INTEGER PRIMARY KEY, {column} INTEGER, \
                     s TEXT DEFAULT '{text}', v VECTOR(2))"
                ),
                (),
            )
            .unwrap();
            db.execute(
                &format!("CREATE INDEX {index} ON {table} (v) USING HNSW"),
                (),
            )
            .unwrap();
            db.execute(
                &format!("INSERT INTO {table} (id, {column}, v) VALUES (1, 7, '[1, 2]')"),
                (),
            )
            .unwrap();
            if checkpoint {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            db.close().unwrap();
        }
        // Reopened, checkpointed, and reopened again
        {
            let db = open(dir.path());
            db.execute(
                &format!("INSERT INTO {table} (id, {column}, v) VALUES (2, 8, '[3, 4]')"),
                (),
            )
            .unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db.close().unwrap();
        }
        let db = open(dir.path());
        let rows: Vec<(i64, i64, bool)> = db
            .query(
                &format!("SELECT id, {column}, s = '{text}' FROM {table} ORDER BY id"),
                (),
            )
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (
                    row.get(0).unwrap(),
                    row.get(1).unwrap(),
                    row.get(2).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            rows,
            vec![(1, 7, true), (2, 8, true)],
            "checkpoint {checkpoint}"
        );
        let indexes = db
            .query(&format!("SHOW INDEXES FROM {table}"), ())
            .unwrap()
            .count();
        assert!(
            indexes >= 1,
            "the HNSW index read back, checkpoint {checkpoint}"
        );
    }
}
