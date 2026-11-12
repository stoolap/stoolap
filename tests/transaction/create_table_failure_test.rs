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

//! A CREATE TABLE that fails on one of its indexes leaves no table, index or
//! foreign key behind, in a transaction or alone, and none after a reopen

use stoolap::Database;

/// Each fails on an index built after another of its indexes
const REFUSED: &[&str] = &[
    "CREATE TABLE c (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, UNIQUE (a))",
    "CREATE TABLE c (id INTEGER PRIMARY KEY, a INTEGER UNIQUE REFERENCES p (id), UNIQUE (a))",
    "CREATE TABLE c (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, \
     UNIQUE (a, b), UNIQUE (a, b), FOREIGN KEY (b) REFERENCES p (id))",
];

#[derive(Debug, Clone, Copy)]
enum Run {
    Alone,
    Committed,
    RolledBack,
    Reopened,
}

/// What the failed CREATE of `c` left in `db`
fn left_behind(db: &Database) -> Vec<String> {
    let mut left = Vec::new();
    if db.query("SELECT * FROM c", ()).is_ok() {
        left.push("table c".to_string());
    }
    if let Err(e) = db.execute("DROP TABLE p", ()) {
        left.push(format!("a key into p: {e}"));
    }
    if let Err(e) = db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY)", ()) {
        left.push(format!("CREATE TABLE c again: {e}"));
        return left;
    }
    for row in db.query("SHOW INDEXES FROM c", ()).unwrap() {
        let row = row.unwrap();
        for i in 0..2 {
            let text: String = row.get(i).unwrap();
            if text.contains("unique_c") || text.contains("fk_c") {
                left.push(format!("index {text}"));
            }
        }
    }
    left
}

fn refuse(db: &Database, sql: &str, run: Run, wrong: &mut Vec<String>) {
    let parent = "CREATE TABLE p (id INTEGER PRIMARY KEY)";
    db.execute(parent, ()).unwrap();
    let in_transaction = matches!(run, Run::Committed | Run::RolledBack);
    if in_transaction {
        db.execute("BEGIN", ()).unwrap();
    }
    if db.execute(sql, ()).is_ok() {
        wrong.push(format!("{run:?} {sql}: accepted"));
    }
    if in_transaction {
        if db.query("SELECT * FROM c", ()).is_ok() {
            wrong.push(format!("{run:?} {sql}: table c inside the transaction"));
        }
        let end = match run {
            Run::Committed => "COMMIT",
            _ => "ROLLBACK",
        };
        db.execute(end, ()).unwrap();
    }
}

#[test]
fn a_failed_create_table_leaves_nothing_behind() {
    let mut wrong = Vec::new();
    for (i, sql) in REFUSED.iter().enumerate() {
        for run in [Run::Alone, Run::Committed, Run::RolledBack] {
            let db = Database::open(&format!("memory://create_table_failure_{run:?}_{i}")).unwrap();
            refuse(&db, sql, run, &mut wrong);
            for left in left_behind(&db) {
                wrong.push(format!("{run:?} {sql}: {left}"));
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}", dir.path().display());
        let db = Database::open(&dsn).unwrap();
        refuse(&db, sql, Run::Reopened, &mut wrong);
        db.close().unwrap();
        drop(db);
        let db = Database::open(&dsn).unwrap();
        for left in left_behind(&db) {
            wrong.push(format!("{:?} {sql}: {left}", Run::Reopened));
        }
        db.close().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
