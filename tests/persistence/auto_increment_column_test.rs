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

//! AUTO_INCREMENT belongs to the INTEGER PRIMARY KEY: CREATE TABLE refuses it
//! on another column, and a table an older build wrote with one opens
//! without it. The fixtures are in tests/fixtures/auto_increment

use std::path::Path;

use stoolap::Database;

/// The Extra column DESCRIBE gives `column` of `table`
fn extra(db: &Database, table: &str, column: &str) -> String {
    db.query(&format!("DESCRIBE {table}"), ())
        .unwrap()
        .map(|row| row.unwrap())
        .find(|row| row.get::<String>(0).unwrap() == column)
        .map(|row| row.get::<String>(5).unwrap())
        .unwrap_or_else(|| panic!("no column {column} in {table}"))
}

#[test]
fn a_column_other_than_the_primary_key_takes_no_auto_increment() {
    let db = Database::open("memory://auto_increment_other_column").unwrap();
    for (sql, column) in [
        (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER AUTO_INCREMENT, v TEXT)",
            "n",
        ),
        (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT AUTO_INCREMENT)",
            "s",
        ),
        (
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, n INTEGER AUTO_INCREMENT)",
            "n",
        ),
        (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER AUTO_INCREMENT NOT NULL)",
            "n",
        ),
        ("CREATE TABLE t (n INTEGER AUTO_INCREMENT, v TEXT)", "n"),
    ] {
        let error = db.execute(sql, ()).expect_err(sql).to_string();
        let says =
            format!("AUTO_INCREMENT is allowed only on the INTEGER PRIMARY KEY, not on '{column}'");
        assert!(error.contains(&says), "{sql}: {error}");
        let tables = db.query("SHOW TABLES", ()).unwrap().count();
        assert_eq!(tables, 0, "{sql} created a table");
    }
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t (v) VALUES ('a')", ()).unwrap();
    let id: i64 = db.query_one("SELECT id FROM t", ()).unwrap();
    assert_eq!(id, 1);
    assert_eq!(extra(&db, "t", "id"), "auto_increment");
    let error = db
        .execute("ALTER TABLE t ADD COLUMN n INTEGER AUTO_INCREMENT", ())
        .expect_err("ADD COLUMN")
        .to_string();
    assert!(
        error.contains("does not take an AUTO_INCREMENT constraint"),
        "{error}"
    );
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn a_table_an_older_build_wrote_with_auto_increment_on_another_column_opens_without_it() {
    for name in ["wal_column", "snapshot_column"] {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/auto_increment")
            .join(name);
        let dir = tempfile::tempdir().unwrap();
        copy_tree(&fixture, dir.path());
        let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
        assert_eq!(extra(&db, "t", "n"), "", "{name}: DESCRIBE");
        let create: String = db
            .query("SHOW CREATE TABLE t", ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(1).unwrap())
            .next()
            .unwrap();
        assert!(!create.contains("AUTO_INCREMENT"), "{name}: {create}");
        db.execute("INSERT INTO t (id, v) VALUES (3, 'c')", ())
            .unwrap();
        let rows: Vec<(i64, Option<i64>)> = db
            .query("SELECT id, n FROM t ORDER BY id", ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (row.get(0).unwrap(), row.get(1).unwrap())
            })
            .collect();
        assert_eq!(rows, [(1, None), (2, None), (3, None)], "{name}");
        db.close().unwrap();
    }
}
