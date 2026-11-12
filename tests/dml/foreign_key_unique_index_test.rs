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

//! A column kept unique serves its foreign key's lookups: a UNIQUE constraint
//! on the column, inline or at table level, stands in for the key's own
//! index, and a unique index created later takes that index's place

use std::path::Path;

use stoolap::Database;

/// The table's indexes, by name, and whether each is unique
fn indexes(db: &Database, table: &str) -> Vec<(String, bool)> {
    let mut found: Vec<(String, bool)> = db
        .query(&format!("SHOW INDEXES FROM {table}"), ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<String>(1).unwrap(), row.get::<bool>(4).unwrap())
        })
        .collect();
    found.sort();
    found
}

/// The foreign key and the uniqueness of `pid` both hold on `table`, which
/// holds the row (1, 1) and no row with id 100 or 101
fn rules_hold(db: &Database, table: &str, unique: bool, wrong: &mut Vec<String>) {
    if db
        .execute(&format!("INSERT INTO {table} VALUES (100, 99)"), ())
        .is_ok()
    {
        wrong.push(format!("{table}: a row with no parent was accepted"));
    }
    let duplicate = db.execute(&format!("INSERT INTO {table} VALUES (101, 1)"), ());
    if unique && duplicate.is_ok() {
        wrong.push(format!("{table}: a duplicate pid was accepted"));
    }
}

fn parent(db: &Database) {
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2), (3)", ())
        .unwrap();
}

#[test]
fn a_unique_constraint_serves_the_foreign_key_of_its_column() {
    let db = Database::open("memory://fk_unique_constraint").unwrap();
    parent(&db);
    let mut wrong = Vec::new();
    for (table, sql, want) in [
        (
            "c1",
            "CREATE TABLE c1 (id INTEGER PRIMARY KEY, pid INTEGER UNIQUE, \
             FOREIGN KEY (pid) REFERENCES p (id))",
            vec![("unique_c1_pid", true)],
        ),
        (
            "c2",
            "CREATE TABLE c2 (id INTEGER PRIMARY KEY, pid INTEGER, UNIQUE (pid), \
             FOREIGN KEY (pid) REFERENCES p (id))",
            vec![("unique_c2_0", true)],
        ),
        (
            "c3",
            "CREATE TABLE c3 (id INTEGER PRIMARY KEY, pid INTEGER, x INTEGER, UNIQUE (pid, x), \
             FOREIGN KEY (pid) REFERENCES p (id))",
            vec![("fk_c3_pid", false), ("unique_c3_0", true)],
        ),
    ] {
        if let Err(e) = db.execute(sql, ()) {
            wrong.push(format!("{table}: {e}"));
            continue;
        }
        let want: Vec<(String, bool)> = want.iter().map(|(n, u)| (n.to_string(), *u)).collect();
        let found: Vec<(String, bool)> = indexes(&db, table)
            .into_iter()
            .filter(|(name, _)| !name.starts_with("__pk"))
            .collect();
        if found != want {
            wrong.push(format!("{table}: indexes {found:?}, want {want:?}"));
        }
        db.execute(&format!("INSERT INTO {table} (id, pid) VALUES (1, 1)"), ())
            .unwrap();
        rules_hold(&db, table, table != "c3", &mut wrong);
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn child(db: &Database, table: &str, rows: &str) {
    db.execute(
        &format!(
            "CREATE TABLE {table} (id INTEGER PRIMARY KEY, pid INTEGER, \
             FOREIGN KEY (pid) REFERENCES p (id))"
        ),
        (),
    )
    .unwrap();
    db.execute(&format!("INSERT INTO {table} VALUES {rows}"), ())
        .unwrap();
}

/// A unique index on `table`'s key column replaces the key's own index, or,
/// over duplicates, fails and leaves it
fn replace(db: &Database, table: &str, create: &str, duplicates: bool, wrong: &mut Vec<String>) {
    let created = db.execute(&create.replace("{t}", table), ());
    let fk = (format!("fk_{table}_pid"), false);
    let unique = (format!("u_{table}"), true);
    let found = indexes(db, table);
    if duplicates {
        if created.is_ok() {
            wrong.push(format!(
                "{table}: a unique index over duplicates was created"
            ));
        }
        if !found.contains(&fk) || found.contains(&unique) {
            wrong.push(format!(
                "{table}: after the failed create the indexes are {found:?}"
            ));
        }
        rules_hold(db, table, false, wrong);
        return;
    }
    if let Err(e) = created {
        wrong.push(format!("{table}: {e}"));
        return;
    }
    if found.contains(&fk) || !found.contains(&unique) {
        wrong.push(format!("{table}: indexes {found:?}"));
    }
    rules_hold(db, table, true, wrong);
}

const CREATES: [&str; 2] = [
    "CREATE UNIQUE INDEX u_{t} ON {t} (pid)",
    "CREATE UNIQUE INDEX IF NOT EXISTS u_{t} ON {t} (pid)",
];

#[test]
fn a_unique_index_takes_the_place_of_the_foreign_keys_own_index() {
    let db = Database::open("memory://fk_unique_index_hot").unwrap();
    parent(&db);
    let mut wrong = Vec::new();
    for (i, create) in CREATES.iter().enumerate() {
        child(&db, &format!("h{i}"), "(1, 1), (2, 2)");
        replace(&db, &format!("h{i}"), create, false, &mut wrong);
        child(&db, &format!("d{i}"), "(1, 1), (2, 1)");
        replace(&db, &format!("d{i}"), create, true, &mut wrong);
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn a_unique_index_replaces_the_key_index_over_sealed_rows_and_after_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let image = tempfile::tempdir().unwrap();
    let dsn = |path: &Path| format!("file://{}?checkpoint_interval=3600", path.display());
    let db = Database::open(&dsn(dir.path())).unwrap();
    parent(&db);
    child(&db, "s", "(1, 1), (2, 2)");
    child(&db, "sd", "(1, 1), (2, 1)");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut wrong = Vec::new();
    replace(&db, "s", CREATES[0], false, &mut wrong);
    replace(&db, "sd", CREATES[0], true, &mut wrong);
    crash_image(dir.path(), image.path());
    db.close().unwrap();

    let db = Database::open(&dsn(image.path())).unwrap();
    let found = indexes(&db, "s");
    if found.contains(&("fk_s_pid".to_string(), false))
        || !found.contains(&("u_s".to_string(), true))
    {
        wrong.push(format!("s after the crash: indexes {found:?}"));
    }
    rules_hold(&db, "s", true, &mut wrong);
    db.close().unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
