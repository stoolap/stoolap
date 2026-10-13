// Copyright 2026 Stoolap Contributors
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

//! Databases an older build left after a column type change open with every
//! layer in the schema type, or refuse to open with their files kept. The
//! fixtures and how they were written are in tests/fixtures/retype

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stoolap::Database;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/retype")
        .join(name)
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

/// Every file under `root` with its content, by path relative to `root`;
/// the lock file an open database holds (locked against reads on Windows)
/// is left out
fn contents(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else if path.file_name().is_some_and(|name| name != "db.lock") {
                let content = std::fs::read(&path).unwrap();
                out.insert(path.strip_prefix(root).unwrap().to_path_buf(), content);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

fn open(name: &str) -> (tempfile::TempDir, stoolap::Result<Database>) {
    let dir = tempfile::tempdir().unwrap();
    copy_tree(&fixture(name), dir.path());
    let db = Database::open(&format!("file://{}", dir.path().display()));
    (dir, db)
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| format!("{:?}", r.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

fn refused_by(db: &Database, sql: &str) -> Option<i64> {
    match db.execute(sql, ()) {
        Ok(_) => None,
        Err(stoolap::Error::UniqueConstraint { row_id, .. }) => Some(row_id),
        Err(e) => panic!("{sql}: {e:?}"),
    }
}

#[test]
fn old_typed_volumes_open_cast_to_the_schema_type() {
    let (_dir, db) = open("old_typed_volumes");
    let db = db.unwrap();
    assert_eq!(
        rows(&db, "SELECT id, k FROM t ORDER BY id"),
        ["Integer(1),Text(\"7\")", "Integer(2),Text(\"8\")"]
    );
    assert_eq!(rows(&db, "SELECT id FROM t WHERE k = '7'"), ["Integer(1)"]);
    assert_eq!(refused_by(&db, "INSERT INTO t VALUES (3, '7')"), Some(1));
}

#[test]
fn volumes_that_collide_once_cast_refuse_to_open_and_keep_their_files() {
    let source = contents(&fixture("colliding_volumes"));
    let (dir, db) = open("colliding_volumes");
    let error = match db {
        Ok(db) => panic!("opened: {:?}", rows(&db, "SELECT id, k FROM t ORDER BY id")),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("t") && error.to_lowercase().contains("unique"),
        "{error}"
    );
    let after = contents(dir.path());
    for (path, content) in &source {
        assert_eq!(after.get(path), Some(content), "{} kept", path.display());
    }
}

#[test]
fn a_row_logged_before_a_change_and_marked_after_it_recovers_in_the_new_type() {
    let (_dir, db) = open("row_alter_marker");
    let db = db.unwrap();
    assert_eq!(
        rows(&db, "SELECT id, k FROM t ORDER BY id"),
        ["Integer(1),Text(\"7\")", "Integer(2),Text(\"8\")"]
    );
    assert_eq!(refused_by(&db, "INSERT INTO t VALUES (3, '8')"), Some(2));
}

#[test]
fn a_type_change_whose_replay_fails_stops_the_open_and_keeps_the_log() {
    let source = contents(&fixture("hot_colliding"));
    let (dir, db) = open("hot_colliding");
    let error = match db {
        Ok(db) => panic!("opened: {:?}", rows(&db, "SELECT id, k FROM t ORDER BY id")),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("column type change"), "{error}");
    let after = contents(dir.path());
    for (path, content) in &source {
        assert_eq!(after.get(path), Some(content), "{} kept", path.display());
    }
}

#[test]
fn replay_casts_at_each_change_in_order() {
    let (_dir, db) = open("text_integer_text");
    let db = db.unwrap();
    assert_eq!(
        rows(&db, "SELECT id, k FROM t"),
        ["Integer(1),Text(\"7\")"],
        "'007' became 7, then '7'"
    );
}

#[test]
fn a_type_change_over_unversioned_renames_is_refused_and_changes_no_file() {
    for (name, column) in [
        ("unversioned_renames", "k"),
        ("sealed_between_renames", "k"),
        ("dropped_before_seal", "k"),
        ("passed_name_added_again", "k"),
        ("dropped_after_reopen", "c"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        copy_tree(&fixture(name), dir.path());
        let dsn = format!("file://{}?checkpoint_interval=3600", dir.path().display());
        let db = Database::open(&dsn).unwrap_or_else(|error| panic!("{name}: {error}"));
        let before = contents(dir.path());
        let refused = db.execute(&format!("ALTER TABLE t MODIFY COLUMN {column} TEXT"), ());
        let message = match refused {
            Err(stoolap::Error::NotSupported(message)) => message,
            other => panic!("{name}: {other:?}"),
        };
        assert!(
            message.contains("table 't'") && message.contains("recreate the table"),
            "{name}: {message}"
        );
        assert_eq!(contents(dir.path()), before, "{name}: a file changed");
        db.close().unwrap();
        Database::open(&dsn).unwrap_or_else(|error| panic!("{name} reopened: {error}"));
    }
}

#[test]
fn an_old_typed_volume_under_unversioned_renames_refuses_to_open_and_keeps_the_files() {
    let source = contents(&fixture("unversioned_old_typed"));
    let (dir, db) = open("unversioned_old_typed");
    let error = match db {
        Ok(db) => panic!("opened: {:?}", rows(&db, "SELECT id, b, k FROM t")),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("table 't'") && error.contains("recreate the table"),
        "{error}"
    );
    let after = contents(dir.path());
    for (path, content) in &source {
        assert_eq!(after.get(path), Some(content), "{} kept", path.display());
    }
}

#[test]
fn a_volume_whose_reading_turns_on_an_old_drop_is_not_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    copy_tree(&fixture("name_dropped_after_reopen"), dir.path());
    let dsn = format!("file://{}?checkpoint_interval=3600", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    let before = contents(dir.path());
    let message = match db.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ()) {
        Err(stoolap::Error::NotSupported(message)) => message,
        other => panic!("{other:?}"),
    };
    assert!(
        message.contains("table 't'")
            && message.contains("'x'")
            && message.contains("recreate the table"),
        "{message}"
    );
    assert_eq!(contents(dir.path()), before, "a file changed");
    db.close().unwrap();
    Database::open(&dsn).unwrap();
}
