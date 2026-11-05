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

//! A parent delete finds its children by the column the foreign key names

use stoolap::Database;

fn open(dir: &tempfile::TempDir) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap()
}

/// The child rows, as `id/k` with `k` NULL written `-`
fn children(db: &Database) -> Vec<String> {
    db.query("SELECT id, k FROM c ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            let k: Option<i64> = row.get(1).ok();
            format!(
                "{}/{}",
                row.get::<i64>(0).unwrap(),
                k.map_or("-".into(), |k| k.to_string())
            )
        })
        .collect()
}

#[test]
fn a_delete_reads_the_key_column_the_foreign_key_names() {
    let mut wrong = Vec::new();
    // Parent 1 holds key 7 and parent 2 key 1; child 1 points at key 7 and
    // child 2 at key 1, the deleted parent's primary key
    for parent_key in [
        "id INTEGER PRIMARY KEY, k INTEGER UNIQUE",
        "id INTEGER, k INTEGER UNIQUE",
    ] {
        for sealed in [false, true] {
            for (action, refused, want) in [
                ("ON DELETE RESTRICT", true, ["1/7", "2/1"].as_slice()),
                ("ON DELETE CASCADE", false, ["2/1"].as_slice()),
                ("ON DELETE SET NULL", false, ["1/-", "2/1"].as_slice()),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let db = open(&dir);
                for sql in [
                    format!("CREATE TABLE p ({parent_key})"),
                    format!(
                        "CREATE TABLE c (id INTEGER PRIMARY KEY, k INTEGER REFERENCES p(k) {action})"
                    ),
                    "INSERT INTO p VALUES (1, 7), (2, 1)".to_string(),
                    "INSERT INTO c VALUES (1, 7), (2, 1)".to_string(),
                ] {
                    db.execute(&sql, ()).unwrap();
                }
                if sealed {
                    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
                }
                let result = db.execute("DELETE FROM p WHERE id = 1", ());
                let got = children(&db);
                if result.is_err() != refused || got != want {
                    wrong.push(format!(
                        "({parent_key}) sealed {sealed} {action}: {result:?}, children {got:?}, \
                         want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
