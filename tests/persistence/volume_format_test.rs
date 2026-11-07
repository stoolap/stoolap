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

use stoolap::Database;

#[test]
fn test_keys_up_to_the_largest_integer_survive_a_checkpoint_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("largest_key").display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE one (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE run (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO one VALUES ($1, 1)", (i64::MAX,))
        .unwrap();
    for id in [i64::MAX - 2, i64::MAX - 1, i64::MAX] {
        db.execute("INSERT INTO run VALUES ($1, 1)", (id,)).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    drop(db);
    for table in ["one", "run"] {
        let volumes = std::fs::read_dir(dir.path().join("largest_key/volumes").join(table))
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|x| x == "vol")
            })
            .count();
        assert_eq!(volumes, 1, "{table} was not sealed");
    }

    let db = Database::open(&dsn).unwrap();
    for (table, rows) in [("one", 1), ("run", 3)] {
        let ids: Vec<i64> = db
            .query(&format!("SELECT id FROM {table} ORDER BY id"), ())
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(ids.len(), rows, "{table}: {ids:?}");
        assert_eq!(ids.last(), Some(&i64::MAX), "{table}");
    }
}
