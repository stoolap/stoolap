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

//! A statement that writes one column more than once is refused, whatever
//! the case of the names and wherever the rows live

use stoolap::Database;

const WRITTEN_TWICE: [&str; 12] = [
    "UPDATE t SET v = 5, v = 6",
    "UPDATE t SET v = 1, V = v + 1 WHERE v >= 1",
    "UPDATE t SET v = 5, s = 'x', V = 6 WHERE id = 1",
    "UPDATE t SET v = $1, v = $2 WHERE id = 1",
    "INSERT INTO t (id, v, v) VALUES (3, 1, 2)",
    "INSERT INTO t (id, v, V) VALUES (3, 1, 2)",
    "INSERT INTO t (id, v, v) SELECT id + 10, v, v FROM t",
    "INSERT INTO t (id, v) VALUES (1, 9) ON CONFLICT (id) DO UPDATE SET v = 5, v = 6",
    "INSERT INTO t (id, v) VALUES (1, 9) ON CONFLICT (id) DO UPDATE SET v = 5, V = EXCLUDED.v",
    "INSERT INTO t (id, v) VALUES (1, 9) ON DUPLICATE KEY UPDATE v = 7, V = 8",
    "INSERT INTO t (id, v) SELECT 1, 9 ON DUPLICATE KEY UPDATE v = 7, v = 8",
    "COPY t (id, v, V) FROM '/nonexistent.csv'",
];

/// t holding (1, 1, 'a') and (2, 2, 'b'), sealed when `sealed`
fn database(sealed: bool, dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 'a'), (2, 2, 'b')", ())
        .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database) -> Vec<(i64, i64, String)> {
    db.query("SELECT id, v, s FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get(0).unwrap(),
                row.get(1).unwrap(),
                row.get(2).unwrap(),
            )
        })
        .collect()
}

fn refused(sql: &str, outcome: stoolap::Result<i64>) {
    let error = outcome.expect_err(sql).to_string();
    assert!(
        error.contains("column v is written more than once"),
        "{sql}: {error}"
    );
}

#[test]
fn written_twice_is_refused_on_every_path() {
    let before = vec![(1, 1, "a".to_string()), (2, 2, "b".to_string())];
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(sealed, &dir);
        for sql in WRITTEN_TWICE {
            refused(sql, db.execute(sql, (5, 6)));
            refused(sql, db.prepare(sql).map(|_| 0));
            let mut tx = db.begin().unwrap();
            refused(sql, tx.execute(sql, (5, 6)));
            tx.rollback().unwrap();
        }
        assert_eq!(rows(&db), before, "sealed {sealed}: the rows are untouched");
    }
}

#[test]
fn written_twice_compares_names_beyond_ascii_without_case() {
    let db = Database::open("memory://written_twice_beyond_ascii").unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, \"größe\" INTEGER)",
        (),
    )
    .unwrap();
    for sql in [
        "UPDATE u SET \"größe\" = 1, \"GRÖßE\" = 2",
        "INSERT INTO u (id, \"größe\", \"Größe\") VALUES (1, 2, 3)",
    ] {
        let error = db.execute(sql, ()).expect_err(sql).to_string();
        assert!(
            error.contains("column größe is written more than once"),
            "{sql}: {error}"
        );
    }
}

#[test]
fn written_twice_allows_each_column_once() {
    let dir = tempfile::tempdir().unwrap();
    let db = database(false, &dir);
    for sql in [
        "UPDATE t SET v = 5, s = 'x' WHERE id = 1",
        "UPDATE t SET V = v + 1, S = s WHERE id = 2",
        "INSERT INTO t (id, V, s) VALUES (3, 3, 'c')",
        "INSERT INTO t (id, v) VALUES (1, 9) ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v, s = 'y'",
    ] {
        db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    assert_eq!(
        rows(&db),
        vec![
            (1, 9, "y".to_string()),
            (2, 3, "b".to_string()),
            (3, 3, "c".to_string()),
        ]
    );
}
