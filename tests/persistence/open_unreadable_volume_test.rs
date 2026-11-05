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

//! A volume the manifest lists holds rows that exist nowhere else, so a
//! table whose listed volume does not read, or is gone, refuses every
//! statement after the open instead of answering without those rows; the
//! other tables of the database answer as before.

use stoolap::Database;

const ROWS: i64 = 2_000;

/// A file database with tables t and u sealed into volumes, closed, with
/// the paths of t's volume files
fn sealed(dir: &std::path::Path, name: &str) -> (String, Vec<std::path::PathBuf>) {
    let db_dir = dir.join(name);
    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER UNIQUE)"),
            (),
        )
        .unwrap();
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=ROWS {
            db.execute(
                &format!("INSERT INTO {table} VALUES ($1, $2, $3)"),
                (i, i * 10, i),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    let mut volumes: Vec<_> = std::fs::read_dir(db_dir.join("volumes").join("t"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "vol"))
        .collect();
    volumes.sort();
    assert!(!volumes.is_empty(), "the checkpoint sealed no volume of t");
    (dsn, volumes)
}

/// The rows `sql` answers, read to the end, or its first error
fn run(db: &Database, sql: &str) -> Result<usize, String> {
    if !sql.starts_with("SELECT") {
        return db
            .execute(sql, ())
            .map(|n| n as usize)
            .map_err(|e| e.to_string());
    }
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut n = 0;
    for row in rows {
        row.map_err(|e| e.to_string())?;
        n += 1;
    }
    Ok(n)
}

fn assert_t_refused(dsn: &str, what: &str) {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "SELECT COUNT(*) FROM t",
        "SELECT SUM(v) FROM t",
        "SELECT MAX(id) FROM t",
        "SELECT MIN(v) FROM t",
        "SELECT v FROM t WHERE id = 5",
        "SELECT id FROM t WHERE w = 50",
        "SELECT * FROM t WHERE v > 100 LIMIT 3",
        "INSERT INTO t VALUES (9999, 1, 9999)",
        "UPDATE t SET v = 0 WHERE id = 5",
        "DELETE FROM t WHERE id = 6",
    ] {
        match run(&db, sql) {
            Err(msg) => assert!(
                msg.contains("'t' is unavailable"),
                "{what}: {sql} should name the unavailable table, got: {msg}"
            ),
            Ok(n) => panic!("{what}: {sql} must be refused, it answered {n} rows"),
        }
    }
    let point = db.prepare("SELECT v FROM t WHERE id = $1").unwrap();
    let prepared = point
        .query((7,))
        .map_err(|e| e.to_string())
        .and_then(|rows| {
            let mut n = 0;
            for row in rows {
                row.map_err(|e| e.to_string())?;
                n += 1;
            }
            Ok(n)
        });
    match prepared {
        Err(msg) => assert!(
            msg.contains("'t' is unavailable"),
            "{what}: a prepared point read should name the table, got: {msg}"
        ),
        Ok(n) => panic!("{what}: a prepared point read must be refused, it answered {n} rows"),
    }
    db.execute("BEGIN", ()).unwrap();
    let in_txn = run(&db, "SELECT v FROM t WHERE id = 8");
    db.execute("ROLLBACK", ()).unwrap();
    match in_txn {
        Err(msg) => assert!(
            msg.contains("'t' is unavailable"),
            "{what}: a read in a transaction should name the table, got: {msg}"
        ),
        Ok(n) => panic!("{what}: a read in a transaction must be refused, it answered {n} rows"),
    }
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM u", ()).unwrap();
    assert_eq!(rows, ROWS, "{what}: the intact table answers whole");
}

#[test]
fn test_a_listed_volume_that_does_not_read_makes_its_table_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "damaged_volume");
    let path = &volumes[0];
    let mut bytes = std::fs::read(path).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xFF;
    std::fs::write(path, &bytes).unwrap();
    assert_t_refused(&dsn, "one flipped byte in a volume");
}

#[test]
fn test_a_listed_volume_whose_file_is_gone_makes_its_table_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "missing_volume");
    for path in &volumes {
        std::fs::remove_file(path).unwrap();
    }
    assert_t_refused(&dsn, "every volume file of t deleted");
}

#[test]
fn test_a_volume_put_back_makes_its_table_available_again() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "restored_volume");
    let path = &volumes[0];
    let intact = std::fs::read(path).unwrap();
    std::fs::remove_file(path).unwrap();
    // An open, a statement refused and a closing checkpoint keep the segment listed
    let db = Database::open(&dsn).unwrap();
    assert!(run(&db, "SELECT COUNT(*) FROM t").is_err());
    db.close().unwrap();
    std::fs::write(path, &intact).unwrap();
    let db = Database::open(&dsn).unwrap();
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(rows, ROWS, "the table answers whole once its file is back");
}

#[test]
fn test_intact_volumes_answer_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, _) = sealed(dir.path(), "intact_volume");
    let db = Database::open(&dsn).unwrap();
    for table in ["t", "u"] {
        let rows: i64 = db
            .query_one(&format!("SELECT COUNT(*) FROM {table}"), ())
            .unwrap();
        assert_eq!(rows, ROWS, "{table}");
    }
}
