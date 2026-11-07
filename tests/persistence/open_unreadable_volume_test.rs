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
        "ALTER TABLE t ADD COLUMN x INTEGER",
        "ALTER TABLE t MODIFY COLUMN v FLOAT",
        "ALTER TABLE t RENAME COLUMN v TO v2",
        "CREATE INDEX t_v ON t(v)",
        "TRUNCATE TABLE t",
        "ALTER TABLE t RENAME TO t2",
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
    let footer = bytes.len() - 8;
    bytes[footer] ^= 0xFF;
    std::fs::write(path, &bytes).unwrap();
    assert_t_refused(&dsn, "one flipped byte in a volume's footer");
}

#[test]
fn test_a_listed_volume_replaced_by_another_file_makes_its_table_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "replaced_volume");
    let other = std::fs::read_dir(volumes[0].parent().unwrap().with_file_name("u"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "vol"))
        .unwrap();
    std::fs::copy(&other, &volumes[0]).unwrap();
    assert_t_refused(&dsn, "a volume of t replaced by a readable file of u");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_a_file_swapped_under_an_evicted_volume_keeps_the_recorded_identity() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "swapped_after_eviction");
    let other = std::fs::read_dir(volumes[0].parent().unwrap().with_file_name("u"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "vol"))
        .unwrap();
    let db = Database::open(&dsn).unwrap();
    let (listed, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!(cold, listed, "t's volumes went cold");
    std::fs::copy(&other, &volumes[0]).unwrap();
    // The reload reads the swapped file, then a checkpoint persists t
    run(&db, "SELECT * FROM t WHERE v > 100 LIMIT 3").unwrap();
    db.execute("INSERT INTO t VALUES (9999, 1, 9999)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    drop(db);
    assert_t_refused(&dsn, "a file swapped under an evicted volume of t");
}

#[test]
fn test_a_damaged_block_fails_the_statements_that_read_it() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "damaged_block");
    let path = &volumes[0];
    let mut bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..4], b"STV5");
    bytes[16] ^= 0xFF;
    std::fs::write(path, &bytes).unwrap();
    let db = Database::open(&dsn).unwrap();
    match run(&db, "SELECT * FROM t") {
        Err(msg) => assert!(msg.contains("CRC mismatch"), "got: {msg}"),
        Ok(n) => panic!("a damaged block was read: {n} rows"),
    }
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM u", ()).unwrap();
    assert_eq!(rows, ROWS, "the intact table answers whole");
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
fn test_a_hot_row_of_an_unavailable_table_is_refused_too() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "hot_row_volume");
    // Rows 5 and 50 change after the checkpoint: the log brings them back hot
    let no_close_checkpoint = format!("{dsn}?checkpoint_on_close=off");
    let db = Database::open(&no_close_checkpoint).unwrap();
    db.execute("UPDATE t SET v = 1 WHERE id = 5", ()).unwrap();
    db.execute("UPDATE t SET v = 1 WHERE id = 50", ()).unwrap();
    db.close().unwrap();
    std::fs::remove_file(&volumes[0]).unwrap();
    let db = Database::open(&no_close_checkpoint).unwrap();
    // The open seals no row of an unavailable table, so the two stay hot
    let hot_rows: i64 = db
        .query("PRAGMA MEMORY_STATS", ())
        .unwrap()
        .map(|row| row.unwrap())
        .find(|row| row.get::<String>(0).unwrap() == "t")
        .map(|row| row.get(1).unwrap())
        .unwrap();
    assert_eq!(hot_rows, 2, "rows 5 and 50 are hot when they are read");
    let point = db.prepare("SELECT v FROM t WHERE id = $1").unwrap();
    for (what, answer) in [
        ("a point read", run(&db, "SELECT v FROM t WHERE id = 5")),
        ("a unique read", run(&db, "SELECT v FROM t WHERE w = 50")),
        (
            "a prepared point read",
            point
                .query((5,))
                .map_err(|e| e.to_string())
                .and_then(|rows| {
                    let mut n = 0;
                    for row in rows {
                        row.map_err(|e| e.to_string())?;
                        n += 1;
                    }
                    Ok(n)
                }),
        ),
    ] {
        match answer {
            Err(msg) => assert!(
                msg.contains("'t' is unavailable"),
                "{what} of a hot row should name the table, got: {msg}"
            ),
            Ok(n) => panic!("{what} of a hot row must be refused, it answered {n} rows"),
        }
    }
}

#[test]
fn test_a_checkpoint_keeps_the_log_of_an_unavailable_table() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "logged_delete_volume");
    // Row 7 lives only in t's volumes; its DELETE only in the log
    let no_close_checkpoint = format!("{dsn}?checkpoint_on_close=off");
    let db = Database::open(&no_close_checkpoint).unwrap();
    db.execute("DELETE FROM t WHERE id = 7", ()).unwrap();
    db.close().unwrap();
    let intact: Vec<_> = (volumes.iter())
        .map(|path| (path.clone(), std::fs::read(path).unwrap()))
        .collect();
    for path in &volumes {
        std::fs::remove_file(path).unwrap();
    }
    // A checkpoint and a closing one while t cannot read its volumes
    let db = Database::open(&dsn).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    for (path, bytes) in &intact {
        std::fs::write(path, bytes).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    let seven: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 7", ())
        .unwrap();
    assert_eq!(seven, 0, "the deleted row stays deleted");
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(rows, ROWS - 1);
}

#[test]
fn test_an_unavailable_table_can_be_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "dropped_volume");
    std::fs::remove_file(&volumes[0]).unwrap();
    let db = Database::open(&dsn).unwrap();
    db.execute("DROP TABLE t", ()).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(rows, 0, "a table created under the name starts empty");
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

/// Every file under `dir`, recursively, with its bytes
fn files_under(dir: &std::path::Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(files_under(&path));
        } else {
            let bytes = std::fs::read(&path).unwrap();
            out.push((path, bytes));
        }
    }
    out.sort();
    out
}

#[test]
fn test_a_snapshot_of_an_unavailable_table_fails_and_keeps_the_last_backup() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volumes) = sealed(dir.path(), "snapshot_volume");
    let dsn = format!("{dsn}?keep_snapshots=1");
    let db = Database::open(&dsn).unwrap();
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();
    db.close().unwrap();
    let snapshots = dir.path().join("snapshot_volume").join("snapshots");
    let backup = files_under(&snapshots);
    std::fs::remove_file(&volumes[0]).unwrap();
    let db = Database::open(&dsn).unwrap();
    match db.execute("PRAGMA SNAPSHOT", ()) {
        Err(e) => assert!(
            e.to_string().contains("'t' is unavailable"),
            "the snapshot should name the unavailable table, got: {e}"
        ),
        Ok(_) => panic!("a snapshot must not be taken without t's missing rows"),
    }
    assert_eq!(
        files_under(&snapshots),
        backup,
        "the last good backup stays as it was"
    );
}

#[test]
fn test_no_compaction_of_an_unavailable_table_drops_a_tombstone() {
    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join("compacted_volume");
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000&target_volume_rows=65536",
        db_dir.display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let ids: Vec<i64> = (1..=65_536).collect();
    for chunk in ids.chunks(8_192) {
        let values: Vec<String> = chunk.iter().map(|id| format!("({id}, 0)")).collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // Newer copies of 5 and of a new row in small volumes
    db.execute("UPDATE t SET v = 1 WHERE id = 5", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (100000, 1)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    // Row 5 deleted with no checkpoint after it: the tombstone comes back
    // from the log, and the next compaction is the first to see it
    let no_close_checkpoint = format!("{dsn}&checkpoint_on_close=off");
    let db = Database::open(&no_close_checkpoint).unwrap();
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
    db.close().unwrap();
    let full = std::fs::read_dir(db_dir.join("volumes").join("t"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "vol"))
        .max_by_key(|path| std::fs::metadata(path).unwrap().len())
        .unwrap();
    let intact = std::fs::read(&full).unwrap();
    std::fs::remove_file(&full).unwrap();
    // With the full volume gone, a compaction of the small ones must not
    // decide that nothing else holds row 5
    let db = Database::open(&dsn).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 1", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    std::fs::write(&full, &intact).unwrap();
    let db = Database::open(&dsn).unwrap();
    let five: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 5", ())
        .unwrap();
    assert_eq!(five, 0, "the deleted row stays deleted");
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(rows, 65_536);
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
