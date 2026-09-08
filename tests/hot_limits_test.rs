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

//! Hot size limits: `hot_max_rows` asks for a seal as soon as a commit
//! passes it, `hot_max_bytes` makes commits wait for that seal, and
//! PRAGMA MEMORY_STATS reports what the hot store holds.

use std::time::{Duration, Instant};
use stoolap::Database;
use tempfile::tempdir;

fn int(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .expect("query")
        .next()
        .expect("one row")
        .expect("row")
        .get::<i64>(0)
        .expect("integer")
}

/// (hot_rows, hot_bytes, chain_entries, admission_waits of the `*` row)
fn memory_stats(db: &Database, table: &str) -> (i64, i64, i64, i64) {
    let mut hot_rows = -1;
    let mut hot_bytes = -1;
    let mut chain_entries = -1;
    let mut waits = -1;
    for row in db.query("PRAGMA MEMORY_STATS", ()).expect("memory stats") {
        let row = row.expect("row");
        let name: String = row.get(0).expect("table_name");
        if name == table {
            hot_rows = row.get(1).expect("hot_rows");
            hot_bytes = row.get(2).expect("hot_bytes");
            chain_entries = row.get(5).expect("chain_entries");
        } else if name == "*" {
            waits = row.get(7).expect("admission_waits");
        }
    }
    (hot_rows, hot_bytes, chain_entries, waits)
}

fn insert_batch(db: &Database, table: &str, from: i64, count: i64) {
    let mut sql = format!("INSERT INTO {table} VALUES ");
    for i in from..from + count {
        if i != from {
            sql.push(',');
        }
        sql.push_str(&format!("({i}, 'row number {i} with a longer text value')"));
    }
    db.execute(&sql, ()).expect("insert");
}

#[test]
fn hot_limit_pragmas_read_the_dsn_and_accept_new_values() {
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&hot_max_rows=5000&hot_max_bytes=123456",
        dir.path().display()
    ))
    .expect("open");
    assert_eq!(int(&db, "PRAGMA HOT_MAX_ROWS"), 5000);
    assert_eq!(int(&db, "PRAGMA HOT_MAX_BYTES"), 123456);
    db.execute("PRAGMA HOT_MAX_ROWS = 0", ()).expect("set rows");
    db.execute("PRAGMA HOT_MAX_BYTES = 0", ())
        .expect("set bytes");
    assert_eq!(int(&db, "PRAGMA HOT_MAX_ROWS"), 0);
    assert_eq!(int(&db, "PRAGMA HOT_MAX_BYTES"), 0);
    assert!(db.execute("PRAGMA HOT_MAX_ROWS = -1", ()).is_err());
}

#[test]
fn memory_stats_follow_inserts_updates_and_vacuum() {
    let db = Database::open("memory://hot_limits_memory_stats").expect("open");
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, t TEXT)", ())
        .expect("create");
    let (rows, bytes, chains, _) = memory_stats(&db, "m");
    assert_eq!((rows, bytes, chains), (0, 0, 0));

    insert_batch(&db, "m", 1, 100);
    let (rows, bytes_after_insert, chains, _) = memory_stats(&db, "m");
    assert_eq!(rows, 100);
    assert_eq!(chains, 0);
    // 100 rows of two 16-byte values plus a heap string each
    assert!(
        bytes_after_insert > 100 * 48,
        "hot_bytes {bytes_after_insert}"
    );

    db.execute(
        "UPDATE m SET t = t || ' and some more text appended to it' WHERE id <= 50",
        (),
    )
    .expect("update");
    let (rows, bytes_after_update, chains, _) = memory_stats(&db, "m");
    assert_eq!(rows, 100);
    assert_eq!(chains, 50, "each updated row keeps one previous version");
    assert!(
        bytes_after_update > bytes_after_insert,
        "longer values must count: {bytes_after_update} <= {bytes_after_insert}"
    );

    db.execute("DELETE FROM m", ()).expect("delete");
    db.execute("VACUUM", ()).expect("vacuum");
    let (rows, bytes, _, _) = memory_stats(&db, "m");
    assert_eq!(rows, 0);
    assert_eq!(bytes, 0, "cleared slots must give their bytes back");
}

#[test]
fn hot_max_rows_seals_the_table_without_waiting_for_the_interval() {
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&hot_max_rows=20000",
        dir.path().display()
    ))
    .expect("open");
    db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, t TEXT)", ())
        .expect("create");
    for batch in 0..25 {
        insert_batch(&db, "s", batch * 2000 + 1, 2000);
    }
    assert_eq!(int(&db, "SELECT COUNT(*) FROM s"), 50_000);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut hot_rows = memory_stats(&db, "s").0;
    while hot_rows >= 20_000 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
        hot_rows = memory_stats(&db, "s").0;
    }
    assert!(
        hot_rows >= 0,
        "PRAGMA MEMORY_STATS has no row for the table"
    );
    assert!(
        hot_rows < 20_000,
        "the hot store still holds {hot_rows} rows: no seal was requested"
    );
    let volumes = db
        .query("PRAGMA VOLUME_STATS", ())
        .expect("volume stats")
        .count();
    assert!(volumes > 0, "the early seal must have written a volume");
    assert!(
        int(&db, "SELECT COUNT(*) FROM s") == 50_000,
        "the sealed rows must still be readable"
    );
    assert_eq!(int(&db, "SELECT MIN(id) FROM s"), 1);
    assert_eq!(int(&db, "SELECT MAX(id) FROM s"), 50_000);
    assert!(
        int(
            &db,
            "SELECT COUNT(*) FROM s WHERE id BETWEEN 19990 AND 20010"
        ) == 21,
        "rows on both sides of the trigger must be there"
    );
}

#[test]
fn hot_max_bytes_makes_commits_wait_for_the_seal() {
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&hot_max_rows=0&hot_max_bytes=200000",
        dir.path().display()
    ))
    .expect("open");
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, t TEXT)", ())
        .expect("create");
    let mut peak_bytes = 0;
    for batch in 0..30 {
        insert_batch(&db, "w", batch * 1000 + 1, 1000);
        peak_bytes = peak_bytes.max(memory_stats(&db, "w").1);
    }
    let (_, _, _, waits) = memory_stats(&db, "w");
    assert!(waits > 0, "no commit waited for admission");
    // One batch is well under 100 KB, so the hot store never holds more
    // than the limit plus the batch that crossed it
    assert!(
        peak_bytes < 300_000,
        "hot bytes reached {peak_bytes} with a 200000 byte limit"
    );
    assert_eq!(int(&db, "SELECT COUNT(*) FROM w"), 30_000);
    assert_eq!(int(&db, "SELECT SUM(id) FROM w"), 30_000 * 30_001 / 2);
}

#[test]
fn an_update_over_the_byte_limit_does_not_wait_on_its_own_claims() {
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&hot_max_rows=0",
        dir.path().display()
    ))
    .expect("open");
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, t TEXT)", ())
        .expect("create");
    insert_batch(&db, "u", 1, 2000);
    db.execute("PRAGMA HOT_MAX_BYTES = 1", ()).expect("limit");
    let started = Instant::now();
    db.execute("UPDATE u SET t = 'changed' WHERE id <= 1500", ())
        .expect("update");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the UPDATE waited {:?} for a seal its own claims block",
        started.elapsed()
    );
    assert_eq!(int(&db, "SELECT COUNT(*) FROM u WHERE t = 'changed'"), 1500);
}

#[test]
fn updates_that_grow_past_the_byte_limit_request_a_seal() {
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&hot_max_rows=0",
        dir.path().display()
    ))
    .expect("open");
    db.execute("CREATE TABLE g (id INTEGER PRIMARY KEY, t TEXT)", ())
        .expect("create");
    insert_batch(&db, "g", 1, 2000);
    let bytes_at_rest = memory_stats(&db, "g").1;
    db.execute(
        &format!("PRAGMA HOT_MAX_BYTES = {}", bytes_at_rest + 20_000),
        (),
    )
    .expect("limit");
    for _ in 0..4 {
        db.execute("UPDATE g SET t = t || ' grown by an update'", ())
            .expect("update");
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut volumes = 0;
    while volumes == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
        volumes = db
            .query("PRAGMA VOLUME_STATS", ())
            .expect("volume stats")
            .count();
    }
    assert!(
        volumes > 0,
        "no seal was requested for a table grown by UPDATEs"
    );
    assert_eq!(
        int(&db, "SELECT COUNT(*) FROM g WHERE t LIKE '%grown%'"),
        2000
    );
}

/// Under `test-filedb` a memory DSN opens a file database, which can seal
#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_memory_database_ignores_the_byte_limit() {
    let db = Database::open("memory://hot_limits_no_persistence").expect("open");
    db.execute("PRAGMA HOT_MAX_BYTES = 1", ()).expect("set");
    db.execute("CREATE TABLE n (id INTEGER PRIMARY KEY, t TEXT)", ())
        .expect("create");
    let started = Instant::now();
    insert_batch(&db, "n", 1, 500);
    insert_batch(&db, "n", 501, 500);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "a commit waited although nothing can seal"
    );
    assert_eq!(memory_stats(&db, "n").3, 0);
}
