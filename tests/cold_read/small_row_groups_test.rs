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

//! A volume written with row groups smaller than the writer's reads as
//! the same rows through every path: scans, filters, point and range
//! reads, an index, a join, and a compaction that rewrites it.

#![cfg(feature = "test-failpoints")]

use stoolap::storage::volume::io::rewrite_volume_with_group_rows;
use stoolap::Database;

const ROWS: i64 = 150_000;

fn k_of(i: i64) -> i64 {
    i % 1000
}

fn s_of(i: i64) -> String {
    format!("s{}", i % 97)
}

/// A file database with t sealed into one volume, rewritten in groups of
/// `group_rows` rows, and u, whose rows name rows of t
fn sealed(dir: &std::path::Path, group_rows: usize) -> String {
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, f FLOAT)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
        db.execute(
            &format!(
                "INSERT INTO t SELECT g.value, g.value % 1000, 's' || (g.value % 97), \
                 g.value / 2.0 FROM generate_series(1, {ROWS}) g"
            ),
            (),
        )
        .unwrap();
        db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, t_id INTEGER)", ())
            .unwrap();
        db.execute(
            "INSERT INTO u SELECT g.value, g.value * 997 FROM generate_series(1, 150) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    let volumes: Vec<_> = std::fs::read_dir(dir.join("volumes").join("t"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "vol"))
        .collect();
    assert_eq!(volumes.len(), 1, "t sealed into one volume");
    rewrite_volume_with_group_rows(&volumes[0], group_rows).unwrap();
    dsn
}

fn int(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn a_volume_of_small_row_groups_answers_every_read() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = sealed(dir.path(), 4096);
    let db = Database::open(&dsn).unwrap();
    for cache_mb in [0, 64] {
        db.execute(&format!("PRAGMA GROUP_CACHE_MB = {cache_mb}"), ())
            .unwrap();
        let what = format!("group cache {cache_mb} MB");
        let small: Vec<i64> = (1..=ROWS).filter(|&i| k_of(i) < 10).collect();
        assert_eq!(
            int(&db, "SELECT COUNT(*) FROM t WHERE k < 10"),
            small.len() as i64,
            "{what}"
        );
        assert_eq!(
            int(&db, "SELECT SUM(k + 0) FROM t WHERE k < 10"),
            small.iter().map(|&i| k_of(i)).sum::<i64>(),
            "{what}"
        );
        let newest: Vec<i64> = (1..=ROWS)
            .rev()
            .filter(|&i| s_of(i) == "s5")
            .take(3)
            .collect();
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t WHERE s = 's5' ORDER BY id DESC LIMIT 3"
            ),
            newest,
            "{what}"
        );
        assert_eq!(
            int(&db, "SELECT k FROM t WHERE id = 123457"),
            k_of(123_457),
            "{what}"
        );
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t WHERE id IN (5, 70000, 149999) ORDER BY id"
            ),
            vec![5, 70_000, 149_999],
            "{what}"
        );
        let range: f64 = (60_000..=70_000).map(|i| i as f64 / 2.0).sum();
        let got: f64 = db
            .query_one("SELECT SUM(f) FROM t WHERE id BETWEEN 60000 AND 70000", ())
            .unwrap();
        assert!((got - range).abs() < 1e-6, "{what}: {got} vs {range}");
        assert_eq!(
            int(&db, "SELECT COUNT(*) FROM t WHERE k = 7"),
            (1..=ROWS).filter(|&i| k_of(i) == 7).count() as i64,
            "{what}"
        );
        assert_eq!(
            int(&db, "SELECT SUM(t.k) FROM u JOIN t ON t.id = u.t_id"),
            (1..=150).map(|i| k_of(i * 997)).sum::<i64>(),
            "{what}"
        );
        assert_eq!(int(&db, "SELECT COUNT(DISTINCT s) FROM t"), 97, "{what}");
        // Rows read out one by one across many groups
        let rows: Vec<(i64, String)> = db
            .query("SELECT id, s FROM t WHERE f > 60000.0 AND f < 66000.0", ())
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get(0).unwrap(), r.get(1).unwrap())
            })
            .collect();
        let want: Vec<(i64, String)> = (120_001..132_000).map(|i| (i, s_of(i))).collect();
        assert_eq!(rows.len(), want.len(), "{what}");
        assert!(rows == want, "{what}: rows read across small groups differ");
    }

    // A compaction reads the small groups and writes the writer's
    db.execute("DELETE FROM t WHERE id <= 1000", ()).unwrap();
    db.execute("UPDATE t SET k = -1 WHERE id = 140000", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 1", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(int(&db, "SELECT COUNT(*) FROM t"), ROWS - 1000);
    assert_eq!(int(&db, "SELECT k FROM t WHERE id = 140000"), -1);
    assert_eq!(
        int(&db, "SELECT SUM(k + 0) FROM t"),
        (1_001..=ROWS).map(k_of).sum::<i64>() - k_of(140_000) - 1
    );
    db.close().unwrap();
    drop(db);
    let db = Database::open(&dsn).unwrap();
    assert_eq!(int(&db, "SELECT COUNT(*) FROM t WHERE k = 7"), {
        (1_001..=ROWS).filter(|&i| k_of(i) == 7).count() as i64
    });
}
