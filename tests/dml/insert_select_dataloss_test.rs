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

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

#[test]
fn test_insert_select_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/insert_select", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        // Source table with data
        db.execute(
            "CREATE TABLE t1m (id INTEGER PRIMARY KEY, ts INTEGER, price FLOAT)",
            (),
        )
        .unwrap();
        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO t1m VALUES ($1, $2, $3)").unwrap();
        for i in 0..100_000i64 {
            stmt.execute((i, i * 60, 100.0 + i as f64 * 0.01)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Seal source table
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM t1m"), 100_000);

        // Derived table via INSERT INTO SELECT (like higher timeframe sync)
        db.execute(
            "CREATE TABLE t5m (id INTEGER PRIMARY KEY, ts INTEGER, avg_price FLOAT)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO t5m SELECT (ts / 300) as id, (ts / 300) * 300 as ts, AVG(price) as avg_price FROM t1m GROUP BY ts / 300",
            (),
        )
        .unwrap();

        let t5m_count = count(&db, "SELECT COUNT(*) FROM t5m");
        eprintln!("t5m count before close: {}", t5m_count);
        assert!(
            t5m_count > 0,
            "t5m should have rows from INSERT INTO SELECT"
        );

        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        let t1m = count(&db, "SELECT COUNT(*) FROM t1m");
        let t5m = count(&db, "SELECT COUNT(*) FROM t5m");
        eprintln!("After reopen: t1m={}, t5m={}", t1m, t5m);

        assert_eq!(t1m, 100_000, "t1m data loss: {}", 100_000 - t1m);
        assert!(t5m > 0, "t5m data loss: all INSERT INTO SELECT rows lost");

        db.close().unwrap();
    }
}

#[test]
fn test_insert_select_upsert_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/insert_select_upsert", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute("CREATE TABLE src (id INTEGER PRIMARY KEY, val INTEGER)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE dst (id INTEGER PRIMARY KEY, total INTEGER)",
            (),
        )
        .unwrap();

        // Insert source data
        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO src VALUES ($1, $2)").unwrap();
        for i in 0..50_000i64 {
            stmt.execute((i, i % 100)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Seal source
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // INSERT INTO SELECT with aggregation (simulates higher timeframe sync)
        db.execute(
            "INSERT INTO dst SELECT val as id, COUNT(*) as total FROM src GROUP BY val",
            (),
        )
        .unwrap();

        let dst_count = count(&db, "SELECT COUNT(*) FROM dst");
        eprintln!("dst count before close: {}", dst_count);
        assert_eq!(dst_count, 100); // 100 distinct val values (0-99)

        // Now upsert into dst (simulates re-sync)
        db.execute(
            "INSERT INTO dst SELECT val as id, COUNT(*) as total FROM src GROUP BY val ON CONFLICT(id) DO UPDATE SET total = EXCLUDED.total",
            (),
        )
        .unwrap();

        assert_eq!(count(&db, "SELECT COUNT(*) FROM dst"), 100);

        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        let src = count(&db, "SELECT COUNT(*) FROM src");
        let dst = count(&db, "SELECT COUNT(*) FROM dst");
        eprintln!("After reopen: src={}, dst={}", src, dst);

        assert_eq!(src, 50_000, "src data loss");
        assert_eq!(dst, 100, "dst data loss: expected 100, got {}", dst);

        db.close().unwrap();
    }
}
