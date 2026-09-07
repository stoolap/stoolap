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

//! The decoded row-group column cache keeps a warm volume's decoded columns
//! between queries within its byte budget, answers the same rows as a fresh
//! decode, and steps aside when its budget is zero.

use stoolap::Database;

fn ts(secs: i64) -> String {
    chrono::DateTime::from_timestamp(secs, 0)
        .unwrap()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

fn stats(db: &Database) -> (i64, i64, i64, i64, i64) {
    let row = db
        .query("PRAGMA GROUP_CACHE_STATS", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    (
        row.get::<i64>(0).unwrap(),
        row.get::<i64>(1).unwrap(),
        row.get::<i64>(2).unwrap(),
        row.get::<i64>(3).unwrap(),
        row.get::<i64>(4).unwrap(),
    )
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// A sealed table of 20 keys over 5,000 minutes, reopened so its columns are
/// compressed in memory
fn warm_table(dir: &tempfile::TempDir, name: &str) -> Database {
    let dsn = format!("file://{}/{name}", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY AUTO_INCREMENT, t TIMESTAMP NOT NULL, \
         k TEXT NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    let base = 1_709_251_200;
    let mut stmt = String::from("INSERT INTO c (t, k, v) VALUES ");
    for s in 0..5_000i64 {
        for k in 0..20 {
            if s > 0 || k > 0 {
                stmt.push(',');
            }
            stmt.push_str(&format!(
                "('{}', 'k{k}', {})",
                ts(base + s * 60),
                (s + k) % 100
            ));
        }
    }
    db.execute(&stmt, ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    drop(db);
    Database::open(&dsn).unwrap()
}

const LATEST: &str = "SELECT id FROM c WHERE k = 'k7' ORDER BY t DESC LIMIT 50";
const SUMMARY: &str =
    "SELECT k, MAX(v), MIN(v) FROM c WHERE t >= '2024-03-03 00:00:00' GROUP BY k ORDER BY k";

#[test]
fn test_cache_keeps_decoded_groups_and_answers_the_same_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = warm_table(&dir, "keeps");
    db.execute("PRAGMA GROUP_CACHE_MB = 64", ()).unwrap();
    let (_, bytes0, entries0, _, _) = stats(&db);

    let first = ids(&db, LATEST);
    let (_, bytes1, entries1, hits1, misses1) = stats(&db);
    assert!(entries1 > entries0 && bytes1 > bytes0, "nothing cached");
    assert!(misses1 > 0);

    let second = ids(&db, LATEST);
    let (_, bytes2, entries2, hits2, misses2) = stats(&db);
    assert_eq!(first, second);
    assert!(hits2 > hits1, "second run did not hit the cache");
    assert_eq!(misses2, misses1, "second run decoded again");
    assert_eq!((bytes2, entries2), (bytes1, entries1));

    // Another shape over the same groups reads them from the cache too
    let summary = db.query(SUMMARY, ()).unwrap().count();
    assert_eq!(summary, 20);
    let (_, _, _, hits3, _) = stats(&db);
    assert!(hits3 > hits2);
}

#[test]
fn test_cache_budget_bounds_the_bytes_and_zero_disables_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = warm_table(&dir, "budget");

    db.execute("PRAGMA GROUP_CACHE_MB = 1", ()).unwrap();
    let reference = ids(&db, LATEST);
    db.query(SUMMARY, ()).unwrap().count();
    let (budget, bytes, _, _, _) = stats(&db);
    assert_eq!(budget, 1024 * 1024);
    assert!(
        bytes <= budget + 4 * 1024 * 1024,
        "bytes {bytes} far past the budget"
    );
    assert_eq!(ids(&db, LATEST), reference);

    db.execute("PRAGMA GROUP_CACHE_MB = 0", ()).unwrap();
    let (budget, bytes, entries, _, _) = stats(&db);
    assert_eq!((budget, bytes, entries), (0, 0, 0));
    assert_eq!(ids(&db, LATEST), reference);
    let (_, _, entries, _, _) = stats(&db);
    assert_eq!(entries, 0, "a zero budget must not cache");

    db.execute("PRAGMA GROUP_CACHE_MB = 64", ()).unwrap();
    assert_eq!(ids(&db, "PRAGMA GROUP_CACHE_MB"), [64]);
}
