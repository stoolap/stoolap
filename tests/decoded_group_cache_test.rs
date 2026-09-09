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

use std::sync::{Mutex, MutexGuard};
use stoolap::storage::volume::group_cache::{DECODED_GROUPS, DEFAULT_BUDGET_BYTES};
use stoolap::Database;

// The cache is one per process and `cargo test` runs these tests in one
// process, so each test takes this lock, starts from an empty cache and
// hands the default budget back when it is done.
static SERIAL: Mutex<()> = Mutex::new(());

struct Serial(#[allow(dead_code)] MutexGuard<'static, ()>);

fn serial() -> Serial {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    DECODED_GROUPS.set_budget_bytes(0);
    Serial(guard)
}

impl Drop for Serial {
    fn drop(&mut self) {
        DECODED_GROUPS.set_budget_bytes(0);
        DECODED_GROUPS.set_budget_bytes(DEFAULT_BUDGET_BYTES);
    }
}

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
    let _serial = serial();
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
    let _serial = serial();
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

    db.execute("PRAGMA GROUP_CACHE_MB = 32", ()).unwrap();
    assert_eq!(ids(&db, "PRAGMA GROUP_CACHE_MB"), [32]);
}

// The cache's own invariants, exercised through its API: one decode shared
// by concurrent readers, no phantom bytes after an in-flight eviction, a
// budget change during a decode, a column larger than the budget, and a
// dictionary shared by the groups of one column counted once.

mod invariants {
    use super::serial;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Barrier};
    use std::time::Duration;
    use stoolap::core::DataType;
    use stoolap::storage::volume::column::ColumnData;
    use stoolap::storage::volume::group_cache::DECODED_GROUPS;

    fn ints(rows: usize) -> ColumnData {
        ColumnData::Int64 {
            values: vec![1; rows],
            nulls: vec![false; rows],
        }
    }

    #[test]
    fn concurrent_readers_share_one_decode() {
        let _serial = serial();
        DECODED_GROUPS.set_budget_bytes(1024);
        let starts = Arc::new(Barrier::new(16));
        let decodes = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..16)
            .map(|_| {
                let starts = Arc::clone(&starts);
                let decodes = Arc::clone(&decodes);
                std::thread::spawn(move || {
                    starts.wait();
                    DECODED_GROUPS
                        .get_or_decode((600, 0, 0), || {
                            decodes.fetch_add(1, Ordering::SeqCst);
                            Ok(ints(1))
                        })
                        .unwrap()
                })
            })
            .collect();
        let columns: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(decodes.load(Ordering::SeqCst), 1);
        assert!(columns.iter().all(|c| Arc::ptr_eq(&columns[0], c)));
        DECODED_GROUPS.remove_store(600);
        assert_eq!(DECODED_GROUPS.stats().bytes, 0);
    }

    #[test]
    fn an_entry_evicted_while_decoding_leaves_no_phantom_bytes() {
        let _serial = serial();
        DECODED_GROUPS.set_budget_bytes(18);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            DECODED_GROUPS
                .get_or_decode((100, 0, 0), || {
                    entered_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                    Ok(ints(1))
                })
                .unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        DECODED_GROUPS
            .get_or_decode((200, 0, 0), || Ok(ints(2)))
            .unwrap();
        DECODED_GROUPS
            .get_or_decode((300, 0, 0), || Ok(ints(2)))
            .unwrap();
        resume_tx.send(()).unwrap();
        worker.join().unwrap();
        for store in [100, 200, 300] {
            DECODED_GROUPS.remove_store(store);
        }
        let stats = DECODED_GROUPS.stats();
        assert_eq!((stats.bytes, stats.entries), (0, 0));
    }

    #[test]
    fn a_budget_of_zero_set_during_a_decode_leaves_the_cache_empty() {
        let _serial = serial();
        DECODED_GROUPS.set_budget_bytes(18);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            DECODED_GROUPS
                .get_or_decode((400, 0, 0), || {
                    entered_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                    Ok(ints(1))
                })
                .unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        DECODED_GROUPS.set_budget_bytes(0);
        resume_tx.send(()).unwrap();
        worker.join().unwrap();
        let stats = DECODED_GROUPS.stats();
        assert_eq!((stats.budget_bytes, stats.bytes, stats.entries), (0, 0, 0));
    }

    #[test]
    fn a_column_larger_than_the_budget_is_not_kept() {
        let _serial = serial();
        DECODED_GROUPS.set_budget_bytes(1024 * 1024);
        DECODED_GROUPS
            .get_or_decode((500, 0, 0), || {
                Ok(ColumnData::Bytes {
                    data: vec![b'x'; 8 * 1024 * 1024],
                    offsets: vec![(0, 8 * 1024 * 1024)],
                    ext_type: DataType::Json,
                    nulls: vec![false],
                })
            })
            .unwrap();
        let stats = DECODED_GROUPS.stats();
        assert!(
            stats.bytes <= stats.budget_bytes,
            "kept {} bytes",
            stats.bytes
        );
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn a_dictionary_shared_by_the_groups_is_counted_once() {
        use stoolap::common::SmartString;
        use stoolap::storage::volume::column::ROW_GROUP_SIZE;
        use stoolap::storage::volume::writer::{CompressedBlockStore, LazyColumns};
        let _serial = serial();
        let rows = 2 * ROW_GROUP_SIZE;
        let dictionary: Arc<[SmartString]> = (0..10_000)
            .map(|i| SmartString::from(format!("{i:010}{}", "x".repeat(90))))
            .collect::<Vec<_>>()
            .into();
        let column = ColumnData::Dictionary {
            ids: (0..rows).map(|i| (i % 10_000) as u32).collect(),
            nulls: vec![false; rows],
            dictionary,
        };
        assert!(column.memory_size() < 2 * 1024 * 1024);
        let columns = LazyColumns::eager(vec![column], vec![DataType::Text]);
        let store =
            CompressedBlockStore::compress_columns(&columns, &[DataType::Text], rows).unwrap();
        DECODED_GROUPS.set_budget_bytes(2 * 1024 * 1024);
        let first = store.group_column(0, 0).unwrap();
        let _second = store.group_column(0, 1).unwrap();
        let again = store.group_column(0, 0).unwrap();
        assert!(Arc::ptr_eq(&first, &again), "the first group was evicted");
    }

    #[test]
    fn a_zero_budget_racing_with_a_charge_leaves_nothing_behind() {
        const READERS: usize = 8;
        const ROUNDS: usize = 20_000;
        let _serial = serial();
        let barrier = Barrier::new(READERS + 1);
        let mut failure = None;
        std::thread::scope(|scope| {
            for worker in 0..READERS {
                let barrier = &barrier;
                scope.spawn(move || {
                    for _ in 0..ROUNDS {
                        barrier.wait();
                        DECODED_GROUPS
                            .get_or_decode((10_000 + worker, 0, 0), || Ok(ints(1)))
                            .unwrap();
                        barrier.wait();
                    }
                });
            }
            for round in 0..ROUNDS {
                DECODED_GROUPS.set_budget_bytes(1024);
                barrier.wait();
                DECODED_GROUPS.set_budget_bytes(0);
                barrier.wait();
                let stats = DECODED_GROUPS.stats();
                if stats.bytes != 0 || stats.entries != 0 {
                    failure.get_or_insert((round, stats.bytes, stats.entries));
                }
            }
        });
        assert_eq!(
            failure, None,
            "a zero budget left data behind (round, bytes, entries)"
        );
    }
}
