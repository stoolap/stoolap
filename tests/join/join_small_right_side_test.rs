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

//! An INNER join whose right table offers no index probe and is small
//! beside a left table with an index on its key reads the right table as
//! the outer side of an index join, as the join written the other way does

use stoolap::Database;

/// `t` has 5,000 rows, k = id % 1000, f = id % 2, with an index on k; `s`
/// has 5 rows, one key past every `t` row; `m` has 2,000 rows and `x` 300
/// with an index on k, neither small beside the other side; `d` has 20,000
/// rows and `e` 10, every key 1, `d` with an index on it and f = 1 on its
/// first row alone
fn tables(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, f INTEGER)",
        "CREATE INDEX t_k ON t(k)",
        "INSERT INTO t SELECT g.value, g.value % 1000, g.value, g.value % 2 \
         FROM generate_series(1, 5000) g",
        "CREATE TABLE d (id INTEGER PRIMARY KEY, k INTEGER, f INTEGER)",
        "CREATE INDEX d_k ON d(k)",
        "INSERT INTO d SELECT g.value, 1, CASE WHEN g.value = 1 THEN 1 ELSE 0 END \
         FROM generate_series(1, 20000) g",
        "CREATE TABLE e (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO e SELECT g.value, 1 FROM generate_series(1, 10) g",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO s VALUES (1, 0), (2, 7), (3, 14), (4, 999), (5, 2000)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO m SELECT g.value, g.value FROM generate_series(1, 2000) g",
        "CREATE TABLE x (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE INDEX x_k ON x(k)",
        "INSERT INTO x SELECT g.value, g.value % 100 FROM generate_series(1, 300) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect()
}

fn plan(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

/// How `sql`'s plan runs its join, without the cost it estimates
fn join_method(db: &Database, sql: &str) -> String {
    let plan = plan(db, sql);
    let line = plan.lines().find(|line| line.contains("(INNER Join)"));
    let line = line.unwrap_or_else(|| panic!("{sql}:\n{plan}")).trim();
    line.split(" (cost=").next().unwrap_or(line).to_string()
}

/// Each `s` row with the `t` rows its key matches
fn written() -> Vec<(i64, i64)> {
    [(1, 0), (2, 7), (3, 14), (4, 999)]
        .into_iter()
        .flat_map(|(s, k)| {
            (0..5).map(move |i| (s, if k == 0 { 1000 * (i + 1) } else { k + 1000 * i }))
        })
        .collect()
}

fn small_right_side_probes_the_left_index(db: &Database) {
    let swapped = "SELECT s.id, t.id FROM t JOIN s ON t.k = s.k";
    let ordered = format!("{swapped} ORDER BY s.id, t.id");
    assert_eq!(pairs(db, &ordered), written(), "{ordered}");
    let written_other_way = "SELECT s.id, t.id FROM s JOIN t ON t.k = s.k ORDER BY s.id, t.id";
    assert_eq!(
        pairs(db, written_other_way),
        written(),
        "{written_other_way}"
    );
    let filtered =
        "SELECT s.id, t.id FROM t JOIN s ON t.k = s.k WHERE s.id <= 2 ORDER BY s.id, t.id";
    assert_eq!(pairs(db, filtered), written()[..10].to_vec(), "{filtered}");
    let other_way = join_method(db, "SELECT s.id, t.id FROM s JOIN t ON t.k = s.k");
    assert!(!other_way.contains("Hash Join"), "{other_way}");
    for sql in [swapped, &ordered, filtered] {
        assert_eq!(join_method(db, sql), other_way, "{sql}");
    }
}

// The file engine plans no index join on a secondary index over hot rows
#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_small_right_table_probes_the_left_tables_index() {
    let db = Database::open("memory://join_small_right_side_hot").unwrap();
    tables(&db);
    small_right_side_probes_the_left_index(&db);
}

#[test]
fn a_small_right_table_probes_the_left_tables_index_when_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    tables(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    small_right_side_probes_the_left_index(&db);
}

#[test]
fn a_left_side_its_conditions_keep_small_is_not_probed_from_the_right() {
    let db = Database::open("memory://join_small_right_side_left_filtered").unwrap();
    tables(&db);
    let sql = "SELECT e.id, d.id FROM d JOIN e ON d.k = e.k WHERE d.id = 1";
    for analyzed in [false, true] {
        if analyzed {
            db.execute("ANALYZE d", ()).unwrap();
            db.execute("ANALYZE e", ()).unwrap();
        }
        assert_eq!(pairs(&db, sql).len(), 10, "{sql}, analyzed {analyzed}");
        let method = join_method(&db, sql);
        assert!(
            method.contains("Hash Join"),
            "{sql}, analyzed {analyzed}: {method}"
        );
    }
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_swapped_join_whose_probe_finds_more_than_the_cap_runs_by_hash() {
    use stoolap::test_failpoints::index_join_outer_rows;
    let db = Database::open("memory://join_small_right_side_probe_cap").unwrap();
    tables(&db);
    let sql = "SELECT e.id, d.id FROM d JOIN e ON d.k = e.k WHERE d.f = 1 ORDER BY e.id";
    let method = join_method(&db, sql);
    assert!(!method.contains("Hash Join"), "{sql} swaps: {method}");
    let before = index_join_outer_rows();
    let found = pairs(&db, sql);
    assert_eq!(found, (1..=10).map(|e| (e, 1)).collect::<Vec<_>>(), "{sql}");
    let read = index_join_outer_rows() - before;
    assert!(read <= 1, "{sql}: the swapped join probed {read} rows of e");
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_probe_past_the_cap_across_key_types_runs_by_hash() {
    use stoolap::test_failpoints::index_join_outer_rows;
    let db = Database::open("memory://join_small_right_side_cap_across").unwrap();
    for sql in [
        "CREATE TABLE dt (id INTEGER PRIMARY KEY, k TEXT, f INTEGER)",
        "CREATE INDEX dt_k ON dt(k)",
        "INSERT INTO dt SELECT g.value, '1', CASE WHEN g.value = 1 THEN 1 ELSE 0 END \
         FROM generate_series(1, 20000) g",
        "CREATE TABLE ei (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ei SELECT g.value, 1 FROM generate_series(1, 10) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT ei.id, dt.id FROM dt JOIN ei ON dt.k = ei.k WHERE dt.f = 1 ORDER BY ei.id";
    let method = join_method(&db, sql);
    assert!(!method.contains("Hash Join"), "{sql} swaps: {method}");
    let before = index_join_outer_rows();
    let found = pairs(&db, sql);
    assert_eq!(found, (1..=10).map(|e| (e, 1)).collect::<Vec<_>>(), "{sql}");
    let read = index_join_outer_rows() - before;
    assert!(
        read <= 1,
        "{sql}: the swapped join probed {read} rows of ei"
    );
}

/// The next of 1, 2, 3, ... on each call
#[derive(Default)]
struct JoinSwapNext;

static JOIN_SWAP_CALLS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

impl stoolap::functions::ScalarFunction for JoinSwapNext {
    fn name(&self) -> &str {
        "JOIN_SWAP_NEXT"
    }

    fn info(&self) -> stoolap::functions::FunctionInfo {
        stoolap::functions::FunctionInfo {
            name: "JOIN_SWAP_NEXT".into(),
            ..stoolap::functions::ScalarFunction::info(&stoolap::functions::scalar::SleepFunction)
        }
    }

    fn evaluate(
        &self,
        _args: &[stoolap::core::Value],
    ) -> stoolap::core::Result<stoolap::core::Value> {
        let next = JOIN_SWAP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        Ok(stoolap::core::Value::Integer(next))
    }

    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(Self)
    }
}

#[test]
fn a_join_past_the_cap_reads_its_outer_side_once() {
    stoolap::functions::global_registry()
        .register_scalar::<JoinSwapNext>()
        .unwrap();
    let db = Database::open("memory://join_small_right_side_outer_once").unwrap();
    for sql in [
        "CREATE TABLE g (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE INDEX g_k ON g(k)",
        "INSERT INTO g SELECT v.value, 1 FROM generate_series(1, 2000) v",
    ] {
        db.execute(sql, ()).unwrap();
    }
    JOIN_SWAP_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    let sql = "SELECT COUNT(*) FROM g JOIN (VALUES (JOIN_SWAP_NEXT())) AS v(k) ON g.k = v.k";
    let count: i64 = db.query_one(sql, ()).unwrap();
    assert_eq!(count, 2000, "{sql}: the VALUES row is read once and keys 1");
}

// The file engine plans no index join on a secondary index over hot rows
#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_left_side_its_conditions_keep_large_is_probed_from_the_small_right() {
    let db = Database::open("memory://join_small_right_side_wide_filter").unwrap();
    tables(&db);
    let sql = "SELECT s.id, t.id FROM t JOIN s ON t.k = s.k WHERE t.f = 0 ORDER BY s.id, t.id";
    let other_way = "SELECT s.id, t.id FROM s JOIN t ON t.k = s.k WHERE t.f = 0";
    let even: Vec<_> = written().into_iter().filter(|(_, t)| t % 2 == 0).collect();
    for analyzed in [false, true] {
        if analyzed {
            db.execute("ANALYZE t", ()).unwrap();
            db.execute("ANALYZE s", ()).unwrap();
        }
        assert_eq!(pairs(&db, sql), even, "{sql}, analyzed {analyzed}");
        let method = join_method(&db, sql);
        assert!(
            !method.contains("Hash Join"),
            "{sql}, analyzed {analyzed}: {method}"
        );
        assert_eq!(
            method,
            join_method(&db, other_way),
            "{sql}, analyzed {analyzed}"
        );
    }
}

#[test]
fn a_right_table_not_small_beside_the_left_keeps_its_join() {
    let db = Database::open("memory://join_small_right_side_kept").unwrap();
    tables(&db);
    for sql in [
        // More rows than a small side holds
        "SELECT m.id, t.id FROM t JOIN m ON t.k = m.k",
        // A left table too small for a probe per right row to pay
        "SELECT s.id, x.id FROM x JOIN s ON x.k = s.k",
        // A LEFT join keeps the side it preserves
        "SELECT s.id, t.id FROM t LEFT JOIN s ON t.k = s.k",
    ] {
        let plan = plan(&db, sql);
        assert!(!plan.contains("Index Nested Loop"), "{sql}:\n{plan}");
    }
    let sql = "SELECT COUNT(*) FROM t JOIN m ON t.k = m.k";
    let count: i64 = db.query_one(sql, ()).unwrap();
    assert_eq!(count, 4995, "{sql}");
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_small_right_table_is_the_outer_side_its_index_join_reads() {
    use stoolap::test_failpoints::index_join_outer_rows;
    let db = Database::open("memory://join_small_right_side_outer").unwrap();
    tables(&db);
    let before = index_join_outer_rows();
    let sql = "SELECT s.id, t.id FROM t JOIN s ON t.k = s.k";
    assert_eq!(pairs(&db, sql).len(), 20, "{sql}");
    assert_eq!(
        index_join_outer_rows() - before,
        5,
        "{sql} reads s as its outer side"
    );
}
