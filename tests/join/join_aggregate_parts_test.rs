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

//! An aggregate over a join reads the joined rows in parts as the join
//! makes them, and answers as the rows' definition says, whatever the part
//! a group or an unmatched row falls in

use std::collections::{BTreeMap, BTreeSet};

use stoolap::Database;

/// A row of l or r: id, k, then g or w
type Fixture = (i64, i64, i64);

/// l: id 1..=40, k = id % 10, g = id % 3
fn l_rows() -> impl Iterator<Item = Fixture> + Clone {
    (1..=40).map(|id| (id, id % 10, id % 3))
}

/// r: id 1..=60, k = id % 10 + 3, w = id % 4; keys 0 to 2 of l and 10 to 12
/// of r meet nothing
fn r_rows() -> impl Iterator<Item = Fixture> + Clone {
    (1..=60).map(|id| (id, id % 10 + 3, id % 4))
}

/// The pairs of l and r with equal keys, as (l, r) rows
fn pairs() -> Vec<(Fixture, Fixture)> {
    let mut out = Vec::new();
    for l in l_rows() {
        for r in r_rows().filter(|r| r.1 == l.1) {
            out.push((l, r));
        }
    }
    out
}

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "INSERT INTO l SELECT value, value % 10, value % 3 FROM generate_series(1, 40)",
        "INSERT INTO r SELECT value, value % 10 + 3, value % 4 FROM generate_series(1, 60)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The rows of `sql`, each its values joined by `|`, in the order returned
fn rows(db: &Database, sql: &str) -> Vec<String> {
    let result = db.query(sql, ()).unwrap();
    let mut out = Vec::new();
    for row in result {
        let row = row.unwrap();
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    out
}

fn int(v: i64) -> String {
    format!("Integer({v})")
}

/// Parts of a few rows, so groups and unmatched rows fall across them
const PART_ROWS: usize = 7;

/// `sql`'s rows, read with parts of `PART_ROWS` rows where a test sets the
/// size, checked to have held no more joined rows at once
fn parted(db: &Database, sql: &str) -> Vec<String> {
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::set_join_part_rows(PART_ROWS);
        stoolap::test_failpoints::reset_join_part_largest();
    }
    let out = rows(db, sql);
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::set_join_part_rows(0);
        let largest = stoolap::test_failpoints::join_part_largest();
        assert!(
            (1..=PART_ROWS).contains(&largest),
            "{sql}: the aggregation held {largest} joined rows at once"
        );
    }
    out
}

#[test]
fn a_count_over_an_inner_join_counts_every_pair() {
    let db = open("join_aggregate_parts_inner");
    let sql = "SELECT COUNT(*), SUM(r.w), MIN(l.id), MAX(r.id) FROM l JOIN r ON l.k = r.k";
    let pairs = pairs();
    let sum: i64 = pairs.iter().map(|(_, r)| r.2).sum();
    let min = pairs.iter().map(|(l, _)| l.0).min().unwrap();
    let max = pairs.iter().map(|(_, r)| r.0).max().unwrap();
    let expected = [int(pairs.len() as i64), int(sum), int(min), int(max)].join("|");
    assert_eq!(parted(&db, sql), vec![expected]);
}

#[test]
fn groups_over_a_join_gather_their_rows_from_every_part() {
    let db = open("join_aggregate_parts_grouped");
    let sql = "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k \
               GROUP BY l.g ORDER BY l.g";
    let mut groups: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (l, r) in pairs() {
        let group = groups.entry(l.2).or_default();
        group.0 += 1;
        group.1 += r.2;
    }
    let expected: Vec<String> = groups
        .iter()
        .map(|(g, (count, sum))| [int(*g), int(*count), int(*sum)].join("|"))
        .collect();
    assert_eq!(parted(&db, sql), expected);
}

#[test]
fn a_distinct_count_per_group_reads_each_row_once() {
    let db = open("join_aggregate_parts_distinct");
    let sql = "SELECT l.g, COUNT(DISTINCT r.w), COUNT(DISTINCT r.id) FROM l JOIN r \
               ON l.k = r.k GROUP BY l.g ORDER BY l.g";
    let mut groups: BTreeMap<i64, (BTreeSet<i64>, BTreeSet<i64>)> = BTreeMap::new();
    for (l, r) in pairs() {
        let group = groups.entry(l.2).or_default();
        group.0.insert(r.2);
        group.1.insert(r.0);
    }
    let expected: Vec<String> = groups
        .iter()
        .map(|(g, (w, id))| [int(*g), int(w.len() as i64), int(id.len() as i64)].join("|"))
        .collect();
    assert_eq!(parted(&db, sql), expected);
}

#[test]
fn a_rollup_over_a_join_totals_every_part() {
    let db = open("join_aggregate_parts_rollup");
    let sql = "SELECT COALESCE(l.g, -1), COUNT(*) FROM l JOIN r ON l.k = r.k \
               GROUP BY ROLLUP(l.g) ORDER BY 1";
    let mut groups: BTreeMap<i64, i64> = BTreeMap::new();
    for (l, _) in pairs() {
        *groups.entry(l.2).or_default() += 1;
    }
    let total: i64 = groups.values().sum();
    let expected: Vec<String> = std::iter::once((-1, total))
        .chain(groups)
        .map(|(g, count)| [int(g), int(count)].join("|"))
        .collect();
    assert_eq!(parted(&db, sql), expected);
}

#[test]
fn outer_joins_count_their_unmatched_rows_in_the_last_parts() {
    let db = open("join_aggregate_parts_outer");
    let pairs = pairs().len() as i64;
    let l_alone = l_rows().filter(|l| !r_rows().any(|r| r.1 == l.1)).count() as i64;
    let r_alone = r_rows().filter(|r| !l_rows().any(|l| l.1 == r.1)).count() as i64;
    assert!(
        l_alone > 0 && r_alone > 0,
        "the fixture leaves rows unmatched"
    );
    let cases = [
        ("LEFT", pairs + l_alone, pairs + l_alone, pairs),
        ("RIGHT", pairs + r_alone, pairs, pairs + r_alone),
        (
            "FULL",
            pairs + l_alone + r_alone,
            pairs + l_alone,
            pairs + r_alone,
        ),
    ];
    for (kind, all, with_l, with_r) in cases {
        let sql =
            format!("SELECT COUNT(*), COUNT(l.id), COUNT(r.id) FROM l {kind} JOIN r ON l.k = r.k");
        let expected = [int(all), int(with_l), int(with_r)].join("|");
        assert_eq!(parted(&db, &sql), vec![expected], "{kind}");
    }
}

#[test]
fn a_where_on_both_sides_and_a_residual_pass_rows_in_every_part() {
    let db = open("join_aggregate_parts_where");
    let below = pairs().iter().filter(|(l, r)| l.0 < r.0).count() as i64;
    for sql in [
        "SELECT COUNT(*) FROM l JOIN r ON l.k = r.k WHERE l.id < r.id",
        "SELECT COUNT(*) FROM l JOIN r ON l.k = r.k AND l.id < r.id",
    ] {
        assert_eq!(parted(&db, sql), vec![int(below)], "{sql}");
    }
}

#[test]
fn a_join_without_equal_keys_is_counted_in_parts() {
    let db = open("join_aggregate_parts_nested");
    let mut below = 0;
    for l in l_rows() {
        below += r_rows().filter(|r| l.0 + 30 < r.0).count() as i64;
    }
    let sql = "SELECT COUNT(*) FROM l JOIN r ON l.id + 30 < r.id";
    assert_eq!(parted(&db, sql), vec![int(below)]);
    let cross = (l_rows().count() * r_rows().count()) as i64;
    assert_eq!(
        parted(&db, "SELECT COUNT(*) FROM l CROSS JOIN r"),
        vec![int(cross)]
    );
}

#[test]
fn a_join_whose_parts_are_whole_counts_each_once() {
    let db = open("join_aggregate_parts_exact");
    let pairs: Vec<_> = pairs()
        .into_iter()
        .filter(|(l, _)| (3..=9).contains(&l.0))
        .collect();
    assert_eq!(pairs.len() % PART_ROWS, 0, "the pairs fill whole parts");
    let sql = "SELECT COUNT(*) FROM l JOIN r ON l.k = r.k WHERE l.id BETWEEN 3 AND 9";
    assert_eq!(parted(&db, sql), vec![int(pairs.len() as i64)]);
}

/// Rows on each side of the joins built across threads
const WIDE: i64 = 12_000;

/// bl: id 1..=WIDE, u = id, g = id % 5; br: id 1..=WIDE, u = id + WIDE / 2,
/// v = id % 7. Row n of br meets row n + WIDE / 2 of bl, so half of each
/// side meets nothing
fn open_wide(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE bl (id INTEGER PRIMARY KEY, u INTEGER, g INTEGER)".to_string(),
        "CREATE TABLE br (id INTEGER PRIMARY KEY, u INTEGER, v INTEGER)".to_string(),
        format!("INSERT INTO bl SELECT value, value, value % 5 FROM generate_series(1, {WIDE})"),
        format!(
            "INSERT INTO br SELECT value, value + {}, value % 7 FROM generate_series(1, {WIDE})",
            WIDE / 2
        ),
    ] {
        db.execute(&sql, ()).unwrap();
    }
    db
}

/// The (bl, br) id pairs with equal u
fn wide_pairs() -> impl Iterator<Item = (i64, i64)> {
    (1..=WIDE / 2).map(|r| (r + WIDE / 2, r))
}

/// `sql`'s rows as `parted` reads them, checked to come from the hash join
/// built across threads
fn parted_across_threads(db: &Database, sql: &str) -> Vec<String> {
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::reset_join_paths();
    let out = parted(db, sql);
    #[cfg(feature = "test-failpoints")]
    assert_eq!(
        stoolap::test_failpoints::join_paths_taken(),
        vec!["parallel_hash"],
        "{sql}"
    );
    out
}

#[test]
fn a_join_built_across_threads_is_grouped_in_parts() {
    let db = open_wide("join_aggregate_parts_wide_grouped");
    let mut groups: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (l, r) in wide_pairs() {
        let group = groups.entry(l % 5).or_default();
        group.0 += 1;
        group.1 += r % 7;
    }
    let expected: Vec<String> = groups
        .iter()
        .map(|(g, (count, sum))| [int(*g), int(*count), int(*sum)].join("|"))
        .collect();
    let sql = "SELECT bl.g, COUNT(*), SUM(br.v) FROM bl JOIN br ON bl.u = br.u \
               GROUP BY bl.g ORDER BY bl.g";
    assert_eq!(parted_across_threads(&db, sql), expected);
}

#[test]
fn outer_joins_built_across_threads_count_every_unmatched_row() {
    let db = open_wide("join_aggregate_parts_wide_outer");
    let (pairs, alone) = (WIDE / 2, WIDE / 2);
    let cases = [
        ("LEFT", pairs + alone, pairs + alone, pairs),
        ("RIGHT", pairs + alone, pairs, pairs + alone),
        ("FULL", pairs + 2 * alone, pairs + alone, pairs + alone),
    ];
    for (kind, all, with_l, with_r) in cases {
        let sql = format!(
            "SELECT COUNT(*), COUNT(bl.id), COUNT(br.id) FROM bl {kind} JOIN br ON bl.u = br.u"
        );
        let expected = [int(all), int(with_l), int(with_r)].join("|");
        assert_eq!(parted_across_threads(&db, &sql), vec![expected], "{kind}");
    }
}

#[test]
fn a_residual_and_a_distinct_count_read_a_join_built_across_threads() {
    let db = open_wide("join_aggregate_parts_wide_residual");
    let kept = wide_pairs().filter(|(_, r)| r % 7 < 3).count() as i64;
    let sql = "SELECT COUNT(*) FROM bl JOIN br ON bl.u = br.u AND br.v < 3";
    assert_eq!(parted_across_threads(&db, sql), vec![int(kept)]);
    let mut groups: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for (l, r) in wide_pairs() {
        groups.entry(l % 5).or_default().insert(r % 7);
    }
    let expected: Vec<String> = groups
        .iter()
        .map(|(g, v)| [int(*g), int(v.len() as i64)].join("|"))
        .collect();
    let sql = "SELECT bl.g, COUNT(DISTINCT br.v) FROM bl JOIN br ON bl.u = br.u \
               GROUP BY bl.g ORDER BY bl.g";
    assert_eq!(parted_across_threads(&db, sql), expected);
}

#[test]
fn a_slice_of_probe_rows_meeting_many_rows_is_probed_on_every_thread() {
    let db = Database::open("memory://join_aggregate_parts_fan_out").unwrap();
    for sql in [
        "CREATE TABLE fl (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE fs (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO fl SELECT value, value % 200 FROM generate_series(1, 10000)",
        "INSERT INTO fs SELECT value, value % 200 FROM generate_series(1, 10000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut per_key: BTreeMap<i64, i64> = BTreeMap::new();
    for l in 1..=10_000i64 {
        *per_key.entry(l % 200).or_default() += 1;
    }
    let (mut count, mut sum) = (0, 0);
    for s in 1..=10_000i64 {
        let met = per_key[&(s % 200)];
        count += met;
        sum += met * s;
    }
    let sql = "SELECT COUNT(*), SUM(fs.id) FROM fs JOIN fl ON fs.k = fl.k";
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::set_join_part_rows(2_000);
        stoolap::test_failpoints::reset_join_probes_narrow();
        stoolap::test_failpoints::reset_join_paths();
    }
    let out = rows(&db, sql);
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::set_join_part_rows(0);
        assert_eq!(
            stoolap::test_failpoints::join_paths_taken(),
            vec!["parallel_hash"],
            "{sql}"
        );
        let narrow = stoolap::test_failpoints::join_probes_narrow();
        assert_eq!(
            narrow, 0,
            "{sql}: {narrow} slices were probed in fewer chunks than the threads could take"
        );
    }
    assert_eq!(out, vec![[int(count), int(sum)].join("|")]);
}

#[cfg(feature = "test-failpoints")]
mod cancel {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    use stoolap::core::{Result, Value};
    use stoolap::functions::registry::global_registry;
    use stoolap::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
    };
    use stoolap::test_failpoints::{keep_deadlines, pass_deadlines, FailpointGuard};
    use stoolap::{Error, ExecutionContext, Executor};

    static ARMED: AtomicBool = AtomicBool::new(false);
    static CALLS: AtomicUsize = AtomicUsize::new(0);

    /// Returns its argument, counts its calls and passes every deadline
    /// while `ARMED`
    #[derive(Default)]
    struct JoinCancelNow;

    impl ScalarFunction for JoinCancelNow {
        fn name(&self) -> &str {
            "JOIN_CANCEL_NOW"
        }
        fn info(&self) -> FunctionInfo {
            FunctionInfo::new(
                "JOIN_CANCEL_NOW",
                FunctionType::Scalar,
                "test cancel",
                FunctionSignature::new(
                    FunctionDataType::Integer,
                    vec![FunctionDataType::Any],
                    1,
                    1,
                ),
            )
        }
        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            CALLS.fetch_add(1, Ordering::Relaxed);
            if ARMED.load(Ordering::Relaxed) {
                pass_deadlines();
            }
            Ok(args[0].clone())
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(JoinCancelNow)
        }
    }

    #[test]
    fn a_cancel_stops_a_join_read_in_parts_within_a_hundred_rows() {
        let _guard = FailpointGuard::new();
        global_registry()
            .register_scalar::<JoinCancelNow>()
            .unwrap();
        for (db, sql) in [
            (
                super::open("join_aggregate_parts_cancel_one"),
                "SELECT COUNT(*) FROM l JOIN r ON l.k = r.k \
                 AND JOIN_CANCEL_NOW(l.id + r.id) = l.id + r.id",
            ),
            (
                super::open_wide("join_aggregate_parts_cancel_wide"),
                "SELECT COUNT(*) FROM bl JOIN br ON bl.u = br.u \
                 AND JOIN_CANCEL_NOW(bl.id + br.id) = bl.id + br.id",
            ),
            (
                super::open("join_aggregate_parts_cancel_argument"),
                "SELECT SUM(JOIN_CANCEL_NOW(l.id + r.id)) FROM l JOIN r ON l.k = r.k",
            ),
            (
                super::open_wide("join_aggregate_parts_cancel_wide_argument"),
                "SELECT SUM(JOIN_CANCEL_NOW(bl.id + br.id)) FROM bl JOIN br ON bl.u = br.u",
            ),
        ] {
            let executor = Executor::new(Arc::clone(db.engine()));
            let mut ctx = ExecutionContext::new();
            ctx.set_timeout_ms(3_600_000);
            ARMED.store(true, Ordering::Relaxed);
            let before = CALLS.load(Ordering::Relaxed);
            let read = executor
                .execute_with_context(sql, &ctx)
                .and_then(|mut rows| {
                    while rows.next() {}
                    rows.last_error().map_or(Ok(()), Err)
                });
            ARMED.store(false, Ordering::Relaxed);
            keep_deadlines();
            let calls = CALLS.load(Ordering::Relaxed) - before;
            assert!(
                matches!(read, Err(Error::QueryCancelled)),
                "{sql}: {read:?}"
            );
            assert!(calls <= 100, "{sql}: {calls} rows joined after the cancel");
        }
    }
}

/// Run in a process of its own: a hash join that does not run across
/// threads leaves the global thread pool for the caller to build
#[cfg(feature = "parallel")]
#[test]
fn a_hash_join_on_one_thread_leaves_the_thread_pool_unbuilt() {
    use stoolap::core::{Row, Value};
    use stoolap::executor::{parallel_hash_join, Deadline, ParallelConfig, ParallelJoinType};

    const NAME: &str =
        "join_aggregate_parts_test::a_hash_join_on_one_thread_leaves_the_thread_pool_unbuilt";
    if std::env::var_os("JOIN_POOL_CHILD").is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--test-threads", "1"])
            .env("JOIN_POOL_CHILD", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    let rows: Vec<Row> = (0..10)
        .map(|i| Row::from_values(vec![Value::Integer(i)]))
        .collect();
    for config in [ParallelConfig::disabled(), ParallelConfig::default()] {
        for (probe, build) in [(&rows[..], &rows[..]), (&[][..], &[][..])] {
            for join_type in [ParallelJoinType::Inner, ParallelJoinType::Full] {
                let none = Deadline::NONE;
                parallel_hash_join(
                    probe,
                    build,
                    &[0],
                    &[0],
                    join_type,
                    1,
                    1,
                    false,
                    &config,
                    none,
                )
                .unwrap();
            }
        }
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global()
        .expect("a hash join on one thread built the global thread pool");
}

/// A row of p: id, g
type Parent = (i64, i64);

/// A row of c: id, pid, v
type Child = (i64, i64, i64);

/// Which (p, c) pairs a statement keeps
type Keep = fn(&Parent, &Child) -> bool;

/// p: id 1..=50, g = id % 3
fn p_rows() -> impl Iterator<Item = Parent> + Clone {
    (1..=50).map(|id| (id, id % 3))
}

/// c: id 1..=300, pid = id % 47, v = id % 7, an index on pid; pid 0 meets
/// no p, and p 47 to 50 meet no c
fn c_rows() -> impl Iterator<Item = Child> + Clone {
    (1..=300).map(|id| (id, id % 47, id % 7))
}

/// The (p, c) pairs with p.id = c.pid
fn keyed_pairs() -> Vec<(Parent, Child)> {
    let mut out = Vec::new();
    for p in p_rows() {
        for c in c_rows().filter(|c| c.1 == p.0) {
            out.push((p, c));
        }
    }
    out
}

fn open_keyed(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, g INTEGER)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER, v INTEGER)",
        "CREATE INDEX c_pid ON c (pid)",
        "INSERT INTO p SELECT value, value % 3 FROM generate_series(1, 50)",
        "INSERT INTO c SELECT value, value % 47, value % 7 FROM generate_series(1, 300)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The path a join probing c's index takes; a file database hashes it
const C_INDEX: &str = if cfg!(feature = "test-filedb") {
    "hash"
} else {
    "index_nested_loop"
};

/// `sql`'s rows as `parted` reads them, checked to come from the index
/// join `path`
fn parted_by(db: &Database, sql: &str, path: &str) -> Vec<String> {
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::reset_join_paths();
    let out = parted(db, sql);
    #[cfg(feature = "test-failpoints")]
    assert_eq!(
        stoolap::test_failpoints::join_paths_taken(),
        vec![path],
        "{sql}"
    );
    #[cfg(not(feature = "test-failpoints"))]
    let _ = path;
    out
}

/// Per p.g of the pairs `keep` keeps: the count and the sum of c.v
fn keyed_groups(keep: impl Fn(&Parent, &Child) -> bool) -> Vec<String> {
    let mut groups: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (p, c) in keyed_pairs().iter().filter(|(p, c)| keep(p, c)) {
        let group = groups.entry(p.1).or_default();
        group.0 += 1;
        group.1 += c.2;
    }
    groups
        .iter()
        .map(|(g, (count, sum))| [int(*g), int(*count), int(*sum)].join("|"))
        .collect()
}

#[test]
fn an_index_join_is_grouped_in_parts() {
    let db = open_keyed("join_aggregate_parts_index_grouped");
    let all = keyed_groups(|_, _| true);
    for (sql, path) in [
        (
            "SELECT p.g, COUNT(*), SUM(c.v) FROM p JOIN c ON c.pid = p.id \
             GROUP BY p.g ORDER BY p.g",
            C_INDEX,
        ),
        (
            "SELECT p.g, COUNT(*), SUM(c.v) FROM c JOIN p ON p.id = c.pid \
             GROUP BY p.g ORDER BY p.g",
            "batch_index_nested_loop",
        ),
    ] {
        assert_eq!(parted_by(&db, sql, path), all, "{sql}");
    }
}

#[test]
fn left_index_joins_count_their_unmatched_rows_in_parts() {
    let db = open_keyed("join_aggregate_parts_index_left");
    let pairs = keyed_pairs();
    let lone_p = p_rows().filter(|p| !c_rows().any(|c| c.1 == p.0)).count() as i64;
    let lone_c = c_rows().filter(|c| !p_rows().any(|p| p.0 == c.1)).count() as i64;
    let sum: i64 = pairs.iter().map(|(_, c)| c.2).sum();
    let n = pairs.len() as i64;
    let sql = "SELECT COUNT(*), COUNT(c.id), SUM(c.v) FROM p LEFT JOIN c ON c.pid = p.id";
    let want = vec![[int(n + lone_p), int(n), int(sum)].join("|")];
    assert_eq!(parted_by(&db, sql, C_INDEX), want);
    let sql = "SELECT COUNT(*), COUNT(p.id) FROM c LEFT JOIN p ON p.id = c.pid";
    let want = vec![[int(n + lone_c), int(n)].join("|")];
    assert_eq!(parted_by(&db, sql, "batch_index_nested_loop"), want);
}

#[test]
fn filters_on_an_index_join_pass_rows_in_parts() {
    let db = open_keyed("join_aggregate_parts_index_filters");
    let cases: [(&str, &str, Keep); 4] = [
        (
            "SELECT p.g, COUNT(*), SUM(c.v) FROM p JOIN c ON c.pid = p.id AND c.v < 3 \
             GROUP BY p.g ORDER BY p.g",
            C_INDEX,
            |_, c| c.2 < 3,
        ),
        (
            "SELECT p.g, COUNT(*), SUM(c.v) FROM p JOIN c ON c.pid = p.id WHERE c.v = 2 \
             GROUP BY p.g ORDER BY p.g",
            "batch_index_nested_loop",
            |_, c| c.2 == 2,
        ),
        (
            "SELECT p.g, COUNT(*), SUM(c.v) FROM p JOIN c ON c.pid = p.id \
             WHERE p.g + c.v > 3 GROUP BY p.g ORDER BY p.g",
            C_INDEX,
            |p, c| p.1 + c.2 > 3,
        ),
        (
            "SELECT p.g, COUNT(*), SUM(c.v) FROM c JOIN p ON p.id = c.pid \
             WHERE p.g + c.v > 3 GROUP BY p.g ORDER BY p.g",
            "batch_index_nested_loop",
            |p, c| p.1 + c.2 > 3,
        ),
    ];
    for (sql, path, keep) in cases {
        assert_eq!(parted_by(&db, sql, path), keyed_groups(keep), "{sql}");
    }
}

#[test]
fn distinct_counts_rollups_and_having_read_an_index_join_in_parts() {
    let db = open_keyed("join_aggregate_parts_index_shapes");
    let pairs = keyed_pairs();
    let mut distinct: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for (p, c) in &pairs {
        distinct.entry(p.1).or_default().insert(c.2);
    }
    let want: Vec<String> = distinct
        .iter()
        .map(|(g, v)| [int(*g), int(v.len() as i64)].join("|"))
        .collect();
    let sql = "SELECT p.g, COUNT(DISTINCT c.v) FROM p JOIN c ON c.pid = p.id \
               GROUP BY p.g ORDER BY p.g";
    assert_eq!(parted_by(&db, sql, C_INDEX), want);

    let mut groups: BTreeMap<i64, i64> = BTreeMap::new();
    for (p, _) in &pairs {
        *groups.entry(p.1).or_default() += 1;
    }
    let want: Vec<String> = std::iter::once((-1, pairs.len() as i64))
        .chain(groups)
        .map(|(g, count)| [int(g), int(count)].join("|"))
        .collect();
    let sql = "SELECT COALESCE(p.g, -1), COUNT(*) FROM p JOIN c ON c.pid = p.id \
               GROUP BY ROLLUP(p.g) ORDER BY 1";
    assert_eq!(parted_by(&db, sql, C_INDEX), want);

    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    for (p, _) in &pairs {
        *counts.entry(p.0).or_default() += 1;
    }
    let want: Vec<String> = counts
        .iter()
        .filter(|(_, n)| **n > 6)
        .map(|(id, n)| [int(*id), int(*n)].join("|"))
        .collect();
    let sql = "SELECT p.id, COUNT(*) FROM c JOIN p ON p.id = c.pid \
               GROUP BY p.id HAVING COUNT(*) > 6 ORDER BY p.id";
    assert_eq!(parted_by(&db, sql, "batch_index_nested_loop"), want);
}

#[test]
fn an_index_join_raises_its_on_error_before_the_where_reads_a_row() {
    let db = open_keyed("join_aggregate_parts_index_on_error");
    let sql = "SELECT c.id FROM c JOIN p ON p.id = c.pid \
               AND CASE WHEN c.id = 1 THEN 9223372036854775807 + c.id ELSE 0 END = 0 \
               WHERE EXISTS (SELECT 1 FROM p a JOIN c b ON a.id = b.pid AND b.v >= a.g - 9 \
               WHERE b.id = c.id)";
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::reset_join_paths();
    let err = match db.query(sql, ()) {
        Ok(rows) => rows.map(|r| r.map(|_| ())).collect::<Result<Vec<_>, _>>(),
        Err(e) => Err(e),
    }
    .expect_err("c.id = 1 overflows the ON");
    #[cfg(feature = "test-failpoints")]
    assert_eq!(
        stoolap::test_failpoints::join_paths_taken().first(),
        Some(&"batch_index_nested_loop")
    );
    assert!(err.to_string().contains("overflow"), "{err}");
}

/// `open`'s l and r, s: r's rows under sid, k, w, so l and s share k alone,
/// and t: r's id and w under tid, x
fn open_merged(name: &str) -> Database {
    let db = open(name);
    for sql in [
        "CREATE TABLE s (sid INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "INSERT INTO s SELECT id, k, w FROM r",
        "CREATE TABLE t (tid INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO t SELECT id, w FROM r",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Per merged k of the rows a `kind` join of l and r on k makes: the count,
/// the l rows, the r rows and the sum of r.w; an unmatched row's merged k is
/// its own side's
fn merged_groups(kind: &str) -> Vec<String> {
    let mut groups: BTreeMap<i64, (i64, i64, i64, i64)> = BTreeMap::new();
    for (l, r) in pairs() {
        let group = groups.entry(l.1).or_default();
        *group = (group.0 + 1, group.1 + 1, group.2 + 1, group.3 + r.2);
    }
    if kind == "LEFT" || kind == "FULL" {
        for l in l_rows().filter(|l| !r_rows().any(|r| r.1 == l.1)) {
            let group = groups.entry(l.1).or_default();
            *group = (group.0 + 1, group.1 + 1, group.2, group.3);
        }
    }
    if kind == "RIGHT" || kind == "FULL" {
        for r in r_rows().filter(|r| !l_rows().any(|l| l.1 == r.1)) {
            let group = groups.entry(r.1).or_default();
            *group = (group.0 + 1, group.1, group.2 + 1, group.3 + r.2);
        }
    }
    groups
        .iter()
        .map(|(k, (n, ls, rs, sum))| [int(*k), int(*n), int(*ls), int(*rs), int(*sum)].join("|"))
        .collect()
}

#[test]
fn using_and_natural_joins_are_grouped_by_their_merged_column_in_parts() {
    let db = open_merged("join_aggregate_parts_using");
    let want = merged_groups("INNER");
    for sql in [
        "SELECT k, COUNT(*), COUNT(l.id), COUNT(r.id), SUM(r.w) FROM l JOIN r USING (k) \
         GROUP BY k ORDER BY k",
        "SELECT k, COUNT(*), COUNT(l.id), COUNT(s.sid), SUM(s.w) FROM l NATURAL JOIN s \
         GROUP BY k ORDER BY k",
    ] {
        assert_eq!(parted(&db, sql), want, "{sql}");
    }
    let mut by_g: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (l, r) in pairs() {
        let group = by_g.entry(l.2).or_default();
        *group = (group.0 + 1, group.1 + r.2);
    }
    let want: Vec<String> = by_g
        .iter()
        .map(|(g, (n, sum))| [int(*g), int(*n), int(*sum)].join("|"))
        .collect();
    let sql = "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r USING (k) GROUP BY l.g ORDER BY l.g";
    assert_eq!(parted(&db, sql), want);
}

#[test]
fn outer_using_joins_group_unmatched_rows_under_their_merged_column_in_parts() {
    let db = open_merged("join_aggregate_parts_using_outer");
    for kind in ["LEFT", "RIGHT", "FULL"] {
        let sql = format!(
            "SELECT k, COUNT(*), COUNT(l.id), COUNT(r.id), COALESCE(SUM(r.w), 0) \
             FROM l {kind} JOIN r USING (k) GROUP BY k ORDER BY k"
        );
        assert_eq!(parted(&db, &sql), merged_groups(kind), "{kind}");
    }
}

#[test]
fn a_where_having_distinct_count_and_rollup_read_a_merged_column_in_parts() {
    let db = open_merged("join_aggregate_parts_using_shapes");
    let pairs = pairs();
    let mut below: BTreeMap<i64, i64> = BTreeMap::new();
    for (l, _) in pairs.iter().filter(|(l, r)| l.1 > 4 && l.0 < r.0) {
        *below.entry(l.1).or_default() += 1;
    }
    let want: Vec<String> = (below.iter())
        .filter(|(_, n)| **n > 5)
        .map(|(k, n)| [int(*k), int(*n)].join("|"))
        .collect();
    let sql = "SELECT k, COUNT(*) FROM l JOIN r USING (k) WHERE k > 4 AND l.id < r.id \
               GROUP BY k HAVING COUNT(*) > 5 ORDER BY k";
    assert_eq!(parted(&db, sql), want);

    let mut distinct: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for (l, r) in &pairs {
        distinct.entry(l.1).or_default().insert(r.2);
    }
    let want: Vec<String> = (distinct.iter())
        .map(|(k, w)| [int(*k), int(w.len() as i64)].join("|"))
        .collect();
    let sql = "SELECT k, COUNT(DISTINCT w) FROM l NATURAL JOIN s GROUP BY k ORDER BY k";
    assert_eq!(parted(&db, sql), want);

    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    for (l, _) in &pairs {
        *counts.entry(l.1).or_default() += 1;
    }
    let want: Vec<String> = std::iter::once((-1, pairs.len() as i64))
        .chain(counts)
        .map(|(k, n)| [int(k), int(n)].join("|"))
        .collect();
    let sql = "SELECT COALESCE(k, -1), COUNT(*) FROM l JOIN r USING (k) \
               GROUP BY ROLLUP(k) ORDER BY 1";
    assert_eq!(parted(&db, sql), want);
}

#[test]
fn a_join_over_a_using_join_is_grouped_by_its_merged_column_in_parts() {
    let db = open_merged("join_aggregate_parts_using_side");
    let mut groups: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (l, r) in pairs() {
        let ts = r_rows().filter(|t| t.2 == r.2).count() as i64;
        let group = groups.entry(l.1).or_default();
        *group = (group.0 + ts, group.1 + ts * r.2);
    }
    let want: Vec<String> = groups
        .iter()
        .map(|(k, (n, sum))| [int(*k), int(*n), int(*sum)].join("|"))
        .collect();
    let sql = "SELECT k, COUNT(*), SUM(t.x) FROM l JOIN r USING (k) \
               JOIN t ON t.x = r.w GROUP BY k ORDER BY k";
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::reset_join_paths();
    assert_eq!(parted(&db, sql), want);
    #[cfg(feature = "test-failpoints")]
    assert_eq!(stoolap::test_failpoints::join_paths_taken(), vec!["hash"]);
}

#[cfg(feature = "test-failpoints")]
mod made_as_read {
    use std::sync::atomic::{AtomicI64, Ordering};

    use stoolap::core::{Result, Value};
    use stoolap::functions::registry::global_registry;
    use stoolap::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
    };

    static TICKS: AtomicI64 = AtomicI64::new(0);

    fn info(name: &str) -> FunctionInfo {
        FunctionInfo::new(
            name,
            FunctionType::Scalar,
            "test counter",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
        .non_deterministic()
    }

    /// Returns its argument and counts the pairs the join asked
    #[derive(Default)]
    struct JoinPartTick;

    impl ScalarFunction for JoinPartTick {
        fn name(&self) -> &str {
            "JOIN_PART_TICK"
        }
        fn info(&self) -> FunctionInfo {
            info("JOIN_PART_TICK")
        }
        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            TICKS.fetch_add(1, Ordering::Relaxed);
            Ok(args[0].clone())
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(JoinPartTick)
        }
    }

    /// The pairs the join had asked when the aggregate read a row
    #[derive(Default)]
    struct JoinPartSeen;

    impl ScalarFunction for JoinPartSeen {
        fn name(&self) -> &str {
            "JOIN_PART_SEEN"
        }
        fn info(&self) -> FunctionInfo {
            info("JOIN_PART_SEEN")
        }
        fn evaluate(&self, _args: &[Value]) -> Result<Value> {
            Ok(Value::Integer(TICKS.load(Ordering::Relaxed)))
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(JoinPartSeen)
        }
    }

    #[test]
    fn an_index_join_makes_its_rows_as_the_aggregate_reads_them() {
        global_registry().register_scalar::<JoinPartTick>().unwrap();
        global_registry().register_scalar::<JoinPartSeen>().unwrap();
        let db = super::open_keyed("join_aggregate_parts_index_made_as_read");
        let pairs = super::keyed_pairs().len() as i64;
        for (sql, path) in [
            (
                "SELECT MIN(JOIN_PART_SEEN(c.id)), COUNT(*) FROM p JOIN c ON c.pid = p.id \
                 AND JOIN_PART_TICK(p.id + c.id) = p.id + c.id",
                super::C_INDEX,
            ),
            (
                "SELECT MIN(JOIN_PART_SEEN(c.id)), COUNT(*) FROM c JOIN p ON p.id = c.pid \
                 AND JOIN_PART_TICK(p.id + c.id) = p.id + c.id",
                "batch_index_nested_loop",
            ),
        ] {
            let before = TICKS.load(Ordering::Relaxed);
            let out = super::parted_by(&db, sql, path);
            let row: Vec<&str> = out[0].split('|').collect();
            assert_eq!(row[1], super::int(pairs), "{sql}");
            let first_seen: i64 = row[0]
                .trim_start_matches("Integer(")
                .trim_end_matches(')')
                .parse()
                .unwrap();
            let asked = first_seen - before;
            assert!(
                asked <= 2 * super::PART_ROWS as i64,
                "{sql}: the join had asked {asked} of {pairs} pairs when the aggregate read its first row"
            );
        }
    }
}

/// A row of q: id, a
type Q = (i64, i64);

/// q: id 1..=20, a = id % 6, which the subqueries below read
fn q_rows() -> impl Iterator<Item = Q> + Clone {
    (1..=20).map(|id| (id, id % 6))
}

fn with_q(db: Database) -> Database {
    for sql in [
        "CREATE TABLE q (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO q SELECT value, value % 6 FROM generate_series(1, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Whether q holds a row of `a` with an id below `id`
fn q_below(a: i64, id: i64) -> bool {
    q_rows().any(|q| q.1 == a && q.0 < id)
}

/// Per l.g of the pairs of l and r that `meet` and `keep` keeps: the count
/// and the sum of r.w
fn lr_groups(
    meet: impl Fn(&Fixture, &Fixture) -> bool,
    keep: impl Fn(&Fixture, &Fixture) -> bool,
) -> Vec<String> {
    let mut groups: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for l in l_rows() {
        for r in r_rows().filter(|r| meet(&l, r) && keep(&l, r)) {
            let group = groups.entry(l.2).or_default();
            group.0 += 1;
            group.1 += r.2;
        }
    }
    groups
        .iter()
        .map(|(g, (count, sum))| [int(*g), int(*count), int(*sum)].join("|"))
        .collect()
}

/// Which (l, r) pairs a statement keeps
type Keep3 = fn(&Fixture, &Fixture) -> bool;

fn equal_keys(l: &Fixture, r: &Fixture) -> bool {
    l.1 == r.1
}

#[test]
fn a_hash_join_asks_its_on_subqueries_in_parts() {
    let db = with_q(open("join_aggregate_parts_on_blocks_hash"));
    let cases: [(&str, Keep3); 3] = [
        (
            "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k \
             AND EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id) \
             GROUP BY l.g ORDER BY l.g",
            |l, r| q_below(r.2, l.0),
        ),
        (
            "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k \
             AND r.id IN (SELECT q.id * 3 FROM q WHERE q.a = l.g) \
             GROUP BY l.g ORDER BY l.g",
            |l, r| q_rows().any(|q| q.1 == l.2 && q.0 * 3 == r.0),
        ),
        (
            "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k \
             AND l.id + r.id > (SELECT MIN(q.id) * 10 FROM q WHERE q.a = 5) \
             GROUP BY l.g ORDER BY l.g",
            |l, r| l.0 + r.0 > 50,
        ),
    ];
    for (sql, keep) in cases {
        assert_eq!(
            parted_by(&db, sql, "hash"),
            lr_groups(equal_keys, keep),
            "{sql}"
        );
    }
}

#[test]
fn outer_hash_joins_ask_their_on_subqueries_in_parts() {
    let db = with_q(open("join_aggregate_parts_on_blocks_outer"));
    let kept = |l: &Fixture, r: &Fixture| l.1 == r.1 && q_below(r.2, l.0);
    let n = l_rows()
        .flat_map(|l| r_rows().filter(move |r| kept(&l, r)))
        .count() as i64;
    let lone_l = l_rows().filter(|l| !r_rows().any(|r| kept(l, &r))).count() as i64;
    let lone_r = r_rows().filter(|r| !l_rows().any(|l| kept(&l, r))).count() as i64;
    let on = "ON l.k = r.k AND EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id)";
    for (kind, total) in [
        ("LEFT", n + lone_l),
        ("RIGHT", n + lone_r),
        ("FULL", n + lone_l + lone_r),
    ] {
        let sql = format!("SELECT COUNT(*), COUNT(l.id), COUNT(r.id) FROM l {kind} JOIN r {on}");
        let want = match kind {
            "LEFT" => [total, total, n],
            "RIGHT" => [total, n, total],
            _ => [total, n + lone_l, n + lone_r],
        };
        let want = vec![want.map(int).join("|")];
        assert_eq!(parted_by(&db, &sql, "hash"), want, "{sql}");
    }
}

#[test]
fn a_nested_loop_asks_its_on_subqueries_in_parts() {
    let db = with_q(open("join_aggregate_parts_on_blocks_nested"));
    let below = |l: &Fixture, r: &Fixture| l.1 < r.1;
    let sql = "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k < r.k \
               AND EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id) \
               GROUP BY l.g ORDER BY l.g";
    assert_eq!(
        parted_by(&db, sql, "nested_loop"),
        lr_groups(below, |l, r| q_below(r.2, l.0))
    );
    let kept = |l: &Fixture, r: &Fixture| l.1 < r.1 && q_below(r.2, l.0);
    let n = l_rows()
        .flat_map(|l| r_rows().filter(move |r| kept(&l, r)))
        .count() as i64;
    let lone_l = l_rows().filter(|l| !r_rows().any(|r| kept(l, &r))).count() as i64;
    let sql = "SELECT COUNT(*), COUNT(r.id) FROM l LEFT JOIN r ON l.k < r.k \
               AND EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id)";
    assert_eq!(
        parted_by(&db, sql, "nested_loop"),
        vec![[int(n + lone_l), int(n)].join("|")]
    );
}

/// Which (bl.id, br.id) pairs a statement keeps
type WideKeep = fn(i64, i64) -> bool;

#[test]
fn a_join_built_across_threads_asks_its_on_subqueries_in_parts() {
    let db = with_q(open_wide("join_aggregate_parts_on_blocks_wide"));
    let cases: [(&str, WideKeep); 2] = [
        (
            "SELECT bl.g, COUNT(*), SUM(br.v) FROM bl JOIN br ON bl.u = br.u \
             AND br.v IN (SELECT q.a FROM q WHERE q.id < 4) GROUP BY bl.g ORDER BY bl.g",
            |_, r| q_rows().any(|q| q.0 < 4 && q.1 == r % 7),
        ),
        (
            "SELECT bl.g, COUNT(*), SUM(br.v) FROM bl JOIN br ON bl.u = br.u \
             AND EXISTS (SELECT 1 FROM q WHERE q.a = bl.g AND q.id < br.v + 2) \
             GROUP BY bl.g ORDER BY bl.g",
            |l, r| q_below(l % 5, r % 7 + 2),
        ),
    ];
    for (sql, keep) in cases {
        let mut groups: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
        for (l, r) in wide_pairs().filter(|(l, r)| keep(*l, *r)) {
            let group = groups.entry(l % 5).or_default();
            group.0 += 1;
            group.1 += r % 7;
        }
        let want: Vec<String> = groups
            .iter()
            .map(|(g, (count, sum))| [int(*g), int(*count), int(*sum)].join("|"))
            .collect();
        assert_eq!(parted_across_threads(&db, sql), want, "{sql}");
    }
}

#[test]
fn an_on_that_fails_beside_a_subquery_running_a_join_raises_its_error_in_parts() {
    let db = with_q(open("join_aggregate_parts_on_blocks_error"));
    for (on, path) in [("l.k = r.k", "hash"), ("l.k < r.k", "nested_loop")] {
        let sql = format!(
            "SELECT COUNT(*) FROM l JOIN r ON {on} \
             AND CASE WHEN l.id = 3 THEN 9223372036854775807 + l.id ELSE 0 END = 0 \
             AND EXISTS (SELECT 1 FROM l a JOIN r b ON a.k = b.k \
             WHERE b.id = r.id AND a.id <> l.id)"
        );
        #[cfg(feature = "test-failpoints")]
        stoolap::test_failpoints::reset_join_paths();
        let err = match db.query(&sql, ()) {
            Ok(rows) => rows.map(|r| r.map(|_| ())).collect::<Result<Vec<_>, _>>(),
            Err(e) => Err(e),
        }
        .expect_err("l.id = 3 overflows the ON");
        #[cfg(feature = "test-failpoints")]
        assert!(
            stoolap::test_failpoints::join_paths_taken().contains(&path),
            "{sql}"
        );
        #[cfg(not(feature = "test-failpoints"))]
        let _ = path;
        assert!(err.to_string().contains("overflow"), "{sql}: {err}");
    }
}

#[test]
fn an_index_join_asks_its_where_and_on_subqueries_in_parts() {
    let db = with_q(open_keyed("join_aggregate_parts_index_blocks"));
    let correlated = "EXISTS (SELECT 1 FROM q WHERE q.a = c.v AND q.id < p.id)";
    let cases: [(String, &str, Keep); 6] = [
        (
            format!("FROM c JOIN p ON p.id = c.pid WHERE {correlated}"),
            "batch_index_nested_loop",
            |p, c| q_below(c.2, p.0),
        ),
        (
            "FROM c JOIN p ON p.id = c.pid WHERE c.v IN (SELECT q.a FROM q WHERE q.id < 4)"
                .to_string(),
            "batch_index_nested_loop",
            |_, c| q_rows().any(|q| q.0 < 4 && q.1 == c.2),
        ),
        (
            format!("FROM c JOIN p ON p.id = c.pid AND {correlated}"),
            "batch_index_nested_loop",
            |p, c| q_below(c.2, p.0),
        ),
        (
            format!("FROM p JOIN c ON c.pid = p.id WHERE {correlated}"),
            C_INDEX,
            |p, c| q_below(c.2, p.0),
        ),
        (
            format!("FROM p JOIN c ON c.pid = p.id AND {correlated}"),
            C_INDEX,
            |p, c| q_below(c.2, p.0),
        ),
        (
            "FROM p JOIN c ON c.pid = p.id WHERE p.g + c.v > (SELECT MIN(q.a) + 2 FROM q)"
                .to_string(),
            C_INDEX,
            |p, c| p.1 + c.2 > 2,
        ),
    ];
    for (from, path, keep) in cases {
        let sql = format!("SELECT p.g, COUNT(*), SUM(c.v) {from} GROUP BY p.g ORDER BY p.g");
        assert_eq!(parted_by(&db, &sql, path), keyed_groups(keep), "{sql}");
    }
}

#[test]
fn a_prepared_join_asks_its_subqueries_again_for_each_parameter_in_parts() {
    let db = with_q(open_keyed("join_aggregate_parts_blocks_prepared"));
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "INSERT INTO l SELECT value, value % 10, value % 3 FROM generate_series(1, 40)",
        "INSERT INTO r SELECT value, value % 10 + 3, value % 4 FROM generate_series(1, 60)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let index = db
        .prepare(
            "SELECT p.g, COUNT(*), SUM(c.v) FROM c JOIN p ON p.id = c.pid \
             WHERE EXISTS (SELECT 1 FROM q WHERE q.a = c.v AND q.id < $1) \
             GROUP BY p.g ORDER BY p.g",
        )
        .unwrap();
    let hash = db
        .prepare(
            "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k \
             AND EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < $1) \
             GROUP BY l.g ORDER BY l.g",
        )
        .unwrap();
    let hash_where = db
        .prepare(
            "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k \
             WHERE EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id + $1) \
             GROUP BY l.g ORDER BY l.g",
        )
        .unwrap();
    let read = |query: &stoolap::Statement, below: i64| -> Vec<String> {
        #[cfg(feature = "test-failpoints")]
        {
            stoolap::test_failpoints::set_join_part_rows(PART_ROWS);
            stoolap::test_failpoints::reset_join_part_largest();
        }
        let mut out = Vec::new();
        for row in query.query((below,)).unwrap() {
            let row = row.unwrap();
            out.push(
                (0..row.len())
                    .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
        #[cfg(feature = "test-failpoints")]
        {
            stoolap::test_failpoints::set_join_part_rows(0);
            let largest = stoolap::test_failpoints::join_part_largest();
            assert!(
                (1..=PART_ROWS).contains(&largest),
                "$1 = {below}: the aggregation held {largest} joined rows at once"
            );
        }
        out
    };
    for below in [5, 15] {
        let want = keyed_groups(|_, c| q_below(c.2, below));
        assert_eq!(read(&index, below), want, "index join, $1 = {below}");
        let want = lr_groups(equal_keys, |_, r| q_below(r.2, below));
        assert_eq!(read(&hash, below), want, "hash join, $1 = {below}");
        let want = lr_groups(equal_keys, |l, r| q_below(r.2, l.0 + below));
        assert_eq!(
            read(&hash_where, below),
            want,
            "hash join WHERE, $1 = {below}"
        );
    }
}

/// z: id 1..=10000, k = id % 100 + 5, an index on k, which an EXISTS on
/// l.k probes or builds a keyed set of
fn with_z(db: Database) -> Database {
    for sql in [
        "CREATE TABLE z (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE INDEX z_k ON z (k)",
        "INSERT INTO z SELECT value, value % 100 + 5 FROM generate_series(1, 10000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The probes and the keyed sets the blocks ran on this thread for `sql`,
/// read with parts of `part_rows` rows (0 for the default), its rows checked
/// against `want`
#[cfg(feature = "test-failpoints")]
fn strategies(db: &Database, sql: &str, part_rows: usize, want: &[String]) -> (usize, usize) {
    use stoolap::test_failpoints::{prepared_strategy_runs_here, PreparedStrategy};
    stoolap::test_failpoints::set_join_part_rows(part_rows);
    stoolap::test_failpoints::reset_join_part_largest();
    let probes = prepared_strategy_runs_here(PreparedStrategy::Probe);
    let keyed = prepared_strategy_runs_here(PreparedStrategy::KeyedSet);
    assert_eq!(rows(db, sql), want, "{sql}");
    let ran = (
        prepared_strategy_runs_here(PreparedStrategy::Probe) - probes,
        prepared_strategy_runs_here(PreparedStrategy::KeyedSet) - keyed,
    );
    stoolap::test_failpoints::set_join_part_rows(0);
    assert!(
        stoolap::test_failpoints::join_part_largest() > 0,
        "{sql}: the aggregation read no part"
    );
    ran
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_where_subquery_after_a_join_read_in_parts_chooses_its_strategy_by_the_first_part() {
    let db = with_z(open("join_aggregate_parts_where_strategy"));
    let sql = "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k \
               WHERE EXISTS (SELECT 1 FROM z WHERE z.k = l.k) \
               GROUP BY l.g ORDER BY l.g";
    let want = lr_groups(equal_keys, |l, _| l.1 >= 5);
    let joined = pairs().len();
    let (probes, keyed) = strategies(&db, sql, 0, &want);
    assert!(
        probes > 0 && keyed == 0,
        "{joined} joined rows in one part over 10000 rows of z probe per row: \
         {probes} probes, {keyed} keyed sets"
    );
    let (probes, keyed) = strategies(&db, sql, PART_ROWS, &want);
    assert_eq!(
        (probes, keyed),
        (0, 1),
        "{joined} joined rows past a part of {PART_ROWS} build the keyed set once"
    );
}

#[test]
fn a_hash_join_asks_its_where_subqueries_in_parts() {
    let db = with_q(open("join_aggregate_parts_where_blocks_hash"));
    let cases: [(&str, Keep3); 4] = [
        ("WHERE l.id + r.id IN (SELECT q.id * 4 FROM q)", |l, r| {
            (l.0 + r.0) % 4 == 0 && (1..=20).contains(&((l.0 + r.0) / 4))
        }),
        (
            "WHERE NOT EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id)",
            |l, r| !q_below(r.2, l.0),
        ),
        (
            "WHERE l.id < ALL (SELECT q.id FROM q WHERE q.a = r.w)",
            |l, r| q_rows().filter(|q| q.1 == r.2).all(|q| l.0 < q.0),
        ),
        (
            "WHERE l.id + r.id > (SELECT MIN(q.id) * 10 FROM q WHERE q.a = 5)",
            |l, r| l.0 + r.0 > 50,
        ),
    ];
    for (where_, keep) in cases {
        let sql = format!(
            "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k {where_} \
             GROUP BY l.g ORDER BY l.g"
        );
        assert_eq!(
            parted_by(&db, &sql, "hash"),
            lr_groups(equal_keys, keep),
            "{sql}"
        );
    }
}

#[test]
fn outer_joins_ask_their_where_subqueries_of_unmatched_rows_in_parts() {
    let db = with_q(open("join_aggregate_parts_where_blocks_outer"));
    let pairs = pairs();
    let kept = pairs.iter().filter(|(l, r)| q_below(r.2, l.0)).count() as i64;
    let lone_l = l_rows().filter(|l| !r_rows().any(|r| r.1 == l.1)).count() as i64;
    let lone_r = r_rows().filter(|r| !l_rows().any(|l| l.1 == r.1)).count() as i64;
    for (kind, want) in [
        ("LEFT", [kept + lone_l, kept]),
        ("RIGHT", [kept + lone_r, kept + lone_r]),
        ("FULL", [kept + lone_l + lone_r, kept + lone_r]),
    ] {
        let sql = format!(
            "SELECT COUNT(*), COUNT(r.id) FROM l {kind} JOIN r ON l.k = r.k \
             WHERE l.id IS NULL OR r.id IS NULL \
             OR EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id)"
        );
        let want = vec![want.map(int).join("|")];
        assert_eq!(parted_by(&db, &sql, "hash"), want, "{sql}");
    }
}

#[test]
fn a_nested_loop_and_a_join_across_threads_ask_their_where_subqueries_in_parts() {
    let db = with_q(open("join_aggregate_parts_where_blocks_nested"));
    let sql = "SELECT l.g, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k < r.k \
               WHERE EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id) \
               GROUP BY l.g ORDER BY l.g";
    assert_eq!(
        parted_by(&db, sql, "nested_loop"),
        lr_groups(|l, r| l.1 < r.1, |l, r| q_below(r.2, l.0))
    );

    let db = with_q(open_wide("join_aggregate_parts_where_blocks_wide"));
    let mut groups: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (l, r) in wide_pairs().filter(|(l, r)| q_below(l % 5, r % 7 + 2)) {
        let group = groups.entry(l % 5).or_default();
        group.0 += 1;
        group.1 += r % 7;
    }
    let want: Vec<String> = groups
        .iter()
        .map(|(g, (count, sum))| [int(*g), int(*count), int(*sum)].join("|"))
        .collect();
    let sql = "SELECT bl.g, COUNT(*), SUM(br.v) FROM bl JOIN br ON bl.u = br.u \
               WHERE EXISTS (SELECT 1 FROM q WHERE q.a = bl.g AND q.id < br.v + 2) \
               GROUP BY bl.g ORDER BY bl.g";
    assert_eq!(parted_across_threads(&db, sql), want);
}

#[test]
fn a_using_join_asks_its_where_subqueries_after_its_merged_column_in_parts() {
    let db = with_q(open_merged("join_aggregate_parts_where_blocks_using"));
    let mut groups: BTreeMap<i64, i64> = BTreeMap::new();
    for (_, r) in pairs().iter().filter(|(l, r)| q_below(r.2, l.0)) {
        *groups.entry(r.1).or_default() += 1;
    }
    let want: Vec<String> = groups
        .iter()
        .map(|(k, count)| [int(*k), int(*count)].join("|"))
        .collect();
    let sql = "SELECT k, COUNT(*) FROM l JOIN s USING (k) \
               WHERE EXISTS (SELECT 1 FROM q WHERE q.a = s.w AND q.id < l.id) \
               GROUP BY k ORDER BY k";
    assert_eq!(parted_by(&db, sql, "hash"), want);
}

#[test]
fn a_where_beside_a_subquery_running_a_join_raises_its_error_in_parts() {
    let db = with_q(open("join_aggregate_parts_where_blocks_error"));
    for (on, path) in [("l.k = r.k", "hash"), ("l.k < r.k", "nested_loop")] {
        let sql = format!(
            "SELECT COUNT(*) FROM l JOIN r ON {on} \
             WHERE CASE WHEN l.id = 3 THEN 9223372036854775807 + r.id ELSE 0 END = 0 \
             AND EXISTS (SELECT 1 FROM l a JOIN r b ON a.k = b.k \
             WHERE b.id = r.id AND a.id <> l.id)"
        );
        #[cfg(feature = "test-failpoints")]
        stoolap::test_failpoints::reset_join_paths();
        let err = match db.query(&sql, ()) {
            Ok(rows) => rows.map(|r| r.map(|_| ())).collect::<Result<Vec<_>, _>>(),
            Err(e) => Err(e),
        }
        .expect_err("l.id = 3 overflows the WHERE");
        #[cfg(feature = "test-failpoints")]
        assert!(
            stoolap::test_failpoints::join_paths_taken().contains(&path),
            "{sql}"
        );
        #[cfg(not(feature = "test-failpoints"))]
        let _ = path;
        assert!(err.to_string().contains("overflow"), "{sql}: {err}");
    }
}

/// The rows of q with `a`
fn q_with(a: i64) -> i64 {
    q_rows().filter(|q| q.1 == a).count() as i64
}

/// Per l.g of the pairs of l and r that `meet` and `keep` keeps, its count
fn lr_counts(
    meet: impl Fn(&Fixture, &Fixture) -> bool,
    keep: impl Fn(&Fixture, &Fixture) -> bool,
) -> BTreeMap<i64, i64> {
    let mut counts = BTreeMap::new();
    for l in l_rows() {
        let met = r_rows().filter(|r| meet(&l, r) && keep(&l, r)).count() as i64;
        if met > 0 {
            *counts.entry(l.2).or_default() += met;
        }
    }
    counts
}

#[test]
fn subqueries_in_a_grouped_joins_output_and_having_run_over_its_parts() {
    let db = with_q(open("join_aggregate_parts_family_hash"));
    let counts = lr_counts(equal_keys, |_, _| true);
    let max_a = q_rows().map(|q| q.1).max().unwrap();
    let row = |g: &i64, values: &[i64]| {
        std::iter::once(int(*g))
            .chain(values.iter().map(|v| int(*v)))
            .collect::<Vec<_>>()
            .join("|")
    };
    let cases: Vec<(&str, Vec<String>)> = vec![
        (
            "SELECT l.g, COUNT(*), (SELECT MAX(q.a) FROM q) FROM l JOIN r ON l.k = r.k \
             GROUP BY l.g ORDER BY l.g",
            counts.iter().map(|(g, n)| row(g, &[*n, max_a])).collect(),
        ),
        (
            "SELECT l.g, COUNT(*) FROM l JOIN r ON l.k = r.k GROUP BY l.g \
             HAVING COUNT(*) > (SELECT COUNT(*) FROM q WHERE q.a < 3) ORDER BY l.g",
            counts
                .iter()
                .filter(|(_, n)| **n > (0..3).map(q_with).sum::<i64>())
                .map(|(g, n)| row(g, &[*n]))
                .collect(),
        ),
        (
            "SELECT l.g, COUNT(*), (SELECT COUNT(*) FROM q WHERE q.a = l.g) FROM l \
             JOIN r ON l.k = r.k GROUP BY l.g ORDER BY l.g",
            counts
                .iter()
                .map(|(g, n)| row(g, &[*n, q_with(*g)]))
                .collect(),
        ),
        (
            "SELECT l.g, COUNT(*) FROM l JOIN r ON l.k = r.k GROUP BY l.g \
             ORDER BY (SELECT MAX(q.id) FROM q WHERE q.a = l.g) DESC",
            {
                let top = |g: i64| q_rows().filter(|q| q.1 == g).map(|q| q.0).max();
                let mut groups: Vec<_> = counts.iter().collect();
                groups.sort_by_key(|(g, _)| std::cmp::Reverse(top(**g)));
                groups.into_iter().map(|(g, n)| row(g, &[*n])).collect()
            },
        ),
        (
            "SELECT DISTINCT ON ((SELECT COUNT(*) FROM q WHERE q.a = l.g)) l.g, COUNT(*) \
             FROM l JOIN r ON l.k = r.k GROUP BY l.g \
             ORDER BY (SELECT COUNT(*) FROM q WHERE q.a = l.g), l.g",
            {
                let mut seen = BTreeSet::new();
                let mut groups: Vec<_> = counts.iter().collect();
                groups.sort_by_key(|(g, _)| (q_with(**g), **g));
                groups
                    .into_iter()
                    .filter(|(g, _)| seen.insert(q_with(**g)))
                    .map(|(g, n)| row(g, &[*n]))
                    .collect()
            },
        ),
        (
            "SELECT l.g, COUNT(*), (SELECT MAX(q.a) FROM q) FROM l JOIN r ON l.k = r.k \
             WHERE EXISTS (SELECT 1 FROM q WHERE q.a = r.w AND q.id < l.id) \
             GROUP BY l.g ORDER BY l.g",
            lr_counts(equal_keys, |l, r| q_below(r.2, l.0))
                .iter()
                .map(|(g, n)| row(g, &[*n, max_a]))
                .collect(),
        ),
        (
            "SELECT l.g, COUNT(DISTINCT r.w), (SELECT MAX(q.a) FROM q) FROM l \
             JOIN r ON l.k = r.k GROUP BY l.g ORDER BY l.g",
            {
                let mut ws: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
                for (l, r) in pairs() {
                    ws.entry(l.2).or_default().insert(r.2);
                }
                ws.iter()
                    .map(|(g, w)| row(g, &[w.len() as i64, max_a]))
                    .collect()
            },
        ),
        (
            "SELECT COALESCE(l.g, -1), COUNT(*) FROM l JOIN r ON l.k = r.k \
             GROUP BY ROLLUP(l.g) HAVING COUNT(*) > (SELECT MIN(q.id) FROM q) ORDER BY 1",
            std::iter::once((-1, counts.values().sum::<i64>()))
                .chain(counts.iter().map(|(g, n)| (*g, *n)))
                .filter(|(_, n)| *n > 1)
                .map(|(g, n)| row(&g, &[n]))
                .collect(),
        ),
    ];
    for (sql, want) in cases {
        assert_eq!(parted_by(&db, sql, "hash"), want, "{sql}");
    }
}

#[test]
fn grouped_joins_of_every_kind_run_their_subqueries_over_their_parts() {
    let db = with_q(open("join_aggregate_parts_family_kinds"));
    let max_a = q_rows().map(|q| q.1).max().unwrap();
    let lone_l = l_rows().filter(|l| !r_rows().any(|r| r.1 == l.1)).count() as i64;
    let matched = pairs().len() as i64;
    let sql = "SELECT COUNT(*), COUNT(r.id), (SELECT MAX(q.a) FROM q) FROM l \
               LEFT JOIN r ON l.k = r.k";
    assert_eq!(
        parted_by(&db, sql, "hash"),
        vec![[matched + lone_l, matched, max_a].map(int).join("|")],
        "{sql}"
    );
    let sql = "SELECT l.g, COUNT(*), (SELECT MAX(q.a) FROM q) FROM l JOIN r ON l.k < r.k \
               GROUP BY l.g ORDER BY l.g";
    let want: Vec<String> = lr_counts(|l, r| l.1 < r.1, |_, _| true)
        .iter()
        .map(|(g, n)| [*g, *n, max_a].map(int).join("|"))
        .collect();
    assert_eq!(parted_by(&db, sql, "nested_loop"), want, "{sql}");

    let db = with_q(open_wide("join_aggregate_parts_family_wide"));
    let mut groups: BTreeMap<i64, i64> = BTreeMap::new();
    for (l, _) in wide_pairs() {
        *groups.entry(l % 5).or_default() += 1;
    }
    let want: Vec<String> = groups
        .iter()
        .map(|(g, n)| [*g, *n, q_with(*g)].map(int).join("|"))
        .collect();
    let sql = "SELECT bl.g, COUNT(*), (SELECT COUNT(*) FROM q WHERE q.a = bl.g) FROM bl \
               JOIN br ON bl.u = br.u GROUP BY bl.g ORDER BY bl.g";
    assert_eq!(parted_across_threads(&db, sql), want, "{sql}");
}

#[test]
fn a_grouped_index_join_runs_its_subqueries_over_its_parts() {
    let db = with_q(open_keyed("join_aggregate_parts_family_index"));
    let mut groups: BTreeMap<i64, i64> = BTreeMap::new();
    for (p, _) in keyed_pairs() {
        *groups.entry(p.1).or_default() += 1;
    }
    let want: Vec<String> = groups
        .iter()
        .map(|(g, n)| [*g, *n, q_with(*g)].map(int).join("|"))
        .collect();
    for (sql, path) in [
        (
            "SELECT p.g, COUNT(*), (SELECT COUNT(*) FROM q WHERE q.a = p.g) FROM c \
             JOIN p ON p.id = c.pid GROUP BY p.g ORDER BY p.g",
            "batch_index_nested_loop",
        ),
        (
            "SELECT p.g, COUNT(*), (SELECT COUNT(*) FROM q WHERE q.a = p.g) FROM p \
             JOIN c ON c.pid = p.id GROUP BY p.g ORDER BY p.g",
            C_INDEX,
        ),
    ] {
        assert_eq!(parted_by(&db, sql, path), want, "{sql}");
    }
}

#[test]
fn a_prepared_grouped_join_and_a_union_branch_run_their_subqueries_over_parts() {
    let db = with_q(open("join_aggregate_parts_family_prepared"));
    let counts = lr_counts(equal_keys, |_, _| true);
    let query = db
        .prepare(
            "SELECT l.g, COUNT(*), (SELECT COUNT(*) FROM q WHERE q.a = l.g AND q.id < $1) \
             FROM l JOIN r ON l.k = r.k GROUP BY l.g ORDER BY l.g",
        )
        .unwrap();
    for below in [5, 15] {
        #[cfg(feature = "test-failpoints")]
        {
            stoolap::test_failpoints::set_join_part_rows(PART_ROWS);
            stoolap::test_failpoints::reset_join_part_largest();
        }
        let got: Vec<String> = query
            .query((below,))
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (0..row.len())
                    .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect();
        #[cfg(feature = "test-failpoints")]
        {
            stoolap::test_failpoints::set_join_part_rows(0);
            let largest = stoolap::test_failpoints::join_part_largest();
            assert!(
                (1..=PART_ROWS).contains(&largest),
                "$1 = {below}: the aggregation held {largest} joined rows at once"
            );
        }
        let below_in = |g: i64| q_rows().filter(|q| q.1 == g && q.0 < below).count() as i64;
        let want: Vec<String> = counts
            .iter()
            .map(|(g, n)| [*g, *n, below_in(*g)].map(int).join("|"))
            .collect();
        assert_eq!(got, want, "$1 = {below}");
    }

    let sql = "SELECT l.g, (SELECT MAX(q.a) FROM q) FROM l JOIN r ON l.k = r.k GROUP BY l.g \
               UNION ALL SELECT 9, 9 ORDER BY 1";
    let max_a = q_rows().map(|q| q.1).max().unwrap();
    let want: Vec<String> = counts
        .keys()
        .map(|g| [*g, max_a].map(int).join("|"))
        .chain(std::iter::once([9, 9].map(int).join("|")))
        .collect();
    assert_eq!(parted_by(&db, sql, "hash"), want, "{sql}");
}
