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

mod cancel {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use stoolap::core::{Result, Value};
    use stoolap::functions::registry::global_registry;
    use stoolap::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
    };
    use stoolap::{Error, ExecutionContext, Executor};

    static TO_CANCEL: Mutex<Option<ExecutionContext>> = Mutex::new(None);
    static CALLS: AtomicUsize = AtomicUsize::new(0);

    /// Returns its argument, counts its calls and cancels `TO_CANCEL`
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
            if let Some(ctx) = TO_CANCEL.lock().unwrap().as_ref() {
                ctx.cancel();
            }
            Ok(args[0].clone())
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(JoinCancelNow)
        }
    }

    #[test]
    fn a_cancel_stops_a_join_read_in_parts_within_a_hundred_rows() {
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
        ] {
            let executor = Executor::new(Arc::clone(db.engine()));
            let ctx = ExecutionContext::new();
            *TO_CANCEL.lock().unwrap() = Some(ctx.clone());
            let before = CALLS.load(Ordering::Relaxed);
            let read = executor
                .execute_with_context(sql, &ctx)
                .and_then(|mut rows| {
                    while rows.next() {}
                    rows.last_error().map_or(Ok(()), Err)
                });
            *TO_CANCEL.lock().unwrap() = None;
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
    use stoolap::executor::{parallel_hash_join, ParallelConfig, ParallelJoinType};

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
                parallel_hash_join(probe, build, &[0], &[0], join_type, 1, 1, false, &config);
            }
        }
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global()
        .expect("a hash join on one thread built the global thread pool");
}
