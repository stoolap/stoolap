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

//! Every read path a statement notes either stops at the call's deadline or
//! is named here with the reason it needs no charge

#![cfg(feature = "test-failpoints")]

use std::fs;
use std::path::Path;

use stoolap::executor::DeadlineSite;
use stoolap::test_failpoints::{
    keep_deadlines, pass_deadline_after, take_read_paths, FailpointGuard,
};
use stoolap::{Database, Error, ExecutionContext, Executor};

const SETUP: [&str; 6] = [
    "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, s TEXT, u INTEGER UNIQUE)",
    "CREATE INDEX t_k ON t (k)",
    "CREATE INDEX t_s ON t (s)",
    "INSERT INTO t SELECT value, value % 50, value, 'x' || (value % 7), value \
     FROM generate_series(1, 5000)",
    "CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)",
    "INSERT INTO o SELECT value, value FROM generate_series(1, 100)",
];

#[derive(Clone, Copy, PartialEq)]
enum Store {
    Hot,
    Sealed,
}

enum Case {
    /// The statement reaches the path, and a deadline passed after the
    /// site's first charge, or the charges `CHARGED_AHEAD` names, stops it
    /// there
    Stops(Store, bool, &'static str, DeadlineSite),
    /// The path reads a bounded number of rows or versions
    Bounded(&'static str),
    /// Left to the follow-up issue on writes, DDL and maintenance
    FollowUp(&'static str),
    /// No statement reaches the path
    Unreached(&'static str),
}

use Case::*;
use DeadlineSite::{Aggregate, StorageAggregate, StorageDelete, StorageRead, StorageUpdate};
use Store::*;

const CASES: &[(&str, Case)] = &[
    (
        "exec.index_get",
        Bounded("the latest committed rows of one key, at most 256"),
    ),
    (
        "exec.pk_lookup_in_transaction",
        Stops(Hot, true, "SELECT * FROM t WHERE id = 7", StorageRead),
    ),
    ("exec.unique_get", Bounded("one latest committed row")),
    (
        "registry.check_committed.locked",
        Bounded("one transaction's status"),
    ),
    (
        "seg.collect_cold_rows",
        Stops(
            Sealed,
            false,
            "SELECT * FROM t AS OF TIMESTAMP '2099-01-01 00:00:00'",
            StorageRead,
        ),
    ),
    (
        "seg.collect_rows_by_ids",
        FollowUp("an HNSW search's k rows and a foreign key cascade's rows"),
    ),
    (
        "seg.collect_rows_with_limit",
        Unreached("the ordered LIMIT read of a table holding volumes"),
    ),
    (
        "seg.collect_rows_with_limit_unordered",
        Stops(Sealed, false, "SELECT * FROM t LIMIT 10", StorageRead),
    ),
    (
        "seg.compute_filtered_aggregates",
        Stops(
            Sealed,
            false,
            "SELECT COUNT(*) FROM t WHERE v > 10",
            StorageAggregate,
        ),
    ),
    (
        "seg.compute_grouped_aggregates",
        Stops(
            Sealed,
            false,
            "SELECT k, SUM(v) FROM t GROUP BY k",
            StorageAggregate,
        ),
    ),
    (
        "seg.count_cold_and_hot",
        Stops(Sealed, true, "SELECT COUNT(*) FROM t", StorageRead),
    ),
    (
        "seg.distinct_column",
        Stops(Sealed, false, "SELECT DISTINCT k FROM t", StorageRead),
    ),
    (
        "seg.fetch_rows_by_ids_into",
        Stops(
            Sealed,
            false,
            "SELECT * FROM t WHERE id IN (1, 2, 3, 4)",
            StorageRead,
        ),
    ),
    (
        "seg.max_column",
        Stops(Sealed, false, "SELECT MAX(v) FROM t", StorageAggregate),
    ),
    (
        "seg.min_column",
        Stops(Sealed, false, "SELECT MIN(v) FROM t", StorageAggregate),
    ),
    (
        "seg.rows_at",
        Stops(
            Sealed,
            false,
            "SELECT * FROM t AS OF TIMESTAMP '2099-01-01 00:00:00'",
            StorageRead,
        ),
    ),
    (
        "seg.scan_cold_and_hot",
        Stops(Sealed, false, "SELECT * FROM t", StorageRead),
    ),
    (
        "seg.scan_top_k",
        Stops(
            Sealed,
            false,
            "SELECT * FROM t ORDER BY v LIMIT 10",
            StorageRead,
        ),
    ),
    (
        "seg.sum_column",
        Stops(Sealed, false, "SELECT SUM(v) FROM t", StorageAggregate),
    ),
    (
        "tbl.walk_btree_groups",
        Stops(
            Hot,
            false,
            "SELECT k, COUNT(*) FROM t GROUP BY k HAVING COUNT(*) > 0",
            Aggregate,
        ),
    ),
    (
        "vs.collect_rows_keyset",
        Stops(
            Hot,
            false,
            "SELECT * FROM t WHERE id > 100 ORDER BY id LIMIT 10",
            StorageRead,
        ),
    ),
    (
        "vs.collect_rows_pk_ordered",
        Stops(
            Hot,
            false,
            "SELECT * FROM t ORDER BY id LIMIT 10 OFFSET 4500",
            StorageRead,
        ),
    ),
    (
        "vs.compute_grouped_aggregates",
        Stops(
            Hot,
            false,
            "SELECT k, SUM(v) FROM t GROUP BY k",
            StorageAggregate,
        ),
    ),
    (
        "vs.for_each_visible",
        Stops(
            Hot,
            false,
            "SELECT * FROM t WHERE k > 3 LIMIT 10",
            StorageRead,
        ),
    ),
    (
        "vs.for_each_visible_filtered",
        Stops(
            Sealed,
            false,
            "SELECT * FROM t WHERE k = 3 OR v > 7 LIMIT 10 OFFSET 4000",
            StorageRead,
        ),
    ),
    (
        "vs.get_all_visible_rows_cached",
        Stops(Hot, true, "SELECT * FROM t", StorageRead),
    ),
    (
        "vs.get_all_visible_rows_filtered",
        Stops(
            Hot,
            false,
            "SELECT COUNT(*) FROM t WHERE v > 10",
            StorageRead,
        ),
    ),
    (
        "vs.get_all_visible_rows_for_update",
        Stops(Hot, false, "UPDATE t SET v = v", StorageUpdate),
    ),
    (
        "vs.get_all_visible_rows_for_update_filtered",
        Stops(Hot, false, "UPDATE t SET v = v WHERE v < 0", StorageUpdate),
    ),
    (
        "vs.get_visible_rows_filtered_with_limit",
        Stops(
            Hot,
            false,
            "SELECT * FROM t WHERE v > 10 LIMIT 10",
            StorageRead,
        ),
    ),
    (
        "vs.get_visible_rows_with_limit",
        Stops(Hot, false, "SELECT * FROM t LIMIT 10", StorageRead),
    ),
    (
        "vs.get_visible_version.arena_probe",
        Stops(Hot, false, "DELETE FROM t WHERE v < 0", StorageDelete),
    ),
    (
        "vs.get_visible_version.chain",
        Stops(Sealed, false, "DELETE FROM t WHERE v < 0", StorageDelete),
    ),
    (
        "vs.get_visible_versions_batch",
        Bounded("the index get's latest committed rows, at most 256"),
    ),
    (
        "vs.get_visible_versions_batch.arena_probe",
        Bounded("the index get's latest committed rows, at most 256"),
    ),
    (
        "vs.get_visible_versions_for_update",
        Stops(Hot, false, "UPDATE t SET v = v WHERE k = 3", StorageUpdate),
    ),
    (
        "vs.max_column",
        Stops(Hot, false, "SELECT MAX(v) FROM t", StorageAggregate),
    ),
    (
        "vs.min_column",
        Stops(Hot, false, "SELECT MIN(v) FROM t", StorageAggregate),
    ),
    (
        "vs.sum_column",
        Stops(Hot, false, "SELECT SUM(v) FROM t", StorageAggregate),
    ),
];

/// Paths whose statement charges the site once per id it finds before the
/// path reads them: the 100 ids `k = 3` names
const CHARGED_AHEAD: &[(&str, usize)] = &[("vs.get_visible_versions_for_update", 100)];

/// Every literal path name the library notes a read under
fn noted_paths() -> Vec<String> {
    let mut names = Vec::new();
    let mut dirs = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }
            let source = fs::read_to_string(&path).unwrap();
            for (at, _) in source.match_indices("note_read(") {
                let rest = source[at + "note_read(".len()..].trim_start();
                if let Some(quoted) = rest.strip_prefix('"') {
                    names.push(quoted[..quoted.find('"').unwrap()].to_string());
                }
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

fn read(executor: &Executor, sql: &str, ctx: &ExecutionContext) -> stoolap::Result<usize> {
    let mut rows = executor.execute_with_context(sql, ctx)?;
    let mut count = 0;
    while rows.next() {
        count += 1;
    }
    rows.last_error().map_or(Ok(count), Err)
}

/// The statement's result and the paths it noted, on a fresh executor so no
/// cache answers it
fn run(
    db: &Database,
    in_txn: bool,
    sql: &str,
    ctx: &ExecutionContext,
) -> (stoolap::Result<usize>, Vec<&'static str>) {
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    if in_txn {
        read(&executor, "BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ctx).unwrap();
    }
    take_read_paths();
    let result = read(&executor, sql, ctx);
    let paths = take_read_paths();
    if in_txn {
        read(&executor, "ROLLBACK", ctx).unwrap();
    }
    (result, paths)
}

#[test]
fn every_noted_read_path_stops_at_the_deadline_or_says_why_not() {
    let _guard = FailpointGuard::new();
    let hot = Database::open("memory://read_path_deadline_hot").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let sealed = Database::open(&dsn).unwrap();
    for db in [&hot, &sealed] {
        for sql in SETUP {
            db.execute(sql, ()).unwrap();
        }
    }
    sealed.engine().seal_for_test().unwrap();
    sealed
        .execute("UPDATE t SET v = v + 1 WHERE id <= 10", ())
        .unwrap();
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);

    let mut wrong = Vec::new();
    let named: Vec<&str> = CASES.iter().map(|(name, _)| *name).collect();
    for name in noted_paths() {
        if !named.contains(&name.as_str()) {
            wrong.push(format!("{name}: noted in src with no case here"));
        }
    }
    let noted = noted_paths();
    for (name, case) in CASES {
        if !noted.iter().any(|noted| noted == name) {
            wrong.push(format!("{name}: a case for a path src no longer notes"));
        }
        let (store, in_txn, sql, site) = match case {
            Stops(store, in_txn, sql, site) => (store, in_txn, sql, site),
            Bounded(why) | FollowUp(why) | Unreached(why) => {
                if why.is_empty() {
                    wrong.push(format!("{name}: no reason given"));
                }
                continue;
            }
        };
        // Only a memory table walks its index; under test-filedb every table
        // can hold volumes
        if cfg!(feature = "test-filedb") && *name == "tbl.walk_btree_groups" {
            continue;
        }
        let db = if *store == Hot { &hot } else { &sealed };
        let (result, paths) = run(db, *in_txn, sql, &ctx);
        if result.is_err() || !paths.contains(name) {
            wrong.push(format!("{name}: {sql} gave {result:?} through {paths:?}"));
        }
        let ahead = CHARGED_AHEAD.iter().find(|(path, _)| path == name);
        pass_deadline_after(*site, ahead.map_or(1, |&(_, charges)| charges));
        let (result, paths) = run(db, *in_txn, sql, &ctx);
        keep_deadlines();
        if !matches!(result, Err(Error::QueryCancelled)) || !paths.contains(name) {
            wrong.push(format!(
                "{name}: {sql} past its deadline at {site:?} gave {result:?} through {paths:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
