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

//! A statement stops at every loop that charges its work against the
//! call's deadline, and runs to its end while the deadline is ahead

#![cfg(feature = "test-failpoints")]

use stoolap::executor::DeadlineSite;
use stoolap::test_failpoints::{
    count_deadline_charges, deadline_charges, keep_deadlines, pass_deadline_after,
    pass_deadline_at, FailpointGuard,
};
use stoolap::{Database, Error, ExecutionContext, Executor};

/// The tables every site's statements read or write
const TABLES: [&str; 12] = [
    "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, a INTEGER, b INTEGER)",
    "INSERT INTO big SELECT value, value % 50, value, value % 10, value FROM generate_series(1, 5000)",
    "CREATE INDEX big_ab ON big (a, b)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
    "INSERT INTO u SELECT value, value % 50 FROM generate_series(1, 200)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER)",
    "CREATE TABLE p (id INTEGER PRIMARY KEY)",
    "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE CASCADE)",
    "CREATE TABLE n (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE SET NULL)",
    "CREATE TABLE q (id INTEGER PRIMARY KEY, code INTEGER UNIQUE)",
    "CREATE TABLE r (id INTEGER PRIMARY KEY, code INTEGER REFERENCES q(code) ON UPDATE CASCADE)",
    "CREATE TABLE wide (id INTEGER PRIMARY KEY, k INTEGER)",
];

/// Rows of the tables a write fills, beside `TABLES`
const FILLED: [&str; 7] = [
    "INSERT INTO wide SELECT value, value FROM generate_series(1, 12000)",
    "INSERT INTO w SELECT value, value % 50 FROM generate_series(1, 3000)",
    "INSERT INTO p SELECT value FROM generate_series(1, 100)",
    "INSERT INTO c SELECT value, value % 100 + 1 FROM generate_series(1, 3000)",
    "INSERT INTO n SELECT value, value % 100 + 1 FROM generate_series(1, 3000)",
    "INSERT INTO q SELECT value, value FROM generate_series(1, 100)",
    "INSERT INTO r SELECT value, value % 100 + 1 FROM generate_series(1, 3000)",
];

/// What the tables hold, which a statement stopped part way leaves as it was
fn holding(executor: &Executor, ctx: &ExecutionContext) -> Vec<String> {
    [
        "SELECT COUNT(*), SUM(v) FROM big",
        "SELECT COUNT(*), SUM(k) FROM w",
        "SELECT COUNT(*) FROM p",
        "SELECT COUNT(*), SUM(pid) FROM c",
        "SELECT COUNT(*), COUNT(pid) FROM n",
        "SELECT COUNT(*), SUM(code) FROM q",
        "SELECT COUNT(*), SUM(code) FROM r",
    ]
    .iter()
    .map(|sql| {
        let mut rows = executor.execute_with_context(sql, ctx).unwrap();
        assert!(rows.next());
        format!("{sql}: {:?}", rows.row())
    })
    .collect()
}

/// A file of `n` rows of `w` for COPY to read
fn csv_of(n: i64) -> tempfile::NamedTempFile {
    use std::io::Write;
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut out = std::io::BufWriter::new(file.as_file());
    for id in 1..=n {
        writeln!(out, "{},{}", 10_000 + id, id % 50).unwrap();
    }
    drop(out);
    file
}

/// A statement whose run reaches `site`; `csv` is the file COPY reads
fn statement(site: DeadlineSite, csv: &str) -> String {
    let sql = match site {
        DeadlineSite::Scan => "SELECT id FROM big WHERE v * 1 > 0 ORDER BY v DESC",
        DeadlineSite::LateKeys => {
            "SELECT k, (SELECT COUNT(*) FROM u WHERE u.k = big.k) AS m FROM big \
             GROUP BY k ORDER BY (SELECT 0) + m, k"
        }
        DeadlineSite::Block => {
            "SELECT COUNT(*) FROM u WHERE EXISTS (SELECT 1 FROM big WHERE big.k = u.k AND big.v * 1 < 0)"
        }
        DeadlineSite::JoinedRows => {
            "SELECT COUNT(*) FROM u a, u b WHERE a.k + b.k > 0"
        }
        DeadlineSite::JoinParts => "SELECT COUNT(*) FROM big JOIN u ON big.k = u.k",
        DeadlineSite::Result => "SELECT id, v FROM big",
        DeadlineSite::IndexWalk => "SELECT a, b FROM big ORDER BY a, b LIMIT 10 OFFSET 1000",
        DeadlineSite::Insert => "INSERT INTO w SELECT id + 20000, k FROM big",
        DeadlineSite::Update => "UPDATE big SET v = v + 1 WHERE k >= 0",
        DeadlineSite::Delete => "DELETE FROM w RETURNING id",
        DeadlineSite::ForeignKey => "DELETE FROM p WHERE id > 0",
        DeadlineSite::Copy => {
            return format!("COPY w FROM '{csv}' WITH (FORMAT CSV, HEADER false)")
        }
        DeadlineSite::HashJoin => "SELECT big.id, u.id FROM big JOIN u ON big.k = u.k",
        DeadlineSite::NestedLoop => "SELECT a.id, b.id FROM u a JOIN u b ON a.k < b.k",
        DeadlineSite::IndexProbe => "SELECT big.id, u.k FROM big JOIN u ON u.id = big.v",
        DeadlineSite::ParallelJoin => "SELECT x.id, y.id FROM wide x JOIN wide y ON x.k = y.k",
        DeadlineSite::Sort => "SELECT id FROM big ORDER BY v * 1 DESC",
        DeadlineSite::Window => {
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY k ORDER BY v) AS n FROM big"
        }
        DeadlineSite::SetOp => "SELECT k FROM big UNION SELECT k FROM u",
        DeadlineSite::RecursiveCte => {
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5000) \
             SELECT n FROM r"
        }
        DeadlineSite::Tvf => "SELECT value FROM generate_series(1, 5000)",
        DeadlineSite::Aggregate => "SELECT k, SUM(v * 2) FROM big GROUP BY k",
        DeadlineSite::StorageDelete => "DELETE FROM w WHERE k >= 0",
        DeadlineSite::StorageUpdate => "UPDATE u SET k = k + 0 WHERE k >= 0",
    };
    sql.to_string()
}

fn read(executor: &Executor, sql: &str, ctx: &ExecutionContext) -> stoolap::Result<usize> {
    let mut rows = executor.execute_with_context(sql, ctx)?;
    let mut count = 0;
    while rows.next() {
        count += 1;
    }
    rows.last_error().map_or(Ok(count), Err)
}

#[test]
fn every_site_stops_its_statement_once_the_deadline_passes_there() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_sites").unwrap();
    for sql in TABLES.iter().chain(&FILLED) {
        db.execute(sql, ()).unwrap();
    }
    let csv = csv_of(3000);
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let mut wrong = Vec::new();
    for site in DeadlineSite::ALL {
        let sql = &statement(site, &csv.path().display().to_string());
        let held = holding(&executor, &ctx);
        pass_deadline_at(site);
        let stopped = read(&executor, sql, &ctx);
        keep_deadlines();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{site:?}: {sql} gave {stopped:?}"));
        }
        if holding(&executor, &ctx) != held {
            wrong.push(format!("{site:?}: {sql} stopped and left a change"));
        }
        if let Err(err) = read(&executor, sql, &ctx) {
            wrong.push(format!(
                "{site:?}: {sql} failed ahead of its deadline: {err}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_write_stopped_at_its_deadline_leaves_no_row_in_or_out_of_a_transaction() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_writes").unwrap();
    for sql in TABLES.iter().chain(&FILLED) {
        db.execute(sql, ()).unwrap();
    }
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let writes = [
        (
            DeadlineSite::Insert,
            "INSERT INTO w VALUES (90001, 1), (90002, 2)",
        ),
        (
            DeadlineSite::Insert,
            "INSERT INTO w VALUES (1, 7), (90003, 3) ON DUPLICATE KEY UPDATE k = 7",
        ),
        (
            DeadlineSite::Insert,
            "INSERT INTO w SELECT id + 20000, k FROM big",
        ),
        (
            DeadlineSite::Update,
            "UPDATE w SET k = k + 1 WHERE k * 1 >= 0",
        ),
        (DeadlineSite::Delete, "DELETE FROM w RETURNING id"),
        (DeadlineSite::ForeignKey, "DELETE FROM p WHERE id > 50"),
        (DeadlineSite::ForeignKey, "UPDATE q SET code = code + 1000"),
        (
            DeadlineSite::ForeignKey,
            "UPDATE c SET pid = 1 WHERE id <= 10",
        ),
    ];
    let mut wrong = Vec::new();
    for in_txn in [false, true] {
        for (site, sql) in writes {
            read(&executor, "BEGIN", &ctx).unwrap();
            let charges = charges_of(&executor, &ctx, site, sql);
            read(&executor, "ROLLBACK", &ctx).unwrap();
            if charges == 0 {
                wrong.push(format!("{sql} (in a transaction: {in_txn}) found no rows"));
                continue;
            }
            // Past its first write, and past its last with every row written
            for passed in [1, charges] {
                if in_txn {
                    read(&executor, "BEGIN", &ctx).unwrap();
                }
                let held = holding(&executor, &ctx);
                pass_deadline_after(site, passed);
                let stopped = read(&executor, sql, &ctx);
                keep_deadlines();
                let at = format!("{sql} (in a transaction: {in_txn}, {passed} of {charges})");
                if !matches!(stopped, Err(Error::QueryCancelled)) {
                    wrong.push(format!("{at} gave {stopped:?}"));
                }
                if holding(&executor, &ctx) != held {
                    wrong.push(format!("{at} left a change"));
                }
                if in_txn {
                    read(&executor, "SELECT 1", &ctx).unwrap();
                    read(&executor, "COMMIT", &ctx).unwrap();
                }
            }
        }
    }
    // A COPY past its last row, and writes by key, which charge nothing,
    // each by key run twice to reach its cached plan
    let csv = csv_of(50);
    let copy = statement(DeadlineSite::Copy, &csv.path().display().to_string());
    let (update, delete) = (
        "UPDATE w SET k = 5 WHERE id = 1",
        "DELETE FROM w WHERE id = 2",
    );
    let late = [
        (DeadlineSite::Copy, copy.as_str(), 50, false),
        (DeadlineSite::Update, update, 0, false),
        (DeadlineSite::Update, update, 0, false),
        (DeadlineSite::Update, update, 0, true),
        (DeadlineSite::Delete, delete, 0, false),
        (DeadlineSite::Delete, delete, 0, false),
        (DeadlineSite::Delete, delete, 0, true),
    ];
    for (site, sql, passed, in_txn) in late {
        if in_txn {
            read(&executor, "BEGIN", &ctx).unwrap();
        }
        let held = holding(&executor, &ctx);
        pass_deadline_after(site, passed);
        let stopped = read(&executor, sql, &ctx);
        keep_deadlines();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!(
                "{sql} (in a transaction: {in_txn}) past its end gave {stopped:?}"
            ));
        }
        if holding(&executor, &ctx) != held {
            wrong.push(format!(
                "{sql} (in a transaction: {in_txn}) past its end left a change"
            ));
        }
        if in_txn {
            read(&executor, "COMMIT", &ctx).unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The charges `sql` makes at `site`, run once
fn charges_of(executor: &Executor, ctx: &ExecutionContext, site: DeadlineSite, sql: &str) -> usize {
    count_deadline_charges(site);
    let ran = read(executor, sql, ctx);
    let charges = deadline_charges();
    keep_deadlines();
    assert!(ran.is_ok(), "{sql} gave {ran:?}");
    charges
}

#[test]
fn a_deadline_passed_late_in_a_statement_still_stops_it() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_late").unwrap();
    for sql in TABLES.iter().chain(&FILLED) {
        db.execute(sql, ()).unwrap();
    }
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let join = statement(DeadlineSite::ParallelJoin, "");
    // The charges each passes, None for all it makes: a sort's after its
    // last, a set operation's in its loops past the rows it read, a join's
    // past every row built and probed, where only its pairs are left
    let late = [
        (
            DeadlineSite::Sort,
            "SELECT id FROM big ORDER BY v * 1 DESC",
            None,
        ),
        (
            DeadlineSite::Sort,
            "SELECT id, k FROM big ORDER BY k, id",
            None,
        ),
        (
            DeadlineSite::Sort,
            "SELECT id FROM big ORDER BY v * 1 DESC LIMIT 10",
            None,
        ),
        (
            DeadlineSite::SetOp,
            "SELECT k FROM big INTERSECT ALL SELECT k FROM u",
            Some(1),
        ),
        (
            DeadlineSite::SetOp,
            "SELECT k FROM big EXCEPT ALL SELECT k FROM u",
            Some(1),
        ),
        (
            DeadlineSite::SetOp,
            "SELECT k FROM big INTERSECT SELECT k FROM u",
            Some(251),
        ),
        (DeadlineSite::ParallelJoin, join.as_str(), Some(24_000)),
    ];
    let mut wrong = Vec::new();
    for (site, sql, passed) in late {
        let passed = passed.unwrap_or_else(|| charges_of(&executor, &ctx, site, sql));
        pass_deadline_after(site, passed);
        let stopped = read(&executor, sql, &ctx);
        keep_deadlines();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{sql} past {passed} charges gave {stopped:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_storage_write_over_sealed_rows_stops_and_leaves_no_row_or_claim() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO s SELECT value, value % 50, value FROM generate_series(1, 3000)",
        (),
    )
    .unwrap();
    db.engine().seal_for_test().unwrap();
    db.execute(
        "INSERT INTO s SELECT value, value % 50, value FROM generate_series(3001, 3200)",
        (),
    )
    .unwrap();
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let held = |executor: &Executor| {
        let mut rows = executor
            .execute_with_context("SELECT COUNT(*), SUM(v) FROM s", &ctx)
            .unwrap();
        assert!(rows.next());
        format!("{:?}", rows.row())
    };
    let writes = [
        (DeadlineSite::StorageDelete, "DELETE FROM s WHERE k >= 0"),
        (
            DeadlineSite::StorageUpdate,
            "UPDATE s SET v = v + 1 WHERE k >= 0",
        ),
    ];
    let mut wrong = Vec::new();
    for in_txn in [false, true] {
        for (site, sql) in writes {
            // Inside the sealed rows, and inside the hot rows, its claims taken
            for passed in [1, 3001] {
                if in_txn {
                    read(&executor, "BEGIN", &ctx).unwrap();
                }
                let before = held(&executor);
                pass_deadline_after(site, passed);
                let stopped = read(&executor, sql, &ctx);
                keep_deadlines();
                let at = format!("{sql} (in a transaction: {in_txn}, past {passed})");
                if !matches!(stopped, Err(Error::QueryCancelled)) {
                    wrong.push(format!("{at} gave {stopped:?}"));
                }
                if held(&executor) != before {
                    wrong.push(format!("{at} left a change"));
                }
                if in_txn {
                    if let Err(err) = read(&executor, sql, &ctx) {
                        wrong.push(format!("{at} could not run again: {err}"));
                    }
                    read(&executor, "ROLLBACK", &ctx).unwrap();
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_aggregation_split_over_the_thread_pool_stops_at_its_deadline() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_split_aggregate").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value % 50, value FROM generate_series(1, 100001)",
        (),
    )
    .unwrap();
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let sql = "SELECT k, SUM(v) FROM t WHERE v * 1 > 0 GROUP BY k";
    // Past all but one row, so only a worker's charge for a row can stop it
    pass_deadline_after(DeadlineSite::Aggregate, 100_000);
    let stopped = read(&executor, sql, &ctx);
    keep_deadlines();
    assert!(
        matches!(stopped, Err(Error::QueryCancelled)),
        "{sql} gave {stopped:?}"
    );
    assert_eq!(read(&executor, sql, &ctx).unwrap(), 50);
}

#[test]
fn a_derived_source_streamed_to_the_caller_keeps_the_deadline() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_source").unwrap();
    for sql in [
        "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO big SELECT value, value % 50 FROM generate_series(1, 5000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let mut wrong = Vec::new();
    for sql in [
        "SELECT d.k FROM (SELECT k, COUNT(*) AS c FROM big GROUP BY k) d WHERE d.k * 1 >= 0",
        "SELECT d.k FROM (SELECT k, COUNT(*) AS c FROM big GROUP BY k) d",
    ] {
        pass_deadline_at(DeadlineSite::Result);
        let stopped = read(&executor, sql, &ctx);
        keep_deadlines();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{sql} gave {stopped:?}"));
        }
        if let Err(err) = read(&executor, sql, &ctx) {
            wrong.push(format!("{sql} failed ahead of its deadline: {err}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_key_read_in_a_transaction_that_outlasts_its_deadline_fails() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_key_read").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    stoolap::test_failpoints::after_pk_lookup_resolved(stoolap::test_failpoints::pass_deadlines);
    let read = db
        .query_with_timeout("SELECT v FROM t WHERE id = $1", (1,), 3_600_000)
        .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>());
    keep_deadlines();
    assert!(
        matches!(read, Err(Error::QueryCancelled)),
        "a key read past its deadline gave {:?}",
        read.map(|rows| rows.len())
    );
    db.execute("ROLLBACK", ()).unwrap();
}
