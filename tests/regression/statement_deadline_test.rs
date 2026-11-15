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
        DeadlineSite::StorageRead => "SELECT id, ROW_NUMBER() OVER (ORDER BY v) AS n FROM big",
        DeadlineSite::StorageAggregate => "SELECT k, SUM(v) FROM big GROUP BY k",
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
    // Reads of the sealed rows, each stopped inside them: a read charges its
    // hot rows once before them, which the deadline lets pass; an aggregate
    // charges only its sealed rows
    let reads = [
        (
            DeadlineSite::StorageRead,
            "SELECT id, ROW_NUMBER() OVER (ORDER BY v) AS n FROM s",
            Some(1),
        ),
        (
            DeadlineSite::StorageAggregate,
            "SELECT k, SUM(v) FROM s GROUP BY k",
            None,
        ),
        (
            DeadlineSite::StorageAggregate,
            "SELECT SUM(v) FROM s WHERE k >= 0",
            None,
        ),
    ];
    for (site, sql, passed) in reads {
        match passed {
            Some(passed) => pass_deadline_after(site, passed),
            None => pass_deadline_at(site),
        }
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
fn a_storage_read_or_write_past_its_first_charges_still_stops() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_storage_late").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO t SELECT value, value % 3, value FROM generate_series(1, 3000)",
        "CREATE INDEX t_k ON t (k)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    // The charges each passes, None for every one it makes: a write by
    // index past its first candidates, a read past its last row
    let late = [
        (
            DeadlineSite::StorageDelete,
            "DELETE FROM t WHERE k = 1",
            Some(2),
        ),
        (
            DeadlineSite::StorageUpdate,
            "UPDATE t SET v = v + 1 WHERE k = 1",
            Some(2),
        ),
        (
            DeadlineSite::StorageRead,
            "SELECT * FROM t WHERE v >= (SELECT 0) AND ABS(v) < 0 LIMIT 1",
            None,
        ),
        (
            DeadlineSite::StorageRead,
            "SELECT COUNT(*) FROM t WHERE v >= (SELECT 0) AND ABS(v) < 0",
            None,
        ),
    ];
    let mut wrong = Vec::new();
    for (site, sql, passed) in late {
        read(&executor, "BEGIN", &ctx).unwrap();
        let passed = passed.unwrap_or_else(|| charges_of(&executor, &ctx, site, sql));
        pass_deadline_after(site, passed);
        let stopped = read(&executor, sql, &ctx);
        keep_deadlines();
        read(&executor, "ROLLBACK", &ctx).unwrap();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{sql} past {passed} charges gave {stopped:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_scan_charges_the_rows_its_pushed_filter_rejects() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_pushed_filter").unwrap();
    for sql in [
        "CREATE TABLE h (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO h SELECT value, value % 3, value FROM generate_series(1, 20000)",
        "CREATE INDEX h_k ON h (k)",
        "CREATE TABLE e (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO e SELECT value, value FROM generate_series(1, 300)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let sealed = Database::open(&dsn).unwrap();
    sealed
        .execute(
            "CREATE TABLE s (id INTEGER PRIMARY KEY, v INTEGER, t TEXT)",
            (),
        )
        .unwrap();
    sealed
        .execute(
            "INSERT INTO s SELECT value, (value * 7919) % 140000, 'x' || (value % 10) \
             FROM generate_series(1, 140000)",
            (),
        )
        .unwrap();
    sealed.engine().seal_for_test().unwrap();
    let hot = Executor::new(std::sync::Arc::clone(db.engine()));
    let cold = Executor::new(std::sync::Arc::clone(sealed.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    // Each passes one charge more than the read makes without the scan's own
    let shapes = [
        (&hot, false, "SELECT id FROM h WHERE v = -1", 1),
        (&hot, true, "SELECT id FROM h WHERE v = -1", 2),
        (&hot, false, "SELECT * FROM h WHERE v = -1 ORDER BY v", 2),
        (&hot, false, "SELECT id FROM h WHERE k = 1 AND v < 0", 1),
        (
            &hot,
            false,
            "SELECT COUNT(*) FROM e a WHERE EXISTS (SELECT 1 FROM e b WHERE b.v = a.v + 1000)",
            1,
        ),
        (&cold, false, "SELECT id FROM s WHERE v = 70000", 2),
        (&cold, false, "SELECT id FROM s WHERE t = 'x7'", 5),
    ];
    let mut wrong = Vec::new();
    for (executor, own_rows, sql, passed) in shapes {
        if own_rows {
            read(executor, "BEGIN", &ctx).unwrap();
            read(executor, "INSERT INTO h VALUES (900000, 1, 5)", &ctx).unwrap();
        }
        pass_deadline_after(DeadlineSite::StorageRead, passed);
        let stopped = read(executor, sql, &ctx);
        keep_deadlines();
        if own_rows {
            read(executor, "ROLLBACK", &ctx).unwrap();
        }
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{sql} past {passed} charges gave {stopped:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_storage_loop_over_filtered_candidates_or_unseen_rows_still_stops() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_filtered_candidates").unwrap();
    for sql in [
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "CREATE INDEX w_k ON w (k)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO r SELECT value, value % 2, value FROM generate_series(1, 20000)",
        "CREATE INDEX r_k ON r (k)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    // Each passes one charge more than the statement makes without the
    // charges of its filter's loop
    let shapes = [
        (
            DeadlineSite::StorageUpdate,
            true,
            "UPDATE w SET v = v + 1 WHERE ABS(v) < 0",
            10_001,
        ),
        (
            DeadlineSite::StorageUpdate,
            true,
            "UPDATE w SET v = v + 1 WHERE k = 1 AND ABS(v) < 0",
            5_001,
        ),
        (
            DeadlineSite::StorageDelete,
            true,
            "DELETE FROM w WHERE k = 1 AND ABS(v) < 0",
            3,
        ),
        (
            DeadlineSite::StorageRead,
            false,
            "SELECT * FROM r WHERE k = 1 AND ABS(v) < 0",
            2,
        ),
    ];
    let mut wrong = Vec::new();
    for (site, own_rows, sql, passed) in shapes {
        read(&executor, "BEGIN", &ctx).unwrap();
        if own_rows {
            let insert =
                "INSERT INTO w SELECT value, value % 2, value FROM generate_series(1, 10000)";
            read(&executor, insert, &ctx).unwrap();
        }
        pass_deadline_after(site, passed);
        let stopped = read(&executor, sql, &ctx);
        keep_deadlines();
        read(&executor, "ROLLBACK", &ctx).unwrap();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{sql} past {passed} charges gave {stopped:?}"));
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let sealed = Database::open(&dsn).unwrap();
    sealed
        .execute(
            "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
            (),
        )
        .unwrap();
    sealed
        .execute(
            "INSERT INTO s SELECT value, value % 50, value FROM generate_series(1, 140000)",
            (),
        )
        .unwrap();
    sealed.engine().seal_for_test().unwrap();
    sealed.execute("DELETE FROM s WHERE id >= 0", ()).unwrap();
    let cold = Executor::new(std::sync::Arc::clone(sealed.engine()));
    let sql = "SELECT k, SUM(v) FROM s GROUP BY k";
    pass_deadline_after(DeadlineSite::StorageAggregate, 1);
    let stopped = read(&cold, sql, &ctx);
    keep_deadlines();
    if !matches!(stopped, Err(Error::QueryCancelled)) {
        wrong.push(format!("{sql} over deleted sealed rows gave {stopped:?}"));
    }

    // Index candidates committed after the snapshot: the store reads them
    // and sees none, so only the charge before each run of ids can stop it
    db.execute(
        "CREATE TABLE q (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX q_k ON q (k)", ()).unwrap();
    read(
        &executor,
        "BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT",
        &ctx,
    )
    .unwrap();
    read(&executor, "SELECT COUNT(*) FROM q", &ctx).unwrap();
    db.execute(
        "INSERT INTO q SELECT value, 1, value FROM generate_series(1, 10000)",
        (),
    )
    .unwrap();
    let sql = "SELECT * FROM q WHERE k = 1 AND ABS(v) < 0";
    pass_deadline_after(DeadlineSite::StorageRead, 1);
    let stopped = read(&executor, sql, &ctx);
    keep_deadlines();
    read(&executor, "ROLLBACK", &ctx).unwrap();
    if !matches!(stopped, Err(Error::QueryCancelled)) {
        wrong.push(format!(
            "{sql} over rows the snapshot misses gave {stopped:?}"
        ));
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

/// What a shape runs before its statement, inside the transaction it reads in
#[derive(Clone, Copy)]
enum Before {
    Nothing,
    OwnRows,
    UnseenRows,
}

#[test]
fn a_table_read_that_took_no_deadline_stops_at_it() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_table_reads").unwrap();
    for sql in [
        "CREATE TABLE h (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, g INTEGER)",
        "INSERT INTO h SELECT value, value % 3, value, value FROM generate_series(1, 20000)",
        "CREATE INDEX h_g ON h (g)",
        "CREATE INDEX h_kg ON h (k, g)",
        "CREATE TABLE q (id INTEGER PRIMARY KEY, v INTEGER)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let sealed = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, t TEXT)",
        "CREATE INDEX s_t ON s (t)",
        "INSERT INTO s SELECT value, value % 50, (value * 7919) % 140000, 'x' || (value % 10) \
         FROM generate_series(1, 140000)",
    ] {
        sealed.execute(sql, ()).unwrap();
    }
    sealed.engine().seal_for_test().unwrap();
    for sql in [
        "DELETE FROM s WHERE id <= 10",
        "INSERT INTO s SELECT value, value % 50, value, 'y' FROM generate_series(140001, 140100)",
    ] {
        sealed.execute(sql, ()).unwrap();
    }
    let hot = Executor::new(std::sync::Arc::clone(db.engine()));
    let cold = Executor::new(std::sync::Arc::clone(sealed.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let rows = DeadlineSite::StorageRead;
    let fold = DeadlineSite::StorageAggregate;
    let as_of = "AS OF TIMESTAMP '2200-01-01 00:00:00'";
    let hot_as_of = format!("SELECT id FROM h {as_of} WHERE v < 0");
    let cold_as_of = format!("SELECT id FROM s {as_of} WHERE v < 0");
    let hot_as_of_sum = format!("SELECT SUM(v) FROM h {as_of}");
    let cold_as_of_sum = format!("SELECT SUM(v) FROM s {as_of}");
    // None charged its site before its read took the deadline
    let shapes = [
        (
            &hot,
            Before::Nothing,
            rows,
            "SELECT id FROM h WHERE k = 1 AND v < 0 ORDER BY g LIMIT 1",
        ),
        (
            &cold,
            Before::Nothing,
            rows,
            "SELECT id FROM s WHERE v = -1 ORDER BY id DESC LIMIT 1",
        ),
        (
            &hot,
            Before::UnseenRows,
            rows,
            "SELECT * FROM q WHERE id > 0 ORDER BY id LIMIT 1",
        ),
        (
            &hot,
            Before::Nothing,
            rows,
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) FROM h LIMIT 1",
        ),
        (
            &cold,
            Before::Nothing,
            rows,
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY t ORDER BY id) FROM s LIMIT 1",
        ),
        (
            &cold,
            Before::Nothing,
            rows,
            "SELECT COUNT(DISTINCT t) FROM s",
        ),
        (&cold, Before::Nothing, rows, "SELECT DISTINCT t FROM s"),
        (&hot, Before::Nothing, rows, "SELECT g FROM h GROUP BY g"),
        (&hot, Before::Nothing, fold, "SELECT SUM(v) FROM h"),
        (&hot, Before::Nothing, fold, "SELECT MIN(v) FROM h"),
        (&hot, Before::Nothing, fold, "SELECT MAX(v) FROM h"),
        (&cold, Before::Nothing, fold, "SELECT SUM(v) FROM s"),
        (&cold, Before::Nothing, fold, "SELECT MIN(v) FROM s"),
        (&cold, Before::Nothing, fold, "SELECT MAX(v) FROM s"),
        (&hot, Before::Nothing, rows, hot_as_of.as_str()),
        (&cold, Before::Nothing, rows, cold_as_of.as_str()),
        (&hot, Before::Nothing, rows, hot_as_of_sum.as_str()),
        (&cold, Before::Nothing, rows, cold_as_of_sum.as_str()),
        (&hot, Before::OwnRows, rows, "SELECT COUNT(*) FROM h"),
        (&cold, Before::UnseenRows, rows, "SELECT COUNT(*) FROM s"),
        (
            &cold,
            Before::Nothing,
            DeadlineSite::StorageDelete,
            "DELETE FROM s WHERE id IN (SELECT 20 UNION SELECT 30)",
        ),
    ];
    let mut wrong = Vec::new();
    // A sealed top-k walked newest first, past the charges of its hot rows
    let sql = "SELECT id FROM s WHERE k = 7 AND v > 139990 AND v < 140000 ORDER BY id DESC LIMIT 1";
    pass_deadline_after(rows, 3);
    let stopped = read(&cold, sql, &ctx);
    keep_deadlines();
    if !matches!(stopped, Err(Error::QueryCancelled)) {
        wrong.push(format!("{sql} past 3 charges gave {stopped:?}"));
    }
    for (executor, before, site, sql) in shapes {
        match before {
            Before::Nothing => {
                read(executor, "BEGIN", &ctx).unwrap();
            }
            Before::OwnRows => {
                read(executor, "BEGIN", &ctx).unwrap();
                read(executor, "INSERT INTO h VALUES (900000, 1, 5, 5)", &ctx).unwrap();
            }
            // Rows another handle commits after the snapshot began
            Before::UnseenRows => {
                let hot_side = std::ptr::eq(executor, &hot);
                let (table, other) = if hot_side { ("q", &db) } else { ("s", &sealed) };
                read(executor, "BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", &ctx).unwrap();
                let begun = format!("SELECT COUNT(*) FROM {table} WHERE id = 1");
                read(executor, &begun, &ctx).unwrap();
                let insert = format!(
                    "INSERT INTO {table} (id, v) SELECT value + 200000, 1 \
                     FROM generate_series(1, 10000)"
                );
                other.execute(&insert, ()).unwrap();
            }
        }
        pass_deadline_at(site);
        let stopped = read(executor, sql, &ctx);
        keep_deadlines();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{sql} gave {stopped:?}"));
        }
        // A read is checked once more after its last charge
        if site != DeadlineSite::StorageDelete {
            count_deadline_charges(site);
            let counted = read(executor, sql, &ctx);
            let charges = deadline_charges();
            pass_deadline_after(site, charges);
            let ended = read(executor, sql, &ctx);
            keep_deadlines();
            if counted.is_err() || charges == 0 || !matches!(ended, Err(Error::QueryCancelled)) {
                wrong.push(format!(
                    "{sql} counted {counted:?} in {charges} charges, past them gave {ended:?}"
                ));
            }
        }
        read(executor, "ROLLBACK", &ctx).unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_long_table_read_charges_each_run_of_its_work() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_long_reads").unwrap();
    for sql in [
        "CREATE TABLE h (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, b INTEGER)",
        "INSERT INTO h SELECT value, value, value, value FROM generate_series(1, 20000)",
        "CREATE INDEX h_g ON h (g)",
        "CREATE INDEX h_x ON h (x) USING HASH",
        "CREATE INDEX h_b ON h (b) USING BITMAP",
        "CREATE TABLE l (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO l VALUES (1, 0)",
        "CREATE TABLE far (id INTEGER PRIMARY KEY)",
        "INSERT INTO far SELECT value + 10000000 FROM generate_series(1, 20000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let sealed = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, n INTEGER, t TEXT, p TEXT)",
        "CREATE INDEX s_n ON s (n)",
        "CREATE INDEX s_t ON s (t)",
        "INSERT INTO s SELECT value, value, 't' || (value % 30000), 'p' || value \
         FROM generate_series(1, 140000)",
    ] {
        sealed.execute(sql, ()).unwrap();
    }
    sealed.engine().seal_for_test().unwrap();
    let hot = Executor::new(std::sync::Arc::clone(db.engine()));
    let cold = Executor::new(std::sync::Arc::clone(sealed.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let rows = DeadlineSite::StorageRead;
    // A snapshot older than 5,000 versions of the one row of l
    read(&hot, "BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", &ctx).unwrap();
    read(&hot, "SELECT v FROM l", &ctx).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(3));
    let before = db
        .query("SELECT CAST(NOW() AS TEXT)", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<String>(0)
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(3));
    for _ in 0..5000 {
        db.execute("UPDATE l SET v = v + 1", ()).unwrap();
    }
    let as_of = format!("SELECT v FROM l AS OF TIMESTAMP '{before}'");
    let mut wrong = Vec::new();
    let mut charged = |executor: &Executor, site: DeadlineSite, sql: &str, least: usize| {
        count_deadline_charges(site);
        let counted = read(executor, sql, &ctx);
        let charges = deadline_charges();
        keep_deadlines();
        if counted.is_err() || charges < least {
            wrong.push(format!(
                "{sql} gave {counted:?} in {charges} charges, under {least}"
            ));
        }
    };
    // Each run of 4,096 units of work is charged: at least work / 4,096 charges
    charged(&hot, rows, "SELECT COUNT(*) FROM l", 1);
    // A file table under a snapshot leaves SUM to a read of its rows
    #[cfg(not(feature = "test-filedb"))]
    charged(
        &hot,
        DeadlineSite::StorageAggregate,
        "SELECT SUM(v) FROM l",
        1,
    );
    // A run of versions, then the last part of one
    charged(
        &hot,
        rows,
        "SELECT * FROM l WHERE id > 0 ORDER BY id LIMIT 1",
        2,
    );
    read(&hot, "ROLLBACK", &ctx).unwrap();
    charged(&hot, rows, &as_of, 1);
    charged(&hot, rows, "SELECT DISTINCT g FROM h", 4);
    charged(&hot, rows, "SELECT DISTINCT x FROM h", 4);
    charged(&hot, rows, "SELECT DISTINCT b FROM h", 4);
    charged(&hot, rows, "SELECT DISTINCT id FROM h", 4);
    // Keys past the primary key's bitset
    charged(&hot, rows, "SELECT DISTINCT id FROM far", 4);
    charged(&cold, rows, "SELECT DISTINCT t FROM s", 7);
    charged(&cold, rows, "SELECT DISTINCT n FROM s", 34);
    charged(
        &cold,
        rows,
        "SELECT p FROM s WHERE n >= 0 ORDER BY n LIMIT 20000",
        20000,
    );
    // A deleted row is hidden by its tombstone and by its hot delete: 120,000
    // hidden ids located one by one, then 80,000 shown positions read
    sealed
        .execute("DELETE FROM s WHERE id <= 60000", ())
        .unwrap();
    charged(&cold, rows, "SELECT DISTINCT n FROM s", 34);
    charged(&cold, rows, "SELECT DISTINCT t FROM s", 29 + 19);
    // More hidden ids than rows: each of the 140,000 row ids is looked up
    sealed
        .execute("DELETE FROM s WHERE id <= 100000", ())
        .unwrap();
    charged(&cold, rows, "SELECT DISTINCT t FROM s", 34 + 9);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_read_that_keeps_few_of_the_rows_it_reads_charges_each_run_of_them() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_few_kept").unwrap();
    for sql in [
        "CREATE TABLE h (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, half INTEGER)",
        "INSERT INTO h SELECT value, value, value, value % 2 FROM generate_series(1, 20000)",
        "CREATE INDEX h_g ON h (g)",
        "CREATE INDEX h_half ON h (half)",
        "CREATE TABLE o (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO o SELECT value, value FROM generate_series(1, 500)",
        "CREATE TABLE one (id INTEGER PRIMARY KEY)",
        "INSERT INTO one VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let sealed = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, half INTEGER)",
        "CREATE INDEX s_half ON s (half)",
        "INSERT INTO s SELECT value, value % 2 FROM generate_series(1, 140000)",
        "CREATE TABLE one (id INTEGER PRIMARY KEY)",
        "INSERT INTO one VALUES (1)",
    ] {
        sealed.execute(sql, ()).unwrap();
    }
    sealed.engine().seal_for_test().unwrap();
    let hot = Executor::new(std::sync::Arc::clone(db.engine()));
    let cold = Executor::new(std::sync::Arc::clone(sealed.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let mut wrong = Vec::new();
    let mut charged = |executor: &Executor, sql: &str, least: usize| {
        count_deadline_charges(DeadlineSite::StorageRead);
        let counted = read(executor, sql, &ctx);
        let charges = deadline_charges();
        keep_deadlines();
        if counted.is_err() || charges < least {
            wrong.push(format!(
                "{sql} gave {counted:?} in {charges} charges, under {least}"
            ));
        }
    };
    // Each run of 4,096 rows read is charged, kept or not
    charged(&hot, "SELECT id FROM h WHERE x = -1 LIMIT 1", 4);
    charged(&hot, "SELECT id FROM h LIMIT 1 OFFSET 19990", 4);
    charged(&hot, "SELECT id FROM h WHERE g > 0 AND x = -1 LIMIT 1", 4);
    charged(&hot, "SELECT id FROM h WHERE g < 3 OR x = -1 LIMIT 5", 4);
    charged(
        &hot,
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY half ORDER BY id) FROM h LIMIT 1",
        2,
    );
    // The keys NOT IN leaves, then their rows
    charged(
        &hot,
        "SELECT id FROM h WHERE id NOT IN (SELECT id FROM one)",
        8,
    );
    // The inner rows each outer key, or each batch of keys, names
    charged(&hot, "SELECT o.id, h.x FROM o JOIN h ON o.g = h.g", 100);
    charged(
        &cold,
        "SELECT id FROM s WHERE id NOT IN (SELECT id FROM one) LIMIT 1",
        34,
    );
    charged(
        &cold,
        "SELECT id FROM s WHERE id NOT IN (SELECT id FROM one)",
        68,
    );
    // The first partition's rows are looked for among every sealed row
    charged(
        &cold,
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY half ORDER BY id) FROM s LIMIT 1",
        34,
    );
    // A transaction's own row leaves a LIMIT to a read of every row
    read(&hot, "BEGIN", &ctx).unwrap();
    read(&hot, "INSERT INTO h VALUES (900000, 0, 0, 0)", &ctx).unwrap();
    charged(&hot, "SELECT id FROM h WHERE x = -1 LIMIT 1", 4);
    read(&hot, "ROLLBACK", &ctx).unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_read_past_old_versions_own_rows_or_hidden_rows_charges_them() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://statement_deadline_unseen_work").unwrap();
    for sql in [
        "CREATE TABLE i (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)",
        "INSERT INTO i SELECT value, CASE WHEN value <= 10000 THEN 1 ELSE value END, value \
         FROM generate_series(1, 50000)",
        "CREATE INDEX i_g ON i (g)",
        "CREATE TABLE o (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO o VALUES (1, 1)",
        "CREATE TABLE e (id INTEGER PRIMARY KEY, x INTEGER)",
        "CREATE TABLE l (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)",
        "CREATE INDEX l_g ON l (g)",
        "INSERT INTO l VALUES (2, 1, 0)",
        "CREATE TABLE one (id INTEGER PRIMARY KEY)",
        "INSERT INTO one VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    );
    let sealed = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, half INTEGER)",
        "CREATE INDEX p_half ON p (half)",
        "INSERT INTO p SELECT value, value % 2 FROM generate_series(1, 140000)",
    ] {
        sealed.execute(sql, ()).unwrap();
    }
    sealed.engine().seal_for_test().unwrap();
    sealed
        .execute("DELETE FROM p WHERE id <= 135000", ())
        .unwrap();
    let hot = Executor::new(std::sync::Arc::clone(db.engine()));
    let cold = Executor::new(std::sync::Arc::clone(sealed.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let mut wrong = Vec::new();
    let mut charged = |executor: &Executor, sql: &str, least: usize| {
        count_deadline_charges(DeadlineSite::StorageRead);
        let counted = read(executor, sql, &ctx);
        let charges = deadline_charges();
        keep_deadlines();
        if counted.is_err() || charges < least {
            wrong.push(format!(
                "{sql} gave {counted:?} in {charges} charges, under {least}"
            ));
        }
    };
    // The 10,000 candidates a correlated EXISTS counted on one key
    charged(
        &hot,
        "SELECT EXISTS(SELECT 1 FROM i WHERE i.g = o.g AND i.v < 0) FROM o LIMIT 1",
        3,
    );
    // A hidden prefix of 135,000 sealed rows before the first shown one,
    // charged a run at a time on top of the 57 charges of the rest
    charged(
        &cold,
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY half ORDER BY id) FROM p LIMIT 1",
        80,
    );
    // The transaction's own 20,000 rows, gathered and asked
    read(&hot, "BEGIN", &ctx).unwrap();
    read(
        &hot,
        "INSERT INTO e SELECT value, value FROM generate_series(1, 20000)",
        &ctx,
    )
    .unwrap();
    charged(&hot, "SELECT id FROM e WHERE x = -1 LIMIT 1", 8);
    charged(
        &hot,
        "SELECT id FROM e WHERE id NOT IN (SELECT id FROM one) LIMIT 1",
        4,
    );
    read(&hot, "ROLLBACK", &ctx).unwrap();
    // A snapshot older than 9,000 versions of the one row of l
    read(&hot, "BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", &ctx).unwrap();
    read(&hot, "SELECT v FROM l", &ctx).unwrap();
    for _ in 0..9000 {
        db.execute("UPDATE l SET v = v + 1", ()).unwrap();
    }
    charged(&hot, "SELECT v FROM l WHERE v >= 0 LIMIT 1", 3);
    charged(&hot, "SELECT v FROM l LIMIT 1", 3);
    charged(&hot, "SELECT v FROM l WHERE g > 0 AND v >= 0 LIMIT 1", 3);
    charged(
        &hot,
        "SELECT id FROM l WHERE id NOT IN (SELECT id FROM one)",
        5,
    );
    // A full read charges a chain's versions in one run after the chain
    charged(&hot, "SELECT v FROM l", 2);
    read(&hot, "ROLLBACK", &ctx).unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
