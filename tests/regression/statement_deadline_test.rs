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
use stoolap::test_failpoints::{keep_deadlines, pass_deadline_at, FailpointGuard};
use stoolap::{Database, Error, ExecutionContext, Executor};

/// A statement whose run reaches `site`
fn statement(site: DeadlineSite) -> &'static str {
    match site {
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
    }
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
    for sql in [
        "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, a INTEGER, b INTEGER)",
        "INSERT INTO big SELECT value, value % 50, value, value % 10, value FROM generate_series(1, 5000)",
        "CREATE INDEX big_ab ON big (a, b)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO u SELECT value, value % 50 FROM generate_series(1, 200)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let mut ctx = ExecutionContext::new();
    ctx.set_timeout_ms(3_600_000);
    let mut wrong = Vec::new();
    for site in DeadlineSite::ALL {
        let sql = statement(site);
        pass_deadline_at(site);
        let stopped = read(&executor, sql, &ctx);
        keep_deadlines();
        if !matches!(stopped, Err(Error::QueryCancelled)) {
            wrong.push(format!("{site:?}: {sql} gave {stopped:?}"));
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
