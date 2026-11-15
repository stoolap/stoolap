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

//! PRAGMA QUERY_TIMEOUT and the DSN's query_timeout: the timeout of a
//! handle's statements whose call sets none

use stoolap::Database;

fn timeout_of(db: &Database) -> i64 {
    db.query_one("PRAGMA QUERY_TIMEOUT", ()).unwrap()
}

#[test]
fn a_handle_reads_and_sets_its_query_timeout() {
    let db = Database::open("memory://query_timeout_pragma_set").unwrap();
    assert_eq!(timeout_of(&db), 0, "no timeout until one is set");
    let set: i64 = db.query_one("PRAGMA QUERY_TIMEOUT = 5000", ()).unwrap();
    assert_eq!((set, timeout_of(&db)), (5000, 5000));
    db.execute("PRAGMA QUERY_TIMEOUT = 0", ()).unwrap();
    assert_eq!(timeout_of(&db), 0);
    db.execute("PRAGMA QUERY_TIMEOUT = 300", ()).unwrap();
    for refused in ["-1", "-0.5"] {
        let sql = format!("PRAGMA QUERY_TIMEOUT = {refused}");
        assert!(db.execute(&sql, ()).is_err(), "{refused} was taken");
        assert_eq!(timeout_of(&db), 300, "{refused} changed the timeout");
    }
}

#[test]
fn each_handle_keeps_its_own_query_timeout() {
    let db = Database::open("memory://query_timeout_pragma_handles").unwrap();
    let other = db.clone();
    db.execute("PRAGMA QUERY_TIMEOUT = 700", ()).unwrap();
    assert_eq!((timeout_of(&db), timeout_of(&other)), (700, 0));
    db.execute("BEGIN", ()).unwrap();
    db.execute("PRAGMA QUERY_TIMEOUT = 900", ()).unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(
        timeout_of(&db),
        900,
        "a session setting, kept past the transaction"
    );
}

#[test]
fn the_dsn_query_timeout_starts_every_handle() {
    let db = Database::open("memory://query_timeout_pragma_dsn?query_timeout=2500").unwrap();
    let other = db.clone();
    other.execute("PRAGMA QUERY_TIMEOUT = 100", ()).unwrap();
    assert_eq!((timeout_of(&db), timeout_of(&other)), (2500, 100));
    assert_eq!(timeout_of(&db.clone()), 2500);
    for dsn in [
        "memory://query_timeout_pragma_bad?query_timeout=-5",
        "memory://query_timeout_pragma_bad?query_timeout=soon",
    ] {
        assert!(Database::open(dsn).is_err(), "{dsn} opened");
    }
}

#[test]
fn a_query_timeout_past_the_clock_runs_to_the_end() {
    let db = Database::open("memory://query_timeout_pragma_far").unwrap();
    db.execute("PRAGMA QUERY_TIMEOUT = 9223372036854775807", ())
        .unwrap();
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM generate_series(1, 1000)", ())
        .unwrap();
    assert_eq!(count, 1000);
}

#[cfg(feature = "test-failpoints")]
mod stops {
    use stoolap::executor::DeadlineSite;
    use stoolap::test_failpoints::{
        keep_deadlines, pass_deadline_after, pass_deadline_at, FailpointGuard,
    };
    use stoolap::{Database, Error};

    const SCAN: &str = "SELECT id FROM t WHERE v * 1 > 0 ORDER BY v DESC";

    fn rows(run: stoolap::Result<stoolap::Rows>) -> stoolap::Result<usize> {
        run?.try_fold(0, |n, row| row.map(|_| n + 1))
    }

    /// `run` with the deadline passed at the scan, if the statement has one
    fn past_the_scan(run: impl FnOnce() -> stoolap::Result<usize>) -> stoolap::Result<usize> {
        pass_deadline_at(DeadlineSite::Scan);
        let result = run();
        keep_deadlines();
        result
    }

    fn table(dsn: &str) -> Database {
        let db = Database::open(dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute(
            "INSERT INTO t SELECT value, value FROM generate_series(1, 2000)",
            (),
        )
        .unwrap();
        db
    }

    #[test]
    fn a_statement_stops_at_its_handles_query_timeout() {
        let _guard = FailpointGuard::new();
        let db = table("memory://query_timeout_pragma_stops");
        assert_eq!(past_the_scan(|| rows(db.query(SCAN, ()))).ok(), Some(2000));
        db.execute("PRAGMA QUERY_TIMEOUT = 3600000", ()).unwrap();
        let cancelled =
            |result: stoolap::Result<usize>| matches!(result, Err(Error::QueryCancelled));
        let statement = db.prepare(SCAN).unwrap();
        let ways = [
            ("text", past_the_scan(|| rows(db.query(SCAN, ())))),
            ("prepared", past_the_scan(|| rows(statement.query(())))),
            (
                "a call timeout of 0",
                past_the_scan(|| rows(db.query_with_timeout(SCAN, (), 0))),
            ),
            (
                "another handle",
                past_the_scan(|| rows(db.clone().query(SCAN, ()))),
            ),
        ];
        let stopped: Vec<(&str, bool)> = ways
            .into_iter()
            .map(|(way, result)| (way, cancelled(result)))
            .collect();
        assert_eq!(
            stopped,
            [
                ("text", true),
                ("prepared", true),
                ("a call timeout of 0", true),
                ("another handle", false),
            ]
        );
        let mut tx = db.begin().unwrap();
        let in_tx = past_the_scan(|| rows(tx.query(SCAN, ())));
        tx.rollback().unwrap();
        assert!(
            matches!(in_tx, Err(Error::QueryCancelled)),
            "a transaction's statement gave {in_tx:?}"
        );
        db.execute("PRAGMA QUERY_TIMEOUT = 0", ()).unwrap();
        assert_eq!(past_the_scan(|| rows(db.query(SCAN, ()))).ok(), Some(2000));
    }

    #[test]
    fn a_cached_statement_with_parameters_stops_at_its_handles_query_timeout() {
        let _guard = FailpointGuard::new();
        let db = table("memory://query_timeout_pragma_params");
        db.execute("CREATE INDEX t_v ON t(v)", ()).unwrap();
        let by_id = "SELECT v FROM t WHERE id = $1";
        let by_v = "SELECT id FROM t WHERE v = $1";
        let prepared = db.prepare(by_v).unwrap();
        let past_every_check = |run: &dyn Fn() -> stoolap::Result<usize>| {
            pass_deadline_after(DeadlineSite::Scan, 0);
            let result = run();
            keep_deadlines();
            result
        };
        let ways = |db: &Database| {
            [
                past_every_check(&|| rows(db.query(by_id, (7,)))),
                past_every_check(&|| rows(db.query(by_v, (7,)))),
                past_every_check(&|| rows(prepared.query((7,)))),
            ]
            .map(|result| match result {
                Err(Error::QueryCancelled) => "cancelled".to_string(),
                other => format!("{other:?}"),
            })
        };
        assert_eq!(ways(&db), ["Ok(1)", "Ok(1)", "Ok(1)"], "no timeout set");
        db.execute("PRAGMA QUERY_TIMEOUT = 3600000", ()).unwrap();
        assert_eq!(ways(&db), ["cancelled", "cancelled", "cancelled"]);
    }

    #[test]
    fn the_dsn_query_timeout_stops_a_statement() {
        let _guard = FailpointGuard::new();
        let db = table("memory://query_timeout_pragma_dsn_stops?query_timeout=3600000");
        let result = past_the_scan(|| rows(db.query(SCAN, ())));
        assert!(
            matches!(result, Err(Error::QueryCancelled)),
            "gave {result:?}"
        );
    }
}
