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

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{prepared_exists_counts, prepared_scalar_runs};
use stoolap::{Database, Params};

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, NULL)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 1, 9), (3, 1, NULL), (4, 2, 7), (5, 2, 3), \
         (6, 3, NULL), (7, 3, 4), (8, 3, 4)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The column each row of `sql` gives at `i`
fn column<P: Params>(db: &Database, sql: &str, i: usize, params: P) -> Vec<Option<i64>> {
    db.query(sql, params)
        .unwrap()
        .map(|row| row.unwrap().get::<Option<i64>>(i).unwrap())
        .collect()
}

fn error(db: &Database, sql: &str) -> String {
    match db.query(sql, ()) {
        Err(e) => e.to_string(),
        Ok(rows) => rows
            .map(|row| row.map(|_| ()))
            .collect::<Result<Vec<()>, _>>()
            .expect_err("expected an error")
            .to_string(),
    }
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn scalar(block: &str) -> String {
    format!("SELECT id, ({block}) FROM t ORDER BY id")
}

#[test]
fn a_scalar_block_answers_from_the_rows_its_order_by_limit_and_offset_keep() {
    let _guard = serial();
    let db = setup("block_order_scalar");
    let n = None;
    for (block, want) in [
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1",
            [Some(5), Some(3), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1",
            [n, Some(7), n, n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC NULLS LAST LIMIT 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x NULLS FIRST LIMIT 1",
            [n, Some(3), n, n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1 OFFSET 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1 OFFSET 2",
            [Some(5), n, Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x OFFSET 2",
            [n, n, n, n],
        ),
        ("SELECT x FROM u WHERE u.k = t.id LIMIT 0", [n, n, n, n]),
        (
            "SELECT u.id FROM u WHERE u.k = t.id ORDER BY x, u.id DESC LIMIT 1",
            [Some(1), Some(5), Some(8), n],
        ),
        (
            "SELECT u.id FROM u WHERE u.k = t.id ORDER BY x LIMIT 1",
            [Some(1), Some(5), Some(7), n],
        ),
        // Rows 7 and 8 tie on x: they stay in the order read
        (
            "SELECT u.id FROM u WHERE u.k = t.id ORDER BY x LIMIT 1 OFFSET 1",
            [Some(2), Some(4), Some(8), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY ABS(x - t.a) LIMIT 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x AS y FROM u WHERE u.k = t.id ORDER BY y DESC NULLS LAST LIMIT 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY 1 / (x - 9) LIMIT 1",
            [Some(5), Some(7), Some(4), n],
        ),
        (
            "SELECT u.x FROM u WHERE u.k = t.id \
             AND u.x > (SELECT MIN(x) FROM u AS w WHERE w.k = u.k) ORDER BY u.x LIMIT 1",
            [Some(9), Some(7), n, n],
        ),
        (
            "SELECT x FROM (VALUES (1, 5), (1, 2), (2, 8)) AS v(k, x) \
             WHERE v.k = t.id ORDER BY x LIMIT 1",
            [Some(2), Some(8), n, n],
        ),
    ] {
        let sql = scalar(block);
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        let runs = prepared_scalar_runs();
        assert_eq!(column(&db, &sql, 1, ()), want, "{sql}");
        // The nested MIN runs for each of the eight candidates too
        let per_call = if block.contains("AS w") { 12 } else { 4 };
        assert_eq!(
            prepared_scalar_runs() - runs,
            per_call,
            "{sql} runs once per row"
        );
    }
    let once = scalar("SELECT x FROM u WHERE u.k = 1 ORDER BY x DESC LIMIT 1");
    let runs = prepared_scalar_runs();
    assert_eq!(column(&db, &once, 1, ()), [n, n, n, n]);
    assert_eq!(
        prepared_scalar_runs() - runs,
        1,
        "an uncorrelated block runs once"
    );
    let params = scalar("SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT $1 OFFSET $2");
    assert_eq!(
        column(&db, &params, 1, (1, 1)),
        [Some(9), Some(7), Some(4), n],
        "LIMIT and OFFSET as parameters"
    );
    let ordered = "SELECT id FROM t \
                   ORDER BY (SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC NULLS LAST LIMIT 1), id";
    assert_eq!(prepared_line(&db, ordered), "Prepared: yes");
    assert_eq!(
        column(&db, ordered, 0, ()),
        [Some(3), Some(2), Some(1), Some(4)]
    );
}

#[test]
fn a_scalar_block_cut_to_more_than_one_row_or_by_a_bad_count_fails() {
    let _guard = serial();
    let db = setup("block_order_errors");
    for (block, want) in [
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 2",
            "scalar subquery returned more than one row",
        ),
        (
            "SELECT x FROM u WHERE u.k = 2 ORDER BY x LIMIT 2",
            "scalar subquery returned more than one row",
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT -1",
            "LIMIT must be non-negative, got -1",
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1 OFFSET -1",
            "OFFSET must be non-negative, got -1",
        ),
    ] {
        let sql = scalar(block);
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        let err = error(&db, &sql);
        assert!(err.contains(want), "{sql}: {err}");
    }
}

#[test]
fn exists_and_member_blocks_answer_from_the_rows_kept() {
    let _guard = serial();
    let db = setup("block_order_members");
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id LIMIT 0) ORDER BY id",
            &[][..],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id LIMIT 1 OFFSET 2) \
             ORDER BY id",
            &[1, 3],
        ),
        (
            "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.id OFFSET 2) \
             ORDER BY id",
            &[2, 4],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id ORDER BY x LIMIT 1) \
             ORDER BY id",
            &[1, 2, 3],
        ),
        (
            "SELECT id FROM t WHERE a / 10 IN (SELECT k FROM u ORDER BY x DESC LIMIT 2) ORDER BY id",
            &[1, 3],
        ),
        (
            "SELECT id FROM t WHERE a / 10 NOT IN (SELECT k FROM u ORDER BY x LIMIT 3) ORDER BY id",
            &[1],
        ),
        (
            "SELECT id FROM t WHERE a / 10 IN (SELECT k FROM u ORDER BY x NULLS FIRST, u.id LIMIT 2) \
             ORDER BY id",
            &[1, 3],
        ),
        (
            "SELECT id FROM t WHERE 5 IN (SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1) \
             ORDER BY id",
            &[1],
        ),
        (
            "SELECT id FROM t WHERE 4 = ANY \
             (SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1 OFFSET 1) ORDER BY id",
            &[3],
        ),
        (
            "SELECT id FROM t WHERE a > ALL \
             (SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1) ORDER BY id",
            &[2, 4],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let want: Vec<_> = want.iter().map(|&id| Some(id)).collect();
        assert_eq!(column(&db, sql, 0, ()), want, "{sql}");
    }
    let sql = "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.k = t.id LIMIT 1 OFFSET 1) FROM t \
               ORDER BY id";
    let probes = prepared_exists_counts().0;
    let found: Vec<bool> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<bool>(1).unwrap())
        .collect();
    assert_eq!(found, [true, true, true, false]);
    assert_eq!(prepared_exists_counts().0 - probes, 4, "one probe per row");
}

#[test]
fn a_cut_block_outside_the_family_says_why() {
    let _guard = serial();
    let db = setup("block_order_refused");
    for (block, reason) in [
        (
            "SELECT MAX(x) FROM u WHERE u.k = t.id ORDER BY 1 LIMIT 1",
            "a grouped subquery block with ORDER BY, LIMIT or OFFSET",
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY 1 LIMIT 1",
            "an ORDER BY position in a subquery block",
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT t.a",
            "a subquery block's LIMIT or OFFSET that is not a literal or a parameter",
        ),
    ] {
        let sql = scalar(block);
        assert_eq!(
            prepared_line(&db, &sql),
            format!("Prepared: no ({reason})"),
            "{sql}"
        );
    }
}
