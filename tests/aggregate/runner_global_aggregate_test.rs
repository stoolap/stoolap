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

//! A statement that aggregates one table's rows into one, with a WHERE the
//! table does not answer, feeds the rows its plan reads to the calls: it
//! answers as the rows themselves add up, for each run's parameters, inside
//! a transaction, hot and sealed, and past the size the feed splits at

use stoolap::Database;

fn fill(db: &Database, rows: i64) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_b ON t(b)", ()).unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value % 97, value % 7 FROM generate_series(1, {rows})"
        ),
        (),
    )
    .unwrap();
}

/// COUNT(*), SUM(a) and MAX(a) of the rows with b above `bound`, added up
/// from the rows a SELECT without aggregates reads
fn expected(db: &Database, bound: i64) -> (i64, i64, Option<i64>) {
    let rows: Vec<i64> = db
        .query("SELECT a FROM t WHERE b > $1", (bound,))
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    let sum = rows.iter().sum();
    (rows.len() as i64, sum, rows.iter().copied().max())
}

fn aggregated(rows: stoolap::Rows) -> (i64, i64, Option<i64>) {
    let row = rows.into_iter().next().unwrap().unwrap();
    let count = row.get::<i64>(0).unwrap();
    let sum = row.get::<Option<i64>>(1).unwrap().unwrap_or(0);
    (count, sum, row.get::<Option<i64>>(2).unwrap())
}

const SQL: &str = "SELECT COUNT(*), SUM(a), MAX(a) FROM t WHERE b > $1";

#[test]
fn a_prepared_aggregate_answers_each_run_s_parameters() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
            false => Database::open("memory://runner_global_aggregate_params").unwrap(),
        };
        fill(&db, 500);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let stmt = db.prepare(SQL).unwrap();
        for bound in [-1, 0, 3, 6, 9] {
            let got = aggregated(stmt.query((bound,)).unwrap());
            assert_eq!(got, expected(&db, bound), "sealed {sealed}, b > {bound}");
        }
    }
}

#[test]
fn a_call_s_argument_read_from_a_parameter_is_read_on_each_run() {
    let db = Database::open("memory://runner_global_aggregate_separator").unwrap();
    fill(&db, 20);
    let stmt = db
        .prepare("SELECT STRING_AGG(CAST(id AS TEXT), $1) FROM t WHERE b > 0 AND id <= 3")
        .unwrap();
    for separator in ["-", "+"] {
        let joined: String = stmt
            .query((separator,))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(
            joined.contains(separator) && joined.split(separator).count() == 3,
            "{separator}: {joined}"
        );
    }
}

#[test]
fn a_parameter_a_like_s_escape_reads_in_a_call_s_argument_is_read_on_each_run() {
    let db = Database::open("memory://runner_global_aggregate_escape").unwrap();
    fill(&db, 20);
    let stmt = db
        .prepare(
            "SELECT STRING_AGG('x', CASE WHEN 'a_' LIKE 'a!_' ESCAPE $1 THEN '-' ELSE '+' END) \
             FROM t WHERE id <= 2",
        )
        .unwrap();
    for (escape, want) in [("!", "x-x"), ("#", "x+x")] {
        let joined: String = stmt
            .query((escape,))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(joined, want, "ESCAPE {escape}");
    }
}

#[test]
fn a_parameter_in_a_call_the_order_by_alone_names_is_read_on_each_run() {
    let db = Database::open("memory://runner_global_aggregate_order_call").unwrap();
    fill(&db, 20);
    let stmt = db
        .prepare(
            "SELECT COUNT(*) FROM t WHERE id <= 2 \
             ORDER BY STRING_AGG(CAST(id AS TEXT), CAST(9223372036854775807 + $1 AS TEXT))",
        )
        .unwrap();
    assert!(stmt.query((0,)).unwrap().next().unwrap().is_ok());
    let overflow = stmt
        .query((1,))
        .and_then(|mut rows| rows.next().unwrap().map(|_| ()));
    assert!(overflow.is_err(), "the second run's argument overflows");
}

#[test]
fn an_aggregate_in_a_transaction_reads_its_own_writes() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
            false => Database::open("memory://runner_global_aggregate_tx").unwrap(),
        };
        fill(&db, 300);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let before = expected(&db, 2);
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (1000, 500, 6)", ())
            .unwrap();
        tx.execute("DELETE FROM t WHERE id = 7", ()).unwrap();
        let got = aggregated(tx.query(SQL, (2,)).unwrap());
        // Row 7 has b = 0, outside the WHERE; the new row is inside it
        let want = (before.0 + 1, before.1 + 500, Some(500));
        assert_eq!(got, want, "sealed {sealed}");
        tx.rollback().unwrap();
        assert_eq!(aggregated(db.query(SQL, (2,)).unwrap()), before);
    }
}

#[test]
fn a_table_past_the_parallel_feed_answers_as_its_rows_add_up() {
    let db = Database::open("memory://runner_global_aggregate_large").unwrap();
    fill(&db, 120_000);
    for bound in [-1, 4] {
        let got = aggregated(db.query(SQL, (bound,)).unwrap());
        assert_eq!(got, expected(&db, bound), "b > {bound}");
    }
}

#[test]
fn explain_shows_the_rows_the_aggregate_reads() {
    let db = Database::open("memory://runner_global_aggregate_explain").unwrap();
    fill(&db, 50);
    let lines: Vec<String> = db
        .query("EXPLAIN SELECT COUNT(*), SUM(a) FROM t WHERE a > 3", ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(
        lines,
        [
            "Aggregate COUNT(*), SUM(a)",
            "  StorageFiltered t (a > 3)",
            "  else SeqScan t (a > 3)",
        ]
    );
}
