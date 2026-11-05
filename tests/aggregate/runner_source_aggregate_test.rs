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

//! A statement that aggregates the rows of a derived table, a view, a CTE,
//! a VALUES list or a table function feeds the rows the source gives to its
//! calls: it answers as those rows add up, for each run's parameters and
//! inside a transaction, and EXPLAIN shows the aggregate over the source

use std::collections::BTreeMap;

use stoolap::Database;

fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, a INTEGER)",
        "INSERT INTO t SELECT value, value % 4, value % 13 FROM generate_series(1, 200)",
        "CREATE VIEW v AS SELECT k, a FROM t WHERE id % 2 = 0",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// Each group of k over the rows `sql` reads, with a above `bound`, its
/// COUNT(*) and SUM(a), added up from the rows themselves
fn groups(db: &Database, sql: &str, bound: i64) -> BTreeMap<i64, (i64, i64)> {
    let mut groups = BTreeMap::new();
    for row in db.query(sql, (bound,)).unwrap() {
        let row = row.unwrap();
        let group: &mut (i64, i64) = groups.entry(row.get::<i64>(0).unwrap()).or_default();
        group.0 += 1;
        group.1 += row.get::<i64>(1).unwrap();
    }
    groups
}

fn grouped(rows: stoolap::Rows) -> BTreeMap<i64, (i64, i64)> {
    rows.map(|row| {
        let row = row.unwrap();
        let (k, n, sum) = (
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
        );
        (k, (n, sum))
    })
    .collect()
}

/// (statement grouping a source, the rows it reads without aggregates)
const SOURCES: &[(&str, &str)] = &[
    (
        "SELECT k, COUNT(*), SUM(a) FROM (SELECT k, a FROM t) d WHERE a > $1 GROUP BY k",
        "SELECT k, a FROM t WHERE a > $1",
    ),
    (
        "SELECT k, COUNT(*), SUM(a) FROM v WHERE a > $1 GROUP BY k",
        "SELECT k, a FROM t WHERE id % 2 = 0 AND a > $1",
    ),
    (
        "WITH c AS (SELECT k, a FROM t WHERE id > 100) \
         SELECT k, COUNT(*), SUM(a) FROM c WHERE a > $1 GROUP BY k",
        "SELECT k, a FROM t WHERE id > 100 AND a > $1",
    ),
];

#[test]
fn a_prepared_grouping_of_a_source_answers_each_run_s_parameters() {
    let db = Database::open("memory://runner_source_aggregate_params").unwrap();
    fill(&db);
    for (sql, rows) in SOURCES {
        let stmt = db.prepare(sql).unwrap();
        for bound in [-1, 3, 7, 12] {
            let got = grouped(stmt.query((bound,)).unwrap());
            assert_eq!(got, groups(&db, rows, bound), "{sql} with a > {bound}");
        }
    }
}

#[test]
fn a_grouping_of_a_source_in_a_transaction_reads_its_own_writes() {
    let db = Database::open("memory://runner_source_aggregate_tx").unwrap();
    fill(&db);
    for (sql, rows) in SOURCES {
        let before = groups(&db, rows, 0);
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (1000, 2, 12)", ())
            .unwrap();
        let got = grouped(tx.query(sql, (0,)).unwrap());
        let mut want = before.clone();
        let group = want.entry(2).or_default();
        group.0 += 1;
        group.1 += 12;
        assert_eq!(got, want, "{sql}");
        tx.rollback().unwrap();
        assert_eq!(grouped(db.query(sql, (0,)).unwrap()), before, "{sql}");
    }
}

#[test]
fn a_limit_cuts_the_aggregate_s_rows_not_the_source_s() {
    let db = Database::open("memory://runner_source_aggregate_limit").unwrap();
    fill(&db);
    for (sql, want) in [
        ("SELECT COUNT(*) FROM (SELECT a FROM t) d LIMIT 1", 200),
        ("SELECT COUNT(*) FROM v LIMIT 1", 100),
        ("SELECT COUNT(*) FROM generate_series(1, 100) LIMIT 1", 100),
        (
            "WITH c AS (SELECT a FROM t) SELECT COUNT(*) FROM c LIMIT 1 OFFSET 0",
            200,
        ),
    ] {
        let got: Vec<i64> = db
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(got, [want], "{sql}");
    }
    let stmt = db
        .prepare("SELECT k, COUNT(*) FROM (SELECT k FROM t) d GROUP BY k ORDER BY k LIMIT $1")
        .unwrap();
    let got: Vec<(i64, i64)> = stmt
        .query((2,))
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect();
    assert_eq!(got, [(0, 50), (1, 50)]);
}

#[test]
fn every_value_of_a_values_list_is_evaluated_under_a_where_that_keeps_no_row() {
    let db = Database::open("memory://runner_source_aggregate_values_error").unwrap();
    for sql in [
        "SELECT COUNT(*) FROM (VALUES (9223372036854775807 + 1)) AS s(x) WHERE FALSE",
        "SELECT x FROM (VALUES (9223372036854775807 + 1)) AS s(x) WHERE FALSE",
    ] {
        let error = db
            .query(sql, ())
            .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("Integer overflow"), "{sql}: {error:?}");
    }
}

#[test]
fn explain_shows_the_aggregate_over_its_source() {
    let db = Database::open("memory://runner_source_aggregate_explain").unwrap();
    fill(&db);
    let lines = |sql: &str| -> Vec<String> {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect()
    };
    assert_eq!(
        lines("SELECT k, COUNT(*) FROM (SELECT k FROM t) d WHERE k > 0 GROUP BY k HAVING COUNT(*) > 1"),
        [
            "Having (COUNT(*) > 1)",
            "  Aggregate k, COUNT(*) by k",
            "    Derived d (k > 0)",
            "      Project k",
            "        SeqScan t",
        ]
    );
    assert_eq!(
        lines("SELECT COUNT(*), SUM(a) FROM v"),
        [
            "Aggregate COUNT(*), SUM(a)",
            "  View v",
            "    Project k, a",
            "      SeqScan t ((id % 2) = 0)",
        ]
    );
    assert_eq!(
        lines("SELECT x, COUNT(*) FROM (VALUES (1), (2)) AS s(x) GROUP BY x ORDER BY x"),
        ["Sort x ASC", "  Aggregate x, COUNT(*) by x", "    Values s"]
    );
    assert_eq!(
        lines("SELECT SUM(value) FROM generate_series(1, 5)"),
        ["Aggregate SUM(value)", "  Function generate_series"]
    );
}
