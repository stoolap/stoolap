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

//! A statement that groups one table's rows feeds the rows its plan reads to
//! the groups: it answers as the rows themselves add up, for each run's
//! parameters, inside a transaction, hot and sealed, past the size the feed
//! splits at, and under a LIMIT that caps its groups

use std::collections::BTreeMap;

use stoolap::Database;

fn fill(db: &Database, rows: i64) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, k INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_b ON t(b)", ()).unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value % 97, value % 7, value % 10 \
             FROM generate_series(1, {rows})"
        ),
        (),
    )
    .unwrap();
}

/// Each group of k over the rows with b above `bound`, its COUNT(*) and
/// SUM(a), added up from the rows a SELECT without aggregates reads
fn groups(db: &Database, bound: i64) -> BTreeMap<i64, (i64, i64)> {
    let mut groups = BTreeMap::new();
    for row in db
        .query("SELECT k, a FROM t WHERE b > $1", (bound,))
        .unwrap()
    {
        let row = row.unwrap();
        let group: &mut (i64, i64) = groups.entry(row.get::<i64>(0).unwrap()).or_default();
        group.0 += 1;
        group.1 += row.get::<i64>(1).unwrap();
    }
    groups
}

/// The groups whose sum is above `least`, in the order of k
fn expected(db: &Database, bound: i64, least: i64) -> Vec<(i64, i64, i64)> {
    groups(db, bound)
        .into_iter()
        .filter(|(_, (_, sum))| *sum > least)
        .map(|(k, (count, sum))| (k, count, sum))
        .collect()
}

fn grouped(rows: stoolap::Rows) -> Vec<(i64, i64, i64)> {
    rows.map(|row| {
        let row = row.unwrap();
        (
            row.get::<i64>(0).unwrap(),
            row.get::<i64>(1).unwrap(),
            row.get::<i64>(2).unwrap(),
        )
    })
    .collect()
}

const SQL: &str =
    "SELECT k, COUNT(*), SUM(a) FROM t WHERE b > $1 GROUP BY k HAVING SUM(a) > $2 ORDER BY k";

#[test]
fn a_prepared_grouping_answers_each_run_s_parameters() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
            false => Database::open("memory://runner_grouped_aggregate_params").unwrap(),
        };
        fill(&db, 500);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let stmt = db.prepare(SQL).unwrap();
        for (bound, least) in [(-1, 0), (0, 1500), (3, 0), (6, 0), (9, 0)] {
            let got = grouped(stmt.query((bound, least)).unwrap());
            assert_eq!(
                got,
                expected(&db, bound, least),
                "sealed {sealed}, b > {bound}, SUM(a) > {least}"
            );
        }
    }
}

#[test]
fn a_having_over_the_groups_the_table_computes_reads_each_run_s_parameter() {
    const HAVING: &str =
        "SELECT k, COUNT(*), SUM(a) FROM t GROUP BY k HAVING SUM(a) > $1 ORDER BY k";
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
            false => Database::open("memory://runner_grouped_aggregate_having").unwrap(),
        };
        fill(&db, 500);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let stmt = db.prepare(HAVING).unwrap();
        for least in [0, 2400, 2500, 1_000_000] {
            let got = grouped(stmt.query((least,)).unwrap());
            assert_eq!(got, expected(&db, -1, least), "sealed {sealed}, {least}");
        }
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (1000, 5000, 6, 3)", ())
            .unwrap();
        let got = grouped(tx.query(HAVING, (2500,)).unwrap());
        let mut want = expected(&db, -1, 0);
        for group in want.iter_mut().filter(|group| group.0 == 3) {
            group.1 += 1;
            group.2 += 5000;
        }
        want.retain(|group| group.2 > 2500);
        assert_eq!(got, want, "sealed {sealed}, the transaction's own row");
        tx.rollback().unwrap();
    }
}

#[test]
fn a_grouping_in_a_transaction_reads_its_own_writes() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
            false => Database::open("memory://runner_grouped_aggregate_tx").unwrap(),
        };
        fill(&db, 300);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let before = expected(&db, 2, 0);
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (1000, 500, 6, 42)", ())
            .unwrap();
        let got = grouped(tx.query(SQL, (2, 0)).unwrap());
        let mut want = before.clone();
        want.push((42, 1, 500));
        assert_eq!(got, want, "sealed {sealed}");
        tx.rollback().unwrap();
        assert_eq!(grouped(db.query(SQL, (2, 0)).unwrap()), before);
    }
}

#[test]
fn a_table_past_the_parallel_feed_answers_as_its_rows_add_up() {
    let db = Database::open("memory://runner_grouped_aggregate_large").unwrap();
    fill(&db, 120_000);
    for bound in [-1, 4] {
        let got = grouped(db.query(SQL, (bound, 0)).unwrap());
        assert_eq!(got, expected(&db, bound, 0), "b > {bound}");
    }
}

#[test]
fn a_limit_without_order_caps_the_groups_each_whole() {
    let db = Database::open("memory://runner_grouped_aggregate_cap").unwrap();
    fill(&db, 500);
    let all = groups(&db, 0);
    let rows = db
        .query(
            "SELECT k, COUNT(*), SUM(a) FROM t WHERE b > 0 GROUP BY k LIMIT 3 OFFSET 1",
            (),
        )
        .unwrap();
    let got = grouped(rows);
    assert_eq!(got.len(), 3);
    for (k, count, sum) in got {
        assert_eq!(all.get(&k), Some(&(count, sum)), "group {k}");
    }
}

#[test]
fn an_empty_grouping_set_over_a_key_read_computes_its_select_list() {
    let db = Database::open("memory://runner_grouped_aggregate_empty_set").unwrap();
    fill(&db, 20);
    let sql = "SELECT 42 AS answer FROM t WHERE id = 1 GROUP BY GROUPING SETS (())";
    let rows = db.query(sql, ()).unwrap();
    assert_eq!(rows.columns(), ["answer"]);
    let got: Vec<i64> = rows.map(|row| row.unwrap().get(0).unwrap()).collect();
    assert_eq!(got, [42]);
}

#[test]
fn an_extra_argument_s_error_reaches_every_source() {
    let db = Database::open("memory://runner_grouped_aggregate_extra_error").unwrap();
    fill(&db, 20);
    for sql in [
        "SELECT k, SUM(a, ABS(-9223372036854775808)) FROM t GROUP BY k",
        "SELECT SUM(a, ABS(-9223372036854775808)) FROM t",
        "SELECT COUNT(1, ABS(-9223372036854775808)) FROM t",
        "SELECT MIN(b, ABS(-9223372036854775808)) FROM t",
    ] {
        let answer = db
            .query(sql, ())
            .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>());
        let error = answer.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            error.contains("Integer overflow in ABS"),
            "{sql}: answered past its argument's error: {error:?}"
        );
    }
}

#[test]
fn explain_shows_the_groups_over_the_rows_the_read_passes() {
    let db = Database::open("memory://runner_grouped_aggregate_explain").unwrap();
    fill(&db, 50);
    let lines: Vec<String> = db
        .query(
            "EXPLAIN SELECT k, COUNT(*) FROM t WHERE b > 3 GROUP BY k \
             HAVING COUNT(*) > 1 ORDER BY k LIMIT 2",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(
        lines,
        [
            "Limit 2",
            "  TopN k ASC",
            "    Having (COUNT(*) > 1)",
            "      Aggregate k, COUNT(*) by k",
            "        IndexScan t (b > 3)",
            "          Index t_b (b range)",
        ]
    );
    let lines: Vec<String> = db
        .query("EXPLAIN SELECT b, COUNT(*) FROM t GROUP BY b", ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(
        lines,
        [
            "Aggregate b, COUNT(*) by b",
            "  StorageGrouped t (b)",
            "  else SortedStream t (b)",
            "  else SeqScan t",
        ]
    );
}
