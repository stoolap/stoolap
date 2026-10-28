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

//! A scan by the access a plan chose reads the rows a scan that chooses its
//! own access reads: hot and sealed, with the transaction's own changes

use stoolap::core::{RowVec, Value};
use stoolap::storage::expression::{
    AndExpr, BetweenExpr, ComparisonExpr, Expression, InListExpr, LikeExpr,
};
use stoolap::storage::mvcc::index_plan::decide;
use stoolap::storage::traits::{Engine, ScanAccess, Scanner};
use stoolap::Database;

/// The rows as `(row id, values)`, by row id
fn sorted(rows: impl IntoIterator<Item = (i64, Vec<Value>)>) -> Vec<(i64, Vec<Value>)> {
    let mut rows: Vec<_> = rows.into_iter().collect();
    rows.sort_by_key(|(id, _)| *id);
    rows
}

fn scanned(mut scanner: Box<dyn Scanner>) -> Vec<(i64, Vec<Value>)> {
    let mut rows = Vec::new();
    while scanner.next() {
        rows.push((scanner.current_row_id(), scanner.row().as_slice().to_vec()));
    }
    assert!(scanner.err().is_none(), "{:?}", scanner.err());
    sorted(rows)
}

fn collected(rows: RowVec) -> Vec<(i64, Vec<Value>)> {
    sorted(rows.into_iter().map(|(id, row)| (id, row.into_values())))
}

/// Each index condition with the filter that holds it and more
fn cases() -> Vec<(Box<dyn Expression>, Box<dyn Expression>)> {
    let eq = || Box::new(ComparisonExpr::eq("a", Value::Integer(3))) as Box<dyn Expression>;
    let range = || {
        Box::new(AndExpr::new(vec![
            Box::new(ComparisonExpr::gte("a", Value::Integer(2))),
            Box::new(ComparisonExpr::lte("a", Value::Integer(5))),
        ])) as Box<dyn Expression>
    };
    let list = || {
        Box::new(InListExpr::new(
            "a",
            vec![Value::Integer(1), Value::Integer(4), Value::Integer(9)],
        )) as Box<dyn Expression>
    };
    let between = || {
        Box::new(BetweenExpr::new("a", Value::Integer(6), Value::Integer(8))) as Box<dyn Expression>
    };
    let prefix = || Box::new(LikeExpr::new("b", "k1%")) as Box<dyn Expression>;
    let with_c = |condition: Box<dyn Expression>| {
        Box::new(AndExpr::new(vec![
            condition,
            Box::new(ComparisonExpr::gt("c", Value::Integer(40))),
        ])) as Box<dyn Expression>
    };
    vec![
        (eq(), eq()),
        (eq(), with_c(eq())),
        (range(), range()),
        (range(), with_c(range())),
        (list(), with_c(list())),
        (between(), with_c(between())),
        (prefix(), with_c(prefix())),
    ]
}

/// The table's rows: `a` and `b` indexed, `c` not
fn fill(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, c INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_a ON t(a)", ()).unwrap();
    db.execute("CREATE INDEX t_b ON t(b)", ()).unwrap();
    for id in 1..=300i64 {
        db.execute(
            "INSERT INTO t VALUES ($1, $2, $3, $4)",
            (id, id % 10, format!("k{}", id % 25), id % 97),
        )
        .unwrap();
    }
}

/// Changes after the seal: rows moved out of and into the conditions,
/// deleted, and new
fn change(db: &Database) {
    db.execute("UPDATE t SET a = 3 WHERE id % 7 = 0", ())
        .unwrap();
    db.execute("UPDATE t SET a = 11, c = 99 WHERE id % 11 = 0", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id % 13 = 0", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1001, 3, 'k10', 50), (1002, 4, 'k1', 60)",
        (),
    )
    .unwrap();
}

fn assert_alike(db: &Database, what: &str) {
    let mut tx = db.engine().begin_transaction().unwrap();
    {
        let mut table = tx.get_table("t").unwrap();
        // The transaction's own rows: a new one, a changed one, a deleted one
        table
            .insert(stoolap::core::Row::from_values(vec![
                Value::Integer(2001),
                Value::Integer(3),
                Value::text("k1"),
                Value::Integer(70),
            ]))
            .unwrap();
    }
    let table = tx.get_table("t").unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    let columns = [0, 1, 2, 3];
    for (mut condition, mut filter) in cases() {
        condition.prepare_for_schema(table.schema());
        filter.prepare_for_schema(table.schema());
        let plan = decide(&store, table.schema(), condition.as_ref())
            .unwrap_or_else(|| panic!("{what}: no index reads {condition:?}"));
        let expected = scanned(table.scan(&columns, Some(filter.as_ref())).unwrap());
        assert!(!expected.is_empty(), "{what}: {filter:?} keeps no row");
        let by_index = table
            .scan_access(
                ScanAccess::Index(&plan),
                &columns,
                None,
                Some(filter.as_ref()),
            )
            .unwrap();
        assert_eq!(
            scanned(by_index),
            expected,
            "{what}: index {condition:?} for {filter:?}"
        );
        let every = table
            .scan_access(ScanAccess::Seq, &columns, None, Some(filter.as_ref()))
            .unwrap();
        assert_eq!(scanned(every), expected, "{what}: every row for {filter:?}");
        let rows = table
            .collect_access_rows(ScanAccess::Index(&plan), Some(filter.as_ref()))
            .unwrap();
        assert_eq!(
            collected(rows),
            expected,
            "{what}: collected by index for {filter:?}"
        );
        for access in [ScanAccess::Index(&plan), ScanAccess::Seq] {
            for limit in [0, 3, expected.len() + 5] {
                let rows = table
                    .collect_access_rows_with_limit_unordered(
                        access,
                        Some(filter.as_ref()),
                        limit,
                        0,
                    )
                    .unwrap();
                let rows = collected(rows);
                assert_eq!(
                    rows.len(),
                    limit.min(expected.len()),
                    "{what}: limit {limit} for {filter:?}"
                );
                assert!(
                    rows.iter().all(|row| expected.contains(row)),
                    "{what}: limit {limit} read a row {filter:?} does not keep"
                );
            }
        }
    }
    #[cfg(feature = "test-failpoints")]
    {
        let mut condition = ComparisonExpr::eq("a", Value::Integer(3));
        condition.prepare_for_schema(table.schema());
        let plan = decide(&store, table.schema(), &condition as &dyn Expression).unwrap();
        let read = stoolap::test_failpoints::hot_index_ids_read;
        let before = read();
        drop(
            table
                .scan_access(ScanAccess::Seq, &columns, None, Some(&condition))
                .unwrap(),
        );
        assert_eq!(read(), before, "{what}: a scan of every row read an index");
        drop(
            table
                .collect_access_rows_with_limit_unordered(ScanAccess::Seq, Some(&condition), 3, 0)
                .unwrap(),
        );
        assert_eq!(
            read(),
            before,
            "{what}: a limited scan of every row read an index"
        );
        drop(
            table
                .scan_access(ScanAccess::Index(&plan), &columns, None, Some(&condition))
                .unwrap(),
        );
        assert!(read() > before, "{what}: the index access read no index");
    }
    tx.rollback().unwrap();
}

#[test]
fn an_index_access_reads_the_rows_a_chosen_scan_reads_hot() {
    let db = Database::open("memory://scan_access_hot").unwrap();
    fill(&db);
    change(&db);
    assert_alike(&db, "hot");
}

#[test]
fn an_index_access_reads_the_rows_a_chosen_scan_reads_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    fill(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    change(&db);
    assert_alike(&db, "sealed");
}

/// The ids `sql` gives, sorted
fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn a_range_after_a_multi_column_prefix_keeps_the_rows_at_its_bounds() {
    let db = Database::open("memory://scan_access_prefix_range").unwrap();
    for table in ["t", "plain"] {
        db.execute(
            &format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)"
            ),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} VALUES (1, 1, 5, 0), (2, 1, 10, 0), (3, 1, 3, 7), \
                 (4, 2, 5, 0), (5, 1, 11, 1), (6, 0, 5, 0)"
            ),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_abc ON t(a, b, c)", ()).unwrap();
    for condition in [
        "a = 1 AND b < 10",
        "a = 1 AND b <= 10",
        "a = 1 AND b > 3 AND b < 10",
        "a = 1 AND b >= 3 AND b <= 10",
        "a = 1 AND b > 3 AND b <= 11",
    ] {
        for tail in ["", " LIMIT 10"] {
            let sql = format!("SELECT id FROM {{t}} WHERE {condition}{tail}");
            assert_eq!(
                ids(&db, &sql.replace("{t}", "t")),
                ids(&db, &sql.replace("{t}", "plain")),
                "{sql}"
            );
        }
    }
}
