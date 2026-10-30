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

//! An UPDATE or DELETE whose subqueries are in the prepared family leaves
//! the table a SELECT computes for the same expressions over the table as
//! the statement found it, and the answer recorded when the general path
//! gave the same, under every EXISTS strategy; the corpus reaches every
//! registered prepared DML site.

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{
    force_exists_strategy, prepared_dml_built, prepared_dml_site_runs, PreparedDmlSite,
};
use stoolap::Database;

mod oracle;

/// The table each statement leaves, and the rows each RETURNING gives
const RECORDED_TABLES: oracle::Answers = answers!("prepared_dml_tables");
const RECORDED_RETURNING: oracle::Answers = answers!("prepared_dml_returning");

/// The forced strategy and the preparation counts are process-wide: tests
/// that force one or prepare a statement take turns
static FORCED: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 1, 1, '1'), (2, 2, 2, '2'), (3, NULL, NULL, NULL), (4, 4, 4, '4'), (5, 5, 5, '5')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, d INTEGER)",
        "INSERT INTO u VALUES (1, 1, 1, 2), (2, 1, 10, 1), (3, 2, NULL, 1), (4, NULL, 4, 1), (5, 4, 4, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn table(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap_or_else(|e| panic!("{sql}: {e}"));
            // A stored NULL carries its column's type, a selected one none
            (0..row.len())
                .map(|i| match row.get_value(i).unwrap() {
                    v if v.is_null() => "NULL".to_string(),
                    v => format!("{v:?}"),
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

const ROWS: &str = "SELECT id, k, x, s FROM t ORDER BY id";

/// The rows of `sql` on the general path, each operand of a UNION ALL apart
fn general_table(db: &Database, sql: &str) -> Vec<String> {
    let operands = sql.split(" UNION ALL ");
    oracle::on_general_path(sql, || {
        operands
            .flat_map(|operand| table(db, &oracle::general_twin(operand)))
            .collect()
    })
}

/// Each statement with a SELECT of the rows it leaves:
/// an UPDATE's as a CASE in the select list, a DELETE's as the ids it keeps
const CORPUS: &[(&str, &str)] = &[
    (
        "UPDATE t SET x = (SELECT MAX(v) FROM u WHERE u.k = t.k)",
        "SELECT id, k, (SELECT MAX(v) FROM u WHERE u.k = t.k), s FROM t ORDER BY id",
    ),
    (
        "UPDATE t SET x = (SELECT MAX(v) FROM u), s = (SELECT CAST(COUNT(*) AS TEXT) FROM u WHERE u.k = t.k)",
        "SELECT id, k, (SELECT MAX(v) FROM u), (SELECT CAST(COUNT(*) AS TEXT) FROM u WHERE u.k = t.k) FROM t ORDER BY id",
    ),
    (
        "UPDATE t SET x = x + 100 WHERE x < (SELECT MAX(v) FROM u WHERE u.k = t.k)",
        "SELECT id, k, CASE WHEN x < (SELECT MAX(v) FROM u WHERE u.k = t.k) THEN x + 100 ELSE x END, s FROM t ORDER BY id",
    ),
    (
        "UPDATE t SET s = 'hit' WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
        "SELECT id, k, x, CASE WHEN EXISTS (SELECT 1 FROM u WHERE u.k = t.k) THEN 'hit' ELSE s END FROM t ORDER BY id",
    ),
    (
        "UPDATE t SET s = 'hit' WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.id = t.id AND u.d = 1)",
        "SELECT id, k, x, CASE WHEN NOT EXISTS (SELECT 1 FROM u WHERE u.id = t.id AND u.d = 1) THEN 'hit' ELSE s END FROM t ORDER BY id",
    ),
    (
        "UPDATE t SET x = (SELECT COUNT(*) FROM u WHERE u.k = t.k) WHERE x IN (SELECT v FROM u)",
        "SELECT id, k, CASE WHEN x IN (SELECT v FROM u) THEN (SELECT COUNT(*) FROM u WHERE u.k = t.k) ELSE x END, s FROM t ORDER BY id",
    ),
    (
        "UPDATE t SET x = (SELECT SUM(t2.x) FROM t t2 WHERE t2.id < t.id)",
        "SELECT id, k, (SELECT SUM(t2.x) FROM t t2 WHERE t2.id < t.id), s FROM t ORDER BY id",
    ),
    (
        "DELETE FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
        "SELECT id, CASE WHEN EXISTS (SELECT 1 FROM u WHERE u.k = t.k) THEN 1 ELSE 0 END FROM t ORDER BY id",
    ),
    (
        "DELETE FROM t WHERE x NOT IN (SELECT v FROM u WHERE u.k = t.k)",
        "SELECT id, CASE WHEN x NOT IN (SELECT v FROM u WHERE u.k = t.k) THEN 1 ELSE 0 END FROM t ORDER BY id",
    ),
    (
        "DELETE FROM t WHERE x > (SELECT AVG(v) FROM u) OR (SELECT COUNT(*) FROM u WHERE u.k = t.k) = 0",
        "SELECT id, CASE WHEN x > (SELECT AVG(v) FROM u) OR (SELECT COUNT(*) FROM u WHERE u.k = t.k) = 0 THEN 1 ELSE 0 END FROM t ORDER BY id",
    ),
    (
        "DELETE FROM t a WHERE EXISTS (SELECT 1 FROM u WHERE u.k = a.k AND u.d = 1)",
        "SELECT id, CASE WHEN EXISTS (SELECT 1 FROM u WHERE u.k = a.k AND u.d = 1) THEN 1 ELSE 0 END FROM t a ORDER BY id",
    ),
    (
        "DELETE FROM t WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.id = t.id - 1)",
        "SELECT id, CASE WHEN EXISTS (SELECT 1 FROM t t2 WHERE t2.id = t.id - 1) THEN 1 ELSE 0 END FROM t ORDER BY id",
    ),
    (
        "INSERT INTO t VALUES (9, 9, (SELECT MAX(v) FROM u WHERE u.k = 1), 'v')",
        "SELECT 9, 9, (SELECT MAX(v) FROM u WHERE u.k = 1), 'v'",
    ),
    (
        "INSERT INTO t VALUES (9, 9, (SELECT COUNT(*) FROM t), (SELECT CAST(MAX(id) AS TEXT) FROM t)), (10, 10, (SELECT COUNT(*) FROM t), 'c')",
        "SELECT 9, 9, (SELECT COUNT(*) FROM t), (SELECT CAST(MAX(id) AS TEXT) FROM t) UNION ALL SELECT 10, 10, (SELECT COUNT(*) FROM t), 'c'",
    ),
    (
        "INSERT INTO t VALUES (9, 9, CASE WHEN EXISTS (SELECT 1 FROM u WHERE u.k = 4) THEN 1 ELSE 0 END, 'e')",
        "SELECT 9, 9, CASE WHEN EXISTS (SELECT 1 FROM u WHERE u.k = 4) THEN 1 ELSE 0 END, 'e'",
    ),
    (
        "INSERT INTO t VALUES (9, 9, CASE WHEN 4 IN (SELECT v FROM u) THEN 1 ELSE 0 END, 'i')",
        "SELECT 9, 9, CASE WHEN 4 IN (SELECT v FROM u) THEN 1 ELSE 0 END, 'i'",
    ),
    (
        "INSERT INTO t VALUES (2, 2, 0, 'n') ON CONFLICT (id) DO UPDATE SET x = (SELECT MAX(v) FROM u WHERE u.k = t.k)",
        "SELECT id, k, CASE WHEN id = 2 THEN (SELECT MAX(v) FROM u WHERE u.k = t.k) ELSE x END, s FROM t ORDER BY id",
    ),
    (
        "INSERT INTO t VALUES (1, 4, 0, 'n') ON CONFLICT (id) DO UPDATE SET x = (SELECT COUNT(*) FROM u WHERE u.k = excluded.k)",
        "SELECT id, k, CASE WHEN id = 1 THEN (SELECT COUNT(*) FROM u WHERE u.k = 4) ELSE x END, s FROM t ORDER BY id",
    ),
    (
        "INSERT INTO t VALUES (4, 4, 0, 'n') ON CONFLICT (id) DO UPDATE SET s = 'w' WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
        "SELECT id, k, x, CASE WHEN id = 4 AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k) THEN 'w' ELSE s END FROM t ORDER BY id",
    ),
    (
        "INSERT INTO t VALUES (1, 0, 0, 'n'), (2, 0, 0, 'n') ON CONFLICT (id) DO UPDATE SET x = (SELECT SUM(t2.x) FROM t t2 WHERE t2.id <> t.id)",
        "SELECT id, k, CASE WHEN id IN (1, 2) THEN (SELECT SUM(t2.x) FROM t t2 WHERE t2.id <> t.id) ELSE x END, s FROM t ORDER BY id",
    ),
];

/// Each statement with RETURNING with a SELECT of the rows
/// it returns, read before the statement runs
const RETURNING_CORPUS: &[(&str, &str)] = &[
    (
        "UPDATE t SET x = 0 WHERE id <= 2 RETURNING id, x, (SELECT MAX(v) FROM u WHERE u.k = t.k)",
        "SELECT id, 0, (SELECT MAX(v) FROM u WHERE u.k = t.k) FROM t WHERE id <= 2",
    ),
    (
        "UPDATE t SET s = 'z' WHERE id = 1 RETURNING *, (SELECT MAX(v) FROM u)",
        "SELECT id, k, x, 'z', (SELECT MAX(v) FROM u) FROM t WHERE id = 1",
    ),
    (
        "DELETE FROM t WHERE id >= 4 RETURNING id, (SELECT COUNT(*) FROM u WHERE u.k = t.k), (SELECT MAX(id) FROM t)",
        "SELECT id, (SELECT COUNT(*) FROM u WHERE u.k = t.k), (SELECT MAX(id) FROM t) FROM t WHERE id >= 4",
    ),
    (
        "INSERT INTO t VALUES (9, 1, 9, 'r') RETURNING id, CASE WHEN EXISTS (SELECT 1 FROM u WHERE u.k = t.k) THEN 1 ELSE 0 END, k IN (SELECT k FROM u)",
        "SELECT 9, CASE WHEN EXISTS (SELECT 1 FROM u WHERE u.k = 1) THEN 1 ELSE 0 END, 1 IN (SELECT k FROM u)",
    ),
    (
        "INSERT INTO t VALUES (2, 2, 0, 'n') ON CONFLICT (id) DO UPDATE SET x = 7 RETURNING id, x, (SELECT SUM(v) FROM u WHERE u.k = t.k)",
        "SELECT 2, 7, (SELECT SUM(v) FROM u WHERE u.k = 2)",
    ),
];

/// The rows an expected SELECT's `rows` name: an UPDATE's or an upsert's
/// whole rows, the rows of t whose id a DELETE's CASE marks 0, or t with
/// the rows an INSERT adds after it
fn expected_rows(db: &Database, dml: &str, rows: Vec<String>) -> Vec<String> {
    if dml.starts_with("INSERT") && !dml.contains("ON CONFLICT") {
        let mut all = table(db, ROWS);
        all.extend(rows);
        return all;
    }
    if !dml.starts_with("DELETE") {
        return rows;
    }
    let all = table(db, ROWS);
    rows.iter()
        .zip(all)
        .filter(|(marked, _)| marked.ends_with("|Integer(0)"))
        .map(|(_, row)| row)
        .collect()
}

#[test]
fn test_every_prepared_dml_site_leaves_the_general_answer_and_is_reached() {
    let _turn = FORCED.lock().unwrap_or_else(|e| e.into_inner());
    let before: Vec<usize> = PreparedDmlSite::ALL
        .iter()
        .map(|s| prepared_dml_site_runs(*s))
        .collect();
    for (n, strategy) in [None, Some(false), Some(true)].into_iter().enumerate() {
        force_exists_strategy(strategy);
        for (i, (dml, expected)) in CORPUS.iter().enumerate() {
            let db = setup(&format!("prepared_dml_corpus_{n}_{i}"));
            let want = expected_rows(&db, dml, table(&db, expected));
            let general =
                oracle::recording().then(|| expected_rows(&db, dml, general_table(&db, expected)));
            db.execute(dml, ())
                .unwrap_or_else(|e| panic!("{dml} under {strategy:?}: {e}"));
            let got = Ok(table(&db, ROWS));
            assert_eq!(got, Ok(want), "{dml} under {strategy:?}");
            let general = || Ok(general.unwrap_or_default());
            let wrong = oracle::check_confirmed(&RECORDED_TABLES, dml, &got, general);
            assert_eq!(wrong, None, "under {strategy:?}");
        }
        for (i, (dml, expected)) in RETURNING_CORPUS.iter().enumerate() {
            let db = setup(&format!("prepared_dml_returning_{n}_{i}"));
            let mut want = table(&db, expected);
            let general = oracle::recording().then(|| {
                let mut rows = general_table(&db, expected);
                rows.sort();
                rows
            });
            let mut got = table(&db, dml);
            want.sort();
            got.sort();
            assert_eq!(got, want, "{dml} under {strategy:?}");
            let general = || Ok(general.unwrap_or_default());
            let wrong = oracle::check_confirmed(&RECORDED_RETURNING, dml, &Ok(got), general);
            assert_eq!(wrong, None, "under {strategy:?}");
        }
    }
    force_exists_strategy(None);
    for (site, before) in PreparedDmlSite::ALL.iter().zip(before) {
        assert!(
            prepared_dml_site_runs(*site) > before,
            "no case of the corpus reaches {site:?}"
        );
    }
}

#[test]
fn test_a_prepared_dml_statement_is_kept_until_the_schema_changes() {
    let _turn = FORCED.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup("prepared_dml_kept");
    let update = db
        .prepare("UPDATE t SET x = (SELECT MAX(v) FROM u WHERE u.k = t.k) WHERE id = $1")
        .unwrap();
    let delete = db
        .prepare("DELETE FROM t WHERE x > (SELECT MAX(v) FROM u WHERE u.k = t.k) + $1")
        .unwrap();
    let start = prepared_dml_built();
    for id in [1i64, 2, 4] {
        update.execute((id,)).unwrap();
        delete.execute((100i64,)).unwrap();
    }
    assert_eq!(
        prepared_dml_built() - start,
        2,
        "each statement prepared once"
    );
    db.execute("ALTER TABLE u ADD COLUMN w INTEGER", ())
        .unwrap();
    update.execute((5i64,)).unwrap();
    assert_eq!(
        prepared_dml_built() - start,
        3,
        "a schema change prepares the statement again"
    );
    assert_eq!(
        table(&db, ROWS),
        [
            "Integer(1)|Integer(1)|Integer(10)|Text(\"1\")",
            "Integer(2)|Integer(2)|NULL|Text(\"2\")",
            "Integer(3)|NULL|NULL|NULL",
            "Integer(4)|Integer(4)|Integer(4)|Text(\"4\")",
            "Integer(5)|Integer(5)|NULL|Text(\"5\")",
        ]
    );
}

#[test]
fn a_keyed_update_after_a_write_of_its_transaction_reads_no_whole_table() {
    let _turn = FORCED.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup("prepared_dml_keyed_after_write");
    let mut tx = db.begin().unwrap();
    tx.execute("UPDATE t SET x = 9 WHERE id = 5", ()).unwrap();
    stoolap::test_failpoints::take_read_paths();
    let updated = tx
        .execute(
            "UPDATE t SET x = x + 1 WHERE id = (SELECT u.id FROM u WHERE u.id = 1)",
            (),
        )
        .unwrap();
    let paths = stoolap::test_failpoints::take_read_paths();
    tx.rollback().unwrap();
    assert_eq!(updated, 1);
    assert!(
        !paths
            .iter()
            .any(|path| path.starts_with("vs.get_all_visible_rows")),
        "a keyed update read the whole table: {paths:?}"
    );
}

#[test]
fn a_delete_reads_the_rows_its_narrowing_names_once() {
    let _turn = FORCED.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup("prepared_dml_delete_narrowed_once");
    let before = stoolap::test_failpoints::prepared_rows_narrowed();
    let deleted = db
        .execute(
            "DELETE FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id)",
            (),
        )
        .unwrap();
    let narrowed = stoolap::test_failpoints::prepared_rows_narrowed() - before;
    assert_eq!(deleted, 3);
    assert_eq!(
        narrowed, 1,
        "the delete read its narrowed rows {narrowed} times"
    );
}
