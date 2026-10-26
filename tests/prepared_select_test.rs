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

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    after_prepared_plan_taken, after_select_table_opened, before_prepared_scalar,
    general_path_hits, prepared_ordered_reads, prepared_projection_runs, prepared_scalar_runs,
    prepared_selects_built, random_draws, tvf_rows, GeneralPath,
};
use stoolap::{Database, Params};

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const S1: &str =
    "SELECT id FROM t WHERE b = $1 AND a > (SELECT v FROM u WHERE u.id = t.id) ORDER BY id";
const S2: &str =
    "SELECT id FROM t WHERE b = $1 AND a > (SELECT v FROM u WHERE u.id = t.id + 1) ORDER BY id";
// The same questions with an inner ORDER BY position, which keeps them out of the family
const S1_OLD: &str = "SELECT id FROM t WHERE b = $1 AND a > (SELECT v FROM u WHERE u.id = t.id ORDER BY 1) ORDER BY id";
const S2_OLD: &str = "SELECT id FROM t WHERE b = $1 AND a > (SELECT v FROM u WHERE u.id = t.id + 1 ORDER BY 1) ORDER BY id";

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 1), (2, 20, 1), (3, 30, 2), (4, 40, 2), (5, 50, 1), (6, 60, 2)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    // id 3 holds NULL, id 4 has no row
    db.execute(
        "INSERT INTO u VALUES (1, 5), (2, 25), (3, NULL), (5, 60), (6, 7)",
        (),
    )
    .unwrap();
    db
}

/// The first column of each row, and the prepared subquery runs it took
fn ids<P: Params>(db: &Database, sql: &str, params: P) -> (Vec<i64>, usize) {
    let before = prepared_scalar_runs();
    let found = db
        .query(sql, params)
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    (found, prepared_scalar_runs() - before)
}

fn error<P: Params>(db: &Database, sql: &str, params: P) -> (String, usize) {
    let before = prepared_scalar_runs();
    let err = match db.query(sql, params) {
        Err(e) => e.to_string(),
        Ok(rows) => rows
            .map(|row| row.map(|_| ()))
            .collect::<Result<Vec<()>, _>>()
            .expect_err("expected an error")
            .to_string(),
    };
    (err, prepared_scalar_runs() - before)
}

/// The line EXPLAIN gives on the prepared family
fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

const ALL_GENERAL: [GeneralPath; 7] = [
    GeneralPath::CorrelatedScan,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::PushdownConstant,
    GeneralPath::ProjectionEvaluator,
    GeneralPath::OrderByEvaluator,
];

fn general_hits() -> Vec<usize> {
    ALL_GENERAL.iter().map(|p| general_path_hits(*p)).collect()
}

#[test]
fn test_one_plan_answers_each_parameter_and_outer_row() {
    let _guard = serial();
    let db = setup("prepared_params");
    let built = prepared_selects_built();
    for (param, s1, s2) in [
        (1, vec![1], vec![5]),
        (2, vec![6], vec![]),
        (1, vec![1], vec![5]),
    ] {
        let (found, runs) = ids(&db, S1, (param,));
        assert_eq!(found, s1, "S1 with b = {param}");
        assert_eq!(runs, 3, "S1 with b = {param} runs once per outer row");
        let (found, runs) = ids(&db, S2, (param,));
        assert_eq!(found, s2, "S2 with b = {param}");
        assert_eq!(runs, 3, "S2 with b = {param} runs once per outer row");
        assert_eq!(ids(&db, S1_OLD, (param,)), (s1, 0), "S1 on the old path");
        assert_eq!(ids(&db, S2_OLD, (param,)), (s2, 0), "S2 on the old path");
    }
    assert_eq!(
        prepared_selects_built() - built,
        2,
        "S1 and S2 are prepared once each across three runs"
    );
}

#[test]
fn test_schema_change_prepares_again() {
    let _guard = serial();
    let db = setup("prepared_schema");
    assert_eq!(ids(&db, S1, (1,)).0, vec![1]);
    db.execute("ALTER TABLE t ADD COLUMN z INTEGER", ())
        .unwrap();
    assert_eq!(ids(&db, S1, (1,)), (vec![1], 3), "a column added outside");
    db.execute("ALTER TABLE u ADD COLUMN w INTEGER", ())
        .unwrap();
    assert_eq!(ids(&db, S1, (1,)), (vec![1], 3), "a column added inside");
    db.execute("ALTER TABLE t DROP COLUMN z", ()).unwrap();
    assert_eq!(ids(&db, S1, (1,)), (vec![1], 3), "the outer column dropped");
    db.execute("DROP TABLE u", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (2, 0), (5, 0)", ())
        .unwrap();
    assert_eq!(
        ids(&db, S1, (1,)),
        (vec![2, 5], 3),
        "the inner table recreated"
    );
}

#[test]
fn test_a_table_rename_prepares_again() {
    let _guard = serial();
    let db = setup("prepared_rename");
    let statement = db.prepare(S1).unwrap();
    let run = || -> Vec<i64> {
        statement
            .query((1,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    };
    assert_eq!(run(), vec![1]);
    let built = prepared_selects_built();
    db.execute("ALTER TABLE t RENAME TO t_moved", ()).unwrap();
    db.execute("ALTER TABLE t_moved RENAME TO t", ()).unwrap();
    assert_eq!(run(), vec![1]);
    assert_eq!(
        prepared_selects_built() - built,
        1,
        "a rename changes the schema the plan was prepared for"
    );
}

#[test]
fn test_used_columns_moved_or_retyped_prepare_again() {
    let _guard = serial();
    let db = Database::open("memory://prepared_moved").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 0, 10, 1), (2, 0, 20, 1), (5, 0, 50, 1)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 5), (2, 25), (5, 60)", ())
        .unwrap();
    assert_eq!(ids(&db, S1, (1,)), (vec![1], 3));
    for change in [
        "ALTER TABLE t DROP COLUMN x",
        "ALTER TABLE t MODIFY COLUMN a FLOAT",
        "ALTER TABLE u MODIFY COLUMN v FLOAT",
    ] {
        db.execute(change, ()).unwrap();
        let built = prepared_selects_built();
        assert_eq!(ids(&db, S1, (1,)), (vec![1], 3), "after {change}");
        assert_eq!(
            prepared_selects_built() - built,
            1,
            "{change} prepares again"
        );
    }
}

#[test]
fn test_a_new_text_prepares_again() {
    let _guard = serial();
    let db = setup("prepared_new_text");
    let built = prepared_selects_built();
    for i in 0..3 {
        let sql = format!(
            "SELECT id AS id{i} FROM t WHERE b = 1 AND a > (SELECT v FROM u WHERE u.id = t.id) ORDER BY id"
        );
        assert_eq!(ids(&db, &sql, ()), (vec![1], 3));
    }
    assert_eq!(
        prepared_selects_built() - built,
        3,
        "each new text is prepared"
    );
}

#[test]
fn test_scalar_subquery_results() {
    let _guard = serial();
    let db = setup("prepared_results");
    let missing = "SELECT id FROM t WHERE (SELECT v FROM u WHERE u.id = t.id) IS NULL ORDER BY id";
    assert_eq!(
        ids(&db, missing, ()),
        (vec![3, 4], 6),
        "no row and a NULL value both read NULL"
    );
    let present =
        "SELECT id FROM t WHERE (SELECT v FROM u WHERE u.id = t.id) IS NOT NULL ORDER BY id";
    assert_eq!(ids(&db, present, ()), (vec![1, 2, 5, 6], 6));

    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO m VALUES (1, 1, 0), (2, 1, 0), (3, 2, 0)", ())
        .unwrap();
    let multi = "SELECT id FROM t WHERE a > (SELECT v FROM m WHERE m.k = t.id) ORDER BY id";
    let (err, runs) = error(&db, multi, ());
    assert!(err.contains("more than one row"), "{err}");
    assert!(runs >= 1, "the error came from the prepared path");
    let multi_old =
        "SELECT id FROM t WHERE a > (SELECT v FROM m WHERE m.k = t.id ORDER BY 1) ORDER BY id";
    assert!(error(&db, multi_old, ()).0.contains("more than one row"));
}

#[test]
fn test_a_transaction_sees_its_own_writes() {
    let _guard = serial();
    for level in ["READ COMMITTED", "SNAPSHOT"] {
        let db = setup(&format!("prepared_own_writes_{}", level.len()));
        db.execute(&format!("SET isolation_level = '{level}'"), ())
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE u SET v = 0 WHERE id = 2", ()).unwrap();
        assert_eq!(
            ids(&db, S1, (1,)),
            (vec![1, 2], 3),
            "{level}: the uncommitted update is read"
        );
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(ids(&db, S1, (1,)), (vec![1], 3), "{level}: rolled back");
    }
}

const COMMIT: &str = "UPDATE u SET v = 0 WHERE id IN (2, 5)";

/// The rows of S1 when another handle commits `COMMIT` as the prepared
/// subquery of the first outer row is about to run
fn prepared_with_commit_inside(db: &Database) -> (Vec<i64>, usize) {
    let writer = db.clone();
    let done = Rc::new(Cell::new(false));
    let flag = Rc::clone(&done);
    before_prepared_scalar(move || {
        writer.execute(COMMIT, ()).unwrap();
        flag.set(true);
    });
    let found = ids(db, S1, (1,));
    assert!(done.get(), "the commit ran inside the statement");
    found
}

/// The rows of S1_OLD when another handle commits `COMMIT` as the
/// subquery of the first outer row opens its table
fn old_with_commit_inside(db: &Database) -> Vec<i64> {
    let writer = db.clone();
    let done = Rc::new(Cell::new(false));
    let flag = Rc::clone(&done);
    // The first scan opened is the outer table's; the second, the first subquery's
    after_select_table_opened(move || {
        after_select_table_opened(move || {
            writer.execute(COMMIT, ()).unwrap();
            flag.set(true);
        });
    });
    let found = ids(db, S1_OLD, (1,)).0;
    assert!(done.get(), "the commit ran inside the statement");
    found
}

#[test]
fn test_every_outer_row_reads_the_statement_view() {
    let _guard = serial();
    for level in ["READ COMMITTED", "SNAPSHOT"] {
        let db = setup(&format!("prepared_one_view_{}", level.len()));
        db.execute(&format!("SET isolation_level = '{level}'"), ())
            .unwrap();
        assert_eq!(
            prepared_with_commit_inside(&db),
            (vec![1], 3),
            "{level}: ids 2 and 5 are read after the commit and must not see it"
        );
        assert_eq!(
            ids(&db, S1, (1,)).0,
            vec![1, 2, 5],
            "{level}: the next statement sees the commit"
        );
    }
}

#[test]
fn test_old_path_reads_one_view() {
    let _guard = serial();
    let db = setup("prepared_one_view_old");
    db.execute("SET isolation_level = 'SNAPSHOT'", ()).unwrap();
    assert_eq!(old_with_commit_inside(&db), vec![1], "old path");
    assert_eq!(ids(&db, S1_OLD, (1,)).0, vec![1, 2, 5]);
}

#[test]
fn test_a_snapshot_transaction_keeps_its_view_between_statements() {
    let _guard = serial();
    let db = setup("prepared_snapshot_between");
    let writer = db.clone();
    db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    assert_eq!(ids(&db, S1, (1,)), (vec![1], 3));
    writer.execute(COMMIT, ()).unwrap();
    assert_eq!(
        ids(&db, S1, (1,)),
        (vec![1], 3),
        "the second SELECT does not see a commit made after the first"
    );
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(ids(&db, S1, (1,)).0, vec![1, 2, 5]);
}

/// Column names, `(id, d)` rows, and the prepared projections the query ran
fn pairs(db: &Database, sql: &str) -> (Vec<String>, Vec<(i64, i64)>, usize) {
    let before = prepared_projection_runs();
    let rows = db.query(sql, ()).unwrap();
    let columns = rows.columns().to_vec();
    let found = rows
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect();
    (columns, found, prepared_projection_runs() - before)
}

#[test]
fn test_projection_and_order_by_run_prepared() {
    let _guard = serial();
    let db = setup("prepared_projection");
    let present = "(SELECT v FROM u WHERE u.id = t.id) IS NOT NULL";
    let present_old = "(SELECT v FROM u WHERE u.id = t.id ORDER BY 1) IS NOT NULL";
    let names = vec!["id".to_string(), "d".to_string()];
    for (order, expected) in [
        ("a DESC", vec![(6, 120), (5, 100), (2, 40), (1, 20)]),
        ("a % 3, id", vec![(6, 120), (1, 20), (2, 40), (5, 100)]),
        ("d DESC", vec![(6, 120), (5, 100), (2, 40), (1, 20)]),
        ("id", vec![(1, 20), (2, 40), (5, 100), (6, 120)]),
    ] {
        let sql = format!("SELECT id, a * 2 AS d FROM t WHERE {present} ORDER BY {order}");
        assert_eq!(
            pairs(&db, &sql),
            (names.clone(), expected.clone(), 1),
            "ORDER BY {order} on the prepared projection"
        );
        let old = format!("SELECT id, a * 2 AS d FROM t WHERE {present_old} ORDER BY {order}");
        assert_eq!(
            pairs(&db, &old),
            (names.clone(), expected, 0),
            "ORDER BY {order} on the old path"
        );
    }
}

#[test]
fn test_star_limit_and_offset() {
    let _guard = serial();
    let db = setup("prepared_star");
    let present = "(SELECT v FROM u WHERE u.id = t.id) IS NOT NULL";
    let rows = |sql: &str| -> (Vec<String>, Vec<Vec<i64>>) {
        let rows = db.query(sql, ()).unwrap();
        let columns = rows.columns().to_vec();
        let found = rows
            .map(|row| {
                let row = row.unwrap();
                (0..3).map(|i| row.get::<i64>(i).unwrap()).collect()
            })
            .collect();
        (columns, found)
    };
    let all: Vec<String> = ["id", "a", "b"].iter().map(|s| s.to_string()).collect();
    assert_eq!(
        rows(&format!(
            "SELECT * FROM t WHERE {present} ORDER BY id DESC LIMIT 2"
        )),
        (all.clone(), vec![vec![6, 60, 2], vec![5, 50, 1]])
    );
    assert_eq!(
        rows(&format!(
            "SELECT t.* FROM t WHERE {present} ORDER BY id LIMIT 2 OFFSET 1"
        )),
        (all.clone(), vec![vec![2, 20, 1], vec![5, 50, 1]])
    );
    let before = prepared_scalar_runs();
    let (_, found) = rows(&format!("SELECT * FROM t WHERE {present} LIMIT 1 OFFSET 1"));
    assert_eq!(found, vec![vec![2, 20, 1]]);
    assert_eq!(
        prepared_scalar_runs() - before,
        2,
        "without ORDER BY the scan stops once LIMIT rows past OFFSET passed"
    );
    let param = db
        .query(
            &format!("SELECT id FROM t WHERE {present} ORDER BY id LIMIT $1 OFFSET $2"),
            (2, 1),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(param, vec![2, 5], "LIMIT and OFFSET as parameters");
}

#[test]
fn test_expressions_see_the_transaction_id() {
    let _guard = serial();
    let db = setup("prepared_txn_id");
    let sql = "SELECT id, CURRENT_TRANSACTION_ID() FROM t WHERE b = 1 \
               AND a > (SELECT v FROM u WHERE u.id = t.id) \
               AND CURRENT_TRANSACTION_ID() IS NOT NULL ORDER BY a";
    let old = "SELECT id, CURRENT_TRANSACTION_ID() FROM t WHERE b = 1 \
               AND a > (SELECT v FROM u WHERE u.id = t.id ORDER BY 1) \
               AND CURRENT_TRANSACTION_ID() IS NOT NULL ORDER BY a";
    db.execute("BEGIN", ()).unwrap();
    let scalar_before = prepared_scalar_runs();
    let (_, found, projections) = pairs(&db, sql);
    assert!(
        prepared_scalar_runs() > scalar_before,
        "the WHERE ran prepared"
    );
    assert_eq!(projections, 1, "the projection ran prepared");
    let (_, expected, _) = pairs(&db, old);
    assert_eq!(expected.len(), 1, "the old path finds id 1");
    assert_eq!(found, expected, "both paths read the same transaction id");
    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn test_inner_values_that_read_no_inner_column() {
    let _guard = serial();
    let db = setup("prepared_inner_values");
    let sql = "SELECT id FROM t WHERE b = 1 \
               AND a < (SELECT v FROM u WHERE u.id = t.id AND u.v <= $1 * 10) ORDER BY id";
    for (param, expected) in [(1, vec![]), (3, vec![2]), (6, vec![2, 5]), (1, vec![])] {
        assert_eq!(ids(&db, sql, (param,)), (expected, 3), "v <= {param} * 10");
    }
    let odd = "SELECT id FROM t WHERE b = 1 \
               AND a > (SELECT v FROM u WHERE u.id = t.id AND u.v % 2 = 1) ORDER BY id";
    assert_eq!(
        ids(&db, odd, ()),
        (vec![1], 3),
        "a filter pushdown cannot take"
    );
}

#[test]
fn test_volatile_values_run_per_inner_row() {
    let _guard = serial();
    let db = setup("prepared_volatile");
    db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE q (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for i in 0..200 {
        db.execute("INSERT INTO r VALUES ($1, 0)", (i,)).unwrap();
        db.execute("INSERT INTO q VALUES ($1, 0)", (i,)).unwrap();
    }
    let per_row =
        "SELECT id FROM r WHERE 0.5 > (SELECT RANDOM() FROM q WHERE q.id = r.id) ORDER BY id";
    let (found, runs) = ids(&db, per_row, ());
    assert_eq!(runs, 200, "one subquery run per outer row");
    assert!(
        !found.is_empty() && found.len() < 200,
        "RANDOM() is drawn per row, not once for the statement: {} of 200",
        found.len()
    );

    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO m VALUES (1, 1, 1), (2, 1, 2), (3, 1, 3), (4, 1, 4), (5, 1, 5), (6, 2, 3)",
        (),
    )
    .unwrap();
    let filter = "SELECT id FROM t WHERE id = 1 \
                  AND a > (SELECT v FROM m WHERE m.k = t.id AND RANDOM() < 2 AND m.v + 0 = 3)";
    let draws = random_draws();
    assert_eq!(ids(&db, filter, ()), (vec![1], 1));
    assert_eq!(
        random_draws() - draws,
        5,
        "RANDOM() is drawn once per inner row reaching the filter, the five with k = 1"
    );
}

#[test]
fn test_pushed_comparisons_bind_values() {
    let _guard = serial();
    let db = setup("prepared_pushed");
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, y INTEGER, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO w VALUES (1, 0, 1, 5), (2, 0, 2, 25), (3, 0, 5, 60)",
        (),
    )
    .unwrap();
    let by_k =
        "SELECT id FROM t WHERE b = 1 AND a > (SELECT v FROM w WHERE w.k = t.id) ORDER BY id";
    assert_eq!(ids(&db, by_k, ()), (vec![1], 3));
    db.execute("ALTER TABLE w DROP COLUMN y", ()).unwrap();
    assert_eq!(
        ids(&db, by_k, ()),
        (vec![1], 3),
        "k moved one position left"
    );

    let lossless = "SELECT id FROM t WHERE b = 1 \
                    AND a > (SELECT v FROM w WHERE w.k = t.id + 0.0) ORDER BY id";
    assert_eq!(
        ids(&db, lossless, ()),
        (vec![1], 3),
        "1.0 matches INTEGER 1"
    );
    let fraction = "SELECT id FROM t WHERE b = 1 \
                    AND a > (SELECT v FROM w WHERE w.k = t.id + 0.5) ORDER BY id";
    assert_eq!(
        ids(&db, fraction, ()),
        (vec![], 3),
        "1.5 matches no INTEGER"
    );

    let bounded = "SELECT id FROM t WHERE b = 1 \
                   AND a > (SELECT v FROM w WHERE w.k = t.id AND w.v < $1) ORDER BY id";
    assert_eq!(ids(&db, bounded, (10,)), (vec![1], 3), "v < 10");
    assert_eq!(ids(&db, bounded, (1,)), (vec![], 3), "v < 1");
    assert_eq!(ids(&db, bounded, (None::<i64>,)), (vec![], 3), "v < NULL");
    assert_eq!(ids(&db, bounded, (10,)), (vec![1], 3), "v < 10 again");
}

#[test]
fn test_a_reused_comparison_takes_each_value() {
    let _guard = serial();
    let db = Database::open("memory://prepared_rebind").unwrap();
    db.execute(
        "CREATE TABLE o (id INTEGER PRIMARY KEY, s TEXT, a INTEGER)",
        (),
    )
    .unwrap();
    // One run binds the same comparison to '1', NULL, 'abc', '1', '2' in turn
    db.execute(
        "INSERT INTO o VALUES (1, '1', 10), (2, NULL, 10), (3, 'abc', 10), (4, '1', 10), (5, '2', 10)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE q (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO q VALUES (1, 1, 0), (2, 2, 20)", ())
        .unwrap();
    let sql = "SELECT id FROM o WHERE a > (SELECT v FROM q WHERE q.k = o.s) ORDER BY id";
    let old = "SELECT id FROM o WHERE a > (SELECT v FROM q WHERE q.k = o.s ORDER BY 1) ORDER BY id";
    let expected = ids(&db, old, ()).0;
    assert_eq!(expected, vec![1, 4], "the old path's answer");
    assert_eq!(
        ids(&db, sql, ()),
        (expected, 5),
        "rows 2 and 3 do not keep row 1's value, row 4 does not keep row 3's"
    );
}

const NESTED: &str = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id \
                      AND u.v < (SELECT w.x FROM w WHERE w.k = u.id AND w.z = t.b AND w.x > $1)) \
                      ORDER BY id";
// The same question with an inner ORDER BY position in the middle scope, out of the family
const NESTED_OLD: &str = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id \
                          AND u.v < (SELECT w.x FROM w WHERE w.k = u.id AND w.z = t.b AND w.x > $1) \
                          ORDER BY 1) ORDER BY id";

fn nested_setup(name: &str) -> Database {
    let db = setup(name);
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER, z INTEGER, x INTEGER)",
        (),
    )
    .unwrap();
    // k reads u.id one scope out, z reads t.b two scopes out
    db.execute(
        "INSERT INTO w VALUES (1, 1, 1, 100), (2, 2, 2, 100), (3, 6, 2, 50), (4, 5, 1, 10)",
        (),
    )
    .unwrap();
    db
}

#[test]
fn test_nested_scopes_read_each_depth() {
    let _guard = serial();
    let db = nested_setup("prepared_nested");
    let built = prepared_selects_built();
    for (param, expected) in [(0, vec![1, 6]), (60, vec![1]), (0, vec![1, 6])] {
        let (found, runs) = ids(&db, NESTED, (param,));
        assert_eq!(found, expected, "w.x > {param}");
        assert_eq!(
            runs, 11,
            "six middle runs, one per outer row, and five innermost, one per u row found"
        );
        assert_eq!(
            ids(&db, NESTED_OLD, (param,)),
            (expected, 0),
            "old path, w.x > {param}"
        );
    }
    assert_eq!(prepared_selects_built() - built, 1, "prepared once");
}

#[test]
fn test_nested_scope_names_and_schema() {
    let _guard = serial();
    let db = nested_setup("prepared_nested_names");
    let aliased = "SELECT o.id FROM t AS o WHERE o.a > (SELECT m.v FROM u AS m WHERE m.id = o.id \
                   AND m.v < (SELECT i.x FROM w AS i WHERE i.k = m.id AND i.z = o.b)) ORDER BY o.id";
    assert_eq!(
        ids(&db, aliased, ()),
        (vec![1, 6], 11),
        "aliases on every scope"
    );

    for change in [
        "ALTER TABLE w ADD COLUMN q INTEGER",
        "ALTER TABLE w DROP COLUMN q",
    ] {
        db.execute(change, ()).unwrap();
        let built = prepared_selects_built();
        assert_eq!(ids(&db, NESTED, (0,)), (vec![1, 6], 11), "after {change}");
        assert_eq!(
            prepared_selects_built() - built,
            1,
            "{change} prepares again"
        );
    }
}

#[test]
fn test_nested_null_and_errors_cross_scopes() {
    let _guard = serial();
    let db = nested_setup("prepared_nested_errors");
    db.execute("UPDATE w SET x = NULL WHERE id = 1", ())
        .unwrap();
    assert_eq!(
        ids(&db, NESTED, (0,)),
        (vec![6], 11),
        "a NULL innermost value drops t1"
    );
    db.execute("INSERT INTO w VALUES (5, 6, 2, 70)", ())
        .unwrap();
    let (err, runs) = error(&db, NESTED, (0,));
    assert!(err.contains("more than one row"), "{err}");
    assert!(runs >= 1, "raised on the prepared path");
    assert!(error(&db, NESTED_OLD, (0,)).0.contains("more than one row"));
}

#[test]
fn test_one_value_reading_two_enclosing_scopes() {
    let _guard = serial();
    let db = setup("prepared_two_scopes");
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        (),
    )
    .unwrap();
    // k = u.id + t.b: (1, b 1) gives 2, (6, b 2) gives 8; 3 = 2 + 1 and
    // 7 = 6 + 1 are what a wrong t.b would find
    db.execute(
        "INSERT INTO s VALUES (1, 2, 100), (2, 8, 50), (3, 3, 0), (4, 7, 0)",
        (),
    )
    .unwrap();
    let sql = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id \
               AND u.v < (SELECT s.x FROM s WHERE s.k = u.id + t.b)) ORDER BY id";
    let old = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id \
               AND u.v < (SELECT s.x FROM s WHERE s.k = u.id + t.b) ORDER BY 1) ORDER BY id";
    assert_eq!(
        ids(&db, sql, ()),
        (vec![1, 6], 11),
        "u.id + t.b on the prepared path"
    );
    assert_eq!(ids(&db, old, ()), (vec![1, 6], 0), "old path");
}

#[test]
fn test_names_resolve_by_the_nearest_scope() {
    let _guard = serial();
    let db = setup("prepared_scopes");
    let aliased = "SELECT o.id FROM t AS o WHERE o.b = 1 \
                   AND o.a > (SELECT x.v FROM u AS x WHERE x.id = o.id) ORDER BY o.id";
    assert_eq!(ids(&db, aliased, ()), (vec![1], 3), "aliases on both sides");
    let cased =
        "SELECT id FROM T WHERE b = 1 AND a > (SELECT v FROM U WHERE U.ID = T.ID) ORDER BY id";
    assert_eq!(ids(&db, cased, ()), (vec![1], 3), "names in another case");

    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO w VALUES (1, 100), (5, 1)", ())
        .unwrap();
    let own_first =
        "SELECT id FROM t WHERE b = 1 AND a > (SELECT a FROM w WHERE w.id = t.id) ORDER BY id";
    assert_eq!(ids(&db, own_first, ()), (vec![5], 3), "a bare a is w.a");

    let self_ref = "SELECT id FROM t WHERE b = 1 \
                    AND a < (SELECT a FROM t AS t2 WHERE t2.id = t.id + 1) ORDER BY id";
    assert_eq!(
        ids(&db, self_ref, ()),
        (vec![1, 2, 5], 3),
        "the outer table again"
    );

    // The inner alias t names w, so t.a inside reads w.a and t.id reads w.id
    let shadowing = "SELECT id FROM t WHERE b = 1 \
                     AND a > (SELECT t.a FROM w AS t WHERE t.id = 5) ORDER BY id";
    assert_eq!(
        ids(&db, shadowing, ()),
        (vec![1, 2, 5], 1),
        "an inner alias shadows the outer table of that name"
    );

    // b exists only in t: the unqualified name reaches the enclosing scope
    let outward =
        "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id AND b = 1) ORDER BY id";
    assert_eq!(
        ids(&db, outward, ()),
        (vec![1], 6),
        "an unqualified outer column resolves"
    );
}

#[test]
fn test_name_errors_are_the_statements() {
    let _guard = serial();
    let db = setup("prepared_name_errors");
    let built = prepared_selects_built();
    let unknown = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE nosuch.id = t.id) ORDER BY id";
    let (err, runs) = error(&db, unknown, ());
    assert!(err.contains("unknown table or alias 'nosuch'"), "{err}");
    assert_eq!(runs, 0);
    let missing = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.nosuch) ORDER BY id";
    let (err, _) = error(&db, missing, ());
    assert!(err.contains("t.nosuch"), "{err}");
    let bare = "SELECT id FROM t WHERE a > (SELECT nosuch FROM u WHERE u.id = t.id) ORDER BY id";
    let (err, _) = error(&db, bare, ());
    assert!(err.contains("nosuch"), "{err}");
    let outer_unknown =
        "SELECT x.id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id) ORDER BY id";
    let (err, _) = error(&db, outer_unknown, ());
    assert!(err.contains("unknown table or alias 'x'"), "{err}");
    assert_eq!(
        prepared_selects_built() - built,
        0,
        "nothing that fails to bind is prepared"
    );
    let explain = db.query(&format!("EXPLAIN {unknown}"), ());
    assert!(
        explain.is_err_and(|e| e.to_string().contains("unknown table or alias 'nosuch'")),
        "the statement fails before any path reads it"
    );
}

#[test]
fn test_named_parameters_bind_by_name_in_any_block() {
    let _guard = serial();
    let db = setup("prepared_named");
    let sql = "SELECT id FROM t WHERE b = :b \
               AND a > (SELECT v FROM u WHERE u.id = t.id AND u.v < :cap) ORDER BY id";
    let mut params = stoolap::NamedParams::new();
    params.insert("b", 1);
    params.insert("cap", 10);
    let found: Vec<i64> = db
        .query_named(sql, params)
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(found, vec![1], "v < 10 for b = 1");

    let mut partial = stoolap::NamedParams::new();
    partial.insert("b", 1);
    let err = match db.query_named(sql, partial) {
        Err(e) => e.to_string(),
        Ok(rows) => rows
            .map(|row| row.map(|_| ()))
            .collect::<Result<Vec<()>, _>>()
            .expect_err("a missing named parameter is an error")
            .to_string(),
    };
    assert!(err.contains(":cap"), "{err}");
}

#[test]
fn test_quoting_is_unchanged() {
    let _guard = serial();
    let db = Database::open("memory://prepared_quoting").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'abc'), (2, 'xyz')", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 'abc'), (2, 'abc')", ())
        .unwrap();
    let quoted_column =
        "SELECT id FROM t WHERE \"S\" = (SELECT u.\"S\" FROM u WHERE u.\"ID\" = t.id) ORDER BY id";
    assert_eq!(
        ids(&db, quoted_column, ()),
        (vec![1], 2),
        "a quoted name resolves by its lowercase form"
    );
    let literal =
        "SELECT id FROM t WHERE s = (SELECT \"abc\" FROM u WHERE u.id = t.id) ORDER BY id";
    assert_eq!(
        ids(&db, literal, ()),
        (vec![1], 2),
        "a quoted name naming no column reads as TEXT"
    );
}

#[test]
fn test_uncorrelated_blocks_run_once_when_needed() {
    let _guard = serial();
    let db = setup("prepared_uncorrelated");
    let once =
        "SELECT id FROM t WHERE b = 1 AND a + 0 > (SELECT v FROM u WHERE u.id = 2) ORDER BY id";
    assert_eq!(ids(&db, once, ()), (vec![5], 1), "once for the execution");
    assert_eq!(
        ids(&db, once, ()),
        (vec![5], 1),
        "again in the next execution"
    );

    let untaken = "SELECT id FROM t WHERE CASE WHEN b = 9 \
                   THEN a > (SELECT v FROM u WHERE u.id = 2) ELSE TRUE END ORDER BY id";
    assert_eq!(
        ids(&db, untaken, ()),
        (vec![1, 2, 3, 4, 5, 6], 0),
        "not at all in a CASE arm no row takes"
    );
    let short = "SELECT id FROM t WHERE id > 0 OR a > (SELECT v FROM u WHERE u.id = 2) ORDER BY id";
    assert_eq!(
        ids(&db, short, ()),
        (vec![1, 2, 3, 4, 5, 6], 0),
        "not at all on the right of an OR every row short-circuits"
    );

    // A column compared with the value is narrowed by it before the scan
    let pushed =
        "SELECT id FROM t WHERE b = 9 AND a > (SELECT v FROM u WHERE u.id = 2) ORDER BY id";
    assert_eq!(
        ids(&db, pushed, ()),
        (vec![], 1),
        "a top-level comparison with a column takes the value to plan the scan"
    );

    let nested = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id \
                  AND u.v < (SELECT v FROM u WHERE u.id = 5)) ORDER BY id";
    assert_eq!(
        ids(&db, nested, ()),
        (vec![1, 6], 7),
        "six correlated runs and one of the uncorrelated block nested in them"
    );

    let draws = random_draws();
    let volatile = "SELECT id FROM t WHERE a > (SELECT v + RANDOM() * 0 FROM u WHERE u.id <= 2 \
                    AND RANDOM() < 2 AND u.v + 0 > 20) ORDER BY id";
    assert_eq!(ids(&db, volatile, ()), (vec![3, 4, 5, 6], 1));
    assert_eq!(
        random_draws() - draws,
        3,
        "RANDOM() per inner row reaching the filter, two, and once for the row kept"
    );
}

#[test]
fn test_scope_chain_depth() {
    let _guard = serial();
    // The general path recurses per scope and needs more than a test thread's stack
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(scope_chain_depth)
        .unwrap()
        .join()
        .unwrap();
}

fn scope_chain_depth() {
    let db = Database::open("memory://prepared_depth").unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    // `scopes` scopes in the innermost chain; the innermost reads the statement's row
    let chain = |scopes: usize| -> String {
        let mut sql = format!(
            "(SELECT s{n}.v FROM d AS s{n} WHERE s{n}.id = s0.id)",
            n = scopes - 1
        );
        for n in (1..scopes - 1).rev() {
            sql = format!(
                "(SELECT s{n}.v FROM d AS s{n} WHERE s{n}.id = s{p}.id AND s{n}.v = {sql})",
                p = n - 1
            );
        }
        format!("SELECT id FROM d AS s0 WHERE s0.v = {sql} ORDER BY id")
    };
    let in_family = chain(32);
    assert_eq!(prepared_line(&db, &in_family), "Prepared: yes");
    let built = prepared_selects_built();
    let (found, runs) = ids(&db, &in_family, ());
    assert_eq!(found, vec![1, 2, 3]);
    assert_eq!(runs, 31 * 3, "31 blocks for each of three rows");
    assert_eq!(prepared_selects_built() - built, 1);

    let too_deep = chain(33);
    assert_eq!(
        prepared_line(&db, &too_deep),
        "Prepared: no (more than 32 scopes in a chain)"
    );
    let built = prepared_selects_built();
    assert_eq!(
        ids(&db, &too_deep, ()),
        (vec![1, 2, 3], 0),
        "the old path answers"
    );
    assert_eq!(prepared_selects_built() - built, 0);
}

#[test]
fn test_classifier_rules() {
    let _guard = serial();
    let db = setup("prepared_classifier");
    db.execute("CREATE VIEW uv AS SELECT id, v FROM u", ())
        .unwrap();
    let sq = "(SELECT v FROM u WHERE u.id = t.id)";
    let yes = [
        format!("SELECT id FROM t WHERE a > {sq}"),
        "SELECT id FROM t AS o WHERE a > (SELECT v FROM u WHERE u.id = o.id)".to_string(),
        format!("SELECT * FROM t WHERE a > {sq}"),
        format!("SELECT t.* FROM t WHERE a > {sq}"),
        format!("SELECT id, a + 1 AS s, CASE WHEN b = 1 THEN 'x' END FROM t WHERE a > {sq}"),
        format!("SELECT id FROM t WHERE a > {sq} ORDER BY a DESC NULLS LAST, id"),
        format!("SELECT id FROM t WHERE a > {sq} LIMIT 2 OFFSET $1"),
        format!("SELECT id FROM t WHERE CASE WHEN {sq} IS NULL THEN 0 ELSE 1 END = 1"),
        format!("SELECT id FROM t WHERE {sq} BETWEEN 1 AND 10"),
        format!("SELECT id FROM t WHERE {sq} IN (5, 7)"),
        format!("SELECT id FROM t WHERE b IN ({sq}, 1)"),
        format!("SELECT id FROM t WHERE CAST({sq} AS TEXT) LIKE '5%'"),
        format!("SELECT id FROM t WHERE {sq} IS DISTINCT FROM a"),
        format!("SELECT id FROM t WHERE ABS({sq}) > 1 AND NOT (b = 2) AND -a < 0"),
        format!("SELECT id FROM t WHERE 'x' || CAST({sq} AS TEXT) = 'x5'"),
        "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = 2)".to_string(),
        format!("SELECT id FROM t WHERE a > (SELECT {sq} FROM u WHERE u.id = 1)"),
        "SELECT id FROM t WHERE a > (SELECT v FROM uv WHERE uv.id = t.id)".to_string(),
        format!("SELECT id, {sq} FROM t WHERE a > {sq}"),
        format!("SELECT id FROM t WHERE a > {sq} ORDER BY {sq}"),
        format!("SELECT id, {sq} FROM t"),
        format!("SELECT id FROM t WHERE b = 1 ORDER BY {sq}"),
        format!("SELECT id, {sq} FROM t ORDER BY {sq} DESC, id LIMIT 3 OFFSET $1"),
        format!("SELECT DISTINCT id FROM t WHERE a > {sq}"),
        format!("SELECT DISTINCT ON (b) id FROM t WHERE a > {sq} ORDER BY b, id"),
        format!("SELECT id FROM t WHERE a > {sq} UNION SELECT id FROM u"),
        "SELECT id FROM t WHERE a > (SELECT q.v FROM (SELECT id, v FROM u) AS q WHERE q.id = t.id)"
            .to_string(),
        "SELECT id FROM t WHERE a > (SELECT MAX(q.v) FROM (SELECT v FROM u WHERE u.id <> t.b) AS q \
         WHERE q.v = t.id)"
            .to_string(),
        format!("SELECT t.id FROM t JOIN (SELECT * FROM u) u ON u.id = t.id WHERE a > {sq}"),
        "SELECT id FROM t WHERE a > (SELECT v FROM u JOIN t AS x ON x.id = u.id WHERE u.id = t.id)"
            .to_string(),
        "SELECT id FROM t WHERE a > (SELECT MAX(q.v) FROM (SELECT u.v FROM u JOIN uv \
         ON uv.id = u.id) AS q WHERE q.v = t.id)"
            .to_string(),
    ];
    let rows = |sql: &str| -> Vec<String> {
        let mut found: Vec<String> = db
            .query(sql, (1,))
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                // A NULL is NULL whatever type it carries
                let values: Vec<_> = (0..row.len())
                    .map(|i| match row.get_value(i) {
                        Some(stoolap::core::Value::Null(_)) => None,
                        other => other,
                    })
                    .collect();
                format!("{values:?}")
            })
            .collect();
        found.sort();
        found
    };
    for sql in &yes {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        // An inner ORDER BY position keeps the same question out of the family
        let old = sql
            .replace(" = t.id)", " = t.id ORDER BY 1)")
            .replace("u.id = 2)", "u.id = 2 ORDER BY 1)");
        let old = old.replace(" = o.id)", " = o.id ORDER BY 1)");
        assert!(
            prepared_line(&db, &old).starts_with("Prepared: no"),
            "{old}"
        );
        let runs = prepared_scalar_runs();
        assert_eq!(rows(sql), rows(&old), "{sql} answers as the old path");
        assert!(prepared_scalar_runs() > runs, "{sql} ran prepared");
    }
    let no = [
        (
            "SELECT id FROM t WHERE (id, a) IN (SELECT id, v, a FROM u)".to_string(),
            "a subquery block selecting another number of values than its row holds",
        ),
        (
            format!("SELECT id FROM t WHERE a > {sq} LIMIT 1 + 1"),
            "LIMIT or OFFSET that is not a literal or a parameter",
        ),
        (
            format!(
                "SELECT id, ROW_NUMBER() OVER w FROM t WHERE a > {sq} WINDOW w AS (ORDER BY id)"
            ),
            "a named window",
        ),
        (
            format!(
                "SELECT t.id FROM t JOIN u ON u.id = t.id JOIN u AS w ON w.id = t.id \
                 WHERE a > {sq}"
            ),
            "a side that is itself a join",
        ),
        (
            "SELECT id FROM t WHERE a > (SELECT MAX(v) FROM u WHERE u.id = t.id ORDER BY 1)"
                .to_string(),
            "a grouped subquery block with DISTINCT ON, ORDER BY, LIMIT or OFFSET",
        ),
        (
            "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id ORDER BY 1 LIMIT 1)"
                .to_string(),
            "an ORDER BY position in a subquery block",
        ),
        (
            "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id LIMIT t.a)".to_string(),
            "a subquery block's LIMIT or OFFSET that is not a literal or a parameter",
        ),
        (
            "SELECT id FROM t WHERE a > (SELECT v, id FROM u WHERE u.id = t.id)".to_string(),
            "a subquery block without exactly one SELECT expression",
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM u JOIN uv ON uv.id = u.id) \
             AS q WHERE q.v = t.a)"
                .to_string(),
            "a derived table selecting * over a join",
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT v FROM u LIMIT t.a) AS q)"
                .to_string(),
            "a derived table's LIMIT or OFFSET naming a column",
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT v FROM u WHERE v IN \
             (SELECT x.v FROM u AS x LIMIT t.a)) AS q)"
                .to_string(),
            "a derived table's LIMIT or OFFSET naming a column",
        ),
        (
            format!(
                "SELECT id FROM t WHERE a > {sq} UNION SELECT id FROM u \
                 ORDER BY (SELECT v FROM u WHERE u.id = 2)"
            ),
            "an ORDER BY subquery over a set operation",
        ),
    ];
    for (sql, reason) in &no {
        assert_eq!(
            prepared_line(&db, sql),
            format!("Prepared: no ({reason})"),
            "{sql}"
        );
    }
    assert_eq!(
        prepared_line(&db, "SELECT id FROM t WHERE b = 1 ORDER BY id"),
        "",
        "a statement without a subquery is no candidate, so EXPLAIN says nothing"
    );
}

#[test]
fn test_in_family_statements_reach_no_general_path_mechanism() {
    let _guard = serial();
    let db = nested_setup("prepared_removed");
    let mut named = stoolap::NamedParams::new();
    named.insert("cap", 50);
    let corpus = [
        S1,
        S2,
        NESTED,
        "SELECT id, a * 2 AS d FROM t WHERE (SELECT v FROM u WHERE u.id = t.id) IS NOT NULL ORDER BY a % 3, id",
        "SELECT * FROM t WHERE b = $1 + 0 AND a > (SELECT v FROM u WHERE u.id = t.id + 1 - 1)",
        "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = 2) ORDER BY id LIMIT 2",
        "SELECT id FROM t WHERE CASE WHEN a > 20 THEN (SELECT v FROM u WHERE u.id = t.id) END IS NULL",
        "SELECT UPPER(CAST(a AS TEXT)) FROM t WHERE (SELECT v FROM u WHERE u.id = t.id) BETWEEN 1 AND 30",
        "SELECT id FROM t WHERE a < (SELECT MAX(v) + t.a FROM u WHERE u.id = t.id)",
        "SELECT id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.v > t.a HAVING COUNT(*) > 0) > 1",
    ];
    let built = prepared_selects_built();
    let runs = prepared_scalar_runs();
    let before = general_hits();
    for sql in corpus {
        db.query(sql, (1,)).unwrap().for_each(|row| {
            row.unwrap();
        });
    }
    db.query_named(
        "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id AND u.v < :cap)",
        named,
    )
    .unwrap()
    .for_each(|row| {
        row.unwrap();
    });
    assert_eq!(
        general_hits(),
        before,
        "every general path counter stays where it was: {ALL_GENERAL:?}"
    );
    assert!(prepared_selects_built() > built, "the corpus was prepared");
    assert!(prepared_scalar_runs() > runs, "the corpus ran prepared");

    let before = general_hits();
    ids(&db, S1_OLD, (1,));
    assert_ne!(general_hits(), before, "the old path moves the counters");
}

/// Per comparison: whether the row passes it, its negation, or neither
fn tri(db: &Database, cmp: &str) -> Option<bool> {
    let pass = |predicate: String| -> bool {
        let sql = format!(
            "SELECT id FROM c WHERE id = 1 AND (SELECT id FROM k WHERE k.id = c.id) = 1 AND {predicate}"
        );
        let (found, runs) = ids(db, &sql, ());
        assert!(runs >= 1, "{sql} ran prepared");
        !found.is_empty()
    };
    let truth = pass(cmp.to_string());
    let negated = pass(format!("NOT ({cmp})"));
    assert!(!(truth && negated), "{cmp} and its negation both pass");
    if truth {
        Some(true)
    } else if negated {
        Some(false)
    } else {
        None
    }
}

#[test]
fn test_comparison_contract() {
    let _guard = serial();
    let db = Database::open("memory://prepared_contract").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, s TEXT, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE k (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO k VALUES (1)", ()).unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, s TEXT)",
        (),
    )
    .unwrap();
    // Inner values read through a subquery, outer through the statement's row
    let inner = |column: &str| format!("(SELECT p.{column} FROM p WHERE p.id = c.id)");
    let cases: Vec<(&str, &str, &str, &str, Option<bool>)> = vec![
        // (outer row values, outer side, operator, inner value, expected)
        ("1, NULL, '1', NULL", "c.n", "=", "i", None),
        ("1, NULL, '1', NULL", "c.i", "=", "n_null", None),
        (
            "9007199254740993, NULL, NULL, NULL",
            "c.i",
            "=",
            "f_big",
            Some(false),
        ),
        (
            "9007199254740993, NULL, NULL, NULL",
            "c.i",
            "<>",
            "f_big",
            Some(true),
        ),
        (
            "9007199254740993, NULL, NULL, NULL",
            "c.i",
            ">",
            "f_big",
            Some(true),
        ),
        ("1, NULL, NULL, NULL", "c.i", "=", "s_1", Some(true)),
        ("-2, NULL, NULL, NULL", "c.i", "=", "s_neg", Some(true)),
        ("1, 1.5, NULL, NULL", "c.f", "=", "s_15", Some(true)),
        ("1, NULL, NULL, NULL", "c.i", "=", "s_abc", None),
        ("1, NULL, NULL, NULL", "c.i", "<>", "s_abc", None),
        ("1, NULL, NULL, NULL", "c.i", "<", "s_empty", None),
        ("1, NULL, NULL, NULL", "c.i", "=", "s_space", None),
        ("1, NULL, 'a', NULL", "c.s", "<", "s_b", Some(true)),
        ("1, NULL, 'a', NULL", "c.s", "=", "s_b", Some(false)),
    ];
    let values = |name: &str| -> &str {
        match name {
            "i" => "1, 1, NULL, NULL",
            "n_null" => "1, NULL, NULL, NULL",
            "f_big" => "1, NULL, 9007199254740992.0, NULL",
            "s_1" => "1, NULL, NULL, '1'",
            "s_neg" => "1, NULL, NULL, '-2'",
            "s_15" => "1, NULL, NULL, '1.5'",
            "s_abc" => "1, NULL, NULL, 'abc'",
            "s_empty" => "1, NULL, NULL, ''",
            "s_space" => "1, NULL, NULL, ' 1'",
            "s_b" => "1, NULL, NULL, 'b'",
            other => panic!("no inner value {other}"),
        }
    };
    let column = |name: &str| match name.split('_').next() {
        Some("n") | Some("i") => "i",
        Some("f") => "f",
        _ => "s",
    };
    let flip = |op: &'static str| -> &'static str {
        match op {
            "<" => ">",
            ">" => "<",
            "<=" => ">=",
            ">=" => "<=",
            other => other,
        }
    };
    for (outer, outer_side, op, inner_value, expected) in cases {
        db.execute("DELETE FROM c", ()).unwrap();
        db.execute("DELETE FROM p", ()).unwrap();
        db.execute(&format!("INSERT INTO c VALUES (1, {outer})"), ())
            .unwrap();
        db.execute(
            &format!("INSERT INTO p VALUES ({})", values(inner_value)),
            (),
        )
        .unwrap();
        let inner_side = inner(column(inner_value));
        let cmp = format!("{outer_side} {op} {inner_side}");
        assert_eq!(
            tri(&db, &cmp),
            expected,
            "{cmp} with {outer} / {inner_value}"
        );
        let flipped = format!("{inner_side} {} {outer_side}", flip(op));
        assert_eq!(tri(&db, &flipped), expected, "flipped: {flipped}");
        let wrapped = [
            format!("CASE WHEN {cmp} THEN TRUE WHEN NOT ({cmp}) THEN FALSE END"),
            format!("({cmp}) = TRUE"),
            format!("COALESCE({cmp}, NULL)"),
        ];
        for wrapped in &wrapped {
            assert_eq!(tri(&db, wrapped), expected, "wrapped: {wrapped}");
        }
        // Pushed: the inner column against the outer value, taken by the scan
        let pushed = format!(
            "(SELECT p.id FROM p WHERE p.{} {} {outer_side} AND p.id = c.id) IS NOT NULL",
            column(inner_value),
            flip(op)
        );
        assert_eq!(
            tri(&db, &pushed).unwrap_or(false),
            expected == Some(true),
            "pushed: {pushed}"
        );
    }

    db.execute("DELETE FROM c", ()).unwrap();
    db.execute("DELETE FROM p", ()).unwrap();
    db.execute("INSERT INTO c VALUES (1, 1, NULL, NULL, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1, NULL, NULL, 'abc')", ())
        .unwrap();
    let s = inner("s");
    let i = inner("i");
    for (cmp, expected) in [
        (format!("c.i IS DISTINCT FROM {s}"), Some(true)),
        (format!("{s} IS DISTINCT FROM c.i"), Some(true)),
        (format!("c.i IS NOT DISTINCT FROM {s}"), Some(false)),
        (format!("c.n IS NOT DISTINCT FROM {i}"), Some(true)),
        (format!("c.n IS DISTINCT FROM {i}"), Some(false)),
        (format!("c.i IS DISTINCT FROM {i}"), Some(true)),
    ] {
        assert_eq!(tri(&db, &cmp), expected, "{cmp}");
    }
}

#[test]
fn test_an_inner_layout_changed_after_the_plan_was_taken() {
    let _guard = serial();
    let db = Database::open("memory://prepared_inner_layout").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, pad INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, 0, 5)", ()).unwrap();
    let sql = "SELECT id FROM t WHERE a > (SELECT v FROM u WHERE u.id = t.id)";
    assert_eq!(ids(&db, sql, ()).0, vec![1], "the plan is prepared");
    let writer = db.clone();
    let done = Rc::new(Cell::new(false));
    let flag = Rc::clone(&done);
    after_prepared_plan_taken(move || {
        writer.execute("ALTER TABLE u DROP COLUMN pad", ()).unwrap();
        flag.set(true);
    });
    let found = ids(&db, sql, ()).0;
    assert!(done.get(), "the column dropped inside the statement");
    assert_eq!(found, vec![1], "v is read where the table now holds it");
}

#[test]
fn test_a_whole_pushed_inner_where_is_not_checked_again() {
    let _guard = serial();
    let db = Database::open("memory://prepared_pushed_whole").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, b BOOLEAN, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, TRUE, '2024-01-02 03:04:05')", ())
        .unwrap();
    for (inner, param) in [("u.b = $1", "true"), ("u.ts = $1", "2024-01-02 03:04:05")] {
        let sql = format!("SELECT t.id FROM t WHERE t.id = (SELECT u.id FROM u WHERE {inner})");
        let old = format!(
            "SELECT t.id FROM t WHERE t.id = (SELECT u.id FROM u WHERE {inner} ORDER BY 1)"
        );
        let expected = ids(&db, &old, (param,)).0;
        assert_eq!(
            expected,
            vec![1],
            "the general path keeps the row for {inner}"
        );
        let (found, runs) = ids(&db, &sql, (param,));
        assert!(runs >= 1, "{sql} ran prepared");
        assert_eq!(found, expected, "{inner} with TEXT {param}");
    }
}

#[test]
fn test_a_missing_positional_parameter_is_an_error() {
    let _guard = serial();
    let db = setup("prepared_missing_positional");
    let sql = "SELECT id FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id AND u.id = $1)";
    let (err, runs) = error(&db, sql, ());
    assert!(err.contains("requires 1 parameters"), "{err}");
    assert_eq!(runs, 0, "nothing ran with a slot standing in for $1");
}

#[test]
fn test_a_named_parameter_under_an_alias_must_have_a_value() {
    let _guard = serial();
    let db = setup("prepared_named_alias");
    let sql = "SELECT :missing AS x FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id)";
    let (err, _) = error(&db, sql, ());
    assert!(err.contains(":missing"), "{err}");
}

#[test]
fn test_a_volatile_escape_runs_per_inner_row() {
    let _guard = serial();
    let db = setup("prepared_volatile_escape");
    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, keep BOOLEAN, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO m VALUES (1, 1, TRUE, 1), (2, 1, TRUE, 2), (3, 1, TRUE, 3), (4, 1, TRUE, 4), (5, 1, TRUE, 5)",
        (),
    )
    .unwrap();
    let sql = "SELECT id FROM t WHERE id = 1 AND a > (SELECT v FROM m WHERE m.k = t.id \
               AND m.keep = ('x' LIKE 'x' ESCAPE CASE WHEN RANDOM() < 2 THEN '!' ELSE '?' END) \
               AND m.v + 0 = 3)";
    let draws = random_draws();
    assert_eq!(ids(&db, sql, ()), (vec![1], 1));
    assert_eq!(
        random_draws() - draws,
        5,
        "the ESCAPE is drawn once per inner row, not once per call"
    );
}

#[test]
fn test_an_order_by_key_reading_an_alias_is_prepared() {
    let _guard = serial();
    let db = setup("prepared_order_alias");
    let sql = "SELECT id, a AS x FROM t WHERE (SELECT v FROM u WHERE u.id = t.id) IS NOT NULL \
               ORDER BY x % 3, id";
    let old = "SELECT id, a AS x FROM t WHERE (SELECT v FROM u WHERE u.id = t.id ORDER BY 1) \
               IS NOT NULL ORDER BY x % 3, id";
    let expected = pairs(&db, old);
    let before = general_path_hits(GeneralPath::OrderByEvaluator);
    let found = pairs(&db, sql);
    assert_eq!(
        general_path_hits(GeneralPath::OrderByEvaluator) - before,
        0,
        "the sort reads the key the projection computed"
    );
    assert_eq!((found.0, found.1), (expected.0, expected.1));
}

#[test]
fn test_a_boolean_or_timestamp_column_reads_a_text_value_as_the_scan_does() {
    let _guard = serial();
    let db = Database::open("memory://prepared_scan_compare").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, TRUE)", ()).unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, b BOOLEAN, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, TRUE, '2024-01-02 03:04:05')", ())
        .unwrap();
    for (inner, param) in [
        ("u.b = $1 AND TRUE", "true"),
        ("u.b = $1 OR u.id = 99", "true"),
        ("NOT (u.b <> $1)", "true"),
        ("u.b IN ($1)", "true"),
        ("u.ts = $1 AND TRUE", "2024-01-02 03:04:05"),
        ("u.ts BETWEEN $1 AND $1", "2024-01-02 03:04:05"),
    ] {
        let sql = format!("SELECT t.id FROM t WHERE t.id = (SELECT u.id FROM u WHERE {inner})");
        let (found, runs) = ids(&db, &sql, (param,));
        assert!(runs >= 1, "{sql} ran prepared");
        assert_eq!(
            found,
            vec![1],
            "{inner} with TEXT {param}, as the scan compares it"
        );
    }
    let outer =
        "SELECT t.id FROM t WHERE t.b = $1 AND t.id = (SELECT u.id FROM u WHERE u.id = t.id)";
    assert_eq!(
        ids(&db, outer, ("true",)).0,
        vec![1],
        "the statement's own WHERE follows the same rule"
    );
}

#[test]
fn test_an_order_by_alias_key_reads_the_projected_value() {
    let _guard = serial();
    let db = setup("prepared_order_alias_value");
    let sql = "SELECT RANDOM() AS r FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) \
               ORDER BY r + 0";
    let draws = random_draws();
    let found: Vec<f64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<f64>(0).unwrap())
        .collect();
    assert_eq!(found.len(), 5, "one row per u row");
    assert_eq!(
        random_draws() - draws,
        5,
        "RANDOM() runs once per row, for r"
    );
    assert!(
        found.windows(2).all(|w| w[0] <= w[1]),
        "the rows are ordered by their own r: {found:?}"
    );
}

#[test]
fn test_an_order_by_alias_inside_cast() {
    let _guard = serial();
    let db = setup("prepared_order_alias_cast");
    let cast = "SELECT a AS x FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) \
                ORDER BY CAST(x AS INTEGER) DESC";
    let before = general_path_hits(GeneralPath::OrderByEvaluator);
    let found: Vec<i64> = db
        .query(cast, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(found, vec![60, 50, 30, 20, 10], "an alias inside CAST");
    assert_eq!(general_path_hits(GeneralPath::OrderByEvaluator) - before, 0);
}

mod registered_volatile {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use stoolap::core::{Result, Value};
    use stoolap::executor::Executor;
    use stoolap::functions::registry::FunctionRegistry;
    use stoolap::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
    };
    use stoolap::storage::mvcc::engine::MVCCEngine;

    static TICKS: AtomicUsize = AtomicUsize::new(0);

    /// Returns its argument and counts its calls; registered non-deterministic
    #[derive(Default)]
    struct Tick;

    impl ScalarFunction for Tick {
        fn name(&self) -> &str {
            "TICK"
        }
        fn info(&self) -> FunctionInfo {
            FunctionInfo::new(
                "TICK",
                FunctionType::Scalar,
                "test volatile",
                FunctionSignature::new(
                    FunctionDataType::Integer,
                    vec![FunctionDataType::Any],
                    1,
                    1,
                ),
            )
            .non_deterministic()
        }
        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            TICKS.fetch_add(1, Ordering::Relaxed);
            Ok(args[0].clone())
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(Tick)
        }
    }

    /// TICK under the name of the built-in NOW
    #[derive(Default)]
    struct NowTick;

    impl ScalarFunction for NowTick {
        fn name(&self) -> &str {
            "NOW"
        }
        fn info(&self) -> FunctionInfo {
            let mut info = Tick.info();
            info.name = "NOW".to_string();
            info
        }
        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            Tick.evaluate(args)
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(NowTick)
        }
    }

    #[test]
    fn test_a_registered_volatile_function_runs_per_inner_row() {
        let _guard = super::serial();
        let registry = FunctionRegistry::new();
        registry.register_scalar::<Tick>();
        assert_eq!(
            calls_per_run(registry, "TICK"),
            5,
            "TICK runs once per inner row reaching the filter, not once per call"
        );
    }

    #[test]
    fn test_a_registered_function_named_like_a_clock_runs_per_inner_row() {
        let _guard = super::serial();
        let registry = FunctionRegistry::new();
        registry.register_scalar::<NowTick>();
        assert_eq!(
            calls_per_run(registry, "NOW"),
            5,
            "a NOW the registry holds non-deterministic is not the built-in clock"
        );
    }

    /// Calls of `function` in one run that finds the third of five inner rows
    fn calls_per_run(registry: FunctionRegistry, function: &str) -> usize {
        let mut engine = MVCCEngine::in_memory();
        stoolap::storage::traits::Engine::open(&mut engine).unwrap();
        let executor = Executor::with_function_registry(Arc::new(engine), Arc::new(registry));
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
            "INSERT INTO t VALUES (1, 10)",
            "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
            "INSERT INTO m VALUES (1, 1, 1), (2, 1, 2), (3, 1, 3), (4, 1, 4), (5, 1, 5)",
        ] {
            executor.execute(sql).unwrap();
        }
        let before = TICKS.load(Ordering::Relaxed);
        let mut rows = executor
            .execute(&format!(
                "SELECT id FROM t WHERE id = 1 \
                 AND a > (SELECT v FROM m WHERE m.k = t.id AND m.v = {function}(t.id) + 2)"
            ))
            .unwrap();
        let mut found = Vec::new();
        while rows.next() {
            found.push(rows.row().get(0).cloned());
        }
        assert_eq!(found, vec![Some(Value::Integer(1))]);
        TICKS.load(Ordering::Relaxed) - before
    }
}

#[test]
fn test_a_subquery_value_against_a_boolean_column_reads_as_the_column() {
    let _guard = serial();
    let db = Database::open("memory://prepared_subquery_typed").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, b BOOLEAN)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, TRUE)", ()).unwrap();
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO w VALUES (1, 'true')", ()).unwrap();
    let sql = "SELECT t.id FROM t WHERE t.id = \
               (SELECT u.id FROM u WHERE u.b = (SELECT w.s FROM w WHERE w.id = 1))";
    let old = "SELECT t.id FROM t WHERE t.id = \
               (SELECT u.id FROM u WHERE u.b = (SELECT w.s FROM w WHERE w.id = 1) ORDER BY 1)";
    assert_eq!(ids(&db, old, ()).0, vec![1], "the general path");
    let (found, runs) = ids(&db, sql, ());
    assert!(runs >= 1, "ran prepared");
    assert_eq!(found, vec![1], "the subquery's TEXT read as BOOLEAN");
}

#[test]
fn test_a_dotted_output_alias_is_no_qualifier() {
    let _guard = serial();
    let db = Database::open("memory://prepared_dotted_alias").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (100, 9)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1), (100)", ()).unwrap();
    let sql =
        "SELECT a AS \"t.id\", a AS x FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) \
               ORDER BY x + t.id";
    let (found, _) = ids(&db, sql, ());
    assert_eq!(
        found,
        vec![10, 9],
        "t.id is the table's id: 10 + 1 before 9 + 100"
    );
}

#[test]
fn test_json_comparisons_give_one_answer() {
    let _guard = serial();
    let db = Database::open("memory://prepared_json_compare").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, j JSON)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '{\"x\":1}')", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, j JSON)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, '{\"x\":1}')", ())
        .unwrap();
    let param = "{\"x\":1}";
    let mut answers = Vec::new();
    for inner in ["u.j = $1", "(u.j = $1) = TRUE"] {
        let sql = format!("SELECT t.id FROM t WHERE t.id = (SELECT u.id FROM u WHERE {inner})");
        let (found, runs) = ids(&db, &sql, (param,));
        assert!(runs >= 1, "{sql} ran prepared");
        answers.push((inner, found));
    }
    let outer =
        "SELECT t.id FROM t WHERE t.j = $1 AND t.id = (SELECT u.id FROM u WHERE u.id = t.id)";
    answers.push(("statement", ids(&db, outer, (param,)).0));
    for (shape, found) in &answers {
        assert_eq!(
            found,
            &Vec::<i64>::new(),
            "{shape}: a JSON column against a TEXT is unknown"
        );
    }
    let as_json = "SELECT t.id FROM t WHERE t.id = \
                   (SELECT u.id FROM u WHERE u.j = CAST($1 AS JSON))";
    assert_eq!(
        ids(&db, as_json, (param,)).0,
        vec![1],
        "a JSON column equals the same JSON value"
    );
}

#[test]
fn test_a_quoted_dotted_alias_reads_the_projected_value() {
    let _guard = serial();
    let db = Database::open("memory://prepared_quoted_dotted_alias").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1), (2)", ()).unwrap();
    let sql = "SELECT a AS \"x.y\", id FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) \
               ORDER BY CAST(\"x.y\" AS INTEGER) DESC, id";
    let (found, runs) = ids(&db, sql, ());
    assert!(runs >= 1, "ran prepared");
    assert_eq!(
        found,
        vec![20, 10],
        "\"x.y\" is the alias, not a qualified name"
    );
}

#[test]
fn test_a_repeated_alias_reads_its_last_projection() {
    let _guard = serial();
    let db = Database::open("memory://prepared_repeated_alias").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1), (2)", ()).unwrap();
    let sql = "SELECT a AS x, -a AS x FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) \
               ORDER BY CAST(x AS INTEGER)";
    let (found, runs) = ids(&db, sql, ());
    assert!(runs >= 1, "ran prepared");
    assert_eq!(found, vec![20, 10], "x in an expression is the last x, -a");
}

#[test]
fn test_an_alias_binding_cannot_meet_a_column_name() {
    let _guard = serial();
    let db = Database::open("memory://prepared_alias_binding_name").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, \"\u{1}0\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 99), (2, 20, 0)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1), (2)", ()).unwrap();
    let sql = "SELECT a AS z FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) \
               ORDER BY CAST(z AS INTEGER) DESC";
    let (found, runs) = ids(&db, sql, ());
    assert!(runs >= 1, "ran prepared");
    assert_eq!(
        found,
        vec![20, 10],
        "z is a, whatever the table's columns are named"
    );
}

#[test]
fn test_a_numeric_column_reads_no_boolean_value() {
    let _guard = serial();
    let db = Database::open("memory://prepared_numeric_boolean").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, i INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
    for (inner, expected) in [
        ("i = TRUE", vec![]),
        ("i = TRUE OR id = 99", vec![]),
        ("(i = TRUE) = TRUE", vec![]),
        ("i <> TRUE", vec![1]),
    ] {
        let sql = format!("SELECT id FROM t WHERE id = (SELECT id FROM u WHERE {inner})");
        let (found, runs) = ids(&db, &sql, ());
        assert!(runs >= 1, "{sql} ran prepared");
        assert_eq!(found, expected, "{inner}: TRUE is not the INTEGER 1");
    }
}

#[test]
fn test_order_by_columns_do_not_show_beside_a_star() {
    let _guard = serial();
    let db = Database::open("memory://prepared_star_order").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1)", ()).unwrap();
    for order in ["1", "t.id", "id + 0"] {
        let sql = format!(
            "SELECT * FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) ORDER BY {order}"
        );
        let rows = db.query(&sql, ()).unwrap();
        assert_eq!(
            rows.columns().to_vec(),
            vec!["id".to_string()],
            "ORDER BY {order}"
        );
        let widths: Vec<usize> = rows.map(|row| row.unwrap().len()).collect();
        assert_eq!(widths, vec![1], "ORDER BY {order}");
    }
}

#[test]
fn test_a_quoted_dotted_column_name_is_the_column() {
    let _guard = serial();
    let db = Database::open("memory://prepared_dotted_column").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER, \"x.y\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, 7, 7)", ()).unwrap();
    let sql = "SELECT t.id FROM t WHERE t.id = \
               (SELECT u.id FROM u WHERE u.id = t.id AND u.v = ABS(\"x.y\"))";
    let (found, runs) = ids(&db, sql, ());
    assert!(runs >= 1, "ran prepared");
    assert_eq!(found, vec![1], "\"x.y\" is u's column");
}

#[test]
fn test_a_text_column_reads_a_boolean_value_as_the_scan_does() {
    let _guard = serial();
    let db = Database::open("memory://prepared_text_boolean").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 'true')", ()).unwrap();
    for inner in ["s = $1", "s = $1 OR id = 99"] {
        let sql = format!("SELECT id FROM t WHERE id = (SELECT id FROM u WHERE {inner})");
        let (found, runs) = ids(&db, &sql, (true,));
        assert!(runs >= 1, "{sql} ran prepared");
        assert_eq!(
            found,
            vec![1],
            "{inner}: BOOLEAN true read as the TEXT 'true'"
        );
    }
}

#[test]
fn test_columns_in_an_in_list_or_a_pattern_stay_columns() {
    let _guard = serial();
    let db = Database::open("memory://prepared_columns_in_lists").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, other INTEGER, s TEXT, pat TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 'abc', 'a%')", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, other INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
    for (shape, sql) in [
        (
            "statement IN",
            "SELECT id FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) AND id IN (other)",
        ),
        (
            "statement LIKE",
            "SELECT id FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) AND s LIKE pat",
        ),
        (
            "subquery IN",
            "SELECT id FROM t WHERE id = (SELECT u.id FROM u WHERE u.id = t.id AND u.id IN (u.other))",
        ),
    ] {
        let (found, runs) = ids(&db, sql, ());
        assert!(runs >= 1, "{shape} ran prepared");
        assert_eq!(found, vec![1], "{shape}: a column, not its name as TEXT");
    }
}

#[test]
fn test_null_safe_equality_reads_the_value_as_equality_does() {
    let _guard = serial();
    let db = Database::open("memory://prepared_null_safe_typed").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, i INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
    for (inner, expected) in [
        ("(u.i = TRUE) = TRUE", vec![]),
        ("u.i IS NOT DISTINCT FROM TRUE", vec![]),
        ("u.i IS DISTINCT FROM TRUE", vec![1]),
        ("u.i IS DISTINCT FROM 'abc'", vec![1]),
    ] {
        let sql = format!("SELECT id FROM t WHERE id = (SELECT id FROM u WHERE {inner})");
        let (found, runs) = ids(&db, &sql, ());
        assert!(runs >= 1, "{sql} ran prepared");
        assert_eq!(found, expected, "{inner}");
    }
}

#[test]
fn test_a_star_shows_every_column_whatever_the_names() {
    let _guard = serial();
    let db = Database::open("memory://prepared_star_dotted_names").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, \"t.b\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 20)", ()).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1)", ()).unwrap();
    for star in ["*", "t.*"] {
        let sql = format!(
            "SELECT {star} FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) ORDER BY id + 0"
        );
        let rows = db.query(&sql, ()).unwrap();
        assert_eq!(rows.columns().len(), 3, "{star}: {:?}", rows.columns());
        let found: Vec<Vec<i64>> = rows
            .map(|row| {
                let row = row.unwrap();
                (0..row.len()).map(|i| row.get::<i64>(i).unwrap()).collect()
            })
            .collect();
        assert_eq!(found, vec![vec![1, 10, 20]], "{star}");
    }
}

#[test]
fn test_rewritten_columns_never_read_as_literals() {
    let _guard = serial();
    let db = Database::open("memory://prepared_rewritten_literals").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, other INTEGER, s TEXT, pat TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (7, 7, 'abc', 'a%')", ())
        .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, \"true\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (7, 7)", ()).unwrap();
    for (shape, sql) in [
        (
            "u.true",
            "SELECT t.id FROM t WHERE t.id = (SELECT u.true FROM u WHERE u.id = t.id)",
        ),
        (
            "u.\"true\"",
            "SELECT t.id FROM t WHERE t.id = (SELECT u.\"true\" FROM u WHERE u.id = t.id)",
        ),
        (
            "t.\"other\" in IN",
            "SELECT id FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) AND id IN (t.\"other\")",
        ),
        (
            "t.\"pat\" in LIKE",
            "SELECT id FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) AND s LIKE t.\"pat\"",
        ),
        (
            "\"other\" in IN",
            "SELECT id FROM t WHERE id = (SELECT id FROM u WHERE u.id = t.id) AND id IN (\"other\")",
        ),
    ] {
        let (found, runs) = ids(&db, sql, ());
        assert!(runs >= 1, "{shape} ran prepared");
        assert_eq!(found, vec![7], "{shape}: a column, not a literal");
    }
}

#[test]
fn test_a_derived_table_runs_prepared_over_its_rows() {
    let _guard = serial();
    let db = setup("prepared_derived");
    const DERIVED: &str = "SELECT s.id FROM (SELECT id, a, b FROM t) AS s \
                           WHERE s.b = $1 AND s.a > (SELECT v FROM u WHERE u.id = s.id) ORDER BY s.id";
    assert_eq!(prepared_line(&db, DERIVED), "Prepared: yes");
    let built = prepared_selects_built();
    let general = general_hits();
    for (param, want) in [(1, vec![1]), (2, vec![6]), (1, vec![1])] {
        let (found, runs) = ids(&db, DERIVED, (param,));
        assert_eq!(found, want, "b = {param}");
        assert_eq!(runs, 3, "b = {param} runs once per derived row");
    }
    assert_eq!(
        prepared_selects_built() - built,
        1,
        "prepared once across three runs"
    );
    assert_eq!(general_hits(), general, "no general path ran");
}

#[test]
fn test_a_derived_table_repeating_a_name_reads_star_by_position() {
    let _guard = serial();
    let db = setup("prepared_derived_repeated_star");
    const STAR: &str = "SELECT * FROM (SELECT id, 10 AS x, 20 AS x FROM t) AS s \
                        WHERE s.id < 3 AND EXISTS (SELECT 1 FROM u WHERE u.id = s.id) ORDER BY 1";
    assert_eq!(prepared_line(&db, STAR), "Prepared: yes");
    let rows: Vec<(i64, i64, i64)> = db
        .query(STAR, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get(0).unwrap(),
                row.get(1).unwrap(),
                row.get(2).unwrap(),
            )
        })
        .collect();
    assert_eq!(rows, vec![(1, 10, 20), (2, 10, 20)], "each x by position");
}

#[test]
fn test_a_name_a_derived_table_repeats_is_ambiguous() {
    let _guard = serial();
    let db = setup("prepared_derived_repeated_name");
    const FROM: &str = "FROM (SELECT id, 10 AS x, 20 AS x FROM t) AS s";
    const EXISTS: &str = "EXISTS (SELECT 1 FROM u WHERE u.id = s.id)";
    for (site, sql) in [
        (
            "WHERE",
            format!("SELECT s.id {FROM} WHERE s.x = 20 AND {EXISTS}"),
        ),
        (
            "bare WHERE",
            format!("SELECT s.id {FROM} WHERE x = 20 AND {EXISTS}"),
        ),
        ("select list", format!("SELECT s.x {FROM} WHERE {EXISTS}")),
        (
            "ORDER BY",
            format!("SELECT s.id {FROM} WHERE {EXISTS} ORDER BY s.x"),
        ),
        (
            "ORDER BY expression over an alias of the name",
            format!("SELECT s.id AS x {FROM} WHERE {EXISTS} ORDER BY x + 0"),
        ),
        (
            "correlated block",
            format!("SELECT s.id {FROM} WHERE EXISTS (SELECT 1 FROM u WHERE u.id = s.x)"),
        ),
    ] {
        let (err, _) = error(&db, &sql, ());
        assert!(err.contains("is ambiguous"), "{site}: {err}");
    }
    for (site, sql) in [
        (
            "bare ORDER BY alias",
            format!("SELECT s.id AS x {FROM} WHERE {EXISTS} ORDER BY x DESC"),
        ),
        (
            "ORDER BY expression over an alias no column has",
            format!("SELECT s.id AS y {FROM} WHERE {EXISTS} ORDER BY 0 - y"),
        ),
    ] {
        assert_eq!(ids(&db, &sql, ()).0, vec![6, 5, 3, 2, 1], "{site}");
    }
}

#[test]
fn test_a_view_runs_prepared_over_its_rows() {
    let _guard = serial();
    let db = setup("prepared_view");
    db.execute("CREATE VIEW tv AS SELECT id, a, b FROM t", ())
        .unwrap();
    const VIEW: &str = "SELECT s.id FROM tv AS s \
                        WHERE s.b = $1 AND s.a > (SELECT v FROM u WHERE u.id = s.id) ORDER BY s.id";
    assert_eq!(prepared_line(&db, VIEW), "Prepared: yes");
    let built = prepared_selects_built();
    let general = general_hits();
    for (param, want) in [(1, vec![1]), (2, vec![6]), (1, vec![1])] {
        let (found, runs) = ids(&db, VIEW, (param,));
        assert_eq!(found, want, "b = {param}");
        assert_eq!(runs, 3, "b = {param} runs once per view row");
    }
    assert_eq!(
        prepared_selects_built() - built,
        1,
        "prepared once across three runs"
    );
    assert_eq!(general_hits(), general, "no general path ran");
}

#[test]
fn test_a_redefined_view_answers_from_its_new_body() {
    let _guard = serial();
    let db = setup("prepared_view_redefined");
    const VIEW: &str =
        "SELECT s.id FROM tv AS s WHERE s.a > (SELECT v FROM u WHERE u.id = s.id) ORDER BY s.id";
    let built = prepared_selects_built();
    for (body, want, prepared) in [
        ("SELECT id, a FROM t WHERE b = 1", vec![1], 1),
        ("SELECT id, a FROM t WHERE b = 2", vec![6], 2),
        ("SELECT a, id FROM t", vec![1, 6], 3),
    ] {
        db.execute("DROP VIEW IF EXISTS tv", ()).unwrap();
        db.execute(&format!("CREATE VIEW tv AS {body}"), ())
            .unwrap();
        assert_eq!(ids(&db, VIEW, ()).0, want, "{body}");
        assert_eq!(
            prepared_selects_built() - built,
            prepared,
            "{body}: a view defined again prepares the plan again"
        );
    }
}

#[test]
fn test_a_view_reading_itself_stops_at_the_nesting_limit() {
    let _guard = serial();
    let db = setup("prepared_view_loop");
    let view = "SELECT id FROM tv WHERE id > (SELECT MIN(id) FROM u)";
    assert!(db
        .execute(&format!("CREATE VIEW tv AS {view}"), ())
        .is_err());
    // Only the engine stores one, as a database an older release wrote can hold
    db.engine()
        .create_view("tv", view.to_string(), None, false)
        .unwrap();
    let (err, _) = error(
        &db,
        "SELECT id FROM tv WHERE id > (SELECT MIN(id) FROM u)",
        (),
    );
    assert!(err.contains("Maximum view nesting depth"), "{err}");
}

#[test]
fn test_a_view_comes_before_a_table_of_its_name() {
    let _guard = serial();
    let db = setup("prepared_view_and_table");
    db.execute(
        "CREATE VIEW tv AS SELECT id + 100 AS id FROM u WHERE id = 1",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE pair (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO pair VALUES (2)", ()).unwrap();
    // Only the engine renames a table into a view's name, as a database
    // an older release wrote can hold one
    db.engine().rename_table("pair", "tv").unwrap();
    const VIEW: &str = "SELECT id FROM tv WHERE id > (SELECT MIN(id) FROM u)";
    assert_eq!(prepared_line(&db, VIEW), "Prepared: yes");
    assert_eq!(
        ids(&db, "SELECT id FROM tv", ()).0,
        vec![101],
        "the general path"
    );
    assert_eq!(ids(&db, VIEW, ()).0, vec![101], "the prepared path");
}

#[test]
fn test_a_values_list_runs_prepared_over_its_rows() {
    let _guard = serial();
    let db = setup("prepared_values");
    const VALUES: &str = "SELECT s.id FROM (VALUES (1, $1), (2, 20), (5, 50)) AS s(id, a) \
                          WHERE s.a > (SELECT v FROM u WHERE u.id = s.id) ORDER BY s.id";
    assert_eq!(prepared_line(&db, VALUES), "Prepared: yes");
    let built = prepared_selects_built();
    let general = general_hits();
    for (param, want) in [(10, vec![1]), (1, vec![]), (10, vec![1])] {
        let (found, runs) = ids(&db, VALUES, (param,));
        assert_eq!(found, want, "a = {param} in the first row");
        assert_eq!(runs, 3, "a = {param} runs once per row");
    }
    assert_eq!(
        prepared_selects_built() - built,
        1,
        "prepared once across three runs"
    );
    assert_eq!(general_hits(), general, "no general path ran");
}

#[test]
fn test_a_values_list_evaluates_every_value_under_a_limit() {
    let _guard = serial();
    let db = setup("prepared_values_limit");
    const FROM: &str = "FROM (VALUES (1), (9223372036854775807 + 1)) AS s(x)";
    let general = format!("SELECT x {FROM} WHERE x > 0 LIMIT 1");
    let prepared = format!("SELECT x {FROM} WHERE x > 0 AND EXISTS (SELECT 1 FROM u) LIMIT 1");
    assert_eq!(prepared_line(&db, &prepared), "Prepared: yes");
    let (general_err, _) = error(&db, &general, ());
    let (prepared_err, _) = error(&db, &prepared, ());
    assert!(general_err.contains("overflow"), "{general_err}");
    assert_eq!(
        prepared_err, general_err,
        "both paths evaluate the second row"
    );
}

#[test]
fn test_a_values_row_holding_a_subquery_is_not_prepared() {
    let _guard = serial();
    let db = setup("prepared_values_row_subquery");
    assert_eq!(
        prepared_line(
            &db,
            "SELECT n FROM (VALUES ((SELECT MAX(v) FROM u)), (1)) AS s(n) WHERE n > (SELECT 0)"
        ),
        "Prepared: no (a VALUES row holding a subquery)"
    );
}

#[test]
fn test_a_cte_runs_prepared_over_its_rows_and_keeps_its_plan() {
    let _guard = serial();
    let db = setup("prepared_cte");
    const WHERE: &str =
        "SELECT s.id FROM s WHERE s.b = $1 AND s.a > (SELECT v FROM u WHERE u.id = s.id) ORDER BY s.id";
    for (cte, with) in [
        (
            "materialized",
            "WITH s(id, a, b) AS (SELECT id, a, b FROM t) ",
        ),
        ("inlined", "WITH s AS (SELECT id, a, b FROM t) "),
    ] {
        let sql = format!("{with}{WHERE}");
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{cte}");
        let built = prepared_selects_built();
        let general = general_hits();
        for (param, want) in [(1, vec![1]), (2, vec![6]), (1, vec![1])] {
            let (found, runs) = ids(&db, &sql, (param,));
            assert_eq!(found, want, "{cte}: b = {param}");
            assert_eq!(runs, 3, "{cte}: b = {param} runs once per CTE row");
        }
        assert_eq!(
            prepared_selects_built() - built,
            1,
            "{cte}: prepared once across three runs"
        );
        assert_eq!(general_hits(), general, "{cte}: no general path ran");
        let columns = db.query(&sql, (1,)).unwrap().columns().to_vec();
        assert_eq!(columns, vec!["id".to_string()], "{cte}: the column's name");
    }
}

#[test]
fn test_a_block_reading_a_cte_is_not_prepared() {
    let _guard = serial();
    let db = setup("prepared_cte_block");
    assert_eq!(
        prepared_line(
            &db,
            "WITH s AS (SELECT id, a FROM t) SELECT x.id FROM s AS x \
             WHERE x.a > (SELECT MIN(a) FROM s)"
        ),
        "Prepared: no (a subquery's FROM names a CTE of the statement's WITH)"
    );
}

#[test]
fn test_a_cte_the_statement_does_not_read_does_not_run() {
    let _guard = serial();
    let db = setup("prepared_cte_unused");
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO m VALUES (1, -9223372036854775808)", ())
        .unwrap();
    const UNUSED: &str = "WITH unused AS (SELECT ABS(x) FROM m), s AS (SELECT id FROM t) \
                          SELECT id FROM s WHERE id < (SELECT MIN(id) FROM u) + 2 ORDER BY id";
    assert_eq!(prepared_line(&db, UNUSED), "Prepared: yes");
    assert_eq!(
        ids(&db, UNUSED, ()).0,
        vec![1, 2],
        "ABS of its row would overflow"
    );
}

#[test]
fn test_a_cte_reads_the_ctes_and_views_around_it() {
    let _guard = serial();
    let db = setup("prepared_cte_scope");
    db.execute(
        "CREATE VIEW tv AS WITH w AS (SELECT id FROM t WHERE b = 2) SELECT id FROM w",
        (),
    )
    .unwrap();
    for (sql, want) in [
        (
            // Chained CTEs do not inline: c is materialized for s to read
            "WITH c(id) AS (SELECT id FROM t WHERE b = 1), s AS (SELECT id FROM c) \
             SELECT s.id FROM s WHERE s.id > (SELECT MIN(id) FROM u) ORDER BY s.id",
            vec![2, 5],
        ),
        (
            "WITH c(id) AS (SELECT 1) SELECT tv.id FROM tv \
             WHERE tv.id > (SELECT MIN(id) FROM u) ORDER BY tv.id",
            vec![3, 4, 6],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        assert_eq!(ids(&db, sql, ()).0, want, "{sql}");
    }
}

#[test]
fn test_a_table_function_runs_prepared_over_its_rows() {
    let _guard = serial();
    let db = setup("prepared_tvf");
    const SERIES: &str = "SELECT s.id FROM generate_series(1, $1) AS s(id) \
                          WHERE s.id * 10 > (SELECT v FROM u WHERE u.id = s.id) ORDER BY s.id";
    assert_eq!(prepared_line(&db, SERIES), "Prepared: yes");
    let built = prepared_selects_built();
    let general = general_hits();
    for (stop, want) in [(6, vec![1, 6]), (3, vec![1]), (6, vec![1, 6])] {
        let (found, runs) = ids(&db, SERIES, (stop,));
        assert_eq!(found, want, "generate_series(1, {stop})");
        assert_eq!(runs, stop as usize, "the block runs once per generated row");
    }
    assert_eq!(
        prepared_selects_built() - built,
        1,
        "prepared once across three runs"
    );
    assert_eq!(general_hits(), general, "no general path ran");
    let columns = db.query(SERIES, (6,)).unwrap().columns().to_vec();
    assert_eq!(columns, vec!["id".to_string()], "the column's name");
}

#[test]
fn test_a_table_function_generates_only_the_range_its_where_allows() {
    let _guard = serial();
    let db = setup("prepared_tvf_range");
    const SERIES: &str = "SELECT s.id FROM generate_series(1, 1000000) AS s(id) \
                          WHERE s.id < 4 AND s.id IN (SELECT id FROM u) ORDER BY s.id";
    assert_eq!(prepared_line(&db, SERIES), "Prepared: yes");
    let before = tvf_rows();
    assert_eq!(ids(&db, SERIES, ()).0, vec![1, 2, 3]);
    assert_eq!(tvf_rows() - before, 3, "the rows below 4, of a million");
}

#[test]
fn test_a_block_reads_a_view_values_list_or_table_function() {
    let _guard = serial();
    let db = setup("prepared_block_sources");
    db.execute("CREATE VIEW uv AS SELECT id, v FROM u", ())
        .unwrap();
    for (sql, want, runs) in [
        (
            "SELECT id FROM t WHERE a > (SELECT v FROM uv WHERE uv.id = t.id) ORDER BY id",
            vec![1, 6],
            6,
        ),
        (
            "SELECT id FROM t WHERE a >= (SELECT MAX(v) FROM uv) ORDER BY id",
            vec![6],
            1,
        ),
        (
            "SELECT id FROM t WHERE a > (SELECT w.x FROM (VALUES (1, 5), (2, 25), (6, 7)) AS w(k, x) \
             WHERE w.k = t.id) ORDER BY id",
            vec![1, 6],
            6,
        ),
        (
            "SELECT id FROM t WHERE a > (SELECT MAX(s.x) * 10 FROM generate_series(1, 3) AS s(x)) \
             ORDER BY id",
            vec![4, 5, 6],
            1,
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        assert_eq!(ids(&db, sql, ()), (want, runs), "{sql}");
    }
}

#[test]
fn test_a_block_reads_a_cte_around_its_statement() {
    let _guard = serial();
    let db = setup("prepared_block_cte");
    const SQL: &str = "WITH c AS (SELECT id, v FROM u), \
                       s AS (SELECT id FROM t WHERE a > (SELECT v FROM c WHERE c.id = t.id)) \
                       SELECT id FROM s ORDER BY id";
    assert_eq!(ids(&db, SQL, ()), (vec![1, 6], 6));
}

#[test]
fn test_a_block_over_a_redefined_view_reads_its_new_body() {
    let _guard = serial();
    let db = setup("prepared_block_view_redefined");
    const SQL: &str =
        "SELECT id FROM t WHERE a > (SELECT v FROM uv WHERE uv.id = t.id) ORDER BY id";
    let built = prepared_selects_built();
    for (body, want, prepared) in [
        ("SELECT id, v FROM u", vec![1, 6], 1),
        ("SELECT id, v + 100 AS v FROM u", vec![], 2),
        ("SELECT v, id FROM u", vec![1, 6], 3),
    ] {
        db.execute("DROP VIEW IF EXISTS uv", ()).unwrap();
        db.execute(&format!("CREATE VIEW uv AS {body}"), ())
            .unwrap();
        assert_eq!(ids(&db, SQL, ()).0, want, "{body}");
        assert_eq!(
            prepared_selects_built() - built,
            prepared,
            "{body}: a view defined again prepares the plan again"
        );
    }
}

#[test]
fn test_a_block_over_a_view_or_cte_reads_each_statement_s_rows() {
    let _guard = serial();
    let db = setup("prepared_block_source_fresh");
    db.execute("CREATE VIEW uv AS SELECT id, v FROM u", ())
        .unwrap();
    for sql in [
        "SELECT id FROM t WHERE a >= (SELECT MAX(v) FROM uv) ORDER BY id",
        "WITH c AS (SELECT v FROM u), s AS (SELECT id, a FROM t WHERE a >= (SELECT MAX(v) FROM c)) \
         SELECT id FROM s ORDER BY id",
    ] {
        db.execute("UPDATE u SET v = 60 WHERE id = 5", ()).unwrap();
        assert_eq!(ids(&db, sql, ()).0, vec![6], "{sql}");
        db.execute("UPDATE u SET v = 30 WHERE id = 5", ()).unwrap();
        assert_eq!(ids(&db, sql, ()).0, vec![3, 4, 5, 6], "{sql}");
    }
}

#[test]
fn test_a_block_source_is_read_when_a_call_first_needs_it() {
    let _guard = serial();
    let db = setup("prepared_block_source_error");
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO m VALUES (1, -9223372036854775808)", ())
        .unwrap();
    db.execute("CREATE VIEW mv AS SELECT id, ABS(x) AS x FROM m", ())
        .unwrap();
    db.execute("CREATE VIEW unnamed AS SELECT id, ABS(x) FROM m", ())
        .unwrap();
    let sources = [
        "mv",
        "(VALUES (1, ABS(-9223372036854775808))) AS mv(id, x)",
        "(SELECT id, ABS(x) AS x FROM m) AS mv",
        "(SELECT value AS id, value AS x FROM generate_series(1, 2, 0)) AS mv",
    ];
    for source in sources {
        for (reached, sql) in [
            (false, format!("SELECT id FROM t WHERE id > 100 AND a > (SELECT x FROM {source} WHERE mv.id = t.id)")),
            (false, format!("SELECT id FROM t WHERE id > 0 OR a > (SELECT x FROM {source} WHERE mv.id = t.id) ORDER BY id")),
            (true, format!("SELECT id FROM t WHERE a > (SELECT x FROM {source} WHERE mv.id = t.id)")),
        ] {
            assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
            let got = db
                .query(&sql, ())
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
                .map(|rows| rows.len())
                .map_err(|e| e.to_string());
            match reached {
                false => assert!(
                    got.is_ok(),
                    "{sql}: no call needs the source, which raises nothing: {got:?}"
                ),
                true => assert!(
                    got.as_ref()
                        .is_err_and(|e| e.contains("overflow") || e.contains("step")),
                    "{sql}: the first call reads the source and raises its error: {got:?}"
                ),
            }
        }
    }
    // A view's columns are recorded when it is created, so one named as written is read lazily too
    let sql =
        "SELECT id FROM t WHERE id > 100 AND a > (SELECT id FROM unnamed WHERE unnamed.id = t.id)";
    assert_eq!(ids(&db, sql, ()).0, Vec::<i64>::new(), "{sql}");
}

#[test]
fn test_a_block_s_table_function_generates_only_the_range_its_where_allows() {
    let _guard = serial();
    let db = setup("prepared_block_tvf_range");
    const SQL: &str = "SELECT id FROM t WHERE a > \
                       (SELECT MAX(value) FROM generate_series(1, 10000001) WHERE value < 3) ORDER BY id";
    assert_eq!(prepared_line(&db, SQL), "Prepared: yes");
    let before = tvf_rows();
    assert_eq!(ids(&db, SQL, ()).0, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(tvf_rows() - before, 2, "the rows below 3, of ten million");
    for none in ["FALSE", "NULL", "1 = 0"] {
        let sql = format!(
            "SELECT id FROM t WHERE EXISTS \
             (SELECT 1 FROM generate_series(1, 10000001) AS s(n) WHERE {none})"
        );
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{none}");
        let before = tvf_rows();
        assert_eq!(ids(&db, &sql, ()).0, Vec::<i64>::new(), "{none}");
        assert_eq!(tvf_rows() - before, 0, "WHERE {none} generates nothing");
    }
    const OUTER: &str = "SELECT id FROM t WHERE \
                         (SELECT COUNT(*) FROM generate_series(1, 10) AS s(a) WHERE t.a = 10) = 10 ORDER BY id";
    assert_eq!(prepared_line(&db, OUTER), "Prepared: yes");
    assert_eq!(
        ids(&db, OUTER, ()).0,
        vec![1],
        "a comparison of an enclosing column narrows nothing"
    );
}

#[test]
fn test_a_block_s_view_runs_once_in_the_run_that_prepares_it() {
    let _guard = serial();
    let db = setup("prepared_block_view_once");
    db.execute("CREATE VIEW rv AS SELECT RANDOM() AS n", ())
        .unwrap();
    const SQL: &str = "SELECT id FROM t WHERE a > (SELECT MAX(n) FROM rv) ORDER BY id";
    let built = prepared_selects_built();
    for run in ["prepares", "keeps its plan"] {
        let before = random_draws();
        assert_eq!(ids(&db, SQL, ()).0, vec![1, 2, 3, 4, 5, 6], "{run}");
        assert_eq!(
            random_draws() - before,
            1,
            "the run that {run} reads the view once"
        );
    }
    assert_eq!(prepared_selects_built() - built, 1);
}

#[test]
fn test_a_returning_block_s_view_over_the_target_runs_once_in_the_run_that_prepares_it() {
    let _guard = serial();
    let db = setup("prepared_returning_view_once");
    db.execute(
        "CREATE VIEW rv AS SELECT id, RANDOM() AS n FROM t WHERE id = 1",
        (),
    )
    .unwrap();
    const SQL: &str =
        "UPDATE t SET b = 11 WHERE id = 1 RETURNING id, (SELECT rv.n FROM rv WHERE rv.id = t.id)";
    for run in ["prepares", "keeps its plan"] {
        let before = random_draws();
        let found: Vec<i64> = db
            .query(SQL, ())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(found, vec![1], "{run}");
        assert_eq!(
            random_draws() - before,
            1,
            "the run that {run} reads the view once"
        );
    }
}

/// Two rows in a column named `n`
struct NamedRows;

impl stoolap::functions::tvf::TableValuedFunction for NamedRows {
    fn name(&self) -> &str {
        "NAMED_ROWS"
    }
    fn column_names(&self) -> Vec<String> {
        vec!["n".to_string()]
    }
    fn generate(
        &self,
        _args: &[stoolap::core::Value],
        _limit: Option<usize>,
    ) -> stoolap::core::Result<stoolap::core::RowVec> {
        let mut rows = stoolap::core::RowVec::new();
        for n in [1i64, 2] {
            rows.push((n, stoolap::core::Row::from_values(vec![n.into()])));
        }
        Ok(rows)
    }
}

#[test]
fn test_a_table_function_generating_nothing_keeps_its_column_names() {
    let _guard = serial();
    stoolap::functions::global_registry()
        .register_tvf("NAMED_ROWS", std::sync::Arc::new(|| Box::new(NamedRows)));
    let db = setup("prepared_tvf_named_columns");
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM named_rows() WHERE FALSE)",
            vec![],
        ),
        (
            "SELECT id FROM t WHERE id IN (SELECT n FROM named_rows() WHERE n > 1)",
            vec![2],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        assert_eq!(ids(&db, sql, ()).0, want, "{sql}");
    }
    let columns = db
        .query("SELECT * FROM named_rows() WHERE FALSE", ())
        .unwrap()
        .columns()
        .to_vec();
    assert_eq!(columns, vec!["n".to_string()]);
}

#[test]
fn test_a_select_list_block_is_asked_of_each_row_returned() {
    let _guard = serial();
    let db = setup("prepared_select_list_rows");
    const SQL: &str = "SELECT id, (SELECT v FROM u WHERE u.id = t.id) FROM t LIMIT 2";
    assert_eq!(prepared_line(&db, SQL), "Prepared: yes");
    let (found, runs) = ids(&db, SQL, ());
    assert_eq!(found.len(), 2);
    assert_eq!(runs, 2, "one run per row returned, not per row read");
}

#[test]
fn test_an_uncorrelated_output_subquery_is_read_before_any_row() {
    let _guard = serial();
    let db = setup("prepared_output_read_first");
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO m VALUES (1, -9223372036854775808)", ())
        .unwrap();
    for sql in [
        "SELECT id, (SELECT ABS(x) FROM m) FROM t WHERE id > 100",
        "SELECT id, a IN (SELECT ABS(x) FROM m) FROM t WHERE id > 100",
        "SELECT id, a > ALL (SELECT ABS(x) FROM m) FROM t WHERE id > 100",
        "SELECT id, EXISTS (SELECT 1 FROM m WHERE ABS(x) > 0) FROM t WHERE id > 100",
        "SELECT id FROM t WHERE id > 100 ORDER BY a IN (SELECT ABS(x) FROM m)",
        "SELECT id, (SELECT ABS(x) FROM m) FROM t ORDER BY id LIMIT 0",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        assert!(
            error(&db, sql, ()).0.contains("overflow"),
            "{sql}: the subquery's error, though no row is returned"
        );
    }
    const CORRELATED: &str = "SELECT id, (SELECT ABS(x) FROM m WHERE m.id = t.id) FROM t \
                              WHERE id > 100";
    assert_eq!(ids(&db, CORRELATED, ()).0, Vec::<i64>::new());
}

#[test]
fn test_select_list_exists_in_and_all_answer_as_their_literals() {
    let _guard = serial();
    let db = setup("prepared_select_list_forms");
    let answers = |sql: &str| -> Vec<(i64, Option<bool>)> {
        db.query(sql, ())
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .map(|row| {
                let row = row.unwrap();
                (row.get(0).unwrap(), row.get(1).unwrap())
            })
            .collect()
    };
    for (sql, literal) in [
        (
            "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.id = t.id) FROM t ORDER BY id",
            "SELECT id, id IN (1, 2, 3, 5, 6) FROM t ORDER BY id",
        ),
        (
            "SELECT id, a IN (SELECT v FROM u WHERE u.id = t.id) FROM t ORDER BY id",
            "SELECT id, CASE WHEN id = 3 THEN NULL ELSE FALSE END FROM t ORDER BY id",
        ),
        (
            "SELECT id, a > ALL (SELECT v FROM u WHERE u.id = t.id) FROM t ORDER BY id",
            "SELECT id, CASE id WHEN 2 THEN FALSE WHEN 3 THEN NULL WHEN 5 THEN FALSE \
             ELSE TRUE END FROM t ORDER BY id",
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        assert_eq!(answers(sql), answers(literal), "{sql}");
    }
}

#[test]
fn test_an_ordered_limit_asks_its_select_list_blocks_of_the_rows_returned() {
    let _guard = serial();
    let db = Database::open("memory://prepared_ordered_limit_rows").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO u VALUES (1, 1, 100), (2, 2, 200), (3, 2, 300), (4, 3, 400), (5, 4, 500)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // Row 2's block answers more than one row: no row it reaches is returned
    for (sql, want) in [
        (
            "SELECT t.id, (SELECT u.v FROM u WHERE u.k = t.id) FROM t ORDER BY t.id LIMIT 1",
            vec![1],
        ),
        (
            "SELECT t.id, (SELECT u.v FROM u WHERE u.k = t.id) FROM t ORDER BY t.a LIMIT 1",
            vec![1],
        ),
        (
            "SELECT t.id, (SELECT u.v FROM u WHERE u.k = t.id) FROM t \
             ORDER BY t.a DESC LIMIT 2",
            vec![4, 3],
        ),
        (
            "SELECT t.id, (SELECT u.v FROM u WHERE u.k = t.id) FROM t \
             ORDER BY t.a DESC LIMIT 1 OFFSET 1",
            vec![3],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let (found, runs) = ids(&db, sql, ());
        assert_eq!(found, want, "{sql}");
        assert_eq!(runs, want.len(), "{sql}: one run per row returned");
    }
}

#[test]
fn test_a_table_function_without_where_or_order_by_generates_the_rows_returned() {
    let _guard = serial();
    let db = setup("prepared_tvf_first_rows");
    const SQL: &str = "SELECT value, (SELECT 1) FROM generate_series(1, 10000001) LIMIT 2 OFFSET 1";
    assert_eq!(prepared_line(&db, SQL), "Prepared: yes");
    let before = tvf_rows();
    assert_eq!(ids(&db, SQL, ()).0, vec![2, 3]);
    assert_eq!(tvf_rows() - before, 3, "LIMIT and OFFSET, of ten million");
}

#[test]
fn test_an_order_by_subquery_naming_a_select_alias_inside_it_is_not_prepared() {
    let _guard = serial();
    let db = setup("prepared_order_by_alias_inside");
    for sql in [
        "SELECT id AS z FROM t ORDER BY (SELECT z + t.id) DESC",
        "SELECT t.id AS z FROM t JOIN u ON u.id = t.id ORDER BY (SELECT z + t.id) DESC",
    ] {
        assert_eq!(
            prepared_line(&db, sql),
            "Prepared: no (an ORDER BY subquery that names a SELECT alias)",
            "{sql}"
        );
        assert!(db.query(sql, ()).is_ok(), "{sql}");
    }
    assert_eq!(
        ids(
            &db,
            "SELECT id AS z FROM t ORDER BY (SELECT z + t.id) DESC",
            ()
        )
        .0,
        vec![6, 5, 4, 3, 2, 1]
    );
}

#[test]
fn test_an_order_by_key_repeating_a_select_item_reads_its_value() {
    let _guard = serial();
    let db = setup("prepared_order_by_repeated_item");
    const SQL: &str = "SELECT id, (SELECT RANDOM() FROM u WHERE u.id = t.id) FROM t \
                       ORDER BY (SELECT RANDOM() FROM u WHERE u.id = t.id), id";
    assert_eq!(prepared_line(&db, SQL), "Prepared: yes");
    let before = random_draws();
    let rows: Vec<(i64, Option<f64>)> = db
        .query(SQL, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect();
    assert_eq!(
        random_draws() - before,
        5,
        "one draw per row u holds a key of"
    );
    let drawn: Vec<f64> = rows.iter().filter_map(|(_, value)| *value).collect();
    assert!(
        drawn.windows(2).all(|pair| pair[0] <= pair[1]),
        "ordered by the values returned: {rows:?}"
    );
}

#[test]
fn test_a_select_list_block_over_values_is_asked_of_the_rows_limit_and_offset_keep() {
    let _guard = serial();
    let db = setup("prepared_values_limit_rows");
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1, 1), (2, 2), (3, 2)", ())
        .unwrap();
    for (sql, want) in [
        (
            "SELECT n, (SELECT d.id FROM d WHERE d.k = q.n) FROM (VALUES (2), (1)) AS q(n) LIMIT 0",
            vec![],
        ),
        (
            "SELECT n, (SELECT d.id FROM d WHERE d.k = q.n) FROM (VALUES (2), (1)) AS q(n) OFFSET 1",
            vec![1],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let (found, runs) = ids(&db, sql, ());
        assert_eq!(found, want, "{sql}");
        assert_eq!(runs, want.len(), "{sql}: one run per row returned");
    }
}

/// t: id 1..4, a = id * 10 indexed, x overflowing ABS on id 2; u holds two
/// rows for key 2 and one for each other key
fn ordered_setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, x INTEGER)",
        "CREATE INDEX t_a ON t(a)",
        "INSERT INTO t VALUES (1, 10, 1), (2, 20, -9223372036854775808), (3, 30, 3), (4, 40, 4)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO u VALUES (1, 1, 100), (2, 2, 200), (3, 2, 300), (4, 3, 400), (5, 4, 500)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn test_an_ordered_limit_on_a_column_reads_its_rows_in_order() {
    let _guard = serial();
    let db = ordered_setup("prepared_ordered_read");
    // None where the table decides whether its top-k scan can read them
    for (sql, want, in_order) in [
        (
            "SELECT t.id, (SELECT u.v FROM u WHERE u.k = t.id) FROM t ORDER BY t.id LIMIT 1",
            vec![1],
            Some(true),
        ),
        (
            "SELECT t.id, (SELECT u.v FROM u WHERE u.k = t.id) FROM t ORDER BY a DESC LIMIT 2",
            vec![4, 3],
            Some(true),
        ),
        (
            "SELECT t.id, (SELECT u.v FROM u WHERE u.k = t.id) FROM t \
             WHERE a > (SELECT MIN(v) FROM u) - 80 ORDER BY t.id LIMIT 1 OFFSET 1",
            vec![4],
            None,
        ),
        (
            "SELECT t.id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id) \
             ORDER BY t.id LIMIT 1",
            vec![1],
            Some(false),
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let reads = prepared_ordered_reads();
        let (found, _) = ids(&db, sql, ());
        assert_eq!(found, want, "{sql}");
        if let Some(in_order) = in_order {
            assert_eq!(
                prepared_ordered_reads() - reads,
                usize::from(in_order),
                "{sql}: read in order through the index"
            );
        }
    }
}

#[test]
fn test_an_ordered_limit_computes_select_items_for_the_rows_it_keeps() {
    let _guard = serial();
    let db = ordered_setup("prepared_ordered_items");
    for (sql, want) in [
        // ABS overflows on row 2, which the LIMIT drops
        (
            "SELECT t.id, ABS(t.x), (SELECT u.v FROM u WHERE u.k = t.id) FROM t \
             ORDER BY t.id + 0 LIMIT 1",
            vec![1],
        ),
        // The key reads z, not the subquery, so row 2's block runs for no row
        (
            "SELECT id AS z, (SELECT u.v FROM u WHERE u.k = t.id) FROM t ORDER BY z + 0 LIMIT 1",
            vec![1],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let (found, runs) = ids(&db, sql, ());
        assert_eq!(found, want, "{sql}");
        assert_eq!(runs, 1, "{sql}: the block runs for the row returned");
    }
}

#[test]
fn test_an_alias_inside_an_aliased_order_by_subquery_item_is_not_prepared() {
    let _guard = serial();
    let db = setup("prepared_order_by_inner_alias");
    const SQL: &str = "SELECT id AS z FROM t ORDER BY (SELECT z + t.id AS k) DESC";
    assert_eq!(
        prepared_line(&db, SQL),
        "Prepared: no (an ORDER BY subquery that names a SELECT alias)"
    );
    assert_eq!(ids(&db, SQL, ()).0, vec![6, 5, 4, 3, 2, 1]);
}

#[test]
fn test_an_uncorrelated_order_by_key_repeating_a_select_item_runs_once() {
    let _guard = serial();
    let db = setup("prepared_order_by_repeated_slot");
    const SQL: &str = "SELECT id, (SELECT RANDOM()) FROM t ORDER BY (SELECT RANDOM()), id";
    assert_eq!(prepared_line(&db, SQL), "Prepared: yes");
    let before = random_draws();
    assert_eq!(ids(&db, SQL, ()).0, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(random_draws() - before, 1, "one draw, which the key reads");
}

#[test]
fn test_a_read_past_its_timeout_fails() {
    let _guard = serial();
    let db = setup("prepared_read_timeout");
    for sql in [
        "SELECT id FROM t WHERE a > (SELECT MIN(v) FROM u)",
        "SELECT id FROM t WHERE a + 0 IN (SELECT v FROM u)",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        after_prepared_plan_taken(|| std::thread::sleep(std::time::Duration::from_millis(150)));
        let read = db
            .query_with_timeout(sql, (), 10)
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
        assert!(read.is_err(), "{sql}: rows read past the timeout");
    }
}

#[test]
fn test_a_read_cancelled_before_its_rows_fails() {
    let _guard = serial();
    let db = setup("prepared_read_cancel");
    let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
    for sql in [
        "SELECT id FROM t WHERE a > (SELECT MIN(v) FROM u)",
        "SELECT id FROM t WHERE a + 0 IN (SELECT v FROM u)",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let ctx = stoolap::ExecutionContext::new();
        let handle = ctx.clone();
        after_prepared_plan_taken(move || handle.cancel());
        let read = executor
            .execute_with_context(sql, &ctx)
            .and_then(|mut rows| {
                while rows.next() {}
                rows.last_error().map_or(Ok(()), Err)
            });
        assert!(
            matches!(read, Err(stoolap::Error::QueryCancelled)),
            "{sql}: {:?}",
            read
        );
    }
}
