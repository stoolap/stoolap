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

//! A view or a CTE on one side of a join is a materialized source, not a
//! base table: a table on the other side is probed by its index from it,
//! and a small one narrows a large table by its keys, index or not

use stoolap::Database;

/// t: a primary key and an unindexed k over 70,000 rows; r drives the
/// views; small holds three keys of each kind
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t SELECT g.value, g.value % 1000, CAST(g.value % 1000 AS TEXT), g.value \
         FROM generate_series(1, 70000) g",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO r VALUES (1, 7), (2, 42), (3, 69999)", ())
        .unwrap();
    db.execute("CREATE VIEW c AS SELECT x FROM r", ()).unwrap();
    db.execute(
        "CREATE TABLE small (id INTEGER PRIMARY KEY, n INTEGER, txt TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO small VALUES (1, 5, '5'), (2, 6, 'x'), (3, NULL, NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE VIEW sv AS SELECT n, txt FROM small", ())
        .unwrap();
    // A table below the size that pays for narrowing from a base table
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO m SELECT g.value, g.value % 1000 FROM generate_series(1, 30000) g",
        (),
    )
    .unwrap();
    db
}

fn answer(db: &Database, sql: &str) -> (Vec<String>, Vec<String>) {
    let result = db.query(sql, ()).unwrap();
    let columns = result.columns().to_vec();
    let mut rows: Vec<String> = result
        .map(|row| format!("{:?}", row.unwrap().as_row().as_slice()))
        .collect();
    rows.sort();
    (columns, rows)
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

#[test]
fn a_table_joined_to_a_view_or_cte_on_its_right_answers_by_its_key() {
    let db = tables("materialized_side_rows");
    let ids = |sql: &str| -> Vec<i64> {
        let mut ids: Vec<i64> = db
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        ids.sort_unstable();
        ids
    };
    let want = vec![7, 42, 69999];
    assert_eq!(ids("SELECT t.v FROM t JOIN c ON c.x = t.id"), want);
    assert_eq!(
        ids("WITH w(x) AS (SELECT x FROM r) SELECT t.v FROM t JOIN w ON w.x = t.id ORDER BY t.v"),
        want
    );
    assert_eq!(
        ids("SELECT COUNT(*) FROM t JOIN c ON c.x = t.id"),
        vec![3],
        "an aggregate over the swapped join"
    );
    assert_eq!(
        ids("SELECT t.id FROM t LEFT JOIN c ON c.x = t.id WHERE t.id <= 3"),
        vec![1, 2, 3],
        "a LEFT join keeps its sides"
    );
    let (columns, rows) = answer(&db, "SELECT * FROM t JOIN c ON c.x = t.id AND t.id = 42");
    assert_eq!(columns, ["t.id", "t.k", "t.s", "t.v", "c.x"]);
    assert_eq!(
        rows,
        ["[Integer(42), Integer(42), Text(\"42\"), Integer(42), Integer(42)]"]
    );
}

#[test]
fn a_small_view_narrows_a_large_unindexed_table_as_its_control_answers() {
    let db = tables("materialized_side_narrow");
    for (sql, want) in [
        ("SELECT COUNT(*) FROM sv JOIN t ON t.k = sv.n", 140),
        ("SELECT COUNT(*) FROM t JOIN sv ON t.k = sv.n", 140),
        ("SELECT COUNT(*) FROM sv JOIN t ON t.s = sv.n", 140),
        ("SELECT COUNT(*) FROM sv JOIN t ON t.k = sv.txt", 70),
        ("SELECT COUNT(*) FROM sv JOIN m ON m.k = sv.n", 60),
        ("SELECT COUNT(*) FROM m JOIN sv ON m.k = sv.txt", 30),
        (
            "SELECT COUNT(*) FROM sv JOIN t ON t.k = sv.n AND t.v > 35000",
            70,
        ),
        (
            "SELECT COUNT(*) FROM sv JOIN t ON t.k = sv.n WHERE t.v % 2 = 0",
            70,
        ),
    ] {
        assert_eq!(count(&db, sql), want, "{sql}");
    }
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn explain_reports_the_probe_from_a_view_on_the_right() {
    let db = tables("materialized_side_explain");
    // A CTE hiding a table of its name is the CTE, as execution reads it
    db.execute("CREATE TABLE w (x INTEGER)", ()).unwrap();
    for sql in [
        "EXPLAIN SELECT t.v FROM t JOIN c ON c.x = t.id",
        "EXPLAIN WITH w(x) AS (SELECT x FROM r) SELECT t.v FROM t JOIN w ON w.x = t.id \
         ORDER BY t.v",
    ] {
        let plan: Vec<String> = db
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect();
        assert!(
            plan.iter()
                .any(|line| line.contains("Index Nested Loop (PK)")),
            "{sql}\n{}",
            plan.join("\n")
        );
    }
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn the_index_join_and_the_narrowing_run_for_materialized_sides() {
    use std::cell::Cell;
    use std::rc::Rc;
    let db = tables("materialized_side_failpoints");
    for sql in [
        "SELECT t.v FROM t JOIN c ON c.x = t.id",
        "WITH w(x) AS (SELECT x FROM r) SELECT t.v FROM t JOIN w ON w.x = t.id ORDER BY t.v",
    ] {
        let opened = Rc::new(Cell::new(false));
        let hook_opened = Rc::clone(&opened);
        stoolap::test_failpoints::after_join_inner_opened(move || hook_opened.set(true));
        answer(&db, sql);
        assert!(opened.get(), "the left table was probed: {sql}");
    }
    for (sql, narrowed) in [
        ("SELECT COUNT(*) FROM sv JOIN t ON t.k = sv.n", 1),
        ("SELECT COUNT(*) FROM t JOIN sv ON t.k = sv.n", 1),
        // A materialized side narrows a table of any size
        ("SELECT COUNT(*) FROM sv JOIN m ON m.k = sv.n", 1),
        ("SELECT COUNT(*) FROM m JOIN sv ON m.k = sv.n", 1),
        // A base table outer narrows only a large table, as before
        ("SELECT COUNT(*) FROM small JOIN m ON m.k = small.n", 0),
    ] {
        let before = stoolap::test_failpoints::small_outer_joins();
        count(&db, sql);
        assert_eq!(
            stoolap::test_failpoints::small_outer_joins() - before,
            narrowed,
            "narrowings: {sql}"
        );
    }
}
