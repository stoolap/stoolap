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

//! A join is planned from the join itself, not from what reads its rows: an
//! aggregate or a window over an index join reads the rows the same join
//! gives a projection, and answers what it answers over a hash join

use stoolap::Database;

/// a, b with an index on b.aid, and plain_b holding b's rows with no index
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    for table in ["b", "plain_b"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, aid INTEGER, w INTEGER)"),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX b_aid ON b(aid)", ()).unwrap();
    let a: Vec<String> = (1..=40)
        .map(|i| format!("({i}, {}, {})", i % 4, i % 7))
        .collect();
    db.execute(&format!("INSERT INTO a VALUES {}", a.join(", ")), ())
        .unwrap();
    let b: Vec<String> = (1..=160)
        .map(|i| {
            let aid = if i % 11 == 0 {
                "NULL".to_string()
            } else {
                (i % 45).to_string()
            };
            format!("({i}, {aid}, {})", i % 10)
        })
        .collect();
    for table in ["b", "plain_b"] {
        db.execute(&format!("INSERT INTO {table} VALUES {}", b.join(", ")), ())
            .unwrap();
    }
    db
}

/// The column names and the sorted rows a statement returns
fn answer(db: &Database, sql: &str) -> (Vec<String>, Vec<String>) {
    let result = db.query(sql, ()).unwrap();
    let columns = result.columns().to_vec();
    let mut rows: Vec<String> = result
        .map(|row| format!("{:?}", row.unwrap().as_row().as_slice()))
        .collect();
    rows.sort();
    (columns, rows)
}

const SHAPES: &[&str] = &[
    "SELECT a.k, SUM(b.w) FROM a JOIN {b} b ON a.id = b.aid GROUP BY a.k",
    "SELECT COUNT(*), COUNT(DISTINCT b.w), MIN(b.w), MAX(a.v) FROM a JOIN {b} b ON a.id = b.aid",
    "SELECT a.k, COUNT(b.id) FROM a LEFT JOIN {b} b ON a.id = b.aid GROUP BY a.k \
     HAVING COUNT(b.id) > 1",
    "SELECT SUM(b.w) FROM a JOIN {b} b ON a.id = b.aid AND b.w > 3",
    "SELECT a.k, SUM(b.w) FROM a JOIN {b} b ON a.id = b.aid WHERE a.v > 2 AND b.w < 9 \
     GROUP BY a.k",
    "SELECT a.id, b.w, ROW_NUMBER() OVER (PARTITION BY a.id ORDER BY b.w, b.id) \
     FROM a JOIN {b} b ON a.id = b.aid",
    "SELECT a.k, SUM(b.w), RANK() OVER (ORDER BY SUM(b.w)) FROM a JOIN {b} b \
     ON a.id = b.aid GROUP BY a.k",
    "SELECT SUM(b.w) FROM {b} b JOIN (SELECT id, k FROM a WHERE v > 1) x ON x.id = b.aid",
    "SELECT *, COUNT(*) OVER () FROM {b} b JOIN (SELECT id, k FROM a WHERE v > 1) x \
     ON x.id = b.aid",
];

/// The rows of the index join's own paths, asked with and without a reader
/// above them: a renamed subquery column, `*` after the sides swap, and a
/// SELECT alias in WHERE
#[test]
fn an_index_join_keeps_projections_column_order_and_aliases() {
    let db = Database::open("memory://join_consumer_paths").unwrap();
    db.execute(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, aid INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX b_aid ON b(aid)", ()).unwrap();
    db.execute("INSERT INTO a VALUES (1, 10, 100), (2, 1, 200)", ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (1, 1, 5), (2, 2, 6)", ())
        .unwrap();
    let renamed = "FROM b JOIN (SELECT id, v AS k FROM a) x ON b.aid = x.id";
    let fewer = "FROM b JOIN (SELECT id, k FROM a) x ON b.aid = x.id";
    let reordered = "FROM b JOIN (SELECT k, id FROM a) x ON b.aid = x.id";
    let respelled = "FROM b JOIN (SELECT id AS \"ID\", k, v FROM a) x ON b.aid = x.id";
    let swapped = "FROM b JOIN (SELECT id, k FROM a WHERE k = 1) x ON b.id = x.id";
    let aliased = "FROM a JOIN b ON a.id = b.aid WHERE total > a.v + b.w - 1";
    let cases: [(String, Vec<&str>, Vec<&str>); 9] = [
        (
            format!("SELECT *, COUNT(*) OVER () {respelled}"),
            vec!["b.id", "b.aid", "b.w", "x.ID", "x.k", "x.v", "COUNT(*) OVER ()"],
            vec![
                "[Integer(1), Integer(1), Integer(5), Integer(1), Integer(10), Integer(100), Integer(2)]",
                "[Integer(2), Integer(2), Integer(6), Integer(2), Integer(1), Integer(200), Integer(2)]",
            ],
        ),
        (
            format!("SELECT *, COUNT(*) OVER () {fewer}"),
            vec!["b.id", "b.aid", "b.w", "x.id", "x.k", "COUNT(*) OVER ()"],
            vec![
                "[Integer(1), Integer(1), Integer(5), Integer(1), Integer(10), Integer(2)]",
                "[Integer(2), Integer(2), Integer(6), Integer(2), Integer(1), Integer(2)]",
            ],
        ),
        (
            format!("SELECT * {reordered}"),
            vec!["b.id", "b.aid", "b.w", "x.k", "x.id"],
            vec![
                "[Integer(1), Integer(1), Integer(5), Integer(10), Integer(1)]",
                "[Integer(2), Integer(2), Integer(6), Integer(1), Integer(2)]",
            ],
        ),
        (
            format!("SELECT x.k {renamed}"),
            vec!["k"],
            vec!["[Integer(100)]", "[Integer(200)]"],
        ),
        (
            format!("SELECT SUM(x.k) {renamed}"),
            vec!["SUM(x.k)"],
            vec!["[Integer(300)]"],
        ),
        (
            format!("SELECT * {swapped}"),
            vec!["b.id", "b.aid", "b.w", "x.id", "x.k"],
            vec!["[Integer(2), Integer(2), Integer(6), Integer(2), Integer(1)]"],
        ),
        (
            format!("SELECT *, COUNT(*) OVER () {swapped}"),
            vec!["b.id", "b.aid", "b.w", "x.id", "x.k", "COUNT(*) OVER ()"],
            vec!["[Integer(2), Integer(2), Integer(6), Integer(2), Integer(1), Integer(1)]"],
        ),
        (
            format!("SELECT a.v + b.w AS total {aliased}"),
            vec!["total"],
            vec!["[Integer(105)]", "[Integer(206)]"],
        ),
        (
            format!("SELECT a.v + b.w AS total, COUNT(*) OVER () {aliased}"),
            vec!["total", "COUNT(*) OVER ()"],
            vec!["[Integer(105), Integer(2)]", "[Integer(206), Integer(2)]"],
        ),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|(sql, columns, rows)| {
            let got = match db.query(sql, ()) {
                Ok(_) => answer(&db, sql),
                Err(e) => return Some(format!("{sql}\n  error {e}")),
            };
            let want = (
                columns.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
                rows.iter().map(|r| r.to_string()).collect::<Vec<_>>(),
            );
            (got != want).then(|| format!("{sql}\n  got {got:?}\n  want {want:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_aggregate_or_window_over_an_index_join_answers_as_over_a_hash_join() {
    let db = tables("join_consumer_equivalence");
    let wrong: Vec<String> = SHAPES
        .iter()
        .filter_map(|shape| {
            let indexed = answer(&db, &shape.replace("{b}", "b"));
            let plain = answer(&db, &shape.replace("{b}", "plain_b"));
            (indexed != plain).then(|| format!("{shape}\n  index: {indexed:?}\n  hash:  {plain:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn explain_reports_the_index_join_under_an_aggregate() {
    let db = tables("join_consumer_explain");
    let plan: Vec<String> = db
        .query(
            "EXPLAIN SELECT a.k, SUM(b.w) FROM a JOIN b ON a.id = b.aid GROUP BY a.k",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert!(
        plan.iter().any(|line| line.contains("Index Nested Loop")),
        "{}",
        plan.join("\n")
    );
    // The left primary key is probed only by the swap execution makes: not
    // from a right table nothing filters
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, aid INTEGER, flag INTEGER)",
        (),
    )
    .unwrap();
    for sql in [
        "EXPLAIN SELECT a.id FROM a JOIN p ON a.id = p.aid",
        "EXPLAIN SELECT COUNT(*) FROM a JOIN p ON a.id = p.aid",
        "EXPLAIN SELECT COUNT(*) FROM a JOIN p ON a.id = p.aid WHERE 1 = 1 AND p.flag > 0",
    ] {
        let plan: Vec<String> = db
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect();
        assert!(
            !plan.iter().any(|line| line.contains("Index Nested Loop")),
            "{sql}\n{}",
            plan.join("\n")
        );
    }
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn an_aggregate_over_a_join_runs_the_index_join() {
    use std::cell::Cell;
    use std::rc::Rc;
    let db = tables("join_consumer_failpoint");
    let opened = Rc::new(Cell::new(false));
    let hook_opened = Rc::clone(&opened);
    stoolap::test_failpoints::after_join_inner_opened(move || hook_opened.set(true));
    answer(
        &db,
        "SELECT a.k, SUM(b.w) FROM a JOIN b ON a.id = b.aid GROUP BY a.k",
    );
    assert!(opened.get(), "the aggregate read the index join's rows");
}
