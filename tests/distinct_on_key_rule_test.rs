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

//! A DISTINCT ON key is read as an ORDER BY key is: one no output column
//! holds is computed for each output row, whatever path produced the rows

use stoolap::{Database, Value};

/// t: id 1..30, a = id % 5, b = id % 3; w: id 1..30, z = id % 2
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, z INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 5, g.value % 3 FROM generate_series(1, 30) g",
        "INSERT INTO w SELECT g.value, g.value % 2 FROM generate_series(1, 30) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// Each statement, and its twin reading the key as a column of a derived table
const SHAPES: &[(&str, &str)] = &[
    (
        "SELECT DISTINCT ON (a % 2) a, COUNT(*) FROM t GROUP BY a ORDER BY a % 2, a",
        "SELECT DISTINCT ON (k) a, n FROM (SELECT a, COUNT(*) AS n, a % 2 AS k FROM t GROUP BY a) s \
         ORDER BY k, a",
    ),
    (
        "SELECT DISTINCT ON (a % 2) COUNT(*) FROM t GROUP BY a ORDER BY a % 2, a",
        "SELECT DISTINCT ON (k) n FROM (SELECT COUNT(*) AS n, a % 2 AS k, a FROM t GROUP BY a) s \
         ORDER BY k, a",
    ),
    (
        "SELECT DISTINCT ON (SUM(id) % 2) a FROM t GROUP BY a ORDER BY SUM(id) % 2, a",
        "SELECT DISTINCT ON (k) a FROM (SELECT a, SUM(id) % 2 AS k FROM t GROUP BY a) s \
         ORDER BY k, a",
    ),
    (
        "SELECT DISTINCT ON (a % 2) a % 2, COUNT(*) FROM t GROUP BY a % 2",
        "SELECT a % 2, COUNT(*) FROM t GROUP BY a % 2",
    ),
    (
        "SELECT DISTINCT ON (a % 2) a, COUNT(*) FROM t GROUP BY ROLLUP(a) ORDER BY a % 2, a",
        "SELECT DISTINCT ON (k) a, n FROM (SELECT a, COUNT(*) AS n, a % 2 AS k FROM t \
         GROUP BY ROLLUP(a)) s ORDER BY k, a",
    ),
    (
        "SELECT DISTINCT ON (a % 2) a, ROW_NUMBER() OVER (ORDER BY id) FROM t ORDER BY a % 2, id",
        "SELECT DISTINCT ON (k) a, r FROM (SELECT a, id, a % 2 AS k, \
         ROW_NUMBER() OVER (ORDER BY id) AS r FROM t) s ORDER BY k, id",
    ),
    (
        "SELECT DISTINCT ON (a % 2) a % 2, ROW_NUMBER() OVER (ORDER BY id) FROM t",
        "SELECT DISTINCT ON (k) k, r FROM (SELECT a % 2 AS k, \
         ROW_NUMBER() OVER (ORDER BY id) AS r FROM t) s",
    ),
    (
        "SELECT DISTINCT ON (a % 2) a, RANK() OVER (ORDER BY a) FROM t GROUP BY a ORDER BY a % 2, a",
        "SELECT DISTINCT ON (k) a, r FROM (SELECT a, a % 2 AS k, RANK() OVER (ORDER BY a) AS r \
         FROM t GROUP BY a) s ORDER BY k, a",
    ),
    (
        "SELECT DISTINCT ON (t.a % 2) t.a, COUNT(*) FROM t JOIN w ON w.id = t.id GROUP BY t.a \
         ORDER BY t.a % 2, t.a",
        "SELECT DISTINCT ON (k) a, n FROM (SELECT t.a, COUNT(*) AS n, t.a % 2 AS k \
         FROM t JOIN w ON w.id = t.id GROUP BY t.a) s ORDER BY k, a",
    ),
    (
        "SELECT DISTINCT ON (s.a % 2) s.a, COUNT(*) FROM (SELECT a FROM t) s GROUP BY s.a \
         ORDER BY s.a % 2, s.a",
        "SELECT DISTINCT ON (k) a, n FROM (SELECT a, COUNT(*) AS n, a % 2 AS k FROM t GROUP BY a) s \
         ORDER BY k, a",
    ),
    (
        "SELECT DISTINCT ON (a % 2) a, id FROM t ORDER BY a % 2, id",
        "SELECT DISTINCT ON (k) a, id FROM (SELECT a, id, a % 2 AS k FROM t) s ORDER BY k, id",
    ),
    (
        "SELECT DISTINCT ON (SUM(id) % 2) a FROM t GROUP BY a ORDER BY a",
        "SELECT DISTINCT ON (k) a FROM (SELECT a, SUM(id) % 2 AS k FROM t GROUP BY a) s ORDER BY a",
    ),
];

#[test]
fn a_key_reads_the_source_and_the_groups_without_an_order_by() {
    let db = Database::open("memory://distinct_on_key_rule_sources").unwrap();
    fill(&db);
    for sql in [
        "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "INSERT INTO r VALUES (1, 0, 10), (2, 1, 12), (3, 2, 14)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for sql in [
        "SELECT DISTINCT ON (r.a % 2) r.a AS x, r.b AS a, COUNT(*) FROM r GROUP BY r.a, r.b",
        "SELECT DISTINCT ON (a % 2) COUNT(*) FROM t GROUP BY a",
    ] {
        assert_eq!(rows(&db, sql).len(), 2, "{sql}");
    }
}

#[test]
fn a_carried_key_never_reaches_the_columns_beside_a_star() {
    let db = Database::open("memory://distinct_on_key_rule_star").unwrap();
    fill(&db);
    let sql = "SELECT DISTINCT ON (a % 2) *, ROW_NUMBER() OVER (ORDER BY id) FROM t";
    assert_eq!(db.query(sql, ()).unwrap().columns().len(), 4, "{sql}");
}

#[test]
fn a_group_by_position_reads_the_select_list_as_written() {
    let db = Database::open("memory://distinct_on_key_rule_position").unwrap();
    fill(&db);
    let err = db
        .query("SELECT DISTINCT ON (a) COUNT(*) FROM t GROUP BY 2", ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .err();
    assert!(err.is_some(), "GROUP BY 2 names no item of a one-item list");
}

#[test]
fn a_set_operation_branch_carries_no_key_column() {
    let db = Database::open("memory://distinct_on_key_rule_union").unwrap();
    fill(&db);
    let columns = |sql: &str| db.query(sql, ()).unwrap().columns().len();
    let sql = "SELECT DISTINCT ON (a % 2) a, COUNT(*) FROM t GROUP BY a \
               UNION ALL SELECT a, COUNT(*) FROM t GROUP BY a";
    let err = db
        .query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .err();
    assert!(
        err.as_ref()
            .is_some_and(|e| e.to_string().contains("names no column")),
        "a set operation's branch carries no key column: {err:?}"
    );
    assert_eq!(
        columns(
            "SELECT DISTINCT ON (a) a, COUNT(*) FROM t GROUP BY a \
             UNION ALL SELECT a, COUNT(*) FROM t GROUP BY a"
        ),
        2
    );
}

#[test]
fn a_rolled_up_key_reads_the_group_item_it_names() {
    let db = Database::open("memory://distinct_on_key_rule_rollup").unwrap();
    for sql in [
        "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO r VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        rows(
            &db,
            "SELECT DISTINCT ON ((a + 1) % 2) a, a + 1 AS k, COUNT(*) FROM r \
             GROUP BY ROLLUP(a, a + 1) ORDER BY a, k"
        ),
        vec![
            "Integer(1)|Integer(2)|Integer(1)",
            "Integer(1)|Null(Null)|Integer(1)"
        ]
    );
}

#[test]
fn a_key_no_column_holds_is_computed_for_each_output_row() {
    let dir = tempfile::tempdir().unwrap();
    let hot = Database::open("memory://distinct_on_key_rule").unwrap();
    fill(&hot);
    let sealed = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for db in [&hot, &sealed] {
        for (sql, twin) in SHAPES {
            assert_eq!(rows(db, sql), rows(db, twin), "{sql}");
        }
    }
}

#[test]
fn a_key_reads_parameters_in_a_prepared_statement() {
    let db = Database::open("memory://distinct_on_key_rule_params").unwrap();
    fill(&db);
    let stmt = db
        .prepare("SELECT DISTINCT ON (a % $1) a, COUNT(*) FROM t GROUP BY a ORDER BY a % $1, a")
        .unwrap();
    for (modulus, want) in [(2i64, vec![0, 1]), (3, vec![0, 1, 2])] {
        let got: Vec<i64> = stmt
            .query((modulus,))
            .unwrap()
            .map(|row| match row.unwrap().get_value(0).unwrap() {
                Value::Integer(a) => *a,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(got, want, "a % {modulus}");
    }
}

#[test]
fn a_key_naming_nothing_is_an_error() {
    let db = Database::open("memory://distinct_on_key_rule_unknown").unwrap();
    fill(&db);
    let err = db
        .query(
            "SELECT DISTINCT ON (nope) a, COUNT(*) FROM t GROUP BY a ORDER BY a",
            (),
        )
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .err()
        .unwrap_or_else(|| panic!("a key naming no column must fail"));
    assert!(err.to_string().contains("nope"), "{err}");
}
