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

//! A subquery's join condition reads the parent row's columns it names

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Result<Vec<Vec<i64>>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| row.get::<Option<i64>>(i).unwrap().unwrap_or(-1))
                .collect(),
        );
    }
    out.sort_unstable();
    Ok(out)
}

fn create(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 1, 'a'), (2, 2, 'b'), (3, 3, 'c')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 1)",
        "CREATE TABLE v (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO v VALUES (1, 1), (2, 2), (3, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn an_inner_join_condition_answers_as_the_same_test_in_the_where() {
    let db = create("memory://an_inner_join_condition_answers_as_the_where");
    let mut failed = Vec::new();
    for (on, in_where) in [
        (
            "(SELECT COUNT(*) FROM u JOIN v ON v.id = u.id AND v.k = t.k)",
            "(SELECT COUNT(*) FROM u JOIN v ON v.id = u.id WHERE v.k = t.k)",
        ),
        (
            "(SELECT MAX(v.id) FROM u JOIN v ON v.id = u.id AND v.k = t.k)",
            "(SELECT MAX(v.id) FROM u JOIN v ON v.id = u.id WHERE v.k = t.k)",
        ),
        (
            "(SELECT COUNT(*) FROM u JOIN v ON v.k = t.k)",
            "(SELECT COUNT(*) FROM u CROSS JOIN v WHERE v.k = t.k)",
        ),
    ] {
        for shape in [
            "SELECT id, {} FROM t",
            "SELECT id FROM t WHERE {} = 1",
            "SELECT id FROM t WHERE s || 'x' <> 'zx' AND {} = 1",
            "SELECT id FROM t WHERE id IN (1, 2, 3) AND {} = 1",
        ] {
            let got = rows(&db, &shape.replace("{}", on));
            let want = rows(&db, &shape.replace("{}", in_where));
            if got != want {
                failed.push(format!(
                    "{}: {got:?}, in the WHERE {want:?}",
                    shape.replace("{}", on)
                ));
            }
        }
    }
    for (on, in_where) in [
        (
            "EXISTS (SELECT 1 FROM u JOIN v ON v.id = u.id AND v.k = t.k)",
            "EXISTS (SELECT 1 FROM u JOIN v ON v.id = u.id WHERE v.k = t.k)",
        ),
        (
            "NOT EXISTS (SELECT 1 FROM u JOIN v ON v.id = u.id AND v.k = t.k)",
            "NOT EXISTS (SELECT 1 FROM u JOIN v ON v.id = u.id WHERE v.k = t.k)",
        ),
        (
            "1 IN (SELECT u.g FROM u JOIN v ON v.id = u.id AND v.k = t.k)",
            "1 IN (SELECT u.g FROM u JOIN v ON v.id = u.id WHERE v.k = t.k)",
        ),
    ] {
        for filter in ["", "s || 'x' <> 'zx' AND ", "id IN (1, 2, 3) AND "] {
            let got = rows(&db, &format!("SELECT id FROM t WHERE {filter}{on}"));
            let want = rows(&db, &format!("SELECT id FROM t WHERE {filter}{in_where}"));
            if got != want {
                failed.push(format!(
                    "WHERE {filter}{on}: {got:?}, in the WHERE {want:?}"
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn an_outer_join_and_a_nested_join_read_the_parent_row_in_their_condition() {
    let db = create("memory://an_outer_join_reads_the_parent_row_in_its_condition");
    let mut failed = Vec::new();
    for (sql, want) in [
        (
            "SELECT id, (SELECT COUNT(v.id) FROM u LEFT JOIN v ON v.id = u.id AND v.k = t.k) FROM t",
            vec![vec![1, 1], vec![2, 1], vec![3, 0]],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM u LEFT JOIN v ON v.id = u.id AND v.k = t.k) FROM t",
            vec![vec![1, 2], vec![2, 2], vec![3, 2]],
        ),
        (
            "SELECT id, (SELECT COUNT(u.id) FROM u RIGHT JOIN v ON v.id = u.id AND v.k = t.k) FROM t",
            vec![vec![1, 1], vec![2, 1], vec![3, 0]],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM u JOIN v ON v.id = u.id \
             JOIN v AS w ON w.id = v.id AND w.k = t.k) FROM t",
            vec![vec![1, 1], vec![2, 1], vec![3, 0]],
        ),
        (
            "SELECT o.id, (SELECT COUNT(*) FROM u JOIN v ON v.id = u.id AND v.k = o.k) FROM t AS o",
            vec![vec![1, 1], vec![2, 1], vec![3, 0]],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM (SELECT u.id FROM u JOIN v \
             ON v.id = u.id AND v.k = t.k) AS d) FROM t",
            vec![vec![1, 1], vec![2, 1], vec![3, 0]],
        ),
        (
            "SELECT id FROM t WHERE s || 'x' <> 'zx' \
             AND (SELECT COUNT(v.id) FROM u LEFT JOIN v ON v.id = u.id AND v.k = t.k) = 1",
            vec![vec![1], vec![2]],
        ),
    ] {
        let got = rows(&db, sql);
        if got.as_ref() != Ok(&want) {
            failed.push(format!("{sql}: {got:?}"));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}
