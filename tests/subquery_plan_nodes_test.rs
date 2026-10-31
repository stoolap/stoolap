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

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "CREATE INDEX w_k ON w (k)",
        "CREATE TABLE r (rid INTEGER PRIMARY KEY)",
        "INSERT INTO r VALUES (1), (2), (3), (4)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 4)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 1, NULL), (3, 2, 7)",
        "INSERT INTO w VALUES (1, 1, 1), (2, 1, 3), (3, 2, 5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The lines under EXPLAIN ANALYZE's `Subqueries:` heading, past its indent
fn block_lines(db: &Database, sql: &str) -> Vec<String> {
    let lines: Vec<String> = db
        .query(&format!("EXPLAIN ANALYZE {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    lines
        .iter()
        .skip_while(|line| line.as_str() != "  Subqueries:")
        .skip(1)
        .map(|line| line.strip_prefix("    ").unwrap_or(line).to_string())
        .collect()
}

#[test]
fn explain_analyze_says_how_each_block_answers() {
    let db = setup("subquery_plan_nodes_each_block");
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.k)",
            &["1: EXISTS over w, per call: a keyed set, or where the index on k makes probing cheaper: its pushed conditions"],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.k AND w.v > 0)",
            &["1: EXISTS over w, per call: a keyed set"],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM w WHERE w.k = t.k) FROM t",
            &["1: scalar over w, per call: a keyed set, or where the index on k makes probing cheaper: the count the index on k holds, else a scan"],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM w WHERE w.k = t.k AND w.v > 0) FROM t",
            &["1: scalar over w, per call: a keyed set"],
        ),
        (
            "SELECT id FROM t WHERE k IN (SELECT k FROM w)",
            &["1: IN over w, once per run"],
        ),
        (
            "SELECT id FROM t WHERE EXISTS \
             (SELECT 1 FROM w WHERE w.k = t.k AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k))",
            &[
                "1: EXISTS over w, per call: its pushed conditions",
                "2: EXISTS over u, once per row of 1",
            ],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM w JOIN r ON r.rid = w.id WHERE w.k = t.k)",
            &["1: EXISTS over a statement, per call"],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM w JOIN r ON r.rid = w.id \
             WHERE w.k = t.k AND EXISTS (SELECT 1 FROM u WHERE u.k = w.k))",
            &[
                "1: EXISTS over a statement, per call",
                "  1.1: EXISTS over u, per call: a keyed set, or where the index on k makes probing cheaper: its pushed conditions",
            ],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.k \
             AND EXISTS (SELECT 1 FROM u JOIN r ON r.rid = u.id WHERE u.k = t.k))",
            &[
                "1: EXISTS over w, per call: its pushed conditions",
                "2: EXISTS over a statement, per call",
            ],
        ),
        (
            "SELECT id, (SELECT v FROM w WHERE w.id = t.k) FROM t",
            &["1: scalar over w, per call: the row its key names"],
        ),
        (
            "SELECT id FROM t WHERE k > ALL (SELECT k FROM u WHERE u.k = t.id AND u.v > 0)",
            &["1: ALL over u, per call: its pushed conditions"],
        ),
        (
            "SELECT k, (SELECT COUNT(*) FROM w WHERE w.k = t.k) FROM t GROUP BY k",
            &["1: scalar over w, per call: a keyed set, or where the index on k makes probing cheaper: the count the index on k holds, else a scan"],
        ),
        (
            "SELECT t.id FROM t JOIN r ON r.rid = t.id WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.k)",
            &["1: EXISTS over w, per call: a keyed set, or where the index on k makes probing cheaper: its pushed conditions"],
        ),
        (
            "SELECT t.k, (SELECT COUNT(*) FROM w WHERE w.k = t.k) FROM t JOIN r ON r.rid = t.id \
             WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.v > 0) GROUP BY t.k",
            &[
                "1: EXISTS over u, per call: a keyed set",
                "2: scalar over w, per call: a keyed set, or where the index on k makes probing cheaper: the count the index on k holds, else a scan",
            ],
        ),
        (
            "UPDATE t SET k = k WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.k AND w.v > 0)",
            &["1: EXISTS over w, per call: a keyed set"],
        ),
    ];
    let mut wrong = Vec::new();
    for (sql, expected) in cases {
        let lines = block_lines(&db, sql);
        if lines != *expected {
            wrong.push(format!("{sql}\n  got {lines:?}\n  want {expected:?}"));
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn explain_analyze_lists_no_subqueries_for_a_statement_without_blocks() {
    let db = setup("subquery_plan_nodes_none");
    let lines: Vec<String> = db
        .query("EXPLAIN ANALYZE SELECT id FROM t WHERE k > 1", ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert!(
        !lines.iter().any(|line| line.contains("Subqueries:")),
        "{lines:?}"
    );
}
