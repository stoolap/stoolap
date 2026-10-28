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

//! A statement the planner plans answers as the general path did: its
//! cache entries, its transaction, its names and its cancel

use stoolap::{Database, ExecutionContext, Executor};

fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db
}

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
fn an_offset_leaves_the_cached_answer_whole() {
    let db = table("planned_offset_cache");
    assert_eq!(ids(&db, "SELECT * FROM t WHERE v > 0 OFFSET 1").len(), 2);
    assert_eq!(
        ids(&db, "SELECT * FROM t WHERE v > 0"),
        [1, 2, 3],
        "the cache kept the rows past the OFFSET as the WHERE's answer"
    );
}

#[test]
fn a_key_read_in_a_transaction_reads_its_id() {
    let db = table("planned_transaction_id");
    db.execute("BEGIN", ()).unwrap();
    let id: Option<i64> = db
        .query_one("SELECT CURRENT_TRANSACTION_ID() FROM t WHERE id = 1", ())
        .unwrap();
    assert!(id.is_some(), "the key read lost the transaction's id");
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE id = 1 AND CURRENT_TRANSACTION_ID() IS NOT NULL"
        ),
        [1]
    );
    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn a_program_keeps_each_statements_output_names() {
    let db = table("planned_output_names");
    let rows = db
        .query(
            "SELECT id AS A FROM t WHERE id = 1; SELECT id AS a FROM t WHERE id = 1",
            (),
        )
        .unwrap();
    assert_eq!(rows.columns(), ["a"]);
}

#[test]
fn a_cancelled_key_read_is_refused() {
    let db = table("planned_cancel");
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let statement = stoolap::parser::parse_sql("SELECT id + 1 FROM t WHERE id = 1")
        .unwrap()
        .remove(0);
    let ctx = ExecutionContext::new();
    ctx.cancel();
    let answer = executor.execute_statement(&statement, &ctx);
    assert!(
        matches!(answer, Err(stoolap::Error::QueryCancelled)),
        "a cancelled statement answered"
    );
}

#[test]
fn computed_members_and_bounds_are_read_by_the_index() {
    let db = table("planned_computed_values");
    db.execute("CREATE INDEX t_v ON t(v)", ()).unwrap();
    for (sql, want) in [
        ("SELECT id FROM t WHERE v IN (1 + 1, 2 + 1)", vec![2, 3]),
        (
            "SELECT id FROM t WHERE v BETWEEN 0 + 1 AND 1 + 1",
            vec![1, 2],
        ),
        ("SELECT id FROM t WHERE id IN (1 + 0, 2 + 1)", vec![1, 3]),
    ] {
        assert_eq!(ids(&db, sql), want, "{sql}");
        let plan: Vec<String> = db
            .query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect();
        assert!(
            plan.iter()
                .any(|line| line.trim_start().starts_with("IndexScan")),
            "{sql} reads no index:\n{}",
            plan.join("\n")
        );
    }
}

#[test]
fn a_failing_value_fails_only_for_a_row_that_reaches_it() {
    let db = table("planned_failing_value");
    db.execute("CREATE INDEX t_v ON t(v)", ()).unwrap();
    for sql in [
        "SELECT id FROM t WHERE FALSE AND id IN (ABS('a'))",
        "SELECT id FROM t WHERE FALSE AND v BETWEEN ABS('a') AND 3",
        "SELECT id FROM t WHERE id > 100 AND v = ABS('a')",
        "SELECT id FROM t WHERE id > 100 AND (v = 3 OR v IN (ABS('a')))",
    ] {
        assert_eq!(ids(&db, sql), Vec::<i64>::new(), "{sql}");
    }
    for sql in [
        "SELECT id FROM t WHERE v = ABS('a')",
        "SELECT id FROM t WHERE v IN (1, ABS('a'))",
    ] {
        let answer = db
            .query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
        assert!(answer.is_err(), "{sql} reached its value and answered");
    }
}

#[test]
fn computed_values_under_an_or_are_read_by_the_index() {
    let db = table("planned_computed_or");
    db.execute("CREATE INDEX t_v ON t(v)", ()).unwrap();
    let sql = "SELECT id FROM t WHERE v IN (0 + 1, 1 + 1) OR v = 3";
    assert_eq!(ids(&db, sql), [1, 2, 3]);
    let plan: Vec<String> = db
        .query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert!(
        plan.iter().any(|line| line.trim_start() == "Union"),
        "{sql} reads no index union:\n{}",
        plan.join("\n")
    );
}
