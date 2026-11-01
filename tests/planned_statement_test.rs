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
fn an_aggregate_plan_keeps_every_clause_of_its_statement() {
    let db = table("planned_aggregate_clauses");
    let answer = |sql: &str| {
        db.query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .map(|rows| {
                rows.iter()
                    .map(|row| format!("{:?}", row.get_value(0)))
                    .collect::<Vec<_>>()
            })
            .map_err(|e| e.to_string())
    };
    for sql in [
        "SELECT DISTINCT ON (2) MIN(id) FROM t",
        "SELECT DISTINCT ON (ABS('a')) MIN(id) FROM t",
        "SELECT DISTINCT ON (1) COUNT(*) FROM t",
        "SELECT DISTINCT COUNT(*) FROM t",
        "SELECT MIN(id) FROM t WINDOW w AS (ORDER BY v)",
    ] {
        // The first run plans the statement, the second reads by the plan
        let planning = answer(sql);
        assert_eq!(
            answer(sql),
            planning,
            "{sql}: the planned run answered otherwise"
        );
    }
    assert!(answer("SELECT DISTINCT ON (2) MIN(id) FROM t").is_err());
}

#[test]
fn a_cancelled_aggregate_plan_is_refused() {
    let db = table("planned_aggregate_cancel");
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    for sql in ["SELECT MIN(id) FROM t", "SELECT COUNT(*) FROM t"] {
        // The first run plans it, the second reads by the plan
        executor.execute(sql).unwrap();
        executor.execute(sql).unwrap();
        let ctx = ExecutionContext::new();
        ctx.cancel();
        let answer = executor.execute_with_context(sql, &ctx);
        assert!(
            matches!(answer, Err(stoolap::Error::QueryCancelled)),
            "{sql}: a cancelled statement answered"
        );
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_planned_aggregate_of_several_calls_reads_the_statement_view() {
    use std::cell::Cell;
    use std::rc::Rc;
    let db = table("planned_aggregate_several_calls");
    let sql = "SELECT COUNT(*) AS a, COUNT(1) AS b FROM t";
    db.query(sql, ()).unwrap().for_each(drop);
    db.query(sql, ()).unwrap().for_each(drop);
    let taken = Rc::new(Cell::new(false));
    let flag = Rc::clone(&taken);
    stoolap::test_failpoints::after_statement_view_taken(move || flag.set(true));
    db.query(sql, ()).unwrap().for_each(drop);
    stoolap::test_failpoints::after_statement_view_taken(|| {});
    assert!(
        taken.get(),
        "two counts read one by one, a commit between them unseen"
    );
}

#[test]
fn a_source_row_past_the_limit_is_never_computed() {
    let db = table("planned_source_row_past_limit");
    let fails_at_3 = "CASE WHEN id = 3 THEN ABS('a') ELSE id END";
    db.execute(
        &format!("CREATE VIEW w AS SELECT id, {fails_at_3} AS q FROM t"),
        (),
    )
    .unwrap();
    let all = db
        .query("SELECT id, q FROM w", ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    assert!(all.is_err(), "the view's third row computed its value");
    for sql in [
        "SELECT id FROM w LIMIT 1".to_string(),
        format!("SELECT id FROM (SELECT id, {fails_at_3} AS q FROM t) AS d LIMIT 1"),
        "SELECT id, q FROM w LIMIT 2".to_string(),
    ] {
        let answer = db
            .query(&sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
        assert!(answer.is_ok(), "{sql}: {:?}", answer.err());
    }
}

#[test]
fn a_cte_answers_alike_whether_or_not_it_hides_a_view() {
    let db = table("planned_cte_hides_view");
    let fails_at_3 = "CASE WHEN i = 3 THEN ABS('a') ELSE i END";
    let statements = [
        format!("WITH c(i) AS (SELECT id FROM t) SELECT {fails_at_3} FROM c LIMIT 1"),
        format!("WITH c AS (SELECT id AS i FROM t) SELECT {fails_at_3} FROM c LIMIT 1"),
    ];
    let read = |sql: &str| {
        db.query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .map(|rows| rows.len())
            .map_err(|e| e.to_string())
    };
    let alone: Vec<_> = statements.iter().map(|sql| read(sql)).collect();
    db.execute("CREATE VIEW c AS SELECT 0 AS i", ()).unwrap();
    for (sql, alone) in statements.iter().zip(alone) {
        assert_eq!(read(sql), alone, "{sql}: a view the CTE hides changed it");
        assert_eq!(
            alone,
            Ok(1),
            "{sql}: a row past the LIMIT computed its value"
        );
    }
}

#[test]
fn a_failing_parameter_or_subquery_fails_only_where_a_row_reaches_it() {
    let db = table("planned_failing_parameter");
    db.execute("CREATE INDEX t_v ON t(v)", ()).unwrap();
    let answer = |sql: &str| {
        db.query(sql, ("a",))
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .map(|rows| rows.len())
    };
    for (sql, want) in [
        ("SELECT id FROM t WHERE FALSE AND id IN (ABS($1))", 0),
        ("SELECT id FROM t WHERE v > 0 OR v = ABS($1)", 3),
        (
            "SELECT id FROM t WHERE id > 100 AND v BETWEEN ABS($1) AND 3",
            0,
        ),
        (
            "SELECT id FROM t WHERE id > 100 AND v = (SELECT v FROM t)",
            0,
        ),
        (
            "SELECT id FROM t WHERE id > 100 AND v IN ((SELECT v FROM t), 2)",
            0,
        ),
    ] {
        for run in 0..2 {
            assert_eq!(answer(sql).ok(), Some(want), "{sql}, run {run}");
        }
    }
    for sql in [
        "SELECT id FROM t WHERE v = ABS($1)",
        "SELECT id FROM t WHERE v < 0 OR v = ABS($1)",
        "SELECT id FROM t WHERE v = (SELECT v FROM t)",
    ] {
        assert!(answer(sql).is_err(), "{sql} reached its value and answered");
    }
}

#[test]
fn explain_names_each_read_a_run_may_take() {
    let db = table("planned_explain_branches");
    let explain = |sql: &str| -> Vec<String> {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect()
    };
    for (sql, plan) in [
        (
            "SELECT id FROM t ORDER BY v LIMIT 2",
            vec![
                "Limit 2",
                "  Project id",
                "    TopK t v ASC",
                "    else TopN v ASC",
                "      SeqScan t",
            ],
        ),
        (
            "SELECT id FROM t ORDER BY v",
            vec!["Sort v ASC", "  Project id", "    SeqScan t"],
        ),
        (
            "SELECT id FROM t WHERE id = 1",
            vec![
                "Project id",
                "  PkGet t id = 1",
                "  else IndexScan t (id = 1)",
                "    PkKey",
            ],
        ),
        (
            "SELECT v FROM t WHERE id = 1 ORDER BY v LIMIT 1",
            vec![
                "Limit 1",
                "  Project v",
                "    PkGet t id = 1",
                "    else TopK t v ASC (id = 1)",
                "    else TopN v ASC",
                "      IndexScan t (id = 1)",
                "        PkKey",
            ],
        ),
        (
            "SELECT DISTINCT v FROM t",
            vec![
                "Distinct",
                "  Project v",
                "    DistinctValues t (v)",
                "    else SeqScan t",
            ],
        ),
        (
            "SELECT DISTINCT v FROM t WHERE v > 1 ORDER BY v LIMIT 1",
            vec![
                "Limit 1",
                "  TopN v ASC",
                "    Distinct",
                "      Project v",
                "        SeqScan t (v > 1)",
            ],
        ),
        (
            "SELECT DISTINCT ON (v) id, v FROM t ORDER BY v, id",
            vec![
                "DistinctOn v",
                "  Sort v ASC, id ASC",
                "    Project id, v",
                "      SeqScan t",
            ],
        ),
        (
            "SELECT DISTINCT ON (v) id, v FROM t ORDER BY v, id LIMIT 1",
            vec![
                "Limit 1",
                "  DistinctOn v",
                "    Sort v ASC, id ASC",
                "      Project id, v",
                "        SeqScan t",
            ],
        ),
        (
            "SELECT id FROM t WHERE v > 1 UNION ALL SELECT id FROM t ORDER BY id LIMIT 2",
            vec![
                "Limit 2",
                "  TopN id ASC",
                "    UNION ALL",
                "      Project id",
                "        SeqScan t (v > 1)",
                "      Project id",
                "        SeqScan t",
            ],
        ),
        (
            "SELECT id FROM t EXCEPT SELECT v FROM t INTERSECT SELECT id FROM t",
            vec![
                "EXCEPT",
                "  Project id",
                "    SeqScan t",
                "  INTERSECT",
                "    Project v",
                "      SeqScan t",
                "    Project id",
                "      SeqScan t",
            ],
        ),
        (
            "SELECT SUM(v), COUNT(*) FROM t WHERE v > 1",
            vec![
                "Aggregate SUM(v), COUNT(*)",
                "  StorageFiltered t (v > 1)",
                "  else SeqScan t (v > 1)",
            ],
        ),
        (
            "SELECT v, COUNT(*) FROM t GROUP BY v HAVING COUNT(*) > 0",
            vec![
                "Having (COUNT(*) > 0)",
                "  Aggregate v, COUNT(*) by v",
                "    SortedStream t (v)",
                "    else SeqScan t",
            ],
        ),
        (
            "SELECT v, COUNT(*) AS n FROM t GROUP BY v ORDER BY n DESC LIMIT 1",
            vec![
                "Limit 1",
                "  TopN n DESC",
                "    Aggregate v, n by v",
                "      StorageGrouped t (v)",
                "      else SeqScan t",
            ],
        ),
        (
            "SELECT id FROM t WHERE 1 = 0 AND v > 1",
            vec![
                "Project id",
                "  Empty unless (1 = 0)",
                "  else SeqScan t ((1 = 0) AND (v > 1))",
            ],
        ),
    ] {
        assert_eq!(explain(sql), plan, "{sql}");
    }
}

#[test]
fn explain_names_the_row_a_key_write_reads() {
    let db = table("planned_key_writes");
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER REFERENCES p(id))",
        (),
    )
    .unwrap();
    let explain = |sql: &str| -> Vec<String> {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect()
    };
    for (sql, plan) in [
        (
            "UPDATE t SET v = v + 1 WHERE id = $1",
            vec![
                "Update t SET v = (v + 1)",
                "  PkGet t id = $1",
                "  else IndexScan t (id = $1)",
                "    PkKey",
            ],
        ),
        (
            "DELETE FROM t WHERE 2 = id",
            vec![
                "Delete t",
                "  PkGet t id = 2",
                "  else IndexScan t (2 = id)",
                "    PkKey",
            ],
        ),
        // A key write the general path runs: a parent's delete keeps its
        // children's keys
        (
            "DELETE FROM p WHERE id = 1",
            vec!["DELETE FROM p", "  Filter: (id = 1)"],
        ),
        (
            "UPDATE t SET v = 1 WHERE id = 1 RETURNING v",
            vec!["UPDATE t", "  Set: 1 column(s)", "  Filter: (id = 1)"],
        ),
    ] {
        assert_eq!(explain(sql), plan, "{sql}");
    }
}

#[test]
fn a_declined_filtered_aggregate_reads_by_the_where_its_scan_reads() {
    let db = table("planned_filtered_aggregate_scan");
    db.execute("CREATE INDEX t_v ON t(v)", ()).unwrap();
    let lines: Vec<String> = db
        .query("EXPLAIN SELECT SUM(v) FROM t WHERE FALSE OR v > 1", ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(
        lines,
        [
            "Aggregate SUM(v)",
            "  StorageFiltered t (FALSE OR (v > 1))",
            "  else SeqScan t (FALSE OR (v > 1))",
        ],
        "the scan's indexes"
    );
}

#[test]
fn explain_names_the_source_a_statement_reads() {
    let db = table("planned_explain_sources");
    db.execute("CREATE VIEW w AS SELECT id, v FROM t", ())
        .unwrap();
    let explain = |sql: &str| -> Vec<String> {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect()
    };
    for (sql, plan) in [
        (
            "SELECT a FROM (SELECT id AS a, v FROM t) d WHERE v > 1 ORDER BY a LIMIT 2",
            vec![
                "Limit 2",
                "  TopN a ASC",
                "    Project a",
                "      Derived d (v > 1)",
                "        Project a, v",
                "          SeqScan t",
            ],
        ),
        (
            "SELECT id FROM w WHERE v > 1 ORDER BY v DESC",
            vec![
                "Sort v DESC",
                "  Project id",
                "    View w (v > 1)",
                "      Project id, v",
                "        SeqScan t",
            ],
        ),
        (
            "WITH c AS (SELECT id, v FROM t) SELECT id FROM c WHERE v = 2",
            vec![
                "Project id",
                "  Derived c (v = 2)",
                "    Project id, v",
                "      SeqScan t",
            ],
        ),
        (
            "WITH c (i, x) AS (SELECT id, v FROM t) SELECT i FROM c WHERE x = 2",
            vec![
                "Project i",
                "  Cte c (x = 2)",
                "    Project id, v",
                "      SeqScan t",
            ],
        ),
        (
            "SELECT b FROM (SELECT x AS b FROM (VALUES (1), (2)) AS z(x) WHERE x > 1) d",
            vec![
                "Project b",
                "  Derived d",
                "    Project b",
                "      Values z (x > 1)",
            ],
        ),
        (
            "SELECT x FROM (VALUES (1), (2)) AS z(x) ORDER BY x LIMIT 1",
            vec!["Limit 1", "  TopN x ASC", "    Project x", "      Values z"],
        ),
        (
            "SELECT value FROM generate_series(1, 5) WHERE value > 2",
            vec!["Project value", "  Function generate_series (value > 2)"],
        ),
        (
            "SELECT id FROM (SELECT id, v FROM t WHERE id = 2) t WHERE v = 2",
            vec![
                "Project id",
                "  Derived t (v = 2)",
                "    Project id, v",
                "      IndexScan t (id = 2)",
                "        PkKey",
            ],
        ),
        ("SELECT 1 + 1", vec!["Project 1 + 1", "  OneRow"]),
    ] {
        assert_eq!(explain(sql), plan, "{sql}");
    }
}

#[test]
fn a_count_read_per_run_takes_the_ordered_read() {
    let db = table("planned_count_per_run");
    for (sql, want, plan) in [
        (
            "SELECT id FROM t ORDER BY id LIMIT $1",
            vec![1, 2],
            vec![
                "Limit $1",
                "  Project id",
                "    IndexOrder t id ASC",
                "    else TopK t id ASC",
                "    else TopN id ASC",
                "      SeqScan t",
            ],
        ),
        (
            "SELECT id FROM t ORDER BY id DESC LIMIT (SELECT 1) OFFSET $1",
            vec![1],
            vec![
                "Limit (SELECT 1) offset $1",
                "  Project id",
                "    IndexOrder t id DESC",
                "    else TopK t id DESC",
                "    else TopN id DESC",
                "      SeqScan t",
            ],
        ),
    ] {
        let got: Vec<i64> = db
            .query(sql, (2,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(got, want, "{sql}");
        let lines: Vec<String> = db
            .query(&format!("EXPLAIN {sql}"), (2,))
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect();
        assert_eq!(lines, plan, "{sql}");
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
