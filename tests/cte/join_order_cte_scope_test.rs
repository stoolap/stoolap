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

//! A join order a statement's plan holds for its tables is not taken where
//! a context's CTEs stand for those tables

use std::sync::Arc;

use stoolap::common::{CompactArc, StringMap};
use stoolap::core::{Row, Value};
use stoolap::executor::{ExecutionContext, Executor};
use stoolap::storage::mvcc::engine::MVCCEngine;

/// A context bringing a CTE t (id, k) of the row (1, 1) and a CTE u
/// (id, k, x) of the row (5, 1, 1)
fn cte_context() -> ExecutionContext {
    let mut ctes = StringMap::default();
    for (name, columns, row) in [
        ("t", vec!["id", "k"], vec![1, 1]),
        ("u", vec!["id", "k", "x"], vec![5, 1, 1]),
    ] {
        ctes.insert(
            name.into(),
            (
                CompactArc::new(columns.iter().map(|c| c.to_string()).collect()),
                CompactArc::new(vec![(
                    0,
                    Row::from_values(row.into_iter().map(Value::Integer).collect()),
                )]),
            ),
        );
    }
    ExecutionContext::new().with_cte_data(Arc::new(ctes))
}

fn lines(result: stoolap::Result<Box<dyn stoolap::storage::traits::QueryResult>>) -> Vec<String> {
    let mut result = result.unwrap();
    let mut lines = Vec::new();
    while result.next() {
        lines.push(format!("{:?}", result.row().get(0)));
    }
    lines
}

fn count(result: stoolap::Result<Box<dyn stoolap::storage::traits::QueryResult>>) -> String {
    match result {
        Ok(mut result) => match (result.next(), result.row().get(0)) {
            (true, Some(Value::Integer(n))) => n.to_string(),
            other => format!("{other:?}"),
        },
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_held_join_order_is_not_taken_over_ctes_of_the_tables_names() {
    let engine = MVCCEngine::in_memory();
    engine.open_engine().unwrap();
    let executor = Executor::new(Arc::new(engine));
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 10, g.value % 100 FROM generate_series(1, 2000) g",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO u SELECT g.value, g.value % 10 FROM generate_series(1, 2000) g",
        "CREATE TABLE s (m INTEGER)",
        "INSERT INTO s SELECT g.value FROM generate_series(0, 99) g",
        "CREATE INDEX t_k ON t (k)",
        "CREATE INDEX u_k ON u (k)",
    ] {
        executor.execute(sql).unwrap();
    }
    // Over the tables x is t.x; over the CTEs, u.x
    let sql = "SELECT COUNT(*) FROM t JOIN u ON t.k = u.k JOIN s ON x = s.m WHERE s.m = 1";
    let explained = lines(executor.execute(&format!("EXPLAIN {sql}")));
    assert!(
        explained
            .iter()
            .any(|line| line.contains("Join order: s, t, u")),
        "{explained:?}"
    );
    assert_eq!(count(executor.execute(sql)), "4000", "over the tables");
    assert_eq!(
        count(executor.execute_with_context(sql, &cte_context())),
        "1",
        "over the CTEs, as written"
    );
    assert_eq!(
        count(executor.execute(sql)),
        "4000",
        "over the tables after"
    );
    let analyze = format!("EXPLAIN ANALYZE {sql}");
    let over_tables = lines(executor.execute(&analyze));
    assert!(
        over_tables
            .iter()
            .any(|line| line.contains("Join order: s, t, u")),
        "{over_tables:?}"
    );
    let over_ctes = lines(executor.execute_with_context(&analyze, &cte_context()));
    assert!(
        !over_ctes.iter().any(|line| line.contains("Join order")),
        "the run over the CTEs took the written order: {over_ctes:?}"
    );
}
