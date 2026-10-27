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

//! A view's query reads the tables and views it names, never a CTE of the
//! statement that reads the view, whichever statement read the view first;
//! a statement run in a context bringing CTEs reads them, and its text's
//! plan stays the one its text alone means, and a plan made for CTEs holds
//! only for CTEs of the same names and columns

use std::sync::Arc;

use stoolap::common::{CompactArc, StringMap};
use stoolap::core::{Row, Value};
use stoolap::executor::{ExecutionContext, Executor};
use stoolap::storage::mvcc::engine::MVCCEngine;
use stoolap::Database;

/// t: 1, 2. u: 1. Each view reads u where a CTE named u would differ
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY)",
    "INSERT INTO t VALUES (1), (2)",
    "INSERT INTO u VALUES (1)",
    "CREATE VIEW v_from AS SELECT id FROM u",
    "CREATE VIEW v_in AS SELECT id FROM t WHERE id IN (SELECT id FROM u)",
    "CREATE VIEW v_exists AS SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id)",
    "CREATE VIEW v_join AS SELECT t.id FROM t JOIN u ON u.id = t.id \
     WHERE t.id <= (SELECT MAX(id) FROM u)",
];

const VIEWS: &[&str] = &["v_from", "v_in", "v_exists", "v_join"];

/// Each way a statement reads `{v}`, with a CTE u of the row 2
const READS: &[&str] = &[
    "WITH u(id) AS (SELECT 2) SELECT id FROM {v}",
    "WITH u(id) AS (SELECT 2) SELECT x.id FROM t JOIN {v} AS x ON x.id = t.id",
    "WITH u(id) AS (SELECT 2) SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {v} AS x WHERE x.id = t.id)",
    "WITH u(id) AS (SELECT 2) SELECT id FROM t WHERE id IN (SELECT id FROM {v}) \
     AND id > (SELECT MIN(id) - 2 FROM u)",
];

fn ids(db: &Database, sql: &str, prepared: bool) -> Result<Vec<i64>, String> {
    let rows = match prepared {
        true => db
            .prepare(sql)
            .and_then(|statement| statement.query(()))
            .map_err(|e| e.to_string())?,
        false => db.query(sql, ()).map_err(|e| e.to_string())?,
    };
    let mut out = rows
        .map(|row| {
            row.and_then(|row| row.get::<i64>(0))
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    out.sort();
    Ok(out)
}

fn open(name: &str, sealed: bool, dir: &tempfile::TempDir) -> Database {
    let db = match sealed {
        true => Database::open(&format!("file://{}/{name}", dir.path().display())).unwrap(),
        false => Database::open(&format!("memory://view_cte_scope_{name}")).unwrap(),
    };
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

#[test]
fn a_view_reads_its_own_tables_under_a_cte_of_the_same_name() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        for prepared in [false, true] {
            for read_first in [true, false] {
                for (index, view) in VIEWS.iter().enumerate() {
                    let dir = tempfile::tempdir().unwrap();
                    let name = format!("{sealed}_{prepared}_{read_first}_{index}");
                    let db = open(&name, sealed, &dir);
                    let plain = format!("SELECT id FROM {view}");
                    if read_first {
                        ids(&db, &plain, prepared).unwrap();
                    }
                    for read in READS {
                        let sql = read.replace("{v}", view);
                        let got = ids(&db, &sql, prepared);
                        if got != Ok(vec![1]) {
                            wrong.push(format!(
                                "sealed {sealed}, prepared {prepared}, read first {read_first}: \
                                 {sql}: {got:?}"
                            ));
                        }
                    }
                    let got = ids(&db, &plain, prepared);
                    if got != Ok(vec![1]) {
                        wrong.push(format!(
                            "sealed {sealed}, prepared {prepared}, read first {read_first}: \
                             {plain} after: {got:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A context bringing a CTE `name` of the row 2
fn cte_context(name: &str) -> ExecutionContext {
    cte_context_of(name, &["id"], vec![Value::Integer(2)])
}

/// A context bringing a CTE `name` of `columns` and the one row `row`
fn cte_context_of(name: &str, columns: &[&str], row: Vec<Value>) -> ExecutionContext {
    let mut ctes = StringMap::default();
    ctes.insert(
        name.into(),
        (
            CompactArc::new(columns.iter().map(|c| c.to_string()).collect()),
            CompactArc::new(vec![(0, Row::from_values(row))]),
        ),
    );
    ExecutionContext::new().with_cte_data(Arc::new(ctes))
}

fn executor_ids(
    result: stoolap::Result<Box<dyn stoolap::storage::traits::QueryResult>>,
) -> Result<Vec<i64>, String> {
    let mut result = result.map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    while result.next() {
        match result.row().get(0) {
            Some(Value::Integer(id)) => out.push(*id),
            other => return Err(format!("{other:?}")),
        }
    }
    out.sort();
    Ok(out)
}

#[test]
fn a_statement_in_a_context_bringing_ctes_keeps_its_text_s_plan_apart() {
    let texts = [
        "SELECT id FROM t WHERE id IN (SELECT id FROM u)",
        "SELECT id FROM u WHERE id = 2",
        "SELECT t.id FROM t JOIN u ON u.id = t.id WHERE t.id <= (SELECT MAX(id) FROM u)",
        "SELECT t.id FROM t JOIN u ON EXISTS (SELECT 1 FROM u w WHERE w.id = t.id)",
    ];
    let mut wrong = Vec::new();
    for (index, text) in texts.iter().enumerate() {
        for context_first in [true, false] {
            let engine = MVCCEngine::in_memory();
            engine.open_engine().unwrap();
            let executor = Executor::new(Arc::new(engine));
            for sql in &SETUP[..4] {
                executor.execute(sql).unwrap();
            }
            let plain = |executor: &Executor| executor_ids(executor.execute(text));
            let want_plain = if index == 1 { vec![] } else { vec![1] };
            if !context_first {
                let got = plain(&executor);
                if got != Ok(want_plain.clone()) {
                    wrong.push(format!("{text}: plain first: {got:?}"));
                }
            }
            let got = executor_ids(executor.execute_with_context(text, &cte_context("u")));
            if got != Ok(vec![2]) {
                wrong.push(format!(
                    "{text}: with the CTE, context first {context_first}: {got:?}"
                ));
            }
            let got = plain(&executor);
            if got != Ok(want_plain.clone()) {
                wrong.push(format!(
                    "{text}: plain after, context first {context_first}: {got:?}"
                ));
            }
            let view = format!("CREATE VIEW v AS {text}");
            let got = executor
                .execute(&view)
                .map_err(|e| e.to_string())
                .and_then(|_| executor_ids(executor.execute("SELECT id FROM v")));
            if got != Ok(want_plain) {
                wrong.push(format!(
                    "{text}: as a view, context first {context_first}: {got:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_join_on_subquery_runs_as_its_blocks_in_a_context_bringing_ctes() {
    let engine = MVCCEngine::in_memory();
    engine.open_engine().unwrap();
    let executor = Executor::new(Arc::new(engine));
    for sql in &SETUP[..4] {
        executor.execute(sql).unwrap();
    }
    let sql = "SELECT t.id FROM t JOIN u ON EXISTS (SELECT 1 FROM u w WHERE w.id = t.id)";
    for context in [
        cte_context("extra"),
        cte_context("extra"),
        cte_context("other"),
    ] {
        assert_eq!(
            executor_ids(executor.execute_with_context(sql, &context)),
            Ok(vec![1]),
            "a CTE the statement does not read leaves its answer"
        );
    }
}

#[test]
fn a_cte_of_the_same_name_with_other_columns_gets_its_own_plan() {
    let engine = MVCCEngine::in_memory();
    engine.open_engine().unwrap();
    let executor = Executor::new(Arc::new(engine));
    for sql in &SETUP[..3] {
        executor.execute(sql).unwrap();
    }
    let texts = [
        "SELECT t.id FROM t JOIN u ON u.id = t.id WHERE t.id <= (SELECT MAX(id) FROM t)",
        "SELECT id FROM t WHERE id IN (SELECT id FROM u)",
        "SELECT id FROM u",
    ];
    let mut wrong = Vec::new();
    for text in texts {
        for (columns, row) in [
            (&["id"][..], vec![Value::Integer(1)]),
            (
                &["extra", "id"][..],
                vec![Value::Integer(99), Value::Integer(1)],
            ),
            (
                &["id", "extra"][..],
                vec![Value::Integer(1), Value::Integer(99)],
            ),
        ] {
            let context = cte_context_of("u", columns, row);
            let got = executor_ids(executor.execute_with_context(text, &context));
            if got != Ok(vec![1]) {
                wrong.push(format!("{text}: u{columns:?}: {got:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_plan_made_in_a_context_bringing_ctes_is_kept_for_the_same_names() {
    use stoolap::test_failpoints::{bridge_compiles, program_compiles};
    let engine = MVCCEngine::in_memory();
    engine.open_engine().unwrap();
    let executor = Executor::new(Arc::new(engine));
    for sql in &SETUP[..4] {
        executor.execute(sql).unwrap();
    }
    let sql = "SELECT t.id FROM t JOIN u ON EXISTS (SELECT 1 FROM u w WHERE w.id = t.id)";
    executor_ids(executor.execute_with_context(sql, &cte_context("u"))).unwrap();
    let compiles = (bridge_compiles(), program_compiles());
    let again = executor_ids(executor.execute_with_context(sql, &cte_context("u")));
    assert_eq!(again, Ok(vec![2]));
    assert_eq!(
        (bridge_compiles(), program_compiles()),
        compiles,
        "the same CTE names keep the plan"
    );
}
