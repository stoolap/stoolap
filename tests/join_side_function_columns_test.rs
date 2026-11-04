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

//! A join's table function side registered again with its columns in
//! another order fails as a changed schema, never reading a column by its
//! old position. The only test of this binary: it registers a function
//! process-wide

use std::sync::Arc;

use stoolap::core::{Result, Row, RowVec, Value};
use stoolap::functions::global_registry;
use stoolap::functions::tvf::TableValuedFunction;
use stoolap::Database;

/// One row, `n` = 1 and `v` = 99, its columns swapped when `.0`
struct MyRows(bool);

impl TableValuedFunction for MyRows {
    fn name(&self) -> &str {
        "MY_ROWS"
    }

    fn column_names(&self) -> Vec<String> {
        match self.0 {
            false => vec!["n".into(), "v".into()],
            true => vec!["v".into(), "n".into()],
        }
    }

    fn generate(&self, _args: &[Value], _limit: Option<usize>) -> Result<RowVec> {
        let (n, v) = (Value::Integer(1), Value::Integer(99));
        let values = match self.0 {
            false => vec![n, v],
            true => vec![v, n],
        };
        let mut rows = RowVec::new();
        rows.push((0, Row::from_values(values)));
        Ok(rows)
    }
}

#[test]
fn a_table_function_side_registered_again_reads_no_column_by_its_old_place() {
    global_registry()
        .register_tvf("MY_ROWS", Arc::new(|| Box::new(MyRows(false))))
        .unwrap();
    let db = Database::open("memory://join_side_function_columns").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY)",
        "INSERT INTO t VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT s.n FROM MY_ROWS() AS s JOIN t ON t.id = s.n LIMIT 1";
    let statement = db.prepare(sql).unwrap();
    let run = || -> std::result::Result<Vec<i64>, String> {
        let rows = statement.query(()).map_err(|e| e.to_string())?;
        rows.map(|row| {
            row.map_err(|e| e.to_string())?
                .get::<i64>(0)
                .map_err(|e| e.to_string())
        })
        .collect()
    };
    let first = run();
    global_registry()
        .register_tvf("MY_ROWS", Arc::new(|| Box::new(MyRows(true))))
        .unwrap();
    let again = run();
    let changed = again.as_ref().is_err_and(|e| e.contains("schema changed"));
    assert!(
        first == Ok(vec![1]) && changed,
        "{sql}: {first:?}, then with its columns swapped {again:?}"
    );
}
