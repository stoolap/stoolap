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

use sqllogictest::{DBOutput, DefaultColumnType, Runner};
use std::path::Path;
use stoolap::Database;

/// Wrapper around stoolap's Database for sqllogictest
struct StoolapDB {
    db: Database,
}

/// Error wrapper that satisfies sqllogictest's requirements
#[derive(Debug, Clone, PartialEq, Eq)]
struct StoolapError(String);

impl std::fmt::Display for StoolapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StoolapError {}

impl sqllogictest::DB for StoolapDB {
    type Error = StoolapError;
    type ColumnType = DefaultColumnType;

    fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        // Use query() for everything - it works for DDL/DML and SELECT
        let mut rows_result = self
            .db
            .query(sql, ())
            .map_err(|e| StoolapError(e.to_string()))?;

        let columns = rows_result.columns().to_vec();

        // If no columns, this is a DDL/DML statement
        if columns.is_empty() {
            let affected = rows_result.rows_affected();
            return Ok(DBOutput::StatementComplete(affected as u64));
        }

        // Collect rows and determine types from the data
        let num_cols = columns.len();
        let mut result_rows: Vec<Vec<String>> = Vec::new();
        let mut col_types: Vec<DefaultColumnType> = vec![DefaultColumnType::Any; num_cols];
        let mut types_determined = vec![false; num_cols];

        for row in &mut rows_result {
            let row = row.map_err(|e| StoolapError(e.to_string()))?;
            let mut string_row = Vec::with_capacity(num_cols);

            for i in 0..num_cols {
                let value = row.get_value(i);
                match value {
                    Some(stoolap::Value::Null(_)) | None => {
                        string_row.push("NULL".to_string());
                    }
                    Some(stoolap::Value::Integer(v)) => {
                        if !types_determined[i] {
                            col_types[i] = DefaultColumnType::Integer;
                            types_determined[i] = true;
                        }
                        string_row.push(v.to_string());
                    }
                    Some(stoolap::Value::Float(v)) => {
                        if !types_determined[i] {
                            col_types[i] = DefaultColumnType::FloatingPoint;
                            types_determined[i] = true;
                        }
                        // sqllogictest convention: 3 decimal places for floats
                        string_row.push(format!("{:.3}", v));
                    }
                    Some(stoolap::Value::Boolean(b)) => {
                        if !types_determined[i] {
                            col_types[i] = DefaultColumnType::Text;
                            types_determined[i] = true;
                        }
                        string_row.push(if *b {
                            "true".to_string()
                        } else {
                            "false".to_string()
                        });
                    }
                    Some(stoolap::Value::Text(s)) => {
                        if !types_determined[i] {
                            col_types[i] = DefaultColumnType::Text;
                            types_determined[i] = true;
                        }
                        let s = s.to_string();
                        if s.is_empty() {
                            string_row.push("(empty)".to_string());
                        } else {
                            string_row.push(s);
                        }
                    }
                    Some(stoolap::Value::Timestamp(ts)) => {
                        if !types_determined[i] {
                            col_types[i] = DefaultColumnType::Text;
                            types_determined[i] = true;
                        }
                        string_row.push(ts.to_rfc3339());
                    }
                    Some(stoolap::Value::Extension(_)) => {
                        if !types_determined[i] {
                            col_types[i] = DefaultColumnType::Text;
                            types_determined[i] = true;
                        }
                        // Use Display impl for extension types (JSON, Vector, etc.)
                        string_row.push(value.unwrap().to_string());
                    }
                }
            }
            result_rows.push(string_row);
        }

        Ok(DBOutput::Rows {
            types: col_types,
            rows: result_rows,
        })
    }

    fn engine_name(&self) -> &str {
        "stoolap"
    }
}

fn run_slt_file(path: &Path) {
    let mut runner = Runner::new(|| async {
        let db = Database::open_in_memory().expect("Failed to create in-memory database");
        Ok(StoolapDB { db })
    });
    runner
        .run_file(path)
        .unwrap_or_else(|e| panic!("Failed to run SLT file: {}: {}", path.display(), e));
}

fn run_slt_dir(dir: &str) {
    let dir_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/slt")
        .join(dir);
    if !dir_path.exists() {
        panic!("SLT directory not found: {}", dir_path.display());
    }

    let mut files: Vec<_> = std::fs::read_dir(&dir_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "slt"))
        .map(|e| e.path())
        .collect();
    files.sort();

    assert!(
        !files.is_empty(),
        "No .slt files found in {}",
        dir_path.display()
    );

    for file in files {
        run_slt_file(&file);
    }
}

// Test functions for each category

#[test]
fn sqllogictest_basic() {
    run_slt_dir("basic");
}

#[test]
fn sqllogictest_aggregate() {
    run_slt_dir("aggregate");
}

#[test]
fn sqllogictest_join() {
    run_slt_dir("join");
}

#[test]
fn sqllogictest_subquery() {
    run_slt_dir("subquery");
}

#[test]
fn sqllogictest_advanced() {
    run_slt_dir("advanced");
}

#[test]
fn sqllogictest_functions() {
    run_slt_dir("functions");
}

#[test]
fn sqllogictest_index() {
    run_slt_dir("index");
}

#[test]
fn sqllogictest_transaction() {
    run_slt_dir("transaction");
}
