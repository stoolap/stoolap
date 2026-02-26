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

//! WASM bindings for Stoolap
//!
//! Provides a JavaScript-friendly API for running Stoolap in the browser.
//! Mirrors the CLI (`src/bin/stoolap.rs`) interface: transaction support,
//! proper SQL splitting, and the same query detection logic.
//!
//! Only available when compiled for `wasm32` targets.

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use crate::api::{Database, Transaction as ApiTransaction};
use crate::common::version::{GIT_COMMIT, VERSION};
use crate::core::types::DataType;
use crate::core::Value;

/// A Stoolap database instance for use from JavaScript.
///
/// Create with `new StoolapDB()`, then call `execute(sql)` to run queries.
/// Returns JSON strings for all results. Supports transactions via
/// `begin()`, `commit()`, and `rollback()`.
#[wasm_bindgen]
pub struct StoolapDB {
    db: Database,
    tx: RefCell<Option<ApiTransaction>>,
}

#[wasm_bindgen]
impl StoolapDB {
    /// Create a new in-memory Stoolap database.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<StoolapDB, JsValue> {
        // Set panic hook once (avoid leaking on repeated new() calls)
        use std::sync::Once;
        static SET_HOOK: Once = Once::new();
        SET_HOOK.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                let msg = format!("stoolap panic: {}", info);
                web_sys::console::error_1(&msg.into());
            }));
        });

        let db = Database::open_in_memory()
            .map_err(|e| JsValue::from_str(&format!("Failed to open database: {}", e)))?;
        Ok(StoolapDB {
            db,
            tx: RefCell::new(None),
        })
    }

    /// Return the Stoolap version string (e.g. "0.3.1-abc1234").
    pub fn version(&self) -> String {
        if GIT_COMMIT != "unknown" {
            let short = if GIT_COMMIT.len() > 7 {
                &GIT_COMMIT[..7]
            } else {
                GIT_COMMIT
            };
            format!("{}-{}", VERSION, short)
        } else {
            VERSION.to_string()
        }
    }

    /// Execute a SQL statement and return the result as a JSON string.
    ///
    /// Returns JSON with one of these shapes:
    /// - `{ "type": "rows", "columns": [...], "rows": [[...], ...], "count": N }`
    /// - `{ "type": "affected", "affected": N }`
    /// - `{ "type": "error", "message": "..." }`
    ///
    /// Transaction commands (BEGIN, COMMIT, ROLLBACK) are handled automatically.
    pub fn execute(&self, sql: &str) -> String {
        self.execute_inner(sql)
    }

    /// Execute multiple semicolon-separated SQL statements.
    /// Handles quotes and comments correctly (like the CLI).
    /// Returns the result of the last statement.
    pub fn execute_batch(&self, sql: &str) -> String {
        let mut last_result = String::from(r#"{"type":"affected","affected":0}"#);
        for stmt in split_sql_statements(sql) {
            let trimmed = stmt.trim();
            if trimmed.is_empty() {
                continue;
            }
            last_result = self.execute_inner(trimmed);
            if last_result.starts_with(r#"{"type":"error"#) {
                return last_result;
            }
        }
        last_result
    }
}

impl StoolapDB {
    fn execute_inner(&self, sql: &str) -> String {
        let trimmed = sql.trim();
        if trimmed.is_empty() {
            return r#"{"type":"affected","affected":0}"#.to_string();
        }

        let upper = trimmed.to_uppercase();
        let upper = upper.trim();

        // Handle transaction commands (mirrors CLI's execute_query)
        if upper.starts_with("BEGIN") {
            return self.begin_transaction();
        } else if upper == "COMMIT" {
            return self.commit_transaction();
        } else if upper == "ROLLBACK" {
            return self.rollback_transaction();
        }

        // Check if it's a query that returns rows (mirrors CLI's detection logic)
        // Include statements with RETURNING clause (INSERT/UPDATE/DELETE RETURNING)
        let is_query = upper.starts_with("SELECT")
            || upper.starts_with("WITH")
            || upper.starts_with("SHOW")
            || upper.starts_with("DESCRIBE")
            || upper.starts_with("DESC ")
            || upper.starts_with("EXPLAIN")
            || upper.starts_with("VACUUM")
            || (upper.starts_with("PRAGMA") && !upper.contains('='))
            || upper.contains(" RETURNING ")
            || upper.ends_with(" RETURNING");

        if is_query {
            self.execute_read_query(trimmed)
        } else {
            self.execute_write_query(trimmed)
        }
    }

    fn begin_transaction(&self) -> String {
        let mut tx_ref = self.tx.borrow_mut();
        if tx_ref.is_some() {
            return error_json("already in a transaction");
        }
        match self.db.begin() {
            Ok(tx) => {
                *tx_ref = Some(tx);
                r#"{"type":"affected","affected":0}"#.to_string()
            }
            Err(e) => error_json(&e.to_string()),
        }
    }

    fn commit_transaction(&self) -> String {
        let mut tx_ref = self.tx.borrow_mut();
        match tx_ref.take() {
            Some(mut tx) => match tx.commit() {
                Ok(_) => r#"{"type":"affected","affected":0}"#.to_string(),
                Err(e) => error_json(&e.to_string()),
            },
            None => error_json("not in a transaction"),
        }
    }

    fn rollback_transaction(&self) -> String {
        let mut tx_ref = self.tx.borrow_mut();
        match tx_ref.take() {
            Some(mut tx) => match tx.rollback() {
                Ok(_) => r#"{"type":"affected","affected":0}"#.to_string(),
                Err(e) => error_json(&e.to_string()),
            },
            None => error_json("not in a transaction"),
        }
    }

    fn execute_read_query(&self, query: &str) -> String {
        let mut tx_ref = self.tx.borrow_mut();
        let rows_result = if let Some(ref mut tx) = *tx_ref {
            match tx.query(query, ()) {
                Ok(r) => r,
                Err(e) => return error_json(&e.to_string()),
            }
        } else {
            drop(tx_ref);
            match self.db.query(query, ()) {
                Ok(r) => r,
                Err(e) => return error_json(&e.to_string()),
            }
        };

        let columns: Vec<String> = rows_result.columns().to_vec();
        let mut result_rows: Vec<Vec<serde_json::Value>> = Vec::new();

        for row_result in rows_result {
            match row_result {
                Ok(row) => {
                    let mut json_row = Vec::with_capacity(columns.len());
                    for i in 0..columns.len() {
                        json_row.push(match row.get_value(i) {
                            Some(v) => value_to_json(v),
                            None => serde_json::Value::Null,
                        });
                    }
                    result_rows.push(json_row);
                }
                Err(e) => return error_json(&e.to_string()),
            }
        }

        let row_count = result_rows.len();

        serde_json::json!({
            "type": "rows",
            "columns": columns,
            "rows": result_rows,
            "count": row_count
        })
        .to_string()
    }

    fn execute_write_query(&self, query: &str) -> String {
        let mut tx_ref = self.tx.borrow_mut();
        let rows_affected = if let Some(ref mut tx) = *tx_ref {
            match tx.execute(query, ()) {
                Ok(n) => n,
                Err(e) => return error_json(&e.to_string()),
            }
        } else {
            drop(tx_ref);
            match self.db.execute(query, ()) {
                Ok(n) => n,
                Err(e) => return error_json(&e.to_string()),
            }
        };

        serde_json::json!({
            "type": "affected",
            "affected": rows_affected
        })
        .to_string()
    }
}

fn error_json(msg: &str) -> String {
    serde_json::json!({
        "type": "error",
        "message": msg
    })
    .to_string()
}

/// Convert a Value to serde_json::Value (mirrors CLI's value_to_json)
fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Null(_) => serde_json::Value::Null,
        Value::Boolean(b) => serde_json::json!(b),
        Value::Integer(i) => serde_json::json!(i),
        Value::Float(f) => {
            // serde_json::json!(f) panics on NaN/Infinity — handle gracefully
            if f.is_finite() {
                serde_json::json!(f)
            } else {
                serde_json::Value::Null
            }
        }
        Value::Text(s) => serde_json::json!(s.as_str()),
        Value::Timestamp(ts) => {
            serde_json::json!(ts.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        }
        Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
            serde_json::json!(std::str::from_utf8(&data[1..]).unwrap_or(""))
        }
        Value::Extension(data) if data.first() == Some(&(DataType::Vector as u8)) => {
            serde_json::json!(crate::core::value::format_vector_bytes(&data[1..]))
        }
        Value::Extension(_) => serde_json::Value::Null,
    }
}

/// Split SQL statements by semicolons, handling quotes and comments.
/// Mirrors the CLI's `split_sql_statements()` function.
fn split_sql_statements(input: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current_statement = String::new();

    let mut in_single_quotes = false;
    let mut in_double_quotes = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;

    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let char = chars[i];

        // Handle end of line comment
        if in_line_comment {
            if char == '\n' {
                in_line_comment = false;
                current_statement.push(char);
            }
            i += 1;
            continue;
        }

        // Handle start of line comment
        if !in_single_quotes
            && !in_double_quotes
            && !in_block_comment
            && char == '-'
            && i + 1 < chars.len()
            && chars[i + 1] == '-'
        {
            let after_second_dash = if i + 2 < chars.len() {
                chars[i + 2]
            } else {
                '\0'
            };
            if after_second_dash == '\0'
                || after_second_dash == ' '
                || after_second_dash == '\t'
                || after_second_dash == '\n'
                || after_second_dash == '\r'
            {
                in_line_comment = true;
                i += 2;
                continue;
            }
        }

        // Handle end of block comment
        if in_block_comment {
            if char == '*' && i + 1 < chars.len() && chars[i + 1] == '/' {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        // Handle start of block comment
        if !in_single_quotes
            && !in_double_quotes
            && char == '/'
            && i + 1 < chars.len()
            && chars[i + 1] == '*'
        {
            in_block_comment = true;
            i += 2;
            continue;
        }

        // Handle quotes
        if !in_block_comment && !in_line_comment {
            if char == '\'' && (i == 0 || chars[i - 1] != '\\') {
                in_single_quotes = !in_single_quotes;
            } else if char == '"' && (i == 0 || chars[i - 1] != '\\') {
                in_double_quotes = !in_double_quotes;
            }
        }

        // Semicolon outside of quotes and comments is a statement delimiter
        if char == ';'
            && !in_single_quotes
            && !in_double_quotes
            && !in_block_comment
            && !in_line_comment
        {
            statements.push(current_statement.clone());
            current_statement.clear();
        } else {
            current_statement.push(char);
        }

        i += 1;
    }

    // Add any remaining statement
    if !current_statement.is_empty() {
        statements.push(current_statement);
    }

    statements
}
