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

//! TiKV query result implementation

use rustc_hash::FxHashMap;

use crate::core::{Result, Row, Value};
use crate::storage::traits::{QueryResult, Scanner};

use super::scanner::TiKVScanner;

/// Query result backed by TiKV scanner
pub struct TiKVQueryResult {
    /// Column names
    columns: Vec<String>,
    /// Underlying scanner
    scanner: TiKVScanner,
    /// Number of rows affected (for DML)
    rows_affected: i64,
}

impl TiKVQueryResult {
    /// Create a new query result from a scanner and column names
    pub fn new(columns: Vec<String>, scanner: TiKVScanner) -> Self {
        Self {
            columns,
            scanner,
            rows_affected: 0,
        }
    }

    /// Create a DML result (INSERT/UPDATE/DELETE)
    pub fn dml(rows_affected: i64) -> Self {
        Self {
            columns: Vec::new(),
            scanner: TiKVScanner::empty(),
            rows_affected,
        }
    }

    /// Create an empty result
    pub fn empty(columns: Vec<String>) -> Self {
        Self {
            columns,
            scanner: TiKVScanner::empty(),
            rows_affected: 0,
        }
    }
}

impl QueryResult for TiKVQueryResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        self.scanner.next()
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        let row = self.scanner.row();
        for (i, d) in dest.iter_mut().enumerate() {
            if let Some(v) = row.get(i) {
                *d = v.clone();
            }
        }
        Ok(())
    }

    fn row(&self) -> &Row {
        self.scanner.row()
    }

    fn take_row(&mut self) -> Row {
        self.scanner.take_row()
    }

    fn close(&mut self) -> Result<()> {
        self.scanner.close()
    }

    fn rows_affected(&self) -> i64 {
        self.rows_affected
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(
        mut self: Box<Self>,
        aliases: FxHashMap<String, String>,
    ) -> Box<dyn QueryResult> {
        // Apply aliases to column names
        for col in &mut self.columns {
            for (alias, original) in &aliases {
                if col == original {
                    *col = alias.clone();
                    break;
                }
            }
        }
        self
    }
}
