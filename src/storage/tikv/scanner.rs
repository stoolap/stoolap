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

//! TiKV scanner implementation

use crate::core::{Result, Row, Value};
use crate::storage::traits::Scanner;

use super::encoding;

/// Scanner over TiKV key-value pairs (loaded into memory)
pub struct TiKVScanner {
    /// Decoded rows with their row IDs
    rows: Vec<(i64, Row)>,
    /// Column indices to project (empty = all columns)
    column_indices: Vec<usize>,
    /// Current position (None = before first)
    position: Option<usize>,
    /// Current projected row (cached for row() to return reference)
    current_row: Row,
}

impl TiKVScanner {
    /// Create a new scanner from raw TiKV KvPairs
    pub fn new(pairs: Vec<tikv_client::KvPair>, column_indices: Vec<usize>) -> Self {
        let mut rows = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let key = pair.key().to_owned();
            let key_bytes: Vec<u8> = key.into();
            let row_id = encoding::extract_row_id_from_data_key(&key_bytes);
            if let Ok(values) = encoding::deserialize_row(pair.value()) {
                let row = encoding::values_to_row(values);
                rows.push((row_id, row));
            }
        }

        Self {
            rows,
            column_indices,
            position: None,
            current_row: Row::new(),
        }
    }

    /// Create an empty scanner
    pub fn empty() -> Self {
        Self {
            rows: Vec::new(),
            column_indices: Vec::new(),
            position: None,
            current_row: Row::new(),
        }
    }

    /// Create from pre-decoded rows
    pub fn from_rows(rows: Vec<(i64, Row)>, column_indices: Vec<usize>) -> Self {
        Self {
            rows,
            column_indices,
            position: None,
            current_row: Row::new(),
        }
    }

    /// Project the current row according to column_indices
    fn project_current(&mut self) {
        if let Some(pos) = self.position {
            if pos < self.rows.len() {
                let (_, ref row) = self.rows[pos];
                if self.column_indices.is_empty() {
                    self.current_row = row.clone();
                } else {
                    let values: Vec<Value> = self
                        .column_indices
                        .iter()
                        .map(|&i| {
                            row.get(i).cloned().unwrap_or_else(Value::null_unknown)
                        })
                        .collect();
                    self.current_row = Row::from_values(values);
                }
            }
        }
    }
}

impl Scanner for TiKVScanner {
    fn next(&mut self) -> bool {
        let next_pos = match self.position {
            None => 0,
            Some(i) => i + 1,
        };

        if next_pos < self.rows.len() {
            self.position = Some(next_pos);
            self.project_current();
            true
        } else {
            false
        }
    }

    fn row(&self) -> &Row {
        &self.current_row
    }

    fn err(&self) -> Option<&crate::core::Error> {
        None
    }

    fn close(&mut self) -> Result<()> {
        self.rows.clear();
        Ok(())
    }

    fn take_row(&mut self) -> Row {
        std::mem::take(&mut self.current_row)
    }

    fn current_row_id(&self) -> i64 {
        match self.position {
            Some(pos) if pos < self.rows.len() => self.rows[pos].0,
            _ => -1,
        }
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.rows.len())
    }

    fn take_row_with_id(&mut self) -> (i64, Row) {
        let id = self.current_row_id();
        (id, self.take_row())
    }
}
