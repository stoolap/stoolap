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

//! FFI type definitions: opaque wrapper structs and `#[repr(C)]` value type.

use std::cell::OnceCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use crate::api::database::DatabaseInner;
use crate::api::transaction::Transaction;
use crate::api::{Database, Rows, Statement};
use crate::common::CompactArc;

/// FFI-safe tagged union for passing parameter values across the C boundary.
#[repr(C)]
pub struct StoolapValue {
    /// One of `STOOLAP_TYPE_*` constants.
    pub value_type: i32,
    pub _padding: i32,
    pub v: StoolapValueData,
}

/// Union payload for [`StoolapValue`].
#[repr(C)]
pub union StoolapValueData {
    pub integer: i64,
    pub float64: f64,
    pub boolean: i32,
    pub text: StoolapTextData,
    pub blob: StoolapBlobData,
    pub timestamp_nanos: i64,
}

/// Text pointer + length (not necessarily null-terminated on input).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StoolapTextData {
    pub ptr: *const c_char,
    pub len: i64,
}

/// Blob pointer + length.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StoolapBlobData {
    pub ptr: *const u8,
    pub len: i64,
}

/// FFI-safe named parameter: a key-value pair for `:name`-style bindings.
#[repr(C)]
pub struct StoolapNamedParam {
    /// Parameter name (without the `:` prefix). Not necessarily null-terminated.
    pub name: *const c_char,
    /// Length of `name` in bytes.
    pub name_len: i32,
    pub _padding: i32,
    /// Parameter value.
    pub value: StoolapValue,
}

/// Opaque handle wrapping a [`Database`] connection.
pub struct StoolapDB {
    pub(crate) db: Database,
    pub(crate) last_error: Option<CString>,
    /// Buffer behind `stoolap_query_params_all()`, reused across calls
    pub(crate) packed: Vec<u8>,
}

/// Opaque handle wrapping a [`Statement`].
pub struct StoolapStmt {
    pub(crate) stmt: Statement,
    pub(crate) last_error: Option<CString>,
    /// Pre-computed CString for `stoolap_stmt_sql()`.
    pub(crate) sql_cstr: CString,
    /// Cached column name CStrings (computed on first query, reused thereafter).
    pub(crate) cached_columns: Option<Arc<Vec<CString>>>,
    /// Buffer behind `stoolap_stmt_query_all()`, reused across calls
    pub(crate) packed: Vec<u8>,
    /// Keeps the originating `DatabaseInner` alive so the `Statement`'s `Weak`
    /// reference can be upgraded, and with it the engine.
    pub(crate) _db_keepalive: Arc<DatabaseInner>,
}

/// Opaque handle wrapping a [`Transaction`].
pub struct StoolapTx {
    pub(crate) tx: Option<Transaction>,
    pub(crate) last_error: Option<CString>,
    /// Keeps the originating `DatabaseInner` alive so the transaction's
    /// storage references remain valid.
    pub(crate) _db_keepalive: Arc<DatabaseInner>,
}

/// Opaque handle wrapping a [`Rows`] result set.
pub struct StoolapRows {
    pub(crate) rows: Option<Rows>,
    pub(crate) has_row: bool,
    pub(crate) last_error: Option<CString>,
    /// Column names of the result, kept after the rows are consumed
    pub(crate) columns: CompactArc<Vec<String>>,
    /// Column names as CStrings, built on the first `stoolap_rows_column_name()`
    /// unless a prepared statement handed over its cached ones
    pub(crate) column_names: OnceCell<Arc<Vec<CString>>>,
    /// Per column, the row it was read for and its text with a trailing NUL.
    /// Buffers are reused from row to row; an entry is current when its row
    /// matches `row_gen`
    pub(crate) text_cache: Vec<(u64, Vec<u8>)>,
    /// Counts the rows `stoolap_rows_next()` advanced to
    pub(crate) row_gen: u64,
    /// Number of rows affected (for DML results).
    pub(crate) rows_affected: i64,
    /// Buffer behind `stoolap_rows_fetch_chunk()`, reused across chunks
    pub(crate) packed: Vec<u8>,
}

impl StoolapDB {
    pub(crate) fn set_error(&mut self, msg: &str) {
        let sanitized = msg.replace('\0', "\\0");
        self.last_error = CString::new(sanitized).ok();
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        match &self.last_error {
            Some(cs) => cs.as_ptr(),
            None => super::error::empty_cstr(),
        }
    }
}

impl StoolapStmt {
    pub(crate) fn set_error(&mut self, msg: &str) {
        let sanitized = msg.replace('\0', "\\0");
        self.last_error = CString::new(sanitized).ok();
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        match &self.last_error {
            Some(cs) => cs.as_ptr(),
            None => super::error::empty_cstr(),
        }
    }
}

impl StoolapTx {
    pub(crate) fn set_error(&mut self, msg: &str) {
        let sanitized = msg.replace('\0', "\\0");
        self.last_error = CString::new(sanitized).ok();
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        match &self.last_error {
            Some(cs) => cs.as_ptr(),
            None => super::error::empty_cstr(),
        }
    }
}

impl StoolapRows {
    /// A handle on `rows`, with a prepared statement's cached column names
    /// when it has them
    pub(crate) fn boxed(rows: Rows, column_names: Option<Arc<Vec<CString>>>) -> *mut Self {
        let rows_affected = rows.rows_affected();
        let handle = Box::new(StoolapRows {
            columns: rows.columns_arc().clone(),
            column_names: column_names.map(OnceCell::from).unwrap_or_default(),
            rows: Some(rows),
            has_row: false,
            last_error: None,
            text_cache: Vec::new(),
            row_gen: 0,
            rows_affected,
            packed: Vec::new(),
        });
        Box::into_raw(handle)
    }

    /// The column names as CStrings, built on first use
    pub(crate) fn column_cstrs(&self) -> &Arc<Vec<CString>> {
        self.column_names.get_or_init(|| {
            Arc::new(
                self.columns
                    .iter()
                    .map(|name| CString::new(name.as_str()).unwrap_or_default())
                    .collect(),
            )
        })
    }

    pub(crate) fn set_error(&mut self, msg: &str) {
        let sanitized = msg.replace('\0', "\\0");
        self.last_error = CString::new(sanitized).ok();
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        match &self.last_error {
            Some(cs) => cs.as_ptr(),
            None => super::error::empty_cstr(),
        }
    }
}
