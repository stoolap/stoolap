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

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use crate::api::database::DatabaseInner;
use crate::api::transaction::Transaction;
use crate::api::{Database, ReadOnlyDatabase, Rows, Statement};
use crate::core::Error;

use super::error::LastErrorState;

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

/// FFI-safe structured error detail. Pointers are valid until the next
/// FFI call on the originating handle. NULL fields indicate "not
/// applicable for this error code". `message` is never NULL — empty
/// string when no error.
#[repr(C)]
pub struct StoolapErrorDetails {
    /// One of `STOOLAP_ERR_*` constants.
    pub code: i32,
    pub _padding: i32,
    /// Always non-NULL. Empty string on success.
    pub message: *const c_char,
    /// Table name for table-scoped errors. NULL otherwise.
    pub table: *const c_char,
    /// Column name for column-scoped errors. NULL otherwise.
    pub column: *const c_char,
    /// Index name (UNIQUE) or referenced table (FK). NULL otherwise.
    pub constraint: *const c_char,
    /// Free-form detail: conflicting value (UNIQUE), CHECK expression,
    /// FK detail message. NULL otherwise.
    pub detail: *const c_char,
}

/// Opaque handle wrapping a [`Database`] connection.
pub struct StoolapDB {
    pub(crate) db: Database,
    pub(crate) last_error: LastErrorState,
    /// Holds a reference to the original (engine-owning) DatabaseInner.
    /// Prevents premature engine shutdown when the original handle is closed
    /// before its clones. `None` for the original handle, `Some` for clones.
    pub(crate) _engine_keepalive: Option<Arc<DatabaseInner>>,
}

/// Opaque handle wrapping a [`ReadOnlyDatabase`] view.
///
/// Mirrors the Rust type split: this handle exposes only read functions
/// (`stoolap_ro_query*`, `stoolap_ro_table_*`, `stoolap_ro_refresh`).
/// There are no `_exec` / `_begin` / savepoint entry points, so attempting
/// to write through a read-only handle is a compile-time error on the C
/// side too — not a runtime `STOOLAP_ERR_READ_ONLY`.
pub struct StoolapRoDB {
    pub(crate) ro: ReadOnlyDatabase,
    pub(crate) last_error: LastErrorState,
    /// One-time cache of the DSN as a CString, populated lazily on
    /// first `stoolap_ro_dsn()` call so the returned pointer is stable
    /// for the lifetime of the handle without leaking on every call.
    pub(crate) dsn_cstr: std::sync::OnceLock<CString>,
}

/// Opaque handle wrapping a [`Statement`].
pub struct StoolapStmt {
    pub(crate) stmt: Statement,
    pub(crate) last_error: LastErrorState,
    /// Pre-computed CString for `stoolap_stmt_sql()`.
    pub(crate) sql_cstr: CString,
    /// Cached column name CStrings (computed on first query, reused thereafter).
    pub(crate) cached_columns: Option<Arc<Vec<CString>>>,
    /// Keeps the originating `DatabaseInner` alive so the `Statement`'s `Weak`
    /// reference can be upgraded. For original handles this is the engine-owning
    /// inner; for clone handles it is the clone's own (non-owning) inner.
    pub(crate) _db_keepalive: Arc<DatabaseInner>,
    /// For statements prepared from a clone handle, holds the engine-owning
    /// `DatabaseInner` to prevent `close_engine()` after the original handle
    /// is closed. `None` when prepared from an original (non-clone) handle.
    pub(crate) _engine_keepalive: Option<Arc<DatabaseInner>>,
}

/// Opaque handle wrapping a [`Transaction`].
pub struct StoolapTx {
    pub(crate) tx: Option<Transaction>,
    pub(crate) last_error: LastErrorState,
    /// Keeps the originating `DatabaseInner` alive so the transaction's
    /// storage references remain valid.
    pub(crate) _db_keepalive: Arc<DatabaseInner>,
    /// For transactions begun from a clone handle, holds the engine-owning
    /// `DatabaseInner` to prevent `close_engine()`. `None` for original handles.
    pub(crate) _engine_keepalive: Option<Arc<DatabaseInner>>,
}

/// Opaque handle wrapping a [`Rows`] result set.
pub struct StoolapRows {
    pub(crate) rows: Option<Rows>,
    pub(crate) has_row: bool,
    pub(crate) last_error: LastErrorState,
    /// Column names as CStrings (shared via Arc for prepared statement reuse).
    pub(crate) column_names: Arc<Vec<CString>>,
    /// Lazy text cache for the current row. Starts empty; grown on demand by
    /// `stoolap_rows_column_text()`. Cleared only when at least one entry was
    /// populated (`text_cache_dirty`), so numeric-only scans pay zero overhead.
    /// Each populated entry is `[text_bytes..., 0]` — the original text (may
    /// contain interior NULs) with a trailing NUL terminator for C compat.
    pub(crate) text_cache: Vec<Option<Vec<u8>>>,
    /// True when at least one entry in `text_cache` was populated for the
    /// current row. Avoids clearing the entire Vec when no text was accessed.
    pub(crate) text_cache_dirty: bool,
    /// Number of rows affected (for DML results).
    pub(crate) rows_affected: i64,
}

impl StoolapDB {
    pub(crate) fn set_error(&mut self, msg: &str) {
        self.last_error.set_message(msg);
    }

    pub(crate) fn set_error_from(&mut self, err: &Error) {
        self.last_error.set_from_error(err);
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        self.last_error.message_ptr()
    }
}

impl StoolapRoDB {
    pub(crate) fn set_error(&mut self, msg: &str) {
        self.last_error.set_message(msg);
    }

    pub(crate) fn set_error_from(&mut self, err: &Error) {
        self.last_error.set_from_error(err);
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        self.last_error.message_ptr()
    }
}

impl StoolapStmt {
    pub(crate) fn set_error(&mut self, msg: &str) {
        self.last_error.set_message(msg);
    }

    pub(crate) fn set_error_from(&mut self, err: &Error) {
        self.last_error.set_from_error(err);
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        self.last_error.message_ptr()
    }
}

impl StoolapTx {
    pub(crate) fn set_error(&mut self, msg: &str) {
        self.last_error.set_message(msg);
    }

    pub(crate) fn set_error_from(&mut self, err: &Error) {
        self.last_error.set_from_error(err);
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        self.last_error.message_ptr()
    }
}

impl StoolapRows {
    pub(crate) fn set_error(&mut self, msg: &str) {
        self.last_error.set_message(msg);
    }

    pub(crate) fn set_error_from(&mut self, err: &Error) {
        self.last_error.set_from_error(err);
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        self.last_error.message_ptr()
    }
}
