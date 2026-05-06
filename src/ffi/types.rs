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
    /// Most-recent column-name `CString` set, keyed by the
    /// `CompactArc<Vec<String>>` pointer identity of the source `Rows`.
    /// Hits when two consecutive queries on this handle resolve to the
    /// same projection plan (very common in HFT-style hot loops); misses
    /// are no worse than the prior unconditional rebuild.
    pub(crate) col_cstr_cache: ColumnCStrCache,
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
    pub(crate) col_cstr_cache: ColumnCStrCache,
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
    pub(crate) col_cstr_cache: ColumnCStrCache,
}

/// Single cached column-name set: the source `CompactArc` we matched
/// against, kept alive so its address can't be recycled, paired with the
/// `CString`s we built from it.
type ColumnCacheEntry = (crate::common::CompactArc<Vec<String>>, Arc<Vec<CString>>);

/// Single-slot column-name cache. Two queries that share the same column
/// `CompactArc` (same projection plan) skip a `Vec<CString>` rebuild.
///
/// The cache holds a strong clone of the source `CompactArc` — comparing
/// raw pointers without keeping the allocation alive is unsound: when the
/// previous `Rows` handle is closed and its `CompactArc` drops, the
/// allocator can reuse that address for a brand-new projection, and a
/// pointer-equality check would falsely report a hit and return the
/// previous projection's `CString`s. Equality is therefore checked via
/// `CompactArc::ptr_eq` against an Arc we own.
#[derive(Default)]
pub(crate) struct ColumnCStrCache {
    held: Option<ColumnCacheEntry>,
}

impl ColumnCStrCache {
    /// Look up by `Rows::columns_arc()` identity. On miss, build fresh
    /// `CString`s, store them alongside an owning clone of the source
    /// `CompactArc`, and return the new cache entry.
    pub(crate) fn get_or_build(
        &mut self,
        arc: &crate::common::CompactArc<Vec<String>>,
    ) -> Arc<Vec<CString>> {
        if let Some((held_arc, cstrs)) = &self.held {
            if crate::common::CompactArc::ptr_eq(held_arc, arc) {
                return Arc::clone(cstrs);
            }
        }
        let cstrs: Arc<Vec<CString>> = Arc::new(
            arc.iter()
                .map(|name| CString::new(name.as_str()).unwrap_or_default())
                .collect(),
        );
        self.held = Some((arc.clone(), Arc::clone(&cstrs)));
        cstrs
    }
}

/// Build a [`StoolapRows`] handle, looking up cached column-name
/// `CString`s by `Rows::columns_arc()` identity. Used by every FFI query
/// entry point (DB / Tx / RO / Statement) so they share one cache hit
/// path.
pub(crate) fn build_rows_handle(
    rows: crate::api::Rows,
    cache: &mut ColumnCStrCache,
) -> Box<StoolapRows> {
    let column_names = cache.get_or_build(rows.columns_arc());
    let affected = rows.rows_affected();
    let col_count = column_names.len();
    Box::new(StoolapRows {
        rows: Some(rows),
        has_row: false,
        last_error: LastErrorState::default(),
        column_names,
        text_cache: vec![Vec::new(); col_count],
        text_cache_dirty: smallvec::SmallVec::new(),
        rows_affected: affected,
    })
}

/// Opaque handle wrapping a [`Rows`] result set.
pub struct StoolapRows {
    pub(crate) rows: Option<Rows>,
    pub(crate) has_row: bool,
    pub(crate) last_error: LastErrorState,
    /// Column names as CStrings (shared via Arc for prepared statement reuse).
    pub(crate) column_names: Arc<Vec<CString>>,
    /// Lazy text cache for the current row. One slot per column; each slot's
    /// `Vec<u8>` is reused across rows (cleared, not dropped) so a long scan
    /// pays at most one allocation per (row, column) pair amortized.
    /// Empty `Vec` (`len == 0`) means "not populated for the current row";
    /// populated buffers always end in a trailing NUL byte (so length >= 1).
    /// Numeric-only scans pay zero overhead because no slot is ever populated.
    pub(crate) text_cache: Vec<Vec<u8>>,
    /// Indices populated for the current row. On row advance we only clear
    /// these (preserving capacity), avoiding an O(N) sweep over wide tables.
    pub(crate) text_cache_dirty: smallvec::SmallVec<[u32; 4]>,
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
