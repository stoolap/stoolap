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

//! Per-handle error storage with typed code + structured details.
//!
//! Each FFI handle holds one `LastErrorState`. The success path is
//! zero-allocation: success leaves all fields `None`. On the error path
//! `set_from_error` builds the message CString once and, for constraint
//! / not-found / type-mismatch cases, also captures `{table, column,
//! constraint, detail}` as small CStrings — only the fields that are
//! present, so a generic error costs one CString (same as today's
//! string-only error path).
//!
//! The thread-local fallback (`THREAD_ERROR`) carries the same shape,
//! used for `stoolap_open` failures and for `commit`/`rollback` whose
//! handle is consumed before we can store the error on it.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

use crate::core::Error;

use super::types::StoolapErrorDetails;

/// Numeric error codes mapped from `crate::core::Error`. Stable wire
/// values: appended-only, never renumbered.
pub mod codes {
    pub const STOOLAP_ERR_OK: i32 = 0;
    pub const STOOLAP_ERR_GENERIC: i32 = 1;
    pub const STOOLAP_ERR_NOT_NULL: i32 = 2;
    pub const STOOLAP_ERR_UNIQUE: i32 = 3;
    pub const STOOLAP_ERR_PRIMARY_KEY: i32 = 4;
    pub const STOOLAP_ERR_FOREIGN_KEY: i32 = 5;
    pub const STOOLAP_ERR_CHECK: i32 = 6;
    pub const STOOLAP_ERR_TABLE_NOT_FOUND: i32 = 7;
    pub const STOOLAP_ERR_TABLE_EXISTS: i32 = 8;
    pub const STOOLAP_ERR_COLUMN_NOT_FOUND: i32 = 9;
    pub const STOOLAP_ERR_INDEX_NOT_FOUND: i32 = 10;
    pub const STOOLAP_ERR_INDEX_EXISTS: i32 = 11;
    pub const STOOLAP_ERR_TYPE_MISMATCH: i32 = 12;
    pub const STOOLAP_ERR_INVALID_ARGUMENT: i32 = 13;
    pub const STOOLAP_ERR_PARSE: i32 = 14;
    pub const STOOLAP_ERR_TX_ABORTED: i32 = 15;
    pub const STOOLAP_ERR_TX_CLOSED: i32 = 16;
    pub const STOOLAP_ERR_READ_ONLY: i32 = 17;
    pub const STOOLAP_ERR_DB_LOCKED: i32 = 18;
    pub const STOOLAP_ERR_IO: i32 = 19;
    pub const STOOLAP_ERR_NOT_SUPPORTED: i32 = 20;
    pub const STOOLAP_ERR_INTERNAL: i32 = 21;
    pub const STOOLAP_ERR_QUERY_CANCELLED: i32 = 22;
    pub const STOOLAP_ERR_DIVISION_BY_ZERO: i32 = 23;
    pub const STOOLAP_ERR_VALUE_TOO_LONG: i32 = 24;
    pub const STOOLAP_ERR_VIEW_NOT_FOUND: i32 = 25;
    pub const STOOLAP_ERR_VIEW_EXISTS: i32 = 26;
    /// SWMR: caller must reopen the read-only handle. Covers
    /// `SchemaChanged`, `SwmrWriterReincarnated`, `SwmrSnapshotExpired`,
    /// `SwmrPendingDdl`, `SwmrPartialReload`, `SwmrOverlayApplyFailed`.
    pub const STOOLAP_ERR_REOPEN_REQUIRED: i32 = 27;
}

use codes::*;

// Re-export commonly-referenced codes at the module root so call sites
// can use `super::error::STOOLAP_ERR_OK` directly.
pub(crate) use codes::STOOLAP_ERR_OK;

/// Per-handle structured error state.
#[derive(Default)]
pub(crate) struct LastErrorState {
    pub(crate) code: i32,
    pub(crate) message: Option<CString>,
    pub(crate) table: Option<CString>,
    pub(crate) column: Option<CString>,
    pub(crate) constraint: Option<CString>,
    pub(crate) detail: Option<CString>,
}

impl LastErrorState {
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.code = STOOLAP_ERR_OK;
        self.message = None;
        self.table = None;
        self.column = None;
        self.constraint = None;
        self.detail = None;
    }

    /// Set message-only error with a generic code. Used for panic catches
    /// and internal precondition checks where no `Error` value exists.
    pub(crate) fn set_message(&mut self, msg: &str) {
        self.code = STOOLAP_ERR_GENERIC;
        self.message = sanitize_cstring(msg);
        self.table = None;
        self.column = None;
        self.constraint = None;
        self.detail = None;
    }

    /// Populate from a typed `Error`, capturing both the formatted message
    /// and any structured fields (table / column / constraint / detail).
    /// One Display allocation for the message; up to three additional
    /// CStrings for constraint sub-fields, only when applicable.
    pub(crate) fn set_from_error(&mut self, err: &Error) {
        // Format message once, into a sanitized CString.
        // Allocation footprint == today's `set_error(&e.to_string())`.
        self.message = sanitize_cstring(&err.to_string());
        self.table = None;
        self.column = None;
        self.constraint = None;
        self.detail = None;
        match err {
            Error::TableNotFound(t) | Error::TableOrViewNotFound(t) => {
                self.code = STOOLAP_ERR_TABLE_NOT_FOUND;
                self.table = sanitize_cstring(t);
            }
            Error::TableAlreadyExists(t) => {
                self.code = STOOLAP_ERR_TABLE_EXISTS;
                self.table = sanitize_cstring(t);
            }
            Error::ColumnNotFound(c) => {
                self.code = STOOLAP_ERR_COLUMN_NOT_FOUND;
                self.column = sanitize_cstring(c);
            }
            Error::IndexNotFound(i) => {
                self.code = STOOLAP_ERR_INDEX_NOT_FOUND;
                self.constraint = sanitize_cstring(i);
            }
            Error::IndexAlreadyExists(i) => {
                self.code = STOOLAP_ERR_INDEX_EXISTS;
                self.constraint = sanitize_cstring(i);
            }
            Error::ViewNotFound(v) => {
                self.code = STOOLAP_ERR_VIEW_NOT_FOUND;
                self.table = sanitize_cstring(v);
            }
            Error::ViewAlreadyExists(v) => {
                self.code = STOOLAP_ERR_VIEW_EXISTS;
                self.table = sanitize_cstring(v);
            }
            Error::NotNullConstraint { column } => {
                self.code = STOOLAP_ERR_NOT_NULL;
                self.column = sanitize_cstring(column);
            }
            Error::PrimaryKeyConstraint { row_id: _ } => {
                self.code = STOOLAP_ERR_PRIMARY_KEY;
            }
            Error::UniqueConstraint {
                index,
                column,
                value,
                row_id: _,
            } => {
                self.code = STOOLAP_ERR_UNIQUE;
                self.column = sanitize_cstring(column);
                self.constraint = sanitize_cstring(index);
                self.detail = sanitize_cstring(value);
            }
            Error::CheckConstraintViolation { column, expression } => {
                self.code = STOOLAP_ERR_CHECK;
                self.column = sanitize_cstring(column);
                self.detail = sanitize_cstring(expression);
            }
            Error::ForeignKeyViolation {
                table,
                column,
                ref_table,
                ref_column: _,
                detail,
            } => {
                self.code = STOOLAP_ERR_FOREIGN_KEY;
                self.table = sanitize_cstring(table);
                self.column = sanitize_cstring(column);
                self.constraint = sanitize_cstring(ref_table);
                self.detail = sanitize_cstring(detail);
            }
            Error::ValueTooLong { column, .. } => {
                self.code = STOOLAP_ERR_VALUE_TOO_LONG;
                self.column = sanitize_cstring(column);
            }
            Error::TableColumnsNotMatch { .. }
            | Error::InvalidColumnType
            | Error::VectorDimensionMismatch { .. }
            | Error::DuplicateColumn
            | Error::TypeConversion { .. }
            | Error::Type(_)
            | Error::IncomparableTypes
            | Error::NullComparison => {
                self.code = STOOLAP_ERR_TYPE_MISMATCH;
            }
            Error::InvalidArgument(_) | Error::InvalidValue => {
                self.code = STOOLAP_ERR_INVALID_ARGUMENT;
            }
            Error::Parse(_) => {
                self.code = STOOLAP_ERR_PARSE;
            }
            Error::TransactionAborted | Error::TransactionCommitted => {
                self.code = STOOLAP_ERR_TX_ABORTED;
            }
            Error::TransactionNotStarted
            | Error::TransactionAlreadyStarted
            | Error::TransactionEnded
            | Error::TransactionClosed => {
                self.code = STOOLAP_ERR_TX_CLOSED;
            }
            Error::ReadOnlyViolation(_) => {
                self.code = STOOLAP_ERR_READ_ONLY;
            }
            Error::DatabaseLocked => {
                self.code = STOOLAP_ERR_DB_LOCKED;
            }
            Error::Io { .. } => {
                self.code = STOOLAP_ERR_IO;
            }
            Error::NotSupported(_) | Error::CannotDropPrimaryKey => {
                self.code = STOOLAP_ERR_NOT_SUPPORTED;
            }
            Error::QueryCancelled => {
                self.code = STOOLAP_ERR_QUERY_CANCELLED;
            }
            Error::DivisionByZero => {
                self.code = STOOLAP_ERR_DIVISION_BY_ZERO;
            }
            Error::SchemaChanged(_)
            | Error::SwmrWriterReincarnated { .. }
            | Error::SwmrSnapshotExpired { .. }
            | Error::SwmrPendingDdl(_)
            | Error::SwmrPartialReload(_)
            | Error::SwmrOverlayApplyFailed(_) => {
                self.code = STOOLAP_ERR_REOPEN_REQUIRED;
            }
            Error::Internal { .. }
            | Error::EngineNotOpen
            | Error::EngineAlreadyOpen
            | Error::TableClosed
            | Error::TableHasActiveTransactions
            | Error::IndexClosed
            | Error::IndexColumnNotFound
            | Error::SegmentNotFound
            | Error::ExpressionEvaluation
            | Error::ExpressionEvaluationWithMessage { .. }
            | Error::WalNotRunning
            | Error::WalFileClosed
            | Error::WalNotInitialized
            | Error::LockAcquisitionFailed(_)
            | Error::NoRowsReturned
            | Error::NoStatementsToExecute
            | Error::ColumnIndexOutOfBounds { .. } => {
                self.code = STOOLAP_ERR_INTERNAL;
            }
        }
    }

    /// Pointer to the message C string (NUL-terminated, or empty cstr).
    #[inline]
    pub(crate) fn message_ptr(&self) -> *const c_char {
        match &self.message {
            Some(cs) => cs.as_ptr(),
            None => empty_cstr(),
        }
    }

    /// Fill the caller's `StoolapErrorDetails` struct from this state.
    /// Pointers are valid until the next set/clear on this handle.
    pub(crate) fn fill_details(&self, out: &mut StoolapErrorDetails) {
        out.code = self.code;
        out._padding = 0;
        out.message = self.message_ptr();
        out.table = ptr_or_null(&self.table);
        out.column = ptr_or_null(&self.column);
        out.constraint = ptr_or_null(&self.constraint);
        out.detail = ptr_or_null(&self.detail);
    }
}

#[inline]
fn ptr_or_null(field: &Option<CString>) -> *const c_char {
    match field {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Build a CString from a &str, replacing any interior NULs with `\0` text.
/// Returns `None` only on impossibly large allocations (CString itself
/// only fails on interior NUL, which we strip).
fn sanitize_cstring(s: &str) -> Option<CString> {
    if s.as_bytes().contains(&0) {
        let owned = s.replace('\0', "\\0");
        CString::new(owned).ok()
    } else {
        // Borrow-only: CString::new copies once into a Vec<u8> + appends NUL.
        CString::new(s).ok()
    }
}

// Thread-local fallback for `stoolap_open` failures (before any handle
// exists) and for `commit`/`rollback` whose `StoolapTx*` is consumed
// before the error can be returned.
thread_local! {
    static THREAD_ERROR: RefCell<LastErrorState> = const {
        RefCell::new(LastErrorState {
            code: STOOLAP_ERR_OK,
            message: None,
            table: None,
            column: None,
            constraint: None,
            detail: None,
        })
    };
}

pub(crate) fn set_global_error(msg: &str) {
    THREAD_ERROR.with(|cell| cell.borrow_mut().set_message(msg));
}

pub(crate) fn set_global_error_from(err: &Error) {
    THREAD_ERROR.with(|cell| cell.borrow_mut().set_from_error(err));
}

pub(crate) fn global_error_ptr() -> *const c_char {
    THREAD_ERROR.with(|cell| {
        // SAFETY: pointer is into a thread_local CString that lives until
        // the next set on this thread. Caller treats it as read-only and
        // does not call into FFI on the same thread before consuming it.
        let borrow = cell.borrow();
        match &borrow.message {
            Some(cs) => cs.as_ptr(),
            None => empty_cstr(),
        }
    })
}

pub(crate) fn global_error_code() -> i32 {
    THREAD_ERROR.with(|cell| cell.borrow().code)
}

pub(crate) fn fill_global_error_details(out: &mut StoolapErrorDetails) {
    THREAD_ERROR.with(|cell| cell.borrow().fill_details(out));
}

/// Returns a pointer to a static empty C string.
pub(crate) fn empty_cstr() -> *const c_char {
    // SAFETY: b"\0" is a valid null-terminated C string.
    static EMPTY: &[u8] = b"\0";
    EMPTY.as_ptr() as *const c_char
}
