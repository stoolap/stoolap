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

//! Prepared statement FFI functions.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic;
use std::sync::Arc;

use crate::api::transaction::Transaction;
use crate::api::{Database, ParamVec, Rows};
use crate::core::{Error, Result};

use super::rows::pack_all;
use super::types::{
    StoolapDB, StoolapNamedParam, StoolapRows, StoolapStmt, StoolapTx, StoolapValue,
};
use super::value;
use super::{STOOLAP_ERROR, STOOLAP_OK};

/// Prepare a SQL statement for repeated execution.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `out_stmt` must be a valid pointer to a `*mut StoolapStmt`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_prepare(
    db: *mut StoolapDB,
    sql: *const c_char,
    out_stmt: *mut *mut StoolapStmt,
) -> i32 {
    if out_stmt.is_null() {
        return STOOLAP_ERROR;
    }
    *out_stmt = std::ptr::null_mut();

    let handle = match db.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    if sql.is_null() {
        handle.set_error("SQL string is NULL");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let sql_str = match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.set_error(&format!("invalid UTF-8 in SQL: {}", e));
                return STOOLAP_ERROR;
            }
        };

        match handle.db.prepare(sql_str) {
            Ok(stmt) => {
                let sql_cstr = CString::new(sql_str).unwrap_or_default();
                let db_keepalive = handle.db.keepalive();
                let stmt_handle = Box::new(StoolapStmt {
                    stmt,
                    last_error: None,
                    sql_cstr,
                    cached_columns: None,
                    packed: Vec::new(),
                    _db_keepalive: db_keepalive,
                });
                *out_stmt = Box::into_raw(stmt_handle);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_prepare");
        STOOLAP_ERROR
    })
}

/// Execute a prepared statement with parameters.
///
/// # Safety
///
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_exec(
    stmt: *mut StoolapStmt,
    params: *const StoolapValue,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
    let handle = match stmt.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        match value::params_to_vec(params, params_len).and_then(|p| handle.stmt.execute(p)) {
            Ok(affected) => {
                if !rows_affected.is_null() {
                    *rows_affected = affected;
                }
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_stmt_exec");
        STOOLAP_ERROR
    })
}

/// Execute a prepared statement as a batch: opens a transaction, executes
/// the statement once per parameter row, then commits. One FFI call replaces
/// `2 + N` calls (begin + N × exec + commit).
///
/// `params` is a flat array of `row_count * params_per_row` StoolapValue
/// structs laid out row-major: row0_p0, row0_p1, ..., rowN_pK.
///
/// On error the transaction is rolled back and the error is stored on `db`.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `stmt` must be a valid `StoolapStmt` pointer prepared from `db`.
/// - `params` must point to `row_count * params_per_row` valid `StoolapValue`
///   structs (or be NULL when `row_count == 0`).
/// - `total_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_exec_batch(
    db: *mut StoolapDB,
    stmt: *const StoolapStmt,
    params: *const StoolapValue,
    params_per_row: i32,
    row_count: i32,
    total_affected: *mut i64,
) -> i32 {
    let db_handle = match db.as_mut() {
        Some(h) => h,
        None => {
            super::error::set_global_error("db handle is NULL");
            return STOOLAP_ERROR;
        }
    };
    db_handle.last_error = None;

    let stmt_handle = match stmt.as_ref() {
        Some(h) => h,
        None => {
            db_handle.set_error("statement handle is NULL");
            return STOOLAP_ERROR;
        }
    };

    if row_count <= 0 {
        if !total_affected.is_null() {
            *total_affected = 0;
        }
        return STOOLAP_OK;
    }
    if params_per_row < 0 || (params_per_row > 0 && params.is_null()) {
        db_handle.set_error("params must hold row_count * params_per_row values");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut tx = match db_handle.db.begin() {
            Ok(t) => t,
            Err(e) => {
                db_handle.set_error(&e.to_string());
                return STOOLAP_ERROR;
            }
        };

        let ppr = params_per_row as usize;
        let mut total: i64 = 0;

        for r in 0..(row_count as usize) {
            let param_vec = if ppr == 0 {
                Ok(ParamVec::new())
            } else {
                value::params_to_vec(params.add(r * ppr), params_per_row)
            };

            match param_vec.and_then(|p| stmt_handle.stmt.execute_in(&mut tx, p)) {
                Ok(affected) => total += affected,
                Err(e) => {
                    let _ = tx.rollback();
                    db_handle.set_error(&e.to_string());
                    return STOOLAP_ERROR;
                }
            }
        }

        match tx.commit() {
            Ok(_) => {
                if !total_affected.is_null() {
                    *total_affected = total;
                }
                STOOLAP_OK
            }
            Err(e) => {
                db_handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        db_handle.set_error("panic during stoolap_stmt_exec_batch");
        STOOLAP_ERROR
    })
}

/// Query using a prepared statement with parameters.
///
/// # Safety
///
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_query(
    stmt: *mut StoolapStmt,
    params: *const StoolapValue,
    params_len: i32,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    if out_rows.is_null() {
        return STOOLAP_ERROR;
    }
    *out_rows = std::ptr::null_mut();

    let handle = match stmt.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        match value::params_to_vec(params, params_len).and_then(|p| handle.stmt.query(p)) {
            Ok(rows) => {
                *out_rows = handle.rows_handle(rows);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_stmt_query");
        STOOLAP_ERROR
    })
}

/// Query using a prepared statement with named parameters (`:name`).
///
/// # Safety
///
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_query_named(
    stmt: *mut StoolapStmt,
    params: *const StoolapNamedParam,
    params_len: i32,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    if out_rows.is_null() {
        return STOOLAP_ERROR;
    }
    *out_rows = std::ptr::null_mut();

    let handle = match stmt.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut value_of = value::ffi_value_of(params, params_len);
        match handle.stmt.run_named(&mut value_of) {
            Ok(result) => {
                *out_rows = handle.rows_handle(Rows::new(result));
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_stmt_query_named");
        STOOLAP_ERROR
    })
}

/// Execute a prepared statement with named parameters (`:name`).
///
/// # Safety
///
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_exec_named(
    stmt: *mut StoolapStmt,
    params: *const StoolapNamedParam,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
    let handle = match stmt.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut value_of = value::ffi_value_of(params, params_len);
        match handle.stmt.run_named(&mut value_of) {
            Ok(result) => {
                if !rows_affected.is_null() {
                    *rows_affected = result.rows_affected();
                }
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_stmt_exec_named");
        STOOLAP_ERROR
    })
}

impl StoolapStmt {
    /// A rows handle on `rows`, with the column names this statement keeps
    /// until a DDL changes them
    fn rows_handle(&mut self, rows: Rows) -> *mut StoolapRows {
        let actual_columns = rows.columns();
        let column_names = match &self.cached_columns {
            Some(cached)
                if cached.len() == actual_columns.len()
                    && cached
                        .iter()
                        .zip(actual_columns.iter())
                        .all(|(c, a)| c.as_bytes() == a.as_str().as_bytes()) =>
            {
                Arc::clone(cached)
            }
            _ => {
                let names: Vec<CString> = actual_columns
                    .iter()
                    .map(|name| CString::new(name.as_str()).unwrap_or_default())
                    .collect();
                let arc = Arc::new(names);
                self.cached_columns = Some(Arc::clone(&arc));
                arc
            }
        };
        StoolapRows::boxed(rows, Some(column_names))
    }

    /// Packs `rows` into this statement's buffer for a `*_query_all` call
    unsafe fn pack_into(
        &mut self,
        rows: Result<Rows>,
        out_buf: *mut *const u8,
        out_len: *mut i64,
    ) -> i32 {
        match rows.and_then(|mut rows| pack_all(&mut rows, &mut self.packed)) {
            Ok(_) => {
                *out_buf = self.packed.as_ptr();
                *out_len = self.packed.len() as i64;
                STOOLAP_OK
            }
            Err(e) => {
                self.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }
}

/// The open transaction behind `tx`: None for NULL, an error once it ended
pub(super) unsafe fn open_transaction<'a>(
    tx: *mut StoolapTx,
) -> Result<Option<&'a mut Transaction>> {
    match tx.as_mut() {
        None => Ok(None),
        Some(tx_handle) => match &mut tx_handle.tx {
            Some(inner_tx) => Ok(Some(inner_tx)),
            None => Err(Error::TransactionEnded),
        },
    }
}

/// `open_transaction`, refusing a `tx` begun on another engine than `db`
pub(super) unsafe fn open_transaction_on<'a>(
    db: &Database,
    tx: *mut StoolapTx,
) -> Result<Option<&'a mut Transaction>> {
    let inner = open_transaction(tx)?;
    if inner.as_ref().is_some_and(|inner_tx| {
        inner_tx
            .session()
            .is_some_and(|session| !db.shares_engine_with(session))
    }) {
        return Err(Error::invalid_argument(
            "transaction belongs to another database",
        ));
    }
    Ok(inner)
}

/// Query a prepared statement and pack every row into one buffer, in the
/// format of `stoolap_rows_fetch_all()`, without a rows handle.
///
/// Runs inside `tx` when it is not NULL. The buffer belongs to the
/// statement and stays valid until the next call on it or
/// `stoolap_stmt_finalize()`. Must NOT be freed. Errors are read with
/// `stoolap_stmt_errmsg()`.
///
/// # Safety
///
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `tx` must be a valid `StoolapTx` pointer or NULL.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `out_buf` and `out_len` must not be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_query_all(
    stmt: *mut StoolapStmt,
    tx: *mut StoolapTx,
    params: *const StoolapValue,
    params_len: i32,
    out_buf: *mut *const u8,
    out_len: *mut i64,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() {
        return STOOLAP_ERROR;
    }
    *out_buf = std::ptr::null();
    *out_len = 0;

    let handle = match stmt.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let rows =
            value::params_to_vec(params, params_len).and_then(|p| match open_transaction(tx)? {
                None => handle.stmt.query(p),
                Some(inner_tx) => handle.stmt.query_in(inner_tx, p),
            });
        handle.pack_into(rows, out_buf, out_len)
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_stmt_query_all");
        STOOLAP_ERROR
    })
}

/// `stoolap_stmt_query_all()` with named parameters (`:name`).
///
/// # Safety
///
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `tx` must be a valid `StoolapTx` pointer or NULL.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `out_buf` and `out_len` must not be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_query_named_all(
    stmt: *mut StoolapStmt,
    tx: *mut StoolapTx,
    params: *const StoolapNamedParam,
    params_len: i32,
    out_buf: *mut *const u8,
    out_len: *mut i64,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() {
        return STOOLAP_ERROR;
    }
    *out_buf = std::ptr::null();
    *out_len = 0;

    let handle = match stmt.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut value_of = value::ffi_value_of(params, params_len);
        let rows = open_transaction(tx).and_then(|inner_tx| {
            Ok(Rows::new(match inner_tx {
                None => handle.stmt.run_named(&mut value_of)?,
                Some(inner_tx) => {
                    inner_tx.execute_prepared_statement_named(&handle.stmt, &mut value_of)?
                }
            }))
        });
        handle.pack_into(rows, out_buf, out_len)
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_stmt_query_named_all");
        STOOLAP_ERROR
    })
}

/// Get the SQL text of a prepared statement.
///
/// Returns a pointer valid for the lifetime of the statement. Must NOT be freed.
///
/// # Safety
///
/// `stmt` must be a valid `StoolapStmt` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_sql(stmt: *const StoolapStmt) -> *const c_char {
    match stmt.as_ref() {
        Some(handle) => handle.sql_cstr.as_ptr(),
        None => super::error::empty_cstr(),
    }
}

/// Finalize (destroy) a prepared statement and free resources.
///
/// Safe to call with NULL (no-op).
///
/// # Safety
///
/// `stmt` must be a pointer returned by `stoolap_prepare`, or NULL.
/// After this call, the pointer is invalid.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_finalize(stmt: *mut StoolapStmt) {
    if stmt.is_null() {
        return;
    }
    drop(Box::from_raw(stmt));
}

/// Get the last error message for a statement handle.
///
/// # Safety
///
/// `stmt` must be a valid `StoolapStmt` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_stmt_errmsg(stmt: *const StoolapStmt) -> *const c_char {
    match stmt.as_ref() {
        Some(handle) => handle.error_ptr(),
        None => super::error::empty_cstr(),
    }
}
