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

use crate::api::Database;

use super::types::{StoolapDB, StoolapRows, StoolapStmt, StoolapValue};
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
                let engine_keepalive = handle._engine_keepalive.clone();
                let stmt_handle = Box::new(StoolapStmt {
                    stmt,
                    last_error: None,
                    sql_cstr,
                    cached_columns: None,
                    _db_keepalive: db_keepalive,
                    _engine_keepalive: engine_keepalive,
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
        let param_vec = value::params_to_vec(params, params_len);

        match handle.stmt.execute(param_vec) {
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

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut tx = match db_handle.db.begin() {
            Ok(t) => t,
            Err(e) => {
                db_handle.set_error(&e.to_string());
                return STOOLAP_ERROR;
            }
        };

        let ast_stmt = stmt_handle.stmt.ast_statement();
        let ppr = params_per_row as usize;
        let mut total: i64 = 0;

        for r in 0..(row_count as usize) {
            let row_params = params.add(r * ppr);
            let param_vec = value::params_to_vec(row_params, params_per_row);

            let exec_result = if let Some(ast) = ast_stmt {
                tx.execute_prepared(ast, param_vec)
            } else {
                let sql_str = stmt_handle.sql_cstr.to_str().unwrap_or("");
                tx.execute(sql_str, param_vec)
            };

            match exec_result {
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
        let param_vec = value::params_to_vec(params, params_len);

        match handle.stmt.query(param_vec) {
            Ok(rows) => {
                let actual_columns = rows.columns();
                let actual_count = actual_columns.len();

                // Validate cache: rebuild if column count or names changed (DDL)
                let cache_valid = handle.cached_columns.as_ref().is_some_and(|cached| {
                    cached.len() == actual_count
                        && cached
                            .iter()
                            .zip(actual_columns.iter())
                            .all(|(c, a)| c.as_bytes() == a.as_str().as_bytes())
                });
                let column_names = if cache_valid {
                    Arc::clone(handle.cached_columns.as_ref().unwrap())
                } else {
                    let names: Vec<CString> = actual_columns
                        .iter()
                        .map(|name| CString::new(name.as_str()).unwrap_or_default())
                        .collect();
                    let arc = Arc::new(names);
                    handle.cached_columns = Some(Arc::clone(&arc));
                    arc
                };
                let affected = rows.rows_affected();

                let rows_handle = Box::new(StoolapRows {
                    rows: Some(rows),
                    has_row: false,
                    last_error: None,
                    column_names,
                    text_cache: Vec::new(),
                    text_cache_dirty: false,
                    rows_affected: affected,
                });
                *out_rows = Box::into_raw(rows_handle);
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
    let handle = Box::from_raw(stmt);

    // Clone the engine-owning Arc before dropping the statement.
    // After the statement drops, this may be the last non-registry reference,
    // so we retry registry cleanup (try_unregister_arc checks strong_count == 2).
    let engine_owning = match &handle._engine_keepalive {
        Some(arc) => Arc::clone(arc),
        None => Arc::clone(&handle._db_keepalive),
    };
    drop(handle);
    Database::try_unregister_arc(&engine_owning);
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
