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

//! Transaction FFI functions.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic;
use std::sync::Arc;

use crate::api::Database;
use crate::core::types::IsolationLevel;

use super::types::{StoolapDB, StoolapRows, StoolapStmt, StoolapTx, StoolapValue};
use super::value;
use super::{
    STOOLAP_ERROR, STOOLAP_ISOLATION_READ_COMMITTED, STOOLAP_ISOLATION_SNAPSHOT, STOOLAP_OK,
};

/// Begin a transaction with the default isolation level (READ COMMITTED).
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `out_tx` must be a valid pointer to a `*mut StoolapTx`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_begin(db: *mut StoolapDB, out_tx: *mut *mut StoolapTx) -> i32 {
    if out_tx.is_null() {
        super::error::set_global_error("out_tx pointer is NULL");
        return STOOLAP_ERROR;
    }
    *out_tx = std::ptr::null_mut();

    let handle = match db.as_mut() {
        Some(h) => h,
        None => {
            super::error::set_global_error("db handle is NULL");
            return STOOLAP_ERROR;
        }
    };
    handle.last_error = None;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| match handle.db.begin() {
        Ok(tx) => {
            let tx_handle = Box::new(StoolapTx {
                tx: Some(tx),
                last_error: None,
                _db_keepalive: handle.db.keepalive(),
                _engine_keepalive: handle._engine_keepalive.clone(),
            });
            *out_tx = Box::into_raw(tx_handle);
            STOOLAP_OK
        }
        Err(e) => {
            handle.set_error(&e.to_string());
            STOOLAP_ERROR
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_begin");
        STOOLAP_ERROR
    })
}

/// Begin a transaction with a specific isolation level.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `isolation` must be `STOOLAP_ISOLATION_READ_COMMITTED` or `STOOLAP_ISOLATION_SNAPSHOT`.
/// - `out_tx` must be a valid pointer to a `*mut StoolapTx`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_begin_with_isolation(
    db: *mut StoolapDB,
    isolation: i32,
    out_tx: *mut *mut StoolapTx,
) -> i32 {
    if out_tx.is_null() {
        super::error::set_global_error("out_tx pointer is NULL");
        return STOOLAP_ERROR;
    }
    *out_tx = std::ptr::null_mut();

    let handle = match db.as_mut() {
        Some(h) => h,
        None => {
            super::error::set_global_error("db handle is NULL");
            return STOOLAP_ERROR;
        }
    };
    handle.last_error = None;

    let level = match isolation {
        STOOLAP_ISOLATION_READ_COMMITTED => IsolationLevel::ReadCommitted,
        STOOLAP_ISOLATION_SNAPSHOT => IsolationLevel::SnapshotIsolation,
        _ => {
            handle.set_error("invalid isolation level");
            return STOOLAP_ERROR;
        }
    };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        match handle.db.begin_with_isolation(level) {
            Ok(tx) => {
                let tx_handle = Box::new(StoolapTx {
                    tx: Some(tx),
                    last_error: None,
                    _db_keepalive: handle.db.keepalive(),
                    _engine_keepalive: handle._engine_keepalive.clone(),
                });
                *out_tx = Box::into_raw(tx_handle);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_begin_with_isolation");
        STOOLAP_ERROR
    })
}

/// Execute a SQL statement within a transaction (no parameters).
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_exec(
    tx: *mut StoolapTx,
    sql: *const c_char,
    rows_affected: *mut i64,
) -> i32 {
    stoolap_tx_exec_params(tx, sql, std::ptr::null(), 0, rows_affected)
}

/// Execute a SQL statement within a transaction (with parameters).
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_exec_params(
    tx: *mut StoolapTx,
    sql: *const c_char,
    params: *const StoolapValue,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    if sql.is_null() {
        handle.set_error("SQL string is NULL");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let inner_tx = match &mut handle.tx {
            Some(t) => t,
            None => {
                handle.set_error("transaction already ended");
                return STOOLAP_ERROR;
            }
        };

        let sql_str = match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.set_error(&format!("invalid UTF-8 in SQL: {}", e));
                return STOOLAP_ERROR;
            }
        };

        let param_vec = value::params_to_vec(params, params_len);

        match inner_tx.execute(sql_str, param_vec) {
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
        handle.set_error("panic during stoolap_tx_exec_params");
        STOOLAP_ERROR
    })
}

/// Query within a transaction (no parameters).
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_query(
    tx: *mut StoolapTx,
    sql: *const c_char,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    stoolap_tx_query_params(tx, sql, std::ptr::null(), 0, out_rows)
}

/// Query within a transaction (with parameters).
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_query_params(
    tx: *mut StoolapTx,
    sql: *const c_char,
    params: *const StoolapValue,
    params_len: i32,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    if out_rows.is_null() {
        return STOOLAP_ERROR;
    }
    *out_rows = std::ptr::null_mut();

    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    if sql.is_null() {
        handle.set_error("SQL string is NULL");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let inner_tx = match &mut handle.tx {
            Some(t) => t,
            None => {
                handle.set_error("transaction already ended");
                return STOOLAP_ERROR;
            }
        };

        let sql_str = match CStr::from_ptr(sql).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.set_error(&format!("invalid UTF-8 in SQL: {}", e));
                return STOOLAP_ERROR;
            }
        };

        let param_vec = value::params_to_vec(params, params_len);

        match inner_tx.query(sql_str, param_vec) {
            Ok(rows) => {
                let column_names: Vec<CString> = rows
                    .columns()
                    .iter()
                    .map(|name| CString::new(name.as_str()).unwrap_or_default())
                    .collect();
                let affected = rows.rows_affected();

                let rows_handle = Box::new(StoolapRows {
                    rows: Some(rows),
                    has_row: false,
                    last_error: None,
                    column_names: Arc::new(column_names),
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
        handle.set_error("panic during stoolap_tx_query_params");
        STOOLAP_ERROR
    })
}

/// Commit a transaction. The tx handle is consumed (freed).
///
/// # Safety
///
/// `tx` must be a valid `StoolapTx` pointer.
/// After this call (success or failure), the pointer is invalid.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_commit(tx: *mut StoolapTx) -> i32 {
    if tx.is_null() {
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut handle = Box::from_raw(tx);

        let rc = match handle.tx.take() {
            Some(mut inner_tx) => match inner_tx.commit() {
                Ok(()) => STOOLAP_OK,
                Err(e) => {
                    // Transaction is consumed regardless; log the error globally
                    super::error::set_global_error(&e.to_string());
                    STOOLAP_ERROR
                }
            },
            None => {
                super::error::set_global_error("transaction already ended");
                STOOLAP_ERROR
            }
        };

        // Retry registry cleanup: the tx keepalive Arcs may have been the
        // last non-registry references to the engine-owning DatabaseInner.
        let engine_owning = match &handle._engine_keepalive {
            Some(arc) => Arc::clone(arc),
            None => Arc::clone(&handle._db_keepalive),
        };
        drop(handle);
        Database::try_unregister_arc(&engine_owning);

        rc
    }));

    result.unwrap_or(STOOLAP_ERROR)
}

/// Rollback a transaction. The tx handle is consumed (freed).
///
/// # Safety
///
/// `tx` must be a valid `StoolapTx` pointer.
/// After this call (success or failure), the pointer is invalid.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_rollback(tx: *mut StoolapTx) -> i32 {
    if tx.is_null() {
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut handle = Box::from_raw(tx);

        let rc = match handle.tx.take() {
            Some(mut inner_tx) => match inner_tx.rollback() {
                Ok(()) => STOOLAP_OK,
                Err(e) => {
                    super::error::set_global_error(&e.to_string());
                    STOOLAP_ERROR
                }
            },
            None => {
                super::error::set_global_error("transaction already ended");
                STOOLAP_ERROR
            }
        };

        let engine_owning = match &handle._engine_keepalive {
            Some(arc) => Arc::clone(arc),
            None => Arc::clone(&handle._db_keepalive),
        };
        drop(handle);
        Database::try_unregister_arc(&engine_owning);

        rc
    }));

    result.unwrap_or(STOOLAP_ERROR)
}

/// Execute a prepared statement within a transaction (with parameters).
///
/// This gives both parse-once performance AND transaction atomicity.
/// The statement must have been created via `stoolap_prepare()`.
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_stmt_exec(
    tx: *mut StoolapTx,
    stmt: *const StoolapStmt,
    params: *const StoolapValue,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let stmt_handle = match stmt.as_ref() {
        Some(h) => h,
        None => {
            handle.set_error("statement handle is NULL");
            return STOOLAP_ERROR;
        }
    };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let inner_tx = match &mut handle.tx {
            Some(t) => t,
            None => {
                handle.set_error("transaction already ended");
                return STOOLAP_ERROR;
            }
        };

        let ast_stmt = match stmt_handle.stmt.ast_statement() {
            Some(s) => s,
            None => {
                // Multi-statement SQL: fall back to SQL-based execution
                let sql_str = stmt_handle.sql_cstr.to_str().unwrap_or("");
                let param_vec = value::params_to_vec(params, params_len);
                match inner_tx.execute(sql_str, param_vec) {
                    Ok(affected) => {
                        if !rows_affected.is_null() {
                            *rows_affected = affected;
                        }
                        return STOOLAP_OK;
                    }
                    Err(e) => {
                        handle.set_error(&e.to_string());
                        return STOOLAP_ERROR;
                    }
                }
            }
        };

        let param_vec = value::params_to_vec(params, params_len);
        match inner_tx.execute_prepared(ast_stmt, param_vec) {
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
        handle.set_error("panic during stoolap_tx_stmt_exec");
        STOOLAP_ERROR
    })
}

/// Query using a prepared statement within a transaction (with parameters).
///
/// This gives both parse-once performance AND transaction atomicity.
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_stmt_query(
    tx: *mut StoolapTx,
    stmt: *const StoolapStmt,
    params: *const StoolapValue,
    params_len: i32,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    if out_rows.is_null() {
        return STOOLAP_ERROR;
    }
    *out_rows = std::ptr::null_mut();

    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error = None;

    let stmt_handle = match stmt.as_ref() {
        Some(h) => h,
        None => {
            handle.set_error("statement handle is NULL");
            return STOOLAP_ERROR;
        }
    };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let inner_tx = match &mut handle.tx {
            Some(t) => t,
            None => {
                handle.set_error("transaction already ended");
                return STOOLAP_ERROR;
            }
        };

        let ast_stmt = match stmt_handle.stmt.ast_statement() {
            Some(s) => s,
            None => {
                // Multi-statement SQL: fall back to SQL-based execution
                let sql_str = stmt_handle.sql_cstr.to_str().unwrap_or("");
                let param_vec = value::params_to_vec(params, params_len);
                match inner_tx.query(sql_str, param_vec) {
                    Ok(rows) => {
                        let column_names: Vec<CString> = rows
                            .columns()
                            .iter()
                            .map(|name| CString::new(name.as_str()).unwrap_or_default())
                            .collect();
                        let affected = rows.rows_affected();

                        let rows_handle = Box::new(StoolapRows {
                            rows: Some(rows),
                            has_row: false,
                            last_error: None,
                            column_names: Arc::new(column_names),
                            text_cache: Vec::new(),
                            text_cache_dirty: false,
                            rows_affected: affected,
                        });
                        *out_rows = Box::into_raw(rows_handle);
                        return STOOLAP_OK;
                    }
                    Err(e) => {
                        handle.set_error(&e.to_string());
                        return STOOLAP_ERROR;
                    }
                }
            }
        };

        let param_vec = value::params_to_vec(params, params_len);
        match inner_tx.query_prepared(ast_stmt, param_vec) {
            Ok(rows) => {
                let column_names: Vec<CString> = rows
                    .columns()
                    .iter()
                    .map(|name| CString::new(name.as_str()).unwrap_or_default())
                    .collect();
                let affected = rows.rows_affected();

                let rows_handle = Box::new(StoolapRows {
                    rows: Some(rows),
                    has_row: false,
                    last_error: None,
                    column_names: Arc::new(column_names),
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
        handle.set_error("panic during stoolap_tx_stmt_query");
        STOOLAP_ERROR
    })
}

/// Get the last error message for a transaction handle.
///
/// # Safety
///
/// `tx` must be a valid `StoolapTx` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_errmsg(tx: *const StoolapTx) -> *const c_char {
    match tx.as_ref() {
        Some(handle) => handle.error_ptr(),
        None => super::error::global_error_ptr(),
    }
}
