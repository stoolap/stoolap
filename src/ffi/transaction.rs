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

use super::error::LastErrorState;
use super::types::{
    StoolapDB, StoolapErrorDetails, StoolapNamedParam, StoolapRows, StoolapStmt, StoolapTx,
    StoolapValue,
};
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
    handle.last_error.clear();

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| match handle.db.begin() {
        Ok(tx) => {
            let tx_handle = Box::new(StoolapTx {
                tx: Some(tx),
                last_error: LastErrorState::default(),
                _db_keepalive: handle.db.keepalive(),
                _engine_keepalive: handle._engine_keepalive.clone(),
            });
            *out_tx = Box::into_raw(tx_handle);
            STOOLAP_OK
        }
        Err(e) => {
            handle.set_error_from(&e);
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
    handle.last_error.clear();

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
                    last_error: LastErrorState::default(),
                    _db_keepalive: handle.db.keepalive(),
                    _engine_keepalive: handle._engine_keepalive.clone(),
                });
                *out_tx = Box::into_raw(tx_handle);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
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
    handle.last_error.clear();

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
                handle.set_error_from(&e);
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
    handle.last_error.clear();

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
                    last_error: LastErrorState::default(),
                    column_names: Arc::new(column_names),
                    text_cache: Vec::new(),
                    text_cache_dirty: false,
                    rows_affected: affected,
                });
                *out_rows = Box::into_raw(rows_handle);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
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
                    super::error::set_global_error_from(&e);
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
                    super::error::set_global_error_from(&e);
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
    handle.last_error.clear();

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
                        handle.set_error_from(&e);
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
                handle.set_error_from(&e);
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
    handle.last_error.clear();

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
                            last_error: LastErrorState::default(),
                            column_names: Arc::new(column_names),
                            text_cache: Vec::new(),
                            text_cache_dirty: false,
                            rows_affected: affected,
                        });
                        *out_rows = Box::into_raw(rows_handle);
                        return STOOLAP_OK;
                    }
                    Err(e) => {
                        handle.set_error_from(&e);
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
                    last_error: LastErrorState::default(),
                    column_names: Arc::new(column_names),
                    text_cache: Vec::new(),
                    text_cache_dirty: false,
                    rows_affected: affected,
                });
                *out_rows = Box::into_raw(rows_handle);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_tx_stmt_query");
        STOOLAP_ERROR
    })
}

/// Execute a SQL statement with named parameters within a transaction.
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_exec_named(
    tx: *mut StoolapTx,
    sql: *const c_char,
    params: *const StoolapNamedParam,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error.clear();

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

        let named = value::named_params_from_ffi(params, params_len);

        match inner_tx.execute_named(sql_str, named) {
            Ok(affected) => {
                if !rows_affected.is_null() {
                    *rows_affected = affected;
                }
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_tx_exec_named");
        STOOLAP_ERROR
    })
}

/// Query with named parameters within a transaction.
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_query_named(
    tx: *mut StoolapTx,
    sql: *const c_char,
    params: *const StoolapNamedParam,
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
    handle.last_error.clear();

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

        let named = value::named_params_from_ffi(params, params_len);

        match inner_tx.query_named(sql_str, named) {
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
                    last_error: LastErrorState::default(),
                    column_names: Arc::new(column_names),
                    text_cache: Vec::new(),
                    text_cache_dirty: false,
                    rows_affected: affected,
                });
                *out_rows = Box::into_raw(rows_handle);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_tx_query_named");
        STOOLAP_ERROR
    })
}

/// Execute a prepared statement with named parameters within a transaction.
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_stmt_exec_named(
    tx: *mut StoolapTx,
    stmt: *const StoolapStmt,
    params: *const StoolapNamedParam,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error.clear();

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

        let named = value::named_params_from_ffi(params, params_len);

        let ast_stmt = match stmt_handle.stmt.ast_statement() {
            Some(s) => s,
            None => {
                // Multi-statement SQL: fall back to SQL-based execution
                let sql_str = stmt_handle.sql_cstr.to_str().unwrap_or("");
                match inner_tx.execute_named(sql_str, named) {
                    Ok(affected) => {
                        if !rows_affected.is_null() {
                            *rows_affected = affected;
                        }
                        return STOOLAP_OK;
                    }
                    Err(e) => {
                        handle.set_error_from(&e);
                        return STOOLAP_ERROR;
                    }
                }
            }
        };

        match inner_tx.execute_prepared_named(ast_stmt, named) {
            Ok(affected) => {
                if !rows_affected.is_null() {
                    *rows_affected = affected;
                }
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_tx_stmt_exec_named");
        STOOLAP_ERROR
    })
}

/// Query using a prepared statement with named parameters within a transaction.
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `stmt` must be a valid `StoolapStmt` pointer.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_stmt_query_named(
    tx: *mut StoolapTx,
    stmt: *const StoolapStmt,
    params: *const StoolapNamedParam,
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
    handle.last_error.clear();

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

        let named = value::named_params_from_ffi(params, params_len);

        let ast_stmt = match stmt_handle.stmt.ast_statement() {
            Some(s) => s,
            None => {
                // Multi-statement SQL: fall back to SQL-based execution
                let sql_str = stmt_handle.sql_cstr.to_str().unwrap_or("");
                match inner_tx.query_named(sql_str, named) {
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
                            last_error: LastErrorState::default(),
                            column_names: Arc::new(column_names),
                            text_cache: Vec::new(),
                            text_cache_dirty: false,
                            rows_affected: affected,
                        });
                        *out_rows = Box::into_raw(rows_handle);
                        return STOOLAP_OK;
                    }
                    Err(e) => {
                        handle.set_error_from(&e);
                        return STOOLAP_ERROR;
                    }
                }
            }
        };

        match inner_tx.query_prepared_named(ast_stmt, named) {
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
                    last_error: LastErrorState::default(),
                    column_names: Arc::new(column_names),
                    text_cache: Vec::new(),
                    text_cache_dirty: false,
                    rows_affected: affected,
                });
                *out_rows = Box::into_raw(rows_handle);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_tx_stmt_query_named");
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

/// Get the typed error code for a transaction handle's last error.
/// Returns `STOOLAP_ERR_OK` when no error is pending.
///
/// # Safety
///
/// `tx` must be a valid `StoolapTx` pointer or NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_errcode(tx: *const StoolapTx) -> i32 {
    match tx.as_ref() {
        Some(handle) => handle.last_error.code,
        None => super::error::global_error_code(),
    }
}

/// Fill the caller's `StoolapErrorDetails` from this tx's last error.
/// Pointers stay valid until the next API call on this tx handle.
///
/// # Safety
///
/// `tx` must be a valid `StoolapTx` pointer or NULL. `out` must point to
/// a writable `StoolapErrorDetails`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_errdetails(
    tx: *const StoolapTx,
    out: *mut StoolapErrorDetails,
) -> i32 {
    if out.is_null() {
        return STOOLAP_ERROR;
    }
    match tx.as_ref() {
        Some(handle) => handle.last_error.fill_details(&mut *out),
        None => super::error::fill_global_error_details(&mut *out),
    }
    STOOLAP_OK
}

/// Returns the row count of `table` visible to this transaction
/// (snapshot-correct, includes uncommitted local changes).
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `table` must be a valid null-terminated UTF-8 string.
/// - `out_count` must be a valid pointer to a `u64`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_table_count(
    tx: *mut StoolapTx,
    table: *const c_char,
    out_count: *mut u64,
) -> i32 {
    if out_count.is_null() {
        return STOOLAP_ERROR;
    }
    *out_count = 0;

    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error.clear();

    if table.is_null() {
        handle.set_error("table name is NULL");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let inner_tx = match &handle.tx {
            Some(t) => t,
            None => {
                handle.set_error("transaction already ended");
                return STOOLAP_ERROR;
            }
        };
        let name = match CStr::from_ptr(table).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.set_error(&format!("invalid UTF-8 in table name: {}", e));
                return STOOLAP_ERROR;
            }
        };
        match inner_tx.table_count(name) {
            Ok(c) => {
                *out_count = c;
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_tx_table_count");
        STOOLAP_ERROR
    })
}

/// Create a savepoint within this transaction. Re-using an existing
/// savepoint name overwrites it.
///
/// `name_len` is the length in bytes of the savepoint name (no NUL
/// required; pass `-1` to use C string length via `strlen`). Use the
/// explicit length when interoperating with non-NUL-terminated buffers
/// (e.g. MariaDB savepoint names from the handlerton chunk).
///
/// # Safety
///
/// - `tx` must be a valid `StoolapTx` pointer.
/// - `name` must point to `name_len` valid UTF-8 bytes (or be a valid
///   null-terminated C string when `name_len < 0`).
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_savepoint(
    tx: *mut StoolapTx,
    name: *const c_char,
    name_len: i32,
) -> i32 {
    savepoint_op(tx, name, name_len, "stoolap_tx_savepoint", |t, n| {
        t.create_savepoint(n)
    })
}

/// Release a savepoint without rolling back.
///
/// # Safety
///
/// See [`stoolap_tx_savepoint`] for `name`/`name_len` semantics.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_release_savepoint(
    tx: *mut StoolapTx,
    name: *const c_char,
    name_len: i32,
) -> i32 {
    savepoint_op(
        tx,
        name,
        name_len,
        "stoolap_tx_release_savepoint",
        |t, n| t.release_savepoint(n),
    )
}

/// Roll back to a named savepoint.
///
/// # Safety
///
/// See [`stoolap_tx_savepoint`] for `name`/`name_len` semantics.
#[no_mangle]
pub unsafe extern "C" fn stoolap_tx_rollback_to_savepoint(
    tx: *mut StoolapTx,
    name: *const c_char,
    name_len: i32,
) -> i32 {
    savepoint_op(
        tx,
        name,
        name_len,
        "stoolap_tx_rollback_to_savepoint",
        |t, n| t.rollback_to_savepoint(n),
    )
}

/// Common savepoint dispatcher. Borrows the name as a `&str` (no
/// allocation) and forwards to the supplied `Transaction` method.
///
/// # Safety
///
/// Same as the public callers (`tx` valid; `name` either NUL-terminated
/// or `name_len`-bytes long).
unsafe fn savepoint_op<F>(
    tx: *mut StoolapTx,
    name: *const c_char,
    name_len: i32,
    panic_label: &'static str,
    op: F,
) -> i32
where
    F: FnOnce(&mut crate::api::Transaction, &str) -> crate::core::Result<()>,
{
    let handle = match tx.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error.clear();

    if name.is_null() {
        handle.set_error("savepoint name is NULL");
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

        // Borrow `name` without owning a copy: when `name_len < 0` use the
        // NUL-terminated form; otherwise treat as raw bytes of length
        // `name_len`. Both paths produce a &str into caller-owned memory
        // that lives for the duration of this FFI call.
        let bytes: &[u8] = if name_len < 0 {
            CStr::from_ptr(name).to_bytes()
        } else {
            std::slice::from_raw_parts(name as *const u8, name_len as usize)
        };
        let name_str = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                handle.set_error(&format!("invalid UTF-8 in savepoint name: {}", e));
                return STOOLAP_ERROR;
            }
        };

        match op(inner_tx, name_str) {
            Ok(()) => STOOLAP_OK,
            Err(e) => {
                handle.set_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error(&format!("panic during {}", panic_label));
        STOOLAP_ERROR
    })
}
