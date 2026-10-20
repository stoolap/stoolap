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

//! Database lifecycle and query execution FFI functions.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic;

use crate::api::Database;
use crate::common::version::VERSION;

use super::error;
use super::rows::pack_all;
use super::types::{StoolapDB, StoolapNamedParam, StoolapRows, StoolapTx, StoolapValue};
use super::value;
use super::{STOOLAP_ERROR, STOOLAP_OK};

/// Version string with static lifetime, initialized once.
static VERSION_CSTR: std::sync::OnceLock<CString> = std::sync::OnceLock::new();

/// Returns the stoolap version string.
///
/// The returned pointer is static and must NOT be freed.
#[no_mangle]
pub extern "C" fn stoolap_version() -> *const c_char {
    VERSION_CSTR
        .get_or_init(|| CString::new(VERSION).unwrap_or_default())
        .as_ptr()
}

/// Open a database connection.
///
/// # Safety
///
/// - `dsn` must be a valid null-terminated UTF-8 string.
/// - `out_db` must be a valid pointer to a `*mut StoolapDB`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_open(dsn: *const c_char, out_db: *mut *mut StoolapDB) -> i32 {
    if out_db.is_null() {
        return STOOLAP_ERROR;
    }
    *out_db = std::ptr::null_mut();

    if dsn.is_null() {
        error::set_global_error("DSN string is NULL");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let dsn_str = match CStr::from_ptr(dsn).to_str() {
            Ok(s) => s,
            Err(e) => {
                error::set_global_error(&format!("invalid UTF-8 in DSN: {}", e));
                return STOOLAP_ERROR;
            }
        };

        match Database::open(dsn_str) {
            Ok(db) => {
                let handle = Box::new(StoolapDB {
                    db,
                    last_error: None,
                    packed: Vec::new(),
                });
                *out_db = Box::into_raw(handle);
                STOOLAP_OK
            }
            Err(e) => {
                error::set_global_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        error::set_global_error("panic during stoolap_open");
        STOOLAP_ERROR
    })
}

/// Open an in-memory database.
///
/// # Safety
///
/// `out_db` must be a valid pointer to a `*mut StoolapDB`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_open_in_memory(out_db: *mut *mut StoolapDB) -> i32 {
    if out_db.is_null() {
        return STOOLAP_ERROR;
    }
    *out_db = std::ptr::null_mut();

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        match Database::open_in_memory() {
            Ok(db) => {
                let handle = Box::new(StoolapDB {
                    db,
                    last_error: None,
                    packed: Vec::new(),
                });
                *out_db = Box::into_raw(handle);
                STOOLAP_OK
            }
            Err(e) => {
                error::set_global_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        error::set_global_error("panic during stoolap_open_in_memory");
        STOOLAP_ERROR
    })
}

/// Close a database connection and free resources.
///
/// Safe to call with NULL (no-op).
///
/// # Safety
///
/// `db` must be a pointer returned by `stoolap_open*`, or NULL.
/// After this call, the pointer is invalid.
#[no_mangle]
pub unsafe extern "C" fn stoolap_close(db: *mut StoolapDB) -> i32 {
    if db.is_null() {
        return STOOLAP_OK;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let handle = Box::from_raw(db);

        // Drop the handle. The underlying engine is reference-counted (Arc)
        // and will be closed automatically when the last handle drops.
        // We intentionally do NOT call handle.db.close() here because it
        // would shut down the shared engine, breaking any cloned handles.
        drop(handle);
        STOOLAP_OK
    }));

    result.unwrap_or(STOOLAP_ERROR)
}

/// Clone a database handle for use in another thread.
///
/// The new handle shares the same underlying engine (data, tables, indexes)
/// but has its own executor and error state. This is the recommended way to
/// use stoolap from multiple threads: clone once per thread.
///
/// The returned handle must be closed independently with `stoolap_close()`.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `out_db` must be a valid pointer to a `*mut StoolapDB`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_clone(db: *const StoolapDB, out_db: *mut *mut StoolapDB) -> i32 {
    if out_db.is_null() {
        error::set_global_error("out_db pointer is NULL");
        return STOOLAP_ERROR;
    }
    *out_db = std::ptr::null_mut();

    let handle = match db.as_ref() {
        Some(h) => h,
        None => {
            error::set_global_error("db handle is NULL");
            return STOOLAP_ERROR;
        }
    };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let cloned = handle.db.clone();
        let new_handle = Box::new(StoolapDB {
            db: cloned,
            last_error: None,
            packed: Vec::new(),
        });
        *out_db = Box::into_raw(new_handle);
        STOOLAP_OK
    }));

    result.unwrap_or_else(|_| {
        error::set_global_error("panic during stoolap_clone");
        STOOLAP_ERROR
    })
}

/// Get the last error message for a database handle.
///
/// Returns `""` if no error. If `db` is NULL, returns the last global error
/// (from `stoolap_open` failures).
///
/// The returned pointer is valid until the next API call on this handle.
///
/// # Safety
///
/// `db` must be a valid `StoolapDB` pointer or NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_errmsg(db: *const StoolapDB) -> *const c_char {
    match db.as_ref() {
        Some(handle) => handle.error_ptr(),
        None => error::global_error_ptr(),
    }
}

/// Execute a SQL statement without parameters.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_exec(
    db: *mut StoolapDB,
    sql: *const c_char,
    rows_affected: *mut i64,
) -> i32 {
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

        match handle.db.execute(sql_str, ()) {
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
        handle.set_error("panic during stoolap_exec");
        STOOLAP_ERROR
    })
}

/// Execute a SQL statement with positional parameters.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL if `params_len` is 0).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_exec_params(
    db: *mut StoolapDB,
    sql: *const c_char,
    params: *const StoolapValue,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
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

        match value::params_to_vec(params, params_len).and_then(|p| handle.db.execute(sql_str, p)) {
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
        handle.set_error("panic during stoolap_exec_params");
        STOOLAP_ERROR
    })
}

/// Execute a query without parameters, returning a result set.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_query(
    db: *mut StoolapDB,
    sql: *const c_char,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    stoolap_query_params(db, sql, std::ptr::null(), 0, out_rows)
}

/// Execute a query with positional parameters, returning a result set.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_query_params(
    db: *mut StoolapDB,
    sql: *const c_char,
    params: *const StoolapValue,
    params_len: i32,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    if out_rows.is_null() {
        return STOOLAP_ERROR;
    }
    *out_rows = std::ptr::null_mut();

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

        match value::params_to_vec(params, params_len).and_then(|p| handle.db.query(sql_str, p)) {
            Ok(rows) => {
                *out_rows = StoolapRows::boxed(rows, None);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_query_params");
        STOOLAP_ERROR
    })
}

/// Run a query with positional parameters and pack every row into one
/// buffer, in the format of `stoolap_rows_fetch_all()`, without a rows
/// handle.
///
/// Runs inside `tx` when it is not NULL; a `tx` begun on another database
/// than `db` (a `stoolap_clone()` of it is the same one) is an error. The
/// buffer belongs to `db` and stays valid until the next call on it or
/// `stoolap_close()`. Must NOT be freed. Errors are read with
/// `stoolap_errmsg(db)`.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `tx` must be a valid `StoolapTx` pointer or NULL.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapValue` structs (or be NULL).
/// - `out_buf` and `out_len` must not be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_query_params_all(
    db: *mut StoolapDB,
    tx: *mut StoolapTx,
    sql: *const c_char,
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

        let rows = value::params_to_vec(params, params_len).and_then(|p| {
            match super::statement::open_transaction_on(&handle.db, tx)? {
                None => handle.db.query(sql_str, p),
                Some(inner_tx) => inner_tx.query(sql_str, p),
            }
        });
        match rows.and_then(|mut rows| pack_all(&mut rows, &mut handle.packed)) {
            Ok(_) => {
                *out_buf = handle.packed.as_ptr();
                *out_len = handle.packed.len() as i64;
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_query_params_all");
        STOOLAP_ERROR
    })
}

/// Execute a SQL statement with named parameters.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `rows_affected` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_exec_named(
    db: *mut StoolapDB,
    sql: *const c_char,
    params: *const StoolapNamedParam,
    params_len: i32,
    rows_affected: *mut i64,
) -> i32 {
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

        let named = value::named_params_from_ffi(params, params_len);

        match named.and_then(|named| handle.db.execute_named(sql_str, named)) {
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
        handle.set_error("panic during stoolap_exec_named");
        STOOLAP_ERROR
    })
}

/// Execute a query with named parameters, returning a result set.
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_query_named(
    db: *mut StoolapDB,
    sql: *const c_char,
    params: *const StoolapNamedParam,
    params_len: i32,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    if out_rows.is_null() {
        return STOOLAP_ERROR;
    }
    *out_rows = std::ptr::null_mut();

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

        let named = value::named_params_from_ffi(params, params_len);

        match named.and_then(|named| handle.db.query_named(sql_str, named)) {
            Ok(rows) => {
                *out_rows = StoolapRows::boxed(rows, None);
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_query_named");
        STOOLAP_ERROR
    })
}

/// `stoolap_query_params_all()` with named parameters (`:name`).
///
/// # Safety
///
/// - `db` must be a valid `StoolapDB` pointer.
/// - `tx` must be a valid `StoolapTx` pointer or NULL.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapNamedParam` structs (or be NULL).
/// - `out_buf` and `out_len` must not be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_query_named_all(
    db: *mut StoolapDB,
    tx: *mut StoolapTx,
    sql: *const c_char,
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

        let rows = value::named_params_from_ffi(params, params_len).and_then(|named| {
            match super::statement::open_transaction_on(&handle.db, tx)? {
                None => handle.db.query_named(sql_str, named),
                Some(inner_tx) => inner_tx.query_named(sql_str, named),
            }
        });
        match rows.and_then(|mut rows| pack_all(&mut rows, &mut handle.packed)) {
            Ok(_) => {
                *out_buf = handle.packed.as_ptr();
                *out_len = handle.packed.len() as i64;
                STOOLAP_OK
            }
            Err(e) => {
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_query_named_all");
        STOOLAP_ERROR
    })
}

/// Free a string allocated by the library.
///
/// Safe to call with NULL (no-op).
///
/// # Safety
///
/// `s` must be a pointer returned by a stoolap function that requires freeing, or NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_string_free(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}
