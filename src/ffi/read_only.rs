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

//! Read-only handle FFI.
//!
//! Mirrors `crate::api::ReadOnlyDatabase` at the C boundary: open returns
//! a `StoolapRoDB*` that exposes only read entry points
//! (`stoolap_ro_query*`, `stoolap_ro_table_*`, `stoolap_ro_refresh`).
//! Write SQL routed through this handle is impossible at link time
//! because no `stoolap_ro_exec` / `stoolap_ro_begin` exists. This
//! mirrors the type-system enforcement that `Database::open_read_only`
//! gives Rust callers.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic;
use std::sync::Arc;

use crate::api::Database;

use super::error::{self, LastErrorState};
use super::types::{
    StoolapErrorDetails, StoolapNamedParam, StoolapRoDB, StoolapRows, StoolapValue,
};
use super::value;
use super::{STOOLAP_ERROR, STOOLAP_OK};

/// Open a read-only handle on `dsn`.
///
/// Accepts the same DSN spellings as `stoolap_open` plus the read-only
/// flags `?read_only=1` / `?readonly=true` / `?mode=ro` (which
/// `stoolap_open` REJECTS). The DSN flag is redundant — the function
/// is the read-only entry point.
///
/// # Safety
///
/// - `dsn` must be a valid null-terminated UTF-8 string.
/// - `out_db` must be a valid pointer to a `*mut StoolapRoDB`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_open_read_only(
    dsn: *const c_char,
    out_db: *mut *mut StoolapRoDB,
) -> i32 {
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

        match Database::open_read_only(dsn_str) {
            Ok(ro) => {
                let handle = Box::new(StoolapRoDB {
                    ro,
                    last_error: LastErrorState::default(),
                    dsn_cstr: std::sync::OnceLock::new(),
                });
                *out_db = Box::into_raw(handle);
                STOOLAP_OK
            }
            Err(e) => {
                error::set_global_error_from(&e);
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        error::set_global_error("panic during stoolap_open_read_only");
        STOOLAP_ERROR
    })
}

/// Clone a read-only handle for multi-threaded use.
///
/// Each clone shares the underlying engine (cold volumes, segment
/// manager, semantic + plan caches, the writer's WAL being tailed)
/// but has its own per-handle state: independent executor, fresh
/// `ReaderAttachment` (its own WAL pin contribution), its own
/// `auto_refresh` flag, its own overlay cursor. A `BEGIN` /
/// `set_auto_refresh` / `refresh` on one clone does not affect
/// the others.
///
/// Recommended pattern for multi-threaded readers: open once,
/// then `stoolap_ro_clone` per worker thread. Each thread owns
/// its clone exclusively. Each clone must be closed independently
/// with `stoolap_ro_close`. Engine resources are released only
/// when the last clone (and any open `StoolapRows` referencing
/// it) drops. Mirrors `stoolap_clone` for writable handles.
///
/// # Safety
///
/// - `db` must be a valid `StoolapRoDB` pointer.
/// - `out_db` must be a valid pointer to a `*mut StoolapRoDB`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_clone(
    db: *const StoolapRoDB,
    out_db: *mut *mut StoolapRoDB,
) -> i32 {
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
        let cloned = handle.ro.clone();
        let new_handle = Box::new(StoolapRoDB {
            ro: cloned,
            last_error: LastErrorState::default(),
            dsn_cstr: std::sync::OnceLock::new(),
        });
        *out_db = Box::into_raw(new_handle);
        STOOLAP_OK
    }));

    result.unwrap_or_else(|_| {
        error::set_global_error("panic during stoolap_ro_clone");
        STOOLAP_ERROR
    })
}

/// Close a read-only handle.
///
/// Safe to call with NULL (no-op).
///
/// # Safety
///
/// `db` must be a pointer returned by `stoolap_open_read_only`, or NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_close(db: *mut StoolapRoDB) {
    if !db.is_null() {
        let _ = Box::from_raw(db);
    }
}

/// Get the last error message for a read-only handle.
///
/// # Safety
///
/// `db` must be a valid `StoolapRoDB` pointer or NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_errmsg(db: *const StoolapRoDB) -> *const c_char {
    match db.as_ref() {
        Some(handle) => handle.error_ptr(),
        None => error::global_error_ptr(),
    }
}

/// Get the typed error code for a read-only handle's last error.
///
/// # Safety
///
/// `db` must be a valid `StoolapRoDB` pointer or NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_errcode(db: *const StoolapRoDB) -> i32 {
    match db.as_ref() {
        Some(handle) => handle.last_error.code,
        None => error::global_error_code(),
    }
}

/// Fill the caller's `StoolapErrorDetails` from this handle's last error.
///
/// # Safety
///
/// `db` must be a valid `StoolapRoDB` pointer or NULL. `out` must point
/// to a writable `StoolapErrorDetails`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_errdetails(
    db: *const StoolapRoDB,
    out: *mut StoolapErrorDetails,
) -> i32 {
    if out.is_null() {
        return STOOLAP_ERROR;
    }
    match db.as_ref() {
        Some(handle) => handle.last_error.fill_details(&mut *out),
        None => error::fill_global_error_details(&mut *out),
    }
    STOOLAP_OK
}

/// Returns the DSN this handle was opened with.
///
/// Pointer is valid for the lifetime of the handle. Must NOT be freed.
/// First call lazily populates a per-handle `CString`; subsequent calls
/// reuse the cached buffer at zero cost.
///
/// # Safety
///
/// `db` must be a valid `StoolapRoDB` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_dsn(db: *const StoolapRoDB) -> *const c_char {
    let handle = match db.as_ref() {
        Some(h) => h,
        None => return error::empty_cstr(),
    };
    handle
        .dsn_cstr
        .get_or_init(|| CString::new(handle.ro.dsn()).unwrap_or_default())
        .as_ptr()
}

/// Returns 1 if `name` exists in this read-only handle's snapshot,
/// 0 if not. Returns negative on error (and records the error on `db`).
///
/// # Safety
///
/// - `db` must be a valid `StoolapRoDB` pointer.
/// - `name` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_table_exists(db: *mut StoolapRoDB, name: *const c_char) -> i32 {
    let handle = match db.as_mut() {
        Some(h) => h,
        None => return -1,
    };
    handle.last_error.clear();

    if name.is_null() {
        handle.set_error("table name is NULL");
        return -1;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let n = match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.set_error(&format!("invalid UTF-8 in table name: {}", e));
                return -1;
            }
        };
        match handle.ro.table_exists(n) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => {
                handle.set_error_from(&e);
                -1
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_ro_table_exists");
        -1
    })
}

/// Returns the committed row count of `table`. O(1).
///
/// # Safety
///
/// - `db` must be a valid `StoolapRoDB` pointer.
/// - `table` must be a valid null-terminated UTF-8 string.
/// - `out_count` must be a valid pointer to a `u64`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_table_count(
    db: *mut StoolapRoDB,
    table: *const c_char,
    out_count: *mut u64,
) -> i32 {
    if out_count.is_null() {
        return STOOLAP_ERROR;
    }
    *out_count = 0;

    let handle = match db.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error.clear();

    if table.is_null() {
        handle.set_error("table name is NULL");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let n = match CStr::from_ptr(table).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.set_error(&format!("invalid UTF-8 in table name: {}", e));
                return STOOLAP_ERROR;
            }
        };
        match handle.ro.table_count(n) {
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
        handle.set_error("panic during stoolap_ro_table_count");
        STOOLAP_ERROR
    })
}

/// Manually advance the read-only handle to the writer's latest visible
/// state. Returns 1 if the snapshot moved, 0 if it was already current,
/// or `STOOLAP_ERROR` on must-reopen errors (which surface via the
/// handle's `errcode`/`errdetails`).
///
/// # Safety
///
/// `db` must be a valid `StoolapRoDB` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_refresh(db: *mut StoolapRoDB) -> i32 {
    let handle = match db.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    handle.last_error.clear();

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| match handle.ro.refresh() {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(e) => {
            handle.set_error_from(&e);
            STOOLAP_ERROR
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_ro_refresh");
        STOOLAP_ERROR
    })
}

/// Toggle automatic refresh on every query.
///
/// `enabled != 0` enables; `0` disables. Default is enabled.
///
/// # Safety
///
/// `db` must be a valid `StoolapRoDB` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_set_auto_refresh(db: *mut StoolapRoDB, enabled: i32) {
    if let Some(handle) = db.as_mut() {
        handle.ro.set_auto_refresh(enabled != 0);
    }
}

/// Query without parameters.
///
/// # Safety
///
/// - `db` must be a valid `StoolapRoDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_query(
    db: *mut StoolapRoDB,
    sql: *const c_char,
    out_rows: *mut *mut StoolapRows,
) -> i32 {
    stoolap_ro_query_params(db, sql, std::ptr::null(), 0, out_rows)
}

/// Query with positional parameters.
///
/// # Safety
///
/// - `db` must be a valid `StoolapRoDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapValue` (or NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_query_params(
    db: *mut StoolapRoDB,
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
    handle.last_error.clear();

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

        let param_vec = value::params_to_vec(params, params_len);

        match handle.ro.query(sql_str, param_vec) {
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
        handle.set_error("panic during stoolap_ro_query_params");
        STOOLAP_ERROR
    })
}

/// Query with named parameters.
///
/// # Safety
///
/// - `db` must be a valid `StoolapRoDB` pointer.
/// - `sql` must be a valid null-terminated UTF-8 string.
/// - `params` must point to `params_len` valid `StoolapNamedParam` (or NULL).
/// - `out_rows` must be a valid pointer to a `*mut StoolapRows`.
#[no_mangle]
pub unsafe extern "C" fn stoolap_ro_query_named(
    db: *mut StoolapRoDB,
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
    handle.last_error.clear();

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

        match handle.ro.query_named(sql_str, named) {
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
        handle.set_error("panic during stoolap_ro_query_named");
        STOOLAP_ERROR
    })
}
