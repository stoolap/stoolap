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

//! Result set iteration FFI functions.

use std::os::raw::c_char;
use std::panic;

use crate::core::types::DataType;
use crate::core::Value;

use super::types::StoolapRows;
use super::{
    STOOLAP_DONE, STOOLAP_ERROR, STOOLAP_OK, STOOLAP_ROW, STOOLAP_TYPE_BLOB, STOOLAP_TYPE_BOOLEAN,
    STOOLAP_TYPE_FLOAT, STOOLAP_TYPE_INTEGER, STOOLAP_TYPE_JSON, STOOLAP_TYPE_NULL,
    STOOLAP_TYPE_TEXT, STOOLAP_TYPE_TIMESTAMP,
};

/// Build a NUL-terminated byte buffer from a string.
///
/// Unlike `CString`, this preserves interior NUL bytes so that callers
/// using the `out_len` parameter can access the full data.  Callers
/// treating the pointer as a C string will naturally see a truncated
/// view at the first embedded NUL — this matches C convention.
fn make_text_buf(s: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(s.len() + 1);
    buf.extend_from_slice(s.as_bytes());
    buf.push(0); // trailing NUL for C compatibility
    buf
}

/// Advance to the next row.
///
/// Returns `STOOLAP_ROW` if a row is available, `STOOLAP_DONE` when exhausted,
/// or `STOOLAP_ERROR` on error.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_next(rows: *mut StoolapRows) -> i32 {
    let handle = match rows.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let rows_inner = match &mut handle.rows {
            Some(r) => r,
            None => return STOOLAP_DONE,
        };

        // Clear text cache from previous row (only if anything was cached)
        if handle.text_cache_dirty {
            for slot in &mut handle.text_cache {
                *slot = None;
            }
            handle.text_cache_dirty = false;
        }

        if rows_inner.advance() {
            handle.has_row = true;
            STOOLAP_ROW
        } else {
            handle.has_row = false;
            // Check for runtime filter errors (e.g., invalid parameterized REGEXP)
            if let Some(err) = rows_inner.error() {
                handle.set_error(&err.to_string());
                return STOOLAP_ERROR;
            }
            STOOLAP_DONE
        }
    }));

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_rows_next");
        STOOLAP_ERROR
    })
}

/// Get the number of columns in the result set.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_count(rows: *const StoolapRows) -> i32 {
    match rows.as_ref() {
        Some(handle) => handle.column_names.len() as i32,
        None => 0,
    }
}

/// Get the name of a column by index (0-based).
///
/// Returns a pointer valid until `stoolap_rows_close()`. Must NOT be freed.
/// Returns NULL if index is out of bounds.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_name(
    rows: *const StoolapRows,
    index: i32,
) -> *const c_char {
    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return std::ptr::null(),
    };

    if index < 0 {
        return std::ptr::null();
    }

    match handle.column_names.get(index as usize) {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Get the current Value at a column index, if the row is valid.
unsafe fn get_current_value(handle: &StoolapRows, index: i32) -> Option<&Value> {
    if !handle.has_row || index < 0 {
        return None;
    }
    let rows_inner = handle.rows.as_ref()?;
    let row = rows_inner.current_row();
    row.get(index as usize)
}

/// Get the type of a column value in the current row (0-based).
///
/// Returns a `STOOLAP_TYPE_*` constant. Only valid after a successful `stoolap_rows_next()`.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_type(rows: *const StoolapRows, index: i32) -> i32 {
    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return STOOLAP_TYPE_NULL,
    };

    match get_current_value(handle, index) {
        Some(val) => {
            if val.is_null() {
                return STOOLAP_TYPE_NULL;
            }
            match val.data_type() {
                DataType::Null => STOOLAP_TYPE_NULL,
                DataType::Integer => STOOLAP_TYPE_INTEGER,
                DataType::Float => STOOLAP_TYPE_FLOAT,
                DataType::Text => STOOLAP_TYPE_TEXT,
                DataType::Boolean => STOOLAP_TYPE_BOOLEAN,
                DataType::Timestamp => STOOLAP_TYPE_TIMESTAMP,
                DataType::Json => STOOLAP_TYPE_JSON,
                DataType::Vector => STOOLAP_TYPE_BLOB,
            }
        }
        None => STOOLAP_TYPE_NULL,
    }
}

/// Get an integer value from the current row. Returns 0 if NULL or not convertible.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_int64(rows: *const StoolapRows, index: i32) -> i64 {
    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return 0,
    };

    match get_current_value(handle, index) {
        Some(val) => val.as_int64().unwrap_or(0),
        None => 0,
    }
}

/// Get a float value from the current row. Returns 0.0 if NULL or not convertible.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_double(rows: *const StoolapRows, index: i32) -> f64 {
    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return 0.0,
    };

    match get_current_value(handle, index) {
        Some(val) => val.as_float64().unwrap_or(0.0),
        None => 0.0,
    }
}

/// Get a text value from the current row.
///
/// Returns a pointer valid until the next `stoolap_rows_next()` call.
/// `out_len`: if non-NULL, receives the byte length (excluding null terminator).
/// Returns NULL if the column is NULL. Must NOT be freed.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer. `out_len` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_text(
    rows: *mut StoolapRows,
    index: i32,
    out_len: *mut i64,
) -> *const c_char {
    let handle = match rows.as_mut() {
        Some(h) => h,
        None => return std::ptr::null(),
    };

    if !handle.has_row || index < 0 {
        return std::ptr::null();
    }

    let idx = index as usize;

    // Check text cache first
    if let Some(Some(ref cached)) = handle.text_cache.get(idx) {
        if !out_len.is_null() {
            // Length excludes the trailing NUL terminator
            *out_len = (cached.len() - 1) as i64;
        }
        return cached.as_ptr() as *const c_char;
    }

    // Get the value and build a NUL-terminated byte buffer.
    // Fast path: for Text/Json values, use as_str() to avoid an intermediate String allocation.
    // Slow path: for other types (Integer, Float, etc.), fall back to as_string().
    //
    // Interior NUL bytes are preserved in the buffer; callers using out_len
    // can access the full data.  Callers treating the pointer as a C string
    // will see a truncated view at the first embedded NUL.
    let buf = {
        let rows_inner = match &handle.rows {
            Some(r) => r,
            None => return std::ptr::null(),
        };
        let row = rows_inner.current_row();
        match row.get(idx) {
            Some(val) => {
                if let Some(s) = val.as_str() {
                    Some(make_text_buf(s))
                } else {
                    val.as_string().map(|s| make_text_buf(&s))
                }
            }
            None => None,
        }
    };

    match buf {
        Some(b) => {
            // Grow the cache lazily to fit this column index
            if handle.text_cache.len() <= idx {
                handle.text_cache.resize_with(idx + 1, || None);
            }
            handle.text_cache[idx] = Some(b);
            handle.text_cache_dirty = true;
            let cached = handle.text_cache[idx].as_ref().unwrap();
            if !out_len.is_null() {
                *out_len = (cached.len() - 1) as i64;
            }
            cached.as_ptr() as *const c_char
        }
        None => std::ptr::null(),
    }
}

/// Get a boolean value from the current row. Returns 0 (false) if NULL or not convertible.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_bool(rows: *const StoolapRows, index: i32) -> i32 {
    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return 0,
    };

    match get_current_value(handle, index) {
        Some(val) if val.as_boolean().unwrap_or(false) => 1,
        Some(_) => 0,
        None => 0,
    }
}

/// Get a timestamp as nanoseconds since Unix epoch (UTC).
/// Returns 0 if NULL or not convertible.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_timestamp(
    rows: *const StoolapRows,
    index: i32,
) -> i64 {
    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return 0,
    };

    match get_current_value(handle, index) {
        Some(val) => match val.as_timestamp() {
            Some(ts) => ts.timestamp_nanos_opt().unwrap_or(0),
            None => 0,
        },
        None => 0,
    }
}

/// Get a blob value (Extension/Vector types) from the current row.
///
/// `out_len`: receives the byte length.
/// Returns a pointer valid until the next `stoolap_rows_next()` call.
/// Returns NULL if the column is NULL or not a blob type. Must NOT be freed.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer. `out_len` must not be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_blob(
    rows: *const StoolapRows,
    index: i32,
    out_len: *mut i64,
) -> *const u8 {
    if !out_len.is_null() {
        *out_len = 0;
    }

    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return std::ptr::null(),
    };

    if !handle.has_row || index < 0 {
        return std::ptr::null();
    }

    let rows_inner = match &handle.rows {
        Some(r) => r,
        None => return std::ptr::null(),
    };

    let row = rows_inner.current_row();
    match row.get(index as usize) {
        Some(Value::Extension(data)) if data.first() == Some(&(DataType::Vector as u8)) => {
            // Skip the internal type-tag byte; expose only the f32 payload.
            // For empty vectors (tag byte only), returns a valid pointer with len=0.
            let payload = &data[1..];
            if !out_len.is_null() {
                *out_len = payload.len() as i64;
            }
            payload.as_ptr()
        }
        _ => std::ptr::null(),
    }
}

/// Check if the current row's column is NULL.
/// Returns 1 if NULL, 0 otherwise.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_column_is_null(rows: *const StoolapRows, index: i32) -> i32 {
    let handle = match rows.as_ref() {
        Some(h) => h,
        None => return 1,
    };

    match get_current_value(handle, index) {
        Some(val) => {
            if val.is_null() {
                1
            } else {
                0
            }
        }
        None => 1,
    }
}

/// Get the number of rows affected (for DML results).
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_affected(rows: *const StoolapRows) -> i64 {
    match rows.as_ref() {
        Some(handle) => handle.rows_affected,
        None => 0,
    }
}

/// Fetch all remaining rows into a packed binary buffer.
///
/// Binary format:
/// ```text
/// [column_count: u32 LE]
/// [for each column: name_len:u16 LE, name_bytes:u8[name_len]]
/// [row_count: u32 LE]
/// [for each row, for each column:
///   type_tag: u8
///   payload (varies by type):
///     NULL(0): empty
///     INTEGER(1): i64 LE (8 bytes)
///     FLOAT(2): f64 LE (8 bytes)
///     TEXT(3): len:u32 LE + bytes:u8[len]
///     BOOLEAN(4): u8 (0 or 1)
///     TIMESTAMP(5): i64 LE (8 bytes, nanos since epoch)
///     JSON(6): len:u32 LE + bytes:u8[len]
///     BLOB(7): len:u32 LE + bytes:u8[len]
/// ]
/// ```
///
/// On success, `*out_buf` receives a heap-allocated buffer and `*out_len`
/// its byte length. The caller MUST free it with `stoolap_buffer_free()`.
/// The rows handle is consumed and closed by this call.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer. `out_buf` and `out_len` must not be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_fetch_all(
    rows: *mut StoolapRows,
    out_buf: *mut *mut u8,
    out_len: *mut i64,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() {
        return STOOLAP_ERROR;
    }
    *out_buf = std::ptr::null_mut();
    *out_len = 0;

    let handle = match rows.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let rows_inner = match &mut handle.rows {
            Some(r) => r,
            None => {
                // Already exhausted — return empty buffer with just headers
                let col_count = handle.column_names.len() as u32;
                let mut buf = Vec::with_capacity(8);
                buf.extend_from_slice(&col_count.to_le_bytes());
                for name in handle.column_names.iter() {
                    let name_bytes = name.to_bytes();
                    buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                    buf.extend_from_slice(name_bytes);
                }
                buf.extend_from_slice(&0u32.to_le_bytes()); // 0 rows
                                                            // into_boxed_slice guarantees len == capacity (unlike shrink_to_fit)
                let boxed = buf.into_boxed_slice();
                let len = boxed.len();
                let ptr = Box::into_raw(boxed) as *mut u8;
                *out_buf = ptr;
                *out_len = len as i64;
                return STOOLAP_OK;
            }
        };

        let mut buf = Vec::with_capacity(4096);

        // Write column count
        let col_count = handle.column_names.len() as u32;
        buf.extend_from_slice(&col_count.to_le_bytes());

        // Write column names
        for name in handle.column_names.iter() {
            let name_bytes = name.to_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }

        // Reserve space for row count (fill in later)
        let row_count_offset = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        let mut row_count: u32 = 0;

        while rows_inner.advance() {
            let row = rows_inner.current_row();
            for i in 0..col_count as usize {
                match row.get(i) {
                    None => {
                        buf.push(0); // TYPE_NULL
                    }
                    Some(val) if val.is_null() => {
                        buf.push(0); // TYPE_NULL
                    }
                    Some(val) => match val.data_type() {
                        DataType::Null => {
                            buf.push(0);
                        }
                        DataType::Integer => {
                            buf.push(1);
                            let v = val.as_int64().unwrap_or(0);
                            buf.extend_from_slice(&v.to_le_bytes());
                        }
                        DataType::Float => {
                            buf.push(2);
                            let v = val.as_float64().unwrap_or(0.0);
                            buf.extend_from_slice(&v.to_le_bytes());
                        }
                        DataType::Text => {
                            buf.push(3);
                            let s = val.as_str().unwrap_or("");
                            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
                            buf.extend_from_slice(s.as_bytes());
                        }
                        DataType::Boolean => {
                            buf.push(4);
                            buf.push(if val.as_boolean().unwrap_or(false) {
                                1
                            } else {
                                0
                            });
                        }
                        DataType::Timestamp => {
                            buf.push(5);
                            let nanos = val
                                .as_timestamp()
                                .map(|ts| ts.timestamp_nanos_opt().unwrap_or(0))
                                .unwrap_or(0);
                            buf.extend_from_slice(&nanos.to_le_bytes());
                        }
                        DataType::Json => {
                            buf.push(6);
                            let s = val.as_str().unwrap_or("");
                            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
                            buf.extend_from_slice(s.as_bytes());
                        }
                        DataType::Vector => {
                            buf.push(7);
                            if let Value::Extension(data) = val {
                                if data.first() == Some(&(DataType::Vector as u8)) {
                                    let payload = &data[1..];
                                    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                                    buf.extend_from_slice(payload);
                                } else {
                                    buf.extend_from_slice(&0u32.to_le_bytes());
                                }
                            } else {
                                buf.extend_from_slice(&0u32.to_le_bytes());
                            }
                        }
                    },
                }
            }
            row_count += 1;
        }

        // Fill in row count
        buf[row_count_offset..row_count_offset + 4].copy_from_slice(&row_count.to_le_bytes());

        // Transfer ownership to caller.
        // into_boxed_slice guarantees len == capacity (unlike shrink_to_fit
        // which is only a hint). This makes stoolap_buffer_free safe.
        let boxed = buf.into_boxed_slice();
        let len = boxed.len();
        let ptr = Box::into_raw(boxed) as *mut u8;

        *out_buf = ptr;
        *out_len = len as i64;
        STOOLAP_OK
    }));

    // Close rows handle after fetching
    handle.rows = None;
    handle.has_row = false;

    result.unwrap_or_else(|_| {
        handle.set_error("panic during stoolap_rows_fetch_all");
        STOOLAP_ERROR
    })
}

/// Free a buffer allocated by `stoolap_rows_fetch_all()`.
///
/// # Safety
///
/// `buf` must be a pointer returned by `stoolap_rows_fetch_all`, or NULL.
/// `len` must be the exact length returned alongside the buffer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_buffer_free(buf: *mut u8, len: i64) {
    if !buf.is_null() && len > 0 {
        // The buffer was created via Box::into_raw(Vec::into_boxed_slice()),
        // so len == capacity is guaranteed. Reconstruct and drop.
        let _ = Vec::from_raw_parts(buf, len as usize, len as usize);
    }
}

/// Close the result set and free resources.
///
/// Safe to call with NULL (no-op).
/// Must be called even if iteration completes (`STOOLAP_DONE`).
///
/// # Safety
///
/// `rows` must be a pointer returned by a stoolap query function, or NULL.
/// After this call, the pointer is invalid.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_close(rows: *mut StoolapRows) {
    if !rows.is_null() {
        let _ = Box::from_raw(rows);
    }
}

/// Get the last error message for a rows handle.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_errmsg(rows: *const StoolapRows) -> *const c_char {
    match rows.as_ref() {
        Some(handle) => handle.error_ptr(),
        None => super::error::empty_cstr(),
    }
}
