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

use crate::api::Rows;
use crate::core::types::DataType;
use crate::core::{Result, Value};

use super::types::StoolapRows;
use super::{
    STOOLAP_DONE, STOOLAP_ERROR, STOOLAP_OK, STOOLAP_ROW, STOOLAP_TYPE_BLOB, STOOLAP_TYPE_BOOLEAN,
    STOOLAP_TYPE_FLOAT, STOOLAP_TYPE_INTEGER, STOOLAP_TYPE_JSON, STOOLAP_TYPE_NULL,
    STOOLAP_TYPE_TEXT, STOOLAP_TYPE_TIMESTAMP,
};

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

        handle.row_gen += 1;
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
        Some(handle) => handle.columns.len() as i32,
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

    match handle.column_cstrs().get(index as usize) {
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
    let row_gen = handle.row_gen;
    let fresh = match handle.text_cache.get(idx) {
        Some((read_at, _)) => *read_at != row_gen,
        None => true,
    };
    if fresh {
        let Some(rows_inner) = &handle.rows else {
            return std::ptr::null();
        };
        let Some(val) = rows_inner.current_row().get(idx) else {
            return std::ptr::null();
        };
        if handle.text_cache.len() <= idx {
            let len = handle.columns.len().max(idx + 1);
            handle
                .text_cache
                .resize_with(len, || (u64::MAX, Vec::new()));
        }
        let (read_at, buf) = &mut handle.text_cache[idx];
        // Interior NULs are kept: out_len gives the full text, a C string
        // reader stops at the first one
        let mut fill = |s: &str| {
            buf.clear();
            buf.reserve(s.len() + 1);
            buf.extend_from_slice(s.as_bytes());
            buf.push(0);
        };
        if let Some(s) = val.as_str() {
            fill(s);
        } else if let Some(s) = val.as_string() {
            fill(&s);
        } else {
            return std::ptr::null();
        }
        *read_at = row_gen;
    }

    let (_, cached) = &handle.text_cache[idx];
    if !out_len.is_null() {
        // Length excludes the trailing NUL terminator
        *out_len = (cached.len() - 1) as i64;
    }
    cached.as_ptr() as *const c_char
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

/// Largest packed buffer a handle keeps for its next call
const PACKED_KEEP_MAX: usize = 1 << 20;

/// Empties a handle's packed buffer for reuse, releasing it instead when
/// an earlier large result left it above `PACKED_KEEP_MAX`
pub(crate) fn reset_packed(buf: &mut Vec<u8>) {
    if buf.capacity() > PACKED_KEEP_MAX {
        *buf = Vec::new();
    } else {
        buf.clear();
    }
}

/// Writes the column names and a zero row count, returning where the
/// row count sits so `write_rows` can patch it
fn write_header<'a>(names: impl ExactSizeIterator<Item = &'a [u8]>, buf: &mut Vec<u8>) -> usize {
    buf.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for name in names {
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name);
    }
    let count_at = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());
    count_at
}

/// Packs up to `max_rows` rows and patches the count at `count_at`. A
/// runtime error that ends the result early is returned, not packed.
fn write_rows(
    rows: &mut Rows,
    col_count: usize,
    max_rows: usize,
    count_at: usize,
    buf: &mut Vec<u8>,
) -> Result<u32> {
    let mut row_count: u32 = 0;
    while (row_count as usize) < max_rows && rows.advance() {
        let row = rows.current_row();
        for i in 0..col_count {
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
    if (row_count as usize) < max_rows {
        if let Some(err) = rows.error() {
            return Err(err);
        }
    }
    buf[count_at..count_at + 4].copy_from_slice(&row_count.to_le_bytes());
    Ok(row_count)
}

/// Packs every row of `rows` into `buf`, which is reset first
pub(crate) fn pack_all(rows: &mut Rows, buf: &mut Vec<u8>) -> Result<u32> {
    reset_packed(buf);
    let count_at = write_header(rows.columns().iter().map(|n| n.as_bytes()), buf);
    let col_count = rows.columns().len();
    write_rows(rows, col_count, usize::MAX, count_at, buf)
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
/// The rows handle is consumed and closed by this call. A runtime error
/// that ends the result early returns `STOOLAP_ERROR` and no buffer.
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
        let mut buf = Vec::with_capacity(4096);
        let count_at = write_header(handle.columns.iter().map(|n| n.as_bytes()), &mut buf);
        let col_count = handle.columns.len();
        let packed = match &mut handle.rows {
            Some(r) => write_rows(r, col_count, usize::MAX, count_at, &mut buf),
            None => Ok(0),
        };
        if let Err(e) = packed {
            handle.set_error(&e.to_string());
            return STOOLAP_ERROR;
        }

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

/// Fetch up to `max_rows` of the remaining rows into a packed buffer, in
/// the format of `stoolap_rows_fetch_all()`, and advance the cursor.
///
/// `*out_rows` receives the number of rows packed; 0 means the result is
/// exhausted. The buffer belongs to the rows handle and stays valid until
/// the next call on it or `stoolap_rows_close()`. Must NOT be freed. A
/// runtime error that ends the result returns `STOOLAP_ERROR`.
///
/// # Safety
///
/// `rows` must be a valid `StoolapRows` pointer. `out_buf`, `out_len` and
/// `out_rows` must not be NULL.
#[no_mangle]
pub unsafe extern "C" fn stoolap_rows_fetch_chunk(
    rows: *mut StoolapRows,
    max_rows: i32,
    out_buf: *mut *const u8,
    out_len: *mut i64,
    out_rows: *mut i32,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() || out_rows.is_null() {
        return STOOLAP_ERROR;
    }
    *out_buf = std::ptr::null();
    *out_len = 0;
    *out_rows = 0;

    let handle = match rows.as_mut() {
        Some(h) => h,
        None => return STOOLAP_ERROR,
    };
    if max_rows <= 0 {
        handle.set_error("max_rows must be positive");
        return STOOLAP_ERROR;
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        handle.has_row = false;
        reset_packed(&mut handle.packed);
        let names = handle.columns.iter().map(|n| n.as_bytes());
        let count_at = write_header(names, &mut handle.packed);
        let col_count = handle.columns.len();
        let max = max_rows as usize;
        let packed = match &mut handle.rows {
            Some(r) => write_rows(r, col_count, max, count_at, &mut handle.packed),
            None => Ok(0),
        };
        match packed {
            Ok(n) => {
                if (n as usize) < max {
                    handle.rows = None;
                }
                *out_buf = handle.packed.as_ptr();
                *out_len = handle.packed.len() as i64;
                *out_rows = n as i32;
                STOOLAP_OK
            }
            Err(e) => {
                handle.rows = None;
                handle.set_error(&e.to_string());
                STOOLAP_ERROR
            }
        }
    }));

    result.unwrap_or_else(|_| {
        handle.rows = None;
        handle.set_error("panic during stoolap_rows_fetch_chunk");
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
