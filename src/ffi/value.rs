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

//! Conversion between FFI `StoolapValue` and Rust `Value`.

use crate::api::ParamVec;
use crate::core::{Error, Result, Value};

use super::types::{StoolapNamedParam, StoolapValue};
use super::{
    STOOLAP_TYPE_BLOB, STOOLAP_TYPE_BOOLEAN, STOOLAP_TYPE_FLOAT, STOOLAP_TYPE_INTEGER,
    STOOLAP_TYPE_JSON, STOOLAP_TYPE_NULL, STOOLAP_TYPE_TEXT, STOOLAP_TYPE_TIMESTAMP,
};

/// Convert a C `StoolapValue` to a Rust `Value`. Text that is not UTF-8
/// and JSON that does not parse are refused, not read as NULL.
///
/// # Safety
///
/// The caller must ensure text/blob pointers are valid for the specified length.
pub(crate) unsafe fn ffi_value_to_rust(v: &StoolapValue) -> Result<Value> {
    Ok(match v.value_type {
        STOOLAP_TYPE_NULL => Value::null_unknown(),
        STOOLAP_TYPE_INTEGER => Value::Integer(v.v.integer),
        STOOLAP_TYPE_FLOAT => Value::Float(v.v.float64),
        STOOLAP_TYPE_TEXT => {
            let text = &v.v.text;
            if text.ptr.is_null() || text.len < 0 {
                Value::null_unknown()
            } else if text.len == 0 {
                Value::text("")
            } else {
                let slice = std::slice::from_raw_parts(text.ptr as *const u8, text.len as usize);
                match std::str::from_utf8(slice) {
                    Ok(s) => Value::text(s),
                    Err(_) => {
                        return Err(Error::invalid_argument("text parameter is not valid UTF-8"))
                    }
                }
            }
        }
        STOOLAP_TYPE_BOOLEAN => Value::Boolean(v.v.boolean != 0),
        STOOLAP_TYPE_TIMESTAMP => {
            let nanos = v.v.timestamp_nanos;
            let secs = nanos.div_euclid(1_000_000_000);
            let nsecs = nanos.rem_euclid(1_000_000_000) as u32;
            match chrono::DateTime::from_timestamp(secs, nsecs) {
                Some(dt) => Value::Timestamp(dt),
                None => Value::null_unknown(),
            }
        }
        STOOLAP_TYPE_JSON => {
            let text = &v.v.text;
            if text.ptr.is_null() || text.len < 0 {
                Value::null_unknown()
            } else {
                let slice = std::slice::from_raw_parts(text.ptr as *const u8, text.len as usize);
                // The parser the JSON functions read with, so a value bound
                // here is one they can read
                match std::str::from_utf8(slice) {
                    Ok(s) if serde_json::from_str::<serde_json::Value>(s).is_ok() => Value::json(s),
                    _ => return Err(Error::invalid_argument("JSON parameter is not valid JSON")),
                }
            }
        }
        STOOLAP_TYPE_BLOB => {
            let blob = &v.v.blob;
            if blob.ptr.is_null() || blob.len < 0 {
                Value::null_unknown()
            } else if blob.len == 0 {
                // Empty vector (zero dimensions)
                Value::vector(vec![])
            } else {
                let len = blob.len as usize;
                if !len.is_multiple_of(4) {
                    // Vector data must be packed f32 (4 bytes each)
                    Value::null_unknown()
                } else {
                    let slice = std::slice::from_raw_parts(blob.ptr, len);
                    Value::vector_from_bytes(slice.into())
                }
            }
        }
        _ => Value::null_unknown(),
    })
}

/// Convert an FFI parameter array to a Rust `ParamVec`.
///
/// # Safety
///
/// `params` must point to `len` valid `StoolapValue` structs (or be null if `len <= 0`).
pub(crate) unsafe fn params_to_vec(params: *const StoolapValue, len: i32) -> Result<ParamVec> {
    if params.is_null() || len <= 0 {
        return Ok(ParamVec::new());
    }
    let slice = std::slice::from_raw_parts(params, len as usize);
    slice.iter().map(|v| ffi_value_to_rust(v)).collect()
}

/// Reads an FFI named-parameter array by name, converting only the values a
/// statement binds; a name given twice reads its last value.
///
/// # Safety
///
/// `params` must point to `len` valid `StoolapNamedParam` structs (or be null if `len <= 0`).
/// Each `name` pointer must be valid for `name_len` bytes.
pub(crate) unsafe fn ffi_value_of<'a>(
    params: *const StoolapNamedParam,
    len: i32,
) -> impl FnMut(&str) -> Result<Option<Value>> + 'a {
    let slice: &'a [StoolapNamedParam] = if params.is_null() || len <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(params, len as usize)
    };
    move |name| {
        // SAFETY: the caller keeps every name valid for `name_len` bytes and
        // every value valid while the statement binds
        unsafe {
            let given = slice.iter().rev().find(|p| {
                !p.name.is_null()
                    && p.name_len > 0
                    && std::slice::from_raw_parts(p.name as *const u8, p.name_len as usize)
                        == name.as_bytes()
            });
            given.map(|p| ffi_value_to_rust(&p.value)).transpose()
        }
    }
}
