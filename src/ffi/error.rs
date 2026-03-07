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

//! Error storage for FFI: global error for `stoolap_open` failures.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

// Thread-local error for stoolap_open() failures (before a handle exists)
// and for stoolap_tx_commit/stoolap_tx_rollback failures (handle consumed).
// Thread-local storage ensures the returned pointer remains valid until the
// next API call on the same thread, without cross-thread races.
thread_local! {
    static THREAD_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub(crate) fn set_global_error(msg: &str) {
    THREAD_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

pub(crate) fn global_error_ptr() -> *const c_char {
    THREAD_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match &*borrow {
            Some(cs) => cs.as_ptr(),
            None => empty_cstr(),
        }
    })
}

/// Returns a pointer to a static empty C string.
pub(crate) fn empty_cstr() -> *const c_char {
    // SAFETY: b"\0" is a valid null-terminated C string.
    static EMPTY: &[u8] = b"\0";
    EMPTY.as_ptr() as *const c_char
}
