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

//! C FFI layer for Stoolap
//!
//! Provides a C API with opaque handles, step-based iteration,
//! and per-handle error storage. Enable with `--features ffi`.
//!
//! Build: `cargo build --release --features ffi` produces `libstoolap.{so,dylib,dll}`

mod database;
mod error;
mod rows;
mod statement;
mod transaction;
mod types;
mod value;

// Re-export all extern "C" functions
pub use database::*;
pub use rows::*;
pub use statement::*;
pub use transaction::*;
pub use types::{
    StoolapBlobData, StoolapDB, StoolapNamedParam, StoolapRows, StoolapStmt, StoolapTextData,
    StoolapTx, StoolapValue, StoolapValueData,
};

// Status codes
pub const STOOLAP_OK: i32 = 0;
pub const STOOLAP_ERROR: i32 = 1;
pub const STOOLAP_ROW: i32 = 100;
pub const STOOLAP_DONE: i32 = 101;

// Value type codes (match DataType repr(u8))
pub const STOOLAP_TYPE_NULL: i32 = 0;
pub const STOOLAP_TYPE_INTEGER: i32 = 1;
pub const STOOLAP_TYPE_FLOAT: i32 = 2;
pub const STOOLAP_TYPE_TEXT: i32 = 3;
pub const STOOLAP_TYPE_BOOLEAN: i32 = 4;
pub const STOOLAP_TYPE_TIMESTAMP: i32 = 5;
pub const STOOLAP_TYPE_JSON: i32 = 6;
pub const STOOLAP_TYPE_BLOB: i32 = 7;

// Isolation levels (match IsolationLevel repr)
pub const STOOLAP_ISOLATION_READ_COMMITTED: i32 = 0;
pub const STOOLAP_ISOLATION_SNAPSHOT: i32 = 1;
