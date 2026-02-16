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

//! Time compatibility shim for WASM targets
//!
//! On native targets, re-exports `std::time::Instant` and `std::time::SystemTime`.
//! On WASM targets, uses `web_time::Instant` and `web_time::SystemTime` which
//! delegate to `performance.now()` and `Date.now()` via web-sys.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::SystemTime;

#[cfg(target_arch = "wasm32")]
pub use web_time::SystemTime;

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::UNIX_EPOCH;

#[cfg(target_arch = "wasm32")]
pub use web_time::UNIX_EPOCH;
