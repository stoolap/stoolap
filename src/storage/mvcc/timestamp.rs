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

//! Fast timestamp generation for MVCC
//!
//! Provides monotonically increasing timestamps suitable for transaction
//! ordering and version tracking, even under heavy concurrent usage.
//!

use std::sync::atomic::{AtomicI64, Ordering};

/// The order transaction-local writes are made in, which savepoints and
/// statement rollback cut at: a counter, so a write reads no clock
static WRITE_ORDER: AtomicI64 = AtomicI64::new(0);

/// The order of the next transaction-local write
pub(crate) fn next_write_order() -> i64 {
    WRITE_ORDER.fetch_add(1, Ordering::AcqRel) + 1
}

/// The order of the latest transaction-local write: the boundary a savepoint
/// or statement rollback takes
pub fn last_write_order() -> i64 {
    WRITE_ORDER.load(Ordering::Acquire)
}
