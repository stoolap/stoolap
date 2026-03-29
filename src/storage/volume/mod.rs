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

//! Frozen Volume Storage
//!
//! A frozen volume is an immutable, column-major storage unit for historical data.
//! It is the on-disk counterpart to the in-memory arena (hot buffer).
//!
//! # Architecture
//!
//! ```text
//! Table
//! ├── Hot Buffer (in-memory VersionStore + Arena)
//! │   └── Current writes, full MVCC, WAL-protected
//! │
//! ├── Volume 0 (frozen, on disk)
//! │   ├── Column data (typed arrays per column)
//! │   ├── String dictionary (deduplicated text values)
//! │   ├── Zone maps (min/max per column for pruning)
//! │   ├── Pre-computed aggregate stats
//! │   └── Visibility bitmap (for logical deletes)
//! │
//! ├── Volume 1 (frozen, on disk)
//! │   └── ...
//! │
//! └── Volume Catalog (tiny metadata file)
//! ```
//!
//! # Design Principles
//!
//! - **Column-major**: Each column stored contiguously for cache-friendly scans.
//!   A query reading 2 of 9 columns reads 22% of the data, not 100%.
//! - **Typed arrays**: Integer/Float/Timestamp columns stored as raw i64/f64 arrays.
//!   No per-value type tag, no enum overhead during scans.
//! - **Dictionary encoding**: Text columns store u32 IDs referencing a deduplicated
//!   string table. Repeated values (e.g., exchange names) use 4 bytes instead of N.
//! - **Zone maps**: Per-column min/max enables skipping entire volumes for range
//!   predicates. `WHERE time >= X` can skip volumes without reading any data.
//! - **Pre-computed stats**: COUNT, SUM, MIN, MAX, AVG stored per column at freeze
//!   time. Common aggregation queries return instantly with zero row scanning.
//! - **Binary search**: Sorted columns (typically time) support O(log n) range
//!   lookups instead of linear scan.
//! - **Platform independent**: The format is a byte layout. Access methods
//!   (mmap on native, buffered I/O, or Vec<u8> on WASM) are behind a trait.
//! - **Backward compatible**: Memory mode uses no volumes. Persistence mode
//!   uses hot buffer + frozen volumes. The executor sees the same Table/Scanner
//!   traits regardless.

pub mod column;
pub mod format;
pub mod io;
pub mod manifest;
pub mod scanner;
pub mod seal;
pub mod stats;
pub mod table;
pub mod writer;
