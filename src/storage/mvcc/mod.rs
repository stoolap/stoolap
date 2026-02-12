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

//! MVCC (Multi-Version Concurrency Control) storage engine
//!
//! This module provides the core MVCC implementation for Stoolap, including:
//!
//! - [`TransactionRegistry`] - Transaction state management and visibility
//! - [`get_fast_timestamp`] - Fast monotonic timestamp generation
//! - [`VersionStore`] - Row version storage and visibility checking
//! - [`TransactionVersionStore`] - Transaction-local changes before commit
//!
//! # Architecture
//!
//! The MVCC engine supports:
//! - **READ COMMITTED** isolation: Sees only committed data at read time
//! - **SNAPSHOT isolation**: Sees consistent snapshot from transaction start
//!
//! # Transaction Lifecycle
//!
//! ```text
//! Begin -> Active -> [Committing] -> Committed
//!                 \-> Aborted
//! ```
//!

pub mod arena;
pub mod bitmap_index;
pub mod btree_index;
pub mod engine;
pub mod file_lock;
pub mod hash_index;
pub mod multi_column_index;
pub mod persistence;
pub mod pk_index;
pub mod registry;
pub mod scanner;
pub mod snapshot;
pub mod streaming_result;
pub mod table;
pub mod timestamp;
pub mod transaction;
pub mod version_store;
pub mod wal_manager;
pub mod zonemap;

// Re-export main types
pub use bitmap_index::BitmapIndex;
pub use btree_index::{
    intersect_multiple_sorted_ids, intersect_sorted_ids, union_multiple_sorted_ids,
    union_sorted_ids, BTreeIndex,
};
pub use engine::{CleanupHandle, MVCCEngine};
pub use hash_index::HashIndex;
pub use multi_column_index::{CompositeKey, MultiColumnIndex};
pub use persistence::{
    deserialize_row_version, deserialize_value, serialize_row_version, serialize_value,
    IndexMetadata, PersistenceManager, PersistenceMeta, DEFAULT_KEEP_SNAPSHOTS,
    DEFAULT_SNAPSHOT_INTERVAL,
};
pub use pk_index::PkIndex;
pub use registry::{TransactionRegistry, INVALID_TRANSACTION_ID, RECOVERY_TRANSACTION_ID};
pub use scanner::{EmptyScanner, MVCCScanner, RangeScanner, SingleRowScanner};
pub use snapshot::{DiskVersionStore, SnapshotReader, SnapshotWriter};
pub use streaming_result::{AggregationScanner, StreamingResult, VisibleRowInfo};
pub use table::MVCCTable;
pub use timestamp::get_fast_timestamp;
pub use transaction::{MvccTransaction, TransactionEngineOperations, TransactionState};
pub use version_store::{
    clear_version_map_pools, AggregateOp, AggregateResult, RowIndex, RowVersion,
    TransactionVersionStore, VersionStore, VisibilityChecker, WriteSetEntry,
};
pub use wal_manager::{
    CheckpointMetadata, WALEntry, WALManager, WALOperationType, DEFAULT_WAL_BUFFER_SIZE,
    DEFAULT_WAL_FLUSH_TRIGGER, DEFAULT_WAL_MAX_SIZE,
};
pub use zonemap::{
    ColumnZoneMap, PruneStats, TableZoneMap, ZoneMapBuilder, ZoneMapEntry, DEFAULT_SEGMENT_SIZE,
};
