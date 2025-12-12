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
pub use registry::{TransactionRegistry, INVALID_TRANSACTION_ID, RECOVERY_TRANSACTION_ID};
pub use scanner::{EmptyScanner, MVCCScanner, RangeScanner, SingleRowScanner};
pub use snapshot::{DiskVersionStore, SnapshotReader, SnapshotWriter};
pub use streaming_result::{AggregationScanner, StreamingResult, VisibleRowInfo};
pub use table::MVCCTable;
pub use timestamp::get_fast_timestamp;
pub use transaction::{MvccTransaction, TransactionEngineOperations, TransactionState};
pub use version_store::{
    RowVersion, TransactionVersionStore, VersionStore, VisibilityChecker, WriteSetEntry,
};
pub use wal_manager::{
    CheckpointMetadata, WALEntry, WALManager, WALOperationType, DEFAULT_WAL_BUFFER_SIZE,
    DEFAULT_WAL_FLUSH_TRIGGER, DEFAULT_WAL_MAX_SIZE,
};
pub use zonemap::{
    ColumnZoneMap, PruneStats, TableZoneMap, ZoneMapBuilder, ZoneMapEntry, DEFAULT_SEGMENT_SIZE,
};

use thiserror::Error;

/// MVCC-specific errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum MvccError {
    /// No transaction provided for operation
    #[error("no transaction provided")]
    NoTransaction,

    /// Primary key violation (duplicate key)
    #[error("primary key violation: duplicate key")]
    PrimaryKeyViolation,

    /// Invalid or unknown table
    #[error("invalid or unknown table")]
    InvalidTable,

    /// Invalid row data
    #[error("invalid row data")]
    InvalidRow,

    /// Transaction is closed
    #[error("transaction already closed")]
    TransactionClosed,

    /// Engine is not open
    #[error("engine is not open")]
    EngineNotOpen,

    /// Engine is already open
    #[error("engine is already open")]
    EngineAlreadyOpen,

    /// Registry not accepting new transactions
    #[error("transaction registry is not accepting new transactions")]
    NotAcceptingTransactions,

    /// Version not found
    #[error("version not found for row {0}")]
    VersionNotFound(i64),

    /// Write conflict detected
    #[error("write conflict detected")]
    WriteConflict,

    /// Serialization error
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Persistence error
    #[error("persistence error: {0}")]
    Persistence(String),
}

impl From<MvccError> for crate::core::Error {
    fn from(err: MvccError) -> Self {
        match err {
            MvccError::NoTransaction => crate::core::Error::internal("no transaction provided"),
            MvccError::PrimaryKeyViolation => crate::core::Error::primary_key_constraint(0),
            MvccError::InvalidTable => crate::core::Error::TableNotFound,
            MvccError::InvalidRow => crate::core::Error::internal("invalid row data"),
            MvccError::TransactionClosed => {
                crate::core::Error::internal("transaction already closed")
            }
            MvccError::EngineNotOpen => crate::core::Error::EngineNotOpen,
            MvccError::EngineAlreadyOpen => crate::core::Error::EngineAlreadyOpen,
            MvccError::NotAcceptingTransactions => {
                crate::core::Error::internal("not accepting new transactions")
            }
            MvccError::VersionNotFound(row_id) => {
                crate::core::Error::internal(format!("version not found for row {}", row_id))
            }
            MvccError::WriteConflict => crate::core::Error::internal("write conflict detected"),
            MvccError::Serialization(msg) => {
                crate::core::Error::internal(format!("serialization error: {}", msg))
            }
            MvccError::Persistence(msg) => {
                crate::core::Error::internal(format!("persistence error: {}", msg))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvcc_error_display() {
        assert_eq!(
            MvccError::NoTransaction.to_string(),
            "no transaction provided"
        );
        assert_eq!(
            MvccError::PrimaryKeyViolation.to_string(),
            "primary key violation: duplicate key"
        );
        assert_eq!(
            MvccError::TransactionClosed.to_string(),
            "transaction already closed"
        );
    }

    #[test]
    fn test_mvcc_error_to_core_error() {
        let err: crate::core::Error = MvccError::InvalidTable.into();
        assert!(matches!(err, crate::core::Error::TableNotFound));

        let err: crate::core::Error = MvccError::PrimaryKeyViolation.into();
        assert!(matches!(
            err,
            crate::core::Error::PrimaryKeyConstraint { .. }
        ));
    }
}
