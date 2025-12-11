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

//! Storage module for Stoolap
//!
//! This module contains the storage layer components including:
//! - Expression system for query filtering
//! - Index structures for efficient lookups
//! - Storage traits (Engine, Transaction, Table, Index, Result, Scanner)
//! - Configuration types
//! - MVCC (Multi-Version Concurrency Control) engine

pub mod config;
pub mod expression;
pub mod index;
pub mod mvcc;
pub mod statistics;
pub mod traits;

// Re-export main expression types at storage level for convenience
pub use expression::{
    AndExpr, BetweenExpr, CastExpr, ComparisonExpr, CompoundExpr, Expression, InListExpr, NotExpr,
    NullCheckExpr, OrExpr, RangeExpr,
};

// Re-export index types
pub use index::{BTree, Int64BTree};

// Re-export config types
pub use config::{Config, PersistenceConfig, SyncMode};

// Re-export trait types
pub use traits::{
    EmptyResult, EmptyScanner, Engine, Index, MemoryResult, QueryResult, Scanner, Table,
    TemporalType, Transaction, VecScanner,
};

// Re-export MVCC types
pub use mvcc::{
    get_fast_timestamp, BTreeIndex, EmptyScanner as MvccEmptyScanner, MVCCEngine, MVCCScanner,
    MVCCTable, MvccError, MvccTransaction, RangeScanner, RowVersion, SingleRowScanner,
    TransactionEngineOperations, TransactionRegistry, TransactionState, TransactionVersionStore,
    VersionStore, VisibilityChecker, WriteSetEntry, INVALID_TRANSACTION_ID,
    RECOVERY_TRANSACTION_ID,
};

// Re-export WAL types
pub use mvcc::{
    CheckpointMetadata, WALEntry, WALManager, WALOperationType, DEFAULT_WAL_BUFFER_SIZE,
    DEFAULT_WAL_FLUSH_TRIGGER, DEFAULT_WAL_MAX_SIZE,
};

// Re-export Persistence types
pub use mvcc::{
    deserialize_row_version, deserialize_value, serialize_row_version, serialize_value,
    IndexMetadata, PersistenceManager, PersistenceMeta, DEFAULT_KEEP_SNAPSHOTS,
    DEFAULT_SNAPSHOT_INTERVAL,
};

// Re-export Zone Map types
pub use mvcc::{
    ColumnZoneMap, PruneStats, TableZoneMap, ZoneMapBuilder, ZoneMapEntry, DEFAULT_SEGMENT_SIZE,
};

// Re-export Statistics types
pub use statistics::{
    is_stats_table, ColumnCorrelations, ColumnStats, Histogram, HistogramOp, SelectivityEstimator,
    TableStats, CREATE_COLUMN_STATS_SQL, CREATE_TABLE_STATS_SQL, DEFAULT_HISTOGRAM_BUCKETS,
    DEFAULT_SAMPLE_SIZE, SYS_COLUMN_STATS, SYS_TABLE_STATS,
};
