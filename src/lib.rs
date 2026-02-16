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

//! # Stoolap - High-performance embedded SQL database
//!
//! Stoolap is a modern embedded SQL database written in pure Rust. It provides
//! full ACID transactions with MVCC, a sophisticated cost-based query optimizer,
//! and features that rival established databases like PostgreSQL and DuckDB.
//!
//! ## Key Features
//!
//! - **MVCC Transactions** - Full multi-version concurrency control with snapshot isolation
//! - **Multiple Index Types** - B-tree, Hash, and Bitmap indexes (auto-selected by data type)
//! - **Multi-Column Indexes** - Composite indexes for complex query patterns
//! - **Cost-Based Optimizer** - PostgreSQL-style optimizer with adaptive execution
//! - **Parallel Query Execution** - Automatic parallelization using Rayon
//! - **Semantic Query Caching** - Intelligent result caching with predicate subsumption
//! - **AS OF Temporal Queries** - Built-in time-travel queries
//! - **Window Functions** - ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, NTILE, etc.
//! - **CTEs** - Common Table Expressions including recursive CTEs
//! - **101+ Built-in Functions** - String, math, date/time, JSON, aggregate, window
//!
//! ## Quick Start
//!
//! ```rust
//! use stoolap::api::Database;
//!
//! // Open in-memory database
//! let db = Database::open_in_memory().unwrap();
//!
//! // Create a table
//! db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)", ()).unwrap();
//!
//! // Insert data
//! db.execute("INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25)", ()).unwrap();
//!
//! // Query with window function
//! let rows = db.query("
//!     SELECT name, age, RANK() OVER (ORDER BY age DESC) as rank
//!     FROM users
//! ", ()).unwrap();
//! ```
//!
//! ## Modules
//!
//! - [`api`] - Public database interface ([`api::Database`])
//! - [`core`] - Core types ([`DataType`], [`Value`], [`Row`], [`Schema`], [`Error`])
//! - [`storage`] - Storage layer with MVCC, indexes, and expressions
//! - [`parser`] - SQL parser
//! - [`functions`] - 101+ SQL functions (scalar, aggregate, window)
//! - [`executor`] - Query executor with parallel execution
//! - [`optimizer`] - Cost-based query optimizer with cardinality feedback
//! - [`common`] - Utilities (BufferPool, I64Map, version)

// Use mimalloc as global allocator when feature is enabled
// (but not when dhat-heap is enabled, as it needs its own allocator)
#[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod api;
pub mod common;
pub mod core;
pub mod executor;
pub mod functions;
pub mod optimizer;
pub mod parser;
pub mod storage;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export main types for convenience
pub use core::{
    DataType, Error, IndexEntry, IndexType, IsolationLevel, Operator, Result, Row, Schema,
    SchemaBuilder, SchemaColumn, Value,
};

// Re-export common utilities
pub use common::{BufferPool, I64Map, I64Set, PoolStats, SemVer, SmartString};

// Re-export storage/expression types
pub use storage::{
    AndExpr, BetweenExpr, CastExpr, ComparisonExpr, CompoundExpr, Expression, InListExpr, NotExpr,
    NullCheckExpr, OrExpr, RangeExpr,
};

// Re-export index types
pub use storage::{BTree, Int64BTree};

// Re-export config types
pub use storage::{Config, PersistenceConfig, SyncMode};

// Re-export storage traits
pub use storage::{
    EmptyResult, EmptyScanner, Engine, Index, MemoryResult, QueryResult, Scanner, Table,
    TemporalType, Transaction, VecScanner,
};

// Re-export MVCC types
pub use storage::{
    BTreeIndex, MVCCEngine, MVCCScanner, MVCCTable, MvccTransaction, RangeScanner, RowVersion,
    SingleRowScanner, TransactionEngineOperations, TransactionRegistry, TransactionState,
    TransactionVersionStore, VersionStore, VisibilityChecker, WriteSetEntry,
    INVALID_TRANSACTION_ID, RECOVERY_TRANSACTION_ID,
};

// Re-export WAL types
pub use storage::{
    CheckpointMetadata, WALEntry, WALManager, WALOperationType, DEFAULT_WAL_BUFFER_SIZE,
    DEFAULT_WAL_FLUSH_TRIGGER, DEFAULT_WAL_MAX_SIZE,
};

// Re-export Persistence types
pub use storage::{
    deserialize_row_version, deserialize_value, serialize_row_version, serialize_value,
    IndexMetadata, PersistenceManager, PersistenceMeta, DEFAULT_KEEP_SNAPSHOTS,
    DEFAULT_SNAPSHOT_INTERVAL,
};

// Re-export function types
pub use functions::{
    AggregateFunction, AvgFunction, CountFunction, FirstFunction, FunctionDataType, FunctionInfo,
    FunctionRegistry, FunctionSignature, FunctionType, LastFunction, MaxFunction, MinFunction,
    ScalarFunction, SumFunction, WindowFunction,
};

// Re-export specific function implementations
pub use functions::{
    AbsFunction, CeilingFunction, CoalesceFunction, ConcatFunction, DenseRankFunction,
    FloorFunction, IfNullFunction, LagFunction, LeadFunction, LengthFunction, LowerFunction,
    NowFunction, NtileFunction, NullIfFunction, RankFunction, RoundFunction, RowNumberFunction,
    SubstringFunction, UpperFunction,
};

// Re-export executor types
pub use executor::{
    CacheStats, CachedQueryPlan, ColumnStatsCache, ExecResult, ExecutionContext, Executor,
    ExecutorResult, QueryCache, QueryPlanner, StatsHealth,
};

// Re-export API types
pub use api::{
    Database, FromRow, FromValue, NamedParams, Params, ResultRow, Rows, Statement, ToParam,
    Transaction as ApiTransaction,
};

#[cfg(test)]
mod size_tests {
    #[test]
    fn check_ast_sizes() {
        use std::mem::size_of;
        println!("\n=== AST Type Sizes ===");
        println!("Token: {} bytes", size_of::<crate::parser::token::Token>());
        println!(
            "Identifier: {} bytes",
            size_of::<crate::parser::ast::Identifier>()
        );
        println!(
            "Expression: {} bytes",
            size_of::<crate::parser::ast::Expression>()
        );
        println!(
            "Statement: {} bytes",
            size_of::<crate::parser::ast::Statement>()
        );
    }
}
