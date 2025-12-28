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

//! Query operators for streaming execution.
//!
//! This module provides Volcano-style operators that implement streaming
//! execution for SQL queries. Each operator implements the `Operator` trait
//! with `open()`, `next()`, `close()` lifecycle.
//!
//! # Available Operators
//!
//! ## Join Operators
//!
//! - `HashJoinOperator` - Streaming hash join with O(N+M) complexity
//! - `MergeJoinOperator` - Merge join for pre-sorted inputs with O(N+M) complexity
//! - `NestedLoopJoinOperator` - Fallback for complex conditions with O(N*M) complexity
//! - `IndexNestedLoopJoinOperator` - Index-based join with O(N*log M) complexity
//! - `BatchIndexNestedLoopJoinOperator` - Batch variant for NO LIMIT queries
//!
//! # Algorithm Selection
//!
//! | Condition | Recommended Operator |
//! |-----------|---------------------|
//! | Equality keys, large tables | `HashJoinOperator` |
//! | Both inputs pre-sorted | `MergeJoinOperator` |
//! | Inner table has index | `IndexNestedLoopJoinOperator` |
//! | No LIMIT + index available | `BatchIndexNestedLoopJoinOperator` |
//! | Non-equality conditions | `NestedLoopJoinOperator` |
//! | CROSS JOIN | `NestedLoopJoinOperator` |
//!
//! # Note
//!
//! For filtering and projection, use the Result Wrapper pattern:
//! - `FilteredResult` with `RowFilter` for WHERE clauses
//! - `StreamingProjectionResult` for column projection
//!
//! See `executor/result.rs` and `expression/evaluator_bridge.rs` for the
//! recommended execution model.

pub mod bloom_filter;
pub mod hash_join;
pub mod index_nested_loop;
pub mod merge_join;
pub mod nested_loop_join;

// Re-export all operators and types
pub use bloom_filter::BloomFilterOperator;
pub use hash_join::{HashJoinOperator, JoinSide, JoinType};
pub use index_nested_loop::{
    BatchIndexNestedLoopJoinOperator, IndexLookupStrategy, IndexNestedLoopJoinOperator,
};
pub use merge_join::MergeJoinOperator;
pub use nested_loop_join::NestedLoopJoinOperator;
