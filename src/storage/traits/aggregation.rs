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

//! Aggregation types shared across storage backends

use crate::core::Value;

/// Aggregate operation type for deferred aggregation
///
/// Used with `compute_aggregates()` to perform multiple aggregations in a single pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregateOp {
    Count,
    CountStar,
    Sum,
    Min,
    Max,
    Avg,
}

/// Result of storage-level grouped aggregation
#[derive(Debug, Clone)]
pub struct GroupedAggregateResult {
    /// Group key values
    pub group_values: Vec<Value>,
    /// Aggregate results in order of requested aggregates
    pub aggregate_values: Vec<Value>,
}
