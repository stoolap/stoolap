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

//! Query optimizer module for cost-based query planning
//!
//! This module provides cost estimation and optimization for query execution.
//! It uses statistics collected by ANALYZE to estimate the cost of different
//! access methods and choose the most efficient query plan.
//!
//! ## Modules
//!
//! - `cost` - Cost estimation for access methods and operations
//! - `join` - Join optimization and algorithm selection
//! - `feedback` - Cardinality feedback for improved estimates
//! - `aqe` - Adaptive Query Execution for runtime plan adaptation
//! - `bloom` - Runtime bloom filter propagation for join optimization
//! - `workload` - Workload learning and edge-aware optimization (Stoolap unique)
//! - `simplify` - Expression simplification and constant folding

pub mod aqe;
pub mod bloom;
pub mod cost;
pub mod feedback;
pub mod join;
pub mod simplify;
pub mod workload;

pub use cost::{
    AccessMethod, BuildSide, CostConstants, CostEstimator, JoinAlgorithm, JoinStats, PlanCost,
    DEFAULT_COST_CONSTANTS,
};

pub use feedback::{
    extract_column_from_predicate, fingerprint_predicate, global_feedback_cache,
    CardinalityFeedback, FeedbackCache, DEFAULT_DECAY_FACTOR, MAX_CORRECTION_FACTOR,
    MIN_CORRECTION_FACTOR, MIN_SAMPLE_COUNT,
};

pub use join::{JoinCondition, JoinNode, JoinOptimizer, JoinPlan, JoinStep, SortOrder};

pub use aqe::{
    decide_join_algorithm, AqeJoinDecision, AqeJoinResult, JoinAqeContext, AQE_SWITCH_THRESHOLD,
    MAX_ROWS_FOR_NESTED_LOOP, MIN_ROWS_FOR_HASH_JOIN,
};

pub use bloom::{BloomFilter, BloomFilterBuilder, RuntimeBloomFilter};

pub use workload::{
    global_workload_learner, EdgeAwarePlanner, EdgeJoinRecommendation, EdgeMode,
    IndexRecommendation, QueryPattern, TemporalPattern, WorkloadConfig, WorkloadHints,
    WorkloadLearner,
};

pub use simplify::{simplify_expression, simplify_expression_fixed_point, ExpressionSimplifier};
