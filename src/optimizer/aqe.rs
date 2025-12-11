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

//! Adaptive Query Execution (AQE)
//!
//! This module implements runtime query plan adaptation based on actual
//! cardinalities observed during execution. When materialization reveals
//! significantly different cardinalities than estimated, AQE can:
//!
//! - Switch join algorithms (e.g., from nested loop to hash join)
//! - Swap join build/probe sides for better performance
//! - Record feedback to improve future estimates
//!
//! ## Decision Points
//!
//! AQE makes decisions at materialization boundaries:
//! - Before join execution (after inputs are materialized)
//! - Before aggregation (after grouping input is materialized)
//!
//! ## Switching Threshold
//!
//! By default, AQE switches strategy when actual cardinality differs from
//! estimate by more than 10x. This threshold balances adaptation benefits
//! against switching overhead.

use super::{BuildSide, JoinAlgorithm};

/// Threshold for estimation error before switching (10x)
pub const AQE_SWITCH_THRESHOLD: f64 = 10.0;

/// Minimum rows to consider hash join (below this, nested loop is fine)
pub const MIN_ROWS_FOR_HASH_JOIN: u64 = 100;

/// Maximum rows for nested loop join (above this, always prefer hash)
pub const MAX_ROWS_FOR_NESTED_LOOP: u64 = 1000;

/// Decision made by AQE for join execution
#[derive(Debug, Clone, PartialEq)]
pub enum AqeJoinDecision {
    /// Use the originally planned algorithm
    KeepPlanned,
    /// Switch to hash join with specified build side
    SwitchToHashJoin { build_side: BuildSide },
    /// Switch to nested loop join
    SwitchToNestedLoop,
}

/// AQE decision context for a join
#[derive(Debug, Clone)]
pub struct JoinAqeContext {
    /// Estimated left input rows
    pub estimated_left: u64,
    /// Estimated right input rows
    pub estimated_right: u64,
    /// Actual left input rows after materialization
    pub actual_left: u64,
    /// Actual right input rows after materialization
    pub actual_right: u64,
    /// Whether there are equality keys (required for hash join)
    pub has_equality_keys: bool,
    /// Original planned algorithm
    pub planned_algorithm: JoinAlgorithm,
}

impl JoinAqeContext {
    /// Create a new AQE context for a join
    pub fn new(
        estimated_left: u64,
        estimated_right: u64,
        actual_left: u64,
        actual_right: u64,
        has_equality_keys: bool,
        planned_algorithm: JoinAlgorithm,
    ) -> Self {
        Self {
            estimated_left,
            estimated_right,
            actual_left,
            actual_right,
            has_equality_keys,
            planned_algorithm,
        }
    }

    /// Calculate estimation error ratio (max of left and right errors)
    pub fn estimation_error(&self) -> f64 {
        let left_error = if self.estimated_left > 0 {
            self.actual_left as f64 / self.estimated_left as f64
        } else if self.actual_left > 0 {
            f64::MAX
        } else {
            1.0
        };

        let right_error = if self.estimated_right > 0 {
            self.actual_right as f64 / self.estimated_right as f64
        } else if self.actual_right > 0 {
            f64::MAX
        } else {
            1.0
        };

        // Return the larger deviation from 1.0
        let left_deviation = if left_error > 1.0 {
            left_error
        } else {
            1.0 / left_error
        };
        let right_deviation = if right_error > 1.0 {
            right_error
        } else {
            1.0 / right_error
        };

        left_deviation.max(right_deviation)
    }

    /// Check if estimation error exceeds threshold
    pub fn should_switch(&self) -> bool {
        self.estimation_error() > AQE_SWITCH_THRESHOLD
    }
}

/// Make AQE decision for join execution
///
/// This function decides whether to keep the planned join algorithm or
/// switch to a different one based on actual cardinalities.
///
/// # Arguments
///
/// * `ctx` - AQE context with estimated and actual cardinalities
///
/// # Returns
///
/// Decision indicating whether to keep planned algorithm or switch
pub fn decide_join_algorithm(ctx: &JoinAqeContext) -> AqeJoinDecision {
    // If estimation is within threshold, keep planned
    if !ctx.should_switch() {
        return AqeJoinDecision::KeepPlanned;
    }

    let total_rows = ctx.actual_left + ctx.actual_right;
    let cross_product = ctx.actual_left as u128 * ctx.actual_right as u128;

    // Very small inputs - nested loop is fine
    if ctx.actual_left <= MIN_ROWS_FOR_HASH_JOIN && ctx.actual_right <= MIN_ROWS_FOR_HASH_JOIN {
        return match &ctx.planned_algorithm {
            JoinAlgorithm::NestedLoop { .. } => AqeJoinDecision::KeepPlanned,
            _ => AqeJoinDecision::SwitchToNestedLoop,
        };
    }

    // No equality keys - must use nested loop
    if !ctx.has_equality_keys {
        return match &ctx.planned_algorithm {
            JoinAlgorithm::NestedLoop { .. } => AqeJoinDecision::KeepPlanned,
            _ => AqeJoinDecision::SwitchToNestedLoop,
        };
    }

    // Large cross product - definitely use hash join
    if cross_product > (MAX_ROWS_FOR_NESTED_LOOP as u128 * MAX_ROWS_FOR_NESTED_LOOP as u128) {
        // Choose build side based on actual sizes
        let build_side = if ctx.actual_left <= ctx.actual_right {
            BuildSide::Left
        } else {
            BuildSide::Right
        };

        return match &ctx.planned_algorithm {
            JoinAlgorithm::HashJoin { build_side: bs, .. } if *bs == build_side => {
                AqeJoinDecision::KeepPlanned
            }
            _ => AqeJoinDecision::SwitchToHashJoin { build_side },
        };
    }

    // Medium sized inputs - check if hash join makes sense
    if total_rows > MIN_ROWS_FOR_HASH_JOIN {
        let build_side = if ctx.actual_left <= ctx.actual_right {
            BuildSide::Left
        } else {
            BuildSide::Right
        };

        // If we're already on hash join with optimal build side, keep it
        if let JoinAlgorithm::HashJoin { build_side: bs, .. } = &ctx.planned_algorithm {
            if *bs == build_side {
                return AqeJoinDecision::KeepPlanned;
            }
            // Switch build side
            return AqeJoinDecision::SwitchToHashJoin { build_side };
        }

        // Currently on nested loop but hash might be better
        return AqeJoinDecision::SwitchToHashJoin { build_side };
    }

    // Default: keep planned
    AqeJoinDecision::KeepPlanned
}

/// Result of AQE join execution
#[derive(Debug)]
pub struct AqeJoinResult {
    /// Whether algorithm was switched
    pub switched: bool,
    /// Original algorithm (for feedback)
    pub original_algorithm: JoinAlgorithm,
    /// Algorithm actually used
    pub used_algorithm: JoinAlgorithm,
    /// Actual left input rows
    pub actual_left_rows: u64,
    /// Actual right input rows
    pub actual_right_rows: u64,
}

impl AqeJoinResult {
    /// Check if this result indicates a significant estimation error
    pub fn has_significant_error(&self, estimated_left: u64, estimated_right: u64) -> bool {
        let ctx = JoinAqeContext::new(
            estimated_left,
            estimated_right,
            self.actual_left_rows,
            self.actual_right_rows,
            true, // doesn't matter for error calculation
            self.original_algorithm.clone(),
        );
        ctx.should_switch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimation_error_calculation() {
        // Estimate 100, actual 1000 → 10x error
        let ctx = JoinAqeContext::new(
            100,
            100,
            1000,
            100,
            true,
            JoinAlgorithm::NestedLoop {
                outer_rows: 100,
                inner_rows: 100,
            },
        );
        assert!((ctx.estimation_error() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_should_switch_above_threshold() {
        let ctx = JoinAqeContext::new(
            100,
            100,
            1100, // 11x error
            100,
            true,
            JoinAlgorithm::NestedLoop {
                outer_rows: 100,
                inner_rows: 100,
            },
        );
        assert!(ctx.should_switch());
    }

    #[test]
    fn test_should_not_switch_below_threshold() {
        let ctx = JoinAqeContext::new(
            100,
            100,
            500, // 5x error, below threshold
            100,
            true,
            JoinAlgorithm::NestedLoop {
                outer_rows: 100,
                inner_rows: 100,
            },
        );
        assert!(!ctx.should_switch());
    }

    #[test]
    fn test_small_inputs_prefer_nested_loop() {
        let ctx = JoinAqeContext::new(
            1000, // estimated larger
            1000,
            50, // actual small
            50,
            true,
            JoinAlgorithm::HashJoin {
                build_side: BuildSide::Left,
                build_rows: 1000,
                probe_rows: 1000,
            },
        );

        // 20x error should trigger switch
        assert!(ctx.should_switch());

        let decision = decide_join_algorithm(&ctx);
        assert_eq!(decision, AqeJoinDecision::SwitchToNestedLoop);
    }

    #[test]
    fn test_large_inputs_prefer_hash_join() {
        let ctx = JoinAqeContext::new(
            100, // estimated small
            100,
            10000, // actual large
            10000,
            true,
            JoinAlgorithm::NestedLoop {
                outer_rows: 100,
                inner_rows: 100,
            },
        );

        let decision = decide_join_algorithm(&ctx);
        match decision {
            AqeJoinDecision::SwitchToHashJoin { .. } => (),
            _ => panic!("Expected switch to hash join"),
        }
    }

    #[test]
    fn test_no_equality_keys_must_use_nested_loop() {
        let ctx = JoinAqeContext::new(
            100,
            100,
            10000,
            10000,
            false, // No equality keys
            JoinAlgorithm::NestedLoop {
                outer_rows: 100,
                inner_rows: 100,
            },
        );

        let decision = decide_join_algorithm(&ctx);
        // Even with large inputs, must use nested loop without equality keys
        assert!(matches!(
            decision,
            AqeJoinDecision::KeepPlanned | AqeJoinDecision::SwitchToNestedLoop
        ));
    }

    #[test]
    fn test_hash_join_build_side_optimization() {
        // Left is smaller, but we're building on right
        let ctx = JoinAqeContext::new(
            1000,
            100,  // estimated: right smaller
            500,  // actual: left smaller
            5000, // actual: right larger
            true,
            JoinAlgorithm::HashJoin {
                build_side: BuildSide::Right, // Wrong side now
                build_rows: 100,
                probe_rows: 1000,
            },
        );

        // Should switch build side
        let decision = decide_join_algorithm(&ctx);
        assert_eq!(
            decision,
            AqeJoinDecision::SwitchToHashJoin {
                build_side: BuildSide::Left
            }
        );
    }
}
