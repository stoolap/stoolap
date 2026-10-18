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

//! Modern streaming JOIN executor using Volcano-style operators.
//!
//! This module provides high-performance JOIN execution with:
//! - **Hash Join**: Build smaller side, probe larger side with O(N+M) complexity
//! - **Nested Loop**: O(N*M) fallback for non-equality joins or small tables
//! - **Early Termination**: LIMIT stops execution immediately
//! - **Residual Filters**: Non-equality conditions applied during streaming
//!
//! # Architecture
//!
//! ```text
//! JoinRequest
//!     │
//!     ▼
//! ┌─────────────────────────────────┐
//! │ JoinExecutor::execute()         │
//! │  1. Analyze join condition      │
//! │  2. Select optimal algorithm    │
//! │  3. Execute with streaming      │
//! │  4. Early terminate at LIMIT    │
//! └─────────────────────────────────┘
//!     │
//!     ▼
//! JoinResult { rows, columns }
//! ```
//!
//! # Design Decisions & Tradeoffs
//!
//! ## Hybrid Execution: Streaming vs Parallel
//!
//! This executor uses a **hybrid approach** that dynamically chooses between
//! Volcano-style streaming and parallel bulk processing based on query characteristics:
//!
//! ### Streaming Volcano Path (default for small datasets or any LIMIT)
//! - **When**: Build side < 10,000 rows OR a LIMIT
//! - **Benefits**:
//!   - O(1) memory for probe side (streaming, not materialized)
//!   - Early termination: LIMIT 10 stops after 10 rows
//!   - Low latency to first row (important for interactive queries)
//!   - Composable operators (Filter → Join → Project → Limit)
//!
//! ### Parallel Hash Join Path (for large analytical queries)
//! - **When**: Build side ≥ 10,000 rows AND no LIMIT
//! - **Benefits**:
//!   - Parallel hash build into partitioned maps, one thread per partition
//!   - Parallel probe with Rayon work-stealing
//!   - 2-4x speedup on multi-core systems for large joins
//!   - Atomic tracking for OUTER join unmatched rows
//!
//! ### Why Not Always Parallel?
//! Parallel execution has overhead (task scheduling, synchronization). For:
//! - Small datasets: overhead exceeds benefit
//! - A LIMIT: streaming stops early; parallel would build every match first
//!
//! ## Bloom Filter Optimization (Not Yet Implemented)
//!
//! Bloom filters can accelerate hash joins by filtering probe rows that
//! definitely won't match before touching the hash table. This is particularly
//! effective for:
//! - High selectivity joins (few matches relative to probe size)
//! - Multi-way joins (filter cascades through the plan)
//!
//! The optimizer has bloom filter propagation logic, but runtime bloom filter
//! checks are not yet integrated into the streaming operators.

use crate::common::CompactArc;
use crate::core::{Result, Row, RowVec, Schema};
use crate::executor::context::ExecutionContext;
use crate::executor::expression::{JoinFilter, RowFilter};
use crate::executor::hash_table::JoinHashTable;
use crate::executor::operator::{ColumnInfo, MaterializedOperator, Operator};
use crate::executor::operators::hash_join::{HashJoinOperator, JoinSide, JoinType};
use crate::executor::operators::nested_loop_join::NestedLoopJoinOperator;
use crate::executor::parallel::{
    parallel_hash_join, ParallelConfig, DEFAULT_PARALLEL_JOIN_THRESHOLD,
};
use crate::executor::planner::{QueryPlanner, RuntimeJoinAlgorithm, RuntimeJoinDecision};
use crate::executor::utils::{column_families, extract_join_keys_and_residual_keeping};
use crate::functions::registry::global_registry;
use crate::optimizer::bloom::RuntimeBloomFilter;
use crate::parser::ast::Expression;

/// Result of a streaming join execution.
#[derive(Debug)]
pub struct JoinResult {
    /// The joined rows with synthetic row IDs.
    pub rows: RowVec,
    /// Column names for the combined result.
    pub columns: Vec<String>,
}

/// Analysis of a join operation for algorithm selection.
#[derive(Debug, Clone)]
pub struct JoinAnalysis {
    /// Left side key column indices for equality join.
    pub left_key_indices: Vec<usize>,
    /// Right side key column indices for equality join.
    pub right_key_indices: Vec<usize>,
    /// Non-equality conditions to apply after hash matching.
    pub residual_conditions: Vec<Expression>,
    /// The parsed join type.
    pub join_type: JoinType,
    /// Join type as string (for compatibility).
    pub join_type_str: String,
}

/// Configuration for join execution.
/// This combines the algorithm choice with execution-specific config.
#[derive(Debug, Clone)]
struct JoinConfig {
    /// The algorithm to use.
    algorithm: RuntimeJoinAlgorithm,
    /// For hash joins: whether to build on the left side.
    build_left: bool,
}

/// Request to execute a join operation.
///
/// Uses CompactArc<Vec<Row>> to enable zero-copy sharing with CTE results.
/// When dropping, only decrements refcount (O(1)) instead of deallocating rows.
/// The caller should pass Arc-wrapped data for CTE sources.
pub struct JoinRequest<'a> {
    /// Left side rows (Arc for zero-copy sharing with CTE results).
    pub left_rows: CompactArc<Vec<Row>>,
    /// Right side rows (Arc for zero-copy sharing with CTE results).
    pub right_rows: CompactArc<Vec<Row>>,
    /// Left side column names.
    pub left_columns: &'a [String],
    /// Right side column names.
    pub right_columns: &'a [String],
    /// Join condition (if any).
    pub condition: Option<&'a Expression>,
    /// Join type string (INNER, LEFT, RIGHT, FULL, CROSS).
    pub join_type: &'a str,
    /// LIMIT for early termination.
    pub limit: Option<u64>,
    /// Execution context for expression evaluation.
    pub ctx: &'a ExecutionContext,
    /// The schema of a side that reads a base table directly, which says
    /// what its key columns hold without reading the rows
    pub left_schema: Option<CompactArc<Schema>>,
    pub right_schema: Option<CompactArc<Schema>>,
}

/// Request to execute a streaming hash join.
///
/// Unlike `JoinRequest`, this takes a streaming operator for the probe side,
/// enabling true streaming without full materialization. This is optimal for
/// LIMIT queries where early termination can stop the probe scan early.
///
/// # Memory Model
///
/// - **Build side**: Fully materialized (required for hash table construction)
/// - **Probe side**: Streams row-by-row from the operator (O(1) memory)
///
/// # When to Use
///
/// Use `StreamingJoinRequest` when:
/// - Query has LIMIT (early termination benefit)
/// - Probe side is large (avoid full materialization)
/// - Join algorithm is Hash Join
pub struct StreamingJoinRequest<'a> {
    /// Build side rows (Arc for zero-copy sharing with CTE results).
    pub build_rows: CompactArc<Vec<Row>>,
    /// Build side column names.
    pub build_columns: &'a [String],
    /// Probe side as streaming operator (NOT materialized).
    pub probe_source: Box<dyn Operator>,
    /// Probe side column names.
    pub probe_columns: Vec<String>,
    /// Join condition (if any).
    pub condition: Option<&'a Expression>,
    /// Join type string (INNER, LEFT, RIGHT, FULL, CROSS).
    pub join_type: &'a str,
    /// Whether build side is left (false = build is right).
    pub build_is_left: bool,
    /// LIMIT for early termination.
    pub limit: Option<u64>,
    /// Execution context for expression evaluation.
    pub ctx: &'a ExecutionContext,
    /// Optional bloom filter built from build side keys.
    /// When provided, probe rows are filtered before hash lookup.
    pub bloom_filter: Option<RuntimeBloomFilter>,
    /// Pre-built hash table (if available). When provided, skips the hash table
    /// build phase in HashJoinOperator, avoiding double iteration of build_rows.
    pub pre_built_hash_table: Option<JoinHashTable>,
    /// The one key's build family, TEXT (true) or numbers (false), when a
    /// probe key of the other family may come and is answered as it streams
    pub cross_key: Option<bool>,
}

/// Modern streaming join executor.
///
/// Uses Volcano-style operators for efficient join execution with:
/// - Streaming probe side (no full materialization)
/// - Early termination for LIMIT
/// - Residual filter application during iteration
pub struct JoinExecutor {}

impl JoinExecutor {
    /// Create a new join executor.
    pub fn new() -> Self {
        Self {}
    }

    /// Execute a join operation.
    ///
    /// This is the main entry point that:
    /// 1. Analyzes the join condition
    /// 2. Chooses the algorithm from the row counts and the keys
    /// 3. Executes with streaming
    /// 4. Applies early termination
    pub fn execute(&self, request: JoinRequest<'_>) -> Result<JoinResult> {
        // Build combined column list
        let mut all_columns = request.left_columns.to_vec();
        all_columns.extend(request.right_columns.iter().cloned());

        // Analyze the join (key extraction)
        let mut analysis = self.analyze(
            request.left_columns,
            request.right_columns,
            request.condition,
            request.join_type,
            &|_, _| true,
        );

        // A key pair holding a number on one side and a TEXT on the other is
        // compared in the residual instead, and the algorithm chosen again
        let cross_pairs: Vec<(usize, usize)> = analysis
            .left_key_indices
            .iter()
            .zip(&analysis.right_key_indices)
            .filter(|(l, r)| {
                let (left_numbers, left_texts) = column_families(
                    &request.left_rows,
                    request.left_columns,
                    request.left_schema.as_deref(),
                    **l,
                );
                if !left_numbers && !left_texts {
                    return false;
                }
                let (right_numbers, right_texts) = column_families(
                    &request.right_rows,
                    request.right_columns,
                    request.right_schema.as_deref(),
                    **r,
                );
                (left_numbers && right_texts) || (left_texts && right_numbers)
            })
            .map(|(l, r)| (*l, *r))
            .collect();
        if !cross_pairs.is_empty() {
            analysis = self.analyze(
                request.left_columns,
                request.right_columns,
                request.condition,
                request.join_type,
                &|l, r| !cross_pairs.contains(&(l, r)),
            );
        }

        // The one place a join's algorithm is chosen, on the keys it runs with
        let decision = QueryPlanner::plan_runtime_join(
            request.left_rows.len(),
            request.right_rows.len(),
            !analysis.left_key_indices.is_empty(),
        );
        let config = self.convert_runtime_decision(&decision, &analysis);
        let rows = self.run(&config, request, &analysis)?;

        Ok(JoinResult {
            rows,
            columns: all_columns,
        })
    }

    /// Run the join with the algorithm `config` names (takes ownership of rows)
    fn run(
        &self,
        config: &JoinConfig,
        request: JoinRequest<'_>,
        analysis: &JoinAnalysis,
    ) -> Result<RowVec> {
        match config.algorithm {
            RuntimeJoinAlgorithm::HashJoin => self.execute_hash_join(
                request.left_rows,
                request.right_rows,
                analysis,
                request.left_columns,
                request.right_columns,
                config.build_left,
                request.limit,
                request.ctx,
            ),
            RuntimeJoinAlgorithm::NestedLoop => self.execute_nested_loop(
                request.left_rows,
                request.right_rows,
                request.condition,
                request.left_columns,
                request.right_columns,
                &analysis.join_type_str,
                request.limit,
                request.ctx,
            ),
        }
    }

    /// Execute a streaming hash join where probe side streams from an operator.
    ///
    /// This is the optimized path for LIMIT queries:
    /// - Build side is materialized (required for hash table)
    /// - Probe side streams row-by-row (O(1) memory)
    /// - Early termination stops probe scan immediately when LIMIT is reached
    ///
    /// # Performance
    ///
    /// For `SELECT ... JOIN ... LIMIT 10`:
    /// - **Old path**: Materialize 10M + 10M rows, then return 10
    /// - **This path**: Materialize 10M rows, stream until 10 matches
    ///
    /// Memory usage is halved, and early termination actually stops work.
    pub fn execute_streaming(&self, request: StreamingJoinRequest<'_>) -> Result<JoinResult> {
        // Build combined column list based on build side position
        let (left_columns, right_columns) = if request.build_is_left {
            (
                request.build_columns.to_vec(),
                request.probe_columns.clone(),
            )
        } else {
            (
                request.probe_columns.clone(),
                request.build_columns.to_vec(),
            )
        };

        let mut all_columns = left_columns.clone();
        all_columns.extend(right_columns.iter().cloned());

        // Analyze the join condition
        let analysis = self.analyze(
            &left_columns,
            &right_columns,
            request.condition,
            request.join_type,
            &|_, _| true,
        );

        // Probe side is already a streaming operator
        let probe_op = request.probe_source;

        let build_side = if request.build_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };

        // Create hash join operator - use pre-built hash table if available
        let mut join_op = if let Some(hash_table) = request.pre_built_hash_table {
            // Fast path: hash table already built (avoids double iteration of build_rows)
            let (probe_keys, build_keys) = if request.build_is_left {
                (&analysis.right_key_indices, &analysis.left_key_indices)
            } else {
                (&analysis.left_key_indices, &analysis.right_key_indices)
            };
            HashJoinOperator::with_prebuilt(
                probe_op,
                request.build_rows,
                hash_table,
                analysis.join_type,
                probe_keys.clone(),
                build_keys.clone(),
                request.build_is_left,
            )
        } else {
            // Standard path: build hash table during open()
            let build_schema: Vec<ColumnInfo> =
                request.build_columns.iter().map(ColumnInfo::new).collect();
            // Unwrap CompactArc if sole owner, otherwise clone (MaterializedOperator needs Vec<Row>)
            let build_rows_vec =
                CompactArc::try_unwrap(request.build_rows).unwrap_or_else(|arc| (*arc).clone());
            let build_op = Box::new(MaterializedOperator::new(build_rows_vec, build_schema));

            let (left_op, right_op): (Box<dyn Operator>, Box<dyn Operator>) =
                if request.build_is_left {
                    (build_op, probe_op)
                } else {
                    (probe_op, build_op)
                };

            HashJoinOperator::new(
                left_op,
                right_op,
                analysis.join_type,
                analysis.left_key_indices.clone(),
                analysis.right_key_indices.clone(),
                build_side,
            )
        };
        if let Some(build_is_text) = request.cross_key {
            join_op = join_op.with_cross_key(build_is_text);
        }

        // Compile the non-equality part of the ON clause
        let is_inner = !analysis.join_type_str.contains("LEFT")
            && !analysis.join_type_str.contains("RIGHT")
            && !analysis.join_type_str.contains("FULL");

        let residual_filters: Vec<RowFilter> = analysis
            .residual_conditions
            .iter()
            .map(|cond| RowFilter::new(cond, &all_columns).map(|f| f.with_context(request.ctx)))
            .collect::<Result<Vec<_>>>()?;

        // OUTER joins have to reject a pair inside the operator so the row it
        // came from still counts as unmatched; INNER joins can filter the
        // output rows, which avoids materializing a candidate per pair.
        let inline_filters: &[RowFilter] = if is_inner {
            &residual_filters
        } else {
            join_op = join_op.with_residual(residual_filters);
            &[]
        };

        // Execute with Volcano model - this is where true streaming happens!
        let rows =
            self.execute_operator_with_filter(&mut join_op, request.limit, inline_filters)?;

        Ok(JoinResult {
            rows,
            columns: all_columns,
        })
    }

    /// Analyze join for algorithm selection and key extraction.
    fn analyze(
        &self,
        left_columns: &[String],
        right_columns: &[String],
        condition: Option<&Expression>,
        join_type_str: &str,
        keep: &dyn Fn(usize, usize) -> bool,
    ) -> JoinAnalysis {
        let join_type = JoinType::parse(join_type_str);

        // Extract equality keys and residual conditions
        let (left_key_indices, right_key_indices, residual_conditions) =
            if let Some(cond) = condition {
                extract_join_keys_and_residual_keeping(cond, left_columns, right_columns, keep)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        JoinAnalysis {
            left_key_indices,
            right_key_indices,
            residual_conditions,
            join_type,
            join_type_str: join_type_str.to_uppercase(),
        }
    }

    /// Convert a RuntimeJoinDecision from QueryPlanner to JoinConfig.
    ///
    /// This bridges the gap between the QueryPlanner's cost-based decisions and
    /// the executor's algorithm implementation.
    fn convert_runtime_decision(
        &self,
        decision: &RuntimeJoinDecision,
        analysis: &JoinAnalysis,
    ) -> JoinConfig {
        let build_left = match decision.algorithm {
            RuntimeJoinAlgorithm::HashJoin => {
                // Use swap_sides hint from QueryPlanner, but respect OUTER join constraints
                let join_type = &analysis.join_type_str;
                if join_type.contains("LEFT") || join_type.contains("FULL") {
                    // LEFT/FULL OUTER: must build on right (left rows must be preserved)
                    false
                } else if join_type.contains("RIGHT") {
                    // RIGHT OUTER: must build on left (right rows must be preserved)
                    true
                } else {
                    // INNER/CROSS: use QueryPlanner's decision based on cost analysis
                    // swap_sides=true means swap, so if left was smaller, build_left=true normally
                    // QueryPlanner computes swap_sides = right < left, so:
                    // - swap_sides=false means left <= right, build on left
                    // - swap_sides=true means right < left, build on right (inverted)
                    !decision.swap_sides
                }
            }
            _ => false, // build_left not used for merge/nested loop
        };

        JoinConfig {
            algorithm: decision.algorithm,
            build_left,
        }
    }

    /// Execute hash join using streaming HashJoinOperator or parallel execution.
    ///
    /// Chooses between:
    /// - **Parallel hash join**: When build side exceeds threshold (10,000 rows) and
    ///   LIMIT is absent or large (> 1,000). Better for bulk analytics.
    /// - **Streaming Volcano**: When LIMIT is small (early termination benefit) or
    ///   data is below parallel threshold. Better for interactive queries.
    #[allow(clippy::too_many_arguments)]
    fn execute_hash_join(
        &self,
        left_rows: CompactArc<Vec<Row>>,
        right_rows: CompactArc<Vec<Row>>,
        analysis: &JoinAnalysis,
        left_columns: &[String],
        right_columns: &[String],
        build_left: bool,
        limit: Option<u64>,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        // Use schema for column counts (not row data - handles empty tables correctly)
        let left_col_count = left_columns.len();
        let right_col_count = right_columns.len();

        // Build combined column list for residual filter compilation
        let mut all_columns = left_columns.to_vec();
        all_columns.extend(right_columns.iter().cloned());

        // Decide between parallel and streaming execution
        // Parallel is beneficial when:
        // 1. Build side row count exceeds threshold (parallel overhead worthwhile)
        // 2. No LIMIT: the parallel join builds every match, streaming stops at the LIMIT
        let build_row_count = if build_left {
            left_rows.len()
        } else {
            right_rows.len()
        };
        // OUTER joins with a non-equality ON condition need the residual applied
        // per candidate pair, which only the streaming operator does.
        let is_outer = analysis.join_type_str.contains("LEFT")
            || analysis.join_type_str.contains("RIGHT")
            || analysis.join_type_str.contains("FULL");
        let use_parallel = build_row_count >= DEFAULT_PARALLEL_JOIN_THRESHOLD
            && limit.is_none()
            && !(is_outer && !analysis.residual_conditions.is_empty());

        if use_parallel {
            // Parallel execution path
            self.execute_hash_join_parallel(
                left_rows,
                right_rows,
                analysis,
                left_col_count,
                right_col_count,
                &all_columns,
                build_left,
                ctx,
            )
        } else {
            // Streaming Volcano execution path
            self.execute_hash_join_streaming(
                left_rows,
                right_rows,
                analysis,
                left_columns,
                right_columns,
                &all_columns,
                build_left,
                limit,
                ctx,
            )
        }
    }

    /// Execute hash join using parallel execution (partitioned build + Rayon).
    ///
    /// Uses parallel hash build and probe phases for bulk analytics workloads.
    /// Taken only by a join without a LIMIT, since it builds every match.
    #[allow(clippy::too_many_arguments)]
    fn execute_hash_join_parallel(
        &self,
        left_rows: CompactArc<Vec<Row>>,
        right_rows: CompactArc<Vec<Row>>,
        analysis: &JoinAnalysis,
        left_col_count: usize,
        right_col_count: usize,
        all_columns: &[String],
        build_left: bool,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        let config = ParallelConfig::default();

        // Determine probe and build sides - use Arc slices directly (zero-copy)
        let (probe_slice, build_slice, probe_key_indices, build_key_indices, swapped) =
            if build_left {
                // Build on left: probe is right, build is left
                (
                    right_rows.as_slice(),
                    left_rows.as_slice(),
                    &analysis.right_key_indices,
                    &analysis.left_key_indices,
                    true, // swapped: left is build, right is probe
                )
            } else {
                // Build on right (default): probe is left, build is right
                (
                    left_rows.as_slice(),
                    right_rows.as_slice(),
                    &analysis.left_key_indices,
                    &analysis.right_key_indices,
                    false, // not swapped: left is probe, right is build
                )
            };

        let (probe_col_count, build_col_count) = if swapped {
            (right_col_count, left_col_count)
        } else {
            (left_col_count, right_col_count)
        };

        // Execute parallel hash join
        let result = parallel_hash_join(
            probe_slice,
            build_slice,
            probe_key_indices,
            build_key_indices,
            analysis.join_type,
            probe_col_count,
            build_col_count,
            swapped,
            &config,
        );

        // Wrap with synthetic row IDs for join results
        let mut rows: RowVec = result
            .rows
            .into_iter()
            .enumerate()
            .map(|(i, row)| (i as i64, row))
            .collect();

        // Only INNER joins reach this path: execute_hash_join() routes OUTER
        // joins carrying a residual to the streaming operator.
        for cond in &analysis.residual_conditions {
            let filter = RowFilter::new(cond, all_columns)?.with_context(ctx);
            filter.retain_checked(&mut rows)?;
        }

        Ok(rows)
    }

    /// Execute hash join using streaming Volcano-style operators.
    ///
    /// Uses iterator-based execution for low latency to first row and
    /// early termination with LIMIT.
    #[allow(clippy::too_many_arguments)]
    fn execute_hash_join_streaming(
        &self,
        left_rows: CompactArc<Vec<Row>>,
        right_rows: CompactArc<Vec<Row>>,
        analysis: &JoinAnalysis,
        left_columns: &[String],
        right_columns: &[String],
        all_columns: &[String],
        build_left: bool,
        limit: Option<u64>,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        // Build schema for operators from column names
        let left_schema: Vec<ColumnInfo> = left_columns.iter().map(ColumnInfo::new).collect();
        let right_schema: Vec<ColumnInfo> = right_columns.iter().map(ColumnInfo::new).collect();

        // Create input operators from Arc (unwraps if sole owner, clones if shared)
        let left_op = Box::new(MaterializedOperator::from_arc(left_rows, left_schema));
        let right_op = Box::new(MaterializedOperator::from_arc(right_rows, right_schema));

        let build_side = if build_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };

        // Create hash join operator
        let mut join_op = HashJoinOperator::new(
            left_op,
            right_op,
            analysis.join_type,
            analysis.left_key_indices.clone(),
            analysis.right_key_indices.clone(),
            build_side,
        );

        // Compile the non-equality part of the ON clause
        let is_inner = !analysis.join_type_str.contains("LEFT")
            && !analysis.join_type_str.contains("RIGHT")
            && !analysis.join_type_str.contains("FULL");

        let residual_filters: Vec<RowFilter> = analysis
            .residual_conditions
            .iter()
            .map(|cond| RowFilter::new(cond, all_columns).map(|f| f.with_context(ctx)))
            .collect::<Result<Vec<_>>>()?;

        // OUTER joins have to reject a pair inside the operator so the row it
        // came from still counts as unmatched; INNER joins can filter the
        // output rows, which avoids materializing a candidate per pair.
        let inline_filters: &[RowFilter] = if is_inner {
            &residual_filters
        } else {
            join_op = join_op.with_residual(residual_filters);
            &[]
        };

        // Execute with Volcano model
        self.execute_operator_with_filter(&mut join_op, limit, inline_filters)
    }

    /// Execute nested loop join using NestedLoopJoinOperator.
    #[allow(clippy::too_many_arguments)]
    fn execute_nested_loop(
        &self,
        left_rows: CompactArc<Vec<Row>>,
        right_rows: CompactArc<Vec<Row>>,
        condition: Option<&Expression>,
        left_columns: &[String],
        right_columns: &[String],
        join_type_str: &str,
        limit: Option<u64>,
        ctx: &ExecutionContext,
    ) -> Result<RowVec> {
        // Build schema for operators
        let left_schema: Vec<ColumnInfo> = left_columns.iter().map(ColumnInfo::new).collect();
        let right_schema: Vec<ColumnInfo> = right_columns.iter().map(ColumnInfo::new).collect();

        // Unwrap CompactArc if sole owner, otherwise clone (MaterializedOperator needs Vec<Row>)
        let left_vec = CompactArc::try_unwrap(left_rows).unwrap_or_else(|arc| (*arc).clone());
        let right_vec = CompactArc::try_unwrap(right_rows).unwrap_or_else(|arc| (*arc).clone());

        // Create input operators - takes ownership, no clone
        let left_op = Box::new(MaterializedOperator::new(left_vec, left_schema));
        let right_op = Box::new(MaterializedOperator::new(right_vec, right_schema));

        // Convert join type string to enum
        let join_type = if join_type_str.contains("CROSS") {
            JoinType::Cross
        } else if join_type_str.contains("FULL") {
            JoinType::Full
        } else if join_type_str.contains("RIGHT") {
            JoinType::Right
        } else if join_type_str.contains("LEFT") {
            JoinType::Left
        } else {
            JoinType::Inner
        };

        // Create nested loop join operator
        let mut nl_op =
            NestedLoopJoinOperator::new(left_op, right_op, join_type, condition.cloned());
        if let Some(cond) = condition {
            let filter = JoinFilter::new(cond, left_columns, right_columns, global_registry())?
                .with_context(ctx);
            nl_op = nl_op.with_filter(filter);
        }

        // Execute with Volcano model
        self.execute_operator_with_filter(&mut nl_op, limit, &[])
    }

    /// Execute operator with Volcano model and optional residual filter.
    fn execute_operator_with_filter(
        &self,
        op: &mut dyn Operator,
        limit: Option<u64>,
        residual_filters: &[RowFilter],
    ) -> Result<RowVec> {
        op.open()?;

        let max_rows = limit.map(|l| l as usize).unwrap_or(usize::MAX);
        let mut rows = RowVec::with_capacity(max_rows.min(1000));
        let mut row_id = 0i64;
        let has_filters = !residual_filters.is_empty();

        while let Some(row_ref) = op.next()? {
            let row = row_ref.into_owned();

            // Apply residual filters - specialized unrolling for common cases (1-4 filters)
            if has_filters {
                let pass = match residual_filters.len() {
                    1 => residual_filters[0].matches_checked(&row)?,
                    2 => {
                        residual_filters[0].matches_checked(&row)?
                            && residual_filters[1].matches_checked(&row)?
                    }
                    3 => {
                        residual_filters[0].matches_checked(&row)?
                            && residual_filters[1].matches_checked(&row)?
                            && residual_filters[2].matches_checked(&row)?
                    }
                    4 => {
                        residual_filters[0].matches_checked(&row)?
                            && residual_filters[1].matches_checked(&row)?
                            && residual_filters[2].matches_checked(&row)?
                            && residual_filters[3].matches_checked(&row)?
                    }
                    _ => {
                        let mut all_pass = true;
                        for f in residual_filters {
                            if !f.matches_checked(&row)? {
                                all_pass = false;
                                break;
                            }
                        }
                        all_pass
                    }
                };
                if !pass {
                    continue;
                }
            }

            rows.push((row_id, row));
            row_id += 1;

            // Early termination
            if rows.len() >= max_rows {
                break;
            }
        }

        op.close()?;
        Ok(rows)
    }
}

impl Default for JoinExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;

    fn make_rows(data: Vec<Vec<i64>>) -> Vec<Row> {
        data.into_iter()
            .map(|vals| Row::from_values(vals.into_iter().map(Value::integer).collect()))
            .collect()
    }

    #[test]
    fn test_inner_join() {
        let executor = JoinExecutor::new();
        let ctx = ExecutionContext::new();

        let left = make_rows(vec![vec![1, 10], vec![2, 20], vec![3, 30]]);
        let right = make_rows(vec![vec![1, 100], vec![3, 300]]);

        let left_cols = vec!["a.id".to_string(), "a.val".to_string()];
        let right_cols = vec!["b.id".to_string(), "b.data".to_string()];

        // Create equality condition: a.id = b.id
        use crate::parser::ast::{Identifier, InfixExpression};
        use crate::parser::token::{Position, Token, TokenType};

        let cond = Expression::Infix(InfixExpression::new(
            Token::new(TokenType::Operator, "=", Position::default()),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, "a.id", Position::default()),
                "a.id".to_string(),
            ))),
            "=".to_string(),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, "b.id", Position::default()),
                "b.id".to_string(),
            ))),
        ));

        let request = JoinRequest {
            left_rows: CompactArc::new(left),
            right_rows: CompactArc::new(right),
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: Some(&cond),
            join_type: "INNER",
            limit: None,
            ctx: &ctx,
            left_schema: None,
            right_schema: None,
        };

        let result = executor.execute(request).unwrap();

        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.columns.len(), 4);
    }

    #[test]
    fn test_left_join() {
        let executor = JoinExecutor::new();
        let ctx = ExecutionContext::new();

        let left = make_rows(vec![vec![1, 10], vec![2, 20], vec![3, 30]]);
        let right = make_rows(vec![vec![1, 100]]);

        let left_cols = vec!["a.id".to_string(), "a.val".to_string()];
        let right_cols = vec!["b.id".to_string(), "b.data".to_string()];

        use crate::parser::ast::{Identifier, InfixExpression};
        use crate::parser::token::{Position, Token, TokenType};

        let cond = Expression::Infix(InfixExpression::new(
            Token::new(TokenType::Operator, "=", Position::default()),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, "a.id", Position::default()),
                "a.id".to_string(),
            ))),
            "=".to_string(),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, "b.id", Position::default()),
                "b.id".to_string(),
            ))),
        ));

        let request = JoinRequest {
            left_rows: CompactArc::new(left),
            right_rows: CompactArc::new(right),
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: Some(&cond),
            join_type: "LEFT",
            limit: None,
            ctx: &ctx,
            left_schema: None,
            right_schema: None,
        };

        let result = executor.execute(request).unwrap();

        // All 3 left rows should be preserved
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_early_termination() {
        let executor = JoinExecutor::new();
        let ctx = ExecutionContext::new();

        let left = make_rows(vec![vec![1], vec![2], vec![3]]);
        let right = make_rows(vec![vec![1], vec![2], vec![3]]);

        let left_cols = vec!["a.id".to_string()];
        let right_cols = vec!["b.id".to_string()];

        use crate::parser::ast::{Identifier, InfixExpression};
        use crate::parser::token::{Position, Token, TokenType};

        let cond = Expression::Infix(InfixExpression::new(
            Token::new(TokenType::Operator, "=", Position::default()),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, "a.id", Position::default()),
                "a.id".to_string(),
            ))),
            "=".to_string(),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, "b.id", Position::default()),
                "b.id".to_string(),
            ))),
        ));

        let request = JoinRequest {
            left_rows: CompactArc::new(left),
            right_rows: CompactArc::new(right),
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: Some(&cond),
            join_type: "INNER",
            limit: Some(1), // Only need 1 row
            ctx: &ctx,
            left_schema: None,
            right_schema: None,
        };

        let result = executor.execute(request).unwrap();

        // Should stop after 1 row
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_cross_join() {
        let executor = JoinExecutor::new();
        let ctx = ExecutionContext::new();

        let left = make_rows(vec![vec![1], vec![2]]);
        let right = make_rows(vec![vec![10], vec![20]]);

        let left_cols = vec!["a.id".to_string()];
        let right_cols = vec!["b.val".to_string()];

        let request = JoinRequest {
            left_rows: CompactArc::new(left),
            right_rows: CompactArc::new(right),
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: None,
            join_type: "CROSS",
            limit: None,
            ctx: &ctx,
            left_schema: None,
            right_schema: None,
        };

        let result = executor.execute(request).unwrap();

        // 2 x 2 = 4 rows
        assert_eq!(result.rows.len(), 4);
    }

    fn on_clause(condition: &str) -> Expression {
        let sql = format!("SELECT * FROM a JOIN b ON {condition}");
        match &crate::parser::parse_sql(&sql).unwrap()[0] {
            crate::parser::ast::Statement::Select(select) => match select.table_expr.as_deref() {
                Some(Expression::JoinSource(join)) => join.condition.as_deref().unwrap().clone(),
                other => panic!("not a join: {other:?}"),
            },
            other => panic!("not a SELECT: {other:?}"),
        }
    }

    #[test]
    fn every_join_algorithm_returns_the_rows_of_the_nested_loop() {
        use RuntimeJoinAlgorithm::{HashJoin, NestedLoop};
        let all = [HashJoin, NestedLoop];
        // A new algorithm fails to compile here until it joins `all`
        match all[0] {
            HashJoin | NestedLoop => {}
        }
        let left: Vec<Row> = [
            (1, Value::integer(1)),
            (2, Value::integer(1)),
            (3, Value::float(2.0)),
            (4, Value::null_unknown()),
            (5, Value::integer(7)),
            (6, Value::integer(3)),
        ]
        .into_iter()
        .map(|(v, k)| Row::from_values(vec![k, Value::integer(v)]))
        .collect();
        let right: Vec<Row> = [
            (Value::integer(2), 5),
            (Value::integer(1), 1),
            (Value::null_unknown(), 9),
            (Value::integer(1), 9),
            (Value::integer(3), 0),
            (Value::integer(8), 2),
        ]
        .into_iter()
        .map(|(k, w)| Row::from_values(vec![k, Value::integer(w)]))
        .collect();
        let left_cols = vec!["a.k".to_string(), "a.v".to_string()];
        let right_cols = vec!["b.k".to_string(), "b.w".to_string()];
        let ctx = ExecutionContext::new();
        let executor = JoinExecutor::new();
        let mut cases = 0;
        for condition in ["a.k = b.k", "a.k = b.k AND a.v < b.w"] {
            let condition = on_clause(condition);
            for join_type in ["INNER", "LEFT", "RIGHT", "FULL"] {
                let analysis = executor.analyze(
                    &left_cols,
                    &right_cols,
                    Some(&condition),
                    join_type,
                    &|_, _| true,
                );
                let rows_of = |algorithm: RuntimeJoinAlgorithm, swap_sides: bool| {
                    let decision = RuntimeJoinDecision {
                        algorithm,
                        swap_sides,
                        explanation: String::new(),
                    };
                    let config = executor.convert_runtime_decision(&decision, &analysis);
                    let request = JoinRequest {
                        left_rows: CompactArc::new(left.clone()),
                        right_rows: CompactArc::new(right.clone()),
                        left_columns: &left_cols,
                        right_columns: &right_cols,
                        condition: Some(&condition),
                        join_type,
                        limit: None,
                        ctx: &ctx,
                        left_schema: None,
                        right_schema: None,
                    };
                    let rows = executor.run(&config, request, &analysis).unwrap();
                    let mut rows: Vec<String> = rows
                        .iter()
                        .map(|(_, row)| format!("{:?}", row.as_slice()))
                        .collect();
                    rows.sort();
                    rows
                };
                let expected = rows_of(NestedLoop, false);
                for algorithm in all {
                    for swap_sides in [false, true] {
                        assert_eq!(
                            rows_of(algorithm, swap_sides),
                            expected,
                            "{algorithm:?} (swap {swap_sides}) on {join_type} JOIN ON {condition}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 2 * 4 * all.len() * 2);
    }
}
