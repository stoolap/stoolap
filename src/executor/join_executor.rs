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
//! - **Merge Join**: O(N+M) when inputs are pre-sorted on join keys
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
//! ## Volcano-Style Single-Threaded Operators
//!
//! This executor uses Volcano-style (iterator) operators rather than parallel bulk
//! processing. This is an intentional design choice:
//!
//! **Benefits:**
//! - **Streaming Memory**: Only one row in flight at a time, enabling O(1) memory
//!   for the probe side of hash joins (vs O(N+M) for bulk materialization)
//! - **Early Termination**: LIMIT queries stop immediately after N rows
//! - **Composability**: Operators chain naturally (Filter → Join → Project → Limit)
//! - **Predictable Latency**: First row returned quickly (important for interactive queries)
//!
//! **Tradeoffs:**
//! - Single-threaded probe phase (parallel build is still possible in future)
//! - Cannot use SIMD for bulk operations within a single operator
//!
//! For large analytical queries where parallelism matters more than streaming,
//! the parallel execution path in `parallel.rs` provides DashMap-based parallel
//! hash joins. The optimizer chooses between these based on query characteristics.
//!
//! ## Merge Join Materialization
//!
//! The current MergeJoin implementation materializes both sides into
//! `MaterializedOperator`. A true streaming merge join would require sorted
//! iterators on both inputs. This is a future optimization opportunity:
//!
//! ```text
//! Current:    left_rows → MaterializedOperator  ┐
//!             right_rows → MaterializedOperator ┘→ MergeJoinOperator
//!
//! Optimal:    left_iter → SortedIterator  ┐
//!             right_iter → SortedIterator ┘→ StreamingMergeJoin
//! ```
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

use crate::core::{Result, Row, Value};
use crate::executor::context::ExecutionContext;
use crate::executor::expression::RowFilter;
use crate::executor::operator::{ColumnInfo, MaterializedOperator, Operator};
use crate::executor::operators::hash_join::{HashJoinOperator, JoinSide, JoinType};
use crate::executor::operators::merge_join::MergeJoinOperator;
use crate::executor::operators::nested_loop_join::NestedLoopJoinOperator;
use crate::executor::planner::{RuntimeJoinAlgorithm, RuntimeJoinDecision};
use crate::executor::utils::{extract_join_keys_and_residual, is_sorted_on_keys};
use crate::parser::ast::Expression;

/// Result of a streaming join execution.
#[derive(Debug)]
pub struct JoinResult {
    /// The joined rows.
    pub rows: Vec<Row>,
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
    /// Whether left input is sorted on join keys.
    pub left_sorted: bool,
    /// Whether right input is sorted on join keys.
    pub right_sorted: bool,
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
/// Takes ownership of row vectors to avoid cloning during materialization.
/// The caller should pass owned data; if you only have references, use `to_vec()`.
pub struct JoinRequest<'a> {
    /// Left side rows (owned to avoid materialization clone).
    pub left_rows: Vec<Row>,
    /// Right side rows (owned to avoid materialization clone).
    pub right_rows: Vec<Row>,
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
    /// Optional algorithm decision from QueryPlanner.
    /// When provided, the executor uses this instead of making its own decision.
    pub algorithm_hint: Option<&'a RuntimeJoinDecision>,
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
    /// 2. Uses provided algorithm hint or selects optimal algorithm
    /// 3. Executes with streaming
    /// 4. Applies early termination
    pub fn execute(&self, request: JoinRequest<'_>) -> Result<JoinResult> {
        // Build combined column list
        let mut all_columns = request.left_columns.to_vec();
        all_columns.extend(request.right_columns.iter().cloned());

        // Analyze the join (key extraction only - sort check is deferred)
        let analysis = self.analyze(
            request.left_columns,
            request.right_columns,
            request.condition,
            request.join_type,
        );

        // Select algorithm: use provided hint from QueryPlanner if available,
        // otherwise fall back to local heuristics
        let config = if let Some(hint) = request.algorithm_hint {
            self.convert_runtime_decision(hint, &analysis)
        } else {
            self.select_algorithm(&analysis, &request.left_rows, &request.right_rows)
        };

        // Execute join based on algorithm (takes ownership of rows)
        let rows = match config.algorithm {
            RuntimeJoinAlgorithm::HashJoin => self.execute_hash_join(
                request.left_rows,
                request.right_rows,
                &analysis,
                request.left_columns,
                request.right_columns,
                config.build_left,
                request.limit,
                request.ctx,
            )?,
            RuntimeJoinAlgorithm::MergeJoin => self.execute_merge_join(
                request.left_rows,
                request.right_rows,
                request.left_columns,
                request.right_columns,
                &analysis,
                request.limit,
            )?,
            RuntimeJoinAlgorithm::NestedLoop => self.execute_nested_loop(
                request.left_rows,
                request.right_rows,
                request.condition,
                request.left_columns,
                request.right_columns,
                &analysis.join_type_str,
                request.limit,
            )?,
        };

        Ok(JoinResult {
            rows,
            columns: all_columns,
        })
    }

    /// Analyze join for algorithm selection and key extraction.
    ///
    /// Note: Sortedness is NOT checked here. When an algorithm hint from QueryPlanner
    /// is provided, the sort check is unnecessary overhead. Sortedness is only checked
    /// lazily in select_algorithm() when fallback heuristics are used.
    fn analyze(
        &self,
        left_columns: &[String],
        right_columns: &[String],
        condition: Option<&Expression>,
        join_type_str: &str,
    ) -> JoinAnalysis {
        let join_type = JoinType::parse(join_type_str);

        // Extract equality keys and residual conditions
        let (left_key_indices, right_key_indices, residual_conditions) =
            if let Some(cond) = condition {
                extract_join_keys_and_residual(cond, left_columns, right_columns)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        // Note: left_sorted/right_sorted are initialized to false.
        // Actual sort check is deferred to select_algorithm() when needed.
        JoinAnalysis {
            left_key_indices,
            right_key_indices,
            residual_conditions,
            left_sorted: false,
            right_sorted: false,
            join_type,
            join_type_str: join_type_str.to_uppercase(),
        }
    }

    /// Select optimal join algorithm based on analysis and cardinalities.
    ///
    /// This is the fallback algorithm selection when QueryPlanner doesn't provide
    /// an algorithm hint. Sort check is performed here (lazily) only when needed.
    fn select_algorithm(
        &self,
        analysis: &JoinAnalysis,
        left_rows: &[Row],
        right_rows: &[Row],
    ) -> JoinConfig {
        let has_equality_keys = !analysis.left_key_indices.is_empty();

        // No equality keys -> must use nested loop
        if !has_equality_keys {
            return JoinConfig {
                algorithm: RuntimeJoinAlgorithm::NestedLoop,
                build_left: false,
            };
        }

        // Check if both sides are sorted on join keys (lazy evaluation)
        // Only perform this O(n) check when we might actually use merge join
        let left_sorted = is_sorted_on_keys(left_rows, &analysis.left_key_indices);
        let right_sorted =
            left_sorted && is_sorted_on_keys(right_rows, &analysis.right_key_indices);

        // Both sides sorted on join keys -> use merge join
        if left_sorted && right_sorted {
            return JoinConfig {
                algorithm: RuntimeJoinAlgorithm::MergeJoin,
                build_left: false,
            };
        }

        // Use hash join with build on smaller side
        // Exception: OUTER joins have restrictions on build side
        let join_type = &analysis.join_type_str;
        let build_left = if join_type.contains("LEFT") || join_type.contains("FULL") {
            // LEFT/FULL OUTER: must build on right (left rows must be preserved)
            false
        } else if join_type.contains("RIGHT") {
            // RIGHT OUTER: must build on left (right rows must be preserved)
            true
        } else {
            // INNER/CROSS: build on smaller side
            left_rows.len() <= right_rows.len()
        };

        JoinConfig {
            algorithm: RuntimeJoinAlgorithm::HashJoin,
            build_left,
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

    /// Execute hash join using streaming HashJoinOperator.
    #[allow(clippy::too_many_arguments)]
    fn execute_hash_join(
        &self,
        left_rows: Vec<Row>,
        right_rows: Vec<Row>,
        analysis: &JoinAnalysis,
        left_columns: &[String],
        right_columns: &[String],
        build_left: bool,
        limit: Option<u64>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Row>> {
        // Use schema for column counts (not row data - handles empty tables correctly)
        let left_col_count = left_columns.len();
        let right_col_count = right_columns.len();

        // Build schema for operators from column names
        let left_schema: Vec<ColumnInfo> = left_columns.iter().map(ColumnInfo::new).collect();
        let right_schema: Vec<ColumnInfo> = right_columns.iter().map(ColumnInfo::new).collect();

        // Build combined column list for residual filter compilation
        let mut all_columns = left_columns.to_vec();
        all_columns.extend(right_columns.iter().cloned());

        // Create input operators - takes ownership, no clone
        let left_op = Box::new(MaterializedOperator::new(left_rows, left_schema));
        let right_op = Box::new(MaterializedOperator::new(right_rows, right_schema));

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

        // Compile residual filters for inline application (INNER JOINs only)
        let is_inner = !analysis.join_type_str.contains("LEFT")
            && !analysis.join_type_str.contains("RIGHT")
            && !analysis.join_type_str.contains("FULL");

        let residual_filters: Vec<RowFilter> = if is_inner {
            analysis
                .residual_conditions
                .iter()
                .map(|cond| RowFilter::new(cond, &all_columns).map(|f| f.with_context(ctx)))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        // Execute with Volcano model
        let rows = self.execute_operator_with_filter(&mut join_op, limit, &residual_filters)?;

        // Apply residual conditions for OUTER joins (after iteration)
        let rows = if !is_inner && !analysis.residual_conditions.is_empty() {
            self.apply_residual_post_join(
                rows,
                &analysis.residual_conditions,
                &all_columns,
                &analysis.join_type_str,
                left_col_count,
                right_col_count,
                ctx,
            )?
        } else {
            rows
        };

        Ok(rows)
    }

    /// Execute merge join for pre-sorted inputs using MergeJoinOperator.
    fn execute_merge_join(
        &self,
        left_rows: Vec<Row>,
        right_rows: Vec<Row>,
        left_columns: &[String],
        right_columns: &[String],
        analysis: &JoinAnalysis,
        limit: Option<u64>,
    ) -> Result<Vec<Row>> {
        // Build schema for operators
        let left_schema: Vec<ColumnInfo> = left_columns.iter().map(ColumnInfo::new).collect();
        let right_schema: Vec<ColumnInfo> = right_columns.iter().map(ColumnInfo::new).collect();

        // Create input operators - takes ownership, no clone
        let left_op = Box::new(MaterializedOperator::new(left_rows, left_schema));
        let right_op = Box::new(MaterializedOperator::new(right_rows, right_schema));

        // Create merge join operator
        let mut merge_op = MergeJoinOperator::new(
            left_op,
            right_op,
            analysis.join_type,
            analysis.left_key_indices.clone(),
            analysis.right_key_indices.clone(),
        );

        // Execute with Volcano model (no residual filters for merge join currently)
        self.execute_operator_with_filter(&mut merge_op, limit, &[])
    }

    /// Execute nested loop join using NestedLoopJoinOperator.
    #[allow(clippy::too_many_arguments)]
    fn execute_nested_loop(
        &self,
        left_rows: Vec<Row>,
        right_rows: Vec<Row>,
        condition: Option<&Expression>,
        left_columns: &[String],
        right_columns: &[String],
        join_type_str: &str,
        limit: Option<u64>,
    ) -> Result<Vec<Row>> {
        // Build schema for operators
        let left_schema: Vec<ColumnInfo> = left_columns.iter().map(ColumnInfo::new).collect();
        let right_schema: Vec<ColumnInfo> = right_columns.iter().map(ColumnInfo::new).collect();

        // Create input operators - takes ownership, no clone
        let left_op = Box::new(MaterializedOperator::new(left_rows, left_schema));
        let right_op = Box::new(MaterializedOperator::new(right_rows, right_schema));

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

        // Execute with Volcano model
        self.execute_operator_with_filter(&mut nl_op, limit, &[])
    }

    /// Execute operator with Volcano model and optional residual filter.
    fn execute_operator_with_filter(
        &self,
        op: &mut dyn Operator,
        limit: Option<u64>,
        residual_filters: &[RowFilter],
    ) -> Result<Vec<Row>> {
        op.open()?;

        let max_rows = limit.map(|l| l as usize).unwrap_or(usize::MAX);
        let mut rows = Vec::with_capacity(max_rows.min(1000));
        let has_filters = !residual_filters.is_empty();

        while let Some(row_ref) = op.next()? {
            let row = row_ref.into_owned();

            // Apply residual filters
            if has_filters && !residual_filters.iter().all(|f| f.matches(&row)) {
                continue;
            }

            rows.push(row);

            // Early termination
            if rows.len() >= max_rows {
                break;
            }
        }

        op.close()?;
        Ok(rows)
    }

    /// Apply residual conditions for OUTER joins.
    ///
    /// For OUTER joins, residual conditions need special handling:
    /// matched rows that fail residual should produce NULL-padded output.
    #[allow(clippy::too_many_arguments)]
    fn apply_residual_post_join(
        &self,
        mut rows: Vec<Row>,
        residual: &[Expression],
        all_columns: &[String],
        join_type: &str,
        left_col_count: usize,
        right_col_count: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<Row>> {
        let is_left_outer = join_type.contains("LEFT");
        let is_right_outer = join_type.contains("RIGHT");
        let is_full_outer = join_type.contains("FULL");

        for cond in residual {
            let filter = RowFilter::new(cond, all_columns)?.with_context(ctx);

            if is_left_outer || is_right_outer || is_full_outer {
                // For OUTER joins, replace non-matching rows with NULL-padded versions
                rows = rows
                    .into_iter()
                    .map(|row| {
                        if filter.matches(&row) {
                            row
                        } else {
                            // Convert to NULL-padded row
                            let values = row.as_slice();
                            if is_left_outer {
                                // Keep left, NULL right
                                let mut new_values = values[..left_col_count].to_vec();
                                new_values.extend(std::iter::repeat_n(
                                    Value::null_unknown(),
                                    right_col_count,
                                ));
                                Row::from_values(new_values)
                            } else if is_right_outer {
                                // NULL left, keep right
                                let mut new_values: Vec<Value> =
                                    std::iter::repeat_n(Value::null_unknown(), left_col_count)
                                        .collect();
                                new_values.extend(values[left_col_count..].iter().cloned());
                                Row::from_values(new_values)
                            } else {
                                // FULL OUTER - keep original for now
                                row
                            }
                        }
                    })
                    .collect();
            } else {
                // INNER join - just filter
                rows.retain(|row| filter.matches(row));
            }
        }

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
            left_rows: left,
            right_rows: right,
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: Some(&cond),
            join_type: "INNER",
            limit: None,
            ctx: &ctx,
            algorithm_hint: None,
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
            left_rows: left,
            right_rows: right,
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: Some(&cond),
            join_type: "LEFT",
            limit: None,
            ctx: &ctx,
            algorithm_hint: None,
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
            left_rows: left,
            right_rows: right,
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: Some(&cond),
            join_type: "INNER",
            limit: Some(1), // Only need 1 row
            ctx: &ctx,
            algorithm_hint: None,
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
            left_rows: left,
            right_rows: right,
            left_columns: &left_cols,
            right_columns: &right_cols,
            condition: None,
            join_type: "CROSS",
            limit: None,
            ctx: &ctx,
            algorithm_hint: None,
        };

        let result = executor.execute(request).unwrap();

        // 2 x 2 = 4 rows
        assert_eq!(result.rows.len(), 4);
    }
}
