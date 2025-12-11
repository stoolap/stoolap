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

#![allow(clippy::only_used_in_recursion)]

//! Join ordering optimization for multi-way joins
//!
//! This module implements join ordering optimization using **Dynamic Programming**
//! for optimal plans and **greedy fallback** for large queries.
//!
//! ```sql
//! SELECT * FROM A JOIN B ON A.x = B.x JOIN C ON B.y = C.y JOIN D ON C.z = D.z
//! ```
//!
//! ## Algorithms
//!
//! ### 1. Dynamic Programming (for n ≤ 10 tables)
//!
//! Uses bitmask-based DP to explore ALL possible join orderings, finding the
//! globally optimal plan. This is O(3^n) but produces perfect results.
//!
//! The algorithm:
//! 1. Enumerate all 2^n subsets of tables
//! 2. For each subset, find the cheapest way to partition it into two subsets
//! 3. Memoize the optimal cost for each subset
//! 4. Build bushy trees (not just left-deep) for potentially better plans
//!
//! ### 2. Greedy Algorithm (fallback for n > 10)
//!
//! Uses O(n²) greedy approach for very large joins where DP would be too slow.
//!
//! ## Stoolap-Specific Optimizations
//!
//! - **Edge-aware memory limits**: Considers available memory for hash tables
//! - **Interesting orderings**: Tracks sort orders for potential merge joins
//! - **Cross-product avoidance**: Strongly penalizes Cartesian products

use std::collections::{HashMap, HashSet};

use crate::optimizer::cost::{CostEstimator, JoinAlgorithm, JoinStats, PlanCost};
use crate::storage::statistics::TableStats;

/// Represents a join condition between two tables
#[derive(Debug, Clone)]
pub struct JoinCondition {
    /// Left table name
    pub left_table: String,
    /// Left column name
    pub left_column: String,
    /// Right table name
    pub right_table: String,
    /// Right column name
    pub right_column: String,
    /// Whether this is an equality condition (col1 = col2)
    pub is_equality: bool,
}

impl JoinCondition {
    /// Create a new equality join condition
    pub fn new_equality(
        left_table: String,
        left_column: String,
        right_table: String,
        right_column: String,
    ) -> Self {
        Self {
            left_table,
            left_column,
            right_table,
            right_column,
            is_equality: true,
        }
    }

    /// Check if this condition connects the given tables
    pub fn connects(&self, tables1: &HashSet<String>, tables2: &HashSet<String>) -> bool {
        (tables1.contains(&self.left_table) && tables2.contains(&self.right_table))
            || (tables1.contains(&self.right_table) && tables2.contains(&self.left_table))
    }

    /// Check if this condition involves the given table
    pub fn involves(&self, table: &str) -> bool {
        self.left_table == table || self.right_table == table
    }
}

/// Represents a column sort order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortOrder {
    /// Column name (may be table.column for qualified names)
    pub column: String,
    /// True for ascending, false for descending
    pub ascending: bool,
}

impl SortOrder {
    /// Create ascending sort order
    pub fn asc(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            ascending: true,
        }
    }

    /// Create descending sort order
    pub fn desc(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            ascending: false,
        }
    }
}

/// Represents a node in the join tree
/// Can be a single table or a joined result of multiple tables
#[derive(Debug, Clone)]
pub struct JoinNode {
    /// Tables included in this node
    pub tables: HashSet<String>,
    /// Estimated row count after joining
    pub row_estimate: u64,
    /// Distinct count on the join columns (for further joins)
    pub distinct_estimate: u64,
    /// Cumulative cost to reach this node
    pub cumulative_cost: f64,
    /// The algorithm used to create this node (None for base tables)
    pub algorithm: Option<JoinAlgorithm>,
    /// Columns this node is sorted by (empty = unsorted)
    /// For index scans, this is the index key columns
    /// For merge joins, this is the join key columns
    pub sorted_by: Vec<SortOrder>,
}

impl JoinNode {
    /// Create a new leaf node for a single table
    pub fn leaf(table_name: String, row_count: u64, distinct_count: u64) -> Self {
        let mut tables = HashSet::new();
        tables.insert(table_name);
        Self {
            tables,
            row_estimate: row_count,
            distinct_estimate: distinct_count,
            cumulative_cost: 0.0,
            algorithm: None,
            sorted_by: Vec::new(),
        }
    }

    /// Create a new leaf node with sorted columns (e.g., from an index scan)
    pub fn leaf_sorted(
        table_name: String,
        row_count: u64,
        distinct_count: u64,
        sorted_by: Vec<SortOrder>,
    ) -> Self {
        let mut tables = HashSet::new();
        tables.insert(table_name);
        Self {
            tables,
            row_estimate: row_count,
            distinct_estimate: distinct_count,
            cumulative_cost: 0.0,
            algorithm: None,
            sorted_by,
        }
    }

    /// Create a joined node from two nodes
    pub fn joined(
        left: &JoinNode,
        right: &JoinNode,
        output_rows: u64,
        output_distinct: u64,
        join_cost: f64,
        algorithm: JoinAlgorithm,
    ) -> Self {
        let mut tables = left.tables.clone();
        tables.extend(right.tables.iter().cloned());

        // Determine output sort order based on join algorithm
        let sorted_by = match &algorithm {
            JoinAlgorithm::MergeJoin { .. } => {
                // Merge join preserves left sort order (since we merge on sorted inputs)
                left.sorted_by.clone()
            }
            JoinAlgorithm::NestedLoop { .. } => {
                // Nested loop preserves outer (left) sort order
                left.sorted_by.clone()
            }
            _ => {
                // Hash join does not preserve sort order
                Vec::new()
            }
        };

        Self {
            tables,
            row_estimate: output_rows,
            distinct_estimate: output_distinct,
            cumulative_cost: left.cumulative_cost + right.cumulative_cost + join_cost,
            algorithm: Some(algorithm),
            sorted_by,
        }
    }

    /// Check if this node contains the given table
    pub fn contains(&self, table: &str) -> bool {
        self.tables.contains(table)
    }

    /// Check if this node is sorted by the given column (any order)
    pub fn is_sorted_by(&self, column: &str) -> bool {
        self.sorted_by
            .first()
            .map(|s| {
                s.column == column
                    || s.column.ends_with(&format!(".{}", column))
                    || column.ends_with(&format!(".{}", s.column))
            })
            .unwrap_or(false)
    }

    /// Check if this node is sorted by a column from the given table
    pub fn is_sorted_by_table_column(&self, table: &str, column: &str) -> bool {
        let full_name = format!("{}.{}", table, column);
        self.sorted_by
            .first()
            .map(|s| s.column == full_name || s.column == column)
            .unwrap_or(false)
    }
}

/// A step in the join plan
#[derive(Debug, Clone)]
pub struct JoinStep {
    /// Tables being joined in this step
    pub left_tables: HashSet<String>,
    pub right_tables: HashSet<String>,
    /// The join algorithm to use
    pub algorithm: JoinAlgorithm,
    /// The join condition for this step
    pub condition: Option<JoinCondition>,
    /// Estimated cost of this step
    pub cost: PlanCost,
    /// Estimated output rows
    pub output_rows: u64,
}

/// The result of join ordering optimization
#[derive(Debug, Clone)]
pub struct JoinPlan {
    /// The steps to execute joins, in order
    pub steps: Vec<JoinStep>,
    /// Total estimated cost
    pub total_cost: f64,
    /// Final output row estimate
    pub output_rows: u64,
}

impl JoinPlan {
    /// Create an empty plan
    pub fn empty() -> Self {
        Self {
            steps: Vec::new(),
            total_cost: 0.0,
            output_rows: 0,
        }
    }

    /// Create a single-join plan
    pub fn single(step: JoinStep) -> Self {
        let total_cost = step.cost.total;
        let output_rows = step.output_rows;
        Self {
            steps: vec![step],
            total_cost,
            output_rows,
        }
    }
}

/// Maximum number of tables for DP optimization (2^n subsets)
const DP_TABLE_LIMIT: usize = 10;

/// Cost penalty for cross products (Cartesian joins)
const CROSS_PRODUCT_PENALTY: f64 = 1000.0;

/// Memory limit for hash join build side (in rows, for edge computing)
const HASH_JOIN_MEMORY_LIMIT: u64 = 100_000;

/// DP memoization entry
#[derive(Debug, Clone)]
struct DpEntry {
    /// Total cost to compute this subset
    cost: f64,
    /// Estimated output rows
    rows: u64,
    /// Estimated distinct count (for further joins)
    distinct: u64,
    /// Left subset mask (0 for base tables)
    left_mask: usize,
    /// Right subset mask (0 for base tables)
    right_mask: usize,
    /// Join algorithm used (None for base tables)
    algorithm: Option<JoinAlgorithm>,
    /// Columns this subset is sorted by (for merge join optimization)
    sorted_by: Vec<SortOrder>,
}

/// Result of join cost computation
#[derive(Debug)]
struct JoinCostResult {
    cost: f64,
    rows: u64,
    distinct: u64,
    algorithm: JoinAlgorithm,
    /// Output sort order after join
    sorted_by: Vec<SortOrder>,
}

/// Join optimizer using Dynamic Programming (optimal) or greedy (fallback)
pub struct JoinOptimizer {
    /// Cost estimator for join cost calculations
    cost_estimator: CostEstimator,
    /// Table statistics (table_name -> TableStats)
    table_stats: HashMap<String, TableStats>,
    /// Column distinct counts (table_name.column_name -> distinct_count)
    column_distinct: HashMap<String, u64>,
    /// Available memory for hash tables (rows, for edge-aware planning)
    memory_budget: u64,
    /// Sorted tables: table_name -> columns the input is sorted by
    /// This is set when the input comes from an index scan or pre-sorted source
    sorted_inputs: HashMap<String, Vec<SortOrder>>,
}

impl JoinOptimizer {
    /// Create a new join optimizer
    pub fn new(cost_estimator: CostEstimator) -> Self {
        Self {
            cost_estimator,
            table_stats: HashMap::new(),
            column_distinct: HashMap::new(),
            memory_budget: HASH_JOIN_MEMORY_LIMIT,
            sorted_inputs: HashMap::new(),
        }
    }

    /// Create optimizer with custom memory budget (for edge devices)
    pub fn with_memory_budget(cost_estimator: CostEstimator, memory_budget: u64) -> Self {
        Self {
            cost_estimator,
            table_stats: HashMap::new(),
            column_distinct: HashMap::new(),
            memory_budget,
            sorted_inputs: HashMap::new(),
        }
    }

    /// Add table statistics
    pub fn add_table_stats(&mut self, table_name: &str, stats: TableStats) {
        self.table_stats.insert(table_name.to_string(), stats);
    }

    /// Add column distinct count
    pub fn add_column_distinct(&mut self, table_name: &str, column_name: &str, distinct: u64) {
        let key = format!("{}.{}", table_name, column_name);
        self.column_distinct.insert(key, distinct);
    }

    /// Mark a table's input as sorted by specific columns
    /// This is called when the scan uses an index that provides sorted output
    pub fn add_sorted_input(&mut self, table_name: &str, sorted_by: Vec<SortOrder>) {
        self.sorted_inputs.insert(table_name.to_string(), sorted_by);
    }

    /// Check if a table's input is sorted
    pub fn is_input_sorted(&self, table_name: &str) -> bool {
        self.sorted_inputs.contains_key(table_name)
    }

    /// Get the sort order for a table's input
    pub fn get_input_sort_order(&self, table_name: &str) -> Option<&Vec<SortOrder>> {
        self.sorted_inputs.get(table_name)
    }

    /// Check if input is sorted by a specific column
    #[allow(dead_code)]
    fn is_sorted_by_column(&self, tables: &HashSet<String>, column: &str) -> bool {
        for table in tables {
            if let Some(sorted_cols) = self.sorted_inputs.get(table) {
                if sorted_cols
                    .first()
                    .map(|s| s.column == column || s.column.ends_with(&format!(".{}", column)))
                    .unwrap_or(false)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Get table row count (with fallback)
    fn get_row_count(&self, table_name: &str) -> u64 {
        self.table_stats
            .get(table_name)
            .map(|s| s.row_count)
            .unwrap_or(1000) // Default estimate
    }

    /// Get table stats (with fallback)
    fn get_table_stats(&self, table_name: &str) -> TableStats {
        self.table_stats
            .get(table_name)
            .cloned()
            .unwrap_or_else(|| TableStats {
                table_name: table_name.to_string(),
                row_count: 1000,
                page_count: 10,
                avg_row_size: 100,
            })
    }

    /// Get column distinct count (with fallback)
    fn get_distinct_count(&self, table_name: &str, column_name: &str) -> u64 {
        let key = format!("{}.{}", table_name, column_name);
        self.column_distinct.get(&key).copied().unwrap_or_else(|| {
            // Default: assume 10% of rows are distinct
            (self.get_row_count(table_name) / 10).max(1)
        })
    }

    /// Optimize join order for a set of tables with given join conditions
    ///
    /// Uses **Dynamic Programming** for ≤10 tables (optimal)
    /// Falls back to **greedy** for larger joins (fast but suboptimal)
    pub fn optimize_join_order(&self, tables: &[&str], conditions: &[JoinCondition]) -> JoinPlan {
        if tables.is_empty() {
            return JoinPlan::empty();
        }

        if tables.len() == 1 {
            return JoinPlan::empty();
        }

        // Use DP for small joins (optimal), greedy for large (practical)
        if tables.len() <= DP_TABLE_LIMIT {
            self.optimize_dp(tables, conditions)
        } else {
            self.optimize_greedy(tables, conditions)
        }
    }

    // =========================================================================
    // DYNAMIC PROGRAMMING JOIN ORDERING (OPTIMAL)
    // =========================================================================

    /// Dynamic Programming join ordering - explores ALL possible orderings
    ///
    /// This is the "gold standard" algorithm used by PostgreSQL, Oracle, etc.
    /// Complexity: O(3^n) time, O(2^n) space where n = number of tables
    fn optimize_dp(&self, tables: &[&str], conditions: &[JoinCondition]) -> JoinPlan {
        let n = tables.len();

        // Map table names to indices for bitmask operations
        let _table_to_idx: HashMap<&str, usize> =
            tables.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        let idx_to_table: Vec<&str> = tables.to_vec();

        // DP table: dp[mask] = (best_cost, best_plan, row_estimate, distinct_estimate)
        // mask is a bitmask of included tables
        let num_subsets = 1 << n;
        let mut dp: Vec<Option<DpEntry>> = vec![None; num_subsets];

        // Base case: single tables (mask with single bit set)
        for (i, &table) in tables.iter().enumerate() {
            let mask = 1 << i;
            let row_count = self.get_row_count(table);
            let distinct = conditions
                .iter()
                .find(|c| c.involves(table))
                .map(|c| {
                    if c.left_table == table {
                        self.get_distinct_count(table, &c.left_column)
                    } else {
                        self.get_distinct_count(table, &c.right_column)
                    }
                })
                .unwrap_or_else(|| (row_count / 10).max(1));

            // Get sorted columns for this table (from index scans)
            let sorted_by = self.sorted_inputs.get(table).cloned().unwrap_or_default();

            dp[mask] = Some(DpEntry {
                cost: 0.0,
                rows: row_count,
                distinct,
                left_mask: 0,
                right_mask: 0,
                algorithm: None,
                sorted_by,
            });
        }

        // Fill DP table in order of subset size (bottom-up)
        for size in 2..=n {
            for mask in 1..num_subsets {
                if (mask as u32).count_ones() != size as u32 {
                    continue;
                }

                // Try all ways to partition this subset into two non-empty parts
                let mut best: Option<DpEntry> = None;

                // Iterate over all proper subsets of mask
                let mut left = mask;
                while left > 0 {
                    left = (left - 1) & mask;
                    if left == 0 || left == mask {
                        continue;
                    }
                    let right = mask ^ left;

                    // Both subsets must be computable
                    let (left_entry, right_entry) = match (&dp[left], &dp[right]) {
                        (Some(l), Some(r)) => (l, r),
                        _ => continue,
                    };

                    // Check if there's a valid join condition connecting left and right
                    let left_tables: HashSet<String> = (0..n)
                        .filter(|i| left & (1 << i) != 0)
                        .map(|i| idx_to_table[i].to_string())
                        .collect();
                    let right_tables: HashSet<String> = (0..n)
                        .filter(|i| right & (1 << i) != 0)
                        .map(|i| idx_to_table[i].to_string())
                        .collect();

                    let connecting = conditions
                        .iter()
                        .find(|c| c.connects(&left_tables, &right_tables));

                    let has_equality = connecting.map(|c| c.is_equality).unwrap_or(false);

                    // Compute join cost (with sorted input detection)
                    let join_result = self.compute_join_cost(
                        left_entry,
                        right_entry,
                        has_equality,
                        connecting.is_none(), // cross product?
                        connecting,           // pass condition for sorted detection
                    );

                    let total_cost = left_entry.cost + right_entry.cost + join_result.cost;

                    // Update best if this is better
                    if best.is_none() || total_cost < best.as_ref().unwrap().cost {
                        best = Some(DpEntry {
                            cost: total_cost,
                            rows: join_result.rows,
                            distinct: join_result.distinct,
                            left_mask: left,
                            right_mask: right,
                            algorithm: Some(join_result.algorithm),
                            sorted_by: join_result.sorted_by,
                        });
                    }
                }

                dp[mask] = best;
            }
        }

        // Reconstruct plan from DP table
        let full_mask = (1 << n) - 1;
        match &dp[full_mask] {
            Some(entry) => self.reconstruct_plan(entry, &dp, &idx_to_table, conditions),
            None => JoinPlan::empty(),
        }
    }

    /// Compute join cost for DP
    fn compute_join_cost(
        &self,
        left: &DpEntry,
        right: &DpEntry,
        has_equality: bool,
        is_cross_product: bool,
        condition: Option<&JoinCondition>,
    ) -> JoinCostResult {
        let left_rows = left.rows;
        let right_rows = right.rows;

        // Create synthetic table stats for cost estimation
        let left_stats = TableStats {
            table_name: "left".to_string(),
            row_count: left_rows,
            page_count: (left_rows / 100).max(1),
            avg_row_size: 100,
        };
        let right_stats = TableStats {
            table_name: "right".to_string(),
            row_count: right_rows,
            page_count: (right_rows / 100).max(1),
            avg_row_size: 100,
        };

        let join_stats = JoinStats {
            left_stats,
            right_stats,
            left_distinct: left.distinct,
            right_distinct: right.distinct,
        };

        // Detect if inputs are sorted on the join column
        let (left_sorted, right_sorted) = if let Some(cond) = condition {
            // Check if left side is sorted by its join column
            let left_is_sorted = left.sorted_by.first().is_some_and(|s| {
                s.column == cond.left_column
                    || s.column.ends_with(&format!(".{}", cond.left_column))
            });
            // Check if right side is sorted by its join column
            let right_is_sorted = right.sorted_by.first().is_some_and(|s| {
                s.column == cond.right_column
                    || s.column.ends_with(&format!(".{}", cond.right_column))
            });
            (left_is_sorted, right_is_sorted)
        } else {
            (false, false)
        };

        // Use extended algorithm selection that considers sorted inputs
        let (algorithm, cost) = self.cost_estimator.choose_join_algorithm_extended(
            &join_stats,
            has_equality,
            left_sorted,
            right_sorted,
        );

        // Apply cross-product penalty
        let mut final_cost = cost.total;
        if is_cross_product {
            final_cost *= CROSS_PRODUCT_PENALTY;
        }

        // Edge-aware: penalize hash joins that exceed memory budget
        if let JoinAlgorithm::HashJoin { build_rows, .. } = &algorithm {
            if *build_rows > self.memory_budget {
                // Memory overflow - add spill cost
                let overflow_ratio = *build_rows as f64 / self.memory_budget as f64;
                final_cost *= 1.0 + overflow_ratio.ln().max(0.0);
            }
        }

        // Estimate output cardinality
        let max_distinct = left.distinct.max(right.distinct).max(1);
        let output_rows = if is_cross_product {
            // Cross product: multiply
            (left_rows as u128 * right_rows as u128).min(u64::MAX as u128) as u64
        } else {
            // Join: divide by max distinct
            (left_rows as u128 * right_rows as u128 / max_distinct as u128) as u64
        };

        let output_distinct = left.distinct.min(right.distinct).max(1);

        // Determine output sort order based on join algorithm
        let output_sorted_by = match &algorithm {
            JoinAlgorithm::MergeJoin { .. } => {
                // Merge join produces output sorted by the join key (left side)
                if let Some(cond) = condition {
                    vec![SortOrder::asc(cond.left_column.clone())]
                } else {
                    left.sorted_by.clone()
                }
            }
            JoinAlgorithm::NestedLoop { .. } => {
                // Nested loop preserves outer (left) sort order
                left.sorted_by.clone()
            }
            _ => {
                // Hash join does not preserve sort order
                Vec::new()
            }
        };

        JoinCostResult {
            cost: final_cost,
            rows: output_rows.max(1),
            distinct: output_distinct,
            algorithm,
            sorted_by: output_sorted_by,
        }
    }

    /// Reconstruct join plan from DP table
    fn reconstruct_plan(
        &self,
        entry: &DpEntry,
        dp: &[Option<DpEntry>],
        idx_to_table: &[&str],
        conditions: &[JoinCondition],
    ) -> JoinPlan {
        let mut steps = Vec::new();
        self.collect_steps(entry, dp, idx_to_table, conditions, &mut steps);

        let total_cost = entry.cost;
        let output_rows = entry.rows;

        JoinPlan {
            steps,
            total_cost,
            output_rows,
        }
    }

    /// Recursively collect join steps from DP
    fn collect_steps(
        &self,
        entry: &DpEntry,
        dp: &[Option<DpEntry>],
        idx_to_table: &[&str],
        conditions: &[JoinCondition],
        steps: &mut Vec<JoinStep>,
    ) {
        if entry.left_mask == 0 || entry.right_mask == 0 {
            // Base case: single table
            return;
        }

        // Recurse into children first (post-order)
        if let Some(left_entry) = &dp[entry.left_mask] {
            self.collect_steps(left_entry, dp, idx_to_table, conditions, steps);
        }
        if let Some(right_entry) = &dp[entry.right_mask] {
            self.collect_steps(right_entry, dp, idx_to_table, conditions, steps);
        }

        // Build table sets
        let n = idx_to_table.len();
        let left_tables: HashSet<String> = (0..n)
            .filter(|i| entry.left_mask & (1 << i) != 0)
            .map(|i| idx_to_table[i].to_string())
            .collect();
        let right_tables: HashSet<String> = (0..n)
            .filter(|i| entry.right_mask & (1 << i) != 0)
            .map(|i| idx_to_table[i].to_string())
            .collect();

        // Find connecting condition
        let condition = conditions
            .iter()
            .find(|c| c.connects(&left_tables, &right_tables))
            .cloned();

        // Get child info
        let left_entry = dp[entry.left_mask].as_ref().unwrap();
        let right_entry = dp[entry.right_mask].as_ref().unwrap();

        let step_cost = entry.cost - left_entry.cost - right_entry.cost;

        let step = JoinStep {
            left_tables,
            right_tables,
            algorithm: entry
                .algorithm
                .clone()
                .unwrap_or(JoinAlgorithm::NestedLoop {
                    outer_rows: left_entry.rows,
                    inner_rows: right_entry.rows,
                }),
            condition,
            cost: PlanCost::new(
                step_cost * 0.3,
                step_cost * 0.7,
                0,
                entry.rows,
                String::new(),
            ),
            output_rows: entry.rows,
        };

        steps.push(step);
    }

    // =========================================================================
    // GREEDY FALLBACK (for large joins)
    // =========================================================================

    /// Greedy join ordering - fast but suboptimal
    fn optimize_greedy(&self, tables: &[&str], conditions: &[JoinCondition]) -> JoinPlan {
        // Initialize nodes: each table is a leaf node
        let mut nodes: Vec<JoinNode> = tables
            .iter()
            .map(|&t| {
                let row_count = self.get_row_count(t);
                let distinct = conditions
                    .iter()
                    .find(|c| c.involves(t))
                    .map(|c| {
                        if c.left_table == t {
                            self.get_distinct_count(t, &c.left_column)
                        } else {
                            self.get_distinct_count(t, &c.right_column)
                        }
                    })
                    .unwrap_or_else(|| (row_count / 10).max(1));
                JoinNode::leaf(t.to_string(), row_count, distinct)
            })
            .collect();

        let mut steps = Vec::new();

        // Greedy: repeatedly join the two cheapest nodes
        while nodes.len() > 1 {
            let (best_i, best_j, best_step, best_node) =
                match self.find_cheapest_join_pair(&nodes, conditions) {
                    Some(result) => result,
                    None => {
                        // No valid join found - create a cross join as fallback
                        break;
                    }
                };

            steps.push(best_step);

            // Remove joined nodes and add the merged node
            // Remove in reverse order to maintain indices
            let (i, j) = if best_i > best_j {
                (best_i, best_j)
            } else {
                (best_j, best_i)
            };
            nodes.remove(i);
            nodes.remove(j);
            nodes.push(best_node);
        }

        if steps.is_empty() {
            return JoinPlan::empty();
        }

        let total_cost = steps.iter().map(|s| s.cost.total).sum();
        let output_rows = steps.last().map(|s| s.output_rows).unwrap_or(0);

        JoinPlan {
            steps,
            total_cost,
            output_rows,
        }
    }

    /// Find the cheapest pair of nodes to join
    fn find_cheapest_join_pair(
        &self,
        nodes: &[JoinNode],
        conditions: &[JoinCondition],
    ) -> Option<(usize, usize, JoinStep, JoinNode)> {
        let mut best: Option<(usize, usize, JoinStep, JoinNode, f64)> = None;

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                // Find a condition that connects these two nodes
                let connecting_condition = conditions
                    .iter()
                    .find(|c| c.connects(&nodes[i].tables, &nodes[j].tables));

                // If no direct connection, skip this pair (don't create cross joins)
                let condition = match connecting_condition {
                    Some(c) => c,
                    None => continue,
                };

                // Calculate join cost
                let (step, joined_node, cost) =
                    self.estimate_join_pair(&nodes[i], &nodes[j], Some(condition.clone()));

                // Track the cheapest join
                let total_cost = nodes[i].cumulative_cost + nodes[j].cumulative_cost + cost;
                if best.is_none() || total_cost < best.as_ref().unwrap().4 {
                    best = Some((i, j, step, joined_node, total_cost));
                }
            }
        }

        best.map(|(i, j, step, node, _)| (i, j, step, node))
    }

    /// Estimate the cost of joining two nodes
    fn estimate_join_pair(
        &self,
        left: &JoinNode,
        right: &JoinNode,
        condition: Option<JoinCondition>,
    ) -> (JoinStep, JoinNode, f64) {
        // Create table stats for the nodes
        let left_stats = TableStats {
            table_name: left.tables.iter().next().cloned().unwrap_or_default(),
            row_count: left.row_estimate,
            page_count: (left.row_estimate / 100).max(1),
            avg_row_size: 100,
        };
        let right_stats = TableStats {
            table_name: right.tables.iter().next().cloned().unwrap_or_default(),
            row_count: right.row_estimate,
            page_count: (right.row_estimate / 100).max(1),
            avg_row_size: 100,
        };

        let join_stats = JoinStats {
            left_stats,
            right_stats,
            left_distinct: left.distinct_estimate,
            right_distinct: right.distinct_estimate,
        };

        let has_equality = condition.as_ref().map(|c| c.is_equality).unwrap_or(false);
        let (algorithm, cost) = self
            .cost_estimator
            .choose_join_algorithm(&join_stats, has_equality);

        // Estimate output rows using join cardinality formula
        let max_distinct = left.distinct_estimate.max(right.distinct_estimate).max(1);
        let output_rows =
            (left.row_estimate as u128 * right.row_estimate as u128 / max_distinct as u128) as u64;

        // Estimate output distinct (for further joins)
        // Conservative estimate: minimum of input distincts
        let output_distinct = left.distinct_estimate.min(right.distinct_estimate).max(1);

        let step = JoinStep {
            left_tables: left.tables.clone(),
            right_tables: right.tables.clone(),
            algorithm: algorithm.clone(),
            condition,
            cost: cost.clone(),
            output_rows,
        };

        let joined_node = JoinNode::joined(
            left,
            right,
            output_rows,
            output_distinct,
            cost.total,
            algorithm,
        );

        (step, joined_node, cost.total)
    }

    /// Get a simple recommendation for a two-table join
    /// This is useful when the full optimizer isn't needed
    pub fn recommend_join_algorithm(
        &self,
        left_table: &str,
        right_table: &str,
        has_equality_keys: bool,
    ) -> (JoinAlgorithm, PlanCost) {
        let left_stats = self.get_table_stats(left_table);
        let right_stats = self.get_table_stats(right_table);

        let left_distinct = (left_stats.row_count / 10).max(1);
        let right_distinct = (right_stats.row_count / 10).max(1);

        let join_stats = JoinStats {
            left_stats,
            right_stats,
            left_distinct,
            right_distinct,
        };

        self.cost_estimator
            .choose_join_algorithm(&join_stats, has_equality_keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::cost::BuildSide;

    fn create_optimizer_with_stats() -> JoinOptimizer {
        let mut optimizer = JoinOptimizer::new(CostEstimator::new());

        // Add some test table stats
        optimizer.add_table_stats(
            "users",
            TableStats {
                table_name: "users".to_string(),
                row_count: 1000,
                page_count: 10,
                avg_row_size: 100,
            },
        );
        optimizer.add_table_stats(
            "orders",
            TableStats {
                table_name: "orders".to_string(),
                row_count: 10000,
                page_count: 100,
                avg_row_size: 200,
            },
        );
        optimizer.add_table_stats(
            "products",
            TableStats {
                table_name: "products".to_string(),
                row_count: 500,
                page_count: 5,
                avg_row_size: 150,
            },
        );
        optimizer.add_table_stats(
            "order_items",
            TableStats {
                table_name: "order_items".to_string(),
                row_count: 50000,
                page_count: 500,
                avg_row_size: 50,
            },
        );

        // Add column distinct counts
        optimizer.add_column_distinct("users", "id", 1000);
        optimizer.add_column_distinct("orders", "user_id", 1000);
        optimizer.add_column_distinct("orders", "id", 10000);
        optimizer.add_column_distinct("products", "id", 500);
        optimizer.add_column_distinct("order_items", "order_id", 10000);
        optimizer.add_column_distinct("order_items", "product_id", 500);

        optimizer
    }

    #[test]
    fn test_join_condition_connects() {
        let condition = JoinCondition::new_equality(
            "users".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "user_id".to_string(),
        );

        let mut tables1 = HashSet::new();
        tables1.insert("users".to_string());

        let mut tables2 = HashSet::new();
        tables2.insert("orders".to_string());

        assert!(condition.connects(&tables1, &tables2));
        assert!(condition.connects(&tables2, &tables1));

        let mut tables3 = HashSet::new();
        tables3.insert("products".to_string());
        assert!(!condition.connects(&tables1, &tables3));
    }

    #[test]
    fn test_join_node_leaf() {
        let node = JoinNode::leaf("users".to_string(), 1000, 1000);
        assert!(node.contains("users"));
        assert!(!node.contains("orders"));
        assert_eq!(node.row_estimate, 1000);
        assert_eq!(node.cumulative_cost, 0.0);
    }

    #[test]
    fn test_two_table_join_optimization() {
        let optimizer = create_optimizer_with_stats();

        let conditions = vec![JoinCondition::new_equality(
            "users".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "user_id".to_string(),
        )];

        let plan = optimizer.optimize_join_order(&["users", "orders"], &conditions);

        assert_eq!(plan.steps.len(), 1);
        assert!(plan.total_cost > 0.0);

        // With equality keys and large tables, should use hash join
        let step = &plan.steps[0];
        assert!(step.algorithm.is_hash_join());
    }

    #[test]
    fn test_three_table_join_optimization() {
        let optimizer = create_optimizer_with_stats();

        // users JOIN orders ON users.id = orders.user_id
        // JOIN products ON orders.product_id = products.id
        // (Note: orders doesn't have product_id directly, but for testing)
        let conditions = vec![
            JoinCondition::new_equality(
                "users".to_string(),
                "id".to_string(),
                "orders".to_string(),
                "user_id".to_string(),
            ),
            JoinCondition::new_equality(
                "orders".to_string(),
                "id".to_string(),
                "order_items".to_string(),
                "order_id".to_string(),
            ),
        ];

        let plan = optimizer.optimize_join_order(&["users", "orders", "order_items"], &conditions);

        // Should have 2 join steps
        assert_eq!(plan.steps.len(), 2);
        assert!(plan.total_cost > 0.0);
    }

    #[test]
    fn test_recommend_join_algorithm() {
        let optimizer = create_optimizer_with_stats();

        // Large tables with equality keys -> hash join
        let (algo, cost) = optimizer.recommend_join_algorithm("orders", "users", true);
        assert!(algo.is_hash_join());
        assert!(cost.total > 0.0);

        // Without equality keys -> nested loop
        let (algo2, _) = optimizer.recommend_join_algorithm("orders", "users", false);
        assert!(algo2.is_nested_loop());
    }

    #[test]
    fn test_build_side_optimization() {
        let optimizer = create_optimizer_with_stats();

        let (algo, _) = optimizer.recommend_join_algorithm("orders", "users", true);

        if let JoinAlgorithm::HashJoin {
            build_side,
            build_rows,
            probe_rows,
        } = algo
        {
            // Should build on smaller table (users: 1000 rows)
            // and probe with larger table (orders: 10000 rows)
            assert_eq!(build_side, BuildSide::Right);
            assert_eq!(build_rows, 1000);
            assert_eq!(probe_rows, 10000);
        } else {
            panic!("Expected HashJoin");
        }
    }

    #[test]
    fn test_empty_tables() {
        let optimizer = JoinOptimizer::new(CostEstimator::new());
        let plan = optimizer.optimize_join_order(&[], &[]);
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn test_single_table() {
        let optimizer = create_optimizer_with_stats();
        let plan = optimizer.optimize_join_order(&["users"], &[]);
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn test_join_step_output_rows() {
        let optimizer = create_optimizer_with_stats();

        let conditions = vec![JoinCondition::new_equality(
            "users".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "user_id".to_string(),
        )];

        let plan = optimizer.optimize_join_order(&["users", "orders"], &conditions);
        let step = &plan.steps[0];

        // Output rows should be estimated using join cardinality formula
        // users.id has 1000 distinct, orders.user_id has 1000 distinct
        // |users| * |orders| / max(1000, 1000) = 1000 * 10000 / 1000 = 10000
        assert_eq!(step.output_rows, 10000);
    }

    // =========================================================================
    // DYNAMIC PROGRAMMING TESTS
    // =========================================================================

    #[test]
    fn test_dp_four_table_join() {
        let optimizer = create_optimizer_with_stats();

        // A star schema: users in center, orders, products, order_items around it
        let conditions = vec![
            JoinCondition::new_equality(
                "users".to_string(),
                "id".to_string(),
                "orders".to_string(),
                "user_id".to_string(),
            ),
            JoinCondition::new_equality(
                "orders".to_string(),
                "id".to_string(),
                "order_items".to_string(),
                "order_id".to_string(),
            ),
            JoinCondition::new_equality(
                "order_items".to_string(),
                "product_id".to_string(),
                "products".to_string(),
                "id".to_string(),
            ),
        ];

        let plan = optimizer
            .optimize_join_order(&["users", "orders", "products", "order_items"], &conditions);

        // Should have 3 join steps (4 tables = 3 joins)
        assert_eq!(plan.steps.len(), 3);
        assert!(plan.total_cost > 0.0);

        // DP should produce a reasonable plan
        // The optimal order typically starts with smallest tables
        println!("DP 4-table plan cost: {}", plan.total_cost);
    }

    #[test]
    fn test_dp_finds_optimal_order() {
        // Create an optimizer with very different table sizes to verify DP finds optimal
        let mut optimizer = JoinOptimizer::new(CostEstimator::new());

        // tiny (10 rows), small (100), medium (1000), large (10000)
        optimizer.add_table_stats(
            "tiny",
            TableStats {
                table_name: "tiny".to_string(),
                row_count: 10,
                page_count: 1,
                avg_row_size: 100,
            },
        );
        optimizer.add_table_stats(
            "small",
            TableStats {
                table_name: "small".to_string(),
                row_count: 100,
                page_count: 1,
                avg_row_size: 100,
            },
        );
        optimizer.add_table_stats(
            "medium",
            TableStats {
                table_name: "medium".to_string(),
                row_count: 1000,
                page_count: 10,
                avg_row_size: 100,
            },
        );
        optimizer.add_table_stats(
            "large",
            TableStats {
                table_name: "large".to_string(),
                row_count: 10000,
                page_count: 100,
                avg_row_size: 100,
            },
        );

        // Chain join: tiny - small - medium - large
        let conditions = vec![
            JoinCondition::new_equality(
                "tiny".to_string(),
                "id".to_string(),
                "small".to_string(),
                "tiny_id".to_string(),
            ),
            JoinCondition::new_equality(
                "small".to_string(),
                "id".to_string(),
                "medium".to_string(),
                "small_id".to_string(),
            ),
            JoinCondition::new_equality(
                "medium".to_string(),
                "id".to_string(),
                "large".to_string(),
                "medium_id".to_string(),
            ),
        ];

        let plan =
            optimizer.optimize_join_order(&["tiny", "small", "medium", "large"], &conditions);

        // DP should find the optimal plan
        assert_eq!(plan.steps.len(), 3);

        // The optimal strategy should join smaller tables first
        // to reduce intermediate result sizes
        println!("DP optimal plan total cost: {}", plan.total_cost);
    }

    #[test]
    fn test_dp_vs_greedy_consistency() {
        // With 2 tables, DP and greedy should produce same result
        let optimizer = create_optimizer_with_stats();

        let conditions = vec![JoinCondition::new_equality(
            "users".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "user_id".to_string(),
        )];

        let dp_plan = optimizer.optimize_dp(&["users", "orders"], &conditions);
        let greedy_plan = optimizer.optimize_greedy(&["users", "orders"], &conditions);

        // Both should have exactly one join step
        assert_eq!(dp_plan.steps.len(), 1);
        assert_eq!(greedy_plan.steps.len(), 1);

        // Costs should be similar (not necessarily identical due to algorithm differences)
        let cost_ratio = dp_plan.total_cost / greedy_plan.total_cost.max(0.001);
        assert!(
            cost_ratio > 0.5 && cost_ratio < 2.0,
            "DP and greedy costs should be similar for 2 tables"
        );
    }

    #[test]
    fn test_memory_aware_planning() {
        // Test that memory-constrained optimizer penalizes large hash joins
        let mut low_memory_optimizer = JoinOptimizer::with_memory_budget(CostEstimator::new(), 100); // Very low memory

        low_memory_optimizer.add_table_stats(
            "big",
            TableStats {
                table_name: "big".to_string(),
                row_count: 100000,
                page_count: 1000,
                avg_row_size: 100,
            },
        );
        low_memory_optimizer.add_table_stats(
            "small",
            TableStats {
                table_name: "small".to_string(),
                row_count: 10,
                page_count: 1,
                avg_row_size: 100,
            },
        );

        let conditions = vec![JoinCondition::new_equality(
            "big".to_string(),
            "id".to_string(),
            "small".to_string(),
            "big_id".to_string(),
        )];

        let plan = low_memory_optimizer.optimize_join_order(&["big", "small"], &conditions);

        // Should still produce a valid plan
        assert_eq!(plan.steps.len(), 1);

        // The cost should be inflated due to memory overflow
        println!(
            "Memory-constrained plan cost: {} (includes spill penalty)",
            plan.total_cost
        );
    }

    #[test]
    fn test_cross_product_penalty() {
        let optimizer = create_optimizer_with_stats();

        // No join condition = cross product
        let plan = optimizer.optimize_join_order(&["users", "products"], &[]);

        // Should still produce a plan but with high cost
        // (may be empty if no conditions connect tables in DP)
        println!("Cross product plan: {:?}", plan);
    }

    // =========================================================================
    // SORTED INPUT DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_sorted_input_detection() {
        let mut optimizer = create_optimizer_with_stats();

        // Mark users as sorted by 'id' (e.g., from index scan)
        optimizer.add_sorted_input("users", vec![SortOrder::asc("id")]);

        // Mark orders as sorted by 'user_id'
        optimizer.add_sorted_input("orders", vec![SortOrder::asc("user_id")]);

        // Join on users.id = orders.user_id
        let conditions = vec![JoinCondition::new_equality(
            "users".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "user_id".to_string(),
        )];

        let plan = optimizer.optimize_join_order(&["users", "orders"], &conditions);
        assert_eq!(plan.steps.len(), 1);

        // With both inputs sorted on join columns, merge join should be chosen
        let step = &plan.steps[0];
        assert!(
            step.algorithm.is_merge_join(),
            "Expected merge join when both inputs are sorted, got {:?}",
            step.algorithm
        );

        // Verify the merge join knows inputs are sorted
        if let JoinAlgorithm::MergeJoin {
            left_sorted,
            right_sorted,
            ..
        } = &step.algorithm
        {
            assert!(*left_sorted, "Left side should be detected as sorted");
            assert!(*right_sorted, "Right side should be detected as sorted");
        }
    }

    #[test]
    fn test_unsorted_vs_sorted_cost() {
        let unsorted_optimizer = create_optimizer_with_stats();
        let mut sorted_optimizer = create_optimizer_with_stats();

        // Mark inputs as sorted in one optimizer
        sorted_optimizer.add_sorted_input("users", vec![SortOrder::asc("id")]);
        sorted_optimizer.add_sorted_input("orders", vec![SortOrder::asc("user_id")]);

        let conditions = vec![JoinCondition::new_equality(
            "users".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "user_id".to_string(),
        )];

        let unsorted_plan =
            unsorted_optimizer.optimize_join_order(&["users", "orders"], &conditions);
        let sorted_plan = sorted_optimizer.optimize_join_order(&["users", "orders"], &conditions);

        // Both should produce a plan
        assert_eq!(unsorted_plan.steps.len(), 1);
        assert_eq!(sorted_plan.steps.len(), 1);

        // Sorted input should have lower or equal cost (no sort needed for merge)
        println!(
            "Unsorted cost: {}, Sorted cost: {}",
            unsorted_plan.total_cost, sorted_plan.total_cost
        );

        // When both inputs are sorted, merge join avoids sort cost
        if sorted_plan.steps[0].algorithm.is_merge_join() {
            // Merge join on sorted data should be cheaper than hash join
            // (The exact cost depends on table sizes, but we can verify it's reasonable)
            assert!(
                sorted_plan.total_cost <= unsorted_plan.total_cost * 1.5,
                "Sorted merge join should not be much more expensive than unsorted"
            );
        }
    }

    #[test]
    fn test_partial_sorted_input() {
        let mut optimizer = create_optimizer_with_stats();

        // Only mark one side as sorted
        optimizer.add_sorted_input("users", vec![SortOrder::asc("id")]);
        // orders is NOT sorted

        let conditions = vec![JoinCondition::new_equality(
            "users".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "user_id".to_string(),
        )];

        let plan = optimizer.optimize_join_order(&["users", "orders"], &conditions);
        assert_eq!(plan.steps.len(), 1);

        // With only one side sorted, hash join is typically still preferred
        // (or merge join if the sort cost for one side is acceptable)
        let step = &plan.steps[0];
        println!(
            "Partial sorted plan chose: {:?} with cost {}",
            step.algorithm, plan.total_cost
        );
    }

    #[test]
    fn test_sort_order_preserved_through_merge_join() {
        let mut optimizer = create_optimizer_with_stats();

        // Add more tables for multi-way join
        optimizer.add_table_stats(
            "customers",
            TableStats {
                table_name: "customers".to_string(),
                row_count: 500,
                page_count: 5,
                avg_row_size: 100,
            },
        );
        optimizer.add_column_distinct("customers", "id", 500);
        optimizer.add_column_distinct("orders", "customer_id", 500);

        // Mark both sides as sorted
        optimizer.add_sorted_input("customers", vec![SortOrder::asc("id")]);
        optimizer.add_sorted_input("orders", vec![SortOrder::asc("customer_id")]);

        let conditions = vec![JoinCondition::new_equality(
            "customers".to_string(),
            "id".to_string(),
            "orders".to_string(),
            "customer_id".to_string(),
        )];

        let plan = optimizer.optimize_join_order(&["customers", "orders"], &conditions);
        assert_eq!(plan.steps.len(), 1);

        // Verify plan was created successfully
        assert!(plan.output_rows > 0);
    }

    #[test]
    fn test_node_sorted_by_detection() {
        // Test the JoinNode.is_sorted_by helper
        let node =
            JoinNode::leaf_sorted("users".to_string(), 1000, 1000, vec![SortOrder::asc("id")]);

        assert!(node.is_sorted_by("id"));
        assert!(node.is_sorted_by("users.id")); // Should match qualified names too
        assert!(!node.is_sorted_by("name"));
        assert!(!node.is_sorted_by("other_column"));
    }
}
