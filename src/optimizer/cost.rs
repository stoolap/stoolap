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

//! Cost model for query optimization
//!
//! This module provides cost estimation for different access methods:
//! - Sequential scan: Full table scan
//! - Index scan: B-tree, hash, or bitmap index lookup
//! - Index-only scan: When index covers all needed columns
//! - Primary key lookup: Direct row access by primary key
//! - Join methods: Hash Join, Merge Join, Nested Loop Join, Semi-Join
//!
//! The cost model uses statistics collected by ANALYZE to estimate:
//! - Number of rows to process
//! - I/O cost (page reads)
//! - CPU cost (tuple processing)
//! - Join cardinality and cost
//!
//! ## Cost Model Approximations
//!
//! This cost model uses simplified approximations suitable for an embedded
//! in-memory database. The constants are relative units, not wall-clock time.
//!
//! ### Key Approximations:
//!
//! 1. **CPU costs** (`cpu_tuple_cost`, `cpu_operator_cost`, etc.) are arbitrary
//!    relative values tuned empirically. They capture the relative expense of
//!    different operations but do not correspond to actual CPU cycles.
//!
//! 2. **I/O costs** (`seq_page_cost`, `random_page_cost`) assume in-memory
//!    operation. For disk-based scenarios, random_page_cost should be 4-10x
//!    higher than seq_page_cost (not 2x as configured).
//!
//! 3. **Index type costs** are approximate:
//!    - B-tree: Uses `log2(cardinality)` for tree height estimation
//!    - Hash: Assumes O(1) lookup with fixed constant
//!    - Bitmap: Assumes O(n/64) bitwise scan cost
//!    - MultiColumn: Hybrid hash lookup model
//!
//! 4. **Parallel execution costs** assume Rayon's work-stealing scheduler.
//!    The startup cost (10.0) and per-tuple overhead (0.001) are empirical.
//!
//! 5. **Selectivity estimation** uses histogram-based estimates when available
//!    from ANALYZE, otherwise falls back to heuristics (10% for equality, etc.)
//!
//! ### Calibration
//!
//! For production tuning, run representative queries with EXPLAIN ANALYZE
//! to compare estimated vs actual costs, then adjust constants accordingly.

use crate::core::IndexType;
use crate::executor::{
    DEFAULT_PARALLEL_FILTER_THRESHOLD, DEFAULT_PARALLEL_JOIN_THRESHOLD,
    DEFAULT_PARALLEL_SORT_THRESHOLD,
};
use crate::storage::statistics::{
    ColumnCorrelations, ColumnStats, SelectivityEstimator, TableStats,
};

/// Cost constants for query optimization
/// These values are tuned for in-memory/edge computing scenarios
#[derive(Debug, Clone)]
pub struct CostConstants {
    /// Cost to process one tuple during sequential scan
    pub cpu_tuple_cost: f64,

    /// Cost to process one index entry
    pub cpu_index_tuple_cost: f64,

    /// Cost to evaluate one operator (comparison, etc.)
    pub cpu_operator_cost: f64,

    /// Cost to read one page sequentially
    pub seq_page_cost: f64,

    /// Cost to read one page randomly (index access)
    pub random_page_cost: f64,

    /// Cost for a primary key lookup (very fast, O(1))
    pub pk_lookup_cost: f64,

    /// Assumed index height for B-tree indexes
    pub default_index_height: u32,

    /// Page size in bytes (for page count estimation)
    pub page_size: usize,

    /// Minimum cost threshold - costs below this are considered negligible
    pub min_cost: f64,

    // Join cost constants
    /// Cost to hash one row and insert into hash table
    pub hash_build_cost: f64,

    /// Cost to probe hash table (hash + lookup)
    pub hash_probe_cost: f64,

    /// Memory overhead factor for hash table (accounts for load factor, pointers)
    pub hash_memory_factor: f64,

    /// Cost for one comparison in nested loop join
    pub nested_loop_compare_cost: f64,

    /// Cost to compare two keys in merge join
    pub merge_compare_cost: f64,

    /// Cost per row for sorting (N log N amortized)
    pub sort_cost_per_row: f64,

    /// Semi-join early termination benefit factor (0-1)
    /// Lower means more benefit from early termination
    pub semi_join_benefit: f64,

    // Parallel execution constants
    /// Whether parallel execution is enabled in cost model
    pub parallel_enabled: bool,

    /// Number of parallel workers (0 = auto-detect from CPU cores)
    pub parallel_workers: usize,

    /// Minimum rows to consider parallel scan
    pub parallel_scan_threshold: u64,

    /// Minimum rows to consider parallel filter
    pub parallel_filter_threshold: u64,

    /// Minimum build rows to consider parallel hash join
    pub parallel_join_threshold: u64,

    /// Minimum rows to consider parallel sort
    pub parallel_sort_threshold: u64,

    /// Parallel startup cost (thread pool overhead)
    pub parallel_startup_cost: f64,

    /// Per-tuple overhead for parallel execution (coordination cost)
    pub parallel_tuple_cost: f64,
}

impl Default for CostConstants {
    fn default() -> Self {
        DEFAULT_COST_CONSTANTS
    }
}

/// Default cost constants optimized for in-memory edge computing
pub const DEFAULT_COST_CONSTANTS: CostConstants = CostConstants {
    // CPU costs (relative units)
    cpu_tuple_cost: 0.01,        // Cost per tuple in sequential scan
    cpu_index_tuple_cost: 0.005, // Cost per index entry (cheaper than tuple)
    cpu_operator_cost: 0.0025,   // Cost per operator evaluation

    // I/O costs (relative units) - lower for in-memory
    seq_page_cost: 1.0,    // Sequential page read
    random_page_cost: 2.0, // Random page read (slightly more expensive)

    // Special access costs
    pk_lookup_cost: 1.0, // Primary key lookup is very cheap

    // Index assumptions
    default_index_height: 3, // Typical B-tree height

    // Page size
    page_size: 8192, // 8KB pages

    // Minimum cost
    min_cost: 0.0001,

    // Join costs
    hash_build_cost: 0.02,          // Cost to hash and insert one row
    hash_probe_cost: 0.01,          // Cost to probe hash table
    hash_memory_factor: 1.5,        // Memory overhead for hash table
    nested_loop_compare_cost: 0.01, // Cost per comparison in nested loop
    merge_compare_cost: 0.005,      // Cost to compare keys in merge join (cheaper than hash)
    sort_cost_per_row: 0.03,        // Cost per row for sorting (N log N amortized)
    semi_join_benefit: 0.5,         // Semi-join typically scans ~50% before early termination

    // Parallel execution costs (use constants from executor::parallel as single source of truth)
    parallel_enabled: true, // Parallel execution enabled by default
    parallel_workers: 0,    // 0 = auto-detect from rayon (CPU cores)
    parallel_scan_threshold: DEFAULT_PARALLEL_FILTER_THRESHOLD as u64, // Min rows for parallel scan
    parallel_filter_threshold: DEFAULT_PARALLEL_FILTER_THRESHOLD as u64, // Min rows for parallel filter
    parallel_join_threshold: DEFAULT_PARALLEL_JOIN_THRESHOLD as u64, // Min rows for parallel hash join build
    parallel_sort_threshold: DEFAULT_PARALLEL_SORT_THRESHOLD as u64, // Min rows for parallel sort
    parallel_startup_cost: 10.0, // One-time cost for parallel execution
    parallel_tuple_cost: 0.001,  // Per-tuple overhead for coordination
};

/// Represents the estimated cost of a query plan
#[derive(Debug, Clone)]
pub struct PlanCost {
    /// Total estimated cost
    pub total: f64,

    /// Startup cost (one-time cost before returning first row)
    pub startup: f64,

    /// Per-row cost (cost to fetch each additional row)
    pub per_row: f64,

    /// Estimated number of rows returned
    pub estimated_rows: u64,

    /// Estimated number of pages accessed
    pub estimated_pages: u64,

    /// Description of how cost was calculated
    pub explanation: String,
}

impl PlanCost {
    /// Create a new plan cost (total is computed as startup + per_row * estimated_rows)
    pub fn new(
        startup: f64,
        per_row: f64,
        estimated_rows: u64,
        estimated_pages: u64,
        explanation: String,
    ) -> Self {
        let total = startup + per_row * estimated_rows as f64;
        Self {
            total,
            startup,
            per_row,
            estimated_rows,
            estimated_pages,
            explanation,
        }
    }

    /// Create a plan cost with explicit total (for cases where simple formula doesn't apply)
    pub fn with_total(
        total: f64,
        startup: f64,
        per_row: f64,
        estimated_rows: u64,
        estimated_pages: u64,
        explanation: String,
    ) -> Self {
        Self {
            total,
            startup,
            per_row,
            estimated_rows,
            estimated_pages,
            explanation,
        }
    }

    /// Create a zero-cost plan (for cached/trivial results)
    pub fn zero() -> Self {
        Self {
            total: 0.0,
            startup: 0.0,
            per_row: 0.0,
            estimated_rows: 0,
            estimated_pages: 0,
            explanation: "No cost (cached/trivial)".to_string(),
        }
    }

    /// Compare costs - returns true if self is cheaper than other
    pub fn is_cheaper_than(&self, other: &PlanCost) -> bool {
        self.total < other.total
    }
}

/// Access method types for cost estimation
#[derive(Debug, Clone, PartialEq)]
pub enum AccessMethod {
    /// Full sequential scan of the table
    SeqScan,

    /// Primary key lookup (equality on PK)
    PkLookup,

    /// Single index scan
    IndexScan {
        index_name: String,
        column: String,
        selectivity: f64,
        index_type: IndexType,
    },

    /// Index-only scan (covering index - no table access needed)
    /// This is significantly cheaper than regular index scan
    IndexOnlyScan {
        index_name: String,
        columns: Vec<String>,
        selectivity: f64,
        index_type: IndexType,
    },

    /// Multiple indexes combined with AND
    MultiIndexAnd {
        indexes: Vec<(String, String, f64, IndexType)>, // (name, column, selectivity, type)
    },

    /// Multiple indexes combined with OR
    MultiIndexOr {
        indexes: Vec<(String, String, f64, IndexType)>, // (name, column, selectivity, type)
    },
}

impl AccessMethod {
    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            AccessMethod::SeqScan => "Sequential Scan".to_string(),
            AccessMethod::PkLookup => "PK Lookup".to_string(),
            AccessMethod::IndexScan {
                index_name,
                index_type,
                ..
            } => {
                format!(
                    "{} Index Scan using {}",
                    index_type.as_str().to_uppercase(),
                    index_name
                )
            }
            AccessMethod::IndexOnlyScan {
                index_name,
                index_type,
                ..
            } => {
                format!(
                    "{} Index Only Scan using {}",
                    index_type.as_str().to_uppercase(),
                    index_name
                )
            }
            AccessMethod::MultiIndexAnd { indexes } => {
                let names: Vec<_> = indexes.iter().map(|(n, _, _, _)| n.as_str()).collect();
                format!("Multi-Index Scan (AND): {}", names.join(", "))
            }
            AccessMethod::MultiIndexOr { indexes } => {
                let names: Vec<_> = indexes.iter().map(|(n, _, _, _)| n.as_str()).collect();
                format!("Multi-Index Scan (OR): {}", names.join(", "))
            }
        }
    }
}

/// Which side of the join to use for building the hash table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildSide {
    /// Build hash table from left input
    Left,
    /// Build hash table from right input
    Right,
}

impl BuildSide {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            BuildSide::Left => "Left",
            BuildSide::Right => "Right",
        }
    }
}

/// Join algorithm types for cost estimation
#[derive(Debug, Clone, PartialEq)]
pub enum JoinAlgorithm {
    /// Hash join: O(N + M) complexity
    /// Build hash table on smaller table, probe with larger table
    HashJoin {
        /// Which side to build the hash table from
        build_side: BuildSide,
        /// Estimated build rows (from the build side)
        build_rows: u64,
        /// Estimated probe rows (from the probe side)
        probe_rows: u64,
    },

    /// Merge join: O(N + M) complexity with O(N log N + M log M) sort
    /// Efficient when inputs are already sorted or can be sorted efficiently
    MergeJoin {
        /// Left table rows
        left_rows: u64,
        /// Right table rows
        right_rows: u64,
        /// True if left input is already sorted
        left_sorted: bool,
        /// True if right input is already sorted
        right_sorted: bool,
    },

    /// Nested loop join: O(N * M) complexity
    /// Used when no equality join keys or for very small tables
    NestedLoop {
        /// Outer table rows
        outer_rows: u64,
        /// Inner table rows
        inner_rows: u64,
    },

    /// Semi-join: Returns rows from left that have at least one match in right
    /// Used for EXISTS subqueries - can terminate early on first match
    SemiJoin {
        /// Algorithm used for the semi-join
        inner_algorithm: Box<JoinAlgorithm>,
        /// Estimated rows from left that will be checked
        left_rows: u64,
        /// Estimated selectivity (fraction of left rows that match)
        selectivity: f64,
    },

    /// Anti-join: Returns rows from left that have NO match in right
    /// Used for NOT EXISTS / NOT IN subqueries
    AntiJoin {
        /// Algorithm used for the anti-join
        inner_algorithm: Box<JoinAlgorithm>,
        /// Estimated rows from left that will be checked
        left_rows: u64,
        /// Estimated selectivity (fraction of left rows that don't match)
        selectivity: f64,
    },
}

impl JoinAlgorithm {
    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            JoinAlgorithm::HashJoin { build_side, .. } => {
                format!("Hash Join (build: {})", build_side.description())
            }
            JoinAlgorithm::MergeJoin {
                left_sorted,
                right_sorted,
                ..
            } => {
                let sort_info = match (left_sorted, right_sorted) {
                    (true, true) => " (both sorted)",
                    (true, false) => " (left sorted)",
                    (false, true) => " (right sorted)",
                    (false, false) => "",
                };
                format!("Merge Join{}", sort_info)
            }
            JoinAlgorithm::NestedLoop { .. } => "Nested Loop".to_string(),
            JoinAlgorithm::SemiJoin {
                inner_algorithm, ..
            } => {
                format!("Semi-Join ({})", inner_algorithm.description())
            }
            JoinAlgorithm::AntiJoin {
                inner_algorithm, ..
            } => {
                format!("Anti-Join ({})", inner_algorithm.description())
            }
        }
    }

    /// Check if this is a hash join
    pub fn is_hash_join(&self) -> bool {
        matches!(self, JoinAlgorithm::HashJoin { .. })
    }

    /// Check if this is a merge join
    pub fn is_merge_join(&self) -> bool {
        matches!(self, JoinAlgorithm::MergeJoin { .. })
    }

    /// Check if this is a nested loop join
    pub fn is_nested_loop(&self) -> bool {
        matches!(self, JoinAlgorithm::NestedLoop { .. })
    }

    /// Check if this is a semi-join
    pub fn is_semi_join(&self) -> bool {
        matches!(self, JoinAlgorithm::SemiJoin { .. })
    }

    /// Check if this is an anti-join
    pub fn is_anti_join(&self) -> bool {
        matches!(self, JoinAlgorithm::AntiJoin { .. })
    }
}

/// Statistics needed for join cost estimation
#[derive(Debug, Clone)]
pub struct JoinStats {
    /// Left table statistics
    pub left_stats: TableStats,
    /// Right table statistics
    pub right_stats: TableStats,
    /// Distinct count on left join column (0 if unknown)
    pub left_distinct: u64,
    /// Distinct count on right join column (0 if unknown)
    pub right_distinct: u64,
}

/// Cost estimator for query planning
pub struct CostEstimator {
    constants: CostConstants,
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostEstimator {
    /// Create a new cost estimator with default constants
    pub fn new() -> Self {
        Self {
            constants: DEFAULT_COST_CONSTANTS,
        }
    }

    /// Create a cost estimator with custom constants
    pub fn with_constants(constants: CostConstants) -> Self {
        Self { constants }
    }

    /// Estimate cost of a sequential scan
    pub fn estimate_seq_scan(&self, table_stats: &TableStats) -> PlanCost {
        let row_count = table_stats.row_count;
        let page_count = table_stats.page_count.max(1);

        // I/O cost: read all pages sequentially
        let io_cost = page_count as f64 * self.constants.seq_page_cost;

        // CPU cost: process each tuple
        let cpu_cost = row_count as f64 * self.constants.cpu_tuple_cost;

        let total = io_cost + cpu_cost;

        PlanCost::new(
            io_cost, // Startup: read first page
            self.constants.cpu_tuple_cost,
            row_count,
            page_count,
            format!(
                "Seq Scan: {} pages × {:.2} + {} rows × {:.4} = {:.2}",
                page_count,
                self.constants.seq_page_cost,
                row_count,
                self.constants.cpu_tuple_cost,
                total
            ),
        )
    }

    /// Estimate cost of a sequential scan with filter (reduces output rows)
    pub fn estimate_seq_scan_with_filter(
        &self,
        table_stats: &TableStats,
        selectivity: f64,
    ) -> PlanCost {
        let row_count = table_stats.row_count;
        let page_count = table_stats.page_count.max(1);

        // I/O cost: still read all pages
        let io_cost = page_count as f64 * self.constants.seq_page_cost;

        // CPU cost: process ALL tuples + filter evaluation (we scan everything)
        let cpu_cost =
            row_count as f64 * (self.constants.cpu_tuple_cost + self.constants.cpu_operator_cost);

        // Output rows (filtered result)
        let output_rows = (row_count as f64 * selectivity).max(1.0) as u64;

        // Total cost: we must scan ALL rows even though output is filtered
        let total = io_cost + cpu_cost;

        PlanCost::with_total(
            total,
            io_cost,
            self.constants.cpu_tuple_cost + self.constants.cpu_operator_cost,
            output_rows,
            page_count,
            format!(
                "Seq Scan + Filter: {} rows × {:.2} selectivity = {} output rows, cost {:.2}",
                row_count, selectivity, output_rows, total
            ),
        )
    }

    /// Get number of parallel workers (auto-detect if 0)
    fn parallel_workers(&self) -> usize {
        if self.constants.parallel_workers == 0 {
            rayon::current_num_threads()
        } else {
            self.constants.parallel_workers
        }
    }

    /// Check if parallel scan should be used for given row count
    pub fn should_parallel_scan(&self, row_count: u64) -> bool {
        self.constants.parallel_enabled && row_count >= self.constants.parallel_scan_threshold
    }

    /// Check if parallel filter should be used for given row count
    pub fn should_parallel_filter(&self, row_count: u64) -> bool {
        self.constants.parallel_enabled && row_count >= self.constants.parallel_filter_threshold
    }

    /// Check if parallel join should be used for given build rows
    pub fn should_parallel_join(&self, build_rows: u64) -> bool {
        self.constants.parallel_enabled && build_rows >= self.constants.parallel_join_threshold
    }

    /// Check if parallel sort should be used for given row count
    pub fn should_parallel_sort(&self, row_count: u64) -> bool {
        self.constants.parallel_enabled && row_count >= self.constants.parallel_sort_threshold
    }

    /// Compute the parallel speedup factor based on row count and workers
    /// Takes into account Amdahl's law - not all operations parallelize perfectly
    fn parallel_speedup_factor(&self, row_count: u64) -> f64 {
        if !self.constants.parallel_enabled {
            return 1.0;
        }

        let workers = self.parallel_workers() as f64;
        if workers <= 1.0 {
            return 1.0;
        }

        // Parallel fraction: larger datasets have better parallelization
        // Small: ~70% parallel, Large: ~90% parallel
        let parallel_fraction = if row_count < 10_000 {
            0.7
        } else if row_count < 100_000 {
            0.8
        } else {
            0.9
        };

        // Amdahl's law: speedup = 1 / ((1 - p) + p/n)
        // where p = parallel fraction, n = workers
        let sequential_fraction = 1.0 - parallel_fraction;
        let speedup = 1.0 / (sequential_fraction + parallel_fraction / workers);

        // Cap speedup at realistic values (never more than 80% of theoretical max)
        speedup.min(workers * 0.8)
    }

    /// Estimate cost of a parallel sequential scan with filter
    pub fn estimate_parallel_seq_scan_with_filter(
        &self,
        table_stats: &TableStats,
        selectivity: f64,
    ) -> PlanCost {
        let row_count = table_stats.row_count;
        let page_count = table_stats.page_count.max(1);

        // Check if parallel is beneficial
        if !self.should_parallel_filter(row_count) {
            return self.estimate_seq_scan_with_filter(table_stats, selectivity);
        }

        let workers = self.parallel_workers();
        let speedup = self.parallel_speedup_factor(row_count);

        // I/O cost: still read all pages (I/O is not parallelized much)
        let io_cost = page_count as f64 * self.constants.seq_page_cost;

        // CPU cost: parallelized across workers
        let base_cpu_cost =
            row_count as f64 * (self.constants.cpu_tuple_cost + self.constants.cpu_operator_cost);
        let parallel_cpu_cost = base_cpu_cost / speedup;

        // Parallel overhead: startup + per-tuple coordination
        let parallel_overhead = self.constants.parallel_startup_cost
            + row_count as f64 * self.constants.parallel_tuple_cost;

        // Output rows (filtered result)
        let output_rows = (row_count as f64 * selectivity).max(1.0) as u64;

        // Total cost
        let total = io_cost + parallel_cpu_cost + parallel_overhead;

        PlanCost::with_total(
            total,
            io_cost + self.constants.parallel_startup_cost, // Startup includes parallel init
            (self.constants.cpu_tuple_cost + self.constants.cpu_operator_cost) / speedup
                + self.constants.parallel_tuple_cost,
            output_rows,
            page_count,
            format!(
                "Parallel Seq Scan + Filter ({} workers): {} rows × {:.2} selectivity = {} output rows, cost {:.2} (speedup {:.1}x)",
                workers, row_count, selectivity, output_rows, total, speedup
            ),
        )
    }

    /// Compare parallel vs sequential scan costs and return the better option
    pub fn choose_scan_method(
        &self,
        table_stats: &TableStats,
        selectivity: f64,
    ) -> (PlanCost, bool) {
        let sequential = self.estimate_seq_scan_with_filter(table_stats, selectivity);

        if !self.constants.parallel_enabled
            || table_stats.row_count < self.constants.parallel_filter_threshold
        {
            return (sequential, false);
        }

        let parallel = self.estimate_parallel_seq_scan_with_filter(table_stats, selectivity);

        if parallel.is_cheaper_than(&sequential) {
            (parallel, true)
        } else {
            (sequential, false)
        }
    }

    /// Estimate cost of a primary key lookup
    pub fn estimate_pk_lookup(&self) -> PlanCost {
        // PK lookup is O(1) - very cheap
        PlanCost::new(
            self.constants.pk_lookup_cost,
            0.0,
            1, // Always returns at most 1 row
            1, // Access at most 1 page
            format!("PK Lookup: cost {:.2}", self.constants.pk_lookup_cost),
        )
    }

    /// Estimate cost of an index scan
    pub fn estimate_index_scan(
        &self,
        table_stats: &TableStats,
        _column_stats: Option<&ColumnStats>,
        selectivity: f64,
        index_name: &str,
        index_type: IndexType,
    ) -> PlanCost {
        let row_count = table_stats.row_count;

        // Estimated rows from index
        let estimated_rows = (row_count as f64 * selectivity).max(1.0) as u64;

        // Index-specific traversal cost
        let (index_io_cost, leaf_io_cost) = match index_type {
            IndexType::Hash => {
                // Hash index: O(1) lookup - no tree traversal
                // Just a single hash computation and bucket lookup
                let hash_lookup_cost = self.constants.random_page_cost * 0.1; // Very cheap
                (hash_lookup_cost, 0.0)
            }
            IndexType::Bitmap => {
                // Bitmap index: Fast for equality, good for batch operations
                // No tree traversal, just bitmap lookup
                let bitmap_cost = self.constants.random_page_cost * 0.2;
                let scan_cost = (estimated_rows as f64 / 64.0).max(1.0) * 0.01; // RoaringBitmap is ~64 bits per word
                (bitmap_cost, scan_cost)
            }
            IndexType::BTree => {
                // B-tree: O(log n) traversal
                let index_height = self.constants.default_index_height;
                let btree_io_cost = index_height as f64 * self.constants.random_page_cost;
                // Leaf page scans (proportional to selectivity)
                let leaf_pages = (table_stats.page_count as f64 * selectivity).max(1.0) as u64;
                let leaf_io_cost = leaf_pages as f64 * self.constants.random_page_cost;
                (btree_io_cost, leaf_io_cost)
            }
            IndexType::MultiColumn => {
                // Multi-column hybrid: O(1) hash lookup for exact matches
                // Falls back to lazy BTree for range queries (but that's rare)
                let hash_lookup_cost = self.constants.random_page_cost * 0.15; // Slightly more than single-col hash
                (hash_lookup_cost, 0.0)
            }
        };

        // CPU cost for index entries
        let index_cpu_cost = match index_type {
            IndexType::Hash => {
                // Hash: Just hash computation + equality check
                estimated_rows as f64 * self.constants.cpu_operator_cost
            }
            IndexType::Bitmap => {
                // Bitmap: Bit operations are very fast
                estimated_rows as f64 * self.constants.cpu_operator_cost * 0.5
            }
            IndexType::BTree => estimated_rows as f64 * self.constants.cpu_index_tuple_cost,
            IndexType::MultiColumn => {
                // Multi-column: Hash computation on composite key + equality
                estimated_rows as f64 * self.constants.cpu_operator_cost * 1.2 // Slightly more than single hash
            }
        };

        // Table lookup cost: RANDOM I/O for each row fetched
        // This is the key cost - each row requires a random page read unless clustered
        // We estimate the number of distinct pages we need to access
        let pages_accessed = if estimated_rows as f64 >= table_stats.page_count as f64 {
            // If we're fetching many rows, we'll hit most pages
            table_stats.page_count as f64
        } else {
            // For fewer rows, estimate pages based on row distribution
            // Using simplified formula: min(estimated_rows, page_count)
            // since each row might be on a different page
            (estimated_rows as f64).min(table_stats.page_count as f64)
        };
        let table_io_cost = pages_accessed.max(1.0) * self.constants.random_page_cost;

        // CPU cost to process fetched tuples
        let table_cpu_cost = estimated_rows as f64 * self.constants.cpu_tuple_cost;

        let total = index_io_cost + leaf_io_cost + index_cpu_cost + table_io_cost + table_cpu_cost;
        let leaf_pages = (table_stats.page_count as f64 * selectivity).max(1.0) as u64;
        let total_pages =
            leaf_pages + self.constants.default_index_height as u64 + pages_accessed as u64;

        PlanCost::with_total(
            total,
            index_io_cost, // Startup: traverse to first leaf
            self.constants.cpu_index_tuple_cost + self.constants.cpu_tuple_cost,
            estimated_rows,
            total_pages,
            format!(
                "{} Index Scan ({}): selectivity {:.4} → {} rows, cost {:.2}",
                index_type.as_str().to_uppercase(),
                index_name,
                selectivity,
                estimated_rows,
                total
            ),
        )
    }

    /// Estimate cost of an index-only scan (covering index)
    ///
    /// Index-only scans are significantly cheaper than regular index scans
    /// because they don't need to access the table at all - all required
    /// data is stored in the index itself.
    ///
    /// This is beneficial when:
    /// - The query only needs columns that are in the index
    /// - The query is a COUNT(*) or aggregate over indexed columns
    pub fn estimate_index_only_scan(
        &self,
        table_stats: &TableStats,
        selectivity: f64,
        index_name: &str,
    ) -> PlanCost {
        let row_count = table_stats.row_count;

        // Estimated rows from index
        let estimated_rows = (row_count as f64 * selectivity).max(1.0) as u64;

        // Index traversal cost (B-tree height)
        let index_height = self.constants.default_index_height;
        let index_io_cost = index_height as f64 * self.constants.random_page_cost;

        // Leaf page scans (proportional to selectivity)
        let leaf_pages = (table_stats.page_count as f64 * selectivity).max(1.0) as u64;
        let leaf_io_cost = leaf_pages as f64 * self.constants.random_page_cost;

        // CPU cost for index entries only - NO table lookup!
        let index_cpu_cost = estimated_rows as f64 * self.constants.cpu_index_tuple_cost;

        let total = index_io_cost + leaf_io_cost + index_cpu_cost;
        let total_pages = leaf_pages + index_height as u64;

        PlanCost::with_total(
            total,
            index_io_cost,
            self.constants.cpu_index_tuple_cost,
            estimated_rows,
            total_pages,
            format!(
                "Index Only Scan ({}): selectivity {:.4} → {} rows, cost {:.2} (no table access)",
                index_name, selectivity, estimated_rows, total
            ),
        )
    }

    /// Estimate cost of multiple indexes combined with AND
    pub fn estimate_multi_index_and(
        &self,
        table_stats: &TableStats,
        selectivities: &[(String, f64)], // (index_name, selectivity)
    ) -> PlanCost {
        if selectivities.is_empty() {
            return self.estimate_seq_scan(table_stats);
        }

        // Combined selectivity for AND is product of individual selectivities
        let combined_selectivity: f64 = selectivities.iter().map(|(_, s)| s).product();
        let row_count = table_stats.row_count;
        let estimated_rows = (row_count as f64 * combined_selectivity).max(1.0) as u64;

        // Cost for each index traversal
        let mut total_index_cost = 0.0;
        let mut total_pages = 0u64;

        for (_, selectivity) in selectivities {
            let index_height = self.constants.default_index_height;
            let index_io = index_height as f64 * self.constants.random_page_cost;
            let leaf_pages = (table_stats.page_count as f64 * selectivity).max(1.0) as u64;
            let leaf_io = leaf_pages as f64 * self.constants.random_page_cost;
            total_index_cost += index_io + leaf_io;
            total_pages += leaf_pages + index_height as u64;
        }

        // Intersection cost (merge sorted lists)
        let intersection_cost = estimated_rows as f64 * self.constants.cpu_operator_cost;

        // Table lookup cost
        let table_lookup_cost = estimated_rows as f64 * self.constants.cpu_tuple_cost;

        let total = total_index_cost + intersection_cost + table_lookup_cost;

        let index_names: Vec<_> = selectivities.iter().map(|(n, _)| n.as_str()).collect();

        PlanCost::with_total(
            total,
            total_index_cost,
            self.constants.cpu_tuple_cost,
            estimated_rows,
            total_pages,
            format!(
                "Multi-Index AND ({}): combined selectivity {:.6} → {} rows, cost {:.2}",
                index_names.join(", "),
                combined_selectivity,
                estimated_rows,
                total
            ),
        )
    }

    /// Estimate cost of multiple indexes combined with OR
    pub fn estimate_multi_index_or(
        &self,
        table_stats: &TableStats,
        selectivities: &[(String, f64)], // (index_name, selectivity)
    ) -> PlanCost {
        if selectivities.is_empty() {
            return self.estimate_seq_scan(table_stats);
        }

        // Combined selectivity for OR: 1 - product(1 - s_i)
        // This assumes independent predicates
        let combined_selectivity =
            1.0 - selectivities.iter().map(|(_, s)| 1.0 - s).product::<f64>();

        let row_count = table_stats.row_count;
        let estimated_rows = (row_count as f64 * combined_selectivity).max(1.0) as u64;

        // Cost for each index traversal
        let mut total_index_cost = 0.0;
        let mut total_pages = 0u64;

        for (_, selectivity) in selectivities {
            let index_height = self.constants.default_index_height;
            let index_io = index_height as f64 * self.constants.random_page_cost;
            let leaf_pages = (table_stats.page_count as f64 * selectivity).max(1.0) as u64;
            let leaf_io = leaf_pages as f64 * self.constants.random_page_cost;
            total_index_cost += index_io + leaf_io;
            total_pages += leaf_pages + index_height as u64;
        }

        // Union cost (merge and deduplicate)
        let union_cost = estimated_rows as f64 * self.constants.cpu_operator_cost * 2.0;

        // Table lookup cost
        let table_lookup_cost = estimated_rows as f64 * self.constants.cpu_tuple_cost;

        let total = total_index_cost + union_cost + table_lookup_cost;

        let index_names: Vec<_> = selectivities.iter().map(|(n, _)| n.as_str()).collect();

        PlanCost::with_total(
            total,
            total_index_cost,
            self.constants.cpu_tuple_cost,
            estimated_rows,
            total_pages,
            format!(
                "Multi-Index OR ({}): combined selectivity {:.4} → {} rows, cost {:.2}",
                index_names.join(", "),
                combined_selectivity,
                estimated_rows,
                total
            ),
        )
    }

    /// Compare access methods and return the cheapest
    pub fn choose_best_access_method(
        &self,
        _table_stats: &TableStats,
        candidates: Vec<(AccessMethod, PlanCost)>,
    ) -> Option<(AccessMethod, PlanCost)> {
        candidates.into_iter().min_by(|(_, cost_a), (_, cost_b)| {
            cost_a
                .total
                .partial_cmp(&cost_b.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Estimate selectivity for an equality predicate
    pub fn estimate_equality_selectivity(&self, column_stats: Option<&ColumnStats>) -> f64 {
        match column_stats {
            Some(stats) if stats.distinct_count > 0 => {
                SelectivityEstimator::equality(stats.distinct_count)
            }
            _ => 0.1, // Default selectivity
        }
    }

    /// Estimate selectivity for a range predicate (simple, no value)
    pub fn estimate_range_selectivity(&self, column_stats: Option<&ColumnStats>) -> f64 {
        // Use histogram if available for better estimate
        match column_stats {
            Some(stats) if stats.parsed_histogram().is_some() => {
                // With histogram, we assume uniform distribution over the range
                // This is the fallback when no specific bound is given
                SelectivityEstimator::range()
            }
            Some(stats) if stats.min_value.is_some() && stats.max_value.is_some() => {
                // With min/max, we can at least verify the query is valid
                SelectivityEstimator::range()
            }
            _ => SelectivityEstimator::range(),
        }
    }

    /// Estimate selectivity for a comparison predicate using histogram
    ///
    /// This uses the histogram for accurate estimates when available.
    /// Operators: <, <=, >, >=
    pub fn estimate_comparison_selectivity(
        &self,
        column_stats: Option<&ColumnStats>,
        value: &crate::Value,
        operator: crate::storage::statistics::HistogramOp,
    ) -> f64 {
        match column_stats {
            Some(stats) => SelectivityEstimator::range_with_histogram(stats, value, operator),
            None => SelectivityEstimator::range(),
        }
    }

    /// Estimate selectivity for a BETWEEN range predicate using histogram
    ///
    /// Uses bucket walk algorithm for accurate estimation when histogram is available.
    /// For WHERE column BETWEEN low AND high.
    pub fn estimate_between_selectivity(
        &self,
        column_stats: Option<&ColumnStats>,
        low: &crate::Value,
        high: &crate::Value,
    ) -> f64 {
        match column_stats {
            Some(stats) => {
                // Try histogram first
                if let Some(histogram) = stats.parsed_histogram() {
                    return histogram.estimate_range_selectivity(low, high);
                }

                // Fall back to min/max linear interpolation
                if let (Some(min_val), Some(max_val)) = (&stats.min_value, &stats.max_value) {
                    // Estimate range fraction using linear interpolation
                    let total_range = Self::value_distance(min_val, max_val);
                    if total_range > 0.0 {
                        let query_range = Self::value_distance(low, high);
                        return (query_range / total_range).clamp(0.0001, 1.0);
                    }
                }

                SelectivityEstimator::range()
            }
            None => SelectivityEstimator::range(),
        }
    }

    /// Calculate numeric distance between two values for interpolation
    fn value_distance(v1: &crate::Value, v2: &crate::Value) -> f64 {
        match (v1, v2) {
            (crate::Value::Integer(a), crate::Value::Integer(b)) => (*b - *a).abs() as f64,
            (crate::Value::Float(a), crate::Value::Float(b)) => (b - a).abs(),
            (crate::Value::Integer(a), crate::Value::Float(b)) => (b - *a as f64).abs(),
            (crate::Value::Float(a), crate::Value::Integer(b)) => (*b as f64 - a).abs(),
            // For timestamps, could add date arithmetic here
            _ => 0.0, // Non-numeric types can't be interpolated
        }
    }

    /// Estimate selectivity for IS NULL
    pub fn estimate_null_selectivity(
        &self,
        table_stats: &TableStats,
        column_stats: Option<&ColumnStats>,
    ) -> f64 {
        match column_stats {
            Some(stats) => SelectivityEstimator::is_null(stats.null_count, table_stats.row_count),
            None => 0.01, // Default null fraction
        }
    }

    /// Estimate selectivity for IS NOT NULL
    pub fn estimate_not_null_selectivity(
        &self,
        table_stats: &TableStats,
        column_stats: Option<&ColumnStats>,
    ) -> f64 {
        match column_stats {
            Some(stats) => {
                SelectivityEstimator::is_not_null(stats.null_count, table_stats.row_count)
            }
            None => 0.99, // Default not null fraction
        }
    }

    /// Estimate selectivity for LIKE predicate
    pub fn estimate_like_selectivity(
        &self,
        pattern: &str,
        column_stats: Option<&ColumnStats>,
    ) -> f64 {
        let distinct = column_stats.map(|s| s.distinct_count).unwrap_or(100);
        SelectivityEstimator::like(pattern, distinct)
    }

    /// Estimate selectivity for IN list
    pub fn estimate_in_list_selectivity(
        &self,
        list_size: usize,
        column_stats: Option<&ColumnStats>,
    ) -> f64 {
        let distinct = column_stats.map(|s| s.distinct_count).unwrap_or(100);
        SelectivityEstimator::in_list(list_size, distinct)
    }

    /// Estimate combined selectivity for multiple predicates with correlation awareness
    ///
    /// This is the KEY method for accurate multi-column selectivity estimation.
    /// It accounts for correlations between columns to avoid severe under-estimation.
    ///
    /// # Problem
    /// Traditional optimizers assume independence: P(A AND B) = P(A) × P(B)
    /// This fails badly for correlated columns:
    ///   WHERE city = 'NYC' AND state = 'NY'
    ///   - Naive: 0.02 × 0.02 = 0.0004 (way too low!)
    ///   - With correlation: ~0.02 (matches reality)
    ///
    /// # Arguments
    /// * `selectivities` - List of (column_name, selectivity) pairs
    /// * `correlations` - Optional correlation data for the table
    ///
    /// # Returns
    /// Combined selectivity accounting for column correlations
    pub fn estimate_combined_selectivity(
        &self,
        selectivities: &[(&str, f64)],
        correlations: Option<&ColumnCorrelations>,
    ) -> f64 {
        SelectivityEstimator::combined_selectivity_with_correlations(selectivities, correlations)
    }

    /// Estimate combined selectivity for AND predicates on multiple columns
    ///
    /// Convenience method that builds selectivity list and applies correlations.
    ///
    /// # Arguments
    /// * `column_selectivities` - Map of column name to individual selectivity
    /// * `correlations` - Optional correlation data for the table
    pub fn estimate_and_selectivity(
        &self,
        column_selectivities: &std::collections::HashMap<String, f64>,
        correlations: Option<&ColumnCorrelations>,
    ) -> f64 {
        let selectivities: Vec<(&str, f64)> = column_selectivities
            .iter()
            .map(|(col, sel)| (col.as_str(), *sel))
            .collect();
        self.estimate_combined_selectivity(&selectivities, correlations)
    }

    /// Estimate cost of a sequential scan with multiple predicates
    ///
    /// This version accounts for column correlations when combining selectivities.
    pub fn estimate_seq_scan_with_predicates(
        &self,
        table_stats: &TableStats,
        selectivities: &[(&str, f64)],
        correlations: Option<&ColumnCorrelations>,
    ) -> PlanCost {
        let combined_selectivity = self.estimate_combined_selectivity(selectivities, correlations);
        self.estimate_seq_scan_with_filter(table_stats, combined_selectivity)
    }

    /// Determine if sequential scan is better than index scan
    /// Rule of thumb: index is better when selectivity < 10-20%
    pub fn should_use_index(&self, table_stats: &TableStats, selectivity: f64) -> bool {
        // For very small tables, sequential scan is always faster
        if table_stats.row_count < 100 {
            return false;
        }

        // For very selective queries, use index
        if selectivity < 0.01 {
            return true;
        }

        // Compare costs
        let seq_cost = self.estimate_seq_scan_with_filter(table_stats, selectivity);
        // Default to BTree for comparison - this is a heuristic method
        let index_cost =
            self.estimate_index_scan(table_stats, None, selectivity, "idx", IndexType::BTree);

        index_cost.is_cheaper_than(&seq_cost)
    }

    // =========================================================================
    // Join Cost Estimation Methods
    // =========================================================================

    /// Estimate cost of a hash join
    ///
    /// Hash join has two phases:
    /// 1. Build phase: Create hash table from smaller input (O(build_rows))
    /// 2. Probe phase: Scan larger input and probe hash table (O(probe_rows))
    ///
    /// Total cost is O(build_rows + probe_rows), much cheaper than nested loop
    /// for large tables.
    pub fn estimate_hash_join(&self, join_stats: &JoinStats) -> PlanCost {
        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        // Determine build and probe sides (build on smaller table)
        let (build_rows, probe_rows, build_side) = if left_rows <= right_rows {
            (left_rows, right_rows, BuildSide::Left)
        } else {
            (right_rows, left_rows, BuildSide::Right)
        };

        // Build phase cost: hash each row + store in hash table
        let build_cost =
            build_rows as f64 * (self.constants.cpu_tuple_cost + self.constants.hash_build_cost);

        // Probe phase cost: hash each probe row + lookup + verify match
        let probe_cost = probe_rows as f64
            * (self.constants.cpu_tuple_cost
                + self.constants.hash_probe_cost
                + self.constants.cpu_operator_cost);

        // Memory cost (estimated, accounts for hash table overhead)
        let avg_row_size = join_stats.left_stats.avg_row_size.max(32) as f64;
        let hash_table_size = build_rows as f64 * avg_row_size * self.constants.hash_memory_factor;
        let memory_pages = (hash_table_size / self.constants.page_size as f64).max(1.0) as u64;

        // Output cardinality using the formula: |L| * |R| / max(distinct_L, distinct_R)
        let left_distinct = if join_stats.left_distinct > 0 {
            join_stats.left_distinct
        } else {
            left_rows.max(1) // Fallback: assume all unique
        };
        let right_distinct = if join_stats.right_distinct > 0 {
            join_stats.right_distinct
        } else {
            right_rows.max(1) // Fallback: assume all unique
        };
        let output_rows = SelectivityEstimator::join_cardinality(
            left_rows,
            right_rows,
            left_distinct,
            right_distinct,
        );

        let total = build_cost + probe_cost;

        PlanCost::with_total(
            total,
            build_cost, // Startup: build entire hash table
            self.constants.cpu_tuple_cost + self.constants.hash_probe_cost,
            output_rows,
            memory_pages,
            format!(
                "Hash Join: build {} rows ({}), probe {} rows → {} output, cost {:.2}",
                build_rows,
                build_side.description(),
                probe_rows,
                output_rows,
                total
            ),
        )
    }

    /// Estimate cost of a nested loop join
    ///
    /// Nested loop join scans the outer table and for each row,
    /// scans the entire inner table looking for matches.
    ///
    /// Cost is O(outer_rows * inner_rows), expensive for large tables
    /// but can be efficient for small tables or when indexes are available.
    pub fn estimate_nested_loop_join(&self, join_stats: &JoinStats) -> PlanCost {
        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        // Use smaller table as outer for fewer iterations
        let (outer_rows, inner_rows) = if left_rows <= right_rows {
            (left_rows, right_rows)
        } else {
            (right_rows, left_rows)
        };

        // Total comparisons: O(outer * inner)
        let total_comparisons = outer_rows as u128 * inner_rows as u128;

        // Cost per comparison (tuple access + operator evaluation)
        let comparison_cost =
            self.constants.cpu_tuple_cost + self.constants.nested_loop_compare_cost;

        // I/O cost: scan inner table for each outer row
        // In memory, this is mostly CPU cost
        let inner_pages = join_stats.right_stats.page_count.max(1);
        let io_cost = outer_rows as f64 * inner_pages as f64 * self.constants.seq_page_cost * 0.1;

        let cpu_cost = total_comparisons as f64 * comparison_cost;
        let total = cpu_cost + io_cost;

        // Output cardinality estimation
        let left_distinct = if join_stats.left_distinct > 0 {
            join_stats.left_distinct
        } else {
            left_rows.max(1)
        };
        let right_distinct = if join_stats.right_distinct > 0 {
            join_stats.right_distinct
        } else {
            right_rows.max(1)
        };
        let output_rows = SelectivityEstimator::join_cardinality(
            left_rows,
            right_rows,
            left_distinct,
            right_distinct,
        );

        PlanCost::with_total(
            total,
            0.0, // Startup: minimal (streaming)
            comparison_cost,
            output_rows,
            inner_pages,
            format!(
                "Nested Loop: {} × {} = {} comparisons → {} output, cost {:.2}",
                outer_rows, inner_rows, total_comparisons, output_rows, total
            ),
        )
    }

    /// Estimate cost of a merge join
    ///
    /// Merge join works by:
    /// 1. Sort both inputs (if not already sorted)
    /// 2. Merge the sorted streams in O(N + M) time
    ///
    /// Best when:
    /// - Inputs are already sorted (index scan on join key)
    /// - Join output is needed in sorted order
    /// - Memory is constrained (sorts can spill to disk)
    pub fn estimate_merge_join(
        &self,
        join_stats: &JoinStats,
        left_sorted: bool,
        right_sorted: bool,
    ) -> PlanCost {
        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        // Sort cost: O(N log N) for each unsorted input
        let left_sort_cost = if left_sorted {
            0.0
        } else {
            let n = left_rows as f64;
            n * n.log2().max(1.0) * self.constants.sort_cost_per_row
        };

        let right_sort_cost = if right_sorted {
            0.0
        } else {
            let n = right_rows as f64;
            n * n.log2().max(1.0) * self.constants.sort_cost_per_row
        };

        // Merge phase cost: O(N + M) comparisons
        let merge_cost = (left_rows + right_rows) as f64 * self.constants.merge_compare_cost;

        // CPU cost for processing tuples
        let cpu_cost = (left_rows + right_rows) as f64 * self.constants.cpu_tuple_cost;

        let total = left_sort_cost + right_sort_cost + merge_cost + cpu_cost;

        // Output cardinality
        let left_distinct = join_stats.left_distinct.max(1);
        let right_distinct = join_stats.right_distinct.max(1);
        let output_rows = SelectivityEstimator::join_cardinality(
            left_rows,
            right_rows,
            left_distinct,
            right_distinct,
        );

        let pages = join_stats.left_stats.page_count + join_stats.right_stats.page_count;

        PlanCost::with_total(
            total,
            left_sort_cost + right_sort_cost, // Startup: sort both inputs
            self.constants.merge_compare_cost,
            output_rows,
            pages,
            format!(
                "Merge Join: {} + {} rows, sort cost {:.2} + {:.2}, merge cost {:.2} → {} output, total {:.2}",
                left_rows, right_rows, left_sort_cost, right_sort_cost, merge_cost, output_rows, total
            ),
        )
    }

    /// Estimate cost of a semi-join (EXISTS subquery optimization)
    ///
    /// Semi-join returns rows from the left table that have at least one
    /// matching row in the right table. Key optimization: can terminate
    /// early on first match for each left row.
    ///
    /// Used for:
    /// - EXISTS (SELECT ... FROM right WHERE left.key = right.key)
    /// - IN (SELECT key FROM right)
    pub fn estimate_semi_join(&self, join_stats: &JoinStats, has_equality_keys: bool) -> PlanCost {
        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        // Estimate selectivity (what fraction of left rows have a match)
        let left_distinct = join_stats.left_distinct.max(1) as f64;
        let right_distinct = join_stats.right_distinct.max(1) as f64;
        let selectivity = (right_distinct / left_distinct).min(1.0);

        // With equality keys, use hash semi-join
        // Build hash table on right, probe with left
        // Early termination benefit: on average probe ~50% of right before finding match
        let base_cost = if has_equality_keys {
            // Build hash table on right
            let build_cost = right_rows as f64
                * (self.constants.cpu_tuple_cost + self.constants.hash_build_cost);

            // Probe with left - early termination means we often find match quickly
            let probe_cost = left_rows as f64
                * (self.constants.cpu_tuple_cost
                    + self.constants.hash_probe_cost * self.constants.semi_join_benefit);

            build_cost + probe_cost
        } else {
            // Nested loop semi-join with early termination
            // On average, scan ~50% of right table per left row
            let comparisons =
                left_rows as f64 * right_rows as f64 * self.constants.semi_join_benefit;
            comparisons * (self.constants.cpu_tuple_cost + self.constants.nested_loop_compare_cost)
        };

        // Output rows: fraction of left that matches
        let output_rows = (left_rows as f64 * selectivity).max(1.0) as u64;

        let pages = join_stats.right_stats.page_count;

        PlanCost::with_total(
            base_cost,
            0.0,
            self.constants.cpu_tuple_cost,
            output_rows,
            pages,
            format!(
                "Semi-Join: {} left × {} right, selectivity {:.2} → {} output, cost {:.2}",
                left_rows, right_rows, selectivity, output_rows, base_cost
            ),
        )
    }

    /// Estimate cost of an anti-join (NOT EXISTS / NOT IN optimization)
    ///
    /// Anti-join returns rows from left that have NO matching row in right.
    /// Unlike semi-join, we must check all of right for each left row
    /// (no early termination on first match).
    pub fn estimate_anti_join(&self, join_stats: &JoinStats, has_equality_keys: bool) -> PlanCost {
        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        // Estimate selectivity (what fraction of left rows have NO match)
        let left_distinct = join_stats.left_distinct.max(1) as f64;
        let right_distinct = join_stats.right_distinct.max(1) as f64;
        let match_selectivity = (right_distinct / left_distinct).min(1.0);
        let anti_selectivity = 1.0 - match_selectivity;

        // With equality keys, use hash anti-join
        let base_cost = if has_equality_keys {
            // Build hash table on right
            let build_cost = right_rows as f64
                * (self.constants.cpu_tuple_cost + self.constants.hash_build_cost);

            // Probe with left - no early termination (must confirm no match)
            let probe_cost =
                left_rows as f64 * (self.constants.cpu_tuple_cost + self.constants.hash_probe_cost);

            build_cost + probe_cost
        } else {
            // Nested loop anti-join - must scan all of right for each left
            let comparisons = left_rows as f64 * right_rows as f64;
            comparisons * (self.constants.cpu_tuple_cost + self.constants.nested_loop_compare_cost)
        };

        // Output rows: fraction of left that doesn't match
        let output_rows = (left_rows as f64 * anti_selectivity).max(1.0) as u64;

        let pages = join_stats.right_stats.page_count;

        PlanCost::with_total(
            base_cost,
            0.0,
            self.constants.cpu_tuple_cost,
            output_rows,
            pages,
            format!(
                "Anti-Join: {} left × {} right, selectivity {:.2} → {} output, cost {:.2}",
                left_rows, right_rows, anti_selectivity, output_rows, base_cost
            ),
        )
    }

    /// Choose the best join algorithm based on cost estimation
    ///
    /// # Arguments
    /// * `join_stats` - Statistics for both tables and join columns
    /// * `has_equality_keys` - Whether the join has equality conditions (col1 = col2)
    ///
    /// # Returns
    /// The chosen join algorithm with estimated parameters
    pub fn choose_join_algorithm(
        &self,
        join_stats: &JoinStats,
        has_equality_keys: bool,
    ) -> (JoinAlgorithm, PlanCost) {
        // Without equality keys, must use nested loop
        if !has_equality_keys {
            let cost = self.estimate_nested_loop_join(join_stats);
            let left_rows = join_stats.left_stats.row_count;
            let right_rows = join_stats.right_stats.row_count;
            let (outer_rows, inner_rows) = if left_rows <= right_rows {
                (left_rows, right_rows)
            } else {
                (right_rows, left_rows)
            };
            return (
                JoinAlgorithm::NestedLoop {
                    outer_rows,
                    inner_rows,
                },
                cost,
            );
        }

        // Compare hash join vs nested loop
        let hash_cost = self.estimate_hash_join(join_stats);
        let nested_cost = self.estimate_nested_loop_join(join_stats);

        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        if hash_cost.is_cheaper_than(&nested_cost) {
            // Hash join is cheaper
            let (build_rows, probe_rows, build_side) = if left_rows <= right_rows {
                (left_rows, right_rows, BuildSide::Left)
            } else {
                (right_rows, left_rows, BuildSide::Right)
            };
            (
                JoinAlgorithm::HashJoin {
                    build_side,
                    build_rows,
                    probe_rows,
                },
                hash_cost,
            )
        } else {
            // Nested loop is cheaper (unusual, but possible for very small tables)
            let (outer_rows, inner_rows) = if left_rows <= right_rows {
                (left_rows, right_rows)
            } else {
                (right_rows, left_rows)
            };
            (
                JoinAlgorithm::NestedLoop {
                    outer_rows,
                    inner_rows,
                },
                nested_cost,
            )
        }
    }

    /// Estimate cost for a specific join algorithm
    pub fn estimate_join(&self, algorithm: &JoinAlgorithm, join_stats: &JoinStats) -> PlanCost {
        match algorithm {
            JoinAlgorithm::HashJoin { .. } => self.estimate_hash_join(join_stats),
            JoinAlgorithm::MergeJoin {
                left_sorted,
                right_sorted,
                ..
            } => self.estimate_merge_join(join_stats, *left_sorted, *right_sorted),
            JoinAlgorithm::NestedLoop { .. } => self.estimate_nested_loop_join(join_stats),
            JoinAlgorithm::SemiJoin { .. } => self.estimate_semi_join(join_stats, true),
            JoinAlgorithm::AntiJoin { .. } => self.estimate_anti_join(join_stats, true),
        }
    }

    /// Choose the best join algorithm including merge join
    ///
    /// Extended version that considers:
    /// - Hash Join (best for equality joins with large tables)
    /// - Merge Join (best when inputs are sorted or output needs sorting)
    /// - Nested Loop (best for small tables or non-equality joins)
    pub fn choose_join_algorithm_extended(
        &self,
        join_stats: &JoinStats,
        has_equality_keys: bool,
        left_sorted: bool,
        right_sorted: bool,
    ) -> (JoinAlgorithm, PlanCost) {
        // Without equality keys, must use nested loop
        if !has_equality_keys {
            let cost = self.estimate_nested_loop_join(join_stats);
            let left_rows = join_stats.left_stats.row_count;
            let right_rows = join_stats.right_stats.row_count;
            let (outer_rows, inner_rows) = if left_rows <= right_rows {
                (left_rows, right_rows)
            } else {
                (right_rows, left_rows)
            };
            return (
                JoinAlgorithm::NestedLoop {
                    outer_rows,
                    inner_rows,
                },
                cost,
            );
        }

        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        // Compare all join algorithms
        let hash_cost = self.estimate_hash_join(join_stats);
        let merge_cost = self.estimate_merge_join(join_stats, left_sorted, right_sorted);
        let nested_cost = self.estimate_nested_loop_join(join_stats);

        // Find the cheapest
        let mut best_cost = hash_cost.clone();
        let mut best_algo = {
            let (build_rows, probe_rows, build_side) = if left_rows <= right_rows {
                (left_rows, right_rows, BuildSide::Left)
            } else {
                (right_rows, left_rows, BuildSide::Right)
            };
            JoinAlgorithm::HashJoin {
                build_side,
                build_rows,
                probe_rows,
            }
        };

        if merge_cost.is_cheaper_than(&best_cost) {
            best_cost = merge_cost;
            best_algo = JoinAlgorithm::MergeJoin {
                left_rows,
                right_rows,
                left_sorted,
                right_sorted,
            };
        }

        if nested_cost.is_cheaper_than(&best_cost) {
            best_cost = nested_cost;
            let (outer_rows, inner_rows) = if left_rows <= right_rows {
                (left_rows, right_rows)
            } else {
                (right_rows, left_rows)
            };
            best_algo = JoinAlgorithm::NestedLoop {
                outer_rows,
                inner_rows,
            };
        }

        (best_algo, best_cost)
    }

    /// Choose join algorithm with LIMIT pushdown optimization
    ///
    /// When a query has a small LIMIT, nested loop joins can short-circuit early,
    /// making them more efficient than hash joins that need to build the entire
    /// hash table before producing output.
    ///
    /// # Arguments
    /// * `join_stats` - Statistics for both tables
    /// * `has_equality_keys` - Whether the join has equality conditions
    /// * `limit` - Optional LIMIT value from the query
    ///
    /// # Returns
    /// The chosen join algorithm optimized for the given LIMIT
    pub fn choose_join_algorithm_with_limit(
        &self,
        join_stats: &JoinStats,
        has_equality_keys: bool,
        limit: Option<u64>,
    ) -> (JoinAlgorithm, PlanCost) {
        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;

        // Without equality keys, must use nested loop
        if !has_equality_keys {
            let cost = self.estimate_nested_loop_join(join_stats);
            let (outer_rows, inner_rows) = if left_rows <= right_rows {
                (left_rows, right_rows)
            } else {
                (right_rows, left_rows)
            };
            return (
                JoinAlgorithm::NestedLoop {
                    outer_rows,
                    inner_rows,
                },
                cost,
            );
        }

        // Threshold for when LIMIT makes nested loop preferable
        // If LIMIT is small and smaller than either input, nested loop can short-circuit
        const LIMIT_THRESHOLD: u64 = 100;

        if let Some(limit_val) = limit {
            if limit_val <= LIMIT_THRESHOLD {
                // For small LIMIT, calculate the effective cost of nested loop with early termination
                // Nested loop cost with LIMIT: outer scans limited rows, stops early
                let effective_outer = limit_val.min(left_rows.min(right_rows));
                let nested_cost_with_limit = PlanCost::with_total(
                    effective_outer as f64 * self.constants.cpu_tuple_cost,
                    0.0,
                    self.constants.cpu_tuple_cost,
                    limit_val,
                    effective_outer,
                    format!(
                        "Nested Loop (LIMIT {}): ~{} iterations",
                        limit_val, effective_outer
                    ),
                );

                // Hash join cost: must build entire hash table first
                let hash_cost = self.estimate_hash_join(join_stats);

                // If nested loop with early termination is cheaper, use it
                if nested_cost_with_limit.total < hash_cost.total * 0.5 {
                    let (outer_rows, inner_rows) = if left_rows <= right_rows {
                        (left_rows, right_rows)
                    } else {
                        (right_rows, left_rows)
                    };
                    return (
                        JoinAlgorithm::NestedLoop {
                            outer_rows,
                            inner_rows,
                        },
                        nested_cost_with_limit,
                    );
                }
            }
        }

        // Fall back to standard algorithm selection
        self.choose_join_algorithm(join_stats, has_equality_keys)
    }

    /// Choose join algorithm considering sorted inputs for merge join optimization
    ///
    /// When both inputs are sorted on the join key (e.g., via B-tree index scans),
    /// merge join can be much more efficient as it avoids the sort cost.
    ///
    /// # Arguments
    /// * `join_stats` - Statistics for both tables
    /// * `has_equality_keys` - Whether the join has equality conditions
    /// * `limit` - Optional LIMIT value from the query
    /// * `left_sorted` - Whether left input is sorted on join key
    /// * `right_sorted` - Whether right input is sorted on join key
    ///
    /// # Returns
    /// The chosen join algorithm optimized for sorted inputs
    pub fn choose_join_algorithm_with_sorted_inputs(
        &self,
        join_stats: &JoinStats,
        has_equality_keys: bool,
        limit: Option<u64>,
        left_sorted: bool,
        right_sorted: bool,
    ) -> (JoinAlgorithm, PlanCost) {
        // If both inputs are sorted on the join key, prefer merge join
        if has_equality_keys && left_sorted && right_sorted {
            let merge_cost = self.estimate_merge_join(join_stats, true, true);
            let hash_cost = self.estimate_hash_join(join_stats);

            // Merge join is preferable when inputs are pre-sorted (avoids sort overhead)
            // Only choose merge join if it's significantly better than hash
            if merge_cost.total < hash_cost.total * 1.2 {
                let left_rows = join_stats.left_stats.row_count;
                let right_rows = join_stats.right_stats.row_count;
                return (
                    JoinAlgorithm::MergeJoin {
                        left_rows,
                        right_rows,
                        left_sorted: true,
                        right_sorted: true,
                    },
                    merge_cost,
                );
            }
        }

        // Fall back to LIMIT-aware algorithm selection
        self.choose_join_algorithm_with_limit(join_stats, has_equality_keys, limit)
    }

    /// Choose join algorithm for semi-join (EXISTS) operations
    pub fn choose_semi_join_algorithm(
        &self,
        join_stats: &JoinStats,
        has_equality_keys: bool,
    ) -> (JoinAlgorithm, PlanCost) {
        let semi_cost = self.estimate_semi_join(join_stats, has_equality_keys);

        // Determine inner algorithm (hash or nested loop)
        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;
        let left_distinct = join_stats.left_distinct.max(1) as f64;
        let right_distinct = join_stats.right_distinct.max(1) as f64;
        let selectivity = (right_distinct / left_distinct).min(1.0);

        let inner_algorithm = if has_equality_keys {
            let (build_rows, probe_rows, build_side) = if left_rows <= right_rows {
                (left_rows, right_rows, BuildSide::Left)
            } else {
                (right_rows, left_rows, BuildSide::Right)
            };
            JoinAlgorithm::HashJoin {
                build_side,
                build_rows,
                probe_rows,
            }
        } else {
            let (outer_rows, inner_rows) = if left_rows <= right_rows {
                (left_rows, right_rows)
            } else {
                (right_rows, left_rows)
            };
            JoinAlgorithm::NestedLoop {
                outer_rows,
                inner_rows,
            }
        };

        (
            JoinAlgorithm::SemiJoin {
                inner_algorithm: Box::new(inner_algorithm),
                left_rows,
                selectivity,
            },
            semi_cost,
        )
    }

    /// Choose join algorithm for anti-join (NOT EXISTS) operations
    pub fn choose_anti_join_algorithm(
        &self,
        join_stats: &JoinStats,
        has_equality_keys: bool,
    ) -> (JoinAlgorithm, PlanCost) {
        let anti_cost = self.estimate_anti_join(join_stats, has_equality_keys);

        let left_rows = join_stats.left_stats.row_count;
        let right_rows = join_stats.right_stats.row_count;
        let left_distinct = join_stats.left_distinct.max(1) as f64;
        let right_distinct = join_stats.right_distinct.max(1) as f64;
        let match_selectivity = (right_distinct / left_distinct).min(1.0);
        let selectivity = 1.0 - match_selectivity;

        let inner_algorithm = if has_equality_keys {
            let (build_rows, probe_rows, build_side) = if left_rows <= right_rows {
                (left_rows, right_rows, BuildSide::Left)
            } else {
                (right_rows, left_rows, BuildSide::Right)
            };
            JoinAlgorithm::HashJoin {
                build_side,
                build_rows,
                probe_rows,
            }
        } else {
            let (outer_rows, inner_rows) = if left_rows <= right_rows {
                (left_rows, right_rows)
            } else {
                (right_rows, left_rows)
            };
            JoinAlgorithm::NestedLoop {
                outer_rows,
                inner_rows,
            }
        };

        (
            JoinAlgorithm::AntiJoin {
                inner_algorithm: Box::new(inner_algorithm),
                left_rows,
                selectivity,
            },
            anti_cost,
        )
    }

    /// Get the cost constants (for testing/debugging)
    pub fn constants(&self) -> &CostConstants {
        &self.constants
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table_stats(rows: u64, pages: u64) -> TableStats {
        TableStats {
            table_name: "test".to_string(),
            row_count: rows,
            page_count: pages,
            avg_row_size: 100,
        }
    }

    #[test]
    fn test_seq_scan_cost() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(10000, 100);

        let cost = estimator.estimate_seq_scan(&stats);
        assert!(cost.total > 0.0);
        assert_eq!(cost.estimated_rows, 10000);
        assert_eq!(cost.estimated_pages, 100);
    }

    #[test]
    fn test_pk_lookup_cost() {
        let estimator = CostEstimator::new();
        let cost = estimator.estimate_pk_lookup();

        assert!(cost.total > 0.0);
        assert_eq!(cost.estimated_rows, 1);
        assert!(cost.total < 10.0); // PK lookup should be very cheap
    }

    #[test]
    fn test_index_scan_cost() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(10000, 100);

        // 1% selectivity with BTree index
        let cost = estimator.estimate_index_scan(&stats, None, 0.01, "idx_test", IndexType::BTree);

        assert!(cost.total > 0.0);
        assert_eq!(cost.estimated_rows, 100); // 1% of 10000
    }

    #[test]
    fn test_index_vs_seq_scan() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(10000, 100);

        // Very selective query (0.1%) - index should win
        let seq_cost = estimator.estimate_seq_scan_with_filter(&stats, 0.001);
        let idx_cost = estimator.estimate_index_scan(&stats, None, 0.001, "idx", IndexType::BTree);
        assert!(idx_cost.is_cheaper_than(&seq_cost));

        // Non-selective query (50%) - seq scan should win
        let seq_cost = estimator.estimate_seq_scan_with_filter(&stats, 0.5);
        let idx_cost = estimator.estimate_index_scan(&stats, None, 0.5, "idx", IndexType::BTree);
        assert!(seq_cost.is_cheaper_than(&idx_cost));
    }

    #[test]
    fn test_multi_index_and() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(10000, 100);

        // Two indexes with 10% selectivity each → 1% combined
        let cost = estimator.estimate_multi_index_and(
            &stats,
            &[("idx1".to_string(), 0.1), ("idx2".to_string(), 0.1)],
        );

        assert_eq!(cost.estimated_rows, 100); // 1% of 10000
    }

    #[test]
    fn test_multi_index_or() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(10000, 100);

        // Two indexes with 10% selectivity each → ~19% combined
        let cost = estimator.estimate_multi_index_or(
            &stats,
            &[("idx1".to_string(), 0.1), ("idx2".to_string(), 0.1)],
        );

        // OR selectivity: 1 - (0.9 * 0.9) = 0.19
        assert!(cost.estimated_rows > 1800 && cost.estimated_rows < 2000);
    }

    #[test]
    fn test_should_use_index() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(10000, 100);

        // Very selective - use index
        assert!(estimator.should_use_index(&stats, 0.001));

        // Not selective - use seq scan
        assert!(!estimator.should_use_index(&stats, 0.5));

        // Small table - always seq scan
        let small_stats = make_table_stats(50, 1);
        assert!(!estimator.should_use_index(&small_stats, 0.01));
    }

    #[test]
    fn test_equality_selectivity() {
        let estimator = CostEstimator::new();

        // With stats: 1/distinct
        let stats = ColumnStats {
            column_name: "col".to_string(),
            null_count: 0,
            distinct_count: 100,
            min_value: None,
            max_value: None,
            avg_width: 8,
            histogram: None,
        };
        let sel = estimator.estimate_equality_selectivity(Some(&stats));
        assert!((sel - 0.01).abs() < 0.001);

        // Without stats: default
        let sel_default = estimator.estimate_equality_selectivity(None);
        assert!((sel_default - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_plan_cost_comparison() {
        let cheap = PlanCost::new(1.0, 0.01, 100, 1, "cheap".to_string());
        let expensive = PlanCost::new(10.0, 0.1, 1000, 10, "expensive".to_string());

        assert!(cheap.is_cheaper_than(&expensive));
        assert!(!expensive.is_cheaper_than(&cheap));
    }

    // =========================================================================
    // Join Cost Estimation Tests
    // =========================================================================

    fn make_join_stats(
        left_rows: u64,
        right_rows: u64,
        left_distinct: u64,
        right_distinct: u64,
    ) -> JoinStats {
        JoinStats {
            left_stats: TableStats {
                table_name: "left".to_string(),
                row_count: left_rows,
                page_count: (left_rows / 100).max(1),
                avg_row_size: 100,
            },
            right_stats: TableStats {
                table_name: "right".to_string(),
                row_count: right_rows,
                page_count: (right_rows / 100).max(1),
                avg_row_size: 100,
            },
            left_distinct,
            right_distinct,
        }
    }

    #[test]
    fn test_hash_join_cost() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 1000, 1000, 1000);

        let cost = estimator.estimate_hash_join(&join_stats);

        // Hash join should have reasonable cost
        assert!(cost.total > 0.0);

        // Output rows should match join cardinality formula
        // |L| * |R| / max(distinct_L, distinct_R) = 10000 * 1000 / 1000 = 10000
        assert_eq!(cost.estimated_rows, 10000);

        // Explanation should mention Hash Join
        assert!(cost.explanation.contains("Hash Join"));
    }

    #[test]
    fn test_nested_loop_cost() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(100, 50, 100, 50);

        let cost = estimator.estimate_nested_loop_join(&join_stats);

        // Nested loop should have positive cost
        assert!(cost.total > 0.0);

        // For small tables, nested loop is valid
        assert!(cost.explanation.contains("Nested Loop"));
    }

    #[test]
    fn test_hash_join_cheaper_for_large_tables() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 5000, 1000, 500);

        let hash_cost = estimator.estimate_hash_join(&join_stats);
        let nested_cost = estimator.estimate_nested_loop_join(&join_stats);

        // For large tables with equality keys, hash join should be much cheaper
        assert!(
            hash_cost.is_cheaper_than(&nested_cost),
            "Hash join cost ({:.2}) should be cheaper than nested loop ({:.2}) for large tables",
            hash_cost.total,
            nested_cost.total
        );

        // The difference should be significant (hash join is O(N+M) vs O(N*M))
        assert!(
            nested_cost.total > hash_cost.total * 10.0,
            "Nested loop should be at least 10x more expensive"
        );
    }

    #[test]
    fn test_choose_join_algorithm_with_equality() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 1000, 1000, 1000);

        // With equality keys, should choose hash join for large tables
        let (algorithm, cost) = estimator.choose_join_algorithm(&join_stats, true);

        assert!(algorithm.is_hash_join());
        assert!(cost.total > 0.0);

        if let JoinAlgorithm::HashJoin {
            build_side,
            build_rows,
            probe_rows,
        } = algorithm
        {
            // Should build on smaller table (right: 1000 rows)
            assert_eq!(build_side, BuildSide::Right);
            assert_eq!(build_rows, 1000);
            assert_eq!(probe_rows, 10000);
        } else {
            panic!("Expected HashJoin");
        }
    }

    #[test]
    fn test_choose_join_algorithm_without_equality() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(100, 100, 100, 100);

        // Without equality keys, must use nested loop
        let (algorithm, cost) = estimator.choose_join_algorithm(&join_stats, false);

        assert!(algorithm.is_nested_loop());
        assert!(cost.total > 0.0);
    }

    #[test]
    fn test_build_side_description() {
        assert_eq!(BuildSide::Left.description(), "Left");
        assert_eq!(BuildSide::Right.description(), "Right");
    }

    #[test]
    fn test_join_algorithm_description() {
        let hash_join = JoinAlgorithm::HashJoin {
            build_side: BuildSide::Left,
            build_rows: 100,
            probe_rows: 1000,
        };
        assert!(hash_join.description().contains("Hash Join"));
        assert!(hash_join.description().contains("Left"));

        let nested_loop = JoinAlgorithm::NestedLoop {
            outer_rows: 100,
            inner_rows: 1000,
        };
        assert_eq!(nested_loop.description(), "Nested Loop");
    }

    #[test]
    fn test_join_output_cardinality() {
        let estimator = CostEstimator::new();

        // Test with different distinct counts
        // |L| = 10000, |R| = 1000, distinct_L = 1000, distinct_R = 100
        // Expected: 10000 * 1000 / max(1000, 100) = 10000
        let join_stats = make_join_stats(10000, 1000, 1000, 100);
        let cost = estimator.estimate_hash_join(&join_stats);
        assert_eq!(cost.estimated_rows, 10000);

        // |L| = 1000, |R| = 1000, distinct_L = 500, distinct_R = 500
        // Expected: 1000 * 1000 / 500 = 2000
        let join_stats2 = make_join_stats(1000, 1000, 500, 500);
        let cost2 = estimator.estimate_hash_join(&join_stats2);
        assert_eq!(cost2.estimated_rows, 2000);
    }

    #[test]
    fn test_estimate_join() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(1000, 500, 100, 50);

        let hash_algo = JoinAlgorithm::HashJoin {
            build_side: BuildSide::Right,
            build_rows: 500,
            probe_rows: 1000,
        };
        let hash_cost = estimator.estimate_join(&hash_algo, &join_stats);
        assert!(hash_cost.explanation.contains("Hash Join"));

        let nested_algo = JoinAlgorithm::NestedLoop {
            outer_rows: 500,
            inner_rows: 1000,
        };
        let nested_cost = estimator.estimate_join(&nested_algo, &join_stats);
        assert!(nested_cost.explanation.contains("Nested Loop"));
    }

    #[test]
    fn test_merge_join_cost() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 1000, 100, 100);

        // Test merge join with pre-sorted inputs
        let cost_sorted = estimator.estimate_merge_join(&join_stats, true, true);
        assert!(cost_sorted.total > 0.0);
        assert!(cost_sorted.explanation.contains("Merge Join"));
        // When pre-sorted, sort cost is 0.00
        assert!(cost_sorted.explanation.contains("sort cost 0.00"));

        // Test merge join with unsorted inputs
        let cost_unsorted = estimator.estimate_merge_join(&join_stats, false, false);
        assert!(cost_unsorted.total > cost_sorted.total);
        // Unsorted inputs have non-zero sort cost
        assert!(!cost_unsorted.explanation.contains("sort cost 0.00 + 0.00"));
    }

    #[test]
    fn test_merge_join_cheaper_when_sorted() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 10000, 100, 100);

        let merge_sorted = estimator.estimate_merge_join(&join_stats, true, true);
        let hash_cost = estimator.estimate_hash_join(&join_stats);

        // When both sides are pre-sorted, merge join should be competitive
        // Merge is O(N+M) vs Hash O(N+M) but merge has lower constant factors
        assert!(
            merge_sorted.total < hash_cost.total * 1.5,
            "Merge join with sorted inputs should be competitive with hash join"
        );
    }

    #[test]
    fn test_semi_join_cost() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 1000, 100, 100);

        // Test with equality keys (hash-based)
        let cost = estimator.estimate_semi_join(&join_stats, true);
        assert!(cost.total > 0.0);
        assert!(cost.explanation.contains("Semi-Join"));

        // Output rows should be fewer due to early termination
        assert!(cost.estimated_rows <= 10000);
    }

    #[test]
    fn test_semi_join_with_and_without_equality() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 1000, 100, 100);

        // Semi-join with equality keys (hash-based) should be cheaper
        let cost_with_eq = estimator.estimate_semi_join(&join_stats, true);
        let cost_without_eq = estimator.estimate_semi_join(&join_stats, false);

        // Hash-based semi-join should be cheaper than nested loop based
        assert!(
            cost_with_eq.total < cost_without_eq.total,
            "Hash-based semi-join should be cheaper"
        );
    }

    #[test]
    fn test_anti_join_cost() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 1000, 100, 100);

        let cost = estimator.estimate_anti_join(&join_stats, true);
        assert!(cost.total > 0.0);
        assert!(cost.explanation.contains("Anti-Join"));

        // Output rows should be some fraction of left rows
        assert!(cost.estimated_rows <= 10000);
    }

    #[test]
    fn test_index_only_scan_cost() {
        let estimator = CostEstimator::new();

        // Create table stats using helper
        let table_stats = make_table_stats(10000, 100);

        // Index-only scan with 10% selectivity
        let cost = estimator.estimate_index_only_scan(&table_stats, 0.1, "idx_test");
        assert!(cost.total > 0.0);
        assert!(cost.explanation.contains("Index Only Scan"));

        // Index-only scan should be cheaper than regular index scan
        let regular_scan_cost =
            estimator.estimate_index_scan(&table_stats, None, 0.1, "idx_test", IndexType::BTree);
        assert!(
            cost.total < regular_scan_cost.total,
            "Index-only scan should be cheaper than regular index scan"
        );
    }

    #[test]
    fn test_index_only_scan_selectivity() {
        let estimator = CostEstimator::new();
        let table_stats = make_table_stats(10000, 100);

        // Low selectivity (few rows)
        let cost_low = estimator.estimate_index_only_scan(&table_stats, 0.01, "idx_test");

        // High selectivity (many rows)
        let cost_high = estimator.estimate_index_only_scan(&table_stats, 0.5, "idx_test");

        // Higher selectivity = more rows = higher cost
        assert!(
            cost_high.total > cost_low.total,
            "Higher selectivity should have higher cost"
        );
        assert!(cost_high.estimated_rows > cost_low.estimated_rows);
    }

    #[test]
    fn test_choose_join_algorithm_extended() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 10000, 100, 100);

        // Test with pre-sorted inputs - should consider merge join
        let (algorithm, cost) =
            estimator.choose_join_algorithm_extended(&join_stats, true, true, true);
        assert!(cost.total > 0.0);

        // Either hash or merge join should be chosen for equality join
        assert!(
            algorithm.is_hash_join() || matches!(algorithm, JoinAlgorithm::MergeJoin { .. }),
            "Should choose hash or merge join for large tables with equality"
        );
    }

    #[test]
    fn test_choose_semi_join_algorithm() {
        let estimator = CostEstimator::new();

        // Large tables with equality - should use hash-based semi-join
        let join_stats_large = make_join_stats(100000, 10000, 1000, 1000);
        let (algorithm, _cost) = estimator.choose_semi_join_algorithm(&join_stats_large, true);

        if let JoinAlgorithm::SemiJoin {
            inner_algorithm, ..
        } = &algorithm
        {
            assert!(
                inner_algorithm.is_hash_join(),
                "Should use hash-based semi-join for large tables"
            );
        } else {
            panic!("Expected SemiJoin algorithm");
        }

        // Without equality keys - should use nested loop based semi-join
        let (algorithm_no_eq, _) = estimator.choose_semi_join_algorithm(&join_stats_large, false);

        if let JoinAlgorithm::SemiJoin {
            inner_algorithm, ..
        } = &algorithm_no_eq
        {
            assert!(
                inner_algorithm.is_nested_loop(),
                "Should use nested loop semi-join without equality keys"
            );
        } else {
            panic!("Expected SemiJoin algorithm");
        }
    }

    #[test]
    fn test_choose_anti_join_algorithm() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 1000, 100, 100);

        let (algorithm, cost) = estimator.choose_anti_join_algorithm(&join_stats, true);
        assert!(cost.total > 0.0);

        if let JoinAlgorithm::AntiJoin { selectivity, .. } = algorithm {
            // Selectivity should be between 0 and 1
            assert!((0.0..=1.0).contains(&selectivity));
        } else {
            panic!("Expected AntiJoin algorithm");
        }
    }

    #[test]
    fn test_merge_join_algorithm_variant() {
        let merge = JoinAlgorithm::MergeJoin {
            left_rows: 1000,
            right_rows: 500,
            left_sorted: true,
            right_sorted: false,
        };

        assert!(merge.description().contains("Merge Join"));
        assert!(!merge.is_hash_join());
        assert!(!merge.is_nested_loop());
    }

    #[test]
    fn test_semi_join_algorithm_variant() {
        let semi = JoinAlgorithm::SemiJoin {
            inner_algorithm: Box::new(JoinAlgorithm::HashJoin {
                build_side: BuildSide::Right,
                build_rows: 100,
                probe_rows: 1000,
            }),
            left_rows: 1000,
            selectivity: 0.5,
        };

        assert!(semi.description().contains("Semi-Join"));
        assert!(!semi.is_hash_join());
    }

    #[test]
    fn test_anti_join_algorithm_variant() {
        let anti = JoinAlgorithm::AntiJoin {
            inner_algorithm: Box::new(JoinAlgorithm::HashJoin {
                build_side: BuildSide::Right,
                build_rows: 100,
                probe_rows: 1000,
            }),
            left_rows: 1000,
            selectivity: 0.3,
        };

        assert!(anti.description().contains("Anti-Join"));
        assert!(!anti.is_hash_join());
    }

    #[test]
    fn test_index_only_scan_access_method() {
        let method = AccessMethod::IndexOnlyScan {
            index_name: "idx_test".to_string(),
            columns: vec!["a".to_string(), "b".to_string()],
            selectivity: 0.1,
            index_type: IndexType::BTree,
        };

        let desc = method.description();
        assert!(desc.contains("Index Only Scan"));
        assert!(desc.contains("idx_test"));
    }

    // =========================================================================
    // Correlation-Aware Selectivity Tests
    // =========================================================================

    #[test]
    fn test_combined_selectivity_independent() {
        let estimator = CostEstimator::new();
        // No correlations provided - should multiply
        let selectivities = [("col1", 0.1), ("col2", 0.1)];
        let combined = estimator.estimate_combined_selectivity(&selectivities, None);

        // Independent columns: 0.1 * 0.1 = 0.01
        assert!(
            (combined - 0.01).abs() < 0.001,
            "Independent columns should multiply: expected 0.01, got {}",
            combined
        );
    }

    #[test]
    fn test_combined_selectivity_correlated() {
        let estimator = CostEstimator::new();
        let mut correlations = ColumnCorrelations::new();
        correlations.add_correlation("city", "state", 0.99); // Highly correlated

        // city = 'NYC' (0.02) AND state = 'NY' (0.02)
        let selectivities = [("city", 0.02), ("state", 0.02)];
        let combined = estimator.estimate_combined_selectivity(&selectivities, Some(&correlations));

        // With high correlation, combined should be much higher than naive 0.0004
        assert!(
            combined > 0.01,
            "Correlated columns: expected > 0.01, got {}",
            combined
        );

        // Should not be higher than most selective single predicate
        assert!(
            combined <= 0.02 * 1.1, // Allow 10% tolerance
            "Combined should not exceed single predicate selectivity significantly: {}",
            combined
        );
    }

    #[test]
    fn test_combined_selectivity_functional_dep() {
        let estimator = CostEstimator::new();
        let mut correlations = ColumnCorrelations::new();
        correlations.add_functional_dep("zip_code", "city"); // FD implies correlation = 1.0

        // zip = '10001' (0.001) AND city = 'NYC' (0.02)
        // Since zip determines city, combined selectivity ≈ 0.001
        let selectivities = [("zip_code", 0.001), ("city", 0.02)];
        let combined = estimator.estimate_combined_selectivity(&selectivities, Some(&correlations));

        // Should be close to the more selective predicate
        assert!(
            (0.0005..=0.005).contains(&combined),
            "FD: combined should be near zip selectivity, got {}",
            combined
        );
    }

    #[test]
    fn test_and_selectivity_convenience() {
        let estimator = CostEstimator::new();
        let mut correlations = ColumnCorrelations::new();
        correlations.add_correlation("a", "b", 0.8);

        let mut selectivities = std::collections::HashMap::new();
        selectivities.insert("a".to_string(), 0.1);
        selectivities.insert("b".to_string(), 0.1);

        let combined = estimator.estimate_and_selectivity(&selectivities, Some(&correlations));

        // With 0.8 correlation, should be between 0.01 and 0.1
        assert!(
            combined > 0.01 && combined < 0.1,
            "Partial correlation: expected between 0.01 and 0.1, got {}",
            combined
        );
    }

    #[test]
    fn test_seq_scan_with_predicates() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(10000, 100);
        let mut correlations = ColumnCorrelations::new();
        correlations.add_correlation("city", "state", 0.95);

        // Without correlations
        let selectivities_no_corr = [("city", 0.02), ("state", 0.02)];
        let cost_no_corr =
            estimator.estimate_seq_scan_with_predicates(&stats, &selectivities_no_corr, None);

        // With correlations
        let cost_with_corr = estimator.estimate_seq_scan_with_predicates(
            &stats,
            &selectivities_no_corr,
            Some(&correlations),
        );

        // With correlations, we expect MORE output rows (higher selectivity)
        assert!(
            cost_with_corr.estimated_rows > cost_no_corr.estimated_rows,
            "Correlation should increase estimated rows: {} > {}",
            cost_with_corr.estimated_rows,
            cost_no_corr.estimated_rows
        );
    }

    #[test]
    fn test_three_column_correlation() {
        let estimator = CostEstimator::new();
        let mut correlations = ColumnCorrelations::new();
        correlations.add_correlation("city", "state", 0.99);
        correlations.add_correlation("city", "country", 0.95);
        correlations.add_correlation("state", "country", 0.90);

        // city = 'NYC' AND state = 'NY' AND country = 'USA'
        let selectivities = [("city", 0.01), ("state", 0.02), ("country", 0.05)];
        let combined = estimator.estimate_combined_selectivity(&selectivities, Some(&correlations));

        // All three are correlated - result should be close to most selective
        assert!(
            combined >= 0.005,
            "Three correlated columns: expected >= 0.005, got {}",
            combined
        );

        // Should be much higher than naive: 0.01 * 0.02 * 0.05 = 0.00001
        let naive: f64 = selectivities.iter().map(|(_, s)| *s).product();
        assert!(
            combined > naive * 100.0,
            "Should be much higher than naive: {} >> {}",
            combined,
            naive
        );
    }

    // =========================================================================
    // LIMIT Pushdown Tests
    // =========================================================================

    #[test]
    fn test_choose_join_algorithm_with_small_limit() {
        let estimator = CostEstimator::new();

        // Large tables where hash join would normally be chosen
        let join_stats = make_join_stats(100000, 100000, 10000, 10000);

        // Without LIMIT - should choose hash join for large tables
        let (algo_no_limit, _) =
            estimator.choose_join_algorithm_with_limit(&join_stats, true, None);
        assert!(
            algo_no_limit.is_hash_join(),
            "Without LIMIT, large tables should use hash join"
        );

        // With small LIMIT - may prefer nested loop for early termination
        let (_algo_small_limit, cost_small) =
            estimator.choose_join_algorithm_with_limit(&join_stats, true, Some(10));

        // The cost explanation should reflect LIMIT awareness
        assert!(cost_small.total > 0.0, "Cost should be positive");

        // With large LIMIT - should behave like no LIMIT
        let (algo_large_limit, _) =
            estimator.choose_join_algorithm_with_limit(&join_stats, true, Some(50000));
        assert!(
            algo_large_limit.is_hash_join(),
            "Large LIMIT should behave like no LIMIT"
        );
    }

    #[test]
    fn test_limit_pushdown_without_equality_keys() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 10000, 1000, 1000);

        // Without equality keys, always use nested loop regardless of LIMIT
        let (algo, _) = estimator.choose_join_algorithm_with_limit(&join_stats, false, Some(10));
        assert!(
            algo.is_nested_loop(),
            "Without equality keys, must use nested loop"
        );

        let (algo_no_limit, _) =
            estimator.choose_join_algorithm_with_limit(&join_stats, false, None);
        assert!(
            algo_no_limit.is_nested_loop(),
            "Without equality keys, must use nested loop"
        );
    }

    #[test]
    fn test_sorted_inputs_prefer_merge_join() {
        let estimator = CostEstimator::new();

        // Medium-sized tables where sorted inputs make merge join attractive
        let join_stats = make_join_stats(10000, 10000, 1000, 1000);

        // Without sorted inputs - should choose hash join
        let (algo_unsorted, _) = estimator.choose_join_algorithm_with_sorted_inputs(
            &join_stats,
            true,  // has_equality_keys
            None,  // no limit
            false, // left not sorted
            false, // right not sorted
        );
        assert!(
            algo_unsorted.is_hash_join(),
            "Without sorted inputs, should prefer hash join"
        );

        // With both inputs sorted - should choose merge join
        let (algo_sorted, cost_sorted) = estimator.choose_join_algorithm_with_sorted_inputs(
            &join_stats,
            true, // has_equality_keys
            None, // no limit
            true, // left sorted
            true, // right sorted
        );
        assert!(
            algo_sorted.is_merge_join(),
            "With both inputs sorted, should prefer merge join"
        );

        // Verify merge join reports sorted status
        if let JoinAlgorithm::MergeJoin {
            left_sorted,
            right_sorted,
            ..
        } = algo_sorted
        {
            assert!(left_sorted, "Merge join should record left as sorted");
            assert!(right_sorted, "Merge join should record right as sorted");
        }

        // The sorted merge join should have lower cost than hash
        let (_, cost_unsorted) = estimator.choose_join_algorithm_with_sorted_inputs(
            &join_stats,
            true,
            None,
            false,
            false,
        );
        assert!(
            cost_sorted.total <= cost_unsorted.total,
            "Sorted merge join should have comparable or lower cost"
        );
    }

    #[test]
    fn test_sorted_inputs_with_only_one_side() {
        let estimator = CostEstimator::new();
        let join_stats = make_join_stats(10000, 10000, 1000, 1000);

        // With only left sorted - should NOT choose merge join (needs both sides sorted)
        let (algo_left_only, _) = estimator.choose_join_algorithm_with_sorted_inputs(
            &join_stats,
            true,  // has_equality_keys
            None,  // no limit
            true,  // left sorted
            false, // right not sorted
        );
        assert!(
            !algo_left_only.is_merge_join(),
            "With only one side sorted, should not use merge join"
        );

        // With only right sorted - should NOT choose merge join
        let (algo_right_only, _) = estimator.choose_join_algorithm_with_sorted_inputs(
            &join_stats,
            true,  // has_equality_keys
            None,  // no limit
            false, // left not sorted
            true,  // right sorted
        );
        assert!(
            !algo_right_only.is_merge_join(),
            "With only one side sorted, should not use merge join"
        );
    }

    #[test]
    fn test_parallel_cost_thresholds() {
        let estimator = CostEstimator::new();

        // Below threshold - should not use parallel
        assert!(!estimator.should_parallel_filter(5_000));
        assert!(!estimator.should_parallel_scan(5_000));

        // At or above threshold - should use parallel
        assert!(estimator.should_parallel_filter(10_000));
        assert!(estimator.should_parallel_scan(10_000));
        assert!(estimator.should_parallel_filter(50_000));

        // Join threshold
        assert!(!estimator.should_parallel_join(3_000));
        assert!(estimator.should_parallel_join(5_000));

        // Sort threshold
        assert!(!estimator.should_parallel_sort(40_000));
        assert!(estimator.should_parallel_sort(50_000));
    }

    #[test]
    fn test_parallel_speedup_factor() {
        let estimator = CostEstimator::new();

        // With multiple workers, speedup should be > 1
        let speedup_small = estimator.parallel_speedup_factor(10_000);
        let speedup_large = estimator.parallel_speedup_factor(500_000);

        // Larger datasets should have better speedup
        assert!(speedup_small >= 1.0);
        assert!(speedup_large >= speedup_small);
    }

    #[test]
    fn test_parallel_vs_sequential_cost() {
        let estimator = CostEstimator::new();

        // Small table - sequential should be used (parallel overhead not worth it)
        let small_stats = make_table_stats(1_000, 10);
        let (cost_small, is_parallel_small) = estimator.choose_scan_method(&small_stats, 0.5);
        assert!(!is_parallel_small, "Small tables should not use parallel");
        assert!(cost_small.total > 0.0);

        // Large table - parallel should be considered
        let large_stats = make_table_stats(100_000, 1_000);
        let (cost_large, is_parallel_large) = estimator.choose_scan_method(&large_stats, 0.5);
        // On multi-core systems, parallel should be cheaper
        if rayon::current_num_threads() > 1 {
            assert!(
                is_parallel_large,
                "Large tables should use parallel on multi-core"
            );
        }
        assert!(cost_large.total > 0.0);
    }

    #[test]
    fn test_parallel_cost_estimate() {
        let estimator = CostEstimator::new();
        let stats = make_table_stats(50_000, 500);

        let sequential = estimator.estimate_seq_scan_with_filter(&stats, 0.1);
        let parallel = estimator.estimate_parallel_seq_scan_with_filter(&stats, 0.1);

        // Parallel should have lower total cost on multi-core (due to speedup)
        if rayon::current_num_threads() > 1 {
            assert!(
                parallel.total < sequential.total,
                "Parallel cost ({:.2}) should be lower than sequential ({:.2})",
                parallel.total,
                sequential.total
            );
        }

        // Both should produce same estimated output rows
        assert_eq!(parallel.estimated_rows, sequential.estimated_rows);

        // Parallel explanation should mention workers
        assert!(
            parallel.explanation.contains("Parallel"),
            "Parallel plan should indicate parallel execution"
        );
    }
}
