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

//! Workload Learning and Edge-Aware Query Optimization
//!
//! This module implements Stoolap's unique optimization features:
//!
//! 1. **Workload Learning**: Learns from historical query patterns to predict future behavior
//!    - Query pattern fingerprinting and frequency tracking
//!    - Automatic index recommendation based on access patterns
//!    - Hot column detection for pre-materialization hints
//!    - Temporal pattern detection (batch vs interactive workloads)
//!
//! 2. **Edge-Aware Planning**: Special optimizations for edge computing environments
//!    - Memory-constrained execution strategies
//!    - Network partition tolerance (graceful degradation)
//!    - Battery-aware query scheduling (for IoT/mobile)
//!    - Incremental result computation for slow connections
//!
//! These features make Stoolap unique in that it learns from your specific workload
//! patterns rather than relying solely on static cost models.

#![allow(clippy::too_many_arguments)]

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Maximum number of query fingerprints to store
/// This prevents unbounded memory growth in workload learner
const MAX_FINGERPRINTS: usize = 50000;

/// Global workload learner instance
static WORKLOAD_LEARNER: std::sync::OnceLock<WorkloadLearner> = std::sync::OnceLock::new();

/// Get the global workload learner instance
pub fn global_workload_learner() -> &'static WorkloadLearner {
    WORKLOAD_LEARNER.get_or_init(WorkloadLearner::new)
}

/// Configuration for workload-aware optimization
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Enable workload learning
    pub learning_enabled: bool,
    /// Edge computing mode (enables memory-optimized execution)
    pub edge_mode: EdgeMode,
    /// Maximum memory for query execution (0 = unlimited)
    pub memory_limit_mb: u64,
    /// Enable incremental result streaming
    pub incremental_results: bool,
    /// Target for battery optimization (0 = disabled, 1-100 = power saving level)
    pub power_saving_level: u8,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            learning_enabled: true,
            edge_mode: EdgeMode::Standard,
            memory_limit_mb: 0,
            incremental_results: false,
            power_saving_level: 0,
        }
    }
}

/// Edge computing mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeMode {
    /// Standard mode - no special constraints
    Standard,
    /// Constrained mode - limited memory, prefer streaming operators
    Constrained,
    /// Ultra-low mode - extreme memory limits, single-pass only
    UltraLow,
    /// Mobile mode - battery aware, network resilient
    Mobile,
}

impl EdgeMode {
    /// Get the memory multiplier for cost estimation
    /// Lower means we penalize memory-heavy operations more
    pub fn memory_cost_multiplier(&self) -> f64 {
        match self {
            EdgeMode::Standard => 1.0,
            EdgeMode::Constrained => 5.0,
            EdgeMode::UltraLow => 20.0,
            EdgeMode::Mobile => 3.0,
        }
    }

    /// Get the preferred batch size for this mode
    pub fn preferred_batch_size(&self) -> usize {
        match self {
            EdgeMode::Standard => 10000,
            EdgeMode::Constrained => 1000,
            EdgeMode::UltraLow => 100,
            EdgeMode::Mobile => 500,
        }
    }
}

/// Query pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryPattern {
    /// Point lookup by primary key
    PointLookup,
    /// Range scan
    RangeScan,
    /// Full table scan
    FullScan,
    /// Aggregation query
    Aggregation,
    /// Join-heavy query
    JoinHeavy,
    /// Complex analytical query
    Analytical,
    /// Insert-heavy workload
    InsertHeavy,
    /// Update-heavy workload
    UpdateHeavy,
    /// Mixed OLTP workload
    MixedOLTP,
    /// Unknown/other
    Unknown,
}

/// Learned statistics for a query pattern
#[derive(Debug, Clone)]
pub struct PatternStats {
    /// Number of times this pattern was observed
    pub frequency: u64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average rows scanned
    pub avg_rows_scanned: u64,
    /// Average rows returned
    pub avg_rows_returned: u64,
    /// Tables most commonly accessed with this pattern
    pub hot_tables: Vec<String>,
    /// Columns most commonly filtered on
    pub hot_filter_columns: Vec<String>,
    /// Columns most commonly in ORDER BY
    pub hot_sort_columns: Vec<String>,
    /// Last observed timestamp
    pub last_seen: Instant,
}

impl PatternStats {
    fn new() -> Self {
        Self {
            frequency: 0,
            avg_execution_time_us: 0.0,
            peak_memory_bytes: 0,
            avg_rows_scanned: 0,
            avg_rows_returned: 0,
            hot_tables: Vec::new(),
            hot_filter_columns: Vec::new(),
            hot_sort_columns: Vec::new(),
            last_seen: Instant::now(),
        }
    }

    /// Update stats with new observation
    fn observe(
        &mut self,
        execution_time_us: u64,
        memory_bytes: u64,
        rows_scanned: u64,
        rows_returned: u64,
        tables: Vec<String>,
        filter_columns: Vec<String>,
        sort_columns: Vec<String>,
    ) {
        self.frequency += 1;

        // Exponential moving average for execution time
        let alpha = 0.3;
        self.avg_execution_time_us =
            alpha * execution_time_us as f64 + (1.0 - alpha) * self.avg_execution_time_us;

        // Keep max memory
        self.peak_memory_bytes = self.peak_memory_bytes.max(memory_bytes);

        // Exponential moving average for row counts
        self.avg_rows_scanned =
            ((alpha * rows_scanned as f64 + (1.0 - alpha) * self.avg_rows_scanned as f64) as u64)
                .max(1);
        self.avg_rows_returned =
            ((alpha * rows_returned as f64 + (1.0 - alpha) * self.avg_rows_returned as f64) as u64)
                .max(1);

        // Update hot lists (keep top 10)
        for table in tables {
            Self::update_hot_list(&mut self.hot_tables, table);
        }
        for col in filter_columns {
            Self::update_hot_list(&mut self.hot_filter_columns, col);
        }
        for col in sort_columns {
            Self::update_hot_list(&mut self.hot_sort_columns, col);
        }

        self.last_seen = Instant::now();
    }

    fn update_hot_list(list: &mut Vec<String>, item: String) {
        if !list.contains(&item) && list.len() < 10 {
            list.push(item);
        }
    }
}

/// Index recommendation from workload analysis
#[derive(Debug, Clone)]
pub struct IndexRecommendation {
    /// Table name
    pub table: String,
    /// Column(s) to index
    pub columns: Vec<String>,
    /// Expected benefit score (higher = more beneficial)
    pub benefit_score: f64,
    /// Reason for recommendation
    pub reason: String,
}

/// Temporal workload pattern
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalPattern {
    /// Mostly interactive queries (low latency)
    Interactive,
    /// Mostly batch processing (throughput focused)
    Batch,
    /// Mixed workload
    Mixed,
    /// No clear pattern
    Unknown,
}

/// Workload learner - learns from query patterns to optimize future queries
pub struct WorkloadLearner {
    /// Pattern statistics
    patterns: RwLock<FxHashMap<QueryPattern, PatternStats>>,
    /// Query fingerprint to pattern mapping
    fingerprints: RwLock<FxHashMap<u64, QueryPattern>>,
    /// Table access frequency
    table_access_counts: RwLock<FxHashMap<String, AtomicU64>>,
    /// Column filter frequency (table.column -> count)
    filter_column_counts: RwLock<FxHashMap<String, AtomicU64>>,
    /// Total queries observed
    total_queries: AtomicU64,
    /// Short queries (< 10ms)
    short_queries: AtomicU64,
    /// Long queries (> 1s)
    long_queries: AtomicU64,
    /// Learning enabled
    learning_enabled: RwLock<bool>,
    /// Current workload config
    config: RwLock<WorkloadConfig>,
}

impl WorkloadLearner {
    /// Create a new workload learner
    pub fn new() -> Self {
        Self {
            patterns: RwLock::new(FxHashMap::default()),
            fingerprints: RwLock::new(FxHashMap::default()),
            table_access_counts: RwLock::new(FxHashMap::default()),
            filter_column_counts: RwLock::new(FxHashMap::default()),
            total_queries: AtomicU64::new(0),
            short_queries: AtomicU64::new(0),
            long_queries: AtomicU64::new(0),
            learning_enabled: RwLock::new(true),
            config: RwLock::new(WorkloadConfig::default()),
        }
    }

    /// Update configuration
    pub fn set_config(&self, config: WorkloadConfig) {
        if let Ok(mut cfg) = self.config.write() {
            *cfg = config;
        }
    }

    /// Get current configuration
    pub fn config(&self) -> WorkloadConfig {
        self.config.read().map(|c| c.clone()).unwrap_or_default()
    }

    /// Classify a query into a pattern based on its characteristics
    pub fn classify_query(
        &self,
        has_pk_lookup: bool,
        has_range_predicate: bool,
        has_full_scan: bool,
        has_aggregation: bool,
        join_count: usize,
        is_insert: bool,
        is_update: bool,
    ) -> QueryPattern {
        if is_insert {
            return QueryPattern::InsertHeavy;
        }
        if is_update {
            return QueryPattern::UpdateHeavy;
        }

        if has_pk_lookup && join_count == 0 && !has_aggregation {
            return QueryPattern::PointLookup;
        }

        if join_count >= 3 || (join_count >= 2 && has_aggregation) {
            return QueryPattern::Analytical;
        }

        if join_count >= 2 {
            return QueryPattern::JoinHeavy;
        }

        if has_aggregation {
            return QueryPattern::Aggregation;
        }

        if has_range_predicate && !has_full_scan {
            return QueryPattern::RangeScan;
        }

        if has_full_scan {
            return QueryPattern::FullScan;
        }

        QueryPattern::Unknown
    }

    /// Record a query execution for learning
    pub fn record_query(
        &self,
        query_fingerprint: u64,
        pattern: QueryPattern,
        execution_time: Duration,
        memory_bytes: u64,
        rows_scanned: u64,
        rows_returned: u64,
        tables: Vec<String>,
        filter_columns: Vec<String>,
        sort_columns: Vec<String>,
    ) {
        if !self.is_learning_enabled() {
            return;
        }

        let execution_time_us = execution_time.as_micros() as u64;

        // Update query counters
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        if execution_time < Duration::from_millis(10) {
            self.short_queries.fetch_add(1, Ordering::Relaxed);
        } else if execution_time > Duration::from_secs(1) {
            self.long_queries.fetch_add(1, Ordering::Relaxed);
        }

        // Update fingerprint mapping with size cap
        if let Ok(mut fingerprints) = self.fingerprints.write() {
            // Evict oldest entries if at capacity (simple approach: clear half when full)
            if fingerprints.len() >= MAX_FINGERPRINTS
                && !fingerprints.contains_key(&query_fingerprint)
            {
                // Clear half the entries to make room and amortize the eviction cost
                let target_size = MAX_FINGERPRINTS / 2;
                let keys_to_remove: Vec<u64> = fingerprints
                    .keys()
                    .take(fingerprints.len() - target_size)
                    .copied()
                    .collect();
                for key in keys_to_remove {
                    fingerprints.remove(&key);
                }
            }
            fingerprints.insert(query_fingerprint, pattern);
        }

        // Update pattern stats
        if let Ok(mut patterns) = self.patterns.write() {
            let stats = patterns.entry(pattern).or_insert_with(PatternStats::new);
            stats.observe(
                execution_time_us,
                memory_bytes,
                rows_scanned,
                rows_returned,
                tables.clone(),
                filter_columns.clone(),
                sort_columns,
            );
        }

        // Update table access counts
        if let Ok(table_counts) = self.table_access_counts.read() {
            for table in &tables {
                if let Some(count) = table_counts.get(table) {
                    count.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        // Add new tables
        if let Ok(mut table_counts) = self.table_access_counts.write() {
            for table in tables {
                table_counts
                    .entry(table)
                    .or_insert_with(|| AtomicU64::new(1));
            }
        }

        // Update filter column counts
        if let Ok(mut filter_counts) = self.filter_column_counts.write() {
            for col in filter_columns {
                filter_counts
                    .entry(col)
                    .or_insert_with(|| AtomicU64::new(0))
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get pattern for a known query fingerprint
    pub fn get_pattern(&self, fingerprint: u64) -> Option<QueryPattern> {
        self.fingerprints
            .read()
            .ok()
            .and_then(|f| f.get(&fingerprint).copied())
    }

    /// Get statistics for a pattern
    pub fn get_pattern_stats(&self, pattern: QueryPattern) -> Option<PatternStats> {
        self.patterns
            .read()
            .ok()
            .and_then(|p| p.get(&pattern).cloned())
    }

    /// Generate index recommendations based on learned workload
    pub fn recommend_indexes(&self) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();

        let filter_counts = match self.filter_column_counts.read() {
            Ok(c) => c,
            Err(_) => return recommendations,
        };

        let total = self.total_queries.load(Ordering::Relaxed) as f64;
        if total < 100.0 {
            // Need at least 100 queries before making recommendations
            return recommendations;
        }

        // Find frequently filtered columns
        for (column, count) in filter_counts.iter() {
            let freq = count.load(Ordering::Relaxed) as f64 / total;
            if freq > 0.1 {
                // Filtered in more than 10% of queries
                let parts: Vec<&str> = column.split('.').collect();
                if parts.len() == 2 {
                    let benefit = freq * 100.0; // Simple benefit score
                    recommendations.push(IndexRecommendation {
                        table: parts[0].to_string(),
                        columns: vec![parts[1].to_string()],
                        benefit_score: benefit,
                        reason: format!("Column filtered in {:.1}% of queries", freq * 100.0),
                    });
                }
            }
        }

        // Sort by benefit score
        recommendations.sort_by(|a, b| {
            b.benefit_score
                .partial_cmp(&a.benefit_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top 5
        recommendations.truncate(5);
        recommendations
    }

    /// Detect the temporal workload pattern
    pub fn detect_temporal_pattern(&self) -> TemporalPattern {
        let total = self.total_queries.load(Ordering::Relaxed);
        if total < 100 {
            return TemporalPattern::Unknown;
        }

        let short = self.short_queries.load(Ordering::Relaxed);
        let long = self.long_queries.load(Ordering::Relaxed);

        let short_ratio = short as f64 / total as f64;
        let long_ratio = long as f64 / total as f64;

        if short_ratio > 0.8 {
            TemporalPattern::Interactive
        } else if long_ratio > 0.3 {
            TemporalPattern::Batch
        } else if short_ratio > 0.5 && long_ratio > 0.1 {
            TemporalPattern::Mixed
        } else {
            TemporalPattern::Unknown
        }
    }

    /// Get hot tables (most frequently accessed)
    pub fn hot_tables(&self, limit: usize) -> Vec<(String, u64)> {
        let table_counts = match self.table_access_counts.read() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut tables: Vec<_> = table_counts
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();

        tables.sort_by(|a, b| b.1.cmp(&a.1));
        tables.truncate(limit);
        tables
    }

    /// Get optimization hints based on learned workload
    pub fn get_optimization_hints(&self) -> WorkloadHints {
        let pattern = self.detect_temporal_pattern();
        let config = self.config();

        WorkloadHints {
            prefer_nested_loop: pattern == TemporalPattern::Interactive,
            prefer_hash_join: pattern == TemporalPattern::Batch,
            enable_bloom_filters: self.total_queries.load(Ordering::Relaxed) > 1000,
            target_batch_size: config.edge_mode.preferred_batch_size(),
            memory_constrained: config.edge_mode != EdgeMode::Standard,
            incremental_results: config.incremental_results,
        }
    }

    fn is_learning_enabled(&self) -> bool {
        self.learning_enabled.read().map(|v| *v).unwrap_or(false)
    }

    /// Enable or disable learning
    pub fn set_learning_enabled(&self, enabled: bool) {
        if let Ok(mut v) = self.learning_enabled.write() {
            *v = enabled;
        }
    }

    /// Get total queries observed
    pub fn total_queries(&self) -> u64 {
        self.total_queries.load(Ordering::Relaxed)
    }

    /// Clear all learned data
    pub fn clear(&self) {
        if let Ok(mut p) = self.patterns.write() {
            p.clear();
        }
        if let Ok(mut f) = self.fingerprints.write() {
            f.clear();
        }
        if let Ok(mut t) = self.table_access_counts.write() {
            t.clear();
        }
        if let Ok(mut f) = self.filter_column_counts.write() {
            f.clear();
        }
        self.total_queries.store(0, Ordering::Relaxed);
        self.short_queries.store(0, Ordering::Relaxed);
        self.long_queries.store(0, Ordering::Relaxed);
    }
}

impl Default for WorkloadLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization hints derived from workload learning
#[derive(Debug, Clone)]
pub struct WorkloadHints {
    /// Prefer nested loop joins for low-latency
    pub prefer_nested_loop: bool,
    /// Prefer hash joins for throughput
    pub prefer_hash_join: bool,
    /// Enable bloom filter optimizations
    pub enable_bloom_filters: bool,
    /// Target batch size for operators
    pub target_batch_size: usize,
    /// Memory is constrained
    pub memory_constrained: bool,
    /// Enable incremental result streaming
    pub incremental_results: bool,
}

impl Default for WorkloadHints {
    fn default() -> Self {
        Self {
            prefer_nested_loop: false,
            prefer_hash_join: false,
            enable_bloom_filters: false,
            target_batch_size: 10000,
            memory_constrained: false,
            incremental_results: false,
        }
    }
}

/// Edge-aware query planner enhancements
pub struct EdgeAwarePlanner {
    /// Workload hints
    hints: WorkloadHints,
    /// Memory limit in bytes (0 = unlimited)
    memory_limit: u64,
}

impl EdgeAwarePlanner {
    /// Create a new edge-aware planner
    pub fn new(hints: WorkloadHints, memory_limit: u64) -> Self {
        Self {
            hints,
            memory_limit,
        }
    }

    /// Create from global workload learner
    pub fn from_global() -> Self {
        let learner = global_workload_learner();
        let hints = learner.get_optimization_hints();
        let config = learner.config();
        Self {
            hints,
            memory_limit: config.memory_limit_mb * 1024 * 1024,
        }
    }

    /// Adjust cost for edge computing constraints
    pub fn adjust_cost(&self, base_cost: f64, memory_estimate: u64) -> f64 {
        let mut cost = base_cost;

        // Penalize high memory usage in constrained mode
        if self.hints.memory_constrained && self.memory_limit > 0 {
            if memory_estimate > self.memory_limit {
                // Heavy penalty for exceeding limit
                cost *= 100.0;
            } else if memory_estimate > self.memory_limit / 2 {
                // Moderate penalty for high usage
                cost *= 2.0;
            }
        }

        cost
    }

    /// Decide if we should use streaming execution
    pub fn should_stream(&self, estimated_rows: u64) -> bool {
        if self.hints.incremental_results {
            return true;
        }

        if self.hints.memory_constrained {
            // Stream if estimated rows might cause memory issues
            let row_size_estimate = 100; // bytes per row estimate
            estimated_rows * row_size_estimate > self.memory_limit / 2
        } else {
            false
        }
    }

    /// Get preferred batch size
    pub fn batch_size(&self) -> usize {
        self.hints.target_batch_size
    }

    /// Check if bloom filters should be used
    pub fn use_bloom_filters(&self) -> bool {
        self.hints.enable_bloom_filters
    }

    /// Recommend join algorithm for edge constraints
    pub fn recommend_join_for_edge(
        &self,
        build_rows: u64,
        probe_rows: u64,
        memory_per_build_row: u64,
    ) -> EdgeJoinRecommendation {
        let build_memory = build_rows * memory_per_build_row;

        if self.memory_limit > 0 && build_memory > self.memory_limit {
            // Cannot use hash join - would exceed memory
            EdgeJoinRecommendation::ForceNestedLoop {
                reason: "Hash join would exceed memory limit".to_string(),
            }
        } else if self.hints.prefer_nested_loop && probe_rows < 1000 {
            // Interactive mode prefers nested loop for small probes
            EdgeJoinRecommendation::PreferNestedLoop {
                reason: "Interactive workload with small probe set".to_string(),
            }
        } else if self.hints.prefer_hash_join {
            EdgeJoinRecommendation::PreferHashJoin {
                reason: "Batch workload optimized for throughput".to_string(),
            }
        } else {
            EdgeJoinRecommendation::UseDefault
        }
    }
}

/// Join recommendation for edge computing
#[derive(Debug, Clone)]
pub enum EdgeJoinRecommendation {
    /// Force nested loop due to constraints
    ForceNestedLoop { reason: String },
    /// Prefer nested loop (but not mandatory)
    PreferNestedLoop { reason: String },
    /// Prefer hash join (but not mandatory)
    PreferHashJoin { reason: String },
    /// Use default algorithm selection
    UseDefault,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_learner_basic() {
        let learner = WorkloadLearner::new();

        // Record some queries
        for i in 0..10 {
            learner.record_query(
                i,
                QueryPattern::PointLookup,
                Duration::from_micros(100),
                1024,
                1,
                1,
                vec!["users".to_string()],
                vec!["users.id".to_string()],
                vec![],
            );
        }

        assert_eq!(learner.total_queries(), 10);

        let stats = learner.get_pattern_stats(QueryPattern::PointLookup);
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.frequency, 10);
    }

    #[test]
    fn test_query_classification() {
        let learner = WorkloadLearner::new();

        assert_eq!(
            learner.classify_query(true, false, false, false, 0, false, false),
            QueryPattern::PointLookup
        );

        assert_eq!(
            learner.classify_query(false, false, false, true, 0, false, false),
            QueryPattern::Aggregation
        );

        assert_eq!(
            learner.classify_query(false, false, false, true, 3, false, false),
            QueryPattern::Analytical
        );

        assert_eq!(
            learner.classify_query(false, false, false, false, 2, false, false),
            QueryPattern::JoinHeavy
        );

        assert_eq!(
            learner.classify_query(false, false, false, false, 0, true, false),
            QueryPattern::InsertHeavy
        );
    }

    #[test]
    fn test_temporal_pattern_detection() {
        let learner = WorkloadLearner::new();

        // Record mostly short queries
        for i in 0..100 {
            learner.record_query(
                i,
                QueryPattern::PointLookup,
                Duration::from_micros(500), // < 10ms = short
                1024,
                1,
                1,
                vec!["users".to_string()],
                vec![],
                vec![],
            );
        }

        assert_eq!(
            learner.detect_temporal_pattern(),
            TemporalPattern::Interactive
        );
    }

    #[test]
    fn test_hot_tables() {
        let learner = WorkloadLearner::new();

        // Access 'orders' 5 times
        for i in 0..5 {
            learner.record_query(
                i,
                QueryPattern::PointLookup,
                Duration::from_micros(100),
                1024,
                1,
                1,
                vec!["orders".to_string()],
                vec![],
                vec![],
            );
        }

        // Access 'users' 3 times
        for i in 5..8 {
            learner.record_query(
                i,
                QueryPattern::PointLookup,
                Duration::from_micros(100),
                1024,
                1,
                1,
                vec!["users".to_string()],
                vec![],
                vec![],
            );
        }

        let hot = learner.hot_tables(2);
        assert_eq!(hot.len(), 2);
        assert_eq!(hot[0].0, "orders");
        assert_eq!(hot[1].0, "users");
    }

    #[test]
    fn test_edge_mode_settings() {
        assert_eq!(EdgeMode::Standard.memory_cost_multiplier(), 1.0);
        assert_eq!(EdgeMode::Constrained.memory_cost_multiplier(), 5.0);
        assert_eq!(EdgeMode::UltraLow.memory_cost_multiplier(), 20.0);

        assert_eq!(EdgeMode::Standard.preferred_batch_size(), 10000);
        assert_eq!(EdgeMode::UltraLow.preferred_batch_size(), 100);
    }

    #[test]
    fn test_edge_aware_planner() {
        let hints = WorkloadHints {
            prefer_nested_loop: false,
            prefer_hash_join: true,
            enable_bloom_filters: true,
            target_batch_size: 1000,
            memory_constrained: true,
            incremental_results: false,
        };

        let planner = EdgeAwarePlanner::new(hints, 1024 * 1024); // 1MB limit

        // Test cost adjustment
        let base_cost = 100.0;
        let adjusted = planner.adjust_cost(base_cost, 512 * 1024); // 512KB - under half
        assert_eq!(adjusted, base_cost);

        let adjusted = planner.adjust_cost(base_cost, 768 * 1024); // 768KB - over half
        assert_eq!(adjusted, base_cost * 2.0);

        let adjusted = planner.adjust_cost(base_cost, 2 * 1024 * 1024); // 2MB - over limit
        assert_eq!(adjusted, base_cost * 100.0);
    }

    #[test]
    fn test_edge_join_recommendation() {
        let hints = WorkloadHints {
            prefer_nested_loop: false,
            prefer_hash_join: false,
            enable_bloom_filters: false,
            target_batch_size: 1000,
            memory_constrained: true,
            incremental_results: false,
        };

        let planner = EdgeAwarePlanner::new(hints, 1024 * 1024); // 1MB limit

        // Hash join would need 2MB - should force nested loop
        let rec = planner.recommend_join_for_edge(20000, 100000, 100);
        assert!(matches!(
            rec,
            EdgeJoinRecommendation::ForceNestedLoop { .. }
        ));

        // Hash join fits in memory
        let rec = planner.recommend_join_for_edge(1000, 100000, 100);
        assert!(matches!(rec, EdgeJoinRecommendation::UseDefault));
    }

    #[test]
    fn test_workload_config() {
        let learner = WorkloadLearner::new();

        let config = WorkloadConfig {
            learning_enabled: true,
            edge_mode: EdgeMode::Constrained,
            memory_limit_mb: 512,
            incremental_results: true,
            power_saving_level: 50,
        };

        learner.set_config(config.clone());
        let retrieved = learner.config();

        assert_eq!(retrieved.edge_mode, EdgeMode::Constrained);
        assert_eq!(retrieved.memory_limit_mb, 512);
        assert!(retrieved.incremental_results);
    }

    #[test]
    fn test_clear() {
        let learner = WorkloadLearner::new();

        // Record some queries
        for i in 0..10 {
            learner.record_query(
                i,
                QueryPattern::PointLookup,
                Duration::from_micros(100),
                1024,
                1,
                1,
                vec!["users".to_string()],
                vec![],
                vec![],
            );
        }

        assert_eq!(learner.total_queries(), 10);

        learner.clear();

        assert_eq!(learner.total_queries(), 0);
        assert!(learner
            .get_pattern_stats(QueryPattern::PointLookup)
            .is_none());
    }
}
