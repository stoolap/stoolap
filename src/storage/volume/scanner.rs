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

//! Scanner implementation for frozen volumes.
//!
//! Implements the `Scanner` trait so frozen volumes can be used by the executor
//! through the same interface as live tables. The scanner supports:
//! - Column projection (only reconstruct Values for needed columns)
//! - Range filtering on sorted columns (binary search start position)
//! - Zone map pruning (skip entire volume if predicate doesn't match)
//!
//! This is the bridge between the column-major volume storage and the
//! row-major executor. Values are reconstructed lazily, one row at a time.

use std::sync::Arc;

use crate::core::{Error, Result, Row, Value};
use crate::storage::traits::Scanner;

use super::writer::FrozenVolume;

// =============================================================================
// Columnar pre-filter: typed predicates evaluated directly on column data.
// Avoids full Value reconstruction for rows that don't match.
// =============================================================================

/// Typed target for columnar pre-filter comparison.
/// Operates on raw column data without constructing Value objects.
enum TypedTarget {
    Int64(i64),
    Float64(f64),
    Bool(bool),
}

/// A predicate that can be evaluated directly on typed column data.
/// Extracted from the WHERE clause during `set_filter()`.
/// Safety invariant: must never reject a row that should match.
struct ColumnPredicate {
    col_idx: usize,
    op: crate::core::Operator,
    target: TypedTarget,
}

/// Scanner over a frozen volume that implements the `Scanner` trait.
///
/// Reconstructs rows lazily from column-major data, projecting only
/// the requested columns. Skips rows marked as deleted in the segment-scoped
/// delete vector. Optionally evaluates a predicate to skip non-matching rows
/// without full Value construction.
pub struct VolumeScanner {
    /// Shared reference to the frozen volume
    volume: Arc<FrozenVolume>,
    /// Column indices to project (empty = all columns)
    project_cols: Vec<usize>,
    /// Pre-computed flag: true when project_cols is an identity mapping over
    /// all volume columns. Avoids recomputing this check on every row.
    is_full_projection: bool,
    /// Current scan position
    current_idx: usize,
    /// End position (exclusive) — may be less than row_count for filtered scans
    end_idx: usize,
    /// Current reconstructed row
    current_row: Row,
    /// Row ID of the current row (from segment's row_ids array)
    current_rid: i64,
    /// Whether we have a valid current row
    has_current: bool,
    /// Precomputed column mapping (None = volume matches current schema).
    /// When set and not identity, replaces per-row name-based normalization
    /// with per-row index lookup through the mapping.
    column_mapping: Option<super::writer::ColumnMapping>,
    /// Any error that occurred
    error: Option<Error>,
    /// Optional predicate filter (from WHERE clause pushdown)
    filter: Option<Box<dyn crate::storage::expression::Expression>>,
    /// Pre-resolved dictionary filters: (col_idx, dict_id) pairs.
    /// Enables O(1) u32 comparison per row instead of full Value reconstruction.
    dict_filters: Vec<(usize, u32)>,
    /// Pre-computed matching row indices (when dictionary filters narrow enough).
    /// When set, iteration skips the linear scan entirely.
    matching_indices: Option<Vec<usize>>,
    /// Current position in matching_indices.
    match_idx: usize,
    /// Per-transaction pending cold deletes (deferred, not yet in shared DV).
    /// The owning transaction sees these as deleted; other transactions don't.
    pending_cold_deletes: Option<Arc<rustc_hash::FxHashSet<i64>>>,
    /// Committed tombstones (shared, immutable Arc reference — no clone).
    /// Kept separate from pending_cold_deletes to avoid cloning the tombstone set.
    /// Map: row_id → commit_seq (for snapshot isolation filtering).
    committed_tombstones: Option<Arc<rustc_hash::FxHashMap<i64, u64>>>,
    /// Snapshot sequence: if Some, only tombstones with commit_seq <= this are visible.
    /// None means auto-commit (all tombstones visible).
    pub snapshot_seq: Option<u64>,
    /// Typed pre-filter predicates extracted from the WHERE clause.
    /// Evaluated directly on column data without Value construction.
    typed_predicates: Vec<ColumnPredicate>,
    /// Precomputed set of columns needed for filter + projection.
    /// When set, the filter path materializes only these columns instead
    /// of all columns. Built in set_filter() from filter's referenced
    /// columns ∪ project_cols. None = materialize all (fallback).
    needed_cols: Option<Vec<bool>>,
    /// Pre-computed row group skip decisions. group_idx → can skip entirely.
    /// None = no row groups (small volume or no filter). Computed in set_filter().
    row_group_skips: Option<Vec<bool>>,
    /// Cached end of the current row group (exclusive index). Avoids per-row
    /// integer division in the slow-path scan loop. Recomputed only on group
    /// boundary crossings. 0 means "not yet initialized".
    next_group_boundary: usize,
}

impl VolumeScanner {
    /// Compute whether `project_cols` is an identity mapping over all volume columns.
    /// Extracted as a helper so both constructors share the same logic.
    #[inline]
    fn compute_is_full_projection(project_cols: &[usize], num_cols: usize) -> bool {
        project_cols.len() == num_cols && project_cols.iter().enumerate().all(|(i, &c)| c == i)
    }

    /// Create a scanner over all rows in the volume.
    pub fn new(
        volume: Arc<FrozenVolume>,
        project_cols: Vec<usize>,
        _delete_vector: Option<()>,
    ) -> Self {
        let project = if project_cols.is_empty() {
            (0..volume.columns.len()).collect()
        } else {
            project_cols
        };
        let is_full_projection = Self::compute_is_full_projection(&project, volume.columns.len());
        Self {
            end_idx: volume.row_count,
            volume,
            project_cols: project,
            is_full_projection,
            current_idx: 0,
            current_row: Row::new(),
            current_rid: 0,
            has_current: false,
            error: None,
            filter: None,
            column_mapping: None,
            dict_filters: Vec::new(),
            matching_indices: None,
            match_idx: 0,
            pending_cold_deletes: None,
            committed_tombstones: None,
            snapshot_seq: None,
            typed_predicates: Vec::new(),
            needed_cols: None,
            row_group_skips: None,
            next_group_boundary: 0,
        }
    }

    /// Create a scanner with a start/end range (for binary-search narrowing).
    pub fn with_range(
        volume: Arc<FrozenVolume>,
        project_cols: Vec<usize>,
        start_idx: usize,
        end_idx: usize,
        _delete_vector: Option<()>,
    ) -> Self {
        let project = if project_cols.is_empty() {
            (0..volume.columns.len()).collect()
        } else {
            project_cols
        };
        let is_full_projection = Self::compute_is_full_projection(&project, volume.columns.len());
        Self {
            volume,
            project_cols: project,
            is_full_projection,
            current_idx: start_idx,
            end_idx,
            current_row: Row::new(),
            current_rid: 0,
            has_current: false,
            error: None,
            filter: None,
            column_mapping: None,
            dict_filters: Vec::new(),
            matching_indices: None,
            match_idx: 0,
            pending_cold_deletes: None,
            committed_tombstones: None,
            snapshot_seq: None,
            typed_predicates: Vec::new(),
            needed_cols: None,
            row_group_skips: None,
            next_group_boundary: 0,
        }
    }

    /// Set per-transaction pending cold deletes. The owning transaction
    /// sees these row_ids as deleted even though the shared DV hasn't
    /// been updated yet (deferred to commit).
    pub fn set_pending_cold_deletes(&mut self, pending: Arc<rustc_hash::FxHashSet<i64>>) {
        self.pending_cold_deletes = Some(pending);
    }

    /// Set both committed tombstones (shared Arc, no clone) and dynamic
    /// skip set (hot row_ids + pending tombstones + per-volume dedup IDs).
    /// This avoids cloning the potentially large committed tombstone set.
    pub fn set_skip_sets(
        &mut self,
        committed: Arc<rustc_hash::FxHashMap<i64, u64>>,
        dynamic: Arc<rustc_hash::FxHashSet<i64>>,
    ) {
        self.committed_tombstones = Some(committed);
        self.pending_cold_deletes = Some(dynamic);
    }

    /// Create an empty scanner (for zone-map-pruned volumes that match nothing).
    pub fn empty() -> Self {
        Self {
            volume: Arc::new(FrozenVolume {
                columns: Vec::new(),
                zone_maps: Vec::new(),
                bloom_filters: Vec::new(),
                stats: super::stats::VolumeAggregateStats::new(0),
                row_count: 0,
                column_names: Vec::new(),
                column_types: Vec::new(),
                row_ids: Vec::new(),
                sorted_columns: Vec::new(),
                column_name_map: ahash::AHashMap::new(),
                unique_indices: parking_lot::RwLock::new(rustc_hash::FxHashMap::default()),
                row_groups: Vec::new(),
            }),
            project_cols: Vec::new(),
            is_full_projection: true,
            current_idx: 0,
            end_idx: 0,
            current_row: Row::new(),
            current_rid: 0,
            has_current: false,
            error: None,
            filter: None,
            column_mapping: None,
            dict_filters: Vec::new(),
            matching_indices: None,
            match_idx: 0,
            pending_cold_deletes: None,
            committed_tombstones: None,
            snapshot_seq: None,
            typed_predicates: Vec::new(),
            needed_cols: None,
            row_group_skips: None,
            next_group_boundary: 0,
        }
    }

    /// Set a predicate filter on this scanner.
    /// Automatically extracts dictionary-based fast filters for text equality predicates.
    pub fn set_filter(&mut self, filter: Box<dyn crate::storage::expression::Expression>) {
        // Extract dictionary filters for fast pre-filtering
        let comparisons = filter.collect_comparisons();
        for &(col_name, op, value) in &comparisons {
            if op != crate::core::Operator::Eq {
                continue;
            }
            if let Value::Text(s) = value {
                if let Some(col_idx) = self.volume.column_index(col_name) {
                    if let Some(dict_id) = self.volume.columns[col_idx].dict_lookup(s.as_str()) {
                        self.dict_filters.push((col_idx, dict_id));
                    } else {
                        // Value not in dictionary = zero matches, set empty range
                        self.current_idx = self.end_idx;
                        self.filter = Some(filter);
                        return;
                    }
                }
            }
        }
        // If we have dictionary filters, pre-compute matching row indices.
        // This turns the scanner from O(N) iteration into O(matches) iteration.
        if !self.dict_filters.is_empty() {
            let mut matches = Vec::new();
            for i in self.current_idx..self.end_idx {
                let mut row_matches = true;
                for &(col_idx, expected_id) in &self.dict_filters {
                    if self.volume.columns[col_idx].is_null(i)
                        || self.volume.columns[col_idx].get_dict_id(i) != expected_id
                    {
                        row_matches = false;
                        break;
                    }
                }
                if row_matches {
                    matches.push(i);
                }
            }
            self.matching_indices = Some(matches);
        }
        // Extract typed pre-filter predicates for non-text columns.
        // These are evaluated directly on raw column data (get_i64, get_f64, get_bool)
        // without constructing Value objects. Only AND-connected simple comparisons
        // are extracted (collect_comparisons handles this).
        for &(col_name, op, value) in &comparisons {
            if !matches!(
                op,
                crate::core::Operator::Eq
                    | crate::core::Operator::Ne
                    | crate::core::Operator::Gt
                    | crate::core::Operator::Gte
                    | crate::core::Operator::Lt
                    | crate::core::Operator::Lte
            ) {
                continue;
            }
            let col_idx = match self.volume.column_index(col_name) {
                Some(idx) => idx,
                None => continue,
            };
            // Extract typed target matching the column's storage type.
            // Text equality is already handled by dict_filters above.
            let target = match (value, &self.volume.columns[col_idx]) {
                (Value::Integer(v), super::column::ColumnData::Int64 { .. }) => {
                    TypedTarget::Int64(*v)
                }
                (Value::Float(v), super::column::ColumnData::Float64 { .. }) => {
                    TypedTarget::Float64(*v)
                }
                (Value::Boolean(v), super::column::ColumnData::Boolean { .. }) => {
                    TypedTarget::Bool(*v)
                }
                (Value::Timestamp(dt), super::column::ColumnData::TimestampNanos { .. }) => {
                    TypedTarget::Int64(dt.timestamp_nanos_opt().unwrap_or_else(|| {
                        dt.timestamp()
                            .saturating_mul(1_000_000_000)
                            .saturating_add(dt.timestamp_subsec_nanos() as i64)
                    }))
                }
                (Value::Integer(v), super::column::ColumnData::Float64 { .. }) => {
                    TypedTarget::Float64(*v as f64)
                }
                _ => continue,
            };
            self.typed_predicates.push(ColumnPredicate {
                col_idx,
                op,
                target,
            });
        }

        // Try to extract which columns the filter references.
        // If successful, combine with project_cols to build a bitmask
        // of columns needed during filter evaluation. This enables
        // column pruning: only those columns are materialized from the
        // column store, skipping expensive Text/JSON clones for
        // unreferenced columns.
        let mut filter_cols = Vec::new();
        if filter.collect_column_indices(&mut filter_cols) {
            let num_cols = self.volume.columns.len();
            // Use the larger of volume columns and mapping sources length
            // to handle schema-evolved volumes.
            let mask_len = if let Some(ref m) = self.column_mapping {
                m.sources.len().max(num_cols)
            } else {
                num_cols
            };
            let mut mask = vec![false; mask_len];
            for &ci in &filter_cols {
                if ci < mask_len {
                    mask[ci] = true;
                }
            }
            for &ci in &self.project_cols {
                if ci < mask_len {
                    mask[ci] = true;
                }
            }
            self.needed_cols = Some(mask);
        }

        // Pre-compute row group skip decisions from per-group zone maps.
        // For each group, if ANY comparison's zone map says "no match",
        // the entire group can be skipped.
        if !self.volume.row_groups.is_empty() && !comparisons.is_empty() {
            let skips: Vec<bool> = self
                .volume
                .row_groups
                .iter()
                .map(|rg| {
                    for &(col_name, op, value) in &comparisons {
                        let col_idx = match self.volume.column_index(col_name) {
                            Some(idx) if idx < rg.zone_maps.len() => idx,
                            _ => continue,
                        };
                        let zm = &rg.zone_maps[col_idx];
                        let dominated = match op {
                            crate::core::Operator::Eq => !zm.may_contain_eq(value),
                            crate::core::Operator::Gt | crate::core::Operator::Gte => {
                                !zm.may_contain_gte(value)
                            }
                            crate::core::Operator::Lt | crate::core::Operator::Lte => {
                                !zm.may_contain_lte(value)
                            }
                            _ => false,
                        };
                        if dominated {
                            return true; // skip this group
                        }
                    }
                    false
                })
                .collect();
            // Only store if at least one group can be skipped
            if skips.iter().any(|&s| s) {
                self.row_group_skips = Some(skips);
            }
        }

        self.filter = Some(filter);
    }

    /// Evaluate typed pre-filter predicates directly on column data.
    /// Returns false only if the row definitely does not match (safe rejection).
    /// NULL columns conservatively pass through (the full filter handles NULL logic).
    #[inline]
    fn evaluate_typed_predicates(&self, idx: usize) -> bool {
        for pred in &self.typed_predicates {
            let col = &self.volume.columns[pred.col_idx];
            if col.is_null(idx) {
                // NULL: conservatively pass through (might match under SQL NULL semantics).
                // The full filter will handle it correctly.
                continue;
            }
            let matches = match &pred.target {
                TypedTarget::Int64(target) => {
                    let val = col.get_i64(idx);
                    match pred.op {
                        crate::core::Operator::Eq => val == *target,
                        crate::core::Operator::Ne => val != *target,
                        crate::core::Operator::Gt => val > *target,
                        crate::core::Operator::Gte => val >= *target,
                        crate::core::Operator::Lt => val < *target,
                        crate::core::Operator::Lte => val <= *target,
                        _ => true,
                    }
                }
                TypedTarget::Float64(target) => {
                    let val = col.get_f64(idx);
                    match pred.op {
                        crate::core::Operator::Eq => val == *target,
                        crate::core::Operator::Ne => val != *target,
                        crate::core::Operator::Gt => val > *target,
                        crate::core::Operator::Gte => val >= *target,
                        crate::core::Operator::Lt => val < *target,
                        crate::core::Operator::Lte => val <= *target,
                        _ => true,
                    }
                }
                TypedTarget::Bool(target) => {
                    let val = col.get_bool(idx);
                    match pred.op {
                        crate::core::Operator::Eq => val == *target,
                        crate::core::Operator::Ne => val != *target,
                        _ => true,
                    }
                }
            };
            if !matches {
                return false;
            }
        }
        true
    }

    /// Set a precomputed column mapping for schema-evolved volumes.
    /// Only stores it if the mapping is non-identity (avoids overhead
    /// when the volume matches the current schema).
    pub fn set_column_mapping(&mut self, mapping: super::writer::ColumnMapping) {
        if !mapping.is_identity {
            self.column_mapping = Some(mapping);
        }
    }

    // =========================================================================
    // Shared helpers for both fast path (matching_indices) and slow path
    // (linear scan). Extracted to eliminate code duplication — a single source
    // of truth for skip checks and row materialization.
    // =========================================================================

    /// Check tombstones and pending deletes for a row index. Returns true if
    /// the row should be skipped.
    #[inline(always)]
    fn should_skip_row(&self, idx: usize) -> bool {
        let rid = self.volume.row_ids[idx];
        if let Some(ref ts) = self.committed_tombstones {
            if let Some(&commit_seq) = ts.get(&rid) {
                if self.snapshot_seq.is_none_or(|ss| commit_seq <= ss) {
                    return true;
                }
            }
        }
        if let Some(ref pending) = self.pending_cold_deletes {
            if pending.contains(&rid) {
                return true;
            }
        }
        false
    }

    /// Check dictionary pre-filters for a row index. Returns true if the row
    /// does NOT match (should be skipped). Only called when dict_filters is
    /// non-empty.
    #[inline(always)]
    fn dict_filters_reject(&self, idx: usize) -> bool {
        for &(col_idx, expected_id) in &self.dict_filters {
            if self.volume.columns[col_idx].is_null(idx)
                || self.volume.columns[col_idx].get_dict_id(idx) != expected_id
            {
                return true;
            }
        }
        false
    }

    /// Materialize a row at `idx`, evaluate the filter (if any), and write
    /// the result into `self.current_row`. Returns false if the filter
    /// rejects the row.
    #[inline(always)]
    fn materialize_row(&mut self, idx: usize) -> bool {
        if let Some(ref filter) = self.filter {
            let full_row = match (&self.needed_cols, &self.column_mapping) {
                (Some(mask), Some(mapping)) => {
                    self.volume.get_row_mapped_needed(idx, mapping, mask)
                }
                (Some(mask), None) => self.volume.get_row_needed(idx, mask),
                (None, Some(mapping)) => self.volume.get_row_mapped(idx, mapping),
                (None, None) => self.volume.get_row(idx),
            };
            if !filter.evaluate_fast(&full_row) {
                return false;
            }
            if self.is_full_projection {
                self.current_row = full_row;
            } else {
                self.current_row = Row::from_values(
                    self.project_cols
                        .iter()
                        .map(|&col| {
                            full_row
                                .get(col)
                                .cloned()
                                .unwrap_or(Value::Null(crate::core::DataType::Null))
                        })
                        .collect(),
                );
            }
        } else if let Some(ref mapping) = self.column_mapping {
            if self.is_full_projection {
                self.current_row = self.volume.get_row_mapped(idx, mapping);
            } else {
                self.current_row =
                    self.volume
                        .get_row_mapped_projected(idx, mapping, &self.project_cols);
            }
        } else if self.is_full_projection {
            self.current_row = self.volume.get_row(idx);
        } else {
            self.current_row = self.volume.get_row_projected(idx, &self.project_cols);
        }
        true
    }
}

impl Scanner for VolumeScanner {
    fn next(&mut self) -> bool {
        if self.error.is_some() {
            self.has_current = false;
            return false;
        }

        // Fast path: use pre-computed matching indices (from dictionary filters).
        // Copy indices length to avoid holding an immutable borrow on self
        // across the mutable materialize_row call.
        if self.matching_indices.is_some() {
            loop {
                let idx = match self.matching_indices.as_ref() {
                    Some(indices) if self.match_idx < indices.len() => {
                        let i = indices[self.match_idx];
                        self.match_idx += 1;
                        i
                    }
                    _ => {
                        self.has_current = false;
                        return false;
                    }
                };

                if self.should_skip_row(idx) {
                    continue;
                }
                if !self.typed_predicates.is_empty() && !self.evaluate_typed_predicates(idx) {
                    continue;
                }
                if !self.materialize_row(idx) {
                    continue;
                }

                self.current_rid = self.volume.row_ids[idx];
                self.has_current = true;
                return true;
            }
        }

        // Slow path: linear scan with row-group skipping
        while self.current_idx < self.end_idx {
            // Row-group skip: jump over entire groups whose zone maps prove
            // no rows can match. Uses a cached boundary to avoid per-row
            // integer division — only recomputed on group transitions.
            if let Some(ref skips) = self.row_group_skips {
                if self.current_idx >= self.next_group_boundary {
                    let group_idx = self.current_idx / super::column::ROW_GROUP_SIZE;
                    self.next_group_boundary =
                        ((group_idx + 1) * super::column::ROW_GROUP_SIZE).min(self.end_idx);
                    if group_idx < skips.len() && skips[group_idx] {
                        self.current_idx = self.next_group_boundary;
                        continue;
                    }
                }
            }

            if self.should_skip_row(self.current_idx) {
                self.current_idx += 1;
                continue;
            }

            let idx = self.current_idx;

            if !self.dict_filters.is_empty() && self.dict_filters_reject(idx) {
                self.current_idx += 1;
                continue;
            }
            if !self.typed_predicates.is_empty() && !self.evaluate_typed_predicates(idx) {
                self.current_idx += 1;
                continue;
            }
            if !self.materialize_row(idx) {
                self.current_idx += 1;
                continue;
            }

            self.current_rid = self.volume.row_ids[idx];
            self.has_current = true;
            self.current_idx += 1;
            return true;
        }

        self.has_current = false;
        false
    }

    fn row(&self) -> &Row {
        &self.current_row
    }

    fn err(&self) -> Option<&Error> {
        self.error.as_ref()
    }

    fn close(&mut self) -> Result<()> {
        self.has_current = false;
        Ok(())
    }

    fn take_row(&mut self) -> Row {
        self.has_current = false;
        std::mem::take(&mut self.current_row)
    }

    fn take_row_with_id(&mut self) -> (i64, Row) {
        let rid = self.current_rid;
        self.has_current = false;
        (rid, std::mem::take(&mut self.current_row))
    }

    fn current_row_id(&self) -> i64 {
        self.current_rid
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.end_idx.saturating_sub(self.current_idx))
    }
}

/// Scanner that merges results from multiple sources (hot buffer + volumes).
///
/// This is the key integration point: a query over a table with frozen volumes
/// first scans the volumes (column-major, possibly zone-map-pruned), then
/// scans the hot buffer (current live rows). The executor sees a single
/// unified Scanner.
pub struct MergingScanner {
    /// Scanners to merge (processed in order: volumes first, hot buffer last)
    sources: Vec<Box<dyn Scanner>>,
    /// Index of the current active source
    current_source: usize,
}

impl MergingScanner {
    /// Create a merging scanner from multiple sources.
    ///
    /// Sources are scanned in order. Typically:
    /// `[volume_0_scanner, volume_1_scanner, ..., hot_buffer_scanner]`
    pub fn new(sources: Vec<Box<dyn Scanner>>) -> Self {
        Self {
            sources,
            current_source: 0,
        }
    }
}

impl Scanner for MergingScanner {
    fn next(&mut self) -> bool {
        while self.current_source < self.sources.len() {
            if self.sources[self.current_source].next() {
                return true;
            }
            // Check for errors before moving to next source
            if self.sources[self.current_source].err().is_some() {
                return false;
            }
            self.current_source += 1;
        }
        false
    }

    fn row(&self) -> &Row {
        debug_assert!(
            self.current_source < self.sources.len(),
            "row() called after iteration completed"
        );
        self.sources[self.current_source].row()
    }

    fn current_row_id(&self) -> i64 {
        if self.current_source < self.sources.len() {
            self.sources[self.current_source].current_row_id()
        } else {
            0
        }
    }

    fn err(&self) -> Option<&Error> {
        if self.current_source < self.sources.len() {
            self.sources[self.current_source].err()
        } else {
            None
        }
    }

    fn close(&mut self) -> Result<()> {
        for source in &mut self.sources {
            source.close()?;
        }
        Ok(())
    }

    fn take_row(&mut self) -> Row {
        debug_assert!(
            self.current_source < self.sources.len(),
            "take_row() called after iteration completed"
        );
        self.sources[self.current_source].take_row()
    }

    fn take_row_with_id(&mut self) -> (i64, Row) {
        debug_assert!(
            self.current_source < self.sources.len(),
            "take_row_with_id() called after iteration completed"
        );
        self.sources[self.current_source].take_row_with_id()
    }

    fn estimated_count(&self) -> Option<usize> {
        let mut total = 0usize;
        for source in &self.sources {
            total += source.estimated_count()?;
        }
        Some(total)
    }
}

/// Scanner backed by a pre-collected RowVec.
///
/// Used by SegmentedTable::scan() to wrap eagerly-collected hot rows.
/// This enables scan() to derive the cold skip set from actual hot results,
/// preventing the race where remove_sealed_rows runs between skip set
/// construction and hot scanner execution.
pub struct RowVecScanner {
    rows: crate::core::RowVec,
    index: usize,
    empty_row: Row,
}

impl RowVecScanner {
    pub fn new(rows: crate::core::RowVec) -> Self {
        Self {
            rows,
            index: 0,
            empty_row: Row::new(),
        }
    }
}

impl Scanner for RowVecScanner {
    fn next(&mut self) -> bool {
        if self.index < self.rows.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn row(&self) -> &Row {
        if self.index > 0 && self.index <= self.rows.len() {
            &self.rows[self.index - 1].1
        } else {
            &self.empty_row
        }
    }

    fn current_row_id(&self) -> i64 {
        if self.index > 0 && self.index <= self.rows.len() {
            self.rows[self.index - 1].0
        } else {
            0
        }
    }

    fn take_row(&mut self) -> Row {
        self.row().clone()
    }

    fn take_row_with_id(&mut self) -> (i64, Row) {
        if self.index > 0 && self.index <= self.rows.len() {
            let (id, ref row) = self.rows[self.index - 1];
            (id, row.clone())
        } else {
            (0, Row::new())
        }
    }

    fn err(&self) -> Option<&Error> {
        None
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.rows.len())
    }
}

#[cfg(test)]
mod tests {
    use super::super::writer::VolumeBuilder;
    use super::*;
    use crate::core::{DataType, SchemaBuilder};

    fn make_test_volume() -> Arc<FrozenVolume> {
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build();

        let mut builder = VolumeBuilder::with_capacity(&schema, 5);
        builder.add_row(
            1,
            &Row::from_values(vec![
                Value::Integer(1),
                Value::text("apple"),
                Value::Float(1.50),
            ]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![
                Value::Integer(2),
                Value::text("banana"),
                Value::Float(0.75),
            ]),
        );
        builder.add_row(
            3,
            &Row::from_values(vec![
                Value::Integer(3),
                Value::text("cherry"),
                Value::Float(3.00),
            ]),
        );
        builder.add_row(
            4,
            &Row::from_values(vec![
                Value::Integer(4),
                Value::text("date"),
                Value::Float(5.00),
            ]),
        );
        builder.add_row(
            5,
            &Row::from_values(vec![
                Value::Integer(5),
                Value::text("elderberry"),
                Value::Float(8.00),
            ]),
        );
        Arc::new(builder.finish())
    }

    #[test]
    fn test_full_scan() {
        let vol = make_test_volume();
        let mut scanner = VolumeScanner::new(vol, vec![], None);

        let mut count = 0;
        while scanner.next() {
            let row = scanner.row();
            assert_eq!(row.len(), 3);
            count += 1;
        }
        assert_eq!(count, 5);
        assert!(scanner.err().is_none());
    }

    #[test]
    fn test_projected_scan() {
        let vol = make_test_volume();
        // Only scan name and price (columns 1, 2)
        let mut scanner = VolumeScanner::new(vol, vec![1, 2], None);

        assert!(scanner.next());
        let row = scanner.row();
        assert_eq!(row.len(), 2);
        assert_eq!(row.get(0), Some(&Value::text("apple")));
        assert_eq!(row.get(1), Some(&Value::Float(1.50)));
    }

    #[test]
    fn test_range_scan() {
        let vol = make_test_volume();
        // Scan rows 2..4 (indices 2, 3)
        let mut scanner = VolumeScanner::with_range(Arc::clone(&vol), vec![], 2, 4, None);

        let mut count = 0;
        let mut ids = Vec::new();
        while scanner.next() {
            if let Some(Value::Integer(id)) = scanner.row().get(0) {
                ids.push(*id);
            }
            count += 1;
        }
        assert_eq!(count, 2);
        assert_eq!(ids, vec![3, 4]); // rows at index 2 and 3
    }

    #[test]
    fn test_empty_scanner() {
        let mut scanner = VolumeScanner::empty();
        assert!(!scanner.next());
        assert!(scanner.err().is_none());
    }

    #[test]
    fn test_take_row() {
        let vol = make_test_volume();
        let mut scanner = VolumeScanner::new(vol, vec![0], None);

        assert!(scanner.next());
        let row = scanner.take_row();
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
    }

    #[test]
    fn test_merging_scanner() {
        let vol = make_test_volume();

        // Create two scanners: first 2 rows, then last 2 rows
        let scanner1 = Box::new(VolumeScanner::with_range(
            Arc::clone(&vol),
            vec![0],
            0,
            2,
            None,
        ));
        let scanner2 = Box::new(VolumeScanner::with_range(
            Arc::clone(&vol),
            vec![0],
            3,
            5,
            None,
        ));

        let mut merger = MergingScanner::new(vec![scanner1, scanner2]);

        let mut ids = Vec::new();
        while merger.next() {
            if let Some(Value::Integer(id)) = merger.row().get(0) {
                ids.push(*id);
            }
        }
        assert_eq!(ids, vec![1, 2, 4, 5]); // rows 0,1 from first, rows 3,4 from second
        assert!(merger.err().is_none());
    }

    #[test]
    fn test_estimated_count() {
        let vol = make_test_volume();
        let scanner = VolumeScanner::new(Arc::clone(&vol), vec![], None);
        assert_eq!(scanner.estimated_count(), Some(5));

        let scanner = VolumeScanner::with_range(vol, vec![], 2, 4, None);
        assert_eq!(scanner.estimated_count(), Some(2));
    }
}
