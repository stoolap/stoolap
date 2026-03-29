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

//! SegmentedTable: wraps an MVCCTable (hot buffer) with immutable segments.
//!
//! Write operations delegate to the inner MVCCTable unchanged.
//! Read operations merge results from segments + hot buffer.
//! Aggregation pushdowns use pre-computed segment stats when possible.
//!
//! Key design: volume rows do NOT live in normal MVCC secondary indexes.
//! Hot indexes only cover hot rows. Cold data has its own immutable access
//! path via segment zone maps, binary search, dictionary pre-filters, and
//! (future) segment sidecars.
//!
//! This is the Table trait implementation that makes the executor unaware
//! of whether data lives in memory or frozen segments.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::{DataType, IndexType, Result, Row, RowVec, Schema, Value, ValueMap, ValueSet};
use crate::storage::expression::Expression;
use crate::storage::mvcc::version_store::{AggregateOp, GroupedAggregateResult};
use crate::storage::traits::table::ScanPlan;
use crate::storage::traits::{Index, QueryResult, Scanner, Table};

use super::manifest::SegmentManager;
use super::scanner::{RowVecScanner, VolumeScanner};
use super::writer::FrozenVolume;

/// A table backed by immutable segments (historical) + an MVCCTable (hot buffer).
///
/// The executor sees a single Table interface. Reads merge across all sources.
/// Writes go exclusively to the hot buffer (MVCCTable).
///
/// Visibility: cold segment rows are skipped via per-volume skip sets built
/// from hot row_ids and tombstones. Newer volumes shadow older ones.
///
/// Normal secondary indexes exist only for hot rows. Volume rows are never
/// inserted into hot indexes. Constraint checks (PK/UNIQUE) against cold data
/// use segment metadata (zone maps, sorted columns, dictionary pre-filters).
pub struct SegmentedTable {
    /// The hot buffer (current in-memory MVCC table for writes)
    hot: Box<dyn Table>,
    /// Segment manager (shared, engine-owned): segments, tombstones, manifest
    segment_mgr: Arc<SegmentManager>,
    /// Snapshot sequence for snapshot isolation transactions.
    /// If Some(seq), only tombstones with commit_seq <= seq are visible,
    /// preserving the snapshot's point-in-time view of cold data.
    /// None for auto-commit transactions (all tombstones visible).
    snapshot_seq: Option<u64>,
}

impl SegmentedTable {
    /// Create a segmented table from a hot buffer and a segment manager.
    pub fn new(hot: Box<dyn Table>, segment_mgr: Arc<SegmentManager>) -> Self {
        Self {
            hot,
            segment_mgr,
            snapshot_seq: None,
        }
    }

    /// Create a segmented table with a snapshot sequence for snapshot isolation.
    /// Only tombstones with commit_seq <= snapshot_seq are visible to this table.
    pub fn with_snapshot_seq(
        hot: Box<dyn Table>,
        segment_mgr: Arc<SegmentManager>,
        snapshot_seq: u64,
    ) -> Self {
        Self {
            hot,
            segment_mgr,
            snapshot_seq: Some(snapshot_seq),
        }
    }

    /// Create a segmented table with no segments (equivalent to plain MVCCTable).
    pub fn hot_only(hot: Box<dyn Table>) -> Self {
        Self {
            segment_mgr: Arc::new(SegmentManager::new("", None)),
            hot,
            snapshot_seq: None,
        }
    }

    /// Get the transaction ID for per-txn tombstone tracking.
    fn txn_id(&self) -> i64 {
        self.hot.txn_id()
    }

    /// Check if a tombstone is visible to this table's snapshot.
    /// For auto-commit (snapshot_seq=None), all tombstones are visible.
    /// For snapshot isolation (snapshot_seq=Some(seq)), only tombstones
    /// with commit_seq <= seq are visible — newer tombstones are invisible,
    /// so the original cold row remains visible to the older snapshot.
    #[inline]
    fn is_tombstone_visible(&self, commit_seq: u64) -> bool {
        self.snapshot_seq.is_none_or(|ss| commit_seq <= ss)
    }

    /// Check if a row_id is tombstoned and visible to this snapshot.
    #[inline]
    fn is_row_tombstoned(&self, tombstones: &FxHashMap<i64, u64>, row_id: i64) -> bool {
        tombstones
            .get(&row_id)
            .is_some_and(|&seq| self.is_tombstone_visible(seq))
    }

    /// Get the schema default for a column, or typed NULL if no default.
    #[inline]
    fn column_default(&self, col_idx: usize) -> Value {
        let schema = self.hot.schema();
        if col_idx < schema.columns.len() {
            let col = &schema.columns[col_idx];
            col.default_value
                .clone()
                .unwrap_or_else(|| Value::null(col.data_type))
        } else {
            Value::Null(crate::core::DataType::Null)
        }
    }

    /// Get the number of frozen segments.
    pub fn segment_count(&self) -> usize {
        self.segment_mgr.segment_count()
    }

    /// Get the segment manager.
    pub fn segment_manager(&self) -> &Arc<SegmentManager> {
        &self.segment_mgr
    }

    /// Validate that cold segment data has no duplicate values for a unique index.
    /// Called before CREATE UNIQUE INDEX to prevent certifying already-invalid data.
    fn validate_cold_unique(&self, index_name: &str, columns: &[&str]) -> Result<()> {
        let schema = self.hot.schema();
        let col_indices: Vec<usize> = columns
            .iter()
            .filter_map(|c| {
                schema
                    .columns
                    .iter()
                    .position(|sc| sc.name_lower == c.to_lowercase())
            })
            .collect();
        if col_indices.len() != columns.len() {
            return Ok(());
        }

        let volumes = self.segment_mgr.get_volumes_newest_first();
        let ts = self.segment_mgr.tombstone_set_arc();
        let mut seen_values: ahash::AHashMap<Vec<Value>, i64> = ahash::AHashMap::new();

        // Build skip set: hot row_ids + pending tombstones for this transaction.
        // Inside an explicit transaction, uncommitted UPDATEs create pending
        // local versions that shadow cold rows. has_row_id only sees committed
        // rows, so we also need pending tombstones to skip rows being modified.
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&ts, rid) {
                    continue;
                }
                // Skip rows shadowed by hot buffer or pending transaction changes
                if hot_skip.contains(&rid) {
                    continue;
                }
                let values: Vec<Value> = col_indices
                    .iter()
                    .map(|&ci| {
                        if ci < vol.columns.len() {
                            vol.columns[ci].get_value(i)
                        } else {
                            self.column_default(ci)
                        }
                    })
                    .collect();
                if values.iter().any(|v| v.is_null()) {
                    continue;
                }
                if let Some(&existing_rid) = seen_values.get(&values) {
                    return Err(crate::core::Error::UniqueConstraint {
                        index: index_name.to_string(),
                        column: columns.join(", "),
                        value: values
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                        row_id: existing_rid,
                    });
                }
                seen_values.insert(values, rid);
            }
        }
        Ok(())
    }

    /// Populate an HNSW index from cold segment data.
    /// Called after index creation on the hot store.
    /// Populate an HNSW (or unique HNSW) index from cold segment data.
    /// Propagates errors so unique-constraint violations are not swallowed.
    fn populate_index_from_cold(&self, name: &str, columns: &[&str]) -> Result<()> {
        let index = match self.hot.get_index(name) {
            Some(idx) => idx,
            None => return Ok(()),
        };
        let schema = self.hot.schema();
        let col_indices: Vec<usize> = columns
            .iter()
            .filter_map(|c| {
                schema
                    .columns
                    .iter()
                    .position(|sc| sc.name_lower == c.to_lowercase())
            })
            .collect();
        if col_indices.is_empty() {
            return Ok(());
        }
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let ts = self.segment_mgr.tombstone_set_arc();
        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&ts, rid) {
                    continue;
                }
                let values: Vec<Value> = col_indices
                    .iter()
                    .map(|&ci| {
                        if ci < vol.columns.len() {
                            vol.columns[ci].get_value(i)
                        } else {
                            self.column_default(ci)
                        }
                    })
                    .collect();
                if !values.iter().any(|v| v.is_null()) {
                    index.add(&values, rid, rid)?;
                }
            }
        }
        Ok(())
    }

    /// Get a fast approximate row count across all segments.
    /// Does NOT deduplicate overlapping row_ids. Use for hints only.
    fn segment_row_count_hint(&self) -> usize {
        self.segment_mgr.total_row_count()
    }

    /// Find the first visible cold row matching a set of equality predicates.
    ///
    /// This is used by both unique constraint checks and ON CONFLICT probing so
    /// volume-backed upserts can find the conflicting cold row_id directly.
    /// Delegates to SegmentManager which handles tombstone filtering internally.
    fn find_segment_row_id_by_values(
        &self,
        col_indices: &[usize],
        values: &[&Value],
    ) -> Option<i64> {
        let snapshot = self.segment_mgr.cold_snapshot();
        self.find_segment_row_id_by_values_with_snapshot(&snapshot, col_indices, values)
    }

    fn find_segment_row_id_by_values_with_snapshot(
        &self,
        snapshot: &super::manifest::ColdSnapshot,
        col_indices: &[usize],
        values: &[&Value],
    ) -> Option<i64> {
        if col_indices.is_empty() || col_indices.len() != values.len() {
            return None;
        }

        let defaults: smallvec::SmallVec<[Value; 4]> = col_indices
            .iter()
            .map(|&ci| self.column_default(ci))
            .collect();
        let result = self.segment_mgr.find_row_id_by_values_with_snapshot(
            snapshot,
            col_indices,
            values,
            &defaults,
        );
        let rid = result?;

        if self.segment_mgr.is_pending_tombstone(self.txn_id(), rid) {
            return None;
        }

        Some(rid)
    }

    /// Check PK and UNIQUE constraints against segment data before INSERT.
    ///
    /// Uses zone maps, bloom filters, dictionary pre-filters, and binary search
    /// on sorted columns for fast rejection. No index population needed.
    /// Check cold unique constraints for UPDATE, excluding the row being updated.
    fn check_cold_unique_for_update(&self, new_row: &Row, exclude_row_id: i64) -> Result<()> {
        let schema = self.hot.schema();
        self.hot
            .for_each_unique_non_pk_index(&mut |idx_name, col_names| {
                let col_indices: Vec<usize> = col_names
                    .iter()
                    .filter_map(|name| schema.columns.iter().position(|c| c.name_lower == *name))
                    .collect();
                if col_indices.len() != col_names.len() {
                    return Ok(());
                }
                let coerced: Vec<Value> = col_indices
                    .iter()
                    .filter_map(|&idx| {
                        let val = new_row.get(idx)?;
                        let target_type = schema.columns[idx].data_type;
                        Some(val.coerce_to_type(target_type))
                    })
                    .collect();
                if coerced.len() != col_indices.len() || coerced.iter().any(|v| v.is_null()) {
                    return Ok(());
                }
                let values: Vec<&Value> = coerced.iter().collect();

                if let Some(found_id) = self.find_segment_row_id_by_values(&col_indices, &values) {
                    if found_id != exclude_row_id {
                        return Err(crate::core::Error::UniqueConstraint {
                            index: idx_name.to_string(),
                            column: col_names.join(", "),
                            value: values
                                .iter()
                                .map(|v| format!("{}", v))
                                .collect::<Vec<_>>()
                                .join(", "),
                            row_id: found_id,
                        });
                    }
                }
                Ok(())
            })
    }

    fn check_segment_constraints(&self, row: &Row) -> Result<()> {
        let snapshot = self.segment_mgr.cold_snapshot();
        self.check_segment_constraints_with_snapshot(&snapshot, row)
    }

    fn check_segment_constraints_with_snapshot(
        &self,
        snapshot: &super::manifest::ColdSnapshot,
        row: &Row,
    ) -> Result<()> {
        let schema = self.hot.schema();

        // 1. PK constraint: binary search on sorted INT column
        if let Some(pk_idx) = schema.pk_column_index() {
            if let Some(pk_value) = row.get(pk_idx) {
                if !pk_value.is_null() {
                    if let Some(row_id) = self
                        .segment_mgr
                        .check_value_exists_with_snapshot(snapshot, pk_idx, pk_value)
                    {
                        // check_value_exists_in_segments filters committed tombstones.
                        // Check pending tombstones (no Vec clone).
                        if self.segment_mgr.is_pending_tombstone(self.txn_id(), row_id) {
                            // This txn deleted this cold row — not a real conflict
                            return Ok(());
                        }
                        return Err(crate::core::Error::PrimaryKeyConstraint { row_id });
                    }
                }
            }
        }

        // 2. UNIQUE constraints: scan cold segments with pruning
        if !self.hot.has_unique_non_pk_indexes() {
            return Ok(());
        }
        self.hot
            .for_each_unique_non_pk_index(&mut |idx_name, col_names| {
                let col_indices: Vec<usize> = col_names
                    .iter()
                    .filter_map(|name| schema.columns.iter().position(|c| c.name_lower == *name))
                    .collect();
                if col_indices.len() != col_names.len() {
                    return Ok(());
                }
                // Coerce values to schema column types before comparing with cold segments.
                // Prepared statements may pass TEXT for TIMESTAMP columns — the row is
                // coerced later in prepare_insert, but we need the correct types NOW for
                // zone map / bloom / dict pruning and value comparison to work.
                let coerced: Vec<Value> = col_indices
                    .iter()
                    .filter_map(|&idx| {
                        let val = row.get(idx)?;
                        let target_type = schema.columns[idx].data_type;
                        Some(val.coerce_to_type(target_type))
                    })
                    .collect();
                if coerced.len() != col_indices.len() || coerced.iter().any(|v| v.is_null()) {
                    return Ok(());
                }
                let values: Vec<&Value> = coerced.iter().collect();

                if let Some(conflict_row_id) = self.find_segment_row_id_by_values_with_snapshot(
                    snapshot,
                    &col_indices,
                    &values,
                ) {
                    return Err(crate::core::Error::UniqueConstraint {
                        index: idx_name.to_string(),
                        column: col_names.join(", "),
                        value: values
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                        row_id: conflict_row_id,
                    });
                }
                Ok(())
            })
    }

    /// Pre-compute bloom filter hashes for equality comparisons.
    /// Call once before the volume loop to avoid redundant hashing per volume.
    fn precompute_bloom_hashes(
        comparisons: &[(&str, crate::core::Operator, &Value)],
    ) -> Vec<Option<u64>> {
        comparisons
            .iter()
            .map(|&(_, op, value)| {
                if op == crate::core::Operator::Eq {
                    Some(super::column::ColumnBloomFilter::hash_value_static(value))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Zone map pruning + binary search narrowing on a single volume.
    /// Returns (should_skip, start, end).
    /// `bloom_hashes` are pre-computed per-comparison to avoid redundant hashing.
    fn prune_volume(
        vol: &FrozenVolume,
        comparisons: &[(&str, crate::core::Operator, &Value)],
        bloom_hashes: &[Option<u64>],
    ) -> (bool, usize, usize) {
        let mut start = 0usize;
        let mut end = vol.row_count;

        for (comp_idx, &(col_name, op, value)) in comparisons.iter().enumerate() {
            if let Some(col_idx) = vol.column_index(col_name) {
                let zm = &vol.zone_maps[col_idx];
                let dominated =
                    match op {
                        crate::core::Operator::Gt | crate::core::Operator::Gte => {
                            !zm.may_contain_gte(value)
                        }
                        crate::core::Operator::Lt | crate::core::Operator::Lte => {
                            !zm.may_contain_lte(value)
                        }
                        crate::core::Operator::Eq => {
                            !zm.may_contain_eq(value)
                                || (col_idx < vol.bloom_filters.len()
                                    && bloom_hashes.get(comp_idx).and_then(|h| *h).is_some_and(
                                        |h| !vol.bloom_filters[col_idx].might_contain_hash(h),
                                    ))
                        }
                        _ => false,
                    };
                if dominated {
                    return (true, 0, 0);
                }

                // Binary search on sorted columns.
                // V4 path: uses row-group zone maps to decompress only the target group.
                if vol.is_sorted(col_idx) {
                    let target = match value {
                        Value::Integer(i) => Some(*i),
                        Value::Timestamp(ts) => Some(
                            ts.timestamp_nanos_opt()
                                .unwrap_or(ts.timestamp() * 1_000_000_000),
                        ),
                        _ => None,
                    };
                    if let Some(target) = target {
                        // Use per-group binary search only when columns are
                        // deferred (not yet loaded). When eager,
                        // full columns are in OnceLock — use them directly.
                        let store = if vol.columns.should_use_group_cache() {
                            vol.columns.compressed_store()
                        } else {
                            None
                        };
                        match op {
                            crate::core::Operator::Gte => {
                                let idx = if let Some(st) = store {
                                    st.binary_search_ge(col_idx, target, &vol.row_groups)
                                } else {
                                    vol.columns[col_idx].binary_search_ge(target)
                                };
                                if idx > start {
                                    start = idx;
                                }
                            }
                            crate::core::Operator::Gt => {
                                let idx = if let Some(st) = store {
                                    st.binary_search_gt(col_idx, target, &vol.row_groups)
                                } else {
                                    vol.columns[col_idx].binary_search_gt(target)
                                };
                                if idx > start {
                                    start = idx;
                                }
                            }
                            crate::core::Operator::Lte => {
                                let idx = if let Some(st) = store {
                                    st.binary_search_gt(col_idx, target, &vol.row_groups)
                                } else {
                                    vol.columns[col_idx].binary_search_gt(target)
                                };
                                if idx < end {
                                    end = idx;
                                }
                            }
                            crate::core::Operator::Lt => {
                                let idx = if let Some(st) = store {
                                    st.binary_search_ge(col_idx, target, &vol.row_groups)
                                } else {
                                    vol.columns[col_idx].binary_search_ge(target)
                                };
                                if idx < end {
                                    end = idx;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        (start >= end, start, end)
    }

    /// Create lazy segment scanners for a read query, applying zone map pruning
    /// and binary search on sorted columns for range predicates.
    ///
    /// Uses the per-row visibility bitmap on each ColdSegment for inter-volume
    /// dedup. The caller-provided `hot_skip` (hot row_ids and pending tombstones)
    /// is passed to every scanner unchanged — no per-volume accumulation needed.
    /// Committed tombstones are shared via Arc (no clone per volume).
    ///
    /// The `hot_skip` MUST be derived from actual hot scan results (not a
    /// separate B-tree read) to prevent the seal race.
    fn create_segment_scanners_filtered(
        &self,
        column_indices: &[usize],
        where_expr: Option<&dyn Expression>,
        hot_skip: FxHashSet<i64>,
    ) -> Vec<Box<dyn Scanner>> {
        let comparisons = where_expr
            .map(|e| e.collect_comparisons())
            .unwrap_or_default();

        let volumes = self.segment_mgr.get_volumes_newest_first();

        if volumes.is_empty() {
            return Vec::new();
        }

        // Committed tombstones are kept as a shared Arc (no clone).
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();

        // hot_skip is shared across all scanners via Arc — no clone per volume.
        let hot_skip_arc = Arc::new(hot_skip);

        let bloom_hashes = Self::precompute_bloom_hashes(&comparisons);
        let mut scanners_reverse: Vec<Box<dyn Scanner>> = Vec::with_capacity(volumes.len());

        for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            let (should_skip, start, end) = Self::prune_volume(vol, &comparisons, &bloom_hashes);
            if should_skip {
                continue;
            }

            let mut scanner = if start > 0 || end < vol.row_count {
                VolumeScanner::with_range(
                    Arc::clone(vol),
                    column_indices.to_vec(),
                    start,
                    end,
                    None,
                )
            } else {
                VolumeScanner::new(Arc::clone(vol), column_indices.to_vec(), None)
            };
            // Each scanner gets the same small hot_skip Arc (no clone of hot IDs).
            // Inter-volume dedup is handled by the per-volume visibility bitmap.
            scanner.set_skip_sets(Arc::clone(&tombstones_arc), Arc::clone(&hot_skip_arc));
            scanner.set_visibility_bitmap(cs.visible.clone());
            scanner.snapshot_seq = self.snapshot_seq;
            let current_schema = self.hot.schema();
            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, current_schema);
            scanner.set_column_mapping(mapping);

            if let Some(expr) = where_expr {
                let filter = expr.with_aliases(&Default::default());
                let mut prepared = filter;
                prepared.prepare_for_schema(current_schema);
                scanner.set_filter(prepared);
            }
            scanners_reverse.push(Box::new(scanner) as Box<dyn Scanner>);
        }

        // Reverse so oldest segments come first (consistent iteration order)
        scanners_reverse.reverse();
        scanners_reverse
    }

    /// Collect rows from segments into a RowVec, with zone map pruning
    /// and binary search on sorted columns.
    ///
    /// Uses the per-row visibility bitmap on each ColdSegment for inter-volume
    /// dedup. The caller-provided `hot_skip` (hot row_ids and pending tombstones)
    /// filters rows that are shadowed by the hot buffer.
    ///
    /// The `hot_skip` MUST be derived from actual hot scan results (not a separate
    /// B-tree read) to avoid races with concurrent `remove_sealed_rows`.
    fn collect_cold_rows(
        &self,
        where_expr: Option<&dyn Expression>,
        hot_skip: FxHashSet<i64>,
    ) -> Result<RowVec> {
        let comparisons = where_expr
            .map(|e| e.collect_comparisons())
            .unwrap_or_default();

        let volumes = self.segment_mgr.get_volumes_newest_first();

        let tombstones_arc = self.segment_mgr.tombstone_set_arc();

        let total: usize = volumes.iter().map(|(_, cs)| cs.volume.row_count).sum();
        let mut rows = RowVec::with_capacity(total.min(64_000));

        // Process newest-first; visibility bitmap handles inter-volume dedup.
        let bloom_hashes = Self::precompute_bloom_hashes(&comparisons);
        let mut per_volume_rows: Vec<RowVec> = Vec::with_capacity(volumes.len());
        for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            let (should_skip, start, end) = Self::prune_volume(vol, &comparisons, &bloom_hashes);
            if should_skip {
                continue;
            }

            let mut start = start;

            let mut dict_filters: Vec<(usize, u32)> = Vec::new();
            let store = vol.columns.compressed_store();
            for &(col_name, op, value) in &comparisons {
                if op != crate::core::Operator::Eq {
                    continue;
                }
                if let Some(col_idx) = vol.column_index(col_name) {
                    if let Value::Text(s) = value {
                        // Use CompressedBlockStore dict_lookup when available
                        // (reads shared_dict directly, no column decompression).
                        let dict_id = if let Some(st) = store {
                            st.dict_lookup(col_idx, s.as_str())
                        } else {
                            vol.columns[col_idx].dict_lookup(s.as_str())
                        };
                        if let Some(id) = dict_id {
                            dict_filters.push((col_idx, id));
                        } else {
                            start = end;
                            break;
                        }
                    }
                }
            }

            if start >= end {
                continue;
            }

            let current_schema = self.hot.schema();
            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, current_schema);

            let mut vol_rows = RowVec::new();
            for i in start..end {
                if !cs.is_visible(i) {
                    continue;
                }
                let row_id = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, row_id) || hot_skip.contains(&row_id) {
                    continue;
                }

                if !dict_filters.is_empty() {
                    let mut matches = true;
                    for &(col_idx, expected_id) in &dict_filters {
                        if vol.columns[col_idx].is_null(i)
                            || vol.columns[col_idx].get_dict_id(i) != expected_id
                        {
                            matches = false;
                            break;
                        }
                    }
                    if !matches {
                        continue;
                    }
                }

                let row = if mapping.is_identity {
                    vol.get_row(i)
                } else {
                    vol.get_row_mapped(i, &mapping)
                };
                if let Some(expr) = where_expr {
                    if !expr.evaluate_fast(&row) {
                        continue;
                    }
                }
                vol_rows.push((row_id, row));
            }
            per_volume_rows.push(vol_rows);
        }

        for vol_rows in per_volume_rows.into_iter().rev() {
            for entry in vol_rows {
                rows.push(entry);
            }
        }
        Ok(rows)
    }

    /// Find a row in segments by row_id. Returns (volume, local_offset) if found
    /// and not tombstoned or hot-shadowed. Uses manifest min/max for fast segment
    /// identification, then binary search within the segment.
    fn find_segment_row(&self, row_id: i64) -> Option<(u64, Arc<FrozenVolume>, usize)> {
        // Hot buffer shadows cold: if the row exists in hot, the cold copy is stale
        if self.hot.has_row_id(row_id) {
            return None;
        }
        // Check committed tombstones (snapshot-aware: newer tombstones are invisible)
        {
            let ts = self.segment_mgr.tombstone_set_arc();
            if self.is_row_tombstoned(&ts, row_id) {
                return None;
            }
        }
        // Check pending tombstones for this transaction (no Vec clone)
        if self.segment_mgr.is_pending_tombstone(self.txn_id(), row_id) {
            return None;
        }

        // CoW snapshot: segment IDs + Arc-clone segments map. Lock-free scan.
        let seg_ids: Vec<u64> = {
            let manifest = self.segment_mgr.manifest();
            manifest
                .segments
                .iter()
                .rev()
                .map(|m| m.segment_id)
                .collect()
        };
        let segs = self.segment_mgr.segments_snapshot();
        for &seg_id in &seg_ids {
            let Some(cold) = segs.get(&seg_id) else {
                continue;
            };
            let vol = &cold.volume;
            if vol.row_ids.is_empty() {
                continue;
            }
            let min_id = vol.row_ids[0];
            let max_id = vol.row_ids[vol.row_count - 1];
            if row_id < min_id || row_id > max_id {
                continue;
            }
            if let Ok(idx) = vol.row_ids.binary_search(&row_id) {
                return Some((seg_id, Arc::clone(vol), idx));
            }
        }
        None
    }
}

impl Table for SegmentedTable {
    // =========================================================================
    // Metadata
    // =========================================================================

    fn name(&self) -> &str {
        self.hot.name()
    }

    fn schema(&self) -> &Schema {
        self.hot.schema()
    }

    fn txn_id(&self) -> i64 {
        self.hot.txn_id()
    }

    // =========================================================================
    // DDL — delegate to hot buffer
    // =========================================================================

    fn create_column(&mut self, name: &str, column_type: DataType, nullable: bool) -> Result<()> {
        self.hot.create_column(name, column_type, nullable)
    }

    fn create_column_with_default(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        default_expr: Option<String>,
    ) -> Result<()> {
        self.hot
            .create_column_with_default(name, column_type, nullable, default_expr)
    }

    fn create_column_with_default_value(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        default_expr: Option<String>,
        default_value: Option<Value>,
    ) -> Result<()> {
        self.hot.create_column_with_default_value(
            name,
            column_type,
            nullable,
            default_expr,
            default_value,
        )
    }

    fn drop_column(&mut self, name: &str) -> Result<()> {
        self.hot.drop_column(name)
    }

    // =========================================================================
    // DML — writes go to hot buffer, constraints checked against segments
    // =========================================================================

    fn insert(&mut self, row: Row) -> Result<Row> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        if self.segment_mgr.has_segments() {
            self.check_segment_constraints(&row)?;
        }
        let result = self.hot.insert(row)?;
        if self.segment_mgr.has_segments() {
            self.segment_mgr.record_txn_seal_generation(self.txn_id());
        }
        Ok(result)
    }

    fn insert_discard(&mut self, row: Row) -> Result<()> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        if self.segment_mgr.has_segments() {
            self.check_segment_constraints(&row)?;
        }
        self.hot.insert_discard(row)?;
        if self.segment_mgr.has_segments() {
            self.segment_mgr.record_txn_seal_generation(self.txn_id());
        }
        Ok(())
    }

    fn insert_batch(&mut self, rows: Vec<Row>) -> Result<()> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        if self.segment_mgr.has_segments() {
            // Snapshot once for the entire batch — eliminates 3 lock reads per row.
            let snapshot = self.segment_mgr.cold_snapshot();
            for row in &rows {
                self.check_segment_constraints_with_snapshot(&snapshot, row)?;
            }
        }
        self.hot.insert_batch(rows)?;
        if self.segment_mgr.has_segments() {
            self.segment_mgr.record_txn_seal_generation(self.txn_id());
        }
        Ok(())
    }

    fn update(
        &mut self,
        where_expr: Option<&dyn Expression>,
        setter: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
    ) -> Result<i32> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        let mut count = self.hot.update(where_expr, setter)?;

        // Update matching segment rows.
        // Build skip set from hot row_ids + tombstones, then iterate volumes newest-first.
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let has_int_pk = self
            .hot
            .schema()
            .columns
            .iter()
            .any(|c| c.primary_key && c.data_type == DataType::Integer);

        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);
        let schema_clone = self.hot.schema().clone();

        for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, &schema_clone);

            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let row_id = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, row_id) || hot_skip.contains(&row_id) {
                    continue;
                }
                let row = if mapping.is_identity {
                    vol.get_row(i)
                } else {
                    vol.get_row_mapped(i, &mapping)
                };
                if let Some(expr) = where_expr {
                    if !expr.evaluate_fast(&row) {
                        continue;
                    }
                }
                let old_row = row.clone();
                let (new_row, changed) = setter(row)?;
                if changed {
                    // Claim the cold row to prevent concurrent lost updates.
                    self.hot.try_claim_row(row_id)?;

                    // Check unique constraints against cold segments.
                    if self.hot.has_unique_non_pk_indexes() {
                        self.check_cold_unique_for_update(&new_row, row_id)?;
                    }

                    // Insert the NEW row into hot. For int PK tables, first
                    // mirror the old row (so UPDATE can find it), then update.
                    // If any step fails, clean up to avoid phantoms.
                    if has_int_pk {
                        match self.hot.insert_discard(old_row) {
                            Ok(())
                            | Err(crate::core::Error::PrimaryKeyConstraint { .. })
                            | Err(crate::core::Error::UniqueConstraint { .. }) => {}
                            Err(e) => {
                                return Err(e);
                            }
                        }
                        let mut new_row_opt = Some(new_row);
                        let update_result = self.hot.update_by_row_ids(&[row_id], &mut |_| {
                            Ok((new_row_opt.take().unwrap_or_else(Row::new), true))
                        });
                        if let Err(e) = update_result {
                            let _ = self.hot.delete_by_row_ids(&[row_id]);
                            return Err(e);
                        }
                    } else {
                        self.hot.insert_discard(new_row)?;
                    }
                    // Add tombstone so row_count() doesn't double-count.
                    // The hot version now shadows the cold version via skip set.
                    self.segment_mgr
                        .add_pending_tombstone(self.txn_id(), row_id);
                    count += 1;
                }
            }
        }
        if count > 0 {
            self.segment_mgr.record_txn_seal_generation(self.txn_id());
        }
        Ok(count)
    }

    fn update_by_row_ids(
        &mut self,
        row_ids: &[i64],
        setter: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
    ) -> Result<i32> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        let mut count = 0i32;
        let mut hot_ids = Vec::new();
        let schema = self.hot.schema().clone();
        let has_int_pk = schema
            .columns
            .iter()
            .any(|c| c.primary_key && c.data_type == DataType::Integer);

        let mut cached_mapping: Option<(
            *const super::writer::FrozenVolume,
            super::writer::ColumnMapping,
        )> = None;
        for &row_id in row_ids {
            if let Some((seg_id, vol, idx)) = self.find_segment_row(row_id) {
                let vol_ptr = &*vol as *const super::writer::FrozenVolume;
                let mapping = match &cached_mapping {
                    Some((ptr, m)) if *ptr == vol_ptr => m,
                    _ => {
                        cached_mapping = Some((
                            vol_ptr,
                            self.segment_mgr.get_volume_mapping(seg_id, &schema),
                        ));
                        &cached_mapping.as_ref().unwrap().1
                    }
                };
                let row = if mapping.is_identity {
                    vol.get_row(idx)
                } else {
                    vol.get_row_mapped(idx, mapping)
                };
                let old_row = row.clone();
                let (new_row, changed) = setter(row)?;
                if changed {
                    // Claim the cold row to prevent concurrent lost updates.
                    self.hot.try_claim_row(row_id)?;
                    if self.hot.has_unique_non_pk_indexes() {
                        self.check_cold_unique_for_update(&new_row, row_id)?;
                    }
                    let result = if has_int_pk {
                        let insert_ok = match self.hot.insert_discard(old_row) {
                            Ok(()) => true,
                            Err(crate::core::Error::PrimaryKeyConstraint { .. }) => true,
                            Err(crate::core::Error::UniqueConstraint { .. }) => true,
                            Err(e) => return Err(e),
                        };
                        if insert_ok {
                            let mut new_row_opt = Some(new_row);
                            self.hot
                                .update_by_row_ids(&[row_id], &mut |_| {
                                    Ok((new_row_opt.take().unwrap_or_else(Row::new), true))
                                })
                                .map(|_| ())
                        } else {
                            Ok(())
                        }
                    } else {
                        self.hot.insert_discard(new_row)
                    };
                    result?;
                    // Add tombstone so row_count() doesn't double-count.
                    // The hot version now shadows the cold version via skip set.
                    self.segment_mgr
                        .add_pending_tombstone(self.txn_id(), row_id);
                    count += 1;
                }
            } else {
                hot_ids.push(row_id);
            }
        }
        if !hot_ids.is_empty() {
            // Same reasoning as update(): skip cold unique check for hot path
            // to avoid false violations during ON CONFLICT DO UPDATE.
            count += self.hot.update_by_row_ids(&hot_ids, setter)?;
        }
        if count > 0 {
            self.segment_mgr.record_txn_seal_generation(self.txn_id());
        }
        Ok(count)
    }

    fn delete_by_row_ids(&mut self, row_ids: &[i64]) -> Result<i32> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        let mut count = 0i32;
        let mut hot_ids = Vec::new();
        let has_int_pk = self
            .hot
            .schema()
            .columns
            .iter()
            .any(|c| c.primary_key && c.data_type == DataType::Integer);

        for &row_id in row_ids {
            if let Some((_seg_id, _vol, _idx)) = self.find_segment_row(row_id) {
                // Claim the cold row to prevent concurrent lost deletes.
                self.hot.try_claim_row(row_id)?;
                if has_int_pk {
                    let _ = self.hot.delete_by_row_ids(&[row_id]);
                }
                // Track tombstone for commit. Pending tombstones are applied on commit
                // and discarded on rollback to prevent isolation violations.
                self.segment_mgr
                    .add_pending_tombstone(self.txn_id(), row_id);
                count += 1;
            } else {
                hot_ids.push(row_id);
            }
        }
        if !hot_ids.is_empty() {
            count += self.hot.delete_by_row_ids(&hot_ids)?;
        }
        Ok(count)
    }

    fn get_active_row_ids(&self) -> Vec<i64> {
        let hot_ids = self.hot.get_active_row_ids();

        if !self.segment_mgr.has_segments() {
            return hot_ids;
        }

        // Build hot_skip from hot row_ids + pending tombstones.
        // Committed tombstones are kept as a shared Arc (no clone).
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        for id in &hot_ids {
            hot_skip.insert(*id);
        }
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        let mut ids = Vec::new();
        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let id = vol.row_ids[i];
                if !self.is_row_tombstoned(&tombstones_arc, id) && !hot_skip.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids.extend(hot_ids);
        ids
    }

    fn delete(&mut self, where_expr: Option<&dyn Expression>) -> Result<i32> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        let mut count = self.hot.delete(where_expr)?;
        let has_int_pk = self
            .hot
            .schema()
            .columns
            .iter()
            .any(|c| c.primary_key && c.data_type == DataType::Integer);

        // Build hot_skip from hot row_ids + pending tombstones.
        // Committed tombstones are kept as a shared Arc (no clone).
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);
        let schema_clone = self.hot.schema().clone();

        // Pre-compute a column-level bitmask from the WHERE expression so that
        // we only decompress/allocate the columns the filter actually references.
        // When `where_expr` is None every row matches, so we skip materialisation
        // entirely.  When `collect_column_indices` returns false (expression type
        // is unknown) we fall back to a full-row materialisation.
        let needed_cols: Option<Vec<bool>> = where_expr.and_then(|expr| {
            let mut cols = Vec::new();
            if expr.collect_column_indices(&mut cols) {
                // Size the mask to the schema width; individual volumes may be
                // wider/narrower after schema evolution — the per-volume mask is
                // clamped inside get_row_needed / get_row_mapped_needed.
                let mask_len = schema_clone.columns.len();
                let mut mask = vec![false; mask_len];
                for &ci in &cols {
                    if ci < mask_len {
                        mask[ci] = true;
                    }
                }
                Some(mask)
            } else {
                None // unknown expression — materialise all columns
            }
        });

        let mut deleted_cold_ids: Vec<i64> = Vec::new();
        // Reusable row for WHERE evaluation — avoids Vec/Arc allocation per row.
        // The CompactVec capacity is set once, then clear+push reuses the buffer.
        let mut reusable_row = Row::with_capacity(schema_clone.columns.len());
        for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, &schema_clone);

            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let row_id = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, row_id) || hot_skip.contains(&row_id) {
                    continue;
                }
                if let Some(expr) = where_expr {
                    // Fill reusable row with only the needed columns.
                    // Zero heap allocation after first iteration (buffer reuse).
                    reusable_row.clear();
                    match (&needed_cols, mapping.is_identity) {
                        (Some(mask), true) => {
                            for ci in 0..vol.columns.len() {
                                if ci < mask.len() && mask[ci] {
                                    reusable_row.push(vol.columns[ci].get_value(i));
                                } else {
                                    reusable_row.push(Value::Null(vol.columns.data_type(ci)));
                                }
                            }
                        }
                        (Some(mask), false) => {
                            for (ci, src) in mapping.sources.iter().enumerate() {
                                if ci < mask.len() && mask[ci] {
                                    match src {
                                        super::writer::ColSource::Volume(vi) => {
                                            reusable_row.push(vol.columns[*vi].get_value(i));
                                        }
                                        super::writer::ColSource::Default(val) => {
                                            reusable_row.push(val.clone());
                                        }
                                    }
                                } else {
                                    match src {
                                        super::writer::ColSource::Volume(vi) => {
                                            reusable_row
                                                .push(Value::Null(vol.columns.data_type(*vi)));
                                        }
                                        super::writer::ColSource::Default(val) => {
                                            reusable_row.push(Value::Null(val.data_type()));
                                        }
                                    }
                                }
                            }
                        }
                        (None, true) => {
                            for ci in 0..vol.columns.len() {
                                reusable_row.push(vol.columns[ci].get_value(i));
                            }
                        }
                        (None, false) => {
                            for src in &mapping.sources {
                                match src {
                                    super::writer::ColSource::Volume(vi) => {
                                        reusable_row.push(vol.columns[*vi].get_value(i));
                                    }
                                    super::writer::ColSource::Default(val) => {
                                        reusable_row.push(val.clone());
                                    }
                                }
                            }
                        }
                    };
                    if !expr.evaluate_fast(&reusable_row) {
                        continue;
                    }
                }
                // Claim the cold row to prevent concurrent lost deletes.
                self.hot.try_claim_row(row_id)?;
                if has_int_pk {
                    let _ = self.hot.delete_by_row_ids(&[row_id]);
                }
                deleted_cold_ids.push(row_id);
                count += 1;
            }
        }
        // Track tombstones for commit
        for rid in deleted_cold_ids {
            self.segment_mgr.add_pending_tombstone(self.txn_id(), rid);
        }
        Ok(count)
    }

    fn truncate(&mut self) -> Result<i32> {
        let _seal_guard = self.segment_mgr.acquire_seal_read();
        let seg_rows = self.segment_mgr.total_row_count() as i32;
        // Clear pending tombstones for this txn (segments are being dropped)
        self.segment_mgr.rollback_pending_tombstones(self.txn_id());
        self.segment_mgr.clear();
        Ok(self.hot.truncate()? + seg_rows)
    }

    // =========================================================================
    // Read operations — merge segments + hot buffer
    // =========================================================================

    fn scan(
        &self,
        column_indices: &[usize],
        where_expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn Scanner>> {
        if !self.segment_mgr.has_segments() {
            return self.hot.scan(column_indices, where_expr);
        }

        // Collect hot rows FIRST to get a consistent snapshot of hot row_ids.
        // The skip set for cold scanners is derived from these actual results,
        // preventing the race where remove_sealed_rows runs between building
        // the skip set and hot scanner execution (which would lose rows).
        let hot_rows = self.hot.collect_all_rows(where_expr)?;

        let mut skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(hot_rows.len(), Default::default());
        for &(id, _) in &hot_rows {
            skip.insert(id);
        }
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut skip);

        // Create lazy cold scanners with the skip set (no eager collection).
        // This avoids O(total_cold_rows) memory allocation that was making
        // ALL queries slow during checkpoint.
        let cold_scanners = self.create_segment_scanners_filtered(column_indices, where_expr, skip);

        // Chain: cold scanners (lazy, streamed) + hot rows (already collected)
        let hot_scanner = Box::new(RowVecScanner::new(hot_rows)) as Box<dyn Scanner>;
        let mut sources: Vec<Box<dyn Scanner>> = cold_scanners;
        sources.push(hot_scanner);

        Ok(Box::new(super::scanner::MergingScanner::new(sources)))
    }

    fn collect_all_rows(&self, where_expr: Option<&dyn Expression>) -> Result<RowVec> {
        if !self.segment_mgr.has_segments() {
            return self.hot.collect_all_rows(where_expr);
        }

        // Scan hot FIRST to get a consistent snapshot. The skip set is
        // derived from actual hot results, not a separate B-tree read.
        // This prevents the race where remove_sealed_rows runs between
        // building the skip set and scanning hot.
        let hot_rows = self.hot.collect_all_rows(where_expr)?;

        let mut skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(hot_rows.len(), Default::default());
        for &(id, _) in &hot_rows {
            skip.insert(id);
        }
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut skip);

        let mut all_rows = self.collect_cold_rows(where_expr, skip)?;
        for entry in hot_rows {
            all_rows.push(entry);
        }
        Ok(all_rows)
    }

    fn collect_all_rows_unsorted(&self) -> Result<RowVec> {
        if !self.segment_mgr.has_segments() {
            return self.hot.collect_all_rows_unsorted();
        }

        let hot_rows = self.hot.collect_all_rows_unsorted()?;

        let mut skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(hot_rows.len(), Default::default());
        for &(id, _) in &hot_rows {
            skip.insert(id);
        }
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut skip);

        let mut all_rows = self.collect_cold_rows(None, skip)?;
        for entry in hot_rows {
            all_rows.push(entry);
        }
        Ok(all_rows)
    }

    fn collect_rows_by_ids(&self, row_ids: &[i64]) -> Result<RowVec> {
        if !self.segment_mgr.has_segments() {
            return self.hot.collect_rows_by_ids(row_ids);
        }

        let schema = self.hot.schema().clone();
        let mut result = RowVec::with_capacity(row_ids.len());
        let mut hot_ids = Vec::new();
        let mut cached_mapping: Option<(
            *const super::writer::FrozenVolume,
            super::writer::ColumnMapping,
        )> = None;

        for &row_id in row_ids {
            if let Some((seg_id, vol, idx)) = self.find_segment_row(row_id) {
                let vol_ptr = &*vol as *const super::writer::FrozenVolume;
                let mapping = match &cached_mapping {
                    Some((ptr, m)) if *ptr == vol_ptr => m,
                    _ => {
                        cached_mapping = Some((
                            vol_ptr,
                            self.segment_mgr.get_volume_mapping(seg_id, &schema),
                        ));
                        &cached_mapping.as_ref().unwrap().1
                    }
                };
                let row = if mapping.is_identity {
                    vol.get_row(idx)
                } else {
                    vol.get_row_mapped(idx, mapping)
                };
                result.push((row_id, row));
            } else {
                hot_ids.push(row_id);
            }
        }

        if !hot_ids.is_empty() {
            let hot_result = self.hot.collect_rows_by_ids(&hot_ids)?;
            for (id, row) in hot_result {
                result.push((id, row));
            }
        }

        Ok(result)
    }

    fn fetch_rows_by_ids(&self, row_ids: &[i64], filter: &dyn Expression) -> RowVec {
        let mut results = RowVec::with_capacity(row_ids.len());
        self.fetch_rows_by_ids_into(row_ids, filter, &mut results);
        results
    }

    fn fetch_rows_by_ids_into(
        &self,
        row_ids: &[i64],
        filter: &dyn Expression,
        buffer: &mut RowVec,
    ) {
        if !self.segment_mgr.has_segments() {
            self.hot.fetch_rows_by_ids_into(row_ids, filter, buffer);
            return;
        }

        let mut hot_ids = Vec::new();
        let schema = self.hot.schema();
        let mut cached_mapping: Option<(
            *const super::writer::FrozenVolume,
            super::writer::ColumnMapping,
        )> = None;

        for &row_id in row_ids {
            if let Some((seg_id, vol, idx)) = self.find_segment_row(row_id) {
                let vol_ptr = &*vol as *const super::writer::FrozenVolume;
                let mapping = match &cached_mapping {
                    Some((ptr, m)) if *ptr == vol_ptr => m,
                    _ => {
                        cached_mapping =
                            Some((vol_ptr, self.segment_mgr.get_volume_mapping(seg_id, schema)));
                        &cached_mapping.as_ref().unwrap().1
                    }
                };
                let row = if mapping.is_identity {
                    vol.get_row(idx)
                } else {
                    vol.get_row_mapped(idx, mapping)
                };
                if filter.evaluate_fast(&row) {
                    buffer.push((row_id, row));
                }
            } else {
                hot_ids.push(row_id);
            }
        }

        if !hot_ids.is_empty() {
            self.hot.fetch_rows_by_ids_into(&hot_ids, filter, buffer);
        }
    }

    // =========================================================================
    // LIMIT pushdowns
    // =========================================================================

    fn collect_rows_with_limit(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<RowVec> {
        if !self.segment_mgr.has_segments() {
            return self.hot.collect_rows_with_limit(where_expr, limit, offset);
        }

        let target = limit + offset;

        let volumes = self.segment_mgr.get_volumes_newest_first();
        if volumes.is_empty() {
            return self.hot.collect_rows_with_limit(where_expr, limit, offset);
        }

        // Phase 1: Build authority map (row_id → volume index).
        // Iterates cold row_ids newest-first so newer volumes win dedup.
        // Uses has_row_id() B-tree point lookup for hot check instead
        // of collecting all hot row_ids. Cost: O(cold_ids) integer ops.
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut pending_tomb: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(1_000, Default::default());
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut pending_tomb);

        let total_cold_rows: usize = volumes.iter().map(|(_, cs)| cs.volume.row_count).sum();
        let mut authority: FxHashMap<i64, usize> = FxHashMap::with_capacity_and_hasher(
            total_cold_rows.min(500_000) * 8 / 7 + 16,
            Default::default(),
        );
        for (nf_idx, (_seg_id, cs)) in volumes.iter().enumerate() {
            for &rid in &cs.volume.row_ids {
                if self.hot.has_row_id(rid)
                    || self.is_row_tombstoned(&tombstones_arc, rid)
                    || pending_tomb.contains(&rid)
                {
                    continue;
                }
                authority.entry(rid).or_insert(nf_idx);
            }
        }

        // Phase 2: Iterate oldest-first, materialize only authoritative
        // rows, stop as soon as we have `target` matches.
        let mut cold_rows = RowVec::with_capacity(target.min(1024));
        let current_schema = self.hot.schema();

        let comparisons = where_expr
            .map(|e| e.collect_comparisons())
            .unwrap_or_default();
        let bloom_hashes = Self::precompute_bloom_hashes(&comparisons);

        'done: for (nf_idx, (seg_id, cs)) in volumes.iter().enumerate().rev() {
            let vol = &cs.volume;
            if !comparisons.is_empty() {
                let (skip, _, _) = Self::prune_volume(vol, &comparisons, &bloom_hashes);
                if skip {
                    continue;
                }
            }

            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, current_schema);

            for i in 0..vol.row_count {
                let rid = vol.row_ids[i];
                if authority.get(&rid) != Some(&nf_idx) {
                    continue;
                }
                let row = if mapping.is_identity {
                    vol.get_row(i)
                } else {
                    vol.get_row_mapped(i, &mapping)
                };
                if let Some(expr) = where_expr {
                    if !expr.evaluate_fast(&row) {
                        continue;
                    }
                }
                cold_rows.push((rid, row));
                if cold_rows.len() >= target {
                    break 'done;
                }
            }
        }

        // Phase 3: If cold didn't fill the target, materialize hot rows
        // for the remainder only.
        if cold_rows.len() < target {
            let remaining = target - cold_rows.len();
            let hot_rows = self.hot.collect_rows_with_limit(where_expr, remaining, 0)?;
            cold_rows.extend(hot_rows);
        }

        Ok(cold_rows.into_iter().skip(offset).take(limit).collect())
    }

    fn collect_rows_with_limit_unordered(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<RowVec> {
        if !self.segment_mgr.has_segments() {
            return self
                .hot
                .collect_rows_with_limit_unordered(where_expr, limit, offset);
        }

        let target = limit + offset;

        // Unordered: hot rows first with early termination.
        // Only scan up to `target` hot rows — avoids O(hot_rows) for small LIMITs.
        let hot_rows = self
            .hot
            .collect_rows_with_limit_unordered(where_expr, target, 0)?;
        if hot_rows.len() >= target {
            return Ok(hot_rows.into_iter().skip(offset).take(limit).collect());
        }

        // Need cold rows. Build hot_skip and scan with early termination.
        let mut result = hot_rows;
        let remaining = target - result.len();

        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        let volumes = self.segment_mgr.get_volumes_newest_first();
        let current_schema = self.hot.schema();
        let mut collected = 0usize;

        // Zone-map + bloom filter setup for pruning.
        let comparisons = where_expr
            .map(|e| e.collect_comparisons())
            .unwrap_or_default();
        let bloom_hashes = Self::precompute_bloom_hashes(&comparisons);

        'outer: for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            // Zone-map pruning: skip entire volume if no rows can match.
            let pruned = if !comparisons.is_empty() {
                let (skip, _, _) = Self::prune_volume(vol, &comparisons, &bloom_hashes);
                skip
            } else {
                false
            };

            if pruned {
                continue;
            }

            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, current_schema);

            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let row_id = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, row_id) || hot_skip.contains(&row_id) {
                    continue;
                }
                let row = if mapping.is_identity {
                    vol.get_row(i)
                } else {
                    vol.get_row_mapped(i, &mapping)
                };
                if let Some(expr) = where_expr {
                    if !expr.evaluate_fast(&row) {
                        continue;
                    }
                }
                result.push((row_id, row));
                collected += 1;
                if collected >= remaining {
                    break 'outer;
                }
            }
        }

        Ok(result.into_iter().skip(offset).take(limit).collect())
    }

    fn collect_rows_sorted_with_limit(
        &self,
        sort_col_idx: usize,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        if !self.segment_mgr.has_segments() {
            return self
                .hot
                .collect_rows_sorted_with_limit(sort_col_idx, ascending, limit, offset);
        }
        // Collect all merged rows, sort, take limit
        let mut rows = self.collect_all_rows(None)?;
        rows.sort_by(|(_, a), (_, b)| {
            let va = a.get(sort_col_idx);
            let vb = b.get(sort_col_idx);
            let cmp = match (va, vb) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(va), Some(vb)) => va.compare(vb).unwrap_or(std::cmp::Ordering::Equal),
            };
            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });
        Ok(rows
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(_, row)| row)
            .collect())
    }

    fn has_row_id(&self, row_id: i64) -> bool {
        if self.hot.has_row_id(row_id) {
            return true;
        }
        if !self.segment_mgr.has_segments() {
            return false;
        }
        // Snapshot-aware: row_exists() checks all tombstones unconditionally,
        // but a snapshot txn should still see rows tombstoned after its begin_seq.
        if self.snapshot_seq.is_some() {
            let ts = self.segment_mgr.tombstone_set_arc();
            if self.is_row_tombstoned(&ts, row_id) {
                return false;
            }
            return self.segment_mgr.is_row_id_in_volume(row_id);
        }
        self.segment_mgr.row_exists(row_id)
    }

    fn collect_hot_row_ids_into(&self, dest: &mut FxHashSet<i64>) {
        self.hot.collect_hot_row_ids_into(dest);
    }

    // =========================================================================
    // Row count
    // =========================================================================

    fn row_count(&self) -> usize {
        // Snapshot isolation: deduped_row_count and the fast path subtract ALL
        // tombstones, but a snapshot may not see newer ones. Use full scan.
        if self.snapshot_seq.is_some() {
            return self.collect_all_rows(None).map_or(0, |r| r.len());
        }
        // During seal, rows temporarily exist in both hot and cold.
        // Use the same O(1) formula with overlap correction. The count
        // may be off by a few rows during the brief overlap window, but
        // this avoids an O(total_rows) HashSet collection that made ALL
        // queries slow during checkpoint.
        let seg = self.segment_mgr.deduped_row_count();
        let pending = self.segment_mgr.pending_tombstone_count(self.txn_id());
        let overlap = self.segment_mgr.seal_overlap();
        seg.saturating_sub(pending) + self.hot.row_count().saturating_sub(overlap)
    }

    fn row_count_hint(&self) -> usize {
        let seg = self.segment_row_count_hint();
        let pending = self.segment_mgr.pending_tombstone_count(self.txn_id());
        let overlap = self.segment_mgr.seal_overlap();
        seg.saturating_sub(pending) + self.hot.row_count_hint().saturating_sub(overlap)
    }

    fn fast_row_count(&self) -> Option<usize> {
        // Snapshot isolation: deduped_row_count subtracts ALL tombstones, but
        // this snapshot may not see newer tombstones. Fall back to scan which
        // correctly filters by snapshot_seq.
        if self.snapshot_seq.is_some() {
            return None;
        }
        let hot_count = self.hot.fast_row_count()?;
        let seg = self.segment_mgr.deduped_row_count();
        let pending = self.segment_mgr.pending_tombstone_count(self.txn_id());
        let overlap = self.segment_mgr.seal_overlap();
        // During seal, rows temporarily exist in both hot and cold.
        // Subtract overlap to avoid double-counting. May drift by a few rows
        // during the brief window, but O(1) vs O(N) is worth it.
        Some(seg.saturating_sub(pending) + hot_count.saturating_sub(overlap))
    }

    // =========================================================================
    // Aggregation pushdown
    // =========================================================================

    fn sum_column(&self, col_idx: usize) -> Option<(f64, usize)> {
        // Snapshot isolation: cold aggregation uses tombstones without snapshot
        // filtering. Bail so the executor falls back to full scan.
        if self.snapshot_seq.is_some() {
            return None;
        }
        let hot_result = self.hot.sum_column(col_idx);

        if !self.segment_mgr.has_segments() {
            return hot_result;
        }

        // During seal, hot+cold overlap — can't reliably sum
        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        let (hot_sum, hot_count) = hot_result?;

        let has_tombstones = !self.segment_mgr.is_tombstone_set_empty()
            || self.segment_mgr.has_pending_tombstones(self.txn_id());

        // Pre-compute default contribution for schema-evolved volumes
        // that are missing this column (added via ALTER TABLE ADD COLUMN).
        let default_val = self.column_default(col_idx);
        let default_f64 = match &default_val {
            Value::Integer(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            _ => None, // NULL or non-numeric default → no contribution
        };

        if !has_tombstones {
            // Fast path: use pre-computed volume stats
            let segments = self.segment_mgr.get_segments_ordered();
            let mut total_sum = hot_sum;
            let mut total_count = hot_count;
            for vol in &segments {
                if col_idx < vol.stats.columns.len() {
                    total_sum += vol.stats.columns[col_idx].sum_as_f64();
                    total_count += vol.stats.columns[col_idx].numeric_count as usize;
                } else if let Some(def) = default_f64 {
                    // Old volume missing this column: every row has the default
                    total_sum += def * vol.row_count as f64;
                    total_count += vol.row_count;
                }
            }
            return Some((total_sum, total_count));
        }

        // Tombstones exist: scan columnar data with dedup (avoids full Row materialization)
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        let mut total_sum = hot_sum;
        let mut total_count = hot_count;

        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            let vol_has_col = col_idx < vol.columns.len();
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                if vol_has_col {
                    if vol.columns[col_idx].is_null(i) {
                        continue;
                    }
                    match &vol.columns[col_idx] {
                        crate::storage::volume::column::ColumnData::Int64 { values, .. } => {
                            total_sum += values[i] as f64;
                            total_count += 1;
                        }
                        crate::storage::volume::column::ColumnData::Float64 { values, .. } => {
                            total_sum += values[i];
                            total_count += 1;
                        }
                        _ => {}
                    }
                } else if let Some(def) = default_f64 {
                    total_sum += def;
                    total_count += 1;
                }
            }
        }
        Some((total_sum, total_count))
    }

    fn min_column(&self, col_idx: usize) -> Option<Option<Value>> {
        if self.snapshot_seq.is_some() {
            return None;
        }
        let hot_result = self.hot.min_column(col_idx);

        if !self.segment_mgr.has_segments() {
            return hot_result;
        }

        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        let hot_min = hot_result?;

        let has_tombstones = !self.segment_mgr.is_tombstone_set_empty()
            || self.segment_mgr.has_pending_tombstones(self.txn_id());

        let default_val = self.column_default(col_idx);
        let has_non_null_default = !default_val.is_null();

        if !has_tombstones {
            // Fast path: use pre-computed volume stats (zone map min)
            let segments = self.segment_mgr.get_segments_ordered();
            let mut overall_min = hot_min;
            for vol in &segments {
                let vol_min = if col_idx < vol.stats.columns.len() {
                    let m = &vol.stats.columns[col_idx].min;
                    if m.is_null() {
                        None
                    } else {
                        Some(m)
                    }
                } else if has_non_null_default && vol.row_count > 0 {
                    Some(&default_val)
                } else {
                    None
                };
                if let Some(vm) = vol_min {
                    match &overall_min {
                        None => overall_min = Some(vm.clone()),
                        Some(current) => {
                            if let Ok(std::cmp::Ordering::Less) = vm.compare(current) {
                                overall_min = Some(vm.clone());
                            }
                        }
                    }
                }
            }
            return Some(overall_min);
        }

        // Tombstones exist: scan columnar data with dedup
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        let mut overall_min = hot_min;
        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            let vol_has_col = col_idx < vol.columns.len();
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                let val = if vol_has_col {
                    if vol.columns[col_idx].is_null(i) {
                        continue;
                    }
                    vol.columns[col_idx].get_value(i)
                } else if has_non_null_default {
                    default_val.clone()
                } else {
                    continue;
                };
                match &overall_min {
                    None => overall_min = Some(val),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Less) = val.compare(current) {
                            overall_min = Some(val);
                        }
                    }
                }
            }
        }
        Some(overall_min)
    }

    fn max_column(&self, col_idx: usize) -> Option<Option<Value>> {
        if self.snapshot_seq.is_some() {
            return None;
        }
        let hot_result = self.hot.max_column(col_idx);

        if !self.segment_mgr.has_segments() {
            return hot_result;
        }

        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        let hot_max = hot_result?;

        let has_tombstones = !self.segment_mgr.is_tombstone_set_empty()
            || self.segment_mgr.has_pending_tombstones(self.txn_id());

        let default_val = self.column_default(col_idx);
        let has_non_null_default = !default_val.is_null();

        if !has_tombstones {
            // Fast path: use pre-computed volume stats (zone map max)
            let segments = self.segment_mgr.get_segments_ordered();
            let mut overall_max = hot_max;
            for vol in &segments {
                let vol_max = if col_idx < vol.stats.columns.len() {
                    let m = &vol.stats.columns[col_idx].max;
                    if m.is_null() {
                        None
                    } else {
                        Some(m)
                    }
                } else if has_non_null_default && vol.row_count > 0 {
                    Some(&default_val)
                } else {
                    None
                };
                if let Some(vm) = vol_max {
                    match &overall_max {
                        None => overall_max = Some(vm.clone()),
                        Some(current) => {
                            if let Ok(std::cmp::Ordering::Greater) = vm.compare(current) {
                                overall_max = Some(vm.clone());
                            }
                        }
                    }
                }
            }
            return Some(overall_max);
        }

        // Tombstones exist: scan columnar data with dedup
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        let mut overall_max = hot_max;
        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            let vol_has_col = col_idx < vol.columns.len();
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                let val = if vol_has_col {
                    if vol.columns[col_idx].is_null(i) {
                        continue;
                    }
                    vol.columns[col_idx].get_value(i)
                } else if has_non_null_default {
                    default_val.clone()
                } else {
                    continue;
                };
                match &overall_max {
                    None => overall_max = Some(val),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Greater) = val.compare(current) {
                            overall_max = Some(val);
                        }
                    }
                }
            }
        }
        Some(overall_max)
    }

    // =========================================================================
    // Partition and index-based pushdowns
    // =========================================================================

    fn get_partition_count(&self, column_name: &str) -> Option<usize> {
        if self.snapshot_seq.is_some() {
            return None;
        }
        if !self.segment_mgr.has_segments() {
            return self.hot.get_partition_count(column_name);
        }

        // During seal, hot+cold overlap — can't reliably count
        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        let schema = self.hot.schema();
        let col_idx = *schema.column_index_map().get(&column_name.to_lowercase())?;

        // Collect hot distinct values from index. If no index exists on this
        // column, bail — we can't enumerate hot values without a full scan.
        let mut distinct: ValueSet = ValueSet::default();
        let hot_values = self.hot.get_partition_values(column_name)?;
        for v in hot_values {
            distinct.insert(v);
        }

        // Build skip set: hot row_ids + tombstones + pending tombstones
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        // Scan cold volumes columnar-only (no Row materialization)
        let default_val = self.column_default(col_idx);
        let has_non_null_default = !default_val.is_null();
        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            let vol_has_col = col_idx < vol.columns.len();
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                if vol_has_col {
                    if !vol.columns[col_idx].is_null(i) {
                        distinct.insert(vol.columns[col_idx].get_value(i));
                    }
                } else if has_non_null_default {
                    distinct.insert(default_val.clone());
                }
            }
        }

        Some(distinct.len())
    }

    fn get_partition_values(&self, column_name: &str) -> Option<Vec<Value>> {
        if self.snapshot_seq.is_some() {
            return None;
        }
        if !self.segment_mgr.has_segments() {
            return self.hot.get_partition_values(column_name);
        }

        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        let schema = self.hot.schema();
        let col_idx = *schema.column_index_map().get(&column_name.to_lowercase())?;

        // Bail if hot has no index on this column — can't enumerate hot values
        // without a full scan. Returning Some with only cold values would be wrong.
        let mut distinct: ValueSet = ValueSet::default();
        let hot_values = self.hot.get_partition_values(column_name)?;
        for v in hot_values {
            distinct.insert(v);
        }

        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

        let default_val = self.column_default(col_idx);
        let has_non_null_default = !default_val.is_null();
        for (_, cs) in volumes.iter() {
            let vol = &cs.volume;
            let vol_has_col = col_idx < vol.columns.len();
            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                if vol_has_col {
                    if !vol.columns[col_idx].is_null(i) {
                        distinct.insert(vol.columns[col_idx].get_value(i));
                    }
                } else if has_non_null_default {
                    distinct.insert(default_val.clone());
                }
            }
        }

        Some(distinct.into_iter().collect())
    }

    fn collect_rows_grouped_by_partition(&self, column_name: &str) -> Option<Vec<(Value, RowVec)>> {
        if self.snapshot_seq.is_some() {
            return None;
        }
        if !self.segment_mgr.has_segments() {
            return self.hot.collect_rows_grouped_by_partition(column_name);
        }

        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        let schema = self.hot.schema();
        let col_idx = *schema.column_index_map().get(&column_name.to_lowercase())?;

        // Start from hot grouped data
        let mut groups: ValueMap<RowVec> = ValueMap::default();
        if let Some(hot_groups) = self.hot.collect_rows_grouped_by_partition(column_name) {
            for (val, rows) in hot_groups {
                groups.insert(val, rows);
            }
        }

        // Build hot_skip: hot row_ids + tombstones + pending tombstones
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);
        let current_schema = self.hot.schema();

        let default_val = self.column_default(col_idx);
        let has_non_null_default = !default_val.is_null();
        for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, current_schema);
            // Resolve partition column through mapping (handles DROP COLUMN ordinal shifts)
            let phys_col = match &mapping.sources[col_idx] {
                super::writer::ColSource::Volume(idx) => Some(*idx),
                super::writer::ColSource::Default(_) => None,
            };

            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                let val = if let Some(pc) = phys_col {
                    if vol.columns[pc].is_null(i) {
                        continue;
                    }
                    vol.columns[pc].get_value(i)
                } else if has_non_null_default {
                    default_val.clone()
                } else {
                    continue;
                };
                let row = if mapping.is_identity {
                    vol.get_row(i)
                } else {
                    vol.get_row_mapped(i, &mapping)
                };
                groups.entry(val).or_default().push((rid, row));
            }
        }

        Some(groups.into_iter().collect())
    }

    fn get_rows_for_partition_value(
        &self,
        column_name: &str,
        partition_value: &Value,
    ) -> Option<RowVec> {
        if self.snapshot_seq.is_some() {
            return None;
        }
        if !self.segment_mgr.has_segments() {
            return self
                .hot
                .get_rows_for_partition_value(column_name, partition_value);
        }

        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        let schema = self.hot.schema();
        let col_idx = *schema.column_index_map().get(&column_name.to_lowercase())?;

        // Get hot rows for this partition value
        let mut result = self
            .hot
            .get_rows_for_partition_value(column_name, partition_value)
            .unwrap_or_default();

        // Build hot_skip: hot row_ids + tombstones + pending tombstones
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);
        let current_schema = self.hot.schema();

        let default_val = self.column_default(col_idx);
        let has_non_null_default = !default_val.is_null();
        for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, current_schema);
            // Resolve partition column through mapping (handles DROP COLUMN ordinal shifts)
            let phys_col = match &mapping.sources[col_idx] {
                super::writer::ColSource::Volume(idx) => Some(*idx),
                super::writer::ColSource::Default(_) => None,
            };
            // Skip volume if missing column and default doesn't match target
            if phys_col.is_none() && (!has_non_null_default || &default_val != partition_value) {
                continue;
            }

            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                let matches = if let Some(pc) = phys_col {
                    if vol.columns[pc].is_null(i) {
                        false
                    } else {
                        &vol.columns[pc].get_value(i) == partition_value
                    }
                } else {
                    // Missing column → all rows have default, already checked match above
                    true
                };
                if matches {
                    let row = if mapping.is_identity {
                        vol.get_row(i)
                    } else {
                        vol.get_row_mapped(i, &mapping)
                    };
                    result.push((rid, row));
                }
            }
        }

        Some(result)
    }

    fn collect_rows_ordered_by_index(
        &self,
        column_name: &str,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<RowVec> {
        if !self.segment_mgr.has_segments() {
            return self
                .hot
                .collect_rows_ordered_by_index(column_name, ascending, limit, offset);
        }
        // Hot index doesn't cover cold data — can't use index ordering
        None
    }

    fn collect_rows_pk_keyset(
        &self,
        start_after: Option<i64>,
        start_from: Option<i64>,
        ascending: bool,
        limit: usize,
    ) -> Option<RowVec> {
        if !self.segment_mgr.has_segments() {
            return self
                .hot
                .collect_rows_pk_keyset(start_after, start_from, ascending, limit);
        }
        // Hot PK index doesn't cover cold data — can't use keyset pagination
        None
    }

    // =========================================================================
    // Transaction operations — delegate to hot buffer
    // =========================================================================

    fn close(&mut self) -> Result<()> {
        self.hot.close()
    }

    fn commit(&mut self) -> Result<()> {
        self.hot.commit()?;
        // Apply pending tombstones to the shared tombstone set.
        // commit_seq=0 means "always visible to all snapshots". This is safe because
        // the main commit path goes through engine.commit_all_tables() which passes
        // the real commit_seq. This fallback is for direct Table::commit() calls.
        let txn_id = self.txn_id();
        self.segment_mgr.commit_pending_tombstones(txn_id, 0);
        self.segment_mgr.clear_txn_seal_generation(txn_id);
        Ok(())
    }

    fn rollback(&mut self) {
        self.hot.rollback();
        let txn_id = self.txn_id();
        self.segment_mgr.rollback_pending_tombstones(txn_id);
        self.segment_mgr.clear_txn_seal_generation(txn_id);
    }

    fn rollback_to_timestamp(&self, timestamp: i64) {
        self.hot.rollback_to_timestamp(timestamp);
    }

    fn has_local_changes(&self) -> bool {
        self.hot.has_local_changes() || self.segment_mgr.has_pending_tombstones(self.txn_id())
    }

    fn get_pending_versions(&self) -> Vec<(i64, Row, bool, i64)> {
        self.hot.get_pending_versions()
    }

    // =========================================================================
    // Index operations
    // =========================================================================

    fn create_index(&self, name: &str, columns: &[&str], is_unique: bool) -> Result<()> {
        // For unique indexes, validate cold data has no duplicates first.
        if is_unique && self.segment_mgr.has_segments() {
            self.validate_cold_unique(name, columns)?;
        }
        self.hot.create_index(name, columns, is_unique)
    }

    fn create_index_with_type(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
        index_type: Option<IndexType>,
    ) -> Result<()> {
        // For unique indexes (non-HNSW), validate cold data has no duplicates first.
        // HNSW unique validation happens during populate_index_from_cold via index.add().
        if is_unique && index_type != Some(IndexType::Hnsw) && self.segment_mgr.has_segments() {
            self.validate_cold_unique(name, columns)?;
        }

        self.hot
            .create_index_with_type(name, columns, is_unique, index_type)?;

        // HNSW indexes store all data (hot + cold). After creating the index
        // on the hot store, populate it from cold segments.
        if index_type == Some(IndexType::Hnsw) && self.segment_mgr.has_segments() {
            if let Err(e) = self.populate_index_from_cold(name, columns) {
                // Roll back the hot index on cold population failure
                let _ = self.hot.drop_index(name);
                return Err(e);
            }
        }
        Ok(())
    }

    fn create_hnsw_index(
        &self,
        name: &str,
        column: &str,
        is_unique: bool,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: crate::storage::index::HnswDistanceMetric,
    ) -> Result<()> {
        // Delegate to hot store which creates the HNSW with custom params
        self.hot.create_hnsw_index(
            name,
            column,
            is_unique,
            m,
            ef_construction,
            ef_search,
            metric,
        )?;

        // Populate from cold segments (HNSW must include all data)
        if self.segment_mgr.has_segments() {
            if let Err(e) = self.populate_index_from_cold(name, &[column]) {
                let _ = self.hot.drop_index(name);
                return Err(e);
            }
        }
        Ok(())
    }

    fn drop_index(&self, name: &str) -> Result<()> {
        self.hot.drop_index(name)
    }

    fn create_btree_index(
        &self,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()> {
        if is_unique && self.segment_mgr.has_segments() {
            self.validate_cold_unique(custom_name.unwrap_or(column_name), &[column_name])?;
        }
        self.hot
            .create_btree_index(column_name, is_unique, custom_name)
    }

    fn drop_btree_index(&self, column_name: &str) -> Result<()> {
        self.hot.drop_btree_index(column_name)
    }

    fn create_multi_column_index(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
    ) -> Result<()> {
        if is_unique && self.segment_mgr.has_segments() {
            self.validate_cold_unique(name, columns)?;
        }
        self.hot.create_multi_column_index(name, columns, is_unique)
    }

    fn has_index_on_column(&self, column_name: &str) -> bool {
        self.hot.has_index_on_column(column_name)
    }

    fn get_index_on_column(&self, column_name: &str) -> Option<Arc<dyn Index>> {
        self.hot.get_index_on_column(column_name)
    }

    fn get_index(&self, name: &str) -> Option<Arc<dyn Index>> {
        self.hot.get_index(name)
    }

    fn get_unique_indexes(&self) -> Vec<(String, Vec<String>)> {
        self.hot.get_unique_indexes()
    }

    fn for_each_unique_non_pk_index(
        &self,
        f: &mut dyn FnMut(&str, &[String]) -> Result<()>,
    ) -> Result<()> {
        self.hot.for_each_unique_non_pk_index(f)
    }

    fn find_unique_conflict_row_id(
        &self,
        _index_name: &str,
        column_name: &str,
        row_values: &[Value],
    ) -> Result<Option<i64>> {
        if !self.segment_mgr.has_segments() {
            return Ok(None);
        }

        let schema = self.hot.schema();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for col_name in column_name.split(", ") {
            let Some(&col_idx) = schema
                .column_index_map()
                .get(col_name.to_lowercase().as_str())
            else {
                return Ok(None);
            };
            let Some(value) = row_values.get(col_idx) else {
                return Ok(None);
            };
            if value.is_null() {
                return Ok(None);
            }
            col_indices.push(col_idx);
            values.push(value);
        }

        Ok(self.find_segment_row_id_by_values(&col_indices, &values))
    }

    fn has_unique_non_pk_indexes(&self) -> bool {
        self.hot.has_unique_non_pk_indexes()
    }

    fn acquire_upsert_lock(&self) -> Option<Box<dyn std::any::Any>> {
        self.hot.acquire_upsert_lock()
    }

    fn get_multi_column_index(
        &self,
        predicate_columns: &[&str],
    ) -> Option<(Arc<dyn Index>, usize)> {
        self.hot.get_multi_column_index(predicate_columns)
    }

    fn get_index_min_value(&self, column_name: &str) -> Option<Value> {
        if !self.segment_mgr.has_segments() {
            return self.hot.get_index_min_value(column_name);
        }
        let hot_min = self.hot.get_index_min_value(column_name);
        let segments = self.segment_mgr.get_segments_ordered();
        let col_idx = segments.first()?.column_index(column_name)?;
        let mut vol_min: Option<Value> = None;
        for vol in &segments {
            let zm_min = &vol.zone_maps[col_idx].min;
            if !zm_min.is_null() {
                match &vol_min {
                    None => vol_min = Some(zm_min.clone()),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Less) = zm_min.compare(current) {
                            vol_min = Some(zm_min.clone());
                        }
                    }
                }
            }
        }
        match (hot_min, vol_min) {
            (Some(h), Some(v)) => {
                if let Ok(std::cmp::Ordering::Less) = v.compare(&h) {
                    Some(v)
                } else {
                    Some(h)
                }
            }
            (Some(h), None) => Some(h),
            (None, Some(v)) => Some(v),
            (None, None) => None,
        }
    }

    fn get_index_max_value(&self, column_name: &str) -> Option<Value> {
        if !self.segment_mgr.has_segments() {
            return self.hot.get_index_max_value(column_name);
        }
        let hot_max = self.hot.get_index_max_value(column_name);
        let segments = self.segment_mgr.get_segments_ordered();
        let col_idx = segments.first()?.column_index(column_name)?;
        let mut vol_max: Option<Value> = None;
        for vol in &segments {
            let zm_max = &vol.zone_maps[col_idx].max;
            if !zm_max.is_null() {
                match &vol_max {
                    None => vol_max = Some(zm_max.clone()),
                    Some(current) => {
                        if let Ok(std::cmp::Ordering::Greater) = zm_max.compare(current) {
                            vol_max = Some(zm_max.clone());
                        }
                    }
                }
            }
        }
        match (hot_max, vol_max) {
            (Some(h), Some(v)) => {
                if let Ok(std::cmp::Ordering::Greater) = v.compare(&h) {
                    Some(v)
                } else {
                    Some(h)
                }
            }
            (Some(h), None) => Some(h),
            (None, Some(v)) => Some(v),
            (None, None) => None,
        }
    }

    // =========================================================================
    // Column operations — delegate to hot buffer
    // =========================================================================

    fn rename_column(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        self.hot.rename_column(old_name, new_name)
    }

    fn modify_column(&mut self, name: &str, column_type: DataType, nullable: bool) -> Result<()> {
        self.hot.modify_column(name, column_type, nullable)
    }

    // =========================================================================
    // Query operations
    // =========================================================================

    fn select(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn QueryResult>> {
        if !self.segment_mgr.has_segments() {
            return self.hot.select(columns, expr);
        }
        let all_rows = self.collect_all_rows(expr)?;
        let col_names: Vec<String> = columns.iter().map(|c| c.to_string()).collect();
        Ok(Box::new(crate::executor::result::ExecutorResult::new(
            col_names, all_rows,
        )))
    }

    fn select_with_aliases(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
    ) -> Result<Box<dyn QueryResult>> {
        if !self.segment_mgr.has_segments() {
            return self.hot.select_with_aliases(columns, expr, aliases);
        }
        self.select(columns, expr)
    }

    fn select_as_of(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        temporal_type: &str,
        temporal_value: i64,
    ) -> Result<Box<dyn QueryResult>> {
        if !self.segment_mgr.has_segments() {
            return self
                .hot
                .select_as_of(columns, expr, temporal_type, temporal_value);
        }

        // Historical point-in-time queries on segment-backed tables are not
        // supported because cold rows lack version chains and create_time.
        let is_current_query = temporal_type.eq_ignore_ascii_case("CURRENT");
        if !is_current_query {
            return Err(crate::core::Error::internal(
                "AS OF temporal queries are not supported on tables with sealed segments. \
                 Sealed rows lack version history for point-in-time reconstruction.",
            ));
        }

        // Hot buffer handles temporal filtering via version chains.
        let mut hot_result = self
            .hot
            .select_as_of(columns, expr, temporal_type, temporal_value)?;

        let schema = self.hot.schema().clone();
        let pk_idx = schema.columns.iter().position(|c| c.primary_key);
        let mut hot_pks: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        let mut all_rows = RowVec::new();

        while hot_result.next() {
            let row = hot_result.take_row();
            if let Some(pi) = pk_idx {
                if let Some(Value::Integer(pk)) = row.get(pi) {
                    hot_pks.insert(*pk);
                }
            }
            all_rows.push((0, row));
        }

        // Cold rows in segments have no version chain and no create_time,
        // so they cannot participate in SYSTEM_TIME temporal queries.
        // Cold segments store raw data without version chains, so they're valid
        // only for "CURRENT" temporal queries. Historical queries return early above.
        if is_current_query {
            // Build hot_skip from hot row_ids + pending tombstones.
            // Committed tombstones are kept as a shared Arc (no clone).
            let volumes = self.segment_mgr.get_volumes_newest_first();
            let tombstones_arc = self.segment_mgr.tombstone_set_arc();
            let mut hot_skip: FxHashSet<i64> =
                FxHashSet::with_capacity_and_hasher(10_000, Default::default());
            self.hot.collect_hot_row_ids_into(&mut hot_skip);
            self.segment_mgr
                .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);

            for (seg_id, cs) in volumes.iter() {
                let vol = &cs.volume;
                let mapping = self.segment_mgr.get_volume_mapping(*seg_id, &schema);
                for i in 0..vol.row_count {
                    if !cs.is_visible(i) {
                        continue;
                    }
                    let row_id = vol.row_ids[i];
                    if self.is_row_tombstoned(&tombstones_arc, row_id) || hot_skip.contains(&row_id)
                    {
                        continue;
                    }
                    if let Some(pi) = pk_idx {
                        if pi < vol.column_types.len()
                            && matches!(
                                vol.column_types[pi],
                                DataType::Integer | DataType::Timestamp
                            )
                            && pi < vol.columns.len()
                            && !vol.columns[pi].is_null(i)
                        {
                            let pk_val = vol.columns[pi].get_i64(i);
                            if hot_pks.contains(&pk_val) {
                                continue;
                            }
                        }
                    }
                    let row = if mapping.is_identity {
                        vol.get_row(i)
                    } else {
                        vol.get_row_mapped(i, &mapping)
                    };
                    if let Some(e) = expr {
                        if !e.evaluate_fast(&row) {
                            continue;
                        }
                    }
                    all_rows.push((row_id, row));
                }
            }
        }
        // For historical temporal queries (SYSTEM_TIME AS OF <timestamp>),
        // cold rows are excluded since they lack version history.
        // The hot buffer's AS OF result is the authoritative source.

        let col_names: Vec<String> = columns.iter().map(|c| c.to_string()).collect();
        Ok(Box::new(crate::executor::result::ExecutorResult::new(
            col_names, all_rows,
        )))
    }

    fn explain_scan(&self, where_expr: Option<&dyn Expression>) -> ScanPlan {
        self.hot.explain_scan(where_expr)
    }

    // =========================================================================
    // Zone maps — delegate to hot buffer
    // =========================================================================

    fn set_zone_maps(&self, zone_maps: crate::storage::mvcc::zonemap::TableZoneMap) {
        self.hot.set_zone_maps(zone_maps)
    }

    fn get_zone_maps(&self) -> Option<Arc<crate::storage::mvcc::zonemap::TableZoneMap>> {
        self.hot.get_zone_maps()
    }

    // =========================================================================
    // Deferred aggregation
    // =========================================================================

    fn compute_grouped_aggregates(
        &self,
        group_by_indices: &[usize],
        aggregates: &[(AggregateOp, usize)],
    ) -> Option<Vec<GroupedAggregateResult>> {
        if self.snapshot_seq.is_some() {
            return None;
        }
        if !self.segment_mgr.has_segments() {
            return self
                .hot
                .compute_grouped_aggregates(group_by_indices, aggregates);
        }

        // During seal, hot+cold overlap — can't reliably aggregate
        if self.segment_mgr.seal_overlap() > 0 {
            return None;
        }

        // Accumulator per group
        #[derive(Clone)]
        struct Accum {
            count: i64,
            int_sum: i128,
            float_sum: f64,
            min: Option<Value>,
            max: Option<Value>,
        }

        impl Default for Accum {
            fn default() -> Self {
                Self {
                    count: 0,
                    int_sum: 0,
                    float_sum: 0.0,
                    min: None,
                    max: None,
                }
            }
        }

        #[inline(always)]
        fn update_accums(accums: &mut [Accum], aggregates: &[(AggregateOp, usize)], row: &Row) {
            for (agg_idx, (op, col_idx)) in aggregates.iter().enumerate() {
                let accum = &mut accums[agg_idx];
                match op {
                    AggregateOp::CountStar => accum.count += 1,
                    AggregateOp::Count => {
                        if let Some(v) = row.get(*col_idx) {
                            if !v.is_null() {
                                accum.count += 1;
                            }
                        }
                    }
                    AggregateOp::Sum | AggregateOp::Avg => {
                        if let Some(v) = row.get(*col_idx) {
                            match v {
                                Value::Integer(i) => {
                                    accum.int_sum += *i as i128;
                                    accum.count += 1;
                                }
                                Value::Float(f) => {
                                    accum.float_sum += *f;
                                    accum.count += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                    AggregateOp::Min => {
                        if let Some(v) = row.get(*col_idx) {
                            if !v.is_null() {
                                match &accum.min {
                                    None => accum.min = Some(v.clone()),
                                    Some(cur) => {
                                        if v < cur {
                                            accum.min = Some(v.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AggregateOp::Max => {
                        if let Some(v) = row.get(*col_idx) {
                            if !v.is_null() {
                                match &accum.max {
                                    None => accum.max = Some(v.clone()),
                                    Some(cur) => {
                                        if v > cur {
                                            accum.max = Some(v.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        fn finalize(aggregates: &[(AggregateOp, usize)], accums: &[Accum]) -> Vec<Value> {
            aggregates
                .iter()
                .zip(accums.iter())
                .map(|((op, _), acc)| match op {
                    AggregateOp::Count | AggregateOp::CountStar => Value::Integer(acc.count),
                    AggregateOp::Sum => {
                        if acc.count > 0 {
                            Value::Float(acc.int_sum as f64 + acc.float_sum)
                        } else {
                            Value::Null(DataType::Float)
                        }
                    }
                    AggregateOp::Avg => {
                        if acc.count > 0 {
                            Value::Float((acc.int_sum as f64 + acc.float_sum) / acc.count as f64)
                        } else {
                            Value::Null(DataType::Float)
                        }
                    }
                    AggregateOp::Min => acc.min.clone().unwrap_or(Value::Null(DataType::Null)),
                    AggregateOp::Max => acc.max.clone().unwrap_or(Value::Null(DataType::Null)),
                })
                .collect()
        }

        // Build group map: group_key -> accumulators
        let mut groups: ValueMap<(Vec<Value>, Vec<Accum>)> = ValueMap::default();

        // Process hot rows
        let hot_rows = self.hot.collect_all_rows(None).ok()?;
        for (_, row) in &hot_rows {
            // Build group key
            let key = if group_by_indices.len() == 1 {
                row.get(group_by_indices[0])
                    .cloned()
                    .unwrap_or(Value::Null(DataType::Null))
            } else {
                // Multi-column group key: use first column as hash key,
                // store full key in the value
                row.get(group_by_indices[0])
                    .cloned()
                    .unwrap_or(Value::Null(DataType::Null))
            };

            let group_key_values: Vec<Value> = group_by_indices
                .iter()
                .map(|&idx| row.get(idx).cloned().unwrap_or(Value::Null(DataType::Null)))
                .collect();

            let entry = groups
                .entry(key)
                .or_insert_with(|| (group_key_values, vec![Accum::default(); aggregates.len()]));
            update_accums(&mut entry.1, aggregates, row);
        }

        // Process cold volumes with visibility bitmap dedup
        let volumes = self.segment_mgr.get_volumes_newest_first();
        let tombstones_arc = self.segment_mgr.tombstone_set_arc();
        let mut hot_skip: FxHashSet<i64> =
            FxHashSet::with_capacity_and_hasher(10_000, Default::default());
        self.hot.collect_hot_row_ids_into(&mut hot_skip);
        self.segment_mgr
            .insert_pending_tombstones_into(self.txn_id(), &mut hot_skip);
        let current_schema = self.hot.schema();

        for (seg_id, cs) in volumes.iter() {
            let vol = &cs.volume;
            let mapping = self.segment_mgr.get_volume_mapping(*seg_id, current_schema);

            for i in 0..vol.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                let rid = vol.row_ids[i];
                if self.is_row_tombstoned(&tombstones_arc, rid) || hot_skip.contains(&rid) {
                    continue;
                }
                let row = if mapping.is_identity {
                    vol.get_row(i)
                } else {
                    vol.get_row_mapped(i, &mapping)
                };

                let key = row
                    .get(group_by_indices[0])
                    .cloned()
                    .unwrap_or(Value::Null(DataType::Null));

                let group_key_values: Vec<Value> = group_by_indices
                    .iter()
                    .map(|&idx| row.get(idx).cloned().unwrap_or(Value::Null(DataType::Null)))
                    .collect();

                let entry = groups.entry(key).or_insert_with(|| {
                    (group_key_values, vec![Accum::default(); aggregates.len()])
                });
                update_accums(&mut entry.1, aggregates, &row);
            }
        }

        // Convert to results
        let results: Vec<GroupedAggregateResult> = groups
            .into_values()
            .map(|(group_values, accums)| GroupedAggregateResult {
                group_values,
                aggregate_values: finalize(aggregates, &accums),
            })
            .collect();

        Some(results)
    }
}

#[cfg(test)]
mod tests {
    use super::super::writer::VolumeBuilder;
    use super::*;
    use crate::core::SchemaBuilder;
    use std::path::PathBuf;

    use super::super::manifest::SegmentMeta;

    /// Helper: create a SegmentManager with one segment from given rows.
    fn make_segment_mgr(schema: &Schema, rows: &[(i64, Row)]) -> Arc<SegmentManager> {
        let mut builder = VolumeBuilder::with_capacity(schema, rows.len());
        for (id, row) in rows {
            builder.add_row(*id, row);
        }
        let vol = Arc::new(builder.finish());
        let min_id = rows.first().map(|(id, _)| *id).unwrap_or(0);
        let max_id = rows.last().map(|(id, _)| *id).unwrap_or(0);

        let mgr = Arc::new(SegmentManager::new("test", None));
        mgr.register_segment(
            1,
            vol,
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("test.vol"),
                row_count: rows.len(),
                min_row_id: min_id,
                max_row_id: max_id,
                schema_version: 0,
                creation_lsn: 0,
                compaction_epoch: 0,
            },
            None,
        );
        mgr
    }

    /// Minimal test table for the hot buffer
    struct MockHotTable {
        schema: Schema,
        rows: Vec<(i64, Row)>,
    }

    impl MockHotTable {
        fn new(schema: Schema, rows: Vec<(i64, Row)>) -> Self {
            Self { schema, rows }
        }
    }

    impl Table for MockHotTable {
        fn name(&self) -> &str {
            "test"
        }
        fn schema(&self) -> &Schema {
            &self.schema
        }
        fn txn_id(&self) -> i64 {
            1
        }
        fn create_column(&mut self, _: &str, _: DataType, _: bool) -> Result<()> {
            Ok(())
        }
        fn drop_column(&mut self, _: &str) -> Result<()> {
            Ok(())
        }
        fn insert(&mut self, row: Row) -> Result<Row> {
            let id = self.rows.len() as i64 + 1000;
            self.rows.push((id, row.clone()));
            Ok(row)
        }
        fn insert_batch(&mut self, rows: Vec<Row>) -> Result<()> {
            for row in rows {
                self.insert(row)?;
            }
            Ok(())
        }
        fn update(
            &mut self,
            _: Option<&dyn Expression>,
            _: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
        ) -> Result<i32> {
            Ok(0)
        }
        fn update_by_row_ids(
            &mut self,
            _: &[i64],
            _: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
        ) -> Result<i32> {
            Ok(0)
        }
        fn delete_by_row_ids(&mut self, _: &[i64]) -> Result<i32> {
            Ok(0)
        }
        fn get_active_row_ids(&self) -> Vec<i64> {
            self.rows.iter().map(|(id, _)| *id).collect()
        }
        fn delete(&mut self, _: Option<&dyn Expression>) -> Result<i32> {
            Ok(0)
        }
        fn scan(&self, _: &[usize], _: Option<&dyn Expression>) -> Result<Box<dyn Scanner>> {
            Ok(Box::new(crate::storage::traits::VecScanner::new(
                self.rows.iter().map(|(_, r)| r.clone()).collect(),
            )))
        }
        fn collect_all_rows(&self, _: Option<&dyn Expression>) -> Result<RowVec> {
            let mut rv = RowVec::with_capacity(self.rows.len());
            for (id, row) in &self.rows {
                rv.push((*id, row.clone()));
            }
            Ok(rv)
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn commit(&mut self) -> Result<()> {
            Ok(())
        }
        fn rollback(&mut self) {}
        fn rollback_to_timestamp(&self, _: i64) {}
        fn has_local_changes(&self) -> bool {
            false
        }
        fn create_index(&self, _: &str, _: &[&str], _: bool) -> Result<()> {
            Ok(())
        }
        fn drop_index(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn create_btree_index(&self, _: &str, _: bool, _: Option<&str>) -> Result<()> {
            Ok(())
        }
        fn drop_btree_index(&self, _: &str) -> Result<()> {
            Ok(())
        }
        fn rename_column(&mut self, _: &str, _: &str) -> Result<()> {
            Ok(())
        }
        fn modify_column(&mut self, _: &str, _: DataType, _: bool) -> Result<()> {
            Ok(())
        }
        fn select(&self, _: &[&str], _: Option<&dyn Expression>) -> Result<Box<dyn QueryResult>> {
            Err(crate::core::Error::internal("not implemented"))
        }
        fn select_with_aliases(
            &self,
            _: &[&str],
            _: Option<&dyn Expression>,
            _: &FxHashMap<String, String>,
        ) -> Result<Box<dyn QueryResult>> {
            Err(crate::core::Error::internal("not implemented"))
        }
        fn select_as_of(
            &self,
            _: &[&str],
            _: Option<&dyn Expression>,
            _: &str,
            _: i64,
        ) -> Result<Box<dyn QueryResult>> {
            Err(crate::core::Error::internal("not implemented"))
        }
        fn row_count(&self) -> usize {
            self.rows.len()
        }
        fn fast_row_count(&self) -> Option<usize> {
            Some(self.rows.len())
        }
        fn max_column(&self, col_idx: usize) -> Option<Option<Value>> {
            let mut max: Option<Value> = None;
            for (_, row) in &self.rows {
                if let Some(val) = row.get(col_idx) {
                    if !val.is_null() {
                        match &max {
                            None => max = Some(val.clone()),
                            Some(current) => {
                                if let Ok(std::cmp::Ordering::Greater) = val.compare(current) {
                                    max = Some(val.clone());
                                }
                            }
                        }
                    }
                }
            }
            Some(max)
        }
        fn sum_column(&self, col_idx: usize) -> Option<(f64, usize)> {
            let mut sum = 0.0;
            let mut count = 0;
            for (_, row) in &self.rows {
                match row.get(col_idx) {
                    Some(Value::Integer(i)) => {
                        sum += *i as f64;
                        count += 1;
                    }
                    Some(Value::Float(f)) => {
                        sum += *f;
                        count += 1;
                    }
                    _ => {}
                }
            }
            Some((sum, count))
        }
    }

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("value", DataType::Float, false, false)
            .build()
    }

    #[test]
    fn test_hot_only() {
        let schema = test_schema();
        let hot = MockHotTable::new(
            schema.clone(),
            vec![
                (
                    1,
                    Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
                ),
                (
                    2,
                    Row::from_values(vec![Value::Integer(2), Value::Float(20.0)]),
                ),
            ],
        );
        let table = SegmentedTable::hot_only(Box::new(hot));

        assert_eq!(table.row_count(), 2);
        assert_eq!(table.segment_count(), 0);
    }

    #[test]
    fn test_row_count_merges() {
        let schema = test_schema();

        let hot = MockHotTable::new(
            schema.clone(),
            vec![
                (
                    100,
                    Row::from_values(vec![Value::Integer(100), Value::Float(500.0)]),
                ),
                (
                    101,
                    Row::from_values(vec![Value::Integer(101), Value::Float(600.0)]),
                ),
            ],
        );

        let seg_rows: Vec<(i64, Row)> = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::Float(20.0)]),
            ),
            (
                3,
                Row::from_values(vec![Value::Integer(3), Value::Float(30.0)]),
            ),
        ];
        let mgr = make_segment_mgr(&schema, &seg_rows);

        let table = SegmentedTable::new(Box::new(hot), mgr);

        assert_eq!(table.row_count(), 5); // 3 segment + 2 hot
        assert_eq!(table.fast_row_count(), Some(5));
    }

    #[test]
    fn test_collect_all_rows_merges() {
        let schema = test_schema();

        let hot = MockHotTable::new(
            schema.clone(),
            vec![(
                100,
                Row::from_values(vec![Value::Integer(100), Value::Float(500.0)]),
            )],
        );

        let seg_rows: Vec<(i64, Row)> = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::Float(20.0)]),
            ),
        ];
        let mgr = make_segment_mgr(&schema, &seg_rows);

        let table = SegmentedTable::new(Box::new(hot), mgr);

        let rows = table.collect_all_rows(None).unwrap();
        assert_eq!(rows.len(), 3); // 2 segment + 1 hot
    }

    #[test]
    fn test_max_column_merges() {
        let schema = test_schema();

        let hot = MockHotTable::new(
            schema.clone(),
            vec![(
                100,
                Row::from_values(vec![Value::Integer(100), Value::Float(500.0)]),
            )],
        );

        let seg_rows: Vec<(i64, Row)> = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::Float(30.0)]),
            ),
        ];
        let mgr = make_segment_mgr(&schema, &seg_rows);

        let table = SegmentedTable::new(Box::new(hot), mgr);

        let max = table.max_column(1);
        assert_eq!(max, Some(Some(Value::Float(500.0))));
    }

    #[test]
    fn test_sum_column_merges() {
        let schema = test_schema();

        let hot = MockHotTable::new(
            schema.clone(),
            vec![(
                100,
                Row::from_values(vec![Value::Integer(100), Value::Float(500.0)]),
            )],
        );

        let seg_rows: Vec<(i64, Row)> = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::Float(20.0)]),
            ),
        ];
        let mgr = make_segment_mgr(&schema, &seg_rows);

        let table = SegmentedTable::new(Box::new(hot), mgr);

        let (sum, count) = table.sum_column(1).unwrap();
        assert_eq!(sum, 530.0);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_scan_merges_via_merging_scanner() {
        let schema = test_schema();

        let hot = MockHotTable::new(
            schema.clone(),
            vec![(
                100,
                Row::from_values(vec![Value::Integer(100), Value::Float(500.0)]),
            )],
        );

        let seg_rows: Vec<(i64, Row)> = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::Float(20.0)]),
            ),
        ];
        let mgr = make_segment_mgr(&schema, &seg_rows);

        let table = SegmentedTable::new(Box::new(hot), mgr);

        let all_col_indices: Vec<usize> = (0..schema.columns.len()).collect();
        let mut scanner = table.scan(&all_col_indices, None).unwrap();

        let mut count = 0;
        let mut ids = Vec::new();
        while scanner.next() {
            if let Some(Value::Integer(id)) = scanner.row().get(0) {
                ids.push(*id);
            }
            count += 1;
        }
        assert_eq!(count, 3);
        assert_eq!(ids, vec![1, 2, 100]);
    }

    #[test]
    fn test_insert_goes_to_hot_buffer() {
        let schema = test_schema();
        let hot = MockHotTable::new(schema.clone(), vec![]);

        let mut table = SegmentedTable::hot_only(Box::new(hot));
        assert_eq!(table.row_count(), 0);

        table
            .insert(Row::from_values(vec![
                Value::Integer(1),
                Value::Float(10.0),
            ]))
            .unwrap();
        assert_eq!(table.row_count(), 1);
    }

    #[test]
    fn test_delete_tombstones_cold_row() {
        let schema = test_schema();
        let hot = MockHotTable::new(schema.clone(), vec![]);

        let seg_rows: Vec<(i64, Row)> = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::Float(20.0)]),
            ),
            (
                3,
                Row::from_values(vec![Value::Integer(3), Value::Float(30.0)]),
            ),
        ];
        let mgr = make_segment_mgr(&schema, &seg_rows);

        let mut table = SegmentedTable::new(Box::new(hot), Arc::clone(&mgr));

        // Delete row 2 via delete_by_row_ids
        let deleted = table.delete_by_row_ids(&[2]).unwrap();
        assert_eq!(deleted, 1);

        // Tombstones are deferred until commit. Before commit, the shared
        // segment manager still sees the row as live.
        assert_eq!(mgr.total_row_count(), 3);
        assert!(!mgr.is_tombstoned(2));

        // After commit, tombstones are applied to the shared segment manager.
        table.commit().unwrap();
        assert_eq!(mgr.total_row_count(), 2);
        assert!(mgr.is_tombstoned(2));
        assert!(mgr.row_exists(1));
        assert!(mgr.row_exists(3));
        assert!(!mgr.row_exists(2));
    }
}
