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
use crate::storage::mvcc::version_store::CapturedHotView;
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

/// Cache holding decompressed columns for a single row group.
/// Avoids decompressing the entire column when only a few groups are needed.
struct GroupColumnCache {
    #[allow(dead_code)]
    group_idx: usize,
    /// Decoded columns for this group (only needed columns populated),
    /// shared with the decoded group cache
    columns: Vec<Option<Arc<super::column::ColumnData>>>,
    /// Global row index where this group starts
    group_start: usize,
}

impl GroupColumnCache {
    /// Get column data and local row index for a global row index.
    #[inline(always)]
    fn col_and_local(
        &self,
        col_idx: usize,
        global_idx: usize,
    ) -> Option<(&super::column::ColumnData, usize)> {
        self.columns[col_idx]
            .as_deref()
            .map(|col| (col, global_idx - self.group_start))
    }
}

/// Statement visibility is resolved once for the requested part of each row
/// group, before any predicate column is loaded. Boxing keeps legacy scanners
/// at one pointer of overhead; the bitmap never grows with table cardinality.
struct CapturedVisibility {
    view: Arc<CapturedHotView>,
    pending: Arc<rustc_hash::FxHashSet<i64>>,
    hidden: smallvec::SmallVec<[u64; 4]>,
    group: Option<usize>,
    start: usize,
    all_hidden: bool,
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
    /// An empty projection requests every logical schema column, including
    /// columns added after this physical volume was written.
    projects_all_columns: bool,
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
    /// Pre-computed inter-volume visibility bitmap.
    /// Bit i is set (1) if row at index i is visible (not overridden by a newer volume).
    /// Stored as packed u64 words: word w covers rows [w*64 .. w*64+63].
    /// When None, all rows are assumed visible (no inter-volume dedup needed).
    visibility_bitmap: Option<Arc<Vec<u64>>>,
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
    /// Per-group column cache: decompresses only needed columns for the current
    /// group instead of the entire column. Active when the volume has a compressed
    /// store (V4 format). Dramatically reduces decompression work for selective scans.
    group_cache: Option<GroupColumnCache>,
    /// Cached end of the current row group (exclusive index). Avoids per-row
    /// integer division in the slow-path scan loop. Recomputed only on group
    /// boundary crossings. 0 means "not yet initialized".
    next_group_boundary: usize,
    /// Walk the range from its end down to its start
    reverse: bool,
    /// The caller consumes rows in the volume's order and may stop early, so
    /// the eager dictionary pre-scan over the whole range is not done
    ordered_walk: bool,
    /// Stop when the raw key of the row under the cursor is past this bound:
    /// (physical column, key, ascending). Only meaningful on a sorted column.
    stop_key: Option<(usize, i64, bool)>,
    /// Rows of the current group that pass the dictionary filters, ascending,
    /// found in one pass over the raw ids; consumed from the back on a
    /// reverse walk
    group_candidates: Vec<usize>,
    /// The group `group_candidates` was computed for
    candidates_group: Option<usize>,
    captured_visibility: Option<Box<CapturedVisibility>>,
}

impl VolumeScanner {
    /// Bind a fixed statement before installing its filter or starting the
    /// scan. The view supplies the same epoch for hot rows and tombstones.
    /// Pending deletes are the transaction's frozen statement overlay.
    pub fn set_captured_visibility(
        &mut self,
        view: Arc<CapturedHotView>,
        pending: Arc<rustc_hash::FxHashSet<i64>>,
    ) {
        // Short ranges, especially a primary-key point read, need only their
        // own bits. Larger scans reuse at most one physical group's bitmap.
        let words = self
            .end_idx
            .saturating_sub(self.current_idx)
            .min(super::column::ROW_GROUP_SIZE)
            .div_ceil(64);
        let mut hidden = smallvec::SmallVec::new();
        hidden.resize(words, 0);
        self.captured_visibility = Some(Box::new(CapturedVisibility {
            view,
            pending,
            hidden,
            group: None,
            start: 0,
            all_hidden: false,
        }));
        // Preserve a previously selected reverse direction. Captured scans
        // cannot eagerly inspect dictionary columns in later hidden groups.
        self.ordered_walk = true;
        self.matching_indices = None;
        self.match_idx = 0;
        self.next_group_boundary = 0;
        self.candidates_group = None;
        self.group_candidates.clear();
    }

    /// Returns true when every requested row in this group is hidden. The
    /// initial bounds remain valid as the forward/reverse cursor advances.
    fn prepare_captured_group(&mut self, group: usize, lo: usize, hi: usize) -> bool {
        let Some(policy) = &mut self.captured_visibility else {
            return false;
        };
        if policy.group == Some(group) {
            return policy.all_hidden;
        }
        let ids = &self.volume.meta.row_ids[lo..hi];
        policy.view.mark_authoritative(ids, &mut policy.hidden);
        policy.start = lo;
        policy.group = Some(group);
        if self.visibility_bitmap.is_none()
            && policy.pending.is_empty()
            && self
                .committed_tombstones
                .as_ref()
                .is_none_or(|tombstones| tombstones.is_empty())
        {
            // The authority writer clears unused tail bits; counting bounded
            // words avoids another row pass when cold exclusions are absent.
            policy.all_hidden = policy.hidden[..ids.len().div_ceil(64)]
                .iter()
                .map(|bits| bits.count_ones() as usize)
                .sum::<usize>()
                == ids.len();
            return policy.all_hidden;
        }
        policy.all_hidden = true;
        for (local, &id) in ids.iter().enumerate() {
            let global = lo + local;
            let hidden = self.visibility_bitmap.as_ref().is_some_and(|bitmap| {
                bitmap
                    .get(global / 64)
                    .is_some_and(|word| word & (1u64 << (global % 64)) == 0)
            }) || policy.pending.contains(&id)
                || self
                    .committed_tombstones
                    .as_ref()
                    .is_some_and(|tombstones| {
                        tombstones.get(&id).is_some_and(|&sequence| {
                            i64::try_from(sequence).is_ok_and(|sequence| {
                                policy.view.epoch().admits_commit_sequence(sequence)
                            })
                        })
                    });
            let bit = 1u64 << (local % 64);
            let word = &mut policy.hidden[local / 64];
            if hidden {
                *word |= bit;
            }
            policy.all_hidden &= *word & bit != 0;
        }
        policy.all_hidden
    }

    /// Rows in the volume's order, from the start (ascending) or from the end
    /// (descending), without the eager dictionary pre-scan over the range
    pub fn set_ordered_walk(&mut self, ascending: bool) {
        self.ordered_walk = true;
        self.reverse = !ascending;
    }

    /// Stop as soon as the row under the cursor has a key past `bound`
    /// (below it walking backwards, above it walking forwards). The column
    /// must be sorted in the volume; only integer and timestamp keys count.
    pub fn set_stop_key(&mut self, col_idx: usize, bound: &Value, ascending: bool) {
        let target = match (bound, self.volume.columns.data_type(col_idx)) {
            (Value::Integer(v), crate::core::DataType::Integer) => *v,
            (Value::Timestamp(dt), crate::core::DataType::Timestamp) => {
                dt.timestamp_nanos_opt().unwrap_or_else(|| {
                    dt.timestamp()
                        .saturating_mul(1_000_000_000)
                        .saturating_add(dt.timestamp_subsec_nanos() as i64)
                })
            }
            _ => return,
        };
        self.stop_key = Some((col_idx, target, ascending));
    }

    /// True when the row at `idx` lies past the stop key
    fn past_stop_key(&self, idx: usize) -> Result<bool> {
        let Some((col_idx, target, ascending)) = self.stop_key else {
            return Ok(false);
        };
        let (col, local) = self.col_and_idx(col_idx, idx)?;
        if col.is_null(local) {
            return Ok(false);
        }
        let key = col.get_i64(local);
        Ok(if ascending {
            key > target
        } else {
            key < target
        })
    }

    /// The rows in `[lo, hi)` that pass every dictionary filter, in one pass
    /// over the raw ids of the group; None when a filter column is not a
    /// dictionary column here and the rows must be tested one by one
    fn dictionary_candidates(&self, lo: usize, hi: usize) -> Result<Option<Vec<usize>>> {
        let mut filters: smallvec::SmallVec<[super::column::DictFilter<'_>; 4]> =
            smallvec::SmallVec::with_capacity(self.dict_filters.len());
        for &(col_idx, expected) in &self.dict_filters {
            let (col, local_lo) = self.col_and_idx(col_idx, lo)?;
            filters.push((col, local_lo, expected));
        }
        let mut candidates = Vec::new();
        if super::column::ColumnData::dict_matching_offsets(&filters, hi - lo, &mut candidates)
            .is_none()
        {
            return Ok(None);
        }
        for idx in &mut candidates {
            *idx += lo;
        }
        Ok(Some(candidates))
    }

    /// The reverse walk: the newest row of the range first. Row groups are
    /// entered from their end; a pruned group is skipped whole; with
    /// dictionary filters the group's candidates are found in one pass.
    fn next_reverse(&mut self) -> Result<bool> {
        let use_group_cache = self.volume.columns.should_use_group_cache();
        while self.current_idx < self.end_idx {
            let idx = self.end_idx - 1;
            let group_idx = idx / super::column::ROW_GROUP_SIZE;
            let group_start = group_idx * super::column::ROW_GROUP_SIZE;
            if self.prepare_captured_group(
                group_idx,
                group_start.max(self.current_idx),
                self.end_idx,
            ) {
                self.end_idx = group_start.max(self.current_idx);
                continue;
            }
            let cached = self.group_cache.as_ref().map(|c| c.group_idx);
            if cached != Some(group_idx) {
                if let Some(ref skips) = self.row_group_skips {
                    if group_idx < skips.len() && skips[group_idx] {
                        self.end_idx = group_start.max(self.current_idx);
                        continue;
                    }
                }
                if use_group_cache {
                    self.load_group_cache(group_idx);
                    if self.error.is_some() {
                        self.has_current = false;
                        return Ok(false);
                    }
                }
            }
            // Captured scans test visibility before even dictionary predicates.
            // Legacy reverse scans can batch candidates for the whole group.
            if self.captured_visibility.is_none()
                && !self.dict_filters.is_empty()
                && self.candidates_group != Some(group_idx)
            {
                let lo = group_start.max(self.current_idx);
                match self.dictionary_candidates(lo, self.end_idx)? {
                    Some(candidates) => {
                        self.group_candidates = candidates;
                        self.candidates_group = Some(group_idx);
                    }
                    None => {
                        self.group_candidates.clear();
                        self.candidates_group = None;
                    }
                }
            }
            let idx = if self.candidates_group == Some(group_idx) {
                match self.group_candidates.pop() {
                    Some(idx) => idx,
                    None => {
                        // The group holds no more candidates: leave it whole
                        self.end_idx = group_start.max(self.current_idx);
                        continue;
                    }
                }
            } else {
                idx
            };
            self.end_idx = idx;
            if self.should_skip_row(idx) {
                continue;
            }
            if self.past_stop_key(idx)? {
                self.end_idx = self.current_idx;
                self.has_current = false;
                return Ok(false);
            }
            if self.candidates_group != Some(group_idx)
                && !self.dict_filters.is_empty()
                && self.dict_filters_reject(idx)?
            {
                continue;
            }
            if !self.typed_predicates.is_empty() && !self.evaluate_typed_predicates(idx)? {
                continue;
            }
            if !self.materialize_row(idx)? {
                continue;
            }
            self.current_rid = self.volume.meta.row_ids[idx];
            self.has_current = true;
            return Ok(true);
        }
        self.has_current = false;
        Ok(false)
    }
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
        let projects_all_columns = project_cols.is_empty();
        let project = if project_cols.is_empty() {
            (0..volume.columns.len()).collect()
        } else {
            project_cols
        };
        let is_full_projection = Self::compute_is_full_projection(&project, volume.columns.len());
        // Stamp with current global eviction epoch so the volume ages correctly.
        volume.mark_accessed();
        let mut s = Self {
            end_idx: volume.meta.row_count,
            volume,
            project_cols: project,
            projects_all_columns,
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
            visibility_bitmap: None,
            pending_cold_deletes: None,
            committed_tombstones: None,
            snapshot_seq: None,
            typed_predicates: Vec::new(),
            needed_cols: None,
            row_group_skips: None,
            group_cache: None,
            next_group_boundary: 0,
            reverse: false,
            ordered_walk: false,
            stop_key: None,
            group_candidates: Vec::new(),
            candidates_group: None,
            captured_visibility: None,
        };
        if !s.is_full_projection && s.volume.columns.should_use_group_cache() {
            let mut mask = vec![false; s.volume.columns.len()];
            for &ci in &s.project_cols {
                if ci < mask.len() {
                    mask[ci] = true;
                }
            }
            s.needed_cols = Some(mask);
        }
        s
    }

    /// Create a scanner with a start/end range (for binary-search narrowing).
    pub fn with_range(
        volume: Arc<FrozenVolume>,
        project_cols: Vec<usize>,
        start_idx: usize,
        end_idx: usize,
        _delete_vector: Option<()>,
    ) -> Self {
        let projects_all_columns = project_cols.is_empty();
        let project = if project_cols.is_empty() {
            (0..volume.columns.len()).collect()
        } else {
            project_cols
        };
        let is_full_projection = Self::compute_is_full_projection(&project, volume.columns.len());
        volume.mark_accessed();
        let mut s = Self {
            volume,
            project_cols: project,
            projects_all_columns,
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
            visibility_bitmap: None,
            pending_cold_deletes: None,
            committed_tombstones: None,
            snapshot_seq: None,
            typed_predicates: Vec::new(),
            needed_cols: None,
            row_group_skips: None,
            group_cache: None,
            next_group_boundary: 0,
            reverse: false,
            ordered_walk: false,
            stop_key: None,
            group_candidates: Vec::new(),
            candidates_group: None,
            captured_visibility: None,
        };
        if !s.is_full_projection && s.volume.columns.should_use_group_cache() {
            let mut mask = vec![false; s.volume.columns.len()];
            for &ci in &s.project_cols {
                if ci < mask.len() {
                    mask[ci] = true;
                }
            }
            s.needed_cols = Some(mask);
        }
        s
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

    /// Set a pre-computed inter-volume visibility bitmap.
    /// Bit i is 1 if the row at position i in this volume is visible (not overridden by
    /// a newer volume). Bit i being 0 means a newer volume has a row with the same row_id,
    /// so this row should be skipped without materialization.
    pub fn set_visibility_bitmap(&mut self, bitmap: Option<Arc<Vec<u64>>>) {
        self.visibility_bitmap = bitmap;
    }

    /// Create an empty scanner (for zone-map-pruned volumes that match nothing).
    pub fn empty() -> Self {
        Self {
            volume: Arc::new(FrozenVolume {
                backing: std::sync::OnceLock::new(),
                columns: super::writer::LazyColumns::empty(),
                meta: Arc::new(super::writer::VolumeMeta {
                    zone_maps: Vec::new(),
                    bloom_filters: Vec::new(),
                    stats: super::stats::VolumeAggregateStats::new(0),
                    row_count: 0,
                    column_names: Vec::new(),
                    column_types: Vec::new(),
                    row_ids: Vec::new(),
                    sorted_columns: Vec::new(),
                    column_name_map: ahash::AHashMap::new(),
                    row_groups: Vec::new(),
                }),
                unique_indices: std::sync::Arc::new(parking_lot::RwLock::new(
                    rustc_hash::FxHashMap::default(),
                )),
                last_access_epoch: std::sync::atomic::AtomicU64::new(0),
            }),
            project_cols: Vec::new(),
            projects_all_columns: true,
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
            visibility_bitmap: None,
            pending_cold_deletes: None,
            committed_tombstones: None,
            snapshot_seq: None,
            typed_predicates: Vec::new(),
            needed_cols: None,
            row_group_skips: None,
            group_cache: None,
            next_group_boundary: 0,
            reverse: false,
            ordered_walk: false,
            stop_key: None,
            group_candidates: Vec::new(),
            candidates_group: None,
            captured_visibility: None,
        }
    }

    /// Set a predicate filter on this scanner.
    /// Automatically extracts dictionary-based fast filters for text equality predicates.
    pub fn set_filter(&mut self, filter: Box<dyn crate::storage::expression::Expression>) {
        if let Err(error) = self.try_set_filter(filter) {
            self.error = Some(error);
            self.has_current = false;
        }
    }

    fn try_set_filter(
        &mut self,
        filter: Box<dyn crate::storage::expression::Expression>,
    ) -> Result<()> {
        // Extract dictionary filters for fast pre-filtering.
        // Uses CompressedBlockStore's shared dict when available (no column decompression).
        // A schema mapping can replace an old physical column with a DEFAULT
        // (including DROP + ADD with the same name). Physical-name prefilters
        // cannot reject rows until that logical mapping has been applied.
        let comparisons = if self.column_mapping.is_some() {
            Vec::new()
        } else {
            filter.collect_comparisons()
        };
        // Only use CompressedBlockStore for dict lookup / group scan when
        // columns are NOT already loaded (deferred volumes from disk).
        // After seal/compaction, eager() pre-loads all OnceLock
        // slots — direct column access is faster than re-decompressing.
        let store = if self.volume.columns.should_use_group_cache() {
            self.volume.columns.compressed_store()
        } else {
            None
        };
        for &(col_name, op, value) in &comparisons {
            if op != crate::core::Operator::Eq {
                continue;
            }
            if let Value::Text(s) = value {
                if let Some(col_idx) = self.volume.column_index(col_name) {
                    let dict_id = if self.captured_visibility.is_some() {
                        // Optional dictionary narrowing must not force a
                        // column read before this group's authority mask.
                        let Some(dictionary) = self.volume.columns.get_column_dictionary(col_idx)
                        else {
                            continue;
                        };
                        dictionary
                            .iter()
                            .position(|value| value.as_str() == s.as_str())
                            .map(|index| index as u32)
                    } else if let Some(st) = store {
                        st.dict_lookup(col_idx, s.as_str())
                    } else {
                        self.volume.columns.get(col_idx)?.dict_lookup(s.as_str())
                    };
                    if let Some(id) = dict_id {
                        self.dict_filters.push((col_idx, id));
                    } else {
                        self.current_idx = self.end_idx;
                        self.filter = Some(filter);
                        return Ok(());
                    }
                }
            }
        }
        // Pre-compute matching row indices from dictionary filters.
        // Skip pre-computation when match rate is too high (>10%) to avoid
        // large Vec allocation — use streaming dict filter in the slow path instead.
        // An ordered walk may stop after a few rows, so it never pays for the
        // whole range up front.
        if !self.dict_filters.is_empty() && !self.ordered_walk {
            let scan_range = self.end_idx - self.current_idx;
            let selectivity_cap = scan_range / 10; // 10% threshold
            let matches = if let Some(st) = store {
                let mut m = Vec::new();
                let first_col = self.dict_filters[0].0;
                let num_grp = st.num_groups(first_col);
                let mut exceeded = false;
                for gi in 0..num_grp {
                    let gs = gi * super::column::ROW_GROUP_SIZE;
                    let ge = ((gi + 1) * super::column::ROW_GROUP_SIZE).min(self.end_idx);
                    if gs >= self.end_idx || ge <= self.current_idx {
                        continue;
                    }
                    let mut group_cols = Vec::with_capacity(self.dict_filters.len());
                    for &(ci, _) in &self.dict_filters {
                        group_cols.push(st.group_column(ci, gi)?);
                    }
                    let lo = gs.max(self.current_idx);
                    let filters: smallvec::SmallVec<[super::column::DictFilter<'_>; 4]> = self
                        .dict_filters
                        .iter()
                        .zip(group_cols.iter())
                        .map(|(&(_, eid), col)| (&**col, lo - gs, eid))
                        .collect();
                    let first = m.len();
                    if super::column::ColumnData::dict_matching_offsets(&filters, ge - lo, &mut m)
                        .is_none()
                    {
                        m.truncate(first);
                        for i in lo..ge {
                            let local = i - gs;
                            let ok = self.dict_filters.iter().zip(group_cols.iter()).all(
                                |(&(_, eid), col)| {
                                    !col.is_null(local) && col.get_dict_id(local) == eid
                                },
                            );
                            if ok {
                                m.push(i);
                            }
                        }
                    } else {
                        for idx in &mut m[first..] {
                            *idx += lo;
                        }
                    }
                    if m.len() > selectivity_cap {
                        exceeded = true;
                        break;
                    }
                }
                if exceeded {
                    None
                } else {
                    Some(m)
                }
            } else {
                let mut m = Vec::new();
                // One row group at a time, so a filter that matches most rows
                // stops at the selectivity cap instead of listing them all
                let mut lo = self.current_idx;
                let mut vectorized = true;
                while lo < self.end_idx && m.len() <= selectivity_cap {
                    let hi = (lo + super::column::ROW_GROUP_SIZE).min(self.end_idx);
                    let filters: smallvec::SmallVec<[super::column::DictFilter<'_>; 4]> = self
                        .dict_filters
                        .iter()
                        .map(|&(ci, eid)| Ok((self.volume.columns.get(ci)?, lo, eid)))
                        .collect::<Result<_>>()?;
                    let first = m.len();
                    if super::column::ColumnData::dict_matching_offsets(&filters, hi - lo, &mut m)
                        .is_none()
                    {
                        m.truncate(first);
                        vectorized = false;
                        break;
                    }
                    for idx in &mut m[first..] {
                        *idx += lo;
                    }
                    lo = hi;
                }
                if !vectorized {
                    m.clear();
                    for i in self.current_idx..self.end_idx {
                        let mut ok = true;
                        for &(ci, eid) in &self.dict_filters {
                            let col = self.volume.columns.get(ci)?;
                            if col.is_null(i) || col.get_dict_id(i) != eid {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            m.push(i);
                        }
                        if m.len() > selectivity_cap {
                            break;
                        }
                    }
                }
                if m.len() > selectivity_cap {
                    None
                } else {
                    Some(m)
                }
            };
            self.matching_indices = matches;
        }
        // Extract typed pre-filter predicates using data_type() (no column decompression).
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
            let col_dt = self.volume.columns.data_type(col_idx);
            let target = match (value, col_dt) {
                (Value::Integer(v), crate::core::DataType::Integer) => TypedTarget::Int64(*v),
                (Value::Float(v), crate::core::DataType::Float) => TypedTarget::Float64(*v),
                (Value::Boolean(v), crate::core::DataType::Boolean) => TypedTarget::Bool(*v),
                (Value::Timestamp(dt), crate::core::DataType::Timestamp) => {
                    TypedTarget::Int64(dt.timestamp_nanos_opt().unwrap_or_else(|| {
                        dt.timestamp()
                            .saturating_mul(1_000_000_000)
                            .saturating_add(dt.timestamp_subsec_nanos() as i64)
                    }))
                }
                // Only lossless integers: `i as f64` rounds above 2^53 and
                // the pre-filter must never reject rows the full filter
                // accepts
                (Value::Integer(v), crate::core::DataType::Float)
                    if crate::core::value::lossless_f64_from_i64(*v).is_some() =>
                {
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
            // Filter and projection positions address the logical schema.
            let mask_len = if let Some(ref m) = self.column_mapping {
                m.sources.len()
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
        } else {
            // Cannot determine filter columns — materialize all columns
            // so the filter evaluates against real data, not Null.
            self.needed_cols = None;
        }

        // Pre-compute row group skip decisions from per-group zone maps.
        // For each group, if ANY comparison's zone map says "no match",
        // the entire group can be skipped.
        if !self.volume.meta.row_groups.is_empty() && !comparisons.is_empty() {
            let skips: Vec<bool> = self
                .volume
                .meta
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
        Ok(())
    }

    /// Evaluate typed pre-filter predicates directly on column data.
    /// Returns false only if the row definitely does not match (safe rejection).
    /// NULL columns conservatively pass through (the full filter handles NULL logic).
    #[inline]
    fn evaluate_typed_predicates(&self, idx: usize) -> Result<bool> {
        for pred in &self.typed_predicates {
            let (col, local) = self.col_and_idx(pred.col_idx, idx)?;
            if col.is_null(local) {
                // NULL: conservatively pass through (might match under SQL NULL semantics).
                // The full filter will handle it correctly.
                continue;
            }
            let matches = match &pred.target {
                TypedTarget::Int64(target) => {
                    let val = col.get_i64(local);
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
                    let val = col.get_f64(local);
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
                    let val = col.get_bool(local);
                    match pred.op {
                        crate::core::Operator::Eq => val == *target,
                        crate::core::Operator::Ne => val != *target,
                        _ => true,
                    }
                }
            };
            if !matches {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Set a precomputed column mapping for schema-evolved volumes.
    /// Install before the filter or iteration. Projection and filter masks use
    /// logical positions; decompression maps
    /// those positions back to physical columns when loading a group.
    pub fn set_column_mapping(&mut self, mapping: super::writer::ColumnMapping) {
        if mapping.sources.iter().any(|source| {
            matches!(source,
            super::writer::ColSource::Volume(index) if *index >= self.volume.columns.len())
        }) {
            self.error = Some(Error::internal("volume column mapping index out of bounds"));
            self.has_current = false;
            return;
        }
        let logical_columns = mapping.sources.len();
        if mapping.is_identity
            && self.column_mapping.is_none()
            && logical_columns == self.volume.columns.len()
        {
            return;
        }
        if self.projects_all_columns && self.project_cols.len() != logical_columns {
            self.project_cols = (0..logical_columns).collect();
        }
        self.is_full_projection =
            Self::compute_is_full_projection(&self.project_cols, logical_columns);
        self.column_mapping = (!mapping.is_identity).then_some(mapping);
        self.group_cache = None;
        self.next_group_boundary = 0;
        self.dict_filters.clear();
        self.typed_predicates.clear();
        self.matching_indices = None;
        self.match_idx = 0;
        self.row_group_skips = None;
        self.group_candidates.clear();
        self.candidates_group = None;
        let mut references = Vec::new();
        if self
            .filter
            .as_ref()
            .is_some_and(|filter| !filter.collect_column_indices(&mut references))
            || (self.filter.is_none() && self.is_full_projection)
        {
            self.needed_cols = None;
        } else {
            let mut mask = vec![false; logical_columns];
            for &index in self.project_cols.iter().chain(&references) {
                if let Some(needed) = mask.get_mut(index) {
                    *needed = true;
                }
            }
            self.needed_cols = Some(mask);
        }
    }

    /// Get (column_data, local_index) for a global row index.
    /// Uses group cache when available, falls back to full volume columns.
    #[inline(always)]
    fn col_and_idx(
        &self,
        col_idx: usize,
        global_idx: usize,
    ) -> Result<(&super::column::ColumnData, usize)> {
        if let Some(ref cache) = self.group_cache {
            if let Some(pair) = cache.col_and_local(col_idx, global_idx) {
                return Ok(pair);
            }
        }
        Ok((self.volume.columns.get(col_idx)?, global_idx))
    }

    /// Load group cache for a new row group. Decompresses only the columns
    /// needed for filtering + projection from the compressed store.
    fn load_group_cache(&mut self, group_idx: usize) {
        let store = match self.volume.columns.compressed_store() {
            Some(s) => s,
            None => {
                // No compressed store: clear stale cache so materialize_row
                // falls through to the full-column path instead of using
                // a cache from a previous group with wrong group_start.
                self.group_cache = None;
                return;
            }
        };
        let col_count = self.volume.columns.len();
        let group_start = group_idx * super::column::ROW_GROUP_SIZE;

        let mut columns: Vec<Option<Arc<super::column::ColumnData>>> = vec![None; col_count];
        if let Some(mapping) = &self.column_mapping {
            for (logical, source) in mapping.sources.iter().enumerate() {
                if self
                    .needed_cols
                    .as_ref()
                    .is_some_and(|needed| !needed.get(logical).copied().unwrap_or(false))
                {
                    continue;
                }
                let super::writer::ColSource::Volume(ci) = *source else {
                    continue;
                };
                if columns[ci].is_some() {
                    continue;
                }
                match store.group_column(ci, group_idx) {
                    Ok(column) => columns[ci] = Some(column),
                    Err(error) => {
                        self.error = Some(error.into());
                        return;
                    }
                }
            }
        } else if let Some(ref needed) = self.needed_cols {
            for (ci, &need) in needed.iter().enumerate() {
                if need && ci < col_count && group_idx < store.num_groups(ci) {
                    match store.group_column(ci, group_idx) {
                        Ok(col) => columns[ci] = Some(col),
                        Err(e) => {
                            self.error = Some(e.into());
                            return;
                        }
                    }
                }
            }
        } else {
            for (ci, slot) in columns.iter_mut().enumerate() {
                if group_idx < store.num_groups(ci) {
                    match store.group_column(ci, group_idx) {
                        Ok(col) => *slot = Some(col),
                        Err(e) => {
                            self.error = Some(e.into());
                            return;
                        }
                    }
                }
            }
        }

        self.group_cache = Some(GroupColumnCache {
            group_idx,
            columns,
            group_start,
        });
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
        if let Some(policy) = &self.captured_visibility {
            debug_assert_eq!(policy.group, Some(idx / super::column::ROW_GROUP_SIZE));
            let local = idx - policy.start;
            return policy.hidden[local / 64] & (1u64 << (local % 64)) != 0;
        }
        // Check pre-computed inter-volume visibility bitmap first (O(1) bit check).
        // A clear bit means a newer volume owns this row_id — skip without materialization.
        if let Some(ref bm) = self.visibility_bitmap {
            let word_idx = idx >> 6;
            if word_idx < bm.len() && (bm[word_idx] >> (idx & 63)) & 1 == 0 {
                return true;
            }
        }
        let rid = self.volume.meta.row_ids[idx];
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
    fn dict_filters_reject(&self, idx: usize) -> Result<bool> {
        for &(col_idx, expected_id) in &self.dict_filters {
            let (col, local) = self.col_and_idx(col_idx, idx)?;
            if col.is_null(local) || col.get_dict_id(local) != expected_id {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Materialize a row at `idx`, evaluate the filter (if any), and write
    /// the result into `self.current_row`. Returns false if the filter
    /// rejects the row.
    #[inline(always)]
    fn materialize_row(&mut self, idx: usize) -> Result<bool> {
        // Both paths use already-loaded group columns. Schema mappings resolve
        // logical positions without reloading entire physical columns.
        if self.group_cache.is_some() && self.column_mapping.is_some() {
            return self.materialize_mapped_row_from_cache(idx);
        }
        if self.group_cache.is_some() {
            return self.materialize_row_from_cache(idx);
        }

        if let Some(ref filter) = self.filter {
            let full_row = match (&self.needed_cols, &self.column_mapping) {
                (Some(mask), Some(mapping)) => {
                    self.volume.get_row_mapped_needed(idx, mapping, mask)?
                }
                (Some(mask), None) => self.volume.get_row_needed(idx, mask)?,
                (None, Some(mapping)) => self.volume.get_row_mapped(idx, mapping)?,
                (None, None) => self.volume.get_row(idx)?,
            };
            if !filter.evaluate(&full_row)? {
                return Ok(false);
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
                self.current_row = self.volume.get_row_mapped(idx, mapping)?;
            } else {
                self.current_row =
                    self.volume
                        .get_row_mapped_projected(idx, mapping, &self.project_cols)?;
            }
        } else if self.is_full_projection {
            self.current_row = self.volume.get_row(idx)?;
        } else {
            self.current_row = self.volume.get_row_projected(idx, &self.project_cols)?;
        }
        Ok(true)
    }

    /// Build a row from the per-group column cache (avoids full-column decompression).
    fn materialize_mapped_row_from_cache(&mut self, idx: usize) -> Result<bool> {
        let mapping = self.column_mapping.as_ref().expect("mapped cache path");
        let mut values = Vec::with_capacity(mapping.sources.len());
        for (logical, source) in mapping.sources.iter().enumerate() {
            let needed = self
                .needed_cols
                .as_ref()
                .is_none_or(|mask| mask.get(logical).copied().unwrap_or(false));
            let value = match source {
                super::writer::ColSource::Volume(physical) => {
                    if needed {
                        let (column, local) = self.col_and_idx(*physical, idx)?;
                        column.get_value(local)
                    } else {
                        Value::Null(self.volume.columns.data_type(*physical))
                    }
                }
                super::writer::ColSource::Default(value) => {
                    if needed {
                        value.clone()
                    } else {
                        Value::Null(value.data_type())
                    }
                }
            };
            values.push(value);
        }
        self.finish_cached_row(Row::from_values(values))
    }

    fn materialize_row_from_cache(&mut self, idx: usize) -> Result<bool> {
        let col_count = self.volume.columns.len();
        let mut values = Vec::with_capacity(col_count);
        for ci in 0..col_count {
            if self
                .needed_cols
                .as_ref()
                .is_none_or(|needed| ci < needed.len() && needed[ci])
            {
                let (col, local) = self.col_and_idx(ci, idx)?;
                values.push(col.get_value(local));
            } else {
                values.push(Value::Null(self.volume.columns.data_type(ci)));
            }
        }
        self.finish_cached_row(Row::from_values(values))
    }

    fn finish_cached_row(&mut self, full_row: Row) -> Result<bool> {
        if self
            .filter
            .as_ref()
            .map(|filter| filter.evaluate(&full_row))
            .transpose()?
            .is_some_and(|matched| !matched)
        {
            return Ok(false);
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
        Ok(true)
    }
}

impl VolumeScanner {
    fn try_next(&mut self) -> Result<bool> {
        if self.error.is_some() {
            self.has_current = false;
            return Ok(false);
        }
        if self.reverse {
            return self.next_reverse();
        }

        // Fast path: use pre-computed matching indices (from dictionary filters).
        let use_group_cache_fast = self.volume.columns.should_use_group_cache();
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
                        return Ok(false);
                    }
                };

                let gi = idx / super::column::ROW_GROUP_SIZE;
                self.prepare_captured_group(
                    gi,
                    gi * super::column::ROW_GROUP_SIZE,
                    ((gi + 1) * super::column::ROW_GROUP_SIZE).min(self.end_idx),
                );
                if self.should_skip_row(idx) {
                    continue;
                }

                // Load group cache on group transition (matching_indices are sorted)
                if use_group_cache_fast {
                    let gi = idx / super::column::ROW_GROUP_SIZE;
                    let need_load = self.group_cache.as_ref().is_none_or(|c| c.group_idx != gi);
                    if need_load {
                        self.load_group_cache(gi);
                        if self.error.is_some() {
                            self.has_current = false;
                            return Ok(false);
                        }
                    }
                }

                if !self.typed_predicates.is_empty() && !self.evaluate_typed_predicates(idx)? {
                    continue;
                }
                if !self.materialize_row(idx)? {
                    continue;
                }

                self.current_rid = self.volume.meta.row_ids[idx];
                self.has_current = true;
                return Ok(true);
            }
        }

        // Slow path: linear scan with row-group skipping + per-group decompression
        let use_group_cache = self.volume.columns.should_use_group_cache();
        while self.current_idx < self.end_idx {
            // Row-group boundary: skip pruned groups + load group cache
            if self.current_idx >= self.next_group_boundary {
                let group_idx = self.current_idx / super::column::ROW_GROUP_SIZE;
                self.next_group_boundary =
                    ((group_idx + 1) * super::column::ROW_GROUP_SIZE).min(self.end_idx);

                if self.prepare_captured_group(
                    group_idx,
                    self.current_idx,
                    self.next_group_boundary,
                ) {
                    self.current_idx = self.next_group_boundary;
                    continue;
                }

                // Zone map skip
                if let Some(ref skips) = self.row_group_skips {
                    if group_idx < skips.len() && skips[group_idx] {
                        self.current_idx = self.next_group_boundary;
                        continue;
                    }
                }

                // Load per-group cache (V4 only)
                if use_group_cache {
                    self.load_group_cache(group_idx);
                    if self.error.is_some() {
                        self.has_current = false;
                        return Ok(false);
                    }
                }
            }

            if self.should_skip_row(self.current_idx) {
                self.current_idx += 1;
                continue;
            }

            if self.stop_key.is_some() && self.past_stop_key(self.current_idx)? {
                self.current_idx = self.end_idx;
                self.has_current = false;
                return Ok(false);
            }

            let idx = self.current_idx;

            if !self.dict_filters.is_empty() && self.dict_filters_reject(idx)? {
                self.current_idx += 1;
                continue;
            }
            if !self.typed_predicates.is_empty() && !self.evaluate_typed_predicates(idx)? {
                self.current_idx += 1;
                continue;
            }
            if !self.materialize_row(idx)? {
                self.current_idx += 1;
                continue;
            }

            self.current_rid = self.volume.meta.row_ids[idx];
            self.has_current = true;
            self.current_idx += 1;
            return Ok(true);
        }

        self.has_current = false;
        Ok(false)
    }
}

impl Scanner for VolumeScanner {
    fn next(&mut self) -> bool {
        match self.try_next() {
            Ok(value) => value,
            Err(error) => {
                self.error = Some(error);
                self.has_current = false;
                false
            }
        }
    }

    fn row(&self) -> &Row {
        &self.current_row
    }

    fn err(&self) -> Option<&Error> {
        self.error.as_ref()
    }

    fn close(&mut self) -> Result<()> {
        self.has_current = false;
        if let Some(policy) = self.captured_visibility.take() {
            // A closed scanner can remain allocated. Release its captured
            // owners and buffers, then the epoch, and never resume the cursor.
            self.current_idx = self.end_idx;
            self.matching_indices = None;
            self.match_idx = 0;
            self.group_cache = None;
            self.group_candidates = Vec::new();
            self.candidates_group = None;
            self.visibility_bitmap = None;
            self.pending_cold_deletes = None;
            self.committed_tombstones = None;
            self.filter = None;
            self.current_row = Row::new();
            drop(policy);
        }
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
    use crate::storage::expression::{ComparisonExpr, Expression};
    use crate::storage::mvcc::registry::TransactionRegistry;
    use crate::storage::mvcc::version_store::{RowVersion, VersionStore};

    fn captured_store() -> (Arc<TransactionRegistry>, VersionStore) {
        let registry = Arc::new(TransactionRegistry::new());
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build();
        let store = VersionStore::with_visibility_checker("test", schema, registry.clone());
        (registry, store)
    }

    fn scan_ids(scanner: &mut VolumeScanner) -> Vec<i64> {
        let mut ids = Vec::new();
        while scanner.next() {
            ids.push(scanner.current_rid);
        }
        assert!(scanner.err().is_none(), "{:?}", scanner.err());
        ids
    }

    #[test]
    fn mapped_projection_keeps_added_defaults_and_explicit_physical_width_prefix() {
        use super::super::writer::{ColSource, ColumnMapping};
        let (registry, store) = captured_store();
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        let old_schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("g", DataType::Integer, false, false)
            .build();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("g", DataType::Integer, false, false)
            .column("added", DataType::Integer, false, false)
            .build();
        let mut builder = VolumeBuilder::new(&old_schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::Integer(20)]),
        );
        let mut volume = builder.finish();
        let (_, compressed) = super::super::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(compressed);
        let warm = Arc::new(volume.to_warm().unwrap());
        let eager = Arc::new(volume);
        let mapping = ColumnMapping {
            sources: vec![
                ColSource::Volume(0),
                ColSource::Volume(1),
                ColSource::Default(Value::Integer(7)),
            ],
            is_identity: false,
        };
        for volume in [eager, warm] {
            for columns in [vec![], vec![0, 1], vec![0, 1, 2], vec![2], vec![2, 0]] {
                for filtered in [false, true] {
                    let mut scanner = VolumeScanner::new(volume.clone(), columns.clone(), None);
                    scanner.set_column_mapping(mapping.clone());
                    scanner.set_captured_visibility(view.clone(), Arc::default());
                    if filtered {
                        let mut filter = ComparisonExpr::eq("id", Value::Integer(1));
                        filter.prepare_for_schema(&schema);
                        scanner.set_filter(Box::new(filter));
                    }
                    assert!(scanner.next(), "{:?}", scanner.err());
                    let values = [Value::Integer(1), Value::Integer(20), Value::Integer(7)];
                    let expected = if columns.is_empty() {
                        values.to_vec()
                    } else {
                        columns.iter().map(|&index| values[index].clone()).collect()
                    };
                    assert_eq!(scanner.row(), &Row::from_values(expected));
                    assert!(!scanner.next());
                    assert!(scanner.err().is_none());
                }
            }
        }
    }

    #[cfg(feature = "test-failpoints")]
    #[test]
    fn mapped_default_filter_does_not_read_dropped_or_omitted_physical_columns() {
        use super::super::writer::{ColSource, ColumnMapping};
        let _guard = crate::test_failpoints::FailpointGuard::new();
        let (registry, store) = captured_store();
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        let mut volume = Arc::try_unwrap(make_test_volume()).ok().unwrap();
        let (_, compressed) = super::super::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(compressed);
        let warm = Arc::new(volume.to_warm().unwrap());
        let schema = SchemaBuilder::new("test")
            .column("name", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build();
        let mapping = ColumnMapping {
            sources: vec![
                ColSource::Default(Value::text("replacement")),
                ColSource::Volume(2),
            ],
            is_identity: false,
        };
        for ascending in [true, false] {
            let mut scanner = VolumeScanner::new(warm.clone(), vec![0], None);
            scanner.set_column_mapping(mapping.clone());
            scanner.set_captured_visibility(view.clone(), Arc::default());
            scanner.set_ordered_walk(ascending);
            let mut filter = ComparisonExpr::eq("name", Value::text("replacement"));
            filter.prepare_for_schema(&schema);
            crate::test_failpoints::fail_cold_read_on(1);
            scanner.set_filter(Box::new(filter));
            let mut count = 0;
            while scanner.next() {
                assert_eq!(
                    scanner.row(),
                    &Row::from_values(vec![Value::text("replacement")])
                );
                count += 1;
            }
            assert_eq!(count, 5);
            assert!(scanner.err().is_none(), "{:?}", scanner.err());
            let mut required = VolumeScanner::new(warm.clone(), vec![2], None);
            assert!(
                !required.next(),
                "the first physical access must retain the fault"
            );
            assert!(required.err().is_some());
        }
    }

    #[test]
    fn captured_authority_precedes_filters_and_keeps_invisible_hot_fallback() {
        let (registry, store) = captured_store();
        let (committed, _) = registry.begin_transaction();
        registry.start_commit(committed);
        store
            .add_version(1, RowVersion::new(committed, Row::new()))
            .unwrap();
        let mut deleted = RowVersion::new(committed, Row::new());
        deleted.deleted_at_txn_id = committed;
        store.add_version(2, deleted).unwrap();
        registry.complete_commit(committed);
        let (inflight, _) = registry.begin_transaction();
        registry.start_commit(inflight);
        store
            .add_version(3, RowVersion::new(inflight, Row::new()))
            .unwrap();
        let epoch = registry.capture_read_epoch();
        let view = Arc::new(CapturedHotView::new(store.capture_hot_root(), epoch, None));
        registry.complete_commit(inflight);
        let pending: Arc<rustc_hash::FxHashSet<i64>> = Arc::new([4].into_iter().collect());
        let volume = make_test_volume();

        for start in 0..=5 {
            for end in start..=5 {
                for ascending in [true, false] {
                    let mut scanner =
                        VolumeScanner::with_range(volume.clone(), vec![], start, end, None);
                    scanner.set_ordered_walk(ascending);
                    scanner.set_captured_visibility(view.clone(), pending.clone());
                    let mut expected: Vec<i64> = [3, 5]
                        .into_iter()
                        .filter(|id| start < *id as usize && (*id as usize - 1) < end)
                        .collect();
                    if !ascending {
                        expected.reverse();
                    }
                    assert_eq!(scan_ids(&mut scanner), expected);
                }
            }
        }

        // The old cold value matches "apple", but visible hot authority must
        // suppress it even though the replacement hot row does not match.
        for ascending in [true, false] {
            let mut scanner = VolumeScanner::new(volume.clone(), vec![], None);
            scanner.set_captured_visibility(view.clone(), pending.clone());
            scanner.set_ordered_walk(ascending);
            let mut filter = ComparisonExpr::eq("name", Value::text("apple"));
            filter.prepare_for_schema(&store.schema());
            scanner.set_filter(Box::new(filter));
            assert!(scan_ids(&mut scanner).is_empty());
            assert!(scanner.matching_indices.is_none());
        }

        // An excluded tombstone remains invisible after its publisher commits.
        let tombstones: Arc<rustc_hash::FxHashMap<i64, u64>> = Arc::new(
            [
                (3, registry.get_commit_sequence(inflight).unwrap() as u64),
                (5, registry.get_commit_sequence(committed).unwrap() as u64),
            ]
            .into_iter()
            .collect(),
        );
        let mut scanner = VolumeScanner::new(volume.clone(), vec![], None);
        scanner.set_captured_visibility(view.clone(), pending.clone());
        scanner.set_skip_sets(tombstones.clone(), pending.clone());
        assert_eq!(scan_ids(&mut scanner), vec![3]);

        let mut scanner = VolumeScanner::new(volume, vec![], None);
        scanner.set_captured_visibility(view, pending.clone());
        scanner.set_skip_sets(tombstones, pending);
        scanner.set_visibility_bitmap(Some(Arc::new(vec![!(1u64 << 2)])));
        assert!(scan_ids(&mut scanner).is_empty());
    }

    #[test]
    fn captured_hidden_range_never_loads_columns_or_stop_keys() {
        let (registry, store) = captured_store();
        let (txn, _) = registry.begin_transaction();
        registry.start_commit(txn);
        for id in [2, 3] {
            store
                .add_version(id, RowVersion::new(txn, Row::new()))
                .unwrap();
        }
        registry.complete_commit(txn);
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        // A metadata-only volume has no column source. Any attempted column
        // access fails, including optional dictionary resolution in set_filter.
        let volume = Arc::new(make_test_volume().to_cold());
        for ascending in [true, false] {
            let mut scanner = VolumeScanner::with_range(volume.clone(), vec![], 1, 3, None);
            scanner.set_ordered_walk(ascending);
            scanner.set_captured_visibility(view.clone(), Arc::default());
            scanner.set_stop_key(0, &Value::Integer(2), ascending);
            let mut filter = ComparisonExpr::eq("name", Value::text("banana"));
            filter.prepare_for_schema(&store.schema());
            scanner.set_filter(Box::new(filter));
            assert!(scan_ids(&mut scanner).is_empty());
        }
        let mut visible = VolumeScanner::new(volume, vec![], None);
        visible.set_captured_visibility(view, Arc::default());
        assert!(!visible.next());
        assert!(
            visible.err().is_some(),
            "visible cold access must still fail"
        );
    }

    #[test]
    fn captured_close_releases_epoch_and_stops_without_discarding_error() {
        let (registry, store) = captured_store();
        for reverse in [false, true] {
            for unreadable in [false, true] {
                assert_eq!(registry.oldest_retention_horizon(), None);
                let epoch = registry.capture_read_epoch();
                let horizon = epoch.retention_horizon();
                let view = Arc::new(CapturedHotView::new(store.capture_hot_root(), epoch, None));
                let volume = make_test_volume();
                let volume = if unreadable {
                    Arc::new(volume.to_cold())
                } else {
                    volume
                };
                let mut scanner = VolumeScanner::new(volume, vec![], None);
                scanner.set_captured_visibility(view, Arc::default());
                scanner.set_ordered_walk(!reverse);
                assert_eq!(registry.oldest_retention_horizon(), Some(horizon));
                assert_eq!(scanner.next(), !unreadable);
                let error = scanner.err().map(ToString::to_string);
                scanner.close().unwrap();
                assert_eq!(registry.oldest_retention_horizon(), None);
                assert!(scanner.captured_visibility.is_none());
                assert!(!scanner.next(), "closed object must not resume scanning");
                assert_eq!(scanner.err().map(ToString::to_string), error);
                scanner.close().unwrap();
                assert!(!scanner.next());
            }
        }
    }

    #[cfg(feature = "test-failpoints")]
    #[test]
    fn captured_hidden_warm_group_skips_dictionary_prescan_and_decompression() {
        let _guard = crate::test_failpoints::FailpointGuard::new();
        let (registry, store) = captured_store();
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        let mut volume = Arc::try_unwrap(make_test_volume()).ok().unwrap();
        let (_, compressed) = super::super::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(compressed);
        let warm = Arc::new(volume.to_warm().unwrap());
        assert!(warm.columns.should_use_group_cache());
        for ascending in [true, false] {
            let mut scanner = VolumeScanner::new(warm.clone(), vec![], None);
            scanner.set_captured_visibility(view.clone(), Arc::new((1..=5).collect()));
            scanner.set_ordered_walk(ascending);
            let mut filter = ComparisonExpr::eq("name", Value::text("apple"));
            filter.prepare_for_schema(&store.schema());
            crate::test_failpoints::fail_cold_read_on(1);
            scanner.set_filter(Box::new(filter));
            assert!(scan_ids(&mut scanner).is_empty());
            // No masked access consumed the fault, and a required access
            // remains fallible rather than treating unreadable data as empty.
            let mut visible = VolumeScanner::new(warm.clone(), vec![], None);
            visible.set_captured_visibility(view.clone(), Arc::default());
            assert!(!visible.next());
            assert!(visible.err().is_some());
        }
    }

    #[test]
    fn captured_point_range_bounds_authority_work_with_many_hot_rows() {
        let (registry, store) = captured_store();
        let schema = SchemaBuilder::new("point")
            .column("id", DataType::Integer, false, true)
            .build();
        let target = 2048usize;
        let mut builder = VolumeBuilder::with_capacity(&schema, 4096);
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        for id in 0..4096 {
            let row = Row::from_values(vec![Value::Integer(id as i64)]);
            builder.add_row(id as i64, &row);
            if id != target {
                store
                    .add_version(id as i64, RowVersion::new(writer, row))
                    .unwrap();
            }
        }
        registry.complete_commit(writer);
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        assert!(!view.is_empty());
        let volume = Arc::new(builder.finish());
        for ascending in [true, false] {
            let mut scanner =
                VolumeScanner::with_range(volume.clone(), vec![0], target, target + 1, None);
            scanner.set_captured_visibility(view.clone(), Arc::default());
            scanner.set_ordered_walk(ascending);
            let mut filter = ComparisonExpr::eq("id", Value::Integer(target as i64));
            filter.prepare_for_schema(&schema);
            scanner.set_filter(Box::new(filter));
            assert_eq!(
                scanner.captured_visibility.as_ref().unwrap().hidden.len(),
                1
            );
            assert!(scanner.next(), "{:?}", scanner.err());
            assert_eq!(scanner.current_row_id(), target as i64);
            let policy = scanner.captured_visibility.as_ref().unwrap();
            assert_eq!(
                policy.start, target,
                "authority begins at the requested point, not the physical group"
            );
            assert_eq!(policy.hidden[0], 0);
            assert_eq!(
                policy.hidden.len(),
                1,
                "a point needs only its own mask word"
            );
            assert!(!scanner.next());
            assert!(scanner.err().is_none());
        }
    }

    #[test]
    fn captured_visibility_reuses_bounded_mask_across_group_and_word_boundaries() {
        let (registry, store) = captured_store();
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        let schema = SchemaBuilder::new("large")
            .column("id", DataType::Integer, false, true)
            .build();
        let count = super::super::column::ROW_GROUP_SIZE + 65;
        let mut builder = VolumeBuilder::with_capacity(&schema, count);
        for id in 0..count {
            builder.add_row(
                id as i64,
                &Row::from_values(vec![Value::Integer(id as i64)]),
            );
        }
        let volume = Arc::new(builder.finish());
        let survivors = [0, 63, 64, count - 66, count - 65, count - 1];
        let pending: Arc<rustc_hash::FxHashSet<i64>> = Arc::new(
            (0..count)
                .filter(|id| !survivors.contains(id))
                .map(|id| id as i64)
                .collect(),
        );
        for ascending in [true, false] {
            let mut scanner = VolumeScanner::with_range(volume.clone(), vec![], 1, count, None);
            scanner.set_captured_visibility(view.clone(), pending.clone());
            scanner.set_ordered_walk(ascending);
            let mut expected: Vec<i64> = survivors[1..].iter().map(|id| *id as i64).collect();
            if !ascending {
                expected.reverse();
            }
            assert_eq!(scan_ids(&mut scanner), expected);
        }
    }

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
